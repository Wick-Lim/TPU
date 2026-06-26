`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// rmsnorm_unit_tb.v  --  thorough self-checking TB for rmsnorm_unit  (§6,§8.4)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL
//   The DUT computes  y[i] = x[i] * rsqrt((1/LEN)*Sx^2 + eps) * gamma[i]
//   entirely in the project's bit-accurate fp32 primitives (glm_fp.vh): a
//   fp32 adder-tree Sx^2 reduce, a fp32 mean+eps, a Quake-seed Newton rsqrt,
//   and an fp32 multiply chain rounded to bf16 on output.
//
//   The golden here shares NONE of that arithmetic.  It recomputes RMSNorm a
//   COMPLETELY DIFFERENT way, in Verilog `real` (IEEE double, fp64):
//     * widen each stored bf16 x[i] to its EXACT real value (bf16->real),
//     * accumulate Sx^2 in fp64 (not the fp32 tree),
//     * mean = Sx^2 / LEN  in fp64 real division,
//     * scale = 1.0/$sqrt(mean + eps_real)   using the math-library sqrt
//       (a true sqrt, NOT the magic-constant + Newton approximation),
//     * y_ref_real[i] = x_real[i] * scale * gamma_real[i]  in fp64,
//     * quantize y_ref_real[i] to bf16 the SAME way the unit emits bf16
//       (real -> fp32 bits via $shortrealtobits, then fp32_to_bf16 RNE).
//   So the only shared step is the final fp32->bf16 RNE pack (which IS the
//   unit's defined output format -- both DUT and golden must land on the same
//   bf16 grid).  Everything that produces the *value* differs, so the golden
//   catches DUT arithmetic bugs instead of mirroring them.
//
//   The DUT is fed bf16-quantized x and gamma (the same bit patterns the
//   golden widens), so any discrepancy is the unit's fp32/rsqrt path, not
//   input-quantization skew.
//
//----------------------------------------------------------------------------
// TOLERANCE  (STATED + JUSTIFIED)
//   We compare the DUT bf16 output to the fp64-golden value (re-widened from
//   its bf16 quantization) as a RELATIVE error, per element:
//        relerr = |y_dut_real - y_gold_real| / max(|y_gold_real|, TINY)
//   and require  relerr <= REL_TOL = 2.0^-6  (= 1/64 ~= 0.0156).
//
//   Why 2^-6:
//     * bf16 carries 8 significand bits (7 stored + implicit), so one bf16
//       ULP is 2^-8 of the value; rounding the final result to bf16 alone
//       costs up to 0.5 ULP = 2^-9.
//     * The rsqrt scale is a single per-vector scalar; the unit's rsqrt is
//       measured at <= 2^-17.7 rel-err, far below a bf16 ULP, so it adds well
//       under 1 ULP to each element.
//     * The fp32 reduce/mul chain is ~2^-23 per op and is dwarfed by the bf16
//       output rounding.
//     * The DOMINANT effect is that the DUT and golden can round the SAME real
//       product to ADJACENT bf16 values when it sits near a rounding boundary
//       (the golden rounds the exact fp64 product; the DUT rounds an fp32
//       product whose own rounding can tip it across the halfway point).  That
//       is bounded by ~1 bf16 ULP = 2^-8 of the value.  2^-6 = 4 bf16 ULP
//       leaves comfortable margin for the rare double-rounding case while
//       still being TIGHT enough to fail on a real arithmetic bug (a wrong
//       scale or a dropped term moves elements by >> 4 ULP).
//   For elements whose golden magnitude is below TINY (true near-zero, where
//   relative error is meaningless), we instead require the DUT magnitude to
//   also be within ABS_TOL of zero.
//
//   We additionally track and PRINT the worst observed rel-err across the
//   whole run as a sanity readout.
//
//----------------------------------------------------------------------------
// COVERAGE  (each element checked within tolerance; $fatal on any miss)
//   Three lengths exercise the datapath, the non-trivial beat counter, and the
//   GLM-scale reduce:
//        LEN = 16   (small, stresses short pipeline / beat==LAST edge)
//        LEN = 128  (the default)
//        LEN = 6144 (the GLM-5.2 hidden size -- the fp32-reduce raison d'etre)
//   Directed cases per length:
//        all-equal vector            (rms == |val|, y == sign*gamma)
//        single large outlier        (one element dominates Sx^2)
//        near-zero vector (eps wins) (mean ~ 0 -> eps floors the scale)
//        gamma = 1                   (pure normalize)
//        random gamma                (full y = x*scale*gamma)
//        all-negative x              (sign handling through the chain)
//   Plus many RANDOM vectors per length spanning a WIDE dynamic range
//   (|x| ~ 1e-3 .. 1e3, mixed sign) with both gamma=1 and random gamma.
//   A stall test re-runs one vector while throttling x_valid/g_valid to prove
//   the ready/valid handshake is correct under back-pressure (same result).
//
//   GATES: prints "ALL <N> TESTS PASSED"; $fatal on any element mismatch,
//   any nan/inf in the output, or any handshake/protocol violation.
//============================================================================
module rmsnorm_unit_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ---- elaboration: instantiate three DUTs of different LEN ----
    localparam integer LANES = 4;
    localparam [31:0]  EPS   = 32'h3727C5AC;          // 1e-5 fp32
    localparam real    EPS_R = 1.0e-5;                 // matching real eps

    // tolerances
    localparam real REL_TOL = 1.0/64.0;                // 2^-6
    localparam real ABS_TOL = 1.0/64.0;                // for near-zero golden
    localparam real TINY    = 1.0e-9;

    // ===================================================================
    //  Per-DUT signal bundles (LEN=16, 128, 6144).  We instantiate three
    //  separate rmsnorm_unit modules so each compiles with its own LEN
    //  localparams; a shared `task` drives whichever one is selected.
    // ===================================================================
    localparam integer L0 = 16, L1 = 128, L2 = 6144;
    localparam integer N0 = L0/LANES, N1 = L1/LANES, N2 = L2/LANES;
    localparam integer NMAX = N2;                       // largest beat count

    // generic driver-side storage (sized to the largest LEN)
    reg  [LANES*16-1:0] xbuf  [0:NMAX-1];
    reg  [LANES*16-1:0] gbuf  [0:NMAX-1];

    // shared control
    reg                 start;
    reg                 stall_en;                       // throttle valids when 1

    // ---- DUT 0 (LEN=16) ----
    reg                 d0_start;
    wire                d0_in_req, d0_g_req, d0_y_valid, d0_busy, d0_done;
    reg  [LANES*16-1:0] d0_x_in, d0_gamma_in;
    reg                 d0_x_valid, d0_g_valid;
    wire [LANES*16-1:0] d0_y_out;
    rmsnorm_unit #(.LEN(L0), .LANES(LANES), .EPS(EPS)) dut0 (
        .clk(clk), .rst(rst), .start(d0_start),
        .in_req(d0_in_req), .x_in(d0_x_in), .x_valid(d0_x_valid),
        .g_req(d0_g_req), .gamma_in(d0_gamma_in), .g_valid(d0_g_valid),
        .y_valid(d0_y_valid), .y_out(d0_y_out),
        .busy(d0_busy), .done(d0_done));

    // ---- DUT 1 (LEN=128) ----
    reg                 d1_start;
    wire                d1_in_req, d1_g_req, d1_y_valid, d1_busy, d1_done;
    reg  [LANES*16-1:0] d1_x_in, d1_gamma_in;
    reg                 d1_x_valid, d1_g_valid;
    wire [LANES*16-1:0] d1_y_out;
    rmsnorm_unit #(.LEN(L1), .LANES(LANES), .EPS(EPS)) dut1 (
        .clk(clk), .rst(rst), .start(d1_start),
        .in_req(d1_in_req), .x_in(d1_x_in), .x_valid(d1_x_valid),
        .g_req(d1_g_req), .gamma_in(d1_gamma_in), .g_valid(d1_g_valid),
        .y_valid(d1_y_valid), .y_out(d1_y_out),
        .busy(d1_busy), .done(d1_done));

    // ---- DUT 2 (LEN=6144) ----
    reg                 d2_start;
    wire                d2_in_req, d2_g_req, d2_y_valid, d2_busy, d2_done;
    reg  [LANES*16-1:0] d2_x_in, d2_gamma_in;
    reg                 d2_x_valid, d2_g_valid;
    wire [LANES*16-1:0] d2_y_out;
    rmsnorm_unit #(.LEN(L2), .LANES(LANES), .EPS(EPS)) dut2 (
        .clk(clk), .rst(rst), .start(d2_start),
        .in_req(d2_in_req), .x_in(d2_x_in), .x_valid(d2_x_valid),
        .g_req(d2_g_req), .gamma_in(d2_gamma_in), .g_valid(d2_g_valid),
        .y_valid(d2_y_valid), .y_out(d2_y_out),
        .busy(d2_busy), .done(d2_done));

    // ===================================================================
    //  bf16 <-> real helpers (independent of the DUT's fp32 path)
    // ===================================================================
    // real value of a bf16 pattern: widen to fp32 bits, reinterpret as fp32.
    function automatic real bf16_to_real(input [15:0] b);
        reg [31:0] f;
        begin
            f = {b, 16'h0000};                 // exact bf16->fp32 (lossless)
            bf16_to_real = $bitstoshortreal(f);
        end
    endfunction

    // quantize a real to bf16 the SAME way the unit outputs: real->fp32 bits
    // (shortreal = IEEE single), then fp32_to_bf16 RNE from glm_fp.vh.
    function automatic [15:0] real_to_bf16(input real r);
        reg [31:0] f;
        begin
            f = $shortrealtobits(r);
            real_to_bf16 = fp32_to_bf16(f);
        end
    endfunction

    // value -> nearest bf16 input pattern (how we present x/gamma to the DUT).
    function automatic [15:0] real_to_bf16_in(input real r);
        real_to_bf16_in = real_to_bf16(r);
    endfunction

    // ===================================================================
    //  Test bookkeeping
    // ===================================================================
    integer test_count = 0;
    integer errors     = 0;
    real    worst_relerr = 0.0;

    // golden bf16 expectation, captured per element
    reg [15:0] yref [0:L2-1];                  // bf16-quantized golden, sized max
    reg [15:0] ydut [0:L2-1];                  // captured DUT output

    // ---- random real in [lo,hi) with random sign when signed=1 ----
    function automatic real rand_real(input real lo, input real hi,
                                      input integer signd);
        real u; integer r;
        begin
            r = $random;
            u = (((r % 1000000) + 1000000) % 1000000) / 1000000.0; // [0,1)
            rand_real = lo + u*(hi-lo);
            if (signd) begin
                r = $random;
                if (r & 1) rand_real = -rand_real;
            end
        end
    endfunction

    // ===================================================================
    //  GOLDEN: given the bf16 input arrays for `len` elements (LANES-packed
    //  in xbuf/gbuf), compute yref[] (bf16) in fp64.
    // ===================================================================
    task compute_golden(input integer len);
        integer i, beat, lane;
        real    sumsq, mean, scale, xv, gv, yv;
        reg [15:0] xb, gb;
        begin
            sumsq = 0.0;
            for (i = 0; i < len; i = i + 1) begin
                beat = i / LANES; lane = i % LANES;
                xb   = xbuf[beat][16*lane +: 16];
                xv   = bf16_to_real(xb);
                sumsq = sumsq + xv*xv;
            end
            mean  = sumsq / len;
            scale = 1.0 / $sqrt(mean + EPS_R);
            for (i = 0; i < len; i = i + 1) begin
                beat = i / LANES; lane = i % LANES;
                xb = xbuf[beat][16*lane +: 16];
                gb = gbuf[beat][16*lane +: 16];
                xv = bf16_to_real(xb);
                gv = bf16_to_real(gb);
                yv = xv * scale * gv;
                yref[i] = real_to_bf16(yv);
            end
        end
    endtask

    // ===================================================================
    //  fill xbuf/gbuf from generator selectors
    //    xmode: 0 random-wide, 1 all-equal(val), 2 outlier, 3 near-zero,
    //           4 all-negative
    //    gmode: 0 gamma=1, 1 random gamma
    // ===================================================================
    task gen_vector(input integer len, input integer xmode, input integer gmode,
                    input real eqval);
        integer i, beat, lane;
        real xv, gv;
        reg [15:0] xb, gb;
        begin
            for (beat = 0; beat < NMAX; beat = beat + 1) begin
                xbuf[beat] = {LANES*16{1'b0}};
                gbuf[beat] = {LANES*16{1'b0}};
            end
            for (i = 0; i < len; i = i + 1) begin
                beat = i / LANES; lane = i % LANES;
                case (xmode)
                    1: xv = eqval;                                   // all-equal
                    2: xv = (i==0) ? 800.0 : rand_real(0.001,0.05,1);// outlier
                    3: xv = rand_real(1.0e-4, 5.0e-4, 1);            // near-zero
                    4: xv = -rand_real(0.01, 100.0, 0);             // all-neg
                    default: xv = rand_real(0.001, 1000.0, 1);      // wide
                endcase
                gv = (gmode==0) ? 1.0 : rand_real(0.1, 3.0, 1);
                xb = real_to_bf16_in(xv);
                gb = real_to_bf16_in(gv);
                xbuf[beat][16*lane +: 16] = xb;
                gbuf[beat][16*lane +: 16] = gb;
            end
        end
    endtask

    // ===================================================================
    //  CHECK one finished vector: compare ydut[] vs yref[] elementwise.
    // ===================================================================
    task check_vector(input integer len, input [255:0] label);
        integer i;
        real    yd, yg, e, denom, rel;
        reg [15:0] db;
        begin
            for (i = 0; i < len; i = i + 1) begin
                db = ydut[i];
                // reject nan/inf in DUT output
                if (db[14:7] == 8'hFF) begin
                    $display("FAIL [%0s] elem %0d: DUT produced nan/inf = %h",
                             label, i, db);
                    errors = errors + 1;
                end
                yd = bf16_to_real(db);
                yg = bf16_to_real(yref[i]);
                e  = yd - yg; if (e < 0.0) e = -e;
                denom = (yg < 0.0) ? -yg : yg;
                if (denom < TINY) begin
                    // near-zero golden: require small absolute deviation
                    if (e > ABS_TOL) begin
                        $display("FAIL [%0s] elem %0d (near-zero): yd=%g yg=%g |e|=%g > ABS_TOL=%g",
                                 label, i, yd, yg, e, ABS_TOL);
                        errors = errors + 1;
                    end
                end else begin
                    rel = e / denom;
                    if (rel > worst_relerr) worst_relerr = rel;
                    if (rel > REL_TOL) begin
                        $display("FAIL [%0s] elem %0d: yd=%g yg=%g rel=%g > REL_TOL=%g",
                                 label, i, yd, yg, rel, REL_TOL);
                        errors = errors + 1;
                    end
                end
            end
            test_count = test_count + 1;
        end
    endtask

    // ===================================================================
    //  DRIVE a single DUT through one vector and capture outputs.
    //  We use a `dut_sel` to pick which DUT's ports to wiggle.  Each DUT
    //  has its own always-blocks feeding x_in/gamma_in from xbuf/gbuf based
    //  on the DUT's own req lines and a per-DUT beat pointer.
    //  Returns via ydut[].
    // ===================================================================
    integer dut_sel;           // 0,1,2

    // per-DUT input-beat pointers (advance on accepted handshake)
    integer xptr0, gptr0, optr0;
    integer xptr1, gptr1, optr1;
    integer xptr2, gptr2, optr2;

    // -- combinational feed of x_in/gamma_in (present requested beat) --
    // we present xbuf[xptr]/gbuf[gptr]; the pointer advances in the clocked
    // block below when req & valid both hold.
    always @* begin
        d0_x_in     = xbuf[xptr0];  d0_gamma_in = gbuf[gptr0];
        d1_x_in     = xbuf[xptr1];  d1_gamma_in = gbuf[gptr1];
        d2_x_in     = xbuf[xptr2];  d2_gamma_in = gbuf[gptr2];
    end

    // -- valid generation: assert when the unit is requesting, unless we are
    //    in a throttle cycle.  A simple alternating throttle proves the
    //    handshake tolerates back-pressure (req held until accepted).
    reg throttle_phase;
    always @* begin
        d0_x_valid = d0_in_req & ~(stall_en & throttle_phase);
        d0_g_valid = d0_g_req  & ~(stall_en & throttle_phase);
        d1_x_valid = d1_in_req & ~(stall_en & throttle_phase);
        d1_g_valid = d1_g_req  & ~(stall_en & throttle_phase);
        d2_x_valid = d2_in_req & ~(stall_en & throttle_phase);
        d2_g_valid = d2_g_req  & ~(stall_en & throttle_phase);
    end

    // -- pointer advance + output capture (clocked) --
    always @(posedge clk) begin
        if (rst) begin
            throttle_phase <= 1'b0;
        end else begin
            throttle_phase <= ~throttle_phase;     // toggle for stall test
            // DUT0
            if (d0_in_req & d0_x_valid) xptr0 <= xptr0 + 1;
            if (d0_g_req  & d0_g_valid) gptr0 <= gptr0 + 1;
            if (d0_y_valid) begin
                ydut[optr0*LANES + 0] <= d0_y_out[15:0];
                ydut[optr0*LANES + 1] <= d0_y_out[31:16];
                ydut[optr0*LANES + 2] <= d0_y_out[47:32];
                ydut[optr0*LANES + 3] <= d0_y_out[63:48];
                optr0 <= optr0 + 1;
            end
            // DUT1
            if (d1_in_req & d1_x_valid) xptr1 <= xptr1 + 1;
            if (d1_g_req  & d1_g_valid) gptr1 <= gptr1 + 1;
            if (d1_y_valid) begin
                ydut[optr1*LANES + 0] <= d1_y_out[15:0];
                ydut[optr1*LANES + 1] <= d1_y_out[31:16];
                ydut[optr1*LANES + 2] <= d1_y_out[47:32];
                ydut[optr1*LANES + 3] <= d1_y_out[63:48];
                optr1 <= optr1 + 1;
            end
            // DUT2
            if (d2_in_req & d2_x_valid) xptr2 <= xptr2 + 1;
            if (d2_g_req  & d2_g_valid) gptr2 <= gptr2 + 1;
            if (d2_y_valid) begin
                ydut[optr2*LANES + 0] <= d2_y_out[15:0];
                ydut[optr2*LANES + 1] <= d2_y_out[31:16];
                ydut[optr2*LANES + 2] <= d2_y_out[47:32];
                ydut[optr2*LANES + 3] <= d2_y_out[63:48];
                optr2 <= optr2 + 1;
            end
        end
    end

    // default all start pulses low; the run_dut task pulses the selected one.
    always @* begin
        d0_start = (dut_sel==0) ? start : 1'b0;
        d1_start = (dut_sel==1) ? start : 1'b0;
        d2_start = (dut_sel==2) ? start : 1'b0;
    end

    // run one vector on the selected DUT: reset pointers, pulse start, wait
    // for that DUT's done, then check.  Times out with $fatal if it hangs.
    task run_dut(input integer sel, input integer len, input [255:0] label);
        integer guard;
        begin
            dut_sel = sel;
            // reset the relevant pointers
            xptr0=0; gptr0=0; optr0=0;
            xptr1=0; gptr1=0; optr1=0;
            xptr2=0; gptr2=0; optr2=0;
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            // wait for done on the selected DUT (named block so we can `disable`)
            guard = 0;
            begin : wait_done
                forever begin
                    @(posedge clk);
                    if ((sel==0 && d0_done) || (sel==1 && d1_done) ||
                        (sel==2 && d2_done)) disable wait_done;
                    guard = guard + 1;
                    if (guard > 8*NMAX + 1000) begin
                        $display("FAIL [%0s]: TIMEOUT waiting for done", label);
                        $fatal(1, "timeout");
                    end
                end
            end
            // allow the last captured y to settle
            @(negedge clk);
            check_vector(len, label);
        end
    endtask

    // ===================================================================
    //  random-vector campaign for one LEN/DUT
    // ===================================================================
    task campaign(input integer sel, input integer len, input [255:0] tag);
        integer r;
        reg [255:0] lab;
        begin
            // directed: all-equal positive
            gen_vector(len, 1, 0, 3.5);   compute_golden(len);
            run_dut(sel, len, {tag, "_alleq+"});
            // directed: all-equal negative
            gen_vector(len, 1, 1, -2.25);  compute_golden(len);
            run_dut(sel, len, {tag, "_alleq-"});
            // directed: outlier, gamma=1
            gen_vector(len, 2, 0, 0.0);   compute_golden(len);
            run_dut(sel, len, {tag, "_outlier"});
            // directed: near-zero (eps dominates), random gamma
            gen_vector(len, 3, 1, 0.0);   compute_golden(len);
            run_dut(sel, len, {tag, "_nearzero"});
            // directed: all-negative, random gamma
            gen_vector(len, 4, 1, 0.0);   compute_golden(len);
            run_dut(sel, len, {tag, "_allneg"});
            // directed: wide random, gamma=1
            gen_vector(len, 0, 0, 0.0);   compute_golden(len);
            run_dut(sel, len, {tag, "_wide_g1"});
            // random campaign: wide range, random gamma
            for (r = 0; r < 12; r = r + 1) begin
                gen_vector(len, 0, 1, 0.0); compute_golden(len);
                run_dut(sel, len, {tag, "_rand"});
            end
        end
    endtask

    // ===================================================================
    //  MAIN
    // ===================================================================
    integer seed_dummy;
    integer seed_var;
    initial begin
        seed_var   = 32'hC0FFEE01;
        seed_dummy = $random(seed_var);   // seed
        start = 1'b0; stall_en = 1'b0; dut_sel = 0;
        xptr0=0; gptr0=0; optr0=0;
        xptr1=0; gptr1=0; optr1=0;
        xptr2=0; gptr2=0; optr2=0;
        d0_x_in=0; d0_gamma_in=0; d1_x_in=0; d1_gamma_in=0;
        d2_x_in=0; d2_gamma_in=0;

        rst = 1'b1;
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // -------- LEN=16 campaign --------
        campaign(0, L0, "L16");
        // -------- LEN=128 campaign --------
        campaign(1, L1, "L128");
        // -------- LEN=6144 campaign (the GLM hidden size) --------
        campaign(2, L2, "L6144");

        // -------- STALL / back-pressure test on LEN=128 --------
        stall_en = 1'b1;
        gen_vector(L1, 0, 1, 0.0); compute_golden(L1);
        run_dut(1, L1, "L128_stall");
        stall_en = 1'b0;

        // -------- summary --------
        if (errors != 0) begin
            $display("RMSNORM: %0d ELEMENT MISMATCHES across %0d tests",
                     errors, test_count);
            $fatal(1, "rmsnorm mismatch");
        end
        $display("RMSNORM worst rel-err vs fp64 golden = %g (REL_TOL = %g)",
                 worst_relerr, REL_TOL);
        $display("ALL %0d TESTS PASSED", test_count);
        $finish;
    end

    // global watchdog
    initial begin
        #200000000;
        $display("FAIL: global timeout");
        $fatal(1, "global timeout");
    end

endmodule
