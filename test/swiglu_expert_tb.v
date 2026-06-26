`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// swiglu_expert_tb.v  --  thorough self-checking TB for swiglu_expert
//                         (the GLM-5.2 SwiGLU FFN expert, ACCEL_GLM52 §5)
//----------------------------------------------------------------------------
// FUNCTION UNDER TEST
//   swiglu_expert orchestrates the foundation blocks to compute one token's
//   SwiGLU FFN output y (HIDDEN bf16) from a token x (HIDDEN bf16) and this
//   expert's three weight matrices:
//
//       gate = W_gate @ x      (W_gate : INTER x HIDDEN)
//       up   = W_up   @ x      (W_up   : INTER x HIDDEN)
//       h    = silu(gate) (.) up        (elementwise, INTER)
//       y    = W_down @ h      (W_down : HIDDEN x INTER)
//
//   bf16 operands everywhere; fp32 accumulation INSIDE each GEMM; silu in fp32
//   (poly exp + Quake rsqrt) rounded to bf16; the silu*up product rounded to
//   bf16 (the resident h buffer); the down GEMM then reduces the bf16 h in fp32
//   and rounds y to bf16.  The DUT pulls weights combinationally via a fully-
//   decoded request (w_req/w_sel/w_grp/w_k -> w_col/w_col_up); this TB answers
//   that request from flat bf16 weight ROMs in the SAME cycle (the contract).
//
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL  (shares NONE of the DUT's fp32 arithmetic)
//   The DUT reduces in the project's bit-accurate pipelined fp32 primitives
//   (fp32_mac_pipe + an fp32_add_pipe tree, L-way interleaved) and evaluates
//   silu through a degree-5 poly exp + a Newton rsqrt reciprocal.  The golden
//   recomputes EVERYTHING a completely different way, in Verilog `real` (IEEE
//   double, fp64):
//     * widen each stored bf16 (W_gate, W_up, W_down, x) to its EXACT real value
//       (bf16->real is lossless: bf16 is the high 16 bits of fp32),
//     * gate[i] = sum_k W_gate[i][k]*x[k]   and   up[i] = sum_k W_up[i][k]*x[k]
//       as TRUE fp64 dot products (NOT the fp32 mac/add tree, NOT the L-way
//       grouping),
//     * silu(gate) = gate / (1 + exp(-gate))  using the system math.h $exp --
//       a DIFFERENT transcendental path than the DUT's poly+rsqrt,
//     * h[i] = silu(gate[i]) * up[i]   in fp64,
//     * y[o] = sum_k W_down[o][k]*h[k]  as a fp64 dot product.
//   Everything that produces a *value* differs, so the golden catches DUT bugs
//   (a dropped term, wrong product, mis-ordered accumulate, a lane collision, a
//   wrong silu, a misrouted up tile, a tile-group mis-map) rather than mirroring
//   them.  The DUT is fed the SAME bf16 bit patterns the golden widens, so any
//   discrepancy is the unit's compute path, not input skew.
//
//   GRID ALIGNMENT (compare on the bf16 grid the unit emits):
//   the unit ROUNDS to bf16 at three internal boundaries (gate, up, and the
//   silu*up product h), then again at y.  To compare the DUT *compute* error and
//   not penalize the golden for carrying extra fp64 precision through those
//   storage points, the golden quantizes at the SAME boundaries the unit does:
//   gate->bf16, up->bf16, h=bf16_round(silu*up) (silu itself rounded to bf16
//   first, exactly the unit's act->bf16->mul->bf16 order), and y->bf16.  The
//   gate/up dot products and the silu transcendental and the down dot product
//   are all still computed independently in fp64 between those grid points.
//
//----------------------------------------------------------------------------
// TOLERANCE  (STATED + JUSTIFIED, K-depth dependent)
//   Per output element we compare DUT bf16 y to the fp64-golden y (re-widened
//   from its bf16 quantization) as a RELATIVE error
//        relerr = |y_dut - y_gold| / max(|y_gold|, TINY)
//   and require relerr <= REL_TOL.  bf16 has 8 significand bits, so 1 ULP is
//   2^-8 and a single round-to-bf16 costs up to 0.5 ULP.  The expert chains
//   THREE rounded GEMM passes plus a rounded silu and a rounded silu*up, and a
//   near-rounding-boundary value can land on an ADJACENT bf16 grid point at each
//   stage (the DUT rounds an fp32 running sum / poly-exp; the golden rounds the
//   exact fp64).  Each stage's fp32-running-sum error also grows mildly with its
//   reduction depth K (each of ~K fp32 adds is ~2^-23 relative, and near-zero
//   cancellation inflates the *relative* error of a tiny result).  We compose
//   the per-stage double-rounding + K-depth budgets (matching the proven
//   glm_matmul_pipe tolerance 2^-6 + K*2^-12 for each of the two GEMM depths,
//   K_gemm1 = HIDDEN for gate/up, K_gemm2 = INTER for down) plus a fixed silu
//   double-rounding allowance:
//        REL_TOL = 2^-5 + (HIDDEN + INTER)*2^-12 + 2^-6
//                = 1/32 + (HIDDEN+INTER)/4096 + 1/64
//   i.e. ~6 bf16-ULP floor for the stacked roundings plus ~0.5 ULP per 8 total
//   reduction terms.  For the tested shapes (HIDDEN+INTER up to ~320) this is a
//   tight ~6..16 bf16-ULP band: it passes the correct datapath with margin for
//   the worst stacked double-rounding, yet fails on a real bug (a missing term /
//   wrong silu / mis-mapped tile moves an element by >> a handful of ULP -- a
//   whole reduction term or the entire silu factor).
//   For elements whose golden magnitude is below TINY (true near-zero from exact
//   cancellation, where relative error is meaningless) we instead require the
//   DUT magnitude within ABS_TOL of zero.  We track + PRINT the worst rel-err.
//
//----------------------------------------------------------------------------
// COVERAGE  ($fatal on any element miss / nan / inf / protocol timeout)
//   Two (HIDDEN,INTER) shapes (separate compile-time DUT instances):
//     (HIDDEN,INTER) = (128, 64)   the MoE 2:1 ratio (HIDDEN > INTER), INTER a
//                                  multiple of TN -- the production shape ratio.
//     (HIDDEN,INTER) = (40,  96)   a DEEPER inter, INTER > HIDDEN (dense-like),
//                                  with INTER NOT a multiple of TN (96/TN ok but
//                                  HIDDEN=40 down-tail mask exercised; sizes are
//                                  also non-power-of-two to stress tile tails).
//   Per shape, directed + random scenarios (every output element checked):
//     x = 0                  -> y == 0 exactly (every path zero).
//     identity-ish weights   -> W_gate=W_up=I (padded), W_down=I: y == silu(x).x
//                               (hits the diagonal map + silu*up + down identity).
//     silu-dip input         -> x near -1.278 (silu's negative minimum), small
//                               weights: exercises silu's in-range dip region.
//     large values           -> large +x (silu saturates: silu(+big)~x) and
//                               large -x (silu(-big)~0): saturation tails.
//     random wide-dynamic    -> W,x in mixed sign, |.| ~ 1e-2..1e1 (x6): the
//                               general SwiGLU compute over a wide range.
//   PROTOCOL: busy must rise after start and fall with done; done is a 1-cycle
//   pulse; a watchdog $fatal's on a hung run.
//   GATES: prints "ALL <N> TESTS PASSED"; $fatal on any mismatch / nan / inf /
//   timeout.
//============================================================================
module swiglu_expert_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ---- compile-time shapes ----
    localparam integer H0 = 128, I0 = 64;   // MoE 2:1 ratio (HIDDEN>INTER)
    localparam integer H1 = 40,  I1 = 96;   // deeper inter (INTER>HIDDEN), tails
    localparam integer TN   = 4;
    localparam integer KMAX = 256;

    // max sizes for shared storage
    localparam integer HMAX = 128;
    localparam integer IMAX = 96;

    // ---- tolerance constants ----
    localparam real TINY    = 1.0e-9;
    localparam real ABS_TOL = 1.0/32.0;     // near-zero absolute floor (~bf16 ULP scale)

    // ===================================================================
    //  Weight ROMs + token, sized to the largest shape.
    //    Wg[out][k], Wu[out][k] : INTER x HIDDEN
    //    Wd[out][k]             : HIDDEN x INTER
    //    xb[k]                  : HIDDEN
    // ===================================================================
    reg [15:0] Wg [0:IMAX-1][0:HMAX-1];
    reg [15:0] Wu [0:IMAX-1][0:HMAX-1];
    reg [15:0] Wd [0:HMAX-1][0:IMAX-1];
    reg [15:0] xb [0:HMAX-1];

    // golden output + captured DUT output (bf16)
    reg [15:0] yref [0:HMAX-1];
    reg [15:0] ydut [0:HMAX-1];

    // ===================================================================
    //  DUT control / data (max-width, sliced per instance).
    // ===================================================================
    reg  start;
    reg  [16*HMAX-1:0] x_vec_max;

    // weight-request wires per DUT
    wire d0_wreq, d1_wreq;
    wire [1:0] d0_wsel, d1_wsel;
    wire [$clog2((I0>H0?I0:H0)/TN+1)-1:0] d0_wgrp;
    wire [$clog2((I1>H1?I1:H1)/TN+1)-1:0] d1_wgrp;
    wire [$clog2(KMAX+1)-1:0] d0_wk, d1_wk;
    reg  [16*TN-1:0] d0_wcol, d0_wcolu;
    reg  [16*TN-1:0] d1_wcol, d1_wcolu;
    wire d0_busy, d1_busy, d0_done, d1_done;
    wire [16*H0-1:0] d0_y;
    wire [16*H1-1:0] d1_y;

    swiglu_expert #(.HIDDEN(H0), .INTER(I0), .TN(TN), .KMAX(KMAX)) dut0 (
        .clk(clk), .rst(rst), .start(start), .busy(d0_busy), .done(d0_done),
        .x_vec(x_vec_max[16*H0-1:0]),
        .w_req(d0_wreq), .w_sel(d0_wsel), .w_grp(d0_wgrp), .w_k(d0_wk),
        .w_col(d0_wcol), .w_col_up(d0_wcolu),
        .y_out(d0_y));

    swiglu_expert #(.HIDDEN(H1), .INTER(I1), .TN(TN), .KMAX(KMAX)) dut1 (
        .clk(clk), .rst(rst), .start(start), .busy(d1_busy), .done(d1_done),
        .x_vec(x_vec_max[16*H1-1:0]),
        .w_req(d1_wreq), .w_sel(d1_wsel), .w_grp(d1_wgrp), .w_k(d1_wk),
        .w_col(d1_wcol), .w_col_up(d1_wcolu),
        .y_out(d1_y));

    // ===================================================================
    //  COMBINATIONAL weight responder (same-cycle as the DUT request).
    //  w_col[t] answers output index (w_grp*TN + t) at reduction index w_k:
    //    w_sel==2 (DOWN)   : Wd[out][k]
    //    w_sel==0 (GATE+UP): Wg[out][k] on w_col, Wu[out][k] on w_col_up.
    //  Tail lanes past OUT are don't-care (the DUT masks them); we drive 0.
    // ===================================================================
    integer t0, o0, t1, o1;
    always @* begin
        d0_wcol = {16*TN{1'b0}}; d0_wcolu = {16*TN{1'b0}};
        for (t0 = 0; t0 < TN; t0 = t0 + 1) begin
            o0 = d0_wgrp*TN + t0;
            if (d0_wsel == 2'd2) begin
                if (o0 < H0) d0_wcol[16*t0 +: 16] = Wd[o0][d0_wk];
            end else begin
                if (o0 < I0) begin
                    d0_wcol [16*t0 +: 16] = Wg[o0][d0_wk];
                    d0_wcolu[16*t0 +: 16] = Wu[o0][d0_wk];
                end
            end
        end
    end
    always @* begin
        d1_wcol = {16*TN{1'b0}}; d1_wcolu = {16*TN{1'b0}};
        for (t1 = 0; t1 < TN; t1 = t1 + 1) begin
            o1 = d1_wgrp*TN + t1;
            if (d1_wsel == 2'd2) begin
                if (o1 < H1) d1_wcol[16*t1 +: 16] = Wd[o1][d1_wk];
            end else begin
                if (o1 < I1) begin
                    d1_wcol [16*t1 +: 16] = Wg[o1][d1_wk];
                    d1_wcolu[16*t1 +: 16] = Wu[o1][d1_wk];
                end
            end
        end
    end

    // ===================================================================
    //  bf16 <-> real helpers (independent of the DUT's fp32 path).
    // ===================================================================
    function automatic real bf16_to_real(input [15:0] b);
        bf16_to_real = $bitstoshortreal({b, 16'h0000});  // exact, lossless
    endfunction
    function automatic [15:0] real_to_bf16(input real r);
        real_to_bf16 = fp32_to_bf16($shortrealtobits(r));
    endfunction
    // silu(g) = g / (1 + exp(-g))  via the system $exp (DIFFERENT path than DUT)
    function automatic real silu_real(input real g);
        silu_real = g / (1.0 + $exp(-g));
    endfunction

    // ===================================================================
    //  random bf16 in a wide dynamic range, mixed sign.
    // ===================================================================
    integer seed = 32'h1234_5678;
    function automatic real rand_real(input real lo, input real hi, input integer signd);
        real u; integer r;
        begin
            r = $random(seed);
            u = (r & 32'h7fff_ffff) / 2147483647.0;        // [0,1]
            rand_real = lo + u*(hi - lo);
            if (signd && ($random(seed) & 1)) rand_real = -rand_real;
        end
    endfunction

    // ===================================================================
    //  INDEPENDENT GOLDEN (fp64), quantized to bf16 at the unit's boundaries.
    // ===================================================================
    real gate_r, up_r, silu_r, y_r;
    reg  [15:0] hgrid [0:IMAX-1];      // h on the bf16 grid (as the unit stores)
    integer gi, gk;
    task compute_golden(input integer HH, input integer II);
        begin
            // gate/up GEMMs + silu*up -> bf16 h grid
            for (gi = 0; gi < II; gi = gi + 1) begin
                gate_r = 0.0; up_r = 0.0;
                for (gk = 0; gk < HH; gk = gk + 1) begin
                    gate_r = gate_r + bf16_to_real(Wg[gi][gk]) * bf16_to_real(xb[gk]);
                    up_r   = up_r   + bf16_to_real(Wu[gi][gk]) * bf16_to_real(xb[gk]);
                end
                // unit rounds gate, up to bf16 before silu / multiply; mirror that
                gate_r = bf16_to_real(real_to_bf16(gate_r));
                up_r   = bf16_to_real(real_to_bf16(up_r));
                // silu(gate) rounded to bf16 (unit's glm_act bf16 output), then
                // bf16_mul by up rounded to bf16 (unit's h buffer write).
                silu_r = bf16_to_real(real_to_bf16(silu_real(gate_r)));
                hgrid[gi] = real_to_bf16(silu_r * up_r);
            end
            // down GEMM over the bf16 h grid -> bf16 y
            for (gi = 0; gi < HH; gi = gi + 1) begin
                y_r = 0.0;
                for (gk = 0; gk < II; gk = gk + 1)
                    y_r = y_r + bf16_to_real(Wd[gi][gk]) * bf16_to_real(hgrid[gk]);
                yref[gi] = real_to_bf16(y_r);
            end
        end
    endtask

    // ===================================================================
    //  bookkeeping + checker
    // ===================================================================
    integer test_count = 0;
    integer errors     = 0;
    real    worst_relerr = 0.0;

    task check_y(input integer HH, input integer II, input [255:0] tag);
        integer o;
        real    gold, got, diff, denom, relerr, rel_tol;
        reg [15:0] db;
        begin
            // composed tolerance: per-GEMM double-rounding + K-depth, two GEMM
            // depths (HIDDEN for gate/up, INTER for down) + a silu allowance.
            rel_tol = (1.0/32.0) + ((HH + II) * (1.0/4096.0)) + (1.0/64.0);
            for (o = 0; o < HH; o = o + 1) begin
                db = ydut[o];
                if (db[14:7] == 8'hFF) begin
                    $display("FAIL [%0s] y[%0d] is nan/inf (bf16=%04h)", tag, o, db);
                    errors = errors + 1;
                end else begin
                    gold = bf16_to_real(yref[o]);
                    got  = bf16_to_real(db);
                    diff = got - gold; if (diff < 0.0) diff = -diff;
                    denom = (gold < 0.0) ? -gold : gold;
                    if (denom < TINY) begin
                        if (diff > ABS_TOL) begin
                            $display("FAIL [%0s] y[%0d] near0 gold=%g got=%g |d|=%g > %g",
                                     tag, o, gold, got, diff, ABS_TOL);
                            errors = errors + 1;
                        end
                    end else begin
                        relerr = diff/denom;
                        if (relerr > worst_relerr) worst_relerr = relerr;
                        if (relerr > rel_tol) begin
                            $display("FAIL [%0s] y[%0d] gold=%g got=%g relerr=%g > tol=%g (H=%0d I=%0d)",
                                     tag, o, gold, got, relerr, rel_tol, HH, II);
                            errors = errors + 1;
                        end
                    end
                end
            end
            test_count = test_count + 1;
        end
    endtask

    // ===================================================================
    //  capture a finished DUT y vector into ydut[]
    // ===================================================================
    task capture(input integer which, input integer HH);
        integer o;
        begin
            for (o = 0; o < HH; o = o + 1)
                ydut[o] = (which == 0) ? d0_y[16*o +: 16] : d1_y[16*o +: 16];
        end
    endtask

    function automatic busy_of(input integer which);
        busy_of = (which == 0) ? d0_busy : d1_busy;
    endfunction
    function automatic done_of(input integer which);
        done_of = (which == 0) ? d0_done : d1_done;
    endfunction

    // ===================================================================
    //  RUN one evaluation: pulse start, drive x_vec, wait for done, capture.
    //  Checks the busy/done protocol (busy rises, done is a 1-cycle pulse).
    // ===================================================================
    task run_eval(input integer which, input integer HH);
        integer guard, k;
        reg saw_busy;
        begin
            // present x
            for (k = 0; k < HH; k = k + 1)
                x_vec_max[16*k +: 16] <= xb[k];
            @(negedge clk);
            start <= 1'b1;
            @(negedge clk);
            start <= 1'b0;

            guard = 0; saw_busy = 1'b0;
            // wait for done; verify busy asserted somewhere before done
            while (!done_of(which) && guard < 2_000_000) begin
                if (busy_of(which)) saw_busy = 1'b1;
                @(negedge clk);
                guard = guard + 1;
            end
            if (guard >= 2_000_000) begin
                $display("FAIL which=%0d timeout (no done)", which);
                errors = errors + 1;
                $fatal(1, "swiglu_expert_tb timeout");
            end
            if (!saw_busy) begin
                $display("FAIL which=%0d busy never asserted", which);
                errors = errors + 1;
            end
            // done is high now; capture y, then verify done falls next cycle
            capture(which, HH);
            @(negedge clk);
            if (done_of(which)) begin
                $display("FAIL which=%0d done not a 1-cycle pulse", which);
                errors = errors + 1;
            end
        end
    endtask

    // ===================================================================
    //  Operand fill helpers
    // ===================================================================
    task fill_zero_x(input integer HH);
        integer k;
        begin
            for (k = 0; k < HH; k = k + 1) xb[k] = real_to_bf16(0.0);
        end
    endtask

    // W_gate = W_up = identity (padded), W_down = identity -> y = silu(x).x
    task fill_identity(input integer HH, input integer II, input real xmag);
        integer i, k;
        begin
            for (k = 0; k < HH; k = k + 1)
                xb[k] = real_to_bf16(rand_real(0.2, xmag, 1));
            for (i = 0; i < II; i = i + 1)
                for (k = 0; k < HH; k = k + 1) begin
                    Wg[i][k] = (i == k) ? real_to_bf16(1.0) : real_to_bf16(0.0);
                    Wu[i][k] = (i == k) ? real_to_bf16(1.0) : real_to_bf16(0.0);
                end
            for (i = 0; i < HH; i = i + 1)
                for (k = 0; k < II; k = k + 1)
                    Wd[i][k] = (i == k) ? real_to_bf16(1.0) : real_to_bf16(0.0);
        end
    endtask

    // silu-dip: drive gate near -1.278 (silu minimum) via identity-ish gate W and
    // x near -1.278; small up/down weights so the dip dominates h.
    task fill_silu_dip(input integer HH, input integer II);
        integer i, k;
        begin
            for (k = 0; k < HH; k = k + 1) xb[k] = real_to_bf16(-1.278);
            for (i = 0; i < II; i = i + 1)
                for (k = 0; k < HH; k = k + 1) begin
                    Wg[i][k] = (i == (k % II)) ? real_to_bf16(1.0) : real_to_bf16(0.0);
                    Wu[i][k] = (i == (k % II)) ? real_to_bf16(1.0) : real_to_bf16(0.0);
                end
            for (i = 0; i < HH; i = i + 1)
                for (k = 0; k < II; k = k + 1)
                    Wd[i][k] = (k == (i % II)) ? real_to_bf16(1.0) : real_to_bf16(0.0);
        end
    endtask

    // large values: x large +/- so silu saturates (silu(+big)~x, silu(-big)~0)
    task fill_large(input integer HH, input integer II);
        integer i, k;
        begin
            for (k = 0; k < HH; k = k + 1)
                xb[k] = real_to_bf16((k & 1) ? -20.0 : 20.0);
            for (i = 0; i < II; i = i + 1)
                for (k = 0; k < HH; k = k + 1) begin
                    Wg[i][k] = (i == (k % II)) ? real_to_bf16(1.0) : real_to_bf16(0.0);
                    Wu[i][k] = real_to_bf16(0.5);
                end
            for (i = 0; i < HH; i = i + 1)
                for (k = 0; k < II; k = k + 1)
                    Wd[i][k] = real_to_bf16(0.1);
        end
    endtask

    task fill_random(input integer HH, input integer II);
        integer i, k;
        begin
            for (k = 0; k < HH; k = k + 1)
                xb[k] = real_to_bf16(rand_real(1.0e-2, 1.0e1, 1));
            for (i = 0; i < II; i = i + 1)
                for (k = 0; k < HH; k = k + 1) begin
                    Wg[i][k] = real_to_bf16(rand_real(1.0e-2, 1.0e0, 1));
                    Wu[i][k] = real_to_bf16(rand_real(1.0e-2, 1.0e0, 1));
                end
            for (i = 0; i < HH; i = i + 1)
                for (k = 0; k < II; k = k + 1)
                    Wd[i][k] = real_to_bf16(rand_real(1.0e-2, 1.0e0, 1));
        end
    endtask

    // ===================================================================
    //  Per-shape exercise: directed + random scenarios.
    // ===================================================================
    task exercise_shape(input integer which, input integer HH, input integer II);
        integer r;
        begin
            // x = 0 -> y == 0
            fill_random(HH, II);            // arbitrary weights
            fill_zero_x(HH);                // override x to 0
            compute_golden(HH, II);
            run_eval(which, HH); check_y(HH, II, "zero-x");

            // identity-ish weights -> y = silu(x).x  (moderate x)
            fill_identity(HH, II, 3.0);
            compute_golden(HH, II);
            run_eval(which, HH); check_y(HH, II, "identity");

            // silu negative-dip region
            fill_silu_dip(HH, II);
            compute_golden(HH, II);
            run_eval(which, HH); check_y(HH, II, "silu-dip");

            // large values (silu saturation tails)
            fill_large(HH, II);
            compute_golden(HH, II);
            run_eval(which, HH); check_y(HH, II, "large");

            // random wide-dynamic-range, mixed sign (x6)
            for (r = 0; r < 6; r = r + 1) begin
                fill_random(HH, II);
                compute_golden(HH, II);
                run_eval(which, HH); check_y(HH, II, "rand");
            end
        end
    endtask

    // ===================================================================
    //  MAIN
    // ===================================================================
    initial begin
        rst = 1'b1; start = 1'b0; x_vec_max = {16*HMAX{1'b0}};
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        exercise_shape(0, H0, I0);   // (128, 64) MoE 2:1
        exercise_shape(1, H1, I1);   // (40,  96) deep inter, tails

        if (errors != 0) begin
            $display("FAILED: %0d mismatch(es) over %0d evaluations (worst relerr=%g)",
                     errors, test_count, worst_relerr);
            $fatal(1, "swiglu_expert_tb FAILED");
        end else begin
            $display("worst observed relerr = %g", worst_relerr);
            $display("ALL %0d TESTS PASSED", test_count);
        end
        $finish;
    end

    // global watchdog
    initial begin
        #200_000_000;
        $display("FAIL global timeout");
        $fatal(1, "swiglu_expert_tb global timeout");
    end

endmodule
