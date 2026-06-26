`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_act_tb.v  --  self-checking TB for glm_act (SIGMOID + SiLU)      (§5)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL
//   The DUT (glm_act) computes, for each bf16 input x:
//       MODE_SIGMOID : y = sigmoid(x) = 1/(1+exp(-x))
//       MODE_SILU    : y = silu(x)    = x*sigmoid(x)
//   in the project's bit-accurate fp32 primitives (glm_fp.vh) via a degree-5
//   Taylor range-reduced exp, a Quake-seed Newton rsqrt reciprocal, and an
//   fp32 multiply, rounded to bf16 on output.
//
//   This golden shares NONE of that arithmetic.  It recomputes the activations
//   a COMPLETELY DIFFERENT way, in Verilog `real` (IEEE double, fp64):
//     * widen the stored bf16 x to its EXACT real value (bf16->real),
//     * sig = 1.0 / (1.0 + $exp(-x_real))   using the C math-library exp
//       (a true exp, NOT the magic-constant range-reduction + Taylor poly),
//     * y_real = (MODE_SILU) ? x_real*sig : sig,
//     * quantize y_real to bf16 the SAME way the unit emits bf16 (real->fp32
//       bits via $shortrealtobits, then fp32_to_bf16 RNE).
//   So the only shared step is the final fp32->bf16 RNE pack (the unit's
//   DEFINED output format -- both DUT and golden must land on the same bf16
//   grid).  Everything that produces the *value* differs (math-lib exp + real
//   division vs. Taylor exp + Newton rsqrt), so the golden catches DUT
//   arithmetic bugs instead of mirroring them.
//
//   The DUT is fed bf16-quantized x (the same bit pattern the golden widens),
//   so any discrepancy is the unit's fp32 transcendental path, not input
//   quantization skew.
//
//----------------------------------------------------------------------------
// SATURATION-AWARE GOLDEN
//   The DUT clamps the SIGMOID exp argument to [-X_SAT,X_SAT] (X_SAT=16) and
//   sanitizes inf/nan to +/-X_SAT.  To compare like-for-like, the golden applies
//   the SAME saturation policy to its OWN argument before the math-lib exp:
//     * inf/nan x          -> +/-X_SAT  (nan -> +X_SAT, matching sanitize_x),
//     * |x| > X_SAT (sig)  -> clamp the sigmoid argument to +/-X_SAT,
//     * SiLU linear factor uses the RAW (sanitized, UNclamped) x.
//   This mirrors the documented module behaviour (silu(+big)~x, silu(-big)~0),
//   so the tolerance need only absorb compute + rounding error, not a
//   modelling mismatch on the rails.
//
//----------------------------------------------------------------------------
// TOLERANCE  (STATED + JUSTIFIED)
//   Per element we compare the DUT bf16 output to the fp64-golden value
//   (re-widened from its bf16 quantization) as an ABSOLUTE error:
//        abserr = |y_dut_real - y_gold_real|
//   and (for non-near-zero golden) also a RELATIVE error
//        relerr = abserr / |y_gold_real|.
//   We PASS an element if EITHER the absolute OR the relative bound holds:
//        abserr <= ABS_TOL = 2^-7   (~0.0078)   OR
//        relerr <= REL_TOL = 2^-6   (1/64 ~0.0156).
//
//   Why these:
//     * bf16 carries 8 significand bits (7 stored + implicit) => one bf16 ULP
//       is 2^-8 of the value; rounding the final result to bf16 alone costs up
//       to 0.5 ULP = 2^-9.
//     * The DUT exp poly is < 2^-18 and the rsqrt reciprocal < 2^-21 (per the
//       module/glm_fp analysis), both far below a bf16 ULP.
//     * The DOMINANT effect is DOUBLE ROUNDING: the golden rounds the exact
//       fp64 value to bf16, while the DUT rounds an fp32 value whose own
//       rounding can tip it across the bf16 halfway point => the two can land
//       on ADJACENT bf16 codes, ~1 bf16 ULP apart.
//     * sigmoid in [0,1]: a 1-ULP gap near the value is <= 2^-8 absolute, so
//       ABS_TOL=2^-7 (2 ULP at value 1, more headroom for smaller values)
//       covers it while still failing a real bug (a wrong sign or dropped term
//       moves the output by >> 2 ULP).
//     * silu spans a wide magnitude (|x| up to 16); there the RELATIVE bound
//       REL_TOL=2^-6 = 4 bf16 ULP governs, with the absolute bound catching the
//       tiny-magnitude dip region.  Either-of bound = the looser, principled
//       envelope for the dual-rounding worst case.
//   For golden magnitudes below TINY (true near-zero, where relative error is
//   meaningless) we require only the absolute bound.
//
//   We also track + PRINT the worst observed abs-err and rel-err.
//
//----------------------------------------------------------------------------
// COVERAGE  (each element checked within tolerance; $fatal on any miss)
//   BOTH modes (MODE_SIGMOID dut, MODE_SILU dut), LANES-packed.  x is swept:
//     * large negative  (sigmoid -> 0,  silu -> 0),
//     * large positive  (sigmoid -> 1,  silu -> x),
//     * around 0        (sigmoid ~ 0.5, silu ~ 0),
//     * the silu MINIMUM region near x ~ -1.278 (silu non-monotonic dip),
//     * a fine dense sweep across [-X_SAT, X_SAT],
//     * directed x = 0, +/-0, +/-inf, qnan, tiny subnormal-ish magnitudes,
//       the exact +/-X_SAT rails and just inside/outside them,
//     * many RANDOM bf16 x of mixed sign and WIDE dynamic range
//       (|x| ~ 1e-6 .. 1e3).
//   Every element is checked; prints "ALL <N> TESTS PASSED"; $fatal on any
//   element mismatch (a "TEST" = one checked element).
//============================================================================
module glm_act_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    localparam integer LANES = 4;
    localparam [31:0]  X_SAT = 32'h41800000;   // 16.0 fp32 (the DUT default)
    localparam real    XSAT_R = 16.0;
    localparam integer LAT   = 5;

    // tolerances (see header)
    localparam real ABS_TOL = 1.0/128.0;       // 2^-7
    localparam real REL_TOL = 1.0/64.0;        // 2^-6
    localparam real TINY    = 1.0e-9;

    // ===================================================================
    //  Two DUTs: one SIGMOID (MODE=0), one SiLU (MODE=1).  Both are fed the
    //  SAME x stream each beat; we capture both outputs LAT cycles later.
    // ===================================================================
    reg                  in_valid;
    reg  [LANES*16-1:0]  x_in;

    wire                 sg_out_valid;
    wire [LANES*16-1:0]  sg_y_out;
    glm_act #(.MODE(0), .LANES(LANES), .X_SAT(X_SAT)) dut_sig (
        .clk(clk), .rst(rst), .in_valid(in_valid), .x_in(x_in),
        .out_valid(sg_out_valid), .y_out(sg_y_out));

    wire                 sl_out_valid;
    wire [LANES*16-1:0]  sl_y_out;
    glm_act #(.MODE(1), .LANES(LANES), .X_SAT(X_SAT)) dut_silu (
        .clk(clk), .rst(rst), .in_valid(in_valid), .x_in(x_in),
        .out_valid(sl_out_valid), .y_out(sl_y_out));

    // ===================================================================
    //  bf16 <-> real helpers (independent of the DUT fp32 path)
    // ===================================================================
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

    // value -> nearest bf16 input pattern.
    function automatic [15:0] real_to_bf16_in(input real r);
        real_to_bf16_in = real_to_bf16(r);
    endfunction

    // ===================================================================
    //  INDEPENDENT GOLDEN: bf16 x pattern -> bf16 expected output, applying
    //  the DUT's documented saturation/sanitize policy, then math-lib exp.
    //    mode: 0 = SIGMOID, 1 = SiLU
    // ===================================================================
    function automatic [15:0] golden(input [15:0] xb, input integer mode);
        reg [31:0] f;
        real x_raw;        // sanitized (inf/nan -> +/-X_SAT) raw value
        real x_sig;        // clamped-to-[-X_SAT,X_SAT] sigmoid argument
        real sig, y;
        reg  is_specexp;   // exp==0xFF -> inf/nan
        begin
            f = {xb, 16'h0000};
            is_specexp = (f[30:23] == 8'hFF);
            if (is_specexp) begin
                // sanitize_x: inf/nan -> +/-X_SAT (sign from input; nan -> +).
                x_raw = (f[31]) ? -XSAT_R : XSAT_R;
            end else begin
                x_raw = $bitstoshortreal(f);
            end
            // clamp_xsat for the sigmoid path
            x_sig = x_raw;
            if (x_sig >  XSAT_R) x_sig =  XSAT_R;
            if (x_sig < -XSAT_R) x_sig = -XSAT_R;
            // true sigmoid via math-lib exp (independent of the DUT's poly)
            sig = 1.0 / (1.0 + $exp(-x_sig));
            // silu uses the RAW (unclamped) linear factor
            y   = (mode == 1) ? (x_raw * sig) : sig;
            golden = real_to_bf16(y);
        end
    endfunction

    // ===================================================================
    //  Test bookkeeping
    // ===================================================================
    integer test_count   = 0;
    integer errors       = 0;
    real    worst_abserr = 0.0;
    real    worst_relerr = 0.0;
    integer worst_mode   = 0;
    reg [15:0] worst_x   = 16'h0;

    // ---- random real in [lo,hi) with optional random sign ----
    integer rng;
    function automatic real rand_real(input real lo, input real hi,
                                      input integer signd);
        real u; integer r;
        begin
            r = $random(rng);
            u = (((r % 1000000) + 1000000) % 1000000) / 1000000.0; // [0,1)
            rand_real = lo + u*(hi-lo);
            if (signd) begin
                r = $random(rng);
                if (r & 1) rand_real = -rand_real;
            end
        end
    endfunction

    // ===================================================================
    //  Stimulus / capture queue.
    //  Pure feed-forward DUT: we push a stream of bf16 x patterns, asserting
    //  in_valid each beat, and capture both DUTs' y_out when out_valid fires.
    //  We store the input pattern per accepted beat and the captured outputs
    //  so the checker can recompute the golden for each.
    // ===================================================================
    localparam integer MAXN = 20000;
    reg  [15:0] xq    [0:MAXN-1];     // input bf16 per element (flattened)
    reg  [15:0] sgq   [0:MAXN-1];     // captured sigmoid DUT output
    reg  [15:0] slq   [0:MAXN-1];     // captured silu    DUT output
    integer     n_in;                 // elements pushed
    integer     n_out_sg;             // elements captured (sigmoid)
    integer     n_out_sl;             // elements captured (silu)

    // capture on out_valid (both DUTs share LAT, fire together)
    always @(posedge clk) begin
        if (!rst) begin
            if (sg_out_valid) begin
                sgq[n_out_sg + 0] <= sg_y_out[15:0];
                sgq[n_out_sg + 1] <= sg_y_out[31:16];
                sgq[n_out_sg + 2] <= sg_y_out[47:32];
                sgq[n_out_sg + 3] <= sg_y_out[63:48];
                n_out_sg <= n_out_sg + LANES;
            end
            if (sl_out_valid) begin
                slq[n_out_sl + 0] <= sl_y_out[15:0];
                slq[n_out_sl + 1] <= sl_y_out[31:16];
                slq[n_out_sl + 2] <= sl_y_out[47:32];
                slq[n_out_sl + 3] <= sl_y_out[63:48];
                n_out_sl <= n_out_sl + LANES;
            end
        end
    end

    // push one LANES-beat of bf16 inputs and record them
    task push_beat(input [LANES*16-1:0] beat);
        integer k;
        begin
            @(negedge clk);
            in_valid = 1'b1;
            x_in     = beat;
            for (k = 0; k < LANES; k = k + 1)
                xq[n_in + k] = beat[16*k +: 16];
            n_in = n_in + LANES;
            @(posedge clk);          // beat accepted this edge
            #1;
            in_valid = 1'b0;
            x_in     = {LANES*16{1'b0}};
        end
    endtask

    // helper: build a LANES-beat from four bf16 patterns
    function automatic [LANES*16-1:0] pack4(input [15:0] a, input [15:0] b,
                                            input [15:0] c, input [15:0] d);
        pack4 = {d, c, b, a};
    endfunction

    // push a single value across all lanes (replicated) -- simple sweeps
    task push_val(input real v);
        reg [15:0] xb;
        begin
            xb = real_to_bf16_in(v);
            push_beat(pack4(xb, xb, xb, xb));
        end
    endtask

    // push a raw bf16 pattern replicated across lanes (for inf/nan/rails)
    task push_pat(input [15:0] xb);
        begin
            push_beat(pack4(xb, xb, xb, xb));
        end
    endtask

    // ===================================================================
    //  CHECK: after all beats have drained, recompute the golden for every
    //  captured element and compare both modes within tolerance.
    // ===================================================================
    task check_all;
        integer i;
        reg [15:0] xb, gref;
        real yd, yg, ae, denom, re;
        begin
            // sigmoid mode
            for (i = 0; i < n_out_sg; i = i + 1) begin
                xb   = xq[i];
                gref = golden(xb, 0);
                // reject nan/inf in DUT output (must always be finite)
                if (sgq[i][14:7] == 8'hFF) begin
                    $display("FAIL [SIGMOID] elem %0d x=%h: DUT out nan/inf=%h",
                             i, xb, sgq[i]);
                    errors = errors + 1;
                end
                yd = bf16_to_real(sgq[i]);
                yg = bf16_to_real(gref);
                ae = yd - yg; if (ae < 0.0) ae = -ae;
                if (ae > worst_abserr) begin
                    worst_abserr = ae; worst_mode = 0; worst_x = xb;
                end
                denom = (yg < 0.0) ? -yg : yg;
                re = (denom < TINY) ? 0.0 : ae/denom;
                if (denom >= TINY && re > worst_relerr) worst_relerr = re;
                // PASS if absolute OR relative bound holds
                if (!((ae <= ABS_TOL) || (denom >= TINY && re <= REL_TOL))) begin
                    $display("FAIL [SIGMOID] elem %0d x=%h: yd=%g yg=%g ae=%g re=%g",
                             i, xb, yd, yg, ae, re);
                    errors = errors + 1;
                end
                test_count = test_count + 1;
            end
            // silu mode
            for (i = 0; i < n_out_sl; i = i + 1) begin
                xb   = xq[i];
                gref = golden(xb, 1);
                if (slq[i][14:7] == 8'hFF) begin
                    $display("FAIL [SILU] elem %0d x=%h: DUT out nan/inf=%h",
                             i, xb, slq[i]);
                    errors = errors + 1;
                end
                yd = bf16_to_real(slq[i]);
                yg = bf16_to_real(gref);
                ae = yd - yg; if (ae < 0.0) ae = -ae;
                if (ae > worst_abserr) begin
                    worst_abserr = ae; worst_mode = 1; worst_x = xb;
                end
                denom = (yg < 0.0) ? -yg : yg;
                re = (denom < TINY) ? 0.0 : ae/denom;
                if (denom >= TINY && re > worst_relerr) worst_relerr = re;
                if (!((ae <= ABS_TOL) || (denom >= TINY && re <= REL_TOL))) begin
                    $display("FAIL [SILU] elem %0d x=%h: yd=%g yg=%g ae=%g re=%g",
                             i, xb, yd, yg, ae, re);
                    errors = errors + 1;
                end
                test_count = test_count + 1;
            end
        end
    endtask

    // ===================================================================
    //  MAIN
    // ===================================================================
    integer i;
    real    v;
    initial begin
        rng       = 32'h5EED_AC71;
        i         = $random(rng);     // seed
        in_valid  = 1'b0;
        x_in      = {LANES*16{1'b0}};
        n_in      = 0;
        n_out_sg  = 0;
        n_out_sl  = 0;

        rst = 1'b1;
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ---------- directed anchors ----------
        push_val( 0.0);                 // sigmoid 0.5, silu 0
        push_pat(16'h8000);             // -0.0
        push_val( 1.0);
        push_val(-1.0);
        push_val( 2.0);
        push_val(-2.0);
        push_val( 0.5);
        push_val(-0.5);

        // ---------- silu minimum dip region near x ~ -1.278 ----------
        push_val(-1.1);
        push_val(-1.2);
        push_val(-1.25);
        push_val(-1.278);
        push_val(-1.3);
        push_val(-1.35);
        push_val(-1.5);

        // ---------- large magnitudes (saturation tails) ----------
        push_val(  8.0);
        push_val( -8.0);
        push_val( 12.0);
        push_val(-12.0);
        push_val( 16.0);               // exact +X_SAT rail
        push_val(-16.0);               // exact -X_SAT rail
        push_val( 15.5);               // just inside
        push_val(-15.5);
        push_val( 20.0);               // beyond the rail -> saturates
        push_val(-20.0);
        push_val(100.0);
        push_val(-100.0);
        push_val(1000.0);
        push_val(-1000.0);

        // ---------- special / pathological inputs ----------
        push_pat(16'h7F80);            // +inf  -> sanitized +X_SAT
        push_pat(16'hFF80);            // -inf  -> sanitized -X_SAT
        push_pat(16'h7FC0);            // qnan  -> sanitized +X_SAT
        push_pat(16'hFFC0);            // qnan(-)-> sanitized +X_SAT (nan->+ per DUT)
        push_pat(16'h0001);            // tiny subnormal-ish +  -> ~0
        push_pat(16'h8001);            // tiny subnormal-ish -  -> ~0
        push_val( 1.0e-6);
        push_val(-1.0e-6);

        // ---------- fine dense sweep across [-X_SAT, X_SAT] ----------
        // step 0.125 over [-16, 16] -> 257 points (covers crossover + dip).
        for (i = 0; i <= 256; i = i + 1) begin
            v = -16.0 + (i * 0.125);
            push_val(v);
        end

        // ---------- random wide-dynamic-range, mixed sign ----------
        // mix small (|x|~1e-6..1e-2), mid (1e-2..3), large (3..1000) buckets.
        for (i = 0; i < 240; i = i + 1) begin
            case (i % 3)
                0: v = rand_real(1.0e-6, 1.0e-2, 1);
                1: v = rand_real(1.0e-2, 3.0,    1);
                default: v = rand_real(3.0, 1000.0, 1);
            endcase
            push_val(v);
        end

        // ---------- random per-lane independent (mixed lanes in a beat) ----
        for (i = 0; i < 120; i = i + 1) begin
            push_beat(pack4(
                real_to_bf16_in(rand_real(1.0e-4, 50.0, 1)),
                real_to_bf16_in(rand_real(1.0e-4, 50.0, 1)),
                real_to_bf16_in(rand_real(1.0e-4, 50.0, 1)),
                real_to_bf16_in(rand_real(1.0e-4, 50.0, 1))));
        end

        // ---------- drain the pipeline ----------
        in_valid = 1'b0;
        repeat (LAT + 4) @(negedge clk);

        // sanity: every pushed element should have produced an output
        if (n_out_sg != n_in || n_out_sl != n_in) begin
            $display("FAIL: capture count mismatch n_in=%0d n_out_sg=%0d n_out_sl=%0d",
                     n_in, n_out_sg, n_out_sl);
            $fatal(1, "count mismatch");
        end

        // ---------- check everything ----------
        check_all;

        if (errors != 0) begin
            $display("GLM_ACT: %0d ELEMENT MISMATCHES across %0d tests",
                     errors, test_count);
            $fatal(1, "glm_act mismatch");
        end
        $display("GLM_ACT worst abs-err = %g, worst rel-err = %g (ABS_TOL=%g REL_TOL=%g)",
                 worst_abserr, worst_relerr, ABS_TOL, REL_TOL);
        $display("  (worst at mode=%0d x=%h)", worst_mode, worst_x);
        $display("ALL %0d TESTS PASSED", test_count);
        $finish;
    end

    // global watchdog
    initial begin
        #500000000;
        $display("FAIL: global timeout");
        $fatal(1, "global timeout");
    end

endmodule
