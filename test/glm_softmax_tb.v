`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_softmax_tb.v  --  self-checking TB for glm_softmax (BF16 stable softmax)
//                                                            (ACCEL_GLM52 §4)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL
//   The DUT (glm_softmax) computes, over a row of LEN bf16 logits x[0..LEN-1]:
//       m   = max_j x_j                       (row max, fp32)
//       e_i = exp(x_i - m)                     (shifted exp, arg <= 0)
//       S   = Sum_k e_k                        (fp32 normalizer)
//       p_i = e_i / S                          (bf16 probability out)
//   using the project's bit-accurate fp32 pipelines (Taylor range-reduced exp,
//   Quake-seed Newton rsqrt for 1/S, fp32 add/mul), rounding only the final
//   probabilities back to bf16.
//
//   This golden shares NONE of that arithmetic.  It recomputes softmax a
//   COMPLETELY DIFFERENT way, in Verilog `real` (IEEE double, fp64):
//     * widen each stored bf16 logit to its EXACT real value (bf16->real),
//     * m_r   = max over the real logits,
//     * e_r_i = $exp(x_r_i - m_r)              (the C math-library exp, NOT the
//                                               DUT's magic-const range-red poly),
//     * S_r   = Sum e_r_i  in fp64 (full double precision, no fp32 reduce),
//     * p_r_i = e_r_i / S_r                     (a true fp64 divide, NOT rsqrt^2),
//     * quantize p_r_i to bf16 the SAME way the unit emits (real->fp32 bits via
//       $shortrealtobits, then fp32_to_bf16 RNE).
//   The only shared step is the final fp32->bf16 RNE pack (the unit's DEFINED
//   output format -- both DUT and golden must land on the same bf16 grid).
//   Everything that produces the *value* differs (math-lib exp + fp64 reduce +
//   true divide vs. Taylor exp + fp32 reduce + rsqrt^2 reciprocal), so the
//   golden catches DUT arithmetic bugs instead of mirroring them.
//
//   STABILITY is exercised directly: rows with large POSITIVE logits would
//   overflow exp() without the max-subtract.  Both DUT and golden subtract the
//   row max first, so exp args are <= 0 -- a correct DUT must reproduce the
//   golden on these rows (a DUT that forgot the max-subtract would overflow to
//   inf/nan and FAIL), proving the documented stability discipline.
//
//----------------------------------------------------------------------------
// TOLERANCE  (STATED + JUSTIFIED)
//   Per element we compare the DUT bf16 probability to the fp64-golden value
//   (re-widened from its bf16 quantization) as an ABSOLUTE error
//        abserr = |p_dut_real - p_gold_real|
//   and (for non-near-zero golden) a RELATIVE error
//        relerr = abserr / |p_gold_real|.
//   An element PASSES if EITHER bound holds:
//        abserr <= ABS_TOL = 2^-7  (~0.0078)   OR
//        relerr <= REL_TOL = 3/64  (~0.0469, = 6 bf16 ULP).
//
//   Why these (the error budget):
//     * Each p_i in [0,1] is rounded to bf16 (8 significand bits): one bf16 ULP
//       is 2^-8 of the value, so the final rounding alone costs up to 0.5 ULP.
//     * DOUBLE ROUNDING dominates: the golden rounds the exact fp64 value to
//       bf16 while the DUT rounds an fp32 value whose own rounding can tip it
//       across the bf16 halfway point => the two can land on ADJACENT bf16
//       codes (~1 bf16 ULP apart).
//     * The DUT exp poly (<2^-18) and the rsqrt-based 1/S (<2^-21 rel) are each
//       far below a bf16 ULP, BUT they enter the normalize multiply e_i*(1/S)
//       which then rounds to bf16 -- a few bf16 ULP of accumulated slack across
//       exp-approx + fp32-reduce-vs-fp64 + reciprocal + the multiply round.
//     * REL_TOL = 6 bf16 ULP (3/64) is the principled envelope for that chain;
//       ABS_TOL = 2^-7 covers the tiny-magnitude probabilities (one-hot tails,
//       large-LEN uniform 1/64 ~ 0.0156) where the relative bound is too tight
//       for a single ULP wobble.  Either-of bound = the looser, principled
//       worst-case envelope; a real bug (wrong sign, missing max-subtract,
//       mis-normalized) moves an output by >> 6 ULP and still fails.
//   For golden magnitudes below TINY (true near-zero, where relative error is
//   meaningless) we require only the absolute bound.
//
//   SUM CHECK: we also assert Sum_i p_dut_real ~= 1 per row.  The DUT sums in
//   fp32 and divides by a ~2^-21-accurate 1/S, then rounds LEN probabilities to
//   bf16; each round is <= 0.5 bf16 ULP, so the row sum can drift by up to ~LEN
//   half-ULPs plus the reciprocal slack.  We allow |Sum p - 1| <= SUM_TOL where
//   SUM_TOL = 2^-5 + LEN*2^-9 (a per-element half-ULP budget + reciprocal slack)
//   -- loose enough for honest bf16 rounding, tight enough to catch a missing or
//   wrong normalization (which breaks the sum by O(1)).
//
//----------------------------------------------------------------------------
// COVERAGE  (each emitted probability checked; a "TEST" = one checked element)
//   LEN in {8, 64}, LANES=2.  For each LEN, directed rows:
//     * all-equal logits      -> uniform p = 1/LEN,
//     * one dominant logit    -> p ~ one-hot (rest ~ 0),
//     * LARGE POSITIVE spread  (stability: max-subtract must prevent exp ovf),
//     * LARGE NEGATIVE logits  (all tiny, still must normalize to sum 1),
//     * mixed-sign moderate range,
//     * a ramp,
//   plus many RANDOM rows with WIDE-RANGE logits (incl. large positive/negative
//   and mixed sign).  Every emitted element is checked within tolerance and the
//   per-row Sum-of-p is asserted ~= 1.  Prints "ALL <N> TESTS PASSED"; $fatal on
//   any element mismatch or any row-sum violation.
//============================================================================
module glm_softmax_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    localparam integer LANES = 2;
    localparam integer LEN8  = 8;
    localparam integer LEN64 = 64;
    localparam integer NB8   = LEN8  / LANES;   // beats for the LEN=8  DUT
    localparam integer NB64  = LEN64 / LANES;   // beats for the LEN=64 DUT

    // tolerances (see header)
    localparam real ABS_TOL = 1.0/128.0;        // 2^-7
    localparam real REL_TOL = 3.0/64.0;         // 6 bf16 ULP
    localparam real TINY    = 1.0e-9;

    `include "glm_fp.vh"

    // ===================================================================
    //  bf16 <-> real helpers (independent of the DUT fp32 path)
    //  Implemented by EXPLICIT IEEE field decode/encode in Verilog `real`
    //  (fp64) -- deliberately NOT using $bitstoshortreal/$shortrealtobits so
    //  the golden is portable across simulators (shortreal is unsupported in
    //  some) and stays an INDEPENDENT path from the DUT's fp32 bit ops.
    // ===================================================================
    // exact bf16 -> fp64 real.  bf16 = 1 sign | 8 exp(bias127) | 7 mant.
    function automatic real bf16_to_real(input [15:0] b);
        reg        s;
        reg [7:0]  e;
        reg [6:0]  m;
        real       mant, val;
        integer    k;
        integer    ei;          // SIGNED unbiased exponent (e is unsigned 8-bit)
        begin
            s = b[15]; e = b[14:7]; m = b[6:0];
            if (e == 8'hFF) begin
                // inf/nan -> a huge magnitude (the softmax stimulus never feeds
                // these; map to a large finite so any leakage is caught, not NaN).
                val = 1.0e38;
            end else if (e == 8'h00) begin
                val = 0.0;                       // FTZ (matches glm_fp.vh policy)
            end else begin
                // significand 1.mmmmmmm
                mant = 1.0;
                for (k = 0; k < 7; k = k + 1)
                    if (m[k]) mant = mant + $pow(2.0, $itor(k-7));
                // 2^(e-127) -- compute e-127 in SIGNED integer arithmetic; doing it
                // directly (e is an unsigned 8-bit reg) underflows to a huge value.
                ei  = e;                         // widen to signed integer first
                val = mant * $pow(2.0, $itor(ei-127));
            end
            bf16_to_real = s ? -val : val;
        end
    endfunction

    // quantize a real to fp32 BITS by explicit IEEE encode (RNE at bit 23),
    // then fp32_to_bf16 RNE from glm_fp.vh (the unit's DEFINED output pack).
    // The two-step (real->fp32 bits->bf16) mirrors the DUT's fp32-compute /
    // bf16-emit contract so both land on the same bf16 grid.
    function automatic [31:0] real_to_fp32bits(input real r);
        reg        s;
        real       a, mant;
        integer    exp_e;
        reg [22:0] frac;
        reg [24:0] sig;       // 25-bit to hold a rounding carry into bit24
        real       scaled;
        integer    ebias;
        reg [7:0]  efield;
        begin
            if (r == 0.0) begin
                real_to_fp32bits = 32'h00000000;
            end else begin
                s = (r < 0.0);
                a = s ? -r : r;
                exp_e = 0;
                // normalize a into [1,2)
                while (a >= 2.0) begin a = a/2.0; exp_e = exp_e + 1; end
                while (a <  1.0) begin a = a*2.0; exp_e = exp_e - 1; end
                // a in [1,2): 24-bit significand with RNE at bit 23.
                scaled = a * $pow(2.0, 23);
                sig = $rtoi(scaled);
                if ((scaled - $itor($rtoi(scaled))) > 0.5)
                    sig = sig + 25'd1;
                else if ((scaled - $itor($rtoi(scaled))) == 0.5)
                    sig = sig + (sig & 25'd1);       // round-half-to-even
                if (sig[24]) begin                   // carry out of significand
                    sig = sig >> 1; exp_e = exp_e + 1;
                end
                frac  = sig[22:0];
                ebias = exp_e + 127;
                if (ebias >= 255) begin
                    efield = 8'hFF; frac = 23'b0;     // overflow -> inf
                end else if (ebias <= 0) begin
                    efield = 8'h00; frac = 23'b0;     // underflow -> FTZ zero
                end else begin
                    efield = ebias[7:0];
                end
                real_to_fp32bits = {s, efield, frac};
            end
        end
    endfunction

    function automatic [15:0] real_to_bf16(input real r);
        real_to_bf16 = fp32_to_bf16(real_to_fp32bits(r));
    endfunction

    // ===================================================================
    //  Two DUTs (LEN is a module parameter -> one instance per LEN).  Each is
    //  driven by its own start/in_valid/x_in and captured on out_valid.
    // ===================================================================
    // ---- LEN = 8 DUT ----
    reg                  s8_start, s8_in_valid;
    reg  [LANES*16-1:0]  s8_x_in;
    wire                 s8_busy, s8_out_valid, s8_done;
    wire [LANES*16-1:0]  s8_p_out;
    glm_softmax #(.LEN(LEN8), .LANES(LANES)) dut8 (
        .clk(clk), .rst(rst), .start(s8_start), .in_valid(s8_in_valid),
        .x_in(s8_x_in), .busy(s8_busy), .out_valid(s8_out_valid),
        .p_out(s8_p_out), .done(s8_done));

    // ---- LEN = 64 DUT ----
    reg                  s64_start, s64_in_valid;
    reg  [LANES*16-1:0]  s64_x_in;
    wire                 s64_busy, s64_out_valid, s64_done;
    wire [LANES*16-1:0]  s64_p_out;
    glm_softmax #(.LEN(LEN64), .LANES(LANES)) dut64 (
        .clk(clk), .rst(rst), .start(s64_start), .in_valid(s64_in_valid),
        .x_in(s64_x_in), .busy(s64_busy), .out_valid(s64_out_valid),
        .p_out(s64_p_out), .done(s64_done));

    // ===================================================================
    //  Test bookkeeping
    // ===================================================================
    integer test_count   = 0;
    integer errors       = 0;
    real    worst_abserr = 0.0;
    real    worst_relerr = 0.0;
    real    worst_sumerr = 0.0;
    integer sum_report   = 0;
    integer rng;

    // ---- per-row storage (sized to the larger LEN) ----
    reg [15:0] row_x    [0:LEN64-1];   // bf16 logits fed to the DUT (row order)
    reg [15:0] row_pgold[0:LEN64-1];   // bf16 golden probabilities
    reg [15:0] row_pdut [0:LEN64-1];   // bf16 DUT probabilities (captured)

    // ---- random real in [lo,hi) with optional random sign ----
    function automatic real rand_real(input real lo, input real hi,
                                       input integer signd);
        real u; integer r;
        begin
            r = $random(rng);
            u = (((r % 1000000) + 1000000) % 1000000) / 1000000.0;  // [0,1)
            rand_real = lo + u*(hi-lo);
            if (signd) begin
                r = $random(rng);
                if (r & 1) rand_real = -rand_real;
            end
        end
    endfunction

    // ===================================================================
    //  INDEPENDENT GOLDEN: fill row_pgold[] from row_x[] (LEN elements), in
    //  fp64: max-subtract, math-lib exp, fp64 sum, true divide, quantize bf16.
    // ===================================================================
    task automatic golden_row(input integer len);
        integer i;
        real    xr, mr, er, sr;
        real    erow [0:LEN64-1];
        begin
            // row max (fp64) over the EXACT bf16 values
            mr = bf16_to_real(row_x[0]);
            for (i = 1; i < len; i = i + 1) begin
                xr = bf16_to_real(row_x[i]);
                if (xr > mr) mr = xr;
            end
            // shifted exp + fp64 sum
            sr = 0.0;
            for (i = 0; i < len; i = i + 1) begin
                xr      = bf16_to_real(row_x[i]);
                er      = $exp(xr - mr);          // arg <= 0 -> er in (0,1]
                erow[i] = er;
                sr      = sr + er;
            end
            // normalize (true fp64 divide) + quantize to bf16
            for (i = 0; i < len; i = i + 1)
                row_pgold[i] = real_to_bf16(erow[i] / sr);
        end
    endtask

    // ===================================================================
    //  CHECK one row: compare row_pdut[] vs row_pgold[] element-wise within
    //  tolerance, and assert Sum_i p_dut ~= 1.
    // ===================================================================
    task automatic check_row(input integer len, input [255:0] label);
        integer i;
        real    pd, pg, abserr, relerr, psum, sum_tol;
        begin
            psum = 0.0;
            for (i = 0; i < len; i = i + 1) begin
                // X-AWARE GUARD: any X/Z bit in a captured DUT probability is a
                // hard failure (bf16_to_real would silently read X bits as 0 and
                // could mask the defect inside the numeric tolerance).  The
                // reduction-XOR is X iff ANY bit of row_pdut[i] is X/Z.
                if (^row_pdut[i] === 1'bx) begin
                    errors = errors + 1;
                    $display("X-OUTPUT [%0s] len=%0d i=%0d : p_dut=%h has X/Z bit(s)",
                             label, len, i, row_pdut[i]);
                end
                pd = bf16_to_real(row_pdut[i]);
                pg = bf16_to_real(row_pgold[i]);
                psum = psum + pd;

                abserr = (pd > pg) ? (pd - pg) : (pg - pd);
                relerr = (pg > TINY || pg < -TINY) ?
                         (abserr / ((pg < 0.0) ? -pg : pg)) : 0.0;
                if (abserr > worst_abserr) worst_abserr = abserr;
                if ((pg > TINY || pg < -TINY) && relerr > worst_relerr)
                    worst_relerr = relerr;

                test_count = test_count + 1;
                if (!((abserr <= ABS_TOL) ||
                      ((pg > TINY || pg < -TINY) && (relerr <= REL_TOL)))) begin
                    errors = errors + 1;
                    $display("MISMATCH [%0s] len=%0d i=%0d : x=%h p_dut=%h(%f) p_gold=%h(%f) abserr=%g relerr=%g",
                             label, len, i, row_x[i],
                             row_pdut[i], pd, row_pgold[i], pg, abserr, relerr);
                end
            end
            // row-sum check: Sum p ~= 1
            sum_tol = (1.0/32.0) + len*(1.0/512.0);    // 2^-5 + LEN*2^-9
            abserr  = (psum > 1.0) ? (psum - 1.0) : (1.0 - psum);
            if (abserr > worst_sumerr) worst_sumerr = abserr;
            if (sum_report < 6) begin
                $display("ROW-SUM [%0s] len=%0d : Sum p_dut = %f (|.-1|=%g, tol=%g)",
                         label, len, psum, abserr, sum_tol);
                sum_report = sum_report + 1;
            end
            if (abserr > sum_tol) begin
                errors = errors + 1;
                $display("ROW-SUM FAIL [%0s] len=%0d : Sum p_dut=%f |.-1|=%g > tol=%g",
                         label, len, psum, abserr, sum_tol);
            end
        end
    endtask

    // ===================================================================
    //  DRIVERS: stream a full row into a DUT, capture all output beats.
    //  (One task per DUT because the port nets differ.)
    // ===================================================================
    integer guard;

    task automatic run_row8;
        integer b, l, oc;
        begin
            // start is a 1-cycle pulse ALONE; the NBEATS input beats follow it
            // (the DUT samples `start` in S_IDLE and enters S_LOAD the next cycle,
            // where it consumes the in_valid beats).
            @(negedge clk);
            s8_start    = 1'b1;
            s8_in_valid = 1'b0;
            @(negedge clk);
            s8_start = 1'b0;
            for (b = 0; b < NB8; b = b + 1) begin
                s8_in_valid = 1'b1;
                for (l = 0; l < LANES; l = l + 1)
                    s8_x_in[16*l +: 16] = row_x[b*LANES + l];
                @(negedge clk);
            end
            s8_in_valid = 1'b0;
            // capture NB8 output beats
            oc = 0;
            guard = 0;
            while (oc < NB8 && guard < 100000) begin
                if (s8_out_valid) begin
                    for (l = 0; l < LANES; l = l + 1)
                        row_pdut[oc*LANES + l] = s8_p_out[16*l +: 16];
                    oc = oc + 1;
                end
                @(negedge clk);
                guard = guard + 1;
            end
            if (oc != NB8) begin
                $display("TIMEOUT len=8 : got %0d/%0d output beats", oc, NB8);
                errors = errors + 1;
            end
            // let done settle
            @(negedge clk);
        end
    endtask

    task automatic run_row64;
        integer b, l, oc;
        begin
            @(negedge clk);
            s64_start    = 1'b1;
            s64_in_valid = 1'b0;
            @(negedge clk);
            s64_start = 1'b0;
            for (b = 0; b < NB64; b = b + 1) begin
                s64_in_valid = 1'b1;
                for (l = 0; l < LANES; l = l + 1)
                    s64_x_in[16*l +: 16] = row_x[b*LANES + l];
                @(negedge clk);
            end
            s64_in_valid = 1'b0;
            oc = 0;
            guard = 0;
            while (oc < NB64 && guard < 100000) begin
                if (s64_out_valid) begin
                    for (l = 0; l < LANES; l = l + 1)
                        row_pdut[oc*LANES + l] = s64_p_out[16*l +: 16];
                    oc = oc + 1;
                end
                @(negedge clk);
                guard = guard + 1;
            end
            if (oc != NB64) begin
                $display("TIMEOUT len=64 : got %0d/%0d output beats", oc, NB64);
                errors = errors + 1;
            end
            @(negedge clk);
        end
    endtask

    // run golden + DUT + check for a populated row_x[] of length `len`.
    task automatic do_row(input integer len, input [255:0] label);
        begin
            golden_row(len);
            if (len == LEN8) run_row8; else run_row64;
            check_row(len, label);
        end
    endtask

    // ---- fill row_x[] from a real value via a generator helper ----
    // (callers set row_x[] directly; these are common patterns.)
    task automatic fill_const(input integer len, input real v);
        integer i; begin
            for (i = 0; i < len; i = i + 1) row_x[i] = real_to_bf16(v);
        end
    endtask

    task automatic fill_random(input integer len, input real lo, input real hi);
        integer i; begin
            for (i = 0; i < len; i = i + 1)
                row_x[i] = real_to_bf16(rand_real(lo, hi, 1));
        end
    endtask

    // ===================================================================
    //  STIMULUS
    // ===================================================================
    integer t, i, k;
    real    v;

    // sweep both LEN values through the same directed + random battery.
    task automatic battery(input integer len, input [255:0] tag);
        integer j;
        begin
            // all-equal -> uniform p = 1/len
            fill_const(len, 1.25);
            do_row(len, {tag, " all-equal"});

            // all-equal at a LARGE POSITIVE value (stability: exp(big) would
            // overflow without max-subtract; shifted args are all 0 -> uniform).
            fill_const(len, 80.0);
            do_row(len, {tag, " all-equal-bigpos"});

            // one dominant logit -> ~one-hot
            for (j = 0; j < len; j = j + 1) row_x[j] = real_to_bf16(-4.0);
            row_x[len/3] = real_to_bf16(12.0);
            do_row(len, {tag, " one-hot"});

            // LARGE POSITIVE spread (stability stress): big positive logits with
            // a clear max; without max-subtract exp() overflows to inf -> nan.
            for (j = 0; j < len; j = j + 1)
                row_x[j] = real_to_bf16(60.0 + 1.0*j);
            do_row(len, {tag, " large-positive-spread"});

            // LARGE NEGATIVE logits (all far below 0, must still normalize to 1)
            for (j = 0; j < len; j = j + 1)
                row_x[j] = real_to_bf16(-50.0 - 0.5*j);
            do_row(len, {tag, " large-negative"});

            // mixed-sign moderate range ramp
            for (j = 0; j < len; j = j + 1)
                row_x[j] = real_to_bf16(-3.0 + 0.5*j);
            do_row(len, {tag, " mixed-ramp"});

            // a few random wide-range rows (mixed sign, wide dynamic range)
            for (j = 0; j < 6; j = j + 1) begin
                fill_random(len, 0.0, 20.0);   // |x| up to 20, both signs
                do_row(len, {tag, " random"});
            end
            // random with very large magnitudes (stability under wide spread)
            for (j = 0; j < 4; j = j + 1) begin
                fill_random(len, 0.0, 90.0);
                do_row(len, {tag, " random-wide"});
            end
        end
    endtask

    initial begin
        rng         = 32'hC0FFEE42;
        rst         = 1'b1;
        s8_start    = 1'b0; s8_in_valid  = 1'b0; s8_x_in  = {LANES*16{1'b0}};
        s64_start   = 1'b0; s64_in_valid = 1'b0; s64_x_in = {LANES*16{1'b0}};
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        battery(LEN8,  "len8 ");
        battery(LEN64, "len64");

        if (errors != 0) begin
            $display("FAILED: %0d error(s) over %0d checked elements.", errors, test_count);
            $display("  worst abserr=%g  worst relerr=%g  worst |sum-1|=%g",
                     worst_abserr, worst_relerr, worst_sumerr);
            $fatal(1, "glm_softmax_tb: MISMATCH");
        end else begin
            $display("worst abserr=%g  worst relerr=%g  worst |sum-1|=%g (ABS_TOL=%g REL_TOL=%g)",
                     worst_abserr, worst_relerr, worst_sumerr, ABS_TOL, REL_TOL);
            $display("ALL %0d TESTS PASSED", test_count);
        end
        $finish;
    end

    // global watchdog
    initial begin
        #50_000_000;
        $display("GLOBAL TIMEOUT");
        $fatal(1, "glm_softmax_tb: global timeout");
    end

endmodule
