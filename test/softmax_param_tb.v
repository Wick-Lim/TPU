`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// softmax_param_tb.v  --  2nd-/3rd-size PARAMETRIC proof for softmax_unit
//----------------------------------------------------------------------------
// PURPOSE
//   The committed unit TB (test/softmax_unit_tb.v) proves softmax_unit at its
//   DEFAULT length (LEN=`SM_LEN=8, NLINES=2).  This TB exercises the NEW `LEN`
//   parameter at TWO OTHER in-range sizes to prove the structural
//   parameterization is correct across the supported envelope:
//       * LEN=4   -> NLINES=1  (single TM line; the partial/short case)
//       * LEN=16  -> NLINES=4  (four TM lines; the multi-line case)
//   It is a SEPARATE small parametric TB (per the task's option), leaving the
//   committed LEN=8 TB and its assertion count untouched.
//
// INDEPENDENT GOLDEN (same method + same tolerance as the committed TB)
//   Each probability is checked against a golden computed in the REAL/`$exp()`
//   domain (NOT mirrored from the DUT's LUT/poly/fixed-point path):
//       ge[i] = exp( (L[i]-max)/256 ) ; p_gold[i] = round( ge[i]/SUM * 65536 ).
//   The golden shares NO arithmetic with the DUT, so it catches real bugs.
//   TOLERANCES (identical to the committed TB):
//       * |p_dut - p_gold| <= 2 LSB of Q0.16 per element,
//       * |SUM p_dut - 0xFFFF| <= 8 LSB,
//       * argmax (low 3 bits, the fixed status port) checked EXACTLY,
//       * done/busy timing checked EXACTLY
//         (LAT = 5 + NLINES + 2*LEN + DIV_CYCLES; the reciprocal is now a
//          multi-cycle radix-2 sequential divider, DIV_CYCLES = 48).
//
// STRUCTURE
//   A reusable harness module `sm_harness #(LEN)` owns its own TM model + DUT +
//   golden and runs DIRECTED + 120 constrained-random vectors, accumulating
//   pass/fail in output ports.  The top instantiates it at LEN=4 and LEN=16 and
//   prints "ALL <N> TESTS PASSED" iff every harness passed.
//============================================================================

// ----------------------------------------------------------------------------
// Reusable length-generic harness.
// ----------------------------------------------------------------------------
module sm_harness #(
    parameter integer LEN = 4
) (
    input  wire        clk,
    input  wire        go,        // pulse to run the whole suite
    output reg         done_all,  // high when this harness has finished
    output integer     pass,
    output integer     fail
);
    localparam integer NLANES = `LINE_LANES;
    localparam integer NLINES = (LEN + NLANES - 1) / NLANES;
    // The reciprocal is now a MULTI-CYCLE radix-2 sequential divider
    // (DIV_CYCLES = DIV_W = 48 cycles in softmax_unit), so the closed-form
    // start-edge..done-edge latency gained exactly DIV_CYCLES vs. the old
    // single-cycle divide:  LAT = 5 + NLINES + 2*LEN + DIV_CYCLES.
    localparam integer DIV_CYCLES = 48;               // softmax_unit DIV_W = 48
    localparam integer LAT    = 5 + NLINES + 2*LEN + DIV_CYCLES; // start..done incl.

    reg rst;

    // ---- DUT control ----
    reg                  start;
    reg  [`TM_IDX_W-1:0] x_base;
    reg  [`TM_IDX_W-1:0] p_base;
    wire                 busy;
    wire                 dut_done;
    wire                 sat;
    wire [2:0]           argmax;

    // ---- DUT <-> TM access ports ----
    wire [`TM_IDX_W-1:0] tm_raddr;
    reg  [`LINE_W-1:0]   tm_rdata;
    wire                 tm_we;
    wire [`TM_IDX_W-1:0] tm_waddr;
    wire [`LINE_W-1:0]   tm_wdata;

    // ---- harness-modelled tile memory ----
    reg [`LINE_W-1:0] tm [0:`TM_LINES-1];
    always @(*) tm_rdata = tm[tm_raddr];
    always @(posedge clk) if (tm_we) tm[tm_waddr] <= tm_wdata;

    // ---- DUT instantiated at THIS LEN ----
    softmax_unit #(.LEN(LEN)) dut (
        .clk      (clk),
        .rst      (rst),
        .start    (start),
        .x_base   (x_base),
        .p_base   (p_base),
        .busy     (busy),
        .done     (dut_done),
        .sat      (sat),
        .argmax   (argmax),
        .tm_raddr (tm_raddr),
        .tm_rdata (tm_rdata),
        .tm_we    (tm_we),
        .tm_waddr (tm_waddr),
        .tm_wdata (tm_wdata)
    );

    // ---- logits + golden ----
    reg signed [15:0] L  [0:LEN-1];
    real    ge [0:LEN-1];
    integer gp [0:LEN-1];
    integer gargmax;
    real    gsum;
    real    gmaxr;
    integer gmax;

    reg [`Q016_W-1:0] DP [0:LEN-1];

    integer i, j, t, cyc, seed, dsum, gmaxp;

    function integer absdiff;
        input integer a, b;
        begin absdiff = (a > b) ? (a - b) : (b - a); end
    endfunction

    // pack LEN logits into NLINES TM lines at `base` (sign-extended, low 16b).
    task write_logits;
        input [`TM_IDX_W-1:0] base;
        integer e, ln, ln_lane;
        reg [`LINE_W-1:0] line;
        begin
            for (ln = 0; ln < NLINES; ln = ln + 1) begin
                line = {`LINE_W{1'b0}};
                for (ln_lane = 0; ln_lane < NLANES; ln_lane = ln_lane + 1) begin
                    e = ln*NLANES + ln_lane;
                    if (e < LEN)
                        line[(ln_lane*`LANE_W) +: `LANE_W] =
                            { {16{L[e][15]}}, L[e] };
                end
                tm[base + ln] = line;
            end
        end
    endtask

    // read back LEN probs from NLINES lines at `base`.
    task read_probs;
        input [`TM_IDX_W-1:0] base;
        integer e, ln, ln_lane;
        begin
            for (ln = 0; ln < NLINES; ln = ln + 1)
                for (ln_lane = 0; ln_lane < NLANES; ln_lane = ln_lane + 1) begin
                    e = ln*NLANES + ln_lane;
                    if (e < LEN)
                        DP[e] = tm[base + ln][(ln_lane*`LANE_W) +: `Q016_W];
                end
        end
    endtask

    // INDEPENDENT real-exp() golden (true softmax, quantized at the boundary).
    task golden;
        integer pq;
        real    pr;
        begin
            gmax    = L[0];
            gargmax = 0;
            for (j = 1; j < LEN; j = j + 1)
                if (L[j] > gmax) begin gmax = L[j]; gargmax = j; end
            gmaxr = gmax;
            gsum = 0.0;
            for (j = 0; j < LEN; j = j + 1) begin
                ge[j] = $exp( (L[j] - gmaxr) / 256.0 );
                gsum  = gsum + ge[j];
            end
            for (j = 0; j < LEN; j = j + 1) begin
                pr = (ge[j] / gsum) * 65536.0;
                pq = $rtoi(pr + 0.5);
                if (pq > 65535) pq = 65535;
                if (pq < 0)     pq = 0;
                gp[j] = pq;
            end
        end
    endtask

    // run one op end-to-end, check against the golden (mirrors the committed TB).
    task run_check;
        input [255:0] tag;
        begin
            golden;
            gmaxp = 0;
            for (j = 0; j < LEN; j = j + 1)
                if (gp[j] > gmaxp) gmaxp = gp[j];

            write_logits(x_base);

            @(negedge clk);
            start = 1'b1;
            @(posedge clk);            // start edge (cyc 1)
            #1; start = 1'b0; cyc = 1;
            if (busy !== 1'b1) begin
                $display("FAIL[L=%0d %0s] busy not asserted after start", LEN, tag);
                fail = fail + 1; $fatal(1, "busy timing");
            end
            while (dut_done !== 1'b1) begin
                @(posedge clk); #1; cyc = cyc + 1;
                if (cyc > LAT + 4) begin
                    $display("FAIL[L=%0d %0s] done never asserted (cyc=%0d)",
                             LEN, tag, cyc);
                    fail = fail + 1; $fatal(1, "done timeout");
                end
            end
            if (cyc != LAT) begin
                $display("FAIL[L=%0d %0s] done at cyc=%0d expected %0d",
                         LEN, tag, cyc, LAT);
                fail = fail + 1; $fatal(1, "latency mismatch");
            end else pass = pass + 1;

            if (busy !== 1'b0) begin
                $display("FAIL[L=%0d %0s] busy still high at done", LEN, tag);
                fail = fail + 1; $fatal(1, "busy-at-done");
            end else pass = pass + 1;

            // argmax: the DUT emits the low 3 bits of the true argmax index on
            // the fixed 3-bit status port; check that EXACTLY against the golden
            // index truncated to 3 bits (valid for any LEN, exact for LEN<=8).
            if (argmax !== gargmax[2:0]) begin
                $display("FAIL[L=%0d %0s] argmax dut=%0d gold(low3)=%0d (true=%0d)",
                         LEN, tag, argmax, gargmax[2:0], gargmax);
                fail = fail + 1; $fatal(1, "argmax mismatch");
            end else pass = pass + 1;

            // sat boundary-tolerant consistency (same policy as committed TB).
            if ((gmaxp <= 65530) && (sat === 1'b1)) begin
                $display("FAIL[L=%0d %0s] spurious sat=1 gmaxp=%0d", LEN, tag, gmaxp);
                fail = fail + 1; $fatal(1, "spurious sat");
            end else if ((sat === 1'b1) && (gmaxp < 65533)) begin
                $display("FAIL[L=%0d %0s] sat=1 but gmaxp=%0d", LEN, tag, gmaxp);
                fail = fail + 1; $fatal(1, "sat without near-1.0");
            end else pass = pass + 1;

            #1; read_probs(p_base);

            dsum = 0;
            for (j = 0; j < LEN; j = j + 1) begin
                dsum = dsum + DP[j];
                if (absdiff(DP[j], gp[j]) > 2) begin
                    $display("FAIL[L=%0d %0s] p[%0d] dut=%0d gold=%0d diff=%0d logit=%0d",
                             LEN, tag, j, DP[j], gp[j], absdiff(DP[j], gp[j]), L[j]);
                    fail = fail + 1; $fatal(1, "prob mismatch");
                end else pass = pass + 1;
            end
            if (absdiff(dsum, 65535) > 8) begin
                $display("FAIL[L=%0d %0s] SUM p dut=%0d expected ~65535 diff=%0d",
                         LEN, tag, dsum, absdiff(dsum, 65535));
                fail = fail + 1; $fatal(1, "sum mismatch");
            end else pass = pass + 1;
        end
    endtask

    // run the whole suite when `go` rises.
    initial begin
        pass = 0; fail = 0; done_all = 1'b0;
        start = 1'b0; rst = 1'b1;
        x_base = 5'd0;
        p_base = 5'd8;        // probs base, clear of the logits lines
        seed   = 32'h1234_5678 + LEN;
        for (i = 0; i < `TM_LINES; i = i + 1) tm[i] = {`LINE_W{1'b0}};

        @(posedge go);
        // synchronous reset
        rst = 1'b1; @(posedge clk); @(posedge clk); rst = 1'b0; @(posedge clk);
        if (busy !== 1'b0 || dut_done !== 1'b0) begin
            $display("FAIL[L=%0d reset] busy=%b done=%b", LEN, busy, dut_done);
            fail = fail + 1; $fatal(1, "reset state");
        end else pass = pass + 1;

        // ---- directed ----
        // all-equal -> uniform 1/LEN, argmax=0
        for (i = 0; i < LEN; i = i + 1) L[i] = 16'sd100;
        run_check("alleq");
        // zero logits -> uniform
        for (i = 0; i < LEN; i = i + 1) L[i] = 16'sd0;
        run_check("zero");
        // one-hot huge at lane 0 -> p~1.0 (sat)
        for (i = 0; i < LEN; i = i + 1) L[i] = -16'sd8000;
        L[0] = 16'sd8000;
        run_check("onehot0");
        if (sat !== 1'b1) begin
            $display("FAIL[L=%0d onehot0] expected sat=1", LEN);
            fail = fail + 1; $fatal(1, "onehot sat");
        end else pass = pass + 1;
        // one-hot at the LAST lane -> argmax = (LEN-1) mod 8
        for (i = 0; i < LEN; i = i + 1) L[i] = -16'sd6000;
        L[LEN-1] = 16'sd6000;
        run_check("onehotlast");
        if (argmax !== ((LEN-1) & 3'b111)) begin
            $display("FAIL[L=%0d onehotlast] argmax=%0d expected %0d",
                     LEN, argmax, (LEN-1) & 7);
            fail = fail + 1; $fatal(1, "last argmax");
        end else pass = pass + 1;
        // ascending ramp -> monotone increasing probs
        for (i = 0; i < LEN; i = i + 1) L[i] = -16'sd700 + i*16'sd180;
        run_check("ramp");
        for (i = 1; i < LEN; i = i + 1)
            if (DP[i] < DP[i-1]) begin
                $display("FAIL[L=%0d ramp] p[%0d]=%0d < p[%0d]=%0d",
                         LEN, i, DP[i], i-1, DP[i-1]);
                fail = fail + 1; $fatal(1, "monotonicity");
            end else pass = pass + 1;
        // negative spread (valid distribution)
        for (i = 0; i < LEN; i = i + 1) L[i] = -16'sd300 - i*16'sd97;
        run_check("neg");
        // saturation extremes: max Q7.8 one-hot at lane 1, rest min Q7.8
        for (i = 0; i < LEN; i = i + 1) L[i] = `Q78_MIN;
        L[1] = `Q78_MAX;
        run_check("satmax");
        if (sat !== 1'b1) begin
            $display("FAIL[L=%0d satmax] expected sat=1", LEN);
            fail = fail + 1; $fatal(1, "sat-extreme");
        end else pass = pass + 1;

        // ---- constrained random ----
        for (t = 0; t < 120; t = t + 1) begin
            if (t < 90) begin
                for (i = 0; i < LEN; i = i + 1) L[i] = $random(seed);
            end else begin
                gmax = $random(seed);
                if (gmax > 32000)  gmax = 32000;
                if (gmax < -32000) gmax = -32000;
                for (i = 0; i < LEN; i = i + 1)
                    L[i] = gmax + ($random(seed) % 129);
            end
            run_check("rand");
        end

        done_all = 1'b1;
    end
endmodule

// ----------------------------------------------------------------------------
// Top: drive the clock, run both harnesses, aggregate the result.
// ----------------------------------------------------------------------------
module softmax_param_tb;
    reg clk;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    reg go;

    wire        done4,  done16;
    integer     pass4,  fail4;
    integer     pass16, fail16;

    sm_harness #(.LEN(4))  h4  (.clk(clk), .go(go),
                                .done_all(done4),  .pass(pass4),  .fail(fail4));
    sm_harness #(.LEN(16)) h16 (.clk(clk), .go(go),
                                .done_all(done16), .pass(pass16), .fail(fail16));

    integer total_pass, total_fail;

    initial begin
        go = 1'b0;
        @(posedge clk); @(posedge clk);
        go = 1'b1;                 // both harnesses start together
        @(posedge clk);
        go = 1'b0;

        // wait for both harnesses to finish.
        wait (done4 === 1'b1);
        wait (done16 === 1'b1);
        @(posedge clk);

        total_pass = pass4 + pass16;
        total_fail = fail4 + fail16;
        if (total_fail != 0) begin
            $display("SOFTMAX_PARAM TB: %0d FAILURES (LEN4 fail=%0d, LEN16 fail=%0d)",
                     total_fail, fail4, fail16);
            $fatal(1, "softmax_param_tb FAILED");
        end
        $display("LEN=4  : %0d checks passed", pass4);
        $display("LEN=16 : %0d checks passed", pass16);
        $display("ALL %0d TESTS PASSED", total_pass);
        $finish;
    end
endmodule
