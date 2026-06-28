`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_softmax.v  --  numerically-stable bf16 softmax over LEN logits
//                                                            (ACCEL_GLM52 §4)
//----------------------------------------------------------------------------
// Computes, over a row of LEN bf16 logits x[0..LEN-1]:
//     m   = max_j x_j                       (row max, fp32)
//     e_i = exp(x_i - m)                     (shifted exp, arg <= 0)
//     S   = Sum_k e_k                        (fp32 normalizer, ALL LEN terms)
//     p_i = e_i / S = e_i * (1/S)            (bf16 probability out)
//
// 1/S is formed as rsqrt(S)^2 (the project reciprocal: fp32_rsqrt then square),
// and exp / add / mul use the FIXED pipelined fp32 primitives in glm_fp_pipe.v.
//
// DENOMINATOR CORRECTNESS (the prior bug, fixed here):
//   The previous unit folded the exp stream with an interleaved/tree accumulator
//   that dropped/double-counted terms, so the row-sum came out ~2.0 at LEN=8 and
//   worse with LEN.  This design instead drives the exp pipe ONE logit per cycle
//   in strict index order, captures the LEN exp outputs into a buffer indexed by
//   a COUNTER that increments once per valid_out, and feeds a single SERIAL fp32
//   accumulator (one fp32_add_pipe, drained to completion before the next add).
//   Each exp result is therefore added to the running sum EXACTLY ONCE, and the
//   buffer write index advances exactly once per exp output -- independent of
//   LEN, LANES, or any pipe latency.  A done-counter (exp_out_cnt == LEN) is the
//   single gate that says "all LEN terms are in the sum and the buffer".
//
// INTERFACE (matches test/glm_softmax_tb.v):
//   parameters : LEN, LANES
//   inputs     : clk, rst (sync, active-high), start (1-cycle pulse alone),
//                in_valid, x_in[LANES*16-1:0]  (LANES bf16 logits / beat)
//   outputs    : busy, out_valid, p_out[LANES*16-1:0] (LANES bf16 probs / beat),
//                done
//   Handshake  : pulse start; the next cycle the DUT enters LOAD and consumes
//                LEN/LANES input beats (each beat = LANES lanes, row order).
//                After compute it emits LEN/LANES output beats with out_valid
//                high (row order), then pulses done.
//
// No latch, no comb loop, deterministic per-row latency, sync active-high reset.
//============================================================================
module glm_softmax #(
    parameter integer LEN   = 8,
    parameter integer LANES = 2
)(
    input                       clk,
    input                       rst,
    input                       start,
    input                       in_valid,
    input      [LANES*16-1:0]   x_in,
    output reg                  busy,
    output reg                  out_valid,
    output reg [LANES*16-1:0]   p_out,
    output reg                  done
);
    `include "glm_fp_pipe_lat.vh"

    localparam integer NBEATS = LEN / LANES;

    // index width helpers (>=1 bit)
    // IW counts 0..LEN (terminal value == LEN), so it needs room for LEN.
    // AW addresses the 0..LEN-1 arrays exactly.
    localparam integer IW  = (LEN   <= 1) ? 1 : $clog2(LEN+1);
    localparam integer AW  = (LEN   <= 1) ? 1 : $clog2(LEN);
    localparam integer BW  = (NBEATS<= 1) ? 1 : $clog2(NBEATS+1);
    // width-matched terminal constants (avoid 32-bit compare warnings)
    localparam [IW-1:0] LEN_IW   = IW'(LEN);
    localparam [IW-1:0] LENM1_IW = IW'(LEN-1);
    localparam [BW-1:0] NBM1_BW  = BW'(NBEATS-1);

    // ---------------- logit buffer (fp32) and exp buffer (fp32) ----------------
    reg [31:0] xbuf [0:LEN-1];      // widened fp32 logits, row order
    reg [31:0] ebuf [0:LEN-1];      // exp(x_i - m), row order

    // ===================================================================
    //  Pipelined FP primitives (instantiated once each, shared across passes)
    // ===================================================================
    // --- max-reduce uses fp32_add_pipe? no: compare combinationally (see below)

    // --- exp pipe ---
    reg          exp_vin;
    reg  [31:0]  exp_x;
    wire         exp_vout;
    wire [31:0]  exp_res;
    fp32_exp_pipe u_exp (
        .clk(clk), .rst(rst), .valid_in(exp_vin), .x(exp_x),
        .valid_out(exp_vout), .result(exp_res));

    // --- add pipe (used for x_i - m in shift pass, and the serial sum) ---
    reg          add_vin;
    reg  [31:0]  add_a, add_b;
    wire         add_vout;
    wire [31:0]  add_res;
    fp32_add_pipe u_add (
        .clk(clk), .rst(rst), .valid_in(add_vin), .a(add_a), .b(add_b),
        .valid_out(add_vout), .result(add_res));

    // --- mul pipe (square rsqrt, and e_i*(1/S)) ---
    reg          mul_vin;
    reg  [31:0]  mul_a, mul_b;
    wire         mul_vout;
    wire [31:0]  mul_res;
    fp32_mul_pipe u_mul (
        .clk(clk), .rst(rst), .valid_in(mul_vin), .a(mul_a), .b(mul_b),
        .valid_out(mul_vout), .result(mul_res));

    // --- rsqrt pipe (1/sqrt(S)) ---
    reg          rsq_vin;
    reg  [31:0]  rsq_x;
    wire         rsq_vout;
    wire [31:0]  rsq_res;
    fp32_rsqrt_pipe u_rsq (
        .clk(clk), .rst(rst), .valid_in(rsq_vin), .x(rsq_x),
        .valid_out(rsq_vout), .result(rsq_res));

    // ===================================================================
    //  FSM
    // ===================================================================
    localparam [3:0]
        S_IDLE   = 4'd0,
        S_LOAD   = 4'd1,   // consume NBEATS input beats -> xbuf[]
        S_MAX    = 4'd2,   // sequential fp32 max reduce over xbuf[]
        S_SHIFT  = 4'd3,   // x_i - m  (serial through add pipe) -> reuse xbuf[]
        S_EXP    = 4'd4,   // exp(shifted) one/cycle; capture -> ebuf[]; sum serially
        S_RSQ    = 4'd6,   // 1/sqrt(S)
        S_RECIP  = 4'd7,   // (1/sqrt(S))^2 = 1/S
        S_NORM   = 4'd8,   // e_i * (1/S) -> bf16 buffer (serial mul)
        S_OUT    = 4'd9,   // emit NBEATS output beats
        S_DONE   = 4'd10;
    reg [3:0] state;

    reg [IW-1:0] idx;       // generic element index
    reg [BW-1:0] beat;      // beat counter for load/out
    reg [IW-1:0] exp_out_cnt;   // # exp results captured (THE denominator gate)

    reg [31:0] maxv;        // running fp32 max
    reg [31:0] sumv;        // running fp32 sum (denominator), ALL LEN terms
    reg [31:0] recip;       // 1/S
    reg [15:0] pbuf [0:LEN-1];  // bf16 probabilities, row order

    // serial-add bookkeeping: we issue one add (sumv + ebuf[k]) and wait for it.
    reg        sum_busy;    // an add is in flight
    reg [IW-1:0] sum_idx;   // which exp term is being summed next
    reg [IW-1:0] sum_done_cnt; // # terms folded into sumv

    // shift-pass bookkeeping (serial through add pipe)
    reg        sh_busy;
    reg [IW-1:0] sh_in_cnt;   // # subtractions issued
    reg [IW-1:0] sh_out_cnt;  // # subtraction results captured

    // norm-pass bookkeeping (serial through mul pipe)
    reg        nm_busy;
    reg [IW-1:0] nm_in_cnt;
    reg [IW-1:0] nm_out_cnt;

    // single-shot guards for the reciprocal sub-pipeline (rsqrt then square).
    // These ensure the rsqrt/square are each issued EXACTLY ONCE and that a
    // leftover valid_out from the square never contaminates the NORM capture.
    reg        rsq_issued;     // rsqrt op has been launched
    reg        rcp_issued;     // square op has been launched

    integer li;

    // fp32 magnitude/compare for max: handle sign correctly.
    // For two FTZ-normal/zero fp32 values, a > b ?
    function automatic fp32_gt(input [31:0] a, input [31:0] b);
        reg sa, sb;
        reg [30:0] ma, mb;
        reg g, eq;
        begin
            sa = a[31]; sb = b[31];
            ma = a[30:0]; mb = b[30:0];
            // single shared magnitude comparator: g=(ma>mb), eq=(ma==mb);
            // ma<mb folds to (!g && !eq) -- identical result, one $gt + $eq only.
            g  = (ma > mb);
            eq = (ma == mb);
            // treat -0 and +0 as equal magnitude 0
            if (sa != sb) begin
                // different signs: positive is greater, unless both are zero
                if (ma == 31'b0 && mb == 31'b0) fp32_gt = 1'b0;
                else fp32_gt = (sb == 1'b1); // a positive, b negative -> a>b
            end else if (sa == 1'b0) begin
                // both non-negative: larger bit pattern is larger
                fp32_gt = g;
            end else begin
                // both negative: larger magnitude is smaller value (ma<mb)
                fp32_gt = (!g && !eq);
            end
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state       <= S_IDLE;
            busy        <= 1'b0;
            out_valid   <= 1'b0;
            done        <= 1'b0;
            p_out       <= {LANES*16{1'b0}};
            idx         <= {IW{1'b0}};
            beat        <= {BW{1'b0}};
            exp_out_cnt <= {IW{1'b0}};
            maxv        <= 32'b0;
            sumv        <= 32'b0;
            recip       <= 32'b0;
            exp_vin     <= 1'b0;
            add_vin     <= 1'b0;
            mul_vin     <= 1'b0;
            rsq_vin     <= 1'b0;
            sum_busy    <= 1'b0;
            sum_idx     <= {IW{1'b0}};
            sum_done_cnt<= {IW{1'b0}};
            sh_busy     <= 1'b0;
            sh_in_cnt   <= {IW{1'b0}};
            sh_out_cnt  <= {IW{1'b0}};
            nm_busy     <= 1'b0;
            nm_in_cnt   <= {IW{1'b0}};
            nm_out_cnt  <= {IW{1'b0}};
            rsq_issued  <= 1'b0;
            rcp_issued  <= 1'b0;
        end else begin
            // default deasserts (single-cycle strobes)
            exp_vin   <= 1'b0;
            add_vin   <= 1'b0;
            mul_vin   <= 1'b0;
            rsq_vin   <= 1'b0;
            out_valid <= 1'b0;
            done      <= 1'b0;

            case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy        <= 1'b1;
                    beat        <= {BW{1'b0}};
                    idx         <= {IW{1'b0}};
                    state       <= S_LOAD;
                end
            end

            // -----------------------------------------------------------
            // Consume NBEATS input beats; widen each lane bf16 -> fp32 -> xbuf.
            S_LOAD: begin
                busy <= 1'b1;
                if (in_valid) begin
                    for (li = 0; li < LANES; li = li + 1)
                        xbuf[beat*LANES + li] <=
                            bf16_to_fp32(x_in[16*li +: 16]);
                    if (beat == NBM1_BW) begin
                        beat  <= {BW{1'b0}};
                        idx   <= {IW{1'b0}};
                        state <= S_MAX;
                    end else begin
                        beat <= beat + 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------
            // Sequential fp32 max over xbuf[0..LEN-1] (combinational compare).
            S_MAX: begin
                busy <= 1'b1;
                if (idx == {IW{1'b0}}) begin
                    maxv <= xbuf[{AW{1'b0}}];
                    idx  <= {{(IW-1){1'b0}}, 1'b1};
                    if (LEN == 1) begin
                        idx        <= {IW{1'b0}};
                        sh_busy    <= 1'b0;
                        sh_in_cnt  <= {IW{1'b0}};
                        sh_out_cnt <= {IW{1'b0}};
                        state      <= S_SHIFT;
                    end
                end else begin
                    if (fp32_gt(xbuf[idx[AW-1:0]], maxv))
                        maxv <= xbuf[idx[AW-1:0]];
                    if (idx == LENM1_IW) begin
                        idx        <= {IW{1'b0}};
                        sh_busy    <= 1'b0;
                        sh_in_cnt  <= {IW{1'b0}};
                        sh_out_cnt <= {IW{1'b0}};
                        state      <= S_SHIFT;
                    end else begin
                        idx <= idx + 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------
            // Serial subtraction x_i - m through the add pipe.  We issue one
            // subtraction at a time and wait for its result, writing it back
            // into xbuf[] in index order.  sh_out_cnt advances once per result.
            S_SHIFT: begin
                busy <= 1'b1;
                // issue next subtraction when pipe is free of our single op
                if (!sh_busy && sh_in_cnt < LEN_IW) begin
                    add_vin   <= 1'b1;
                    add_a     <= xbuf[sh_in_cnt[AW-1:0]];
                    // m negated:  x - m = x + (-m)
                    add_b     <= {~maxv[31], maxv[30:0]};
                    sh_busy   <= 1'b1;
                    sh_in_cnt <= sh_in_cnt + 1'b1;
                end
                // capture result
                if (add_vout) begin
                    xbuf[sh_out_cnt[AW-1:0]] <= add_res;
                    sh_out_cnt <= sh_out_cnt + 1'b1;
                    sh_busy    <= 1'b0;
                    if (sh_out_cnt == LENM1_IW) begin
                        // all shifted; set up the exp/sum pass
                        idx          <= {IW{1'b0}};
                        exp_out_cnt  <= {IW{1'b0}};
                        sumv         <= 32'b0;
                        sum_busy     <= 1'b0;
                        sum_idx      <= {IW{1'b0}};
                        sum_done_cnt <= {IW{1'b0}};
                        state        <= S_EXP;
                    end
                end
            end

            // -----------------------------------------------------------
            // EXP + SERIAL SUM.
            //  * Drive the exp pipe one shifted logit per cycle (idx order).
            //  * Capture each exp output into ebuf[exp_out_cnt]; exp_out_cnt
            //    counts the LEN exp terms EXACTLY (one increment per valid_out).
            //  * A single serial fp32 accumulator folds every captured ebuf
            //    term into sumv exactly once (sum_done_cnt counts folds).
            S_EXP: begin
                busy <= 1'b1;
                // ---- feed exp pipe ----
                if (idx < LEN_IW) begin
                    exp_vin <= 1'b1;
                    exp_x   <= xbuf[idx[AW-1:0]];
                    idx     <= idx + 1'b1;
                end
                // ---- capture exp outputs (in order) ----
                if (exp_vout) begin
                    ebuf[exp_out_cnt[AW-1:0]] <= exp_res;
                    exp_out_cnt <= exp_out_cnt + 1'b1;
                end
                // ---- serial accumulate: fold ebuf terms into sumv one at a ----
                // time.  We may add a term as soon as it is captured.  Use the
                // exp result directly on the cycle it appears OR the buffered
                // value for already-captured terms.
                if (!sum_busy && sum_idx < exp_out_cnt) begin
                    // there is at least one captured-but-unfolded term
                    if (sum_done_cnt == {IW{1'b0}}) begin
                        // first term: sumv starts at 0, just register the term
                        // by adding to +0 (keeps a uniform serial-add path).
                        add_vin <= 1'b1;
                        add_a   <= 32'h00000000;
                        add_b   <= ebuf[sum_idx[AW-1:0]];
                    end else begin
                        add_vin <= 1'b1;
                        add_a   <= sumv;
                        add_b   <= ebuf[sum_idx[AW-1:0]];
                    end
                    sum_busy <= 1'b1;
                    sum_idx  <= sum_idx + 1'b1;
                end
                if (add_vout) begin
                    sumv         <= add_res;
                    sum_done_cnt <= sum_done_cnt + 1'b1;
                    sum_busy     <= 1'b0;
                end
                // ---- all LEN terms captured AND folded? ----
                if (exp_out_cnt == LEN_IW &&
                    sum_done_cnt == LEN_IW) begin
                    rsq_issued <= 1'b0;
                    state      <= S_RSQ;
                end
            end

            // (S_SUMW removed: dead state -- S_EXP transitions directly to S_RSQ
            //  and nothing assigns S_SUMW; the S_EXP completion gate covers it.)

            // -----------------------------------------------------------
            // 1/sqrt(S)  -- issue EXACTLY once (rsq_issued one-shot guard).
            S_RSQ: begin
                busy <= 1'b1;
                if (!rsq_issued) begin
                    rsq_vin    <= 1'b1;
                    rsq_x      <= sumv;
                    rsq_issued <= 1'b1;
                end
                if (rsq_vout) begin
                    recip      <= rsq_res;   // temporarily holds 1/sqrt(S)
                    rcp_issued <= 1'b0;
                    state      <= S_RECIP;
                end
            end

            // -----------------------------------------------------------
            // (1/sqrt(S))^2 = 1/S  -- issue the square EXACTLY once.  We capture
            // its result and ARM the NORM pass; the NORM pass only begins issuing
            // on the NEXT cycle, so this square's valid_out cannot be mistaken for
            // a NORM multiply result.
            S_RECIP: begin
                busy <= 1'b1;
                if (!rcp_issued) begin
                    mul_vin    <= 1'b1;
                    mul_a      <= recip;
                    mul_b      <= recip;
                    rcp_issued <= 1'b1;
                end
                if (rcp_issued && mul_vout) begin
                    recip      <= mul_res;     // now 1/S
                    nm_busy    <= 1'b0;
                    nm_in_cnt  <= {IW{1'b0}};
                    nm_out_cnt <= {IW{1'b0}};
                    state      <= S_NORM;
                end
            end

            // -----------------------------------------------------------
            // Serial normalize: p_i = e_i * (1/S), round to bf16 -> pbuf[].
            S_NORM: begin
                busy <= 1'b1;
                if (!nm_busy && nm_in_cnt < LEN_IW) begin
                    mul_vin   <= 1'b1;
                    mul_a     <= ebuf[nm_in_cnt[AW-1:0]];
                    mul_b     <= recip;
                    nm_busy   <= 1'b1;
                    nm_in_cnt <= nm_in_cnt + 1'b1;
                end
                if (mul_vout) begin
                    pbuf[nm_out_cnt[AW-1:0]] <= fp32_to_bf16(mul_res);
                    nm_out_cnt <= nm_out_cnt + 1'b1;
                    nm_busy    <= 1'b0;
                    if (nm_out_cnt == LENM1_IW) begin
                        beat  <= {BW{1'b0}};
                        state <= S_OUT;
                    end
                end
            end

            // -----------------------------------------------------------
            // Emit NBEATS output beats (LANES lanes each), row order.
            S_OUT: begin
                busy      <= 1'b1;
                out_valid <= 1'b1;
                for (li = 0; li < LANES; li = li + 1)
                    p_out[16*li +: 16] <= pbuf[beat*LANES + li];
                if (beat == NBM1_BW) begin
                    beat  <= {BW{1'b0}};
                    state <= S_DONE;
                end else begin
                    beat <= beat + 1'b1;
                end
            end

            // -----------------------------------------------------------
            S_DONE: begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
