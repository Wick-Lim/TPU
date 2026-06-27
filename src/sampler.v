`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// sampler.v  --  GLM-5.2 TOKEN SAMPLER  (ACCEL_GLM52 §3 row 21, §LM-head)
//----------------------------------------------------------------------------
// PURPOSE
//   Turn the final LM-head logits over the vocabulary into ONE sampled token
//   index.  The pipeline is the standard decode-time sampler:
//
//     (1) TEMPERATURE SCALE   z_i = logit_i * (1/T)   (fp32 multiply by inv-T)
//     (2) TOP-K FILTER        keep the TOPK highest z, mask the rest to -inf
//                             (shared topk_select); optional TOP-P (nucleus)
//                             keeps the smallest descending-prob set summing >=p
//     (3) SOFTMAX             p = softmax(kept z)      (shared glm_softmax)
//     (4) MULTINOMIAL DRAW    u ~ U[0,1) from an LFSR PRNG; walk the cumulative
//                             probability of the kept tokens in DESCENDING-prob
//                             order and emit the FIRST token whose running
//                             cumsum >= u.
//
//   GREEDY / ARGMAX  : as T -> 0 the softmax concentrates all mass on the max
//   logit, so the multinomial walk ALWAYS returns the argmax.  We expose this
//   limit exactly: when `greedy` is asserted (or inv_temp is +inf, i.e. T=0)
//   the unit bypasses softmax+draw and emits topk slot-0 (the argmax) directly.
//   This makes the T->0 behaviour bit-exact (no dependence on exp underflow).
//
//   DETERMINISM  : the only randomness is the LFSR, seeded by the SEED
//   parameter and reset to SEED on `rst`.  Given the seed the sampled token is
//   a deterministic function of the logits/temperature/top-p -> fully testable.
//
//----------------------------------------------------------------------------
// WHY THIS COMPOSES THE SHARED UNITS
//   * topk_select (src/topk_select.v): fp32 top-K of N with a deterministic
//     lower-index tie-break, returning the K indices AND scores in DESCENDING
//     score order (slot 0 = largest).  We feed it the temperature-scaled fp32
//     logits; its descending order is exactly the order the nucleus walk and
//     the cumulative-probability draw want, so no extra sort is needed.
//   * glm_softmax (src/glm_softmax.v): numerically-stable bf16 softmax (fp32
//     reduce).  We run it over a LEN=TOPK row built from the kept (top-k) fp32
//     scores, narrowed to bf16, so the probabilities cover exactly the kept
//     tokens (the masked-out vocab contributes 0 mass by construction).
//
//   Because softmax runs only over the TOPK kept scores (not all VOCAB), the
//   "mask the rest to -inf then softmax over VOCAB" and "softmax over the kept
//   set" are identical (exp(-inf)=0): the dropped tokens contribute nothing.
//   This keeps the softmax row small (TOPK) and the result already in
//   descending-prob order aligned with the topk index slots.
//
//----------------------------------------------------------------------------
// INTERFACE  (start / stream-load / done handshake)
//   parameters : VOCAB (logit count, default 16; scales to 154880),
//                TOPK   (top-k filter width, default 4),
//                SEED   (32-bit LFSR seed, default nonzero),
//                LFSR_W (LFSR width, default 32; maximal-length taps for 32).
//   clk, rst                 : synchronous, active-high reset.
//   start                    : 1-cycle pulse to begin a draw.
//   greedy                   : sample mode select latched at start; 1 => argmax
//                              (T->0).  Also forced greedy if inv_temp == +inf.
//   inv_temp [15:0]          : bf16 reciprocal temperature 1/T (T>0).  +inf
//                              (0x7F80) => greedy/argmax.  1.0 => no scaling.
//   topp     [15:0]          : bf16 nucleus threshold p in (0,1].  topp >= 1.0
//                              (e.g. 0x3F80) disables top-p (keep all TOPK).
//   --- logit LOAD (unit pulls one bf16 logit/cycle, vocab index order) ---
//   load_req                 : high while the unit wants the next logit.
//   logit_in [15:0]          : one bf16 logit (vocab index = load beat count).
//   logit_valid              : producer asserts when logit_in holds a logit.
//   --- result (held from done until next start) ---
//   token_o  [IDXW-1:0]      : the sampled vocab token index.
//   done                     : 1-cycle pulse when token_o is valid.
//   busy                     : high from start until done.
//
//----------------------------------------------------------------------------
// STYLE / CORRECTNESS INVARIANTS
//   * `timescale + `include "glm_fp.vh" + this header; sync active-high reset.
//   * Every reg assigned on every path (no inferred latch); Moore-ish FSM.
//   * No combinational loop: topk tree + softmax pipe are feed-forward; control
//     is a registered FSM; the LFSR advances on a clock edge.
//   * Deterministic given SEED.
//============================================================================
module sampler #(
    parameter integer VOCAB  = 16,
    parameter integer TOPK   = 4,
    parameter [31:0]  SEED   = 32'hACE1_2345,
    parameter integer LFSR_W = 32,
    // IDXW is DERIVED ($clog2(VOCAB)); exposed only to size the output port.
    parameter integer IDXW   = (VOCAB <= 1) ? 1 : $clog2(VOCAB)
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    input  wire              greedy,
    input  wire [15:0]       inv_temp,   // bf16  1/T
    input  wire [15:0]       topp,       // bf16  nucleus threshold p
    // logit load (unit pulls one bf16 logit / cycle, vocab order)
    output reg               load_req,
    input  wire [15:0]       logit_in,
    input  wire              logit_valid,
    // result
    output reg  [IDXW-1:0]   token_o,
    output reg               done,
    output reg               busy
);
    `include "glm_fp_pipe_lat.vh"

    // effective number of kept candidates = min(TOPK, VOCAB)
    localparam integer KEFF = (TOPK < VOCAB) ? TOPK : VOCAB;
    localparam integer KW   = (KEFF <= 1) ? 1 : $clog2(KEFF);
    localparam integer VW   = (VOCAB <= 1) ? 1 : $clog2(VOCAB);
    localparam integer VC   = (VOCAB <= 1) ? 1 : $clog2(VOCAB+1); // counts 0..VOCAB
    localparam integer KC   = (KEFF  <= 1) ? 1 : $clog2(KEFF+1);  // counts 0..KEFF

    localparam [31:0] FP32_ONE = 32'h3F800000;
    localparam [15:0] BF16_INF = 16'h7F80;

    // ===================================================================
    //  logit buffer (fp32, widened from bf16) + temperature-scaled scores
    // ===================================================================
    reg  [31:0] zbuf [0:VOCAB-1];   // temperature-scaled fp32 logits

    // ===================================================================
    //  TOP-K SELECTOR  (shared topk_select), fed the scaled fp32 logits.
    //  Returns KEFF indices + scores in DESCENDING score order (slot 0 = max).
    // ===================================================================
    reg                       tk_start;
    wire                      tk_load_req;
    reg  [31:0]               tk_score_in;
    reg                       tk_score_valid;
    wire [TOPK*VW-1:0]        tk_idx;
    wire [TOPK*32-1:0]        tk_score;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [TOPK-1:0]           tk_valid;   // surplus-slot flags (unused: KEFF only)
    wire [VOCAB-1:0]          tk_mask;    // union mask (unused: we use the order)
    wire                      tk_busy;    // status mirror (unused)
    /* verilator lint_on UNUSEDSIGNAL */
    wire                      tk_done;

    topk_select #(
        .N(VOCAB), .K(TOPK), .SCORE_W(32), .LANES_IN(1)
    ) u_topk (
        .clk(clk), .rst(rst), .start(tk_start),
        .load_req(tk_load_req),
        .score_in(tk_score_in), .score_valid(tk_score_valid),
        .sel_idx_o(tk_idx), .sel_score_o(tk_score), .sel_valid_o(tk_valid),
        .mask_o(tk_mask), .busy(tk_busy), .done(tk_done)
    );

    // captured top-k results (descending score order)
    reg  [VW-1:0]  kidx  [0:KEFF-1];   // vocab index of kept slot s
    reg  [31:0]    kscore[0:KEFF-1];   // fp32 scaled score of kept slot s

    // ===================================================================
    //  SOFTMAX over the KEFF kept scores (shared glm_softmax), LANES=1.
    //  Row = KEFF bf16-narrowed kept scores in descending order.
    // ===================================================================
    reg                  sm_start;
    reg                  sm_in_valid;
    reg  [15:0]          sm_x_in;
    /* verilator lint_off UNUSEDSIGNAL */
    wire                 sm_busy;    // status mirror (unused)
    wire                 sm_done;    // coincides with last out_valid (unused)
    /* verilator lint_on UNUSEDSIGNAL */
    wire                 sm_out_valid;
    wire [15:0]          sm_p_out;

    glm_softmax #( .LEN(KEFF), .LANES(1) ) u_softmax (
        .clk(clk), .rst(rst), .start(sm_start),
        .in_valid(sm_in_valid), .x_in(sm_x_in),
        .busy(sm_busy), .out_valid(sm_out_valid), .p_out(sm_p_out),
        .done(sm_done)
    );

    // captured softmax probabilities (descending order, fp32 for the walk)
    reg  [31:0]    pprob [0:KEFF-1];

    // ===================================================================
    //  LFSR PRNG  ->  u in [0,1)  (deterministic, seeded by SEED).
    //  Galois LFSR, taps for a maximal-length sequence at LFSR_W=32.
    //  u is built from the top mantissa bits of the LFSR as a fp32 in [0,1).
    // ===================================================================
    reg  [LFSR_W-1:0] lfsr;
    // 32-bit maximal-length tap mask (x^32+x^22+x^2+x+1): 0x80200003.
    localparam [31:0] TAP32 = 32'h80200003;

    // next-state of the Galois LFSR (one step).  Pure comb of `lfsr`.
    wire              lfsr_lsb = lfsr[0];
    wire [LFSR_W-1:0] lfsr_next =
        lfsr_lsb ? ((lfsr >> 1) ^ TAP32[LFSR_W-1:0]) : (lfsr >> 1);

    // u in [0,1): take the high 23 bits of the LFSR as the fp32 mantissa of a
    // value in [1,2), then subtract 1.0.  Deterministic, dense on the grid.
    localparam [31:0] FP32_NEG_ONE = 32'hBF800000;   // -1.0
    // pad amount when LFSR is narrower than the 23-bit fraction (clamped >=0 so
    // the replication count is always a legal non-negative constant).
    localparam integer FRAC_PAD = (LFSR_W >= 23) ? 0 : (23 - LFSR_W);
    function automatic [31:0] lfsr_to_u(input [LFSR_W-1:0] s);
        reg [22:0]              frac;
        reg [31:0]              onepu;
        reg [LFSR_W+FRAC_PAD-1:0] wide;   // s, then left-padded to >=23 bits
        begin
            // Build a >=23-bit word: the LFSR bits in the HIGH positions, zeros
            // in the low FRAC_PAD bits (a no-op shift when FRAC_PAD==0), then
            // take the top 23 bits as the fp32 fraction.
            wide = {{(FRAC_PAD){1'b0}}, s};       // zero-extend high (width ok)
            wide = wide << FRAC_PAD;              // move LFSR bits to the top
            frac = wide[LFSR_W+FRAC_PAD-1 -: 23];
            onepu     = {1'b0, 8'd127, frac};        // 1.frac in [1,2)
            lfsr_to_u = fp32_add(onepu, FP32_NEG_ONE);  // u = 1.frac - 1.0 in [0,1)
        end
    endfunction

    // ===================================================================
    //  greedy decision: explicit `greedy` input OR inv_temp == +inf (T=0).
    // ===================================================================
    reg greedy_mode;   // latched at start

    // ===================================================================
    //  nucleus (top-p) enable + threshold (fp32), latched at start.
    //  Disabled when topp >= 1.0.
    // ===================================================================
    reg        topp_en;
    reg [31:0] topp_f;

    // ===================================================================
    //  FSM
    // ===================================================================
    localparam [3:0]
        S_IDLE   = 4'd0,
        S_LOAD   = 4'd1,   // pull VOCAB bf16 logits -> scale by inv_temp -> zbuf
        S_TKSTRT = 4'd2,   // pulse topk start
        S_TKFEED = 4'd3,   // stream scaled fp32 scores into topk
        S_TKWAIT = 4'd4,   // wait topk done, capture kidx/kscore
        S_GREEDY = 4'd5,   // emit argmax (kidx[0]) and finish
        S_SMSTRT = 4'd6,   // pulse softmax start
        S_SMFEED = 4'd7,   // stream KEFF kept bf16 scores into softmax
        S_SMWAIT = 4'd8,   // capture KEFF probabilities (descending)
        S_NUCLEUS= 4'd9,   // optional top-p: find nucleus cutoff slot
        S_DRAW   = 4'd10,  // pick u, walk cumulative prob, choose token
        S_DONE   = 4'd11;
    reg [3:0] state;

    reg [VC-1:0]  vbeat;     // load beat / topk-feed index (0..VOCAB)
    reg [KC-1:0]  kfeed;     // softmax feed index (0..KEFF)
    reg [KC-1:0]  kcap;      // softmax capture index (0..KEFF)
    reg [KC-1:0]  walk;      // draw/nucleus walk index (0..KEFF)
    reg [KC-1:0]  nuc_last;  // last kept slot index for nucleus (inclusive)

    reg [31:0]    cum;       // running cumulative probability (fp32)
    reg [31:0]    uval;      // the drawn u (fp32) latched at draw start

    // temperature-scaled score for the logit currently being fed to topk.
    // (we pre-store zbuf during LOAD so topk-feed is a plain memory read.)

    integer i;

    always @(posedge clk) begin
        if (rst) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            load_req       <= 1'b0;
            token_o        <= {IDXW{1'b0}};
            tk_start       <= 1'b0;
            tk_score_valid <= 1'b0;
            tk_score_in    <= 32'b0;
            sm_start       <= 1'b0;
            sm_in_valid    <= 1'b0;
            sm_x_in        <= 16'b0;
            vbeat          <= {VC{1'b0}};
            kfeed          <= {KC{1'b0}};
            kcap           <= {KC{1'b0}};
            walk           <= {KC{1'b0}};
            nuc_last       <= {KC{1'b0}};
            cum            <= 32'b0;
            uval           <= 32'b0;
            greedy_mode    <= 1'b0;
            topp_en        <= 1'b0;
            topp_f         <= 32'b0;
            lfsr           <= SEED[LFSR_W-1:0];
        end else begin
            // default single-cycle strobes (no latch)
            done           <= 1'b0;
            load_req       <= 1'b0;
            tk_start       <= 1'b0;
            tk_score_valid <= 1'b0;
            sm_start       <= 1'b0;
            sm_in_valid    <= 1'b0;

            case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy        <= 1'b1;
                    vbeat       <= {VC{1'b0}};
                    // latch sample-mode controls
                    greedy_mode <= greedy || (inv_temp == BF16_INF);
                    topp_en     <= !(fp32_gt_eq(bf16_to_fp32(topp), FP32_ONE));
                    topp_f      <= bf16_to_fp32(topp);
                    load_req    <= 1'b1;
                    state       <= S_LOAD;
                end
            end

            // -----------------------------------------------------------
            // pull VOCAB bf16 logits, widen to fp32, scale by inv_temp.
            //   z_i = logit_i * (1/T).   (fp32_mul is combinational here.)
            S_LOAD: begin
                busy     <= 1'b1;
                load_req <= 1'b1;
                if (logit_valid) begin
                    // In greedy mode argmax is temperature-invariant (any T>0
                    // preserves the order), and inv_temp may be +inf (T=0) which
                    // would push every scaled logit to +inf/NaN.  So scale by 1.0
                    // (raw logits) when greedy; otherwise multiply by 1/T.
                    zbuf[vbeat[VW-1:0]] <= greedy_mode
                        ? bf16_to_fp32(logit_in)
                        : fp32_mul(bf16_to_fp32(logit_in), bf16_to_fp32(inv_temp));
                    if (vbeat == VC'(VOCAB-1)) begin
                        load_req <= 1'b0;
                        vbeat    <= {VC{1'b0}};
                        state    <= S_TKSTRT;
                    end else begin
                        vbeat <= vbeat + 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------
            S_TKSTRT: begin
                busy     <= 1'b1;
                tk_start <= 1'b1;
                vbeat    <= {VC{1'b0}};
                state    <= S_TKFEED;
            end

            // -----------------------------------------------------------
            // feed the VOCAB scaled fp32 scores to topk_select, one/cycle,
            // when it pulls (load_req).
            S_TKFEED: begin
                busy <= 1'b1;
                if (tk_load_req) begin
                    tk_score_in    <= zbuf[vbeat[VW-1:0]];
                    tk_score_valid <= 1'b1;
                    if (vbeat == VC'(VOCAB-1)) begin
                        vbeat <= {VC{1'b0}};
                        state <= S_TKWAIT;
                    end else begin
                        vbeat <= vbeat + 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------
            // wait for topk done; capture the KEFF kept idx/score (slot 0 =
            // largest).  Surplus slots (when TOPK>VOCAB) are masked invalid
            // by topk; we only ever use the first KEFF=min(TOPK,VOCAB).
            S_TKWAIT: begin
                busy <= 1'b1;
                if (tk_done) begin
                    for (i = 0; i < KEFF; i = i + 1) begin
                        kidx[i]   <= tk_idx[i*VW +: VW];
                        kscore[i] <= tk_score[i*32 +: 32];
                    end
                    if (greedy_mode) state <= S_GREEDY;
                    else             state <= S_SMSTRT;
                end
            end

            // -----------------------------------------------------------
            // GREEDY / argmax: slot 0 of topk is the highest-scored token.
            S_GREEDY: begin
                busy    <= 1'b1;
                token_o <= IDXW'(kidx[0]);
                state   <= S_DONE;
            end

            // -----------------------------------------------------------
            S_SMSTRT: begin
                busy     <= 1'b1;
                sm_start <= 1'b1;
                kfeed    <= {KC{1'b0}};
                kcap     <= {KC{1'b0}};
                state    <= S_SMFEED;
            end

            // -----------------------------------------------------------
            // stream the KEFF kept fp32 scores (narrowed to bf16) into the
            // softmax, in descending order, one lane/beat (LANES=1).
            S_SMFEED: begin
                busy <= 1'b1;
                if (kfeed < KC'(KEFF)) begin
                    sm_in_valid <= 1'b1;
                    sm_x_in     <= fp32_to_bf16(kscore[kfeed[KW-1:0]]);
                    if (kfeed == KC'(KEFF-1)) begin
                        kfeed <= {KC{1'b0}};
                        state <= S_SMWAIT;
                    end else begin
                        kfeed <= kfeed + 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------
            // capture KEFF softmax probabilities (descending order),
            // widened to fp32 for the cumulative walk.
            S_SMWAIT: begin
                busy <= 1'b1;
                if (sm_out_valid) begin
                    pprob[kcap[KW-1:0]] <= bf16_to_fp32(sm_p_out);
                    if (kcap == KC'(KEFF-1)) begin
                        kcap     <= {KC{1'b0}};
                        cum      <= 32'b0;
                        walk     <= {KC{1'b0}};
                        nuc_last <= KC'(KEFF-1);    // default nucleus: keep all
                        if (topp_en) state <= S_NUCLEUS;
                        else         state <= S_DRAW;
                    end else begin
                        kcap <= kcap + 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------
            // TOP-P (nucleus): walk descending probs, accumulate; the FIRST
            // slot whose running cumsum >= topp is the LAST kept slot.  Tokens
            // after it are dropped (the draw walk stops at nuc_last).  `cum`
            // here is the prefix sum INCLUDING the current slot.
            S_NUCLEUS: begin
                busy <= 1'b1;
                cum  <= fp32_add(cum, pprob[walk[KW-1:0]]);   // running prefix
                if (fp32_gt_eq(fp32_add(cum, pprob[walk[KW-1:0]]), topp_f) ||
                    walk == KC'(KEFF-1)) begin
                    nuc_last <= walk;                // last kept slot (inclusive)
                    cum      <= 32'b0;               // reset for the draw pass
                    walk     <= {KC{1'b0}};
                    state    <= S_DRAW;
                end else begin
                    walk <= walk + 1'b1;
                end
            end

            // -----------------------------------------------------------
            // MULTINOMIAL DRAW.  `cum` is the running prefix sum INCLUDING the
            // current slot.  We emit the FIRST slot whose prefix-cumsum >= u,
            // where u = lfsr_u * nucleus_mass (so u lives in [0, kept-mass) and
            // the walk over the kept set [0..nuc_last] is exact even when top-p
            // dropped mass).  The LFSR advances once per draw (determinism).
            //   walk==0 special-cases latching u and advancing the LFSR.
            S_DRAW: begin
                busy <= 1'b1;
                if (walk == {KC{1'b0}}) begin
                    uval <= fp32_mul(lfsr_to_u(lfsr), nucleus_mass());
                    lfsr <= lfsr_next;                       // prime next draw
                    cum  <= pprob[0];                        // prefix incl slot 0
                    if (fp32_gt_eq(pprob[0],
                            fp32_mul(lfsr_to_u(lfsr), nucleus_mass())) ||
                        nuc_last == {KC{1'b0}}) begin
                        token_o <= IDXW'(kidx[0]);
                        state   <= S_DONE;
                    end else begin
                        walk <= walk + 1'b1;
                    end
                end else begin
                    cum <= fp32_add(cum, pprob[walk[KW-1:0]]);
                    if (fp32_gt_eq(fp32_add(cum, pprob[walk[KW-1:0]]), uval) ||
                        walk == nuc_last) begin
                        token_o <= IDXW'(kidx[walk[KW-1:0]]);
                        state   <= S_DONE;
                    end else begin
                        walk <= walk + 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------
            S_DONE: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // total nucleus probability mass = Sum of kept slots [0..nuc_last].
    // Pure comb fold over pprob[] up to nuc_last (constant-bounded loop).
    // Used to scale u so the draw is correct even when the nucleus drops mass.
    // -----------------------------------------------------------------------
    function automatic [31:0] nucleus_mass;
        reg [31:0] acc;
        integer    q;
        begin
            acc = 32'b0;
            for (q = 0; q < KEFF; q = q + 1)
                if (q <= nuc_last) acc = fp32_add(acc, pprob[q]);
            nucleus_mass = acc;
        end
    endfunction

    // -----------------------------------------------------------------------
    // fp32 "a >= b" under IEEE ordering (sign+magnitude aware; +0==-0; NaN
    // treated as smallest).  Used for nucleus + cumulative-walk comparisons.
    // -----------------------------------------------------------------------
    function automatic fp32_gt_eq(input [31:0] a, input [31:0] b);
        reg sa, sb;
        reg [30:0] ma, mb;
        reg a_nan, b_nan;
        begin
            sa = a[31]; sb = b[31];
            ma = a[30:0]; mb = b[30:0];
            a_nan = (a[30:23] == 8'hFF) && (a[22:0] != 0);
            b_nan = (b[30:23] == 8'hFF) && (b[22:0] != 0);
            if (a_nan)            fp32_gt_eq = 1'b0;        // NaN >= x : false
            else if (b_nan)       fp32_gt_eq = 1'b1;        // x >= NaN : true
            else if (ma==0 && mb==0) fp32_gt_eq = 1'b1;     // +-0 >= +-0
            else if (sa != sb)    fp32_gt_eq = (sa == 1'b0);// pos >= neg
            else if (sa == 1'b0)  fp32_gt_eq = (ma >= mb);  // both >=0
            else                  fp32_gt_eq = (ma <= mb);  // both <0
        end
    endfunction

endmodule
