`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// dsa_indexer.v  --  GLM-5.2 DSA "IndexShare" sparse-attention INDEXER
//                                                       (ACCEL_GLM52 §4.2 / §3 row 7-8)
//----------------------------------------------------------------------------
// FUNCTION  (the lightweight key-selection front of DSA sparse attention)
//   A CHEAP indexer decides WHICH past keys the expensive MLA attention will
//   actually read.  For a single query it scores the query's small "index"
//   vector q_idx against every causal key's index vector k_idx[j] with a cheap
//   dot product in a SMALL index dimension (IDX_DIM, e.g. 16 -- vs the 512-wide
//   real attention head), then keeps the TOPK highest-scoring keys.  The output
//   is the per-query list of selected key INDICES (+ a valid count) that the
//   downstream scatter_gather (§4.2 "cache gather") uses to read exactly TOPK
//   rows of the KV cache, capping attention FLOPs regardless of context length.
//
//       score_j = Σ_{d=0..IDX_DIM-1}  q_idx[d] * k_idx[j][d]     (j = 0..S-1)
//       (sel_idx, count) = TOP-K( score_0 .. score_{S-1} )       (lower-idx tie)
//
//   bf16 index operands (q_idx / k_idx), fp32 score accumulate (§6 numerics:
//   bf16 storage / fp32 reduce).  The dot product reuses the foundation
//   fp32_mac_pipe (a*b+c, 0-ULP to the glm_fp.vh contract); the selection
//   reuses topk_select UNCHANGED -- nothing re-implemented.
//
//----------------------------------------------------------------------------
// DENSE FALLBACK  (S <= TOPK  =>  the indexer is a NO-OP, §4.2 "Dense fallback")
//   When the causal set is no larger than the budget, EVERY key is kept: the
//   output is simply indices 0,1,..,S-1 in order, count = S.  This is exact
//   dense attention -- no scoring/selection error -- and it is detected
//   structurally from S (registered at start), NOT from the scores.
//
//----------------------------------------------------------------------------
// CAUSALITY  (the caller supplies exactly the causal set)
//   Keys k_idx[0..S-1] are the PAST/CURRENT keys for this query (the caller
//   streams exactly the causal window; future keys are simply not presented),
//   so NO extra causal mask is needed in this unit.  Unused score slots
//   (indices S..S_MAX-1) are filled with -inf so topk_select can never pick
//   them (its comparator treats -inf/NaN as the smallest value -- a never-
//   selectable lane), keeping the selection over precisely the S real keys.
//   NOTE: if a design MANDATES always keeping the most-recent key(s) (a sliding
//   "recent window" on top of top-k, §3 row 8 "+causal recent window"), that is
//   a CONTROL overlay added by the sequencer (e.g. forcing the last index into
//   the set); for this slice pure top-K over the provided causal set is the
//   contract and is exact.  IndexShare reuse (freq 4 / offset 3) is likewise a
//   sequencer concern -- THIS unit just produces one fresh per-query list.
//
//----------------------------------------------------------------------------
// INTERFACE  (start / key-stream / done handshake; deterministic latency)
//   clk, rst              : synchronous, ACTIVE-HIGH reset.
//   start                 : 1-cycle pulse.  Latches q_idx and S, begins.
//   q_idx [IDX_DIM*16]    : the query index vector (IDX_DIM bf16 lanes), latched
//                           at start.  lane d at q_idx[16*d +: 16].
//   s_len                 : sequence length S (number of causal keys, <=S_MAX),
//                           latched at start.
//   --- key index-vector stream (unit PULLS one key-vector per beat) ---
//   key_req               : high while the unit wants the next key vector.
//   k_idx [IDX_DIM*16]    : the current key's index vector (IDX_DIM bf16 lanes),
//                           lane d at k_idx[16*d +: 16].  Read when key_valid.
//   key_valid             : producer asserts when k_idx holds key `key_idx`.
//   key_idx [IDXW]        : which causal key (0..S-1) the unit is requesting now.
//   --- results (held stable from done until next start) ---
//   sel_idx  [TOPK*IDXW]  : selected key indices, slot s at sel_idx[IDXW*s +:].
//                           SPARSE (S>TOPK): top-K by score, slot 0 = highest
//                           score, descending (topk_select order, lower-index
//                           tie-break).  DENSE (S<=TOPK): 0,1,..,S-1 in order.
//   sel_count [IDXW+1]    : number of VALID slots = min(S, TOPK).
//   done                  : 1-cycle pulse; sel_idx/sel_count valid.
//   busy                  : high from start until done.
//
//----------------------------------------------------------------------------
// LATENCY  (deterministic, data-independent for a given S)
//   Let MAC = `FP_MAC_LAT (fp32_mac_pipe latency, 7).  Per query:
//     load q+S         : 1
//     SPARSE (S>TOPK):
//       score S keys   : S*IDX_DIM beats issued (one MAC/beat) + MAC drain,
//                        sequentially, key-by-key  ->  S*IDX_DIM + MAC + ~2/key
//       topk_select    : S_MAX load beats + min(S_MAX,TOPK) extract + handshake
//     DENSE  (S<=TOPK) : no scoring/select -- emit 0..S-1 directly  (~1 cycle)
//   Every count is a function of (IDX_DIM, S, S_MAX, TOPK) -- no data-dependent
//   branch on the SCORE values -- so the latency is fixed per (S, params).
//
//----------------------------------------------------------------------------
// STYLE / CORRECTNESS INVARIANTS
//   * `timescale + this header ; synchronous ACTIVE-HIGH reset.
//   * NO latch (every reg assigned on every path); NO combinational loop (the
//     MAC accumulate feeds back ONLY through the pipeline registers of
//     fp32_mac_pipe and the score_mem register; control is a Moore FSM).
//   * Reuses fp32_mac_pipe + topk_select UNCHANGED.  fp32 score accumulate.
//============================================================================
module dsa_indexer #(
    parameter integer IDX_DIM = 16,    // small index/scoring dim (real ~128)
    parameter integer S_MAX   = 32,    // max causal keys (real up to 1M ring)
    parameter integer TOPK    = 8,     // index_topk budget (real 2048)
    // IDXW DERIVED ($clog2(S_MAX)); exposed only to size the index ports.
    // Do NOT override -- always leave default.
    parameter integer IDXW    = (S_MAX <= 1) ? 1 : $clog2(S_MAX)
)(
    input  wire                     clk,
    input  wire                     rst,        // sync, active-high

    // ---- control handshake ----
    input  wire                     start,      // begin one query's index pass
    output reg                      busy,
    output reg                      done,       // 1-cycle pulse, outputs valid

    // ---- query index vector + sequence length (latched at start) ----
    input  wire [IDX_DIM*16-1:0]    q_idx,      // IDX_DIM bf16 query index lanes
    input  wire [IDXW:0]            s_len,      // S = causal key count (<= S_MAX)

    // ---- key index-vector pull (unit requests one key-vector per beat) ----
    output reg                      key_req,    // need the next key vector
    output reg  [IDXW-1:0]          key_idx,    // which causal key (0..S-1)
    input  wire [IDX_DIM*16-1:0]    k_idx,      // IDX_DIM bf16 key index lanes
    input  wire                     key_valid,  // k_idx holds key `key_idx`

    // ---- selected key-index list (held from done until next start) ----
    output reg  [TOPK*IDXW-1:0]     sel_idx,    // selected key indices
    output reg  [IDXW:0]            sel_count   // number of valid slots = min(S,TOPK)
);
    `include "glm_fp.vh"

    //------------------------------------------------------------------------
    // derived sizes / constants
    //------------------------------------------------------------------------
    localparam integer DIW = (IDX_DIM <= 1) ? 1 : $clog2(IDX_DIM); // dim counter
    // -inf : the strict lower bound topk_select treats as never-selectable, used
    // to pad the unused score slots (keys S..S_MAX-1) so selection sees only the
    // S real keys (matches topk_select's NEG_INF / NaN==smallest contract).
    localparam [31:0] NEG_INF = 32'hFF800000;

    //------------------------------------------------------------------------
    // latched query + control
    //------------------------------------------------------------------------
    reg [15:0]  qbuf [0:IDX_DIM-1];      // query index vector (bf16), latched
    reg [IDXW:0] s_reg;                  // S, latched at start
    integer qi;

    //------------------------------------------------------------------------
    // per-key score storage (fp32).  score_mem[j] = dot(q_idx, k_idx[j]).
    // Unused slots (j >= S) hold NEG_INF so topk_select never picks them.
    //------------------------------------------------------------------------
    reg [31:0]  score_mem [0:S_MAX-1];

    //------------------------------------------------------------------------
    // (1) DOT-PRODUCT engine : one fp32_mac_pipe, accumulates over IDX_DIM.
    //   For the current key, we issue IDX_DIM MAC beats  acc = q[d]*k[d] + acc,
    //   feeding the running accumulator back as `c`.  The accumulator update is
    //   sequential (one term per emitted result), so we issue a term only after
    //   the previous term's result has landed -- a simple, deterministic,
    //   in-order accumulate (throughput is traded for zero hazard logic; IDX_DIM
    //   is small).  acc starts at +0.0.
    //------------------------------------------------------------------------
    reg          mac_valid_in;
    reg  [31:0]  mac_a, mac_b, mac_c;
    wire         mac_valid_out;
    wire [31:0]  mac_result;
    fp32_mac_pipe u_mac (
        .clk(clk), .rst(rst), .valid_in(mac_valid_in),
        .a(mac_a), .b(mac_b), .c(mac_c),
        .valid_out(mac_valid_out), .result(mac_result)
    );

    reg [31:0]   acc;                    // running fp32 dot accumulator
    reg [DIW:0]  dim_issue;              // index-dim terms ISSUED for this key
    reg [DIW:0]  dim_done;               // index-dim results LANDED for this key

    // current key's bf16 lanes, latched when key_valid.
    reg [15:0]   kbuf [0:IDX_DIM-1];
    integer ki;

    //------------------------------------------------------------------------
    // (2) TOP-K selector : pick TOPK highest scores -> their indices.
    //   topk_select pulls scores 1/cycle (LANES_IN=1); we answer
    //   score_mem[tk_addr].  N = S_MAX (we always present S_MAX scores; the
    //   j>=S slots are NEG_INF so only the S real keys can be selected).
    //------------------------------------------------------------------------
    reg                  tk_start;
    reg                  tk_score_valid;
    reg  [31:0]          tk_score_in;
    wire                 tk_load_req;
    wire [TOPK*IDXW-1:0] tk_sel_idx;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [TOPK*32-1:0]   tk_sel_score;   // selected scores (unused: indices only)
    wire [TOPK-1:0]      tk_sel_valid;
    wire [S_MAX-1:0]     tk_mask;
    wire                 tk_busy;
    /* verilator lint_on UNUSEDSIGNAL */
    wire                 tk_done;
    topk_select #(.N(S_MAX), .K(TOPK), .SCORE_W(32), .LANES_IN(1)) u_topk (
        .clk(clk), .rst(rst), .start(tk_start),
        .load_req(tk_load_req), .score_in(tk_score_in),
        .score_valid(tk_score_valid),
        .sel_idx_o(tk_sel_idx), .sel_score_o(tk_sel_score),
        .sel_valid_o(tk_sel_valid), .mask_o(tk_mask),
        .busy(tk_busy), .done(tk_done)
    );

    // score-load address: which score_mem slot we hand topk this beat (0..S_MAX).
    localparam integer SAW = (S_MAX > 1) ? $clog2(S_MAX) : 1;
    reg [SAW:0]  tk_addr;

    //------------------------------------------------------------------------
    // FSM
    //------------------------------------------------------------------------
    localparam [2:0] S_IDLE   = 3'd0,   // wait start; latch q,S
                     S_SCORE  = 3'd1,   // dot-product every key (sequential MAC)
                     S_TKL    = 3'd2,   // feed topk score-pull, wait done
                     S_DENSE  = 3'd3,   // emit 0..S-1 (no-op fallback)
                     S_DONE   = 3'd4;
    reg [2:0]   state;
    reg [IDXW:0] kcnt;                  // which key we are currently scoring

    integer t;

    always @(posedge clk) begin
        if (rst) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            key_req        <= 1'b0;
            key_idx        <= {IDXW{1'b0}};
            s_reg          <= {(IDXW+1){1'b0}};
            mac_valid_in   <= 1'b0;
            mac_a          <= 32'b0;
            mac_b          <= 32'b0;
            mac_c          <= 32'b0;
            acc            <= 32'b0;
            dim_issue      <= {(DIW+1){1'b0}};
            dim_done       <= {(DIW+1){1'b0}};
            kcnt           <= {(IDXW+1){1'b0}};
            tk_start       <= 1'b0;
            tk_score_valid <= 1'b0;
            tk_score_in    <= 32'b0;
            tk_addr        <= {(SAW+1){1'b0}};
            sel_idx        <= {TOPK*IDXW{1'b0}};
            sel_count      <= {(IDXW+1){1'b0}};
            for (qi = 0; qi < IDX_DIM; qi = qi + 1) qbuf[qi] <= 16'b0;
            for (ki = 0; ki < IDX_DIM; ki = ki + 1) kbuf[ki] <= 16'b0;
            for (t  = 0; t  < S_MAX;   t  = t  + 1) score_mem[t] <= NEG_INF;
        end else begin
            // ---- pulse defaults (deassert) ----
            done           <= 1'b0;
            mac_valid_in   <= 1'b0;
            tk_start       <= 1'b0;
            tk_score_valid <= 1'b0;

            case (state)
            // ----------------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy  <= 1'b1;
                    // latch query vector + S
                    for (qi = 0; qi < IDX_DIM; qi = qi + 1)
                        qbuf[qi] <= q_idx[16*qi +: 16];
                    s_reg <= s_len;
                    // DENSE no-op when S <= TOPK.
                    if (s_len <= TOPK[IDXW:0]) begin
                        state   <= S_DENSE;
                    end else begin
                        // pre-fill all score slots with -inf (so j>=S can never
                        // win).  Only the SPARSE path reads score_mem (via topk),
                        // so the DENSE path never needs this prefill.
                        for (t = 0; t < S_MAX; t = t + 1) score_mem[t] <= NEG_INF;
                        // begin scoring key 0
                        kcnt      <= {(IDXW+1){1'b0}};
                        acc       <= 32'b0;
                        dim_issue <= {(DIW+1){1'b0}};
                        dim_done  <= {(DIW+1){1'b0}};
                        key_req   <= 1'b1;            // pull key 0's vector
                        key_idx   <= {IDXW{1'b0}};
                        state     <= S_SCORE;
                    end
                end
            end

            // ----------------------------------------------------------------
            // SCORE : for each key j=0..S-1 compute score_j = dot(q, k_j).
            //   - When key_valid lands, latch the key's IDX_DIM bf16 lanes and
            //     start issuing MAC terms.
            //   - Issue one term per cycle (acc = q[d]*k[d] + acc) IN ORDER:
            //     issue term `dim_issue` only after term `dim_issue-1` result has
            //     landed, so `c` (the running acc) is always the up-to-date sum.
            //   - When all IDX_DIM results have landed, store acc into
            //     score_mem[j], advance to key j+1 (or launch topk after the last).
            // ----------------------------------------------------------------
            S_SCORE: begin
                // (a) capture the requested key vector when the producer answers.
                if (key_req && key_valid) begin
                    for (ki = 0; ki < IDX_DIM; ki = ki + 1)
                        kbuf[ki] <= k_idx[16*ki +: 16];
                    key_req   <= 1'b0;               // got it; stop requesting
                    acc       <= 32'b0;              // accumulator for this key
                    dim_issue <= {(DIW+1){1'b0}};
                    dim_done  <= {(DIW+1){1'b0}};
                end

                // (b) issue the next MAC term once we hold the key vector and the
                //     previous term's result has landed (in-order accumulate).
                if (!key_req && (dim_issue < IDX_DIM[DIW:0]) &&
                    (dim_issue == dim_done)) begin
                    mac_valid_in <= 1'b1;
                    mac_a        <= bf16_to_fp32(qbuf[dim_issue[DIW-1:0]]);
                    mac_b        <= bf16_to_fp32(kbuf[dim_issue[DIW-1:0]]);
                    mac_c        <= acc;             // running sum so far
                    dim_issue    <= dim_issue + 1'b1;
                end

                // (c) absorb a landed MAC result into the accumulator.
                if (mac_valid_out) begin
                    acc      <= mac_result;
                    dim_done <= dim_done + 1'b1;
                end

                // (d) key complete: all IDX_DIM terms landed.  Store the score
                //     and move on.  (acc holds the final dot product this cycle
                //     when the last result just landed -> use mac_result.)
                if (!key_req && (dim_done == IDX_DIM[DIW:0] - 1'b1) &&
                    mac_valid_out) begin
                    score_mem[kcnt[IDXW-1:0]] <= mac_result;
                    if (kcnt == s_reg - 1'b1) begin
                        // scored the last causal key -> launch top-K.
                        tk_start <= 1'b1;
                        tk_addr  <= {(SAW+1){1'b0}};
                        state    <= S_TKL;
                    end else begin
                        kcnt    <= kcnt + 1'b1;      // next key
                        key_req <= 1'b1;
                        key_idx <= kcnt[IDXW-1:0] + 1'b1;
                    end
                end
            end

            // ----------------------------------------------------------------
            // TKL : feed topk's score-pull (1/cycle), wait its done, capture the
            //   TOPK selected indices.  topk returns indices in descending-score
            //   order (lower-index tie-break).  count = TOPK (S>TOPK here).
            // ----------------------------------------------------------------
            S_TKL: begin
                if (tk_load_req) begin
                    tk_score_valid <= 1'b1;
                    tk_score_in    <= score_mem[tk_addr[SAW-1:0]];
                    tk_addr        <= tk_addr + 1'b1;
                end
                if (tk_done) begin
                    sel_idx   <= tk_sel_idx;
                    sel_count <= TOPK[IDXW:0];       // S>TOPK -> exactly TOPK valid
                    state     <= S_DONE;
                end
            end

            // ----------------------------------------------------------------
            // DENSE : S <= TOPK no-op.  Emit indices 0,1,..,S-1 in order, the
            //   surplus slots (s>=S) zero-filled but flagged by sel_count=S.
            // ----------------------------------------------------------------
            S_DENSE: begin
                for (t = 0; t < TOPK; t = t + 1) begin
                    if (t < s_reg)
                        sel_idx[IDXW*t +: IDXW] <= t[IDXW-1:0];
                    else
                        sel_idx[IDXW*t +: IDXW] <= {IDXW{1'b0}};
                end
                sel_count <= s_reg;                  // all S kept
                state     <= S_DONE;
            end

            // ----------------------------------------------------------------
            S_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
