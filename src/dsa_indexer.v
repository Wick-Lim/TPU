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
// LATENCY  (deterministic, data-independent for a given S)        [PARALLELIZED]
//   Let MAC = `FP_MAC_LAT (fp32_mac_pipe latency, 7).  The dot product is the
//   throughput wall (at 1M context the indexer scores up to ~S keys); the naive
//   build issued the IDX_DIM MAC terms of ONE key strictly in order -- it waited
//   ~MAC cycles between terms (issue term d only after term d-1 landed), so each
//   key cost ~IDX_DIM*MAC and S keys cost ~S*IDX_DIM*MAC.  THIS build SOFTWARE-
//   PIPELINES across keys: it processes keys in GROUPS of LANES, and issues ONE
//   MAC term every cycle by ROUND-ROBINing the LANES independent in-flight keys
//   through the single shared fp32_mac_pipe.  Each key still walks its OWN terms
//   strictly in order (acc = q[d]*k[d] + acc, c = its own running sum), so its
//   dot product is the IDENTICAL fp32 FMA chain -- the scores are BIT-EXACT to
//   the serial build.  When LANES >= MAC the pipe never stalls -> throughput is
//   1 term/cycle, so scoring costs ~S*IDX_DIM (down from ~S*IDX_DIM*MAC, a ~MACx
//   = ~7x cut at the wall) plus one MAC drain + ~1 fetch/key.  A short tail group
//   (gsize < MAC) simply stalls per-lane as before -- still bit-exact.
//     load q+S         : 1
//     SPARSE (S>TOPK):  ceil(S/LANES) groups; per group fetch ~gsize + score
//                       IDX_DIM*gsize (1/cycle when gsize>=MAC) + MAC drain
//                       ->  ~S*IDX_DIM + ~S + ceil(S/LANES)*MAC   (vs S*IDX_DIM*MAC)
//       topk_select    : S_MAX load beats + min(S_MAX,TOPK) extract + handshake
//     DENSE  (S<=TOPK) : no scoring/select -- emit 0..S-1 directly  (~1 cycle)
//   Every count is a function of (IDX_DIM, S, S_MAX, TOPK, LANES) -- no data-
//   dependent branch on the SCORE values -- so the latency is fixed per (S,params).
//
//----------------------------------------------------------------------------
// STYLE / CORRECTNESS INVARIANTS
//   * `timescale + this header ; synchronous ACTIVE-HIGH reset.
//   * NO latch (every reg assigned on every path); NO combinational loop (each
//     key's accumulate feeds back ONLY through the pipeline registers of
//     fp32_mac_pipe and the per-lane acc register; control is a Moore FSM).
//   * Reuses fp32_mac_pipe + topk_select UNCHANGED.  fp32 score accumulate.
//   * BIT-EXACT to the serial build: interleaving keys does NOT reorder any
//     single key's term-accumulation order, so every score is identical; only
//     the IDLE pipe slots between a key's terms are now filled by OTHER keys.
//============================================================================
module dsa_indexer #(
    parameter integer IDX_DIM = 16,    // small index/scoring dim (real ~128)
    parameter integer S_MAX   = 32,    // max causal keys (real up to 1M ring)
    parameter integer TOPK    = 8,     // index_topk budget (real 2048)
    // LANES : number of keys processed concurrently (software-pipelined through
    // the single MAC).  >= `FP_MAC_LAT (7) fully hides the MAC latency -> 1 term/
    // cycle.  Smaller still correct (just stalls).  Result is bit-exact for ANY
    // LANES (per-key term order is preserved).
    parameter integer LANES   = 8,
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
    localparam integer LW  = (LANES   <= 1) ? 1 : $clog2(LANES);   // lane index
    // tag-FIFO depth: holds the lane-id of every MAC term in flight (<= `FP_MAC_LAT)
    // with slack, sized to a power of two so the head/tail pointers wrap freely.
    localparam integer TQD = 16;
    localparam integer TQW = $clog2(TQD);                          // = 4
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
    // (1) DOT-PRODUCT engine : one SHARED fp32_mac_pipe, software-pipelined over
    //   LANES independent keys.  Keys are scored in GROUPS of up to LANES.  Every
    //   cycle we issue ONE MAC term  acc[g] = q[d]*k[d] + acc[g]  for some lane g
    //   that is READY (its previous term has landed), round-robining the lanes so
    //   the pipe stays full.  Because lane g only ever issues its OWN term d AFTER
    //   its term d-1 has landed (dim_issue[g]==dim_done[g]), each key's accumulate
    //   is the IDENTICAL in-order fp32 FMA chain as the serial build -> BIT-EXACT
    //   scores; the LANES interleave merely fills the pipe's idle slots.  When a
    //   result lands, a lane-id FIFO (pushed at issue, popped at land -- the pipe
    //   is in-order & conserves count) names the lane whose acc it updates.
    //   acc[g] starts at +0.0.
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

    // per-lane state for the LANES in-flight keys of the current group.
    reg [31:0]  acc_l       [0:LANES-1];           // running fp32 dot accumulator
    reg [DIW:0] dim_issue_l [0:LANES-1];           // terms ISSUED for this lane
    reg [DIW:0] dim_done_l  [0:LANES-1];           // terms LANDED for this lane
    reg [15:0]  kbuf_l      [0:LANES-1][0:IDX_DIM-1]; // each lane's key bf16 vector
    integer ki, lg;

    // group bookkeeping.  All counts/sizes/indices share gbase's width [IDXW:0]
    // (IDXW+1 bits) so every arithmetic op is width-matched (warning-free); the
    // LANES-deep arrays are addressed with an explicit [LW-1:0] slice.
    reg [IDXW:0] gbase;                  // first key index of the current group
    reg [IDXW:0] gsize;                  // # keys in this group = min(LANES, S-gbase)
    reg [IDXW:0] fetch_issue;            // key requests issued for this group
    reg [IDXW:0] fetch_got;              // key vectors captured for this group
    reg [IDXW:0] rr;                     // round-robin start lane for issue

    // lane-id FIFO (matches each landed MAC result to its lane, latency-agnostic)
    reg [LW-1:0] tagq [0:TQD-1];
    reg [TQW-1:0] tq_head, tq_tail;
    integer tqi;

    //------------------------------------------------------------------------
    // COMBINATIONAL group/selection logic (Moore inputs to the FSM; pure
    // functions of the FSM registers -- no feedback, no latch).
    //   grem  : keys remaining (S - gbase)
    //   gbnd  : this group's key count = min(LANES, grem)
    //   isel_found/isel_lane : the round-robin-chosen READY lane to issue (a lane
    //     is ready when its previous term has landed: dim_issue==dim_done<IDX_DIM)
    //   grp_all_done : every active lane has landed all IDX_DIM terms
    //------------------------------------------------------------------------
    reg [IDXW:0] grem;
    reg [IDXW:0] gbnd;
    reg          isel_found;
    reg [IDXW:0] isel_lane;
    reg [IDXW:0] isel_scan;
    reg [IDXW:0] isel_cand;
    reg          grp_all_done;
    integer      cg;

    always @(*) begin
        grem      = s_reg - gbase;
        gbnd      = (grem >= LANES[IDXW:0]) ? LANES[IDXW:0] : grem;
        // round-robin scan for the next ready lane to issue.
        isel_found = 1'b0;
        isel_lane  = {(IDXW+1){1'b0}};
        isel_scan  = {(IDXW+1){1'b0}};
        isel_cand  = {(IDXW+1){1'b0}};
        for (cg = 0; cg < LANES; cg = cg + 1) begin
            isel_scan = rr + cg[IDXW:0];
            if (isel_scan >= LANES[IDXW:0]) isel_scan = isel_scan - LANES[IDXW:0];
            isel_cand = isel_scan;
            if (!isel_found && (isel_cand < gsize) &&
                (dim_issue_l[isel_cand[LW-1:0]] < IDX_DIM[DIW:0]) &&
                (dim_issue_l[isel_cand[LW-1:0]] == dim_done_l[isel_cand[LW-1:0]])) begin
                isel_found = 1'b1;
                isel_lane  = isel_cand;
            end
        end
        // group completion: all active lanes have landed all terms.
        grp_all_done = 1'b1;
        for (cg = 0; cg < LANES; cg = cg + 1)
            if ((cg[IDXW:0] < gsize) && (dim_done_l[cg[LW-1:0]] != IDX_DIM[DIW:0]))
                grp_all_done = 1'b0;
    end

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
                     S_FETCH  = 3'd1,   // stream this group's key vectors into lanes
                     S_SCORE  = 3'd2,   // interleaved dot-product (pipelined MACs)
                     S_TKL    = 3'd3,   // feed topk score-pull, wait done
                     S_DENSE  = 3'd4,   // emit 0..S-1 (no-op fallback)
                     S_DONE   = 3'd5;
    reg [2:0]   state;

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
            gbase          <= {(IDXW+1){1'b0}};
            gsize          <= {(IDXW+1){1'b0}};
            fetch_issue    <= {(IDXW+1){1'b0}};
            fetch_got      <= {(IDXW+1){1'b0}};
            rr             <= {(IDXW+1){1'b0}};
            tq_head        <= {TQW{1'b0}};
            tq_tail        <= {TQW{1'b0}};
            tk_start       <= 1'b0;
            tk_score_valid <= 1'b0;
            tk_score_in    <= 32'b0;
            tk_addr        <= {(SAW+1){1'b0}};
            sel_idx        <= {TOPK*IDXW{1'b0}};
            sel_count      <= {(IDXW+1){1'b0}};
            for (qi = 0; qi < IDX_DIM; qi = qi + 1) qbuf[qi] <= 16'b0;
            for (lg = 0; lg < LANES; lg = lg + 1) begin
                acc_l[lg]       <= 32'b0;
                dim_issue_l[lg] <= {(DIW+1){1'b0}};
                dim_done_l[lg]  <= {(DIW+1){1'b0}};
                for (ki = 0; ki < IDX_DIM; ki = ki + 1) kbuf_l[lg][ki] <= 16'b0;
            end
            for (tqi = 0; tqi < TQD;   tqi = tqi + 1) tagq[tqi] <= {LW{1'b0}};
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
                        // begin the first key-group at base 0 -> fetch its vectors.
                        gbase       <= {(IDXW+1){1'b0}};
                        fetch_issue <= {(IDXW+1){1'b0}};
                        fetch_got   <= {(IDXW+1){1'b0}};
                        tq_head     <= {TQW{1'b0}};
                        tq_tail     <= {TQW{1'b0}};
                        key_req     <= 1'b0;
                        state       <= S_FETCH;
                    end
                end
            end

            // ----------------------------------------------------------------
            // FETCH : stream the current group's key vectors into the lane
            //   buffers.  gsize = min(LANES, S - gbase) keys (indices gbase..),
            //   pulled at up to 1/cycle (key_req held high, key_idx incremented).
            //   The producer answers each request one beat later, in order, so
            //   the n-th arriving vector belongs to lane n.  When all gsize are
            //   captured, prime the per-lane accumulators and start scoring.
            // ----------------------------------------------------------------
            S_FETCH: begin
                // this group's key count = min(LANES, S - gbase) (comb: gbnd).
                gsize <= gbnd;

                // (a) capture an arriving key vector into the next lane slot.
                if (key_valid) begin
                    for (ki = 0; ki < IDX_DIM; ki = ki + 1)
                        kbuf_l[fetch_got[LW-1:0]][ki] <= k_idx[16*ki +: 16];
                    acc_l[fetch_got[LW-1:0]]       <= 32'b0;   // c for term 0
                    dim_issue_l[fetch_got[LW-1:0]] <= {(DIW+1){1'b0}};
                    dim_done_l[fetch_got[LW-1:0]]  <= {(DIW+1){1'b0}};
                    fetch_got                      <= fetch_got + 1'b1;
                end

                // (b) issue the next key request (1/cycle) until gbnd issued.
                if (fetch_issue < gbnd) begin
                    key_req     <= 1'b1;
                    key_idx     <= gbase[IDXW-1:0] + fetch_issue[IDXW-1:0];
                    fetch_issue <= fetch_issue + 1'b1;
                end else begin
                    key_req     <= 1'b0;
                end

                // (c) all key vectors captured -> begin interleaved scoring.
                if (fetch_got == gbnd) begin
                    key_req <= 1'b0;
                    rr      <= {(IDXW+1){1'b0}};
                    state   <= S_SCORE;
                end
            end

            // ----------------------------------------------------------------
            // SCORE : interleaved dot products for the up-to-LANES keys of this
            //   group.  Each cycle: ISSUE one MAC term for a ready lane (round
            //   robin), ABSORB any landed result into its lane's acc, and when
            //   every lane has landed all IDX_DIM terms, store the group's scores
            //   and advance to the next group (or launch top-K after the last).
            //   A lane g is READY to issue its term `dim_issue_l[g]` only once
            //   that term's predecessor has landed (dim_issue_l[g]==dim_done_l[g]),
            //   so each key's FMA chain stays strictly in order -> BIT-EXACT.
            // ----------------------------------------------------------------
            S_SCORE: begin
                // (a) the ready lane to issue is chosen combinationally (isel_*).

                // (b) issue the chosen lane's next term (a*b + its running acc).
                if (isel_found) begin
                    mac_valid_in <= 1'b1;
                    mac_a        <= bf16_to_fp32(qbuf[dim_issue_l[isel_lane[LW-1:0]][DIW-1:0]]);
                    mac_b        <= bf16_to_fp32(kbuf_l[isel_lane[LW-1:0]][dim_issue_l[isel_lane[LW-1:0]][DIW-1:0]]);
                    mac_c        <= acc_l[isel_lane[LW-1:0]];
                    dim_issue_l[isel_lane[LW-1:0]] <= dim_issue_l[isel_lane[LW-1:0]] + 1'b1;
                    // advance round-robin pointer (wrap at LANES).
                    rr           <= (isel_lane == LANES[IDXW:0]-1'b1)
                                    ? {(IDXW+1){1'b0}} : (isel_lane + 1'b1);
                    // push the issued lane id into the result-routing FIFO.
                    tagq[tq_tail] <= isel_lane[LW-1:0];
                    tq_tail       <= tq_tail + 1'b1;
                end

                // (c) absorb a landed result into the lane named by the FIFO head.
                if (mac_valid_out) begin
                    acc_l[tagq[tq_head]]      <= mac_result;
                    dim_done_l[tagq[tq_head]] <= dim_done_l[tagq[tq_head]] + 1'b1;
                    tq_head                   <= tq_head + 1'b1;
                end

                // (d) group complete (comb grp_all_done): every active lane has
                //     landed all terms.  Store scores, advance group / launch top-K.
                if (grp_all_done) begin
                    // store this group's scores (acc_l holds the final dots).
                    for (lg = 0; lg < LANES; lg = lg + 1)
                        if (lg[IDXW:0] < gsize)
                            score_mem[gbase[IDXW-1:0] + lg[IDXW-1:0]] <= acc_l[lg];

                    if ((gbase + gsize) >= s_reg) begin
                        // scored the last causal key -> launch top-K.
                        tk_start <= 1'b1;
                        tk_addr  <= {(SAW+1){1'b0}};
                        state    <= S_TKL;
                    end else begin
                        // advance to the next group.
                        gbase       <= gbase + gsize;
                        fetch_issue <= {(IDXW+1){1'b0}};
                        fetch_got   <= {(IDXW+1){1'b0}};
                        tq_head     <= {TQW{1'b0}};
                        tq_tail     <= {TQW{1'b0}};
                        key_req     <= 1'b0;
                        state       <= S_FETCH;
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
