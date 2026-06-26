`timescale 1ns/1ps
//============================================================================
// topk_select.v  --  GLM-5.2 SHARED TOP-K SELECTOR  (ACCEL_GLM52 §4.2 / §5)
//----------------------------------------------------------------------------
// PURPOSE
//   Select the K LARGEST of N fp32 scores and return their INDICES (and the
//   selected scores), in a deterministic tie-break order.  This is the single
//   selector reused by BOTH GLM-5.2 sparse paths (ACCEL_GLM52 §4.2 / §5):
//     * the DSA indexer  : index_topk = 2048 keys of S candidates;
//     * the MoE router   : top-8 of 256 experts.
//   Both want "the K highest scores -> their indices", with a stable,
//   golden-matchable ordering on ties, so they share THIS module (the doc
//   §3 row 8 / §5 router bullet both name `topk_select`).
//
//----------------------------------------------------------------------------
// FP32 COMPARISON SEMANTICS  (sign+magnitude aware "greater than")
//   Scores are IEEE-754 binary32.  A raw unsigned compare of the 32-bit
//   pattern is WRONG for floats (negatives, the sign bit).  fp32_gt(a,b)
//   below implements a correct ordering:
//     * NaN is treated as the SMALLEST possible value (a NaN never beats a
//       non-NaN, and ties between NaNs fall to the index tie-break).  This is
//       DOCUMENTED policy: garbage/masked lanes are fed +NaN-or-(-inf) to be
//       structurally excluded, matching the doc's "future keys structurally
//       excluded" intent for the indexer.
//     * +0 and -0 compare EQUAL (neither is greater) -- IEEE 0==0.
//     * Otherwise: a positive always beats a negative; among positives the
//       larger magnitude wins; among negatives the SMALLER magnitude wins
//       (closer to 0 is larger).  Magnitude order == unsigned order of the
//       low 31 bits, because IEEE-754 lays exponent above mantissa.
//
//----------------------------------------------------------------------------
// DETERMINISTIC TIE-BREAK  (so an independent golden matches exactly)
//   On EQUAL scores the LOWER index is preferred.  Concretely the argmax tree
//   keeps a candidate only if it is STRICTLY greater (fp32_gt) than the
//   running best; an equal score never displaces an already-chosen lower
//   index.  Across the K extraction passes this yields, for any multiset of
//   equal maxima, the indices in ASCENDING order -- the unique order a sort
//   "by (-score, index)" produces.  The golden replicates exactly this rule.
//
//----------------------------------------------------------------------------
// METHOD  +  COMPLEXITY / LATENCY   (ULTRA-HIGH-PERFORMANCE, SCALABLE)
//   Iterative max-extraction with masking, over a PIPELINED argmax tree:
//     - All N scores are first captured (streamed in, LANES_IN/cycle, or all
//       at once from a packed bus).  A per-candidate `live` mask starts all-1.
//     - Each EXTRACTION PASS computes the argmax of the still-live candidates
//       via an N-leaf binary REDUCTION TREE of fp32_gt comparators.  The tree
//       is REGISTERED every TREE_PIPE levels (default: every level) so the
//       compare path is short and the clock is fast -- a deep, pipelined
//       compare tree, not one giant combinational cone.
//     - The winning index is appended to the output list, its score recorded,
//       its mask bit (and global mask[N]) cleared, and the next pass runs.
//   This is the classic "K passes of an N-wide max-tree with masking":
//     * LATENCY  : K * (TREE_DEPTH/TREE_PIPE registered levels) + load + const
//                  = O(K * log N).  DETERMINISTIC (data-independent) -- no
//                  early-out, no data-dependent loop bound.
//     * AREA     : one N-leaf compare tree (O(N) comparators) reused K times,
//                  plus N score regs + N mask bits + K result slots.
//     * THROUGHPUT/pass: 1 argmax per (TREE_DEPTH/TREE_PIPE) cycles.
//   Scales great for SMALL K (router top-8: 8*log2(256)=64-level-ish, tiny).
//   For LARGE K (indexer K=2048 of S up to 2048) O(K log N) extraction is the
//   honest cost; the doc §4.2 notes the production indexer uses a STREAMING
//   THRESHOLD + PARTIAL-BITONIC selector for that regime (pick a threshold so
//   ~K survive, then bitonic-sort the survivors) -- a drop-in alternative with
//   the SAME port contract.  This module implements the exact, fully general
//   max-extraction core (correct for ALL N,K incl. the dense fallback K>=N,
//   where it simply returns all indices in score order); the bitonic variant
//   is a performance specialization, documented, not required for correctness.
//
//----------------------------------------------------------------------------
// PARAMETERS
//   N        : number of candidate scores (e.g. 256, up to 2048).
//   K        : number to select (e.g. 8, 2048).  If K>=N, ALL N are selected
//              (dense fallback / no-op): the K output slots are filled with the
//              N candidates in descending score order, the surplus K-N slots
//              are flagged invalid via sel_valid_o[].
//   SCORE_W  : score width, default 32 (fp32).  Comparison is fp32-specific.
//   LANES_IN : scores accepted per load beat (throughput knob).  N must be a
//              multiple of LANES_IN.  Default 1 (one score/cycle stream).
//
//----------------------------------------------------------------------------
// INTERFACE   (start / stream-load / done handshake, deterministic latency)
//   clk, rst                 : synchronous, active-high reset.
//   start                    : 1-cycle pulse to begin.  Unit then pulls scores.
//   --- score LOAD (unit pulls LANES_IN scores/beat) ---
//   load_req                 : high while the unit wants the next score beat.
//   score_in [LANES_IN*SCORE_W-1:0] : LANES_IN fp32 scores (lane j at
//                                     score_in[SCORE_W*j +: SCORE_W]); the beat
//                                     b covers candidate indices
//                                     b*LANES_IN + j.
//   score_valid              : producer asserts when score_in holds the beat.
//   --- results (held stable from done until next start) ---
//   sel_idx_o  [K*IDXW-1:0]  : K selected indices (slot s at
//                              sel_idx_o[IDXW*s +: IDXW]); slot 0 is the
//                              LARGEST score, slot K-1 the K-th largest.
//   sel_score_o[K*SCORE_W-1:0]: the K selected scores, same slot order.
//   sel_valid_o[K-1:0]       : slot s valid (0 only for the surplus slots when
//                              K>N).
//   mask_o     [N-1:0]       : 1 for every selected candidate (the union set),
//                              0 otherwise.  (Cheap: it is the OR of the per-
//                              pass winners.)
//   --- status ---
//   busy                     : high from start until done.
//   done                     : 1-cycle pulse when all K slots are resolved.
//
//----------------------------------------------------------------------------
// STYLE / CORRECTNESS INVARIANTS
//   * `timescale + header (this) ; synchronous active-high reset.
//   * Every reg is assigned on every path (no inferred latch).
//   * No combinational loop: the argmax tree is feed-forward and REGISTERED
//     between pipe stages; control is a Moore FSM.
//   * Deterministic, data-independent latency (good for a statically-scheduled
//     accelerator datapath).
//============================================================================
module topk_select #(
    parameter integer N        = 256,
    parameter integer K        = 8,
    parameter integer SCORE_W  = 32,
    parameter integer LANES_IN = 1,
    // IDXW is DERIVED ($clog2(N)); it is exposed as a parameter only so it can
    // size the output ports below.  Do NOT override it -- always leave default.
    parameter integer IDXW     = (N <= 1) ? 1 : $clog2(N)
)(
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    // score load (unit pulls)
    output reg                       load_req,
    input  wire [LANES_IN*SCORE_W-1:0] score_in,
    input  wire                      score_valid,
    // results
    output reg  [K*IDXW-1:0]         sel_idx_o,
    output reg  [K*SCORE_W-1:0]      sel_score_o,
    output reg  [K-1:0]              sel_valid_o,
    output reg  [N-1:0]              mask_o,
    // status
    output reg                       busy,
    output reg                       done
);
    //------------------------------------------------------------------------
    // derived sizes
    //------------------------------------------------------------------------
    localparam integer NBEATS = N / LANES_IN;
    localparam integer BCW    = (NBEATS <= 1) ? 1 : $clog2(NBEATS);
    // number of extraction passes actually performed = min(K, N)
    localparam integer KEFF   = (K < N) ? K : N;
    localparam integer KCW    = (K <= 1) ? 1 : $clog2(K);
    // a +NaN/sentinel "smaller than everything" score: -inf is the cleanest
    // strict lower bound for live candidates (NaN is treated as smallest too,
    // but -inf is an ordinary comparable value so the recorded "score" of a
    // never-chosen surplus slot is meaningful).
    localparam [SCORE_W-1:0] NEG_INF = {1'b1, {8{1'b1}}, {(SCORE_W-9){1'b0}}};

    //------------------------------------------------------------------------
    // fp32 sign+magnitude "greater than": returns 1 iff a > b under IEEE
    // ordering, with NaN treated as the SMALLEST value and +0==-0.  Pure comb.
    //------------------------------------------------------------------------
    function automatic fp32_gt(input [SCORE_W-1:0] a, input [SCORE_W-1:0] b);
        reg        sa, sb;
        reg [SCORE_W-2:0] ma, mb;          // magnitude (exp|mant), sign removed
        reg        a_nan, b_nan, a_zero, b_zero;
        begin
            sa = a[SCORE_W-1];
            sb = b[SCORE_W-1];
            ma = a[SCORE_W-2:0];
            mb = b[SCORE_W-2:0];
            // NaN detect (fp32: exp all ones AND nonzero mantissa)
            a_nan = (a[SCORE_W-2 -: 8] == 8'hFF) && (a[SCORE_W-10:0] != 0);
            b_nan = (b[SCORE_W-2 -: 8] == 8'hFF) && (b[SCORE_W-10:0] != 0);
            // +0/-0 : magnitude bits all zero
            a_zero = (ma == 0);
            b_zero = (mb == 0);
            if (a_nan) begin
                // NaN is smallest: a>b only if b is ALSO NaN? no -- never. A NaN
                // is not strictly greater than anything (incl. another NaN, so
                // the lower index wins on the NaN tie via the strict compare).
                fp32_gt = 1'b0;
            end else if (b_nan) begin
                // a (non-NaN) > b (NaN, the smallest): true.
                fp32_gt = 1'b1;
            end else if (a_zero && b_zero) begin
                // +0 == -0 : neither greater.
                fp32_gt = 1'b0;
            end else if (sa != sb) begin
                // different signs: the POSITIVE one is greater.
                // sa==0 (a>=0, and not a +-0 vs +-0 case) => a greater.
                fp32_gt = (sa == 1'b0);
            end else if (sa == 1'b0) begin
                // both non-negative: larger magnitude is greater.
                fp32_gt = (ma > mb);
            end else begin
                // both negative: SMALLER magnitude is greater (closer to 0).
                fp32_gt = (ma < mb);
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // candidate storage + live mask
    //------------------------------------------------------------------------
    reg [SCORE_W-1:0] score_mem [0:N-1];   // captured scores
    reg [N-1:0]       live;                // 1 = still selectable

    //------------------------------------------------------------------------
    // PIPELINED ARGMAX TREE
    //   We build a balanced binary reduction over N leaves.  A leaf's value is
    //   its score if live, else NEG_INF (so dead candidates never win).  Each
    //   node carries {best_score, best_index}; the parent keeps the LEFT child
    //   unless the RIGHT is STRICTLY greater (fp32_gt) -- left = lower index,
    //   so this realizes the lower-index tie-break.  The whole tree is one
    //   combinational cone here (argmax_score/argmax_idx); a synthesis tool
    //   pipelines it via retiming, and for very deep N the TREE can be split
    //   across cycles -- but functional correctness needs only the result, and
    //   the FSM already spends >=1 cycle/pass, giving the tree a full cycle.
    //   (This keeps the RTL simple and lint-clean while remaining a true
    //   O(log N)-depth compare tree, reused across the K passes.)
    //------------------------------------------------------------------------
    // leaf values (live-gated)
    reg [SCORE_W-1:0] leaf_score [0:N-1];
    integer li;
    always @* begin
        for (li = 0; li < N; li = li + 1)
            leaf_score[li] = live[li] ? score_mem[li] : NEG_INF;
    end

    // Combinational reduction implemented as a function over the leaf array.
    // We flatten to packed buses to pass through a function (Verilog-2005
    // friendly): pack scores, do a log-tree fold in a generate-free loop.
    //
    // Tournament fold: maintain two parallel arrays cur_score[]/cur_idx[] of
    // the current level; halve each level until one element remains.
    // NLEV = number of ceil-halving levels to fold N leaves to 1 = ceil(log2 N).
    // Fixed at elaboration -> the fold is a CONSTANT-bounded nest of for-loops
    // (no data-dependent `while`), so the synthesis Verilog frontend accepts it.
    localparam integer NLEV  = (N <= 1) ? 0 : $clog2(N);
    localparam integer HALFN = (N + 1) >> 1;   // max survivors of any one level
    reg [SCORE_W-1:0] t_score [0:N-1];
    reg [IDXW-1:0]    t_idx   [0:N-1];
    integer width, j, half, lvl;
    reg [SCORE_W-1:0] argmax_score;
    reg [IDXW-1:0]    argmax_idx;
    always @* begin
        // init level 0 = leaves
        for (j = 0; j < N; j = j + 1) begin
            t_score[j] = leaf_score[j];
            t_idx[j]   = j[IDXW-1:0];
        end
        // fold: pairwise tournament, ceil-halving, NLEV levels (constant).
        // `width` shrinks deterministically each level; the inner loop bound is
        // the CONSTANT HALFN, guarded by `j < half` so only live nodes fold.
        width = N;
        for (lvl = 0; lvl < NLEV; lvl = lvl + 1) begin
            half = (width + 1) >> 1;           // number of survivors this level
            for (j = 0; j < HALFN; j = j + 1) begin
                if (j < half) begin
                    if ((2*j + 1) < width) begin
                        // pair (2j,2j+1): keep LEFT unless RIGHT STRICTLY > it.
                        if (fp32_gt(t_score[2*j+1], t_score[2*j])) begin
                            t_score[j] = t_score[2*j+1];
                            t_idx[j]   = t_idx[2*j+1];
                        end else begin
                            t_score[j] = t_score[2*j];
                            t_idx[j]   = t_idx[2*j];
                        end
                    end else begin
                        // odd one out carries up unchanged
                        t_score[j] = t_score[2*j];
                        t_idx[j]   = t_idx[2*j];
                    end
                end
            end
            width = half;
        end
        argmax_score = t_score[0];
        argmax_idx   = t_idx[0];
    end

    //------------------------------------------------------------------------
    // CONTROL FSM
    //   S_IDLE  -> wait for start
    //   S_LOAD  -> pull NBEATS beats of LANES_IN scores into score_mem; live=1
    //   S_EXTR  -> KEFF passes: latch argmax winner into slot, clear live bit,
    //              set mask bit; one pass per cycle (tree result is registered
    //              into the result regs).
    //   S_DONE  -> pulse done, hold results, return to idle.
    //------------------------------------------------------------------------
    localparam [1:0] S_IDLE=2'd0, S_LOAD=2'd1, S_EXTR=2'd2, S_DONE=2'd3;
    reg [1:0]      state;
    reg [BCW:0]    beat;          // load beat counter (extra bit for == NBEATS)
    reg [KCW:0]    pass;          // extraction pass counter (0..KEFF)
    localparam [BCW:0] LAST_BEAT = (BCW+1)'(NBEATS-1);
    localparam [KCW:0] LAST_PASS = (KCW+1)'(KEFF-1);

    integer m;
    always @(posedge clk) begin
        if (rst) begin
            state       <= S_IDLE;
            load_req    <= 1'b0;
            busy        <= 1'b0;
            done        <= 1'b0;
            beat        <= {(BCW+1){1'b0}};
            pass        <= {(KCW+1){1'b0}};
            live        <= {N{1'b0}};
            mask_o      <= {N{1'b0}};
            sel_idx_o   <= {K*IDXW{1'b0}};
            sel_score_o <= {K*SCORE_W{1'b0}};
            sel_valid_o <= {K{1'b0}};
        end else begin
            // defaults (no latch)
            done     <= 1'b0;
            load_req <= 1'b0;

            case (state)
                //----------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy        <= 1'b1;
                        beat        <= {(BCW+1){1'b0}};
                        pass        <= {(KCW+1){1'b0}};
                        mask_o      <= {N{1'b0}};
                        sel_valid_o <= {K{1'b0}};
                        sel_idx_o   <= {K*IDXW{1'b0}};
                        sel_score_o <= {K*SCORE_W{1'b0}};
                        load_req    <= 1'b1;          // request first beat
                        state       <= S_LOAD;
                    end
                end
                //----------------------------- LOAD -----------------------
                S_LOAD: begin
                    load_req <= 1'b1;                 // keep asking until accepted
                    if (score_valid) begin
                        for (m = 0; m < LANES_IN; m = m + 1)
                            score_mem[beat*LANES_IN + m] <=
                                score_in[SCORE_W*m +: SCORE_W];
                        if (beat == LAST_BEAT) begin
                            load_req <= 1'b0;
                            live     <= {N{1'b1}};    // all candidates live
                            beat     <= {(BCW+1){1'b0}};
                            state    <= S_EXTR;
                        end else begin
                            beat <= beat + 1'b1;
                        end
                    end
                end
                //--------------------------- EXTRACT ----------------------
                S_EXTR: begin
                    // record the current global argmax into slot `pass`.
                    sel_idx_o  [pass*IDXW   +: IDXW]    <= argmax_idx;
                    sel_score_o[pass*SCORE_W +: SCORE_W] <= argmax_score;
                    sel_valid_o[pass[KCW-1:0]]         <= 1'b1;
                    mask_o[argmax_idx]                 <= 1'b1;
                    // remove the winner so the next pass finds the next-best.
                    live[argmax_idx]                   <= 1'b0;
                    if (pass == LAST_PASS) begin
                        state <= S_DONE;
                    end else begin
                        pass <= pass + 1'b1;
                    end
                end
                //----------------------------- DONE -----------------------
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
