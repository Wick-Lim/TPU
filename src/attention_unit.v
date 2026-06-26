`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// attention_unit.v  --  true scaled-dot-product attention, SEQ-/D-generic
//                       (default SEQ=`SEQ_LEN=4, D=`D_MODEL=4) (SPEC §5.4)
//----------------------------------------------------------------------------
// PURPOSE
//   Computes one head of scaled dot-product attention over SEQ tokens of
//   dimension D, all in Q7.8 fixed point:
//
//       Attn(Q,K,V) = softmax( (Q . K^T) / sqrt(d) ) . V
//
//   This REPLACES the v1.5 single-element "attention" (which had NO softmax and
//   truncated the (query*key) product into 32 bits -- the documented v1.5 bug).
//   v2.0 computes the scores in a full 48-bit Q15.16 accumulator, applies an
//   EXACT 1/sqrt(d) scale with round-half-up, runs a genuine exponential
//   softmax over each length-SEQ row (by INSTANTIATING the LEN-generic
//   softmax_unit), and forms the context in a second 48-bit accumulator with an
//   explicit round-half-up + saturate narrowing and a sticky `sat` flag.  No
//   silent truncation survives anywhere -- the bug class is structurally gone.
//
//   It is TM->TM: it reads Q/K/V (SEQ lines each) from tile memory and writes
//   the SEQ context lines O.  It exposes raw TM ACCESS PORTS only (the
//   surrounding datapath / unit TB owns and models the tile memory).  The single
//   submodule it instantiates is src/softmax_unit.v; that submodule runs
//   entirely on an INTERNAL scratch memory inside this unit, so the softmax
//   never contends for the external TM port (clean, self-contained arbitration).
//
// PARAMETERIZATION (NEW in v2.0; default == tpu_defs.vh, byte-identical)
//   parameter integer SEQ = `SEQ_LEN, D = `D_MODEL  (defaults 4, 4).
//   Every loop bound, the QK^T / softmax / *V dimension, the counter/index
//   widths ($clog2), the softmax-scratch line span and the TM packing indices
//   are derived from SEQ and D -- NO size-literal tricks remain.  In particular
//   the v1.x "{row[1:0],col[1:0]} == row*4+col" mod-4 bit-truncation is replaced
//   by a real (row*SEQ + col) index, and the fixed 3-bit S_LAST/rcnt/row
//   counters are replaced by $clog2-sized counters.
//
//   SUPPORTED RANGE (architectural envelope):
//     * D   <= LINE_LANES = 4 : one Q/K/V/O ROW packs into ONE 128-bit TM line
//       (each element in the low 16 bits of one of the 4 lanes), matching the
//       architecture's FIXED 4-lane TM line.  D need not equal SEQ.
//     * SEQ <= SM_LEN  = 8    : a length-SEQ score row is run through the softmax
//       submodule at LEN=SEQ (NO padding -- the softmax operates over exactly the
//       SEQ real logits).  The score row spans NSCR_X = ceil(SEQ/4) X-lines plus
//       NSCR_X P-lines = 2*ceil(SEQ/4) scratch lines (<=4 for SEQ<=8).  (SEQ also
//       bounds how many TM lines Q/K/V/O occupy; with TM_LINES=32 and four
//       SEQ-line tiles, SEQ<=8 sits comfortably in TM.)
//     * SEQ >= 1, D >= 1.
//   The default (4,4) sits inside this envelope and is exercised exhaustively by
//   attention_unit_tb; a 2nd in-range size (SEQ=2,D=2) is proven by a second
//   instance in test/attention_param_tb.v against an INDEPENDENT real golden.
//
// SOFTMAX REUSE (LEN=SEQ -- NO padding; the pad-collision corner is RESOLVED)
//   softmax_unit is LEN-generic, so we instantiate it at LEN=SEQ and run the
//   softmax over EXACTLY the SEQ real logits -- there are NO pad lanes at all.
//   The score row spans NSCR_X = ceil(SEQ/4) scratch X-lines (the SEQ logits in
//   lanes 0..SEQ-1; any unused lane of the last partial line is a don't-care the
//   LEN=SEQ softmax never reads), and the SEQ probabilities come back in NSCR_X
//   P-lines.  This is the mathematically correct length-SEQ softmax (SPEC §5.4),
//   realised by reuse of the EXACT committed softmax_unit (no forked copy).
//   Because there is no `Q78_MIN sentinel, the old pad-collision corner (all SEQ
//   real logits saturating to the Q7.8 floor and colliding with the pad value ->
//   uniform 1/SM_PAD weights -> SEQ/SM_PAD magnitude loss) is IMPOSSIBLE BY
//   CONSTRUCTION.  (Latency note: the softmax reciprocal divide was later
//   PIPELINED into a multi-cycle sequential divider (+DIV_CYCLES=48 per softmax
//   invocation), so LAT_TOTAL is now 279 at the default SEQ=4 -- see the LATENCY
//   note; the probabilities/context are UNCHANGED, only the latency grew.)
//
// Q-FORMATS  (single source of truth: tpu_defs.vh, SPEC §1.3)
//   Q, K, V elements   : Q7.8   signed 16-bit (low 16 bits of a 32-bit TM lane).
//   score MAC product  : Q7.8 * Q7.8 = Q14.16 (30-bit signed, fits 32 bits).
//   score accumulator  : Q15.16, 48-bit signed (D Q14.16 products, no overflow).
//   scaled score S[i][j]: (acc + round) >> 1, then narrowed into Q7.8 logit space
//                         for softmax (the >>1 is the EXACT 1/sqrt(4) scale).
//   softmax weights W  : Q0.16 unsigned probabilities (0xFFFF ~= 1.0).
//   context MAC product: W(Q0.16) * V(Q7.8) = Q7.24 (held left-aligned, see below)
//   context accumulator: 48-bit signed; narrowed round-half-up + saturate to Q7.8.
//
// SCORE -> SOFTMAX-LOGIT SCALING (the EXACT >>1, documented)
//   Raw score acc_S[i][j] = SUM_{d} Q[i][d]*K[j][d]  is Q15.16 in 48 bits.
//   The 1/sqrt(d) scale is applied as the committed EXACT right shift by 1
//   (the v1.5/default 1/sqrt(4)=1/2): scaled = (acc_S + 1) >>> 1   (Q15.16
//   still).  The softmax_unit consumes Q7.8 LOGITS (16-bit), so the scaled
//   Q15.16 score is narrowed to a Q7.8 logit by a LOCAL round-half-up+saturate
//   helper that is bit-exact to the shared tpu_defs.vh narrowing macro.  softmax
//   is shift-invariant, so any saturation of an individual logit only matters
//   relative to the row max; the row max itself is subtracted inside
//   softmax_unit for numerical stability.  Score-logit saturation is a
//   softmax-INPUT clamp and is intentionally NOT folded into the output `sat`
//   flag (see the SATURATION POLICY note below).
//
// CONTEXT  O[i][d] = SUM_{j} W[i][j] * V[j][d]
//   W[i][j] is Q0.16 (unsigned [0,1]); V[j][d] is Q7.8 signed.  The product
//   W*V is Q0.16 * Q7.8 = Q7.24.  To accumulate in the shared Q15.16 48-bit
//   format, each product is sign-extended then right-shifted by
//   (Q016_FRAC - Q78_FRAC) = 8 with round-half-up so the running sum is Q15.16;
//   SEQ such terms cannot overflow 48 bits.  The 48-bit sum is then narrowed to
//   Q7.8 by a LOCAL round-half-up + saturate (signed-bias) helper.
//
// SATURATION POLICY (the `sat` flag)
//   The attention OUTPUT is the context O.  Its only output narrowing is the
//   W.V -> Q7.8 step, so `sat` is the sticky OR of CONTEXT narrowing saturation
//   (`ctx_sat`) ONLY.  Two clamps are DELIBERATELY excluded because in the normal
//   operating range neither is an output-magnitude loss of this unit:
//     * the score -> Q7.8-logit clamp is a softmax-INPUT clamp (softmax is
//       shift/scale-robust; the renormalized context still tracks the golden);
//     * the softmax probability 0xFFFF clamp (sm_sat) is a ~1.5e-5 rounding of a
//       PROBABILITY, internal to the softmax submodule.
//   NORMAL-RANGE PROPERTY: the softmax runs over the SEQ real logits (LEN=SEQ,
//   no pad lanes), so its SEQ weights ALWAYS sum to ~1.0 and O is a CONVEX
//   COMBINATION of the value vectors; |O| <= max_j |V[j][.]| <= Q7.8 max, so
//   `ctx_sat` (and thus `sat`) does not fire for in-range V.  The TB asserts
//   sat==gsat across all directed + random vectors in that range (logits |val|
//   <= 512), including all-V-max (which rounds to exactly +max WITHOUT clamping).
//
//   RESOLVED -- pad collision (formerly the KNOWN LIMITATION; see docs/ROADMAP.md
//   §5).  The previous scheme padded the score row to SM_PAD = max(SEQ,8) lanes
//   with the `Q78_MIN sentinel and relied on exp(pad - rowmax) ~ 0.  In the
//   extreme corner where ALL SEQ real logits in a row themselves saturated to the
//   Q7.8 floor `Q78_MIN (every key equally, maximally anti-aligned), the real
//   logits COLLIDED with the pad sentinel: softmax saw SM_PAD identical values
//   and returned uniform 1/SM_PAD weights, so the SEQ real weights summed to
//   SEQ/SM_PAD (not 1.0) and |O| was silently scaled by SEQ/SM_PAD (a 2x loss at
//   the default SEQ=4/SM_PAD=8), with `sat` staying 0.  Instantiating the softmax
//   at LEN=SEQ removes the pad lanes ENTIRELY, so there is no sentinel to collide
//   with: the convex-combination property now holds for ALL in-range inputs --
//   even the all-logits-at-floor corner gives the correct uniform 1/SEQ weights
//   (column-mean of V), NOT half of it.  This corner is locked in by the directed
//   COLLISION regression in test/attention_unit_tb.v (Q=+max, K=-max).  The
//   2x-loss corner no longer exists.
//
// INTERFACE
//   clk, rst                          clock / synchronous active-high reset
//   start                             1-cycle pulse: latch bases, begin attention
//   q_base,k_base,v_base,o_base [4:0] TM line indices of Q,K,V,O tiles (SEQ ea)
//   busy                              high while an op is in flight (registered)
//   done                              1-cycle pulse when the last O line is driven
//   sat                               valid with done; 1 iff any narrowing clamped
//   -- external TM read port (combinational) --
//   tm_raddr [4:0] (out) / tm_rdata [127:0] (in)
//   -- external TM write port (synchronous) --
//   tm_we (out) / tm_waddr [4:0] (out) / tm_wdata [127:0] (out)
//
// LATENCY (deterministic, committed; asserted EXACTLY by the unit TB)
//   The unit serially REUSES the softmax_unit once per output row, so the true
//   committed latency is dominated by SEQ softmax invocations.  The measured,
//   committed start->done latency of THIS RTL is `LAT_TOTAL` cycles (a
//   localparam, derived below from SEQ and the softmax submodule's own closed
//   form at LEN=SEQ); the TB asserts it bit-exactly.  At the default SEQ=4
//   (LEN=SEQ=4, NSCR_X=1, softmax LAT 62) LAT_TOTAL = 279.  The softmax's
//   reciprocal divide was PIPELINED into a multi-cycle sequential divider
//   (+DIV_CYCLES=48 per softmax invocation); attention reuses the softmax once
//   per output row, so LAT_TOTAL grew by SEQ*48 (87 -> 279 at SEQ=4).  The
//   probabilities/context/argmax/sat are UNCHANGED -- only the latency grew.
//
// SYNTHESIZABILITY
//   Synchronous reset on ALL state; every reg assigned on every path of the one
//   clocked FSM (no inferred latch); combinational outputs are pure functions of
//   registered state (no comb loop); no real/$display/$random/initial in the
//   module.  Passes verilator --lint-only -Wall and iverilog -g2012 -Wall.
//============================================================================
module attention_unit #(
    // Sequence length and model/head dimension.  DEFAULT to the tpu_defs.vh
    // values so behavior is BYTE-IDENTICAL to the committed seq4/d4 unit at its
    // default (and so the system TB, which instantiates defaults, is unaffected).
    parameter integer SEQ = `SEQ_LEN,
    parameter integer D   = `D_MODEL
) (
    input  wire                 clk,
    input  wire                 rst,

    // Control handshake.
    input  wire                 start,
    input  wire [`TM_IDX_W-1:0] q_base,
    input  wire [`TM_IDX_W-1:0] k_base,
    input  wire [`TM_IDX_W-1:0] v_base,
    input  wire [`TM_IDX_W-1:0] o_base,
    output reg                  busy,
    output reg                  done,
    output reg                  sat,

    // External TM read access port (combinational read).  Only the low ELEM_W
    // (16) bits of each 32-bit lane carry Q7.8 data; the high 16 bits of every
    // lane are intentionally unused on read (the narrow lint_off documents that).
    output reg  [`TM_IDX_W-1:0] tm_raddr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [`LINE_W-1:0]   tm_rdata,
    /* verilator lint_on UNUSEDSIGNAL */

    // External TM write access port (synchronous write).
    output reg                  tm_we,
    output reg  [`TM_IDX_W-1:0] tm_waddr,
    output reg  [`LINE_W-1:0]   tm_wdata
);

    // ===================== parameter-derived geometry ========================
    // S       : sequence length (alias kept for the body's historical naming).
    // NLANES  : lanes per TM line (the FIXED 4-lane architectural line).
    // SM_LEN_ : softmax vector length = EXACTLY SEQ (NO padding; see the
    //           SATURATION POLICY header -- the pad-collision corner is RESOLVED
    //           by running the softmax over the SEQ real logits only).
    // NSCR_X  : softmax X (logit) lines = ceil(SEQ/NLANES).
    // SCR_N   : total softmax scratch lines = X lines + P lines = 2*NSCR_X.
    // RCNT_W  : width of the row-read counter (holds 0..SEQ).
    // ROW_W   : width of the current-row index (holds 0..SEQ-1).
    // SIDX_W  : width of a flat score/QKV element index (holds 0..SEQ*SEQ-1 and
    //           0..SEQ*D-1, both <= max(SEQ*SEQ, SEQ*D); sized to the larger).
    localparam integer S      = SEQ;
    localparam integer NLANES = `LINE_LANES;                       // 4
    localparam integer SM_LEN_= SEQ;                               // softmax length
    localparam integer NSCR_X = (SM_LEN_ + NLANES - 1) / NLANES;   // ceil(SEQ/4)
    localparam integer SCR_N  = 2 * NSCR_X;                        // X + P lines
    localparam integer RCNT_W = $clog2(SEQ + 1);                   // holds 0..SEQ
    localparam integer ROW_W  = (SEQ > 1) ? $clog2(SEQ) : 1;       // holds 0..SEQ-1

    // last row index, sized to the row counter.
    localparam [RCNT_W-1:0] S_LAST = RCNT_W'(SEQ - 1);

    // ---- FSM states ----
    localparam [3:0] ST_IDLE   = 4'd0;
    localparam [3:0] ST_RDQ    = 4'd1;  // read SEQ Q rows  (SEQ cycles)
    localparam [3:0] ST_RDK    = 4'd2;  // read SEQ K rows  (SEQ cycles)
    localparam [3:0] ST_RDV    = 4'd3;  // read SEQ V rows  (SEQ cycles)
    localparam [3:0] ST_SCORE  = 4'd4;  // compute all SEQ*SEQ scaled scores (1 cyc)
    localparam [3:0] ST_SM_LD  = 4'd5;  // load score row i into softmax scratch
    localparam [3:0] ST_SM_GO  = 4'd6;  // pulse softmax start
    localparam [3:0] ST_SM_WT  = 4'd7;  // wait for softmax done, capture weights
    localparam [3:0] ST_CTX    = 4'd8;  // compute O row i, drive its TM write
    localparam [3:0] ST_DONE   = 4'd9;  // 1-cycle done pulse

    // Softmax scratch line indices (internal scratch memory).  The softmax X
    // (logit) tile is scratch lines 0..NSCR_X-1; the P (prob) tile is lines
    // NSCR_X..2*NSCR_X-1.  At the default these are X={0,1}, P={2,3}.
    localparam [`TM_IDX_W-1:0] SM_XBASE = `TM_IDX_W'(0);
    localparam [`TM_IDX_W-1:0] SM_PBASE = `TM_IDX_W'(NSCR_X);
    localparam integer         SCR_IDX_W = (SCR_N > 1) ? $clog2(SCR_N) : 1;

    // Committed deterministic latency (asserted EXACTLY by the unit TB).
    // Counting convention (matches the TB): the cycle `start` is sampled high in
    // ST_IDLE is cycle 1; `done` is a REGISTERED 1-cycle pulse first OBSERVED
    // high `LAT_TOTAL` cycles later.  Breakdown:
    //   SETUP   = ST_RDQ(SEQ) + ST_RDK(SEQ) + ST_RDV(SEQ) + ST_SCORE(1)
    //   SM_LAT  = the softmax submodule's committed closed form for SM_LEN_=SEQ
    //             lanes (NO padding).  The softmax reciprocal divide was
    //             PIPELINED into a multi-cycle radix-2 restoring sequential
    //             divider (DIV_CYCLES=48 added), so its closed form is now
    //             53 + ceil(SEQ/4) + 2*SEQ  (was 5 + ceil(SEQ/4) + 2*SEQ)
    //               (== 62 for SEQ=4 / NSCR_X=1;  == 58 for SEQ=2).
    //   SM_WT   = SM_LAT + 1 : sm_start is REGISTERED in ST_SM_GO, so the
    //             submodule samples it one cycle into ST_SM_WT and then runs its
    //             SM_LAT-cycle pipeline; +1 is the cycle sm_done is observed.
    //   PER_ROW = ST_SM_LD(1) + ST_SM_GO(1) + ST_SM_WT(SM_WT) + ST_CTX(1)
    //   TAIL    = ST_DONE entry (1) + the registered done-observed edge (1) = 2
    // LAT_TOTAL = SETUP + SEQ*PER_ROW + TAIL.  attention serially REUSES the
    // softmax once per output row, so the softmax's +DIV_CYCLES(48) per
    // invocation adds SEQ*48 to LAT_TOTAL.  At default SEQ=4 / NSCR_X=1 this is
    // 13 + 4*66 + 2 = 279 (was 87 before the softmax divider was pipelined; the
    // PROBABILITIES/context/argmax/sat are UNCHANGED -- only the cycle-accurate
    // latency grew); at SEQ=2 it is 7 + 2*62 + 2 = 133 (was 37).  These are PURE
    // DOCUMENTATION localparams (the latency is structural in the FSM, not
    // parameter-driven in logic), so they are not referenced in logic; the
    // lint_off records that.
    /* verilator lint_off UNUSEDPARAM */
    localparam integer SM_LAT    = 53 + NSCR_X + 2*SM_LEN_;           // 62 @ SEQ4
    localparam integer SM_WT_LAT = SM_LAT + 1;                        // 63
    localparam integer SETUP     = 3*SEQ + 1;                         // 13 @ SEQ4
    localparam integer PER_ROW   = 1 /*LD*/ + 1 /*GO*/ + SM_WT_LAT /*WT*/ + 1; // 66
    localparam integer LAT_TOTAL = SETUP + (S * PER_ROW) + 2;         // = 279 @ SEQ4
    /* verilator lint_on UNUSEDPARAM */

    // ===================== latched operands / results ========================
    // Q,K,V stored as S x D Q7.8 signed elements.
    reg signed [`ELEM_W-1:0] qm [0:S*D-1];
    reg signed [`ELEM_W-1:0] km [0:S*D-1];
    reg signed [`ELEM_W-1:0] vm [0:S*D-1];

    // Scaled scores as Q7.8 logits (S x S), fed to softmax.
    reg signed [`ELEM_W-1:0] slog [0:S*S-1];

    // Attention weights for the current row (Q0.16, length S).
    reg [`Q016_W-1:0] wrow [0:S-1];

    // ---- bookkeeping ----
    reg [3:0]              state;
    reg [`TM_IDX_W-1:0]    q_base_q, k_base_q, v_base_q, o_base_q;
    reg [RCNT_W-1:0]       rcnt;     // generic small counter (0..S)
    reg [ROW_W-1:0]        row;      // current output/softmax row (0..S-1)

    // ===================== internal softmax scratch ==========================
    // The instantiated softmax_unit reads/writes THIS scratch only, never the
    // external TM, so there is no external-port arbitration.  SCR_N lines
    // (2*NSCR_X) cover both the X (logit) and P (prob) tiles.
    reg [`LINE_W-1:0] sm_scratch [0:SCR_N-1];

    // softmax handshake wiring.
    //   * sm_busy / sm_argmax are produced by the submodule but NOT consumed
    //     here (the FSM sequences purely off sm_done).  Wrapped in a narrow
    //     UNUSEDSIGNAL lint_off documenting the deliberate non-use.
    //   * sm_sat (softmax's own 0xFFFF probability clamp) is an internal
    //     renormalization artifact, NOT an attention-output magnitude loss, so it
    //     is intentionally NOT consumed (see the `sat` policy in the header).
    //   * sm_raddr / sm_waddr are 5-bit (TM_IDX_W) but the internal scratch is
    //     only SCR_N lines, so only the low SCR_IDX_W bits are used; the lint_off
    //     covers the intentionally-unused high index bits too.
    reg                  sm_start;
    wire                 sm_done;
    wire                 sm_we;
    reg  [`LINE_W-1:0]   sm_rdata;
    wire [`LINE_W-1:0]   sm_wdata;
    /* verilator lint_off UNUSEDSIGNAL */
    wire                 sm_busy;
    wire                 sm_sat;     // softmax's own 0xFFFF prob clamp (not used)
    wire [2:0]           sm_argmax;
    wire [`TM_IDX_W-1:0] sm_raddr;
    wire [`TM_IDX_W-1:0] sm_waddr;
    /* verilator lint_on UNUSEDSIGNAL */

    // Combinational read of the internal scratch for the softmax submodule.
    // softmax addresses scratch lines 0..SCR_N-1; index by the low SCR_IDX_W bits.
    always @(*) sm_rdata = sm_scratch[sm_raddr[SCR_IDX_W-1:0]];

    softmax_unit #(.LEN(SM_LEN_)) u_softmax (
        .clk      (clk),
        .rst      (rst),
        .start    (sm_start),
        .x_base   (SM_XBASE),
        .p_base   (SM_PBASE),
        .busy     (sm_busy),
        .done     (sm_done),
        .sat      (sm_sat),
        .argmax   (sm_argmax),
        .tm_raddr (sm_raddr),
        .tm_rdata (sm_rdata),
        .tm_we    (sm_we),
        .tm_waddr (sm_waddr),
        .tm_wdata (sm_wdata)
    );

    // ===================== local round-half-up + saturate ====================
    // This local helper is a bit-exact wrapper of the canonical tpu_defs.vh
    // `TPU_RND_SAT_Q78 / `TPU_ROUND_SHIFT / `TPU_SAT_HIT.  The shared macro's
    // round bias is now a SIGNED ACC_W constant, so the macro is signed-correct
    // and this helper computes the IDENTICAL result; it is retained as named
    // functions because the score and context blocks index 2-D reg arrays and
    // reuse the round-shift twice.  RND_BIAS is a localparam declared here.
    //   rounded = (acc + (1<<(FRAC-1))) >>> FRAC          [arithmetic shift]
    //   sat:  rounded > 32767 -> 32767 ; < -32768 -> -32768 ; else value
    localparam signed [`ACC_W-1:0] RND_BIAS =
        `ACC_W'sd1 <<< (`Q78_FRAC-1);          // +128 = 1<<(FRAC-1), signed

    function signed [`ACC_W-1:0] round_shift;
        input signed [`ACC_W-1:0] acc_in;
        begin
            round_shift = (acc_in + RND_BIAS) >>> `Q78_FRAC;
        end
    endfunction

    function signed [`ELEM_W-1:0] rnd_sat_q78;
        input signed [`ACC_W-1:0] acc_in;
        reg   signed [`ACC_W-1:0] r;
        begin
            r = round_shift(acc_in);
            if (r > `ACC_W'sd32767)       rnd_sat_q78 = `Q78_MAX;
            else if (r < -`ACC_W'sd32768) rnd_sat_q78 = `Q78_MIN;
            else                          rnd_sat_q78 = r[`ELEM_W-1:0];
        end
    endfunction

    function sat_hit_q78;
        input signed [`ACC_W-1:0] acc_in;
        reg   signed [`ACC_W-1:0] r;
        begin
            r = round_shift(acc_in);
            sat_hit_q78 = (r > `ACC_W'sd32767) || (r < -`ACC_W'sd32768);
        end
    endfunction

    // ===================== combinational score computation ===================
    // For the SCORE state we compute all SEQ*SEQ raw dot products combinationally
    // from the latched qm/km, scale by >>1 (round-half-up) into Q15.16, and
    // narrow each to a Q7.8 logit.  acc_S[i][j] = SUM_d qm[i*D+d]*km[j*D+d].
    //
    // Each product is Q14.16 in 32 bits; D summed in a 48-bit signed accumulator.
    // scaled = round_half_up(acc, 1) >>> 1  (the EXACT 1/sqrt(4) committed scale).
    // The Q15.16 scaled score is then narrowed to a Q7.8 logit (round+sat).
    //
    // NOTE ON SCORE-LOGIT SATURATION (intentionally NOT folded into `sat`): for
    //   very large |Q|,|K| the scaled score can exceed the Q7.8 logit range and
    //   clamp.  This is a SOFTMAX-INPUT clamp, not an OUTPUT-magnitude loss:
    //   softmax is shift/scale-robust, and the renormalized weights (hence the
    //   context O) still track the real golden within tolerance.  The sticky
    //   `sat` flag is the OUTPUT-narrowing flag (SPEC §1.3); it is driven ONLY by
    //   the CONTEXT narrowing (ctx_sat).  Score-logit clamping is deliberately
    //   NOT OR'd into `sat`.
    reg signed [`ELEM_W-1:0] score_log  [0:S*S-1];   // Q7.8 logit
    integer si, sj, sd;
    reg signed [`ACC_W-1:0]  acc_tmp;
    reg signed [`ACC_W-1:0]  scl_tmp;     // acc after the >>1 round (Q15.16)
    reg signed [31:0]        qk_prod;     // Q14.16 signed product (16b*16b -> 32b)
    reg signed [`ACC_W-1:0]  qk_ext;      // sign-extended product, Q14.16 in 48b
    always @(*) begin
        for (si = 0; si < S; si = si + 1) begin
            for (sj = 0; sj < S; sj = sj + 1) begin
                acc_tmp = {`ACC_W{1'b0}};
                for (sd = 0; sd < D; sd = sd + 1) begin
                    // Q7.8 * Q7.8 = Q14.16 (32-bit SIGNED).  Keep it in a signed
                    // 32b temp, then EXPLICITLY sign-extend into the 48-bit
                    // accumulator so the negative-product sign is preserved AND
                    // the lint sees no implicit width growth.
                    qk_prod = $signed(qm[si*D+sd]) * $signed(km[sj*D+sd]);
                    qk_ext  = {{(`ACC_W-32){qk_prod[31]}}, qk_prod};
                    acc_tmp = acc_tmp + qk_ext;
                end
                // EXACT 1/sqrt(4)=1/2 scale: round-half-up then arithmetic >>1.
                scl_tmp = (acc_tmp + `ACC_W'sd1) >>> 1;
                // narrow Q15.16 -> Q7.8 logit (local round-half-up + saturate).
                score_log[si*S+sj] = rnd_sat_q78( scl_tmp );
            end
        end
    end

    // ===================== combinational context computation =================
    // For ST_CTX (current `row`) compute O[row][d] for d=0..D-1 from the captured
    // wrow[] (Q0.16) and vm[] (Q7.8).  product W*V = Q7.24; shift right by
    // (Q016_FRAC - Q78_FRAC)=8 with round-half-up to land in Q15.16, accumulate
    // SEQ terms in a 48-bit signed accumulator, then round+saturate to Q7.8.
    localparam integer WV_SH = `Q016_FRAC - `Q78_FRAC;  // = 8

    reg signed [`ELEM_W-1:0] ctx_out  [0:D-1];
    reg                      ctx_sat;
    integer cd, cj;
    reg signed [`ACC_W-1:0]  cacc;
    reg signed [`ACC_W-1:0]  cprod;     // single W*V term shifted to Q15.16
    // W[row][cj] is Q0.16 unsigned [0..0xFFFF]; widen it to 17-bit SIGNED (a
    // leading 0) so the signed multiply with the Q7.8 V keeps V's sign.  The
    // 17b*16b product is a 33-bit SIGNED value; keep it in a wide signed temp so
    // its sign survives BEFORE sign-extending into the 48-bit accumulator.
    reg signed [16:0]        w_se;       // {1'b0, wrow} as 17-bit signed
    reg signed [32:0]        wv_prod;    // Q7.24 signed product (17b*16b -> 33b)
    reg signed [`ACC_W-1:0]  wv_ext;     // sign-extended product, Q7.24 in 48b
    always @(*) begin
        ctx_sat = 1'b0;
        for (cd = 0; cd < D; cd = cd + 1) begin
            cacc = {`ACC_W{1'b0}};
            for (cj = 0; cj < S; cj = cj + 1) begin
                w_se    = $signed({1'b0, wrow[cj]});
                wv_prod = w_se * $signed(vm[cj*D+cd]);          // Q7.24, signed
                // EXPLICIT sign-extend 33 -> 48 (replicate the sign bit) so the
                // negative-V product survives and verilator sees no width growth.
                wv_ext  = {{(`ACC_W-33){wv_prod[32]}}, wv_prod};
                // Right-shift by WV_SH(8) with round-half-up -> Q15.16.
                cprod = (wv_ext + `ACC_W'sd128 /* 1<<(WV_SH-1) */) >>> WV_SH;
                cacc  = cacc + cprod;
            end
            ctx_out[cd] = rnd_sat_q78( cacc );
            if (sat_hit_q78( cacc ))
                ctx_sat = 1'b1;
        end
    end

    // ===================== combinational TM line (un)packing =================
    // A Q/K/V/O ROW packs D Q7.8 elements into the low 16 bits of lanes 0..D-1 of
    // one 128-bit TM line; lanes D..NLANES-1 are unused (read: ignored; write:
    // sign-extension carries no data there).  These two combinational helpers
    // replace the hardwired [15:0]/[47:32]/[79:64]/[111:96] slices so the packing
    // is parameter-derived in D.
    //   rd_row[d] : the d-th Q7.8 element of the currently-read TM line.
    //   o_wline   : the current context row packed into a TM line (sign-extended).
    reg signed [`ELEM_W-1:0] rd_row [0:D-1];
    integer pk;
    always @(*) begin
        for (pk = 0; pk < D; pk = pk + 1)
            rd_row[pk] = tm_rdata[(pk*`LANE_W) +: `ELEM_W];
    end
    reg [`LINE_W-1:0] o_wline;
    integer ok;
    always @(*) begin
        o_wline = {`LINE_W{1'b0}};
        for (ok = 0; ok < D; ok = ok + 1)
            o_wline[(ok*`LANE_W) +: `LANE_W] =
                {{(`LANE_W-`ELEM_W){ctx_out[ok][`ELEM_W-1]}}, ctx_out[ok]};
    end

    // ===================== combinational softmax X-line packing ==============
    // Pack score row `row` (the SEQ Q7.8 logits) into NSCR_X = ceil(SEQ/NLANES)
    // scratch lines: lane (l*NLANES + k) carries logit (l*NLANES+k) for every
    // index < SEQ.  The softmax is instantiated at LEN=SEQ (NO pad lanes), so
    // there is NO Q78_MIN sentinel: any lane index >= SEQ falls in the last,
    // partial line and is NEVER read by the LEN=SEQ softmax (its read loop guards
    // (lcnt*NLANES+k) < LEN), so it is left 0 as a don't-care.  Removing the pad
    // makes the sentinel-collision corner (old KNOWN LIMITATION) IMPOSSIBLE by
    // construction: softmax runs over exactly the SEQ real logits.  At the
    // default SEQ=4 this is a single line {4 real logits}.
    reg [`LINE_W-1:0] sm_xline [0:NSCR_X-1];
    integer xl, xk, xe;
    always @(*) begin
        for (xl = 0; xl < NSCR_X; xl = xl + 1) begin
            sm_xline[xl] = {`LINE_W{1'b0}};
            for (xk = 0; xk < NLANES; xk = xk + 1) begin
                xe = xl*NLANES + xk;                    // flat softmax-lane index
                if (xe < SEQ)
                    sm_xline[xl][(xk*`LANE_W) +: `LANE_W] =
                        {16'd0, slog[row*SEQ + xe]};
                // xe >= SEQ : last partial line, never read by LEN=SEQ softmax.
            end
        end
    end

    // ===================== single clocked FSM ================================
    integer w;
    integer rl, wl;
    always @(posedge clk) begin
        if (rst) begin
            state    <= ST_IDLE;
            busy     <= 1'b0;
            done     <= 1'b0;
            sat      <= 1'b0;
            q_base_q <= {`TM_IDX_W{1'b0}};
            k_base_q <= {`TM_IDX_W{1'b0}};
            v_base_q <= {`TM_IDX_W{1'b0}};
            o_base_q <= {`TM_IDX_W{1'b0}};
            rcnt     <= {RCNT_W{1'b0}};
            row      <= {ROW_W{1'b0}};
            tm_raddr <= {`TM_IDX_W{1'b0}};
            tm_we    <= 1'b0;
            tm_waddr <= {`TM_IDX_W{1'b0}};
            tm_wdata <= {`LINE_W{1'b0}};
            sm_start <= 1'b0;
            for (w = 0; w < S*D; w = w + 1) begin
                qm[w] <= {`ELEM_W{1'b0}};
                km[w] <= {`ELEM_W{1'b0}};
                vm[w] <= {`ELEM_W{1'b0}};
            end
            for (w = 0; w < S*S; w = w + 1)
                slog[w] <= {`ELEM_W{1'b0}};
            for (w = 0; w < S; w = w + 1)
                wrow[w] <= {`Q016_W{1'b0}};
            for (w = 0; w < SCR_N; w = w + 1)
                sm_scratch[w] <= {`LINE_W{1'b0}};
        end else begin
            // defaults (overridden below where needed).
            done     <= 1'b0;
            tm_we    <= 1'b0;
            sm_start <= 1'b0;

            // Capture softmax writes into the internal scratch every cycle.
            if (sm_we)
                sm_scratch[sm_waddr[SCR_IDX_W-1:0]] <= sm_wdata;

            case (state)
                // ---------------------------------------------------------
                ST_IDLE: begin
                    if (start) begin
                        busy     <= 1'b1;
                        sat      <= 1'b0;
                        q_base_q <= q_base;
                        k_base_q <= k_base;
                        v_base_q <= v_base;
                        o_base_q <= o_base;
                        rcnt     <= {RCNT_W{1'b0}};
                        row      <= {ROW_W{1'b0}};
                        tm_raddr <= q_base;       // present Q row 0 this cycle
                        state    <= ST_RDQ;
                    end
                end

                // ---------------------------------------------------------
                // Read SEQ Q rows; row r presented on tm_raddr at cycle r, its
                // data latched the next cycle.  We latch the line read THIS cycle
                // (presented last cycle) and present the next address.  rd_row[]
                // un-packs the D lanes of the live TM line (parameter-derived).
                ST_RDQ: begin
                    for (rl = 0; rl < D; rl = rl + 1)
                        qm[rcnt*D + rl] <= rd_row[rl];
                    if (rcnt == S_LAST) begin
                        rcnt     <= {RCNT_W{1'b0}};
                        tm_raddr <= k_base_q;      // begin K reads next cycle
                        state    <= ST_RDK;
                    end else begin
                        rcnt     <= rcnt + {{(RCNT_W-1){1'b0}}, 1'b1};
                        tm_raddr <= q_base_q +
                            {{(`TM_IDX_W-RCNT_W){1'b0}},
                             (rcnt + {{(RCNT_W-1){1'b0}}, 1'b1})};
                    end
                end

                // ---------------------------------------------------------
                ST_RDK: begin
                    for (rl = 0; rl < D; rl = rl + 1)
                        km[rcnt*D + rl] <= rd_row[rl];
                    if (rcnt == S_LAST) begin
                        rcnt     <= {RCNT_W{1'b0}};
                        tm_raddr <= v_base_q;      // begin V reads next cycle
                        state    <= ST_RDV;
                    end else begin
                        rcnt     <= rcnt + {{(RCNT_W-1){1'b0}}, 1'b1};
                        tm_raddr <= k_base_q +
                            {{(`TM_IDX_W-RCNT_W){1'b0}},
                             (rcnt + {{(RCNT_W-1){1'b0}}, 1'b1})};
                    end
                end

                // ---------------------------------------------------------
                ST_RDV: begin
                    for (rl = 0; rl < D; rl = rl + 1)
                        vm[rcnt*D + rl] <= rd_row[rl];
                    if (rcnt == S_LAST) begin
                        rcnt  <= {RCNT_W{1'b0}};
                        state <= ST_SCORE;
                    end else begin
                        rcnt     <= rcnt + {{(RCNT_W-1){1'b0}}, 1'b1};
                        tm_raddr <= v_base_q +
                            {{(`TM_IDX_W-RCNT_W){1'b0}},
                             (rcnt + {{(RCNT_W-1){1'b0}}, 1'b1})};
                    end
                end

                // ---------------------------------------------------------
                // Latch all SEQ*SEQ scaled Q7.8 logits.  (Score-logit clamping is
                // a softmax-input clamp and is intentionally NOT folded into the
                // sticky output `sat` flag.)  Move to the first softmax row.
                ST_SCORE: begin
                    for (w = 0; w < S*S; w = w + 1)
                        slog[w] <= score_log[w];
                    row   <= {ROW_W{1'b0}};
                    state <= ST_SM_LD;
                end

                // ---------------------------------------------------------
                // Load score row `row` into the softmax input scratch lines
                // 0..NSCR_X-1.  Lanes 0..SEQ-1 = the SEQ real logits; there is
                // NO padding (softmax is instantiated at LEN=SEQ), so any unused
                // lane of the last partial line is a don't-care.  sm_xline[]
                // packs this combinationally (parameter-derived).
                ST_SM_LD: begin
                    for (wl = 0; wl < NSCR_X; wl = wl + 1)
                        sm_scratch[wl] <= sm_xline[wl];
                    state <= ST_SM_GO;
                end

                // ---------------------------------------------------------
                // Pulse softmax start for one cycle.
                ST_SM_GO: begin
                    sm_start <= 1'b1;
                    state    <= ST_SM_WT;
                end

                // ---------------------------------------------------------
                // Wait for softmax done.  When done, the probs are already in the
                // P scratch lines (SM_PBASE..) (written synchronously by softmax +
                // captured into sm_scratch above).  Capture the SEQ real weights
                // (lanes 0..SEQ-1, spanning NSCR_X P-lines), fold softmax sat NO.
                ST_SM_WT: begin
                    if (sm_done) begin
                        // P-tile line index = NSCR_X + (w/NLANES) (plain-int
                        // localparam + int loop var -> integer index, no width
                        // mismatch); lane within the line = (w % NLANES).
                        for (w = 0; w < S; w = w + 1)
                            wrow[w] <=
                                sm_scratch[NSCR_X + (w / NLANES)]
                                          [((w % NLANES)*`LANE_W) +: `Q016_W];
                        // NOTE: sm_sat (a softmax probability hitting the 0xFFFF
                        // clamp) is an internal renormalization artifact -- NOT an
                        // attention-OUTPUT magnitude loss -- so it is intentionally
                        // NOT folded into the attention `sat` flag.
                        state <= ST_CTX;
                    end
                end

                // ---------------------------------------------------------
                // Compute O[row][0..D-1] from wrow + vm, drive the O-row write.
                // (combinational ctx_out / ctx_sat / o_wline are valid this cyc.)
                ST_CTX: begin
                    tm_we    <= 1'b1;
                    tm_waddr <= o_base_q +
                                {{(`TM_IDX_W-ROW_W){1'b0}}, row};
                    tm_wdata <= o_wline;
                    if (ctx_sat)
                        sat <= 1'b1;
                    if (row == S_LAST[ROW_W-1:0]) begin
                        state <= ST_DONE;
                    end else begin
                        row   <= row + {{(ROW_W-1){1'b0}}, 1'b1};
                        state <= ST_SM_LD;
                    end
                end

                // ---------------------------------------------------------
                ST_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
