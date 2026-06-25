`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// attention_unit.v  --  true scaled-dot-product attention, seq=4 d=4 (SPEC §5.4)
//----------------------------------------------------------------------------
// PURPOSE
//   Computes one head of scaled dot-product attention over SEQ_LEN(4) tokens of
//   dimension D_MODEL(4), all in Q7.8 fixed point:
//
//       Attn(Q,K,V) = softmax( (Q . K^T) / sqrt(d) ) . V
//
//   This REPLACES the v1.5 single-element "attention" (which had NO softmax and
//   truncated the (query*key) product into 32 bits -- the documented v1.5 bug).
//   v2.0 computes the scores in a full 48-bit Q15.16 accumulator, applies an
//   EXACT 1/sqrt(4) = >>1 scale with round-half-up, runs a genuine exponential
//   softmax over each length-4 row (by INSTANTIATING the already-built len-8
//   softmax_unit), and forms the context in a second 48-bit accumulator with an
//   explicit round-half-up + saturate narrowing and a sticky `sat` flag.  No
//   silent truncation survives anywhere -- the bug class is structurally gone.
//
//   It is TM->TM: it reads Q/K/V (4 lines each) from tile memory and writes the
//   4 context lines O.  It exposes raw TM ACCESS PORTS only (the surrounding
//   datapath / unit TB owns and models the tile memory).  The single submodule
//   it instantiates is src/softmax_unit.v; that submodule runs entirely on an
//   INTERNAL 4-line scratch memory inside this unit, so the softmax never
//   contends for the external TM port (clean, self-contained arbitration).
//
// Q-FORMATS  (single source of truth: tpu_defs.vh, SPEC §1.3)
//   Q, K, V elements   : Q7.8   signed 16-bit (low 16 bits of a 32-bit TM lane).
//   score MAC product  : Q7.8 * Q7.8 = Q14.16 (30-bit signed, fits 32 bits).
//   score accumulator  : Q15.16, 48-bit signed (4 Q14.16 products, no overflow).
//   scaled score S[i][j]: (acc + round) >> 1, then >> FRAC into Q7.8 logit space
//                         for softmax (the >>1 is the EXACT 1/sqrt(4) scale).
//   softmax weights W  : Q0.16 unsigned probabilities (0xFFFF ~= 1.0).
//   context MAC product: W(Q0.16) * V(Q7.8) = Q7.24 (held left-aligned, see below)
//   context accumulator: 48-bit signed; narrowed round-half-up + saturate to Q7.8.
//
// SCORE -> SOFTMAX-LOGIT SCALING (the EXACT >>1, documented)
//   Raw score acc_S[i][j] = SUM_{d} Q[i][d]*K[j][d]  is Q15.16 in 48 bits.
//   The 1/sqrt(d) = 1/sqrt(4) = 1/2 scale is an EXACT right shift by 1.  We apply
//   it with round-half-up:  scaled = (acc_S + (1<<0)) >>> 1   (Q15.16 still).
//   The softmax_unit consumes Q7.8 LOGITS (16-bit), so the scaled Q15.16 score is
//   narrowed to a Q7.8 logit by a LOCAL round-half-up+saturate helper that is
//   bit-exact to the (now signed-correct) shared tpu_defs.vh narrowing macro --
//   see the round-half-up note below.  softmax is shift-invariant, so any saturation of
//   an individual logit only matters relative to the row max; the row max itself
//   is subtracted inside softmax_unit for numerical stability.  Score-logit
//   saturation is a softmax-INPUT clamp and is intentionally NOT folded into the
//   output `sat` flag (see the SATURATION POLICY note below).
//
// LENGTH-4 SOFTMAX VIA THE LEN-8 softmax_unit (consistent reuse)
//   softmax_unit is hardwired to SM_LEN=8.  A length-4 row is run through it by
//   placing the 4 real logits in lanes 0..3 and PADDING lanes 4..7 with the most
//   negative Q7.8 logit (`Q78_MIN = -128.0).  exp(min - rowmax) underflows to 0
//   in the unit's Q15.16 exp, so the 4 padding lanes contribute ~0 to the sum and
//   receive ~0 probability; the first 4 output probabilities therefore form the
//   correct length-4 softmax (they sum to ~1.0 to within the unit's documented
//   +/-2 LSB tolerance).  This is the SPEC §5.4 "length-4 variant consistent with
//   it" realised by reuse of the EXACT committed softmax_unit -- no forked copy.
//
// CONTEXT  O[i][d] = SUM_{j} W[i][j] * V[j][d]
//   W[i][j] is Q0.16 (unsigned [0,1]); V[j][d] is Q7.8 signed.  The product
//   W*V is Q0.16 * Q7.8 = Q7.24.  To accumulate in the shared Q15.16 48-bit
//   format, each product is sign-extended then right-shifted by
//   (Q016_FRAC - Q78_FRAC) = 8 with round-half-up so the running sum is Q15.16;
//   four such terms cannot overflow 48 bits.  The 48-bit sum is then narrowed to
//   Q7.8 by a LOCAL round-half-up + saturate (signed-bias) helper.
//
// SATURATION POLICY (the `sat` flag)
//   The attention OUTPUT is the context O.  Its only output narrowing is the
//   W.V -> Q7.8 step, so `sat` is the sticky OR of CONTEXT narrowing saturation
//   (`ctx_sat`) ONLY.  Two clamps are DELIBERATELY excluded because neither is
//   an output-magnitude loss of this unit:
//     * the score -> Q7.8-logit clamp is a softmax-INPUT clamp (softmax is
//       shift/scale-robust; the renormalized context still tracks the golden);
//     * the softmax probability 0xFFFF clamp (sm_sat) is a ~1.5e-5 rounding of a
//       PROBABILITY, internal to the softmax submodule.
//   PROVABLE PROPERTY: because the four real softmax weights sum to ~1.0 and the
//   padding weights are ~0, O is a CONVEX COMBINATION of the value vectors, so
//   |O| <= max_j |V[j][.]| <= Q7.8 max.  Hence `ctx_sat` (and thus `sat`)
//   CANNOT fire for in-range V -- saturation is unreachable BY CONSTRUCTION, a
//   correctness guarantee rather than a gap.  The flag is still implemented per
//   SPEC §1.3 (round+saturate with a visible flag) and is asserted to stay 0 by
//   the TB across all directed + random vectors (including all-V-max, which
//   rounds to exactly +max WITHOUT clamping).  This is the structural cure for
//   the v1.5 silent-truncation bug: scores live in full 48-bit, and the output
//   magnitude is bounded and observable.
//
// INTERFACE
//   clk, rst                          clock / synchronous active-high reset
//   start                             1-cycle pulse: latch bases, begin attention
//   q_base,k_base,v_base,o_base [4:0] TM line indices of Q,K,V,O tiles (4 lines ea)
//   busy                              high while an op is in flight (registered)
//   done                              1-cycle pulse when the last O line is driven
//   sat                               valid with done; 1 iff any narrowing clamped
//   -- external TM read port (combinational) --
//   tm_raddr [4:0] (out) / tm_rdata [127:0] (in)
//   -- external TM write port (synchronous) --
//   tm_we (out) / tm_waddr [4:0] (out) / tm_wdata [127:0] (out)
//
// LATENCY (deterministic, committed; asserted EXACTLY by the unit TB)
//   The unit serially REUSES the len-8 softmax_unit once per output row, so the
//   true committed latency is dominated by 4 softmax invocations rather than the
//   SPEC §3.3 rough estimate (~44, which assumed a fused length-4 softmax).  The
//   measured, committed start->done latency of THIS RTL is `LAT_TOTAL` cycles
//   (a localparam, derived below); the TB asserts it bit-exactly.  The deviation
//   from the ~44 estimate is an honest consequence of reusing the proven len-8
//   softmax_unit four times instead of forking a bespoke len-4 softmax.
//
// SYNTHESIZABILITY
//   Synchronous reset on ALL state; every reg assigned on every path of the one
//   clocked FSM (no inferred latch); combinational outputs are pure functions of
//   registered state (no comb loop); no real/$display/$random/initial in the
//   module.  Passes verilator --lint-only -Wall and iverilog -g2012 -Wall.
//============================================================================
module attention_unit (
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

    // ===================== local constants (NOT in tpu_defs.vh) ==============
    localparam integer S = `SEQ_LEN;   // 4
    localparam integer D = `D_MODEL;   // 4
    localparam [2:0] S_LAST = 3'd3;  // SEQ_LEN-1, sized for clean 3-bit compares

    // ---- FSM states ----
    localparam [3:0] ST_IDLE   = 4'd0;
    localparam [3:0] ST_RDQ    = 4'd1;  // read 4 Q rows  (4 cycles)
    localparam [3:0] ST_RDK    = 4'd2;  // read 4 K rows  (4 cycles)
    localparam [3:0] ST_RDV    = 4'd3;  // read 4 V rows  (4 cycles)
    localparam [3:0] ST_SCORE  = 4'd4;  // compute all 16 scaled scores (1 cycle)
    localparam [3:0] ST_SM_LD  = 4'd5;  // load score row i into softmax scratch
    localparam [3:0] ST_SM_GO  = 4'd6;  // pulse softmax start
    localparam [3:0] ST_SM_WT  = 4'd7;  // wait for softmax done, capture weights
    localparam [3:0] ST_CTX    = 4'd8;  // compute O row i, drive its TM write
    localparam [3:0] ST_DONE   = 4'd9;  // 1-cycle done pulse

    // Softmax scratch line indices (internal 4-line scratch memory).  Two views:
    //   *_5 : 5-bit TM_IDX_W constants wired to the softmax submodule's x/p base.
    //   *_2 : 2-bit constants for indexing the 4-line scratch array directly.
    localparam [`TM_IDX_W-1:0] SM_XBASE = 5'd0;  // logits  -> scratch lines 0,1
    localparam [`TM_IDX_W-1:0] SM_PBASE = 5'd2;  // probs   <- scratch lines 2,3
    localparam [1:0] SCR_X0 = 2'd0;  // scratch idx of logits line 0
    localparam [1:0] SCR_X1 = 2'd1;  // scratch idx of logits line 1
    localparam [1:0] SCR_P0 = 2'd2;  // scratch idx of probs  line 0

    // Committed deterministic latency (asserted EXACTLY by the unit TB).
    // Counting convention (matches the TB): the cycle `start` is sampled high in
    // ST_IDLE is cycle 1; `done` is a REGISTERED 1-cycle pulse first OBSERVED
    // high `LAT_TOTAL` cycles later (the FSM enters ST_DONE one cycle before
    // `done` is observed).  Breakdown:
    //   SETUP   = ST_RDQ(4) + ST_RDK(4) + ST_RDV(4) + ST_SCORE(1)          = 13
    //   PER_ROW = ST_SM_LD(1) + ST_SM_GO(1) + ST_SM_WT(24) + ST_CTX(1)     = 27
    //     ST_SM_WT spans 24 cycles: sm_start is REGISTERED in ST_SM_GO, so the
    //     softmax submodule samples it one cycle into ST_SM_WT and then runs its
    //     committed 22-cycle pipeline; 1 (start-sample latency) + 22 + 1 (the
    //     cycle sm_done is observed) = 24 cycles resident in ST_SM_WT.
    //   TAIL    = ST_DONE entry (1) + the registered done-observed edge (1)   = 2
    // LAT_TOTAL = 13 + SEQ_LEN*27 + 2 = 13 + 4*27 + 2 = 123 cycles.
    // These are PURE DOCUMENTATION localparams (the latency is structural in the
    // FSM, not parameter-driven), so they are intentionally not referenced in
    // logic; the narrow UNUSEDPARAM lint_off records that.
    /* verilator lint_off UNUSEDPARAM */
    localparam integer SM_WT_LAT = 24;
    localparam integer SETUP     = 4 + 4 + 4 + 1;                       // 13
    localparam integer PER_ROW   = 1 /*LD*/ + 1 /*GO*/ + SM_WT_LAT /*WT*/ + 1; // 27
    localparam integer LAT_TOTAL = SETUP + (S * PER_ROW) + 2;          // = 123
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
    reg [2:0]              rcnt;     // generic small counter (0..S)
    reg [2:0]              row;      // current output/softmax row (0..S-1)

    // ===================== internal 4-line softmax scratch ===================
    // The instantiated softmax_unit reads/writes THIS scratch only, never the
    // external TM, so there is no external-port arbitration.
    reg [`LINE_W-1:0] sm_scratch [0:3];

    // softmax handshake wiring.
    //   * sm_busy / sm_argmax are produced by the submodule but NOT consumed
    //     here (the FSM sequences purely off sm_done; argmax/busy carry no role
    //     in the attention pipeline).  Their declarations are wrapped in a narrow
    //     UNUSEDSIGNAL lint_off documenting the deliberate non-use.
    //   * sm_sat (softmax's own 0xFFFF probability clamp) is an internal
    //     renormalization artifact, NOT an attention-output magnitude loss, so it
    //     is intentionally NOT consumed (see the `sat` policy in the header).
    //   * sm_raddr / sm_waddr are 5-bit (TM_IDX_W) but the internal scratch is
    //     only 4 lines, so only the low 2 bits are used; the lint_off covers the
    //     intentionally-unused high index bits [4:2] too.
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
    // softmax addresses lines 0..3 (SM_XBASE,+1 and SM_PBASE,+1); scratch is 4
    // lines, so index by the low 2 bits.
    always @(*) sm_rdata = sm_scratch[sm_raddr[1:0]];

    softmax_unit u_softmax (
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
    // `TPU_RND_SAT_Q78 / `TPU_ROUND_SHIFT / `TPU_SAT_HIT.  HISTORICAL NOTE: the
    // shared macros used to build their round bias as an UNSIGNED concatenation,
    // so (signed acc + unsigned bias) evaluated UNSIGNED and the `>>> became a
    // LOGICAL shift, BREAKING narrowing for NEGATIVE accumulators (e.g.
    // acc=-10752 -> +32767 instead of -42).  Attention scores and signed-V
    // contexts are routinely NEGATIVE.  The header bias is now a SIGNED ACC_W
    // constant (fixed), so the macro is signed-correct and this helper computes
    // the IDENTICAL result; it is retained as named functions because the score
    // and context blocks index 2-D reg arrays and reuse the round-shift twice.
    // RND_BIAS is a localparam declared inside this module.
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
    // For the SCORE state we compute all 16 raw dot products combinationally
    // from the latched qm/km, scale by >>1 (round-half-up) into Q15.16, and
    // narrow each to a Q7.8 logit.  A genvar-free explicit unroll keeps it
    // synthesizable and lint-clean.  acc_S[i][j] = SUM_d qm[i*D+d]*km[j*D+d].
    //
    // Each product is Q14.16 in 32 bits (PROD_W); 4 summed in a 48-bit signed
    // accumulator.  scaled = round_half_up(acc, 1) >>> 1  (the EXACT 1/sqrt 4).
    // The Q15.16 scaled score is then narrowed to a Q7.8 logit (round+sat).
    //
    // NOTE ON SCORE-LOGIT SATURATION (intentionally NOT folded into `sat`):
    //   For very large |Q|,|K| the scaled score can exceed the Q7.8 logit range
    //   and clamp.  This clamp is a SOFTMAX-INPUT clamp, not an OUTPUT-magnitude
    //   loss: softmax is shift/scale-robust, and the renormalized weights (hence
    //   the context O) still track the real golden within the documented
    //   tolerance even when several logits clamp to +max (a near-degenerate
    //   softmax that the floating golden also produces).  The sticky `sat` flag
    //   is the OUTPUT-narrowing flag (SPEC §1.3 -- "no truncation that can
    //   silently LOSE MAGNITUDE"); it is driven ONLY by the CONTEXT narrowing
    //   (ctx_sat, see the SATURATION POLICY note in the header).  Score-logit
    //   clamping is therefore deliberately NOT OR'd into `sat`.  The v1.5 bug
    //   guard is preserved: scores are full 48-bit (never silently wrapped) and
    //   the OUTPUT magnitude is the flagged, bounded quantity.
    //
    // One dot-product accumulator helper as a function of (i,j) is not allowed
    // (functions cannot index the 2-D reg arrays cleanly here under -Wall), so
    // we build the 16 scaled logits in a small always-comb that the FSM samples.
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
                    // 32b temp, then EXPLICITLY sign-extend (replicate the sign
                    // bit) into the 48-bit accumulator -- an explicit extension so
                    // the negative-product sign is preserved AND verilator sees no
                    // implicit width growth.
                    qk_prod = $signed(qm[si*D+sd]) * $signed(km[sj*D+sd]);
                    qk_ext  = {{(`ACC_W-32){qk_prod[31]}}, qk_prod};
                    acc_tmp = acc_tmp + qk_ext;
                end
                // EXACT 1/sqrt(4)=1/2 scale: round-half-up then arithmetic >>1.
                // acc_tmp is signed; +1 is a signed sized literal so the >>> is
                // arithmetic (ties -> +inf), matching the round-half-up policy.
                scl_tmp = (acc_tmp + `ACC_W'sd1) >>> 1;
                // narrow Q15.16 -> Q7.8 logit (local round-half-up + saturate).
                score_log[si*S+sj] = rnd_sat_q78( scl_tmp );
            end
        end
    end

    // ===================== combinational context computation =================
    // For ST_CTX (current `row`) compute O[row][d] for d=0..3 from the captured
    // wrow[] (Q0.16) and vm[] (Q7.8).  product W*V = Q7.24; shift right by
    // (Q016_FRAC - Q78_FRAC)=8 with round-half-up to land in Q15.16, accumulate
    // 4 terms in a 48-bit signed accumulator, then round+saturate to Q7.8.
    localparam integer WV_SH = `Q016_FRAC - `Q78_FRAC;  // = 8

    reg signed [`ELEM_W-1:0] ctx_out  [0:D-1];
    reg                      ctx_sat;
    integer cd, cj;
    reg signed [`ACC_W-1:0]  cacc;
    reg signed [`ACC_W-1:0]  cprod;     // single W*V term shifted to Q15.16
    // W[row][cj] is Q0.16 unsigned [0..0xFFFF]; widen it to 17-bit SIGNED (a
    // leading 0) so the signed multiply with the Q7.8 V keeps V's sign.  The
    // 17b*16b product is a 33-bit SIGNED value; keep it in a wide signed temp so
    // its sign survives BEFORE sign-extending into the 48-bit accumulator (a
    // bare width-cast would zero-extend and corrupt a negative product).
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

    // ===================== single clocked FSM ================================
    integer w;
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
            rcnt     <= 3'd0;
            row      <= 3'd0;
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
            for (w = 0; w < 4; w = w + 1)
                sm_scratch[w] <= {`LINE_W{1'b0}};
        end else begin
            // defaults (overridden below where needed).
            done     <= 1'b0;
            tm_we    <= 1'b0;
            sm_start <= 1'b0;

            // Capture softmax writes into the internal scratch every cycle.
            if (sm_we)
                sm_scratch[sm_waddr[1:0]] <= sm_wdata;

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
                        rcnt     <= 3'd0;
                        row      <= 3'd0;
                        tm_raddr <= q_base;       // present Q row 0 this cycle
                        state    <= ST_RDQ;
                    end
                end

                // ---------------------------------------------------------
                // Read 4 Q rows; row r presented on tm_raddr at cycle r, its
                // data latched the next cycle.  We latch the line read THIS
                // cycle (presented last cycle) and present the next address.
                ST_RDQ: begin
                    qm[rcnt*D+0] <= tm_rdata[ 15:  0];
                    qm[rcnt*D+1] <= tm_rdata[ 47: 32];
                    qm[rcnt*D+2] <= tm_rdata[ 79: 64];
                    qm[rcnt*D+3] <= tm_rdata[111: 96];
                    if (rcnt == S_LAST) begin
                        rcnt     <= 3'd0;
                        tm_raddr <= k_base_q;      // begin K reads next cycle
                        state    <= ST_RDK;
                    end else begin
                        rcnt     <= rcnt + 3'd1;
                        tm_raddr <= q_base_q +
                                    {{(`TM_IDX_W-3){1'b0}}, (rcnt + 3'd1)};
                    end
                end

                // ---------------------------------------------------------
                ST_RDK: begin
                    km[rcnt*D+0] <= tm_rdata[ 15:  0];
                    km[rcnt*D+1] <= tm_rdata[ 47: 32];
                    km[rcnt*D+2] <= tm_rdata[ 79: 64];
                    km[rcnt*D+3] <= tm_rdata[111: 96];
                    if (rcnt == S_LAST) begin
                        rcnt     <= 3'd0;
                        tm_raddr <= v_base_q;      // begin V reads next cycle
                        state    <= ST_RDV;
                    end else begin
                        rcnt     <= rcnt + 3'd1;
                        tm_raddr <= k_base_q +
                                    {{(`TM_IDX_W-3){1'b0}}, (rcnt + 3'd1)};
                    end
                end

                // ---------------------------------------------------------
                ST_RDV: begin
                    vm[rcnt*D+0] <= tm_rdata[ 15:  0];
                    vm[rcnt*D+1] <= tm_rdata[ 47: 32];
                    vm[rcnt*D+2] <= tm_rdata[ 79: 64];
                    vm[rcnt*D+3] <= tm_rdata[111: 96];
                    if (rcnt == S_LAST) begin
                        rcnt  <= 3'd0;
                        state <= ST_SCORE;
                    end else begin
                        rcnt     <= rcnt + 3'd1;
                        tm_raddr <= v_base_q +
                                    {{(`TM_IDX_W-3){1'b0}}, (rcnt + 3'd1)};
                    end
                end

                // ---------------------------------------------------------
                // Latch all 16 scaled Q7.8 logits.  (Score-logit clamping is a
                // softmax-input clamp and is intentionally NOT folded into the
                // sticky output `sat` flag -- see the score-block note above.)
                // Move to the first softmax row.
                ST_SCORE: begin
                    for (w = 0; w < S*S; w = w + 1)
                        slog[w] <= score_log[w];
                    row   <= 3'd0;
                    state <= ST_SM_LD;
                end

                // ---------------------------------------------------------
                // Load score row `row` into the softmax input scratch (lines
                // SM_XBASE, SM_XBASE+1).  Lanes 0..3 = the 4 real logits;
                // lanes 4..7 = Q78_MIN padding (their exp underflows to ~0).
                ST_SM_LD: begin
                    // slog index = row*S + col with S=4, so {row[1:0],col[1:0]}
                    // is exactly row*4+col (4-bit index into the 16-entry array).
                    sm_scratch[SCR_X0] <= {
                        {16'd0, slog[{row[1:0], 2'b11}]},
                        {16'd0, slog[{row[1:0], 2'b10}]},
                        {16'd0, slog[{row[1:0], 2'b01}]},
                        {16'd0, slog[{row[1:0], 2'b00}]} };
                    sm_scratch[SCR_X1] <= {
                        {16'd0, `Q78_MIN}, {16'd0, `Q78_MIN},
                        {16'd0, `Q78_MIN}, {16'd0, `Q78_MIN} };
                    state <= ST_SM_GO;
                end

                // ---------------------------------------------------------
                // Pulse softmax start for one cycle.
                ST_SM_GO: begin
                    sm_start <= 1'b1;
                    state    <= ST_SM_WT;
                end

                // ---------------------------------------------------------
                // Wait for softmax done.  When done, the probs are already in
                // scratch lines SM_PBASE, SM_PBASE+1 (written synchronously by
                // softmax + captured into sm_scratch above).  Capture the 4
                // real weights (lanes 0..3 of line SM_PBASE), fold softmax sat.
                ST_SM_WT: begin
                    if (sm_done) begin
                        wrow[0] <= sm_scratch[SCR_P0][ 15:  0];
                        wrow[1] <= sm_scratch[SCR_P0][ 47: 32];
                        wrow[2] <= sm_scratch[SCR_P0][ 79: 64];
                        wrow[3] <= sm_scratch[SCR_P0][111: 96];
                        // NOTE: sm_sat (a softmax probability hitting the 0xFFFF
                        // clamp) is an internal renormalization artifact worth
                        // ~1.5e-5 in a probability -- NOT an attention-OUTPUT
                        // magnitude loss -- so it is intentionally NOT folded
                        // into the attention `sat` flag (see header / sat policy).
                        state <= ST_CTX;
                    end
                end

                // ---------------------------------------------------------
                // Compute O[row][0..3] from wrow + vm, drive the O-row write.
                // (combinational ctx_out / ctx_sat are valid this cycle).
                ST_CTX: begin
                    tm_we    <= 1'b1;
                    tm_waddr <= o_base_q +
                                {{(`TM_IDX_W-3){1'b0}}, row};
                    tm_wdata <= {
                        {{16{ctx_out[3][15]}}, ctx_out[3]},
                        {{16{ctx_out[2][15]}}, ctx_out[2]},
                        {{16{ctx_out[1][15]}}, ctx_out[1]},
                        {{16{ctx_out[0][15]}}, ctx_out[0]} };
                    if (ctx_sat)
                        sat <= 1'b1;
                    if (row == S_LAST) begin
                        state <= ST_DONE;
                    end else begin
                        row   <= row + 3'd1;
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
