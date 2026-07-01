`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
//============================================================================
// mla_attn_fp8.v  --  GLM-5.2 MLA latent attention, ONE query token (decode),
//                     FP8-NATIVE WEIGHT PROJECTIONS.            (ACCEL_GLM52 §4.1,§6)
//----------------------------------------------------------------------------
// FUNCTION  (the FP8 sibling of mla_attn.v -- identical FSM/dataflow/latency
//   structure; ONLY the SEVEN WEIGHT-MATRIX GEMMs change numerics)
//
//   This is the SAME large orchestrator FSM as mla_attn.v.  Every stage, buffer,
//   handshake and the deterministic latency are mirrored EXACTLY.  The ONE change
//   is the GEMM datapath used for the seven LARGE LINEAR WEIGHT projections:
//
//     W_dq, W_uq, W_dkv, W_kr(W_krope), W_uk, W_uv, W_o
//        -> glm_matmul_fp8  (official GLM-5.2-FP8 numerics), exactly like
//           swiglu_expert_fp8 wires glm_matmul_fp8.
//
//   EVERYTHING ELSE STAYS bf16, UNCHANGED FROM mla_attn.v (RMSNorm, decoupled
//   RoPE, the per-head q.K SCORE matmul (ACTIVATION x ACTIVATION, bf16 engine),
//   the weighted-V context, glm_softmax, dsa_indexer, the c_kv / k_rope caches).
//
//============================================================================
// PE_M BATCHING (B query-token ROWS share ONE weight fetch)        (ULTRA_PERF#2)
//----------------------------------------------------------------------------
//   PE_M (default 1 == byte-identical to the original single-token MLA decode) is
//   the number of QUERY-TOKEN ROWS pushed through the SAME projection weights in
//   one pass.  glm_matmul_fp8 / glm_matmul_pipe are already PE_M-ready: each
//   streams PE_M activation lanes (a_col[16*PE_M], a_shift[8*PE_M]) against ONE
//   weight column (w_row, SHARED) and emits PE_M*PE_N results, time-sharing the
//   weight stream + the dequant multipliers.  So widening PE_M costs activation-
//   lane area + per-row attention state but adds ZERO extra weight bandwidth: the
//   w_req / w_sel / w_grp / w_k request stream and the w_col / w_scale responses
//   are IDENTICAL to PE_M=1 -- ONE Flash fetch feeds all B rows.
//
//   WHICH PROJECTIONS BATCH OVER QUERY ROWS:
//     * W_dq, W_uq, W_dkv, W_kr, W_o : activation is per-query-row (x / qlora_n /
//       ctx).  These BATCH: B rows' activations stream against the one shared
//       weight column, each row carrying its OWN dynamic per-vector pow2 a_shift
//       (from its own activation's exp-max).  Row r's projection output is
//       BIT-IDENTICAL to a PE_M=1 run on row r (glm_matmul_fp8 accumulates every
//       (row,col) independently).
//     * W_uk, W_uv : the activation is ckv_n = RMSNorm(c_kv[key]), a CACHE-KEY
//       latent that is SHARED across all query rows (it depends only on the key,
//       not the query).  These are computed ONCE PER KEY (weights fetched once
//       per key, NOT per query row) and the resulting K/V are shared by every
//       row's score/context.  (In the matmul they are driven on lane 0; PE_M>1
//       lanes are don't-care here.)
//
//   PER-ROW ATTENTION (kept per-row, replicated PE_M-wide, lockstep):
//     RMSNorm(q_lora), decoupled RoPE(q), the q.K SCORE matmul (per-row q against
//     the SHARED key K via the bf16 engine's PE_M lanes), glm_softmax, and the
//     weighted-V context all fan out to PE_M.  The sub-units (rmsnorm_q, rope,
//     softmax) are REPLICATED PE_M times and run in lockstep off ONE shared
//     control handshake (their control timing is data-independent), each fed its
//     own row's data -- so the FSM cycle structure is UNCHANGED and PE_M=1 folds
//     to exactly the committed single-row datapath.
//
//   PER-ROW QUERY POSITION (pos_vec):  each row r carries its OWN query position
//     pos_r (pos_vec[POSW*r +: POSW]; row 0 = the scalar `pos`).  pos_r drives ONLY
//     the per-row QUERY RoPE rotation (qrot[r]) -- and the per-row current-token
//     k_rope coverage pass -- which then flows through the ALREADY-per-row score /
//     softmax / weighted-V context / W_o.  So row r's output is EXACTLY the single-
//     token mla_attn_fp8 result for (x_r, pos_r, s_len).  rope_interleave_unit
//     captures pos at start and pos affects ONLY its angle datapath (never its FSM
//     timing), so the PE_M replicas stay in perfect lockstep off ONE shared control
//     handshake even with different pos_r.  At PE_M=1 (or all pos_r equal) every
//     RoPE replica sees the same angle -> byte-identical to the committed module.
//
//   SHARED-CONTEXT ASSUMPTION (documented; holds for batched decode at one step):
//     s_len and the KV cache (kc_*) -- i.e. the CAUSAL PREFIX EXTENT and the cached
//     key latents -- are SHARED across the B rows (same context window); the KEY
//     projections (W_uk/W_uv, the c_kv RMSNorm, V) depend only on the key, not the
//     query, so they too are shared/computed-once-per-key.  Rows differ in their
//     token activation x AND now their query position pos_r.  (Keeping the causal
//     extent = the shared s_len is also REQUIRED for byte-identicality: the
//     committed datapath attends all s_len keys regardless of pos, e.g. pos<s_len.)
//     The DSA top-K selection is driven from row 0's q and SHARED across rows --
//     EXACT in the dense fallback (S <= TOPK, where dsa_indexer ignores q and
//     keeps keys 0..S-1), which is the regime the TB and this decode slice use.
//     (Sparse per-row selection divergence is out of scope for PE_M>1.)
//
//   At PE_M=1 every PE_M-indexed construct constant-folds to the original single-
//   row datapath -> the committed test/mla_attn_fp8_tb.v instantiates this
//   unchanged (identical ports) and passes byte-identically.
//----------------------------------------------------------------------------
// STYLE: sync active-high reset; NO latch; NO comb loop; deterministic,
//   handshake-driven latency (absorbs the FP8 matmul's own latency via out_valid).
//============================================================================
module mla_attn_fp8 #(
    parameter integer MODEL_DIM = 128,
    parameter integer H_HEADS   = 4,
    parameter integer NOPE      = 16,
    parameter integer ROPE      = 16,
    parameter integer V_DIM     = 32,
    parameter integer Q_LORA    = 64,
    parameter integer KV_LORA   = 32,
    parameter integer S_MAX     = 8,
    parameter integer TOPK      = 8,
    parameter integer THETA     = 8000000,
    parameter integer PE_N      = 4,    // matmul tile width (output lanes/pass)
    parameter integer POSW      = 20,
    parameter integer BLK       = 128,  // weight block size along K -- [128,128]
    parameter integer PE_M      = 1,    // query-token ROWS (batch B) sharing one weight fetch
    // PER_ROW_POS=0 (default): every row decodes RoPE at the SHARED scalar `pos`
    //   (pos_vec IGNORED) -- byte-identical to the pre-per-row path and SAFE when a
    //   caller leaves pos_vec unconnected (no silent position-0 corruption).  =1:
    //   rows 1..PE_M-1 decode at their OWN pos_vec slice (row 0 still = `pos`).
    parameter integer PER_ROW_POS = 0,
    // ---- derived (do NOT override) ----
    parameter integer QK_DIM    = NOPE + ROPE,
    parameter integer IDXW      = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer HQK       = H_HEADS * QK_DIM,   // q width
    parameter integer HNOPE     = H_HEADS * NOPE,     // k_nope width
    parameter integer HV        = H_HEADS * V_DIM,    // v width (and W_o K)
    // largest reduction K across all projections (matmul counter + w_k width)
    parameter integer KMAX      = (MODEL_DIM > Q_LORA) ?
                               ((MODEL_DIM > KV_LORA) ?
                                ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV))
                             : ((Q_LORA > KV_LORA) ?
                                ((Q_LORA > HV) ? Q_LORA : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV)),
    // largest output length across projections (tile-group counter sizing)
    parameter integer OMAX      = (HQK > MODEL_DIM) ?
                               ((HQK > HNOPE) ?
                                 ((HQK > HV) ? HQK : HV)
                               : ((HNOPE > HV) ? HNOPE : HV))
                             : ((MODEL_DIM > HNOPE) ?
                                 ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                               : ((HNOPE > HV) ? HNOPE : HV)),
    parameter integer NGMAX     = (OMAX + PE_N - 1) / PE_N,
    parameter integer GRPW      = (NGMAX <= 1) ? 1 : $clog2(NGMAX),
    parameter integer KCW       = (KMAX  <= 1) ? 1 : $clog2(KMAX),
    parameter integer KW        = $clog2(KMAX + 1),       // matmul k_len width
    parameter integer NB        = (KMAX + BLK - 1) / BLK  // #K-blocks (weight scales)
)(
    input  wire                        clk,
    input  wire                        rst,        // sync, active-high

    // ---- control ----
    input  wire                        start,
    output reg                         busy,
    output reg                         done,
    input  wire [POSW-1:0]             pos,        // token position (for RoPE) -- ROW 0 / PE_M=1 (shared default)
    // PER-ROW query positions (ONLY consulted when PER_ROW_POS=1): row r decodes
    //   RoPE at pos_vec[POSW*r +: POSW].  Row 0 always uses the scalar `pos`.  With
    //   the default PER_ROW_POS=0 this port is ignored and every row shares `pos`,
    //   so an unconnected pos_vec is SAFE (shared-pos, byte-identical).
    input  wire [POSW*PE_M-1:0]        pos_vec,    // per-row query positions (rows 1..; row 0 = `pos`)
    input  wire [IDXW:0]               s_len,      // S causal keys (<= S_MAX) -- SHARED across rows (KV prefix)

    // ---- x input (latched at start) -- PE_M rows, row-major packed ----
    //   row r element k = x_vec[16*(MODEL_DIM*r + k) +: 16]
    input  wire [MODEL_DIM*16*PE_M-1:0] x_vec,     // PE_M * MODEL_DIM bf16

    // ---- weight pull (combinational responder; FP8 codes + block scales) -- SHARED by all rows ----
    output reg                         w_req,
    output reg  [3:0]                  w_sel,      // 0..6 projection select
    output reg  [GRPW-1:0]             w_grp,      // output tile-group index
    output reg  [KCW-1:0]              w_k,        // reduction index k of this beat
    input  wire [PE_N*8-1:0]           w_col,      // PE_N FP8 E4M3 weight lanes
    input  wire [16*PE_N*NB-1:0]       w_scale,    // bf16 block scales (sel,grp)

    // ---- cache read (past-key latents from caller's KV cache) -- SHARED across rows ----
    output reg                         kc_req,
    output reg  [IDXW-1:0]             kc_idx,     // requested causal key j
    input  wire [KV_LORA*16-1:0]       kc_ckv,     // cached c_kv[j]   (bf16)
    input  wire [ROPE*16-1:0]          kc_krope,   // cached k_rope[j] (bf16, roped)
    input  wire                        kc_valid,

    // ---- output -- PE_M rows, row-major packed ----
    //   row r element o = out[16*(MODEL_DIM*r + o) +: 16]
    output reg  [MODEL_DIM*16*PE_M-1:0] out        // PE_M * MODEL_DIM bf16
);
    `include "glm_fp.vh"

    //========================================================================
    // weight-select codes (w_sel)
    //========================================================================
    localparam [3:0] SEL_DQ=4'd0, SEL_UQ=4'd1, SEL_DKV=4'd2, SEL_KR=4'd3,
                     SEL_UK=4'd4, SEL_UV=4'd5, SEL_O=4'd6;

    // RoPE LANES: process the rope vector in lanes that divide its pair count.
    localparam integer ROPE_LANES = 1;

    integer tt;
    integer rr;        // PE_M row loop variable (sequential blocks)

    //========================================================================
    // DYNAMIC per-vector pow2 ACTIVATION SCALE (a_shift) -- pure exp maxes.
    //   a_shift = clamp(134 - emax) so the max element's scaled exp becomes 7.
    //   emax==0 (all-zero vector) -> a_shift = 0.  Identical to swiglu_expert_fp8.
    //========================================================================
    function automatic signed [7:0] dyn_shift(input [7:0] emax);
        reg signed [9:0] sh;
        begin
            if (emax == 8'd0) dyn_shift = 8'sd0;
            else begin
                sh = 10'sd134 - $signed({2'b0, emax});
                if (sh > 10'sd127)  sh = 10'sd127;
                if (sh < -10'sd128) sh = -10'sd128;
                dyn_shift = sh[7:0];
            end
        end
    endfunction

    //========================================================================
    // INPUT / INTERMEDIATE BUFFERS  (bf16 storage; fp32 only inside sub-units)
    //   PER-ROW buffers carry a leading [0:PE_M-1] dim; SHARED (key-derived)
    //   buffers do not.
    //========================================================================
    reg [15:0] xbuf      [0:PE_M-1][0:MODEL_DIM-1];   // latched x (per row)
    reg [POSW*PE_M-1:0] pos_qr;                        // PER-ROW query positions (latched): row0=pos, rows=pos_vec
    reg [IDXW:0]   s_reg;                              // shared S causal keys (KV prefix extent)

    reg [15:0] qlora     [0:PE_M-1][0:Q_LORA-1];       // x*W_dq        (per row)
    reg [15:0] qlora_n   [0:PE_M-1][0:Q_LORA-1];       // RMSNorm(qlora)(per row)
    reg [15:0] qfull     [0:PE_M-1][0:HQK-1];          // qlora_n*W_uq  (per row)
    reg [15:0] qrot      [0:PE_M-1][0:HQK-1];          // roped q       (per row)

    // ckv_cur = x*W_dkv current-token latent: exercised for FP8 datapath coverage /
    // X-freeness but (as in mla_attn_tb) NOT consumed downstream -> write-only.
    /* verilator lint_off UNUSEDSIGNAL */
    reg [15:0] ckv_cur   [0:PE_M-1][0:KV_LORA-1];      // x*W_dkv  (per-row datapath coverage)
    /* verilator lint_on UNUSEDSIGNAL */
    reg [15:0] krope_cur [0:PE_M-1][0:ROPE-1];         // x*W_kr -> roped (per-row coverage)

    reg [15:0] ckv_key   [0:KV_LORA-1];                // cache key latent c_kv[j] (SHARED)
    reg [15:0] ckv_n     [0:KV_LORA-1];                // RMSNorm(c_kv[j])         (SHARED)
    reg [15:0] knope_j   [0:HNOPE-1];                  // ckv_n*W_uk (per head)    (SHARED)
    reg [15:0] v_j       [0:HV-1];                     // ckv_n*W_uv (per head)    (SHARED)
    reg [15:0] krope_j   [0:ROPE-1];                   // cached k_rope[j]         (SHARED)

    reg [15:0] scores    [0:PE_M-1][0:H_HEADS-1][0:S_MAX-1];           // per row
    reg [15:0] vstore    [0:H_HEADS-1][0:S_MAX-1][0:V_DIM-1];          // SHARED (key V)
    reg [15:0] probs     [0:PE_M-1][0:H_HEADS-1][0:S_MAX-1];           // per row
    reg [15:0] ctx       [0:PE_M-1][0:HV-1];                           // per row (O concat)
    reg [15:0] outbuf    [0:PE_M-1][0:MODEL_DIM-1];                    // per row

    // DSA selection results (SHARED -- dense-fallback / row-0-driven; see header)
    reg [IDXW-1:0] sel_list [0:TOPK-1];
    reg [IDXW:0]   sel_cnt;

    //========================================================================
    // SHARED GEMV ENGINES.  PE_M activation rows, ONE shared weight stream.
    //   The SEVEN WEIGHT projections go to the FP8 engine (glm_matmul_fp8); the
    //   q.K SCORE pass (ACTIVATION x ACTIVATION) goes to the bf16 engine
    //   (glm_matmul_pipe).  Both PE_M wide, PE_N tile width.  out_valid / c_out
    //   are muxed on gv_score (stable across a pass).
    //========================================================================
    reg                  mm_start;
    reg  [KW-1:0]        mm_klen;
    reg                  mm_in_valid;
    reg  [16*PE_M-1:0]   mm_a;                    // PE_M packed A elements (one/row)
    // this GEMV pass is a q.K SCORE (-> bf16 engine) rather than a weight pass.
    reg                  gv_score;

    // ---- FP8 weight-projection engine ----
    wire                 fp8_busy, fp8_ov;
    wire [16*PE_M*PE_N-1:0] fp8_c;
    // ---- bf16 score (activation x activation) engine ----
    wire                 bf16_busy, bf16_ov;
    wire [16*PE_M*PE_N-1:0] bf16_c;
    reg  [PE_N*16-1:0]   score_w_lanes;           // assembled K lanes (lane0 meaningful; SHARED)

    // dynamic per-ROW activation pow2 scale (packed signed-8 per row).
    wire [8*PE_M-1:0]    a_shift_comb;

    glm_matmul_fp8 #(.PE_M(PE_M), .PE_N(PE_N), .KMAX(KMAX), .BLK(BLK)) u_mm_fp8 (
        .clk(clk), .rst(rst),
        .start(mm_start & ~gv_score), .k_len(mm_klen),
        .in_valid(mm_in_valid & ~gv_score), .a_col(mm_a), .w_row(w_col),
        .a_shift(a_shift_comb), .w_scale(w_scale),
        .busy(fp8_busy), .out_valid(fp8_ov), .c_out(fp8_c)
    );

    glm_matmul_pipe #(.PE_M(PE_M), .PE_N(PE_N), .KMAX(KMAX)) u_mm_bf16 (
        .clk(clk), .rst(rst),
        .start(mm_start & gv_score), .k_len(mm_klen),
        .in_valid(mm_in_valid & gv_score), .a_col(mm_a), .w_row(score_w_lanes),
        .busy(bf16_busy), .out_valid(bf16_ov), .c_out(bf16_c)
    );

    // muxed matmul result (gv_score is stable for the whole pass).
    wire                    mm_out_valid = gv_score ? bf16_ov : fp8_ov;
    wire [16*PE_M*PE_N-1:0] mm_c         = gv_score ? bf16_c  : fp8_c;

    //========================================================================
    // SUB-UNIT : rmsnorm_unit for q_lora -- PE_M replicated, lockstep.
    //   (the cache-key c_kv RMSNorm below is SHARED -> single instance.)
    //========================================================================
    reg               rnq_start;
    wire [PE_M-1:0]   rnq_in_req, rnq_g_req, rnq_y_valid, rnq_busy, rnq_done;
    wire [16*PE_M-1:0] rnq_y_out;
    reg  [16*PE_M-1:0] rnq_x_in, rnq_gamma_in;
    reg               rnq_x_valid, rnq_g_valid;
    genvar gq;
    generate
    for (gq = 0; gq < PE_M; gq = gq + 1) begin : RNQ
        rmsnorm_unit #(.LEN(Q_LORA), .LANES(1)) u_rn_q (
            .clk(clk), .rst(rst), .start(rnq_start),
            .in_req(rnq_in_req[gq]), .x_in(rnq_x_in[16*gq +: 16]), .x_valid(rnq_x_valid),
            .g_req(rnq_g_req[gq]), .gamma_in(rnq_gamma_in[16*gq +: 16]), .g_valid(rnq_g_valid),
            .y_valid(rnq_y_valid[gq]), .y_out(rnq_y_out[16*gq +: 16]),
            .busy(rnq_busy[gq]), .done(rnq_done[gq])
        );
    end
    endgenerate

    // cache-key c_kv RMSNorm -- SHARED (single instance).
    reg               rnk_start;
    wire              rnk_in_req, rnk_g_req, rnk_y_valid, rnk_busy, rnk_done;
    wire [15:0]       rnk_y_out;
    reg  [15:0]       rnk_x_in, rnk_gamma_in;
    reg               rnk_x_valid, rnk_g_valid;
    rmsnorm_unit #(.LEN(KV_LORA), .LANES(1)) u_rn_k (
        .clk(clk), .rst(rst), .start(rnk_start),
        .in_req(rnk_in_req), .x_in(rnk_x_in), .x_valid(rnk_x_valid),
        .g_req(rnk_g_req), .gamma_in(rnk_gamma_in), .g_valid(rnk_g_valid),
        .y_valid(rnk_y_valid), .y_out(rnk_y_out), .busy(rnk_busy), .done(rnk_done)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _busy_unused = &{1'b0, rnq_busy, rnk_busy, fp8_busy, bf16_busy, rnq_g_req[0]};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // SUB-UNIT : rope_interleave_unit -- PE_M replicated, lockstep.  Serves the
    //   q_rope per-head pass and the (per-row) current-token k_rope pass.
    //========================================================================
    reg               rp_start;
    reg  [POSW*PE_M-1:0] rp_pos;                       // PER-ROW RoPE position (one slice / replica)
    wire [PE_M-1:0]   rp_in_req, rp_y_valid, rp_busy, rp_done;
    reg  [ROPE_LANES*32*PE_M-1:0] rp_x_in;
    wire [ROPE_LANES*32*PE_M-1:0] rp_y_out;
    reg               rp_x_valid;
    genvar gp;
    generate
    for (gp = 0; gp < PE_M; gp = gp + 1) begin : RP
        rope_interleave_unit #(.ROT_DIM(ROPE), .THETA(THETA),
                               .LANES(ROPE_LANES), .POSW(POSW)) u_rope (
            .clk(clk), .rst(rst), .start(rp_start), .pos(rp_pos[POSW*gp +: POSW]),
            .in_req(rp_in_req[gp]), .x_in(rp_x_in[ROPE_LANES*32*gp +: ROPE_LANES*32]),
            .x_valid(rp_x_valid),
            .y_valid(rp_y_valid[gp]), .y_out(rp_y_out[ROPE_LANES*32*gp +: ROPE_LANES*32]),
            .busy(rp_busy[gp]), .done(rp_done[gp])
        );
    end
    endgenerate
    /* verilator lint_off UNUSEDSIGNAL */
    wire _rp_busy_unused = &{1'b0, rp_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // SUB-UNIT : dsa_indexer (top-K key selection) -- SHARED (single instance,
    //   driven from row 0's q; EXACT in the dense fallback, see header).
    //========================================================================
    reg                    dsa_start;
    wire                   dsa_busy, dsa_done;
    reg  [NOPE*16-1:0]     dsa_qidx;
    reg  [IDXW:0]          dsa_slen;
    wire                   dsa_key_req;
    wire [IDXW-1:0]        dsa_key_idx;
    reg  [NOPE*16-1:0]     dsa_kidx;
    reg                    dsa_key_valid;
    wire [TOPK*IDXW-1:0]   dsa_sel_idx;
    wire [IDXW:0]          dsa_sel_count;
    dsa_indexer #(.IDX_DIM(NOPE), .S_MAX(S_MAX), .TOPK(TOPK)) u_dsa (
        .clk(clk), .rst(rst), .start(dsa_start),
        .busy(dsa_busy), .done(dsa_done),
        .q_idx(dsa_qidx), .s_len(dsa_slen),
        .key_req(dsa_key_req), .key_idx(dsa_key_idx),
        .k_idx(dsa_kidx), .key_valid(dsa_key_valid),
        .sel_idx(dsa_sel_idx), .sel_count(dsa_sel_count)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _dsa_unused = &{1'b0, dsa_busy, dsa_key_req, dsa_key_idx};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // SUB-UNIT : glm_softmax -- PE_M replicated, lockstep (per-row attention).
    //========================================================================
    reg               sm_start;
    reg               sm_in_valid;
    reg  [16*PE_M-1:0] sm_x_in;
    wire [PE_M-1:0]   sm_busy, sm_out_valid, sm_done;
    wire [16*PE_M-1:0] sm_p_out;
    genvar gs;
    generate
    for (gs = 0; gs < PE_M; gs = gs + 1) begin : SM
        glm_softmax #(.LEN(S_MAX), .LANES(1)) u_softmax (
            .clk(clk), .rst(rst), .start(sm_start),
            .in_valid(sm_in_valid), .x_in(sm_x_in[16*gs +: 16]),
            .busy(sm_busy[gs]), .out_valid(sm_out_valid[gs]), .p_out(sm_p_out[16*gs +: 16]),
            .done(sm_done[gs])
        );
    end
    endgenerate
    /* verilator lint_off UNUSEDSIGNAL */
    wire _sm_unused = &{1'b0, sm_busy};
    /* verilator lint_on UNUSEDSIGNAL */
    localparam [15:0] NEG_BIG = 16'hFF80;   // -inf bf16 (masks unused slots)

    //========================================================================
    // CONTEXT accumulate (fp32) : O_h[d] = sum_s p[h][s] * V[h][s][d], PER ROW.
    //========================================================================
    reg [31:0] ctx_acc [0:PE_M-1];

    //========================================================================
    // MASTER FSM
    //========================================================================
    localparam [4:0]
        S_IDLE  = 5'd0,
        S_QDQ   = 5'd1,    // x*W_dq -> qlora                 (FP8, batched rows)
        S_QNORM = 5'd2,    // RMSNorm(qlora) -> qlora_n       (bf16, per row)
        S_QUQ   = 5'd3,    // qlora_n*W_uq -> qfull           (FP8, batched rows)
        S_QROPE = 5'd4,    // rope q_rope per head            (bf16, per row)
        S_KVDKV = 5'd5,    // x*W_dkv -> ckv_cur              (FP8, batched rows)
        S_KVKR  = 5'd6,    // x*W_kr -> krope_cur             (FP8, batched rows)
        S_KRROPE= 5'd7,    // rope shared k_rope              (bf16, per row)
        S_DSA   = 5'd8,    // dsa_indexer select keys
        S_KEY   = 5'd9,    // per key: norm/W_uk/W_uv (shared)/assemble/score (per row)
        S_SOFT  = 5'd10,   // per head softmax over scores    (bf16, per row)
        S_CTX   = 5'd11,   // weighted-V context              (bf16 fp32-acc, per row)
        S_OUT   = 5'd12,   // ctx*W_o -> out                  (FP8, batched rows)
        S_DONE  = 5'd13;
    reg [4:0] state;

    // ---- GEMV micro-sequencer sub-state (shared by every projection stage) ----
    localparam [1:0] GV_IDLE=2'd0, GV_START=2'd1, GV_RUN=2'd2, GV_WAIT=2'd3;
    reg [1:0]        gv_st;
    reg [GRPW-1:0]   gv_grp;       // current tile-group
    reg [GRPW-1:0]   gv_ng;        // number of tile-groups for this pass
    reg [KCW-1:0]    gv_k;         // current K beat
    reg [KW-1:0]     gv_klen;      // K length for this pass
    wire [KCW-1:0]   gv_klast = gv_klen[KCW-1:0] - 1'b1;  // last operand index
    reg [3:0]        gv_sel;       // weight select for this pass
    reg [1:0]        gv_dst;       // destination buffer code (see GVD_*)
    reg              gv_go;        // request: launch a GEMV pass
    reg              gv_done;      // a GEMV pass finished (1-cycle)
    // destination codes
    localparam [1:0] GVD_QLORA=2'd0, GVD_QFULL=2'd1, GVD_CKV=2'd2, GVD_KR=2'd3;

    // A-source selection (which buffer feeds a_col[k]).
    localparam [2:0] AS_X=3'd0, AS_QLN=3'd1, AS_CTX=3'd2, AS_Q=3'd3, AS_CKVN=3'd4;
    reg [2:0]        gv_asrc;
    reg [$clog2(H_HEADS+1)-1:0] gv_head;   // head for the score pass

    // ---- per-key loop bookkeeping (S_KEY) ----
    localparam [3:0]
        K_RDREQ=4'd0,  // request cache read for selected key s
        K_RDWAIT=4'd1, // wait kc_valid; latch c_kv[j], k_rope[j]  (SHARED)
        K_NWAIT=4'd3,  // RMSNorm(c_kv[j]) -> ckv_n                (SHARED)
        K_UK=4'd4,     // ckv_n*W_uk -> knope_j (per head)   FP8   (SHARED)
        K_UV=4'd5,     // ckv_n*W_uv -> v_j     (per head)   FP8   (SHARED)
        K_SCORE=4'd7,  // per head: q_h . K_{h,j} -> scores[row][h][s]  bf16 (per row)
        K_NEXTH=4'd8,  // advance head in score loop
        K_NEXT=4'd9;   // advance selected key s
    reg [3:0]        kst;
    reg [IDXW:0]     ksel;         // index into sel_list (0..sel_cnt-1)

    // ---- softmax loop bookkeeping (S_SOFT) ----
    localparam [2:0] SF_FEED=3'd0, SF_CAP=3'd2, SF_NEXT=3'd3;
    reg [2:0]        sfst;
    reg [$clog2(H_HEADS+1)-1:0] sf_head;
    reg [IDXW:0]     sf_feed_i;    // logits fed
    reg [IDXW:0]     sf_cap_i;     // probs captured

    // ---- context loop bookkeeping (S_CTX) ----
    localparam [2:0] CX_INIT=3'd0, CX_ACC=3'd1, CX_STORE=3'd2, CX_NEXT=3'd3;
    reg [2:0]        cxst;
    reg [$clog2(H_HEADS+1)-1:0] cx_head;
    reg [$clog2(V_DIM+1)-1:0]   cx_d;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] ctx_lin = cx_head*V_DIM + {{(32-$clog2(V_DIM+1)){1'b0}}, cx_d};
    /* verilator lint_on UNUSEDSIGNAL */
    reg [IDXW:0]     cx_s;

    //========================================================================
    // GEMV micro-sequencer (combinational operand drive + sequential control).
    //========================================================================
    // combinational: PE_M a_col elements for current K beat from the selected
    // source.  Each row presents its own activation; AS_CKVN is a SHARED cache-key
    // latent broadcast to all lanes (W_uk/W_uv use lane 0).  AS_Q (score) presents
    // each row's q_h element against the SHARED key column (score_w_lanes).
    reg [16*PE_M-1:0] gv_a_elem;
    integer           ga;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0]     q_lin = gv_head*QK_DIM + {{(32-KCW){1'b0}}, gv_k};
    /* verilator lint_on UNUSEDSIGNAL */
    always @* begin
        for (ga = 0; ga < PE_M; ga = ga + 1) begin
            case (gv_asrc)
                AS_X:    gv_a_elem[16*ga +: 16] = xbuf   [ga][gv_k[$clog2(MODEL_DIM)-1:0]];
                AS_QLN:  gv_a_elem[16*ga +: 16] = qlora_n[ga][gv_k[$clog2(Q_LORA)-1:0]];
                AS_CTX:  gv_a_elem[16*ga +: 16] = ctx    [ga][gv_k[$clog2(HV)-1:0]];
                AS_Q:    gv_a_elem[16*ga +: 16] = qrot   [ga][q_lin[$clog2(HQK)-1:0]];
                AS_CKVN: gv_a_elem[16*ga +: 16] = ckv_n     [gv_k[$clog2(KV_LORA)-1:0]];
                default: gv_a_elem[16*ga +: 16] = 16'h0;
            endcase
        end
    end

    //------------------------------------------------------------------------
    // DYNAMIC ACTIVATION SHIFT : combinational max bf16 exponent over the active
    // A-source vector PER ROW, then dyn_shift.  Stable across all tile-groups of a
    // pass; latched by the FP8 matmul at its start.  (Score pass uses the bf16
    // engine -> a_shift unused.)  AS_CKVN is shared -> identical across rows.
    //------------------------------------------------------------------------
    reg [7:0] a_emax [0:PE_M-1];
    integer   ae_i, ae_r;
    always @* begin
        for (ae_r = 0; ae_r < PE_M; ae_r = ae_r + 1) begin
            a_emax[ae_r] = 8'd0;
            case (gv_asrc)
                AS_X:    for (ae_i=0; ae_i<MODEL_DIM; ae_i=ae_i+1)
                             if (xbuf   [ae_r][ae_i][14:7] > a_emax[ae_r]) a_emax[ae_r] = xbuf   [ae_r][ae_i][14:7];
                AS_QLN:  for (ae_i=0; ae_i<Q_LORA;   ae_i=ae_i+1)
                             if (qlora_n[ae_r][ae_i][14:7] > a_emax[ae_r]) a_emax[ae_r] = qlora_n[ae_r][ae_i][14:7];
                AS_CTX:  for (ae_i=0; ae_i<HV;       ae_i=ae_i+1)
                             if (ctx    [ae_r][ae_i][14:7] > a_emax[ae_r]) a_emax[ae_r] = ctx    [ae_r][ae_i][14:7];
                AS_CKVN: for (ae_i=0; ae_i<KV_LORA;  ae_i=ae_i+1)
                             if (ckv_n        [ae_i][14:7] > a_emax[ae_r]) a_emax[ae_r] = ckv_n        [ae_i][14:7];
                default: a_emax[ae_r] = 8'd0;
            endcase
        end
    end
    // POWER: register the per-row max-tree result on the PRE-START beat and feed
    // the FP8 matmul the REGISTERED a_shift (see the original single-row note).
    reg  [7:0] a_emax_q [0:PE_M-1];
    wire       a_emax_cap = ((gv_st == GV_IDLE) && gv_go) ||
                            ((gv_st == GV_WAIT) && mm_out_valid);
    integer    aq_i;
    always @(posedge clk) begin
        if (rst)             for (aq_i=0; aq_i<PE_M; aq_i=aq_i+1) a_emax_q[aq_i] <= 8'd0;
        else if (a_emax_cap) for (aq_i=0; aq_i<PE_M; aq_i=aq_i+1) a_emax_q[aq_i] <= a_emax[aq_i];
    end
    genvar gsh;
    generate
    for (gsh = 0; gsh < PE_M; gsh = gsh + 1) begin : ASH
        assign a_shift_comb[8*gsh +: 8] = dyn_shift(a_emax_q[gsh]);
    end
    endgenerate

    // combinational: assembled K lanes for the bf16 score engine (SHARED across
    // rows).  First NOPE come from knope_j[head], the rest ROPE from krope_j.
    // Only lane 0 meaningful.
    reg [15:0]        score_k_elem;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0]       knope_lin = gv_head*NOPE + {{(32-KCW){1'b0}}, gv_k};
    wire [31:0]       krope_lin = {{(32-KCW){1'b0}}, gv_k} - NOPE;
    /* verilator lint_on UNUSEDSIGNAL */
    always @* begin
        if (gv_k < KCW'(NOPE))
            score_k_elem = knope_j[knope_lin[$clog2(HNOPE)-1:0]];
        else
            score_k_elem = krope_j[krope_lin[$clog2(ROPE)-1:0]];
        score_w_lanes        = {PE_N*16{1'b0}};
        score_w_lanes[15:0]  = score_k_elem;     // only lane 0 meaningful
    end

    //========================================================================
    // COMBINATIONAL MATMUL / WEIGHT-PULL DRIVE.  (Weight request stream is
    //   independent of PE_M -- ONE fetch shared by all rows.)
    //========================================================================
    always @* begin
        mm_klen     = gv_klen;
        mm_start    = (gv_st == GV_START);
        mm_in_valid = (gv_st == GV_RUN);
        mm_a        = gv_a_elem;
        w_req = (gv_st == GV_RUN) && ~gv_score;
        w_sel = gv_sel;
        w_grp = gv_grp;
        w_k   = gv_k;
    end

    //========================================================================
    // SEQUENTIAL CONTROL
    //========================================================================
    integer s_i, h_i, d_i;

    reg [$clog2(Q_LORA+1)-1:0]   rn_idx_q;   // qlora reduce read index  (shared lockstep)
    reg [$clog2(Q_LORA+1)-1:0]   rn_yidx_q;  // qlora_n write index
    reg [$clog2(KV_LORA+1)-1:0]  rn_idx_k;
    reg [$clog2(KV_LORA+1)-1:0]  rn_yidx_k;
    reg [$clog2((ROPE/2)+1)-1:0] rope_pair;  // rope input pair index
    reg [$clog2((ROPE/2)+1)-1:0] rope_yp;    // rope output pair index
    reg                          sm_in_valid_able; // softmax feed gate

    always @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            out        <= {MODEL_DIM*16*PE_M{1'b0}};
            kc_req     <= 1'b0; kc_idx <= {IDXW{1'b0}};
            pos_qr     <= {POSW*PE_M{1'b0}};
            s_reg      <= {(IDXW+1){1'b0}};
            rnq_start  <= 1'b0; rnk_start <= 1'b0;
            rnq_x_valid<= 1'b0; rnq_g_valid <= 1'b0;
            rnk_x_valid<= 1'b0; rnk_g_valid <= 1'b0;
            rnq_x_in   <= {16*PE_M{1'b0}}; rnq_gamma_in <= {16*PE_M{1'b0}};
            rnk_x_in   <= 16'h0; rnk_gamma_in <= 16'h0;
            rp_start   <= 1'b0; rp_pos <= {POSW*PE_M{1'b0}};
            rp_x_valid <= 1'b0; rp_x_in <= {ROPE_LANES*32*PE_M{1'b0}};
            dsa_start  <= 1'b0; dsa_qidx <= {NOPE*16{1'b0}};
            dsa_slen   <= {(IDXW+1){1'b0}};
            dsa_kidx   <= {NOPE*16{1'b0}}; dsa_key_valid <= 1'b0;
            sm_start   <= 1'b0; sm_in_valid <= 1'b0; sm_x_in <= {16*PE_M{1'b0}};
            gv_st      <= GV_IDLE; gv_grp <= {GRPW{1'b0}}; gv_ng <= {GRPW{1'b0}};
            gv_k       <= {KCW{1'b0}}; gv_klen <= {KW{1'b0}}; gv_sel <= 4'd0;
            gv_dst     <= GVD_QLORA; gv_go <= 1'b0; gv_done <= 1'b0;
            gv_asrc    <= AS_X; gv_score <= 1'b0; gv_head <= {$clog2(H_HEADS+1){1'b0}};
            kst        <= K_RDREQ; ksel <= {(IDXW+1){1'b0}};
            sfst       <= SF_FEED; sf_head <= {$clog2(H_HEADS+1){1'b0}};
            sf_feed_i  <= {(IDXW+1){1'b0}}; sf_cap_i <= {(IDXW+1){1'b0}};
            cxst       <= CX_INIT; cx_head <= {$clog2(H_HEADS+1){1'b0}};
            cx_d       <= {$clog2(V_DIM+1){1'b0}}; cx_s <= {(IDXW+1){1'b0}};
            sel_cnt    <= {(IDXW+1){1'b0}};
            for (rr=0; rr<PE_M; rr=rr+1) begin
                ctx_acc[rr] <= 32'h0;
                for (s_i=0; s_i<MODEL_DIM; s_i=s_i+1) begin xbuf[rr][s_i]<=16'h0; outbuf[rr][s_i]<=16'h0; end
                for (s_i=0; s_i<Q_LORA;   s_i=s_i+1) begin qlora[rr][s_i]<=16'h0; qlora_n[rr][s_i]<=16'h0; end
                for (s_i=0; s_i<HQK;      s_i=s_i+1) begin qfull[rr][s_i]<=16'h0; qrot[rr][s_i]<=16'h0; end
                for (s_i=0; s_i<KV_LORA;  s_i=s_i+1) ckv_cur[rr][s_i]<=16'h0;
                for (s_i=0; s_i<ROPE;     s_i=s_i+1) krope_cur[rr][s_i]<=16'h0;
                for (s_i=0; s_i<HV;       s_i=s_i+1) ctx[rr][s_i]<=16'h0;
                for (h_i=0; h_i<H_HEADS;  h_i=h_i+1)
                    for (s_i=0; s_i<S_MAX; s_i=s_i+1) begin
                        scores[rr][h_i][s_i]<=16'h0; probs[rr][h_i][s_i]<=16'h0;
                    end
            end
            for (s_i=0; s_i<KV_LORA;  s_i=s_i+1) begin ckv_key[s_i]<=16'h0; ckv_n[s_i]<=16'h0; end
            for (s_i=0; s_i<ROPE;     s_i=s_i+1) krope_j[s_i]<=16'h0;
            for (s_i=0; s_i<HNOPE;    s_i=s_i+1) knope_j[s_i]<=16'h0;
            for (s_i=0; s_i<HV;       s_i=s_i+1) v_j[s_i]<=16'h0;
            for (h_i=0; h_i<H_HEADS;  h_i=h_i+1)
                for (s_i=0; s_i<S_MAX; s_i=s_i+1)
                    for (d_i=0; d_i<V_DIM; d_i=d_i+1) vstore[h_i][s_i][d_i]<=16'h0;
            for (s_i=0; s_i<TOPK; s_i=s_i+1) sel_list[s_i]<={IDXW{1'b0}};
        end else begin
            // ---- default pulse deasserts ----
            done          <= 1'b0;
            rnq_start     <= 1'b0; rnk_start <= 1'b0;
            rp_start      <= 1'b0;
            dsa_start     <= 1'b0;
            sm_start      <= 1'b0;
            gv_done       <= 1'b0;

            //================================================================
            // GEMV MICRO-SEQUENCER (runs whenever gv_go pulses; drives the
            // active engine).  GV_RUN issues a valid beat for gv_k=0..klen-1.
            //================================================================
            case (gv_st)
                GV_IDLE: begin
                    if (gv_go) begin
                        gv_grp   <= {GRPW{1'b0}};
                        gv_k     <= {KCW{1'b0}};
                        gv_st    <= GV_START;
                    end
                end
                GV_START: begin
                    gv_k  <= {KCW{1'b0}};
                    gv_st <= GV_RUN;
                end
                GV_RUN: begin
                    if (gv_k == gv_klast) begin
                        gv_st <= GV_WAIT;        // issued all klen beats
                    end else begin
                        gv_k <= gv_k + 1'b1;
                    end
                end
                GV_WAIT: begin
                    if (mm_out_valid) begin
                        // PER-ROW capture: lane (row r, col t) at mm_c[16*(r*PE_N+t)].
                        for (rr = 0; rr < PE_M; rr = rr + 1)
                          for (tt = 0; tt < PE_N; tt = tt + 1) begin
                            case (gv_dst)
                            GVD_QLORA: if (gv_grp*PE_N+tt < Q_LORA)
                                          qlora[rr][gv_grp*PE_N+tt]   <= mm_c[16*(rr*PE_N+tt) +:16];
                            GVD_QFULL: if (gv_grp*PE_N+tt < HQK)
                                          qfull[rr][gv_grp*PE_N+tt]   <= mm_c[16*(rr*PE_N+tt) +:16];
                            GVD_CKV:   if (gv_grp*PE_N+tt < KV_LORA)
                                          ckv_cur[rr][gv_grp*PE_N+tt] <= mm_c[16*(rr*PE_N+tt) +:16];
                            GVD_KR:    if (gv_grp*PE_N+tt < ROPE)
                                          krope_cur[rr][gv_grp*PE_N+tt] <= mm_c[16*(rr*PE_N+tt) +:16];
                            endcase
                          end
                        if (gv_grp == gv_ng - 1'b1) begin
                            gv_st   <= GV_IDLE;
                            gv_done <= 1'b1;        // whole GEMV pass complete
                        end else begin
                            gv_grp   <= gv_grp + 1'b1;
                            gv_k     <= {KCW{1'b0}};
                            gv_st    <= GV_START;   // launch next tile-group
                        end
                    end
                end
                default: gv_st <= GV_IDLE;
            endcase

            //================================================================
            // MASTER STAGE FSM
            //================================================================
            case (state)
            // -------------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy  <= 1'b1;
                    // latch PER-ROW query positions.  Row 0 = scalar `pos` always.
                    //   PER_ROW_POS=0 (default): rows 1.. ALSO use `pos` (shared) --
                    //     byte-identical to the pre-per-row path; a caller that never
                    //     connects pos_vec is SAFE (no silent position-0 decode).
                    //   PER_ROW_POS=1: rows 1.. take their own pos_vec slice.
                    for (rr=0; rr<PE_M; rr=rr+1)
                        pos_qr[POSW*rr +: POSW] <=
                            ((rr==0) || (PER_ROW_POS==0)) ? pos
                                                          : pos_vec[POSW*rr +: POSW];
                    s_reg <= s_len;
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (s_i=0; s_i<MODEL_DIM; s_i=s_i+1)
                            xbuf[rr][s_i] <= x_vec[16*(MODEL_DIM*rr + s_i) +: 16];
                    // launch FP8 GEMV: x*W_dq -> qlora  (K=MODEL_DIM, OUT=Q_LORA)
                    gv_asrc  <= AS_X;
                    gv_sel   <= SEL_DQ;
                    gv_klen  <= KW'(MODEL_DIM);
                    gv_ng    <= GRPW'((Q_LORA + PE_N - 1)/PE_N);
                    gv_dst   <= GVD_QLORA;
                    gv_score <= 1'b0;
                    gv_go    <= 1'b1;
                    state    <= S_QDQ;
                end
            end
            // ------------------------------------------------------------- Q W_dq
            S_QDQ: begin
                if (gv_go) gv_go <= 1'b0;
                else if (gv_done) begin
                    rnq_start <= 1'b1;
                    state     <= S_QNORM;
                end
            end
            // ------------------------------------------------------------- Q norm
            S_QNORM: begin
                rnq_x_valid <= 1'b0; rnq_g_valid <= 1'b0;
                if (rnq_in_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        rnq_x_in[16*rr +: 16] <= qlora[rr][rn_idx_q[$clog2(Q_LORA)-1:0]];
                    rnq_x_valid <= 1'b1;
                end
                if (rnq_g_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        rnq_gamma_in[16*rr +: 16] <= 16'h3F80;   // bf16 1.0
                    rnq_g_valid  <= 1'b1;
                end
                if (rnq_y_valid[0])
                    for (rr=0; rr<PE_M; rr=rr+1)
                        qlora_n[rr][rn_yidx_q[$clog2(Q_LORA)-1:0]] <= rnq_y_out[16*rr +: 16];
                if (rnq_done[0]) begin
                    // FP8 GEMV: qlora_n*W_uq -> qfull  (K=Q_LORA, OUT=HQK)
                    gv_asrc  <= AS_QLN;
                    gv_sel   <= SEL_UQ;
                    gv_klen  <= KW'(Q_LORA);
                    gv_ng    <= GRPW'((HQK + PE_N - 1)/PE_N);
                    gv_dst   <= GVD_QFULL;
                    gv_score <= 1'b0;
                    gv_go    <= 1'b1;
                    state    <= S_QUQ;
                end
            end
            // ------------------------------------------------------------- Q W_uq
            S_QUQ: begin
                if (gv_go) gv_go <= 1'b0;
                else if (gv_done) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (h_i=0; h_i<H_HEADS; h_i=h_i+1)
                            for (d_i=0; d_i<NOPE; d_i=d_i+1)
                                qrot[rr][h_i*QK_DIM + d_i] <= qfull[rr][h_i*QK_DIM + d_i];
                    for (rr=0; rr<PE_M; rr=rr+1)            // per-row q RoPE position
                        rp_pos[POSW*rr +: POSW] <= pos_qr[POSW*rr +: POSW];
                    rp_start <= 1'b1;
                    gv_head  <= {$clog2(H_HEADS+1){1'b0}};
                    state    <= S_QROPE;
                end
            end
            // ------------------------------------------------------------- Q rope
            S_QROPE: begin
                rp_x_valid <= 1'b0;
                if (rp_in_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1) begin
                        rp_x_in[32*rr +: 16]      <= qfull[rr][gv_head*QK_DIM + NOPE + 2*rope_pair];
                        rp_x_in[32*rr + 16 +: 16] <= qfull[rr][gv_head*QK_DIM + NOPE + 2*rope_pair+1];
                    end
                    rp_x_valid     <= 1'b1;
                end
                if (rp_y_valid[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1) begin
                        qrot[rr][gv_head*QK_DIM + NOPE + 2*rope_yp]   <= rp_y_out[32*rr +: 16];
                        qrot[rr][gv_head*QK_DIM + NOPE + 2*rope_yp+1] <= rp_y_out[32*rr + 16 +: 16];
                    end
                end
                if (rp_done[0]) begin
                    if (gv_head == H_HEADS[$clog2(H_HEADS+1)-1:0] - 1'b1) begin
                        // FP8 GEMV: x*W_dkv -> ckv_cur
                        gv_asrc  <= AS_X;
                        gv_sel   <= SEL_DKV;
                        gv_klen  <= KW'(MODEL_DIM);
                        gv_ng    <= GRPW'((KV_LORA + PE_N - 1)/PE_N);
                        gv_dst   <= GVD_CKV;
                        gv_score <= 1'b0;
                        gv_go    <= 1'b1;
                        state    <= S_KVDKV;
                    end else begin
                        gv_head  <= gv_head + 1'b1;
                        for (rr=0; rr<PE_M; rr=rr+1)        // per-row q RoPE position
                            rp_pos[POSW*rr +: POSW] <= pos_qr[POSW*rr +: POSW];
                        rp_start <= 1'b1;       // rope next head
                    end
                end
            end
            // ------------------------------------------------------------- W_dkv
            S_KVDKV: begin
                if (gv_go) gv_go <= 1'b0;
                else if (gv_done) begin
                    // FP8 GEMV: x*W_kr -> krope_cur  (K=MODEL_DIM, OUT=ROPE)
                    gv_asrc  <= AS_X;
                    gv_sel   <= SEL_KR;
                    gv_klen  <= KW'(MODEL_DIM);
                    gv_ng    <= GRPW'((ROPE + PE_N - 1)/PE_N);
                    gv_dst   <= GVD_KR;
                    gv_score <= 1'b0;
                    gv_go    <= 1'b1;
                    state    <= S_KVKR;
                end
            end
            // ------------------------------------------------------------- W_kr
            S_KVKR: begin
                if (gv_go) gv_go <= 1'b0;
                else if (gv_done) begin
                    for (rr=0; rr<PE_M; rr=rr+1)            // per-row current-token k RoPE position
                        rp_pos[POSW*rr +: POSW] <= pos_qr[POSW*rr +: POSW];
                    rp_start <= 1'b1;          // rope the per-row k_rope
                    state    <= S_KRROPE;
                end
            end
            // ------------------------------------------------------------- k_rope
            S_KRROPE: begin
                rp_x_valid <= 1'b0;
                if (rp_in_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1) begin
                        rp_x_in[32*rr +: 16]      <= krope_cur[rr][2*rope_pair];
                        rp_x_in[32*rr + 16 +: 16] <= krope_cur[rr][2*rope_pair+1];
                    end
                    rp_x_valid     <= 1'b1;
                end
                if (rp_y_valid[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1) begin
                        krope_cur[rr][2*rope_yp]   <= rp_y_out[32*rr +: 16];
                        krope_cur[rr][2*rope_yp+1] <= rp_y_out[32*rr + 16 +: 16];
                    end
                end
                if (rp_done[0]) begin
                    // DSA selection driven from row 0's q (head0 nope) -- shared.
                    for (d_i=0; d_i<NOPE; d_i=d_i+1)
                        dsa_qidx[16*d_i +: 16] <= qrot[0][d_i];
                    dsa_slen  <= s_reg;
                    dsa_start <= 1'b1;
                    state     <= S_DSA;
                end
            end
            // ------------------------------------------------------------- DSA
            S_DSA: begin
                dsa_key_valid <= 1'b0;
                if (dsa_key_req) begin
                    dsa_kidx      <= {NOPE*16{1'b0}};
                    dsa_key_valid <= 1'b1;
                end
                if (dsa_done) begin
                    for (s_i=0; s_i<TOPK; s_i=s_i+1)
                        sel_list[s_i] <= dsa_sel_idx[IDXW*s_i +: IDXW];
                    sel_cnt <= dsa_sel_count;
                    ksel    <= {(IDXW+1){1'b0}};
                    kst     <= K_RDREQ;
                    state   <= S_KEY;
                end
            end
            // ------------------------------------------------------------- per-key
            S_KEY: begin
                case (kst)
                    K_RDREQ: begin
                        kc_idx  <= sel_list[ksel[IDXW-1:0]];
                        kc_req  <= 1'b1;
                        kst     <= K_RDWAIT;
                    end
                    K_RDWAIT: begin
                        if (kc_valid) begin
                            kc_req <= 1'b0;
                            // cache key latent + rope are SHARED across rows.
                            for (d_i=0; d_i<KV_LORA; d_i=d_i+1)
                                ckv_key[d_i] <= kc_ckv[16*d_i +: 16];
                            for (d_i=0; d_i<ROPE; d_i=d_i+1)
                                krope_j[d_i] <= kc_krope[16*d_i +: 16];
                            rnk_start <= 1'b1;
                            kst       <= K_NWAIT;
                        end
                    end
                    K_NWAIT: begin
                        rnk_x_valid <= 1'b0; rnk_g_valid <= 1'b0;
                        if (rnk_in_req) begin
                            rnk_x_in    <= ckv_key[rn_idx_k[$clog2(KV_LORA)-1:0]];
                            rnk_x_valid <= 1'b1;
                        end
                        if (rnk_g_req) begin
                            rnk_gamma_in <= 16'h3F80;   // 1.0
                            rnk_g_valid  <= 1'b1;
                        end
                        if (rnk_y_valid) ckv_n[rn_yidx_k[$clog2(KV_LORA)-1:0]] <= rnk_y_out;
                        if (rnk_done) begin
                            // FP8 GEMV: ckv_n*W_uk -> knope_j (K=KV_LORA, OUT=HNOPE) SHARED
                            gv_asrc  <= AS_CKVN;
                            gv_sel   <= SEL_UK;
                            gv_klen  <= KW'(KV_LORA);
                            gv_ng    <= GRPW'((HNOPE + PE_N - 1)/PE_N);
                            gv_dst   <= GVD_QFULL;
                            gv_score <= 1'b0;
                            gv_go    <= 1'b1;
                            kst      <= K_UK;
                        end
                    end
                    K_UK: begin
                        if (gv_go) gv_go <= 1'b0;
                        // shared result: capture lane 0 (row 0) only.
                        if (mm_out_valid && gv_st==GV_WAIT) begin
                            for (tt=0; tt<PE_N; tt=tt+1)
                                if (gv_grp*PE_N+tt < HNOPE)
                                    knope_j[gv_grp*PE_N+tt] <= mm_c[16*tt +:16];
                        end
                        if (gv_done) begin
                            // FP8 GEMV: ckv_n*W_uv -> v_j (K=KV_LORA, OUT=HV) SHARED
                            gv_asrc  <= AS_CKVN;
                            gv_sel   <= SEL_UV;
                            gv_klen  <= KW'(KV_LORA);
                            gv_ng    <= GRPW'((HV + PE_N - 1)/PE_N);
                            gv_score <= 1'b0;
                            gv_go    <= 1'b1;
                            kst      <= K_UV;
                        end
                    end
                    K_UV: begin
                        if (gv_go) gv_go <= 1'b0;
                        if (mm_out_valid && gv_st==GV_WAIT) begin
                            for (tt=0; tt<PE_N; tt=tt+1)
                                if (gv_grp*PE_N+tt < HV)
                                    v_j[gv_grp*PE_N+tt] <= mm_c[16*tt +:16];
                        end
                        if (gv_done) begin
                            for (h_i=0; h_i<H_HEADS; h_i=h_i+1)
                                for (d_i=0; d_i<V_DIM; d_i=d_i+1)
                                    vstore[h_i][ksel[IDXW-1:0]][d_i]
                                        <= v_j[h_i*V_DIM + d_i];
                            gv_head <= {$clog2(H_HEADS+1){1'b0}};
                            kst     <= K_SCORE;
                        end
                    end
                    // bf16 SCORE pass: q_h . K_{h,j} (GEMV over QK_DIM, OUT=1), PER ROW.
                    K_SCORE: begin
                        gv_asrc  <= AS_Q;
                        gv_klen  <= KW'(QK_DIM);
                        gv_ng    <= GRPW'(1);
                        gv_score <= 1'b1;        // -> bf16 score engine
                        gv_go    <= 1'b1;
                        kst      <= K_NEXTH;
                    end
                    K_NEXTH: begin
                        if (gv_go) gv_go <= 1'b0;
                        if (mm_out_valid && gv_st==GV_WAIT)
                            for (rr=0; rr<PE_M; rr=rr+1)
                                scores[rr][gv_head[$clog2(H_HEADS)-1:0]][ksel[IDXW-1:0]]
                                    <= mm_c[16*(rr*PE_N) +:16];
                        if (gv_done) begin
                            if (gv_head == H_HEADS[$clog2(H_HEADS+1)-1:0]-1'b1) begin
                                gv_score <= 1'b0;
                                kst      <= K_NEXT;
                            end else begin
                                gv_head  <= gv_head + 1'b1;
                                kst      <= K_SCORE;
                            end
                        end
                    end
                    K_NEXT: begin
                        if (ksel == sel_cnt - 1'b1) begin
                            sf_head   <= {$clog2(H_HEADS+1){1'b0}};
                            sfst      <= SF_FEED;
                            sf_feed_i <= {(IDXW+1){1'b0}};
                            sf_cap_i  <= {(IDXW+1){1'b0}};
                            sm_start  <= 1'b1;
                            state     <= S_SOFT;
                        end else begin
                            ksel <= ksel + 1'b1;
                            kst  <= K_RDREQ;
                        end
                    end
                    default: kst <= K_RDREQ;
                endcase
            end
            // ------------------------------------------------------------- softmax
            S_SOFT: begin
                case (sfst)
                    SF_FEED: begin
                        sm_in_valid <= 1'b0;
                        if (sm_in_valid_able) begin
                            sm_in_valid <= 1'b1;
                            for (rr=0; rr<PE_M; rr=rr+1)
                                sm_x_in[16*rr +: 16] <= (sf_feed_i < sel_cnt) ?
                                       scores[rr][sf_head[$clog2(H_HEADS)-1:0]][sf_feed_i[IDXW-1:0]]
                                     : NEG_BIG;
                            sf_feed_i   <= sf_feed_i + 1'b1;
                            if (sf_feed_i == S_MAX[IDXW:0]-1'b1)
                                sfst <= SF_CAP;
                        end
                    end
                    SF_CAP: begin
                        if (sm_out_valid[0]) begin
                            for (rr=0; rr<PE_M; rr=rr+1)
                                probs[rr][sf_head[$clog2(H_HEADS)-1:0]][sf_cap_i[IDXW-1:0]] <=
                                    (sf_cap_i < sel_cnt) ? sm_p_out[16*rr +: 16] : 16'h0;
                            sf_cap_i <= sf_cap_i + 1'b1;
                        end
                        if (sm_done[0]) sfst <= SF_NEXT;
                    end
                    SF_NEXT: begin
                        if (sf_head == H_HEADS[$clog2(H_HEADS+1)-1:0]-1'b1) begin
                            cx_head <= {$clog2(H_HEADS+1){1'b0}};
                            cx_d    <= {$clog2(V_DIM+1){1'b0}};
                            cxst    <= CX_INIT;
                            state   <= S_CTX;
                        end else begin
                            sf_head   <= sf_head + 1'b1;
                            sf_feed_i <= {(IDXW+1){1'b0}};
                            sf_cap_i  <= {(IDXW+1){1'b0}};
                            sm_start  <= 1'b1;
                            sfst      <= SF_FEED;
                        end
                    end
                    default: sfst <= SF_FEED;
                endcase
            end
            // ------------------------------------------------------------- context
            S_CTX: begin
                case (cxst)
                    CX_INIT: begin
                        for (rr=0; rr<PE_M; rr=rr+1) ctx_acc[rr] <= 32'h0;
                        cx_s    <= {(IDXW+1){1'b0}};
                        cxst    <= CX_ACC;
                    end
                    CX_ACC: begin
                        for (rr=0; rr<PE_M; rr=rr+1)
                            ctx_acc[rr] <= fp32_add(ctx_acc[rr],
                                         fp32_mul(
                                           bf16_to_fp32(probs[rr][cx_head[$clog2(H_HEADS)-1:0]][cx_s[IDXW-1:0]]),
                                           bf16_to_fp32(vstore[cx_head[$clog2(H_HEADS)-1:0]][cx_s[IDXW-1:0]][cx_d[$clog2(V_DIM)-1:0]])));
                        if (cx_s == sel_cnt - 1'b1) cxst <= CX_STORE;
                        else cx_s <= cx_s + 1'b1;
                    end
                    CX_STORE: begin
                        for (rr=0; rr<PE_M; rr=rr+1)
                            ctx[rr][ctx_lin[$clog2(HV)-1:0]] <= fp32_to_bf16(ctx_acc[rr]);
                        cxst <= CX_NEXT;
                    end
                    CX_NEXT: begin
                        if (cx_d == V_DIM[$clog2(V_DIM+1)-1:0]-1'b1) begin
                            if (cx_head==H_HEADS[$clog2(H_HEADS+1)-1:0]-1'b1) begin
                                // FP8 GEMV: ctx*W_o -> out  (K=HV, OUT=MODEL_DIM)
                                gv_asrc  <= AS_CTX;
                                gv_sel   <= SEL_O;
                                gv_klen  <= KW'(HV);
                                gv_ng    <= GRPW'((MODEL_DIM + PE_N - 1)/PE_N);
                                gv_score <= 1'b0;
                                gv_go    <= 1'b1;
                                state    <= S_OUT;
                            end else begin
                                cx_head <= cx_head + 1'b1;
                                cx_d    <= {$clog2(V_DIM+1){1'b0}};
                                cxst    <= CX_INIT;
                            end
                        end else begin
                            cx_d <= cx_d + 1'b1;
                            cxst <= CX_INIT;
                        end
                    end
                    default: cxst <= CX_INIT;
                endcase
            end
            // ------------------------------------------------------------- W_o
            S_OUT: begin
                if (gv_go) gv_go <= 1'b0;
                if (mm_out_valid && gv_st==GV_WAIT) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (tt=0; tt<PE_N; tt=tt+1)
                            if (gv_grp*PE_N+tt < MODEL_DIM)
                                outbuf[rr][gv_grp*PE_N+tt] <= mm_c[16*(rr*PE_N+tt) +:16];
                end
                if (gv_done) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (s_i=0; s_i<MODEL_DIM; s_i=s_i+1)
                            out[16*(MODEL_DIM*rr + s_i) +: 16] <= outbuf[rr][s_i];
                    state <= S_DONE;
                end
            end
            // -------------------------------------------------------------
            S_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end

    //========================================================================
    // small index helpers for the rmsnorm/rope pull answers.  All sub-units run
    // in lockstep, so instance-0's handshake drives the shared counters.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            rn_idx_q <= 0; rn_yidx_q <= 0; rn_idx_k <= 0; rn_yidx_k <= 0;
            rope_pair <= 0; rope_yp <= 0; sm_in_valid_able <= 1'b0;
        end else begin
            if (rnq_start) begin rn_idx_q <= 0; rn_yidx_q <= 0; end
            else begin
                if (rnq_in_req[0]) rn_idx_q <= rn_idx_q + 1'b1;
                if (rnq_y_valid[0]) rn_yidx_q <= rn_yidx_q + 1'b1;
            end
            if (rnk_start) begin rn_idx_k <= 0; rn_yidx_k <= 0; end
            else begin
                if (rnk_in_req) rn_idx_k <= rn_idx_k + 1'b1;
                if (rnk_y_valid) rn_yidx_k <= rn_yidx_k + 1'b1;
            end
            if (rp_start) begin rope_pair <= 0; rope_yp <= 0; end
            else begin
                if (rp_in_req[0]) rope_pair <= rope_pair + 1'b1;
                if (rp_y_valid[0]) rope_yp <= rope_yp + 1'b1;
            end
            if (sm_start) sm_in_valid_able <= 1'b1;
            else if (state==S_SOFT && sfst==SF_FEED &&
                     sf_feed_i==S_MAX[IDXW:0]-1'b1 && sm_in_valid_able)
                     sm_in_valid_able <= 1'b0;
        end
    end

endmodule
