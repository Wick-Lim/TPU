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
//        -> glm_matmul_fp8  (official GLM-5.2-FP8 numerics:
//             * weights E4M3 (8-bit) + per-[128,128]-block bf16 dequant scales
//               (DeepSeek-V3 / GLM-5.2-FP8 weight_block_size=[128,128]),
//             * activations bf16, DYNAMICALLY quantized to E4M3 with a per-vector
//               (per-token) power-of-two scale a_shift,
//             * 4x4-mantissa fp8 products, fp32-accumulated per K-block, block-
//               scaled, de-scaled by 2^-a_shift, rounded to bf16),
//        exactly like swiglu_expert_fp8 wires glm_matmul_fp8.
//
//   EVERYTHING ELSE STAYS bf16, UNCHANGED FROM mla_attn.v:
//     * RMSNorm (q low-rank norm, c_kv norm)         -- rmsnorm_unit
//     * decoupled RoPE (q_rope per head, shared k_rope) -- rope_interleave_unit
//     * the per-head q.K SCORE matmul   (ACTIVATION x ACTIVATION -> bf16 via the
//                                        bf16 glm_matmul_pipe score engine)
//     * the weighted-V (P.V) CONTEXT accumulate (ACTIVATION x ACTIVATION -> bf16,
//                                        the fp32-accumulate-to-bf16 path)
//     * glm_softmax, dsa_indexer (+topk_select), the c_kv / k_rope caches.
//   So ONLY the seven weight-matrix GEMMs are FP8; the attention math is bf16.
//
//   FLOW / FSM STAGES / CACHE interface are byte-for-byte the same intent as
//   mla_attn.v -- read that file's header for the full per-stage description.
//
//----------------------------------------------------------------------------
// WEIGHT PULL INTERFACE (now FP8 + block scales, was bf16 weights)
//   w_req            : high on a beat that needs a weight column (NORMAL passes).
//   w_sel  [3:0]     : 0=W_dq 1=W_uq 2=W_dkv 3=W_kr 4=W_uk 5=W_uv 6=W_o.
//   w_grp  [GRPW]    : output tile-group g (outputs g*PE_N..+PE_N-1).
//   w_k    [KCW]     : reduction index k of this beat.
//   w_col  [PE_N*8]  : the PE_N **FP8 E4M3** weight lanes  W_sel[g*PE_N+t , w_k],
//                      presented COMBINATIONALLY the same cycle (was bf16).
//   w_scale[16*PE_N*NB] : bf16 BLOCK dequant scale per (output col pj, K-block bj)
//                      for the addressed (w_sel, w_grp) tile-group, packed
//                      w_scale[16*(bj*PE_N+pj)+:16].  Presented COMBINATIONALLY
//                      from w_sel/w_grp; latched by glm_matmul_fp8 at its start
//                      (NB = ceil(KMAX/BLK) K-blocks; default slice NB=1).
//   The q.K SCORE pass reads K from the reconstructed-key buffers (internal,
//   bf16) -- NOT a streamed weight -- so it does NOT assert w_req (unchanged).
//
//   The CACHE-READ interface (kc_*) and all other ports are IDENTICAL to mla_attn.
//
//   Activation a_shift is derived ON-CHIP (per pass, from the A-source vector's
//   max bf16 exponent) -- the caller supplies NO a_shift, exactly mirroring
//   swiglu_expert_fp8's dynamic per-vector activation quant.
//----------------------------------------------------------------------------
// STYLE: sync active-high reset; NO latch; NO comb loop (all FP feedback rides
//   the matmul / rmsnorm / rope / softmax pipeline registers); deterministic,
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
    input  wire [POSW-1:0]             pos,        // token position (for RoPE)
    input  wire [IDXW:0]               s_len,      // S causal keys (<= S_MAX)

    // ---- x input (latched at start) ----
    input  wire [MODEL_DIM*16-1:0]     x_vec,      // MODEL_DIM bf16

    // ---- weight pull (combinational responder; FP8 codes + block scales) ----
    output reg                         w_req,
    output reg  [3:0]                  w_sel,      // 0..6 projection select
    output reg  [GRPW-1:0]             w_grp,      // output tile-group index
    output reg  [KCW-1:0]              w_k,        // reduction index k of this beat
    input  wire [PE_N*8-1:0]           w_col,      // PE_N FP8 E4M3 weight lanes
    input  wire [16*PE_N*NB-1:0]       w_scale,    // bf16 block scales (sel,grp)

    // ---- cache read (past-key latents from caller's KV cache) ----
    output reg                         kc_req,
    output reg  [IDXW-1:0]             kc_idx,     // requested causal key j
    input  wire [KV_LORA*16-1:0]       kc_ckv,     // cached c_kv[j]   (bf16)
    input  wire [ROPE*16-1:0]          kc_krope,   // cached k_rope[j] (bf16, roped)
    input  wire                        kc_valid,

    // ---- output ----
    output reg  [MODEL_DIM*16-1:0]     out         // MODEL_DIM bf16
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

    //========================================================================
    // DYNAMIC per-vector pow2 ACTIVATION SCALE (a_shift) -- pure exp maxes.
    //   a_shift = clamp(134 - emax) so the max element's scaled exp becomes 7
    //   (value in [128,256), well under the 448 E4M3 max -> no saturation).
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
    //========================================================================
    reg [15:0] xbuf      [0:MODEL_DIM-1];   // latched x
    reg [POSW-1:0] pos_q;
    reg [IDXW:0]   s_reg;                    // S causal keys

    reg [15:0] qlora     [0:Q_LORA-1];       // x*W_dq
    reg [15:0] qlora_n   [0:Q_LORA-1];       // RMSNorm(qlora)
    reg [15:0] qfull     [0:HQK-1];          // qlora_n*W_uq  (per head QK_DIM)
    reg [15:0] qrot      [0:HQK-1];

    reg [15:0] ckv_cur   [0:KV_LORA-1];      // x*W_dkv  (current token latent)
    reg [15:0] krope_cur [0:ROPE-1];         // x*W_kr -> roped (shared)

    reg [15:0] ckv_n     [0:KV_LORA-1];      // RMSNorm(c_kv[j])
    reg [15:0] knope_j   [0:HNOPE-1];        // ckv_n*W_uk  (per head NOPE)
    reg [15:0] v_j       [0:HV-1];           // ckv_n*W_uv  (per head V_DIM)

    reg [15:0] scores    [0:H_HEADS-1][0:S_MAX-1];
    reg [15:0] vstore    [0:H_HEADS-1][0:S_MAX-1][0:V_DIM-1];
    reg [15:0] probs     [0:H_HEADS-1][0:S_MAX-1];   // softmax weights
    reg [15:0] ctx       [0:HV-1];                    // O concat (H*V_DIM)
    reg [15:0] outbuf    [0:MODEL_DIM-1];

    // DSA selection results
    reg [IDXW-1:0] sel_list [0:TOPK-1];
    reg [IDXW:0]   sel_cnt;

    //========================================================================
    // SHARED GEMV ENGINES.  The micro-sequencer drives mm_* operands; the
    // SEVEN WEIGHT projections go to the FP8 engine (glm_matmul_fp8), the q.K
    // SCORE pass (ACTIVATION x ACTIVATION) goes to the bf16 engine
    // (glm_matmul_pipe).  Both PE_M=1 (single query row), PE_N tile width.
    // out_valid / c_out are muxed on gv_score (stable across a pass).
    //========================================================================
    reg              mm_start;
    reg  [KW-1:0]    mm_klen;
    reg              mm_in_valid;
    reg  [15:0]      mm_a;                    // the single A element (PE_M=1)
    // declared here (ahead of the engine instances that gate on it): this GEMV
    // pass is a q.K SCORE (-> bf16 engine) rather than a weight pass (-> FP8).
    reg              gv_score;

    // ---- FP8 weight-projection engine ----
    wire             fp8_busy, fp8_ov;
    wire [PE_N*16-1:0] fp8_c;
    // ---- bf16 score (activation x activation) engine ----
    wire             bf16_busy, bf16_ov;
    wire [PE_N*16-1:0] bf16_c;
    reg  [PE_N*16-1:0] score_w_lanes;        // assembled K lanes (lane0 meaningful)

    // dynamic per-pass activation pow2 scale (combinational over the A-source).
    wire signed [7:0] a_shift_comb;

    glm_matmul_fp8 #(.PE_M(1), .PE_N(PE_N), .KMAX(KMAX), .BLK(BLK)) u_mm_fp8 (
        .clk(clk), .rst(rst),
        .start(mm_start & ~gv_score), .k_len(mm_klen),
        .in_valid(mm_in_valid & ~gv_score), .a_col(mm_a), .w_row(w_col),
        .a_shift(a_shift_comb), .w_scale(w_scale),
        .busy(fp8_busy), .out_valid(fp8_ov), .c_out(fp8_c)
    );

    glm_matmul_pipe #(.PE_M(1), .PE_N(PE_N), .KMAX(KMAX)) u_mm_bf16 (
        .clk(clk), .rst(rst),
        .start(mm_start & gv_score), .k_len(mm_klen),
        .in_valid(mm_in_valid & gv_score), .a_col(mm_a), .w_row(score_w_lanes),
        .busy(bf16_busy), .out_valid(bf16_ov), .c_out(bf16_c)
    );

    // muxed matmul result (gv_score is stable for the whole pass).
    wire               mm_out_valid = gv_score ? bf16_ov : fp8_ov;
    wire [PE_N*16-1:0] mm_c         = gv_score ? bf16_c  : fp8_c;

    //========================================================================
    // SUB-UNIT : rmsnorm_unit (reused for q_lora and c_kv).  LANES=1 (slice).
    //========================================================================
    reg              rnq_start, rnk_start;
    wire             rnq_in_req, rnq_g_req, rnq_y_valid, rnq_busy, rnq_done;
    wire [15:0]      rnq_y_out;
    wire             rnk_in_req, rnk_g_req, rnk_y_valid, rnk_busy, rnk_done;
    wire [15:0]      rnk_y_out;
    reg  [15:0]      rnq_x_in, rnq_gamma_in, rnk_x_in, rnk_gamma_in;
    reg              rnq_x_valid, rnq_g_valid, rnk_x_valid, rnk_g_valid;

    rmsnorm_unit #(.LEN(Q_LORA), .LANES(1)) u_rn_q (
        .clk(clk), .rst(rst), .start(rnq_start),
        .in_req(rnq_in_req), .x_in(rnq_x_in), .x_valid(rnq_x_valid),
        .g_req(rnq_g_req), .gamma_in(rnq_gamma_in), .g_valid(rnq_g_valid),
        .y_valid(rnq_y_valid), .y_out(rnq_y_out), .busy(rnq_busy), .done(rnq_done)
    );
    rmsnorm_unit #(.LEN(KV_LORA), .LANES(1)) u_rn_k (
        .clk(clk), .rst(rst), .start(rnk_start),
        .in_req(rnk_in_req), .x_in(rnk_x_in), .x_valid(rnk_x_valid),
        .g_req(rnk_g_req), .gamma_in(rnk_gamma_in), .g_valid(rnk_g_valid),
        .y_valid(rnk_y_valid), .y_out(rnk_y_out), .busy(rnk_busy), .done(rnk_done)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _busy_unused = &{1'b0, rnq_busy, rnk_busy, fp8_busy, bf16_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // SUB-UNIT : rope_interleave_unit (q_rope per head + shared k_rope).
    //========================================================================
    reg              rp_start;
    reg  [POSW-1:0]  rp_pos;
    wire             rp_in_req;
    reg  [ROPE_LANES*32-1:0] rp_x_in;
    reg              rp_x_valid;
    wire             rp_y_valid;
    wire [ROPE_LANES*32-1:0] rp_y_out;
    wire             rp_busy, rp_done;
    rope_interleave_unit #(.ROT_DIM(ROPE), .THETA(THETA),
                           .LANES(ROPE_LANES), .POSW(POSW)) u_rope (
        .clk(clk), .rst(rst), .start(rp_start), .pos(rp_pos),
        .in_req(rp_in_req), .x_in(rp_x_in), .x_valid(rp_x_valid),
        .y_valid(rp_y_valid), .y_out(rp_y_out), .busy(rp_busy), .done(rp_done)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _rp_busy_unused = &{1'b0, rp_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // SUB-UNIT : dsa_indexer (top-K key selection; dense fallback when S<=TOPK).
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
    // SUB-UNIT : glm_softmax (per-head attention weights over selected keys).
    //========================================================================
    reg              sm_start;
    reg              sm_in_valid;
    reg  [15:0]      sm_x_in;
    wire             sm_busy, sm_out_valid, sm_done;
    wire [15:0]      sm_p_out;
    glm_softmax #(.LEN(S_MAX), .LANES(1)) u_softmax (
        .clk(clk), .rst(rst), .start(sm_start),
        .in_valid(sm_in_valid), .x_in(sm_x_in),
        .busy(sm_busy), .out_valid(sm_out_valid), .p_out(sm_p_out),
        .done(sm_done)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _sm_unused = &{1'b0, sm_busy};
    /* verilator lint_on UNUSEDSIGNAL */
    localparam [15:0] NEG_BIG = 16'hFF80;   // -inf bf16 (masks unused slots)

    //========================================================================
    // CONTEXT accumulate (fp32) : O_h[d] = sum_s p[h][s] * V[h][s][d].
    //========================================================================
    reg [31:0] ctx_acc;

    //========================================================================
    // MASTER FSM
    //========================================================================
    localparam [4:0]
        S_IDLE  = 5'd0,
        S_QDQ   = 5'd1,    // x*W_dq -> qlora                 (FP8)
        S_QNORM = 5'd2,    // RMSNorm(qlora) -> qlora_n       (bf16)
        S_QUQ   = 5'd3,    // qlora_n*W_uq -> qfull           (FP8)
        S_QROPE = 5'd4,    // rope q_rope per head            (bf16)
        S_KVDKV = 5'd5,    // x*W_dkv -> ckv_cur              (FP8)
        S_KVKR  = 5'd6,    // x*W_kr -> krope_cur             (FP8)
        S_KRROPE= 5'd7,    // rope shared k_rope              (bf16)
        S_DSA   = 5'd8,    // dsa_indexer select keys
        S_KEY   = 5'd9,    // per key: norm/W_uk/W_uv (FP8)/assemble/score (bf16)
        S_SOFT  = 5'd10,   // per head softmax over scores    (bf16)
        S_CTX   = 5'd11,   // weighted-V context              (bf16 fp32-acc)
        S_OUT   = 5'd12,   // ctx*W_o -> out                  (FP8)
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
    // score pass: stream q_h (A) and K_{h,j} (W) from internal buffers (bf16 eng).
    // (gv_score is declared up by the engine instances it gates.)
    reg [$clog2(H_HEADS+1)-1:0] gv_head;   // head for the score pass

    // ---- per-key loop bookkeeping (S_KEY) ----
    localparam [3:0]
        K_RDREQ=4'd0,  // request cache read for selected key s
        K_RDWAIT=4'd1, // wait kc_valid; latch c_kv[j], k_rope[j]
        K_NWAIT=4'd3,  // RMSNorm(c_kv[j]) -> ckv_n
        K_UK=4'd4,     // ckv_n*W_uk -> knope_j (per head)   FP8
        K_UV=4'd5,     // ckv_n*W_uv -> v_j     (per head)   FP8
        K_SCORE=4'd7,  // for each head: q_h . K_{h,j} -> scores[h][s]  bf16
        K_NEXTH=4'd8,  // advance head in score loop
        K_NEXT=4'd9;   // advance selected key s
    reg [3:0]        kst;
    reg [IDXW:0]     ksel;         // index into sel_list (0..sel_cnt-1)
    reg [15:0]       krope_j [0:ROPE-1];   // latched k_rope[j]

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
    // combinational: a_col element for current K beat from the selected source.
    reg [15:0]      gv_a_elem;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0]     q_lin = gv_head*QK_DIM + {{(32-KCW){1'b0}}, gv_k};
    /* verilator lint_on UNUSEDSIGNAL */
    always @* begin
        case (gv_asrc)
            AS_X:    gv_a_elem = xbuf   [gv_k[$clog2(MODEL_DIM)-1:0]];
            AS_QLN:  gv_a_elem = qlora_n[gv_k[$clog2(Q_LORA)-1:0]];
            AS_CTX:  gv_a_elem = ctx    [gv_k[$clog2(HV)-1:0]];
            AS_Q:    gv_a_elem = qrot   [q_lin[$clog2(HQK)-1:0]];   // q_h element
            AS_CKVN: gv_a_elem = ckv_n  [gv_k[$clog2(KV_LORA)-1:0]];
            default: gv_a_elem = 16'h0;
        endcase
    end

    //------------------------------------------------------------------------
    // DYNAMIC ACTIVATION SHIFT : combinational max bf16 exponent over the active
    // A-source vector (the SAME vector that feeds the whole FP8 pass), then
    // dyn_shift.  Stable across all tile-groups of a pass; latched by the FP8
    // matmul at its start.  (Score pass uses the bf16 engine -> a_shift unused.)
    //------------------------------------------------------------------------
    reg [7:0] a_emax;
    integer   ae_i;
    always @* begin
        a_emax = 8'd0;
        case (gv_asrc)
            AS_X:    for (ae_i=0; ae_i<MODEL_DIM; ae_i=ae_i+1)
                         if (xbuf   [ae_i][14:7] > a_emax) a_emax = xbuf   [ae_i][14:7];
            AS_QLN:  for (ae_i=0; ae_i<Q_LORA;   ae_i=ae_i+1)
                         if (qlora_n[ae_i][14:7] > a_emax) a_emax = qlora_n[ae_i][14:7];
            AS_CTX:  for (ae_i=0; ae_i<HV;       ae_i=ae_i+1)
                         if (ctx    [ae_i][14:7] > a_emax) a_emax = ctx    [ae_i][14:7];
            AS_CKVN: for (ae_i=0; ae_i<KV_LORA;  ae_i=ae_i+1)
                         if (ckv_n  [ae_i][14:7] > a_emax) a_emax = ckv_n  [ae_i][14:7];
            default: a_emax = 8'd0;
        endcase
    end
    assign a_shift_comb = dyn_shift(a_emax);

    // combinational: assembled K lanes for the bf16 score engine.  First NOPE
    // come from knope_j[head], the rest ROPE from krope_j.  Only lane 0 means.
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
    // COMBINATIONAL MATMUL / WEIGHT-PULL DRIVE.
    //   mm_start pulses in GV_START (latches k_len + FP8 a_shift/w_scale).
    //   mm_in_valid high through GV_RUN (one beat/cycle, operand index gv_k).
    //   The weight request (w_req/w_sel/w_grp/w_k) describes the SAME beat so the
    //   external combinational responder returns FP8 w_col + w_scale in time.
    //   (FP8 engine takes start/in_valid when ~gv_score; bf16 score engine when
    //    gv_score -- gated at the instances above.)
    //========================================================================
    always @* begin
        mm_klen     = gv_klen;
        mm_start    = (gv_st == GV_START);
        mm_in_valid = (gv_st == GV_RUN);
        mm_a        = gv_a_elem;
        // weight request: NORMAL (non-score) passes only.  w_sel/w_grp also
        // address the per-pass block scales (latched by the FP8 matmul at start).
        w_req = (gv_st == GV_RUN) && ~gv_score;
        w_sel = gv_sel;
        w_grp = gv_grp;
        w_k   = gv_k;
    end

    //========================================================================
    // SEQUENTIAL CONTROL
    //========================================================================
    integer s_i, h_i, d_i;

    reg [$clog2(Q_LORA+1)-1:0]   rn_idx_q;   // qlora reduce read index
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
            out        <= {MODEL_DIM*16{1'b0}};
            kc_req     <= 1'b0; kc_idx <= {IDXW{1'b0}};
            pos_q      <= {POSW{1'b0}};
            s_reg      <= {(IDXW+1){1'b0}};
            rnq_start  <= 1'b0; rnk_start <= 1'b0;
            rnq_x_valid<= 1'b0; rnq_g_valid <= 1'b0;
            rnk_x_valid<= 1'b0; rnk_g_valid <= 1'b0;
            rnq_x_in   <= 16'h0; rnq_gamma_in <= 16'h0;
            rnk_x_in   <= 16'h0; rnk_gamma_in <= 16'h0;
            rp_start   <= 1'b0; rp_pos <= {POSW{1'b0}};
            rp_x_valid <= 1'b0; rp_x_in <= {ROPE_LANES*32{1'b0}};
            dsa_start  <= 1'b0; dsa_qidx <= {NOPE*16{1'b0}};
            dsa_slen   <= {(IDXW+1){1'b0}};
            dsa_kidx   <= {NOPE*16{1'b0}}; dsa_key_valid <= 1'b0;
            sm_start   <= 1'b0; sm_in_valid <= 1'b0; sm_x_in <= 16'h0;
            gv_st      <= GV_IDLE; gv_grp <= {GRPW{1'b0}}; gv_ng <= {GRPW{1'b0}};
            gv_k       <= {KCW{1'b0}}; gv_klen <= {KW{1'b0}}; gv_sel <= 4'd0;
            gv_dst     <= GVD_QLORA; gv_go <= 1'b0; gv_done <= 1'b0;
            gv_asrc    <= AS_X; gv_score <= 1'b0; gv_head <= {$clog2(H_HEADS+1){1'b0}};
            kst        <= K_RDREQ; ksel <= {(IDXW+1){1'b0}};
            sfst       <= SF_FEED; sf_head <= {$clog2(H_HEADS+1){1'b0}};
            sf_feed_i  <= {(IDXW+1){1'b0}}; sf_cap_i <= {(IDXW+1){1'b0}};
            cxst       <= CX_INIT; cx_head <= {$clog2(H_HEADS+1){1'b0}};
            cx_d       <= {$clog2(V_DIM+1){1'b0}}; cx_s <= {(IDXW+1){1'b0}};
            ctx_acc    <= 32'h0; sel_cnt <= {(IDXW+1){1'b0}};
            for (s_i=0; s_i<MODEL_DIM; s_i=s_i+1) begin xbuf[s_i]<=16'h0; outbuf[s_i]<=16'h0; end
            for (s_i=0; s_i<Q_LORA;   s_i=s_i+1) begin qlora[s_i]<=16'h0; qlora_n[s_i]<=16'h0; end
            for (s_i=0; s_i<HQK;      s_i=s_i+1) begin qfull[s_i]<=16'h0; qrot[s_i]<=16'h0; end
            for (s_i=0; s_i<KV_LORA;  s_i=s_i+1) begin ckv_cur[s_i]<=16'h0; ckv_n[s_i]<=16'h0; end
            for (s_i=0; s_i<ROPE;     s_i=s_i+1) begin krope_cur[s_i]<=16'h0; krope_j[s_i]<=16'h0; end
            for (s_i=0; s_i<HNOPE;    s_i=s_i+1) knope_j[s_i]<=16'h0;
            for (s_i=0; s_i<HV;       s_i=s_i+1) begin v_j[s_i]<=16'h0; ctx[s_i]<=16'h0; end
            for (h_i=0; h_i<H_HEADS;  h_i=h_i+1) begin
                for (s_i=0; s_i<S_MAX; s_i=s_i+1) begin
                    scores[h_i][s_i]<=16'h0; probs[h_i][s_i]<=16'h0;
                    for (d_i=0; d_i<V_DIM; d_i=d_i+1) vstore[h_i][s_i][d_i]<=16'h0;
                end
            end
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
                        for (tt = 0; tt < PE_N; tt = tt + 1) begin
                            case (gv_dst)
                            GVD_QLORA: if (gv_grp*PE_N+tt < Q_LORA)
                                          qlora[gv_grp*PE_N+tt]   <= mm_c[16*tt +:16];
                            GVD_QFULL: if (gv_grp*PE_N+tt < HQK)
                                          qfull[gv_grp*PE_N+tt]   <= mm_c[16*tt +:16];
                            GVD_CKV:   if (gv_grp*PE_N+tt < KV_LORA)
                                          ckv_cur[gv_grp*PE_N+tt] <= mm_c[16*tt +:16];
                            GVD_KR:    if (gv_grp*PE_N+tt < ROPE)
                                          krope_cur[gv_grp*PE_N+tt] <= mm_c[16*tt +:16];
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
                    pos_q <= pos;
                    s_reg <= s_len;
                    for (s_i=0; s_i<MODEL_DIM; s_i=s_i+1)
                        xbuf[s_i] <= x_vec[16*s_i +: 16];
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
                if (rnq_in_req) begin
                    rnq_x_in    <= qlora[rn_idx_q[$clog2(Q_LORA)-1:0]];
                    rnq_x_valid <= 1'b1;
                end
                if (rnq_g_req) begin
                    rnq_gamma_in <= 16'h3F80;   // bf16 1.0
                    rnq_g_valid  <= 1'b1;
                end
                if (rnq_y_valid) qlora_n[rn_yidx_q[$clog2(Q_LORA)-1:0]] <= rnq_y_out;
                if (rnq_done) begin
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
                    for (h_i=0; h_i<H_HEADS; h_i=h_i+1)
                        for (d_i=0; d_i<NOPE; d_i=d_i+1)
                            qrot[h_i*QK_DIM + d_i] <= qfull[h_i*QK_DIM + d_i];
                    rp_pos   <= pos_q;
                    rp_start <= 1'b1;
                    gv_head  <= {$clog2(H_HEADS+1){1'b0}};
                    state    <= S_QROPE;
                end
            end
            // ------------------------------------------------------------- Q rope
            S_QROPE: begin
                rp_x_valid <= 1'b0;
                if (rp_in_req) begin
                    rp_x_in[15:0]  <= qfull[gv_head*QK_DIM + NOPE + 2*rope_pair];
                    rp_x_in[31:16] <= qfull[gv_head*QK_DIM + NOPE + 2*rope_pair+1];
                    rp_x_valid     <= 1'b1;
                end
                if (rp_y_valid) begin
                    qrot[gv_head*QK_DIM + NOPE + 2*rope_yp]   <= rp_y_out[15:0];
                    qrot[gv_head*QK_DIM + NOPE + 2*rope_yp+1] <= rp_y_out[31:16];
                end
                if (rp_done) begin
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
                        rp_pos   <= pos_q;
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
                    rp_pos   <= pos_q;
                    rp_start <= 1'b1;          // rope the shared k_rope
                    state    <= S_KRROPE;
                end
            end
            // ------------------------------------------------------------- k_rope
            S_KRROPE: begin
                rp_x_valid <= 1'b0;
                if (rp_in_req) begin
                    rp_x_in[15:0]  <= krope_cur[2*rope_pair];
                    rp_x_in[31:16] <= krope_cur[2*rope_pair+1];
                    rp_x_valid     <= 1'b1;
                end
                if (rp_y_valid) begin
                    krope_cur[2*rope_yp]   <= rp_y_out[15:0];
                    krope_cur[2*rope_yp+1] <= rp_y_out[31:16];
                end
                if (rp_done) begin
                    for (d_i=0; d_i<NOPE; d_i=d_i+1)
                        dsa_qidx[16*d_i +: 16] <= qrot[d_i];   // head0 nope
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
                            for (d_i=0; d_i<KV_LORA; d_i=d_i+1)
                                ckv_cur[d_i] <= kc_ckv[16*d_i +: 16];
                            for (d_i=0; d_i<ROPE; d_i=d_i+1)
                                krope_j[d_i] <= kc_krope[16*d_i +: 16];
                            rnk_start <= 1'b1;
                            kst       <= K_NWAIT;
                        end
                    end
                    K_NWAIT: begin
                        rnk_x_valid <= 1'b0; rnk_g_valid <= 1'b0;
                        if (rnk_in_req) begin
                            rnk_x_in    <= ckv_cur[rn_idx_k[$clog2(KV_LORA)-1:0]];
                            rnk_x_valid <= 1'b1;
                        end
                        if (rnk_g_req) begin
                            rnk_gamma_in <= 16'h3F80;   // 1.0
                            rnk_g_valid  <= 1'b1;
                        end
                        if (rnk_y_valid) ckv_n[rn_yidx_k[$clog2(KV_LORA)-1:0]] <= rnk_y_out;
                        if (rnk_done) begin
                            // FP8 GEMV: ckv_n*W_uk -> knope_j (K=KV_LORA, OUT=HNOPE)
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
                        if (mm_out_valid && gv_st==GV_WAIT) begin
                            for (tt=0; tt<PE_N; tt=tt+1)
                                if (gv_grp*PE_N+tt < HNOPE)
                                    knope_j[gv_grp*PE_N+tt] <= mm_c[16*tt +:16];
                        end
                        if (gv_done) begin
                            // FP8 GEMV: ckv_n*W_uv -> v_j (K=KV_LORA, OUT=HV)
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
                    // bf16 SCORE pass: q_h . K_{h,j} (GEMV over QK_DIM, OUT=1)
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
                            scores[gv_head[$clog2(H_HEADS)-1:0]][ksel[IDXW-1:0]] <= mm_c[15:0];
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
                            sm_x_in     <= (sf_feed_i < sel_cnt) ?
                                           scores[sf_head[$clog2(H_HEADS)-1:0]][sf_feed_i[IDXW-1:0]]
                                         : NEG_BIG;
                            sf_feed_i   <= sf_feed_i + 1'b1;
                            if (sf_feed_i == S_MAX[IDXW:0]-1'b1)
                                sfst <= SF_CAP;
                        end
                    end
                    SF_CAP: begin
                        if (sm_out_valid) begin
                            probs[sf_head[$clog2(H_HEADS)-1:0]][sf_cap_i[IDXW-1:0]] <=
                                (sf_cap_i < sel_cnt) ? sm_p_out : 16'h0;
                            sf_cap_i <= sf_cap_i + 1'b1;
                        end
                        if (sm_done) sfst <= SF_NEXT;
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
                        ctx_acc <= 32'h0;
                        cx_s    <= {(IDXW+1){1'b0}};
                        cxst    <= CX_ACC;
                    end
                    CX_ACC: begin
                        ctx_acc <= fp32_add(ctx_acc,
                                     fp32_mul(
                                       bf16_to_fp32(probs[cx_head[$clog2(H_HEADS)-1:0]][cx_s[IDXW-1:0]]),
                                       bf16_to_fp32(vstore[cx_head[$clog2(H_HEADS)-1:0]][cx_s[IDXW-1:0]][cx_d[$clog2(V_DIM)-1:0]])));
                        if (cx_s == sel_cnt - 1'b1) cxst <= CX_STORE;
                        else cx_s <= cx_s + 1'b1;
                    end
                    CX_STORE: begin
                        ctx[ctx_lin[$clog2(HV)-1:0]] <= fp32_to_bf16(ctx_acc);
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
                    for (tt=0; tt<PE_N; tt=tt+1)
                        if (gv_grp*PE_N+tt < MODEL_DIM)
                            outbuf[gv_grp*PE_N+tt] <= mm_c[16*tt +:16];
                end
                if (gv_done) begin
                    for (s_i=0; s_i<MODEL_DIM; s_i=s_i+1)
                        out[16*s_i +: 16] <= outbuf[s_i];
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
    // small index helpers for the rmsnorm/rope pull answers.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            rn_idx_q <= 0; rn_yidx_q <= 0; rn_idx_k <= 0; rn_yidx_k <= 0;
            rope_pair <= 0; rope_yp <= 0; sm_in_valid_able <= 1'b0;
        end else begin
            if (rnq_start) begin rn_idx_q <= 0; rn_yidx_q <= 0; end
            else begin
                if (rnq_in_req) rn_idx_q <= rn_idx_q + 1'b1;
                if (rnq_y_valid) rn_yidx_q <= rn_yidx_q + 1'b1;
            end
            if (rnk_start) begin rn_idx_k <= 0; rn_yidx_k <= 0; end
            else begin
                if (rnk_in_req) rn_idx_k <= rn_idx_k + 1'b1;
                if (rnk_y_valid) rn_yidx_k <= rn_yidx_k + 1'b1;
            end
            if (rp_start) begin rope_pair <= 0; rope_yp <= 0; end
            else begin
                if (rp_in_req) rope_pair <= rope_pair + 1'b1;
                if (rp_y_valid) rope_yp <= rope_yp + 1'b1;
            end
            if (sm_start) sm_in_valid_able <= 1'b1;
            else if (state==S_SOFT && sfst==SF_FEED &&
                     sf_feed_i==S_MAX[IDXW:0]-1'b1 && sm_in_valid_able)
                     sm_in_valid_able <= 1'b0;
        end
    end

endmodule
