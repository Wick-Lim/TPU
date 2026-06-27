`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
//============================================================================
// mla_attn.v  --  GLM-5.2 MLA latent attention, ONE query token (decode step)
//                                                       (ACCEL_GLM52 §4.1)
//----------------------------------------------------------------------------
// FUNCTION  (single decode-step Multi-head Latent Attention, bf16 / fp32-acc)
//   This is the large ORCHESTRATOR FSM that wires the verified GLM-5.2 sub-units
//   into one MLA attention step for a single query token.  It reimplements NO
//   arithmetic -- every GEMM/GEMV is glm_matmul_pipe, every RMSNorm is
//   rmsnorm_unit, every rotary embed is rope_interleave_unit, the key selection
//   is dsa_indexer (+ its topk_select), and the per-head attention weights are
//   glm_softmax.  fp32 accumulate inside each sub-unit; bf16 operands + output.
//
//   FLOW (per docs §4.1):
//     Q PATH  (low-rank):
//        x --W_dq--> q_lora[Q_LORA]
//        q_lora --RMSNorm--> q_lora_n
//        q_lora_n --W_uq--> q[H_HEADS*QK_DIM]
//        split each head's QK_DIM into  q_nope[NOPE] | q_rope[ROPE]
//        rope_interleave(q_rope, pos)  (per head; NOPE gets NO position enc)
//     KV PATH (the MLA compression -- what is CACHED):
//        x --W_dkv--> c_kv[KV_LORA]        (cached per token; here it is the
//                                           CURRENT token's latent)
//        x --W_kr-->  k_rope_cur[ROPE]      (cached, SHARED across heads)
//        rope_interleave(k_rope_cur, pos)
//        For each PAST key j in the causal set the caller supplies the cached
//        c_kv[j] and (already-roped) k_rope[j] via the CACHE-READ interface; we
//        reconstruct per head:
//           c_kv[j] --RMSNorm--> --W_uk--> k_nope_j[H_HEADS*NOPE]
//           c_kv[j] --RMSNorm--> --W_uv--> v_j   [H_HEADS*V_DIM]
//           K_{h,j} = [ k_nope_j[h][NOPE] | k_rope[j][ROPE] ]   (QK_DIM)
//     SCORE / SOFTMAX / CONTEXT:
//        dsa_indexer selects the top-K key subset (dense when S<=TOPK -- still
//        instantiated/exercised); for each selected key j and head h:
//           score_{h,j} = q_h . K_{h,j}   over QK_DIM   (glm_matmul_pipe GEMV)
//        glm_softmax over the selected keys per head -> p_{h,j}
//        O_h = sum_j p_{h,j} * V_{h,j}   over V_DIM
//     OUTPUT PROJECTION:
//        concat heads (H_HEADS*V_DIM) --W_o--> out[MODEL_DIM]   (attention_bias=0)
//
//----------------------------------------------------------------------------
// SUB-UNIT REUSE (read each src for ports; none modified here)
//   glm_matmul_pipe  : ALL projections + the q.K scores (GEMV: PE_M=1).
//   rmsnorm_unit     : the q_lora norm and the c_kv norm.
//   rope_interleave_unit : q_rope (per head) and the single shared k_rope.
//   dsa_indexer      : top-K causal key selection (dense fallback for the slice).
//   glm_softmax      : per-head attention weights over the selected keys.
//
//----------------------------------------------------------------------------
// PARAMETERS (small-but-faithful slice; ALL parameterized)
//   MODEL_DIM, H_HEADS, NOPE, ROPE (QK_DIM=NOPE+ROPE), V_DIM, Q_LORA, KV_LORA,
//   S_MAX (max causal keys), TOPK (DSA budget; dense for the slice), THETA, PE_N
//   (matmul tile width = output lanes/pass), POSW.  IDX_DIM for the indexer is
//   derived = NOPE (the indexer scores a small slice of the nope part).
//
//----------------------------------------------------------------------------
// INTERFACES
//   CONTROL  : start (pulse), busy, done (pulse), pos, s_len (S causal keys).
//   X INPUT  : x_vec[MODEL_DIM bf16], latched at start.
//   WEIGHT PULL (combinational, like swiglu_expert/moe_router):
//        w_req            : high on a beat that needs a weight column.
//        w_sel  [3:0]     : which projection: 0=W_dq 1=W_uq 2=W_dkv 3=W_kr
//                           4=W_uk 5=W_uv 6=W_o.  (the q.K score reads K from
//                           the reconstructed-key buffers, NOT a streamed weight.)
//        w_grp  [GRPW]    : output tile-group index g (outputs g*PE_N..+PE_N-1).
//        w_k    [KW]      : the reduction index k of this beat.
//        w_col  [PE_N*16] : the PE_N bf16 weight lanes  W_sel[g*PE_N + t , w_k],
//                           presented COMBINATIONALLY the same cycle.
//   CACHE READ (past-key latents supplied by the caller's KV cache / TB):
//        kc_req           : high while the unit wants a cached key.
//        kc_idx [IDXW]    : which causal key j (0..S-1) is requested.
//        kc_ckv [KV_LORA*16]  : cached latent c_kv[j]   (bf16).
//        kc_krope [ROPE*16]   : cached, already-roped k_rope[j] (bf16, shared/head).
//        kc_valid         : producer asserts when kc_ckv/kc_krope hold key kc_idx.
//   OUTPUT   : out[MODEL_DIM bf16], done pulse.
//
//----------------------------------------------------------------------------
// FSM STAGES (deterministic latency; sync active-high reset; no latch/comb loop)
//   S_IDLE     : wait start; latch x, pos, s_len.
//   S_QDQ      : x --W_dq--> q_lora            (GEMV over MODEL_DIM)
//   S_QNORM    : q_lora --RMSNorm--> q_lora_n
//   S_QUQ      : q_lora_n --W_uq--> q[H*QK_DIM] (GEMV over Q_LORA)
//   S_QROPE    : rope q_rope per head (NOPE part copied through unchanged)
//   S_KVDKV    : x --W_dkv--> c_kv_cur         (GEMV over MODEL_DIM)
//   S_KVKR     : x --W_kr--> k_rope_cur        (GEMV over MODEL_DIM)
//   S_KRROPE   : rope the shared k_rope_cur
//   S_DSA      : dsa_indexer selects keys (dense for slice) -> sel list
//   S_KEY      : for each selected key j: read cache, RMSNorm(c_kv[j]),
//                W_uk->k_nope, W_uv->v, assemble K_{h,j}, V_{h,j}; for head h
//                score_{h,j} = q_h . K_{h,j} (GEMV) into the score row.
//   S_SOFT     : per head, glm_softmax over the S_sel scores -> p row.
//   S_CTX      : O_h = sum_j p_{h,j} * V_{h,j}  (fp32 accumulate -> bf16).
//   S_OUT      : concat O --W_o--> out          (GEMV over H*V_DIM)
//   S_DONE     : pulse done; outputs held.
//
//   Latency is fully data-independent for a given (S, params): each GEMV pass is
//   K + `FP_MAC_LAT + 3*`FP_ADD_LAT + 1 cycles, the norms/ropes/softmax are their
//   documented fixed latencies, and the per-key loop runs S_sel*H deterministic
//   score passes.  No data-dependent stall anywhere.
//============================================================================
module mla_attn #(
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
    parameter integer KW        = $clog2(KMAX + 1)        // matmul k_len width
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

    // ---- weight pull (combinational responder, like swiglu_expert) ----
    output reg                         w_req,
    output reg  [3:0]                  w_sel,      // 0..6 projection select
    output reg  [GRPW-1:0]             w_grp,      // output tile-group index
    output reg  [KCW-1:0]              w_k,        // reduction index k of this beat
    input  wire [PE_N*16-1:0]          w_col,      // PE_N bf16 weight lanes

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
    // ROPE/2 pairs; pick LANES = 1 for generality (slice ROPE small).
    localparam integer ROPE_LANES = 1;

    integer tt;

    //========================================================================
    // INPUT / INTERMEDIATE BUFFERS  (bf16 storage; fp32 only inside sub-units)
    //========================================================================
    reg [15:0] xbuf      [0:MODEL_DIM-1];   // latched x
    reg [POSW-1:0] pos_q;
    reg [IDXW:0]   s_reg;                    // S causal keys

    reg [15:0] qlora     [0:Q_LORA-1];       // x*W_dq
    reg [15:0] qlora_n   [0:Q_LORA-1];       // RMSNorm(qlora)
    reg [15:0] qfull     [0:HQK-1];          // qlora_n*W_uq  (per head QK_DIM)
    // q after rope (nope copied, rope rotated), per head laid out as QK_DIM
    reg [15:0] qrot      [0:HQK-1];

    reg [15:0] ckv_cur   [0:KV_LORA-1];      // x*W_dkv  (current token latent)
    reg [15:0] krope_cur [0:ROPE-1];         // x*W_kr -> roped (shared)

    // per-current-key reconstruction scratch
    reg [15:0] ckv_n     [0:KV_LORA-1];      // RMSNorm(c_kv[j])
    reg [15:0] knope_j   [0:HNOPE-1];        // ckv_n*W_uk  (per head NOPE)
    reg [15:0] v_j       [0:HV-1];           // ckv_n*W_uv  (per head V_DIM)

    // assembled per-(head,selected-key) score + per-(head,key) V copy for ctx.
    // scores[h][s] = q_h . K_{h,j(s)} ; vstore[h][s][d] = V_{h,j(s)}[d]
    reg [15:0] scores    [0:H_HEADS-1][0:S_MAX-1];
    reg [15:0] vstore    [0:H_HEADS-1][0:S_MAX-1][0:V_DIM-1];
    reg [15:0] probs     [0:H_HEADS-1][0:S_MAX-1];   // softmax weights
    reg [15:0] ctx       [0:HV-1];                    // O concat (H*V_DIM)
    reg [15:0] outbuf    [0:MODEL_DIM-1];

    // DSA selection results
    reg [IDXW-1:0] sel_list [0:TOPK-1];
    reg [IDXW:0]   sel_cnt;

    //========================================================================
    // SHARED GEMV ENGINE :  ONE glm_matmul_pipe, PE_M=1 (single query row),
    // PE_N tile width.  A 1xPE_N output tile per pass; K reduced by streaming.
    // The mm_* regs are driven combinationally by the GEMV micro-sequencer.
    //========================================================================
    reg              mm_start;
    reg  [KW-1:0]    mm_klen;
    reg              mm_in_valid;
    reg  [15:0]      mm_a;                    // the single A element (PE_M=1)
    reg  [PE_N*16-1:0] mm_w;                  // PE_N bf16 weight lanes
    wire             mm_busy;
    wire             mm_out_valid;
    wire [PE_N*16-1:0] mm_c;                  // 1xPE_N bf16 output tile

    glm_matmul_pipe #(.PE_M(1), .PE_N(PE_N), .KMAX(KMAX)) u_mm (
        .clk(clk), .rst(rst),
        .start(mm_start), .k_len(mm_klen),
        .in_valid(mm_in_valid), .a_col(mm_a), .w_row(mm_w),
        .busy(mm_busy), .out_valid(mm_out_valid), .c_out(mm_c)
    );

    //========================================================================
    // SUB-UNIT : rmsnorm_unit (reused for q_lora and c_kv).  LANES=1 (slice).
    //   It PULLS x (reduce pass) then gamma (normalize pass); we answer from the
    //   selected source buffer.  gamma is all-ones here (identity scale) so the
    //   norm is the pure RMS scaling; a real design streams the learned gamma.
    //========================================================================
    // rmsnorm_unit is fixed-LEN per instance; we need TWO lengths (Q_LORA and
    // KV_LORA).  Instantiate ONE per length and mux which one is driven.
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
    wire _rn_busy_unused = &{1'b0, rnq_busy, rnk_busy, mm_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // SUB-UNIT : rope_interleave_unit (q_rope per head + shared k_rope).
    //   ROT_DIM=ROPE.  It PULLS pairs (LANES pairs/beat); we feed from the
    //   rope source buffer and capture the rotated pairs back.
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
    //   IDX_DIM = NOPE (it scores a small slice of the nope part of head-0).
    //   It PULLS each key's index vector; we answer with head-0 k_nope... but for
    //   the slice S<=TOPK the indexer is a NO-OP (emits 0..S-1) and never pulls
    //   keys, so we still EXERCISE its interface without reconstructing keys here.
    //   q_idx is head-0's q_nope slice (latched at indexer start).
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
    //   LEN = S_MAX, LANES = 1.  We pad unselected slots (s>=S_sel) with a very
    //   negative logit so their probability is ~0 (a clean per-head softmax over
    //   exactly the selected keys; the slice keeps all keys so S_sel == S).
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
    // serial fp32 mac-free accumulate using the combinational glm_fp ops.
    //========================================================================
    reg [31:0] ctx_acc;

    //========================================================================
    // MASTER FSM
    //========================================================================
    localparam [4:0]
        S_IDLE  = 5'd0,
        S_QDQ   = 5'd1,    // x*W_dq -> qlora
        S_QNORM = 5'd2,    // RMSNorm(qlora) -> qlora_n
        S_QUQ   = 5'd3,    // qlora_n*W_uq -> qfull
        S_QROPE = 5'd4,    // rope q_rope per head
        S_KVDKV = 5'd5,    // x*W_dkv -> ckv_cur
        S_KVKR  = 5'd6,    // x*W_kr -> krope_cur
        S_KRROPE= 5'd7,    // rope shared k_rope
        S_DSA   = 5'd8,    // dsa_indexer select keys
        S_KEY   = 5'd9,    // per selected key: norm/W_uk/W_uv/assemble/score
        S_SOFT  = 5'd10,   // per head softmax over scores
        S_CTX   = 5'd11,   // weighted-V context
        S_OUT   = 5'd12,   // ctx*W_o -> out
        S_DONE  = 5'd13;
    reg [4:0] state;

    // ---- GEMV micro-sequencer sub-state (shared by every projection stage) ----
    // A GEMV pass computes OUT outputs in NG=ceil(OUT/PE_N) tile-groups, each a
    // matmul tile over K beats.  gv_* track the current group and K beat.
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
    // (W_uk/W_uv/W_o/score destinations are captured inline in S_KEY/S_OUT by
    //  snooping mm_out_valid, so no extra destination code is needed.)

    // The A-source for the GEMV (which buffer feeds a_col[k]) is selected by
    // gv_asrc; weight lanes always come from the external w_col responder EXCEPT
    // for the score pass which streams K from the reconstructed-key buffers.
    localparam [2:0] AS_X=3'd0, AS_QLN=3'd1, AS_CTX=3'd2, AS_Q=3'd3, AS_CKVN=3'd4;
    reg [2:0]        gv_asrc;
    // score pass: stream q_h (A) and K_{h,j} (W) from internal buffers.
    reg              gv_score;     // this GEMV pass is a q.K score (internal W)
    reg [$clog2(H_HEADS+1)-1:0] gv_head;   // head for the score pass
    // score weight lanes come from K assembled buffer; we provide them via
    // kw_lane() below.  Since PE_N may exceed 1, the score uses PE_N=... but to
    // keep scores scalar (one score per (h,j)) we run the score with a 1-wide
    // view: gv_ng=1 and only lane 0 of the tile is meaningful.

    // ---- per-key loop bookkeeping (S_KEY) ----
    localparam [3:0]
        K_RDREQ=4'd0,  // request cache read for selected key s
        K_RDWAIT=4'd1, // wait kc_valid; latch c_kv[j], k_rope[j]
        K_NWAIT=4'd3,  // RMSNorm(c_kv[j]) -> ckv_n
        K_UK=4'd4,     // ckv_n*W_uk -> knope_j (per head)
        K_UV=4'd5,     // ckv_n*W_uv -> v_j     (per head)
        K_SCORE=4'd7,  // for each head: q_h . K_{h,j} -> scores[h][s]
        K_NEXTH=4'd8,  // advance head in score loop
        K_NEXT=4'd9;   // advance selected key s
    reg [3:0]        kst;
    reg [IDXW:0]     ksel;         // index into sel_list (0..sel_cnt-1)
    reg [15:0]       krope_j [0:ROPE-1];   // latched k_rope[j]
    // assembled K for the current (head being scored): NOPE from knope_j[head],
    // ROPE from krope_j.  Built combinationally per beat in the score GEMV.

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
    // sized linear address ctx[head*V_DIM + d]; low $clog2(HV) bits address ctx.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] ctx_lin = cx_head*V_DIM + {{(32-$clog2(V_DIM+1)){1'b0}}, cx_d};
    /* verilator lint_on UNUSEDSIGNAL */
    reg [IDXW:0]     cx_s;

    //========================================================================
    // GEMV micro-sequencer (combinational operand drive + sequential control).
    // Drives mm_* from the chosen A-source buffer and the external w_col (or the
    // assembled-K for a score pass).  Writes the 1xPE_N output tile into the
    // destination buffer at out_valid.
    //========================================================================
    // combinational: a_col element for current K beat from the selected source.
    reg [15:0]      gv_a_elem;
    // 32-bit linear address for the q_h element of the score pass (head*QK+k);
    // only the low $clog2(HQK) bits address qrot -- waive the unused upper bits.
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
    // combinational: weight lanes for current beat.  Normal pass = external
    // w_col.  Score pass = assembled K_{gv_head, j}[gv_k] in lane 0.
    reg [PE_N*16-1:0] gv_w_lanes;
    reg [15:0]        score_k_elem;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0]       knope_lin = gv_head*NOPE + {{(32-KCW){1'b0}}, gv_k};
    wire [31:0]       krope_lin = {{(32-KCW){1'b0}}, gv_k} - NOPE;
    /* verilator lint_on UNUSEDSIGNAL */
    always @* begin
        // K element for the score: first NOPE come from knope_j[head], the rest
        // ROPE come from krope_j.
        if (gv_k < KCW'(NOPE))
            score_k_elem = knope_j[knope_lin[$clog2(HNOPE)-1:0]];
        else
            score_k_elem = krope_j[krope_lin[$clog2(ROPE)-1:0]];
        if (gv_score) begin
            gv_w_lanes = {PE_N*16{1'b0}};
            gv_w_lanes[15:0] = score_k_elem;     // only lane 0 meaningful
        end else begin
            gv_w_lanes = w_col;
        end
    end

    //========================================================================
    // COMBINATIONAL MATMUL / WEIGHT-PULL DRIVE.
    //   mm_start pulses in GV_START (latches k_len, clears ps; no beat issued).
    //   mm_in_valid is high through GV_RUN (one beat/cycle, operand index gv_k);
    //   mm_a / mm_w present the A element and the PE_N weight lanes for gv_k.
    //   The weight request (w_req/w_sel/w_grp/w_k) describes the SAME beat, so the
    //   external combinational responder returns w_col in time (score pass uses
    //   internal K instead, so it does not assert w_req).
    //========================================================================
    always @* begin
        mm_klen     = gv_klen;
        mm_start    = (gv_st == GV_START);
        mm_in_valid = (gv_st == GV_RUN);
        mm_a        = gv_a_elem;
        mm_w        = gv_w_lanes;
        // weight request: assert during the beat stream of a NORMAL (non-score)
        // pass so the responder presents w_col on the same cycle.
        w_req = (gv_st == GV_RUN) && ~gv_score;
        w_sel = gv_sel;
        w_grp = gv_grp;
        w_k   = gv_k;
    end

    //========================================================================
    // SEQUENTIAL CONTROL
    //========================================================================
    integer s_i, h_i, d_i;

    // ---- helper counters for the rmsnorm/rope pull answers (declared here so
    // they are in scope for the FSM that reads them; their update logic is the
    // small always block at the end of the module).  Because LANES=1 the beat
    // index IS the element index.
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
            // sub-unit strobes (mm_*/w_* are combinational -- not reset here)
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
            // gemv seq
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
            // GEMV MICRO-SEQUENCER (runs whenever gv_go pulses; drives u_mm).
            // mm_start / mm_in_valid / mm_a / mm_w and the w_* weight request are
            // driven COMBINATIONALLY (see the always @* below) from {gv_st,gv_k,
            // gv_grp,...}; here we only sequence the counters.  GV_RUN issues a
            // valid beat for gv_k = 0..klen-1 (exactly klen beats), then waits.
            //================================================================
            case (gv_st)
                GV_IDLE: begin
                    if (gv_go) begin
                        gv_grp   <= {GRPW{1'b0}};
                        gv_k     <= {KCW{1'b0}};
                        gv_st    <= GV_START;
                    end
                end
                // one cycle: assert mm_start (latches k_len, clears ps); no valid.
                GV_START: begin
                    gv_k  <= {KCW{1'b0}};
                    gv_st <= GV_RUN;
                end
                // stream K beats for the current tile-group (combinational drive):
                // mm_in_valid high, operand index gv_k = 0..klen-1.
                GV_RUN: begin
                    if (gv_k == gv_klast) begin
                        gv_st <= GV_WAIT;        // issued all klen beats
                    end else begin
                        gv_k <= gv_k + 1'b1;
                    end
                end
                // wait for this tile-group's out_valid, store it, advance group
                GV_WAIT: begin
                    if (mm_out_valid) begin
                        // store the 1xPE_N output tile into the destination
                        for (tt = 0; tt < PE_N; tt = tt + 1) begin
                            // global output index for this lane
                            // (group base + lane); guarded by destination length
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
                    // launch GEMV: x*W_dq -> qlora  (K=MODEL_DIM, OUT=Q_LORA)
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
                if (gv_go) gv_go <= 1'b0;          // 1-cycle launch pulse
                else if (gv_done) begin
                    // start RMSNorm over qlora (Q_LORA) -- pull x then gamma
                    rnq_start <= 1'b1;
                    state     <= S_QNORM;
                end
            end
            // ------------------------------------------------------------- Q norm
            S_QNORM: begin
                // answer rmsnorm pulls: x = qlora[k] (reduce), gamma = 1.0 (norm)
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
                    // launch GEMV: qlora_n*W_uq -> qfull  (K=Q_LORA, OUT=HQK)
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
                    // copy NOPE parts through; queue rope of head-0 q_rope first.
                    for (h_i=0; h_i<H_HEADS; h_i=h_i+1)
                        for (d_i=0; d_i<NOPE; d_i=d_i+1)
                            qrot[h_i*QK_DIM + d_i] <= qfull[h_i*QK_DIM + d_i];
                    rp_pos   <= pos_q;
                    rp_start <= 1'b1;
                    gv_head  <= {$clog2(H_HEADS+1){1'b0}};   // rope head 0 first
                    state    <= S_QROPE;
                end
            end
            // ------------------------------------------------------------- Q rope
            // rope the ROPE slice of head gv_head; when done advance head; after
            // last head, move to KV path.
            S_QROPE: begin
                rp_x_valid <= 1'b0;
                if (rp_in_req) begin
                    // feed pair (lane 0): even=q_rope[2i], odd=q_rope[2i+1]
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
                        // all heads roped -> KV path: x*W_dkv -> ckv_cur
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
                    // x*W_kr -> krope_cur  (K=MODEL_DIM, OUT=ROPE)
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
                    // launch DSA key selection.  q_idx = head-0 q_nope slice.
                    for (d_i=0; d_i<NOPE; d_i=d_i+1)
                        dsa_qidx[16*d_i +: 16] <= qrot[d_i];   // head0 nope
                    dsa_slen  <= s_reg;
                    dsa_start <= 1'b1;
                    state     <= S_DSA;
                end
            end
            // ------------------------------------------------------------- DSA
            S_DSA: begin
                // answer key pulls if the indexer scores (only when S>TOPK).
                dsa_key_valid <= 1'b0;
                if (dsa_key_req) begin
                    // present head-0 nope of key dsa_key_idx... but for the slice
                    // S<=TOPK the indexer is dense and never pulls.  Provide zeros
                    // here to keep the interface exercised/safe.
                    dsa_kidx      <= {NOPE*16{1'b0}};
                    dsa_key_valid <= 1'b1;
                end
                if (dsa_done) begin
                    for (s_i=0; s_i<TOPK; s_i=s_i+1)
                        sel_list[s_i] <= dsa_sel_idx[IDXW*s_i +: IDXW];
                    sel_cnt <= dsa_sel_count;
                    // begin per-key reconstruction loop
                    ksel    <= {(IDXW+1){1'b0}};
                    kst     <= K_RDREQ;
                    state   <= S_KEY;
                end
            end
            // ------------------------------------------------------------- per-key
            S_KEY: begin
                case (kst)
                    // request cache read for selected key sel_list[ksel]
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
                            // RMSNorm c_kv[j] (KV_LORA)
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
                            // ckv_n*W_uk -> knope_j (K=KV_LORA, OUT=HNOPE)
                            gv_asrc  <= AS_CKVN;
                            gv_sel   <= SEL_UK;
                            gv_klen  <= KW'(KV_LORA);
                            gv_ng    <= GRPW'((HNOPE + PE_N - 1)/PE_N);
                            gv_dst   <= GVD_QFULL; // reuse path? no -> handle UK
                            gv_score <= 1'b0;
                            gv_go    <= 1'b1;
                            kst      <= K_UK;
                        end
                    end
                    // W_uk pass: capture knope_j from the matmul output directly
                    // (override gv_dst store path: we snoop mm_out_valid here).
                    K_UK: begin
                        if (gv_go) gv_go <= 1'b0;
                        // capture tile outputs into knope_j as they retire
                        if (mm_out_valid && gv_st==GV_WAIT) begin
                            for (tt=0; tt<PE_N; tt=tt+1)
                                if (gv_grp*PE_N+tt < HNOPE)
                                    knope_j[gv_grp*PE_N+tt] <= mm_c[16*tt +:16];
                        end
                        if (gv_done) begin
                            // ckv_n*W_uv -> v_j (K=KV_LORA, OUT=HV)
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
                            // copy V_{h,j} into vstore for the context stage
                            for (h_i=0; h_i<H_HEADS; h_i=h_i+1)
                                for (d_i=0; d_i<V_DIM; d_i=d_i+1)
                                    vstore[h_i][ksel[IDXW-1:0]][d_i]
                                        <= v_j[h_i*V_DIM + d_i];
                            gv_head <= {$clog2(H_HEADS+1){1'b0}};
                            kst     <= K_SCORE;
                        end
                    end
                    // score head gv_head: q_h . K_{h,j} (GEMV over QK_DIM, OUT=1)
                    K_SCORE: begin
                        // launch a 1-output score GEMV (gv_ng=1, score mode)
                        gv_asrc  <= AS_Q;
                        gv_klen  <= KW'(QK_DIM);
                        gv_ng    <= GRPW'(1);
                        gv_score <= 1'b1;
                        gv_go    <= 1'b1;
                        kst      <= K_NEXTH;
                    end
                    K_NEXTH: begin
                        if (gv_go) gv_go <= 1'b0;
                        // capture the scalar score (lane 0) when the pass retires
                        if (mm_out_valid && gv_st==GV_WAIT)
                            scores[gv_head[$clog2(H_HEADS)-1:0]][ksel[IDXW-1:0]] <= mm_c[15:0];
                        if (gv_done) begin
                            if (gv_head == H_HEADS[$clog2(H_HEADS+1)-1:0]-1'b1) begin
                                gv_score <= 1'b0;
                                kst      <= K_NEXT;
                            end else begin
                                gv_head  <= gv_head + 1'b1;
                                kst      <= K_SCORE;   // score next head
                            end
                        end
                    end
                    K_NEXT: begin
                        if (ksel == sel_cnt - 1'b1) begin
                            // all selected keys processed -> softmax stage
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
            // per head: feed S_MAX logits (real scores for s<sel_cnt, NEG_BIG pad
            // for s>=sel_cnt), capture S_MAX probs, mask padding to 0.
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
                            // all heads done -> context
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
            // O_h[d] = sum_s probs[h][s]*vstore[h][s][d]   (fp32 acc, bf16 out)
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
                                // ctx*W_o -> out  (K=HV, OUT=MODEL_DIM)
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
    // small index helpers for the rmsnorm/rope pull answers.  rmsnorm_unit pulls
    // x in order then gamma in order; we mirror its internal beat counter with a
    // local counter that advances on each accepted pull.  Because LANES=1 the
    // beat IS the element index.  We track them as free-running counters reset at
    // each rnq/rnk start and the rope pair index similarly.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            rn_idx_q <= 0; rn_yidx_q <= 0; rn_idx_k <= 0; rn_yidx_k <= 0;
            rope_pair <= 0; rope_yp <= 0; sm_in_valid_able <= 1'b0;
        end else begin
            // qlora rmsnorm counters
            if (rnq_start) begin rn_idx_q <= 0; rn_yidx_q <= 0; end
            else begin
                if (rnq_in_req) rn_idx_q <= rn_idx_q + 1'b1;
                if (rnq_y_valid) rn_yidx_q <= rn_yidx_q + 1'b1;
            end
            // ckv rmsnorm counters
            if (rnk_start) begin rn_idx_k <= 0; rn_yidx_k <= 0; end
            else begin
                if (rnk_in_req) rn_idx_k <= rn_idx_k + 1'b1;
                if (rnk_y_valid) rn_yidx_k <= rn_yidx_k + 1'b1;
            end
            // rope pair counters (reset at each rope start)
            if (rp_start) begin rope_pair <= 0; rope_yp <= 0; end
            else begin
                if (rp_in_req) rope_pair <= rope_pair + 1'b1;
                if (rp_y_valid) rope_yp <= rope_yp + 1'b1;
            end
            // softmax feed-enable: high while softmax is busy and feeding logits.
            if (sm_start) sm_in_valid_able <= 1'b1;
            else if (state==S_SOFT && sfst==SF_FEED &&
                     sf_feed_i==S_MAX[IDXW:0]-1'b1 && sm_in_valid_able)
                     sm_in_valid_able <= 1'b0;
        end
    end

endmodule
