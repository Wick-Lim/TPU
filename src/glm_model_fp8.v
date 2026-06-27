`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_model_fp8.v  --  GLM-5.2-FP8 FULL FORWARD PASS for ONE token position
//                      at the small-but-faithful slice  (ACCEL_GLM52 §2/§6/§8)
//----------------------------------------------------------------------------
// FUNCTION  (one decode step, single query token at position `pos`)
//
//    x0   = embed(token_id)                                     // bf16 table lookup
//    x_{l+1} = decoder_block_fp8( x_l , layer=l , mode_l )  l=0..L-1
//                where mode_l = DENSE   for l <  N_DENSE  (first_k_dense_replace)
//                              = MoE     for l >= N_DENSE
//    xN   = rmsnorm_final( x_L )                                // final RMSNorm (bf16)
//    logits[V] = W_lm[V,MODEL_DIM] . xN                         // LM head GEMV (bf16)
//    argmax = arg max_v logits[v]                               // next token
//
//   This is the FP8-NATIVE sibling of glm_model.v.  It is byte-for-byte the SAME
//   orchestrator -- the SAME master FSM, the SAME serial layer walk over ONE
//   decoder-block instance with a running bf16 residual hidden state, the SAME
//   per-layer dense/MoE mode + DSA IndexShare schedule, the SAME final RMSNorm
//   and LM-head GEMV -- with ONE structural change:
//
//     * the L transformer layers run on ONE glm_decoder_block_FP8 instance (the
//       verified GLM-5.2-FP8 layer) instead of glm_decoder_block, so the big
//       LINEAR WEIGHT matmuls inside each layer (attention projections, router
//       gate, swiglu gate/up/down) consume OFFICIAL GLM-5.2-FP8 weights: E4M3
//       codes + per-[128,128]-block bf16 dequant scales + on-chip dynamic
//       per-token activation->E4M3 quant.  This module FORWARDS those FP8 codes
//       (aw_col/rw_col/fw_col*) AND the bf16 block scales (aw_scale/rw_scale/
//       fw_scale_*) straight out, annotated with db_layer, exactly like the
//       bf16 capstone forwards its bf16 weight pulls.
//
//   FP8 SPLIT (modules_to_not_convert preserved).  KEPT bf16, NOT FP8:
//     * the token EMBEDDING lookup (em_* pull, bf16 element);
//     * the FINAL rmsnorm_unit (bf16, gamma pulled bf16 via fn_*);
//     * the LM-head GEMV (glm_matmul_pipe, bf16 W_lm via lw_*);
//     * the running residual hidden state (bf16 across all layers);
//     * and INSIDE each FP8 layer, all norms/softmax/rope/sigmoid/topk/residual
//       (the decoder_block_fp8 keeps those bf16).
//   ONLY the per-layer attention/router/expert WEIGHT matmuls are FP8.
//
//   PURE ORCHESTRATOR: reimplements NO arithmetic; orchestrates ONE
//   glm_decoder_block_fp8 (serial layer reuse), ONE rmsnorm_unit (final norm),
//   ONE glm_matmul_pipe (LM-head GEMV).  Embedding is a combinational table pull.
//
//----------------------------------------------------------------------------
// LATENCY  (deterministic, data-independent for given params + S)
//   L_model = L_embed (MODEL_DIM beats load)
//           + L * L_decoder_block_fp8(params, S)       // serial block reuse
//           + L_rmsnorm(MODEL_DIM)                      // final norm (bf16)
//           + LM head: (VOCAB/LM_TN) tiles, each
//                 MODEL_DIM (K stream) + `FP_MAC_LAT + 3*`FP_ADD_LAT + few
//           + VOCAB (argmax scan)
//   No data-dependent stall; sync active-high reset; no latch; no comb loop.
//
//----------------------------------------------------------------------------
// CONVENTIONS: `timescale + glm_fp.vh; synchronous ACTIVE-HIGH reset; no latch;
//   no combinational loop.  All weight / cache / embedding delivery is via
//   combinational PULL interfaces answered the SAME cycle by the system/TB.
//============================================================================
module glm_model_fp8 #(
    // ---- model / slice config (small-but-faithful, ACCEL_GLM52 §8.1) ----
    parameter integer MODEL_DIM  = 128,
    parameter integer L          = 6,           // total layers (3 dense + 3 MoE)
    parameter integer N_DENSE    = 3,            // first_k_dense_replace
    parameter integer VOCAB      = 256,
    // ---- decoder_block_fp8 slice params (passed straight through) ----
    parameter integer H_HEADS    = 4,
    parameter integer NOPE       = 16,
    parameter integer ROPE       = 16,
    parameter integer V_DIM      = 32,
    parameter integer Q_LORA     = 64,
    parameter integer KV_LORA    = 32,
    parameter integer S_MAX      = 8,
    parameter integer TOPK_ATTN  = 8,           // DSA budget (dense for slice)
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 4,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 8,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 64,
    parameter integer INTER_DENSE= 256,
    parameter [31:0]  RSCALE     = 32'h40200000,// 2.5 fp32
    parameter integer TN         = 4,
    // ---- FP8 weight block size -- GLM-5.2-FP8 weight_block_size=[128,128] ----
    parameter integer BLK        = 128,
    // ---- LM-head GEMV tile width (VOCAB cols/pass).  VOCAB % LM_TN == 0. ----
    parameter integer LM_TN      = 4,
    // ====================================================================
    // derived (do NOT override) -- mirror decoder_block_fp8's port-width derivations
    // ====================================================================
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer HQK        = H_HEADS * QK_DIM,
    parameter integer HNOPE      = H_HEADS * NOPE,
    parameter integer HV         = H_HEADS * V_DIM,
    parameter integer EIDXW      = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer A_KMAX     = (MODEL_DIM > Q_LORA) ?
                               ((MODEL_DIM > KV_LORA) ?
                                ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV))
                             : ((Q_LORA > KV_LORA) ?
                                ((Q_LORA > HV) ? Q_LORA : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV)),
    parameter integer A_OMAX     = (HQK > MODEL_DIM) ?
                               ((HQK > HNOPE) ?
                                 ((HQK > HV) ? HQK : HV)
                               : ((HNOPE > HV) ? HNOPE : HV))
                             : ((MODEL_DIM > HNOPE) ?
                                 ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                               : ((HNOPE > HV) ? HNOPE : HV)),
    parameter integer A_NGMAX    = (A_OMAX + PE_N - 1) / PE_N,
    parameter integer A_GRPW     = (A_NGMAX <= 1) ? 1 : $clog2(A_NGMAX),
    parameter integer A_KCW      = (A_KMAX  <= 1) ? 1 : $clog2(A_KMAX),
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1),
    parameter integer FF_KMAX_D  = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM,
    parameter integer FF_KWD     = $clog2(FF_KMAX_D + 1),
    parameter integer FF_KMAX_M  = (INTER_MOE  > MODEL_DIM) ? INTER_MOE  : MODEL_DIM,
    parameter integer R_KW       = $clog2(FF_KMAX_M + 1),
    // ---- FP8 [128,128]-block scale counts (#K-blocks per weight family) ----
    parameter integer A_NB       = (A_KMAX    + BLK - 1) / BLK,  // attention scales
    parameter integer FF_NB_D    = (FF_KMAX_D + BLK - 1) / BLK,  // dense FFN scales
    parameter integer R_NB       = (FF_KMAX_M + BLK - 1) / BLK,  // router scales
    // model-level derived
    parameter integer LAYW       = (L     <= 1) ? 1 : $clog2(L),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    parameter integer LMKW       = $clog2(MODEL_DIM + 1)   // matmul_pipe k_len width
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    // ---- control ----
    input  wire                          start,      // 1-cycle pulse: begin token
    output reg                           busy,
    output reg                           done,       // 1-cycle pulse: logits valid
    input  wire [TOKW-1:0]               token_id,   // input token to embed
    input  wire [POSW-1:0]               pos,        // query position (RoPE)
    input  wire [IDXW:0]                 s_len,      // S causal keys (<= S_MAX)

    // ---- outputs ----
    output reg  [VOCAB*16-1:0]           logits,     // VOCAB bf16 next-token logits
    output reg  [TOKW-1:0]               argmax,     // arg max logit (next token)

    // ---- embedding table pull (combinational, BF16): em_val = Emb[token_id][em_idx] ----
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,     // = token_id (for the ROM)
    output wire [DIMW-1:0]               em_idx,     // element index 0..MODEL_DIM-1
    input  wire [15:0]                   em_val,     // bf16 embedding element

    // ---- per-layer index (annotates ALL decoder_block_fp8 pulls below) ----
    output reg  [LAYW-1:0]               db_layer,   // current layer being run
    // ---- IndexShare schedule (DSA, ACCEL_GLM52 #9) ----
    output reg                           idx_fresh,  // 1 = (layer mod 4 == 3): fresh
    output reg  [LAYW-1:0]               idx_win,    // share-window id = layer>>2

    // ---- decoder_block_fp8 RMSNorm gamma pull (BF16; per-layer; pre-attn/pre-FFN) ----
    output wire                          gn_req,
    output wire                          gn_which,   // 0=pre-attn, 1=pre-FFN
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,

    // ---- decoder_block_fp8 attention weight pull (per-layer; FP8 E4M3 + scales) ----
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*8-1:0]             aw_col,     // PE_N FP8 E4M3 weight lanes
    input  wire [16*PE_N*A_NB-1:0]       aw_scale,   // bf16 [128,128] block scales

    // ---- decoder_block_fp8 attention KV-cache read (per-layer; BF16) ----
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,
    input  wire                          kc_valid,

    // ---- decoder_block_fp8 MoE router weight pull (per-layer, W_g col; FP8 + scales) ----
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [8*N_EXPERT-1:0]         rw_col,     // N_EXPERT FP8 E4M3 = W_g[k,*]
    input  wire [16*N_EXPERT*R_NB-1:0]   rw_scale,   // bf16 [128,128] block scales

    // ---- decoder_block_fp8 FFN expert weight pull (per-layer, qualified; FP8 + scales) ----
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [8*TN-1:0]               fw_col,     // GATE/DOWN FP8 E4M3 lanes
    input  wire [8*TN-1:0]               fw_col_up,  // UP companion FP8 E4M3 lanes
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_g, // GATE/DOWN bf16 block scales
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_u, // UP bf16 block scales

    // ---- final RMSNorm gamma pull (BF16, final-norm learned scale) ----
    output wire                          fn_req,     // need a gamma element
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,     // gamma_final[fn_idx] (bf16)

    // ---- LM-head weight pull (combinational, BF16): lm_col[t]=W_lm[lm_k][vtile*LM_TN+t]
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,   // which VOCAB tile (cols group)
    output wire [DIMW-1:0]               lw_k,       // reduction index 0..MODEL_DIM-1
    input  wire [LM_TN*16-1:0]           lw_col      // LM_TN bf16 weight lanes
);
    `include "glm_fp.vh"

    integer ii;

    //========================================================================
    // running hidden state (bf16, MODEL_DIM elements) + packed view for the
    // decoder_block_fp8 / final rmsnorm.  Residual stays BF16 across all layers.
    //========================================================================
    reg [15:0] xcur [0:MODEL_DIM-1];
    reg [MODEL_DIM*16-1:0] xcur_vec;
    always @* begin
        for (ii = 0; ii < MODEL_DIM; ii = ii + 1)
            xcur_vec[16*ii +: 16] = xcur[ii];
    end

    // latched final logits scratch (bf16) before publishing
    reg [15:0] lbuf [0:VOCAB-1];

    //========================================================================
    // ONE decoder_block_fp8 instance (serially reused across the L layers).
    //   All its pulls are forwarded straight to this module's matching ports;
    //   db_layer (driven by the FSM) annotates which layer's weights the system
    //   must answer.  The big linear WEIGHT matmuls inside are FP8 E4M3 + [128,128]
    //   block scales; norms/softmax/rope/residual inside stay bf16.
    //========================================================================
    reg                       db_start;
    reg                       db_mode;       // 0=DENSE, 1=MoE
    wire                      db_busy, db_done;
    wire [MODEL_DIM*16-1:0]   db_y;
    glm_decoder_block_fp8 #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW),
        .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), .BLK(BLK)
    ) u_block (
        .clk(clk), .rst(rst), .start(db_start), .busy(db_busy), .done(db_done),
        .mode(db_mode), .pos(pos), .s_len(s_len),
        .x_vec(xcur_vec), .y_out(db_y),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_col(aw_col), .aw_scale(aw_scale),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col), .rw_scale(rw_scale),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_col(fw_col), .fw_col_up(fw_col_up),
        .fw_scale_g(fw_scale_g), .fw_scale_u(fw_scale_u)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _db_busy_unused = &{1'b0, db_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // FINAL RMSNorm (rmsnorm_unit, LEN=MODEL_DIM, LANES=1).  BF16, NOT FP8
    // (modules_to_not_convert).  It PULLS x (reduce pass) from xcur, then gamma
    // (normalize pass) from the system via fn_*.
    //========================================================================
    reg              fn_start;
    wire             fn_in_req, fn_g_req, fn_y_valid, fn_busy, fn_done;
    wire [15:0]      fn_y_out;
    reg  [15:0]      fn_x_in;
    reg              fn_x_valid;
    reg  [15:0]      fn_gamma_in;
    reg              fn_g_valid;
    rmsnorm_unit #(.LEN(MODEL_DIM), .LANES(1)) u_fnorm (
        .clk(clk), .rst(rst), .start(fn_start),
        .in_req(fn_in_req), .x_in(fn_x_in), .x_valid(fn_x_valid),
        .g_req(fn_g_req), .gamma_in(fn_gamma_in), .g_valid(fn_g_valid),
        .y_valid(fn_y_valid), .y_out(fn_y_out), .busy(fn_busy), .done(fn_done)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _fn_busy_unused = &{1'b0, fn_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // final-norm beat counters (LANES=1 -> beat == element index)
    reg [DIMW:0] fn_ridx;   // reduce read index (x pull)
    reg [DIMW:0] fn_widx;   // normalize write index (y store)
    reg [DIMW:0] fn_gidx;   // gamma pull index

    // gamma pull is COMBINATIONAL (answered same cycle), registered 1 cycle.
    assign fn_req = fn_g_req;
    assign fn_idx = fn_gidx[DIMW-1:0];

    // normalized vector store (fed to the LM head)
    reg [15:0] xn [0:MODEL_DIM-1];

    //========================================================================
    // LM HEAD : glm_matmul_pipe as a 1xLM_TN GEMV tile, K=MODEL_DIM reduction.
    //   BF16 (modules_to_not_convert -- the LM head stays bf16).
    //   A row (M=1) = xN[1,MODEL_DIM] ; W tile (N=LM_TN) = W_lm[k, vtile*LM_TN+.]
    //   We stream MODEL_DIM K-beats; on beat k present a_col = xN[k] (1 lane) and
    //   w_row = lw_col = W_lm[k][vtile group] (LM_TN lanes).  out_valid -> LM_TN
    //   logits for this vtile.  Loop vtile = 0..NVTILE-1.
    //========================================================================
    reg                  mm_start;
    reg                  mm_in_valid;
    reg  [LMKW-1:0]      mm_klen;
    reg  [15:0]          mm_a;            // xN[k] (1 lane)
    wire                 mm_busy, mm_ov;
    wire [16*LM_TN-1:0]  mm_c;            // 1 x LM_TN result tile (bf16)
    glm_matmul_pipe #(.PE_M(1), .PE_N(LM_TN), .KMAX(MODEL_DIM)) u_lm (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_klen),
        .in_valid(mm_in_valid), .a_col(mm_a), .w_row(lw_col),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _mm_busy_unused = &{1'b0, mm_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // LM-head sequencing registers
    reg [VTW-1:0]  vtile;          // current VOCAB tile
    reg [DIMW:0]   lm_k;           // current K beat (0..MODEL_DIM)
    reg            lm_streaming;   // streaming K beats for the current tile
    // lk_present : the K index whose xN is CURRENTLY registered in mm_a (i.e. the
    //   beat the matmul samples THIS cycle).  The weight pull MUST use this index,
    //   not lm_k (which has already advanced), so a_col=xN[k] and w_row=W_lm[k]
    //   are aligned on the same beat.
    reg [DIMW-1:0] lk_present;     // 0..MODEL_DIM-1 (only presented beats latched)
    reg            mm_pres_valid;  // = mm_in_valid, mirrors a presented beat

    assign lw_req   = mm_pres_valid;
    assign lw_vtile = vtile;
    assign lw_k     = lk_present;

    // embedding pull is combinational (answered same cycle), registered 1 cycle.
    reg [DIMW:0] em_ridx;          // embedding load index
    reg          em_loading;
    assign em_req = em_loading;
    assign em_tok = token_id;
    assign em_idx = em_ridx[DIMW-1:0];

    //========================================================================
    // MASTER FSM
    //========================================================================
    localparam [3:0]
        M_IDLE   = 4'd0,
        M_EMBED  = 4'd1,    // load embed(token_id) -> xcur
        M_LAYER  = 4'd2,    // run decoder_block_fp8 for current layer
        M_LWAIT  = 4'd3,    // wait db_done; xcur <= y; advance layer
        M_FNORM  = 4'd4,    // final rmsnorm(xcur) -> xn
        M_LMTILE = 4'd5,    // stream K beats for current vtile
        M_LMWAIT = 4'd6,    // wait mm_ov; store LM_TN logits; next vtile
        M_ARGMAX = 4'd7,    // scan lbuf for argmax (fp32 compare)
        M_DONE   = 4'd8;
    reg [3:0] state;

    reg [LAYW:0]   lcur;           // current layer 0..L
    reg [TOKW:0]   am_i;           // argmax scan index
    reg [31:0]     am_best;        // best logit value (fp32)
    reg [TOKW-1:0] am_arg;

    always @(posedge clk) begin
        if (rst) begin
            state       <= M_IDLE;
            busy        <= 1'b0;
            done        <= 1'b0;
            logits      <= {VOCAB*16{1'b0}};
            argmax      <= {TOKW{1'b0}};
            db_start    <= 1'b0;
            db_mode     <= 1'b0;
            db_layer    <= {LAYW{1'b0}};
            idx_fresh   <= 1'b0;
            idx_win     <= {LAYW{1'b0}};
            fn_start    <= 1'b0;
            fn_x_in     <= 16'h0; fn_x_valid <= 1'b0;
            fn_gamma_in <= 16'h0; fn_g_valid <= 1'b0;
            fn_ridx     <= {(DIMW+1){1'b0}};
            fn_widx     <= {(DIMW+1){1'b0}};
            mm_start    <= 1'b0; mm_in_valid <= 1'b0;
            mm_klen     <= {LMKW{1'b0}}; mm_a <= 16'h0;
            vtile       <= {VTW{1'b0}};
            lm_k        <= {(DIMW+1){1'b0}};
            lm_streaming<= 1'b0;
            lk_present  <= {DIMW{1'b0}};
            mm_pres_valid<= 1'b0;
            em_ridx     <= {(DIMW+1){1'b0}};
            em_loading  <= 1'b0;
            lcur        <= {(LAYW+1){1'b0}};
            am_i        <= {(TOKW+1){1'b0}};
            am_best     <= 32'h0;
            am_arg      <= {TOKW{1'b0}};
            for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                xcur[ii] <= 16'h0; xn[ii] <= 16'h0;
            end
            for (ii=0; ii<VOCAB; ii=ii+1) lbuf[ii] <= 16'h0;
        end else begin
            // ---- default pulse deassert ----
            done     <= 1'b0;
            db_start <= 1'b0;
            fn_start <= 1'b0;
            mm_start <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            M_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy       <= 1'b1;
                    // launch embedding load: pull MODEL_DIM elements of Emb[tok]
                    em_loading <= 1'b1;
                    em_ridx    <= {(DIMW+1){1'b0}};
                    state      <= M_EMBED;
                end
            end
            //---------------------------------------------------------------- embed
            // The embedding pull is combinational: em_req/em_tok/em_idx are driven
            // from em_loading + em_ridx; em_val is answered the SAME cycle and
            // stored on the NEXT edge (1-cycle registered, like every other pull).
            M_EMBED: begin
                if (em_loading) begin
                    xcur[em_ridx[DIMW-1:0]] <= em_val;
                    if (em_ridx == MODEL_DIM[DIMW:0]-1'b1) begin
                        em_loading <= 1'b0;
                        // launch layer 0
                        lcur       <= {(LAYW+1){1'b0}};
                        db_layer   <= {LAYW{1'b0}};
                        db_mode    <= (N_DENSE > 0) ? 1'b0 : 1'b1;
                        idx_fresh  <= (0 % 4 == 3);   // layer 0: reuse window
                        idx_win    <= {LAYW{1'b0}};
                        db_start   <= 1'b1;
                        state      <= M_LAYER;
                    end else begin
                        em_ridx <= em_ridx + 1'b1;
                    end
                end
            end
            //---------------------------------------------------------------- run layer
            M_LAYER: begin
                // db_start was pulsed entering this state (or from M_LWAIT); wait.
                state <= M_LWAIT;
            end
            M_LWAIT: begin
                if (db_done) begin
                    // x_{l+1} = decoder_block_fp8(x_l)
                    for (ii=0; ii<MODEL_DIM; ii=ii+1)
                        xcur[ii] <= db_y[16*ii +: 16];
                    if (lcur == (L[LAYW:0]-1'b1)) begin
                        // last layer done -> final rmsnorm.  (rmsnorm pulls xcur,
                        // which we have just updated; the reduce pass starts next
                        // cycle so the new xcur is in place.)
                        fn_start <= 1'b1;
                        fn_ridx  <= {(DIMW+1){1'b0}};
                        fn_widx  <= {(DIMW+1){1'b0}};
                        state    <= M_FNORM;
                    end else begin
                        // advance to next layer
                        lcur      <= lcur + 1'b1;
                        db_layer  <= lcur[LAYW-1:0] + 1'b1;
                        db_mode   <= ((lcur + 1'b1) < N_DENSE[LAYW:0]) ? 1'b0 : 1'b1;
                        idx_fresh <= ((lcur[1:0] + 2'd1) == 2'd3);
                        idx_win   <= (lcur[LAYW-1:0] + 1'b1) >> 2;
                        db_start  <= 1'b1;
                        state     <= M_LAYER;
                    end
                end
            end
            //---------------------------------------------------------------- final norm
            M_FNORM: begin
                fn_x_valid <= 1'b0; fn_g_valid <= 1'b0;
                if (fn_in_req) begin
                    fn_x_in    <= xcur[fn_ridx[DIMW-1:0]];
                    fn_x_valid <= 1'b1;
                end
                if (fn_g_req) begin
                    fn_gamma_in <= fn_val;      // combinational gamma answer
                    fn_g_valid  <= 1'b1;
                end
                if (fn_y_valid) xn[fn_widx[DIMW-1:0]] <= fn_y_out;
                if (fn_done) begin
                    // launch LM head: first vtile, stream MODEL_DIM K beats.
                    vtile        <= {VTW{1'b0}};
                    mm_klen      <= MODEL_DIM[LMKW-1:0];
                    mm_start     <= 1'b1;
                    lm_streaming <= 1'b1;
                    lm_k         <= {(DIMW+1){1'b0}};
                    mm_in_valid  <= 1'b0;       // first beat presented in M_LMTILE
                    state        <= M_LMTILE;
                end
            end
            //---------------------------------------------------------------- LM head tile stream
            // Present xN[k] as a_col and W_lm[k][vtile] (lw_col) as w_row on each
            // K beat.  lw_req/lw_vtile/lw_k are driven combinationally from
            // lm_streaming/vtile/lm_k so the system answers lw_col the same cycle.
            M_LMTILE: begin
                if (lm_streaming) begin
                    if (lm_k < MODEL_DIM[DIMW:0]) begin
                        // present beat lm_k: register xN[lm_k] into a_col AND latch
                        // lk_present=lm_k so the weight pull (lw_k=lk_present) lines
                        // up with this very beat on the next edge.
                        mm_a          <= xn[lm_k[DIMW-1:0]];
                        lk_present    <= lm_k[DIMW-1:0];
                        mm_in_valid   <= 1'b1;
                        mm_pres_valid <= 1'b1;
                        lm_k          <= lm_k + 1'b1;
                    end else begin
                        mm_in_valid   <= 1'b0;
                        mm_pres_valid <= 1'b0;
                        lm_streaming  <= 1'b0;
                        state         <= M_LMWAIT;
                    end
                end
            end
            //---------------------------------------------------------------- LM head tile wait
            M_LMWAIT: begin
                mm_in_valid   <= 1'b0;
                mm_pres_valid <= 1'b0;
                if (mm_ov) begin
                    // store this tile's LM_TN logits
                    for (ii=0; ii<LM_TN; ii=ii+1)
                        lbuf[vtile*LM_TN + ii] <= mm_c[16*ii +: 16];
                    if (vtile == (NVTILE[VTW-1:0]-1'b1)) begin
                        // all vocab tiles done -> argmax scan
                        am_i    <= {(TOKW+1){1'b0}};
                        am_best <= 32'hFF80_0000;   // -inf (fp32) as starting best
                        am_arg  <= {TOKW{1'b0}};
                        state   <= M_ARGMAX;
                    end else begin
                        // next vtile
                        vtile        <= vtile + 1'b1;
                        mm_klen      <= MODEL_DIM[LMKW-1:0];
                        mm_start     <= 1'b1;
                        lm_streaming <= 1'b1;
                        lm_k         <= {(DIMW+1){1'b0}};
                        state        <= M_LMTILE;
                    end
                end
            end
            //---------------------------------------------------------------- argmax
            // Scan lbuf for the max logit (fp32 compare, lower-index tie-break:
            // strictly-greater keeps the first occurrence).  One element/cycle.
            M_ARGMAX: begin
                if (am_i < VOCAB[TOKW:0]) begin
                    if (fp32_gt(bf16_to_fp32(lbuf[am_i[TOKW-1:0]]), am_best)) begin
                        am_best <= bf16_to_fp32(lbuf[am_i[TOKW-1:0]]);
                        am_arg  <= am_i[TOKW-1:0];
                    end
                    am_i <= am_i + 1'b1;
                end else begin
                    // publish logits + argmax
                    for (ii=0; ii<VOCAB; ii=ii+1)
                        logits[16*ii +: 16] <= lbuf[ii];
                    argmax <= am_arg;
                    state  <= M_DONE;
                end
            end
            //----------------------------------------------------------------
            M_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= M_IDLE;
            end
            default: state <= M_IDLE;
            endcase
        end
    end

    //========================================================================
    // final-norm pull beat counters (mirror the unit's beat order; LANES=1 so
    // beat == element index).  Reset at each fn_start.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            fn_gidx <= {(DIMW+1){1'b0}};
        end else begin
            if (fn_start) begin
                fn_gidx <= {(DIMW+1){1'b0}};
            end else begin
                if (fn_in_req)  fn_ridx <= fn_ridx + 1'b1;
                if (fn_y_valid) fn_widx <= fn_widx + 1'b1;
                if (fn_g_req)   fn_gidx <= fn_gidx + 1'b1;
            end
        end
    end

    //========================================================================
    // fp32 greater-than (strict).  Treats -0 == +0; ignores nan (the LM logits
    // are finite normals).  Used for the argmax compare ONLY.
    //========================================================================
    function automatic fp32_gt(input [31:0] a, input [31:0] b);
        reg sa, sb;
        reg [30:0] ma, mb;
        begin
            sa = a[31]; sb = b[31];
            ma = a[30:0]; mb = b[30:0];
            if (sa != sb) begin
                // different signs: positive is greater, unless both are zero.
                if ((ma == 31'b0) && (mb == 31'b0)) fp32_gt = 1'b0; // +0 vs -0
                else fp32_gt = (sb == 1'b1);                        // a>=0 > b<0
            end else if (sa == 1'b0) begin
                fp32_gt = (ma > mb);                  // both positive
            end else begin
                fp32_gt = (ma < mb);                  // both negative: smaller mag bigger
            end
        end
    endfunction

endmodule
/* verilator lint_on DECLFILENAME */
