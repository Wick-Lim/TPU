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
//   This is the FP8-NATIVE sibling of glm_model.v.  ONLY the per-layer attention/
//   router/expert WEIGHT matmuls are FP8 (E4M3 codes + per-[128,128]-block bf16
//   dequant scales + on-chip dynamic per-token activation->E4M3 quant); the token
//   embedding, the final RMSNorm, the LM-head GEMV and the running residual stay
//   bf16 (modules_to_not_convert).
//
//============================================================================
// PE_M BATCHING (B query tokens decoded in lockstep)              (ULTRA_PERF#2)
//----------------------------------------------------------------------------
//   PE_M (default 1 == byte-identical to the committed single-token forward) is
//   the number of query tokens pushed through the model at once.  The B tokens
//   share the SAME decode step: pos, s_len, the KV cache and ALL weights -- only
//   their token_id (and hence embedding + activations) differ.  The one
//   glm_decoder_block_fp8 instance runs at PE_M, carrying a B-WIDE bf16 residual
//   hidden; ONE weight fetch per (layer, projection, expert) feeds all B rows.
//
//   PER-ROW (replicated B-wide, lockstep):
//     * the running residual hidden xcur[r][.] (bf16);
//     * the embedding load (SERIAL over rows -- the em_* table pull is one
//       row/element at a time, so widening costs em-load cycles but keeps the
//       bf16 em_* interface width PE_M-independent);
//     * the final RMSNorm (PE_M replicated rmsnorm_units, lockstep off ONE shared
//       final-gamma pull);
//     * the LM-head GEMV (glm_matmul_pipe at PE_M: PE_M activation rows stream
//       against ONE shared W_lm column tile -> PE_M*LM_TN logits/pass);
//     * the per-row argmax scan.
//   The weight / cache pull request streams (aw_*/rw_*/fw_*/kc_*/gn_*/fn_*/lw_*)
//   are IDENTICAL to PE_M=1 -- ONE fetch shared by all B rows.
//
//   At PE_M=1 every PE_M-indexed construct constant-folds to the committed
//   single-token datapath (identical ports), so the committed TBs instantiate
//   this unchanged.
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
    // ---- PE_M : query tokens decoded in lockstep (batch B) ----
    parameter integer PE_M       = 1,
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
    parameter integer LMKW       = $clog2(MODEL_DIM + 1),   // matmul_pipe k_len width
    parameter integer ROWCW      = $clog2(PE_M + 1)         // embedding row counter
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    // ---- control ----
    input  wire                          start,      // 1-cycle pulse: begin token
    output reg                           busy,
    output reg                           done,       // 1-cycle pulse: logits valid
    input  wire [PE_M*TOKW-1:0]          token_id,   // PE_M input tokens to embed
    input  wire [POSW-1:0]               pos,        // query position (RoPE) -- SHARED
    input  wire [IDXW:0]                 s_len,      // S causal keys (<= S_MAX) -- SHARED

    // ---- outputs (PE_M rows, row-major) ----
    output reg  [PE_M*VOCAB*16-1:0]      logits,     // PE_M * VOCAB bf16 next-token logits
    output reg  [PE_M*TOKW-1:0]          argmax,     // PE_M arg max logit (next token)

    // ---- embedding table pull (combinational, BF16): em_val = Emb[em_tok][em_idx] ----
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,     // current row's token (for the ROM)
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

    // ---- final RMSNorm gamma pull (BF16, final-norm learned scale) -- SHARED ----
    output wire                          fn_req,     // need a gamma element
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,     // gamma_final[fn_idx] (bf16)

    // ---- LM-head weight pull (combinational, BF16): lm_col[t]=W_lm[lm_k][vtile*LM_TN+t]
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,   // which VOCAB tile (cols group)
    output wire [DIMW-1:0]               lw_k,       // reduction index 0..MODEL_DIM-1
    input  wire [LM_TN*16-1:0]           lw_col,     // LM_TN bf16 weight lanes -- SHARED

    // ---- hidden-state out (BACKWARD-COMPATIBLE, ADDITIVE) -- PE_M rows ----
    //   xN[r] = the final-RMSNorm result of row r (= row r's LM-head input vector),
    //   packed bf16, MODEL_DIM elements/row.  Continuously reflects the xn[] buffer;
    //   STABLE from `done` until the next `start`.  Existing TBs leave it dangling.
    output wire [PE_M*MODEL_DIM*16-1:0]  h_state
);
    `include "glm_fp.vh"

    integer ii;
    integer rr;

    //========================================================================
    // running hidden state (bf16, MODEL_DIM elements) -- PER ROW + packed view for
    // the decoder_block_fp8 / final rmsnorm.  Residual stays BF16 across all layers.
    //========================================================================
    reg [15:0] xcur [0:PE_M-1][0:MODEL_DIM-1];
    reg [PE_M*MODEL_DIM*16-1:0] xcur_vec;
    always @* begin
        for (rr = 0; rr < PE_M; rr = rr + 1)
            for (ii = 0; ii < MODEL_DIM; ii = ii + 1)
                xcur_vec[16*(MODEL_DIM*rr + ii) +: 16] = xcur[rr][ii];
    end

    // latched final logits scratch (bf16) before publishing -- PER ROW
    reg [15:0] lbuf [0:PE_M-1][0:VOCAB-1];

    //========================================================================
    // ONE decoder_block_fp8 instance (serially reused across the L layers), PE_M
    //   wide.  All its pulls are forwarded straight to this module's matching ports
    //   (shared by all B rows); db_layer annotates which layer's weights to answer.
    //========================================================================
    reg                       db_start;
    reg                       db_mode;       // 0=DENSE, 1=MoE
    wire                      db_busy, db_done;
    wire [PE_M*MODEL_DIM*16-1:0] db_y;
    glm_decoder_block_fp8 #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW),
        .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), .BLK(BLK), .PE_M(PE_M)
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
    // FINAL RMSNorm -- PE_M replicated rmsnorm_units, lockstep off ONE shared
    // final-gamma pull (fn_*).  BF16, NOT FP8 (modules_to_not_convert).
    //========================================================================
    reg              fn_start;
    wire [PE_M-1:0]  fn_in_req, fn_g_req, fn_y_valid, fn_busy, fn_done;
    wire [16*PE_M-1:0] fn_y_out;
    reg  [16*PE_M-1:0] fn_x_in;
    reg  [16*PE_M-1:0] fn_gamma_in;
    reg              fn_x_valid;
    reg              fn_g_valid;
    genvar gfn;
    generate
    for (gfn = 0; gfn < PE_M; gfn = gfn + 1) begin : FNORM
        rmsnorm_unit #(.LEN(MODEL_DIM), .LANES(1)) u_fnorm (
            .clk(clk), .rst(rst), .start(fn_start),
            .in_req(fn_in_req[gfn]), .x_in(fn_x_in[16*gfn +: 16]), .x_valid(fn_x_valid),
            .g_req(fn_g_req[gfn]), .gamma_in(fn_gamma_in[16*gfn +: 16]), .g_valid(fn_g_valid),
            .y_valid(fn_y_valid[gfn]), .y_out(fn_y_out[16*gfn +: 16]),
            .busy(fn_busy[gfn]), .done(fn_done[gfn])
        );
    end
    endgenerate
    /* verilator lint_off UNUSEDSIGNAL */
    wire _fn_busy_unused = &{1'b0, fn_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // final-norm beat counters (LANES=1 -> beat == element index).  SHARED across
    // the lockstep units -> instance-0's handshake drives them.
    reg [DIMW:0] fn_ridx;   // reduce read index (x pull)
    reg [DIMW:0] fn_widx;   // normalize write index (y store)
    reg [DIMW:0] fn_gidx;   // gamma pull index

    // gamma pull is COMBINATIONAL (answered same cycle), registered 1 cycle.  SHARED.
    assign fn_req = fn_g_req[0];
    assign fn_idx = fn_gidx[DIMW-1:0];

    // normalized vector store (fed to the LM head) -- PER ROW
    reg [15:0] xn [0:PE_M-1][0:MODEL_DIM-1];

    // ---- ADDITIVE hidden-out: packed view of xn (the final-norm / LM-head input). ----
    reg [PE_M*MODEL_DIM*16-1:0] h_state_r;
    integer hr, hi;
    always @* begin
        for (hr = 0; hr < PE_M; hr = hr + 1)
            for (hi = 0; hi < MODEL_DIM; hi = hi + 1)
                h_state_r[16*(MODEL_DIM*hr + hi) +: 16] = xn[hr][hi];
    end
    assign h_state = h_state_r;

    //========================================================================
    // LM HEAD : glm_matmul_pipe as a PE_M x LM_TN GEMV tile, K=MODEL_DIM reduction.
    //   BF16 (modules_to_not_convert).  PE_M activation rows (xN[r]) stream against
    //   ONE shared W tile (N=LM_TN) = W_lm[k, vtile*LM_TN+.] -> PE_M*LM_TN logits
    //   for this vtile.  Loop vtile = 0..NVTILE-1.  ONE shared lw_* weight stream.
    //========================================================================
    reg                  mm_start;
    reg                  mm_in_valid;
    reg  [LMKW-1:0]      mm_klen;
    reg  [16*PE_M-1:0]   mm_a;            // xN[r][k] (PE_M lanes)
    wire                 mm_busy, mm_ov;
    wire [16*PE_M*LM_TN-1:0] mm_c;        // PE_M x LM_TN result tile (bf16)
    glm_matmul_pipe #(.PE_M(PE_M), .PE_N(LM_TN), .KMAX(MODEL_DIM)) u_lm (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_klen),
        .in_valid(mm_in_valid), .a_col(mm_a), .w_row(lw_col),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _mm_busy_unused = &{1'b0, mm_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // LM-head sequencing registers (SHARED across rows: same K stream, same weights)
    reg [VTW-1:0]  vtile;          // current VOCAB tile
    reg [DIMW:0]   lm_k;           // current K beat (0..MODEL_DIM)
    reg            lm_streaming;   // streaming K beats for the current tile
    reg [DIMW-1:0] lk_present;     // 0..MODEL_DIM-1 (only presented beats latched)
    reg            mm_pres_valid;  // = mm_in_valid, mirrors a presented beat

    assign lw_req   = mm_pres_valid;
    assign lw_vtile = vtile;
    assign lw_k     = lk_present;

    // embedding pull is combinational (answered same cycle), registered 1 cycle.
    // SERIAL over rows: em_row selects which token's embedding is loading.
    reg [ROWCW-1:0] em_row;        // current embedding row 0..PE_M-1
    reg [DIMW:0]    em_ridx;       // embedding load element index
    reg             em_loading;
    assign em_req = em_loading;
    assign em_tok = token_id[TOKW*em_row +: TOKW];
    assign em_idx = em_ridx[DIMW-1:0];

    //========================================================================
    // MASTER FSM
    //========================================================================
    localparam [3:0]
        M_IDLE   = 4'd0,
        M_EMBED  = 4'd1,    // load embed(token_id) -> xcur (serial over rows)
        M_LWAIT  = 4'd3,    // run decoder_block_fp8; wait db_done; xcur <= y; advance
        M_FNORM  = 4'd4,    // final rmsnorm(xcur) -> xn
        M_LMTILE = 4'd5,    // stream K beats for current vtile
        M_LMWAIT = 4'd6,    // wait mm_ov; store LM_TN logits/row; next vtile
        M_ARGMAX = 4'd7,    // scan lbuf for per-row argmax (fp32 compare)
        M_DONE   = 4'd8;
    reg [3:0] state;

    reg [LAYW:0]   lcur;           // current layer 0..L
    reg [TOKW:0]   am_i;           // argmax scan index (SHARED)
    reg [15:0]     am_best [0:PE_M-1];  // best logit value (bf16 raw) per row
    reg [TOKW-1:0] am_arg  [0:PE_M-1];  // argmax per row

    always @(posedge clk) begin
        if (rst) begin
            state       <= M_IDLE;
            busy        <= 1'b0;
            done        <= 1'b0;
            logits      <= {PE_M*VOCAB*16{1'b0}};
            argmax      <= {PE_M*TOKW{1'b0}};
            db_start    <= 1'b0;
            db_mode     <= 1'b0;
            db_layer    <= {LAYW{1'b0}};
            idx_fresh   <= 1'b0;
            idx_win     <= {LAYW{1'b0}};
            fn_start    <= 1'b0;
            fn_x_in     <= {16*PE_M{1'b0}}; fn_x_valid <= 1'b0;
            fn_gamma_in <= {16*PE_M{1'b0}}; fn_g_valid <= 1'b0;
            fn_ridx     <= {(DIMW+1){1'b0}};
            fn_widx     <= {(DIMW+1){1'b0}};
            mm_start    <= 1'b0; mm_in_valid <= 1'b0;
            mm_klen     <= {LMKW{1'b0}}; mm_a <= {16*PE_M{1'b0}};
            vtile       <= {VTW{1'b0}};
            lm_k        <= {(DIMW+1){1'b0}};
            lm_streaming<= 1'b0;
            lk_present  <= {DIMW{1'b0}};
            mm_pres_valid<= 1'b0;
            em_row      <= {ROWCW{1'b0}};
            em_ridx     <= {(DIMW+1){1'b0}};
            em_loading  <= 1'b0;
            lcur        <= {(LAYW+1){1'b0}};
            am_i        <= {(TOKW+1){1'b0}};
            for (rr=0; rr<PE_M; rr=rr+1) begin
                am_best[rr] <= 16'h0;
                am_arg[rr]  <= {TOKW{1'b0}};
                for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                    xcur[rr][ii] <= 16'h0; xn[rr][ii] <= 16'h0;
                end
                for (ii=0; ii<VOCAB; ii=ii+1) lbuf[rr][ii] <= 16'h0;
            end
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
                    // launch embedding load: pull MODEL_DIM elements of Emb[tok],
                    // serial over the PE_M rows starting at row 0.
                    em_loading <= 1'b1;
                    em_row     <= {ROWCW{1'b0}};
                    em_ridx    <= {(DIMW+1){1'b0}};
                    state      <= M_EMBED;
                end
            end
            //---------------------------------------------------------------- embed
            // The embedding pull is combinational: em_req/em_tok/em_idx are driven
            // from em_loading + em_row + em_ridx; em_val answered the SAME cycle and
            // stored on the NEXT edge.  Rows are loaded serially (em_row).
            M_EMBED: begin
                if (em_loading) begin
                    xcur[em_row[ROWCW-1:0]][em_ridx[DIMW-1:0]] <= em_val;
                    if (em_ridx == MODEL_DIM[DIMW:0]-1'b1) begin
                        if (em_row == (PE_M[ROWCW-1:0]-1'b1)) begin
                            em_loading <= 1'b0;
                            // launch layer 0
                            lcur       <= {(LAYW+1){1'b0}};
                            db_layer   <= {LAYW{1'b0}};
                            db_mode    <= (N_DENSE > 0) ? 1'b0 : 1'b1;
                            idx_fresh  <= (0 % 4 == 3);   // layer 0: reuse window
                            idx_win    <= {LAYW{1'b0}};
                            db_start   <= 1'b1;
                            state      <= M_LWAIT;
                        end else begin
                            em_row  <= em_row + 1'b1;
                            em_ridx <= {(DIMW+1){1'b0}};
                        end
                    end else begin
                        em_ridx <= em_ridx + 1'b1;
                    end
                end
            end
            //---------------------------------------------------------------- run layer / wait
            M_LWAIT: begin
                if (db_done) begin
                    // x_{l+1} = decoder_block_fp8(x_l)  (per row)
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (ii=0; ii<MODEL_DIM; ii=ii+1)
                            xcur[rr][ii] <= db_y[16*(MODEL_DIM*rr + ii) +: 16];
                    if (lcur == (L[LAYW:0]-1'b1)) begin
                        // last layer done -> final rmsnorm
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
                        state     <= M_LWAIT;
                    end
                end
            end
            //---------------------------------------------------------------- final norm
            M_FNORM: begin
                fn_x_valid <= 1'b0; fn_g_valid <= 1'b0;
                if (fn_in_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        fn_x_in[16*rr +: 16] <= xcur[rr][fn_ridx[DIMW-1:0]];
                    fn_x_valid <= 1'b1;
                end
                if (fn_g_req[0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        fn_gamma_in[16*rr +: 16] <= fn_val;     // SAME gamma to every row
                    fn_g_valid  <= 1'b1;
                end
                if (fn_y_valid[0])
                    for (rr=0; rr<PE_M; rr=rr+1)
                        xn[rr][fn_widx[DIMW-1:0]] <= fn_y_out[16*rr +: 16];
                if (fn_done[0]) begin
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
            M_LMTILE: begin
                if (lm_streaming) begin
                    if (lm_k < MODEL_DIM[DIMW:0]) begin
                        // present beat lm_k: register xN[r][lm_k] into a_col (PE_M
                        // lanes) AND latch lk_present so the shared weight pull lines
                        // up with this very beat on the next edge.
                        for (rr=0; rr<PE_M; rr=rr+1)
                            mm_a[16*rr +: 16] <= xn[rr][lm_k[DIMW-1:0]];
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
                    // store this tile's LM_TN logits PER ROW
                    for (rr=0; rr<PE_M; rr=rr+1)
                        for (ii=0; ii<LM_TN; ii=ii+1)
                            lbuf[rr][vtile*LM_TN + ii] <= mm_c[16*(rr*LM_TN + ii) +: 16];
                    if (vtile == (NVTILE[VTW-1:0]-1'b1)) begin
                        // all vocab tiles done -> argmax scan
                        am_i    <= {(TOKW+1){1'b0}};
                        for (rr=0; rr<PE_M; rr=rr+1) begin
                            am_best[rr] <= 16'hFF80;        // -inf (bf16) start
                            am_arg[rr]  <= {TOKW{1'b0}};
                        end
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
            //---------------------------------------------------------------- argmax (per row)
            // Scan lbuf for the max logit (fp32 compare, lower-index tie-break:
            // strictly-greater keeps the first occurrence).  One element/cycle, all
            // PE_M rows in lockstep.
            M_ARGMAX: begin
                if (am_i < VOCAB[TOKW:0]) begin
                    for (rr=0; rr<PE_M; rr=rr+1)
                        if (bf16_gt(lbuf[rr][am_i[TOKW-1:0]], am_best[rr])) begin
                            am_best[rr] <= lbuf[rr][am_i[TOKW-1:0]];
                            am_arg[rr]  <= am_i[TOKW-1:0];
                        end
                    am_i <= am_i + 1'b1;
                end else begin
                    // publish logits + argmax (per row)
                    for (rr=0; rr<PE_M; rr=rr+1) begin
                        for (ii=0; ii<VOCAB; ii=ii+1)
                            logits[16*(VOCAB*rr + ii) +: 16] <= lbuf[rr][ii];
                        argmax[TOKW*rr +: TOKW] <= am_arg[rr];
                    end
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
    // beat == element index).  Reset at each fn_start.  SHARED (lockstep units).
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            fn_gidx <= {(DIMW+1){1'b0}};
        end else begin
            if (fn_start) begin
                fn_gidx <= {(DIMW+1){1'b0}};
            end else begin
                if (fn_in_req[0])  fn_ridx <= fn_ridx + 1'b1;
                if (fn_y_valid[0]) fn_widx <= fn_widx + 1'b1;
                if (fn_g_req[0])   fn_gidx <= fn_gidx + 1'b1;
            end
        end
    end

    //========================================================================
    // bf16 greater-than (strict).  Used for the argmax compare ONLY.  Bit-for-bit
    // the old fp32_gt(bf16_to_fp32(a),bf16_to_fp32(b)).
    //========================================================================
    function automatic bf16_gt(input [15:0] a, input [15:0] b);
        reg sa, sb;
        reg [14:0] ma, mb;
        begin
            sa = a[15]; sb = b[15];
            ma = a[14:0]; mb = b[14:0];
            if (sa != sb) begin
                if ((ma == 15'b0) && (mb == 15'b0)) bf16_gt = 1'b0; // +0 vs -0
                else bf16_gt = (sb == 1'b1);                        // a>=0 > b<0
            end else if (sa == 1'b0) begin
                bf16_gt = (ma > mb);                  // both positive
            end else begin
                bf16_gt = (ma < mb);                  // both negative
            end
        end
    endfunction

endmodule
/* verilator lint_on DECLFILENAME */
