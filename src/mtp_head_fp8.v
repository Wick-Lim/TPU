`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// mtp_head_fp8.v  --  GLM-5.2-FP8 Multi-Token Prediction head (the FP8-NATIVE
//                     sibling of mtp_head.v).  num_nextn_predict_layers=1.
//----------------------------------------------------------------------------
// FUNCTION  (identical math + FSM + flow to mtp_head.v; the ONLY change is that
//   the big WEIGHT matmuls run in the OFFICIAL zai-org/GLM-5.2-FP8 numerics so
//   the published FP8 checkpoint runs with NO re-quantization)
//
//     a    = RMSNorm(h_t)                          // bf16 (modules_to_not_convert)
//     b    = RMSNorm(embed(tok_{t+1}))             // bf16
//     cat  = [ a ; b ]                              // concat -> 2*MODEL_DIM (bf16)
//     h'   = W_proj @ cat                           // *** FP8 *** glm_matmul_fp8
//     y    = decoder_block_fp8( h' , pos , kv )     // *** FP8 *** one GLM-5.2 layer
//     xN   = RMSNorm_final( y )                     // bf16 shared final norm
//     logits[V] = W_lm[V,MODEL_DIM] . xN            // *** bf16 *** glm_matmul_pipe
//     argmax    = arg max_v logits[v]               // speculative next-next token
//
//   FP8 SPLIT (GLM-5.2-FP8 modules_to_not_convert preserved -- byte-for-byte the
//   same boundary glm_decoder_block_fp8.v draws):
//     * FP8 (E4M3 weights + per-[128,128]-block bf16 dequant scales + on-chip
//       dynamic per-token activation->E4M3 quant):
//         - the W_proj combine projection  (glm_matmul_fp8)
//         - everything inside the decoder layer's big linears (glm_decoder_block_fp8)
//     * bf16 (UNCHANGED):
//         - the THREE RMSNorms (a, b, final)        -- rmsnorm_unit (bf16)
//         - the residual stream + softmax/rope tails inside the decoder block
//         - the LM head GEMV                          -- glm_matmul_pipe (bf16).
//           The GLM-5.2-FP8 LM head ("lm_head") is in modules_to_not_convert and
//           stays bf16, so it routes through the bf16 matmul exactly as mtp_head.v.
//
//   PURE ORCHESTRATOR -- REIMPLEMENTS NO ARITHMETIC.  Same serial-reuse discipline
//   as mtp_head.v:
//     * ONE rmsnorm_unit reused for the 3 norms (cn_which selects the gamma).
//     * ONE glm_matmul_fp8(PE_M=1,PE_N=PROJ_TN,KMAX=2*MODEL_DIM,BLK) as the FP8
//       combine projection, walked over MODEL_DIM/PROJ_TN output tiles.  The
//       activation is the bf16 concat `cat`, dynamically quantized to E4M3 by a
//       per-vector pow2 shift (csh) reduced combinationally from cat's exponents
//       -- the SAME a_shift discipline swiglu_expert_fp8 / mla_attn_fp8 use.
//     * ONE glm_decoder_block_fp8 (the verified FP8 layer) run ONCE on h'.
//     * ONE glm_matmul_pipe(PE_M=1,PE_N=LM_TN,KMAX=MODEL_DIM) bf16 LM-head GEMV.
//     * a 1-elt/cycle argmax scan -- identical to mtp_head.v's tail.
//
//----------------------------------------------------------------------------
// DYNAMIC ACTIVATION QUANT for the W_proj projection (activation_scheme=dynamic)
//   glm_matmul_fp8 takes a per-row signed-8 pow2 exponent a_shift and prescales
//   the bf16 activation by 2^a_shift before the E4M3 encode (undoing 2^-a_shift at
//   dequant).  PE_M=1 here, so ONE a_shift for the whole `cat` vector:
//       csh = clamp_{[-128,127]}( 134 - emax )      (emax = max bf16 exp field)
//   so the max element's scaled exponent lands at 7, giving E4M3 maximum mantissa
//   headroom before underflow.  emax==0 (all-zero cat) -> csh = 0.  cbuf (= cat)
//   is fully populated by the time the projection streams, so csh is a stable
//   combinational reduction over cbuf, latched into the matmul at each tile start.
//
//----------------------------------------------------------------------------
// WEIGHT PULL INTERFACE  (vs mtp_head.v: the FP8 matmuls now pull FP8 codes +
//   [128,128] block scales; the bf16 LM head + the three gammas are UNCHANGED)
//   * combine/final RMSNorm gamma (cn_*)   : bf16, UNCHANGED.
//   * decoder RMSNorm gamma (gn_*)         : bf16, UNCHANGED.
//   * W_proj weight pull (pw_*)            : pw_col = PROJ_TN FP8 E4M3 lanes (was
//                                            bf16); pw_scale = bf16 [128,128] block
//                                            scales for the addressed pw_ptile.
//   * decoder attention weight pull (aw_*) : aw_col FP8 + aw_scale block scales.
//   * decoder cache read (kc_*)            : bf16 latent KV, UNCHANGED.
//   * decoder router weight pull (rw_*)    : rw_col FP8 + rw_scale block scales.
//   * decoder FFN expert weight pull (fw_*): fw_col/fw_col_up FP8 + fw_scale_g/_u.
//   * shared LM-head weight pull (lw_*)    : lw_col bf16, UNCHANGED (bf16 LM head).
//   FP8 codes are pulled per K-beat (qualified by the *_req + index); block scales
//   are answered from the current tile/expert selector (latched at the matmul's
//   start cycle, exactly as glm_decoder_block_fp8 / swiglu_expert_fp8 expect).
//
//----------------------------------------------------------------------------
// LATENCY  (deterministic; handshake-driven so each FP8 leaf unit's own latency
//   is absorbed via its busy/out_valid -- structurally identical to mtp_head.v,
//   only the per-tile matmul latency term changes from the bf16 to the FP8 pipe):
//   L_mtp = 2*L_rmsnorm(MODEL_DIM)
//         + (MODEL_DIM/PROJ_TN) * L_matmul_fp8(K=2*MODEL_DIM, PE_N=PROJ_TN)
//         + L_decoder_block_fp8(params, S)
//         + L_rmsnorm(MODEL_DIM)
//         + (VOCAB/LM_TN) * L_matmul_pipe(K=MODEL_DIM, PE_N=LM_TN)   // bf16 LM head
//         + VOCAB                                                    // argmax scan
//   No data-dependent stall; sync active-high reset; no latch; no comb loop (all
//   feedback rides the rmsnorm / decoder_block / matmul pipeline registers).
//============================================================================
module mtp_head_fp8 #(
    // ---- model / slice config (small-but-faithful, ACCEL_GLM52 §8.1) ----
    parameter integer MODEL_DIM  = 128,
    parameter integer VOCAB      = 256,
    // ---- decoder_block slice params (passed straight through) ----
    parameter integer H_HEADS    = 4,
    parameter integer NOPE       = 16,
    parameter integer ROPE       = 16,
    parameter integer V_DIM      = 32,
    parameter integer Q_LORA     = 64,
    parameter integer KV_LORA    = 32,
    parameter integer S_MAX      = 8,
    parameter integer TOPK_ATTN  = 8,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 4,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 8,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 64,
    parameter integer INTER_DENSE= 256,
    parameter [31:0]  RSCALE     = 32'h40200000,// 2.5 fp32
    parameter integer TN         = 4,
    // ---- FP8 weight block size -- DeepSeek-V3 / GLM-5.2-FP8 weight_block_size=[128,128]
    parameter integer BLK        = 128,
    // ---- GEMV tile widths.  VOCAB % LM_TN == 0 ; MODEL_DIM % PROJ_TN == 0. ----
    parameter integer LM_TN      = 4,           // LM-head VOCAB cols/pass
    parameter integer PROJ_TN    = 4,           // combine-proj output cols/pass
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
    // head-level derived
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    parameter integer LMKW       = $clog2(MODEL_DIM + 1),    // LM matmul k_len width
    // combine-projection derived
    parameter integer CK         = 2 * MODEL_DIM,            // concat length (K)
    parameter integer CKIW       = $clog2(CK),               // concat index width
    parameter integer PKW        = $clog2(CK + 1),           // proj matmul k_len width
    parameter integer NPTILE     = MODEL_DIM / PROJ_TN,      // proj output tiles
    parameter integer PTW        = (NPTILE <= 1) ? 1 : $clog2(NPTILE),
    parameter integer PROJ_NB    = (CK + BLK - 1) / BLK      // proj FP8 K-block scales
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    // ---- control ----
    input  wire                          start,      // 1-cycle pulse: begin
    output reg                           busy,
    output reg                           done,       // 1-cycle pulse: logits valid
    input  wire                          mode,       // 0=DENSE FFN, 1=MoE FFN (block)
    input  wire [POSW-1:0]               pos,        // query position t (RoPE)
    input  wire [IDXW:0]                 s_len,      // S causal keys (<= S_MAX)

    // ---- data in (bf16) ----
    input  wire [MODEL_DIM*16-1:0]       h_t,        // main-model hidden state @ t
    input  wire [MODEL_DIM*16-1:0]       emb_t1,     // embedding of predicted tok t+1

    // ---- outputs ----
    output reg  [VOCAB*16-1:0]           logits,     // VOCAB bf16 t+2 logits
    output reg  [TOKW-1:0]               argmax,     // arg max logit (spec. t+2 token)

    // ---- combine/final RMSNorm gamma pull (cn_which: 0=h_t,1=emb,2=final) ----
    output wire                          cn_req,
    output wire [1:0]                    cn_which,
    output wire [DIMW-1:0]               cn_idx,
    input  wire [15:0]                   cn_val,

    // ---- FP8 combine-projection weight pull ----
    //   pw_col[t]   = FP8 E4M3 W_proj[ptile*PROJ_TN+t][pw_k]
    //   pw_scale    = bf16 [128,128] block scales for the pw_ptile output tile,
    //                 packed pw_scale[16*(bj*PROJ_TN + t) +: 16]  (bj = K-block).
    output wire                          pw_req,
    output wire [PTW-1:0]                pw_ptile,   // which MODEL_DIM output tile
    output wire [CKIW-1:0]               pw_k,       // concat reduction index 0..2*MD-1
    input  wire [PROJ_TN*8-1:0]          pw_col,     // PROJ_TN FP8 E4M3 weight lanes
    input  wire [16*PROJ_TN*PROJ_NB-1:0] pw_scale,   // bf16 [128,128] block scales

    // ---- decoder_block RMSNorm gamma pull (pre-attn/pre-FFN) ----
    output wire                          gn_req,
    output wire                          gn_which,   // 0=pre-attn, 1=pre-FFN
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,

    // ---- decoder_block attention weight pull (FP8 codes + block scales) ----
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*8-1:0]             aw_col,     // PE_N FP8 E4M3 weight lanes
    input  wire [16*PE_N*A_NB-1:0]       aw_scale,   // bf16 [128,128] block scales

    // ---- decoder_block attention KV-cache read ----
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,
    input  wire                          kc_valid,

    // ---- decoder_block MoE router weight pull (W_g column; FP8 codes + scales) ----
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [8*N_EXPERT-1:0]         rw_col,     // N_EXPERT FP8 E4M3 = W_g[k,*]
    input  wire [16*N_EXPERT*R_NB-1:0]   rw_scale,   // bf16 [128,128] block scales

    // ---- decoder_block FFN expert weight pull (qualified; FP8 codes + scales) ----
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

    // ---- shared LM-head weight pull (bf16, UNCHANGED): lw_col[t]=W_lm[vtile*LM_TN+t][lw_k]
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col
);
    `include "glm_fp.vh"
    `include "fp8_e4m3.vh"

    integer ii;

    //========================================================================
    // latched inputs + working buffers (bf16)
    //========================================================================
    reg [15:0] hbuf [0:MODEL_DIM-1];   // latched h_t  (RMSNorm source, phase 0)
    reg [15:0] ebuf [0:MODEL_DIM-1];   // latched emb  (RMSNorm source, phase 1)
    reg [15:0] cbuf [0:CK-1];          // concat [a;b]  (= RMSNorm outputs)  (FP8 proj act)
    reg [15:0] hprime [0:MODEL_DIM-1]; // h' = W_proj @ cat  (decoder block input)
    reg [15:0] xcur   [0:MODEL_DIM-1]; // decoder-block output y (final-norm source)
    reg [15:0] xn     [0:MODEL_DIM-1]; // final-normed (LM-head input)
    reg [15:0] lbuf   [0:VOCAB-1];     // LM-head logits scratch (bf16)

    reg            mode_q;
    reg [POSW-1:0] pos_q;
    reg [IDXW:0]   slen_q;

    // packed view of h' for the decoder block's wide x_vec port
    reg [MODEL_DIM*16-1:0] hp_vec;
    always @* begin
        for (ii = 0; ii < MODEL_DIM; ii = ii + 1)
            hp_vec[16*ii +: 16] = hprime[ii];
    end

    //========================================================================
    // ONE rmsnorm_unit (LEN=MODEL_DIM, LANES=1) reused for the 3 norms.  bf16,
    //   UNCHANGED from mtp_head.v (modules_to_not_convert).  It PULLS x (reduce
    //   pass) from the phase-selected source, then gamma (normalize pass) via cn_*
    //   (cn_which = phase tells the system which gamma).
    //========================================================================
    reg              cn_start;
    reg  [1:0]       cn_phase;          // 0=norm(h_t),1=norm(emb),2=final norm
    wire             cn_in_req, cn_g_req, cn_y_valid, cn_busy, cn_done;
    wire [15:0]      cn_y_out;
    reg  [15:0]      cn_x_in;
    reg              cn_x_valid;
    reg  [15:0]      cn_gamma_in;
    reg              cn_g_valid;
    rmsnorm_unit #(.LEN(MODEL_DIM), .LANES(1)) u_norm (
        .clk(clk), .rst(rst), .start(cn_start),
        .in_req(cn_in_req), .x_in(cn_x_in), .x_valid(cn_x_valid),
        .g_req(cn_g_req), .gamma_in(cn_gamma_in), .g_valid(cn_g_valid),
        .y_valid(cn_y_valid), .y_out(cn_y_out), .busy(cn_busy), .done(cn_done)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _cn_busy_unused = &{1'b0, cn_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // norm beat counters (LANES=1 -> beat == element index).  Reset at cn_start.
    reg [DIMW:0] cn_ridx;   // reduce read index (x pull)
    reg [DIMW:0] cn_widx;   // normalize write index (y store)
    reg [DIMW:0] cn_gidx;   // gamma pull index

    // gamma pull is COMBINATIONAL (answered same cycle), registered 1 cycle.
    assign cn_req   = cn_g_req;
    assign cn_which = cn_phase;
    assign cn_idx   = cn_gidx[DIMW-1:0];

    //========================================================================
    // Dynamic per-vector pow2 activation scale for the W_proj projection.
    //   The activation is the bf16 concat `cat` (cbuf).  csh = clamp(134-emax)
    //   so the max element's scaled exponent lands at 7 (max E4M3 headroom).
    //   emax==0 (all-zero cat) -> csh = 0.  PE_M=1 -> ONE shift for the vector.
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

    // Running exponent-max over cbuf, updated as the concat is written during
    // S_NORM (phases 0/1).  cbuf is fully populated before the projection
    // streams, so cemax_r is bit-identical to a combinational max over cbuf,
    // but this replaces the CK-deep (256-element) sequential fmax chain --
    // the design's longest path -- with a single 8-bit compare per write.
    // (max is associative -> identical result.)
    reg [7:0] cemax_r;
    wire signed [7:0] csh = dyn_shift(cemax_r);   // proj activation pow2 shift

    //========================================================================
    // ONE glm_decoder_block_fp8 (the verified FP8 GLM-5.2 layer) run ONCE on h'.
    //   All FP8 weight / cache pulls forwarded straight out.
    //========================================================================
    reg                       db_start;
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
        .mode(mode_q), .pos(pos_q), .s_len(slen_q),
        .x_vec(hp_vec), .y_out(db_y),
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
    // FP8 COMBINE-PROJECTION GEMV : glm_matmul_fp8 as a 1xPROJ_TN tile, K=2*MD,
    //   [128,128] block scales.  A row (M=1) = cat[1,2*MODEL_DIM] (bf16, dynamic
    //   E4M3 via csh) ; W tile (N=PROJ_TN) = W_proj[ptile..][k] (FP8 E4M3) +
    //   pw_scale block scales.  On beat k present pp_a = cat[k] and pw_col = W col.
    //========================================================================
    reg                  pp_start;
    reg                  pp_in_valid;
    reg  [PKW-1:0]       pp_klen;
    reg  [15:0]          pp_a;            // cat[k] (1 bf16 lane, drop-in activation)
    wire                 pp_busy, pp_ov;
    wire [16*PROJ_TN-1:0] pp_c;           // 1 x PROJ_TN result tile (bf16)
    glm_matmul_fp8 #(.PE_M(1), .PE_N(PROJ_TN), .KMAX(CK), .BLK(BLK)) u_proj (
        .clk(clk), .rst(rst), .start(pp_start), .k_len(pp_klen),
        .in_valid(pp_in_valid), .a_col(pp_a), .w_row(pw_col),
        .a_shift(csh), .w_scale(pw_scale),
        .busy(pp_busy), .out_valid(pp_ov), .c_out(pp_c)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _pp_busy_unused = &{1'b0, pp_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // projection sequencing
    reg [PTW-1:0]   ptile;          // current output tile
    reg [CKIW:0]    pk;             // current K beat (0..CK)
    reg             pp_streaming;
    reg [CKIW-1:0]  pk_present;     // K index currently registered in pp_a
    reg             pp_pres_valid;  // mirrors a presented beat (weight pull qualifier)

    assign pw_req   = pp_pres_valid;
    assign pw_ptile = ptile;        // also selects pw_scale (block scales for tile)
    assign pw_k     = pk_present;

    //========================================================================
    // SHARED LM-HEAD GEMV : glm_matmul_pipe as a 1xLM_TN tile, K=MODEL_DIM.
    //   bf16 (GLM-5.2-FP8 lm_head is in modules_to_not_convert -> stays bf16),
    //   identical to mtp_head.v's tail.
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

    reg [VTW-1:0]  vtile;          // current VOCAB tile
    reg [DIMW:0]   lm_k;           // current K beat (0..MODEL_DIM)
    reg            lm_streaming;
    reg [DIMW-1:0] lk_present;     // K index currently registered in mm_a
    reg            mm_pres_valid;  // mirrors a presented beat

    assign lw_req   = mm_pres_valid;
    assign lw_vtile = vtile;
    assign lw_k     = lk_present;

    //========================================================================
    // MASTER FSM  (byte-for-byte the same control as mtp_head.v)
    //========================================================================
    localparam [3:0]
        S_IDLE   = 4'd0,
        S_NORM   = 4'd1,    // run rmsnorm pass (phase 0/1/2)
        S_PROJ   = 4'd2,    // stream K beats of combine projection (current ptile)
        S_PROJW  = 4'd3,    // wait pp_ov; store PROJ_TN h' elts; next ptile / block
        S_DBW    = 4'd5,    // db_start pulsed in S_PROJW; wait db_done; xcur<=y; final norm
        S_LMTILE = 4'd6,    // stream K beats for current vtile
        S_LMWAIT = 4'd7,    // wait mm_ov; store LM_TN logits; next vtile
        S_ARGMAX = 4'd8,    // scan lbuf for argmax (fp32 compare)
        S_DONE   = 4'd9;
    reg [3:0] state;

    reg [TOKW:0]   am_i;           // argmax scan index
    reg [15:0]     am_best;        // best logit value (bf16; direct magnitude compare)
    reg [TOKW-1:0] am_arg;

    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            logits       <= {VOCAB*16{1'b0}};
            argmax       <= {TOKW{1'b0}};
            mode_q       <= 1'b0;
            pos_q        <= {POSW{1'b0}};
            slen_q       <= {(IDXW+1){1'b0}};
            cn_start     <= 1'b0; cn_phase <= 2'd0;
            cn_x_in      <= 16'h0; cn_x_valid <= 1'b0;
            cn_gamma_in  <= 16'h0; cn_g_valid <= 1'b0;
            db_start     <= 1'b0;
            pp_start     <= 1'b0; pp_in_valid <= 1'b0;
            pp_klen      <= {PKW{1'b0}}; pp_a <= 16'h0;
            ptile        <= {PTW{1'b0}};
            pk           <= {(CKIW+1){1'b0}};
            pp_streaming <= 1'b0;
            pk_present   <= {CKIW{1'b0}};
            pp_pres_valid<= 1'b0;
            mm_start     <= 1'b0; mm_in_valid <= 1'b0;
            mm_klen      <= {LMKW{1'b0}}; mm_a <= 16'h0;
            vtile        <= {VTW{1'b0}};
            lm_k         <= {(DIMW+1){1'b0}};
            lm_streaming <= 1'b0;
            lk_present   <= {DIMW{1'b0}};
            mm_pres_valid<= 1'b0;
            am_i         <= {(TOKW+1){1'b0}};
            am_best      <= 16'h0;
            am_arg       <= {TOKW{1'b0}};
            cemax_r      <= 8'd0;
            for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                hbuf[ii] <= 16'h0; ebuf[ii] <= 16'h0;
                hprime[ii] <= 16'h0; xcur[ii] <= 16'h0; xn[ii] <= 16'h0;
            end
            for (ii=0; ii<CK; ii=ii+1)    cbuf[ii] <= 16'h0;
            for (ii=0; ii<VOCAB; ii=ii+1) lbuf[ii] <= 16'h0;
        end else begin
            // ---- default pulse deassert ----
            done     <= 1'b0;
            cn_start <= 1'b0;
            db_start <= 1'b0;
            pp_start <= 1'b0;
            mm_start <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy   <= 1'b1;
                    mode_q <= mode;
                    pos_q  <= pos;
                    slen_q <= s_len;
                    for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                        hbuf[ii] <= h_t   [16*ii +: 16];
                        ebuf[ii] <= emb_t1[16*ii +: 16];
                    end
                    // launch RMSNorm(h_t) : phase 0 (reduce source = hbuf)
                    cemax_r  <= 8'd0;        // fresh running exponent-max for cbuf
                    cn_phase <= 2'd0;
                    cn_start <= 1'b1;
                    state    <= S_NORM;
                end
            end
            //---------------------------------------------------------------- rmsnorm pass
            // Reduce pass: answer cn_in_req from the phase source.  Normalize pass:
            // answer cn_g_req with the combinational gamma (registered 1 cycle).
            // Store y to the phase destination: phase0 -> cbuf[0..MD-1] (=a),
            // phase1 -> cbuf[MD..2MD-1] (=b), phase2 -> xn (final-normed).
            S_NORM: begin
                cn_x_valid <= 1'b0; cn_g_valid <= 1'b0;
                if (cn_in_req) begin
                    case (cn_phase)
                        2'd0:    cn_x_in <= hbuf[cn_ridx[DIMW-1:0]];
                        2'd1:    cn_x_in <= ebuf[cn_ridx[DIMW-1:0]];
                        default: cn_x_in <= xcur[cn_ridx[DIMW-1:0]];
                    endcase
                    cn_x_valid <= 1'b1;
                end
                if (cn_g_req) begin
                    cn_gamma_in <= cn_val;      // combinational gamma answer
                    cn_g_valid  <= 1'b1;
                end
                if (cn_y_valid) begin
                    case (cn_phase)
                        2'd0:    cbuf[cn_widx[CKIW-1:0]]                            <= cn_y_out;
                        2'd1:    cbuf[MODEL_DIM[CKIW-1:0] + cn_widx[CKIW-1:0]]      <= cn_y_out;
                        default: xn[cn_widx[DIMW-1:0]]                              <= cn_y_out;
                    endcase
                    // accumulate running exponent-max over the concat cbuf
                    // (phases 0/1 only; phase 2 writes xn, not cbuf).
                    if (!cn_phase[1] && (cn_y_out[14:7] > cemax_r))
                        cemax_r <= cn_y_out[14:7];
                end
                if (cn_done) begin
                    if (cn_phase == 2'd0) begin
                        // a done -> run RMSNorm(emb) : phase 1
                        cn_phase <= 2'd1;
                        cn_start <= 1'b1;
                        state    <= S_NORM;
                    end else if (cn_phase == 2'd1) begin
                        // concat ready -> launch FP8 combine projection (tile 0)
                        ptile        <= {PTW{1'b0}};
                        pp_klen      <= CK[PKW-1:0];
                        pp_start     <= 1'b1;
                        pp_streaming <= 1'b1;
                        pk           <= {(CKIW+1){1'b0}};
                        pp_in_valid  <= 1'b0;       // first beat presented in S_PROJ
                        state        <= S_PROJ;
                    end else begin
                        // final norm done -> launch LM head (vtile 0)
                        vtile        <= {VTW{1'b0}};
                        mm_klen      <= MODEL_DIM[LMKW-1:0];
                        mm_start     <= 1'b1;
                        lm_streaming <= 1'b1;
                        lm_k         <= {(DIMW+1){1'b0}};
                        mm_in_valid  <= 1'b0;       // first beat presented in S_LMTILE
                        state        <= S_LMTILE;
                    end
                end
            end
            //---------------------------------------------------------------- proj tile stream
            // Present cat[k] as a_col and FP8 W_proj[ptile][k] (pw_col) as w_row
            // each K beat.  pk_present latches the beat index so the weight pull
            // aligns.  pw_scale + csh (a_shift) were latched into u_proj at start.
            S_PROJ: begin
                if (pp_streaming) begin
                    if (pk < CK[CKIW:0]) begin
                        pp_a          <= cbuf[pk[CKIW-1:0]];
                        pk_present    <= pk[CKIW-1:0];
                        pp_in_valid   <= 1'b1;
                        pp_pres_valid <= 1'b1;
                        pk            <= pk + 1'b1;
                    end else begin
                        pp_in_valid   <= 1'b0;
                        pp_pres_valid <= 1'b0;
                        pp_streaming  <= 1'b0;
                        state         <= S_PROJW;
                    end
                end
            end
            //---------------------------------------------------------------- proj tile wait
            S_PROJW: begin
                pp_in_valid   <= 1'b0;
                pp_pres_valid <= 1'b0;
                if (pp_ov) begin
                    for (ii=0; ii<PROJ_TN; ii=ii+1)
                        hprime[ptile*PROJ_TN + ii] <= pp_c[16*ii +: 16];
                    if (ptile == (NPTILE[PTW-1:0]-1'b1)) begin
                        // all output tiles done -> run the FP8 decoder block on h'
                        // (db_start pulses here; go straight to the wait state)
                        db_start <= 1'b1;
                        state    <= S_DBW;
                    end else begin
                        ptile        <= ptile + 1'b1;
                        pp_klen      <= CK[PKW-1:0];
                        pp_start     <= 1'b1;
                        pp_streaming <= 1'b1;
                        pk           <= {(CKIW+1){1'b0}};
                        state        <= S_PROJ;
                    end
                end
            end
            //---------------------------------------------------------------- decoder block
            S_DBW: begin
                if (db_done) begin
                    for (ii=0; ii<MODEL_DIM; ii=ii+1)
                        xcur[ii] <= db_y[16*ii +: 16];
                    // launch final rmsnorm over xcur : phase 2.  (xcur is updated
                    // this edge; the reduce pass starts next cycle so it is in place.)
                    cn_phase <= 2'd2;
                    cn_start <= 1'b1;
                    state    <= S_NORM;
                end
            end
            //---------------------------------------------------------------- LM head tile stream
            S_LMTILE: begin
                if (lm_streaming) begin
                    if (lm_k < MODEL_DIM[DIMW:0]) begin
                        mm_a          <= xn[lm_k[DIMW-1:0]];
                        lk_present    <= lm_k[DIMW-1:0];
                        mm_in_valid   <= 1'b1;
                        mm_pres_valid <= 1'b1;
                        lm_k          <= lm_k + 1'b1;
                    end else begin
                        mm_in_valid   <= 1'b0;
                        mm_pres_valid <= 1'b0;
                        lm_streaming  <= 1'b0;
                        state         <= S_LMWAIT;
                    end
                end
            end
            //---------------------------------------------------------------- LM head tile wait
            S_LMWAIT: begin
                mm_in_valid   <= 1'b0;
                mm_pres_valid <= 1'b0;
                if (mm_ov) begin
                    for (ii=0; ii<LM_TN; ii=ii+1)
                        lbuf[vtile*LM_TN + ii] <= mm_c[16*ii +: 16];
                    if (vtile == (NVTILE[VTW-1:0]-1'b1)) begin
                        am_i    <= {(TOKW+1){1'b0}};
                        am_best <= 16'hFF80;        // -inf (bf16)
                        am_arg  <= {TOKW{1'b0}};
                        state   <= S_ARGMAX;
                    end else begin
                        vtile        <= vtile + 1'b1;
                        mm_klen      <= MODEL_DIM[LMKW-1:0];
                        mm_start     <= 1'b1;
                        lm_streaming <= 1'b1;
                        lm_k         <= {(DIMW+1){1'b0}};
                        state        <= S_LMTILE;
                    end
                end
            end
            //---------------------------------------------------------------- argmax
            S_ARGMAX: begin
                if (am_i < VOCAB[TOKW:0]) begin
                    if (bf16_gt(lbuf[am_i[TOKW-1:0]], am_best)) begin
                        am_best <= lbuf[am_i[TOKW-1:0]];
                        am_arg  <= am_i[TOKW-1:0];
                    end
                    am_i <= am_i + 1'b1;
                end else begin
                    for (ii=0; ii<VOCAB; ii=ii+1)
                        logits[16*ii +: 16] <= lbuf[ii];
                    argmax <= am_arg;
                    state  <= S_DONE;
                end
            end
            //----------------------------------------------------------------
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
    // rmsnorm pull beat counters (mirror the unit's beat order; LANES=1 so
    // beat == element index).  Reset at each cn_start.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            cn_ridx <= {(DIMW+1){1'b0}};
            cn_widx <= {(DIMW+1){1'b0}};
            cn_gidx <= {(DIMW+1){1'b0}};
        end else begin
            if (cn_start) begin
                cn_ridx <= {(DIMW+1){1'b0}};
                cn_widx <= {(DIMW+1){1'b0}};
                cn_gidx <= {(DIMW+1){1'b0}};
            end else begin
                if (cn_in_req)  cn_ridx <= cn_ridx + 1'b1;
                if (cn_y_valid) cn_widx <= cn_widx + 1'b1;
                if (cn_g_req)   cn_gidx <= cn_gidx + 1'b1;
            end
        end
    end

    //========================================================================
    // bf16 greater-than (strict) on a direct sign + 15-bit |.| magnitude compare.
    // Treats -0 == +0; ignores nan (finite logits).  Used for the argmax ONLY.
    // bf16->fp32 is a pure low-zero-extend, so this ordering is byte-for-byte
    // identical to the previous fp32 compare -- the comparator is just 31b->15b
    // and am_best 32b->16b.
    //========================================================================
    function automatic bf16_gt(input [15:0] a, input [15:0] b);
        reg sa, sb;
        reg [14:0] ma, mb;
        begin
            sa = a[15]; sb = b[15];
            ma = a[14:0]; mb = b[14:0];
            if (sa != sb) begin
                if ((ma == 15'b0) && (mb == 15'b0)) bf16_gt = 1'b0; // +0 vs -0
                else bf16_gt = (sb == 1'b1);
            end else if (sa == 1'b0) begin
                bf16_gt = (ma > mb);
            end else begin
                bf16_gt = (ma < mb);
            end
        end
    endfunction

endmodule
/* verilator lint_on DECLFILENAME */
