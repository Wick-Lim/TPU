`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_decoder_block_fp8.v  --  ONE GLM-5.2-FP8 decoder layer (ACCEL_GLM52 §2,§6)
//----------------------------------------------------------------------------
// FUNCTION  (the FP8-NATIVE sibling of glm_decoder_block.v -- IDENTICAL FSM,
//   dataflow, dense/MoE modes, streamed residual adds and FFN combine; the ONLY
//   change is that the big LINEAR WEIGHT matmuls run FP8 instead of bf16)
//
//     h = x + mla_attn_fp8( rmsnorm(x), pos, kv_cache )      // attention sub-block
//     y = h + FFN(          rmsnorm(h) )                      // FFN sub-block
//
//   FFN is selected by the MODE input:
//     MODE==0 (DENSE, first_k_dense_replace layers): ONE swiglu_expert_fp8 at
//             INTER_DENSE, no router.    FFN(z) = swiglu(z).
//     MODE==1 (MoE):  moe_router_fp8(z) picks TOPK of N_EXPERT experts; each routed
//             expert runs swiglu_expert_fp8(z, expert_weights) scaled by its routed
//             gate; the ALWAYS-ON shared expert runs swiglu_expert_fp8(z, shared)
//             with weight 1; combine = Σ_e gate_e * y_e + y_shared (fp32 accum).
//
//   FP8 SPLIT (modules_to_not_convert preserved):  the bf16 residual STREAM, the
//   two pre-RMSNorms, softmax/rope/sigmoid/topk tails stay bf16 EXACTLY as
//   glm_decoder_block.v.  ONLY the leaf units' big weight GEMMs are FP8 (E4M3
//   weights + per-[128,128]-block bf16 dequant scales + on-chip dynamic per-token
//   activation->E4M3 quant).  The residual adds are bf16 (glm_fp.vh, §6).
//
//   This module is a PURE ORCHESTRATOR FSM -- it REIMPLEMENTS NO ARITHMETIC:
//     * rmsnorm_unit       : the two pre-norms (pre-attn, pre-FFN), bf16, gamma
//                            pulled from the system.
//     * mla_attn_fp8       : the whole MLA + DSA attention sub-block (FP8 weight
//                            projections; owns the cache read interface).
//     * swiglu_expert_fp8  : the FFN expert(s) -- one INTER_DENSE instance for
//                            dense mode, one INTER_MOE instance reused SERIALLY
//                            across the TOPK routed experts AND the shared expert.
//     * moe_router_fp8     : top-K routing (FP8 gate GEMV -> sigmoid -> top-K ->
//                            renorm -> *SCALE).
//
//----------------------------------------------------------------------------
// EXPERT REUSE STRATEGY (MoE mode -- SERIAL, one swiglu_expert_fp8 instance)
//   ONE swiglu_expert_fp8(INTER_MOE) instance is time-multiplexed over the TOPK
//   routed experts followed by the 1 shared expert (TOPK+1 evaluations).  Before
//   each evaluation the orchestrator drives the FFN weight-pull select
//   (fw_eidx / fw_shared) ALONGSIDE the expert's own (w_sel/w_grp/w_k + block-
//   scale) request, so the system answers the correct expert's FP8 column +
//   [128,128] scales that cycle.  After each routed expert's y the orchestrator
//   scales it by that expert's routed gate (fp32 mul) and adds into a per-element
//   fp32 accumulator; the shared expert is added with weight 1.
//
//----------------------------------------------------------------------------
// WEIGHT PULL INTERFACE (now FP8 E4M3 codes + per-[128,128]-block bf16 scales)
//   Every big weight matrix is delivered as FP8 codes + block scales:
//     * gamma pull (gn_*)            : bf16 RMSNorm learned scale (UNCHANGED).
//     * attention weight pull (aw_*) : aw_col = PE_N FP8 E4M3 lanes (was bf16);
//                                      aw_scale = bf16 [128,128] block scales for
//                                      the addressed (aw_sel, aw_grp) tile-group.
//     * attention cache read (kc_*)  : latent KV cache (UNCHANGED, bf16).
//     * router weight pull (rw_*)    : rw_col = N_EXPERT FP8 E4M3 lanes of W_g[k,*];
//                                      rw_scale = bf16 block scales of W_g.
//     * FFN expert weight pull (fw_*): fw_col / fw_col_up = TN FP8 E4M3 lanes
//                                      (GATE|DOWN / UP); fw_scale_g / fw_scale_u =
//                                      bf16 [128,128] block scales; qualified by
//                                      fw_shared / fw_eidx so the system selects
//                                      WHICH expert (routed e, or shared).
//   The FFN scale/grp/k ports are sized to the DENSE family (the wider one); the
//   MoE expert's narrower scales slice the low K-blocks (zero-extend for grp/k).
//
//----------------------------------------------------------------------------
// LATENCY  (deterministic, data-independent for given params + S; handshake-driven
//   so the FP8 leaf units' own latencies are absorbed via their busy/done)
//   L = L_rmsnorm(MODEL_DIM)                       // pre-attn norm
//     + L_mla_attn_fp8(params, S)                  // attention sub-block
//     + MODEL_DIM (residual add, 1 elt/cycle) + few
//     + L_rmsnorm(MODEL_DIM)                       // pre-FFN norm
//     + FFN:  DENSE -> L_swiglu_fp8(INTER_DENSE)
//             MoE   -> L_router_fp8 + (TOPK+1)*L_swiglu_fp8(INTER_MOE)
//                      + combine (per-expert scale+accumulate)
//     + MODEL_DIM (residual add) + few
//   sync active-high reset; no latch; no comb loop; no data-dependent stall.
//============================================================================
module glm_decoder_block_fp8 #(
    // ---- model / slice config (small-but-faithful) ----
    parameter integer MODEL_DIM  = 128,
    // ---- mla_attn_fp8 slice params (passed straight through) ----
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
    // ---- MoE / FFN config ----
    parameter integer N_EXPERT   = 8,           // routed experts
    parameter integer TOPK       = 2,           // experts/token
    parameter integer INTER_MOE  = 64,          // MoE expert inter size
    parameter integer INTER_DENSE= 256,         // dense-front inter size
    parameter [31:0]  RSCALE     = 32'h40200000,// routed_scaling_factor 2.5 fp32
    parameter integer TN         = 4,           // swiglu output-tile width
    // ---- FP8 weight block size -- DeepSeek-V3 / GLM-5.2-FP8 weight_block_size=[128,128]
    parameter integer BLK        = 128,
    // ---- derived (do NOT override) ----
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer HQK        = H_HEADS * QK_DIM,
    parameter integer HNOPE      = H_HEADS * NOPE,
    parameter integer HV         = H_HEADS * V_DIM,
    parameter integer EIDXW      = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    // mla_attn_fp8 re-derived sizings (mirror its own derivations for port widths)
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
    parameter integer TKIW       = (TOPK <= 1) ? 1 : $clog2(TOPK), // sel_* index w
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1),
    parameter integer FF_GWM     = $clog2(((INTER_MOE >MODEL_DIM)?INTER_MOE :MODEL_DIM)/TN + 1),
    parameter integer FF_KMAX_D  = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM,
    parameter integer FF_KMAX_M  = (INTER_MOE  > MODEL_DIM) ? INTER_MOE  : MODEL_DIM,
    parameter integer FF_KWD     = $clog2(FF_KMAX_D + 1),
    parameter integer FF_KWM     = $clog2(FF_KMAX_M + 1),
    parameter integer R_KW       = $clog2(FF_KMAX_M + 1), // router KMAX = same family
    // ---- FP8 [128,128]-block scale counts (#K-blocks per weight family) ----
    parameter integer A_NB       = (A_KMAX    + BLK - 1) / BLK,  // attention scales
    parameter integer FF_NB_D    = (FF_KMAX_D + BLK - 1) / BLK,  // dense FFN scales
    parameter integer FF_NB_M    = (FF_KMAX_M + BLK - 1) / BLK,  // MoE  FFN scales
    parameter integer R_NB       = (FF_KMAX_M + BLK - 1) / BLK   // router scales
)(
    input  wire                         clk,
    input  wire                         rst,        // sync, active-high

    // ---- control ----
    input  wire                         start,      // 1-cycle pulse: begin layer
    output reg                          busy,
    output reg                          done,       // 1-cycle pulse: y valid
    input  wire                         mode,       // 0=DENSE FFN, 1=MoE FFN
    input  wire [POSW-1:0]              pos,        // token position (RoPE)
    input  wire [IDXW:0]                s_len,      // S causal keys (<= S_MAX)

    // ---- residual stream in / out (bf16) ----
    input  wire [MODEL_DIM*16-1:0]      x_vec,      // MODEL_DIM bf16
    output reg  [MODEL_DIM*16-1:0]      y_out,      // MODEL_DIM bf16

    // ---- RMSNorm gamma pull (combinational, which norm: 0=pre-attn,1=pre-FFN) ----
    output wire                         gn_req,     // need a gamma element this cyc
    output wire                         gn_which,   // 0=pre-attn, 1=pre-FFN
    output wire [$clog2(MODEL_DIM)-1:0] gn_idx,     // gamma element index
    input  wire [15:0]                  gn_val,     // gamma[gn_idx] (bf16)

    // ---- attention weight pull (forwarded mla_attn_fp8 w_*; FP8 codes + scales) ----
    output wire                         aw_req,
    output wire [3:0]                   aw_sel,     // 0=W_dq..6=W_o
    output wire [A_GRPW-1:0]            aw_grp,
    output wire [A_KCW-1:0]             aw_k,
    input  wire [PE_N*8-1:0]            aw_col,     // PE_N FP8 E4M3 weight lanes
    input  wire [16*PE_N*A_NB-1:0]      aw_scale,   // bf16 [128,128] block scales

    // ---- attention cache read (forwarded mla_attn_fp8 kc_*) ----
    output wire                         kc_req,
    output wire [IDXW-1:0]              kc_idx,
    input  wire [KV_LORA*16-1:0]        kc_ckv,
    input  wire [ROPE*16-1:0]           kc_krope,
    input  wire                         kc_valid,

    // ---- MoE router weight pull (W_g column; FP8 codes + scales) ----
    output wire                         rw_req,
    output wire [R_KW-1:0]              rw_k,
    input  wire [8*N_EXPERT-1:0]        rw_col,     // N_EXPERT FP8 E4M3 = W_g[k,*]
    input  wire [16*N_EXPERT*R_NB-1:0]  rw_scale,   // bf16 [128,128] block scales

    // ---- FFN expert weight pull (swiglu_expert_fp8 w_*), qualified by which expert ----
    output wire                         fw_req,
    output wire [1:0]                   fw_sel,     // 0=GATE,1=UP,2=DOWN (swiglu)
    output wire [FF_GWD-1:0]            fw_grp,     // dense-sized group (>= moe)
    output wire [FF_KWD-1:0]            fw_k,       // dense-sized k (>= moe)
    output reg                          fw_shared,  // 1 = shared expert weights
    output reg  [EIDXW-1:0]             fw_eidx,    // routed expert id (MoE)
    input  wire [8*TN-1:0]              fw_col,     // GATE/DOWN FP8 E4M3 lanes
    input  wire [8*TN-1:0]              fw_col_up,  // UP companion FP8 E4M3 lanes
    input  wire [16*TN*FF_NB_D-1:0]     fw_scale_g, // GATE/DOWN bf16 block scales
    input  wire [16*TN*FF_NB_D-1:0]     fw_scale_u  // UP bf16 block scales
);
    `include "glm_fp.vh"

    integer ii;

    //========================================================================
    // residual-stream buffers (bf16)
    //========================================================================
    reg [15:0] xbuf [0:MODEL_DIM-1];   // latched x (residual base for attn add)
    reg [15:0] nrm  [0:MODEL_DIM-1];   // rmsnorm output (fed to attn / ffn)
    reg [15:0] hbuf [0:MODEL_DIM-1];   // h = x + attn  (residual base for ffn add)
    reg [15:0] fbuf [0:MODEL_DIM-1];   // FFN output    (before final residual add)
    reg        mode_q;
    reg [POSW-1:0] pos_q;
    reg [IDXW:0]   slen_q;

    // packed views for sub-unit wide ports
    reg [MODEL_DIM*16-1:0] nrm_vec;    // normalized vector (combinational pack)
    always @* begin
        for (ii = 0; ii < MODEL_DIM; ii = ii + 1)
            nrm_vec[16*ii +: 16] = nrm[ii];
    end


    //========================================================================
    // RMSNorm sub-unit (shared for both pre-norms; LEN=MODEL_DIM, LANES=1).
    //   bf16, UNCHANGED from glm_decoder_block.v (modules_to_not_convert).
    //========================================================================
    reg              rn_start;
    reg              rn_src;            // 0=xbuf, 1=hbuf  (reduce-pass source)
    wire             rn_in_req, rn_g_req, rn_y_valid, rn_busy, rn_done;
    wire [15:0]      rn_y_out;
    reg  [15:0]      rn_x_in;
    reg              rn_x_valid;
    reg  [15:0]      rn_gamma_in;
    reg              rn_g_valid;
    rmsnorm_unit #(.LEN(MODEL_DIM), .LANES(1)) u_rn (
        .clk(clk), .rst(rst), .start(rn_start),
        .in_req(rn_in_req), .x_in(rn_x_in), .x_valid(rn_x_valid),
        .g_req(rn_g_req), .gamma_in(rn_gamma_in), .g_valid(rn_g_valid),
        .y_valid(rn_y_valid), .y_out(rn_y_out), .busy(rn_busy), .done(rn_done)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _rn_busy_unused = &{1'b0, rn_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // rmsnorm beat counters (LANES=1 -> beat index == element index)
    reg [$clog2(MODEL_DIM+1)-1:0] rn_ridx;   // reduce read index (x pull)
    reg [$clog2(MODEL_DIM+1)-1:0] rn_widx;    // normalize write index (y store)
    reg [$clog2(MODEL_DIM+1)-1:0] rn_gidx;    // gamma pull index (per g_req beat)
    reg                            rn_gwhich; // which norm is active (0/1)

    //========================================================================
    // gamma pull is COMBINATIONAL: assert gn_req whenever the shared rmsnorm_unit
    // requests a gamma element (rn_g_req); present the current gamma beat index
    // (rn_gidx) and which norm (rn_gwhich) so the system answers gn_val the SAME
    // cycle.  rn_gidx mirrors the unit's gamma-pass beat (LANES=1 -> beat==idx).
    //========================================================================
    assign gn_req   = rn_g_req;
    assign gn_which = rn_gwhich;
    assign gn_idx   = rn_gidx[$clog2(MODEL_DIM)-1:0];

    //========================================================================
    // mla_attn_fp8 sub-block (full attention; FP8 weight projections).  All its
    // w_*/kc_* pulls are forwarded to this module's aw_*/kc_* ports (caller answers).
    //========================================================================
    reg                       at_start;
    wire                      at_busy, at_done;
    wire [MODEL_DIM*16-1:0]   at_out;
    mla_attn_fp8 #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .BLK(BLK)
    ) u_attn (
        .clk(clk), .rst(rst), .start(at_start), .busy(at_busy), .done(at_done),
        .pos(pos_q), .s_len(slen_q), .x_vec(nrm_vec),
        .w_req(aw_req), .w_sel(aw_sel), .w_grp(aw_grp), .w_k(aw_k),
        .w_col(aw_col), .w_scale(aw_scale),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid), .out(at_out)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _at_busy_unused = &{1'b0, at_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // moe_router_fp8 (MoE mode).  Its FP8 W_g pull is forwarded to rw_*.
    //========================================================================
    reg                       rt_start;
    wire                      rt_busy, rt_done;
    wire [TOPK*EIDXW-1:0]     rt_sel_idx;
    wire [TOPK*16-1:0]        rt_sel_weight;
    moe_router_fp8 #(
        .HIDDEN(MODEL_DIM), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .SCALE(RSCALE), .KMAX(FF_KMAX_M), .BLK(BLK)
    ) u_router (
        .clk(clk), .rst(rst), .start(rt_start), .busy(rt_busy), .done(rt_done),
        .x_vec(nrm_vec),
        .w_req(rw_req), .w_k(rw_k), .w_col(rw_col), .w_scale(rw_scale),
        .sel_idx(rt_sel_idx), .sel_weight(rt_sel_weight)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _rt_busy_unused = &{1'b0, rt_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // captured routing result
    reg [EIDXW-1:0] sel_e   [0:TOPK-1];
    reg [15:0]      sel_w   [0:TOPK-1];

    //========================================================================
    // FFN expert (one DENSE swiglu_expert_fp8 + one MoE instance).  Only the
    // instance matching `mode_q` is started; the other stays idle.  Both share
    // the normalized input nrm_vec and pull their FP8 weights+scales through the
    // SAME fw_* ports (qualified by fw_shared / fw_eidx).
    //   FFN col/scale ports are the DENSE family (wider); the MoE instance's
    //   narrower block scales slice the low FF_NB_M K-blocks of fw_scale_*.
    //========================================================================
    // ---- dense expert ----
    reg                  ed_start;
    wire                 ed_busy, ed_done;
    wire [MODEL_DIM*16-1:0] ed_y;
    wire                 ed_wreq;
    wire [1:0]           ed_wsel;
    wire [FF_GWD-1:0]    ed_wgrp;
    wire [FF_KWD-1:0]    ed_wk;
    swiglu_expert_fp8 #(
        .HIDDEN(MODEL_DIM), .INTER(INTER_DENSE), .TN(TN), .KMAX(FF_KMAX_D), .BLK(BLK)
    ) u_dense (
        .clk(clk), .rst(rst), .start(ed_start), .busy(ed_busy), .done(ed_done),
        .x_vec(nrm_vec),
        .w_req(ed_wreq), .w_sel(ed_wsel), .w_grp(ed_wgrp), .w_k(ed_wk),
        .w_col(fw_col), .w_col_up(fw_col_up),
        .w_scale_g(fw_scale_g), .w_scale_u(fw_scale_u), .y_out(ed_y)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _ed_busy_unused = &{1'b0, ed_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- moe expert (serial reuse) ----
    reg                  em_start;
    wire                 em_busy, em_done;
    wire [MODEL_DIM*16-1:0] em_y;
    wire                 em_wreq;
    wire [1:0]           em_wsel;
    wire [FF_GWM-1:0]    em_wgrp;
    wire [FF_KWM-1:0]    em_wk;
    swiglu_expert_fp8 #(
        .HIDDEN(MODEL_DIM), .INTER(INTER_MOE), .TN(TN), .KMAX(FF_KMAX_M), .BLK(BLK)
    ) u_moe (
        .clk(clk), .rst(rst), .start(em_start), .busy(em_busy), .done(em_done),
        .x_vec(nrm_vec),
        .w_req(em_wreq), .w_sel(em_wsel), .w_grp(em_wgrp), .w_k(em_wk),
        .w_col(fw_col), .w_col_up(fw_col_up),
        .w_scale_g(fw_scale_g[16*TN*FF_NB_M-1:0]),
        .w_scale_u(fw_scale_u[16*TN*FF_NB_M-1:0]), .y_out(em_y)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _em_busy_unused = &{1'b0, em_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // FFN weight-pull MUX: forward whichever expert instance is active.  Widths:
    // dense grp/k are the WIDE family (FF_GWD/FF_KWD >= moe's), so the moe fields
    // zero-extend into them.  The FP8 col/scale responses are width-shared (same
    // TN) -- dense uses all FF_NB_D scale K-blocks, moe slices the low FF_NB_M.
    assign fw_req = mode_q ? em_wreq : ed_wreq;
    assign fw_sel = mode_q ? em_wsel : ed_wsel;
    assign fw_grp = mode_q ? {{(FF_GWD-FF_GWM){1'b0}}, em_wgrp} : ed_wgrp;
    assign fw_k   = mode_q ? {{(FF_KWD-FF_KWM){1'b0}}, em_wk}   : ed_wk;

    //========================================================================
    // FFN combine accumulator (MoE): acc[d] += gate_e * y_e[d]  (fp32) ; shared
    // added with weight 1.  Per-element fp32 accumulator array.
    //========================================================================
    reg [31:0] facc [0:MODEL_DIM-1];

    //========================================================================
    // MASTER FSM  (byte-for-byte the same control as glm_decoder_block.v)
    //========================================================================
    localparam [4:0]
        T_IDLE   = 5'd0,
        T_RN1    = 5'd1,    // pre-attn rmsnorm(x) -> nrm
        T_ATTN   = 5'd2,    // mla_attn_fp8(nrm) -> at_out
        T_RADD1  = 5'd3,    // h = x + at_out (bf16, per-elt)
        T_RN2    = 5'd4,    // pre-ffn rmsnorm(h) -> nrm
        T_FFN_D  = 5'd5,    // dense swiglu(nrm) -> fbuf
        T_ROUTE  = 5'd6,    // moe_router_fp8(nrm) -> sel
        // T_EXP (5'd7) REMOVED: cur_gate_f is now latched at the em_start pulse
        // (T_ROUTE / T_ACC next-expert branches), so the 1-cycle filler that only
        // latched the gate before T_EXPW is gone -- saves 1 cycle/expert.
        T_EXPW   = 5'd8,    // wait expert done
        T_ACC    = 5'd9,    // scale+accumulate expert y into facc (1 elt/cycle)
        T_FCOMB  = 5'd10,   // finalize fbuf from facc (MoE, 1 elt/cycle)
        T_RADD2  = 5'd11,   // y = h + fbuf (bf16, per-elt)
        T_DONE   = 5'd12;
    reg [4:0] state;

    // residual-add element cursor (shared by T_RADD1 / T_RADD2)
    reg [$clog2(MODEL_DIM+1)-1:0] radd_i;
    // FFN combine element cursor (shared by T_ACC / T_FCOMB), streamed ONE element
    // per cycle (one fp32_mul+fp32_add) -- bounded combinational path, identical
    // numerics to a parallel fp32 datapath.
    reg [$clog2(MODEL_DIM+1)-1:0] comb_i;

    // MoE expert loop bookkeeping
    reg [$clog2(TOPK+2)-1:0] exp_i;   // 0..TOPK-1 routed, TOPK = shared
    reg                      exp_is_shared;

    // current routed gate (fp32) for the scale step
    reg [31:0] cur_gate_f;
    // next routed-expert index (exp_i+1), used only when exp_i < TOPK-1 so it
    // never exceeds TOPK-1 (TKIW bits suffice to address sel_e).
    wire [TKIW-1:0] exp_nxt = exp_i[TKIW-1:0] + 1'b1;

    //========================================================================
    // SHARED fp32 add + bf16 narrow datapath (RESOURCE MERGE).  The four
    // mutually-exclusive streaming states T_RADD1 / T_RADD2 / T_ACC / T_FCOMB
    // each need at most ONE fp32 add and ONE fp32->bf16 narrow per cycle on a
    // single element.  The per-state inline fp32_to_bf16(fp32_add(...)) copies
    // block synthesis from sharing the adder/narrower; factor them into ONE
    // shared adder + ONE shared narrower fed by state-muxed operands and a
    // state-muxed element index.  Bit-identical to the original expressions.
    //   T_RADD1 : hbuf <- bf16( x + attn )
    //   T_RADD2 : y    <- bf16( h + ffn  )
    //   T_ACC   : facc <- facc + gate*em_y      (stays fp32; no narrow)
    //   T_FCOMB : fbuf <- bf16( facc )          (narrow only; no add)
    //========================================================================
    reg  [$clog2(MODEL_DIM)-1:0] sh_idx;     // state-muxed element index
    reg  [31:0]                  sh_add_a;   // state-muxed fp32 addend A
    reg  [31:0]                  sh_add_b;   // state-muxed fp32 addend B
    reg                          sh_nsel;    // narrow src: 0=add result, 1=facc
    wire [31:0] sh_add_s     = fp32_add(sh_add_a, sh_add_b);
    wire [31:0] sh_narrow_in = sh_nsel ? facc[sh_idx] : sh_add_s;
    wire [15:0] sh_narrow_bf = fp32_to_bf16(sh_narrow_in);
    // em_y element as fp32 for the MoE accumulate.  POWER: in the SHARED expert
    // (cur_gate_f==1.0) the routed-gate multiply is bypassed -- em_y is added
    // straight into facc, holding the multiplier inputs stable.  1.0*x==x in
    // fp32_mul (mantissa<<23, exponent unchanged, zero guard/sticky), so this is
    // BIT-IDENTICAL while saving MODEL_DIM fp32 multiplies per layer.
    wire [31:0] sh_emy_f = bf16_to_fp32(em_y[16*comb_i +: 16]);
    always @* begin
        sh_idx   = radd_i[$clog2(MODEL_DIM)-1:0];
        sh_add_a = 32'h0;
        sh_add_b = 32'h0;
        sh_nsel  = 1'b0;
        case (state)
        T_RADD1: begin
            sh_idx   = radd_i[$clog2(MODEL_DIM)-1:0];
            sh_add_a = bf16_to_fp32(xbuf[sh_idx]);
            sh_add_b = bf16_to_fp32(at_out[16*radd_i +: 16]);
        end
        T_RADD2: begin
            sh_idx   = radd_i[$clog2(MODEL_DIM)-1:0];
            sh_add_a = bf16_to_fp32(hbuf[sh_idx]);
            sh_add_b = bf16_to_fp32(fbuf[sh_idx]);
        end
        T_ACC: begin
            sh_idx   = comb_i[$clog2(MODEL_DIM)-1:0];
            sh_add_a = facc[sh_idx];
            sh_add_b = exp_is_shared ? sh_emy_f
                                     : fp32_mul(cur_gate_f, sh_emy_f);
        end
        T_FCOMB: begin
            sh_idx   = comb_i[$clog2(MODEL_DIM)-1:0];
            sh_nsel  = 1'b1;
        end
        default: ;
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            state      <= T_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            y_out      <= {MODEL_DIM*16{1'b0}};
            mode_q     <= 1'b0;
            pos_q      <= {POSW{1'b0}};
            slen_q     <= {(IDXW+1){1'b0}};
            rn_start   <= 1'b0; rn_src <= 1'b0; rn_gwhich <= 1'b0;
            rn_x_in    <= 16'h0; rn_x_valid <= 1'b0;
            rn_gamma_in<= 16'h0; rn_g_valid <= 1'b0;
            // NOTE: rn_ridx / rn_widx / rn_gidx are owned EXCLUSIVELY by the
            // dedicated rmsnorm beat-counter always-block below (single-driver).
            at_start   <= 1'b0;
            rt_start   <= 1'b0;
            ed_start   <= 1'b0; em_start <= 1'b0;
            fw_shared  <= 1'b0; fw_eidx <= {EIDXW{1'b0}};
            radd_i     <= {$clog2(MODEL_DIM+1){1'b0}};
            comb_i     <= {$clog2(MODEL_DIM+1){1'b0}};
            exp_i      <= {$clog2(TOPK+2){1'b0}};
            exp_is_shared <= 1'b0;
            cur_gate_f <= 32'h0;
            for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                xbuf[ii] <= 16'h0; nrm[ii] <= 16'h0; hbuf[ii] <= 16'h0;
                fbuf[ii] <= 16'h0; facc[ii] <= 32'h0;
            end
            for (ii=0; ii<TOPK; ii=ii+1) begin
                sel_e[ii] <= {EIDXW{1'b0}}; sel_w[ii] <= 16'h0;
            end
        end else begin
            // ---- default pulse deassert ----
            done     <= 1'b0;
            rn_start <= 1'b0;
            at_start <= 1'b0;
            rt_start <= 1'b0;
            ed_start <= 1'b0;
            em_start <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            T_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy   <= 1'b1;
                    mode_q <= mode;
                    pos_q  <= pos;
                    slen_q <= s_len;
                    for (ii=0; ii<MODEL_DIM; ii=ii+1)
                        xbuf[ii] <= x_vec[16*ii +: 16];
                    // launch pre-attn rmsnorm over x (reduce source = xbuf)
                    rn_src    <= 1'b0;
                    rn_gwhich <= 1'b0;          // pre-attn gamma
                    rn_start  <= 1'b1;
                    state     <= T_RN1;
                end
            end
            //---------------------------------------------------------------- pre-attn norm
            T_RN1: begin
                rn_x_valid <= 1'b0; rn_g_valid <= 1'b0;
                if (rn_in_req) begin
                    rn_x_in    <= xbuf[rn_ridx[$clog2(MODEL_DIM)-1:0]];
                    rn_x_valid <= 1'b1;
                end
                if (rn_g_req) begin
                    // gamma pull is COMBINATIONAL (gn_*); register the answer one
                    // cycle, mirroring the x pull's 1-cycle response.
                    rn_gamma_in  <= gn_val;
                    rn_g_valid   <= 1'b1;
                end
                if (rn_y_valid) nrm[rn_widx[$clog2(MODEL_DIM)-1:0]] <= rn_y_out;
                if (rn_done) begin
                    at_start <= 1'b1;                // launch MLA attention (FP8)
                    state    <= T_ATTN;
                end
            end
            //---------------------------------------------------------------- attention
            T_ATTN: begin
                if (at_done) begin
                    // h = x + attn  : start residual add (combinational fp32 add)
                    radd_i <= {$clog2(MODEL_DIM+1){1'b0}};
                    state  <= T_RADD1;
                end
            end
            //---------------------------------------------------------------- residual add 1
            T_RADD1: begin
                // h = x + attn  (shared fp32 add + bf16 narrow; see sh_* datapath)
                hbuf[radd_i[$clog2(MODEL_DIM)-1:0]] <= sh_narrow_bf;
                if (radd_i == MODEL_DIM[$clog2(MODEL_DIM+1)-1:0]-1'b1) begin
                    // launch pre-ffn rmsnorm over h
                    rn_src    <= 1'b1;
                    rn_gwhich <= 1'b1;          // pre-ffn gamma
                    rn_start  <= 1'b1;
                    state     <= T_RN2;
                end else begin
                    radd_i <= radd_i + 1'b1;
                end
            end
            //---------------------------------------------------------------- pre-ffn norm
            T_RN2: begin
                rn_x_valid <= 1'b0; rn_g_valid <= 1'b0;
                if (rn_in_req) begin
                    rn_x_in    <= hbuf[rn_ridx[$clog2(MODEL_DIM)-1:0]];
                    rn_x_valid <= 1'b1;
                end
                if (rn_g_req) begin
                    rn_gamma_in <= gn_val;          // combinational gamma answer
                    rn_g_valid  <= 1'b1;
                end
                if (rn_y_valid) nrm[rn_widx[$clog2(MODEL_DIM)-1:0]] <= rn_y_out;
                if (rn_done) begin
                    if (mode_q) begin
                        // MoE: route first
                        rt_start <= 1'b1;
                        state    <= T_ROUTE;
                    end else begin
                        // DENSE: single swiglu expert
                        ed_start  <= 1'b1;
                        fw_shared <= 1'b0;
                        fw_eidx   <= {EIDXW{1'b0}};
                        state     <= T_FFN_D;
                    end
                end
            end
            //---------------------------------------------------------------- dense ffn
            T_FFN_D: begin
                if (ed_done) begin
                    for (ii=0; ii<MODEL_DIM; ii=ii+1)
                        fbuf[ii] <= ed_y[16*ii +: 16];
                    radd_i <= {$clog2(MODEL_DIM+1){1'b0}};
                    state  <= T_RADD2;
                end
            end
            //---------------------------------------------------------------- moe route
            T_ROUTE: begin
                if (rt_done) begin
                    for (ii=0; ii<TOPK; ii=ii+1) begin
                        sel_e[ii] <= rt_sel_idx[EIDXW*ii +: EIDXW];
                        sel_w[ii] <= rt_sel_weight[16*ii +: 16];
                    end
                    // clear combine accumulator
                    for (ii=0; ii<MODEL_DIM; ii=ii+1) facc[ii] <= 32'h0;
                    exp_i         <= {$clog2(TOPK+2){1'b0}};
                    exp_is_shared <= 1'b0;
                    // select first routed expert's weights and start swiglu.
                    // Latch its gate AT the em_start pulse (was the T_EXP filler):
                    // first routed expert -> sel_w[0] (== rt_sel_weight[0] this cyc).
                    fw_shared  <= 1'b0;
                    fw_eidx    <= rt_sel_idx[EIDXW*0 +: EIDXW];
                    cur_gate_f <= bf16_to_fp32(rt_sel_weight[16*0 +: 16]);
                    em_start   <= 1'b1;
                    state      <= T_EXPW;
                end
            end
            //---------------------------------------------------------------- expert wait
            T_EXPW: begin
                // em_y (u_moe.y_out) holds stable after em_done until the next
                // em_start, so T_ACC can stream it across MODEL_DIM cycles.
                if (em_done) begin
                    comb_i <= {$clog2(MODEL_DIM+1){1'b0}};
                    state  <= T_ACC;
                end
            end
            //---------------------------------------------------------------- scale+accumulate (1 elt/cycle)
            T_ACC: begin
                // facc[comb_i] += cur_gate_f * em_y[comb_i]   (one fp32 MAC/cycle)
                // via the SHARED fp32 adder (sh_add_s); the routed-gate multiply
                // is bypassed for the shared expert (sh_add_b mux, see sh_* block).
                facc[comb_i[$clog2(MODEL_DIM)-1:0]] <= sh_add_s;
                if (comb_i == MODEL_DIM[$clog2(MODEL_DIM+1)-1:0]-1'b1) begin
                    // last element accumulated -> advance to next expert / finalize
                    if (exp_is_shared) begin
                        comb_i <= {$clog2(MODEL_DIM+1){1'b0}};
                        state  <= T_FCOMB;
                    end else if (exp_i == TOPK[$clog2(TOPK+2)-1:0]-1'b1) begin
                        // routed experts done -> run shared expert (weight 1).
                        // Latch gate=1.0 AT the em_start pulse (was T_EXP filler).
                        exp_is_shared <= 1'b1;
                        fw_shared     <= 1'b1;
                        fw_eidx       <= {EIDXW{1'b0}};
                        cur_gate_f    <= 32'h3F80_0000;  // 1.0 (shared, weight 1)
                        em_start      <= 1'b1;
                        state         <= T_EXPW;
                    end else begin
                        // next routed expert.  Latch its gate AT the em_start
                        // pulse (was T_EXP filler): next routed -> sel_w[exp_nxt].
                        exp_i      <= exp_i + 1'b1;
                        fw_shared  <= 1'b0;
                        fw_eidx    <= sel_e[exp_nxt]; // next expert id
                        cur_gate_f <= bf16_to_fp32(sel_w[exp_nxt]);
                        em_start   <= 1'b1;
                        state      <= T_EXPW;
                    end
                end else begin
                    comb_i <= comb_i + 1'b1;
                end
            end
            //---------------------------------------------------------------- moe combine finalize (1 elt/cycle)
            T_FCOMB: begin
                // narrow each fp32 accumulator to bf16 -> fbuf, one elt/cycle
                // (shared fp32->bf16 narrower; see sh_* datapath)
                fbuf[comb_i[$clog2(MODEL_DIM)-1:0]] <= sh_narrow_bf;
                if (comb_i == MODEL_DIM[$clog2(MODEL_DIM+1)-1:0]-1'b1) begin
                    radd_i <= {$clog2(MODEL_DIM+1){1'b0}};
                    state  <= T_RADD2;
                end else begin
                    comb_i <= comb_i + 1'b1;
                end
            end
            //---------------------------------------------------------------- residual add 2
            T_RADD2: begin
                // y = h + ffn  (shared fp32 add + bf16 narrow; see sh_* datapath)
                y_out[16*radd_i +: 16] <= sh_narrow_bf;
                if (radd_i == MODEL_DIM[$clog2(MODEL_DIM+1)-1:0]-1'b1)
                    state <= T_DONE;
                else
                    radd_i <= radd_i + 1'b1;
            end
            //----------------------------------------------------------------
            T_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= T_IDLE;
            end
            default: state <= T_IDLE;
            endcase
        end
    end

    //========================================================================
    // rmsnorm pull beat counters (mirror the unit's internal beat order;
    // LANES=1 so beat == element index).  Reset at each rn_start.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            rn_ridx <= {$clog2(MODEL_DIM+1){1'b0}};
            rn_widx <= {$clog2(MODEL_DIM+1){1'b0}};
            rn_gidx <= {$clog2(MODEL_DIM+1){1'b0}};
        end else begin
            if (rn_start) begin
                rn_ridx <= {$clog2(MODEL_DIM+1){1'b0}};
                rn_widx <= {$clog2(MODEL_DIM+1){1'b0}};
                rn_gidx <= {$clog2(MODEL_DIM+1){1'b0}};
            end else begin
                if (rn_in_req)  rn_ridx <= rn_ridx + 1'b1;
                if (rn_y_valid) rn_widx <= rn_widx + 1'b1;
                if (rn_g_req)   rn_gidx <= rn_gidx + 1'b1;
            end
        end
    end

    // rn_src is the reduce-pass source selector; it is read implicitly by the
    // T_RN1/T_RN2 states answering rn_in_req from xbuf/hbuf respectively, so the
    // signal itself is informational.  Waive unused.
    /* verilator lint_off UNUSEDSIGNAL */
    wire _rn_src_unused = &{1'b0, rn_src};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
/* verilator lint_on DECLFILENAME */
