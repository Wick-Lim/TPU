`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// spec_chain_top.v  --  GLM-5.2-FP8 K-STEP MTP-CHAIN SPECULATIVE-DECODE TOP
//                       (P1.3b -> B8 : PROMOTED pull ports, simulable)
//----------------------------------------------------------------------------
// PURPOSE  (the K>1 generalisation of spec_decode_top's K=1 loop)
//   DeepSeek-MTP runs its single predict layer RECURRENTLY to mint K speculative
//   drafts from ONE main-model pass: each step feeds the PRIOR step's decoder-
//   block hidden state (the chain state) back in as the next step's h_t.  P1.3b
//   exposed exactly that state on mtp_head_fp8 via the additive `h_mtp` port, so
//   this orchestrator can chain it:
//
//     0. MAIN  : glm_model_fp8(PE_M=1, token=cur_tok, pos=cur_pos)
//                  -> verified m_0 = argmax , h_0 = h_state (final-norm output).
//     1. CHAIN : for k = 0 .. K-1  (K recurrent mtp_head_fp8 runs):
//                  h_in_0   = h_0  (main model hidden state)              [seed]
//                  h_in_k   = h_mtp from step k-1   (*** the P1.3b chain state ***)
//                  emb_k    = embed(prev token: m_0 for k=0 else draft d_{k-1})
//                  d_k      = mtp_head_fp8(h_in_k, emb_k, pos=cur_pos+1+k).argmax
//                  h_mtp    = its packed decoder-block state -> feeds step k+1.
//                -> K drafts {d_0 .. d_{K-1}}.
//     2. VERIFY: assemble the PE_M=K+1 batch {cur_tok, d_0 .. d_{K-1}} and run
//                glm_model_fp8(PE_M=K+1) in ONE weight-load (the spec_batched_top
//                contract) -> K+1 truths {m_0 .. m_K}.
//     3. FEED  : pulse spec_decode_seq.pass_valid with draft_vec={d_0..d_{K-1}},
//                truth_vec={m_0..m_K}, n_draft=K -> commit the longest accepted
//                prefix; advance cur_tok/cur_pos over the committed frontier.
//
//   PURE ORCHESTRATOR -- reimplements NO math.  It instantiates the committed
//   units (glm_model_fp8 x2, mtp_head_fp8, spec_decode_seq) and routes their
//   weight/cache/embedding PULL interfaces up to the top exactly as
//   spec_decode_top / spec_batched_top do, using THREE independent pull buses:
//     m_* : the PE_M=1 main model         (prefix m_)
//     t_* : the recurrent MTP head        (prefix t_)
//     v_* : the PE_M=K+1 verify model     (prefix v_)
//   plus a whole-vector embedding pull (em_req/em_tok/em_vec) that answers the
//   MTP head's emb_t1 = embed(prev token).
//
// *** SEED CONVENTION (documented; now implemented) ***
//   The chain is seeded with TWO hidden-state conventions:
//     * step 0   : h_chain[0] = u_main.h_state  -- the POST-final-RMSNorm hidden
//                  state (the model's normalised output h).
//     * step k>=1: h_chain[k] = u_mtp.h_mtp     -- the PRE-final-norm db_y decoder-
//                  block state carried out of the prior chain step.
//   This is a SEED-scale mismatch: it perturbs the DRAFT distribution and hence
//   K_eff (acceptance rate), NOT the spec==greedy safety property -- spec_decode_seq
//   only ever commits the VERIFY model's own argmaxes (truth_vec), never a raw
//   draft, so a mis-seeded draft can only be REJECTED, never wrongly committed.
//
// DISCIPLINE: synchronous active-high reset; every output registered; no latch;
//   no combinational loop; handshake-driven (each sub-unit launch waits its done).
//============================================================================
module spec_chain_top #(
    // ---- model / slice config (mirrors glm_model_fp8 / mtp_head_fp8) ----
    parameter integer MODEL_DIM  = 128,
    parameter integer L          = 6,
    parameter integer N_DENSE    = 3,
    parameter integer VOCAB      = 256,
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
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 4,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 4,
    parameter integer PROJ_TN    = 4,           // MTP combine-proj cols/pass
    parameter integer DRAFT_K    = 2,           // # chained MTP drafts per pass (K)
    // ====================================================================
    // derived (do NOT override) -- copied verbatim from glm_model_fp8 /
    //                              mtp_head_fp8 / spec_decode_seq
    // ====================================================================
    parameter integer B          = DRAFT_K + 1,    // PE_M batch = K+1 verify rows
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
    parameter integer A_NB       = (A_KMAX    + BLK - 1) / BLK,
    parameter integer FF_NB_D    = (FF_KMAX_D + BLK - 1) / BLK,
    parameter integer R_NB       = (FF_KMAX_M + BLK - 1) / BLK,
    parameter integer LAYW       = (L     <= 1) ? 1 : $clog2(L),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    // MTP combine-projection derived (mtp_head_fp8)
    parameter integer CK         = 2 * MODEL_DIM,
    parameter integer CKIW       = $clog2(CK),
    parameter integer NPTILE     = MODEL_DIM / PROJ_TN,
    parameter integer PTW        = (NPTILE <= 1) ? 1 : $clog2(NPTILE),
    parameter integer PROJ_NB    = (CK + BLK - 1) / BLK,
    // spec_decode_seq batch-interface widths
    parameter integer DKW        = (DRAFT_K <= 1) ? 1 : $clog2(DRAFT_K + 1),
    parameter integer OCW        = $clog2(DRAFT_K + 2),   // 0..K+1 prefix count width
    // chain-step counter width
    parameter integer KW         = (DRAFT_K <= 1) ? 1 : $clog2(DRAFT_K)
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    // ---- loop control ----
    input  wire                          start,      // 1-cycle pulse: begin loop
    input  wire [TOKW-1:0]               prompt_tok, // first token to decode from
    input  wire [POSW-1:0]               start_pos,  // first query position
    input  wire [IDXW:0]                 s_len,      // S causal keys (<= S_MAX)
    input  wire                          mtp_mode,   // MTP decoder block: 0=DENSE,1=MoE
    input  wire [15:0]                   num_passes, // # main passes to run (>=1)
    output reg                           busy,
    output reg                           done,       // 1-cycle pulse: loop complete

    // ---- committed-token stream + counters (straight from spec_decode_seq) ----
    output wire                          commit_valid,
    output wire [TOKW-1:0]               commit_tok,
    output wire                          accepted,
    output wire [31:0]                   total_tokens,
    output wire [31:0]                   main_passes,
    output wire [31:0]                   accepts,
    output wire [31:0]                   rejects,

    // ====================================================================
    // whole-vector embedding pull : answers the MTP head's emb_t1 = embed(prev tok)
    // ====================================================================
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,
    input  wire [MODEL_DIM*16-1:0]       em_vec,

    // ====================================================================
    // m_* : PE_M=1 MAIN model pull bus (routed straight up)
    // ====================================================================
    output wire                          m_em_req,
    output wire [TOKW-1:0]               m_em_tok,
    output wire [DIMW-1:0]               m_em_idx,
    input  wire [15:0]                   m_em_val,
    output wire [LAYW-1:0]               m_db_layer,
    output wire                          m_idx_fresh,
    output wire [LAYW-1:0]               m_idx_win,
    output wire                          m_gn_req,
    output wire                          m_gn_which,
    output wire [DIMW-1:0]               m_gn_idx,
    input  wire [15:0]                   m_gn_val,
    output wire                          m_aw_req,
    output wire [3:0]                    m_aw_sel,
    output wire [A_GRPW-1:0]             m_aw_grp,
    output wire [A_KCW-1:0]              m_aw_k,
    input  wire [PE_N*8-1:0]             m_aw_col,
    input  wire [16*PE_N*A_NB-1:0]       m_aw_scale,
    output wire                          m_kc_req,
    output wire [IDXW-1:0]               m_kc_idx,
    input  wire [KV_LORA*16-1:0]         m_kc_ckv,
    input  wire [ROPE*16-1:0]            m_kc_krope,
    input  wire                          m_kc_valid,
    output wire                          m_rw_req,
    output wire [R_KW-1:0]               m_rw_k,
    input  wire [8*N_EXPERT-1:0]         m_rw_col,
    input  wire [16*N_EXPERT*R_NB-1:0]   m_rw_scale,
    output wire                          m_fw_req,
    output wire [1:0]                    m_fw_sel,
    output wire [FF_GWD-1:0]             m_fw_grp,
    output wire [FF_KWD-1:0]             m_fw_k,
    output wire                          m_fw_shared,
    output wire [EIDXW-1:0]              m_fw_eidx,
    input  wire [8*TN-1:0]               m_fw_col,
    input  wire [8*TN-1:0]               m_fw_col_up,
    input  wire [16*TN*FF_NB_D-1:0]      m_fw_scale_g,
    input  wire [16*TN*FF_NB_D-1:0]      m_fw_scale_u,
    output wire                          m_fn_req,
    output wire [DIMW-1:0]               m_fn_idx,
    input  wire [15:0]                   m_fn_val,
    output wire                          m_lw_req,
    output wire [VTW-1:0]                m_lw_vtile,
    output wire [DIMW-1:0]               m_lw_k,
    input  wire [LM_TN*16-1:0]           m_lw_col,

    // ====================================================================
    // t_* : recurrent MTP head pull bus (routed straight up)
    // ====================================================================
    output wire                          t_cn_req,
    output wire [1:0]                    t_cn_which,
    output wire [DIMW-1:0]               t_cn_idx,
    input  wire [15:0]                   t_cn_val,
    output wire                          t_pw_req,
    output wire [PTW-1:0]                t_pw_ptile,
    output wire [CKIW-1:0]               t_pw_k,
    input  wire [PROJ_TN*8-1:0]          t_pw_col,
    input  wire [16*PROJ_TN*PROJ_NB-1:0] t_pw_scale,
    output wire                          t_gn_req,
    output wire                          t_gn_which,
    output wire [DIMW-1:0]               t_gn_idx,
    input  wire [15:0]                   t_gn_val,
    output wire                          t_aw_req,
    output wire [3:0]                    t_aw_sel,
    output wire [A_GRPW-1:0]             t_aw_grp,
    output wire [A_KCW-1:0]              t_aw_k,
    input  wire [PE_N*8-1:0]             t_aw_col,
    input  wire [16*PE_N*A_NB-1:0]       t_aw_scale,
    output wire                          t_kc_req,
    output wire [IDXW-1:0]               t_kc_idx,
    input  wire [KV_LORA*16-1:0]         t_kc_ckv,
    input  wire [ROPE*16-1:0]            t_kc_krope,
    input  wire                          t_kc_valid,
    output wire                          t_rw_req,
    output wire [R_KW-1:0]               t_rw_k,
    input  wire [8*N_EXPERT-1:0]         t_rw_col,
    input  wire [16*N_EXPERT*R_NB-1:0]   t_rw_scale,
    output wire                          t_fw_req,
    output wire [1:0]                    t_fw_sel,
    output wire [FF_GWD-1:0]             t_fw_grp,
    output wire [FF_KWD-1:0]             t_fw_k,
    output wire                          t_fw_shared,
    output wire [EIDXW-1:0]              t_fw_eidx,
    input  wire [8*TN-1:0]               t_fw_col,
    input  wire [8*TN-1:0]               t_fw_col_up,
    input  wire [16*TN*FF_NB_D-1:0]      t_fw_scale_g,
    input  wire [16*TN*FF_NB_D-1:0]      t_fw_scale_u,
    output wire                          t_lw_req,
    output wire [VTW-1:0]                t_lw_vtile,
    output wire [DIMW-1:0]               t_lw_k,
    input  wire [LM_TN*16-1:0]           t_lw_col,

    // ====================================================================
    // v_* : PE_M=K+1 VERIFY model pull bus (routed straight up)
    // ====================================================================
    output wire                          v_em_req,
    output wire [TOKW-1:0]               v_em_tok,
    output wire [DIMW-1:0]               v_em_idx,
    input  wire [15:0]                   v_em_val,
    output wire [LAYW-1:0]               v_db_layer,
    output wire                          v_idx_fresh,
    output wire [LAYW-1:0]               v_idx_win,
    output wire                          v_gn_req,
    output wire                          v_gn_which,
    output wire [DIMW-1:0]               v_gn_idx,
    input  wire [15:0]                   v_gn_val,
    output wire                          v_aw_req,
    output wire [3:0]                    v_aw_sel,
    output wire [A_GRPW-1:0]             v_aw_grp,
    output wire [A_KCW-1:0]              v_aw_k,
    input  wire [PE_N*8-1:0]             v_aw_col,
    input  wire [16*PE_N*A_NB-1:0]       v_aw_scale,
    output wire                          v_kc_req,
    output wire [IDXW-1:0]               v_kc_idx,
    input  wire [KV_LORA*16-1:0]         v_kc_ckv,
    input  wire [ROPE*16-1:0]            v_kc_krope,
    input  wire                          v_kc_valid,
    output wire                          v_rw_req,
    output wire [R_KW-1:0]               v_rw_k,
    input  wire [8*N_EXPERT-1:0]         v_rw_col,
    input  wire [16*N_EXPERT*R_NB-1:0]   v_rw_scale,
    output wire                          v_fw_req,
    output wire [1:0]                    v_fw_sel,
    output wire [FF_GWD-1:0]             v_fw_grp,
    output wire [FF_KWD-1:0]             v_fw_k,
    output wire                          v_fw_shared,
    output wire [EIDXW-1:0]              v_fw_eidx,
    input  wire [8*TN-1:0]               v_fw_col,
    input  wire [8*TN-1:0]               v_fw_col_up,
    input  wire [16*TN*FF_NB_D-1:0]      v_fw_scale_g,
    input  wire [16*TN*FF_NB_D-1:0]      v_fw_scale_u,
    output wire                          v_fn_req,
    output wire [DIMW-1:0]               v_fn_idx,
    input  wire [15:0]                   v_fn_val,
    output wire                          v_lw_req,
    output wire [VTW-1:0]                v_lw_vtile,
    output wire [DIMW-1:0]               v_lw_k,
    input  wire [LM_TN*16-1:0]           v_lw_col
);
    integer ci;

    //========================================================================
    // cursor + pass counter
    //========================================================================
    reg [TOKW-1:0] cur_tok;
    reg [POSW-1:0] cur_pos;
    reg [POSW-1:0] pos_q;
    reg [IDXW:0]   slen_q;
    reg            mode_q;
    reg [15:0]     passes_left;
    reg [TOKW-1:0] prev_tok;      // token whose embedding seeds the next chain step

    //========================================================================
    // per-step chain buffers.  h_chain[0] = main-model h_state; h_chain[k] for
    // k>=1 = the h_mtp packed decoder-block state minted by chain step k-1.
    //========================================================================
    reg  [MODEL_DIM*16-1:0] h_chain [0:DRAFT_K];      // one extra: final h_mtp
    reg  [TOKW-1:0]         draft_id [0:DRAFT_K-1];
    reg  [TOKW-1:0]         truth_id [0:DRAFT_K];      // K+1 verify argmaxes

    // packed views spec_decode_seq consumes (row-major, step 0 in LSBs)
    reg [DRAFT_K*TOKW-1:0]     draft_q;
    reg [(DRAFT_K+1)*TOKW-1:0] truth_q;
    always @* begin
        for (ci = 0; ci < DRAFT_K; ci = ci + 1)
            draft_q[ci*TOKW +: TOKW] = draft_id[ci];
        for (ci = 0; ci <= DRAFT_K; ci = ci + 1)
            truth_q[ci*TOKW +: TOKW] = truth_id[ci];
    end

    // embedding pull : answer emb_t1 = embed(prev_tok) during the chain-prep beat.
    assign em_req = 1'b1;
    assign em_tok = prev_tok;

    //========================================================================
    // u_main : the FULL FP8 model at PE_M=1 -- one main pass -> verified m_0 and
    //   the hidden state h_state (final-RMSNorm output) that seeds the chain.
    //========================================================================
    reg                        main_start;
    wire                       main_busy, main_done;
    wire [VOCAB*16-1:0]        main_logits;
    wire [TOKW-1:0]            main_argmax;
    wire [MODEL_DIM*16-1:0]    main_hstate;
    reg  [TOKW-1:0]            main_tok;

    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PE_M(1)
    ) u_main (
        .clk(clk), .rst(rst), .start(main_start), .busy(main_busy), .done(main_done),
        .token_id(main_tok), .pos(pos_q), .s_len(slen_q),
        .logits(main_logits), .argmax(main_argmax),
        .em_req(m_em_req), .em_tok(m_em_tok), .em_idx(m_em_idx), .em_val(m_em_val),
        .db_layer(m_db_layer), .idx_fresh(m_idx_fresh), .idx_win(m_idx_win),
        .gn_req(m_gn_req), .gn_which(m_gn_which), .gn_idx(m_gn_idx), .gn_val(m_gn_val),
        .aw_req(m_aw_req), .aw_sel(m_aw_sel), .aw_grp(m_aw_grp), .aw_k(m_aw_k),
        .aw_col(m_aw_col), .aw_scale(m_aw_scale),
        .kc_req(m_kc_req), .kc_idx(m_kc_idx), .kc_ckv(m_kc_ckv),
        .kc_krope(m_kc_krope), .kc_valid(m_kc_valid),
        .rw_req(m_rw_req), .rw_k(m_rw_k), .rw_col(m_rw_col), .rw_scale(m_rw_scale),
        .fw_req(m_fw_req), .fw_sel(m_fw_sel), .fw_grp(m_fw_grp), .fw_k(m_fw_k),
        .fw_shared(m_fw_shared), .fw_eidx(m_fw_eidx),
        .fw_col(m_fw_col), .fw_col_up(m_fw_col_up),
        .fw_scale_g(m_fw_scale_g), .fw_scale_u(m_fw_scale_u),
        .fn_req(m_fn_req), .fn_idx(m_fn_idx), .fn_val(m_fn_val),
        .lw_req(m_lw_req), .lw_vtile(m_lw_vtile), .lw_k(m_lw_k), .lw_col(m_lw_col),
        .h_state(main_hstate)
    );

    //========================================================================
    // u_mtp : ONE mtp_head_fp8 serially reused across the K chain steps.  Its
    //   h_t input is the phase-selected chain state; its h_mtp output (P1.3b) is
    //   captured and fed back as the next step's h_t.  emb_t1 = embed(prev token).
    //========================================================================
    reg                        mtp_start;
    reg  [KW:0]                 chain_k;         // current chain step 0..K
    wire                       mtp_busy, mtp_done;
    wire [VOCAB*16-1:0]        mtp_logits;
    wire [TOKW-1:0]            mtp_argmax;
    wire [MODEL_DIM*16-1:0]    mtp_h_mtp;        // *** the chained hidden state ***
    reg  [MODEL_DIM*16-1:0]    mtp_h_t;          // selected chain state
    reg  [MODEL_DIM*16-1:0]    mtp_emb;          // embed(prev token)

    mtp_head_fp8 #(
        .MODEL_DIM(MODEL_DIM), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PROJ_TN(PROJ_TN)
    ) u_mtp (
        .clk(clk), .rst(rst), .start(mtp_start), .busy(mtp_busy), .done(mtp_done),
        .mode(mode_q), .pos(pos_q), .s_len(slen_q),
        .h_t(mtp_h_t), .emb_t1(mtp_emb),
        .logits(mtp_logits), .argmax(mtp_argmax),
        .h_mtp(mtp_h_mtp),                       // *** P1.3b chain state out ***
        .cn_req(t_cn_req), .cn_which(t_cn_which), .cn_idx(t_cn_idx), .cn_val(t_cn_val),
        .pw_req(t_pw_req), .pw_ptile(t_pw_ptile), .pw_k(t_pw_k),
        .pw_col(t_pw_col), .pw_scale(t_pw_scale),
        .gn_req(t_gn_req), .gn_which(t_gn_which), .gn_idx(t_gn_idx), .gn_val(t_gn_val),
        .aw_req(t_aw_req), .aw_sel(t_aw_sel), .aw_grp(t_aw_grp), .aw_k(t_aw_k),
        .aw_col(t_aw_col), .aw_scale(t_aw_scale),
        .kc_req(t_kc_req), .kc_idx(t_kc_idx), .kc_ckv(t_kc_ckv),
        .kc_krope(t_kc_krope), .kc_valid(t_kc_valid),
        .rw_req(t_rw_req), .rw_k(t_rw_k), .rw_col(t_rw_col), .rw_scale(t_rw_scale),
        .fw_req(t_fw_req), .fw_sel(t_fw_sel), .fw_grp(t_fw_grp), .fw_k(t_fw_k),
        .fw_shared(t_fw_shared), .fw_eidx(t_fw_eidx),
        .fw_col(t_fw_col), .fw_col_up(t_fw_col_up),
        .fw_scale_g(t_fw_scale_g), .fw_scale_u(t_fw_scale_u),
        .lw_req(t_lw_req), .lw_vtile(t_lw_vtile), .lw_k(t_lw_k), .lw_col(t_lw_col)
    );

    //========================================================================
    // u_verify : the FULL FP8 model at PE_M=K+1 -- verify {cur_tok,d_0..d_{K-1}}
    //   in ONE weight-load -> {m_0 .. m_K} (spec_batched_top PE_M weight-share).
    //========================================================================
    reg                        ver_start;
    wire                       ver_busy, ver_done;
    wire [B*VOCAB*16-1:0]      ver_logits;
    wire [B*TOKW-1:0]          ver_argmax;
    wire [B*MODEL_DIM*16-1:0]  ver_hstate;
    reg  [B*TOKW-1:0]          ver_tok;         // {cur_tok, d_0 .. d_{K-1}}

    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PE_M(B)
    ) u_verify (
        .clk(clk), .rst(rst), .start(ver_start), .busy(ver_busy), .done(ver_done),
        .token_id(ver_tok), .pos(pos_q), .s_len(slen_q),
        .logits(ver_logits), .argmax(ver_argmax),
        .em_req(v_em_req), .em_tok(v_em_tok), .em_idx(v_em_idx), .em_val(v_em_val),
        .db_layer(v_db_layer), .idx_fresh(v_idx_fresh), .idx_win(v_idx_win),
        .gn_req(v_gn_req), .gn_which(v_gn_which), .gn_idx(v_gn_idx), .gn_val(v_gn_val),
        .aw_req(v_aw_req), .aw_sel(v_aw_sel), .aw_grp(v_aw_grp), .aw_k(v_aw_k),
        .aw_col(v_aw_col), .aw_scale(v_aw_scale),
        .kc_req(v_kc_req), .kc_idx(v_kc_idx), .kc_ckv(v_kc_ckv),
        .kc_krope(v_kc_krope), .kc_valid(v_kc_valid),
        .rw_req(v_rw_req), .rw_k(v_rw_k), .rw_col(v_rw_col), .rw_scale(v_rw_scale),
        .fw_req(v_fw_req), .fw_sel(v_fw_sel), .fw_grp(v_fw_grp), .fw_k(v_fw_k),
        .fw_shared(v_fw_shared), .fw_eidx(v_fw_eidx),
        .fw_col(v_fw_col), .fw_col_up(v_fw_col_up),
        .fw_scale_g(v_fw_scale_g), .fw_scale_u(v_fw_scale_u),
        .fn_req(v_fn_req), .fn_idx(v_fn_idx), .fn_val(v_fn_val),
        .lw_req(v_lw_req), .lw_vtile(v_lw_vtile), .lw_k(v_lw_k), .lw_col(v_lw_col),
        .h_state(ver_hstate)
    );

    //========================================================================
    // u_seq : the accept/reject controller (batch path: commit the longest
    //   accepted prefix of {m_0..m_K} against drafts {d_0..d_{K-1}}).
    //========================================================================
    reg                        seq_arm;
    reg                        seq_pass;
    reg  [TOKW-1:0]            seq_verified;
    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(DRAFT_K)) u_seq (
        .clk(clk), .rst(rst), .start(seq_arm),
        .pass_valid(seq_pass), .verified_tok(seq_verified),
        .draft_tok(draft_id[0]), .draft_present(1'b0),
        .commit_valid(commit_valid), .commit_tok(commit_tok), .accepted(accepted),
        .total_tokens(total_tokens), .main_passes(main_passes),
        .accepts(accepts), .rejects(rejects),
        .draft_vec(draft_q), .truth_vec(truth_q), .n_draft(DRAFT_K[DKW-1:0])
    );

    //========================================================================
    // LOCAL longest-accepted-prefix (cursor advance only) -- ported verbatim
    //   from spec_batched_top: the SAME function spec_decode_seq uses, so
    //   cur_tok/cur_pos track the committed frontier (m_{p} = last committed
    //   token; +p+1 positions advanced).  The chain always mints the full
    //   DRAFT_K drafts, so n_draft is fixed at K here.
    //========================================================================
    function automatic [OCW-1:0] acc_prefix;
        input [DRAFT_K*TOKW-1:0]     dv;
        input [(DRAFT_K+1)*TOKW-1:0] tv;
        input [OCW-1:0]              ndi;
        integer       fj;
        reg           fb;
        reg [OCW-1:0] fp;
        begin
            fp = {OCW{1'b0}};
            fb = 1'b0;
            for (fj = 0; fj < DRAFT_K; fj = fj + 1) begin
                if (!fb && (fj < ndi) &&
                    (dv[fj*TOKW +: TOKW] == tv[fj*TOKW +: TOKW]))
                    fp = fp + 1'b1;
                else
                    fb = 1'b1;
            end
            acc_prefix = fp;
        end
    endfunction

    localparam [OCW-1:0] K_OCW = DRAFT_K[OCW-1:0];
    wire [OCW-1:0]  pfx_w        = acc_prefix(draft_q, truth_q, K_OCW);   // p (0..K)
    // next committed-frontier token = m_{p} = argmax row p
    wire [TOKW-1:0] frontier_tok = truth_q[TOKW*pfx_w +: TOKW];
    // position advance p+1 (zero-extended to POSW)
    wire [POSW-1:0] pos_adv      = {{(POSW-OCW){1'b0}}, pfx_w}
                                 + {{(POSW-1){1'b0}}, 1'b1};

    //========================================================================
    // MASTER FSM  (five phases: MAIN -> PREP/CHAIN xK -> VERIFY -> FEED)
    //========================================================================
    localparam [2:0]
        C_IDLE   = 3'd0,
        C_MAIN   = 3'd1,    // run PE_M=1 main pass -> m_0, h_0
        C_PREP   = 3'd2,    // capture embed(prev tok); launch a chain step
        C_CHAIN  = 3'd3,    // wait mtp_head_fp8 done, chaining h_mtp
        C_VERIFY = 3'd4,    // run PE_M=K+1 batched verify -> m_0..m_K
        C_FEED   = 3'd5,    // pulse spec_decode_seq; advance cursor
        C_DRAIN  = 3'd6,    // drain final pass's bonus commit beats
        C_DONE   = 3'd7;
    reg [2:0] state;
    reg       ver_wait;     // 1 = verify batch launched, waiting on ver_done
    reg [7:0] drain_cnt;    // drain the final pass's commit beats

    always @(posedge clk) begin
        if (rst) begin
            state       <= C_IDLE;
            busy        <= 1'b0;
            done        <= 1'b0;
            main_start  <= 1'b0;
            mtp_start   <= 1'b0;
            ver_start   <= 1'b0;
            ver_wait    <= 1'b0;
            seq_arm     <= 1'b0;
            seq_pass    <= 1'b0;
            seq_verified<= {TOKW{1'b0}};
            cur_tok     <= {TOKW{1'b0}};
            cur_pos     <= {POSW{1'b0}};
            pos_q       <= {POSW{1'b0}};
            slen_q      <= {(IDXW+1){1'b0}};
            mode_q      <= 1'b0;
            passes_left <= 16'h0;
            drain_cnt   <= 8'd0;
            chain_k     <= {(KW+1){1'b0}};
            prev_tok    <= {TOKW{1'b0}};
            main_tok    <= {TOKW{1'b0}};
            mtp_h_t     <= {MODEL_DIM*16{1'b0}};
            mtp_emb     <= {MODEL_DIM*16{1'b0}};
            for (ci = 0; ci <= DRAFT_K; ci = ci + 1) begin
                h_chain[ci]  <= {MODEL_DIM*16{1'b0}};
                truth_id[ci] <= {TOKW{1'b0}};
            end
            for (ci = 0; ci < DRAFT_K; ci = ci + 1)
                draft_id[ci] <= {TOKW{1'b0}};
        end else begin
            // ---- default pulse deassert ----
            done       <= 1'b0;
            main_start <= 1'b0;
            mtp_start  <= 1'b0;
            ver_start  <= 1'b0;
            seq_arm    <= 1'b0;
            seq_pass   <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            C_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy        <= 1'b1;
                    cur_tok     <= prompt_tok;
                    cur_pos     <= start_pos;
                    pos_q       <= start_pos;
                    slen_q      <= s_len;
                    mode_q      <= mtp_mode;
                    passes_left <= num_passes;
                    seq_arm     <= 1'b1;         // arm the commit controller
                    main_tok    <= prompt_tok;
                    main_start  <= 1'b1;
                    state       <= C_MAIN;
                end
            end
            //---------------------------------------------------------------- main pass
            C_MAIN: begin
                if (main_done) begin
                    // m_0 = verified argmax ; h_0 = final-norm hidden state
                    truth_id[0] <= main_argmax;
                    h_chain[0]  <= main_hstate;
                    // seed chain step 0 : h_t = h_0, emb = embed(m_0)
                    chain_k     <= {(KW+1){1'b0}};
                    mtp_h_t     <= main_hstate;
                    prev_tok    <= main_argmax;       // -> em_vec = embed(m_0)
                    pos_q       <= cur_pos + 1'b1;    // chain predicts pos+1+k
                    state       <= C_PREP;
                end
            end
            //---------------------------------------------------------------- chain prep
            // em_vec has settled to embed(prev_tok); capture it and launch the head.
            C_PREP: begin
                mtp_emb   <= em_vec;
                mtp_start <= 1'b1;
                state     <= C_CHAIN;
            end
            //---------------------------------------------------------------- K-step MTP chain
            // Each finished head: capture d_k = argmax and h_mtp -> h_chain[k+1].
            // Feed h_chain[k+1] as the next step's h_t (*** the P1.3b chain ***).
            C_CHAIN: begin
                if (mtp_done) begin
                    draft_id[chain_k[KW-1:0]] <= mtp_argmax;
                    h_chain[chain_k + 1'b1]   <= mtp_h_mtp;   // chain the state
                    if (chain_k == (DRAFT_K[KW:0] - 1'b1)) begin
                        // all K drafts minted -> assemble the verify batch
                        pos_q    <= cur_pos;        // verify batch shares base pos
                        state    <= C_VERIFY;
                        ver_wait <= 1'b0;
                    end else begin
                        // advance to the next chain step
                        chain_k   <= chain_k + 1'b1;
                        mtp_h_t   <= mtp_h_mtp;               // feed chained state
                        prev_tok  <= mtp_argmax;              // emb(d_k) next
                        pos_q     <= pos_q + 1'b1;
                        state     <= C_PREP;
                    end
                end
            end
            //---------------------------------------------------------------- batched verify
            C_VERIFY: begin
                if (!ver_wait) begin
                    ver_start <= 1'b1;              // launch PE_M=K+1 pass (1 pulse)
                    ver_wait  <= 1'b1;
                end else if (ver_done) begin
                    for (ci = 0; ci <= DRAFT_K; ci = ci + 1)
                        truth_id[ci] <= ver_argmax[ci*TOKW +: TOKW];
                    state <= C_FEED;
                end
            end
            //---------------------------------------------------------------- feed controller
            // commit longest accepted prefix; advance cursor over the WHOLE
            // accepted frontier (m_{p}, +p+1 positions) -- ported from
            // spec_batched_top B_FEED so a multi-pass run no longer restarts
            // inside already-committed tokens.
            C_FEED: begin
                seq_verified <= frontier_tok;
                seq_pass     <= 1'b1;
                cur_tok      <= frontier_tok;        // m_{p} (last committed token)
                cur_pos      <= cur_pos + pos_adv;   // advance p+1 positions
                pos_q        <= cur_pos + pos_adv;
                passes_left  <= passes_left - 1'b1;
                if (passes_left <= 16'h1) begin
                    drain_cnt <= 8'd0;
                    state     <= C_DRAIN;
                end else begin
                    // next pass: re-launch the main model on the new cursor
                    main_tok   <= frontier_tok;
                    main_start <= 1'b1;
                    state      <= C_MAIN;
                end
            end
            //---------------------------------------------------------------- drain
            // The final pass_valid drains up to K bonus commit beats over the next
            // p idle cycles; hold K+1 cycles so they all land before going idle
            // (mirrors spec_batched_top B_DRAIN -- 'done' no longer races them).
            C_DRAIN: begin
                if (drain_cnt == DRAFT_K[7:0]) state <= C_DONE;
                else drain_cnt <= drain_cnt + 8'd1;
            end
            //----------------------------------------------------------------
            C_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= C_IDLE;
            end
            default: state <= C_IDLE;
            endcase
        end
    end

    //========================================================================
    // PE_M=K+1 verify batch token vector : {cur_tok, d_0 .. d_{K-1}}, row-major
    // (row 0 = cur_tok in the LSBs).  Registered so it stays comb-loop-free and
    // holds stable across the model's serial embed phase.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            ver_tok <= {B*TOKW{1'b0}};
        end else begin
            ver_tok[0 +: TOKW] <= cur_tok;
            for (ci = 0; ci < DRAFT_K; ci = ci + 1)
                ver_tok[(ci+1)*TOKW +: TOKW] <= draft_id[ci];
        end
    end

    //========================================================================
    // unused-net tie-off
    //========================================================================
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, main_busy, main_logits, mtp_busy, mtp_logits,
        ver_busy, ver_logits, ver_hstate, h_chain[DRAFT_K]};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
/* verilator lint_on DECLFILENAME */
