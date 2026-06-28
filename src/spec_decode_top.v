`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// spec_decode_top.v  --  GLM-5.2 MTP SPECULATIVE-DECODE LOOP TOP
//                        (docs/SYSTEM_SINGLE_PACKAGE.md -- the FP8 throughput lever)
//----------------------------------------------------------------------------
// PURPOSE
//   The on-chip autoregressive decode loop that ties the three verified units
//   together into DeepSeek-V3-style K=1 MTP speculative decoding:
//
//     * glm_model_fp8  (u_main) -- the MAIN model: one full FP8 forward pass ->
//                       the VERIFIED next token (argmax) AND the hidden state h_t
//                       (its final-RMSNorm output, exposed via the additive
//                       h_state port -- the LM-head input vector).
//     * mtp_head_fp8   (u_mtp)  -- the DRAFT head: given (h_t, embed(verified))
//                       it produces the speculative t+2 DRAFT token (argmax).
//     * spec_decode_seq(u_seq)  -- the accept/reject CONTROLLER: per main pass it
//                       commits the verified token (+ a bonus when the prior
//                       draft was confirmed) and maintains total_tokens /
//                       main_passes / accepts / rejects.
//
//   This module is a PURE ORCHESTRATOR.  It instantiates the three committed
//   units, reimplements NONE of their math, and routes every weight / cache /
//   embedding PULL interface straight up to the top so the system (TB) answers
//   them.  The main-model pulls (m_*) and the MTP-head pulls (t_*) are exposed as
//   SEPARATE, independently-answerable port sets -- the two cores run at disjoint
//   times in the loop, but giving each its own pull set avoids any muxing /
//   contention and keeps the orchestrator combinational-loop-free.
//
//----------------------------------------------------------------------------
// THE LOOP (one main pass per outer iteration; K=1 draft)
//   token cursor: cur_tok @ cur_pos.  For pass i = 0 .. num_passes-1:
//     1. MAIN  : run glm_model_fp8(token_id=cur_tok, pos=cur_pos)
//                  -> verified = argmax , h_t = h_state.
//     2. EMBED : gather embed(verified) from the system (e_* pull) -> emb_vec.
//     3. MTP   : run mtp_head_fp8(h_t, emb_vec, pos=cur_pos)  -> draft = argmax.
//     4. FEED  : pulse spec_decode_seq.pass_valid with
//                  verified_tok = verified,
//                  draft_tok    = draft,
//                  draft_present= (i != 0).
//                The controller compares verified against the PRIOR pass's draft
//                (held internally) -> ACCEPT (commit 2) or REJECT (commit 1),
//                and emits the committed-token stream + counters.
//                Advance the cursor: cur_tok <= verified ; cur_pos <= cur_pos+1.
//   So the next main pass decodes the position the prior draft predicted -- i.e.
//   it VERIFIES that draft (its argmax IS the true token there), exactly the
//   semantics spec_decode_seq expects.  pass_valid pulses are naturally many
//   cycles apart (a full main+MTP run between them), so the controller's one-beat
//   accepted-draft "bonus" always has its idle slot before the next pass.
//
// DISCIPLINE: synchronous active-high reset; no latch; no combinational loop;
//   handshake-driven (every sub-unit launch waits that unit's done pulse).
//============================================================================
module spec_decode_top #(
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
    parameter integer DRAFT_K    = 1,           // GLM-5.2 ships K=1
    // ====================================================================
    // derived (do NOT override) -- copied verbatim from glm_model_fp8 /
    //                              mtp_head_fp8 so every pull port matches width.
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
    parameter integer PROJ_NB    = (CK + BLK - 1) / BLK
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
    // TOP-LEVEL EMBEDDING PULL for the MTP head's emb_t1 (= embed(verified)).
    //   Combinational, BF16, answered the SAME cycle: e_val = Emb[e_tok][e_idx].
    // ====================================================================
    output wire                          e_req,
    output wire [TOKW-1:0]               e_tok,
    output wire [DIMW-1:0]               e_idx,
    input  wire [15:0]                   e_val,

    // ====================================================================
    // MAIN-MODEL (glm_model_fp8) PULL INTERFACES -- routed up, prefix m_*
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
    // MTP-HEAD (mtp_head_fp8) PULL INTERFACES -- routed up, prefix t_*
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
    input  wire [LM_TN*16-1:0]           t_lw_col
);
    integer gi;

    //========================================================================
    // loop state
    //========================================================================
    reg                    m_start, t_start, seq_arm, seq_pass;
    reg  [TOKW-1:0]        cur_tok, verified_q, draft_q;
    reg  [POSW-1:0]       cur_pos;
    reg  [IDXW:0]         s_len_q;
    reg                    mtp_mode_q;
    reg                    dpresent_q;
    reg  [15:0]           pass_idx, npass_q;
    reg  [MODEL_DIM*16-1:0] hstate_q;          // latched main h_t (MTP input)

    // top embedding gather (embed(verified) -> emb_vec for the MTP head)
    reg  [15:0]           emb_buf [0:MODEL_DIM-1];
    reg  [MODEL_DIM*16-1:0] emb_vec;
    always @* begin
        for (gi = 0; gi < MODEL_DIM; gi = gi + 1)
            emb_vec[16*gi +: 16] = emb_buf[gi];
    end
    reg  [DIMW:0]         e_cnt;
    reg                    e_loading;
    assign e_req = e_loading;
    assign e_tok = verified_q;
    assign e_idx = e_cnt[DIMW-1:0];

    //========================================================================
    // sub-unit data wires
    //========================================================================
    wire                       m_busy, m_done;
    wire [VOCAB*16-1:0]        m_logits;
    wire [TOKW-1:0]            m_argmax;
    wire [MODEL_DIM*16-1:0]    m_hstate;

    wire                       t_busy, t_done;
    wire [VOCAB*16-1:0]        t_logits;
    wire [TOKW-1:0]            t_argmax;

    //========================================================================
    // u_main : the FULL FP8 main model (verified next token + hidden state).
    //========================================================================
    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN)
    ) u_main (
        .clk(clk), .rst(rst), .start(m_start), .busy(m_busy), .done(m_done),
        .token_id(cur_tok), .pos(cur_pos), .s_len(s_len_q),
        .logits(m_logits), .argmax(m_argmax),
        .em_req(m_em_req), .em_tok(m_em_tok), .em_idx(m_em_idx), .em_val(m_em_val),
        .db_layer(m_db_layer), .idx_fresh(m_idx_fresh), .idx_win(m_idx_win),
        .gn_req(m_gn_req), .gn_which(m_gn_which), .gn_idx(m_gn_idx), .gn_val(m_gn_val),
        .aw_req(m_aw_req), .aw_sel(m_aw_sel), .aw_grp(m_aw_grp), .aw_k(m_aw_k),
        .aw_col(m_aw_col), .aw_scale(m_aw_scale),
        .kc_req(m_kc_req), .kc_idx(m_kc_idx), .kc_ckv(m_kc_ckv), .kc_krope(m_kc_krope),
        .kc_valid(m_kc_valid),
        .rw_req(m_rw_req), .rw_k(m_rw_k), .rw_col(m_rw_col), .rw_scale(m_rw_scale),
        .fw_req(m_fw_req), .fw_sel(m_fw_sel), .fw_grp(m_fw_grp), .fw_k(m_fw_k),
        .fw_shared(m_fw_shared), .fw_eidx(m_fw_eidx),
        .fw_col(m_fw_col), .fw_col_up(m_fw_col_up),
        .fw_scale_g(m_fw_scale_g), .fw_scale_u(m_fw_scale_u),
        .fn_req(m_fn_req), .fn_idx(m_fn_idx), .fn_val(m_fn_val),
        .lw_req(m_lw_req), .lw_vtile(m_lw_vtile), .lw_k(m_lw_k), .lw_col(m_lw_col),
        .h_state(m_hstate)
    );

    //========================================================================
    // u_mtp : the FP8 MTP draft head (h_t + embed(verified) -> t+2 draft).
    //========================================================================
    mtp_head_fp8 #(
        .MODEL_DIM(MODEL_DIM), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PROJ_TN(PROJ_TN)
    ) u_mtp (
        .clk(clk), .rst(rst), .start(t_start), .busy(t_busy), .done(t_done),
        .mode(mtp_mode_q), .pos(cur_pos), .s_len(s_len_q),
        .h_t(hstate_q), .emb_t1(emb_vec),
        .logits(t_logits), .argmax(t_argmax),
        .cn_req(t_cn_req), .cn_which(t_cn_which), .cn_idx(t_cn_idx), .cn_val(t_cn_val),
        .pw_req(t_pw_req), .pw_ptile(t_pw_ptile), .pw_k(t_pw_k),
        .pw_col(t_pw_col), .pw_scale(t_pw_scale),
        .gn_req(t_gn_req), .gn_which(t_gn_which), .gn_idx(t_gn_idx), .gn_val(t_gn_val),
        .aw_req(t_aw_req), .aw_sel(t_aw_sel), .aw_grp(t_aw_grp), .aw_k(t_aw_k),
        .aw_col(t_aw_col), .aw_scale(t_aw_scale),
        .kc_req(t_kc_req), .kc_idx(t_kc_idx), .kc_ckv(t_kc_ckv), .kc_krope(t_kc_krope),
        .kc_valid(t_kc_valid),
        .rw_req(t_rw_req), .rw_k(t_rw_k), .rw_col(t_rw_col), .rw_scale(t_rw_scale),
        .fw_req(t_fw_req), .fw_sel(t_fw_sel), .fw_grp(t_fw_grp), .fw_k(t_fw_k),
        .fw_shared(t_fw_shared), .fw_eidx(t_fw_eidx),
        .fw_col(t_fw_col), .fw_col_up(t_fw_col_up),
        .fw_scale_g(t_fw_scale_g), .fw_scale_u(t_fw_scale_u),
        .lw_req(t_lw_req), .lw_vtile(t_lw_vtile), .lw_k(t_lw_k), .lw_col(t_lw_col)
    );

    //========================================================================
    // u_seq : the accept/reject controller (commit stream + counters).
    //========================================================================
    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(DRAFT_K)) u_seq (
        .clk(clk), .rst(rst), .start(seq_arm),
        .pass_valid(seq_pass), .verified_tok(verified_q),
        .draft_tok(draft_q), .draft_present(dpresent_q),
        .commit_valid(commit_valid), .commit_tok(commit_tok), .accepted(accepted),
        .total_tokens(total_tokens), .main_passes(main_passes),
        .accepts(accepts), .rejects(rejects)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, m_busy, t_busy, m_logits, t_logits};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // LOOP FSM
    //========================================================================
    localparam [3:0]
        T_IDLE   = 4'd0,
        T_MWAIT  = 4'd1,    // main pass running; wait m_done
        T_EMB    = 4'd2,    // gather embed(verified) -> emb_vec
        T_TWAIT  = 4'd3,    // MTP pass running; wait t_done
        T_FEED   = 4'd4,    // pulse pass_valid; advance cursor / next pass
        T_DRAIN  = 4'd5,    // let the last accepted-draft bonus beat emit
        T_DONE   = 4'd6;
    reg [3:0] state;

    always @(posedge clk) begin
        if (rst) begin
            state      <= T_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            m_start    <= 1'b0;
            t_start    <= 1'b0;
            seq_arm    <= 1'b0;
            seq_pass   <= 1'b0;
            cur_tok    <= {TOKW{1'b0}};
            verified_q <= {TOKW{1'b0}};
            draft_q    <= {TOKW{1'b0}};
            cur_pos    <= {POSW{1'b0}};
            s_len_q    <= {(IDXW+1){1'b0}};
            mtp_mode_q <= 1'b0;
            dpresent_q <= 1'b0;
            pass_idx   <= 16'd0;
            npass_q    <= 16'd0;
            hstate_q   <= {MODEL_DIM*16{1'b0}};
            e_cnt      <= {(DIMW+1){1'b0}};
            e_loading  <= 1'b0;
            for (gi = 0; gi < MODEL_DIM; gi = gi + 1)
                emb_buf[gi] <= 16'h0;
        end else begin
            // ---- default pulse deassert ----
            done     <= 1'b0;
            m_start  <= 1'b0;
            t_start  <= 1'b0;
            seq_arm  <= 1'b0;
            seq_pass <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            T_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy       <= 1'b1;
                    cur_tok    <= prompt_tok;
                    cur_pos    <= start_pos;
                    s_len_q    <= s_len;
                    mtp_mode_q <= mtp_mode;
                    npass_q    <= num_passes;
                    pass_idx   <= 16'd0;
                    seq_arm    <= 1'b1;       // arm the controller
                    m_start    <= 1'b1;       // launch the first main pass
                    state      <= T_MWAIT;
                end
            end
            //---------------------------------------------------------------- main
            T_MWAIT: begin
                if (m_done) begin
                    verified_q <= m_argmax;
                    hstate_q   <= m_hstate;   // latch h_t for the MTP head
                    e_loading  <= 1'b1;       // start embed(verified) gather
                    e_cnt      <= {(DIMW+1){1'b0}};
                    state      <= T_EMB;
                end
            end
            //---------------------------------------------------------------- embed gather
            // e_* is combinational (e_req/e_tok/e_idx from e_loading/verified_q/e_cnt);
            // e_val is answered the SAME cycle and stored on the NEXT edge.
            T_EMB: begin
                if (e_loading) begin
                    emb_buf[e_cnt[DIMW-1:0]] <= e_val;
                    if (e_cnt == MODEL_DIM[DIMW:0]-1'b1) begin
                        e_loading <= 1'b0;
                        t_start   <= 1'b1;    // launch the MTP draft pass
                        state     <= T_TWAIT;
                    end else begin
                        e_cnt <= e_cnt + 1'b1;
                    end
                end
            end
            //---------------------------------------------------------------- mtp
            T_TWAIT: begin
                if (t_done) begin
                    draft_q    <= t_argmax;
                    dpresent_q <= (pass_idx != 16'd0);  // prior draft to verify?
                    state      <= T_FEED;
                end
            end
            //---------------------------------------------------------------- feed controller
            // Pulse pass_valid with the latched (verified_q, draft_q, dpresent_q);
            // they stay stable until the next pass overwrites them.  Advance the
            // cursor so the next main pass decodes (== verifies) the drafted pos.
            T_FEED: begin
                seq_pass <= 1'b1;
                cur_tok  <= verified_q;
                cur_pos  <= cur_pos + 1'b1;
                if (pass_idx == npass_q - 16'd1) begin
                    state <= T_DRAIN;
                end else begin
                    pass_idx <= pass_idx + 16'd1;
                    m_start  <= 1'b1;         // launch the next main pass
                    state    <= T_MWAIT;
                end
            end
            //---------------------------------------------------------------- drain
            // pass_valid pulsed last cycle; an accepted pass queues its bonus beat
            // for THIS cycle.  Hold one cycle so it lands before we go idle.
            T_DRAIN: begin
                state <= T_DONE;
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

endmodule
/* verilator lint_on DECLFILENAME */
