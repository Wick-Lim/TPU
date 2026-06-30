`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// spec_batched_top.v  --  GLM-5.2-FP8 BATCHED-VERIFY SPECULATIVE-DECODE LOOP
//                         (docs/SYSTEM_SINGLE_PACKAGE.md -- the Flash/K lever)
//----------------------------------------------------------------------------
// PURPOSE  (KEY IDEA #5 -- ÷K weight-loads)
//   The committed K=1 loop (spec_decode_top) runs ONE full glm_model_fp8 weight-
//   load to verify ONE drafted position per pass.  This module verifies K drafted
//   positions in ONE weight-load by making the K+1 verify positions a PE_M=K+1
//   BATCH through a SINGLE glm_model_fp8 instance:
//
//     verify rows (PE_M = DRAFT_K+1):
//        row 0      = cur_tok   (the last committed token, position t)
//        row j (1..K) = d_j     (the j-th chained MTP draft, position t+j)
//
//   glm_model_fp8 at PE_M=K+1 pushes all K+1 rows through the model in LOCKSTEP:
//   ONE weight fetch per (layer, projection, expert) feeds every row (the aw_*/
//   rw_*/fw_*/kc_*/gn_*/fn_*/lw_* pull streams are IDENTICAL to a single-token
//   pass -- the model's documented PE_M weight-share contract).  So the whole
//   K+1-row verify costs ONE model weight-load, not K+1 serial single-token
//   passes  =>  weight-loads ÷ (K+1)  (the "Flash ÷K" of the brief).
//
//   The model returns K+1 per-row argmaxes:
//        argmax row j = m_{j+1} = the model's TRUE next-token at verify row j.
//   These are EXACTLY the truth_vec {m_1..m_{K+1}} that the committed
//   spec_decode_seq (DRAFT_K=K, batch path g_kn) consumes, alongside the K drafts
//   {d_1..d_K} (draft_vec).  spec_decode_seq commits the LONGEST ACCEPTED PREFIX
//   m_1..m_{p+1} (p = #leading drafts equal to their model argmax, stop at first
//   mismatch).  Because spec_decode_seq ONLY ever commits the model's OWN argmaxes
//   (truth_vec), never a raw draft, the committed stream is structurally a prefix
//   of the model's greedy stream for ANY K -- rejected drafts never commit.
//
//   The K drafts arrive on a TB-driven port (draft_in / n_draft); this module is
//   focused on the BATCHED-VERIFY mechanism and the ÷K weight-load (chaining an
//   mtp_head to MINT the drafts is orthogonal -- spec_decode_top already shows it).
//
//----------------------------------------------------------------------------
// THE LOOP  (one model weight-load per outer pass; K drafts verified per pass)
//   token cursor cur_tok @ cur_pos.  For pass i = 0 .. num_passes-1:
//     1. SETUP : latch the K drafts {d_1..d_K} (draft_in) + n_draft.
//     2. LAUNCH: assemble the PE_M=K+1 batch {cur_tok, d_1..d_K} and pulse
//                glm_model_fp8.start  -> ONE weight-load.  weight_loads++.
//     3. MWAIT : wait u_main.done -> capture the K+1 argmaxes {m_1..m_{K+1}}.
//     4. FEED  : pulse spec_decode_seq.pass_valid with draft_vec={d_1..d_K},
//                truth_vec={m_1..m_{K+1}}, n_draft -> commits m_1..m_{p+1}.
//                Advance: cur_tok <= m_{p+1} (last committed), cur_pos += p+1
//                (p = local longest-accepted-prefix, recomputed combinationally
//                with the SAME function spec_decode_seq uses -> the cursor tracks
//                the committed frontier).
//   pass_valid pulses are naturally many cycles apart (a full model pass between
//   them), so spec_decode_seq's drain of m_2..m_{p+1} (p<=K idle beats) always
//   finishes before the next pass; a final DRAIN state covers the last pass.
//
// DISCIPLINE: synchronous active-high reset; no latch; no combinational loop;
//   handshake-driven (the model pass waits u_main.done).  All weight / cache /
//   embedding delivery is via combinational PULL interfaces routed straight to
//   the top so the system (TB) answers them (ONE shared set -- single model).
//============================================================================
module spec_batched_top #(
    // ---- model / slice config (mirrors glm_model_fp8) ----
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
    // ---- DRAFT_K : drafted positions verified per ONE model weight-load (K>1) ----
    parameter integer DRAFT_K    = 2,
    // ====================================================================
    // derived (do NOT override) -- copied verbatim from glm_model_fp8 so every
    //                             routed-up pull port matches width.
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
    // spec_decode_seq batch-interface widths
    parameter integer DKW        = (DRAFT_K <= 1) ? 1 : $clog2(DRAFT_K + 1),
    parameter integer OCW        = $clog2(DRAFT_K + 2)   // 0..K+1 prefix count width
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    // ---- loop control ----
    input  wire                          start,      // 1-cycle pulse: begin loop
    input  wire [TOKW-1:0]               prompt_tok, // first token to decode from
    input  wire [POSW-1:0]               start_pos,  // first query position t
    input  wire [IDXW:0]                 s_len,      // S causal keys (<= S_MAX)
    input  wire [15:0]                   num_passes, // # batched-verify passes (>=1)

    // ---- K draft tokens for this pass (TB-driven; d_1..d_K = positions t+1..t+K)
    input  wire [DRAFT_K*TOKW-1:0]       draft_in,   // draft_in[j*TOKW +: TOKW]=d_{j+1}
    input  wire [DKW-1:0]                n_draft,    // #valid drafts this pass (<=K)

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

    // ---- weight-load counter : ONE model weight-load per K-verify pass ----
    output reg  [31:0]                   weight_loads,

    // ====================================================================
    // glm_model_fp8 PULL INTERFACES -- routed straight up (single shared model).
    //   ONE set answers ALL B=K+1 verify rows (the PE_M weight-share contract).
    // ====================================================================
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,
    output wire [DIMW-1:0]               em_idx,
    input  wire [15:0]                   em_val,
    output wire [LAYW-1:0]               db_layer,
    output wire                          idx_fresh,
    output wire [LAYW-1:0]               idx_win,
    output wire                          gn_req,
    output wire                          gn_which,
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*8-1:0]             aw_col,
    input  wire [16*PE_N*A_NB-1:0]       aw_scale,
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,
    input  wire                          kc_valid,
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [8*N_EXPERT-1:0]         rw_col,
    input  wire [16*N_EXPERT*R_NB-1:0]   rw_scale,
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [8*TN-1:0]               fw_col,
    input  wire [8*TN-1:0]               fw_col_up,
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_g,
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_u,
    output wire                          fn_req,
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col
);
    genvar gj;

    //========================================================================
    // loop state
    //========================================================================
    reg                    mdl_start, seq_arm, seq_pass;
    reg  [TOKW-1:0]        cur_tok;
    reg  [POSW-1:0]       cur_pos;
    reg  [IDXW:0]         s_len_q;
    reg  [15:0]           pass_idx, npass_q;
    reg  [DRAFT_K*TOKW-1:0]     draft_q;    // latched K drafts d_1..d_K (this pass)
    reg  [DKW-1:0]             nd_q;        // latched n_draft (this pass)
    reg  [B*TOKW-1:0]         truth_q;      // captured K+1 argmaxes m_1..m_{K+1}
    reg  [7:0]                drain_cnt;    // drain the final pass's commit beats

    //========================================================================
    // PE_M = K+1 verify batch token vector : {cur_tok, d_1..d_K}, row-major.
    //   row 0 = cur_tok (position t) ; row j+1 = d_{j+1} (position t+j+1).
    //   Built combinationally from the STABLE registers cur_tok + draft_q (cur_tok
    //   changes only at FEED, draft_q only at SETUP), so it holds for the whole
    //   model pass -- glm_model_fp8 reads token_id across its serial embed phase.
    //========================================================================
    wire [B*TOKW-1:0] mdl_token_id;
    assign mdl_token_id[TOKW*0 +: TOKW] = cur_tok;
    generate
    for (gj = 0; gj < DRAFT_K; gj = gj + 1) begin : G_TOK
        assign mdl_token_id[TOKW*(gj+1) +: TOKW] = draft_q[TOKW*gj +: TOKW];
    end
    endgenerate

    //========================================================================
    // u_main : the FULL FP8 model at PE_M = K+1 (ONE weight-load -> K+1 argmaxes).
    //========================================================================
    wire                  mdl_busy, mdl_done;
    wire [B*VOCAB*16-1:0] mdl_logits;
    wire [B*TOKW-1:0]     mdl_argmax;
    wire [B*MODEL_DIM*16-1:0] mdl_hstate;   // unused (verify only needs argmax)
    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PE_M(B)
    ) u_main (
        .clk(clk), .rst(rst), .start(mdl_start), .busy(mdl_busy), .done(mdl_done),
        .token_id(mdl_token_id), .pos(cur_pos), .s_len(s_len_q),
        .logits(mdl_logits), .argmax(mdl_argmax),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_col(aw_col), .aw_scale(aw_scale),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col), .rw_scale(rw_scale),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_col(fw_col), .fw_col_up(fw_col_up),
        .fw_scale_g(fw_scale_g), .fw_scale_u(fw_scale_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col),
        .h_state(mdl_hstate)
    );

    //========================================================================
    // u_seq : the batch longest-accepted-prefix controller (DRAFT_K = K, g_kn).
    //   verified_tok/draft_tok/draft_present are the K=1 ports -> dead at K>1
    //   (sunk inside spec_decode_seq); tie them to 0.
    //========================================================================
    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(DRAFT_K)) u_seq (
        .clk(clk), .rst(rst), .start(seq_arm),
        .pass_valid(seq_pass),
        .verified_tok({TOKW{1'b0}}), .draft_tok({TOKW{1'b0}}), .draft_present(1'b0),
        .commit_valid(commit_valid), .commit_tok(commit_tok), .accepted(accepted),
        .total_tokens(total_tokens), .main_passes(main_passes),
        .accepts(accepts), .rejects(rejects),
        .draft_vec(draft_q), .truth_vec(truth_q), .n_draft(nd_q)
    );

    //========================================================================
    // LOCAL longest-accepted-prefix (cursor advance only) -- the SAME function
    //   spec_decode_seq uses, so cur_tok/cur_pos track the committed frontier
    //   (m_{p+1} = last committed token; +p+1 positions advanced).
    //========================================================================
    function automatic [OCW-1:0] acc_prefix;
        input [DRAFT_K*TOKW-1:0]     dv;
        input [B*TOKW-1:0]           tv;
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
    /* verilator lint_off WIDTHEXPAND */
    wire [OCW-1:0] nd_ext = nd_q;
    /* verilator lint_on WIDTHEXPAND */
    wire [OCW-1:0] nd_w   = (nd_ext > K_OCW) ? K_OCW : nd_ext;
    wire [OCW-1:0] pfx_w  = acc_prefix(draft_q, truth_q, nd_w);   // p (0..K)
    // next committed-frontier token = m_{p+1} = argmax row p
    wire [TOKW-1:0] frontier_tok = truth_q[TOKW*pfx_w +: TOKW];
    // position advance p+1 (zero-extended to POSW)
    wire [POSW-1:0] pos_adv = {{(POSW-OCW){1'b0}}, pfx_w} + {{(POSW-1){1'b0}}, 1'b1};

    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, mdl_busy, mdl_logits, mdl_hstate};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // LOOP FSM
    //========================================================================
    localparam [2:0]
        B_IDLE   = 3'd0,
        B_SETUP  = 3'd1,    // latch this pass's K drafts + n_draft
        B_LAUNCH = 3'd2,    // assemble PE_M=K+1 batch; pulse model start
        B_MWAIT  = 3'd3,    // model pass running; wait done -> capture K+1 argmaxes
        B_FEED   = 3'd4,    // pulse pass_valid; advance cursor / next pass
        B_DRAIN  = 3'd5,    // drain final pass's commit beats
        B_DONE   = 3'd6;
    reg [2:0] state;

    always @(posedge clk) begin
        if (rst) begin
            state        <= B_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            mdl_start    <= 1'b0;
            seq_arm      <= 1'b0;
            seq_pass     <= 1'b0;
            cur_tok      <= {TOKW{1'b0}};
            cur_pos      <= {POSW{1'b0}};
            s_len_q      <= {(IDXW+1){1'b0}};
            pass_idx     <= 16'd0;
            npass_q      <= 16'd0;
            draft_q      <= {DRAFT_K*TOKW{1'b0}};
            nd_q         <= {DKW{1'b0}};
            truth_q      <= {B*TOKW{1'b0}};
            drain_cnt    <= 8'd0;
            weight_loads <= 32'd0;
        end else begin
            // ---- default pulse deassert ----
            done      <= 1'b0;
            mdl_start <= 1'b0;
            seq_arm   <= 1'b0;
            seq_pass  <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            B_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy     <= 1'b1;
                    cur_tok  <= prompt_tok;
                    cur_pos  <= start_pos;
                    s_len_q  <= s_len;
                    npass_q  <= num_passes;
                    pass_idx <= 16'd0;
                    seq_arm  <= 1'b1;            // arm the controller
                    state    <= B_SETUP;
                end
            end
            //---------------------------------------------------------------- latch drafts
            B_SETUP: begin
                draft_q <= draft_in;
                nd_q    <= n_draft;
                state   <= B_LAUNCH;
            end
            //---------------------------------------------------------------- launch model
            // ONE weight-load for the whole K+1-row verify batch.
            B_LAUNCH: begin
                mdl_start    <= 1'b1;
                weight_loads <= weight_loads + 32'd1;
                state        <= B_MWAIT;
            end
            //---------------------------------------------------------------- model wait
            B_MWAIT: begin
                if (mdl_done) begin
                    // capture the K+1 per-row argmaxes m_1..m_{K+1} (= truth_vec).
                    truth_q <= mdl_argmax;
                    state   <= B_FEED;
                end
            end
            //---------------------------------------------------------------- feed controller
            // Pulse pass_valid with {draft_q, truth_q, nd_q}; spec_decode_seq
            // commits m_1..m_{p+1}.  Advance the cursor to the committed frontier.
            B_FEED: begin
                seq_pass <= 1'b1;
                cur_tok  <= frontier_tok;        // m_{p+1} (last committed token)
                cur_pos  <= cur_pos + pos_adv;   // advance p+1 positions
                if (pass_idx == npass_q - 16'd1) begin
                    drain_cnt <= 8'd0;
                    state     <= B_DRAIN;
                end else begin
                    pass_idx <= pass_idx + 16'd1;
                    state    <= B_SETUP;
                end
            end
            //---------------------------------------------------------------- drain
            // The final pass_valid drains up to K bonus commit beats over the next
            // p idle cycles; hold K+1 cycles so they all land before going idle.
            B_DRAIN: begin
                if (drain_cnt == DRAFT_K[7:0]) state <= B_DONE;
                else drain_cnt <= drain_cnt + 8'd1;
            end
            //----------------------------------------------------------------
            B_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= B_IDLE;
            end
            default: state <= B_IDLE;
            endcase
        end
    end

endmodule
/* verilator lint_on DECLFILENAME */
