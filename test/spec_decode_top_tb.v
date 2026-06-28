`timescale 1ns/1ps
//============================================================================
// spec_decode_top_tb.v -- EXACTNESS TB for the MTP speculative-decode loop top.
//----------------------------------------------------------------------------
// WHAT THIS PROVES
//   Speculative decoding must be EXACT: the sequence of tokens it commits must
//   be IDENTICAL to what plain GREEDY decoding (argmax every step) of the SAME
//   model + SAME weights would produce.  This TB proves that token-for-token.
//
//   GOLDEN (reference greedy):  a STANDALONE glm_model_fp8 (u_ref) is driven
//     greedily by the TB -- run, take argmax, append, feed it back as the next
//     token at pos+1, repeat -- for N short steps from a fixed prompt + fixed
//     weights.  That argmax chain greedy[0..N-1] is the reference sequence.
//
//   DUT:  spec_decode_top (dut) is run with the SAME prompt + SAME weights
//     (the main-model pull set m_* is answered from the SAME weight ROMs that
//     feed u_ref, via an in_spec phase mux, so dut.u_main and u_ref are the
//     bit-identical model).  Its committed-token stream is captured.
//
//   THE LOOP'S COMMIT STREAM (spec_decode_seq contract, K=1):
//     every main pass commits its VERIFIED token (the main model's greedy
//     argmax for that position); when the PRIOR pass's MTP draft was confirmed,
//     an extra "bonus" beat is emitted one cycle later (the accepted draft).
//     The bonus beat is flagged by `accepted` pulsing on the verified beat that
//     precedes it.  So the committed stream is the VERIFIED subsequence (one
//     beat per main pass == the greedy tokens) with accepted-draft bonus beats
//     interleaved.  Speculative-decoding EXACTNESS == the verified subsequence
//     equals greedy, token-for-token.  When no draft is accepted (accepts==0)
//     the FULL committed stream is byte-identical to greedy and that stronger
//     equality is asserted too.
//
//   BINDING CHECKS ($fatal on violation, X-aware):
//     (1) verified-subsequence == greedy[0..N-1], token-for-token;
//     (2) #verified beats == num_passes == main_passes (no token lost/dup'd);
//     (3) #bonus beats == accepts ; #captured beats == total_tokens;
//     (4) acc+rej == passes-1 ; busy clears ; done pulses ; no X on any token.
//   REPORTED: the token count N, the sequence-identity verdict, accepts/rejects,
//   and the effective tokens/main-pass (= total_tokens/main_passes).
//
//   SIM DEPTH: each glm_model_fp8 forward is deep, so this uses the SMALL slice
//   the leaf units' own TBs use (MODEL_DIM=16/L=2/VOCAB=16) and a SHORT N=5
//   generation.  The orchestrator FSM + controller stream under test are
//   identical at any slice; only the leaf math is smaller, so the run finishes
//   under iverilog.  Random (deterministic) FP8 weights make the greedy
//   sequence non-degenerate so reorder/loss bugs are observable.
//============================================================================
module spec_decode_top_tb;
    // ---- SMALL-but-valid slice (mirrors the leaf units' own small TBs) ----
    localparam integer MODEL_DIM = 16, L = 2, N_DENSE = 1, VOCAB = 16;
    localparam integer H_HEADS = 2, NOPE = 4, ROPE = 4, V_DIM = 4;
    localparam integer Q_LORA = 8, KV_LORA = 8, S_MAX = 2, TOPK_ATTN = 2;
    localparam integer THETA = 8000000, PE_N = 2, POSW = 20, N_EXPERT = 4;
    localparam integer TOPK = 2, INTER_MOE = 8, INTER_DENSE = 32;
    localparam [31:0]  RSCALE = 32'h40200000;
    localparam integer TN = 4, BLK = 128, LM_TN = 4, PROJ_TN = 4;

    // ---- derived widths (mirror the DUT) ----
    localparam integer IDXW   = (S_MAX <= 1) ? 1 : $clog2(S_MAX);
    localparam integer QK_DIM = NOPE + ROPE;
    localparam integer HQK    = H_HEADS * QK_DIM;
    localparam integer HNOPE  = H_HEADS * NOPE;
    localparam integer HV     = H_HEADS * V_DIM;
    localparam integer EIDXW  = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT);
    localparam integer A_KMAX = (MODEL_DIM > Q_LORA) ?
                              ((MODEL_DIM > KV_LORA) ?
                               ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                             : ((KV_LORA > HV) ? KV_LORA : HV))
                            : ((Q_LORA > KV_LORA) ?
                               ((Q_LORA > HV) ? Q_LORA : HV)
                             : ((KV_LORA > HV) ? KV_LORA : HV));
    localparam integer A_OMAX = (HQK > MODEL_DIM) ?
                              ((HQK > HNOPE) ? ((HQK > HV) ? HQK : HV)
                                             : ((HNOPE > HV) ? HNOPE : HV))
                            : ((MODEL_DIM > HNOPE) ? ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                                                   : ((HNOPE > HV) ? HNOPE : HV));
    localparam integer A_NGMAX = (A_OMAX + PE_N - 1) / PE_N;
    localparam integer A_GRPW  = (A_NGMAX <= 1) ? 1 : $clog2(A_NGMAX);
    localparam integer A_KCW   = (A_KMAX  <= 1) ? 1 : $clog2(A_KMAX);
    localparam integer FF_GWD  = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1);
    localparam integer FF_KMAX_D = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM;
    localparam integer FF_KWD  = $clog2(FF_KMAX_D + 1);
    localparam integer FF_KMAX_M = (INTER_MOE > MODEL_DIM) ? INTER_MOE : MODEL_DIM;
    localparam integer R_KW    = $clog2(FF_KMAX_M + 1);
    localparam integer A_NB    = (A_KMAX    + BLK - 1) / BLK;
    localparam integer FF_NB_D = (FF_KMAX_D + BLK - 1) / BLK;
    localparam integer R_NB    = (FF_KMAX_M + BLK - 1) / BLK;
    localparam integer LAYW    = (L     <= 1) ? 1 : $clog2(L);
    localparam integer TOKW    = (VOCAB <= 1) ? 1 : $clog2(VOCAB);
    localparam integer DIMW    = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM);
    localparam integer NVTILE  = VOCAB / LM_TN;
    localparam integer VTW     = (NVTILE <= 1) ? 1 : $clog2(NVTILE);
    localparam integer CK      = 2 * MODEL_DIM;
    localparam integer CKIW    = $clog2(CK);
    localparam integer NPTILE  = MODEL_DIM / PROJ_TN;
    localparam integer PTW     = (NPTILE <= 1) ? 1 : $clog2(NPTILE);
    localparam integer PROJ_NB = (CK + BLK - 1) / BLK;

    localparam integer NGEN = 5;        // generated tokens (4-6 per the spec)
    localparam integer CAP  = 64;

    // ---- clock / reset ----
    reg clk = 1'b0;  always #5 clk = ~clk;
    reg rst;

    // phase select: 0 = greedy (u_ref active), 1 = spec (dut active).
    reg in_spec;

    //========================================================================
    // SHARED WEIGHT ROMs (main model) -- one copy, feeds BOTH u_ref and dut.m_*
    //========================================================================
    reg [15:0] EMB [0:VOCAB-1][0:MODEL_DIM-1];
    reg [15:0] GF  [0:MODEL_DIM-1];
    reg [15:0] Wlm [0:VOCAB-1][0:MODEL_DIM-1];
    reg [15:0] G1 [0:L-1][0:MODEL_DIM-1];
    reg [15:0] G2 [0:L-1][0:MODEL_DIM-1];
    reg [7:0] W_dq  [0:L-1][0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:L-1][0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:L-1][0:KV_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_kr  [0:L-1][0:ROPE-1][0:MODEL_DIM-1];
    reg [7:0] W_uk  [0:L-1][0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] W_uv  [0:L-1][0:HV-1][0:KV_LORA-1];
    reg [7:0] W_o   [0:L-1][0:MODEL_DIM-1][0:HV-1];
    reg [15:0] ScW_dq[0:L-1], ScW_uq[0:L-1], ScW_dkv[0:L-1], ScW_kr[0:L-1],
               ScW_uk[0:L-1], ScW_uv[0:L-1], ScW_o[0:L-1];
    reg [15:0] CKV [0:L-1][0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP [0:L-1][0:S_MAX-1][0:ROPE-1];
    reg [7:0]  Wg [0:L-1][0:MODEL_DIM-1][0:N_EXPERT-1];
    reg [15:0] ScWg[0:L-1];
    reg [7:0] Dg [0:L-1][0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [7:0] Du [0:L-1][0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [7:0] Dd [0:L-1][0:MODEL_DIM-1][0:INTER_DENSE-1];
    reg [15:0] ScDg [0:L-1][0:FF_NB_D-1];
    reg [15:0] ScDu [0:L-1][0:FF_NB_D-1];
    reg [15:0] ScDd [0:L-1][0:FF_NB_D-1];
    reg [7:0] Mg [0:L-1][0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] Mu [0:L-1][0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] Md [0:L-1][0:N_EXPERT-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [7:0] SHg [0:L-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] SHu [0:L-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] SHd [0:L-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [15:0] ScMg [0:L-1][0:N_EXPERT-1], ScMu [0:L-1][0:N_EXPERT-1], ScMd [0:L-1][0:N_EXPERT-1];
    reg [15:0] ScSHg[0:L-1], ScSHu[0:L-1], ScSHd[0:L-1];
    // MTP-head extra ROMs (combine proj + concat norms)
    reg [15:0] GA  [0:MODEL_DIM-1];
    reg [15:0] GB  [0:MODEL_DIM-1];
    reg [7:0]  Wp  [0:MODEL_DIM-1][0:CK-1];
    reg [15:0] ScWp[0:PROJ_NB-1];

    // ---- deterministic stimulus generators (independent, X-free) ----
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer hh; begin
        hh=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=hh[3];
        if (band==1) e=8'd125+hh[6:4]; else e=8'd124+hh[5:4];
        m=hh[12:6];
        gen_bf16={s,e,m};
    end endfunction
    function [7:0] gen_e4m3; input integer seed; input integer band;
        reg s; reg [3:0] e; reg [2:0] m; integer hh; begin
        hh=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=hh[3];
        if (band==1) e = 4'd7 + {3'b0,hh[4]}; else e = 4'd6 + {3'b0,hh[4]};
        m = hh[12:10];
        gen_e4m3 = {s,e,m};
    end endfunction
    function [15:0] gen_scale; input integer seed;
        reg [7:0] e; reg [6:0] m; integer hh; begin
        hh=(seed*2654435761)^(seed<<13)^(seed*22229);
        e = 8'd122 + {7'b0,hh[2]};
        m = hh[10:4];
        gen_scale={1'b0,e,m};
    end endfunction

    integer i,j,e,gl,sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc=seed0;
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin EMB[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (gl=0;gl<L;gl=gl+1) begin
            for (i=0;i<MODEL_DIM;i=i+1) begin G1[gl][i]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) begin G2[gl][i]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScW_dq[gl]=gen_scale(sc); sc=sc+1;  ScW_uq[gl]=gen_scale(sc); sc=sc+1;
            ScW_dkv[gl]=gen_scale(sc); sc=sc+1; ScW_kr[gl]=gen_scale(sc); sc=sc+1;
            ScW_uk[gl]=gen_scale(sc); sc=sc+1;  ScW_uv[gl]=gen_scale(sc); sc=sc+1;
            ScW_o[gl]=gen_scale(sc); sc=sc+1;
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[gl][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[gl][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<N_EXPERT;j=j+1) begin Wg[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScWg[gl]=gen_scale(sc); sc=sc+1;
            for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Dg[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Du[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_DENSE;j=j+1) begin Dd[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<FF_NB_D;i=i+1) begin ScDg[gl][i]=gen_scale(sc); sc=sc+1; end
            for (i=0;i<FF_NB_D;i=i+1) begin ScDu[gl][i]=gen_scale(sc); sc=sc+1; end
            for (i=0;i<FF_NB_D;i=i+1) begin ScDd[gl][i]=gen_scale(sc); sc=sc+1; end
            for (e=0;e<N_EXPERT;e=e+1) begin
                for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mg[gl][e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
                for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mu[gl][e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
                for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin Md[gl][e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
                ScMg[gl][e]=gen_scale(sc); sc=sc+1; ScMu[gl][e]=gen_scale(sc); sc=sc+1; ScMd[gl][e]=gen_scale(sc); sc=sc+1;
            end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHg[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHu[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin SHd[gl][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScSHg[gl]=gen_scale(sc); sc=sc+1; ScSHu[gl]=gen_scale(sc); sc=sc+1; ScSHd[gl]=gen_scale(sc); sc=sc+1;
        end
        for (i=0;i<MODEL_DIM;i=i+1) begin GF[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Wlm[i][j]=gen_bf16(sc,band); sc=sc+1; end
        // MTP-head extras (only affect the DRAFT, never the verified sequence)
        for (i=0;i<MODEL_DIM;i=i+1) begin GA[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin GB[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<CK;j=j+1) begin Wp[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<PROJ_NB;i=i+1) begin ScWp[i]=gen_scale(sc); sc=sc+1; end
    end endtask

    //========================================================================
    // u_ref (standalone greedy reference) port wires
    //========================================================================
    reg               ref_start;
    reg  [TOKW-1:0]   ref_token;
    reg  [POSW-1:0]   ref_pos;
    reg  [IDXW:0]     ref_slen;
    wire              ref_busy, ref_done;
    wire [VOCAB*16-1:0] ref_logits;
    wire [TOKW-1:0]   ref_argmax;
    wire [MODEL_DIM*16-1:0] ref_hstate;
    wire              r_em_req;  wire [TOKW-1:0] r_em_tok; wire [DIMW-1:0] r_em_idx;
    wire [LAYW-1:0]   r_db_layer; wire r_idx_fresh; wire [LAYW-1:0] r_idx_win;
    wire              r_gn_req, r_gn_which; wire [DIMW-1:0] r_gn_idx;
    wire              r_aw_req; wire [3:0] r_aw_sel; wire [A_GRPW-1:0] r_aw_grp; wire [A_KCW-1:0] r_aw_k;
    wire              r_kc_req; wire [IDXW-1:0] r_kc_idx;
    wire              r_rw_req; wire [R_KW-1:0] r_rw_k;
    wire              r_fw_req; wire [1:0] r_fw_sel; wire [FF_GWD-1:0] r_fw_grp; wire [FF_KWD-1:0] r_fw_k;
    wire              r_fw_shared; wire [EIDXW-1:0] r_fw_eidx;
    wire              r_fn_req; wire [DIMW-1:0] r_fn_idx;
    wire              r_lw_req; wire [VTW-1:0] r_lw_vtile; wire [DIMW-1:0] r_lw_k;

    //========================================================================
    // dut (spec_decode_top) port wires
    //========================================================================
    reg            start;
    reg [TOKW-1:0] prompt_tok;
    reg [POSW-1:0] start_pos;
    reg [IDXW:0]   s_len;
    reg            mtp_mode;
    reg [15:0]     num_passes;
    wire busy, done;
    wire commit_valid, accepted; wire [TOKW-1:0] commit_tok;
    wire [31:0] total_tokens, main_passes, accepts, rejects;
    wire              e_req;     wire [TOKW-1:0] e_tok;  wire [DIMW-1:0] e_idx;
    wire              m_em_req;  wire [TOKW-1:0] m_em_tok; wire [DIMW-1:0] m_em_idx;
    wire [LAYW-1:0]   m_db_layer; wire m_idx_fresh; wire [LAYW-1:0] m_idx_win;
    wire              m_gn_req, m_gn_which; wire [DIMW-1:0] m_gn_idx;
    wire              m_aw_req; wire [3:0] m_aw_sel; wire [A_GRPW-1:0] m_aw_grp; wire [A_KCW-1:0] m_aw_k;
    wire              m_kc_req; wire [IDXW-1:0] m_kc_idx;
    wire              m_rw_req; wire [R_KW-1:0] m_rw_k;
    wire              m_fw_req; wire [1:0] m_fw_sel; wire [FF_GWD-1:0] m_fw_grp; wire [FF_KWD-1:0] m_fw_k;
    wire              m_fw_shared; wire [EIDXW-1:0] m_fw_eidx;
    wire              m_fn_req; wire [DIMW-1:0] m_fn_idx;
    wire              m_lw_req; wire [VTW-1:0] m_lw_vtile; wire [DIMW-1:0] m_lw_k;
    wire              t_cn_req; wire [1:0] t_cn_which; wire [DIMW-1:0] t_cn_idx;
    wire              t_pw_req; wire [PTW-1:0] t_pw_ptile; wire [CKIW-1:0] t_pw_k;
    wire              t_gn_req, t_gn_which; wire [DIMW-1:0] t_gn_idx;
    wire              t_aw_req; wire [3:0] t_aw_sel; wire [A_GRPW-1:0] t_aw_grp; wire [A_KCW-1:0] t_aw_k;
    wire              t_kc_req; wire [IDXW-1:0] t_kc_idx;
    wire              t_rw_req; wire [R_KW-1:0] t_rw_k;
    wire              t_fw_req; wire [1:0] t_fw_sel; wire [FF_GWD-1:0] t_fw_grp; wire [FF_KWD-1:0] t_fw_k;
    wire              t_fw_shared; wire [EIDXW-1:0] t_fw_eidx;
    wire              t_lw_req; wire [VTW-1:0] t_lw_vtile; wire [DIMW-1:0] t_lw_k;

    //========================================================================
    // PHASE-MUXED main-model pull inputs (greedy: u_ref ; spec: dut.u_main).
    //   The two models run at DISJOINT times, so one response set feeds both.
    //========================================================================
    wire [LAYW-1:0]   mu_layer = in_spec ? m_db_layer : r_db_layer;
    wire [TOKW-1:0]   mu_em_tok = in_spec ? m_em_tok : r_em_tok;
    wire [DIMW-1:0]   mu_em_idx = in_spec ? m_em_idx : r_em_idx;
    wire              mu_gn_which = in_spec ? m_gn_which : r_gn_which;
    wire [DIMW-1:0]   mu_gn_idx = in_spec ? m_gn_idx : r_gn_idx;
    wire [3:0]        mu_aw_sel = in_spec ? m_aw_sel : r_aw_sel;
    wire [A_GRPW-1:0] mu_aw_grp = in_spec ? m_aw_grp : r_aw_grp;
    wire [A_KCW-1:0]  mu_aw_k  = in_spec ? m_aw_k  : r_aw_k;
    wire [IDXW-1:0]   mu_kc_idx = in_spec ? m_kc_idx : r_kc_idx;
    wire              mu_kc_req = in_spec ? m_kc_req : r_kc_req;
    wire [R_KW-1:0]   mu_rw_k  = in_spec ? m_rw_k  : r_rw_k;
    wire [1:0]        mu_fw_sel = in_spec ? m_fw_sel : r_fw_sel;
    wire [FF_GWD-1:0] mu_fw_grp = in_spec ? m_fw_grp : r_fw_grp;
    wire [FF_KWD-1:0] mu_fw_k  = in_spec ? m_fw_k  : r_fw_k;
    wire              mu_fw_shared = in_spec ? m_fw_shared : r_fw_shared;
    wire [EIDXW-1:0]  mu_fw_eidx = in_spec ? m_fw_eidx : r_fw_eidx;
    wire [DIMW-1:0]   mu_fn_idx = in_spec ? m_fn_idx : r_fn_idx;
    wire [VTW-1:0]    mu_lw_vtile = in_spec ? m_lw_vtile : r_lw_vtile;
    wire [DIMW-1:0]   mu_lw_k  = in_spec ? m_lw_k  : r_lw_k;

    //========================================================================
    // main-model combinational pull responders (shared regs -> both models)
    //========================================================================
    integer t, re, ft, fo, cd, obd, lt, lay;
    reg [15:0] em_val, gn_val, fn_val, sc_a;
    reg [PE_N*8-1:0]   aw_col;   reg [16*PE_N*A_NB-1:0]  aw_scale;
    reg [KV_LORA*16-1:0] kc_ckv; reg [ROPE*16-1:0]       kc_krope; reg kc_valid;
    reg [8*N_EXPERT-1:0] rw_col; reg [16*N_EXPERT*R_NB-1:0] rw_scale;
    reg [8*TN-1:0] fw_col, fw_col_up; reg [16*TN*FF_NB_D-1:0] fw_scale_g, fw_scale_u;
    reg [LM_TN*16-1:0] lw_col;
    reg [15:0] kb1;

    always @* em_val = EMB[mu_em_tok][mu_em_idx];
    always @* fn_val = GF[mu_fn_idx];
    always @* gn_val = mu_gn_which ? G2[mu_layer][mu_gn_idx] : G1[mu_layer][mu_gn_idx];
    always @* begin
        lw_col = {LM_TN*16{1'b0}};
        for (lt=0;lt<LM_TN;lt=lt+1) lw_col[16*lt+:16] = Wlm[mu_lw_vtile*LM_TN + lt][mu_lw_k];
    end
    always @* begin
        lay = mu_layer;
        aw_col   = {PE_N*8{1'b0}};
        aw_scale = {16*PE_N*A_NB{1'b0}};
        for (t=0;t<PE_N;t=t+1) begin
            case (mu_aw_sel)
            4'd0: if (mu_aw_grp*PE_N+t < Q_LORA)   aw_col[8*t+:8]=W_dq [lay][mu_aw_grp*PE_N+t][mu_aw_k];
            4'd1: if (mu_aw_grp*PE_N+t < HQK)      aw_col[8*t+:8]=W_uq [lay][mu_aw_grp*PE_N+t][mu_aw_k];
            4'd2: if (mu_aw_grp*PE_N+t < KV_LORA)  aw_col[8*t+:8]=W_dkv[lay][mu_aw_grp*PE_N+t][mu_aw_k];
            4'd3: if (mu_aw_grp*PE_N+t < ROPE)     aw_col[8*t+:8]=W_kr [lay][mu_aw_grp*PE_N+t][mu_aw_k];
            4'd4: if (mu_aw_grp*PE_N+t < HNOPE)    aw_col[8*t+:8]=W_uk [lay][mu_aw_grp*PE_N+t][mu_aw_k];
            4'd5: if (mu_aw_grp*PE_N+t < HV)       aw_col[8*t+:8]=W_uv [lay][mu_aw_grp*PE_N+t][mu_aw_k];
            4'd6: if (mu_aw_grp*PE_N+t < MODEL_DIM)aw_col[8*t+:8]=W_o  [lay][mu_aw_grp*PE_N+t][mu_aw_k];
            default: aw_col[8*t+:8]=8'h0;
            endcase
        end
        case (mu_aw_sel)
            4'd0: sc_a=ScW_dq[lay]; 4'd1: sc_a=ScW_uq[lay]; 4'd2: sc_a=ScW_dkv[lay];
            4'd3: sc_a=ScW_kr[lay]; 4'd4: sc_a=ScW_uk[lay]; 4'd5: sc_a=ScW_uv[lay];
            4'd6: sc_a=ScW_o[lay]; default: sc_a=16'h3F80;
        endcase
        for (t=0;t<PE_N;t=t+1) aw_scale[16*t+:16]=sc_a;
    end
    always @* begin
        kc_ckv   = {KV_LORA*16{1'b0}};
        kc_krope = {ROPE*16{1'b0}};
        for (cd=0;cd<KV_LORA;cd=cd+1) kc_ckv[16*cd+:16]   = CKV[mu_layer][mu_kc_idx][cd];
        for (cd=0;cd<ROPE;cd=cd+1)    kc_krope[16*cd+:16] = KRP[mu_layer][mu_kc_idx][cd];
    end
    always @(posedge clk) begin
        if (rst) kc_valid <= 1'b0;
        else     kc_valid <= mu_kc_req;
    end
    always @* begin
        rw_col   = {8*N_EXPERT{1'b0}};
        rw_scale = {16*N_EXPERT*R_NB{1'b0}};
        for (re=0;re<N_EXPERT;re=re+1) begin
            rw_col[8*re+:8]    = Wg[mu_layer][mu_rw_k][re];
            rw_scale[16*re+:16]= ScWg[mu_layer];
        end
    end
    // FFN expert weight + scale.  FF_NB_D==1 at this slice (every GEMM is one
    // K-block), so only K-block 0 is populated; the FF_NB_D>1 path is written
    // generically (dead here) with a variable index so it never OOBs.
    always @* begin
        lay = mu_layer;
        kb1 = (FF_NB_D>1) ? 16'd1 : 16'd0;
        fw_col     = {8*TN{1'b0}};      fw_col_up  = {8*TN{1'b0}};
        fw_scale_g = {16*TN*FF_NB_D{1'b0}}; fw_scale_u = {16*TN*FF_NB_D{1'b0}};
        obd = (mu_fw_grp*TN) / BLK;
        for (ft=0;ft<TN;ft=ft+1) begin
            fo = mu_fw_grp*TN + ft;
            if (lay < N_DENSE) begin
                if (mu_fw_sel==2'd2) begin
                    if (fo<MODEL_DIM) fw_col[8*ft+:8]=Dd[lay][fo][mu_fw_k];
                end else if (fo<INTER_DENSE) begin
                    fw_col   [8*ft+:8]=Dg[lay][fo][mu_fw_k];
                    fw_col_up[8*ft+:8]=Du[lay][fo][mu_fw_k];
                end
            end else begin
                if (mu_fw_shared) begin
                    if (mu_fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[8*ft+:8]=SHd[lay][fo][mu_fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [8*ft+:8]=SHg[lay][fo][mu_fw_k];
                        fw_col_up[8*ft+:8]=SHu[lay][fo][mu_fw_k];
                    end
                end else begin
                    if (mu_fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[8*ft+:8]=Md[lay][mu_fw_eidx][fo][mu_fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [8*ft+:8]=Mg[lay][mu_fw_eidx][fo][mu_fw_k];
                        fw_col_up[8*ft+:8]=Mu[lay][mu_fw_eidx][fo][mu_fw_k];
                    end
                end
            end
        end
        for (ft=0;ft<TN;ft=ft+1) begin
            if (lay < N_DENSE) begin
                if (mu_fw_sel==2'd2) begin
                    fw_scale_g[16*(0*TN+ft)+:16]=ScDd[lay][0];
                    if (FF_NB_D>1) fw_scale_g[16*(1*TN+ft)+:16]=ScDd[lay][kb1];
                end else begin
                    fw_scale_g[16*(0*TN+ft)+:16]=ScDg[lay][obd];
                    fw_scale_u[16*(0*TN+ft)+:16]=ScDu[lay][obd];
                    if (FF_NB_D>1) begin
                        fw_scale_g[16*(1*TN+ft)+:16]=ScDg[lay][obd];
                        fw_scale_u[16*(1*TN+ft)+:16]=ScDu[lay][obd];
                    end
                end
            end else begin
                if (mu_fw_shared) begin
                    if (mu_fw_sel==2'd2) fw_scale_g[16*(0*TN+ft)+:16]=ScSHd[lay];
                    else begin
                        fw_scale_g[16*(0*TN+ft)+:16]=ScSHg[lay];
                        fw_scale_u[16*(0*TN+ft)+:16]=ScSHu[lay];
                    end
                end else begin
                    if (mu_fw_sel==2'd2) fw_scale_g[16*(0*TN+ft)+:16]=ScMd[lay][mu_fw_eidx];
                    else begin
                        fw_scale_g[16*(0*TN+ft)+:16]=ScMg[lay][mu_fw_eidx];
                        fw_scale_u[16*(0*TN+ft)+:16]=ScMu[lay][mu_fw_eidx];
                    end
                end
            end
        end
    end

    //========================================================================
    // MTP-head pull responders (single layer; read main ROMs at layer 0 +
    //   the MTP-only proj/concat-norm ROMs).  Only affect the DRAFT.
    //========================================================================
    localparam integer TMTPL = 0;                       // MTP decoder uses layer-0 weights
    reg [15:0] t_cn_val, t_gn_val;
    reg [PROJ_TN*8-1:0] t_pw_col; reg [16*PROJ_TN*PROJ_NB-1:0] t_pw_scale;
    reg [PE_N*8-1:0] t_aw_col; reg [16*PE_N*A_NB-1:0] t_aw_scale;
    reg [KV_LORA*16-1:0] t_kc_ckv; reg [ROPE*16-1:0] t_kc_krope; reg t_kc_valid;
    reg [8*N_EXPERT-1:0] t_rw_col; reg [16*N_EXPERT*R_NB-1:0] t_rw_scale;
    reg [8*TN-1:0] t_fw_col, t_fw_col_up; reg [16*TN*FF_NB_D-1:0] t_fw_scale_g, t_fw_scale_u;
    reg [LM_TN*16-1:0] t_lw_col;
    integer tt, tre, tft, tfo, tcd, tobd, tlt, tpq;
    reg [15:0] t_sca;

    always @* begin
        case (t_cn_which)
            2'd0:    t_cn_val = GA[t_cn_idx];
            2'd1:    t_cn_val = GB[t_cn_idx];
            default: t_cn_val = GF[t_cn_idx];
        endcase
    end
    always @* begin
        t_pw_col   = {PROJ_TN*8{1'b0}};
        t_pw_scale = {16*PROJ_TN*PROJ_NB{1'b0}};
        for (tpq=0; tpq<PROJ_TN; tpq=tpq+1)
            t_pw_col[8*tpq +: 8] = Wp[t_pw_ptile*PROJ_TN + tpq][t_pw_k];
        for (tcd=0; tcd<PROJ_NB; tcd=tcd+1)
            for (tpq=0; tpq<PROJ_TN; tpq=tpq+1)
                t_pw_scale[16*(tcd*PROJ_TN + tpq) +: 16] = ScWp[tcd];
    end
    always @* t_gn_val = t_gn_which ? G2[TMTPL][t_gn_idx] : G1[TMTPL][t_gn_idx];
    always @* begin
        t_aw_col   = {PE_N*8{1'b0}};
        t_aw_scale = {16*PE_N*A_NB{1'b0}};
        for (tt=0;tt<PE_N;tt=tt+1) begin
            case (t_aw_sel)
            4'd0: if (t_aw_grp*PE_N+tt < Q_LORA)   t_aw_col[8*tt+:8]=W_dq [TMTPL][t_aw_grp*PE_N+tt][t_aw_k];
            4'd1: if (t_aw_grp*PE_N+tt < HQK)      t_aw_col[8*tt+:8]=W_uq [TMTPL][t_aw_grp*PE_N+tt][t_aw_k];
            4'd2: if (t_aw_grp*PE_N+tt < KV_LORA)  t_aw_col[8*tt+:8]=W_dkv[TMTPL][t_aw_grp*PE_N+tt][t_aw_k];
            4'd3: if (t_aw_grp*PE_N+tt < ROPE)     t_aw_col[8*tt+:8]=W_kr [TMTPL][t_aw_grp*PE_N+tt][t_aw_k];
            4'd4: if (t_aw_grp*PE_N+tt < HNOPE)    t_aw_col[8*tt+:8]=W_uk [TMTPL][t_aw_grp*PE_N+tt][t_aw_k];
            4'd5: if (t_aw_grp*PE_N+tt < HV)       t_aw_col[8*tt+:8]=W_uv [TMTPL][t_aw_grp*PE_N+tt][t_aw_k];
            4'd6: if (t_aw_grp*PE_N+tt < MODEL_DIM)t_aw_col[8*tt+:8]=W_o  [TMTPL][t_aw_grp*PE_N+tt][t_aw_k];
            default: t_aw_col[8*tt+:8]=8'h0;
            endcase
        end
        case (t_aw_sel)
            4'd0: t_sca=ScW_dq[TMTPL]; 4'd1: t_sca=ScW_uq[TMTPL]; 4'd2: t_sca=ScW_dkv[TMTPL];
            4'd3: t_sca=ScW_kr[TMTPL]; 4'd4: t_sca=ScW_uk[TMTPL]; 4'd5: t_sca=ScW_uv[TMTPL];
            4'd6: t_sca=ScW_o[TMTPL]; default: t_sca=16'h3F80;
        endcase
        for (tt=0;tt<PE_N;tt=tt+1) t_aw_scale[16*tt+:16]=t_sca;
    end
    always @* begin
        t_kc_ckv   = {KV_LORA*16{1'b0}};
        t_kc_krope = {ROPE*16{1'b0}};
        for (tcd=0;tcd<KV_LORA;tcd=tcd+1) t_kc_ckv[16*tcd+:16]   = CKV[TMTPL][t_kc_idx][tcd];
        for (tcd=0;tcd<ROPE;tcd=tcd+1)    t_kc_krope[16*tcd+:16] = KRP[TMTPL][t_kc_idx][tcd];
    end
    always @(posedge clk) begin
        if (rst) t_kc_valid <= 1'b0;
        else     t_kc_valid <= t_kc_req;
    end
    always @* begin
        t_rw_col   = {8*N_EXPERT{1'b0}};
        t_rw_scale = {16*N_EXPERT*R_NB{1'b0}};
        for (tre=0;tre<N_EXPERT;tre=tre+1) begin
            t_rw_col[8*tre+:8]    = Wg[TMTPL][t_rw_k][tre];
            t_rw_scale[16*tre+:16]= ScWg[TMTPL];
        end
    end
    // MTP runs the decoder in mtp_mode (DENSE here, mtp_mode=0) -> dense FFN weights.
    always @* begin
        t_fw_col     = {8*TN{1'b0}};      t_fw_col_up  = {8*TN{1'b0}};
        t_fw_scale_g = {16*TN*FF_NB_D{1'b0}}; t_fw_scale_u = {16*TN*FF_NB_D{1'b0}};
        tobd = (t_fw_grp*TN) / BLK;
        for (tft=0;tft<TN;tft=tft+1) begin
            tfo = t_fw_grp*TN + tft;
            if (mtp_mode==1'b0) begin
                if (t_fw_sel==2'd2) begin
                    if (tfo<MODEL_DIM) t_fw_col[8*tft+:8]=Dd[TMTPL][tfo][t_fw_k];
                end else if (tfo<INTER_DENSE) begin
                    t_fw_col   [8*tft+:8]=Dg[TMTPL][tfo][t_fw_k];
                    t_fw_col_up[8*tft+:8]=Du[TMTPL][tfo][t_fw_k];
                end
            end else begin
                if (t_fw_shared) begin
                    if (t_fw_sel==2'd2) begin
                        if (tfo<MODEL_DIM) t_fw_col[8*tft+:8]=SHd[TMTPL][tfo][t_fw_k];
                    end else if (tfo<INTER_MOE) begin
                        t_fw_col   [8*tft+:8]=SHg[TMTPL][tfo][t_fw_k];
                        t_fw_col_up[8*tft+:8]=SHu[TMTPL][tfo][t_fw_k];
                    end
                end else begin
                    if (t_fw_sel==2'd2) begin
                        if (tfo<MODEL_DIM) t_fw_col[8*tft+:8]=Md[TMTPL][t_fw_eidx][tfo][t_fw_k];
                    end else if (tfo<INTER_MOE) begin
                        t_fw_col   [8*tft+:8]=Mg[TMTPL][t_fw_eidx][tfo][t_fw_k];
                        t_fw_col_up[8*tft+:8]=Mu[TMTPL][t_fw_eidx][tfo][t_fw_k];
                    end
                end
            end
        end
        for (tft=0;tft<TN;tft=tft+1) begin
            if (mtp_mode==1'b0) begin
                if (t_fw_sel==2'd2) t_fw_scale_g[16*(0*TN+tft)+:16]=ScDd[TMTPL][0];
                else begin
                    t_fw_scale_g[16*(0*TN+tft)+:16]=ScDg[TMTPL][tobd];
                    t_fw_scale_u[16*(0*TN+tft)+:16]=ScDu[TMTPL][tobd];
                end
            end else begin
                if (t_fw_shared) begin
                    if (t_fw_sel==2'd2) t_fw_scale_g[16*(0*TN+tft)+:16]=ScSHd[TMTPL];
                    else begin
                        t_fw_scale_g[16*(0*TN+tft)+:16]=ScSHg[TMTPL];
                        t_fw_scale_u[16*(0*TN+tft)+:16]=ScSHu[TMTPL];
                    end
                end else begin
                    if (t_fw_sel==2'd2) t_fw_scale_g[16*(0*TN+tft)+:16]=ScMd[TMTPL][t_fw_eidx];
                    else begin
                        t_fw_scale_g[16*(0*TN+tft)+:16]=ScMg[TMTPL][t_fw_eidx];
                        t_fw_scale_u[16*(0*TN+tft)+:16]=ScMu[TMTPL][t_fw_eidx];
                    end
                end
            end
        end
    end
    always @* begin
        t_lw_col = {LM_TN*16{1'b0}};
        for (tlt=0;tlt<LM_TN;tlt=tlt+1) t_lw_col[16*tlt+:16] = Wlm[t_lw_vtile*LM_TN + tlt][t_lw_k];
    end

    // MTP embed(verified) pull: e_val = EMB[e_tok][e_idx] (same embedding table)
    wire [15:0] em_for_mtp = EMB[e_tok][e_idx];

    //========================================================================
    // u_ref : standalone greedy reference model
    //========================================================================
    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN)
    ) u_ref (
        .clk(clk), .rst(rst), .start(ref_start), .busy(ref_busy), .done(ref_done),
        .token_id(ref_token), .pos(ref_pos), .s_len(ref_slen),
        .logits(ref_logits), .argmax(ref_argmax),
        .em_req(r_em_req), .em_tok(r_em_tok), .em_idx(r_em_idx), .em_val(em_val),
        .db_layer(r_db_layer), .idx_fresh(r_idx_fresh), .idx_win(r_idx_win),
        .gn_req(r_gn_req), .gn_which(r_gn_which), .gn_idx(r_gn_idx), .gn_val(gn_val),
        .aw_req(r_aw_req), .aw_sel(r_aw_sel), .aw_grp(r_aw_grp), .aw_k(r_aw_k),
        .aw_col(aw_col), .aw_scale(aw_scale),
        .kc_req(r_kc_req), .kc_idx(r_kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid),
        .rw_req(r_rw_req), .rw_k(r_rw_k), .rw_col(rw_col), .rw_scale(rw_scale),
        .fw_req(r_fw_req), .fw_sel(r_fw_sel), .fw_grp(r_fw_grp), .fw_k(r_fw_k),
        .fw_shared(r_fw_shared), .fw_eidx(r_fw_eidx),
        .fw_col(fw_col), .fw_col_up(fw_col_up),
        .fw_scale_g(fw_scale_g), .fw_scale_u(fw_scale_u),
        .fn_req(r_fn_req), .fn_idx(r_fn_idx), .fn_val(fn_val),
        .lw_req(r_lw_req), .lw_vtile(r_lw_vtile), .lw_k(r_lw_k), .lw_col(lw_col),
        .h_state(ref_hstate)
    );

    //========================================================================
    // dut : the speculative-decode loop top
    //========================================================================
    spec_decode_top #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PROJ_TN(PROJ_TN)
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .prompt_tok(prompt_tok), .start_pos(start_pos), .s_len(s_len),
        .mtp_mode(mtp_mode), .num_passes(num_passes),
        .busy(busy), .done(done),
        .commit_valid(commit_valid), .commit_tok(commit_tok), .accepted(accepted),
        .total_tokens(total_tokens), .main_passes(main_passes),
        .accepts(accepts), .rejects(rejects),
        .e_req(e_req), .e_tok(e_tok), .e_idx(e_idx), .e_val(em_for_mtp),
        .m_em_req(m_em_req), .m_em_tok(m_em_tok), .m_em_idx(m_em_idx), .m_em_val(em_val),
        .m_db_layer(m_db_layer), .m_idx_fresh(m_idx_fresh), .m_idx_win(m_idx_win),
        .m_gn_req(m_gn_req), .m_gn_which(m_gn_which), .m_gn_idx(m_gn_idx), .m_gn_val(gn_val),
        .m_aw_req(m_aw_req), .m_aw_sel(m_aw_sel), .m_aw_grp(m_aw_grp), .m_aw_k(m_aw_k),
        .m_aw_col(aw_col), .m_aw_scale(aw_scale),
        .m_kc_req(m_kc_req), .m_kc_idx(m_kc_idx),
        .m_kc_ckv(kc_ckv), .m_kc_krope(kc_krope), .m_kc_valid(kc_valid),
        .m_rw_req(m_rw_req), .m_rw_k(m_rw_k), .m_rw_col(rw_col), .m_rw_scale(rw_scale),
        .m_fw_req(m_fw_req), .m_fw_sel(m_fw_sel), .m_fw_grp(m_fw_grp), .m_fw_k(m_fw_k),
        .m_fw_shared(m_fw_shared), .m_fw_eidx(m_fw_eidx),
        .m_fw_col(fw_col), .m_fw_col_up(fw_col_up),
        .m_fw_scale_g(fw_scale_g), .m_fw_scale_u(fw_scale_u),
        .m_fn_req(m_fn_req), .m_fn_idx(m_fn_idx), .m_fn_val(fn_val),
        .m_lw_req(m_lw_req), .m_lw_vtile(m_lw_vtile), .m_lw_k(m_lw_k), .m_lw_col(lw_col),
        .t_cn_req(t_cn_req), .t_cn_which(t_cn_which), .t_cn_idx(t_cn_idx), .t_cn_val(t_cn_val),
        .t_pw_req(t_pw_req), .t_pw_ptile(t_pw_ptile), .t_pw_k(t_pw_k),
        .t_pw_col(t_pw_col), .t_pw_scale(t_pw_scale),
        .t_gn_req(t_gn_req), .t_gn_which(t_gn_which), .t_gn_idx(t_gn_idx), .t_gn_val(t_gn_val),
        .t_aw_req(t_aw_req), .t_aw_sel(t_aw_sel), .t_aw_grp(t_aw_grp), .t_aw_k(t_aw_k),
        .t_aw_col(t_aw_col), .t_aw_scale(t_aw_scale),
        .t_kc_req(t_kc_req), .t_kc_idx(t_kc_idx),
        .t_kc_ckv(t_kc_ckv), .t_kc_krope(t_kc_krope), .t_kc_valid(t_kc_valid),
        .t_rw_req(t_rw_req), .t_rw_k(t_rw_k), .t_rw_col(t_rw_col), .t_rw_scale(t_rw_scale),
        .t_fw_req(t_fw_req), .t_fw_sel(t_fw_sel), .t_fw_grp(t_fw_grp), .t_fw_k(t_fw_k),
        .t_fw_shared(t_fw_shared), .t_fw_eidx(t_fw_eidx),
        .t_fw_col(t_fw_col), .t_fw_col_up(t_fw_col_up),
        .t_fw_scale_g(t_fw_scale_g), .t_fw_scale_u(t_fw_scale_u),
        .t_lw_req(t_lw_req), .t_lw_vtile(t_lw_vtile), .t_lw_k(t_lw_k), .t_lw_col(t_lw_col)
    );

    // soak unused request / status lines
    wire _unused = &{1'b0, ref_busy, ref_logits, ref_hstate,
        r_em_req, r_idx_fresh, r_idx_win, r_gn_req, r_aw_req, r_kc_req, r_rw_req,
        r_fw_req, r_fn_req, r_lw_req,
        busy, e_req, m_em_req, m_idx_fresh, m_idx_win, m_gn_req, m_aw_req, m_kc_req,
        m_rw_req, m_fw_req, m_fn_req, m_lw_req,
        t_cn_req, t_pw_req, t_gn_req, t_aw_req, t_kc_req, t_rw_req, t_fw_req, t_lw_req};

    //========================================================================
    // greedy reference + committed-stream capture
    //========================================================================
    reg [TOKW-1:0] greedy [0:NGEN-1];

    // captured committed beats (token + accepted flag at the beat's cycle)
    reg [TOKW-1:0] beat_tok [0:CAP-1];
    reg            beat_acc [0:CAP-1];
    integer        beat_n;
    reg            cap_en;

    always @(negedge clk) if (cap_en && commit_valid) begin
        if (^commit_tok === 1'bx) begin
            $display("FAIL: X on commit_tok at beat %0d", beat_n);
            $fatal(1, "X on committed token");
        end
        if (beat_n < CAP) begin
            beat_tok[beat_n] = commit_tok;
            beat_acc[beat_n] = accepted;
        end
        beat_n = beat_n + 1;
    end

    // greedy run helper (drives u_ref one step; argmax holds after done)
    task run_ref_step; begin
        @(negedge clk);
        ref_start = 1'b1; @(negedge clk); ref_start = 1'b0;
        wait (ref_done === 1'b1);
        @(negedge clk);
    end endtask

    integer n, k, vcount, bcount, prev_acc, mism;
    reg [TOKW-1:0] cur;
    reg [POSW-1:0] p;
    integer tests;
    real eff;

    // safety timeout
    initial begin
        #500000000;
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end

    initial begin
        tests = 0; beat_n = 0; cap_en = 1'b0; in_spec = 1'b0;
        ref_start = 1'b0; ref_token = {TOKW{1'b0}}; ref_pos = {POSW{1'b0}};
        ref_slen = {(IDXW+1){1'b0}};
        start = 1'b0; prompt_tok = {TOKW{1'b0}}; start_pos = {POSW{1'b0}};
        s_len = {(IDXW+1){1'b0}}; mtp_mode = 1'b0; num_passes = 16'd0;

        // fixed prompt + fixed weights
        build_stimulus(424242, 0);
        prompt_tok = 4'd3; start_pos = 20'd0; s_len = 2'd1; mtp_mode = 1'b0;
        num_passes = NGEN[15:0];

        rst = 1'b1;
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ================= PHASE 1 : GREEDY GOLDEN (u_ref alone) =================
        in_spec = 1'b0;
        cur = prompt_tok; p = start_pos;
        for (n = 0; n < NGEN; n = n + 1) begin
            ref_token = cur; ref_pos = p; ref_slen = s_len;
            run_ref_step();
            if (^ref_argmax === 1'bx) begin
                $display("FAIL: greedy step %0d argmax X", n);
                $fatal(1, "X greedy argmax");
            end
            greedy[n] = ref_argmax;
            $display("GREEDY[%0d] tok=%0d -> argmax=%0d (pos=%0d)", n, cur, ref_argmax, p);
            tests = tests + 1;                    // each greedy token produced, non-X
            cur = ref_argmax; p = p + 20'd1;
        end

        // ================= PHASE 2 : DUT spec-decode loop =================
        in_spec = 1'b1;
        @(negedge clk);
        beat_n = 0; cap_en = 1'b1;
        @(negedge clk);
        start = 1'b1; @(negedge clk); start = 1'b0;

        k = 0;
        while (!done && k < 5000000) begin @(negedge clk); k = k + 1; end
        if (!done) $fatal(1, "loop never finished (no done)");
        @(negedge clk); @(negedge clk);          // let trailing bonus beat land
        cap_en = 1'b0;

        // ================= CHECKS =================
        // (a) main_passes == NGEN
        if (main_passes !== NGEN) begin
            $display("FAIL: main_passes=%0d exp %0d", main_passes, NGEN);
            $fatal(1, "main_passes mismatch");
        end
        tests = tests + 1;

        // (b) counters / busy non-X and clean
        if (^{total_tokens, main_passes, accepts, rejects} === 1'bx)
            $fatal(1, "counters X");
        tests = tests + 1;
        if (busy !== 1'b0) $fatal(1, "busy stuck high after done");
        tests = tests + 1;

        // (c) captured beat count == total_tokens (nothing lost/duplicated on stream)
        if (beat_n !== total_tokens) begin
            $display("FAIL: captured beats %0d != total_tokens %0d", beat_n, total_tokens);
            $fatal(1, "beat/total invariant broken");
        end
        tests = tests + 1;

        // (d) reconstruct VERIFIED subsequence: a beat is a bonus iff the prior
        //     beat had accepted=1 (the accepted-draft beat lands one cycle later).
        vcount = 0; bcount = 0; prev_acc = 0; mism = 0;
        for (k = 0; k < beat_n; k = k + 1) begin
            if (prev_acc == 1) begin
                bcount = bcount + 1;             // this beat is the accepted-draft bonus
            end else begin
                // verified beat -> must equal greedy[vcount]
                if (vcount < NGEN) begin
                    if (beat_tok[k] !== greedy[vcount]) begin
                        $display("FAIL: verified[%0d]=%0d != greedy[%0d]=%0d",
                                 vcount, beat_tok[k], vcount, greedy[vcount]);
                        mism = mism + 1;
                    end else tests = tests + 1;  // token-for-token identity
                end
                vcount = vcount + 1;
            end
            prev_acc = beat_acc[k];
        end
        if (mism != 0) $fatal(1, "spec-decode NOT exact vs greedy");

        // (e) #verified beats == NGEN == main_passes (one verified per main pass)
        if (vcount !== NGEN) begin
            $display("FAIL: verified beats %0d != NGEN %0d", vcount, NGEN);
            $fatal(1, "verified count mismatch");
        end
        tests = tests + 1;
        if (vcount !== main_passes) $fatal(1, "verified count != main_passes");
        tests = tests + 1;

        // (f) #bonus beats == accepts
        if (bcount !== accepts) begin
            $display("FAIL: bonus beats %0d != accepts %0d", bcount, accepts);
            $fatal(1, "bonus/accepts mismatch");
        end
        tests = tests + 1;

        // (g) acc + rej == passes - 1 (first pass has no prior draft to verify)
        if ((accepts + rejects) !== (main_passes - 32'd1))
            $fatal(1, "acc+rej != passes-1");
        tests = tests + 1;

        // (h) when no draft accepted, the FULL committed stream == greedy
        if (accepts == 32'd0) begin
            if (beat_n !== NGEN) $fatal(1, "accepts==0 but stream length != NGEN");
            for (k = 0; k < NGEN; k = k + 1)
                if (beat_tok[k] !== greedy[k]) $fatal(1, "full stream != greedy (accepts==0)");
            tests = tests + 1;
        end

        eff = (main_passes == 0) ? 0.0 :
              (total_tokens * 1.0) / (main_passes * 1.0);
        $display("----------------------------------------------------------------");
        $display("SPEC-DECODE EXACTNESS: committed verified-token sequence == greedy");
        $display("  generated tokens (N)         = %0d", NGEN);
        $display("  greedy sequence              = %0d %0d %0d %0d %0d",
                 greedy[0], greedy[1], greedy[2], greedy[3], greedy[4]);
        $display("  main_passes / total_tokens   = %0d / %0d", main_passes, total_tokens);
        $display("  accepts / rejects            = %0d / %0d", accepts, rejects);
        $display("  effective tokens / main-pass = %0f", eff);
        $display("  verified beats / bonus beats = %0d / %0d", vcount, bcount);
        $display("----------------------------------------------------------------");
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end
endmodule
