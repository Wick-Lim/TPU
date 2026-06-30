`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// spec_batched_top_tb.v -- BINDING spec==greedy + ÷K test for the BATCHED-
//                          VERIFY speculative-decode loop (spec_batched_top).
//----------------------------------------------------------------------------
// WHAT IT BINDS (per the brief):
//
//  (1) spec == greedy  (EXACT, the safety invariant):
//      An INDEPENDENT reference (a separate PE_M=1 glm_model_fp8 instance + a
//      private copy of spec_decode_seq's longest-accepted-prefix rule) computes,
//      for the SAME prompt / pos / s_len / draft schedule, the EXACT committed
//      token stream the loop must produce.  Because the loop only ever commits
//      the model's OWN argmaxes (truth_vec), never a raw draft, that stream is
//      the model's GREEDY rollout -- the drafts only gate HOW MANY tokens commit
//      per weight-load.  We assert, beat-for-beat and X-free, that the DUT's
//      committed stream EXACTLY equals this reference, AND (separately) that each
//      committed token equals an independently-recomputed pure-greedy rollout
//      (ng[k] = argmax(model(ng[k-1]))).  Two ways, same tokens => spec==greedy.
//      Covered for K=2 AND K=3, over ALL-ACCEPT, ALL-REJECT and MIXED drafts.
//
//  (2) ÷K weight-loads (the Flash lever):
//      The model WEIGHT-LOAD fires exactly ONCE per K-verify pass
//          weight_loads == num_passes
//      (vs K+1 serial single-token passes in the naive loop), and the per-pass
//      attention WEIGHT pull is fetched ONCE for all K+1 rows
//          aw_req(batched)/pass == aw_req(PE_M=1 single pass)
//      proving the K+1 verify positions share ONE weight-load.  We also report
//      the effective committed-tokens / pass (= total_tokens / num_passes).
//
//  The model weight ROMs + pull responders are the committed tiny slice (copied
//  from glm_model_fp8_pem_tb.v); the K drafts are TB-driven.  K=2 and K=3 each
//  run in their own parameterized engine; the top aggregates and prints
//  "ALL <N> TESTS PASSED"; $fatal on any spec!=greedy / miscount / X / timeout.
//============================================================================

//============================================================================
//  sbt_engine : one full DRAFT_K-parameterized binding harness (self-checking)
//============================================================================
module sbt_engine #(
    parameter integer DRAFT_K = 2
)(
    input  wire        clk,
    output reg  [31:0] tests_out,
    output reg  [31:0] errors_out,
    output reg         finished
);
    reg rst;

    // ================= tiny faithful slice (== glm_model_fp8_pem_tb) =================
    localparam integer MODEL_DIM  = 16;
    localparam integer L          = 2;
    localparam integer N_DENSE    = 1;
    localparam integer VOCAB      = 16;
    localparam integer H_HEADS    = 2;
    localparam integer NOPE       = 4;
    localparam integer ROPE       = 4;
    localparam integer V_DIM      = 4;
    localparam integer Q_LORA     = 8;
    localparam integer KV_LORA    = 8;
    localparam integer S_MAX      = 2;
    localparam integer TOPK_ATTN  = 2;
    localparam integer THETA      = 8000000;
    localparam integer PE_N       = 2;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = 4;
    localparam integer TOPK       = 2;
    localparam integer INTER_MOE  = 8;
    localparam integer INTER_DENSE= 160;
    localparam [31:0]  RSCALE     = 32'h40200000;
    localparam integer TN         = 2;
    localparam integer BLK        = 128;
    localparam integer LM_TN      = 2;
    localparam integer B          = DRAFT_K + 1;  // PE_M batch rows

    // ---- derived ----
    localparam integer QK_DIM = NOPE + ROPE;
    localparam integer IDXW   = (S_MAX<=1)?1:$clog2(S_MAX);
    localparam integer HQK    = H_HEADS*QK_DIM;
    localparam integer HNOPE  = H_HEADS*NOPE;
    localparam integer HV     = H_HEADS*V_DIM;
    localparam integer EIDXW  = (N_EXPERT<=1)?1:$clog2(N_EXPERT);
    localparam integer A_KMAX = (MODEL_DIM>Q_LORA)?
                       ((MODEL_DIM>KV_LORA)?((MODEL_DIM>HV)?MODEL_DIM:HV):((KV_LORA>HV)?KV_LORA:HV))
                     :((Q_LORA>KV_LORA)?((Q_LORA>HV)?Q_LORA:HV):((KV_LORA>HV)?KV_LORA:HV));
    localparam integer A_OMAX = (HQK>MODEL_DIM)?
                       ((HQK>HNOPE)?((HQK>HV)?HQK:HV):((HNOPE>HV)?HNOPE:HV))
                     :((MODEL_DIM>HNOPE)?((MODEL_DIM>HV)?MODEL_DIM:HV):((HNOPE>HV)?HNOPE:HV));
    localparam integer A_NGMAX = (A_OMAX+PE_N-1)/PE_N;
    localparam integer A_GRPW  = (A_NGMAX<=1)?1:$clog2(A_NGMAX);
    localparam integer A_KCW   = (A_KMAX <=1)?1:$clog2(A_KMAX);
    localparam integer FF_KMAX_D = (INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM;
    localparam integer FF_KMAX_M = (INTER_MOE >MODEL_DIM)?INTER_MOE :MODEL_DIM;
    localparam integer FF_GWD = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN+1);
    localparam integer FF_KWD = $clog2(FF_KMAX_D+1);
    localparam integer R_KW   = $clog2(FF_KMAX_M+1);
    localparam integer A_NB    = (A_KMAX   +BLK-1)/BLK;
    localparam integer FF_NB_D = (FF_KMAX_D+BLK-1)/BLK;
    localparam integer R_NB    = (FF_KMAX_M+BLK-1)/BLK;
    localparam integer LAYW   = (L<=1)?1:$clog2(L);
    localparam integer TOKW   = (VOCAB<=1)?1:$clog2(VOCAB);
    localparam integer DIMW   = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM);
    localparam integer NVTILE = VOCAB/LM_TN;
    localparam integer VTW    = (NVTILE<=1)?1:$clog2(NVTILE);
    localparam integer DKW    = (DRAFT_K<=1)?1:$clog2(DRAFT_K+1);

    integer test_count;
    integer errors;

    // ================= shared per-layer WEIGHT ROMs (== pem_tb) =================
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

    // ================= deterministic generators (== pem_tb) =================
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e=8'd125+h[6:4]; else e=8'd124+h[5:4];
        m=h[12:6]; gen_bf16={s,e,m};
    end endfunction
    function [7:0] gen_e4m3; input integer seed; input integer band;
        reg s; reg [3:0] e; reg [2:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e = 4'd7 + {3'b0,h[4]}; else e = 4'd6 + {3'b0,h[4]};
        m = h[12:10]; gen_e4m3 = {s,e,m};
    end endfunction
    function [15:0] gen_scale; input integer seed;
        reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*22229);
        e = 8'd122 + {7'b0,h[2]}; m = h[10:4]; gen_scale={1'b0,e,m};
    end endfunction

    integer i,j,e,GLY,sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc=seed0;
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin EMB[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (GLY=0;GLY<L;GLY=GLY+1) begin
            for (i=0;i<MODEL_DIM;i=i+1) begin G1[GLY][i]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) begin G2[GLY][i]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScW_dq[GLY]=gen_scale(sc); sc=sc+1;  ScW_uq[GLY]=gen_scale(sc); sc=sc+1;
            ScW_dkv[GLY]=gen_scale(sc); sc=sc+1; ScW_kr[GLY]=gen_scale(sc); sc=sc+1;
            ScW_uk[GLY]=gen_scale(sc); sc=sc+1;  ScW_uv[GLY]=gen_scale(sc); sc=sc+1;
            ScW_o[GLY]=gen_scale(sc); sc=sc+1;
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[GLY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[GLY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<N_EXPERT;j=j+1) begin Wg[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScWg[GLY]=gen_scale(sc); sc=sc+1;
            for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Dg[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Du[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_DENSE;j=j+1) begin Dd[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<FF_NB_D;i=i+1) begin ScDg[GLY][i]=gen_scale(sc); sc=sc+1; end
            for (i=0;i<FF_NB_D;i=i+1) begin ScDu[GLY][i]=gen_scale(sc); sc=sc+1; end
            for (i=0;i<FF_NB_D;i=i+1) begin ScDd[GLY][i]=gen_scale(sc); sc=sc+1; end
            for (e=0;e<N_EXPERT;e=e+1) begin
                for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mg[GLY][e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
                for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mu[GLY][e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
                for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin Md[GLY][e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
                ScMg[GLY][e]=gen_scale(sc); sc=sc+1; ScMu[GLY][e]=gen_scale(sc); sc=sc+1; ScMd[GLY][e]=gen_scale(sc); sc=sc+1;
            end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHg[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHu[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin SHd[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScSHg[GLY]=gen_scale(sc); sc=sc+1; ScSHu[GLY]=gen_scale(sc); sc=sc+1; ScSHd[GLY]=gen_scale(sc); sc=sc+1;
        end
        for (i=0;i<MODEL_DIM;i=i+1) begin GF[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Wlm[i][j]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ================= DUT : spec_batched_top (PE_M=B internally) =================
    reg                       start;
    reg  [TOKW-1:0]           prompt_tok;
    reg  [POSW-1:0]           start_pos;
    reg  [IDXW:0]             s_len;
    reg  [15:0]               num_passes;
    reg  [DRAFT_K*TOKW-1:0]   draft_in;
    reg  [DKW-1:0]            n_draft;
    wire                      busy, done;
    wire                      commit_valid; wire [TOKW-1:0] commit_tok; wire accepted;
    wire [31:0]               total_tokens, main_passes, accepts, rejects, weight_loads;

    wire                      em_req; wire [TOKW-1:0] em_tok; wire [DIMW-1:0] em_idx; reg [15:0] em_val;
    wire [LAYW-1:0]           db_layer; wire idx_fresh; wire [LAYW-1:0] idx_win;
    wire                      gn_req, gn_which; wire [DIMW-1:0] gn_idx; reg [15:0] gn_val;
    wire                      aw_req; wire [3:0] aw_sel; wire [A_GRPW-1:0] aw_grp; wire [A_KCW-1:0] aw_k;
    reg  [PE_N*8-1:0]         aw_col; reg [16*PE_N*A_NB-1:0] aw_scale;
    wire                      kc_req; wire [IDXW-1:0] kc_idx; reg [KV_LORA*16-1:0] kc_ckv; reg [ROPE*16-1:0] kc_krope; reg kc_valid;
    wire                      rw_req; wire [R_KW-1:0] rw_k; reg [8*N_EXPERT-1:0] rw_col; reg [16*N_EXPERT*R_NB-1:0] rw_scale;
    wire                      fw_req; wire [1:0] fw_sel; wire [FF_GWD-1:0] fw_grp; wire [FF_KWD-1:0] fw_k;
    wire                      fw_shared; wire [EIDXW-1:0] fw_eidx;
    reg  [8*TN-1:0]           fw_col, fw_col_up; reg [16*TN*FF_NB_D-1:0] fw_scale_g, fw_scale_u;
    wire                      fn_req; wire [DIMW-1:0] fn_idx; reg [15:0] fn_val;
    wire                      lw_req; wire [VTW-1:0] lw_vtile; wire [DIMW-1:0] lw_k; reg [LM_TN*16-1:0] lw_col;

    spec_batched_top #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .DRAFT_K(DRAFT_K)
    ) dut (
        .clk(clk), .rst(rst),
        .start(start), .prompt_tok(prompt_tok), .start_pos(start_pos),
        .s_len(s_len), .num_passes(num_passes),
        .draft_in(draft_in), .n_draft(n_draft),
        .busy(busy), .done(done),
        .commit_valid(commit_valid), .commit_tok(commit_tok), .accepted(accepted),
        .total_tokens(total_tokens), .main_passes(main_passes),
        .accepts(accepts), .rejects(rejects), .weight_loads(weight_loads),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_col(aw_col), .aw_scale(aw_scale),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope), .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col), .rw_scale(rw_scale),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_col(fw_col), .fw_col_up(fw_col_up), .fw_scale_g(fw_scale_g), .fw_scale_u(fw_scale_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col)
    );
    always @(posedge clk) begin if (rst) kc_valid <= 1'b0; else kc_valid <= kc_req; end
    /* verilator lint_off UNUSEDSIGNAL */
    wire _u = &{1'b0, busy, em_req, aw_req, fw_req, rw_req, gn_req, fn_req, lw_req,
                idx_fresh, idx_win, accepted};
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- DUT pull responder (combinational ROM reads keyed on routed-up ports) ----
    integer ts, res, fts, fos, cds, obds; reg [15:0] scas; reg dmodes;
    always @* em_val = EMB[em_tok][em_idx];
    always @* fn_val = GF[fn_idx];
    always @* begin lw_col = {LM_TN*16{1'b0}};
        for (ts=0;ts<LM_TN;ts=ts+1) lw_col[16*ts+:16] = Wlm[lw_vtile*LM_TN + ts][lw_k]; end
    always @* gn_val = gn_which ? G2[db_layer][gn_idx] : G1[db_layer][gn_idx];
    always @* begin
        aw_col = {PE_N*8{1'b0}}; aw_scale = {16*PE_N*A_NB{1'b0}};
        for (ts=0;ts<PE_N;ts=ts+1) case (aw_sel)
        4'd0: if (aw_grp*PE_N+ts<Q_LORA)   aw_col[8*ts+:8]=W_dq [db_layer][aw_grp*PE_N+ts][aw_k];
        4'd1: if (aw_grp*PE_N+ts<HQK)      aw_col[8*ts+:8]=W_uq [db_layer][aw_grp*PE_N+ts][aw_k];
        4'd2: if (aw_grp*PE_N+ts<KV_LORA)  aw_col[8*ts+:8]=W_dkv[db_layer][aw_grp*PE_N+ts][aw_k];
        4'd3: if (aw_grp*PE_N+ts<ROPE)     aw_col[8*ts+:8]=W_kr [db_layer][aw_grp*PE_N+ts][aw_k];
        4'd4: if (aw_grp*PE_N+ts<HNOPE)    aw_col[8*ts+:8]=W_uk [db_layer][aw_grp*PE_N+ts][aw_k];
        4'd5: if (aw_grp*PE_N+ts<HV)       aw_col[8*ts+:8]=W_uv [db_layer][aw_grp*PE_N+ts][aw_k];
        4'd6: if (aw_grp*PE_N+ts<MODEL_DIM)aw_col[8*ts+:8]=W_o  [db_layer][aw_grp*PE_N+ts][aw_k];
        default: aw_col[8*ts+:8]=8'h0; endcase
        case (aw_sel) 4'd0:scas=ScW_dq[db_layer]; 4'd1:scas=ScW_uq[db_layer]; 4'd2:scas=ScW_dkv[db_layer];
        4'd3:scas=ScW_kr[db_layer]; 4'd4:scas=ScW_uk[db_layer]; 4'd5:scas=ScW_uv[db_layer];
        4'd6:scas=ScW_o[db_layer]; default:scas=16'h3F80; endcase
        for (ts=0;ts<PE_N;ts=ts+1) aw_scale[16*ts+:16]=scas;
    end
    always @* begin kc_ckv={KV_LORA*16{1'b0}}; kc_krope={ROPE*16{1'b0}};
        for (cds=0;cds<KV_LORA;cds=cds+1) kc_ckv[16*cds+:16]=CKV[db_layer][kc_idx][cds];
        for (cds=0;cds<ROPE;cds=cds+1)    kc_krope[16*cds+:16]=KRP[db_layer][kc_idx][cds]; end
    always @* begin rw_col={8*N_EXPERT{1'b0}}; rw_scale={16*N_EXPERT*R_NB{1'b0}};
        for (res=0;res<N_EXPERT;res=res+1) begin rw_col[8*res+:8]=Wg[db_layer][rw_k][res]; rw_scale[16*res+:16]=ScWg[db_layer]; end end
    always @* begin dmodes=(db_layer<N_DENSE)?1'b0:1'b1;
        fw_col={8*TN{1'b0}}; fw_col_up={8*TN{1'b0}}; fw_scale_g={16*TN*FF_NB_D{1'b0}}; fw_scale_u={16*TN*FF_NB_D{1'b0}};
        obds=(fw_grp*TN)/BLK;
        for (fts=0;fts<TN;fts=fts+1) begin fos=fw_grp*TN+fts;
            if (dmodes==1'b0) begin
                if (fw_sel==2'd2) begin if (fos<MODEL_DIM) fw_col[8*fts+:8]=Dd[db_layer][fos][fw_k]; end
                else begin if (fos<INTER_DENSE) begin fw_col[8*fts+:8]=Dg[db_layer][fos][fw_k]; fw_col_up[8*fts+:8]=Du[db_layer][fos][fw_k]; end end
            end else begin
                if (fw_shared) begin
                    if (fw_sel==2'd2) begin if (fos<MODEL_DIM) fw_col[8*fts+:8]=SHd[db_layer][fos][fw_k]; end
                    else if (fos<INTER_MOE) begin fw_col[8*fts+:8]=SHg[db_layer][fos][fw_k]; fw_col_up[8*fts+:8]=SHu[db_layer][fos][fw_k]; end
                end else begin
                    if (fw_sel==2'd2) begin if (fos<MODEL_DIM) fw_col[8*fts+:8]=Md[db_layer][fw_eidx][fos][fw_k]; end
                    else if (fos<INTER_MOE) begin fw_col[8*fts+:8]=Mg[db_layer][fw_eidx][fos][fw_k]; fw_col_up[8*fts+:8]=Mu[db_layer][fw_eidx][fos][fw_k]; end
                end end end
        for (fts=0;fts<TN;fts=fts+1) begin
            if (dmodes==1'b0) begin
                if (fw_sel==2'd2) begin fw_scale_g[16*(0*TN+fts)+:16]=ScDd[db_layer][0]; fw_scale_g[16*(1*TN+fts)+:16]=ScDd[db_layer][1]; end
                else begin fw_scale_g[16*(0*TN+fts)+:16]=ScDg[db_layer][obds]; fw_scale_g[16*(1*TN+fts)+:16]=ScDg[db_layer][obds];
                    fw_scale_u[16*(0*TN+fts)+:16]=ScDu[db_layer][obds]; fw_scale_u[16*(1*TN+fts)+:16]=ScDu[db_layer][obds]; end
            end else begin
                if (fw_shared) begin
                    if (fw_sel==2'd2) fw_scale_g[16*(0*TN+fts)+:16]=ScSHd[db_layer];
                    else begin fw_scale_g[16*(0*TN+fts)+:16]=ScSHg[db_layer]; fw_scale_u[16*(0*TN+fts)+:16]=ScSHu[db_layer]; end
                end else begin
                    if (fw_sel==2'd2) fw_scale_g[16*(0*TN+fts)+:16]=ScMd[db_layer][fw_eidx];
                    else begin fw_scale_g[16*(0*TN+fts)+:16]=ScMg[db_layer][fw_eidx]; fw_scale_u[16*(0*TN+fts)+:16]=ScMu[db_layer][fw_eidx]; end
                end end end
    end

    // ================= committed-stream capture (X-aware) =================
    integer commit_n;
    reg [TOKW-1:0] got [0:4095];
    reg cap;
    always @(negedge clk) if (cap && commit_valid) begin
        if (^commit_tok === 1'bx) begin
            $display("FAIL K=%0d: X on commit beat %0d", DRAFT_K, commit_n); errors=errors+1;
        end
        got[commit_n] = commit_tok; commit_n = commit_n + 1;
    end

    // ================= INDEPENDENT reference model (PE_M=1) =================
    //   Used to compute the greedy/argmax reference stream AND the single-pass
    //   aw_req count -- fully separate from the DUT's own internal model.
    reg                       s1_start; wire s1_busy, s1_done;
    reg  [TOKW-1:0]           ref_tok;
    reg  [POSW-1:0]           ref_pos;
    reg  [IDXW:0]             ref_slen;
    wire [VOCAB*16-1:0]       s1_logits; wire [TOKW-1:0] s1_argmax;
    wire                      s1_em_req; wire [TOKW-1:0] s1_em_tok; wire [DIMW-1:0] s1_em_idx; reg [15:0] s1_em_val;
    wire [LAYW-1:0]           s1_db_layer; wire s1_idx_fresh; wire [LAYW-1:0] s1_idx_win;
    wire                      s1_gn_req, s1_gn_which; wire [DIMW-1:0] s1_gn_idx; reg [15:0] s1_gn_val;
    wire                      s1_aw_req; wire [3:0] s1_aw_sel; wire [A_GRPW-1:0] s1_aw_grp; wire [A_KCW-1:0] s1_aw_k;
    reg  [PE_N*8-1:0]         s1_aw_col; reg [16*PE_N*A_NB-1:0] s1_aw_scale;
    wire                      s1_kc_req; wire [IDXW-1:0] s1_kc_idx; reg [KV_LORA*16-1:0] s1_kc_ckv; reg [ROPE*16-1:0] s1_kc_krope; reg s1_kc_valid;
    wire                      s1_rw_req; wire [R_KW-1:0] s1_rw_k; reg [8*N_EXPERT-1:0] s1_rw_col; reg [16*N_EXPERT*R_NB-1:0] s1_rw_scale;
    wire                      s1_fw_req; wire [1:0] s1_fw_sel; wire [FF_GWD-1:0] s1_fw_grp; wire [FF_KWD-1:0] s1_fw_k;
    wire                      s1_fw_shared; wire [EIDXW-1:0] s1_fw_eidx;
    reg  [8*TN-1:0]           s1_fw_col, s1_fw_col_up; reg [16*TN*FF_NB_D-1:0] s1_fw_scale_g, s1_fw_scale_u;
    wire                      s1_fn_req; wire [DIMW-1:0] s1_fn_idx; reg [15:0] s1_fn_val;
    wire                      s1_lw_req; wire [VTW-1:0] s1_lw_vtile; wire [DIMW-1:0] s1_lw_k; reg [LM_TN*16-1:0] s1_lw_col;

    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PE_M(1)
    ) ref1 (
        .clk(clk), .rst(rst), .start(s1_start), .busy(s1_busy), .done(s1_done),
        .token_id(ref_tok), .pos(ref_pos), .s_len(ref_slen),
        .logits(s1_logits), .argmax(s1_argmax),
        .em_req(s1_em_req), .em_tok(s1_em_tok), .em_idx(s1_em_idx), .em_val(s1_em_val),
        .db_layer(s1_db_layer), .idx_fresh(s1_idx_fresh), .idx_win(s1_idx_win),
        .gn_req(s1_gn_req), .gn_which(s1_gn_which), .gn_idx(s1_gn_idx), .gn_val(s1_gn_val),
        .aw_req(s1_aw_req), .aw_sel(s1_aw_sel), .aw_grp(s1_aw_grp), .aw_k(s1_aw_k),
        .aw_col(s1_aw_col), .aw_scale(s1_aw_scale),
        .kc_req(s1_kc_req), .kc_idx(s1_kc_idx), .kc_ckv(s1_kc_ckv), .kc_krope(s1_kc_krope), .kc_valid(s1_kc_valid),
        .rw_req(s1_rw_req), .rw_k(s1_rw_k), .rw_col(s1_rw_col), .rw_scale(s1_rw_scale),
        .fw_req(s1_fw_req), .fw_sel(s1_fw_sel), .fw_grp(s1_fw_grp), .fw_k(s1_fw_k),
        .fw_shared(s1_fw_shared), .fw_eidx(s1_fw_eidx),
        .fw_col(s1_fw_col), .fw_col_up(s1_fw_col_up), .fw_scale_g(s1_fw_scale_g), .fw_scale_u(s1_fw_scale_u),
        .fn_req(s1_fn_req), .fn_idx(s1_fn_idx), .fn_val(s1_fn_val),
        .lw_req(s1_lw_req), .lw_vtile(s1_lw_vtile), .lw_k(s1_lw_k), .lw_col(s1_lw_col),
        .h_state()
    );
    always @(posedge clk) begin if (rst) s1_kc_valid <= 1'b0; else s1_kc_valid <= s1_kc_req; end
    /* verilator lint_off UNUSEDSIGNAL */
    wire _u1 = &{1'b0, s1_busy, s1_em_req, s1_aw_req, s1_fw_req, s1_rw_req, s1_gn_req,
                s1_fn_req, s1_lw_req, s1_idx_fresh, s1_idx_win, s1_logits};
    /* verilator lint_on UNUSEDSIGNAL */

    integer rts, rres, rfts, rfos, rcds, robds; reg [15:0] rscas; reg rdmodes;
    always @* s1_em_val = EMB[s1_em_tok][s1_em_idx];
    always @* s1_fn_val = GF[s1_fn_idx];
    always @* begin s1_lw_col = {LM_TN*16{1'b0}};
        for (rts=0;rts<LM_TN;rts=rts+1) s1_lw_col[16*rts+:16] = Wlm[s1_lw_vtile*LM_TN + rts][s1_lw_k]; end
    always @* s1_gn_val = s1_gn_which ? G2[s1_db_layer][s1_gn_idx] : G1[s1_db_layer][s1_gn_idx];
    always @* begin
        s1_aw_col = {PE_N*8{1'b0}}; s1_aw_scale = {16*PE_N*A_NB{1'b0}};
        for (rts=0;rts<PE_N;rts=rts+1) case (s1_aw_sel)
        4'd0: if (s1_aw_grp*PE_N+rts<Q_LORA)   s1_aw_col[8*rts+:8]=W_dq [s1_db_layer][s1_aw_grp*PE_N+rts][s1_aw_k];
        4'd1: if (s1_aw_grp*PE_N+rts<HQK)      s1_aw_col[8*rts+:8]=W_uq [s1_db_layer][s1_aw_grp*PE_N+rts][s1_aw_k];
        4'd2: if (s1_aw_grp*PE_N+rts<KV_LORA)  s1_aw_col[8*rts+:8]=W_dkv[s1_db_layer][s1_aw_grp*PE_N+rts][s1_aw_k];
        4'd3: if (s1_aw_grp*PE_N+rts<ROPE)     s1_aw_col[8*rts+:8]=W_kr [s1_db_layer][s1_aw_grp*PE_N+rts][s1_aw_k];
        4'd4: if (s1_aw_grp*PE_N+rts<HNOPE)    s1_aw_col[8*rts+:8]=W_uk [s1_db_layer][s1_aw_grp*PE_N+rts][s1_aw_k];
        4'd5: if (s1_aw_grp*PE_N+rts<HV)       s1_aw_col[8*rts+:8]=W_uv [s1_db_layer][s1_aw_grp*PE_N+rts][s1_aw_k];
        4'd6: if (s1_aw_grp*PE_N+rts<MODEL_DIM)s1_aw_col[8*rts+:8]=W_o  [s1_db_layer][s1_aw_grp*PE_N+rts][s1_aw_k];
        default: s1_aw_col[8*rts+:8]=8'h0; endcase
        case (s1_aw_sel) 4'd0:rscas=ScW_dq[s1_db_layer]; 4'd1:rscas=ScW_uq[s1_db_layer]; 4'd2:rscas=ScW_dkv[s1_db_layer];
        4'd3:rscas=ScW_kr[s1_db_layer]; 4'd4:rscas=ScW_uk[s1_db_layer]; 4'd5:rscas=ScW_uv[s1_db_layer];
        4'd6:rscas=ScW_o[s1_db_layer]; default:rscas=16'h3F80; endcase
        for (rts=0;rts<PE_N;rts=rts+1) s1_aw_scale[16*rts+:16]=rscas;
    end
    always @* begin s1_kc_ckv={KV_LORA*16{1'b0}}; s1_kc_krope={ROPE*16{1'b0}};
        for (rcds=0;rcds<KV_LORA;rcds=rcds+1) s1_kc_ckv[16*rcds+:16]=CKV[s1_db_layer][s1_kc_idx][rcds];
        for (rcds=0;rcds<ROPE;rcds=rcds+1)    s1_kc_krope[16*rcds+:16]=KRP[s1_db_layer][s1_kc_idx][rcds]; end
    always @* begin s1_rw_col={8*N_EXPERT{1'b0}}; s1_rw_scale={16*N_EXPERT*R_NB{1'b0}};
        for (rres=0;rres<N_EXPERT;rres=rres+1) begin s1_rw_col[8*rres+:8]=Wg[s1_db_layer][s1_rw_k][rres]; s1_rw_scale[16*rres+:16]=ScWg[s1_db_layer]; end end
    always @* begin rdmodes=(s1_db_layer<N_DENSE)?1'b0:1'b1;
        s1_fw_col={8*TN{1'b0}}; s1_fw_col_up={8*TN{1'b0}}; s1_fw_scale_g={16*TN*FF_NB_D{1'b0}}; s1_fw_scale_u={16*TN*FF_NB_D{1'b0}};
        robds=(s1_fw_grp*TN)/BLK;
        for (rfts=0;rfts<TN;rfts=rfts+1) begin rfos=s1_fw_grp*TN+rfts;
            if (rdmodes==1'b0) begin
                if (s1_fw_sel==2'd2) begin if (rfos<MODEL_DIM) s1_fw_col[8*rfts+:8]=Dd[s1_db_layer][rfos][s1_fw_k]; end
                else begin if (rfos<INTER_DENSE) begin s1_fw_col[8*rfts+:8]=Dg[s1_db_layer][rfos][s1_fw_k]; s1_fw_col_up[8*rfts+:8]=Du[s1_db_layer][rfos][s1_fw_k]; end end
            end else begin
                if (s1_fw_shared) begin
                    if (s1_fw_sel==2'd2) begin if (rfos<MODEL_DIM) s1_fw_col[8*rfts+:8]=SHd[s1_db_layer][rfos][s1_fw_k]; end
                    else if (rfos<INTER_MOE) begin s1_fw_col[8*rfts+:8]=SHg[s1_db_layer][rfos][s1_fw_k]; s1_fw_col_up[8*rfts+:8]=SHu[s1_db_layer][rfos][s1_fw_k]; end
                end else begin
                    if (s1_fw_sel==2'd2) begin if (rfos<MODEL_DIM) s1_fw_col[8*rfts+:8]=Md[s1_db_layer][s1_fw_eidx][rfos][s1_fw_k]; end
                    else if (rfos<INTER_MOE) begin s1_fw_col[8*rfts+:8]=Mg[s1_db_layer][s1_fw_eidx][rfos][s1_fw_k]; s1_fw_col_up[8*rfts+:8]=Mu[s1_db_layer][s1_fw_eidx][rfos][s1_fw_k]; end
                end end end
        for (rfts=0;rfts<TN;rfts=rfts+1) begin
            if (rdmodes==1'b0) begin
                if (s1_fw_sel==2'd2) begin s1_fw_scale_g[16*(0*TN+rfts)+:16]=ScDd[s1_db_layer][0]; s1_fw_scale_g[16*(1*TN+rfts)+:16]=ScDd[s1_db_layer][1]; end
                else begin s1_fw_scale_g[16*(0*TN+rfts)+:16]=ScDg[s1_db_layer][robds]; s1_fw_scale_g[16*(1*TN+rfts)+:16]=ScDg[s1_db_layer][robds];
                    s1_fw_scale_u[16*(0*TN+rfts)+:16]=ScDu[s1_db_layer][robds]; s1_fw_scale_u[16*(1*TN+rfts)+:16]=ScDu[s1_db_layer][robds]; end
            end else begin
                if (s1_fw_shared) begin
                    if (s1_fw_sel==2'd2) s1_fw_scale_g[16*(0*TN+rfts)+:16]=ScSHd[s1_db_layer];
                    else begin s1_fw_scale_g[16*(0*TN+rfts)+:16]=ScSHg[s1_db_layer]; s1_fw_scale_u[16*(0*TN+rfts)+:16]=ScSHu[s1_db_layer]; end
                end else begin
                    if (s1_fw_sel==2'd2) s1_fw_scale_g[16*(0*TN+rfts)+:16]=ScMd[s1_db_layer][s1_fw_eidx];
                    else begin s1_fw_scale_g[16*(0*TN+rfts)+:16]=ScMg[s1_db_layer][s1_fw_eidx]; s1_fw_scale_u[16*(0*TN+rfts)+:16]=ScMu[s1_db_layer][s1_fw_eidx]; end
                end end end
    end

    // ================= reference-step : ONE PE_M=1 model pass -> argmax =========
    //   Returns the model's argmax for (tk @ pz) in `ref_arg`, and (re)captures
    //   the single-pass aw_req count in `aw_single` (structural constant for the
    //   active weight set -> used by the ÷K weight-sharing check).
    reg  [TOKW-1:0] ref_arg;
    integer aw_single, ref_aw;
    task run_ref_step; input [TOKW-1:0] tk; input [POSW-1:0] pz;
        integer w; begin
            ref_tok = tk; ref_pos = pz;
            @(negedge clk); s1_start = 1'b1; @(negedge clk); s1_start = 1'b0;
            w = 0; ref_aw = 0;
            while (!s1_done && w < 2000000) begin
                @(negedge clk);
                if (s1_aw_req) ref_aw = ref_aw + 1;
                w = w + 1;
            end
            if (!s1_done) begin $display("FAIL K=%0d: ref step TIMEOUT", DRAFT_K); errors=errors+1; end
            ref_arg   = s1_argmax;
            aw_single = ref_aw;
            if (^ref_arg === 1'bx) begin $display("FAIL K=%0d: ref argmax is X", DRAFT_K); errors=errors+1; end
        end
    endtask

    // ================= INDEPENDENT reference stream (= what must commit) ========
    //   Replays the loop with a SEPARATE model + a private copy of the longest-
    //   accepted-prefix rule.  Per pass, at the SHARED pos (as the DUT does):
    //     truth[0]   = argmax(model(cur_tok @ pos))             (= m_1, greedy)
    //     truth[j]   = argmax(model(d_j     @ pos))  (j=1..K)   (= m_{j+1})
    //     p          = longest prefix with d_j == truth[j-1]
    //     commit truth[0..p]  ; advance cur_tok=truth[p], pos+=p+1
    //   PLUS an independent pure-greedy rollout ng[k]=argmax(model(ng[k-1]@pos))
    //   that must equal the committed truth[0..p] (drafts only gate the count).
    reg [TOKW-1:0] ref_commit [0:4095];
    integer ref_n, ref_acc, ref_rej, ref_tot;
    reg [TOKW-1:0] truthv [0:DRAFT_K];
    reg [TOKW-1:0] gng;
    task compute_ref;
        input [TOKW-1:0]            p0;
        input [POSW-1:0]            pos0;
        input [IDXW:0]             sl;
        input [DRAFT_K*TOKW-1:0]    dvec;
        input integer              nd;
        input integer              np;
        reg [TOKW-1:0] tok; reg [POSW-1:0] pos;
        integer pass, jj, p, brk, ndw; reg [TOKW-1:0] dj;
        begin
            ref_n=0; ref_acc=0; ref_rej=0; ref_tot=0;
            tok=p0; pos=pos0; ref_slen=sl;
            ndw = (nd>DRAFT_K)?DRAFT_K:nd;
            for (pass=0; pass<np; pass=pass+1) begin
                // the K+1 batch-row argmaxes at the SHARED pass pos
                run_ref_step(tok, pos); truthv[0]=ref_arg;
                for (jj=1; jj<=DRAFT_K; jj=jj+1) begin
                    dj = dvec[(jj-1)*TOKW +: TOKW];
                    run_ref_step(dj, pos); truthv[jj]=ref_arg;
                end
                // longest accepted prefix p (== spec_decode_seq's rule)
                p=0; brk=0;
                for (jj=1; jj<=DRAFT_K; jj=jj+1) begin
                    dj = dvec[(jj-1)*TOKW +: TOKW];
                    if (!brk && (jj<=ndw) && (dj==truthv[jj-1])) p=p+1; else brk=1;
                end
                // INDEPENDENT pure-greedy rollout must equal the committed truth
                gng = truthv[0];
                for (jj=1; jj<=p; jj=jj+1) begin
                    run_ref_step(gng, pos);
                    if (ref_arg !== truthv[jj]) begin
                        $display("FAIL K=%0d pass=%0d: greedy rollout %0d != committed truth %0d (idx %0d)",
                                 DRAFT_K, pass, ref_arg, truthv[jj], jj);
                        errors=errors+1;
                    end
                    gng = ref_arg;
                end
                // commit truth[0..p]
                for (jj=0; jj<=p; jj=jj+1) begin ref_commit[ref_n]=truthv[jj]; ref_n=ref_n+1; end
                ref_acc=ref_acc+p; ref_rej=ref_rej+(ndw-p); ref_tot=ref_tot+(p+1);
                tok=truthv[p]; pos=pos+(p+1);
            end
        end
    endtask

    // ================= pass-0 greedy rollout (to MINT draft schedules) =========
    reg [TOKW-1:0] g [0:DRAFT_K];
    task greedy0; input [TOKW-1:0] ptok; input [POSW-1:0] pos0; input [IDXW:0] sl;
        integer k; begin
            rst=1'b1; repeat(3) @(negedge clk); rst=1'b0; @(negedge clk);
            ref_slen = sl;
            run_ref_step(ptok, pos0); g[0]=ref_arg;
            for (k=1;k<=DRAFT_K;k=k+1) begin run_ref_step(g[k-1], pos0); g[k]=ref_arg; end
        end
    endtask

    // build the static draft vector from g[]; `mp` = forced-mismatch slot
    //   mp>=DRAFT_K -> ALL-ACCEPT ; mp==0 -> ALL-REJECT ; 0<mp<K -> MIXED
    reg [DRAFT_K*TOKW-1:0] dbuild;
    task build_dvec; input integer mp; integer k; begin
        for (k=0;k<DRAFT_K;k=k+1) begin
            if (k==mp) dbuild[k*TOKW +: TOKW] = (g[k]+1) & (VOCAB-1);  // != g[k] -> reject here
            else       dbuild[k*TOKW +: TOKW] = g[k];                  // == g[k] -> accept
        end
    end endtask

    // ================= one binding case : reference vs DUT =====================
    integer wd, aw_batched;
    integer ci;
    task run_case;
        input integer              np;
        input [TOKW-1:0]            ptok;
        input [POSW-1:0]            pos0;
        input [IDXW:0]             sl;
        input [DRAFT_K*TOKW-1:0]    dvec;
        input integer              nd;
        input integer              exp_acc;   // expected ref accepts (-1 = skip)
        input integer              exp_rej;   // expected ref rejects (-1 = skip)
        input [8*12-1:0]           nm;
        begin
            test_count = test_count + 1;
            $display("..  K=%0d %0s: start (np=%0d)", DRAFT_K, nm, np); $fflush;

            // (a) compute the INDEPENDENT reference stream (DUT idle / in reset)
            rst=1'b1; repeat(3) @(negedge clk); rst=1'b0; @(negedge clk);
            compute_ref(ptok, pos0, sl, dvec, nd, np);

            // coverage sanity : the minted drafts realized the intended scenario
            if (exp_acc >= 0 && ref_acc !== exp_acc) begin
                $display("FAIL K=%0d %0s: ref accepts %0d != expected %0d (scenario not realized)",
                         DRAFT_K, nm, ref_acc, exp_acc); errors=errors+1;
            end
            if (exp_rej >= 0 && ref_rej !== exp_rej) begin
                $display("FAIL K=%0d %0s: ref rejects %0d != expected %0d",
                         DRAFT_K, nm, ref_rej, exp_rej); errors=errors+1;
            end

            // (b) run the DUT loop on the SAME inputs (fresh counters)
            rst=1'b1; repeat(3) @(negedge clk); rst=1'b0; @(negedge clk);
            prompt_tok = ptok; start_pos = pos0; s_len = sl;
            draft_in = dvec; n_draft = nd[DKW-1:0]; num_passes = np[15:0];
            commit_n = 0; cap = 1'b1;
            @(negedge clk); start=1'b1; @(negedge clk); start=1'b0;
            wd = 0; aw_batched = 0;
            while (!done && wd < 8000000) begin
                @(negedge clk);
                if (aw_req) aw_batched = aw_batched + 1;
                wd = wd + 1;
            end
            cap = 1'b0;
            if (!done) begin $display("FAIL K=%0d %0s: DUT loop TIMEOUT", DRAFT_K, nm); errors=errors+1; disable run_case; end
            @(negedge clk);

            // ---- spec == greedy : EXACT committed-stream equality ----
            if (commit_n !== ref_n) begin
                $display("FAIL K=%0d %0s: committed beats %0d != reference %0d", DRAFT_K, nm, commit_n, ref_n);
                errors=errors+1;
            end else begin
                for (ci=0; ci<ref_n; ci=ci+1)
                    if (got[ci] !== ref_commit[ci]) begin
                        $display("FAIL K=%0d %0s: commit[%0d]=%0d != greedy ref %0d (spec!=greedy)",
                                 DRAFT_K, nm, ci, got[ci], ref_commit[ci]);
                        errors=errors+1;
                    end
            end
            // ---- counters mirror spec_decode_seq exactly ----
            if (total_tokens !== ref_tot) begin $display("FAIL K=%0d %0s: total_tokens %0d != ref %0d", DRAFT_K, nm, total_tokens, ref_tot); errors=errors+1; end
            if (main_passes !== np)        begin $display("FAIL K=%0d %0s: main_passes %0d != %0d",    DRAFT_K, nm, main_passes, np);     errors=errors+1; end
            if (accepts !== ref_acc)       begin $display("FAIL K=%0d %0s: accepts %0d != ref %0d",     DRAFT_K, nm, accepts, ref_acc);    errors=errors+1; end
            if (rejects !== ref_rej)       begin $display("FAIL K=%0d %0s: rejects %0d != ref %0d",     DRAFT_K, nm, rejects, ref_rej);    errors=errors+1; end
            // ---- ÷K : ONE model weight-load per K-verify pass ----
            if (weight_loads !== np) begin
                $display("FAIL K=%0d %0s: weight_loads %0d != num_passes %0d (NOT one weight-load/pass)",
                         DRAFT_K, nm, weight_loads, np); errors=errors+1;
            end
            // ---- ÷K : attention weight pull shared across the K+1 rows ----
            if (aw_batched !== np*aw_single) begin
                $display("FAIL K=%0d %0s: aw_req batched=%0d != %0d*single(%0d) (weight NOT shared across K+1 rows)",
                         DRAFT_K, nm, aw_batched, np, aw_single); errors=errors+1;
            end else begin
                $display("ok  K=%0d %0s: passes=%0d commits=%0d(=%0.2f/pass) acc=%0d rej=%0d | weight_loads=%0d(==passes, K+1=%0d verify rows/load -> Flash/%0d) | aw_req=%0d==%0d*%0d(shared) | spec==greedy EXACT",
                         DRAFT_K, nm, np, commit_n, (1.0*commit_n)/np, accepts, rejects,
                         weight_loads, B, B, aw_batched, np, aw_single);
            end
            $fflush;
        end
    endtask

    // ================= the binding sequence (K=2 and K=3 each) =================
    initial begin
        start=1'b0; s1_start=1'b0; prompt_tok=0; ref_tok=0; ref_pos=0; ref_slen=1;
        start_pos=0; s_len=1; num_passes=0; draft_in=0; n_draft=0;
        cap=1'b0; commit_n=0; aw_single=0;
        test_count=0; errors=0; finished=1'b0;
        tests_out=0; errors_out=0;
        rst=1'b1; repeat(4) @(negedge clk); rst=1'b0; @(negedge clk);

        // ---- weight config A : prompt=3 @ pos0, s_len=1 ----
        //   one greedy rollout g[] (pos0) mints all three draft schedules.
        build_stimulus(500,0);
        greedy0(4'd3, 0, 1);
        // ALL-REJECT : draft 1 forced != m_1 -> p=0, every draft discarded
        build_dvec(0);
        run_case(1, 4'd3, 0, 1, dbuild, DRAFT_K, 0, DRAFT_K, "all-reject");
        // ALL-ACCEPT : drafts == the greedy rollout -> p=K, commit K+1 in one load
        build_dvec(DRAFT_K);
        run_case(1, 4'd3, 0, 1, dbuild, DRAFT_K, DRAFT_K, 0, "all-accept");
        // MIXED : accept draft 1, reject draft 2 -> p=1
        build_dvec(1);
        run_case(1, 4'd3, 0, 1, dbuild, DRAFT_K, 1, DRAFT_K-1, "mixed");

        // ---- weight config B : prompt=11 @ pos37, s_len=2, MULTI-PASS ----
        //   pass-0 drafts are all-accept; later passes use the SAME static drafts
        //   against a moved cursor -> mixed acceptance across passes.  The cursor
        //   advance (cur_tok/cur_pos += p+1) + ÷K over multiple passes is exercised.
        build_stimulus(7000,0);
        greedy0(4'd11, 37, 2); build_dvec(DRAFT_K);
        run_case(2, 4'd11, 37, 2, dbuild, DRAFT_K, -1, -1, "multipass");

        tests_out=test_count; errors_out=errors; finished=1'b1;
    end
endmodule

//============================================================================
//  spec_batched_top_tb : run the K=2 and K=3 binding engines, aggregate
//============================================================================
module spec_batched_top_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    wire [31:0] t2, e2, t3, e3;
    wire        f2, f3;

    sbt_engine #(.DRAFT_K(2)) engK2 (.clk(clk), .tests_out(t2), .errors_out(e2), .finished(f2));
    sbt_engine #(.DRAFT_K(3)) engK3 (.clk(clk), .tests_out(t3), .errors_out(e3), .finished(f3));

    // global watchdog
    initial begin #1000000000; $display("FAIL: global timeout"); $fatal(1,"timeout"); end

    initial begin
        wait (f2 === 1'b1 && f3 === 1'b1);
        @(negedge clk);
        if ((e2 + e3) != 32'd0) begin
            $display("FAILED: %0d error(s) across %0d tests (K=2 err=%0d, K=3 err=%0d)",
                     e2+e3, t2+t3, e2, e3);
            $fatal(1,"fail");
        end
        $display("ALL %0d TESTS PASSED", t2 + t3);
        $finish;
    end
endmodule
