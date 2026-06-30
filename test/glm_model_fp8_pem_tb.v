`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_model_fp8_pem_tb.v -- PE_M BATCHING equivalence + weight-share smoke for
//                           the FULL GLM-5.2-FP8 forward (glm_model_fp8.v)
//----------------------------------------------------------------------------
// VERIFY STANCE  (RTL-vs-RTL)
//   The committed glm_model_fp8_tb.v pins the PE_M=1 forward against a faithful
//   fp64 FP8 golden (argmax 4/31/20).  This TB pins the PE_M>1 WIDENING contract:
//
//     "row r of a PE_M=B forward is the SAME as a standalone PE_M=1 forward on
//      token r (per-row argmax + logits), AND the per-layer attention weight
//      stream is fetched ONCE for all B rows (not B times)."
//
//   It drives B=2 query tokens two ways against ONE shared weight set:
//     (a) REFERENCE : a PE_M=1 DUT, run once per token, argmax/logits captured.
//     (b) BATCHED   : a PE_M=2 DUT, both tokens decoded in lockstep.
//   The B rows share pos, s_len, the KV cache and ALL weights; only token_id
//   differs.  With the slice TOPK=2 the per-row MoE combine is fp32-add
//   commutative, so the batched per-row result is BIT-EXACT to its PE_M=1 run.
//
//   CHECKS:
//     * argmax[row r] === reference argmax for token r        (EXACT, X-aware)
//     * logits[row r] === reference logits for token r        (BIT-EXACT bf16)
//     * aw_req pulses(batched) == aw_req pulses(single)        (weight stream shared)
//   Prints "ALL <N> TESTS PASSED"; $fatal on any mismatch / X / timeout.
//============================================================================
module glm_model_fp8_pem_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ================= slice config =================
    //   TINY faithful slice: the PE_M widening is a CONFIG-INDEPENDENT structural
    //   property (per-row residual + per-row tail + shared weight fetch), and the
    //   committed glm_model_fp8_tb already pins the FULL-slice PE_M=1 datapath
    //   (argmax 4/31/20).  This smoke shrinks every dim so the dual-DUT equivalence
    //   run is fast, while still exercising dense+MoE layers, PER-ROW MoE routing
    //   divergence (N_EXPERT=4, TOPK=2) and the shared attention weight stream.
    //   INTER_DENSE stays >128 so the dense FFN keeps 2 K-blocks (FF_NB_D=2), the
    //   same scale-bank structure the responder (copied from the committed TB) uses.
    localparam integer MODEL_DIM  = 16;
    localparam integer L          = 2;            // 1 dense + 1 MoE
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
    localparam integer INTER_DENSE= 160;          // >128 -> dense keeps 2 K-blocks
    localparam [31:0]  RSCALE     = 32'h40200000;
    localparam integer TN         = 2;
    localparam integer BLK        = 128;
    localparam integer LM_TN      = 2;
    localparam integer B          = 2;            // batch rows for the widened DUT

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

    integer test_count = 0;
    integer errors     = 0;

    // ================= shared per-layer WEIGHT ROMs =================
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

    // ================= deterministic generators (== committed) =================
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

    // ================= shared control =================
    reg  [POSW-1:0] pos;
    reg  [IDXW:0]   s_len;

    //========================================================================
    // REFERENCE DUT (PE_M = 1)
    //========================================================================
    reg                       start_s;
    wire                      busy_s, done_s;
    reg  [TOKW-1:0]           token_id_s;
    wire [VOCAB*16-1:0]       logits_s;
    wire [TOKW-1:0]           argmax_s;
    wire                      em_req_s; wire [TOKW-1:0] em_tok_s; wire [DIMW-1:0] em_idx_s; reg [15:0] em_val_s;
    wire [LAYW-1:0]           db_layer_s; wire idx_fresh_s; wire [LAYW-1:0] idx_win_s;
    wire                      gn_req_s, gn_which_s; wire [DIMW-1:0] gn_idx_s; reg [15:0] gn_val_s;
    wire                      aw_req_s; wire [3:0] aw_sel_s; wire [A_GRPW-1:0] aw_grp_s; wire [A_KCW-1:0] aw_k_s;
    reg  [PE_N*8-1:0]         aw_col_s; reg [16*PE_N*A_NB-1:0] aw_scale_s;
    wire                      kc_req_s; wire [IDXW-1:0] kc_idx_s; reg [KV_LORA*16-1:0] kc_ckv_s; reg [ROPE*16-1:0] kc_krope_s; reg kc_valid_s;
    wire                      rw_req_s; wire [R_KW-1:0] rw_k_s; reg [8*N_EXPERT-1:0] rw_col_s; reg [16*N_EXPERT*R_NB-1:0] rw_scale_s;
    wire                      fw_req_s; wire [1:0] fw_sel_s; wire [FF_GWD-1:0] fw_grp_s; wire [FF_KWD-1:0] fw_k_s;
    wire                      fw_shared_s; wire [EIDXW-1:0] fw_eidx_s;
    reg  [8*TN-1:0]           fw_col_s, fw_col_up_s; reg [16*TN*FF_NB_D-1:0] fw_scale_g_s, fw_scale_u_s;
    wire                      fn_req_s; wire [DIMW-1:0] fn_idx_s; reg [15:0] fn_val_s;
    wire                      lw_req_s; wire [VTW-1:0] lw_vtile_s; wire [DIMW-1:0] lw_k_s; reg [LM_TN*16-1:0] lw_col_s;

    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PE_M(1)
    ) dut_s (
        .clk(clk), .rst(rst), .start(start_s), .busy(busy_s), .done(done_s),
        .token_id(token_id_s), .pos(pos), .s_len(s_len),
        .logits(logits_s), .argmax(argmax_s),
        .em_req(em_req_s), .em_tok(em_tok_s), .em_idx(em_idx_s), .em_val(em_val_s),
        .db_layer(db_layer_s), .idx_fresh(idx_fresh_s), .idx_win(idx_win_s),
        .gn_req(gn_req_s), .gn_which(gn_which_s), .gn_idx(gn_idx_s), .gn_val(gn_val_s),
        .aw_req(aw_req_s), .aw_sel(aw_sel_s), .aw_grp(aw_grp_s), .aw_k(aw_k_s),
        .aw_col(aw_col_s), .aw_scale(aw_scale_s),
        .kc_req(kc_req_s), .kc_idx(kc_idx_s), .kc_ckv(kc_ckv_s), .kc_krope(kc_krope_s), .kc_valid(kc_valid_s),
        .rw_req(rw_req_s), .rw_k(rw_k_s), .rw_col(rw_col_s), .rw_scale(rw_scale_s),
        .fw_req(fw_req_s), .fw_sel(fw_sel_s), .fw_grp(fw_grp_s), .fw_k(fw_k_s),
        .fw_shared(fw_shared_s), .fw_eidx(fw_eidx_s),
        .fw_col(fw_col_s), .fw_col_up(fw_col_up_s), .fw_scale_g(fw_scale_g_s), .fw_scale_u(fw_scale_u_s),
        .fn_req(fn_req_s), .fn_idx(fn_idx_s), .fn_val(fn_val_s),
        .lw_req(lw_req_s), .lw_vtile(lw_vtile_s), .lw_k(lw_k_s), .lw_col(lw_col_s),
        .h_state()
    );
    always @(posedge clk) begin if (rst) kc_valid_s <= 1'b0; else kc_valid_s <= kc_req_s; end
    /* verilator lint_off UNUSEDSIGNAL */
    wire _u_s = &{1'b0, busy_s, em_req_s, em_tok_s, aw_req_s, fw_req_s, rw_req_s, gn_req_s, fn_req_s, lw_req_s, idx_fresh_s, idx_win_s};
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- REFERENCE responder (reads shared ROMs, keyed on dut_s pull ports) ----
    integer ts, res, fts, fos, cds, obds; reg [15:0] scas; reg dmodes;
    always @(em_tok_s or em_idx_s or start_s) em_val_s = EMB[em_tok_s][em_idx_s];
    always @(fn_idx_s or start_s) fn_val_s = GF[fn_idx_s];
    always @(lw_vtile_s or lw_k_s or start_s) begin lw_col_s = {LM_TN*16{1'b0}};
        for (ts=0;ts<LM_TN;ts=ts+1) lw_col_s[16*ts+:16] = Wlm[lw_vtile_s*LM_TN + ts][lw_k_s]; end
    always @(gn_which_s or gn_idx_s or db_layer_s or start_s) gn_val_s = gn_which_s ? G2[db_layer_s][gn_idx_s] : G1[db_layer_s][gn_idx_s];
    always @(aw_sel_s or aw_grp_s or aw_k_s or db_layer_s or start_s) begin
        aw_col_s = {PE_N*8{1'b0}}; aw_scale_s = {16*PE_N*A_NB{1'b0}};
        for (ts=0;ts<PE_N;ts=ts+1) case (aw_sel_s)
        4'd0: if (aw_grp_s*PE_N+ts<Q_LORA)   aw_col_s[8*ts+:8]=W_dq [db_layer_s][aw_grp_s*PE_N+ts][aw_k_s];
        4'd1: if (aw_grp_s*PE_N+ts<HQK)      aw_col_s[8*ts+:8]=W_uq [db_layer_s][aw_grp_s*PE_N+ts][aw_k_s];
        4'd2: if (aw_grp_s*PE_N+ts<KV_LORA)  aw_col_s[8*ts+:8]=W_dkv[db_layer_s][aw_grp_s*PE_N+ts][aw_k_s];
        4'd3: if (aw_grp_s*PE_N+ts<ROPE)     aw_col_s[8*ts+:8]=W_kr [db_layer_s][aw_grp_s*PE_N+ts][aw_k_s];
        4'd4: if (aw_grp_s*PE_N+ts<HNOPE)    aw_col_s[8*ts+:8]=W_uk [db_layer_s][aw_grp_s*PE_N+ts][aw_k_s];
        4'd5: if (aw_grp_s*PE_N+ts<HV)       aw_col_s[8*ts+:8]=W_uv [db_layer_s][aw_grp_s*PE_N+ts][aw_k_s];
        4'd6: if (aw_grp_s*PE_N+ts<MODEL_DIM)aw_col_s[8*ts+:8]=W_o  [db_layer_s][aw_grp_s*PE_N+ts][aw_k_s];
        default: aw_col_s[8*ts+:8]=8'h0; endcase
        case (aw_sel_s) 4'd0:scas=ScW_dq[db_layer_s]; 4'd1:scas=ScW_uq[db_layer_s]; 4'd2:scas=ScW_dkv[db_layer_s];
        4'd3:scas=ScW_kr[db_layer_s]; 4'd4:scas=ScW_uk[db_layer_s]; 4'd5:scas=ScW_uv[db_layer_s];
        4'd6:scas=ScW_o[db_layer_s]; default:scas=16'h3F80; endcase
        for (ts=0;ts<PE_N;ts=ts+1) aw_scale_s[16*ts+:16]=scas;
    end
    always @(kc_idx_s or db_layer_s or start_s) begin kc_ckv_s={KV_LORA*16{1'b0}}; kc_krope_s={ROPE*16{1'b0}};
        for (cds=0;cds<KV_LORA;cds=cds+1) kc_ckv_s[16*cds+:16]=CKV[db_layer_s][kc_idx_s][cds];
        for (cds=0;cds<ROPE;cds=cds+1)    kc_krope_s[16*cds+:16]=KRP[db_layer_s][kc_idx_s][cds]; end
    always @(rw_k_s or db_layer_s or start_s) begin rw_col_s={8*N_EXPERT{1'b0}}; rw_scale_s={16*N_EXPERT*R_NB{1'b0}};
        for (res=0;res<N_EXPERT;res=res+1) begin rw_col_s[8*res+:8]=Wg[db_layer_s][rw_k_s][res]; rw_scale_s[16*res+:16]=ScWg[db_layer_s]; end end
    always @(fw_grp_s or fw_k_s or fw_sel_s or fw_shared_s or fw_eidx_s or db_layer_s or start_s) begin dmodes=(db_layer_s<N_DENSE)?1'b0:1'b1;
        fw_col_s={8*TN{1'b0}}; fw_col_up_s={8*TN{1'b0}}; fw_scale_g_s={16*TN*FF_NB_D{1'b0}}; fw_scale_u_s={16*TN*FF_NB_D{1'b0}};
        obds=(fw_grp_s*TN)/BLK;
        for (fts=0;fts<TN;fts=fts+1) begin fos=fw_grp_s*TN+fts;
            if (dmodes==1'b0) begin
                if (fw_sel_s==2'd2) begin if (fos<MODEL_DIM) fw_col_s[8*fts+:8]=Dd[db_layer_s][fos][fw_k_s]; end
                else begin if (fos<INTER_DENSE) begin fw_col_s[8*fts+:8]=Dg[db_layer_s][fos][fw_k_s]; fw_col_up_s[8*fts+:8]=Du[db_layer_s][fos][fw_k_s]; end end
            end else begin
                if (fw_shared_s) begin
                    if (fw_sel_s==2'd2) begin if (fos<MODEL_DIM) fw_col_s[8*fts+:8]=SHd[db_layer_s][fos][fw_k_s]; end
                    else if (fos<INTER_MOE) begin fw_col_s[8*fts+:8]=SHg[db_layer_s][fos][fw_k_s]; fw_col_up_s[8*fts+:8]=SHu[db_layer_s][fos][fw_k_s]; end
                end else begin
                    if (fw_sel_s==2'd2) begin if (fos<MODEL_DIM) fw_col_s[8*fts+:8]=Md[db_layer_s][fw_eidx_s][fos][fw_k_s]; end
                    else if (fos<INTER_MOE) begin fw_col_s[8*fts+:8]=Mg[db_layer_s][fw_eidx_s][fos][fw_k_s]; fw_col_up_s[8*fts+:8]=Mu[db_layer_s][fw_eidx_s][fos][fw_k_s]; end
                end end end
        for (fts=0;fts<TN;fts=fts+1) begin
            if (dmodes==1'b0) begin
                if (fw_sel_s==2'd2) begin fw_scale_g_s[16*(0*TN+fts)+:16]=ScDd[db_layer_s][0]; fw_scale_g_s[16*(1*TN+fts)+:16]=ScDd[db_layer_s][1]; end
                else begin fw_scale_g_s[16*(0*TN+fts)+:16]=ScDg[db_layer_s][obds]; fw_scale_g_s[16*(1*TN+fts)+:16]=ScDg[db_layer_s][obds];
                    fw_scale_u_s[16*(0*TN+fts)+:16]=ScDu[db_layer_s][obds]; fw_scale_u_s[16*(1*TN+fts)+:16]=ScDu[db_layer_s][obds]; end
            end else begin
                if (fw_shared_s) begin
                    if (fw_sel_s==2'd2) fw_scale_g_s[16*(0*TN+fts)+:16]=ScSHd[db_layer_s];
                    else begin fw_scale_g_s[16*(0*TN+fts)+:16]=ScSHg[db_layer_s]; fw_scale_u_s[16*(0*TN+fts)+:16]=ScSHu[db_layer_s]; end
                end else begin
                    if (fw_sel_s==2'd2) fw_scale_g_s[16*(0*TN+fts)+:16]=ScMd[db_layer_s][fw_eidx_s];
                    else begin fw_scale_g_s[16*(0*TN+fts)+:16]=ScMg[db_layer_s][fw_eidx_s]; fw_scale_u_s[16*(0*TN+fts)+:16]=ScMu[db_layer_s][fw_eidx_s]; end
                end end end
    end

    //========================================================================
    // BATCHED DUT (PE_M = B)
    //========================================================================
    reg                       start_w;
    wire                      busy_w, done_w;
    reg  [B*TOKW-1:0]         token_id_w;
    wire [B*VOCAB*16-1:0]     logits_w;
    wire [B*TOKW-1:0]         argmax_w;
    wire                      em_req_w; wire [TOKW-1:0] em_tok_w; wire [DIMW-1:0] em_idx_w; reg [15:0] em_val_w;
    wire [LAYW-1:0]           db_layer_w; wire idx_fresh_w; wire [LAYW-1:0] idx_win_w;
    wire                      gn_req_w, gn_which_w; wire [DIMW-1:0] gn_idx_w; reg [15:0] gn_val_w;
    wire                      aw_req_w; wire [3:0] aw_sel_w; wire [A_GRPW-1:0] aw_grp_w; wire [A_KCW-1:0] aw_k_w;
    reg  [PE_N*8-1:0]         aw_col_w; reg [16*PE_N*A_NB-1:0] aw_scale_w;
    wire                      kc_req_w; wire [IDXW-1:0] kc_idx_w; reg [KV_LORA*16-1:0] kc_ckv_w; reg [ROPE*16-1:0] kc_krope_w; reg kc_valid_w;
    wire                      rw_req_w; wire [R_KW-1:0] rw_k_w; reg [8*N_EXPERT-1:0] rw_col_w; reg [16*N_EXPERT*R_NB-1:0] rw_scale_w;
    wire                      fw_req_w; wire [1:0] fw_sel_w; wire [FF_GWD-1:0] fw_grp_w; wire [FF_KWD-1:0] fw_k_w;
    wire                      fw_shared_w; wire [EIDXW-1:0] fw_eidx_w;
    reg  [8*TN-1:0]           fw_col_w, fw_col_up_w; reg [16*TN*FF_NB_D-1:0] fw_scale_g_w, fw_scale_u_w;
    wire                      fn_req_w; wire [DIMW-1:0] fn_idx_w; reg [15:0] fn_val_w;
    wire                      lw_req_w; wire [VTW-1:0] lw_vtile_w; wire [DIMW-1:0] lw_k_w; reg [LM_TN*16-1:0] lw_col_w;

    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PE_M(B)
    ) dut_w (
        .clk(clk), .rst(rst), .start(start_w), .busy(busy_w), .done(done_w),
        .token_id(token_id_w), .pos(pos), .s_len(s_len),
        .logits(logits_w), .argmax(argmax_w),
        .em_req(em_req_w), .em_tok(em_tok_w), .em_idx(em_idx_w), .em_val(em_val_w),
        .db_layer(db_layer_w), .idx_fresh(idx_fresh_w), .idx_win(idx_win_w),
        .gn_req(gn_req_w), .gn_which(gn_which_w), .gn_idx(gn_idx_w), .gn_val(gn_val_w),
        .aw_req(aw_req_w), .aw_sel(aw_sel_w), .aw_grp(aw_grp_w), .aw_k(aw_k_w),
        .aw_col(aw_col_w), .aw_scale(aw_scale_w),
        .kc_req(kc_req_w), .kc_idx(kc_idx_w), .kc_ckv(kc_ckv_w), .kc_krope(kc_krope_w), .kc_valid(kc_valid_w),
        .rw_req(rw_req_w), .rw_k(rw_k_w), .rw_col(rw_col_w), .rw_scale(rw_scale_w),
        .fw_req(fw_req_w), .fw_sel(fw_sel_w), .fw_grp(fw_grp_w), .fw_k(fw_k_w),
        .fw_shared(fw_shared_w), .fw_eidx(fw_eidx_w),
        .fw_col(fw_col_w), .fw_col_up(fw_col_up_w), .fw_scale_g(fw_scale_g_w), .fw_scale_u(fw_scale_u_w),
        .fn_req(fn_req_w), .fn_idx(fn_idx_w), .fn_val(fn_val_w),
        .lw_req(lw_req_w), .lw_vtile(lw_vtile_w), .lw_k(lw_k_w), .lw_col(lw_col_w),
        .h_state()
    );
    always @(posedge clk) begin if (rst) kc_valid_w <= 1'b0; else kc_valid_w <= kc_req_w; end
    /* verilator lint_off UNUSEDSIGNAL */
    wire _u_w = &{1'b0, busy_w, em_req_w, em_tok_w, aw_req_w, fw_req_w, rw_req_w, gn_req_w, fn_req_w, lw_req_w, idx_fresh_w, idx_win_w};
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- BATCHED responder (reads shared ROMs, keyed on dut_w pull ports) ----
    integer tw, rew, ftw, fow, cdw, obdw; reg [15:0] scaw; reg dmodew;
    always @(em_tok_w or em_idx_w or start_w) em_val_w = EMB[em_tok_w][em_idx_w];
    always @(fn_idx_w or start_w) fn_val_w = GF[fn_idx_w];
    always @(lw_vtile_w or lw_k_w or start_w) begin lw_col_w = {LM_TN*16{1'b0}};
        for (tw=0;tw<LM_TN;tw=tw+1) lw_col_w[16*tw+:16] = Wlm[lw_vtile_w*LM_TN + tw][lw_k_w]; end
    always @(gn_which_w or gn_idx_w or db_layer_w or start_w) gn_val_w = gn_which_w ? G2[db_layer_w][gn_idx_w] : G1[db_layer_w][gn_idx_w];
    always @(aw_sel_w or aw_grp_w or aw_k_w or db_layer_w or start_w) begin
        aw_col_w = {PE_N*8{1'b0}}; aw_scale_w = {16*PE_N*A_NB{1'b0}};
        for (tw=0;tw<PE_N;tw=tw+1) case (aw_sel_w)
        4'd0: if (aw_grp_w*PE_N+tw<Q_LORA)   aw_col_w[8*tw+:8]=W_dq [db_layer_w][aw_grp_w*PE_N+tw][aw_k_w];
        4'd1: if (aw_grp_w*PE_N+tw<HQK)      aw_col_w[8*tw+:8]=W_uq [db_layer_w][aw_grp_w*PE_N+tw][aw_k_w];
        4'd2: if (aw_grp_w*PE_N+tw<KV_LORA)  aw_col_w[8*tw+:8]=W_dkv[db_layer_w][aw_grp_w*PE_N+tw][aw_k_w];
        4'd3: if (aw_grp_w*PE_N+tw<ROPE)     aw_col_w[8*tw+:8]=W_kr [db_layer_w][aw_grp_w*PE_N+tw][aw_k_w];
        4'd4: if (aw_grp_w*PE_N+tw<HNOPE)    aw_col_w[8*tw+:8]=W_uk [db_layer_w][aw_grp_w*PE_N+tw][aw_k_w];
        4'd5: if (aw_grp_w*PE_N+tw<HV)       aw_col_w[8*tw+:8]=W_uv [db_layer_w][aw_grp_w*PE_N+tw][aw_k_w];
        4'd6: if (aw_grp_w*PE_N+tw<MODEL_DIM)aw_col_w[8*tw+:8]=W_o  [db_layer_w][aw_grp_w*PE_N+tw][aw_k_w];
        default: aw_col_w[8*tw+:8]=8'h0; endcase
        case (aw_sel_w) 4'd0:scaw=ScW_dq[db_layer_w]; 4'd1:scaw=ScW_uq[db_layer_w]; 4'd2:scaw=ScW_dkv[db_layer_w];
        4'd3:scaw=ScW_kr[db_layer_w]; 4'd4:scaw=ScW_uk[db_layer_w]; 4'd5:scaw=ScW_uv[db_layer_w];
        4'd6:scaw=ScW_o[db_layer_w]; default:scaw=16'h3F80; endcase
        for (tw=0;tw<PE_N;tw=tw+1) aw_scale_w[16*tw+:16]=scaw;
    end
    always @(kc_idx_w or db_layer_w or start_w) begin kc_ckv_w={KV_LORA*16{1'b0}}; kc_krope_w={ROPE*16{1'b0}};
        for (cdw=0;cdw<KV_LORA;cdw=cdw+1) kc_ckv_w[16*cdw+:16]=CKV[db_layer_w][kc_idx_w][cdw];
        for (cdw=0;cdw<ROPE;cdw=cdw+1)    kc_krope_w[16*cdw+:16]=KRP[db_layer_w][kc_idx_w][cdw]; end
    always @(rw_k_w or db_layer_w or start_w) begin rw_col_w={8*N_EXPERT{1'b0}}; rw_scale_w={16*N_EXPERT*R_NB{1'b0}};
        for (rew=0;rew<N_EXPERT;rew=rew+1) begin rw_col_w[8*rew+:8]=Wg[db_layer_w][rw_k_w][rew]; rw_scale_w[16*rew+:16]=ScWg[db_layer_w]; end end
    always @(fw_grp_w or fw_k_w or fw_sel_w or fw_shared_w or fw_eidx_w or db_layer_w or start_w) begin dmodew=(db_layer_w<N_DENSE)?1'b0:1'b1;
        fw_col_w={8*TN{1'b0}}; fw_col_up_w={8*TN{1'b0}}; fw_scale_g_w={16*TN*FF_NB_D{1'b0}}; fw_scale_u_w={16*TN*FF_NB_D{1'b0}};
        obdw=(fw_grp_w*TN)/BLK;
        for (ftw=0;ftw<TN;ftw=ftw+1) begin fow=fw_grp_w*TN+ftw;
            if (dmodew==1'b0) begin
                if (fw_sel_w==2'd2) begin if (fow<MODEL_DIM) fw_col_w[8*ftw+:8]=Dd[db_layer_w][fow][fw_k_w]; end
                else begin if (fow<INTER_DENSE) begin fw_col_w[8*ftw+:8]=Dg[db_layer_w][fow][fw_k_w]; fw_col_up_w[8*ftw+:8]=Du[db_layer_w][fow][fw_k_w]; end end
            end else begin
                if (fw_shared_w) begin
                    if (fw_sel_w==2'd2) begin if (fow<MODEL_DIM) fw_col_w[8*ftw+:8]=SHd[db_layer_w][fow][fw_k_w]; end
                    else if (fow<INTER_MOE) begin fw_col_w[8*ftw+:8]=SHg[db_layer_w][fow][fw_k_w]; fw_col_up_w[8*ftw+:8]=SHu[db_layer_w][fow][fw_k_w]; end
                end else begin
                    if (fw_sel_w==2'd2) begin if (fow<MODEL_DIM) fw_col_w[8*ftw+:8]=Md[db_layer_w][fw_eidx_w][fow][fw_k_w]; end
                    else if (fow<INTER_MOE) begin fw_col_w[8*ftw+:8]=Mg[db_layer_w][fw_eidx_w][fow][fw_k_w]; fw_col_up_w[8*ftw+:8]=Mu[db_layer_w][fw_eidx_w][fow][fw_k_w]; end
                end end end
        for (ftw=0;ftw<TN;ftw=ftw+1) begin
            if (dmodew==1'b0) begin
                if (fw_sel_w==2'd2) begin fw_scale_g_w[16*(0*TN+ftw)+:16]=ScDd[db_layer_w][0]; fw_scale_g_w[16*(1*TN+ftw)+:16]=ScDd[db_layer_w][1]; end
                else begin fw_scale_g_w[16*(0*TN+ftw)+:16]=ScDg[db_layer_w][obdw]; fw_scale_g_w[16*(1*TN+ftw)+:16]=ScDg[db_layer_w][obdw];
                    fw_scale_u_w[16*(0*TN+ftw)+:16]=ScDu[db_layer_w][obdw]; fw_scale_u_w[16*(1*TN+ftw)+:16]=ScDu[db_layer_w][obdw]; end
            end else begin
                if (fw_shared_w) begin
                    if (fw_sel_w==2'd2) fw_scale_g_w[16*(0*TN+ftw)+:16]=ScSHd[db_layer_w];
                    else begin fw_scale_g_w[16*(0*TN+ftw)+:16]=ScSHg[db_layer_w]; fw_scale_u_w[16*(0*TN+ftw)+:16]=ScSHu[db_layer_w]; end
                end else begin
                    if (fw_sel_w==2'd2) fw_scale_g_w[16*(0*TN+ftw)+:16]=ScMd[db_layer_w][fw_eidx_w];
                    else begin fw_scale_g_w[16*(0*TN+ftw)+:16]=ScMg[db_layer_w][fw_eidx_w]; fw_scale_u_w[16*(0*TN+ftw)+:16]=ScMu[db_layer_w][fw_eidx_w]; end
                end end end
    end

    //========================================================================
    // captured reference results (per row) + weight-fetch count
    //========================================================================
    reg [TOKW-1:0]      ref_argmax [0:B-1];
    reg [VOCAB*16-1:0]  ref_logits [0:B-1];
    integer             aw_single;     // aw_req pulses in ONE PE_M=1 run

    integer rc;
    task run_ref_row; input integer b; input integer tokb; integer wd; begin
        token_id_s = tokb[TOKW-1:0];
        @(negedge clk); start_s = 1'b1; @(negedge clk); start_s = 1'b0;
        wd = 0; rc = 0;
        while (!done_s && wd < 2000000) begin
            @(negedge clk);
            if (aw_req_s) rc = rc + 1;
            wd = wd + 1;
        end
        @(negedge clk);
        ref_argmax[b] = argmax_s;
        ref_logits[b] = logits_s;
        if (b == 0) aw_single = rc;
    end endtask

    integer wd2, rc2, b, fail_this;
    reg [TOKW-1:0]     gi; reg [VOCAB*16-1:0] gl;
    task run_batched_and_check; input integer tokA; input integer tokB; begin
        test_count = test_count + 1;
        token_id_w = { tokB[TOKW-1:0], tokA[TOKW-1:0] };   // row0=A (low), row1=B
        @(negedge clk); start_w = 1'b1; @(negedge clk); start_w = 1'b0;
        wd2 = 0; rc2 = 0;
        while (!done_w && wd2 < 2000000) begin
            @(negedge clk);
            if (aw_req_w) rc2 = rc2 + 1;
            wd2 = wd2 + 1;
        end
        fail_this = 0;
        if (!done_w) begin $display("FAIL: batched TIMEOUT"); errors=errors+1; disable run_batched_and_check; end
        @(negedge clk);
        for (b = 0; b < B; b = b + 1) begin
            gi = argmax_w[TOKW*b +: TOKW];
            gl = logits_w[VOCAB*16*b +: VOCAB*16];
            if (^gi === 1'bx || ^gl === 1'bx) begin
                $display("FAIL row %0d: X in batched outputs", b); fail_this=1;
            end else begin
                if (gi !== ref_argmax[b]) begin
                    $display("FAIL row %0d: argmax %0d != ref %0d", b, gi, ref_argmax[b]); fail_this=1;
                end
                if (gl !== ref_logits[b]) begin
                    $display("FAIL row %0d: logits mismatch vs ref (bit-exact)", b); fail_this=1;
                end
            end
        end
        if (rc2 != aw_single) begin
            $display("FAIL: batched aw_req=%0d != single %0d (attn weight stream NOT shared)", rc2, aw_single);
            fail_this=1;
        end
        if (fail_this) errors = errors + 1;
        else $display("ok  PE_M=%0d tokens {%0d,%0d}: row argmax {%0d,%0d} == ref; logits bit-exact; aw_req batched=%0d single=%0d (shared)",
                      B, tokA, tokB, ref_argmax[0], ref_argmax[1], rc2, aw_single);
    end endtask

    task do_case; input integer tokA; input integer tokB; begin
        run_ref_row(0, tokA);
        run_ref_row(1, tokB);
        run_batched_and_check(tokA, tokB);
    end endtask

    // ---- watchdog ----
    initial begin #2000000000; $display("FAIL: global timeout"); $fatal; end

    initial begin
        start_s = 1'b0; start_w = 1'b0;
        token_id_s = {TOKW{1'b0}}; token_id_w = {B*TOKW{1'b0}};
        pos = {POSW{1'b0}}; s_len = {(IDXW+1){1'b0}};
        rst = 1'b1;
        repeat(4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // two distinct tokens (different embeddings -> divergent per-row MoE routing)
        pos = 0;  s_len = 1;     build_stimulus(500,  0); do_case(3, 11);
        pos = 37; s_len = 2;     build_stimulus(7000, 0); do_case(11, 5);
        pos = 42; s_len = S_MAX; build_stimulus(90000,1); do_case(5, 9);

        if (errors != 0) begin
            $display("FAILED: %0d mismatch(es) across %0d tests", errors, test_count);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED", test_count);
        $finish;
    end
endmodule
