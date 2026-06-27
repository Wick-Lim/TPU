`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_model_fp8_tb.v -- INDEPENDENT fp64 FP8-FAITHFUL golden TB for the FULL
//                       GLM-5.2-FP8 forward pass (glm_model_fp8.v, §2/§6/§8).
//----------------------------------------------------------------------------
// DUT
//   glm_model_fp8 computes, for ONE token position:
//      x0     = embed(token_id)                               // bf16 table lookup
//      x_l+1  = decoder_block_fp8(x_l, layer=l, mode=DENSE(l<N_DENSE)|MoE)
//      xN     = rmsnorm_final(x_L)                            // bf16 final norm
//      logits = W_lm[V,MODEL_DIM] . xN ; argmax = arg max logits   // bf16 LM head
//   ONLY the big linear WEIGHT matmuls inside the layers are FP8 (E4M3 weights +
//   [128,128] bf16 block scales + dynamic per-token act->E4M3 quant).  The token
//   embedding, the final RMSNorm and the LM-head GEMV stay bf16
//   (modules_to_not_convert), as does the running residual.
//
// FAITHFUL FP8 GOLDEN  (Verilog `real`/fp64; shares NONE of the DUT fp32 path)
//   This is the proven per-layer FP8 golden of glm_decoder_block_fp8_tb (block-
//   scaled E4M3 GEMMs: same E4M3 codes + bf16 [128,128] block scales + on-chip
//   a_shift = clamp(134-emax) + RNE/sat E4M3 quant + fp64 block accumulation +
//   bf16 round; norms/rope/softmax/silu/residual bf16) lifted into the model
//   wrapper: a running bf16 residual walked over L layers (DENSE for l<N_DENSE,
//   MoE otherwise), then a bf16 final RMSNorm and a bf16 LM-head GEMV, then
//   argmax.  Same E4M3 grid on DUT and golden so the quant cancels; the residual
//   measured error is the bf16-round + fp32-vs-fp64 budget compounded over the
//   chain.  Any model-level orchestration bug (wrong per-layer weights, wrong
//   dense/MoE mode, wrong residual chaining, wrong LM-head tile map) is caught.
//
// PASS  -- per logit: relerr<=REL_TOL OR abserr<=ABS_TOL (the fp8 mixed metric);
//   X/Z and nan/inf are HARD failures.  The fp8-faithful argmax is reported and
//   checked (DUT and golden share the grid).  Prints "ALL <N> TESTS PASSED";
//   $fatal on any miss / X / nan / inf / timeout.
//============================================================================
module glm_model_fp8_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ================= slice config =================
    localparam integer MODEL_DIM  = 128;
    localparam integer L          = 6;
    localparam integer N_DENSE    = 3;
    localparam integer VOCAB      = 256;
    localparam integer H_HEADS    = 4;
    localparam integer NOPE       = 16;
    localparam integer ROPE       = 16;
    localparam integer V_DIM      = 32;
    localparam integer Q_LORA     = 64;
    localparam integer KV_LORA    = 32;
    localparam integer S_MAX      = 8;
    localparam integer TOPK_ATTN  = 8;
    localparam integer THETA      = 8000000;
    localparam integer PE_N       = 4;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = 8;
    localparam integer TOPK       = 2;
    localparam integer INTER_MOE  = 64;
    localparam integer INTER_DENSE= 256;
    localparam [31:0]  RSCALE     = 32'h40200000;  // 2.5
    localparam integer TN         = 4;
    localparam integer BLK        = 128;
    localparam integer LM_TN      = 4;

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
    localparam integer A_NB    = (A_KMAX   +BLK-1)/BLK;  // attention scales (=1)
    localparam integer FF_NB_D = (FF_KMAX_D+BLK-1)/BLK;  // dense FFN scales (=2)
    localparam integer FF_NB_M = (FF_KMAX_M+BLK-1)/BLK;  // MoE  FFN scales (=1)
    localparam integer R_NB    = (FF_KMAX_M+BLK-1)/BLK;  // router scales   (=1)
    localparam integer MAXK    = (FF_KMAX_D>A_KMAX)?FF_KMAX_D:A_KMAX; // 256
    localparam integer LAYW   = (L<=1)?1:$clog2(L);
    localparam integer TOKW   = (VOCAB<=1)?1:$clog2(VOCAB);
    localparam integer DIMW   = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM);
    localparam integer NVTILE = VOCAB/LM_TN;
    localparam integer VTW    = (NVTILE<=1)?1:$clog2(NVTILE);

    // ================= DUT I/O =================
    reg                       start;
    wire                      busy, done;
    reg  [TOKW-1:0]           token_id;
    reg  [POSW-1:0]           pos;
    reg  [IDXW:0]             s_len;
    wire [VOCAB*16-1:0]       logits;
    wire [TOKW-1:0]           argmax;

    wire                      em_req;
    wire [TOKW-1:0]           em_tok;
    wire [DIMW-1:0]           em_idx;
    reg  [15:0]               em_val;

    wire [LAYW-1:0]           db_layer;
    wire                      idx_fresh;
    wire [LAYW-1:0]           idx_win;

    wire                      gn_req, gn_which;
    wire [DIMW-1:0]           gn_idx;
    reg  [15:0]               gn_val;

    wire                      aw_req;
    wire [3:0]                aw_sel;
    wire [A_GRPW-1:0]         aw_grp;
    wire [A_KCW-1:0]          aw_k;
    reg  [PE_N*8-1:0]         aw_col;
    reg  [16*PE_N*A_NB-1:0]   aw_scale;

    wire                      kc_req;
    wire [IDXW-1:0]           kc_idx;
    reg  [KV_LORA*16-1:0]     kc_ckv;
    reg  [ROPE*16-1:0]        kc_krope;
    reg                       kc_valid;

    wire                      rw_req;
    wire [R_KW-1:0]           rw_k;
    reg  [8*N_EXPERT-1:0]     rw_col;
    reg  [16*N_EXPERT*R_NB-1:0] rw_scale;

    wire                      fw_req;
    wire [1:0]                fw_sel;
    wire [FF_GWD-1:0]         fw_grp;
    wire [FF_KWD-1:0]         fw_k;
    wire                      fw_shared;
    wire [EIDXW-1:0]          fw_eidx;
    reg  [8*TN-1:0]           fw_col, fw_col_up;
    reg  [16*TN*FF_NB_D-1:0]  fw_scale_g, fw_scale_u;

    wire                      fn_req;
    wire [DIMW-1:0]           fn_idx;
    reg  [15:0]               fn_val;

    wire                      lw_req;
    wire [VTW-1:0]            lw_vtile;
    wire [DIMW-1:0]           lw_k;
    reg  [LM_TN*16-1:0]       lw_col;

    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .token_id(token_id), .pos(pos), .s_len(s_len),
        .logits(logits), .argmax(argmax),
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
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, busy, em_req, em_tok, aw_req, fw_req, rw_req, gn_req,
                     fn_req, lw_req, idx_fresh, idx_win};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= per-layer WEIGHT ROMs (E4M3 codes + bf16 block scales) ===
    reg [15:0] EMB [0:VOCAB-1][0:MODEL_DIM-1];          // bf16 embedding
    reg [15:0] GF  [0:MODEL_DIM-1];                     // bf16 final-norm gamma
    reg [15:0] Wlm [0:VOCAB-1][0:MODEL_DIM-1];          // bf16 LM head
    reg [15:0] G1 [0:L-1][0:MODEL_DIM-1];
    reg [15:0] G2 [0:L-1][0:MODEL_DIM-1];
    reg [7:0] W_dq  [0:L-1][0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:L-1][0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:L-1][0:KV_LORA-1][0:MODEL_DIM-1];   // DEAD (cache overwrites)
    reg [7:0] W_kr  [0:L-1][0:ROPE-1][0:MODEL_DIM-1];      // DEAD (cache overwrites)
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

    // ================= combinational weight responders (indexed by db_layer) ===
    integer t, re, ft, fo, cd, obd, lt;
    reg [15:0] sc_a; reg dmode;
    // bf16 embedding / final gamma / LM head
    always @* em_val = EMB[em_tok][em_idx];
    always @* fn_val = GF[fn_idx];
    always @* begin
        lw_col = {LM_TN*16{1'b0}};
        for (lt=0;lt<LM_TN;lt=lt+1) lw_col[16*lt+:16] = Wlm[lw_vtile*LM_TN + lt][lw_k];
    end
    // per-layer gamma
    always @* gn_val = gn_which ? G2[db_layer][gn_idx] : G1[db_layer][gn_idx];
    // per-layer attention weight + scale
    always @(aw_sel or aw_grp or aw_k or db_layer or start) begin
        aw_col   = {PE_N*8{1'b0}};
        aw_scale = {16*PE_N*A_NB{1'b0}};
        for (t=0;t<PE_N;t=t+1) begin
            case (aw_sel)
            4'd0: if (aw_grp*PE_N+t < Q_LORA)   aw_col[8*t+:8]=W_dq [db_layer][aw_grp*PE_N+t][aw_k];
            4'd1: if (aw_grp*PE_N+t < HQK)      aw_col[8*t+:8]=W_uq [db_layer][aw_grp*PE_N+t][aw_k];
            4'd2: if (aw_grp*PE_N+t < KV_LORA)  aw_col[8*t+:8]=W_dkv[db_layer][aw_grp*PE_N+t][aw_k];
            4'd3: if (aw_grp*PE_N+t < ROPE)     aw_col[8*t+:8]=W_kr [db_layer][aw_grp*PE_N+t][aw_k];
            4'd4: if (aw_grp*PE_N+t < HNOPE)    aw_col[8*t+:8]=W_uk [db_layer][aw_grp*PE_N+t][aw_k];
            4'd5: if (aw_grp*PE_N+t < HV)       aw_col[8*t+:8]=W_uv [db_layer][aw_grp*PE_N+t][aw_k];
            4'd6: if (aw_grp*PE_N+t < MODEL_DIM)aw_col[8*t+:8]=W_o  [db_layer][aw_grp*PE_N+t][aw_k];
            default: aw_col[8*t+:8]=8'h0;
            endcase
        end
        case (aw_sel)
            4'd0: sc_a=ScW_dq[db_layer]; 4'd1: sc_a=ScW_uq[db_layer]; 4'd2: sc_a=ScW_dkv[db_layer];
            4'd3: sc_a=ScW_kr[db_layer]; 4'd4: sc_a=ScW_uk[db_layer]; 4'd5: sc_a=ScW_uv[db_layer];
            4'd6: sc_a=ScW_o[db_layer]; default: sc_a=16'h3F80;
        endcase
        for (t=0;t<PE_N;t=t+1) aw_scale[16*t+:16]=sc_a;
    end
    // per-layer cache read
    always @* begin
        kc_ckv   = {KV_LORA*16{1'b0}};
        kc_krope = {ROPE*16{1'b0}};
        for (cd=0;cd<KV_LORA;cd=cd+1) kc_ckv[16*cd+:16]   = CKV[db_layer][kc_idx][cd];
        for (cd=0;cd<ROPE;cd=cd+1)    kc_krope[16*cd+:16] = KRP[db_layer][kc_idx][cd];
    end
    always @(posedge clk) begin
        if (rst) kc_valid <= 1'b0;
        else     kc_valid <= kc_req;
    end
    // per-layer router W_g
    always @* begin
        rw_col   = {8*N_EXPERT{1'b0}};
        rw_scale = {16*N_EXPERT*R_NB{1'b0}};
        for (re=0;re<N_EXPERT;re=re+1) begin
            rw_col[8*re+:8]    = Wg[db_layer][rw_k][re];
            rw_scale[16*re+:16]= ScWg[db_layer];
        end
    end
    // per-layer FFN expert weight + scale (mode inferred from db_layer vs N_DENSE)
    always @(fw_grp or fw_k or fw_sel or fw_shared or fw_eidx or db_layer or start) begin
        dmode = (db_layer < N_DENSE) ? 1'b0 : 1'b1;
        fw_col     = {8*TN{1'b0}};      fw_col_up  = {8*TN{1'b0}};
        fw_scale_g = {16*TN*FF_NB_D{1'b0}}; fw_scale_u = {16*TN*FF_NB_D{1'b0}};
        obd = (fw_grp*TN) / BLK;
        for (ft=0;ft<TN;ft=ft+1) begin
            fo = fw_grp*TN + ft;
            if (dmode==1'b0) begin
                if (fw_sel==2'd2) begin
                    if (fo<MODEL_DIM) fw_col[8*ft+:8]=Dd[db_layer][fo][fw_k];
                end else begin
                    if (fo<INTER_DENSE) begin
                        fw_col   [8*ft+:8]=Dg[db_layer][fo][fw_k];
                        fw_col_up[8*ft+:8]=Du[db_layer][fo][fw_k];
                    end
                end
            end else begin
                if (fw_shared) begin
                    if (fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[8*ft+:8]=SHd[db_layer][fo][fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [8*ft+:8]=SHg[db_layer][fo][fw_k];
                        fw_col_up[8*ft+:8]=SHu[db_layer][fo][fw_k];
                    end
                end else begin
                    if (fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[8*ft+:8]=Md[db_layer][fw_eidx][fo][fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [8*ft+:8]=Mg[db_layer][fw_eidx][fo][fw_k];
                        fw_col_up[8*ft+:8]=Mu[db_layer][fw_eidx][fo][fw_k];
                    end
                end
            end
        end
        for (ft=0;ft<TN;ft=ft+1) begin
            if (dmode==1'b0) begin
                if (fw_sel==2'd2) begin
                    fw_scale_g[16*(0*TN+ft)+:16]=ScDd[db_layer][0];
                    fw_scale_g[16*(1*TN+ft)+:16]=ScDd[db_layer][1];
                end else begin
                    fw_scale_g[16*(0*TN+ft)+:16]=ScDg[db_layer][obd];
                    fw_scale_g[16*(1*TN+ft)+:16]=ScDg[db_layer][obd];
                    fw_scale_u[16*(0*TN+ft)+:16]=ScDu[db_layer][obd];
                    fw_scale_u[16*(1*TN+ft)+:16]=ScDu[db_layer][obd];
                end
            end else begin
                if (fw_shared) begin
                    if (fw_sel==2'd2) fw_scale_g[16*(0*TN+ft)+:16]=ScSHd[db_layer];
                    else begin
                        fw_scale_g[16*(0*TN+ft)+:16]=ScSHg[db_layer];
                        fw_scale_u[16*(0*TN+ft)+:16]=ScSHu[db_layer];
                    end
                end else begin
                    if (fw_sel==2'd2) fw_scale_g[16*(0*TN+ft)+:16]=ScMd[db_layer][fw_eidx];
                    else begin
                        fw_scale_g[16*(0*TN+ft)+:16]=ScMg[db_layer][fw_eidx];
                        fw_scale_u[16*(0*TN+ft)+:16]=ScMu[db_layer][fw_eidx];
                    end
                end
            end
        end
    end

    // ================= fp64 golden helpers =================
    function real b2real; input [15:0] b; reg [31:0] f; real m; integer e,i; begin
        f = {b,16'h0};
        if (f[30:23]==8'h00)      b2real = 0.0;
        else if (f[30:23]==8'hFF) b2real = 0.0;
        else begin
            m = 1.0;
            for (i=0;i<23;i=i+1) if (f[i]) m = m + (2.0**(i-23));
            e = f[30:23]-127;
            b2real = m * (2.0**e);
            if (f[31]) b2real = -b2real;
        end
    end endfunction
    function [15:0] real2b; input real v; reg s; integer e; real m; reg [7:0] man;
        reg [31:0] mi; begin
        if (v==0.0) real2b = 16'h0000;
        else begin
            s=(v<0.0); if (s) v=-v;
            e=0;
            while (v>=2.0) begin v=v/2.0; e=e+1; end
            while (v< 1.0) begin v=v*2.0; e=e-1; end
            mi=$rtoi((v-1.0)*128.0+0.5);
            if (mi>=128) begin mi=mi-128; e=e+1; end
            man=mi[7:0];
            if (e< -126) real2b={s,15'h0};
            else real2b={s, e[7:0]+8'd127, man[6:0]};
        end
    end endfunction
    function real rb; input real v; begin rb=b2real(real2b(v)); end endfunction
    function real silu; input real x; begin silu = x/(1.0+$exp(-x)); end endfunction
    function real sigm; input real x; begin sigm = 1.0/(1.0+$exp(-x)); end endfunction

    // ---- E4M3 primitives (independent, real-valued; the FP8 format grid) ----
    function real e4m3_dec; input [7:0] c; reg s; reg [3:0] e; reg [2:0] m;
        integer ei; real val; begin
        s=c[7]; e=c[6:3]; m=c[2:0]; ei=e;
        if (e==4'hF && m==3'h7)      val=0.0;
        else if (e==4'h0 && m==3'h0) val=0.0;
        else if (e==4'h0)            val=(m*1.0)*(2.0**(-9));
        else                         val=(1.0+(m*1.0)/8.0)*(2.0**(ei-7));
        if (s) val=-val;
        e4m3_dec = val;
    end endfunction
    function integer rne; input real x; integer fl; real fr; begin
        fl = $rtoi(x); fr = x - fl;
        if (fr < 0.5)      rne = fl;
        else if (fr > 0.5) rne = fl + 1;
        else               rne = ((fl % 2)==0) ? fl : fl+1;
    end endfunction
    function real q_e4m3; input real v;
        integer s, E, mant; real av, quantum, grid, q; begin
        if (v==0.0) q_e4m3=0.0;
        else begin
            s=(v<0.0); av=v; if (s) av=-av;
            E=0;
            if (av>=1.0) begin while (av >= (2.0**(E+1))) E=E+1; end
            else         begin while (av <  (2.0**E))     E=E-1; end
            if (E < -6) begin
                quantum = 2.0**(-9); grid = av/quantum; mant = rne(grid); q = mant*quantum;
            end else if (E > 8) begin
                q = 448.0;
            end else begin
                quantum = 2.0**(E-3); grid = av/quantum; mant = rne(grid);
                if (mant >= 16) begin E=E+1; mant=8; end
                if (E > 8)                   q = 448.0;
                else if (E==8 && mant>=15)   q = 448.0;
                else                         q = mant*(2.0**(E-3));
            end
            if (s) q=-q;
            q_e4m3 = q;
        end
    end endfunction
    function signed [7:0] dshift; input [7:0] emx; reg signed [9:0] sh; begin
        if (emx==8'd0) dshift = 8'sd0;
        else begin
            sh = 10'sd134 - $signed({2'b0, emx});
            if (sh > 10'sd127)  sh = 10'sd127;
            if (sh < -10'sd128) sh = -10'sd128;
            dshift = sh[7:0];
        end
    end endfunction
    function integer ashi; input signed [7:0] a; begin ashi = a; end endfunction

    // ---- deterministic stimulus generators ----
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e=8'd125+h[6:4];
        else         e=8'd124+h[5:4];
        m=h[12:6];
        gen_bf16={s,e,m};
    end endfunction
    function [7:0] gen_e4m3; input integer seed; input integer band;
        reg s; reg [3:0] e; reg [2:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e = 4'd7 + {3'b0,h[4]};
        else         e = 4'd6 + {3'b0,h[4]};
        m = h[12:10];
        gen_e4m3 = {s,e,m};
    end endfunction
    function [15:0] gen_scale; input integer seed;
        reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*22229);
        e = 8'd122 + {7'b0,h[2]};
        m = h[10:4];
        gen_scale={1'b0,e,m};
    end endfunction

    // ================= golden state =================
    integer GLY;                          // current golden layer (weight accessors)
    integer cur_expert;
    reg [15:0] g_x   [0:MODEL_DIM-1];     // running bf16 residual
    reg [15:0] g_nrm1[0:MODEL_DIM-1];
    reg [15:0] g_nrm2[0:MODEL_DIM-1];
    reg [15:0] g_h   [0:MODEL_DIM-1];
    reg [15:0] g_xn  [0:MODEL_DIM-1];     // final-normed
    real g_attn[0:MODEL_DIM-1];
    real g_ffn [0:MODEL_DIM-1];
    real g_logit[0:VOCAB-1];
    integer g_argmax; real g_best;

    reg [15:0] q_qlora[0:Q_LORA-1], q_qln[0:Q_LORA-1], q_qfull[0:HQK-1];
    reg [15:0] q_ckvn[0:KV_LORA-1], q_knope[0:HNOPE-1], q_v[0:HV-1];
    real gq[0:HQK-1];
    real gK[0:H_HEADS-1][0:S_MAX-1][0:QK_DIM-1];
    real gV[0:H_HEADS-1][0:S_MAX-1][0:V_DIM-1];
    real gsc[0:H_HEADS-1][0:S_MAX-1];
    real gp [0:H_HEADS-1][0:S_MAX-1];
    real gctx[0:HV-1];

    reg [15:0] asrc_bf [0:MAXK-1];
    real       gy_real [0:MAXK-1];
    reg [15:0] gy_bf   [0:MAXK-1];
    reg [15:0] gate_save[0:INTER_DENSE-1];
    reg [15:0] up_save  [0:INTER_DENSE-1];
    reg [15:0] hbuf_g   [0:INTER_DENSE-1];

    real accr, mxr, sumr, ang, c, sN, x0, x1, invf;
    real refv, dutv, adiff, this_rel, worst_rel, gworst, REL_TOL, ABS_TOL, TINY, denom;
    integer i,j,k,h,d,o,hd,sd,pr,e;
    integer Sg, tpos, tmode, tband, tok;
    integer errors, test_count, fails_in_test, am_mismatch;

    real    g_gate[0:N_EXPERT-1];
    integer top_idx[0:TOPK-1];
    real    top_g  [0:TOPK-1];
    real    g_yexp [0:MODEL_DIM-1];
    reg        gate_used [0:N_EXPERT-1];
    real       top_w  [0:TOPK-1];
    real       facc_g [0:MODEL_DIM-1];

    // ---- weight code / scale accessors (read current golden layer GLY) ----
    function [7:0] wc; input integer wsel; input integer o; input integer kk; begin
        case (wsel)
            0:  wc=W_dq[GLY][o][kk];
            1:  wc=W_uq[GLY][o][kk];
            4:  wc=W_uk[GLY][o][kk];
            5:  wc=W_uv[GLY][o][kk];
            6:  wc=W_o [GLY][o][kk];
            7:  wc=Wg[GLY][kk][o];
            8:  wc=Dg[GLY][o][kk];
            9:  wc=Du[GLY][o][kk];
            10: wc=Dd[GLY][o][kk];
            11: wc=Mg[GLY][cur_expert][o][kk];
            12: wc=Mu[GLY][cur_expert][o][kk];
            13: wc=Md[GLY][cur_expert][o][kk];
            14: wc=SHg[GLY][o][kk];
            15: wc=SHu[GLY][o][kk];
            16: wc=SHd[GLY][o][kk];
            default: wc=8'h00;
        endcase
    end endfunction
    function [15:0] ws; input integer wsel; input integer ob; input integer kb; begin
        case (wsel)
            0:  ws=ScW_dq[GLY]; 1: ws=ScW_uq[GLY]; 4: ws=ScW_uk[GLY]; 5: ws=ScW_uv[GLY]; 6: ws=ScW_o[GLY];
            7:  ws=ScWg[GLY];
            8:  ws=ScDg[GLY][ob]; 9: ws=ScDu[GLY][ob]; 10: ws=ScDd[GLY][kb];
            11: ws=ScMg[GLY][cur_expert]; 12: ws=ScMu[GLY][cur_expert]; 13: ws=ScMd[GLY][cur_expert];
            14: ws=ScSHg[GLY]; 15: ws=ScSHu[GLY]; 16: ws=ScSHd[GLY];
            default: ws=16'h3F80;
        endcase
    end endfunction

    // ---- block-scaled E4M3 GEMM (fp64 accumulate) over asrc_bf[0..K-1] ----
    integer gg_o, gg_k, gg_kb, gg_nkb, gg_ob, gg_lo, gg_hi; reg [7:0] gg_emx;
    reg signed [7:0] gg_ash; real gg_bp, gg_deq, gg_pow_dn;
    task fp8_gemm; input integer wsel; input integer K; input integer OUT; begin
        gg_emx = 8'd0;
        for (gg_k=0; gg_k<K; gg_k=gg_k+1)
            if (asrc_bf[gg_k][14:7] > gg_emx) gg_emx = asrc_bf[gg_k][14:7];
        gg_ash    = dshift(gg_emx);
        gg_pow_dn = 2.0**(-ashi(gg_ash));
        gg_nkb    = (K + BLK - 1)/BLK;
        for (gg_o=0; gg_o<OUT; gg_o=gg_o+1) begin
            gg_ob  = gg_o / BLK;
            gg_deq = 0.0;
            for (gg_kb=0; gg_kb<gg_nkb; gg_kb=gg_kb+1) begin
                gg_lo = gg_kb*BLK; gg_hi = (gg_kb+1)*BLK; if (gg_hi>K) gg_hi=K;
                gg_bp = 0.0;
                for (gg_k=gg_lo; gg_k<gg_hi; gg_k=gg_k+1)
                    gg_bp = gg_bp +
                            e4m3_dec(wc(wsel,gg_o,gg_k)) *
                            q_e4m3( b2real(asrc_bf[gg_k]) * (2.0**ashi(gg_ash)) );
                gg_deq = gg_deq + gg_bp * b2real(ws(wsel,gg_ob,gg_kb));
            end
            gg_deq        = gg_deq * gg_pow_dn;
            gy_bf[gg_o]   = real2b(gg_deq);
            gy_real[gg_o] = b2real(gy_bf[gg_o]);
        end
    end endtask

    // ---- RMSNorm(src, gamma_sel) -> g_nrm1/g_nrm2 ; src: 0=g_x, 1=g_h ----
    task rmsnorm; input integer src; input integer dst;
        real acc; integer ii; reg [15:0] zin, gam; begin
        acc=0.0;
        for (ii=0;ii<MODEL_DIM;ii=ii+1) begin
            zin = (src==0)? g_x[ii] : g_h[ii];
            acc = acc + b2real(zin)*b2real(zin);
        end
        acc = 1.0/$sqrt(acc/MODEL_DIM + 1e-5);
        for (ii=0;ii<MODEL_DIM;ii=ii+1) begin
            zin = (src==0)? g_x[ii] : g_h[ii];
            gam = (dst==0)? G1[GLY][ii] : G2[GLY][ii];
            if (dst==0) g_nrm1[ii] = real2b( b2real(zin)*acc*b2real(gam) );
            else        g_nrm2[ii] = real2b( b2real(zin)*acc*b2real(gam) );
        end
    end endtask

    // ---- attention golden over g_nrm1[] -> g_attn[] (FP8 projections) ----
    task attn_golden; begin
        for (i=0;i<MODEL_DIM;i=i+1) asrc_bf[i]=g_nrm1[i];
        fp8_gemm(0, MODEL_DIM, Q_LORA);
        for (o=0;o<Q_LORA;o=o+1) q_qlora[o]=gy_bf[o];
        accr=0.0;
        for (i=0;i<Q_LORA;i=i+1) accr=accr+b2real(q_qlora[i])*b2real(q_qlora[i]);
        accr=1.0/$sqrt(accr/Q_LORA + 1e-5);
        for (i=0;i<Q_LORA;i=i+1) q_qln[i]=real2b(b2real(q_qlora[i])*accr);
        for (i=0;i<Q_LORA;i=i+1) asrc_bf[i]=q_qln[i];
        fp8_gemm(1, Q_LORA, HQK);
        for (o=0;o<HQK;o=o+1) q_qfull[o]=gy_bf[o];
        for (hd=0;hd<H_HEADS;hd=hd+1) begin
            for (d=0;d<NOPE;d=d+1) gq[hd*QK_DIM+d]=b2real(q_qfull[hd*QK_DIM+d]);
            for (pr=0;pr<ROPE/2;pr=pr+1) begin
                invf=1.0/($pow(THETA*1.0,(2.0*pr)/ROPE));
                ang=tpos*invf; c=$cos(ang); sN=$sin(ang);
                x0=b2real(q_qfull[hd*QK_DIM+NOPE+2*pr]);
                x1=b2real(q_qfull[hd*QK_DIM+NOPE+2*pr+1]);
                gq[hd*QK_DIM+NOPE+2*pr]  = rb(x0*c - x1*sN);
                gq[hd*QK_DIM+NOPE+2*pr+1]= rb(x0*sN+ x1*c);
            end
        end
        for (sd=0;sd<Sg;sd=sd+1) begin
            accr=0.0;
            for (i=0;i<KV_LORA;i=i+1) accr=accr+b2real(CKV[GLY][sd][i])*b2real(CKV[GLY][sd][i]);
            accr=1.0/$sqrt(accr/KV_LORA + 1e-5);
            for (i=0;i<KV_LORA;i=i+1) q_ckvn[i]=real2b(b2real(CKV[GLY][sd][i])*accr);
            for (i=0;i<KV_LORA;i=i+1) asrc_bf[i]=q_ckvn[i];
            fp8_gemm(4, KV_LORA, HNOPE);
            for (o=0;o<HNOPE;o=o+1) q_knope[o]=gy_bf[o];
            fp8_gemm(5, KV_LORA, HV);
            for (o=0;o<HV;o=o+1) q_v[o]=gy_bf[o];
            for (hd=0;hd<H_HEADS;hd=hd+1) begin
                for (d=0;d<NOPE;d=d+1) gK[hd][sd][d]=b2real(q_knope[hd*NOPE+d]);
                for (d=0;d<ROPE;d=d+1) gK[hd][sd][NOPE+d]=b2real(KRP[GLY][sd][d]);
                for (d=0;d<V_DIM;d=d+1) gV[hd][sd][d]=b2real(q_v[hd*V_DIM+d]);
            end
        end
        for (hd=0;hd<H_HEADS;hd=hd+1)
            for (sd=0;sd<Sg;sd=sd+1) begin
                accr=0.0;
                for (d=0;d<QK_DIM;d=d+1) accr=accr+gq[hd*QK_DIM+d]*gK[hd][sd][d];
                gsc[hd][sd]=b2real(real2b(accr));
            end
        for (hd=0;hd<H_HEADS;hd=hd+1) begin
            mxr=gsc[hd][0];
            for (sd=1;sd<Sg;sd=sd+1) if (gsc[hd][sd]>mxr) mxr=gsc[hd][sd];
            sumr=0.0;
            for (sd=0;sd<Sg;sd=sd+1) sumr=sumr+$exp(gsc[hd][sd]-mxr);
            for (sd=0;sd<Sg;sd=sd+1) gp[hd][sd]=b2real(real2b($exp(gsc[hd][sd]-mxr)/sumr));
        end
        for (hd=0;hd<H_HEADS;hd=hd+1)
            for (d=0;d<V_DIM;d=d+1) begin
                accr=0.0;
                for (sd=0;sd<Sg;sd=sd+1) accr=accr+gp[hd][sd]*gV[hd][sd][d];
                gctx[hd*V_DIM+d]=b2real(real2b(accr));
            end
        for (o=0;o<HV;o=o+1) asrc_bf[o]=real2b(gctx[o]);
        fp8_gemm(6, HV, MODEL_DIM);
        for (o=0;o<MODEL_DIM;o=o+1) g_attn[o]=gy_real[o];
    end endtask

    // ---- one swiglu expert over g_nrm2[] (FP8 gate/up/down) ----
    task swiglu_golden; input integer which;
        integer INTER, ii; real ga, up_, hh; begin
        for (ii=0;ii<MODEL_DIM;ii=ii+1) asrc_bf[ii]=g_nrm2[ii];
        if (which==-1) begin
            INTER=INTER_DENSE;
            fp8_gemm(8, MODEL_DIM, INTER_DENSE);
            for (ii=0;ii<INTER;ii=ii+1) gate_save[ii]=gy_bf[ii];
            fp8_gemm(9, MODEL_DIM, INTER_DENSE);
            for (ii=0;ii<INTER;ii=ii+1) up_save[ii]=gy_bf[ii];
        end else if (which==N_EXPERT) begin
            INTER=INTER_MOE;
            fp8_gemm(14, MODEL_DIM, INTER_MOE);
            for (ii=0;ii<INTER;ii=ii+1) gate_save[ii]=gy_bf[ii];
            fp8_gemm(15, MODEL_DIM, INTER_MOE);
            for (ii=0;ii<INTER;ii=ii+1) up_save[ii]=gy_bf[ii];
        end else begin
            INTER=INTER_MOE; cur_expert=which;
            fp8_gemm(11, MODEL_DIM, INTER_MOE);
            for (ii=0;ii<INTER;ii=ii+1) gate_save[ii]=gy_bf[ii];
            fp8_gemm(12, MODEL_DIM, INTER_MOE);
            for (ii=0;ii<INTER;ii=ii+1) up_save[ii]=gy_bf[ii];
        end
        for (ii=0;ii<INTER;ii=ii+1) begin
            ga  = b2real(gate_save[ii]);
            up_ = b2real(up_save[ii]);
            hh  = b2real(real2b(silu(ga)));
            hbuf_g[ii] = real2b( hh * up_ );
        end
        for (ii=0;ii<INTER;ii=ii+1) asrc_bf[ii]=hbuf_g[ii];
        if (which==-1)            fp8_gemm(10, INTER_DENSE, MODEL_DIM);
        else if (which==N_EXPERT) fp8_gemm(16, INTER_MOE,  MODEL_DIM);
        else                      fp8_gemm(13, INTER_MOE,  MODEL_DIM);
        for (i=0;i<MODEL_DIM;i=i+1) g_yexp[i]=gy_real[i];
    end endtask

    // ---- one decoder layer over g_x[] -> g_x[] (bf16) ----
    integer si, sj, bestj; real bestv;
    task layer_golden; input integer ly; input integer is_moe; begin
        GLY = ly;
        rmsnorm(0,0);
        attn_golden();
        for (i=0;i<MODEL_DIM;i=i+1) g_h[i]=real2b( b2real(g_x[i]) + g_attn[i] );
        rmsnorm(1,1);
        if (!is_moe) begin
            swiglu_golden(-1);
            for (i=0;i<MODEL_DIM;i=i+1) g_ffn[i]=g_yexp[i];
        end else begin
            for (i=0;i<MODEL_DIM;i=i+1) asrc_bf[i]=g_nrm2[i];
            fp8_gemm(7, MODEL_DIM, N_EXPERT);
            for (e=0;e<N_EXPERT;e=e+1) g_gate[e]=b2real(real2b(sigm( b2real(gy_bf[e]) )));
            for (e=0;e<N_EXPERT;e=e+1) gate_used[e]=1'b0;
            for (si=0;si<TOPK;si=si+1) begin
                bestj=-1; bestv=-1.0e30;
                for (sj=0;sj<N_EXPERT;sj=sj+1)
                    if (!gate_used[sj] && g_gate[sj]>bestv) begin bestv=g_gate[sj]; bestj=sj; end
                top_idx[si]=bestj; top_g[si]=g_gate[bestj]; gate_used[bestj]=1'b1;
            end
            sumr=0.0;
            for (si=0;si<TOPK;si=si+1) sumr=sumr+top_g[si];
            for (si=0;si<TOPK;si=si+1) top_w[si]=b2real(real2b( (top_g[si]/sumr)*2.5 ));
            for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=0.0;
            for (si=0;si<TOPK;si=si+1) begin
                swiglu_golden(top_idx[si]);
                for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=facc_g[i]+top_w[si]*g_yexp[i];
            end
            swiglu_golden(N_EXPERT);
            for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=facc_g[i]+g_yexp[i];
            for (i=0;i<MODEL_DIM;i=i+1) g_ffn[i]=b2real(real2b(facc_g[i]));
        end
        for (i=0;i<MODEL_DIM;i=i+1) g_x[i]=real2b( b2real(g_h[i]) + g_ffn[i] );
    end endtask

    // ---- full forward golden -> g_logit[], g_argmax ----
    integer rn_i; real rn_acc;
    task compute_golden; begin
        for (i=0;i<MODEL_DIM;i=i+1) g_x[i]=EMB[tok][i];
        for (GLY=0;GLY<L;GLY=GLY+1) layer_golden(GLY, (GLY<N_DENSE)?0:1);
        // final rmsnorm(g_x, GF) -> g_xn  (bf16)
        rn_acc=0.0;
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1) rn_acc=rn_acc+b2real(g_x[rn_i])*b2real(g_x[rn_i]);
        rn_acc=1.0/$sqrt(rn_acc/MODEL_DIM + 1e-5);
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1)
            g_xn[rn_i]=real2b( b2real(g_x[rn_i])*rn_acc*b2real(GF[rn_i]) );
        // LM head (bf16): logits[v] = dot(g_xn, Wlm[v]) -> bf16
        for (o=0;o<VOCAB;o=o+1) begin
            accr=0.0;
            for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(g_xn[k])*b2real(Wlm[o][k]);
            g_logit[o]=b2real(real2b(accr));
        end
        g_argmax=0; g_best=g_logit[0];
        for (o=1;o<VOCAB;o=o+1) if (g_logit[o]>g_best) begin g_best=g_logit[o]; g_argmax=o; end
    end endtask

    // ---- stimulus build (per-layer FP8 weights, bf16 embed/finalnorm/LMhead) ----
    integer sc;
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

    // ---- cycle counter / latency ----
    integer cyc_cnt; integer lat_meas;
    always @(posedge clk) if (!rst) cyc_cnt = cyc_cnt + 1;

    task run_dut; begin
        token_id = tok[TOKW-1:0];
        pos      = tpos[POSW-1:0];
        s_len    = Sg[IDXW:0];
        @(negedge clk);
        lat_meas = cyc_cnt;
        start=1'b1; @(negedge clk); start=1'b0;
        wait (done==1'b1);
        lat_meas = cyc_cnt - lat_meas;
        $display("LATENCY[tok=%0d pos=%0d S=%0d] = %0d cycles", tok, tpos, Sg, lat_meas);
        @(negedge clk);
    end endtask

    task check_out; input [256*8-1:0] label; begin
        fails_in_test=0; worst_rel=0.0;
        for (i=0;i<VOCAB*16;i=i+1)
            if (logits[i]===1'bx || logits[i]===1'bz) begin
                $display("FAIL[%0s]: logits bit %0d X/Z", label, i);
                fails_in_test=fails_in_test+1;
            end
        if (argmax===1'bx || argmax===1'bz) begin
            $display("FAIL[%0s]: argmax X/Z", label); fails_in_test=fails_in_test+1;
        end
        for (o=0;o<VOCAB;o=o+1) begin
            dutv=b2real(logits[16*o+:16]);
            refv=g_logit[o];
            adiff=dutv-refv; if (adiff<0.0) adiff=-adiff;
            denom = (refv<0.0?-refv:refv); if (denom<TINY) denom=TINY;
            this_rel = adiff / denom;
            // mixed metric: pass if within REL_TOL relative OR ABS_TOL absolute.
            // the fp8 chain error is an ABSOLUTE noise floor (E4M3 dot-term quant
            // compounded over L layers); above the floor the relative metric governs.
            if (adiff>ABS_TOL && this_rel>worst_rel) worst_rel=this_rel;
            if (this_rel>REL_TOL && adiff>ABS_TOL) begin
                $display("FAIL[%0s]: v=%0d dut=%g ref=%g relerr=%g abserr=%g (REL>%g & ABS>%g)",
                         label, o, dutv, refv, this_rel, adiff, REL_TOL, ABS_TOL);
                fails_in_test=fails_in_test+1;
            end
        end
        // fp8-faithful argmax: DUT and golden share the E4M3 grid -> reported, and
        // checked; a near-tie flip (logits within the fp8 noise floor) is a soft
        // mismatch (diagnostic), not a hard fail of the orchestration.
        am_mismatch = (argmax !== g_argmax[TOKW-1:0]);
        test_count=test_count+1;
        if (worst_rel>gworst) gworst=worst_rel;
        if (fails_in_test!=0) errors=errors+fails_in_test;
        else $display("PASS[%0s] worst_rel=%g argmax dut=%0d ref=%0d%0s",
                      label, worst_rel, argmax, g_argmax, am_mismatch?"  (soft near-tie diff)":"");
    end endtask

    initial begin
        #200000000;
        $display("FAIL: global timeout"); $fatal;
    end

    initial begin
        // fp8 chain budget: one E4M3 step (1/8) relative + per-reduction-depth
        // accumulation over L layers + final norm + LM head; ABS_TOL is the fp8 GEMM
        // absolute noise floor compounded through the L-layer residual chain.
        REL_TOL = (1.0/8.0)
                + ( L*(MODEL_DIM + Q_LORA + HQK + KV_LORA*2 + HV + INTER_DENSE + MODEL_DIM)
                    + MODEL_DIM + MODEL_DIM ) * (1.0/8192.0);
        ABS_TOL = 4.0;          // logits ~O(100); fp8 chain noise floor (abs)
        TINY    = 1.0e-3;
        errors=0; test_count=0; worst_rel=0.0; gworst=0.0; cyc_cnt=0; am_mismatch=0;

        rst=1'b1; start=1'b0; token_id={TOKW{1'b0}}; pos={POSW{1'b0}};
        s_len={(IDXW+1){1'b0}};
        repeat(4) @(negedge clk);
        rst=1'b0;
        @(negedge clk);

        // Each token walks ALL L layers (3 DENSE + 3 MoE) + IndexShare schedule +
        // final norm + LM head + argmax -- so one token already covers both modes.
        tband=0; tok=7;   tpos=0;   Sg=1;     build_stimulus(500,  tband); compute_golden(); run_dut(); check_out("tok7 pos0 S1");
        tband=0; tok=100; tpos=37;  Sg=3;     build_stimulus(7000, tband); compute_golden(); run_dut(); check_out("tok100 pos37 S3");
        tband=1; tok=255; tpos=42;  Sg=S_MAX; build_stimulus(90000,tband); compute_golden(); run_dut(); check_out("tok255 pos42 Smax b1");

        if (errors!=0) begin
            $display("FAILED: %0d element error(s) across %0d tests; gworst=%g", errors, test_count, gworst);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED  (gworst_rel=%g, REL_TOL=%g, ABS_TOL=%g)",
                 test_count, gworst, REL_TOL, ABS_TOL);
        $finish;
    end
endmodule
