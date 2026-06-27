`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// mtp_head_tb.v  --  INDEPENDENT fp64 golden TB for the GLM-5.2 MTP head
//                    (DeepSeek-V3-style t+2 speculative head, mtp_head.v).
//----------------------------------------------------------------------------
// DUT
//   mtp_head computes, for ONE (h_t, emb_t1, pos, S):
//      a    = RMSNorm(h_t , gamma_a)
//      b    = RMSNorm(emb , gamma_b)
//      h'   = W_proj @ [a;b]                         (MODEL_DIM x 2*MODEL_DIM)
//      y    = decoder_block(h', pos, kv, mode)
//      xN   = RMSNorm_final(y, gamma_final)
//      logits[V] = W_lm[V,MODEL_DIM] . xN ;  argmax = arg max logits
//
// INDEPENDENT GOLDEN  (Verilog `real`/fp64; shares NONE of the DUT fp32 path).
//   Every bf16 widened losslessly to real; every dot a true fp64 reduce; RMSNorm
//   rsqrt via $sqrt; RoPE via $cos/$sin; softmax/silu/sigmoid via $exp.  bf16
//   quantization applied at EXACTLY the boundaries the hardware rounds (both
//   combine-norm outputs, each projection output, then the proven decoder_block
//   golden, final-norm outputs, each LM-head logit).  The MoE top-K runs on the
//   bf16 gate with lower-index tie-break to match the DUT.  Any orchestration bug
//   (wrong norm gamma, wrong concat order, wrong W_proj transpose/tile mapping,
//   wrong residual chaining inside the block, wrong LM-head tile mapping, wrong
//   argmax) is caught.
//
// PASS / TOLERANCE
//   Per logit RELATIVE error vs the fp64 golden <= REL_TOL, AND exact argmax
//   match (ACCEL_GLM52 §6).  X/Z and nan/inf are hard failures.  Prints
//   "ALL <N> TESTS PASSED"; $fatal on any miss / X / nan / inf / argmax mismatch.
//============================================================================
module mtp_head_tb;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ================= slice config =================
    localparam integer MODEL_DIM  = 128;
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
    localparam integer LM_TN      = 4;
    localparam integer PROJ_TN    = 4;

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
    localparam integer TOKW   = (VOCAB<=1)?1:$clog2(VOCAB);
    localparam integer DIMW   = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM);
    localparam integer NVTILE = VOCAB/LM_TN;
    localparam integer VTW    = (NVTILE<=1)?1:$clog2(NVTILE);
    localparam integer CK     = 2*MODEL_DIM;
    localparam integer CKIW   = $clog2(CK);
    localparam integer NPTILE = MODEL_DIM/PROJ_TN;
    localparam integer PTW    = (NPTILE<=1)?1:$clog2(NPTILE);

    // ================= DUT I/O =================
    reg                       start;
    wire                      busy, done;
    reg                       mode;
    reg  [POSW-1:0]           pos;
    reg  [IDXW:0]             s_len;
    reg  [MODEL_DIM*16-1:0]   h_t;
    reg  [MODEL_DIM*16-1:0]   emb_t1;
    wire [VOCAB*16-1:0]       logits;
    wire [TOKW-1:0]           argmax;

    wire                      cn_req;
    wire [1:0]                cn_which;
    wire [DIMW-1:0]           cn_idx;
    reg  [15:0]               cn_val;

    wire                      pw_req;
    wire [PTW-1:0]            pw_ptile;
    wire [CKIW-1:0]           pw_k;
    reg  [PROJ_TN*16-1:0]     pw_col;

    wire                      gn_req, gn_which;
    wire [DIMW-1:0]           gn_idx;
    reg  [15:0]               gn_val;

    wire                      aw_req;
    wire [3:0]                aw_sel;
    wire [A_GRPW-1:0]         aw_grp;
    wire [A_KCW-1:0]          aw_k;
    reg  [PE_N*16-1:0]        aw_col;

    wire                      kc_req;
    wire [IDXW-1:0]           kc_idx;
    reg  [KV_LORA*16-1:0]     kc_ckv;
    reg  [ROPE*16-1:0]        kc_krope;
    reg                       kc_valid;

    wire                      rw_req;
    wire [R_KW-1:0]           rw_k;
    reg  [16*N_EXPERT-1:0]    rw_col;

    wire                      fw_req;
    wire [1:0]                fw_sel;
    wire [FF_GWD-1:0]         fw_grp;
    wire [FF_KWD-1:0]         fw_k;
    wire                      fw_shared;
    wire [EIDXW-1:0]          fw_eidx;
    reg  [16*TN-1:0]          fw_col, fw_col_up;

    wire                      lw_req;
    wire [VTW-1:0]            lw_vtile;
    wire [DIMW-1:0]           lw_k;
    reg  [LM_TN*16-1:0]       lw_col;

    mtp_head #(
        .MODEL_DIM(MODEL_DIM), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .LM_TN(LM_TN), .PROJ_TN(PROJ_TN)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .mode(mode), .pos(pos), .s_len(s_len),
        .h_t(h_t), .emb_t1(emb_t1),
        .logits(logits), .argmax(argmax),
        .cn_req(cn_req), .cn_which(cn_which), .cn_idx(cn_idx), .cn_val(cn_val),
        .pw_req(pw_req), .pw_ptile(pw_ptile), .pw_k(pw_k), .pw_col(pw_col),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k), .aw_col(aw_col),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx), .fw_col(fw_col), .fw_col_up(fw_col_up),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, busy, aw_req, fw_req, rw_req, gn_req, cn_req, pw_req,
                     lw_req};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= WEIGHT ROMs =================
    reg [15:0] GA [0:MODEL_DIM-1];                          // RMSNorm(h_t) gamma
    reg [15:0] GB [0:MODEL_DIM-1];                          // RMSNorm(emb) gamma
    reg [15:0] GF [0:MODEL_DIM-1];                          // final-norm gamma
    reg [15:0] Wp [0:MODEL_DIM-1][0:CK-1];                  // W_proj[out][in]
    // decoder-block weights (single layer)
    reg [15:0] G1 [0:MODEL_DIM-1];
    reg [15:0] G2 [0:MODEL_DIM-1];
    reg [15:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [15:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [15:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];          // exercised, unused val
    reg [15:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];             // exercised, unused val
    reg [15:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [15:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [15:0] W_o   [0:MODEL_DIM-1][0:HV-1];
    reg [15:0] CKV [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP [0:S_MAX-1][0:ROPE-1];
    reg [15:0] Wg [0:MODEL_DIM-1][0:N_EXPERT-1];
    reg [15:0] Dg [0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [15:0] Du [0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [15:0] Dd [0:MODEL_DIM-1][0:INTER_DENSE-1];
    reg [15:0] Mg [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] Mu [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] Md [0:N_EXPERT-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [15:0] SHg [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] SHu [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] SHd [0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [15:0] Wlm [0:VOCAB-1][0:MODEL_DIM-1];

    // ================= combinational pull responders =================
    integer t;
    // combine/final norm gamma : cn_which 0=h_t(GA),1=emb(GB),2=final(GF)
    always @* begin
        case (cn_which)
            2'd0:    cn_val = GA[cn_idx];
            2'd1:    cn_val = GB[cn_idx];
            default: cn_val = GF[cn_idx];
        endcase
    end
    // combine-projection weight : pw_col[t] = W_proj[ptile*PROJ_TN+t][pw_k]
    integer pt;
    always @* begin
        pw_col = {PROJ_TN*16{1'b0}};
        for (pt=0;pt<PROJ_TN;pt=pt+1) pw_col[16*pt+:16] = Wp[pw_ptile*PROJ_TN+pt][pw_k];
    end
    // decoder-block pre-norm gamma (0=pre-attn G1, 1=pre-FFN G2)
    always @* gn_val = gn_which ? G2[gn_idx] : G1[gn_idx];
    // attention weight pull
    always @(aw_sel or aw_grp or aw_k or start) begin
        aw_col = {PE_N*16{1'b0}};
        for (t=0;t<PE_N;t=t+1) begin
            case (aw_sel)
            4'd0: if (aw_grp*PE_N+t < Q_LORA)   aw_col[16*t+:16]=W_dq [aw_grp*PE_N+t][aw_k];
            4'd1: if (aw_grp*PE_N+t < HQK)      aw_col[16*t+:16]=W_uq [aw_grp*PE_N+t][aw_k];
            4'd2: if (aw_grp*PE_N+t < KV_LORA)  aw_col[16*t+:16]=W_dkv[aw_grp*PE_N+t][aw_k];
            4'd3: if (aw_grp*PE_N+t < ROPE)     aw_col[16*t+:16]=W_kr [aw_grp*PE_N+t][aw_k];
            4'd4: if (aw_grp*PE_N+t < HNOPE)    aw_col[16*t+:16]=W_uk [aw_grp*PE_N+t][aw_k];
            4'd5: if (aw_grp*PE_N+t < HV)       aw_col[16*t+:16]=W_uv [aw_grp*PE_N+t][aw_k];
            4'd6: if (aw_grp*PE_N+t < MODEL_DIM)aw_col[16*t+:16]=W_o  [aw_grp*PE_N+t][aw_k];
            default: aw_col[16*t+:16]=16'h0;
            endcase
        end
    end
    // cache read pull
    integer cd;
    always @* begin
        kc_ckv   = {KV_LORA*16{1'b0}};
        kc_krope = {ROPE*16{1'b0}};
        for (cd=0;cd<KV_LORA;cd=cd+1) kc_ckv[16*cd+:16]   = CKV[kc_idx][cd];
        for (cd=0;cd<ROPE;cd=cd+1)    kc_krope[16*cd+:16] = KRP[kc_idx][cd];
    end
    always @(posedge clk) begin
        if (rst) kc_valid <= 1'b0;
        else     kc_valid <= kc_req;
    end
    // router W_g pull
    integer re;
    always @* begin
        rw_col = {16*N_EXPERT{1'b0}};
        for (re=0;re<N_EXPERT;re=re+1) rw_col[16*re+:16] = Wg[rw_k][re];
    end
    // FFN expert weight pull (mode from the DUT `mode` input)
    integer ft, fo;
    always @(fw_grp or fw_k or fw_sel or fw_shared or fw_eidx or mode or start) begin
        fw_col = {16*TN{1'b0}}; fw_col_up = {16*TN{1'b0}};
        for (ft=0;ft<TN;ft=ft+1) begin
            fo = fw_grp*TN + ft;
            if (mode==1'b0) begin
                if (fw_sel==2'd2) begin
                    if (fo<MODEL_DIM) fw_col[16*ft+:16]=Dd[fo][fw_k];
                end else begin
                    if (fo<INTER_DENSE) begin
                        fw_col   [16*ft+:16]=Dg[fo][fw_k];
                        fw_col_up[16*ft+:16]=Du[fo][fw_k];
                    end
                end
            end else begin
                if (fw_shared) begin
                    if (fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[16*ft+:16]=SHd[fo][fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [16*ft+:16]=SHg[fo][fw_k];
                        fw_col_up[16*ft+:16]=SHu[fo][fw_k];
                    end
                end else begin
                    if (fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[16*ft+:16]=Md[fw_eidx][fo][fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [16*ft+:16]=Mg[fw_eidx][fo][fw_k];
                        fw_col_up[16*ft+:16]=Mu[fw_eidx][fo][fw_k];
                    end
                end
            end
        end
    end
    // LM-head weight pull
    integer lt;
    always @* begin
        lw_col = {LM_TN*16{1'b0}};
        for (lt=0;lt<LM_TN;lt=lt+1) lw_col[16*lt+:16] = Wlm[lw_vtile*LM_TN + lt][lw_k];
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

    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e=8'd125+h[6:4];
        else         e=8'd124+h[5:4];
        m=h[12:6];
        gen_bf16={s,e,m};
    end endfunction

    // ================= golden state =================
    reg [15:0] g_a   [0:MODEL_DIM-1];   // RMSNorm(h_t)
    reg [15:0] g_b   [0:MODEL_DIM-1];   // RMSNorm(emb)
    reg [15:0] g_cat [0:CK-1];          // concat
    reg [15:0] g_hp  [0:MODEL_DIM-1];   // h' = W_proj @ cat
    reg [15:0] g_x   [0:MODEL_DIM-1];   // residual stream through the block
    reg [15:0] g_nrm1[0:MODEL_DIM-1];
    reg [15:0] g_nrm2[0:MODEL_DIM-1];
    reg [15:0] g_h   [0:MODEL_DIM-1];
    reg [15:0] g_xn  [0:MODEL_DIM-1];
    real g_attn[0:MODEL_DIM-1];
    real g_ffn [0:MODEL_DIM-1];
    real g_logit[0:VOCAB-1];

    // attention golden scratch
    reg [15:0] q_qlora[0:Q_LORA-1], q_qln[0:Q_LORA-1], q_qfull[0:HQK-1];
    reg [15:0] q_ckvn[0:KV_LORA-1], q_knope[0:HNOPE-1], q_v[0:HV-1];
    real gq[0:HQK-1];
    real gK[0:H_HEADS-1][0:S_MAX-1][0:QK_DIM-1];
    real gV[0:H_HEADS-1][0:S_MAX-1][0:V_DIM-1];
    real gsc[0:H_HEADS-1][0:S_MAX-1];
    real gp [0:H_HEADS-1][0:S_MAX-1];
    real gctx[0:HV-1];

    real accr, mxr, sumr, ang, c, sN, x0, x1, invf;
    real refv, dutv, adiff, this_rel, worst_rel, REL_TOL, TINY;
    integer i,j,k,h,d,o,hd,sd,pr,e;
    integer Sg, tpos, tband, tmode;
    real    rn_acc; integer rn_i;
    real    g_gate[0:N_EXPERT-1];
    integer top_idx[0:TOPK-1];
    real    top_g  [0:TOPK-1];
    real    g_yexp [0:MODEL_DIM-1];
    reg [15:0] hbuf_g [0:INTER_DENSE-1];
    reg        gate_used [0:N_EXPERT-1];
    real       top_w  [0:TOPK-1];
    real       facc_g [0:MODEL_DIM-1];
    integer    g_argmax; real g_best;
    integer    si, sj;
    integer    errors, test_count, fails_in_test;

    // ---- stimulus build ----
    integer sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc=seed0;
        for (i=0;i<MODEL_DIM;i=i+1) begin GA[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin GB[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin GF[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<CK;j=j+1) begin Wp[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin G1[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin G2[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<N_EXPERT;j=j+1) begin Wg[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Dg[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Du[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_DENSE;j=j+1) begin Dd[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (e=0;e<N_EXPERT;e=e+1) begin
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mg[e][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mu[e][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin Md[e][i][j]=gen_bf16(sc,band); sc=sc+1; end
        end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHg[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHu[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin SHd[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Wlm[i][j]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // input vectors (bf16) -- built into h_t/emb_t1 buses
    reg [15:0] in_h [0:MODEL_DIM-1];
    reg [15:0] in_e [0:MODEL_DIM-1];
    task build_inputs; input integer seed0; input integer band; begin
        for (i=0;i<MODEL_DIM;i=i+1) begin in_h[i]=gen_bf16(seed0+i,band);        end
        for (i=0;i<MODEL_DIM;i=i+1) begin in_e[i]=gen_bf16(seed0+777+i*3,band);  end
    end endtask

    // ---- attention golden over g_nrm1[] -> g_attn[] ----
    task attn_golden; begin
        for (o=0;o<Q_LORA;o=o+1) begin
            accr=0.0;
            for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(g_nrm1[k])*b2real(W_dq[o][k]);
            q_qlora[o]=real2b(accr);
        end
        accr=0.0;
        for (i=0;i<Q_LORA;i=i+1) accr=accr+b2real(q_qlora[i])*b2real(q_qlora[i]);
        accr=1.0/$sqrt(accr/Q_LORA + 1e-5);
        for (i=0;i<Q_LORA;i=i+1) q_qln[i]=real2b(b2real(q_qlora[i])*accr);
        for (o=0;o<HQK;o=o+1) begin
            accr=0.0;
            for (k=0;k<Q_LORA;k=k+1) accr=accr+b2real(q_qln[k])*b2real(W_uq[o][k]);
            q_qfull[o]=real2b(accr);
        end
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
            for (i=0;i<KV_LORA;i=i+1) accr=accr+b2real(CKV[sd][i])*b2real(CKV[sd][i]);
            accr=1.0/$sqrt(accr/KV_LORA + 1e-5);
            for (i=0;i<KV_LORA;i=i+1) q_ckvn[i]=real2b(b2real(CKV[sd][i])*accr);
            for (o=0;o<HNOPE;o=o+1) begin
                accr=0.0;
                for (k=0;k<KV_LORA;k=k+1) accr=accr+b2real(q_ckvn[k])*b2real(W_uk[o][k]);
                q_knope[o]=real2b(accr);
            end
            for (o=0;o<HV;o=o+1) begin
                accr=0.0;
                for (k=0;k<KV_LORA;k=k+1) accr=accr+b2real(q_ckvn[k])*b2real(W_uv[o][k]);
                q_v[o]=real2b(accr);
            end
            for (hd=0;hd<H_HEADS;hd=hd+1) begin
                for (d=0;d<NOPE;d=d+1) gK[hd][sd][d]=b2real(q_knope[hd*NOPE+d]);
                for (d=0;d<ROPE;d=d+1) gK[hd][sd][NOPE+d]=b2real(KRP[sd][d]);
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
        for (o=0;o<MODEL_DIM;o=o+1) begin
            accr=0.0;
            for (k=0;k<HV;k=k+1) accr=accr+gctx[k]*b2real(W_o[o][k]);
            g_attn[o]=b2real(real2b(accr));
        end
    end endtask

    // ---- one swiglu expert over g_nrm2[]; which: -1 dense, 0..N_EXPERT-1 routed,
    //      N_EXPERT shared. ----
    task swiglu_golden; input integer which;
        integer INTER, sii, skk, soo; real ga, up_, hh; begin
        INTER = (which==-1)? INTER_DENSE : INTER_MOE;
        for (sii=0;sii<INTER;sii=sii+1) begin
            ga=0.0; up_=0.0;
            for (skk=0;skk<MODEL_DIM;skk=skk+1) begin
                if (which==-1) begin
                    ga = ga + b2real(g_nrm2[skk])*b2real(Dg[sii][skk]);
                    up_= up_+ b2real(g_nrm2[skk])*b2real(Du[sii][skk]);
                end else if (which==N_EXPERT) begin
                    ga = ga + b2real(g_nrm2[skk])*b2real(SHg[sii][skk]);
                    up_= up_+ b2real(g_nrm2[skk])*b2real(SHu[sii][skk]);
                end else begin
                    ga = ga + b2real(g_nrm2[skk])*b2real(Mg[which][sii][skk]);
                    up_= up_+ b2real(g_nrm2[skk])*b2real(Mu[which][sii][skk]);
                end
            end
            ga = b2real(real2b(ga));
            up_= b2real(real2b(up_));
            hh = b2real(real2b(silu(ga)));
            hbuf_g[sii] = real2b( hh * up_ );
        end
        for (soo=0;soo<MODEL_DIM;soo=soo+1) begin
            accr=0.0;
            for (skk=0;skk<INTER;skk=skk+1) begin
                if (which==-1)            accr=accr+b2real(hbuf_g[skk])*b2real(Dd[soo][skk]);
                else if (which==N_EXPERT) accr=accr+b2real(hbuf_g[skk])*b2real(SHd[soo][skk]);
                else                      accr=accr+b2real(hbuf_g[skk])*b2real(Md[which][soo][skk]);
            end
            g_yexp[soo]=b2real(real2b(accr));
        end
    end endtask

    // ---- the ONE decoder layer over g_x[] -> g_x[] (matches glm_decoder_block) ----
    task block_golden; input integer is_moe; begin
        // pre-attn rmsnorm(g_x, G1)
        rn_acc=0.0;
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1) rn_acc=rn_acc+b2real(g_x[rn_i])*b2real(g_x[rn_i]);
        rn_acc=1.0/$sqrt(rn_acc/MODEL_DIM + 1e-5);
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1)
            g_nrm1[rn_i]=real2b( b2real(g_x[rn_i])*rn_acc*b2real(G1[rn_i]) );
        attn_golden();
        for (i=0;i<MODEL_DIM;i=i+1) g_h[i]=real2b( b2real(g_x[i]) + g_attn[i] );
        // pre-ffn rmsnorm(g_h, G2)
        rn_acc=0.0;
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1) rn_acc=rn_acc+b2real(g_h[rn_i])*b2real(g_h[rn_i]);
        rn_acc=1.0/$sqrt(rn_acc/MODEL_DIM + 1e-5);
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1)
            g_nrm2[rn_i]=real2b( b2real(g_h[rn_i])*rn_acc*b2real(G2[rn_i]) );
        if (!is_moe) begin
            swiglu_golden(-1);
            for (i=0;i<MODEL_DIM;i=i+1) g_ffn[i]=g_yexp[i];
        end else begin
            for (e=0;e<N_EXPERT;e=e+1) begin
                accr=0.0;
                for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(g_nrm2[k])*b2real(Wg[k][e]);
                accr=b2real(real2b(accr));
                g_gate[e]=b2real(real2b(sigm(accr)));
            end
            for (e=0;e<N_EXPERT;e=e+1) gate_used[e]=1'b0;
            for (si=0;si<TOPK;si=si+1) begin
                sj=-1;
                for (j=0;j<N_EXPERT;j=j+1)
                    if (!gate_used[j] && (sj==-1 || g_gate[j]>g_gate[sj])) sj=j;
                top_idx[si]=sj; top_g[si]=g_gate[sj]; gate_used[sj]=1'b1;
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

    // ---- full MTP golden -> g_logit[], g_argmax ----
    task compute_golden; begin
        // a = RMSNorm(in_h, GA)
        rn_acc=0.0;
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1) rn_acc=rn_acc+b2real(in_h[rn_i])*b2real(in_h[rn_i]);
        rn_acc=1.0/$sqrt(rn_acc/MODEL_DIM + 1e-5);
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1)
            g_a[rn_i]=real2b( b2real(in_h[rn_i])*rn_acc*b2real(GA[rn_i]) );
        // b = RMSNorm(in_e, GB)
        rn_acc=0.0;
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1) rn_acc=rn_acc+b2real(in_e[rn_i])*b2real(in_e[rn_i]);
        rn_acc=1.0/$sqrt(rn_acc/MODEL_DIM + 1e-5);
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1)
            g_b[rn_i]=real2b( b2real(in_e[rn_i])*rn_acc*b2real(GB[rn_i]) );
        // concat [a;b]
        for (i=0;i<MODEL_DIM;i=i+1) g_cat[i]            = g_a[i];
        for (i=0;i<MODEL_DIM;i=i+1) g_cat[MODEL_DIM+i]  = g_b[i];
        // h' = W_proj @ cat
        for (o=0;o<MODEL_DIM;o=o+1) begin
            accr=0.0;
            for (k=0;k<CK;k=k+1) accr=accr+b2real(g_cat[k])*b2real(Wp[o][k]);
            g_hp[o]=real2b(accr);
        end
        // run the ONE decoder block on h'
        for (i=0;i<MODEL_DIM;i=i+1) g_x[i]=g_hp[i];
        block_golden(tmode);
        // final rmsnorm(g_x, GF)
        rn_acc=0.0;
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1) rn_acc=rn_acc+b2real(g_x[rn_i])*b2real(g_x[rn_i]);
        rn_acc=1.0/$sqrt(rn_acc/MODEL_DIM + 1e-5);
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1)
            g_xn[rn_i]=real2b( b2real(g_x[rn_i])*rn_acc*b2real(GF[rn_i]) );
        // LM head
        for (o=0;o<VOCAB;o=o+1) begin
            accr=0.0;
            for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(g_xn[k])*b2real(Wlm[o][k]);
            g_logit[o]=b2real(real2b(accr));
        end
        g_argmax=0; g_best=g_logit[0];
        for (o=1;o<VOCAB;o=o+1) if (g_logit[o]>g_best) begin g_best=g_logit[o]; g_argmax=o; end
    end endtask

    // ---- cycle counter / latency ----
    integer cyc_cnt; integer lat_meas;
    always @(posedge clk) if (!rst) cyc_cnt = cyc_cnt + 1;

    // ---- drive one MTP query ----
    task run_dut; begin
        for (i=0;i<MODEL_DIM;i=i+1) begin
            h_t   [16*i +: 16] = in_h[i];
            emb_t1[16*i +: 16] = in_e[i];
        end
        mode  = tmode[0];
        pos   = tpos[POSW-1:0];
        s_len = Sg[IDXW:0];
        @(negedge clk);
        lat_meas = cyc_cnt;
        start=1'b1; @(negedge clk); start=1'b0;
        wait (done==1'b1);
        lat_meas = cyc_cnt - lat_meas;
        $display("LATENCY[mode=%0d pos=%0d S=%0d] = %0d cycles", tmode, tpos, Sg, lat_meas);
        @(negedge clk);
    end endtask

    // ---- check ----
    task check_out; input [256*8-1:0] label; begin
        fails_in_test=0; worst_rel=0.0;
        for (i=0;i<VOCAB*16;i=i+1)
            if (logits[i]===1'bx || logits[i]===1'bz) begin
                $display("FAIL[%0s]: logits bit %0d X/Z", label, i);
                fails_in_test=fails_in_test+1;
            end
        for (o=0;o<VOCAB;o=o+1) begin
            dutv=b2real(logits[16*o+:16]);
            refv=g_logit[o];
            adiff=dutv-refv; if (adiff<0.0) adiff=-adiff;
            this_rel = adiff / ((refv<0.0?-refv:refv) > TINY ? (refv<0.0?-refv:refv) : TINY);
            if (this_rel>worst_rel) worst_rel=this_rel;
            if (this_rel>REL_TOL) begin
                $display("FAIL[%0s]: v=%0d dut=%g ref=%g relerr=%g (>%g)",
                         label, o, dutv, refv, this_rel, REL_TOL);
                fails_in_test=fails_in_test+1;
            end
        end
        if (argmax !== g_argmax[TOKW-1:0]) begin
            $display("FAIL[%0s]: argmax dut=%0d ref=%0d", label, argmax, g_argmax);
            fails_in_test=fails_in_test+1;
        end
        test_count=test_count+1;
        if (fails_in_test!=0) errors=errors+fails_in_test;
        else $display("PASS[%0s] worst_rel=%g argmax=%0d", label, worst_rel, argmax);
    end endtask

    // ---- watchdog ----
    initial begin
        #50000000;
        $display("FAIL: global timeout"); $fatal;
    end

    // ---- main ----
    initial begin
        // tolerance: the combine norms + projection (2*MODEL_DIM reduce) + the
        // proven decoder_block per-layer budget + final norm + LM-head reduction.
        REL_TOL = (1.0/16.0)
                + ( CK + (MODEL_DIM + Q_LORA + HQK + KV_LORA*4 + HV
                          + INTER_DENSE + MODEL_DIM)
                    + MODEL_DIM + MODEL_DIM ) * (1.0/4096.0);
        TINY    = 1.0e-3;
        errors=0; test_count=0; worst_rel=0.0; cyc_cnt=0;

        rst=1'b1; start=1'b0; mode=1'b0; pos={POSW{1'b0}}; s_len={(IDXW+1){1'b0}};
        h_t={MODEL_DIM*16{1'b0}}; emb_t1={MODEL_DIM*16{1'b0}};
        repeat(4) @(negedge clk);
        rst=1'b0;
        @(negedge clk);

        // === DENSE-mode MTP tests ===
        tmode=0; tpos=0;   Sg=1;     band_run(500,  0, "dense pos0 S1");
        tmode=0; tpos=5;   Sg=2;     band_run(1000, 0, "dense pos5 S2");
        tmode=0; tpos=37;  Sg=3;     band_run(7000, 0, "dense pos37 S3");
        // === MoE-mode MTP tests ===
        tmode=1; tpos=42;  Sg=5;     band_run(20000,0, "moe pos42 S5");
        tmode=1; tpos=129; Sg=S_MAX; band_run(90000,1, "moe pos129 Smax b1");

        if (errors!=0) begin
            $display("FAILED: %0d element error(s) across %0d tests", errors, test_count);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED", test_count);
        $finish;
    end

    // helper: build stimulus+inputs, golden, run, check
    task band_run; input integer seed; input integer band; input [256*8-1:0] label; begin
        tband=band;
        build_stimulus(seed, band);
        build_inputs(seed+333333, band);
        compute_golden();
        run_dut();
        check_out(label);
    end endtask

endmodule
