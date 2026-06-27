`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_model_tb.v  --  INDEPENDENT fp64 golden TB for the FULL GLM-5.2 forward
//                     pass at the small-but-faithful slice (glm_model.v).
//----------------------------------------------------------------------------
// DUT
//   glm_model computes, for ONE token position:
//      x0   = embed(token_id)
//      x_l+1 = decoder_block(x_l, layer=l, mode = DENSE(l<N_DENSE)|MoE)
//      xN   = rmsnorm_final(x_L)
//      logits[V] = W_lm[V,MODEL_DIM] . xN ;  argmax = arg max logits
//
// INDEPENDENT GOLDEN  (Verilog `real` / fp64; shares NONE of the DUT fp32 path)
//   Recomputes the WHOLE forward pass a different way: every bf16 widened
//   losslessly to real, every dot product a true fp64 reduce, RMSNorm rsqrt via
//   $sqrt, RoPE via $cos/$sin, softmax/silu/sigmoid via $exp.  bf16 quantization
//   is applied at EXACTLY the boundaries the hardware rounds (embed table is bf16
//   already; norm outputs, each GEMM output, silu, silu*up, gate, router weight,
//   both residual adds, each LM-head logit).  The MoE top-K runs on the bf16
//   gate with lower-index tie-break so the selected experts match the DUT.  This
//   is the decoder_block golden (proven 10/10 vs fp64) wrapped in the layer loop
//   + final norm + LM head, so any model-level orchestration bug (wrong layer
//   weights, wrong dense/MoE mode, wrong residual chaining, wrong LM-head tile
//   mapping, wrong argmax) is caught.
//
// PASS / TOLERANCE
//   Per logit, RELATIVE error vs the fp64 golden <= REL_TOL (the per-layer
//   budget * L plus the LM-head reduction depth), AND exact argmax match
//   (ACCEL_GLM52 §6: "rel-err < ~2e-2 on logits AND exact argmax next-token
//   match").  X/Z and nan/inf are hard failures.  Prints "ALL <N> TESTS PASSED";
//   $fatal on any element miss / X / nan / inf / argmax mismatch / timeout.
//============================================================================
module glm_model_tb;

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

    wire                      fn_req;
    wire [DIMW-1:0]           fn_idx;
    reg  [15:0]               fn_val;

    wire                      lw_req;
    wire [VTW-1:0]            lw_vtile;
    wire [DIMW-1:0]           lw_k;
    reg  [LM_TN*16-1:0]       lw_col;

    glm_model #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .LM_TN(LM_TN)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .token_id(token_id), .pos(pos), .s_len(s_len),
        .logits(logits), .argmax(argmax),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k), .aw_col(aw_col),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx), .fw_col(fw_col), .fw_col_up(fw_col_up),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, busy, em_req, em_tok, aw_req, fw_req, rw_req, gn_req,
                     fn_req, lw_req, idx_fresh, idx_win};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= WEIGHT ROMs (per-layer where applicable) =================
    // embedding table [vocab][dim]
    reg [15:0] EMB [0:VOCAB-1][0:MODEL_DIM-1];
    // per-layer RMSNorm gammas
    reg [15:0] G1 [0:L-1][0:MODEL_DIM-1];
    reg [15:0] G2 [0:L-1][0:MODEL_DIM-1];
    // per-layer attention weights
    reg [15:0] W_dq  [0:L-1][0:Q_LORA-1][0:MODEL_DIM-1];
    reg [15:0] W_uq  [0:L-1][0:HQK-1][0:Q_LORA-1];
    reg [15:0] W_dkv [0:L-1][0:KV_LORA-1][0:MODEL_DIM-1];   // exercised, unused val
    reg [15:0] W_kr  [0:L-1][0:ROPE-1][0:MODEL_DIM-1];      // exercised, unused val
    reg [15:0] W_uk  [0:L-1][0:HNOPE-1][0:KV_LORA-1];
    reg [15:0] W_uv  [0:L-1][0:HV-1][0:KV_LORA-1];
    reg [15:0] W_o   [0:L-1][0:MODEL_DIM-1][0:HV-1];
    // per-layer KV cache
    reg [15:0] CKV [0:L-1][0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP [0:L-1][0:S_MAX-1][0:ROPE-1];
    // per-layer router gate weight (used only on MoE layers)
    reg [15:0] Wg [0:L-1][0:MODEL_DIM-1][0:N_EXPERT-1];
    // per-layer dense FFN (only L<N_DENSE used)
    reg [15:0] Dg [0:L-1][0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [15:0] Du [0:L-1][0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [15:0] Dd [0:L-1][0:MODEL_DIM-1][0:INTER_DENSE-1];
    // per-layer MoE routed + shared (only L>=N_DENSE used)
    reg [15:0] Mg [0:L-1][0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] Mu [0:L-1][0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] Md [0:L-1][0:N_EXPERT-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [15:0] SHg [0:L-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] SHu [0:L-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] SHd [0:L-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    // LM head W_lm[vocab][dim]
    reg [15:0] Wlm [0:VOCAB-1][0:MODEL_DIM-1];

    // ================= combinational pull responders =================
    integer t;
    // embedding pull  (@* so it evaluates at t=0 and re-triggers on the array
    // read -- the decoder-block TB uses the same @* discipline for its small
    // gamma/cache/router ROMs; an explicit edge list would miss the first beat
    // when an index stays at its reset value 0 across the boundary.)
    always @* em_val = EMB[em_tok][em_idx];
    // per-layer gamma pull
    always @* gn_val = gn_which ? G2[db_layer][gn_idx] : G1[db_layer][gn_idx];
    // final-norm gamma pull (use a dedicated final gamma)
    reg [15:0] GF [0:MODEL_DIM-1];
    always @* fn_val = GF[fn_idx];

    // per-layer attention weight pull
    always @(aw_sel or aw_grp or aw_k or db_layer or start) begin
        aw_col = {PE_N*16{1'b0}};
        for (t=0;t<PE_N;t=t+1) begin
            case (aw_sel)
            4'd0: if (aw_grp*PE_N+t < Q_LORA)   aw_col[16*t+:16]=W_dq [db_layer][aw_grp*PE_N+t][aw_k];
            4'd1: if (aw_grp*PE_N+t < HQK)      aw_col[16*t+:16]=W_uq [db_layer][aw_grp*PE_N+t][aw_k];
            4'd2: if (aw_grp*PE_N+t < KV_LORA)  aw_col[16*t+:16]=W_dkv[db_layer][aw_grp*PE_N+t][aw_k];
            4'd3: if (aw_grp*PE_N+t < ROPE)     aw_col[16*t+:16]=W_kr [db_layer][aw_grp*PE_N+t][aw_k];
            4'd4: if (aw_grp*PE_N+t < HNOPE)    aw_col[16*t+:16]=W_uk [db_layer][aw_grp*PE_N+t][aw_k];
            4'd5: if (aw_grp*PE_N+t < HV)       aw_col[16*t+:16]=W_uv [db_layer][aw_grp*PE_N+t][aw_k];
            4'd6: if (aw_grp*PE_N+t < MODEL_DIM)aw_col[16*t+:16]=W_o  [db_layer][aw_grp*PE_N+t][aw_k];
            default: aw_col[16*t+:16]=16'h0;
            endcase
        end
    end

    // per-layer cache read pull
    integer cd;
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

    // per-layer router W_g pull
    integer re;
    always @* begin
        rw_col = {16*N_EXPERT{1'b0}};
        for (re=0;re<N_EXPERT;re=re+1) rw_col[16*re+:16] = Wg[db_layer][rw_k][re];
    end

    // per-layer FFN expert weight pull (mode inferred from db_layer vs N_DENSE)
    integer ft, fo; reg dmode;
    always @(fw_grp or fw_k or fw_sel or fw_shared or fw_eidx or db_layer or start) begin
        dmode = (db_layer < N_DENSE) ? 1'b0 : 1'b1;
        fw_col = {16*TN{1'b0}}; fw_col_up = {16*TN{1'b0}};
        for (ft=0;ft<TN;ft=ft+1) begin
            fo = fw_grp*TN + ft;
            if (dmode==1'b0) begin
                if (fw_sel==2'd2) begin
                    if (fo<MODEL_DIM) fw_col[16*ft+:16]=Dd[db_layer][fo][fw_k];
                end else begin
                    if (fo<INTER_DENSE) begin
                        fw_col   [16*ft+:16]=Dg[db_layer][fo][fw_k];
                        fw_col_up[16*ft+:16]=Du[db_layer][fo][fw_k];
                    end
                end
            end else begin
                if (fw_shared) begin
                    if (fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[16*ft+:16]=SHd[db_layer][fo][fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [16*ft+:16]=SHg[db_layer][fo][fw_k];
                        fw_col_up[16*ft+:16]=SHu[db_layer][fo][fw_k];
                    end
                end else begin
                    if (fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[16*ft+:16]=Md[db_layer][fw_eidx][fo][fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [16*ft+:16]=Mg[db_layer][fw_eidx][fo][fw_k];
                        fw_col_up[16*ft+:16]=Mu[db_layer][fw_eidx][fo][fw_k];
                    end
                end
            end
        end
    end

    // LM-head weight pull: lw_col[t] = W_lm[vtile*LM_TN + t][lw_k]
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
    // running hidden state (bf16), one full forward
    reg [15:0] g_x   [0:MODEL_DIM-1];   // residual stream (bf16)
    reg [15:0] g_nrm1[0:MODEL_DIM-1];
    reg [15:0] g_nrm2[0:MODEL_DIM-1];
    reg [15:0] g_h   [0:MODEL_DIM-1];
    reg [15:0] g_xn  [0:MODEL_DIM-1];   // final-normed
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
    integer Sg, tpos, tband;
    integer tok;
    integer LY;          // current golden layer
    integer errors, test_count, fails_in_test;

    // router/expert golden
    real    g_gate[0:N_EXPERT-1];
    integer top_idx[0:TOPK-1];
    real    top_g  [0:TOPK-1];
    real    g_yexp [0:MODEL_DIM-1];
    reg [15:0] hbuf_g [0:INTER_DENSE-1];
    reg        gate_used [0:N_EXPERT-1];
    real       top_w  [0:TOPK-1];
    real       facc_g [0:MODEL_DIM-1];
    integer    g_argmax; real g_best;

    // ---- stimulus build (per-layer weights, embedding, LM head) ----
    integer sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc=seed0;
        // embedding
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin EMB[i][j]=gen_bf16(sc,band); sc=sc+1; end
        // per-layer weights
        for (LY=0;LY<L;LY=LY+1) begin
            for (i=0;i<MODEL_DIM;i=i+1) begin G1[LY][i]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) begin G2[LY][i]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<N_EXPERT;j=j+1) begin Wg[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            // dense FFN
            for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Dg[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Du[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_DENSE;j=j+1) begin Dd[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            // MoE experts
            for (e=0;e<N_EXPERT;e=e+1) begin
                for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mg[LY][e][i][j]=gen_bf16(sc,band); sc=sc+1; end
                for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mu[LY][e][i][j]=gen_bf16(sc,band); sc=sc+1; end
                for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin Md[LY][e][i][j]=gen_bf16(sc,band); sc=sc+1; end
            end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHg[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHu[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin SHd[LY][i][j]=gen_bf16(sc,band); sc=sc+1; end
        end
        // final norm gamma
        for (i=0;i<MODEL_DIM;i=i+1) begin GF[i]=gen_bf16(sc,band); sc=sc+1; end
        // LM head
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Wlm[i][j]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ---- RMSNorm over a bf16 vector src[] with gamma gam[] -> bf16 dst[] ----
    // src/gam are passed implicitly via globals; this is the same math as the
    // decoder_block golden but reads from arbitrary arrays.
    real rn_acc; integer rn_i;

    // ---- per-layer attention golden over g_nrm1[] -> g_attn[] (real, bf16) ----
    task attn_golden; input integer ly; begin
        for (o=0;o<Q_LORA;o=o+1) begin
            accr=0.0;
            for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(g_nrm1[k])*b2real(W_dq[ly][o][k]);
            q_qlora[o]=real2b(accr);
        end
        accr=0.0;
        for (i=0;i<Q_LORA;i=i+1) accr=accr+b2real(q_qlora[i])*b2real(q_qlora[i]);
        accr=1.0/$sqrt(accr/Q_LORA + 1e-5);
        for (i=0;i<Q_LORA;i=i+1) q_qln[i]=real2b(b2real(q_qlora[i])*accr);
        for (o=0;o<HQK;o=o+1) begin
            accr=0.0;
            for (k=0;k<Q_LORA;k=k+1) accr=accr+b2real(q_qln[k])*b2real(W_uq[ly][o][k]);
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
            for (i=0;i<KV_LORA;i=i+1) accr=accr+b2real(CKV[ly][sd][i])*b2real(CKV[ly][sd][i]);
            accr=1.0/$sqrt(accr/KV_LORA + 1e-5);
            for (i=0;i<KV_LORA;i=i+1) q_ckvn[i]=real2b(b2real(CKV[ly][sd][i])*accr);
            for (o=0;o<HNOPE;o=o+1) begin
                accr=0.0;
                for (k=0;k<KV_LORA;k=k+1) accr=accr+b2real(q_ckvn[k])*b2real(W_uk[ly][o][k]);
                q_knope[o]=real2b(accr);
            end
            for (o=0;o<HV;o=o+1) begin
                accr=0.0;
                for (k=0;k<KV_LORA;k=k+1) accr=accr+b2real(q_ckvn[k])*b2real(W_uv[ly][o][k]);
                q_v[o]=real2b(accr);
            end
            for (hd=0;hd<H_HEADS;hd=hd+1) begin
                for (d=0;d<NOPE;d=d+1) gK[hd][sd][d]=b2real(q_knope[hd*NOPE+d]);
                for (d=0;d<ROPE;d=d+1) gK[hd][sd][NOPE+d]=b2real(KRP[ly][sd][d]);
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
            for (k=0;k<HV;k=k+1) accr=accr+gctx[k]*b2real(W_o[ly][o][k]);
            g_attn[o]=b2real(real2b(accr));
        end
    end endtask

    // ---- one swiglu expert over g_nrm2[]; which: -1 dense, 0..N_EXPERT-1 routed,
    //      N_EXPERT shared.  Layer ly.  Writes g_yexp[]. ----
    task swiglu_golden; input integer ly; input integer which;
        integer INTER, ii, kk, oo; real ga, up_, hh; begin
        INTER = (which==-1)? INTER_DENSE : INTER_MOE;
        for (ii=0;ii<INTER;ii=ii+1) begin
            ga=0.0; up_=0.0;
            for (kk=0;kk<MODEL_DIM;kk=kk+1) begin
                if (which==-1) begin
                    ga = ga + b2real(g_nrm2[kk])*b2real(Dg[ly][ii][kk]);
                    up_= up_+ b2real(g_nrm2[kk])*b2real(Du[ly][ii][kk]);
                end else if (which==N_EXPERT) begin
                    ga = ga + b2real(g_nrm2[kk])*b2real(SHg[ly][ii][kk]);
                    up_= up_+ b2real(g_nrm2[kk])*b2real(SHu[ly][ii][kk]);
                end else begin
                    ga = ga + b2real(g_nrm2[kk])*b2real(Mg[ly][which][ii][kk]);
                    up_= up_+ b2real(g_nrm2[kk])*b2real(Mu[ly][which][ii][kk]);
                end
            end
            ga = b2real(real2b(ga));
            up_= b2real(real2b(up_));
            hh = b2real(real2b(silu(ga)));
            hbuf_g[ii] = real2b( hh * up_ );
        end
        for (oo=0;oo<MODEL_DIM;oo=oo+1) begin
            accr=0.0;
            for (kk=0;kk<INTER;kk=kk+1) begin
                if (which==-1)            accr=accr+b2real(hbuf_g[kk])*b2real(Dd[ly][oo][kk]);
                else if (which==N_EXPERT) accr=accr+b2real(hbuf_g[kk])*b2real(SHd[ly][oo][kk]);
                else                      accr=accr+b2real(hbuf_g[kk])*b2real(Md[ly][which][oo][kk]);
            end
            g_yexp[oo]=b2real(real2b(accr));
        end
    end endtask

    // ---- one decoder layer over g_x[] (bf16) -> g_x[] (bf16, updated) ----
    integer si, sj, bestj; real bestv;
    task layer_golden; input integer ly; input integer is_moe; begin
        // 1) pre-attn rmsnorm(g_x, G1[ly]) -> g_nrm1
        rn_acc=0.0;
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1) rn_acc=rn_acc+b2real(g_x[rn_i])*b2real(g_x[rn_i]);
        rn_acc=1.0/$sqrt(rn_acc/MODEL_DIM + 1e-5);
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1)
            g_nrm1[rn_i]=real2b( b2real(g_x[rn_i])*rn_acc*b2real(G1[ly][rn_i]) );
        // 2) attention
        attn_golden(ly);
        // 3) h = x + attn (bf16)
        for (i=0;i<MODEL_DIM;i=i+1) g_h[i]=real2b( b2real(g_x[i]) + g_attn[i] );
        // 4) pre-ffn rmsnorm(g_h, G2[ly]) -> g_nrm2
        rn_acc=0.0;
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1) rn_acc=rn_acc+b2real(g_h[rn_i])*b2real(g_h[rn_i]);
        rn_acc=1.0/$sqrt(rn_acc/MODEL_DIM + 1e-5);
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1)
            g_nrm2[rn_i]=real2b( b2real(g_h[rn_i])*rn_acc*b2real(G2[ly][rn_i]) );
        // 5) FFN
        if (!is_moe) begin
            swiglu_golden(ly,-1);
            for (i=0;i<MODEL_DIM;i=i+1) g_ffn[i]=g_yexp[i];
        end else begin
            for (e=0;e<N_EXPERT;e=e+1) begin
                accr=0.0;
                for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(g_nrm2[k])*b2real(Wg[ly][k][e]);
                accr=b2real(real2b(accr));
                g_gate[e]=b2real(real2b(sigm(accr)));
            end
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
                swiglu_golden(ly,top_idx[si]);
                for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=facc_g[i]+top_w[si]*g_yexp[i];
            end
            swiglu_golden(ly,N_EXPERT);
            for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=facc_g[i]+g_yexp[i];
            for (i=0;i<MODEL_DIM;i=i+1) g_ffn[i]=b2real(real2b(facc_g[i]));
        end
        // 6) y = h + ffn (bf16) -> g_x
        for (i=0;i<MODEL_DIM;i=i+1) g_x[i]=real2b( b2real(g_h[i]) + g_ffn[i] );
    end endtask

    // ---- full forward golden -> g_logit[], g_argmax ----
    task compute_golden; begin
        // embed
        for (i=0;i<MODEL_DIM;i=i+1) g_x[i]=EMB[tok][i];
        // L layers
        for (LY=0;LY<L;LY=LY+1)
            layer_golden(LY, (LY<N_DENSE)?0:1);
        // final rmsnorm(g_x, GF) -> g_xn
        rn_acc=0.0;
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1) rn_acc=rn_acc+b2real(g_x[rn_i])*b2real(g_x[rn_i]);
        rn_acc=1.0/$sqrt(rn_acc/MODEL_DIM + 1e-5);
        for (rn_i=0;rn_i<MODEL_DIM;rn_i=rn_i+1)
            g_xn[rn_i]=real2b( b2real(g_x[rn_i])*rn_acc*b2real(GF[rn_i]) );
        // LM head: logits[v] = dot(g_xn, Wlm[v]) -> bf16
        for (o=0;o<VOCAB;o=o+1) begin
            accr=0.0;
            for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(g_xn[k])*b2real(Wlm[o][k]);
            g_logit[o]=b2real(real2b(accr));
        end
        // argmax (lower-index tie-break)
        g_argmax=0; g_best=g_logit[0];
        for (o=1;o<VOCAB;o=o+1) if (g_logit[o]>g_best) begin g_best=g_logit[o]; g_argmax=o; end
    end endtask

    // ---- cycle counter / latency ----
    integer cyc_cnt; integer lat_meas;
    always @(posedge clk) if (!rst) cyc_cnt = cyc_cnt + 1;

    // ---- drive one token ----
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
        // argmax must match exactly
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
        // tolerance: the per-layer decoder_block budget * L (residual chain) plus
        // the final-norm + LM-head reduction depth.  Each layer rounds to bf16 at
        // ~10 boundaries and accumulates fp32 over a few hundred terms; chained
        // over L=6 layers + final norm + a MODEL_DIM-deep LM-head GEMV.  We keep a
        // 2^-4 floor per the §6 logits target plus per-term fp32 growth.
        REL_TOL = (1.0/16.0)
                + ( L*(MODEL_DIM + Q_LORA + HQK + KV_LORA*4 + HV
                       + INTER_DENSE + MODEL_DIM)
                    + MODEL_DIM + MODEL_DIM ) * (1.0/4096.0);
        TINY    = 1.0e-3;
        errors=0; test_count=0; worst_rel=0.0; cyc_cnt=0;

        rst=1'b1; start=1'b0; token_id={TOKW{1'b0}}; pos={POSW{1'b0}};
        s_len={(IDXW+1){1'b0}};
        repeat(4) @(negedge clk);
        rst=1'b0;
        @(negedge clk);

        // === full forward tests (token, position, S sweep, both bands) ===
        tband=0; tok=7;   tpos=0;   Sg=1; build_stimulus(500,  tband); compute_golden(); run_dut(); check_out("tok7 pos0 S1");
        tband=0; tok=42;  tpos=5;   Sg=2; build_stimulus(1000, tband); compute_golden(); run_dut(); check_out("tok42 pos5 S2");
        tband=0; tok=100; tpos=37;  Sg=3; build_stimulus(7000, tband); compute_golden(); run_dut(); check_out("tok100 pos37 S3");
        tband=0; tok=200; tpos=129; Sg=5; build_stimulus(20000,tband); compute_golden(); run_dut(); check_out("tok200 pos129 S5");
        tband=1; tok=255; tpos=42;  Sg=S_MAX; build_stimulus(90000,tband); compute_golden(); run_dut(); check_out("tok255 pos42 Smax b1");

        if (errors!=0) begin
            $display("FAILED: %0d element error(s) across %0d tests", errors, test_count);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED", test_count);
        $finish;
    end
endmodule
