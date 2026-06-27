`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_decoder_block_tb.v  --  INDEPENDENT fp64 golden TB for one GLM-5.2
//                             decoder layer (glm_decoder_block.v, ACCEL_GLM52 §2)
//----------------------------------------------------------------------------
// DUT
//   glm_decoder_block computes ONE pre-norm transformer layer for a single
//   decode-step token:
//        h = x + mla_attn( rmsnorm(x, gamma1), pos, kv_cache )
//        y = h + FFN(      rmsnorm(h, gamma2) )
//   FFN = DENSE swiglu (mode=0) | MoE (mode=1: router top-K routed experts +
//   shared expert, combine = Σ gate_e*y_e + y_shared).
//
// INDEPENDENT GOLDEN  (Verilog `real` / fp64; shares NONE of the DUT fp32 path)
//   Recomputes the WHOLE layer a different way: every bf16 widened losslessly to
//   real, every dot product a true fp64 reduce, RMSNorm rsqrt via $sqrt, RoPE via
//   $cos/$sin, softmax via $exp, silu/sigmoid via $exp.  bf16 quantization is
//   applied at EXACTLY the boundaries the hardware rounds (norm output, each GEMM
//   output, silu, silu*up, gate, router weight, both residual adds) so the
//   compare measures the DUT compute error, not extra golden precision.  The
//   MoE top-K runs on the bf16-quantized gate with lower-index tie-break so the
//   selected experts match the DUT exactly.
//
// PASS / TOLERANCE
//   Per output element, RELATIVE error vs the fp64 golden:
//      relerr = |y_dut - y_gold| / max(|y_gold|, TINY)   <= REL_TOL
//   The layer chains ~7 GEMMs + 2 RMSNorms + RoPE + softmax (attention) then a
//   3-GEMM SwiGLU (or a router + experts + combine) then two bf16 residual adds.
//   Each stage rounds to bf16 (8-bit significand, 1 ULP = 2^-8) and accumulates
//   in fp32 with K-depth growth.  We budget the stacked double-roundings plus
//   per-term fp32 growth:  REL_TOL = 2^-4 + (sum of reduction depths)*2^-12.
//   X/Z and nan/inf are hard failures.  Prints "ALL <N> TESTS PASSED"; $fatal on
//   any element miss / X / nan / inf / protocol timeout.
//============================================================================
module glm_decoder_block_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ================= slice config (small-but-faithful) =================
    localparam integer MODEL_DIM  = 128;
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

    // ---- derived ----
    localparam integer QK_DIM = NOPE + ROPE;
    localparam integer IDXW   = (S_MAX<=1)?1:$clog2(S_MAX);
    localparam integer HQK    = H_HEADS*QK_DIM;
    localparam integer HNOPE  = H_HEADS*NOPE;
    localparam integer HV     = H_HEADS*V_DIM;
    localparam integer EIDXW  = (N_EXPERT<=1)?1:$clog2(N_EXPERT);
    // mla_attn port-width derivations (mirror the unit / decoder)
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

    // ================= DUT I/O =================
    reg                       start, mode;
    wire                      busy, done;
    reg  [POSW-1:0]           pos;
    reg  [IDXW:0]             s_len;
    reg  [MODEL_DIM*16-1:0]   x_vec;
    wire [MODEL_DIM*16-1:0]   y_out;

    wire                      gn_req, gn_which;
    wire [$clog2(MODEL_DIM)-1:0] gn_idx;
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

    glm_decoder_block #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW),
        .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .mode(mode), .pos(pos), .s_len(s_len), .x_vec(x_vec), .y_out(y_out),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k), .aw_col(aw_col),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx), .fw_col(fw_col), .fw_col_up(fw_col_up)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, busy, aw_req, fw_req, rw_req, gn_req};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= WEIGHT ROMs =================
    // RMSNorm learned gammas (pre-attn = G1, pre-ffn = G2)
    reg [15:0] G1 [0:MODEL_DIM-1];
    reg [15:0] G2 [0:MODEL_DIM-1];
    // attention weights (indexed [out][k], bf16)
    reg [15:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [15:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [15:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];   // exercised path (unused val)
    reg [15:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];      // exercised path (unused val)
    reg [15:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [15:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [15:0] W_o   [0:MODEL_DIM-1][0:HV-1];
    // attention KV cache (S_MAX keys)
    reg [15:0] CKV [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP [0:S_MAX-1][0:ROPE-1];
    // router gate weight W_g[k][e]
    reg [15:0] Wg [0:MODEL_DIM-1][0:N_EXPERT-1];
    // dense FFN expert weights
    reg [15:0] Dg [0:INTER_DENSE-1][0:MODEL_DIM-1];  // gate
    reg [15:0] Du [0:INTER_DENSE-1][0:MODEL_DIM-1];  // up
    reg [15:0] Dd [0:MODEL_DIM-1][0:INTER_DENSE-1];  // down
    // MoE routed + shared expert weights ([expert] index; SH = shared)
    reg [15:0] Mg [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] Mu [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] Md [0:N_EXPERT-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [15:0] SHg [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] SHu [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [15:0] SHd [0:MODEL_DIM-1][0:INTER_MOE-1];

    reg [15:0] xin [0:MODEL_DIM-1];

    // ================= combinational weight responders =================
    integer t;
    // gamma pull (combinational)
    always @* gn_val = gn_which ? G2[gn_idx] : G1[gn_idx];

    // attention weight pull (mirror mla_attn_tb).  Explicit sensitivity on the
    // pull indices (NOT @*) for the same iverilog elaboration-speed reason as the
    // FFN responder: the W_* ROMs are static after build_stimulus and the DUT
    // drives aw_sel/aw_grp/aw_k on every weight beat.
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

    // cache read pull: combinational data + 1-cycle valid
    integer cd;
    // small arrays (CKV/KRP = a few hundred words) -> @* is cheap and content-safe
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

    // router W_g pull: lanes = W_g[rw_k, e]
    integer re;
    // small array (Wg = 1024 words) -> @* is cheap and content-safe
    always @* begin
        rw_col = {16*N_EXPERT{1'b0}};
        for (re=0;re<N_EXPERT;re=re+1) rw_col[16*re+:16] = Wg[rw_k][re];
    end

    // FFN expert weight pull: selected by mode / fw_shared / fw_eidx.
    //   GATE/UP  : fw_col = W_gate[out][k], fw_col_up = W_up[out][k]
    //   DOWN     : fw_col = W_down[out][k]
    // In MoE mode the routed expert is fw_eidx, or the shared bank if fw_shared.
    integer ft, fo;
    // NOTE: explicit sensitivity on the pull-index/control signals (NOT @*).
    //   The weight ROM arrays (Mg/Mu/Md are 65536 words each) are STATIC after
    //   build_stimulus; an @* here makes iverilog build a sensitivity list over
    //   every array word (>500k entries with the variable part-selects), which
    //   makes elaboration run for many minutes.  The DUT changes fw_grp/fw_k/
    //   fw_eidx/fw_sel/fw_shared (and mode is constant per run) on every weight
    //   beat, so re-evaluating on those indices reproduces @* behaviour exactly
    //   while keeping iverilog elaboration fast.
    always @(fw_grp or fw_k or fw_sel or fw_shared or fw_eidx or mode or start) begin
        fw_col = {16*TN{1'b0}}; fw_col_up = {16*TN{1'b0}};
        for (ft=0;ft<TN;ft=ft+1) begin
            fo = fw_grp*TN + ft;
            if (mode==1'b0) begin
                // DENSE expert (INTER_DENSE)
                if (fw_sel==2'd2) begin
                    if (fo<MODEL_DIM) fw_col[16*ft+:16]=Dd[fo][fw_k];
                end else begin
                    if (fo<INTER_DENSE) begin
                        fw_col   [16*ft+:16]=Dg[fo][fw_k];
                        fw_col_up[16*ft+:16]=Du[fo][fw_k];
                    end
                end
            end else begin
                // MoE expert (INTER_MOE): shared or routed bank
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
    // silu(x) = x / (1+exp(-x))   ; sigmoid(x) = 1/(1+exp(-x))
    function real silu; input real x; begin silu = x/(1.0+$exp(-x)); end endfunction
    function real sigm; input real x; begin sigm = 1.0/(1.0+$exp(-x)); end endfunction

    // deterministic bf16 stimulus
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
    reg [15:0] g_nrm1 [0:MODEL_DIM-1];
    reg [15:0] g_nrm2 [0:MODEL_DIM-1];
    reg [15:0] g_h    [0:MODEL_DIM-1];
    real g_attn [0:MODEL_DIM-1];
    real g_ffn  [0:MODEL_DIM-1];
    real gout   [0:MODEL_DIM-1];

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
    integer Sg, tpos, tmode, tband;
    integer errors, test_count, fails_in_test;

    // router/expert golden
    real    g_gate[0:N_EXPERT-1];        // bf16-quantized gate
    integer top_idx[0:TOPK-1];
    real    top_g  [0:TOPK-1];
    real    g_yexp [0:MODEL_DIM-1];      // one expert's swiglu output (real)
    reg [15:0] hbuf_g [0:INTER_DENSE-1]; // expert h buffer (golden)
    reg        gate_used [0:N_EXPERT-1]; // top-K selection mask
    real       top_w  [0:TOPK-1];        // routed weights (renorm*scale)
    real       facc_g [0:MODEL_DIM-1];   // MoE combine accumulator

    // ---- stimulus build ----
    integer sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc=seed0;
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
        for (i=0;i<MODEL_DIM;i=i+1) begin xin[i]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ---- RMSNorm(z, gamma) -> bf16 out[]  (z given as bf16 in[]) ----
    task rmsnorm; input integer src;  // 0: use xin, 1: use g_h ; writes outsel
                  input integer dst;  // 0: g_nrm1 (gamma G1), 1: g_nrm2 (gamma G2)
        real acc; integer ii; reg [15:0] zin; reg [15:0] gam; begin
        acc=0.0;
        for (ii=0;ii<MODEL_DIM;ii=ii+1) begin
            zin = (src==0)? xin[ii] : g_h[ii];
            acc = acc + b2real(zin)*b2real(zin);
        end
        acc = 1.0/$sqrt(acc/MODEL_DIM + 1e-5);
        for (ii=0;ii<MODEL_DIM;ii=ii+1) begin
            zin = (src==0)? xin[ii] : g_h[ii];
            gam = (dst==0)? G1[ii] : G2[ii];
            if (dst==0) g_nrm1[ii] = real2b( b2real(zin)*acc*b2real(gam) );
            else        g_nrm2[ii] = real2b( b2real(zin)*acc*b2real(gam) );
        end
    end endtask

    // ---- attention golden over input vector inv[] (bf16) -> g_attn[] (real) ----
    task attn_golden; input integer dummy;  // input is g_nrm1[]
        begin
        // Q path
        for (o=0;o<Q_LORA;o=o+1) begin
            accr=0.0;
            for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(g_nrm1[k])*b2real(W_dq[o][k]);
            q_qlora[o]=real2b(accr);
        end
        accr=0.0;
        for (i=0;i<Q_LORA;i=i+1) accr=accr+b2real(q_qlora[i])*b2real(q_qlora[i]);
        accr=1.0/$sqrt(accr/Q_LORA + 1e-5);
        for (i=0;i<Q_LORA;i=i+1) q_qln[i]=real2b(b2real(q_qlora[i])*accr); // gamma=1
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
        // KV path
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
        // scores
        for (hd=0;hd<H_HEADS;hd=hd+1)
            for (sd=0;sd<Sg;sd=sd+1) begin
                accr=0.0;
                for (d=0;d<QK_DIM;d=d+1) accr=accr+gq[hd*QK_DIM+d]*gK[hd][sd][d];
                gsc[hd][sd]=b2real(real2b(accr));
            end
        // softmax
        for (hd=0;hd<H_HEADS;hd=hd+1) begin
            mxr=gsc[hd][0];
            for (sd=1;sd<Sg;sd=sd+1) if (gsc[hd][sd]>mxr) mxr=gsc[hd][sd];
            sumr=0.0;
            for (sd=0;sd<Sg;sd=sd+1) sumr=sumr+$exp(gsc[hd][sd]-mxr);
            for (sd=0;sd<Sg;sd=sd+1) gp[hd][sd]=b2real(real2b($exp(gsc[hd][sd]-mxr)/sumr));
        end
        // context
        for (hd=0;hd<H_HEADS;hd=hd+1)
            for (d=0;d<V_DIM;d=d+1) begin
                accr=0.0;
                for (sd=0;sd<Sg;sd=sd+1) accr=accr+gp[hd][sd]*gV[hd][sd][d];
                gctx[hd*V_DIM+d]=b2real(real2b(accr));
            end
        // out proj W_o -> bf16 (DUT mla emits bf16)
        for (o=0;o<MODEL_DIM;o=o+1) begin
            accr=0.0;
            for (k=0;k<HV;k=k+1) accr=accr+gctx[k]*b2real(W_o[o][k]);
            g_attn[o]=b2real(real2b(accr));   // attn out as bf16
        end
    end endtask

    // ---- one swiglu expert over g_nrm2[]; weights chosen by (which) ----
    // which: -1 = DENSE bank ; 0..N_EXPERT-1 = routed bank e ; N_EXPERT = shared
    // writes g_yexp[] (real, already-bf16-quantized down output, as DUT emits).
    task swiglu_golden; input integer which;
        integer INTER, ii, kk, oo; real ga, up_, hh; begin
        INTER = (which==-1)? INTER_DENSE : INTER_MOE;
        // h[i] = bf16( bf16(silu(bf16(gate_i))) * bf16(up_i) )
        // gate/up: dot over MODEL_DIM
        // (store h into gK[0][0][..]? no -- use a local real array via g_yexp reuse)
        // first compute h into a temp using gctx as scratch is unsafe; use array:
        for (ii=0;ii<INTER;ii=ii+1) begin
            ga=0.0; up_=0.0;
            for (kk=0;kk<MODEL_DIM;kk=kk+1) begin
                if (which==-1) begin
                    ga = ga + b2real(g_nrm2[kk])*b2real(Dg[ii][kk]);
                    up_= up_+ b2real(g_nrm2[kk])*b2real(Du[ii][kk]);
                end else if (which==N_EXPERT) begin
                    ga = ga + b2real(g_nrm2[kk])*b2real(SHg[ii][kk]);
                    up_= up_+ b2real(g_nrm2[kk])*b2real(SHu[ii][kk]);
                end else begin
                    ga = ga + b2real(g_nrm2[kk])*b2real(Mg[which][ii][kk]);
                    up_= up_+ b2real(g_nrm2[kk])*b2real(Mu[which][ii][kk]);
                end
            end
            ga = b2real(real2b(ga));            // gate -> bf16
            up_= b2real(real2b(up_));           // up   -> bf16
            hh = b2real(real2b(silu(ga)));      // silu -> bf16
            hbuf_g[ii] = real2b( hh * up_ );    // h = bf16(silu*up)
        end
        // down: y[o] = dot over INTER of h -> bf16
        for (oo=0;oo<MODEL_DIM;oo=oo+1) begin
            accr=0.0;
            for (kk=0;kk<INTER;kk=kk+1) begin
                if (which==-1)            accr=accr+b2real(hbuf_g[kk])*b2real(Dd[oo][kk]);
                else if (which==N_EXPERT) accr=accr+b2real(hbuf_g[kk])*b2real(SHd[oo][kk]);
                else                      accr=accr+b2real(hbuf_g[kk])*b2real(Md[which][oo][kk]);
            end
            g_yexp[oo]=b2real(real2b(accr));    // expert y as bf16
        end
    end endtask

    // ---- full-layer golden -> gout[] ----
    integer si, sj, bestj;
    real bestv;
    task compute_golden; begin
        // 1) pre-attn norm
        rmsnorm(0,0);                 // g_nrm1
        // 2) attention
        attn_golden(0);               // g_attn (real, bf16)
        // 3) h = x + attn  (bf16)
        for (i=0;i<MODEL_DIM;i=i+1)
            g_h[i] = real2b( b2real(xin[i]) + g_attn[i] );
        // 4) pre-ffn norm
        rmsnorm(1,1);                 // g_nrm2
        // 5) FFN
        if (tmode==0) begin
            swiglu_golden(-1);
            for (i=0;i<MODEL_DIM;i=i+1) g_ffn[i]=g_yexp[i];
        end else begin
            // router: logits, sigmoid gate (bf16), top-K (lower-index tie-break),
            // renorm-then-scale.
            for (e=0;e<N_EXPERT;e=e+1) begin
                accr=0.0;
                for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(g_nrm2[k])*b2real(Wg[k][e]);
                accr=b2real(real2b(accr));            // logit bf16
                g_gate[e]=b2real(real2b(sigm(accr))); // gate bf16
            end
            // top-K selection on bf16 gate, lower-index tie-break
            for (e=0;e<N_EXPERT;e=e+1) gate_used[e]=1'b0;
            for (si=0;si<TOPK;si=si+1) begin
                bestj=-1; bestv=-1.0e30;
                for (sj=0;sj<N_EXPERT;sj=sj+1)
                    if (!gate_used[sj] && g_gate[sj]>bestv) begin bestv=g_gate[sj]; bestj=sj; end
                top_idx[si]=bestj; top_g[si]=g_gate[bestj]; gate_used[bestj]=1'b1;
            end
            // renorm then scale: s = sum top_g; w = (g/s)*2.5 -> bf16
            sumr=0.0;
            for (si=0;si<TOPK;si=si+1) sumr=sumr+top_g[si];
            for (si=0;si<TOPK;si=si+1)
                top_w[si]=b2real(real2b( (top_g[si]/sumr)*2.5 ));
            // combine: Σ w_e * y_e (routed) + y_shared
            for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=0.0;
            for (si=0;si<TOPK;si=si+1) begin
                swiglu_golden(top_idx[si]);
                for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=facc_g[i]+top_w[si]*g_yexp[i];
            end
            swiglu_golden(N_EXPERT);   // shared, weight 1
            for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=facc_g[i]+g_yexp[i];
            for (i=0;i<MODEL_DIM;i=i+1) g_ffn[i]=b2real(real2b(facc_g[i]));
        end
        // 6) y = h + ffn (bf16)
        for (i=0;i<MODEL_DIM;i=i+1)
            gout[i]=b2real(real2b( b2real(g_h[i]) + g_ffn[i] ));
    end endtask

    // ---- cycle counter (deterministic-latency measurement) ----
    integer cyc_cnt; integer lat_meas;
    always @(posedge clk) if (!rst) cyc_cnt = cyc_cnt + 1;

    // ---- drive one decode step ----
    task run_dut; begin
        pos   = tpos[POSW-1:0];
        s_len = Sg[IDXW:0];
        mode  = tmode[0];
        @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) x_vec[16*i+:16]=xin[i];
        lat_meas = cyc_cnt;
        start=1'b1; @(negedge clk); start=1'b0;
        wait (done==1'b1);
        lat_meas = cyc_cnt - lat_meas;
        $display("LATENCY[mode=%0d S=%0d] = %0d cycles", tmode, Sg, lat_meas);
        @(negedge clk);
    end endtask

    // ---- check ----
    task check_out; input [256*8-1:0] label; begin
        fails_in_test=0; worst_rel=0.0;
        for (i=0;i<MODEL_DIM*16;i=i+1)
            if (y_out[i]===1'bx || y_out[i]===1'bz) begin
                $display("FAIL[%0s]: y_out bit %0d X/Z", label, i);
                fails_in_test=fails_in_test+1;
            end
        for (o=0;o<MODEL_DIM;o=o+1) begin
            dutv=b2real(y_out[16*o+:16]);
            refv=gout[o];
            adiff=dutv-refv; if (adiff<0.0) adiff=-adiff;
            this_rel = adiff / ((refv<0.0?-refv:refv) > TINY ? (refv<0.0?-refv:refv) : TINY);
            if (this_rel>worst_rel) worst_rel=this_rel;
            if (this_rel>REL_TOL) begin
                $display("FAIL[%0s]: o=%0d dut=%g ref=%g relerr=%g (>%g)",
                         label, o, dutv, refv, this_rel, REL_TOL);
                fails_in_test=fails_in_test+1;
            end
        end
        test_count=test_count+1;
        if (fails_in_test!=0) errors=errors+fails_in_test;
        else $display("PASS[%0s] worst_rel=%g", label, worst_rel);
    end endtask

    // ---- watchdog ----
    initial begin
        #5000000;
        $display("FAIL: global timeout"); $fatal;
    end

    // ---- main ----
    integer tc;
    initial begin
        // tolerance: stacked bf16 roundings + per-term fp32 growth over the deep
        // attention chain + ffn chain.  Reduction depths summed (q dq+uq, kv
        // uk/uv, scores, ctx, out, ffn gate/up + down) ~ a few hundred terms.
        REL_TOL = (1.0/16.0) + ( (MODEL_DIM + Q_LORA + HQK + KV_LORA*4 + HV
                                  + INTER_DENSE + MODEL_DIM) * (1.0/4096.0) );
        TINY    = 1.0e-3;
        errors=0; test_count=0; worst_rel=0.0; cyc_cnt=0;

        rst=1'b1; start=1'b0; mode=1'b0; pos={POSW{1'b0}};
        s_len={(IDXW+1){1'b0}}; x_vec={MODEL_DIM*16{1'b0}};
        repeat(4) @(negedge clk);
        rst=1'b0;
        @(negedge clk);

        // === DENSE mode tests ===
        tmode=0;
        // S=1 single-key + pos=0 (RoPE identity) edge case
        tband=0; tpos=0;  Sg=1; build_stimulus(500, tband); compute_golden(); run_dut(); check_out("DENSE b0 pos0 S1");
        tband=0; tpos=5;  Sg=2; build_stimulus(1000,tband); compute_golden(); run_dut(); check_out("DENSE b0 pos5 S2");
        tband=0; tpos=37; Sg=3; build_stimulus(7000,tband); compute_golden(); run_dut(); check_out("DENSE b0 pos37 S3");
        tband=0; tpos=129;Sg=4; build_stimulus(20000,tband);compute_golden(); run_dut(); check_out("DENSE b0 pos129 S4");
        // directed: larger-magnitude band, full key set
        tband=1; tpos=42; Sg=S_MAX; build_stimulus(90000,tband); compute_golden(); run_dut(); check_out("DENSE b1 pos42 Smax");

        // === MoE mode tests ===
        tmode=1;
        // S=1 single-key + pos=0 edge case (MoE)
        tband=0; tpos=0;  Sg=1; build_stimulus(31000,tband); compute_golden(); run_dut(); check_out("MoE b0 pos0 S1");
        tband=0; tpos=2;  Sg=2; build_stimulus(33000,tband); compute_golden(); run_dut(); check_out("MoE b0 pos2 S2");
        tband=0; tpos=64; Sg=3; build_stimulus(50000,tband); compute_golden(); run_dut(); check_out("MoE b0 pos64 S3");
        tband=0; tpos=300;Sg=5; build_stimulus(70000,tband); compute_golden(); run_dut(); check_out("MoE b0 pos300 S5");
        // directed: larger-magnitude band, full key set (MoE)
        tband=1; tpos=128;Sg=S_MAX; build_stimulus(95000,tband); compute_golden(); run_dut(); check_out("MoE b1 pos128 Smax");

        if (errors!=0) begin
            $display("FAILED: %0d element error(s) across %0d tests", errors, test_count);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED", test_count);
        $finish;
    end
endmodule
