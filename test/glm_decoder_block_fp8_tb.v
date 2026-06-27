`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_decoder_block_fp8_tb.v -- INDEPENDENT fp64 golden TB for ONE GLM-5.2-FP8
//                               decoder layer (glm_decoder_block_fp8.v, §2,§6)
//----------------------------------------------------------------------------
// DUT
//   glm_decoder_block_fp8 computes one pre-norm transformer layer for a single
//   decode-step token, with the big LINEAR WEIGHT matmuls in OFFICIAL GLM-5.2-FP8
//   numerics (E4M3 weights + [128,128] block scales + dynamic per-token act->E4M3
//   quant), everything else (norms/rope/softmax/silu/residual) bf16:
//        h = x + mla_attn_fp8( rmsnorm(x,g1), pos, kv_cache )
//        y = h + FFN(          rmsnorm(h,g2) )
//   FFN = DENSE swiglu (mode=0) | MoE (mode=1: router top-K routed experts +
//   shared expert, combine = Sum gate_e*y_e + y_shared).
//
// FAITHFUL FP8 GOLDEN  (Verilog `real`/fp64; shares NONE of the DUT fp32 path)
//   Recomputes the WHOLE layer independently.  bf16 widened via b2real/real2b
//   (the TB's own bf16), reductions a true fp64 reduce, RMSNorm via $sqrt, RoPE
//   via $cos/$sin, softmax/silu via $exp.  The attention weight projections
//   (W_dq,W_uq,W_uk,W_uv,W_o -- W_dkv/W_kr are computed-but-DEAD, the cache
//   overwrites their outputs, so the golden ignores them exactly like mla_attn),
//   the dense/MoE swiglu GEMMs (gate/up/down) and the router gate GEMV are each
//   recomputed as BLOCK-SCALED E4M3:  the SAME E4M3 weight codes + bf16 [128,128]
//   block scales the DUT is fed, an on-chip-identical per-vector pow2 activation
//   shift (a_shift = clamp(134-emax)), per-element E4M3 quant of the prescaled
//   activation (RNE + saturate to 448 + subnormals, my OWN real-valued q_e4m3),
//   exact-product fp64 accumulation per K-block, bf16 block-scale, undo 2^-a_shift,
//   bf16 round.  Scores / softmax / context / both residual adds stay bf16.
//
//   The fp8 quantization grid is IDENTICAL on DUT and golden (same E4M3 codes,
//   same block scales, same on-chip a_shift rule), so it cancels: the measured
//   error is the bf16-rounding + fp32-vs-fp64 block-accumulation budget of the
//   bf16 sibling, PLUS headroom for intermediate-bf16 divergences that can shift
//   a per-vector a_shift by one E4M3 step on a boundary element.
//
// PASS / TOLERANCE  -- per output element, RELATIVE error vs the fp64 golden:
//      relerr = |y_dut - y_gold| / max(|y_gold|, TINY)   <= REL_TOL
//   X/Z and nan/inf are hard failures.  Prints "ALL <N> TESTS PASSED"; $fatal on
//   any element miss / X / nan / inf / protocol timeout.  Covers DENSE + MoE,
//   several (x, weights, pos, S) cases, and a DENSE re-arm.
//============================================================================
module glm_decoder_block_fp8_tb;

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
    localparam integer BLK        = 128;           // weight_block_size=[128,128]

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
    // ---- FP8 [128,128]-block scale counts (#K-blocks per weight family) ----
    localparam integer A_NB    = (A_KMAX   +BLK-1)/BLK;  // attention scales (=1)
    localparam integer FF_NB_D = (FF_KMAX_D+BLK-1)/BLK;  // dense FFN scales (=2)
    localparam integer FF_NB_M = (FF_KMAX_M+BLK-1)/BLK;  // MoE  FFN scales (=1)
    localparam integer R_NB    = (FF_KMAX_M+BLK-1)/BLK;  // router scales   (=1)
    localparam integer MAXK    = (FF_KMAX_D>A_KMAX)?FF_KMAX_D:A_KMAX; // 256

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
    reg  [PE_N*8-1:0]         aw_col;        // FP8 E4M3 lanes
    reg  [16*PE_N*A_NB-1:0]   aw_scale;      // bf16 block scales

    wire                      kc_req;
    wire [IDXW-1:0]           kc_idx;
    reg  [KV_LORA*16-1:0]     kc_ckv;
    reg  [ROPE*16-1:0]        kc_krope;
    reg                       kc_valid;

    wire                      rw_req;
    wire [R_KW-1:0]           rw_k;
    reg  [8*N_EXPERT-1:0]     rw_col;        // FP8 E4M3 lanes
    reg  [16*N_EXPERT*R_NB-1:0] rw_scale;    // bf16 block scales

    wire                      fw_req;
    wire [1:0]                fw_sel;
    wire [FF_GWD-1:0]         fw_grp;
    wire [FF_KWD-1:0]         fw_k;
    wire                      fw_shared;
    wire [EIDXW-1:0]          fw_eidx;
    reg  [8*TN-1:0]           fw_col, fw_col_up;          // FP8 E4M3 lanes
    reg  [16*TN*FF_NB_D-1:0]  fw_scale_g, fw_scale_u;     // bf16 block scales

    glm_decoder_block_fp8 #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW),
        .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), .BLK(BLK)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .mode(mode), .pos(pos), .s_len(s_len), .x_vec(x_vec), .y_out(y_out),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_col(aw_col), .aw_scale(aw_scale),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col), .rw_scale(rw_scale),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx), .fw_col(fw_col), .fw_col_up(fw_col_up),
        .fw_scale_g(fw_scale_g), .fw_scale_u(fw_scale_u)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, busy, aw_req, fw_req, rw_req, gn_req};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= WEIGHT ROMs (E4M3 codes + bf16 block scales) =========
    // RMSNorm gammas + cache stay bf16 (modules_to_not_convert).
    reg [15:0] G1 [0:MODEL_DIM-1];
    reg [15:0] G2 [0:MODEL_DIM-1];
    // attention weight CODES (E4M3, [out][k])
    reg [7:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];   // DEAD (cache overwrites)
    reg [7:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];      // DEAD (cache overwrites)
    reg [7:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [7:0] W_o   [0:MODEL_DIM-1][0:HV-1];
    // attention block scales ([128,128] -> single scale per matrix here)
    reg [15:0] ScW_dq, ScW_uq, ScW_dkv, ScW_kr, ScW_uk, ScW_uv, ScW_o;
    // attention KV cache (bf16)
    reg [15:0] CKV [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP [0:S_MAX-1][0:ROPE-1];
    // router gate weight codes W_g[k][e] + scale
    reg [7:0]  Wg [0:MODEL_DIM-1][0:N_EXPERT-1];
    reg [15:0] ScWg;
    // dense FFN expert weight codes
    reg [7:0] Dg [0:INTER_DENSE-1][0:MODEL_DIM-1];  // gate
    reg [7:0] Du [0:INTER_DENSE-1][0:MODEL_DIM-1];  // up
    reg [7:0] Dd [0:MODEL_DIM-1][0:INTER_DENSE-1];  // down
    reg [15:0] ScDg [0:FF_NB_D-1];   // gate: per out-block (2)
    reg [15:0] ScDu [0:FF_NB_D-1];   // up  : per out-block (2)
    reg [15:0] ScDd [0:FF_NB_D-1];   // down: per K-block   (2)
    // MoE routed + shared expert weight codes
    reg [7:0] Mg [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] Mu [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] Md [0:N_EXPERT-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [7:0] SHg [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] SHu [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] SHd [0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [15:0] ScMg [0:N_EXPERT-1], ScMu [0:N_EXPERT-1], ScMd [0:N_EXPERT-1];
    reg [15:0] ScSHg, ScSHu, ScSHd;

    reg [15:0] xin [0:MODEL_DIM-1];

    // ================= combinational weight responders =================
    integer t, re, ft, fo, cd, obd;
    reg [15:0] sc_a;
    // gamma pull (combinational)
    always @* gn_val = gn_which ? G2[gn_idx] : G1[gn_idx];

    // attention weight + scale pull (explicit sensitivity for elab speed).
    always @(aw_sel or aw_grp or aw_k or start) begin
        aw_col   = {PE_N*8{1'b0}};
        aw_scale = {16*PE_N*A_NB{1'b0}};
        for (t=0;t<PE_N;t=t+1) begin
            case (aw_sel)
            4'd0: if (aw_grp*PE_N+t < Q_LORA)   aw_col[8*t+:8]=W_dq [aw_grp*PE_N+t][aw_k];
            4'd1: if (aw_grp*PE_N+t < HQK)      aw_col[8*t+:8]=W_uq [aw_grp*PE_N+t][aw_k];
            4'd2: if (aw_grp*PE_N+t < KV_LORA)  aw_col[8*t+:8]=W_dkv[aw_grp*PE_N+t][aw_k];
            4'd3: if (aw_grp*PE_N+t < ROPE)     aw_col[8*t+:8]=W_kr [aw_grp*PE_N+t][aw_k];
            4'd4: if (aw_grp*PE_N+t < HNOPE)    aw_col[8*t+:8]=W_uk [aw_grp*PE_N+t][aw_k];
            4'd5: if (aw_grp*PE_N+t < HV)       aw_col[8*t+:8]=W_uv [aw_grp*PE_N+t][aw_k];
            4'd6: if (aw_grp*PE_N+t < MODEL_DIM)aw_col[8*t+:8]=W_o  [aw_grp*PE_N+t][aw_k];
            default: aw_col[8*t+:8]=8'h0;
            endcase
        end
        case (aw_sel)
            4'd0: sc_a=ScW_dq; 4'd1: sc_a=ScW_uq; 4'd2: sc_a=ScW_dkv; 4'd3: sc_a=ScW_kr;
            4'd4: sc_a=ScW_uk; 4'd5: sc_a=ScW_uv; 4'd6: sc_a=ScW_o; default: sc_a=16'h3F80;
        endcase
        for (t=0;t<PE_N;t=t+1) aw_scale[16*t+:16]=sc_a;   // A_NB=1 -> bj=0
    end

    // cache read pull (bf16, unchanged)
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

    // router W_g pull: lanes = W_g[rw_k, e] (FP8), scale = ScWg (all experts ob0)
    always @* begin
        rw_col   = {8*N_EXPERT{1'b0}};
        rw_scale = {16*N_EXPERT*R_NB{1'b0}};
        for (re=0;re<N_EXPERT;re=re+1) begin
            rw_col[8*re+:8]    = Wg[rw_k][re];
            rw_scale[16*re+:16]= ScWg;       // R_NB=1 -> bj=0
        end
    end

    // FFN expert weight + scale pull (explicit sensitivity for elab speed).
    always @(fw_grp or fw_k or fw_sel or fw_shared or fw_eidx or mode or start) begin
        fw_col     = {8*TN{1'b0}};      fw_col_up  = {8*TN{1'b0}};
        fw_scale_g = {16*TN*FF_NB_D{1'b0}}; fw_scale_u = {16*TN*FF_NB_D{1'b0}};
        obd = (fw_grp*TN) / BLK;        // out-block for gate/up
        for (ft=0;ft<TN;ft=ft+1) begin
            fo = fw_grp*TN + ft;
            if (mode==1'b0) begin
                // DENSE expert (INTER_DENSE)
                if (fw_sel==2'd2) begin
                    if (fo<MODEL_DIM) fw_col[8*ft+:8]=Dd[fo][fw_k];
                end else begin
                    if (fo<INTER_DENSE) begin
                        fw_col   [8*ft+:8]=Dg[fo][fw_k];
                        fw_col_up[8*ft+:8]=Du[fo][fw_k];
                    end
                end
            end else begin
                // MoE expert (INTER_MOE): shared or routed bank
                if (fw_shared) begin
                    if (fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[8*ft+:8]=SHd[fo][fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [8*ft+:8]=SHg[fo][fw_k];
                        fw_col_up[8*ft+:8]=SHu[fo][fw_k];
                    end
                end else begin
                    if (fw_sel==2'd2) begin
                        if (fo<MODEL_DIM) fw_col[8*ft+:8]=Md[fw_eidx][fo][fw_k];
                    end else if (fo<INTER_MOE) begin
                        fw_col   [8*ft+:8]=Mg[fw_eidx][fo][fw_k];
                        fw_col_up[8*ft+:8]=Mu[fw_eidx][fo][fw_k];
                    end
                end
            end
        end
        // block scales (bf16): bj index 0..FF_NB_D-1, packed [16*(bj*TN+pj)+:16]
        for (ft=0;ft<TN;ft=ft+1) begin
            if (mode==1'b0) begin
                if (fw_sel==2'd2) begin
                    // DENSE down: out-block 0, per K-block scale
                    fw_scale_g[16*(0*TN+ft)+:16]=ScDd[0];
                    fw_scale_g[16*(1*TN+ft)+:16]=ScDd[1];
                end else begin
                    // DENSE gate/up: per out-block, K=128 -> only bj=0 matters
                    fw_scale_g[16*(0*TN+ft)+:16]=ScDg[obd];
                    fw_scale_g[16*(1*TN+ft)+:16]=ScDg[obd];
                    fw_scale_u[16*(0*TN+ft)+:16]=ScDu[obd];
                    fw_scale_u[16*(1*TN+ft)+:16]=ScDu[obd];
                end
            end else begin
                // MoE: FF_NB_M=1 -> the expert slices only the low bj=0 lanes
                if (fw_shared) begin
                    if (fw_sel==2'd2) fw_scale_g[16*(0*TN+ft)+:16]=ScSHd;
                    else begin
                        fw_scale_g[16*(0*TN+ft)+:16]=ScSHg;
                        fw_scale_u[16*(0*TN+ft)+:16]=ScSHu;
                    end
                end else begin
                    if (fw_sel==2'd2) fw_scale_g[16*(0*TN+ft)+:16]=ScMd[fw_eidx];
                    else begin
                        fw_scale_g[16*(0*TN+ft)+:16]=ScMg[fw_eidx];
                        fw_scale_u[16*(0*TN+ft)+:16]=ScMu[fw_eidx];
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
    // decode an E4M3 code -> real (exact; every E4M3 value is representable).
    function real e4m3_dec; input [7:0] c; reg s; reg [3:0] e; reg [2:0] m;
        integer ei; real val; begin
        s=c[7]; e=c[6:3]; m=c[2:0]; ei=e;
        if (e==4'hF && m==3'h7)      val=0.0;                  // NaN slot (weights never)
        else if (e==4'h0 && m==3'h0) val=0.0;                  // signed zero
        else if (e==4'h0)            val=(m*1.0)*(2.0**(-9));  // subnormal m*2^-9
        else                         val=(1.0+(m*1.0)/8.0)*(2.0**(ei-7));
        if (s) val=-val;
        e4m3_dec = val;
    end endfunction
    // round half to even (x >= 0)
    function integer rne; input real x; integer fl; real fr; begin
        fl = $rtoi(x);                 // x>=0 -> truncates toward 0 = floor
        fr = x - fl;
        if (fr < 0.5)      rne = fl;
        else if (fr > 0.5) rne = fl + 1;
        else               rne = ((fl % 2)==0) ? fl : fl+1;
    end endfunction
    // quantize a real to the nearest E4M3 value (RNE + saturate 448 + subnormals),
    // returning the decoded real -- matches fp32_to_fp8e4m3 for bf16-precision inputs.
    function real q_e4m3; input real v;
        integer s, E, mant; real av, quantum, grid, q; begin
        if (v==0.0) q_e4m3=0.0;
        else begin
            s=(v<0.0); av=v; if (s) av=-av;
            // E = floor(log2(av))
            E=0;
            if (av>=1.0) begin while (av >= (2.0**(E+1))) E=E+1; end
            else         begin while (av <  (2.0**E))     E=E-1; end
            if (E < -6) begin
                quantum = 2.0**(-9);
                grid    = av/quantum;
                mant    = rne(grid);            // 0..8 (8 -> smallest normal)
                q       = mant*quantum;
            end else if (E > 8) begin
                q = 448.0;
            end else begin
                quantum = 2.0**(E-3);
                grid    = av/quantum;           // [8,16)
                mant    = rne(grid);            // 8..16
                if (mant >= 16) begin E=E+1; mant=8; end
                if (E > 8)                   q = 448.0;
                else if (E==8 && mant>=15)   q = 448.0;   // f=7 is the NaN slot
                else                         q = mant*(2.0**(E-3));
            end
            if (s) q=-q;
            q_e4m3 = q;
        end
    end endfunction

    // a_shift = clamp(134 - emax) ; emax==0 -> 0   (mirrors dyn_shift)
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

    // deterministic bf16 stimulus (gammas / cache / input residual)
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e=8'd125+h[6:4];
        else         e=8'd124+h[5:4];
        m=h[12:6];
        gen_bf16={s,e,m};
    end endfunction
    // deterministic E4M3 weight code (moderate magnitude, never NaN)
    function [7:0] gen_e4m3; input integer seed; input integer band;
        reg s; reg [3:0] e; reg [2:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e = 4'd7 + {3'b0,h[4]};   // 7..8  -> 1..3.75
        else         e = 4'd6 + {3'b0,h[4]};   // 6..7  -> 0.5..1.875
        m = h[12:10];
        gen_e4m3 = {s,e,m};                    // e<=8 -> never the NaN slot
    end endfunction
    // deterministic bf16 [128,128] block scale (~2^-5..2^-4)
    function [15:0] gen_scale; input integer seed;
        reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*22229);
        e = 8'd122 + {7'b0,h[2]};               // 122..123 -> 2^-5..2^-4
        m = h[10:4];
        gen_scale={1'b0,e,m};
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

    // fp8 GEMM scratch
    reg [15:0] asrc_bf [0:MAXK-1];      // current activation vector (bf16)
    real       gy_real [0:MAXK-1];      // GEMM output (real, post bf16 round)
    reg [15:0] gy_bf   [0:MAXK-1];      // GEMM output bf16 code
    reg [15:0] gate_save[0:INTER_DENSE-1];
    reg [15:0] up_save  [0:INTER_DENSE-1];
    reg [15:0] hbuf_g   [0:INTER_DENSE-1];
    integer    cur_expert;

    real accr, mxr, sumr, ang, c, sN, x0, x1, invf;
    real refv, dutv, adiff, this_rel, worst_rel, gworst, REL_TOL, ABS_TOL, TINY, denom;
    integer i,j,k,h,d,o,hd,sd,pr,e;
    integer Sg, tpos, tmode, tband;
    integer errors, test_count, fails_in_test;

    // router/expert golden
    real    g_gate[0:N_EXPERT-1];
    integer top_idx[0:TOPK-1];
    real    top_g  [0:TOPK-1];
    real    g_yexp [0:MODEL_DIM-1];
    reg        gate_used [0:N_EXPERT-1];
    real       top_w  [0:TOPK-1];
    real       facc_g [0:MODEL_DIM-1];

    // ---- weight code accessors (by selector) ----
    function [7:0] wc; input integer wsel; input integer o; input integer kk; begin
        case (wsel)
            0:  wc=W_dq[o][kk];
            1:  wc=W_uq[o][kk];
            4:  wc=W_uk[o][kk];
            5:  wc=W_uv[o][kk];
            6:  wc=W_o [o][kk];
            7:  wc=Wg[kk][o];                 // router W_g[k][e], o=expert
            8:  wc=Dg[o][kk];
            9:  wc=Du[o][kk];
            10: wc=Dd[o][kk];
            11: wc=Mg[cur_expert][o][kk];
            12: wc=Mu[cur_expert][o][kk];
            13: wc=Md[cur_expert][o][kk];
            14: wc=SHg[o][kk];
            15: wc=SHu[o][kk];
            16: wc=SHd[o][kk];
            default: wc=8'h00;
        endcase
    end endfunction
    function [15:0] ws; input integer wsel; input integer ob; input integer kb; begin
        case (wsel)
            0:  ws=ScW_dq; 1: ws=ScW_uq; 4: ws=ScW_uk; 5: ws=ScW_uv; 6: ws=ScW_o;
            7:  ws=ScWg;
            8:  ws=ScDg[ob]; 9: ws=ScDu[ob]; 10: ws=ScDd[kb];
            11: ws=ScMg[cur_expert]; 12: ws=ScMu[cur_expert]; 13: ws=ScMd[cur_expert];
            14: ws=ScSHg; 15: ws=ScSHu; 16: ws=ScSHd;
            default: ws=16'h3F80;
        endcase
    end endfunction

    // ---- block-scaled E4M3 GEMM (fp64 accumulate) over asrc_bf[0..K-1] ----
    //   writes gy_bf[0..OUT-1] (bf16 code) + gy_real[0..OUT-1] (decoded real).
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
            gg_deq        = gg_deq * gg_pow_dn;       // undo 2^a_shift (exact)
            gy_bf[gg_o]   = real2b(gg_deq);
            gy_real[gg_o] = b2real(gy_bf[gg_o]);
        end
    end endtask

    // ---- stimulus build ----
    integer sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc=seed0;
        for (i=0;i<MODEL_DIM;i=i+1) begin G1[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin G2[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        ScW_dq=gen_scale(sc); sc=sc+1;  ScW_uq=gen_scale(sc); sc=sc+1;
        ScW_dkv=gen_scale(sc); sc=sc+1; ScW_kr=gen_scale(sc); sc=sc+1;
        ScW_uk=gen_scale(sc); sc=sc+1;  ScW_uv=gen_scale(sc); sc=sc+1;
        ScW_o=gen_scale(sc); sc=sc+1;
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<N_EXPERT;j=j+1) begin Wg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        ScWg=gen_scale(sc); sc=sc+1;
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Dg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Du[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_DENSE;j=j+1) begin Dd[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin ScDg[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin ScDu[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin ScDd[i]=gen_scale(sc); sc=sc+1; end
        for (e=0;e<N_EXPERT;e=e+1) begin
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mg[e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mu[e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin Md[e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScMg[e]=gen_scale(sc); sc=sc+1; ScMu[e]=gen_scale(sc); sc=sc+1; ScMd[e]=gen_scale(sc); sc=sc+1;
        end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHu[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin SHd[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        ScSHg=gen_scale(sc); sc=sc+1; ScSHu=gen_scale(sc); sc=sc+1; ScSHd=gen_scale(sc); sc=sc+1;
        for (i=0;i<MODEL_DIM;i=i+1) begin xin[i]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ---- RMSNorm(z, gamma) -> bf16 out[] (bf16 path, unchanged) ----
    task rmsnorm; input integer src; input integer dst;
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

    // ---- attention golden over g_nrm1[] -> g_attn[] (FP8 weight projections) ----
    task attn_golden; begin
        // Q low-rank: x*W_dq (FP8)
        for (i=0;i<MODEL_DIM;i=i+1) asrc_bf[i]=g_nrm1[i];
        fp8_gemm(0, MODEL_DIM, Q_LORA);
        for (o=0;o<Q_LORA;o=o+1) q_qlora[o]=gy_bf[o];
        // q low-rank RMSNorm (gamma=1, bf16)
        accr=0.0;
        for (i=0;i<Q_LORA;i=i+1) accr=accr+b2real(q_qlora[i])*b2real(q_qlora[i]);
        accr=1.0/$sqrt(accr/Q_LORA + 1e-5);
        for (i=0;i<Q_LORA;i=i+1) q_qln[i]=real2b(b2real(q_qlora[i])*accr);
        // Q full: qlora_n*W_uq (FP8)
        for (i=0;i<Q_LORA;i=i+1) asrc_bf[i]=q_qln[i];
        fp8_gemm(1, Q_LORA, HQK);
        for (o=0;o<HQK;o=o+1) q_qfull[o]=gy_bf[o];
        // RoPE q per head (bf16)
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
        // KV path per cached key (FP8 W_uk, W_uv ; bf16 norm/rope-from-cache)
        for (sd=0;sd<Sg;sd=sd+1) begin
            accr=0.0;
            for (i=0;i<KV_LORA;i=i+1) accr=accr+b2real(CKV[sd][i])*b2real(CKV[sd][i]);
            accr=1.0/$sqrt(accr/KV_LORA + 1e-5);
            for (i=0;i<KV_LORA;i=i+1) q_ckvn[i]=real2b(b2real(CKV[sd][i])*accr);
            for (i=0;i<KV_LORA;i=i+1) asrc_bf[i]=q_ckvn[i];
            fp8_gemm(4, KV_LORA, HNOPE);                 // ckv_n*W_uk
            for (o=0;o<HNOPE;o=o+1) q_knope[o]=gy_bf[o];
            fp8_gemm(5, KV_LORA, HV);                    // ckv_n*W_uv (same a_shift)
            for (o=0;o<HV;o=o+1) q_v[o]=gy_bf[o];
            for (hd=0;hd<H_HEADS;hd=hd+1) begin
                for (d=0;d<NOPE;d=d+1) gK[hd][sd][d]=b2real(q_knope[hd*NOPE+d]);
                for (d=0;d<ROPE;d=d+1) gK[hd][sd][NOPE+d]=b2real(KRP[sd][d]);
                for (d=0;d<V_DIM;d=d+1) gV[hd][sd][d]=b2real(q_v[hd*V_DIM+d]);
            end
        end
        // scores (bf16 q.K engine -> bf16)
        for (hd=0;hd<H_HEADS;hd=hd+1)
            for (sd=0;sd<Sg;sd=sd+1) begin
                accr=0.0;
                for (d=0;d<QK_DIM;d=d+1) accr=accr+gq[hd*QK_DIM+d]*gK[hd][sd][d];
                gsc[hd][sd]=b2real(real2b(accr));
            end
        // softmax (bf16)
        for (hd=0;hd<H_HEADS;hd=hd+1) begin
            mxr=gsc[hd][0];
            for (sd=1;sd<Sg;sd=sd+1) if (gsc[hd][sd]>mxr) mxr=gsc[hd][sd];
            sumr=0.0;
            for (sd=0;sd<Sg;sd=sd+1) sumr=sumr+$exp(gsc[hd][sd]-mxr);
            for (sd=0;sd<Sg;sd=sd+1) gp[hd][sd]=b2real(real2b($exp(gsc[hd][sd]-mxr)/sumr));
        end
        // context (bf16 fp32-acc -> bf16)
        for (hd=0;hd<H_HEADS;hd=hd+1)
            for (d=0;d<V_DIM;d=d+1) begin
                accr=0.0;
                for (sd=0;sd<Sg;sd=sd+1) accr=accr+gp[hd][sd]*gV[hd][sd][d];
                gctx[hd*V_DIM+d]=b2real(real2b(accr));
            end
        // out proj: ctx*W_o (FP8)
        for (o=0;o<HV;o=o+1) asrc_bf[o]=real2b(gctx[o]);
        fp8_gemm(6, HV, MODEL_DIM);
        for (o=0;o<MODEL_DIM;o=o+1) g_attn[o]=gy_real[o];
    end endtask

    // ---- one swiglu expert over g_nrm2[] (FP8 gate/up/down) ----
    // which: -1 DENSE ; 0..N_EXPERT-1 routed bank ; N_EXPERT shared
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
        // h = bf16( bf16(silu(gate)) * up )  (bf16 tail)
        for (ii=0;ii<INTER;ii=ii+1) begin
            ga  = b2real(gate_save[ii]);
            up_ = b2real(up_save[ii]);
            hh  = b2real(real2b(silu(ga)));
            hbuf_g[ii] = real2b( hh * up_ );
        end
        // down: h*W_down (FP8)
        for (ii=0;ii<INTER;ii=ii+1) asrc_bf[ii]=hbuf_g[ii];
        if (which==-1)            fp8_gemm(10, INTER_DENSE, MODEL_DIM);
        else if (which==N_EXPERT) fp8_gemm(16, INTER_MOE,  MODEL_DIM);
        else                      fp8_gemm(13, INTER_MOE,  MODEL_DIM);
        for (i=0;i<MODEL_DIM;i=i+1) g_yexp[i]=gy_real[i];
    end endtask

    // ---- full-layer golden -> gout[] ----
    integer si, sj, bestj;
    real bestv;
    task compute_golden; begin
        rmsnorm(0,0);                 // pre-attn norm -> g_nrm1
        attn_golden();                // FP8 attention -> g_attn
        for (i=0;i<MODEL_DIM;i=i+1)
            g_h[i] = real2b( b2real(xin[i]) + g_attn[i] );   // h = x + attn (bf16)
        rmsnorm(1,1);                 // pre-ffn norm -> g_nrm2
        if (tmode==0) begin
            swiglu_golden(-1);
            for (i=0;i<MODEL_DIM;i=i+1) g_ffn[i]=g_yexp[i];
        end else begin
            // router: FP8 gate GEMV -> bf16 logits -> sigmoid -> top-K -> renorm*2.5
            for (i=0;i<MODEL_DIM;i=i+1) asrc_bf[i]=g_nrm2[i];
            fp8_gemm(7, MODEL_DIM, N_EXPERT);
            for (e=0;e<N_EXPERT;e=e+1)
                g_gate[e]=b2real(real2b(sigm( b2real(gy_bf[e]) )));
            for (e=0;e<N_EXPERT;e=e+1) gate_used[e]=1'b0;
            for (si=0;si<TOPK;si=si+1) begin
                bestj=-1; bestv=-1.0e30;
                for (sj=0;sj<N_EXPERT;sj=sj+1)
                    if (!gate_used[sj] && g_gate[sj]>bestv) begin bestv=g_gate[sj]; bestj=sj; end
                top_idx[si]=bestj; top_g[si]=g_gate[bestj]; gate_used[bestj]=1'b1;
            end
            sumr=0.0;
            for (si=0;si<TOPK;si=si+1) sumr=sumr+top_g[si];
            for (si=0;si<TOPK;si=si+1)
                top_w[si]=b2real(real2b( (top_g[si]/sumr)*2.5 ));
            for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=0.0;
            for (si=0;si<TOPK;si=si+1) begin
                swiglu_golden(top_idx[si]);
                for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=facc_g[i]+top_w[si]*g_yexp[i];
            end
            swiglu_golden(N_EXPERT);   // shared, weight 1
            for (i=0;i<MODEL_DIM;i=i+1) facc_g[i]=facc_g[i]+g_yexp[i];
            for (i=0;i<MODEL_DIM;i=i+1) g_ffn[i]=b2real(real2b(facc_g[i]));
        end
        for (i=0;i<MODEL_DIM;i=i+1)
            gout[i]=b2real(real2b( b2real(g_h[i]) + g_ffn[i] ));   // y = h + ffn (bf16)
    end endtask

    // ---- cycle counter ----
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
            // y = h + ffn is a bf16 residual add; its absolute error is bounded by
            // the fp8/bf16 error of the operands h and ffn, NOT of the (possibly
            // near-cancelled) result y.  Normalize by the LARGEST of |y|,|h|,|ffn|
            // (and TINY) so catastrophic cancellation does not spuriously inflate
            // the relative metric -- this is the principled fp8 residual tolerance.
            denom = (refv<0.0?-refv:refv);
            if ((b2real(g_h[o])<0.0?-b2real(g_h[o]):b2real(g_h[o])) > denom)
                denom = (b2real(g_h[o])<0.0?-b2real(g_h[o]):b2real(g_h[o]));
            if ((g_ffn[o]<0.0?-g_ffn[o]:g_ffn[o]) > denom)
                denom = (g_ffn[o]<0.0?-g_ffn[o]:g_ffn[o]);
            if (denom < TINY) denom = TINY;
            this_rel = adiff / denom;
            // MIXED tolerance: an element passes if it is within REL_TOL relative
            // OR within ABS_TOL absolute.  ABS_TOL is the fp8 GEMM ABSOLUTE noise
            // floor: a near-cancelled output element (|y| a few * 2^-5) is dominated
            // by E4M3 quantization noise of its dot terms (sqrt(K)*|act*wgt|*2^-4 ~
            // a few * 2^-6), so a relative metric there is meaningless.  Above the
            // floor the relative metric governs.  worst_rel reports the worst
            // ABOVE-floor relative error (the meaningful one).
            if (adiff>ABS_TOL && this_rel>worst_rel) worst_rel=this_rel;
            if (this_rel>REL_TOL && adiff>ABS_TOL) begin
                $display("FAIL[%0s]: o=%0d dut=%g ref=%g relerr=%g abserr=%g (REL>%g & ABS>%g)",
                         label, o, dutv, refv, this_rel, adiff, REL_TOL, ABS_TOL);
                fails_in_test=fails_in_test+1;
            end
        end
        test_count=test_count+1;
        if (worst_rel>gworst) gworst=worst_rel;
        if (fails_in_test!=0) errors=errors+fails_in_test;
        else $display("PASS[%0s] worst_rel=%g", label, worst_rel);
    end endtask

    // ---- watchdog ----
    initial begin
        #80000000;
        $display("FAIL: global timeout"); $fatal;
    end

    // ---- main ----
    initial begin
        // The fp8 weight-projection grid (E4M3, 3-bit mantissa) is shared IDENTICALLY
        // by DUT and golden (same codes, scales, on-chip a_shift), so it cancels; the
        // ABOVE-floor relative error is then just the bf16-rounding + fp32-vs-fp64
        // block-accumulation budget compounded through the chain.  REL_TOL = one
        // E4M3 step (1/8, headroom for an a_shift flip on a boundary element) + a
        // per-reduction-depth accumulation term.  ABS_TOL = 2^-5 is the fp8 GEMM
        // absolute noise floor below which a (near-cancelled) output is dominated by
        // E4M3 dot-term quantization noise and a relative metric is meaningless.
        REL_TOL = (1.0/8.0)
                + ( (MODEL_DIM + Q_LORA + HQK + KV_LORA*2 + HV + INTER_DENSE + MODEL_DIM)
                    * (1.0/8192.0) );
        ABS_TOL = 2.0**(-5);
        TINY    = 1.0e-3;
        errors=0; test_count=0; worst_rel=0.0; gworst=0.0; cyc_cnt=0;

        rst=1'b1; start=1'b0; mode=1'b0; pos={POSW{1'b0}};
        s_len={(IDXW+1){1'b0}}; x_vec={MODEL_DIM*16{1'b0}};
        repeat(4) @(negedge clk);
        rst=1'b0;
        @(negedge clk);

        // === DENSE mode ===
        tmode=0;
        tband=0; tpos=0;  Sg=1; build_stimulus(500, tband); compute_golden(); run_dut(); check_out("DENSE b0 pos0 S1");
        tband=0; tpos=5;  Sg=2; build_stimulus(1000,tband); compute_golden(); run_dut(); check_out("DENSE b0 pos5 S2");
        tband=0; tpos=129;Sg=4; build_stimulus(20000,tband);compute_golden(); run_dut(); check_out("DENSE b0 pos129 S4");
        tband=1; tpos=42; Sg=S_MAX; build_stimulus(90000,tband); compute_golden(); run_dut(); check_out("DENSE b1 pos42 Smax");

        // === MoE mode ===
        tmode=1;
        tband=0; tpos=0;  Sg=1; build_stimulus(31000,tband); compute_golden(); run_dut(); check_out("MoE b0 pos0 S1");
        tband=0; tpos=2;  Sg=2; build_stimulus(33000,tband); compute_golden(); run_dut(); check_out("MoE b0 pos2 S2");
        tband=0; tpos=300;Sg=5; build_stimulus(70000,tband); compute_golden(); run_dut(); check_out("MoE b0 pos300 S5");
        tband=1; tpos=128;Sg=S_MAX; build_stimulus(95000,tband); compute_golden(); run_dut(); check_out("MoE b1 pos128 Smax");

        // === DENSE re-arm (determinism) ===
        tmode=0;
        tband=0; tpos=5; Sg=2; build_stimulus(1000,tband); compute_golden(); run_dut(); check_out("DENSE re-arm b0 pos5 S2");

        if (errors!=0) begin
            $display("FAILED: %0d element error(s) across %0d tests; gworst=%g", errors, test_count, gworst);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED  (gworst_rel=%g, REL_TOL=%g, ABS_TOL=%g)",
                 test_count, gworst, REL_TOL, ABS_TOL);
        $finish;
    end
endmodule
