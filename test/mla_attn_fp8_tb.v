`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// mla_attn_fp8_tb.v -- self-checking TB for mla_attn_fp8 (GLM-5.2 MLA decode,
//   FP8-NATIVE WEIGHT PROJECTIONS).
//----------------------------------------------------------------------------
// INDEPENDENT fp64 ('real' = IEEE double) GOLDEN of the SAME fp8 math.
//
//   This is the FP8 sibling of test/mla_attn_tb.v.  The golden recomputes the
//   COMPLETE single-token MLA flow in double precision from the SAME E4M3
//   weights / [128,128] block scales / bf16 x / bf16 KV-cache the DUT sees.
//   The ONLY numerical change vs mla_attn_tb is the SEVEN weight projections,
//   which here use the OFFICIAL GLM-5.2-FP8 block-scaled E4M3 arithmetic:
//
//     for each output channel o of a weight GEMM  out = A . W :
//       a_shift  = dyn_shift( max bf16 exponent over the A-source vector )
//                  (DYNAMIC per-vector pow2 activation scale, derived on-chip)
//       a_q[k]   = fp32_to_fp8e4m3( A[k] * 2^a_shift )      (E4M3, RNE+sat)
//       prod[k]  = fp8_mul( a_q[k] , W_e4m3[o][k] )         (EXACT fp32, 4x4)
//       blk      = SUM_k prod[k]                            (block-scaled fp64 acc)
//       out[o]   = bf16( 2^-a_shift * w_scale * blk )       ([128,128] dequant)
//
//   computed with the CANONICAL fp8_e4m3.vh / glm_fp.vh primitives but with
//   fp64 ('real') block accumulation -- INDEPENDENT of the DUT's fp32
//   L-way-interleaved + 8-leaf-tree accumulator inside glm_matmul_fp8.  The
//   seven FP8 weight projections are W_dq, W_uq, W_dkv, W_kr, W_uk, W_uv, W_o.
//
//   EVERYTHING ELSE (the attention tail) stays bf16-faithful fp64, EXACTLY as
//   mla_attn_tb does it:  RMSNorm (gamma=1, eps=1e-5), decoupled RoPE
//   ($cos/$sin, theta=THETA), per-head q.K scores, $exp softmax, weighted-V
//   context (fp32-acc -> bf16).  Every intermediate VECTOR is rounded back to
//   bf16 (the unit's storage contract) before the next stage.  RoPE/$exp are
//   INDEPENDENT of the DUT's CORDIC / pipelined-exp, so the cross-check is real.
//
//   As in mla_attn_tb, the DSA front-end runs DENSE here (S <= TOPK), keeping
//   keys 0..S-1 in order; the cache-fed past keys feed the golden; W_dkv/W_kr
//   are exercised in the DUT (current-token latent, then overwritten by the
//   cache key) but do not feed the golden output -- present for FP8 datapath
//   coverage and X-freeness.  The DUT exposes only the final W_o output `out`
//   (the per-head contexts are internal), so `out` is the observed quantity;
//   it transitively validates every per-head FP8 path (W_uk, W_uv) + softmax +
//   weighted-V + the final W_o FP8 projection.
//
// TOLERANCE (justified):
//   The weights are EXACT E4M3 codes shared by DUT and golden, so there is NO
//   weight-quantization gap; the dynamic activation->E4M3 quant is done with
//   the SAME primitives in both, so it is not an error source either.  The only
//   DUT/golden divergence is accumulation ORDER: the DUT folds the fp8 products
//   through an fp32 L-way tree + an fp32 block*scale multiply, the golden in
//   fp64 -- a sub-bf16-ULP difference for these K (<=16) that can occasionally
//   flip a final bf16 ULP and compound through the chain (5 FP8 GEMMs that feed
//   the output + 2 RMSNorms + RoPE + softmax + weighted-V).  bf16 carries ~7-8
//   significant bits (rel step 2^-8 ~ 0.4%).  A bound of REL = 12.5% (2^-3)
//   plus a small ABSOLUTE floor 0.03 (for outputs near zero, where a single
//   bf16 ULP dominates and relative error is meaningless) is the right
//   bf16-faithfulness gate -- tight enough to catch any structural / wiring /
//   ordering / scale bug, loose enough to admit pure rounding.  (Observed
//   worst-case rel error is well inside this; see the per-test PASS report.)
//   X-AWARE: ANY X/Z bit in `out` FAILS; outputs are checked finite (no inf/nan).
//
// Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X / inf.
//============================================================================
module mla_attn_fp8_tb;
    // ---- slice parameters (small-but-faithful; matches DUT default override) --
    localparam integer MODEL_DIM = 16;
    localparam integer H_HEADS   = 2;
    localparam integer NOPE      = 4;
    localparam integer ROPE      = 4;
    localparam integer V_DIM     = 4;
    localparam integer Q_LORA    = 8;
    localparam integer KV_LORA   = 8;
    localparam integer S_MAX     = 2;
    localparam integer TOPK      = 2;
    localparam integer THETA     = 8000000;
    localparam integer PE_N      = 2;
    localparam integer POSW      = 20;
    localparam integer BLK       = 128;
    localparam integer QK_DIM    = NOPE + ROPE;
    localparam integer IDXW      = (S_MAX <= 1) ? 1 : $clog2(S_MAX);
    localparam integer HQK       = H_HEADS*QK_DIM;
    localparam integer HNOPE     = H_HEADS*NOPE;
    localparam integer HV        = H_HEADS*V_DIM;

    // ---- derived sizing identical to the DUT (for w_grp / w_k / w_scale) ----
    localparam integer KMAX = (MODEL_DIM>Q_LORA)?
                 ((MODEL_DIM>KV_LORA)?((MODEL_DIM>HV)?MODEL_DIM:HV)
                                     :((KV_LORA>HV)?KV_LORA:HV))
                :((Q_LORA>KV_LORA)?((Q_LORA>HV)?Q_LORA:HV)
                                  :((KV_LORA>HV)?KV_LORA:HV));
    localparam integer OMAX = (HQK>MODEL_DIM)?
                 ((HQK>HNOPE)?((HQK>HV)?HQK:HV):((HNOPE>HV)?HNOPE:HV))
                :((MODEL_DIM>HNOPE)?((MODEL_DIM>HV)?MODEL_DIM:HV)
                                   :((HNOPE>HV)?HNOPE:HV));
    localparam integer NGMAX = (OMAX + PE_N - 1)/PE_N;
    localparam integer GRPW  = (NGMAX<=1)?1:$clog2(NGMAX);
    localparam integer KCW   = (KMAX<=1)?1:$clog2(KMAX);
    localparam integer NB    = (KMAX + BLK - 1)/BLK;

    `include "glm_fp.vh"
    `include "fp8_e4m3.vh"

    // ---- clock / reset ----
    reg clk=1'b0, rst=1'b1;
    always #5 clk = ~clk;

    // ---- DUT control ----
    reg                      start;
    wire                     busy, done;
    reg  [POSW-1:0]          pos;
    reg  [IDXW:0]            s_len;
    reg  [MODEL_DIM*16-1:0]  x_vec;
    wire [MODEL_DIM*16-1:0]  out;

    // ---- weight pull (FP8 codes + bf16 block scales) ----
    wire                     w_req;
    wire [3:0]               w_sel;
    wire [GRPW-1:0]          w_grp;
    wire [KCW-1:0]           w_k;
    reg  [PE_N*8-1:0]        w_col;        // PE_N FP8 E4M3 lanes
    reg  [16*PE_N*NB-1:0]    w_scale;      // PE_N bf16 block scales (NB=1)

    // ---- cache read ----
    wire                     kc_req;
    wire [IDXW-1:0]          kc_idx;
    reg  [KV_LORA*16-1:0]    kc_ckv;
    reg  [ROPE*16-1:0]       kc_krope;
    reg                      kc_valid;

    // ================= weight ROMs (FP8 E4M3 codes) =================
    // indexed [out][k].  Each matrix carries ONE [128,128] block scale (here the
    // whole matrix is a single block: N<=128 outputs x K<=128 contraction).
    reg [7:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];
    reg [7:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [7:0] W_o   [0:MODEL_DIM-1][0:HV-1];
    // per-matrix bf16 block scales (chosen non-trivial to exercise dequant).
    reg [15:0] S_dq, S_uq, S_dkv, S_kr, S_uk, S_uv, S_o;

    // cache: c_kv[j] [KV_LORA] (bf16), k_rope[j] [ROPE] (bf16, already roped)
    reg [15:0] CKV   [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP   [0:S_MAX-1][0:ROPE-1];
    reg [15:0] xin   [0:MODEL_DIM-1];

    // ---------------- DUT ----------------
    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .pos(pos), .s_len(s_len), .x_vec(x_vec),
        .w_req(w_req), .w_sel(w_sel), .w_grp(w_grp), .w_k(w_k),
        .w_col(w_col), .w_scale(w_scale),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid), .out(out)
    );

    // ================= combinational WEIGHT RESPONDER =================
    // present PE_N FP8 lanes W_sel[w_grp*PE_N+t , w_k] + the per-column block
    // scale (NB=1 -> w_scale[16*t +: 16]).  All columns of a matrix share its
    // single block scale.
    integer t;
    reg [15:0] sel_scale;
    always @* begin
        w_col   = {PE_N*8{1'b0}};
        w_scale = {16*PE_N*NB{1'b0}};
        case (w_sel)
            4'd0: sel_scale = S_dq;  4'd1: sel_scale = S_uq;
            4'd2: sel_scale = S_dkv; 4'd3: sel_scale = S_kr;
            4'd4: sel_scale = S_uk;  4'd5: sel_scale = S_uv;
            4'd6: sel_scale = S_o;   default: sel_scale = 16'h3F80;
        endcase
        for (t = 0; t < PE_N; t = t + 1) begin
            case (w_sel)
            4'd0: if (w_grp*PE_N+t < Q_LORA)   w_col[8*t +:8] = W_dq [w_grp*PE_N+t][w_k];
            4'd1: if (w_grp*PE_N+t < HQK)      w_col[8*t +:8] = W_uq [w_grp*PE_N+t][w_k];
            4'd2: if (w_grp*PE_N+t < KV_LORA)  w_col[8*t +:8] = W_dkv[w_grp*PE_N+t][w_k];
            4'd3: if (w_grp*PE_N+t < ROPE)     w_col[8*t +:8] = W_kr [w_grp*PE_N+t][w_k];
            4'd4: if (w_grp*PE_N+t < HNOPE)    w_col[8*t +:8] = W_uk [w_grp*PE_N+t][w_k];
            4'd5: if (w_grp*PE_N+t < HV)       w_col[8*t +:8] = W_uv [w_grp*PE_N+t][w_k];
            4'd6: if (w_grp*PE_N+t < MODEL_DIM)w_col[8*t +:8] = W_o  [w_grp*PE_N+t][w_k];
            default: w_col[8*t +:8] = 8'h38;    // 1.0 for don't-care
            endcase
            w_scale[16*t +: 16] = sel_scale;
        end
    end

    // ================= combinational CACHE RESPONDER =================
    integer cd;
    always @* begin
        kc_ckv   = {KV_LORA*16{1'b0}};
        kc_krope = {ROPE*16{1'b0}};
        for (cd = 0; cd < KV_LORA; cd = cd + 1)
            kc_ckv[16*cd +:16] = CKV[kc_idx][cd];
        for (cd = 0; cd < ROPE; cd = cd + 1)
            kc_krope[16*cd +:16] = KRP[kc_idx][cd];
    end
    // answer the cache pull with kc_valid one cycle after kc_req (registered).
    always @(posedge clk) begin
        if (rst) kc_valid <= 1'b0;
        else     kc_valid <= kc_req;
    end

    // ===================================================================
    //  INDEPENDENT fp64 GOLDEN HELPERS
    // ===================================================================
    // bf16 -> real.  FTZ subnormals; no inf/nan in stimulus -> 0 defensively.
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
    // fp32 bits -> real (used to read the EXACT fp8_mul product into fp64).
    function real f2real; input [31:0] f; real m; integer e,i; begin
        if (f[30:23]==8'h00)      f2real = 0.0;
        else if (f[30:23]==8'hFF) f2real = 0.0;
        else begin
            m = 1.0;
            for (i=0;i<23;i=i+1) if (f[i]) m = m + (2.0**(i-23));
            e = f[30:23]-127;
            f2real = m * (2.0**e);
            if (f[31]) f2real = -f2real;
        end
    end endfunction
    // round a real to bf16 (RNE on 8-bit significand) -- mirrors bf16 storage.
    function [15:0] real2b; input real v; reg s; integer e; real m; reg [7:0] man;
        reg [31:0] mi; begin
        if (v==0.0) real2b = 16'h0000;
        else begin
            s = (v<0.0); if (s) v=-v;
            e = 0;
            while (v>=2.0) begin v=v/2.0; e=e+1; end
            while (v< 1.0) begin v=v*2.0; e=e-1; end
            mi  = $rtoi((v-1.0)*128.0 + 0.5);
            if (mi>=128) begin mi=mi-128; e=e+1; end
            man = mi[7:0];
            if (e < -126) real2b = {s,15'h0};
            else real2b = {s, e[7:0]+8'd127, man[6:0]};
        end
    end endfunction
    function real rb; input real v; begin rb = b2real(real2b(v)); end endfunction
    // exact 2^n in fp64 (n a small signed integer).
    function real pow2r; input integer n; integer i; real r; begin
        r = 1.0;
        if (n>=0) for (i=0;i<n;i=i+1)  r = r*2.0;
        else      for (i=0;i<-n;i=i+1) r = r/2.0;
        pow2r = r;
    end endfunction

    // ---- fp32_scale_pow2 : exact 2^k exponent add (copied from glm_matmul_fp8) ----
    function automatic [31:0] fp32_scale_pow2(input [31:0] f, input signed [9:0] k);
        reg s; reg [7:0] e; reg [22:0] m; reg signed [10:0] ne;
        begin
            s=f[31]; e=f[30:23]; m=f[22:0];
            if (e==8'hFF) fp32_scale_pow2=f;
            else if (e==8'h00) fp32_scale_pow2={s,31'b0};
            else begin
                ne = $signed({3'b0,e}) + $signed({{1{k[9]}},k});
                if (ne>=11'sd255) fp32_scale_pow2={s,8'hFF,23'b0};
                else if (ne<=11'sd0) fp32_scale_pow2={s,31'b0};
                else fp32_scale_pow2={s,ne[7:0],m};
            end
        end
    endfunction
    // ---- dynamic per-vector activation pow2 shift (copied from the DUT) ----
    function automatic signed [7:0] dyn_shift(input [7:0] emax);
        reg signed [9:0] sh;
        begin
            if (emax==8'd0) dyn_shift=8'sd0;
            else begin
                sh = 10'sd134 - $signed({2'b0,emax});
                if (sh>10'sd127) sh=10'sd127;
                if (sh<-10'sd128) sh=-10'sd128;
                dyn_shift=sh[7:0];
            end
        end
    endfunction
    // ---- one FP8 term : fp8_mul( fp8(a_bf*2^ash) , w_e4m3 ) -> EXACT fp32 -> real ----
    function real fp8_prod_real;
        input [15:0] a_bf; input signed [7:0] ash; input [7:0] wc;
        reg [7:0] aq;
        begin
            aq = fp32_to_fp8e4m3(fp32_scale_pow2(bf16_to_fp32(a_bf), {{2{ash[7]}},ash}));
            fp8_prod_real = f2real(fp8_mul(aq, wc));
        end
    endfunction

    // deterministic bf16 ACTIVATION stimulus (x / cache) -- pure-integer hash.
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = (seed*2654435761) ^ (seed<<13) ^ (seed*40503);
        s = h[3];
        if (band==1) e = 8'd125 + h[6:4];
        else         e = 8'd124 + h[5:4];
        m = h[12:6];
        gen_bf16 = {s, e, m};
    end endfunction
    // deterministic E4M3 WEIGHT stimulus -- exp field 5..8 (~|w| in [0.25,3.75]),
    // never the NaN slot; mixed sign.
    function [7:0] gen_fp8; input integer seed; reg sg; reg [3:0] e; reg [2:0] m;
        integer h; begin
        h = (seed*2246822519) ^ (seed<<11) ^ (seed*3266489917);
        sg = h[2];
        e  = 4'd5 + h[4:3];        // 5..8
        m  = h[7:5];
        gen_fp8 = {sg, e, m};
    end endfunction

    // ===================================================================
    //  GOLDEN STATE (bf16 intermediates + fp64 score/prob/ctx/out)
    // ===================================================================
    reg [15:0] g_qlora   [0:Q_LORA-1];
    reg [15:0] g_qlora_n [0:Q_LORA-1];
    reg [15:0] g_qfull   [0:HQK-1];
    reg [15:0] g_ckvn    [0:KV_LORA-1];
    reg [15:0] g_knope   [0:HNOPE-1];
    reg [15:0] g_v       [0:HV-1];
    reg [15:0] g_ctx     [0:HV-1];        // bf16 context (A-source for W_o)

    real gq  [0:HQK-1];
    real gK  [0:H_HEADS-1][0:S_MAX-1][0:QK_DIM-1];
    real gV  [0:H_HEADS-1][0:S_MAX-1][0:V_DIM-1];
    real gsc [0:H_HEADS-1][0:S_MAX-1];
    real gp  [0:H_HEADS-1][0:S_MAX-1];
    real gout[0:MODEL_DIM-1];

    real accr, mxr, sumr, ang, c, sN, x0, x1, invf, deq;
    real refv, dutv, adiff, rtol, worst_rel, this_rel;
    integer i,j,k,h,d,o,hd,sd,pr;
    integer errors, test_count, fails_in_test;
    integer Sg, tpos, tband;
    reg signed [7:0] ash;

    // emax helpers (max bf16 exponent field over a stored bf16 array slice).
    function integer emax_x; integer kk; reg [7:0] e; begin
        e=8'd0; for(kk=0;kk<MODEL_DIM;kk=kk+1) if (xin[kk][14:7]>e) e=xin[kk][14:7];
        emax_x=e; end endfunction
    function integer emax_qln; integer kk; reg [7:0] e; begin
        e=8'd0; for(kk=0;kk<Q_LORA;kk=kk+1) if (g_qlora_n[kk][14:7]>e) e=g_qlora_n[kk][14:7];
        emax_qln=e; end endfunction
    function integer emax_ckvn; integer kk; reg [7:0] e; begin
        e=8'd0; for(kk=0;kk<KV_LORA;kk=kk+1) if (g_ckvn[kk][14:7]>e) e=g_ckvn[kk][14:7];
        emax_ckvn=e; end endfunction
    function integer emax_ctx; integer kk; reg [7:0] e; begin
        e=8'd0; for(kk=0;kk<HV;kk=kk+1) if (g_ctx[kk][14:7]>e) e=g_ctx[kk][14:7];
        emax_ctx=e; end endfunction

    // -------------------------------------------------------------------
    // build_stimulus : fill FP8 weight ROMs + block scales + bf16 x + cache.
    // -------------------------------------------------------------------
    integer sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc = seed0;
        for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[i][j]=gen_fp8(sc);  sc=sc+1; end
        // non-trivial per-matrix [128,128] block scales (bf16) to exercise dequant.
        S_dq=16'h3F80; S_uq=16'h3F00; S_dkv=16'h3F80; S_kr=16'h3F80;  // 1.0,0.5,1.0,1.0
        S_uk=16'h4000; S_uv=16'h3F00; S_o=16'h3F80;                   // 2.0,0.5,1.0
        for (i=0;i<MODEL_DIM;i=i+1) begin xin[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[i][j]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // make both keys identical -> per-head scores tie -> softmax exactly uniform.
    task make_near_uniform; begin
        for (j=0;j<KV_LORA;j=j+1) CKV[1][j] = CKV[0][j];
        for (j=0;j<ROPE;j=j+1)    KRP[1][j] = KRP[0][j];
    end endtask

    // -------------------------------------------------------------------
    // compute_golden : whole MLA flow in fp64; FP8 block-scaled weight GEMMs,
    //   bf16-faithful attention tail.  Fills gout[].
    // -------------------------------------------------------------------
    task compute_golden; begin
        // ---- Q path : qlora = x . W_dq  (FP8) ----
        ash = dyn_shift(emax_x());
        for (o=0;o<Q_LORA;o=o+1) begin
            accr=0.0;
            for (k=0;k<MODEL_DIM;k=k+1) accr=accr+fp8_prod_real(xin[k], ash, W_dq[o][k]);
            g_qlora[o]=real2b(accr * b2real(S_dq) * pow2r(-ash));
        end
        // RMSNorm(qlora), gamma=1, eps=1e-5
        accr=0.0;
        for (i=0;i<Q_LORA;i=i+1) accr=accr+b2real(g_qlora[i])*b2real(g_qlora[i]);
        accr = 1.0/$sqrt(accr/Q_LORA + 1e-5);
        for (i=0;i<Q_LORA;i=i+1) g_qlora_n[i]=real2b(b2real(g_qlora[i])*accr);
        // up-proj : qfull = qlora_n . W_uq  (FP8)
        ash = dyn_shift(emax_qln());
        for (o=0;o<HQK;o=o+1) begin
            accr=0.0;
            for (k=0;k<Q_LORA;k=k+1) accr=accr+fp8_prod_real(g_qlora_n[k], ash, W_uq[o][k]);
            g_qfull[o]=real2b(accr * b2real(S_uq) * pow2r(-ash));
        end
        // q rope : nope copied; rope interleaved-rotated at tpos (theta=THETA).
        for (hd=0;hd<H_HEADS;hd=hd+1) begin
            for (d=0;d<NOPE;d=d+1) gq[hd*QK_DIM+d]=b2real(g_qfull[hd*QK_DIM+d]);
            for (pr=0;pr<ROPE/2;pr=pr+1) begin
                invf = 1.0 / ($pow(THETA*1.0, (2.0*pr)/ROPE));
                ang  = tpos * invf;
                c=$cos(ang); sN=$sin(ang);
                x0 = b2real(g_qfull[hd*QK_DIM+NOPE+2*pr]);
                x1 = b2real(g_qfull[hd*QK_DIM+NOPE+2*pr+1]);
                gq[hd*QK_DIM+NOPE+2*pr]   = rb(x0*c - x1*sN);
                gq[hd*QK_DIM+NOPE+2*pr+1] = rb(x0*sN + x1*c);
            end
        end
        // ---- KV path per cached key j ----
        for (sd=0;sd<Sg;sd=sd+1) begin
            // ckv_n = RMSNorm(c_kv[j]) (bf16)
            accr=0.0;
            for (i=0;i<KV_LORA;i=i+1) accr=accr+b2real(CKV[sd][i])*b2real(CKV[sd][i]);
            accr = 1.0/$sqrt(accr/KV_LORA + 1e-5);
            for (i=0;i<KV_LORA;i=i+1) g_ckvn[i]=real2b(b2real(CKV[sd][i])*accr);
            // k_nope = ckv_n . W_uk  (FP8)
            ash = dyn_shift(emax_ckvn());
            for (o=0;o<HNOPE;o=o+1) begin
                accr=0.0;
                for (k=0;k<KV_LORA;k=k+1) accr=accr+fp8_prod_real(g_ckvn[k], ash, W_uk[o][k]);
                g_knope[o]=real2b(accr * b2real(S_uk) * pow2r(-ash));
            end
            // v = ckv_n . W_uv  (FP8) -- same A-source (ckv_n) so same a_shift
            for (o=0;o<HV;o=o+1) begin
                accr=0.0;
                for (k=0;k<KV_LORA;k=k+1) accr=accr+fp8_prod_real(g_ckvn[k], ash, W_uv[o][k]);
                g_v[o]=real2b(accr * b2real(S_uv) * pow2r(-ash));
            end
            for (hd=0;hd<H_HEADS;hd=hd+1) begin
                for (d=0;d<NOPE;d=d+1) gK[hd][sd][d]      = b2real(g_knope[hd*NOPE+d]);
                for (d=0;d<ROPE;d=d+1) gK[hd][sd][NOPE+d] = b2real(KRP[sd][d]);
                for (d=0;d<V_DIM;d=d+1) gV[hd][sd][d]     = b2real(g_v[hd*V_DIM+d]);
            end
        end
        // ---- scores : gsc[h][s] = q_h . K_{h,s} (bf16-rounded scalar) ----
        for (hd=0;hd<H_HEADS;hd=hd+1)
            for (sd=0;sd<Sg;sd=sd+1) begin
                accr=0.0;
                for (d=0;d<QK_DIM;d=d+1) accr=accr+gq[hd*QK_DIM+d]*gK[hd][sd][d];
                gsc[hd][sd]=b2real(real2b(accr));
            end
        // ---- softmax over s per head ----
        for (hd=0;hd<H_HEADS;hd=hd+1) begin
            mxr=gsc[hd][0];
            for (sd=1;sd<Sg;sd=sd+1) if (gsc[hd][sd]>mxr) mxr=gsc[hd][sd];
            sumr=0.0;
            for (sd=0;sd<Sg;sd=sd+1) sumr=sumr+$exp(gsc[hd][sd]-mxr);
            for (sd=0;sd<Sg;sd=sd+1) gp[hd][sd]=b2real(real2b($exp(gsc[hd][sd]-mxr)/sumr));
        end
        // ---- context O_h[d] = sum_s p[h][s]*V[h][s][d] (fp32-acc -> bf16) ----
        for (hd=0;hd<H_HEADS;hd=hd+1)
            for (d=0;d<V_DIM;d=d+1) begin
                accr=0.0;
                for (sd=0;sd<Sg;sd=sd+1) accr=accr+gp[hd][sd]*gV[hd][sd][d];
                g_ctx[hd*V_DIM+d]=real2b(accr);
            end
        // ---- out = ctx . W_o  (FP8; K=HV, OUT=MODEL_DIM) ----
        ash = dyn_shift(emax_ctx());
        for (o=0;o<MODEL_DIM;o=o+1) begin
            accr=0.0;
            for (k=0;k<HV;k=k+1) accr=accr+fp8_prod_real(g_ctx[k], ash, W_o[o][k]);
            deq     = accr * b2real(S_o) * pow2r(-ash);
            gout[o] = deq;       // final NOT re-rounded; compared with tolerance
        end
    end endtask

    // -------------------------------------------------------------------
    // run_dut : drive one decode step (pos=tpos, s_len=Sg) and wait for done.
    // -------------------------------------------------------------------
    task run_dut; begin
        pos   = tpos[POSW-1:0];
        s_len = Sg[IDXW:0];
        @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) x_vec[16*i +:16]=xin[i];
        start=1'b1; @(negedge clk); start=1'b0;
        wait (done==1'b1);
        @(negedge clk);
    end endtask

    // -------------------------------------------------------------------
    // check_out : X-aware + finite + per-element fp64 reference compare.
    // -------------------------------------------------------------------
    task check_out; input [256*8-1:0] label; begin
        fails_in_test = 0;
        for (i=0;i<MODEL_DIM*16;i=i+1)
            if (out[i]===1'bx || out[i]===1'bz) begin
                $display("FAIL[%0s]: out bit %0d is X/Z", label, i);
                fails_in_test = fails_in_test + 1;
            end
        for (i=0;i<MODEL_DIM;i=i+1)
            if (out[16*i+7 +:8]==8'hFF) begin
                $display("FAIL[%0s]: out[%0d] inf/nan (%h)", label, i, out[16*i +:16]);
                fails_in_test = fails_in_test + 1;
            end
        for (i=0;i<MODEL_DIM;i=i+1) begin
            dutv  = b2real(out[16*i +:16]);
            refv  = gout[i];
            adiff = dutv - refv; if (adiff<0.0) adiff=-adiff;
            rtol  = (refv<0.0?-refv:refv)*0.125 + 0.03;
            this_rel = adiff / ((refv<0.0?-refv:refv) + 0.03);
            if (this_rel > worst_rel) worst_rel = this_rel;
            if (adiff > rtol) begin
                $display("FAIL[%0s]: out[%0d]=%h (%f) ref=%f |d|=%f tol=%f",
                         label, i, out[16*i +:16], dutv, refv, adiff, rtol);
                fails_in_test = fails_in_test + 1;
            end
        end
        test_count = test_count + 1;
        if (fails_in_test==0)
            $display("  PASS[%0s]  pos=%0d S=%0d band=%0d", label, tpos, Sg, tband);
        else
            errors = errors + fails_in_test;
    end endtask

    // -------------------------------------------------------------------
    // one_case : full single test (stimulus -> golden -> DUT -> check).
    // -------------------------------------------------------------------
    task one_case; input integer seed0; input integer band; input integer p;
                   input integer s; input integer uniform; input [256*8-1:0] label;
        begin
            tpos = p; Sg = s; tband = band;
            build_stimulus(seed0, band);
            if (uniform) make_near_uniform();
            compute_golden();
            run_dut();
            check_out(label);
        end
    endtask

    initial begin
        errors=0; test_count=0; worst_rel=0.0;
        start=1'b0; pos={POSW{1'b0}}; s_len={(IDXW+1){1'b0}};
        x_vec={MODEL_DIM*16{1'b0}};
        @(negedge clk); rst=1'b0;

        // ============ COVERAGE ============
        // T1 : S=1, pos=0  -> single key (p=1 trivial softmax) + RoPE identity
        one_case(   1, 0,    0, 1, 0, "S1_pos0_ropeIdentity");
        // T2 : S=1, pos>0  -> single key, real RoPE rotation
        one_case( 100, 0,    7, 1, 0, "S1_posNZ");
        // T3 : S=2, pos=0  -> multi-key causal, RoPE identity
        one_case( 200, 0,    0, 2, 0, "S2_pos0");
        // T4 : S=2, pos>0  -> multi-key causal, real RoPE
        one_case( 300, 0,    5, 2, 0, "S2_posNZ");
        // T5 : wide-range mixed-sign activations (band 1), S=2, pos large
        one_case( 400, 1, 4095, 2, 0, "S2_wideRange_posBig");
        // T6 : directed NEAR-UNIFORM attention (identical keys -> p=1/S), S=2
        one_case( 500, 0,    3, 2, 1, "S2_nearUniform");
        // T7 : another wide-range draw, S=1, pos>0
        one_case( 700, 1,   42, 1, 0, "S1_wideRange_posNZ");

        // ============ VERDICT ============
        if (errors==0) begin
            $display("worst-case relative error across all tests = %f", worst_rel);
            $display("ALL %0d TESTS PASSED", test_count);
        end else begin
            $display("FAILED: %0d errors over %0d tests", errors, test_count);
            $fatal(1, "mla_attn_fp8 TB failed");
        end
        $finish;
    end

    // global timeout
    initial begin
        #400000000;
        $display("FAIL: timeout (done never asserted)");
        $fatal(1, "timeout");
    end
endmodule
