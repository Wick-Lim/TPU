`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// mla_attn_tb.v -- self-checking TB for mla_attn (GLM-5.2 MLA decode step).
//----------------------------------------------------------------------------
// INDEPENDENT fp64 ('real' = IEEE double) GOLDEN.  The golden recomputes the
// COMPLETE single-token MLA flow in double precision from the SAME bf16
// weights / x / KV-cache the DUT sees:
//
//   Q PATH :  qlora   = x . W_dq                  (low-rank down-projection)
//             qlora_n = RMSNorm(qlora)            (gamma=1, eps=1e-5)
//             qfull   = qlora_n . W_uq            (up-projection -> H*QK_DIM)
//             q_rope  = interleaved RoPE(qfull[ROPE], pos)  (theta=THETA),
//                       NOPE part copied through unchanged
//   KV PATH:  per causal key j supplied by the cache:
//             ckv_n   = RMSNorm(c_kv[j])          (gamma=1, eps=1e-5)
//             k_nope  = ckv_n . W_uk              (per head NOPE)
//             v       = ckv_n . W_uv              (per head V_DIM)
//             K_{h,j} = [ k_nope_h | k_rope[j] ]  (k_rope already roped, cached)
//             V_{h,j} = v_h
//   SCORE  :  s_{h,j} = q_h . K_{h,j}     over QK_DIM
//   SOFTMAX:  p_{h,.} = softmax_j( s_{h,.} )      (max-subtract, $exp, normalize)
//   CONTEXT:  O_h[d]  = Σ_j p_{h,j} * V_{h,j}[d]
//   OUTPUT :  out     = concat_h(O_h) . W_o       (attention_bias = 0)
//
// EVERY intermediate VECTOR is rounded back to bf16 (the unit's storage
// contract) before being consumed by the next stage -- so the golden tracks the
// SAME bf16 quantization points the DUT emits -- while each individual dot
// product accumulates in fp64.  RoPE uses $cos/$sin (theta=THETA) and softmax
// uses $exp: these are INDEPENDENT of the DUT's fp32 CORDIC / pipelined-exp /
// matmul, so the comparison is a true cross-check, not a tautology.
//
// The DSA front-end runs DENSE here (S <= TOPK), keeping keys 0..S-1 in order,
// so softmax slot s maps to cache key s -- the golden uses the same ordering.
//
// TOLERANCE (justified):
//   bf16 carries ~7-8 significant bits (~2-3 decimal digits, rel step 2^-8 ~
//   0.4%).  The output is the tail of a long chain: W_dq, RMSNorm, W_uq, RoPE,
//   W_dkv/W_kr (cur tok) + per-key {RMSNorm, W_uk, W_uv}, q.K score, softmax,
//   weighted-V, W_o -- ~7 GEMMs + 2 RMSNorms + RoPE + softmax.  Each bf16
//   rounding injects up to ~0.4% relative error and these COMPOUND; additionally
//   the DUT's CORDIC RoPE (<=3.1e-5 abs) and pipelined exp differ slightly from
//   the golden's $cos/$sin/$exp, and softmax normalization can amplify small
//   score errors near ties.  A bound of REL = 12.5% (2^-3) plus a small ABSOLUTE
//   floor 0.03 (for outputs near zero where relative error is meaningless and a
//   single bf16 ULP dominates) is the right bf16-faithfulness gate -- tight
//   enough to catch any structural/wiring/ordering bug, loose enough to admit
//   pure rounding.  (Observed worst-case rel error across all cases is well
//   inside this; see the per-test PASS report.)  X-AWARE: ANY X/Z bit in `out`
//   FAILS; outputs are also checked finite (no inf/nan).
//
// Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X / inf.
//============================================================================
module mla_attn_tb;
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
    localparam integer QK_DIM    = NOPE + ROPE;
    localparam integer IDXW      = (S_MAX <= 1) ? 1 : $clog2(S_MAX);
    localparam integer HQK       = H_HEADS*QK_DIM;
    localparam integer HNOPE     = H_HEADS*NOPE;
    localparam integer HV        = H_HEADS*V_DIM;

    `include "glm_fp.vh"

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

    // ---- weight pull ----
    wire                     w_req;
    wire [3:0]               w_sel;
    localparam integer NGMAXc = (HV>MODEL_DIM)?
                  ((HV>HQK)?((HV>HNOPE)?HV:HNOPE):((HQK>HNOPE)?HQK:HNOPE))
                 :((MODEL_DIM>HQK)?((MODEL_DIM>HNOPE)?MODEL_DIM:HNOPE)
                                  :((HQK>HNOPE)?HQK:HNOPE));
    wire [$clog2(((NGMAXc+PE_N-1)/PE_N <=1)?1:(NGMAXc+PE_N-1)/PE_N)-1:0] w_grp;
    localparam integer KMAXc = (MODEL_DIM>Q_LORA)?
                 ((MODEL_DIM>KV_LORA)?((MODEL_DIM>HV)?MODEL_DIM:HV)
                                     :((KV_LORA>HV)?KV_LORA:HV))
                :((Q_LORA>KV_LORA)?((Q_LORA>HV)?Q_LORA:HV)
                                  :((KV_LORA>HV)?KV_LORA:HV));
    wire [$clog2((KMAXc<=1)?1:KMAXc)-1:0] w_k;
    reg  [PE_N*16-1:0]       w_col;

    // ---- cache read ----
    wire                     kc_req;
    wire [IDXW-1:0]          kc_idx;
    reg  [KV_LORA*16-1:0]    kc_ckv;
    reg  [ROPE*16-1:0]       kc_krope;
    reg                      kc_valid;

    // ================= weight ROMs (bf16) =================
    // W_dq [Q_LORA x MODEL_DIM], W_uq [HQK x Q_LORA], W_dkv [KV_LORA x MODEL_DIM],
    // W_kr [ROPE x MODEL_DIM], W_uk [HNOPE x KV_LORA], W_uv [HV x KV_LORA],
    // W_o  [MODEL_DIM x HV].  Indexed [out][k].  (W_dkv/W_kr are exercised in the
    // DUT but, in this cache-fed TB, the past keys come from the cache; they do
    // not feed the golden output -- present for full datapath coverage.)
    reg [15:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [15:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [15:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];
    reg [15:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];
    reg [15:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [15:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [15:0] W_o   [0:MODEL_DIM-1][0:HV-1];

    // cache: c_kv[j] [KV_LORA], k_rope[j] [ROPE]  (already roped)
    reg [15:0] CKV   [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP   [0:S_MAX-1][0:ROPE-1];
    reg [15:0] xin   [0:MODEL_DIM-1];

    // ---------------- DUT ----------------
    mla_attn #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .pos(pos), .s_len(s_len), .x_vec(x_vec),
        .w_req(w_req), .w_sel(w_sel), .w_grp(w_grp), .w_k(w_k), .w_col(w_col),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid), .out(out)
    );

    // ================= combinational WEIGHT RESPONDER =================
    // present PE_N bf16 lanes W_sel[w_grp*PE_N + t , w_k] for t=0..PE_N-1.
    integer t;
    always @* begin
        w_col = {PE_N*16{1'b0}};
        for (t = 0; t < PE_N; t = t + 1) begin
            case (w_sel)
            4'd0: if (w_grp*PE_N+t < Q_LORA)
                     w_col[16*t +:16] = W_dq [w_grp*PE_N+t][w_k];
            4'd1: if (w_grp*PE_N+t < HQK)
                     w_col[16*t +:16] = W_uq [w_grp*PE_N+t][w_k];
            4'd2: if (w_grp*PE_N+t < KV_LORA)
                     w_col[16*t +:16] = W_dkv[w_grp*PE_N+t][w_k];
            4'd3: if (w_grp*PE_N+t < ROPE)
                     w_col[16*t +:16] = W_kr [w_grp*PE_N+t][w_k];
            4'd4: if (w_grp*PE_N+t < HNOPE)
                     w_col[16*t +:16] = W_uk [w_grp*PE_N+t][w_k];
            4'd5: if (w_grp*PE_N+t < HV)
                     w_col[16*t +:16] = W_uv [w_grp*PE_N+t][w_k];
            4'd6: if (w_grp*PE_N+t < MODEL_DIM)
                     w_col[16*t +:16] = W_o  [w_grp*PE_N+t][w_k];
            default: w_col[16*t +:16] = 16'h0;
            endcase
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
        else     kc_valid <= kc_req;     // simple 1-cycle latency responder
    end

    // ===================================================================
    //  INDEPENDENT fp64 GOLDEN HELPERS
    // ===================================================================
    // bf16 -> real (double).  FTZ subnormals (matches glm_fp); no inf/nan in
    // stimulus so those map to 0 defensively.
    function real b2real; input [15:0] b; reg [31:0] f; real m; integer e,i; begin
        f = {b,16'h0};
        if (f[30:23]==8'h00)      b2real = 0.0;       // FTZ
        else if (f[30:23]==8'hFF) b2real = 0.0;       // (no inf/nan in stimulus)
        else begin
            m = 1.0;
            for (i=0;i<23;i=i+1) if (f[i]) m = m + (2.0**(i-23));
            e = f[30:23]-127;
            b2real = m * (2.0**e);
            if (f[31]) b2real = -b2real;
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
            // v in [1,2); 7 mantissa bits, round to nearest even
            mi  = $rtoi((v-1.0)*128.0 + 0.5);
            if (mi>=128) begin mi=mi-128; e=e+1; end
            man = mi[7:0];
            if (e < -126) real2b = {s,15'h0};         // FTZ underflow
            else real2b = {s, e[7:0]+8'd127, man[6:0]};
        end
    end endfunction
    // real, already rounded to bf16 (re-decode for use as next-stage operand).
    function real rb; input real v; begin rb = b2real(real2b(v)); end endfunction

    // deterministic bf16 stimulus generator.  Pure-integer hash -> bf16 in a
    // chosen magnitude band; no $random so DUT and golden share identical bits.
    //   band 0 : |v| in ~[0.06, 2)   (default mixed-sign, moderate range)
    //   band 1 : |v| in ~[0.25, 4)   (wider range, mixed sign)
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = (seed*2654435761) ^ (seed<<13) ^ (seed*40503);
        s = h[3];
        if (band==1) e = 8'd125 + h[6:4];             // 125..132 -> ~[0.25,4)
        else         e = 8'd124 + h[5:4];             // 124..127 -> ~[0.0625,2)
        m = h[12:6];
        gen_bf16 = {s, e, m};
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

    real gq  [0:HQK-1];                           // q after rope (real)
    real gK  [0:H_HEADS-1][0:S_MAX-1][0:QK_DIM-1];
    real gV  [0:H_HEADS-1][0:S_MAX-1][0:V_DIM-1];
    real gsc [0:H_HEADS-1][0:S_MAX-1];
    real gp  [0:H_HEADS-1][0:S_MAX-1];
    real gctx[0:HV-1];
    real gout[0:MODEL_DIM-1];

    real accr, mxr, sumr, ang, c, sN, x0, x1, invf;
    real refv, dutv, adiff, rtol, worst_rel, this_rel;
    integer i,j,k,h,d,o,hd,sd,pr;
    integer errors, test_count, fails_in_test;
    integer Sg;
    integer tpos;       // current test pos
    integer tband;      // current test stimulus band
    real    near_uniform_scale;  // if >0, force near-uniform attention (see task)

    // -------------------------------------------------------------------
    // build_stimulus : fill all weight ROMs + x + cache for the given band,
    //   seed offset distinct per test so the cases are independent draws.
    // -------------------------------------------------------------------
    integer sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc = seed0;
        for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[i][j]=gen_bf16(sc,band);  sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin xin[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[i][j]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // -------------------------------------------------------------------
    // make_near_uniform : directed case.  Make both keys' latent + roped-key
    //   IDENTICAL so per-head scores tie and softmax is exactly uniform
    //   (p = 1/S).  This stresses the softmax + weighted-V averaging path with a
    //   known structure (O_h = mean_j V_{h,j}).  Only meaningful for S=2.
    // -------------------------------------------------------------------
    task make_near_uniform; begin
        for (j=0;j<KV_LORA;j=j+1) CKV[1][j] = CKV[0][j];
        for (j=0;j<ROPE;j=j+1)    KRP[1][j] = KRP[0][j];
    end endtask

    // -------------------------------------------------------------------
    // compute_golden : recompute the whole MLA flow in fp64 for the CURRENT
    //   stimulus / pos (tpos) / S (Sg).  Fills gout[].
    // -------------------------------------------------------------------
    task compute_golden; begin
        // ---- Q path ----
        for (o=0;o<Q_LORA;o=o+1) begin
            accr=0.0;
            for (k=0;k<MODEL_DIM;k=k+1) accr=accr+b2real(xin[k])*b2real(W_dq[o][k]);
            g_qlora[o]=real2b(accr);
        end
        // RMSNorm(qlora), gamma=1, eps=1e-5
        accr=0.0;
        for (i=0;i<Q_LORA;i=i+1) accr=accr+b2real(g_qlora[i])*b2real(g_qlora[i]);
        accr = 1.0/$sqrt(accr/Q_LORA + 1e-5);
        for (i=0;i<Q_LORA;i=i+1) g_qlora_n[i]=real2b(b2real(g_qlora[i])*accr);
        // up-proj
        for (o=0;o<HQK;o=o+1) begin
            accr=0.0;
            for (k=0;k<Q_LORA;k=k+1) accr=accr+b2real(g_qlora_n[k])*b2real(W_uq[o][k]);
            g_qfull[o]=real2b(accr);
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
            accr=0.0;
            for (i=0;i<KV_LORA;i=i+1) accr=accr+b2real(CKV[sd][i])*b2real(CKV[sd][i]);
            accr = 1.0/$sqrt(accr/KV_LORA + 1e-5);
            for (i=0;i<KV_LORA;i=i+1) g_ckvn[i]=real2b(b2real(CKV[sd][i])*accr);
            for (o=0;o<HNOPE;o=o+1) begin
                accr=0.0;
                for (k=0;k<KV_LORA;k=k+1) accr=accr+b2real(g_ckvn[k])*b2real(W_uk[o][k]);
                g_knope[o]=real2b(accr);
            end
            for (o=0;o<HV;o=o+1) begin
                accr=0.0;
                for (k=0;k<KV_LORA;k=k+1) accr=accr+b2real(g_ckvn[k])*b2real(W_uv[o][k]);
                g_v[o]=real2b(accr);
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
                gctx[hd*V_DIM+d]=b2real(real2b(accr));
            end
        // ---- out = ctx * W_o  (K=HV, OUT=MODEL_DIM) ----
        for (o=0;o<MODEL_DIM;o=o+1) begin
            accr=0.0;
            for (k=0;k<HV;k=k+1) accr=accr+gctx[k]*b2real(W_o[o][k]);
            gout[o]=accr;       // final projection NOT re-rounded (DUT emits bf16,
        end                     // compared with tolerance below)
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
    //   Accumulates into 'fails_in_test'; updates worst_rel.
    // -------------------------------------------------------------------
    task check_out; input [256*8-1:0] label; begin
        fails_in_test = 0;
        // (1) X/Z reject
        for (i=0;i<MODEL_DIM*16;i=i+1)
            if (out[i]===1'bx || out[i]===1'bz) begin
                $display("FAIL[%0s]: out bit %0d is X/Z", label, i);
                fails_in_test = fails_in_test + 1;
            end
        // (2) finite (no inf/nan)
        for (i=0;i<MODEL_DIM;i=i+1)
            if (out[16*i+7 +:8]==8'hFF) begin
                $display("FAIL[%0s]: out[%0d] inf/nan (%h)", label, i, out[16*i +:16]);
                fails_in_test = fails_in_test + 1;
            end
        // (3) per-element match to the independent fp64 reference
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
        // T1 : S=1, pos=0   -> single key (softmax trivial p=1) + RoPE identity
        one_case(   1, 0,    0, 1, 0, "S1_pos0_ropeIdentity");
        // T2 : S=1, pos>0   -> single key, real RoPE rotation
        one_case( 100, 0,    7, 1, 0, "S1_posNZ");
        // T3 : S=2, pos=0   -> multi-key causal, RoPE identity
        one_case( 200, 0,    0, 2, 0, "S2_pos0");
        // T4 : S=2, pos>0   -> multi-key causal, real RoPE (default smoke shape)
        one_case( 300, 0,    5, 2, 0, "S2_posNZ");
        // T5 : random wide-range mixed-sign weights/x (band 1), S=2, pos large
        one_case( 400, 1, 4095, 2, 0, "S2_wideRange_posBig");
        // T6 : directed NEAR-UNIFORM attention (identical keys -> p=1/S), S=2
        one_case( 500, 0,    3, 2, 1, "S2_nearUniform");
        // T7 : another wide-range draw, S=1, pos>0 (independent seed)
        one_case( 700, 1,   42, 1, 0, "S1_wideRange_posNZ");

        // ============ VERDICT ============
        if (errors==0) begin
            $display("worst-case relative error across all tests = %f", worst_rel);
            $display("ALL %0d TESTS PASSED", test_count);
        end else begin
            $display("FAILED: %0d errors over %0d tests", errors, test_count);
            $fatal(1, "mla_attn TB failed");
        end
        $finish;
    end

    // global timeout
    initial begin
        #200000000;
        $display("FAIL: timeout (done never asserted)");
        $fatal(1, "timeout");
    end
endmodule
