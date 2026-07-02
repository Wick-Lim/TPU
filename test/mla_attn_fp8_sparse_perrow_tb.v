`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// mla_attn_fp8_sparse_perrow_tb.v
//   PE_M batching oracle for mla_attn_fp8 that combines SPARSE DSA selection
//   (S_MAX > TOPK) with PER-ROW query position / causal extent.
//----------------------------------------------------------------------------
// WHAT THIS PROVES (a DUT-vs-DUT, same-arithmetic, EXACT === oracle):
//   Instantiate ONE batched DUT (PE_M=B=3, PER_ROW_POS=1, PER_ROW_SLEN=1) and,
//   for the SAME shared weight ROMs / block scales / KV cache, ONE PE_M=1
//   reference re-run once per row on THAT row's own (x_r, pos_r, s_len_r).  For
//   each enabled case we assert TWO things:
//     (1) BIT-EXACT per-row equivalence -- batched row r's MODEL_DIM bf16 `out`
//         === the PE_M=1 reference on row r's own inputs (X/Z-aware, exact ===).
//     (2) FETCH SHARING (one fetch per DISTINCT key/weight, NOT B fetches) --
//         the batched DUT asserts w_req / kc_req on EXACTLY the same number of
//         cycles as ONE PE_M=1 reference run that covers the shared max extent
//         (s_reg = max_r s_len_r).  B independent standalone runs would fetch
//         those weights/keys B times; the batch shares the one fetch.
//
// SCOPE (the CURRENT row-0-shared DSA selection):  batched-row == standalone-row
//   holds EXACTLY when the shared, row-0-driven top-K selection equals every
//   row's own selection.  That is true for:
//     * DENSE fallback (max extent <= TOPK): dsa_indexer keeps keys 0..S-1 in
//       order, q-independent -> sel_list[s]=s for every row, and the per-row
//       causal-extent mask (via sel_list[sf_feed_i], the B1 fix) trims each row
//       to its own extent exactly.  (distinct x AND distinct per-row s_len OK.)
//     * ALL-EQUAL-x rows (any S, incl. SPARSE S>TOPK): every row's q is identical
//       so the shared row-0 selection == each row's own selection; with a SHARED
//       extent the batched rows are bit-identical to the single-row runs.
//   THE PENDING-B6 CASE -- DISTINCT-x SPARSE (S>TOPK):  the batch forces row-0's
//   DSA selection on every row, so once the indexer's key set becomes q-DEPENDENT
//   and thus per-row, batched row r would diverge from standalone row r.  That
//   per-row DSA selection is task B6.  This case lives here behind the
//   `SPARSE_XFAIL` guard (DEFAULT OFF) as the forward-looking B6 output oracle.
//   NOTE ON THIS SLICE:  mla_attn_fp8 currently drives the indexer with ZERO key
//   index vectors (S_DSA: dsa_kidx<=0), so every key scores 0 and topk_select
//   keeps keys 0..min(S,TOPK)-1 by its lower-index tie-break -- q-INDEPENDENTLY.
//   Hence at THIS slice even the distinct-x sparse selection is identical across
//   rows and the guarded compare also passes today; it is guarded (not asserted
//   by default) because that q-independence is NOT guaranteed once B6 lands
//   per-row/q-dependent selection.  The FETCH-SHARING assertion is data-
//   independent (given sel_cnt) and is checked UNCONDITIONALLY for these cases.
//
//   S_MAX=8, TOPK=4  ->  S in 1..4 is dense, S in 5..8 is sparse.
//
//   Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X.
//============================================================================
module mla_attn_fp8_sparse_perrow_tb;
    // ---- small slice: S_MAX(8) > TOPK(4) so sparse selection is exercised ----
    localparam integer MODEL_DIM = 16;
    localparam integer H_HEADS   = 2;
    localparam integer NOPE      = 4;
    localparam integer ROPE      = 4;
    localparam integer V_DIM     = 4;
    localparam integer Q_LORA    = 8;
    localparam integer KV_LORA   = 8;
    localparam integer S_MAX     = 8;
    localparam integer TOPK      = 4;
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

    localparam integer PE_M  = 3;

    // DSA_REAL_IDX: under -DSPARSE_XFAIL drive the indexer with REAL, query-dependent
    //   key index vectors (c_kv[j][0:NOPE]) so per-row top-K selection genuinely
    //   depends on each row's query -> the distinct-x SPARSE rows diverge (the B6
    //   divergent path becomes LIVE and OBSERVABLE).  In the default build it stays 0
    //   (zero index vectors, q-independent) so every regression folds byte-identically.
    //
    // SWIN_TB: the B7 attention scratch window.  Its committed default is TOPK, whose
    //   invariant "union u_cnt <= SWIN" holds ONLY when every row selects the SAME
    //   top-K set (q-independent slice: u_cnt == TOPK).  Once selection is genuinely
    //   per-row (DSA_REAL_IDX=1) the rows' union can hold up to min(PE_M*TOPK, S_MAX)
    //   DISTINCT keys -- MORE than TOPK -- so the union-slot scratch must be sized for
    //   that worst case, else the union slot index wraps and corrupts.  So under the
    //   divergent build we size SWIN = min(PE_M*TOPK, S_MAX); the default build keeps
    //   SWIN = TOPK (byte-identical to the committed regression).
`ifdef SPARSE_XFAIL
    localparam integer DSA_REAL_IDX = 1;
    localparam integer SWIN_TB = (PE_M*TOPK < S_MAX) ? PE_M*TOPK : S_MAX;
`else
    localparam integer DSA_REAL_IDX = 0;
    localparam integer SWIN_TB = TOPK;
`endif

    `include "glm_fp.vh"
    `include "fp8_e4m3.vh"

    // ---- clock / reset ----
    reg clk=1'b0, rst=1'b1;
    always #5 clk = ~clk;

    // ================= shared stimulus ROMs (FP8 E4M3 codes + scales) =========
    reg [7:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];
    reg [7:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [7:0] W_o   [0:MODEL_DIM-1][0:HV-1];
    reg [15:0] S_dq, S_uq, S_dkv, S_kr, S_uk, S_uv, S_o;
    reg [15:0] CKV   [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP   [0:S_MAX-1][0:ROPE-1];
    reg [15:0] xr    [0:PE_M-1][0:MODEL_DIM-1];   // per-row token activations

    // deterministic stimulus generators (same hashing style as committed TBs).
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = (seed*2654435761) ^ (seed<<13) ^ (seed*40503);
        s = h[3];
        if (band==1) e = 8'd125 + h[6:4];
        else         e = 8'd124 + h[5:4];
        m = h[12:6];
        gen_bf16 = {s, e, m};
    end endfunction
    function [7:0] gen_fp8; input integer seed; reg sg; reg [3:0] e; reg [2:0] m;
        integer h; begin
        h = (seed*2246822519) ^ (seed<<11) ^ (seed*3266489917);
        sg = h[2];
        e  = 4'd5 + h[4:3];
        m  = h[7:5];
        gen_fp8 = {sg, e, m};
    end endfunction

    integer ii,jj,kk,sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc = seed0;
        for (ii=0;ii<Q_LORA;ii=ii+1) for (jj=0;jj<MODEL_DIM;jj=jj+1) begin W_dq[ii][jj]=gen_fp8(sc); sc=sc+1; end
        for (ii=0;ii<HQK;ii=ii+1)    for (jj=0;jj<Q_LORA;jj=jj+1)    begin W_uq[ii][jj]=gen_fp8(sc); sc=sc+1; end
        for (ii=0;ii<KV_LORA;ii=ii+1)for (jj=0;jj<MODEL_DIM;jj=jj+1) begin W_dkv[ii][jj]=gen_fp8(sc); sc=sc+1; end
        for (ii=0;ii<ROPE;ii=ii+1)   for (jj=0;jj<MODEL_DIM;jj=jj+1) begin W_kr[ii][jj]=gen_fp8(sc); sc=sc+1; end
        for (ii=0;ii<HNOPE;ii=ii+1)  for (jj=0;jj<KV_LORA;jj=jj+1)   begin W_uk[ii][jj]=gen_fp8(sc); sc=sc+1; end
        for (ii=0;ii<HV;ii=ii+1)     for (jj=0;jj<KV_LORA;jj=jj+1)   begin W_uv[ii][jj]=gen_fp8(sc); sc=sc+1; end
        for (ii=0;ii<MODEL_DIM;ii=ii+1)for (jj=0;jj<HV;jj=jj+1)      begin W_o[ii][jj]=gen_fp8(sc);  sc=sc+1; end
        S_dq=16'h3F80; S_uq=16'h3F00; S_dkv=16'h3F80; S_kr=16'h3F80;
        S_uk=16'h4000; S_uv=16'h3F00; S_o=16'h3F80;
        // DISTINCT per-row token activations (rows differ in x by default).
        for (kk=0;kk<PE_M;kk=kk+1)
            for (ii=0;ii<MODEL_DIM;ii=ii+1) begin xr[kk][ii]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<KV_LORA;jj=jj+1) begin CKV[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<ROPE;jj=jj+1)    begin KRP[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // force ALL rows to share row-0's x (all-equal-x regime).
    task equalize_x; integer r,c; begin
        for (r=1;r<PE_M;r=r+1) for (c=0;c<MODEL_DIM;c=c+1) xr[r][c]=xr[0][c];
    end endtask

    // ===========================================================================
    //  BATCHED DUT UNDER TEST : PE_M=3, per-row pos + per-row causal extent.
    // ===========================================================================
    reg                      d_start;
    wire                     d_busy, d_done;
    reg  [POSW-1:0]          d_pos;
    reg  [POSW*PE_M-1:0]     d_pos_vec;
    reg  [IDXW:0]            d_slen;
    reg  [(IDXW+1)*PE_M-1:0] d_slen_vec;
    reg  [MODEL_DIM*16*PE_M-1:0] d_xvec;
    wire [MODEL_DIM*16*PE_M-1:0] d_out;
    wire                     d_wreq; wire [3:0] d_wsel;
    wire [GRPW-1:0]          d_wgrp; wire [KCW-1:0] d_wk;
    reg  [PE_N*8-1:0]        d_wcol; reg [16*PE_N*NB-1:0] d_wscale;
    wire                     d_kcreq; wire [IDXW-1:0] d_kcidx;
    reg  [KV_LORA*16-1:0]    d_kcckv; reg [ROPE*16-1:0] d_kckrope; reg d_kcvalid;

    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(PE_M), .SWIN(SWIN_TB),
               .PER_ROW_POS(1), .PER_ROW_SLEN(1), .DSA_REAL_IDX(DSA_REAL_IDX)) dutM (
        .clk(clk), .rst(rst), .start(d_start), .busy(d_busy), .done(d_done),
        .pos(d_pos), .pos_vec(d_pos_vec), .s_len(d_slen), .s_len_vec(d_slen_vec),
        .x_vec(d_xvec),
        .w_req(d_wreq), .w_sel(d_wsel), .w_grp(d_wgrp), .w_k(d_wk),
        .w_col(d_wcol), .w_scale(d_wscale),
        .kc_req(d_kcreq), .kc_idx(d_kcidx), .kc_ckv(d_kcckv),
        .kc_krope(d_kckrope), .kc_valid(d_kcvalid), .out(d_out)
    );

    // ===========================================================================
    //  PE_M=1 REFERENCE  (single-token; re-run once per row at THAT row's inputs)
    // ===========================================================================
    reg                      r1_start;
    wire                     r1_busy, r1_done;
    reg  [POSW-1:0]          r1_pos;
    reg  [POSW*1-1:0]        r1_pos_vec;
    reg  [IDXW:0]            r1_slen;
    reg  [(IDXW+1)*1-1:0]    r1_slen_vec;
    reg  [MODEL_DIM*16-1:0]  r1_xvec;
    wire [MODEL_DIM*16-1:0]  r1_out;
    wire                     r1_wreq; wire [3:0] r1_wsel;
    wire [GRPW-1:0]          r1_wgrp; wire [KCW-1:0] r1_wk;
    reg  [PE_N*8-1:0]        r1_wcol; reg [16*PE_N*NB-1:0] r1_wscale;
    wire                     r1_kcreq; wire [IDXW-1:0] r1_kcidx;
    reg  [KV_LORA*16-1:0]    r1_kcckv; reg [ROPE*16-1:0] r1_kckrope; reg r1_kcvalid;

    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(1), .SWIN(SWIN_TB), .DSA_REAL_IDX(DSA_REAL_IDX)) ref1 (
        .clk(clk), .rst(rst), .start(r1_start), .busy(r1_busy), .done(r1_done),
        .pos(r1_pos), .pos_vec(r1_pos_vec), .s_len(r1_slen), .s_len_vec(r1_slen_vec),
        .x_vec(r1_xvec),
        .w_req(r1_wreq), .w_sel(r1_wsel), .w_grp(r1_wgrp), .w_k(r1_wk),
        .w_col(r1_wcol), .w_scale(r1_wscale),
        .kc_req(r1_kcreq), .kc_idx(r1_kcidx), .kc_ckv(r1_kcckv),
        .kc_krope(r1_kckrope), .kc_valid(r1_kcvalid), .out(r1_out)
    );

    // ================= combinational WEIGHT responders (one per instance) ======
    integer tdM, tr1; reg [15:0] ssc_dM, ssc_r1;
    always @* begin
        d_wcol={PE_N*8{1'b0}}; d_wscale={16*PE_N*NB{1'b0}};
        case (d_wsel)
            4'd0: ssc_dM=S_dq;  4'd1: ssc_dM=S_uq;  4'd2: ssc_dM=S_dkv; 4'd3: ssc_dM=S_kr;
            4'd4: ssc_dM=S_uk;  4'd5: ssc_dM=S_uv;  4'd6: ssc_dM=S_o;   default: ssc_dM=16'h3F80;
        endcase
        for (tdM=0;tdM<PE_N;tdM=tdM+1) begin
            case (d_wsel)
            4'd0: if (d_wgrp*PE_N+tdM < Q_LORA)   d_wcol[8*tdM +:8] = W_dq [d_wgrp*PE_N+tdM][d_wk];
            4'd1: if (d_wgrp*PE_N+tdM < HQK)      d_wcol[8*tdM +:8] = W_uq [d_wgrp*PE_N+tdM][d_wk];
            4'd2: if (d_wgrp*PE_N+tdM < KV_LORA)  d_wcol[8*tdM +:8] = W_dkv[d_wgrp*PE_N+tdM][d_wk];
            4'd3: if (d_wgrp*PE_N+tdM < ROPE)     d_wcol[8*tdM +:8] = W_kr [d_wgrp*PE_N+tdM][d_wk];
            4'd4: if (d_wgrp*PE_N+tdM < HNOPE)    d_wcol[8*tdM +:8] = W_uk [d_wgrp*PE_N+tdM][d_wk];
            4'd5: if (d_wgrp*PE_N+tdM < HV)       d_wcol[8*tdM +:8] = W_uv [d_wgrp*PE_N+tdM][d_wk];
            4'd6: if (d_wgrp*PE_N+tdM < MODEL_DIM)d_wcol[8*tdM +:8] = W_o  [d_wgrp*PE_N+tdM][d_wk];
            default: d_wcol[8*tdM +:8] = 8'h38;
            endcase
            d_wscale[16*tdM +: 16] = ssc_dM;
        end
    end
    always @* begin
        r1_wcol={PE_N*8{1'b0}}; r1_wscale={16*PE_N*NB{1'b0}};
        case (r1_wsel)
            4'd0: ssc_r1=S_dq;  4'd1: ssc_r1=S_uq;  4'd2: ssc_r1=S_dkv; 4'd3: ssc_r1=S_kr;
            4'd4: ssc_r1=S_uk;  4'd5: ssc_r1=S_uv;  4'd6: ssc_r1=S_o;   default: ssc_r1=16'h3F80;
        endcase
        for (tr1=0;tr1<PE_N;tr1=tr1+1) begin
            case (r1_wsel)
            4'd0: if (r1_wgrp*PE_N+tr1 < Q_LORA)   r1_wcol[8*tr1 +:8] = W_dq [r1_wgrp*PE_N+tr1][r1_wk];
            4'd1: if (r1_wgrp*PE_N+tr1 < HQK)      r1_wcol[8*tr1 +:8] = W_uq [r1_wgrp*PE_N+tr1][r1_wk];
            4'd2: if (r1_wgrp*PE_N+tr1 < KV_LORA)  r1_wcol[8*tr1 +:8] = W_dkv[r1_wgrp*PE_N+tr1][r1_wk];
            4'd3: if (r1_wgrp*PE_N+tr1 < ROPE)     r1_wcol[8*tr1 +:8] = W_kr [r1_wgrp*PE_N+tr1][r1_wk];
            4'd4: if (r1_wgrp*PE_N+tr1 < HNOPE)    r1_wcol[8*tr1 +:8] = W_uk [r1_wgrp*PE_N+tr1][r1_wk];
            4'd5: if (r1_wgrp*PE_N+tr1 < HV)       r1_wcol[8*tr1 +:8] = W_uv [r1_wgrp*PE_N+tr1][r1_wk];
            4'd6: if (r1_wgrp*PE_N+tr1 < MODEL_DIM)r1_wcol[8*tr1 +:8] = W_o  [r1_wgrp*PE_N+tr1][r1_wk];
            default: r1_wcol[8*tr1 +:8] = 8'h38;
            endcase
            r1_wscale[16*tr1 +: 16] = ssc_r1;
        end
    end

    // ---- combinational cache responders + registered valid (1-cycle latency) ----
    integer cdM, cd1;
    always @* begin
        d_kcckv = {KV_LORA*16{1'b0}}; d_kckrope = {ROPE*16{1'b0}};
        for (cdM=0;cdM<KV_LORA;cdM=cdM+1) d_kcckv[16*cdM +:16]  = CKV[d_kcidx][cdM];
        for (cdM=0;cdM<ROPE;cdM=cdM+1)    d_kckrope[16*cdM +:16] = KRP[d_kcidx][cdM];
    end
    always @* begin
        r1_kcckv = {KV_LORA*16{1'b0}}; r1_kckrope = {ROPE*16{1'b0}};
        for (cd1=0;cd1<KV_LORA;cd1=cd1+1) r1_kcckv[16*cd1 +:16]  = CKV[r1_kcidx][cd1];
        for (cd1=0;cd1<ROPE;cd1=cd1+1)    r1_kckrope[16*cd1 +:16] = KRP[r1_kcidx][cd1];
    end
    always @(posedge clk) begin
        if (rst) begin d_kcvalid<=1'b0; r1_kcvalid<=1'b0; end
        else     begin d_kcvalid<=d_kcreq; r1_kcvalid<=r1_kcreq; end
    end

    // ================= fetch-beat counters (w_req / kc_req cycles) =============
    //   reset on each run's start pulse; count every cycle the request is high.
    reg [31:0] bw_cnt, bkc_cnt, rw_cnt, rkc_cnt;
    always @(posedge clk) begin
        if (d_start)  begin bw_cnt<=0;  bkc_cnt<=0;  end
        else          begin if (d_wreq)  bw_cnt <=bw_cnt +1'b1; if (d_kcreq)  bkc_cnt <=bkc_cnt +1'b1; end
        if (r1_start) begin rw_cnt<=0;  rkc_cnt<=0;  end
        else          begin if (r1_wreq) rw_cnt <=rw_cnt +1'b1; if (r1_kcreq) rkc_cnt <=rkc_cnt +1'b1; end
    end

    // ===========================================================================
    //  DRIVERS
    // ===========================================================================
    integer i;
    // batched PE_M=3 run: shared pos p; per-row extents s0,s1,s2; rows xr[0..2].
    reg [31:0] bw_run, bkc_run;   // captured batched fetch-beat counts
    task run_dutM; input integer p; input integer s0; input integer s1; input integer s2; begin
        d_pos     = p[POSW-1:0];
        d_pos_vec = {PE_M{p[POSW-1:0]}};                 // shared query position
        d_slen    = s0[IDXW:0];                          // row 0 uses scalar s_len
        d_slen_vec[(IDXW+1)*0 +: (IDXW+1)] = s0[IDXW:0]; // row-0 slice unused (row0 = scalar)
        d_slen_vec[(IDXW+1)*1 +: (IDXW+1)] = s1[IDXW:0];
        d_slen_vec[(IDXW+1)*2 +: (IDXW+1)] = s2[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) begin
            d_xvec[16*(MODEL_DIM*0 + i) +:16] = xr[0][i];
            d_xvec[16*(MODEL_DIM*1 + i) +:16] = xr[1][i];
            d_xvec[16*(MODEL_DIM*2 + i) +:16] = xr[2][i];
        end
        @(negedge clk); d_start=1'b1; @(negedge clk); d_start=1'b0;
        wait (d_done==1'b1); @(negedge clk);
        bw_run = bw_cnt; bkc_run = bkc_cnt;              // stable until next d_start
    end endtask

    // PE_M=1 reference on one row's (xr[row], pos p, s_len s); capture ro[] + counts.
    reg [15:0] ro [0:MODEL_DIM-1];
    reg [31:0] rw_run, rkc_run;
    task run_ref; input integer row; input integer p; input integer s; begin
        r1_pos     = p[POSW-1:0];
        r1_pos_vec = p[POSW-1:0];                        // unused at PE_M=1, drive sanely
        r1_slen    = s[IDXW:0];
        r1_slen_vec= s[IDXW:0];                          // unused at PE_M=1, drive sanely
        for (i=0;i<MODEL_DIM;i=i+1)
            r1_xvec[16*i +:16] = xr[row][i];
        @(negedge clk); r1_start=1'b1; @(negedge clk); r1_start=1'b0;
        wait (r1_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) ro[i] = r1_out[16*i +:16];
        rw_run = rw_cnt; rkc_run = rkc_cnt;
    end endtask

    // ===========================================================================
    //  CHECKS (exact, X-aware)
    // ===========================================================================
    integer errors, test_count, fails;
    // batched row `row` === last-captured reference ro[].
    task check_row; input integer row; input [256*8-1:0] label; begin
        fails=0;
        for (i=0;i<MODEL_DIM;i=i+1) begin
            if (^d_out[16*(MODEL_DIM*row+i) +:16] === 1'bx) begin
                $display("FAIL[%0s] row%0d out[%0d] X/Z", label, row, i);
                fails=fails+1;
            end else if (d_out[16*(MODEL_DIM*row+i) +:16] !== ro[i]) begin
                $display("FAIL[%0s] row%0d out[%0d] dut=%h ref=%h", label, row, i,
                         d_out[16*(MODEL_DIM*row+i) +:16], ro[i]);
                fails=fails+1;
            end
        end
        test_count=test_count+1;
        if (fails==0) $display("  PASS[%0s] row%0d (exact match)", label, row);
        else errors=errors+fails;
    end endtask

    // FETCH SHARING: batched fetch-beat counts == ONE reference run over the shared
    //   max extent (rw_run/rkc_run must be captured by a run_ref at s = s_reg first).
    task check_fetch_share; input [256*8-1:0] label; begin
        test_count=test_count+1;
        if (bw_run !== rw_run) begin
            $display("FAIL[%0s] w_req beats: batch=%0d != 1-ref=%0d (weight fetch not shared)",
                     label, bw_run, rw_run);
            errors=errors+1;
        end else if (bkc_run !== rkc_run) begin
            $display("FAIL[%0s] kc_req beats: batch=%0d != 1-ref=%0d (key fetch not shared)",
                     label, bkc_run, rkc_run);
            errors=errors+1;
        end else
            $display("  PASS[%0s] fetch-share: w=%0d kc=%0d (one fetch per distinct key/weight)",
                     label, bw_run, bkc_run);
    end endtask

    function integer imax3; input integer a; input integer b; input integer c;
        integer mx; begin mx=a; if (b>mx) mx=b; if (c>mx) mx=c; imax3=mx; end
    endfunction

    // ---------------------------------------------------------------------------
    //  DIVERGENCE PROOF (the whole deliverable): with real query-dependent index
    //    vectors the batched DUT's PER-ROW DSA selection sel_list_r[r] must actually
    //    DIFFER across rows whose queries differ.  Read the DUT's internal per-row
    //    selection lists (hierarchical), print them, and assert at least one row's
    //    (count,list) differs from row 0 -- i.e. the divergent B6/B7 path is LIVE.
    // ---------------------------------------------------------------------------
    integer dvr, dvs; reg dv_diff;
    task check_divergence; input [256*8-1:0] label; begin
        test_count = test_count + 1;
        for (dvr=0; dvr<PE_M; dvr=dvr+1) begin
            $write("    [%0s] row%0d sel_cnt=%0d sel_list=", label, dvr, dutM.sel_cnt_r[dvr]);
            for (dvs=0; dvs<TOPK; dvs=dvs+1) $write(" %0d", dutM.sel_list_r[dvr][dvs]);
            $write("\n");
        end
        dv_diff = 1'b0;
        for (dvr=1; dvr<PE_M; dvr=dvr+1) begin
            if (dutM.sel_cnt_r[dvr] !== dutM.sel_cnt_r[0]) dv_diff = 1'b1;
            for (dvs=0; dvs<TOPK; dvs=dvs+1)
                if (dutM.sel_list_r[dvr][dvs] !== dutM.sel_list_r[0][dvs]) dv_diff = 1'b1;
        end
        if (dv_diff)
            $display("  PASS[%0s] per-row DSA selection is q-DEPENDENT (rows select DIFFERENT key sets)", label);
        else begin
            $display("FAIL[%0s] per-row DSA selection IDENTICAL across distinct-x rows (not q-dependent)", label);
            errors = errors + 1;
        end
    end endtask

    // ---------------------------------------------------------------------------
    //  FETCH-SHARING under genuine per-row divergence.  Once selection is q-dependent
    //    the batch selects the UNION of the rows' key sets, so its fetch count no
    //    longer equals ONE single-row run (the pre-B6 exact check).  The correct,
    //    divergence-proof property is that the batch fetches each distinct union
    //    key/weight ONCE -- i.e. STRICTLY FEWER beats than PE_M INDEPENDENT single-row
    //    decodes (fetch shared, not replicated), yet AT LEAST one row's demand (the
    //    union covers every row).  Bounds: max_single <= batch < sum_of_rows.
    // ---------------------------------------------------------------------------
    reg [31:0] sxw [0:PE_M-1];
    reg [31:0] sxkc[0:PE_M-1];
    task check_fetch_share_union; input [256*8-1:0] label; begin
        test_count = test_count + 1;
        if (!(bw_run < (sxw[0]+sxw[1]+sxw[2]) && bw_run >= imax3(sxw[0],sxw[1],sxw[2]))) begin
            $display("FAIL[%0s] w_req beats: batch=%0d not in [%0d,%0d) (weight fetch not shared)",
                     label, bw_run, imax3(sxw[0],sxw[1],sxw[2]), sxw[0]+sxw[1]+sxw[2]);
            errors = errors + 1;
        end else if (!(bkc_run < (sxkc[0]+sxkc[1]+sxkc[2]) && bkc_run >= imax3(sxkc[0],sxkc[1],sxkc[2]))) begin
            $display("FAIL[%0s] kc_req beats: batch=%0d not in [%0d,%0d) (key fetch not shared)",
                     label, bkc_run, imax3(sxkc[0],sxkc[1],sxkc[2]), sxkc[0]+sxkc[1]+sxkc[2]);
            errors = errors + 1;
        end else
            $display("  PASS[%0s] fetch-share(union): w=%0d in [%0d,%0d)  kc=%0d in [%0d,%0d)",
                     label, bw_run, imax3(sxw[0],sxw[1],sxw[2]), sxw[0]+sxw[1]+sxw[2],
                     bkc_run, imax3(sxkc[0],sxkc[1],sxkc[2]), sxkc[0]+sxkc[1]+sxkc[2]);
    end endtask

    // ---------------------------------------------------------------------------
    //  ENABLED case: batched rows MUST equal the per-row PE_M=1 references.
    //    (dense fallback and/or all-equal-x -- the shared row-0 selection == each
    //     row's own selection, so equivalence is exact.)
    // ---------------------------------------------------------------------------
    integer smax_c;
    task case_match; input integer seed0; input integer band; input integer eqx;
                     input integer p; input integer s0; input integer s1; input integer s2;
                     input [256*8-1:0] label; begin
        build_stimulus(seed0, band);
        if (eqx) equalize_x;
        smax_c = imax3(s0,s1,s2);
        run_dutM(p, s0, s1, s2);
        // (1) per-row bit-exact equivalence
        run_ref(0, p, s0); check_row(0, label);
        run_ref(1, p, s1); check_row(1, label);
        run_ref(2, p, s2); check_row(2, label);
        // (2) fetch sharing vs ONE reference run covering the shared max extent
        run_ref(0, p, smax_c);
        check_fetch_share(label);
        $display("    (%0s: pos=%0d S=(%0d,%0d,%0d) eqx=%0d band=%0d %s)",
                 label, p, s0, s1, s2, eqx, band,
                 (smax_c<=TOPK)?"DENSE":"SPARSE");
    end endtask

    // ---------------------------------------------------------------------------
    //  DISTINCT-x SPARSE case (the forward-looking B6 output oracle).  Fetch-
    //    sharing is checked unconditionally; the per-row OUTPUT equivalence is
    //    checked ONLY under `SPARSE_XFAIL.  Rationale: the batch forces row-0's
    //    DSA selection on every row, so once selection becomes q-dependent/per-row
    //    (task B6) batched row r may diverge from standalone row r.  On THIS slice
    //    the indexer is fed zero key vectors (q-independent selection) so the
    //    compare also passes today, but it is guarded because that is not
    //    guaranteed to survive B6.
    // ---------------------------------------------------------------------------
    task case_sparse_xfail; input integer seed0; input integer band;
                            input integer p; input integer s0; input integer s1; input integer s2;
                            input [256*8-1:0] label; begin
        build_stimulus(seed0, band);           // distinct x (NOT equalized)
        smax_c = imax3(s0,s1,s2);
        run_dutM(p, s0, s1, s2);
`ifdef SPARSE_XFAIL
        // (0) PROVE the divergent path is live: per-row selection is q-dependent.
        check_divergence(label);
        // (1) per-row BIT-EXACT equivalence -- batched row r (its OWN q-dependent
        //     selection) === the PE_M=1 reference on row r's own inputs (same module,
        //     same real index vectors, same KV cache -> identical selection & output).
        run_ref(0, p, s0); check_row(0, label); sxw[0]=rw_run; sxkc[0]=rkc_run;
        run_ref(1, p, s1); check_row(1, label); sxw[1]=rw_run; sxkc[1]=rkc_run;
        run_ref(2, p, s2); check_row(2, label); sxw[2]=rw_run; sxkc[2]=rkc_run;
        // (2) fetch-sharing under divergence (union fetched once; bounds vs 3 runs).
        check_fetch_share_union(label);
`else
        $display("  SKIP[%0s] per-row output compare (distinct-x sparse == B6 gap; define SPARSE_XFAIL to enable)", label);
        run_ref(0, p, smax_c);
        check_fetch_share(label);
`endif
        $display("    (%0s: pos=%0d S=(%0d,%0d,%0d) distinct-x SPARSE)", label, p, s0, s1, s2);
    end endtask

    initial begin
        errors=0; test_count=0;
        d_start=1'b0; r1_start=1'b0;
        d_pos=0; d_pos_vec=0; d_slen=0; d_slen_vec=0; d_xvec=0;
        r1_pos=0; r1_pos_vec=0; r1_slen=0; r1_slen_vec=0; r1_xvec=0;
        bw_cnt=0; bkc_cnt=0; rw_cnt=0; rkc_cnt=0;
        @(negedge clk); rst=1'b0; @(negedge clk);

        // ---------------- ENABLED (must PASS now) ----------------
        // A. DENSE fallback (max extent <= TOPK=4), DISTINCT x, SHARED extent.
        case_match(  11, 0, 0,   7, 3, 3, 3, "denseA_distinctx_shared");
        case_match( 123, 0, 0,   0, 4, 4, 4, "denseB_distinctx_S4");
        // B. DENSE fallback, DISTINCT x, PER-ROW DISTINCT extents (per-row mask).
        case_match(  55, 0, 0,   9, 2, 4, 3, "denseC_distinctx_perrowS");
        case_match(1010, 1, 0, 250, 3, 1, 4, "denseD_distinctx_midshort");
        // C. SPARSE (max extent > TOPK), ALL-EQUAL x, SHARED extent -> exact.
        case_match( 321, 0, 1,   5, 6, 6, 6, "sparseE_equalx_S6");
        case_match( 909, 1, 1, 100, 8, 8, 8, "sparseF_equalx_S8");
        // D. PE_M=1-fold: ALL-EQUAL x, tiny SHARED dense extent (degenerate batch).
        case_match( 777, 0, 1,  42, 2, 2, 2, "foldG_equalx_S2");

        // ---------------- PENDING B6 (distinct-x SPARSE; guarded) ----------------
        // Fetch-sharing still asserted; per-row output compare only under SPARSE_XFAIL.
        case_sparse_xfail( 202, 0,  63, 6, 5, 7, "sparseX_distinctx_S567");
        case_sparse_xfail(1313, 1,  17, 8, 6, 5, "sparseY_distinctx_S865");

        if (errors==0) begin
            $display("ALL %0d TESTS PASSED", test_count);
        end else begin
            $display("FAILED: %0d errors over %0d checks", errors, test_count);
            $fatal(1, "mla_attn_fp8 sparse/per-row batching TB failed");
        end
        $finish;
    end

    initial begin
        #2000000000;
        $display("FAIL: timeout (done never asserted)");
        $fatal(1, "timeout");
    end
endmodule
