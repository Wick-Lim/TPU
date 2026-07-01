`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// mla_attn_fp8_pslen_tb.v -- PER-ROW CAUSAL EXTENT equivalence TB.
//----------------------------------------------------------------------------
// PURPOSE
//   Proves the PE_M widening of mla_attn_fp8 to PER-ROW causal extent
//   (s_len_vec, PER_ROW_SLEN=1) is EXACT: for a PE_M batch whose rows attend
//   DIFFERENT numbers of causal keys (s_len_0 != s_len_1 [ != s_len_2 ]) while
//   sharing ONE weight fetch + the SAME KV prefix / cache / query positions,
//   EACH row's MODEL_DIM bf16 output is BIT-IDENTICAL to the SAME mla_attn_fp8
//   module instantiated at PE_M=1 and run on THAT row's own (x_r, pos, s_len_r).
//
//   The KV prefix / key stream is SHARED across rows (rows share context, differ
//   only in extent): the batched DUT scores/caches keys 0..max(s_len_r)-1 once,
//   then masks row r's scores for keys j>=s_len_r to bf16 -inf before its
//   per-row softmax -- so row r attends exactly keys 0..s_len_r-1.
//
//   This is a DUT-vs-DUT (same module, same arithmetic) check, so the compare
//   is EXACT (===), not a tolerance -- any per-row-extent wiring bug (a row
//   attending too many/few keys, a shared-extent regression, a lockstep break)
//   flips a bit and fails.  X/Z aware: any X/Z in either output fails.
//
//   Regime: S_MAX = TOPK (dense DSA fallback, keys 0..S-1 kept in order,
//   q-independent) -- the documented regime for PE_M>1, so the shared row-0-
//   driven DSA selection over max(s_len_r) keys gives sel_list[s]=s and the
//   softmax feed index IS the key index (the extent mask is exact).
//
//   Also includes an ALL-EQUAL-slen case (byte-identical fold): all rows same
//   s_len -> the per-row-extent path constant-folds to the shared-extent
//   behaviour (all rows identical, equal to the PE_M=1 reference).
//
//   COVERAGE: PE_M=2 (rows differ in extent) and PE_M=3 (three distinct
//   extents, incl. an S=1 edge row and the shared-max row).
//
// Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X.
//============================================================================
module mla_attn_fp8_pslen_tb;
    // ---- slice parameters (S_MAX=TOPK=4 -> room for extents 1..4; dense DSA) ----
    localparam integer MODEL_DIM = 16;
    localparam integer H_HEADS   = 2;
    localparam integer NOPE      = 4;
    localparam integer ROPE      = 4;
    localparam integer V_DIM     = 4;
    localparam integer Q_LORA    = 8;
    localparam integer KV_LORA   = 8;
    localparam integer S_MAX     = 4;
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
    reg [15:0] xr    [0:2][0:MODEL_DIM-1];     // up to 3 rows' token activations

    // deterministic stimulus generators (same hashing style as committed TB).
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
        // DISTINCT per-row token activations (rows differ in x AND causal extent).
        for (kk=0;kk<3;kk=kk+1)
            for (ii=0;ii<MODEL_DIM;ii=ii+1) begin xr[kk][ii]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<KV_LORA;jj=jj+1) begin CKV[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<ROPE;jj=jj+1)    begin KRP[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ===========================================================================
    //  PE_M=3 DUT  (the batch under test; PE_M=2 cases drive only rows 0..1)
    // ===========================================================================
    localparam integer PE_M = 3;
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
               .POSW(POSW), .BLK(BLK), .PE_M(PE_M), .PER_ROW_SLEN(1)) dutM (
        .clk(clk), .rst(rst), .start(d_start), .busy(d_busy), .done(d_done),
        .pos(d_pos), .pos_vec(d_pos_vec), .s_len(d_slen), .s_len_vec(d_slen_vec),
        .x_vec(d_xvec),
        .w_req(d_wreq), .w_sel(d_wsel), .w_grp(d_wgrp), .w_k(d_wk),
        .w_col(d_wcol), .w_scale(d_wscale),
        .kc_req(d_kcreq), .kc_idx(d_kcidx), .kc_ckv(d_kcckv),
        .kc_krope(d_kckrope), .kc_valid(d_kcvalid), .out(d_out)
    );

    // ===========================================================================
    //  PE_M=2 DUT  (byte-identical fold guard uses this too; drives 2 rows)
    // ===========================================================================
    localparam integer PE_M2 = 2;
    reg                      d2_start;
    wire                     d2_busy, d2_done;
    reg  [POSW-1:0]          d2_pos;
    reg  [POSW*PE_M2-1:0]    d2_pos_vec;
    reg  [IDXW:0]            d2_slen;
    reg  [(IDXW+1)*PE_M2-1:0] d2_slen_vec;
    reg  [MODEL_DIM*16*PE_M2-1:0] d2_xvec;
    wire [MODEL_DIM*16*PE_M2-1:0] d2_out;
    wire                     d2_wreq; wire [3:0] d2_wsel;
    wire [GRPW-1:0]          d2_wgrp; wire [KCW-1:0] d2_wk;
    reg  [PE_N*8-1:0]        d2_wcol; reg [16*PE_N*NB-1:0] d2_wscale;
    wire                     d2_kcreq; wire [IDXW-1:0] d2_kcidx;
    reg  [KV_LORA*16-1:0]    d2_kcckv; reg [ROPE*16-1:0] d2_kckrope; reg d2_kcvalid;

    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(PE_M2), .PER_ROW_SLEN(1)) dut2 (
        .clk(clk), .rst(rst), .start(d2_start), .busy(d2_busy), .done(d2_done),
        .pos(d2_pos), .pos_vec(d2_pos_vec), .s_len(d2_slen), .s_len_vec(d2_slen_vec),
        .x_vec(d2_xvec),
        .w_req(d2_wreq), .w_sel(d2_wsel), .w_grp(d2_wgrp), .w_k(d2_wk),
        .w_col(d2_wcol), .w_scale(d2_wscale),
        .kc_req(d2_kcreq), .kc_idx(d2_kcidx), .kc_ckv(d2_kcckv),
        .kc_krope(d2_kckrope), .kc_valid(d2_kcvalid), .out(d2_out)
    );

    // ===========================================================================
    //  PE_M=1 REFERENCE  (single-token; run once per row at THAT row's s_len)
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
               .POSW(POSW), .BLK(BLK), .PE_M(1)) ref1 (
        .clk(clk), .rst(rst), .start(r1_start), .busy(r1_busy), .done(r1_done),
        .pos(r1_pos), .pos_vec(r1_pos_vec), .s_len(r1_slen), .s_len_vec(r1_slen_vec),
        .x_vec(r1_xvec),
        .w_req(r1_wreq), .w_sel(r1_wsel), .w_grp(r1_wgrp), .w_k(r1_wk),
        .w_col(r1_wcol), .w_scale(r1_wscale),
        .kc_req(r1_kcreq), .kc_idx(r1_kcidx), .kc_ckv(r1_kcckv),
        .kc_krope(r1_kckrope), .kc_valid(r1_kcvalid), .out(r1_out)
    );

    // ================= combinational WEIGHT responders (one per instance) ======
    //  (PE_M does NOT change the weight bus width -- one responder per request
    //   stream serves every instance.)  Index the W ROM ARRAYS DIRECTLY inside the
    //   always@* so the implicit sensitivity covers the array words.  SEPARATE loop
    //   vars per block so no shared loop reg ping-pongs at time 0.
    integer tdM, td2, tr1; reg [15:0] ssc_dM, ssc_d2, ssc_r1;
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
        d2_wcol={PE_N*8{1'b0}}; d2_wscale={16*PE_N*NB{1'b0}};
        case (d2_wsel)
            4'd0: ssc_d2=S_dq;  4'd1: ssc_d2=S_uq;  4'd2: ssc_d2=S_dkv; 4'd3: ssc_d2=S_kr;
            4'd4: ssc_d2=S_uk;  4'd5: ssc_d2=S_uv;  4'd6: ssc_d2=S_o;   default: ssc_d2=16'h3F80;
        endcase
        for (td2=0;td2<PE_N;td2=td2+1) begin
            case (d2_wsel)
            4'd0: if (d2_wgrp*PE_N+td2 < Q_LORA)   d2_wcol[8*td2 +:8] = W_dq [d2_wgrp*PE_N+td2][d2_wk];
            4'd1: if (d2_wgrp*PE_N+td2 < HQK)      d2_wcol[8*td2 +:8] = W_uq [d2_wgrp*PE_N+td2][d2_wk];
            4'd2: if (d2_wgrp*PE_N+td2 < KV_LORA)  d2_wcol[8*td2 +:8] = W_dkv[d2_wgrp*PE_N+td2][d2_wk];
            4'd3: if (d2_wgrp*PE_N+td2 < ROPE)     d2_wcol[8*td2 +:8] = W_kr [d2_wgrp*PE_N+td2][d2_wk];
            4'd4: if (d2_wgrp*PE_N+td2 < HNOPE)    d2_wcol[8*td2 +:8] = W_uk [d2_wgrp*PE_N+td2][d2_wk];
            4'd5: if (d2_wgrp*PE_N+td2 < HV)       d2_wcol[8*td2 +:8] = W_uv [d2_wgrp*PE_N+td2][d2_wk];
            4'd6: if (d2_wgrp*PE_N+td2 < MODEL_DIM)d2_wcol[8*td2 +:8] = W_o  [d2_wgrp*PE_N+td2][d2_wk];
            default: d2_wcol[8*td2 +:8] = 8'h38;
            endcase
            d2_wscale[16*td2 +: 16] = ssc_d2;
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
    integer cdM, cd2, cd1;
    always @* begin
        d_kcckv = {KV_LORA*16{1'b0}}; d_kckrope = {ROPE*16{1'b0}};
        for (cdM=0;cdM<KV_LORA;cdM=cdM+1) d_kcckv[16*cdM +:16]  = CKV[d_kcidx][cdM];
        for (cdM=0;cdM<ROPE;cdM=cdM+1)    d_kckrope[16*cdM +:16] = KRP[d_kcidx][cdM];
    end
    always @* begin
        d2_kcckv = {KV_LORA*16{1'b0}}; d2_kckrope = {ROPE*16{1'b0}};
        for (cd2=0;cd2<KV_LORA;cd2=cd2+1) d2_kcckv[16*cd2 +:16]  = CKV[d2_kcidx][cd2];
        for (cd2=0;cd2<ROPE;cd2=cd2+1)    d2_kckrope[16*cd2 +:16] = KRP[d2_kcidx][cd2];
    end
    always @* begin
        r1_kcckv = {KV_LORA*16{1'b0}}; r1_kckrope = {ROPE*16{1'b0}};
        for (cd1=0;cd1<KV_LORA;cd1=cd1+1) r1_kcckv[16*cd1 +:16]  = CKV[r1_kcidx][cd1];
        for (cd1=0;cd1<ROPE;cd1=cd1+1)    r1_kckrope[16*cd1 +:16] = KRP[r1_kcidx][cd1];
    end
    always @(posedge clk) begin
        if (rst) begin d_kcvalid<=1'b0; d2_kcvalid<=1'b0; r1_kcvalid<=1'b0; end
        else     begin d_kcvalid<=d_kcreq; d2_kcvalid<=d2_kcreq; r1_kcvalid<=r1_kcreq; end
    end

    // ===========================================================================
    //  DRIVERS
    // ===========================================================================
    integer i;
    // run the PE_M=3 DUT: shared pos p, per-row s_len s0,s1,s2, rows use xr[0..2].
    task run_dutM; input integer p; input integer s0; input integer s1; input integer s2; begin
        d_pos     = p[POSW-1:0];
        d_pos_vec = {PE_M{p[POSW-1:0]}};
        d_slen    = s0[IDXW:0];                                  // row 0 uses scalar s_len
        d_slen_vec[(IDXW+1)*0 +: (IDXW+1)] = s0[IDXW:0];         // row-0 slice unused (row0 = scalar)
        d_slen_vec[(IDXW+1)*1 +: (IDXW+1)] = s1[IDXW:0];
        d_slen_vec[(IDXW+1)*2 +: (IDXW+1)] = s2[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) begin
            d_xvec[16*(MODEL_DIM*0 + i) +:16] = xr[0][i];
            d_xvec[16*(MODEL_DIM*1 + i) +:16] = xr[1][i];
            d_xvec[16*(MODEL_DIM*2 + i) +:16] = xr[2][i];
        end
        @(negedge clk); d_start=1'b1; @(negedge clk); d_start=1'b0;
        wait (d_done==1'b1); @(negedge clk);
    end endtask

    // run the PE_M=2 DUT: shared pos p, per-row s_len s0,s1, rows use xr[0..1].
    task run_dut2; input integer p; input integer s0; input integer s1; begin
        d2_pos     = p[POSW-1:0];
        d2_pos_vec = {PE_M2{p[POSW-1:0]}};
        d2_slen    = s0[IDXW:0];
        d2_slen_vec[(IDXW+1)*0 +: (IDXW+1)] = s0[IDXW:0];
        d2_slen_vec[(IDXW+1)*1 +: (IDXW+1)] = s1[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) begin
            d2_xvec[16*(MODEL_DIM*0 + i) +:16] = xr[0][i];
            d2_xvec[16*(MODEL_DIM*1 + i) +:16] = xr[1][i];
        end
        @(negedge clk); d2_start=1'b1; @(negedge clk); d2_start=1'b0;
        wait (d2_done==1'b1); @(negedge clk);
    end endtask

    // run the PE_M=1 reference on one row's (xr[row], pos p, s_len s); capture ro[].
    reg [15:0] ro [0:MODEL_DIM-1];
    task run_ref; input integer row; input integer p; input integer s; begin
        r1_pos     = p[POSW-1:0];
        r1_pos_vec = p[POSW-1:0];                     // unused at PE_M=1, drive sanely
        r1_slen    = s[IDXW:0];
        r1_slen_vec= s[IDXW:0];                       // unused at PE_M=1, drive sanely
        for (i=0;i<MODEL_DIM;i=i+1)
            r1_xvec[16*i +:16] = xr[row][i];
        @(negedge clk); r1_start=1'b1; @(negedge clk); r1_start=1'b0;
        wait (r1_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) ro[i] = r1_out[16*i +:16];
    end endtask

    // ===========================================================================
    //  CHECK  (exact, X-aware): dut row `row` (of width `pem`) === reference.
    // ===========================================================================
    integer errors, test_count, fails;
    task check_rowM; input integer row; input [256*8-1:0] label; begin
        fails=0;
        for (i=0;i<MODEL_DIM;i=i+1) begin
            if (^d_out[16*(MODEL_DIM*row+i) +:16] === 1'bx) begin
                $display("FAIL[%0s] pem3 row%0d out[%0d] X/Z", label, row, i);
                fails=fails+1;
            end else if (d_out[16*(MODEL_DIM*row+i) +:16] !== ro[i]) begin
                $display("FAIL[%0s] pem3 row%0d out[%0d] dut=%h ref=%h", label, row, i,
                         d_out[16*(MODEL_DIM*row+i) +:16], ro[i]);
                fails=fails+1;
            end
        end
        test_count=test_count+1;
        if (fails==0) $display("  PASS[%0s] pem3 row%0d (exact match)", label, row);
        else errors=errors+fails;
    end endtask
    task check_row2; input integer row; input [256*8-1:0] label; begin
        fails=0;
        for (i=0;i<MODEL_DIM;i=i+1) begin
            if (^d2_out[16*(MODEL_DIM*row+i) +:16] === 1'bx) begin
                $display("FAIL[%0s] pem2 row%0d out[%0d] X/Z", label, row, i);
                fails=fails+1;
            end else if (d2_out[16*(MODEL_DIM*row+i) +:16] !== ro[i]) begin
                $display("FAIL[%0s] pem2 row%0d out[%0d] dut=%h ref=%h", label, row, i,
                         d2_out[16*(MODEL_DIM*row+i) +:16], ro[i]);
                fails=fails+1;
            end
        end
        test_count=test_count+1;
        if (fails==0) $display("  PASS[%0s] pem2 row%0d (exact match)", label, row);
        else errors=errors+fails;
    end endtask

    // full PE_M=3 per-row-extent case: drive DUT at (p,s0,s1,s2), ref each row.
    task caseM; input integer seed0; input integer band; input integer p;
                input integer s0; input integer s1; input integer s2;
                input [256*8-1:0] label; begin
        build_stimulus(seed0, band);
        run_dutM(p, s0, s1, s2);
        run_ref(0, p, s0); check_rowM(0, label);   // row 0 vs single-token(x0,p,s0)
        run_ref(1, p, s1); check_rowM(1, label);   // row 1 vs single-token(x1,p,s1)
        run_ref(2, p, s2); check_rowM(2, label);   // row 2 vs single-token(x2,p,s2)
        $display("    (case %0s: pos=%0d S=(%0d,%0d,%0d) band=%0d)", label, p, s0, s1, s2, band);
    end endtask

    // full PE_M=2 per-row-extent case.
    task case2; input integer seed0; input integer band; input integer p;
                input integer s0; input integer s1;
                input [256*8-1:0] label; begin
        build_stimulus(seed0, band);
        run_dut2(p, s0, s1);
        run_ref(0, p, s0); check_row2(0, label);
        run_ref(1, p, s1); check_row2(1, label);
        $display("    (case %0s: pos=%0d S=(%0d,%0d) band=%0d)", label, p, s0, s1, band);
    end endtask

    initial begin
        errors=0; test_count=0;
        d_start=1'b0; d2_start=1'b0; r1_start=1'b0;
        d_pos=0; d_pos_vec=0; d_slen=0; d_slen_vec=0; d_xvec=0;
        d2_pos=0; d2_pos_vec=0; d2_slen=0; d2_slen_vec=0; d2_xvec=0;
        r1_pos=0; r1_pos_vec=0; r1_slen=0; r1_slen_vec=0; r1_xvec=0;
        @(negedge clk); rst=1'b0; @(negedge clk);

        // ---- PE_M=2 cases ----
        // C1: row0 SHORTER extent (1) than row1 (2) -> row0 masks key 1.
        case2(  11, 0,   7, 1, 2, "pem2_short_row0");
        // C2: row0 LONGER extent (4) than row1 (2) -> row1 masks keys 2,3.
        case2( 123, 0,   5, 4, 2, "pem2_short_row1");
        // C3: ALL-EQUAL extent (byte-identical fold) -> both rows attend all.
        case2( 321, 1,  42, 3, 3, "pem2_equal_fold");
        // C4: wide spread (row0=1, row1=4) at a large shared position.
        case2( 909, 0, 100, 1, 4, "pem2_spread");

        // ---- PE_M=3 cases ----
        // C5: three DISTINCT extents (1,2,4) incl. an S=1 edge row and the max row.
        caseM(  55, 0,   9, 1, 2, 4, "pem3_1_2_4");
        // C6: descending extents (4,3,2) -> row0 is the shared max.
        caseM( 202, 1,  63, 4, 3, 2, "pem3_4_3_2");
        // C7: ALL-EQUAL extent (byte-identical fold) across 3 rows.
        caseM( 777, 0,  42, 2, 2, 2, "pem3_equal_fold");
        // C8: middle row shortest (3,1,4) at a large shared position.
        caseM(1010, 0, 250, 3, 1, 4, "pem3_mid_short");

        if (errors==0) begin
            $display("ALL %0d TESTS PASSED", test_count);
        end else begin
            $display("FAILED: %0d errors over %0d row-checks", errors, test_count);
            $fatal(1, "mla_attn_fp8 per-row s_len TB failed");
        end
        $finish;
    end

    initial begin
        #1200000000;
        $display("FAIL: timeout (done never asserted)");
        $fatal(1, "timeout");
    end
endmodule
