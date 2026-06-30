`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// mla_attn_fp8_ppos_tb.v -- BINDING PER-POSITION equivalence TB for the
//   per-row query-position widening of mla_attn_fp8 (GLM-5.2 MLA decode, FP8).
//----------------------------------------------------------------------------
// WHAT THIS PROVES (the binding claim of the per-row pos_vec path)
//   A batch of B query ROWS, each decoding at its OWN query position pos_r
//   (consecutive t, t+1, ... and an edge row at position 0), pushed through ONE
//   shared weight fetch + the SAME KV prefix / cache / s_len, produces -- for
//   EACH row r -- EXACTLY the result of an INDEPENDENT, committed-style PE_M=1
//   mla_attn_fp8 run on THAT row's own (x_r, pos_r, s_len).
//
//   i.e. row r is computed AT ITS OWN POSITION: per-row query RoPE (qrot[r] is
//   rotated by pos_r), and the per-row current-token k_rope coverage pass.  The
//   causal extent over the cached keys is the SHARED s_len (the committed
//   single-token datapath's causal == s_len, NOT pos -- see the module header),
//   so the per-row reference is also run at that same s_len.
//
//   Because the DUT (PE_M=B) and the reference (PE_M=1) are the SAME module with
//   the SAME fp32 arithmetic, the binding compare is EXACT (===), X/Z aware --
//   strictly stronger than the committed 0.39% bf16 tolerance.  We ALSO assert
//   the per-row ARGMAX over the MODEL_DIM output matches the reference (the
//   token-bearing quantity), making the check argmax-bearing.
//
//   COVERAGE:
//     PE_M=4 :  consecutive positions incl. a position-0 edge row, far-apart
//               spread, an all-equal broadcast fold, zeros interspersed, S=1/S=2.
//     PE_M=2 :  re-confirm the 2-row batch at consecutive + diff positions.
//
//   REGIME (documented for PE_M>1): S_MAX = TOPK -> dense DSA fallback (keys
//   0..S-1 kept in order, q-independent), so the shared row-0-driven DSA
//   selection equals each row's own selection -- the regime in which per-row
//   batching is exact.  The KV PREFIX is SHARED across rows by construction
//   (same context window); only x and pos differ per row.
//
//   Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X / argmax flip.
//============================================================================
module mla_attn_fp8_ppos_tb;
    // ---- slice parameters (same small-but-faithful slice as the committed TB) --
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
    localparam integer PE_M4     = 4;   // 4-row batch under test
    localparam integer PE_M2     = 2;   // 2-row batch under test
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

    // ================= shared stimulus ROMs (FP8 E4M3 codes + scales + KV) =====
    reg [7:0] W_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:KV_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_kr  [0:ROPE-1][0:MODEL_DIM-1];
    reg [7:0] W_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] W_uv  [0:HV-1][0:KV_LORA-1];
    reg [7:0] W_o   [0:MODEL_DIM-1][0:HV-1];
    reg [15:0] S_dq, S_uq, S_dkv, S_kr, S_uk, S_uv, S_o;
    reg [15:0] CKV   [0:S_MAX-1][0:KV_LORA-1];     // SHARED KV prefix (c_kv latents)
    reg [15:0] KRP   [0:S_MAX-1][0:ROPE-1];        // SHARED KV prefix (k_rope)
    reg [15:0] xrow  [0:PE_M4-1][0:MODEL_DIM-1];   // per-row token activation (<=4 rows)

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

    integer ii,jj,sc;
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
        // DISTINCT per-row token activations (rows differ in x AND in pos).
        for (ii=0;ii<PE_M4;ii=ii+1)
            for (jj=0;jj<MODEL_DIM;jj=jj+1) begin xrow[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
        // SHARED KV prefix (same context window for all rows).
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<KV_LORA;jj=jj+1) begin CKV[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<ROPE;jj=jj+1)    begin KRP[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ===========================================================================
    //  PE_M=4 DUT
    // ===========================================================================
    reg                          d4_start;
    wire                         d4_busy, d4_done;
    reg  [POSW-1:0]              d4_pos;
    reg  [POSW*PE_M4-1:0]        d4_pos_vec;
    reg  [IDXW:0]               d4_slen;
    reg  [MODEL_DIM*16*PE_M4-1:0] d4_xvec;
    wire [MODEL_DIM*16*PE_M4-1:0] d4_out;
    wire                         d4_wreq; wire [3:0] d4_wsel;
    wire [GRPW-1:0]              d4_wgrp; wire [KCW-1:0] d4_wk;
    reg  [PE_N*8-1:0]           d4_wcol; reg [16*PE_N*NB-1:0] d4_wscale;
    wire                         d4_kcreq; wire [IDXW-1:0] d4_kcidx;
    reg  [KV_LORA*16-1:0]       d4_kcckv; reg [ROPE*16-1:0] d4_kckrope; reg d4_kcvalid;

    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(PE_M4)) dut4 (
        .clk(clk), .rst(rst), .start(d4_start), .busy(d4_busy), .done(d4_done),
        .pos(d4_pos), .pos_vec(d4_pos_vec), .s_len(d4_slen), .x_vec(d4_xvec),
        .w_req(d4_wreq), .w_sel(d4_wsel), .w_grp(d4_wgrp), .w_k(d4_wk),
        .w_col(d4_wcol), .w_scale(d4_wscale),
        .kc_req(d4_kcreq), .kc_idx(d4_kcidx), .kc_ckv(d4_kcckv),
        .kc_krope(d4_kckrope), .kc_valid(d4_kcvalid), .out(d4_out)
    );

    // ===========================================================================
    //  PE_M=2 DUT
    // ===========================================================================
    reg                          d2_start;
    wire                         d2_busy, d2_done;
    reg  [POSW-1:0]              d2_pos;
    reg  [POSW*PE_M2-1:0]        d2_pos_vec;
    reg  [IDXW:0]               d2_slen;
    reg  [MODEL_DIM*16*PE_M2-1:0] d2_xvec;
    wire [MODEL_DIM*16*PE_M2-1:0] d2_out;
    wire                         d2_wreq; wire [3:0] d2_wsel;
    wire [GRPW-1:0]              d2_wgrp; wire [KCW-1:0] d2_wk;
    reg  [PE_N*8-1:0]           d2_wcol; reg [16*PE_N*NB-1:0] d2_wscale;
    wire                         d2_kcreq; wire [IDXW-1:0] d2_kcidx;
    reg  [KV_LORA*16-1:0]       d2_kcckv; reg [ROPE*16-1:0] d2_kckrope; reg d2_kcvalid;

    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(PE_M2)) dut2 (
        .clk(clk), .rst(rst), .start(d2_start), .busy(d2_busy), .done(d2_done),
        .pos(d2_pos), .pos_vec(d2_pos_vec), .s_len(d2_slen), .x_vec(d2_xvec),
        .w_req(d2_wreq), .w_sel(d2_wsel), .w_grp(d2_wgrp), .w_k(d2_wk),
        .w_col(d2_wcol), .w_scale(d2_wscale),
        .kc_req(d2_kcreq), .kc_idx(d2_kcidx), .kc_ckv(d2_kcckv),
        .kc_krope(d2_kckrope), .kc_valid(d2_kcvalid), .out(d2_out)
    );

    // ===========================================================================
    //  PE_M=1 REFERENCE  (the INDEPENDENT committed-style single-token run)
    // ===========================================================================
    reg                      r1_start;
    wire                     r1_busy, r1_done;
    reg  [POSW-1:0]          r1_pos;
    reg  [POSW*1-1:0]        r1_pos_vec;
    reg  [IDXW:0]            r1_slen;
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
        .pos(r1_pos), .pos_vec(r1_pos_vec), .s_len(r1_slen), .x_vec(r1_xvec),
        .w_req(r1_wreq), .w_sel(r1_wsel), .w_grp(r1_wgrp), .w_k(r1_wk),
        .w_col(r1_wcol), .w_scale(r1_wscale),
        .kc_req(r1_kcreq), .kc_idx(r1_kcidx), .kc_ckv(r1_kcckv),
        .kc_krope(r1_kckrope), .kc_valid(r1_kcvalid), .out(r1_out)
    );

    // ================= combinational WEIGHT responders (one per instance) ======
    // Index the W ROM ARRAYS DIRECTLY inside always@* (a function call would hide
    // the array reads from @*); SEPARATE loop vars per block.
    integer td4, td2, tr1; reg [15:0] ssc_d4, ssc_d2, ssc_r1;
    always @* begin
        d4_wcol={PE_N*8{1'b0}}; d4_wscale={16*PE_N*NB{1'b0}};
        case (d4_wsel)
            4'd0: ssc_d4=S_dq;  4'd1: ssc_d4=S_uq;  4'd2: ssc_d4=S_dkv; 4'd3: ssc_d4=S_kr;
            4'd4: ssc_d4=S_uk;  4'd5: ssc_d4=S_uv;  4'd6: ssc_d4=S_o;   default: ssc_d4=16'h3F80;
        endcase
        for (td4=0;td4<PE_N;td4=td4+1) begin
            case (d4_wsel)
            4'd0: if (d4_wgrp*PE_N+td4 < Q_LORA)   d4_wcol[8*td4 +:8] = W_dq [d4_wgrp*PE_N+td4][d4_wk];
            4'd1: if (d4_wgrp*PE_N+td4 < HQK)      d4_wcol[8*td4 +:8] = W_uq [d4_wgrp*PE_N+td4][d4_wk];
            4'd2: if (d4_wgrp*PE_N+td4 < KV_LORA)  d4_wcol[8*td4 +:8] = W_dkv[d4_wgrp*PE_N+td4][d4_wk];
            4'd3: if (d4_wgrp*PE_N+td4 < ROPE)     d4_wcol[8*td4 +:8] = W_kr [d4_wgrp*PE_N+td4][d4_wk];
            4'd4: if (d4_wgrp*PE_N+td4 < HNOPE)    d4_wcol[8*td4 +:8] = W_uk [d4_wgrp*PE_N+td4][d4_wk];
            4'd5: if (d4_wgrp*PE_N+td4 < HV)       d4_wcol[8*td4 +:8] = W_uv [d4_wgrp*PE_N+td4][d4_wk];
            4'd6: if (d4_wgrp*PE_N+td4 < MODEL_DIM)d4_wcol[8*td4 +:8] = W_o  [d4_wgrp*PE_N+td4][d4_wk];
            default: d4_wcol[8*td4 +:8] = 8'h38;
            endcase
            d4_wscale[16*td4 +: 16] = ssc_d4;
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
    // SHARED KV prefix: all three instances index the SAME CKV/KRP ROMs.
    integer cd4, cd2, cd1;
    always @* begin
        d4_kcckv = {KV_LORA*16{1'b0}}; d4_kckrope = {ROPE*16{1'b0}};
        for (cd4=0;cd4<KV_LORA;cd4=cd4+1) d4_kcckv[16*cd4 +:16]  = CKV[d4_kcidx][cd4];
        for (cd4=0;cd4<ROPE;cd4=cd4+1)    d4_kckrope[16*cd4 +:16] = KRP[d4_kcidx][cd4];
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
        if (rst) begin d4_kcvalid<=1'b0; d2_kcvalid<=1'b0; r1_kcvalid<=1'b0; end
        else begin
            d4_kcvalid<=d4_kcreq; d2_kcvalid<=d2_kcreq; r1_kcvalid<=r1_kcreq;
        end
    end

    // ===========================================================================
    //  HELPERS
    // ===========================================================================
    // bf16 -> real (for argmax over the output vector); FTZ subnormals.
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

    // ===========================================================================
    //  PER-ROW POSITION DRIVERS
    // ===========================================================================
    integer i;
    // pack a per-row position into the DUT's pos / pos_vec (row 0 -> scalar pos).
    reg [POSW-1:0] posv [0:PE_M4-1];

    task run_dut4; input integer s; begin
        d4_pos = posv[0][POSW-1:0];                 // row 0 uses scalar pos
        for (i=0;i<PE_M4;i=i+1)
            d4_pos_vec[POSW*i +: POSW] = posv[i];   // row 0 slice unused; rows 1.. live
        d4_slen = s[IDXW:0];
        for (i=0;i<PE_M4;i=i+1) begin : LD4
            integer kk;
            for (kk=0;kk<MODEL_DIM;kk=kk+1)
                d4_xvec[16*(MODEL_DIM*i + kk) +:16] = xrow[i][kk];
        end
        @(negedge clk); d4_start=1'b1; @(negedge clk); d4_start=1'b0;
        wait (d4_done==1'b1); @(negedge clk);
    end endtask

    task run_dut2; input integer s; begin
        d2_pos = posv[0][POSW-1:0];
        for (i=0;i<PE_M2;i=i+1)
            d2_pos_vec[POSW*i +: POSW] = posv[i];
        d2_slen = s[IDXW:0];
        for (i=0;i<PE_M2;i=i+1) begin : LD2
            integer kk;
            for (kk=0;kk<MODEL_DIM;kk=kk+1)
                d2_xvec[16*(MODEL_DIM*i + kk) +:16] = xrow[i][kk];
        end
        @(negedge clk); d2_start=1'b1; @(negedge clk); d2_start=1'b0;
        wait (d2_done==1'b1); @(negedge clk);
    end endtask

    // INDEPENDENT PE_M=1 run on row `row` at ITS OWN (x_row, posv[row], s); -> ro[].
    reg [15:0] ro [0:MODEL_DIM-1];
    task run_ref; input integer row; input integer s; begin
        r1_pos     = posv[row][POSW-1:0];
        r1_pos_vec = posv[row][POSW-1:0];                 // unused at PE_M=1
        r1_slen    = s[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1)
            r1_xvec[16*i +:16] = xrow[row][i];
        @(negedge clk); r1_start=1'b1; @(negedge clk); r1_start=1'b0;
        wait (r1_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) ro[i] = r1_out[16*i +:16];
    end endtask

    // ===========================================================================
    //  BINDING CHECK (exact ===, X-aware, finite, + argmax-bearing)
    //    dval = one row slice of a DUT output; compared against ro[] (the ref).
    // ===========================================================================
    integer errors, test_count, fails;
    reg [15:0] dval [0:MODEL_DIM-1];
    integer dut_argmax, ref_argmax;
    real dmax, rmax, dv, rv;
    task check_vec; input integer pos_r; input [256*8-1:0] label; begin
        fails=0;
        // 1) exact, X/Z-aware per-element match to the single-token reference
        for (i=0;i<MODEL_DIM;i=i+1) begin
            if (^dval[i] === 1'bx) begin
                $display("FAIL[%0s] pos=%0d out[%0d] X/Z (%h)", label, pos_r, i, dval[i]);
                fails=fails+1;
            end else if (dval[i][14:7]==8'hFF) begin
                $display("FAIL[%0s] pos=%0d out[%0d] inf/nan (%h)", label, pos_r, i, dval[i]);
                fails=fails+1;
            end else if (dval[i] !== ro[i]) begin
                $display("FAIL[%0s] pos=%0d out[%0d] dut=%h ref=%h", label, pos_r, i, dval[i], ro[i]);
                fails=fails+1;
            end
        end
        // 2) argmax-bearing: the token index (argmax over the vector) must match.
        dut_argmax=0; ref_argmax=0;
        dmax=b2real(dval[0]); rmax=b2real(ro[0]);
        for (i=1;i<MODEL_DIM;i=i+1) begin
            dv=b2real(dval[i]); rv=b2real(ro[i]);
            if (dv>dmax) begin dmax=dv; dut_argmax=i; end
            if (rv>rmax) begin rmax=rv; ref_argmax=i; end
        end
        if (dut_argmax !== ref_argmax) begin
            $display("FAIL[%0s] pos=%0d ARGMAX dut=%0d ref=%0d", label, pos_r, dut_argmax, ref_argmax);
            fails=fails+1;
        end
        test_count=test_count+1;
        if (fails==0)
            $display("  PASS[%0s] pos=%0d (exact match + argmax=%0d)", label, pos_r, ref_argmax);
        else errors=errors+fails;
    end endtask

    // copy a PE_M4-DUT row slice into dval[].
    task grab_d4; input integer row; begin
        for (i=0;i<MODEL_DIM;i=i+1) dval[i] = d4_out[16*(MODEL_DIM*row+i) +:16];
    end endtask
    task grab_d2; input integer row; begin
        for (i=0;i<MODEL_DIM;i=i+1) dval[i] = d2_out[16*(MODEL_DIM*row+i) +:16];
    end endtask

    // ===========================================================================
    //  CASE RUNNERS
    // ===========================================================================
    integer rr;
    // PE_M=4 case: posv[0..3] preset; drive dut4 + 4 independent refs; compare.
    task case4; input integer seed0; input integer band; input integer s;
                input [256*8-1:0] label; begin
        build_stimulus(seed0, band);
        run_dut4(s);
        for (rr=0;rr<PE_M4;rr=rr+1) begin
            run_ref(rr, s);
            grab_d4(rr);
            check_vec(posv[rr], label);
        end
        $display("    (case %0s PE_M=4: pos=[%0d,%0d,%0d,%0d] S=%0d band=%0d)",
                 label, posv[0], posv[1], posv[2], posv[3], s, band);
    end endtask

    // PE_M=2 case: posv[0..1] preset; drive dut2 + 2 independent refs; compare.
    task case2; input integer seed0; input integer band; input integer s;
                input [256*8-1:0] label; begin
        build_stimulus(seed0, band);
        run_dut2(s);
        for (rr=0;rr<PE_M2;rr=rr+1) begin
            run_ref(rr, s);
            grab_d2(rr);
            check_vec(posv[rr], label);
        end
        $display("    (case %0s PE_M=2: pos=[%0d,%0d] S=%0d band=%0d)",
                 label, posv[0], posv[1], s, band);
    end endtask

    task setpos4; input integer p0; input integer p1; input integer p2; input integer p3; begin
        posv[0]=p0[POSW-1:0]; posv[1]=p1[POSW-1:0]; posv[2]=p2[POSW-1:0]; posv[3]=p3[POSW-1:0];
    end endtask

    initial begin
        errors=0; test_count=0;
        d4_start=1'b0; d2_start=1'b0; r1_start=1'b0;
        d4_pos=0; d4_pos_vec=0; d4_slen=0; d4_xvec=0;
        d2_pos=0; d2_pos_vec=0; d2_slen=0; d2_xvec=0;
        r1_pos=0; r1_pos_vec=0; r1_slen=0; r1_xvec=0;
        @(negedge clk); rst=1'b0; @(negedge clk);

        // ===== PE_M=4 : rows at their OWN positions vs independent single runs =====
        // C1: CONSECUTIVE positions incl a position-0 EDGE row (batched-verify t..t+3), S=2
        setpos4(0, 1, 2, 3);          case4(  11, 0, 2, "pem4_consec_edge0_S2");
        // C2: consecutive higher positions (no zero), S=2, wide-range band
        setpos4(100, 101, 102, 103);  case4( 123, 1, 2, "pem4_consec_hi_S2");
        // C3: FAR-APART spread with a position-0 edge, S=2
        setpos4(0, 7, 64, 4095);      case4( 321, 0, 2, "pem4_spread_edge0_S2");
        // C4: ALL-EQUAL positions (broadcast fold), S=2
        setpos4(42, 42, 42, 42);      case4( 777, 0, 2, "pem4_equalpos_fold_S2");
        // C5: zeros interspersed (RoPE identity on rows 1,3), S=1 single key
        setpos4(5, 0, 9, 0);          case4( 909, 1, 1, "pem4_zeros_interspersed_S1");

        // ===== PE_M=2 : re-confirm the 2-row batch =====
        // C6: consecutive t,t+1 with edge0, S=2
        setpos4(0, 1, 0, 0);          case2(  55, 0, 2, "pem2_consec_edge0_S2");
        // C7: far-apart positions, S=1
        setpos4(200, 201, 0, 0);      case2( 444, 1, 1, "pem2_diffpos_S1");

        if (errors==0) begin
            $display("ALL %0d TESTS PASSED", test_count);
        end else begin
            $display("FAILED: %0d errors over %0d row-position checks", errors, test_count);
            $fatal(1, "mla_attn_fp8 per-position TB failed");
        end
        $finish;
    end

    initial begin
        #1500000000;
        $display("FAIL: timeout (done never asserted)");
        $fatal(1, "timeout");
    end
endmodule
