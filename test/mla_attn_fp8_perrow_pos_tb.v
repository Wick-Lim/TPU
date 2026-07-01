`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// mla_attn_fp8_perrow_pos_tb.v -- PER-ROW QUERY POSITION equivalence TB.
//----------------------------------------------------------------------------
// PURPOSE
//   Proves the PE_M widening of mla_attn_fp8 to PER-ROW query positions
//   (pos_vec) is EXACT: for a PE_M=2 batch whose two rows decode at DIFFERENT
//   query positions pos0 != pos1 (sharing ONE weight fetch + the SAME KV
//   prefix / cache / s_len), EACH row's MODEL_DIM bf16 output is BIT-IDENTICAL
//   to the SAME mla_attn_fp8 module instantiated at PE_M=1 and run on THAT
//   row's own (x_r, pos_r, s_len).
//
//   This is a DUT-vs-DUT (same module, same arithmetic) check, so the compare
//   is EXACT (===), not a tolerance -- any per-row-position wiring bug (a row
//   using the wrong pos, a shared-pos regression, a lockstep break) flips a bit
//   and fails.  X/Z aware: any X/Z in either output fails.
//
//   Regime: S_MAX = TOPK (dense DSA fallback, keys 0..S-1 kept in order,
//   q-independent) -- the documented regime for PE_M>1, so the shared row-0-
//   driven DSA selection matches each row's own selection exactly.
//
//   Also includes an ALL-EQUAL-pos case (pos0==pos1): confirms the per-row
//   path constant-folds to the shared-pos behaviour (both rows identical and
//   equal to the PE_M=1 reference) -- the byte-identical fold.
//
// Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X.
//============================================================================
module mla_attn_fp8_perrow_pos_tb;
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
    localparam integer PE_M      = 2;       // the BATCH under test
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
    reg [15:0] x0    [0:MODEL_DIM-1];     // row-0 token activation
    reg [15:0] x1    [0:MODEL_DIM-1];     // row-1 token activation

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
        // DISTINCT row activations (rows differ in x AND position).
        for (ii=0;ii<MODEL_DIM;ii=ii+1) begin x0[ii]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<MODEL_DIM;ii=ii+1) begin x1[ii]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<KV_LORA;jj=jj+1) begin CKV[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<ROPE;jj=jj+1)    begin KRP[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ================= combinational WEIGHT responders (one per instance) ======
    // (PE_M does NOT change the weight bus width -- one responder per request
    //  stream serves both the PE_M=2 DUT and the PE_M=1 reference.)
    //  IMPORTANT (iverilog): index the W ROM ARRAYS DIRECTLY inside the always@*
    //  so the implicit sensitivity covers the array words (a function call HIDES
    //  the array reads from @*, leaving wcol stale/X).  Separate loop vars per
    //  block (a shared module-level loop reg written by two always@* blocks would
    //  be in their own sensitivity and ping-pong at time 0).

    // ===========================================================================
    //  PE_M=2 DUT  (the batch under test)
    // ===========================================================================
    reg                      d2_start;
    wire                     d2_busy, d2_done;
    reg  [POSW-1:0]          d2_pos;
    reg  [POSW*PE_M-1:0]     d2_pos_vec;
    reg  [IDXW:0]            d2_slen;
    reg  [MODEL_DIM*16*PE_M-1:0] d2_xvec;
    wire [MODEL_DIM*16*PE_M-1:0] d2_out;
    wire                     d2_wreq; wire [3:0] d2_wsel;
    wire [GRPW-1:0]          d2_wgrp; wire [KCW-1:0] d2_wk;
    reg  [PE_N*8-1:0]        d2_wcol; reg [16*PE_N*NB-1:0] d2_wscale;
    wire                     d2_kcreq; wire [IDXW-1:0] d2_kcidx;
    reg  [KV_LORA*16-1:0]    d2_kcckv; reg [ROPE*16-1:0] d2_kckrope; reg d2_kcvalid;

    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .THETA(THETA), .PE_N(PE_N),
               .POSW(POSW), .BLK(BLK), .PE_M(PE_M), .PER_ROW_POS(1)) dut2 (
        .clk(clk), .rst(rst), .start(d2_start), .busy(d2_busy), .done(d2_done),
        .pos(d2_pos), .pos_vec(d2_pos_vec), .s_len(d2_slen), .x_vec(d2_xvec),
        .w_req(d2_wreq), .w_sel(d2_wsel), .w_grp(d2_wgrp), .w_k(d2_wk),
        .w_col(d2_wcol), .w_scale(d2_wscale),
        .kc_req(d2_kcreq), .kc_idx(d2_kcidx), .kc_ckv(d2_kcckv),
        .kc_krope(d2_kckrope), .kc_valid(d2_kcvalid), .out(d2_out)
    );

    // ===========================================================================
    //  PE_M=1 REFERENCE  (single-token; run once per row)
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

    integer td2, tr1; reg [15:0] ssc_d2, ssc_r1;
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
    // (SEPARATE loop vars per block -- a shared module-level loop reg written by two
    //  always@* blocks is in their own sensitivity and ping-pongs at time 0.)
    integer cd2, cd1;
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
        if (rst) begin d2_kcvalid<=1'b0; r1_kcvalid<=1'b0; end
        else     begin d2_kcvalid<=d2_kcreq; r1_kcvalid<=r1_kcreq; end
    end

    // ===========================================================================
    //  DRIVERS
    // ===========================================================================
    integer i;
    task run_dut2; input integer p0; input integer p1; input integer s; begin
        d2_pos     = p0[POSW-1:0];                         // row 0 uses scalar pos
        d2_pos_vec[POSW*0 +: POSW] = p0[POSW-1:0];         // row-0 slice unused (row0 uses scalar pos)
        d2_pos_vec[POSW*1 +: POSW] = p1[POSW-1:0];         // row 1 uses its own slice
        d2_slen    = s[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) begin
            d2_xvec[16*(MODEL_DIM*0 + i) +:16] = x0[i];
            d2_xvec[16*(MODEL_DIM*1 + i) +:16] = x1[i];
        end
        @(negedge clk); d2_start=1'b1; @(negedge clk); d2_start=1'b0;
        wait (d2_done==1'b1); @(negedge clk);
    end endtask

    // run the PE_M=1 reference on one row's (x, pos, s); capture into ro[].
    reg [15:0] ro [0:MODEL_DIM-1];
    task run_ref; input integer usex1; input integer p; input integer s; begin
        r1_pos     = p[POSW-1:0];
        r1_pos_vec = p[POSW-1:0];                          // unused at PE_M=1, drive sanely
        r1_slen    = s[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1)
            r1_xvec[16*i +:16] = usex1 ? x1[i] : x0[i];
        @(negedge clk); r1_start=1'b1; @(negedge clk); r1_start=1'b0;
        wait (r1_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) ro[i] = r1_out[16*i +:16];
    end endtask

    // ===========================================================================
    //  CHECK  (exact, X-aware): dut2 row `row` === reference for that row.
    // ===========================================================================
    integer errors, test_count, fails;
    task check_row; input integer row; input [256*8-1:0] label; begin
        fails=0;
        for (i=0;i<MODEL_DIM;i=i+1) begin
            // X/Z aware
            if (^d2_out[16*(MODEL_DIM*row+i) +:16] === 1'bx) begin
                $display("FAIL[%0s] row%0d out[%0d] X/Z", label, row, i);
                fails=fails+1;
            end else if (d2_out[16*(MODEL_DIM*row+i) +:16] !== ro[i]) begin
                $display("FAIL[%0s] row%0d out[%0d] dut2=%h ref=%h", label, row, i,
                         d2_out[16*(MODEL_DIM*row+i) +:16], ro[i]);
                fails=fails+1;
            end
        end
        test_count=test_count+1;
        if (fails==0) $display("  PASS[%0s] row%0d (exact match)", label, row);
        else errors=errors+fails;
    end endtask

    // one full per-row case: drive PE_M=2 at (p0,p1,s), reference each row, compare.
    task one_case; input integer seed0; input integer band;
                   input integer p0; input integer p1; input integer s;
                   input [256*8-1:0] label; begin
        build_stimulus(seed0, band);
        run_dut2(p0, p1, s);
        run_ref(0, p0, s); check_row(0, label);   // row 0 vs single-token(x0,p0,s)
        run_ref(1, p1, s); check_row(1, label);   // row 1 vs single-token(x1,p1,s)
        $display("    (case %0s: pos0=%0d pos1=%0d S=%0d band=%0d)", label, p0, p1, s, band);
    end endtask

    initial begin
        errors=0; test_count=0;
        d2_start=1'b0; r1_start=1'b0;
        d2_pos=0; d2_pos_vec=0; d2_slen=0; d2_xvec=0;
        r1_pos=0; r1_pos_vec=0; r1_slen=0; r1_xvec=0;
        @(negedge clk); rst=1'b0; @(negedge clk);

        // C1: DIFFERENT pos per row, RoPE identity on row0 vs real rotation row1, S=1
        one_case(  11, 0,    0,    7, 1, "diffpos_S1");
        // C2: both rows real rotation, far-apart positions, S=2 (multi-key causal)
        one_case( 123, 0,    5, 4095, 2, "diffpos_S2_far");
        // C3: consecutive batched-verify positions t+1,t+2 sharing prefix, S=2
        one_case( 321, 1,  100,  101, 2, "consecutive_S2");
        // C4: ALL-EQUAL pos (broadcast fold): both rows same pos -> identical, S=2
        one_case( 777, 0,   42,   42, 2, "equalpos_fold_S2");
        // C5: row0 nonzero, row1 zero (RoPE identity on row1), S=2
        one_case( 909, 1,   63,    0, 2, "diffpos_row1ident_S2");

        if (errors==0) begin
            $display("ALL %0d TESTS PASSED", test_count);
        end else begin
            $display("FAILED: %0d errors over %0d row-checks", errors, test_count);
            $fatal(1, "mla_attn_fp8 per-row position TB failed");
        end
        $finish;
    end

    initial begin
        #800000000;
        $display("FAIL: timeout (done never asserted)");
        $fatal(1, "timeout");
    end
endmodule
