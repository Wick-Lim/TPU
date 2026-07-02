`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// mla_attn_fp8_swin_decouple_tb.v  --  B7 SCRATCH-DECOUPLE demonstration.
//----------------------------------------------------------------------------
// PROVES the attention SCRATCH is decoupled from S_MAX:  with S_MAX=64 (>> the
//   DSA budget TOPK=8) instantiate TWO PE_M=1 mla_attn_fp8 that differ ONLY in
//   the new SWIN parameter, over the SAME weight ROMs / block scales / KV cache:
//
//     DUT : SWIN = 8   -- scores/probs/vstore + glm_softmax LEN sized by the
//                         SMALL window (8), NOT by S_MAX (64).  This is the
//                         "decoupled" scratch that lets attention scale to a
//                         1M-position range without the scratch exploding.
//     REF : SWIN = 64  (= S_MAX) -- the pre-B7 sizing (scratch == full position
//                         range).  Serves as the reference the decouple must match.
//
//   For each directed case (dense S<=TOPK and SPARSE S>TOPK) we assert the two
//   instances' MODEL_DIM bf16 `out` are BIT-EXACT (X/Z-aware, exact ===).  Since
//   the ONLY difference is SWIN, a bit-exact match proves the SWIN=8 scratch
//   (indexed by the COMPACT union slot) reproduces the full S_MAX=64 scratch
//   result -- i.e. the key INDICES still span S_MAX while the SCRATCH shrinks.
//
//   Also elaborates the design at S_MAX=64,SWIN=8 (compile == elaboration proof).
//   Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X.
//============================================================================
module mla_attn_fp8_swin_decouple_tb;
    // slice with S_MAX(64) >> TOPK(8): sparse selection over a large range.
    localparam integer MODEL_DIM = 16;
    localparam integer H_HEADS   = 2;
    localparam integer NOPE      = 4;
    localparam integer ROPE      = 4;
    localparam integer V_DIM     = 4;
    localparam integer Q_LORA    = 8;
    localparam integer KV_LORA   = 8;
    localparam integer S_MAX     = 64;   // full position range (>> TOPK)
    localparam integer TOPK      = 8;    // DSA top-K budget
    localparam integer SWIN_DUT  = 8;    // decoupled scratch window (== TOPK)
    localparam integer SWIN_REF  = 64;   // pre-B7 sizing (== S_MAX)
    localparam integer THETA     = 8000000;
    localparam integer PE_N      = 2;
    localparam integer POSW      = 20;
    localparam integer BLK       = 128;
    localparam integer QK_DIM    = NOPE + ROPE;
    localparam integer IDXW      = (S_MAX <= 1) ? 1 : $clog2(S_MAX);
    localparam integer HQK       = H_HEADS*QK_DIM;
    localparam integer HNOPE     = H_HEADS*NOPE;
    localparam integer HV        = H_HEADS*V_DIM;

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
    reg [15:0] xrow  [0:MODEL_DIM-1];

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
        for (ii=0;ii<MODEL_DIM;ii=ii+1) begin xrow[ii]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<KV_LORA;jj=jj+1) begin CKV[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
        for (ii=0;ii<S_MAX;ii=ii+1) for (jj=0;jj<ROPE;jj=jj+1)    begin KRP[ii][jj]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ===================== DUT (SWIN=8) and REF (SWIN=64) =====================
    reg                      a_start;   wire a_busy, a_done;
    reg  [POSW-1:0]          a_pos;     reg [POSW*1-1:0] a_posv;
    reg  [IDXW:0]            a_slen;    reg [(IDXW+1)*1-1:0] a_slenv;
    reg  [MODEL_DIM*16-1:0]  a_xvec;    wire [MODEL_DIM*16-1:0] a_out;
    wire                     a_wreq;    wire [3:0] a_wsel;
    wire [GRPW-1:0]          a_wgrp;    wire [KCW-1:0] a_wk;
    reg  [PE_N*8-1:0]        a_wcol;    reg [16*PE_N*NB-1:0] a_wscale;
    wire                     a_kcreq;   wire [IDXW-1:0] a_kcidx;
    reg  [KV_LORA*16-1:0]    a_kcckv;   reg [ROPE*16-1:0] a_kckrope; reg a_kcvalid;

    reg                      b_start;   wire b_busy, b_done;
    reg  [POSW-1:0]          b_pos;     reg [POSW*1-1:0] b_posv;
    reg  [IDXW:0]            b_slen;    reg [(IDXW+1)*1-1:0] b_slenv;
    reg  [MODEL_DIM*16-1:0]  b_xvec;    wire [MODEL_DIM*16-1:0] b_out;
    wire                     b_wreq;    wire [3:0] b_wsel;
    wire [GRPW-1:0]          b_wgrp;    wire [KCW-1:0] b_wk;
    reg  [PE_N*8-1:0]        b_wcol;    reg [16*PE_N*NB-1:0] b_wscale;
    wire                     b_kcreq;   wire [IDXW-1:0] b_kcidx;
    reg  [KV_LORA*16-1:0]    b_kcckv;   reg [ROPE*16-1:0] b_kckrope; reg b_kcvalid;

    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .SWIN(SWIN_DUT), .THETA(THETA),
               .PE_N(PE_N), .POSW(POSW), .BLK(BLK), .PE_M(1)) dut (
        .clk(clk), .rst(rst), .start(a_start), .busy(a_busy), .done(a_done),
        .pos(a_pos), .pos_vec(a_posv), .s_len(a_slen), .s_len_vec(a_slenv),
        .x_vec(a_xvec),
        .w_req(a_wreq), .w_sel(a_wsel), .w_grp(a_wgrp), .w_k(a_wk),
        .w_col(a_wcol), .w_scale(a_wscale),
        .kc_req(a_kcreq), .kc_idx(a_kcidx), .kc_ckv(a_kcckv),
        .kc_krope(a_kckrope), .kc_valid(a_kcvalid), .out(a_out)
    );

    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE),
               .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
               .S_MAX(S_MAX), .TOPK(TOPK), .SWIN(SWIN_REF), .THETA(THETA),
               .PE_N(PE_N), .POSW(POSW), .BLK(BLK), .PE_M(1)) ref0 (
        .clk(clk), .rst(rst), .start(b_start), .busy(b_busy), .done(b_done),
        .pos(b_pos), .pos_vec(b_posv), .s_len(b_slen), .s_len_vec(b_slenv),
        .x_vec(b_xvec),
        .w_req(b_wreq), .w_sel(b_wsel), .w_grp(b_wgrp), .w_k(b_wk),
        .w_col(b_wcol), .w_scale(b_wscale),
        .kc_req(b_kcreq), .kc_idx(b_kcidx), .kc_ckv(b_kcckv),
        .kc_krope(b_kckrope), .kc_valid(b_kcvalid), .out(b_out)
    );

    // ---------------- weight responders (one per instance) ----------------
    integer ta, tb; reg [15:0] ssc_a, ssc_b;
    always @* begin
        a_wcol={PE_N*8{1'b0}}; a_wscale={16*PE_N*NB{1'b0}};
        case (a_wsel)
            4'd0: ssc_a=S_dq;  4'd1: ssc_a=S_uq;  4'd2: ssc_a=S_dkv; 4'd3: ssc_a=S_kr;
            4'd4: ssc_a=S_uk;  4'd5: ssc_a=S_uv;  4'd6: ssc_a=S_o;   default: ssc_a=16'h3F80;
        endcase
        for (ta=0;ta<PE_N;ta=ta+1) begin
            case (a_wsel)
            4'd0: if (a_wgrp*PE_N+ta < Q_LORA)   a_wcol[8*ta +:8] = W_dq [a_wgrp*PE_N+ta][a_wk];
            4'd1: if (a_wgrp*PE_N+ta < HQK)      a_wcol[8*ta +:8] = W_uq [a_wgrp*PE_N+ta][a_wk];
            4'd2: if (a_wgrp*PE_N+ta < KV_LORA)  a_wcol[8*ta +:8] = W_dkv[a_wgrp*PE_N+ta][a_wk];
            4'd3: if (a_wgrp*PE_N+ta < ROPE)     a_wcol[8*ta +:8] = W_kr [a_wgrp*PE_N+ta][a_wk];
            4'd4: if (a_wgrp*PE_N+ta < HNOPE)    a_wcol[8*ta +:8] = W_uk [a_wgrp*PE_N+ta][a_wk];
            4'd5: if (a_wgrp*PE_N+ta < HV)       a_wcol[8*ta +:8] = W_uv [a_wgrp*PE_N+ta][a_wk];
            4'd6: if (a_wgrp*PE_N+ta < MODEL_DIM)a_wcol[8*ta +:8] = W_o  [a_wgrp*PE_N+ta][a_wk];
            default: a_wcol[8*ta +:8] = 8'h38;
            endcase
            a_wscale[16*ta +: 16] = ssc_a;
        end
    end
    always @* begin
        b_wcol={PE_N*8{1'b0}}; b_wscale={16*PE_N*NB{1'b0}};
        case (b_wsel)
            4'd0: ssc_b=S_dq;  4'd1: ssc_b=S_uq;  4'd2: ssc_b=S_dkv; 4'd3: ssc_b=S_kr;
            4'd4: ssc_b=S_uk;  4'd5: ssc_b=S_uv;  4'd6: ssc_b=S_o;   default: ssc_b=16'h3F80;
        endcase
        for (tb=0;tb<PE_N;tb=tb+1) begin
            case (b_wsel)
            4'd0: if (b_wgrp*PE_N+tb < Q_LORA)   b_wcol[8*tb +:8] = W_dq [b_wgrp*PE_N+tb][b_wk];
            4'd1: if (b_wgrp*PE_N+tb < HQK)      b_wcol[8*tb +:8] = W_uq [b_wgrp*PE_N+tb][b_wk];
            4'd2: if (b_wgrp*PE_N+tb < KV_LORA)  b_wcol[8*tb +:8] = W_dkv[b_wgrp*PE_N+tb][b_wk];
            4'd3: if (b_wgrp*PE_N+tb < ROPE)     b_wcol[8*tb +:8] = W_kr [b_wgrp*PE_N+tb][b_wk];
            4'd4: if (b_wgrp*PE_N+tb < HNOPE)    b_wcol[8*tb +:8] = W_uk [b_wgrp*PE_N+tb][b_wk];
            4'd5: if (b_wgrp*PE_N+tb < HV)       b_wcol[8*tb +:8] = W_uv [b_wgrp*PE_N+tb][b_wk];
            4'd6: if (b_wgrp*PE_N+tb < MODEL_DIM)b_wcol[8*tb +:8] = W_o  [b_wgrp*PE_N+tb][b_wk];
            default: b_wcol[8*tb +:8] = 8'h38;
            endcase
            b_wscale[16*tb +: 16] = ssc_b;
        end
    end

    // ---- cache responders + registered valid (1-cycle latency) ----
    integer ca, cb;
    always @* begin
        a_kcckv = {KV_LORA*16{1'b0}}; a_kckrope = {ROPE*16{1'b0}};
        for (ca=0;ca<KV_LORA;ca=ca+1) a_kcckv[16*ca +:16]  = CKV[a_kcidx][ca];
        for (ca=0;ca<ROPE;ca=ca+1)    a_kckrope[16*ca +:16] = KRP[a_kcidx][ca];
    end
    always @* begin
        b_kcckv = {KV_LORA*16{1'b0}}; b_kckrope = {ROPE*16{1'b0}};
        for (cb=0;cb<KV_LORA;cb=cb+1) b_kcckv[16*cb +:16]  = CKV[b_kcidx][cb];
        for (cb=0;cb<ROPE;cb=cb+1)    b_kckrope[16*cb +:16] = KRP[b_kcidx][cb];
    end
    always @(posedge clk) begin
        if (rst) begin a_kcvalid<=1'b0; b_kcvalid<=1'b0; end
        else     begin a_kcvalid<=a_kcreq; b_kcvalid<=b_kcreq; end
    end

    // ===========================================================================
    //  DRIVERS
    // ===========================================================================
    integer i;
    reg [15:0] ao [0:MODEL_DIM-1];
    reg [15:0] bo [0:MODEL_DIM-1];
    task run_dut; input integer p; input integer s; begin
        a_pos=p[POSW-1:0]; a_posv=p[POSW-1:0];
        a_slen=s[IDXW:0];  a_slenv=s[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) a_xvec[16*i +:16] = xrow[i];
        @(negedge clk); a_start=1'b1; @(negedge clk); a_start=1'b0;
        wait (a_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) ao[i] = a_out[16*i +:16];
    end endtask
    task run_ref; input integer p; input integer s; begin
        b_pos=p[POSW-1:0]; b_posv=p[POSW-1:0];
        b_slen=s[IDXW:0];  b_slenv=s[IDXW:0];
        for (i=0;i<MODEL_DIM;i=i+1) b_xvec[16*i +:16] = xrow[i];
        @(negedge clk); b_start=1'b1; @(negedge clk); b_start=1'b0;
        wait (b_done==1'b1); @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) bo[i] = b_out[16*i +:16];
    end endtask

    integer errors, test_count, fails;
    task check; input [256*8-1:0] label; input integer p; input integer s; begin
        fails=0;
        for (i=0;i<MODEL_DIM;i=i+1) begin
            if (^ao[i] === 1'bx) begin
                $display("FAIL[%0s] out[%0d] DUT(SWIN=%0d) X/Z", label, i, SWIN_DUT); fails=fails+1;
            end else if (ao[i] !== bo[i]) begin
                $display("FAIL[%0s] out[%0d] SWIN8=%h SWIN64=%h", label, i, ao[i], bo[i]); fails=fails+1;
            end
        end
        test_count=test_count+1;
        if (fails==0) $display("  PASS[%0s] pos=%0d S=%0d (%s): SWIN=8 out === SWIN=64 out (BIT-EXACT over %0d dims)",
                               label, p, s, (s<=TOPK)?"DENSE":"SPARSE", MODEL_DIM);
        else errors=errors+fails;
    end endtask

    task both; input [256*8-1:0] label; input integer seed0; input integer band;
               input integer p; input integer s; begin
        build_stimulus(seed0, band);
        run_dut(p, s);
        run_ref(p, s);
        check(label, p, s);
    end endtask

    initial begin
        errors=0; test_count=0;
        a_start=0; b_start=0;
        a_pos=0; a_posv=0; a_slen=0; a_slenv=0; a_xvec=0;
        b_pos=0; b_posv=0; b_slen=0; b_slenv=0; b_xvec=0;
        @(negedge clk); rst=1'b0; @(negedge clk);

        $display("S_MAX=%0d  TOPK=%0d  SWIN(DUT)=%0d  SWIN(REF)=%0d  (S_MAX >> SWIN decouple)",
                 S_MAX, TOPK, SWIN_DUT, SWIN_REF);
        $display("scratch dims: scores[PE_M][H_HEADS][SWIN] vstore[H_HEADS][SWIN][V_DIM] probs[PE_M][H_HEADS][SWIN]");
        $display("  DUT SWIN=%0d -> scores/probs 3rd-dim=%0d, vstore 2nd-dim=%0d (NOT S_MAX=%0d)",
                 SWIN_DUT, SWIN_DUT, SWIN_DUT, S_MAX);

        // DENSE (S <= TOPK): dsa keeps keys 0..S-1.
        both("denseA",  11, 0,   7, 4);
        both("denseB", 123, 0,  20, 8);
        // SPARSE (S > TOPK=8, up to the full S_MAX=64 range): dsa selects TOPK keys
        //   over a 16 / 32 / 64 position range -- the SWIN=8 scratch must still match.
        both("sparse16", 55, 0,  40, 16);
        both("sparse32",909, 1, 300, 32);
        both("sparse64",321, 0, 999, 64);
        both("sparse48",777, 1,  63, 48);

        if (errors==0) $display("ALL %0d TESTS PASSED", test_count);
        else begin
            $display("FAILED: %0d errors over %0d checks", errors, test_count);
            $fatal(1, "mla_attn_fp8 SWIN decouple TB failed");
        end
        $finish;
    end

    initial begin
        #2000000000;
        $display("FAIL: timeout (done never asserted)");
        $fatal(1, "timeout");
    end
endmodule
