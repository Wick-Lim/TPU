`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// mla_attn_fp8_pem_tb.v -- PE_M batching equivalence + weight-share TB for
//   mla_attn_fp8 (GLM-5.2 MLA decode, FP8-native weight projections).
//----------------------------------------------------------------------------
// GOAL (not an fp64 golden -- that is mla_attn_fp8_tb's job):
//   Prove the PE_M widening is a pure throughput transform:
//     (1) BIT-EXACT per-row equivalence -- driving B query rows through ONE
//         PE_M=B wrapper yields, for EACH row r, the SAME MODEL_DIM bf16 `out`
//         as an independent PE_M=1 run on row r's own activation (with the SAME
//         shared pos / s_len / KV-cache).  Compared bit-for-bit (X-aware).
//     (2) WEIGHT-BW AMORTIZATION B->1 -- the batched PE_M=B run asserts w_req on
//         EXACTLY the same number of cycles as ONE PE_M=1 run (the B rows share
//         the one weight fetch); B independent PE_M=1 runs would assert it B x
//         as often.  So weight bandwidth per token drops by B.
//
//   We instantiate THREE structurally-identical DUTs over the SAME weight ROMs /
//   block scales / KV cache: a PE_M=1 reference, a PE_M=2 batch, a PE_M=4 batch.
//   Rows differ ONLY in their token activation x (pos/s_len/cache shared) -- the
//   documented batched-decode regime.  S <= TOPK (dense DSA fallback) so the
//   shared, row-0-driven key selection equals every row's own selection -> the
//   per-row attention is bit-identical to the single-row runs.
//
//   Prints "ALL <N> TESTS PASSED" and $fatal on any mismatch / X.
//============================================================================
module mla_attn_fp8_pem_tb;
    // ---- slice parameters (match the committed mla_attn_fp8_tb) ----
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

    // ---- shared stimulus ----
    reg  [POSW-1:0]          pos;
    reg  [IDXW:0]            s_len;

    // ================= weight ROMs (FP8 E4M3 codes) + block scales =================
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

    // per-row token activations (up to 4 rows)
    reg [15:0] xrow  [0:3][0:MODEL_DIM-1];

    // ---- deterministic stimulus generators (same hashes as committed TB) ----
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

    integer i,j,sc,rw;
    task build_stimulus; input integer seed0; input integer band; begin
        sc = seed0;
        for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[i][j]=gen_fp8(sc); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[i][j]=gen_fp8(sc);  sc=sc+1; end
        S_dq=16'h3F80; S_uq=16'h3F00; S_dkv=16'h3F80; S_kr=16'h3F80;
        S_uk=16'h4000; S_uv=16'h3F00; S_o=16'h3F80;
        // 4 DISTINCT token rows (each from its own seed band) -> rows truly differ
        for (rw=0; rw<4; rw=rw+1)
            for (i=0;i<MODEL_DIM;i=i+1) begin xrow[rw][i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[i][j]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ================= a combinational FP8 weight responder (macro) =================
    // each DUT gets its own copy keyed off ITS w_sel/w_grp/w_k.
    `define WRESP(WSEL,WGRP,WK,WCOL,WSCALE)                                     \
        integer WCOL``_t; reg [15:0] WCOL``_ss;                                 \
        always @* begin                                                        \
            WCOL   = {PE_N*8{1'b0}};                                           \
            WSCALE = {16*PE_N*NB{1'b0}};                                       \
            case (WSEL)                                                        \
                4'd0: WCOL``_ss=S_dq;  4'd1: WCOL``_ss=S_uq;                   \
                4'd2: WCOL``_ss=S_dkv; 4'd3: WCOL``_ss=S_kr;                   \
                4'd4: WCOL``_ss=S_uk;  4'd5: WCOL``_ss=S_uv;                   \
                4'd6: WCOL``_ss=S_o;   default: WCOL``_ss=16'h3F80;            \
            endcase                                                            \
            for (WCOL``_t=0; WCOL``_t<PE_N; WCOL``_t=WCOL``_t+1) begin         \
                case (WSEL)                                                    \
                4'd0: if(WGRP*PE_N+WCOL``_t<Q_LORA)  WCOL[8*WCOL``_t+:8]=W_dq [WGRP*PE_N+WCOL``_t][WK]; \
                4'd1: if(WGRP*PE_N+WCOL``_t<HQK)     WCOL[8*WCOL``_t+:8]=W_uq [WGRP*PE_N+WCOL``_t][WK]; \
                4'd2: if(WGRP*PE_N+WCOL``_t<KV_LORA) WCOL[8*WCOL``_t+:8]=W_dkv[WGRP*PE_N+WCOL``_t][WK]; \
                4'd3: if(WGRP*PE_N+WCOL``_t<ROPE)    WCOL[8*WCOL``_t+:8]=W_kr [WGRP*PE_N+WCOL``_t][WK]; \
                4'd4: if(WGRP*PE_N+WCOL``_t<HNOPE)   WCOL[8*WCOL``_t+:8]=W_uk [WGRP*PE_N+WCOL``_t][WK]; \
                4'd5: if(WGRP*PE_N+WCOL``_t<HV)      WCOL[8*WCOL``_t+:8]=W_uv [WGRP*PE_N+WCOL``_t][WK]; \
                4'd6: if(WGRP*PE_N+WCOL``_t<MODEL_DIM)WCOL[8*WCOL``_t+:8]=W_o [WGRP*PE_N+WCOL``_t][WK]; \
                default: WCOL[8*WCOL``_t+:8]=8'h38;                            \
                endcase                                                        \
                WSCALE[16*WCOL``_t+:16]=WCOL``_ss;                             \
            end                                                                \
        end

    // ================= a combinational cache responder (macro) =================
    `define CRESP(KIDX,KCKV,KKRP)                                              \
        integer KCKV``_d;                                                      \
        always @* begin                                                       \
            KCKV = {KV_LORA*16{1'b0}};  KKRP = {ROPE*16{1'b0}};               \
            for (KCKV``_d=0; KCKV``_d<KV_LORA; KCKV``_d=KCKV``_d+1)            \
                KCKV[16*KCKV``_d+:16]=CKV[KIDX][KCKV``_d];                     \
            for (KCKV``_d=0; KCKV``_d<ROPE; KCKV``_d=KCKV``_d+1)              \
                KKRP[16*KCKV``_d+:16]=KRP[KIDX][KCKV``_d];                     \
        end

    // ====================== DUT 1 : PE_M=1 reference ======================
    reg                      start1;
    wire                     busy1, done1;
    reg  [MODEL_DIM*16-1:0]  x1;
    wire [MODEL_DIM*16-1:0]  out1;
    wire                     w_req1; wire [3:0] w_sel1; wire [GRPW-1:0] w_grp1;
    wire [KCW-1:0] w_k1; reg [PE_N*8-1:0] w_col1; reg [16*PE_N*NB-1:0] w_scale1;
    wire kc_req1; wire [IDXW-1:0] kc_idx1; reg [KV_LORA*16-1:0] kc_ckv1;
    reg [ROPE*16-1:0] kc_krope1; reg kc_valid1;
    `WRESP(w_sel1,w_grp1,w_k1,w_col1,w_scale1)
    `CRESP(kc_idx1,kc_ckv1,kc_krope1)
    always @(posedge clk) if (rst) kc_valid1<=1'b0; else kc_valid1<=kc_req1;
    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM),.H_HEADS(H_HEADS),.NOPE(NOPE),.ROPE(ROPE),
        .V_DIM(V_DIM),.Q_LORA(Q_LORA),.KV_LORA(KV_LORA),.S_MAX(S_MAX),.TOPK(TOPK),
        .THETA(THETA),.PE_N(PE_N),.POSW(POSW),.BLK(BLK),.PE_M(1)) dut1 (
        .clk(clk),.rst(rst),.start(start1),.busy(busy1),.done(done1),
        .pos(pos),.s_len(s_len),.x_vec(x1),
        .w_req(w_req1),.w_sel(w_sel1),.w_grp(w_grp1),.w_k(w_k1),.w_col(w_col1),.w_scale(w_scale1),
        .kc_req(kc_req1),.kc_idx(kc_idx1),.kc_ckv(kc_ckv1),.kc_krope(kc_krope1),.kc_valid(kc_valid1),
        .out(out1));

    // ====================== DUT 2 : PE_M=2 batch ======================
    reg                      start2;
    wire                     busy2, done2;
    reg  [MODEL_DIM*16*2-1:0] x2;
    wire [MODEL_DIM*16*2-1:0] out2;
    wire                     w_req2; wire [3:0] w_sel2; wire [GRPW-1:0] w_grp2;
    wire [KCW-1:0] w_k2; reg [PE_N*8-1:0] w_col2; reg [16*PE_N*NB-1:0] w_scale2;
    wire kc_req2; wire [IDXW-1:0] kc_idx2; reg [KV_LORA*16-1:0] kc_ckv2;
    reg [ROPE*16-1:0] kc_krope2; reg kc_valid2;
    `WRESP(w_sel2,w_grp2,w_k2,w_col2,w_scale2)
    `CRESP(kc_idx2,kc_ckv2,kc_krope2)
    always @(posedge clk) if (rst) kc_valid2<=1'b0; else kc_valid2<=kc_req2;
    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM),.H_HEADS(H_HEADS),.NOPE(NOPE),.ROPE(ROPE),
        .V_DIM(V_DIM),.Q_LORA(Q_LORA),.KV_LORA(KV_LORA),.S_MAX(S_MAX),.TOPK(TOPK),
        .THETA(THETA),.PE_N(PE_N),.POSW(POSW),.BLK(BLK),.PE_M(2)) dut2 (
        .clk(clk),.rst(rst),.start(start2),.busy(busy2),.done(done2),
        .pos(pos),.s_len(s_len),.x_vec(x2),
        .w_req(w_req2),.w_sel(w_sel2),.w_grp(w_grp2),.w_k(w_k2),.w_col(w_col2),.w_scale(w_scale2),
        .kc_req(kc_req2),.kc_idx(kc_idx2),.kc_ckv(kc_ckv2),.kc_krope(kc_krope2),.kc_valid(kc_valid2),
        .out(out2));

    // ====================== DUT 4 : PE_M=4 batch ======================
    reg                      start4;
    wire                     busy4, done4;
    reg  [MODEL_DIM*16*4-1:0] x4;
    wire [MODEL_DIM*16*4-1:0] out4;
    wire                     w_req4; wire [3:0] w_sel4; wire [GRPW-1:0] w_grp4;
    wire [KCW-1:0] w_k4; reg [PE_N*8-1:0] w_col4; reg [16*PE_N*NB-1:0] w_scale4;
    wire kc_req4; wire [IDXW-1:0] kc_idx4; reg [KV_LORA*16-1:0] kc_ckv4;
    reg [ROPE*16-1:0] kc_krope4; reg kc_valid4;
    `WRESP(w_sel4,w_grp4,w_k4,w_col4,w_scale4)
    `CRESP(kc_idx4,kc_ckv4,kc_krope4)
    always @(posedge clk) if (rst) kc_valid4<=1'b0; else kc_valid4<=kc_req4;
    mla_attn_fp8 #(.MODEL_DIM(MODEL_DIM),.H_HEADS(H_HEADS),.NOPE(NOPE),.ROPE(ROPE),
        .V_DIM(V_DIM),.Q_LORA(Q_LORA),.KV_LORA(KV_LORA),.S_MAX(S_MAX),.TOPK(TOPK),
        .THETA(THETA),.PE_N(PE_N),.POSW(POSW),.BLK(BLK),.PE_M(4)) dut4 (
        .clk(clk),.rst(rst),.start(start4),.busy(busy4),.done(done4),
        .pos(pos),.s_len(s_len),.x_vec(x4),
        .w_req(w_req4),.w_sel(w_sel4),.w_grp(w_grp4),.w_k(w_k4),.w_col(w_col4),.w_scale(w_scale4),
        .kc_req(kc_req4),.kc_idx(kc_idx4),.kc_ckv(kc_ckv4),.kc_krope(kc_krope4),.kc_valid(kc_valid4),
        .out(out4));

    // ================= w_req cycle counters (weight-fetch beats) =================
    // reset on the run's start pulse; count every cycle w_req is high.
    reg [31:0] cnt1, cnt2, cnt4;
    always @(posedge clk) begin
        if (start1) cnt1 <= 32'd0; else if (w_req1) cnt1 <= cnt1 + 1'b1;
        if (start2) cnt2 <= 32'd0; else if (w_req2) cnt2 <= cnt2 + 1'b1;
        if (start4) cnt4 <= 32'd0; else if (w_req4) cnt4 <= cnt4 + 1'b1;
    end

    // ================= storage for reference per-row outputs =================
    reg [MODEL_DIM*16-1:0] ref_out [0:3];
    reg [31:0]             ref_cnt [0:3];

    integer errors, test_count;

    // ---- drive tasks ----
    task run_ref; input integer row; begin
        @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) x1[16*i +:16]=xrow[row][i];
        start1=1'b1; @(negedge clk); start1=1'b0;
        wait (done1==1'b1); @(negedge clk);
        ref_out[row]=out1; ref_cnt[row]=cnt1;
    end endtask

    task run_b2; begin
        @(negedge clk);
        for (i=0;i<MODEL_DIM;i=i+1) begin
            x2[16*(MODEL_DIM*0+i) +:16]=xrow[0][i];
            x2[16*(MODEL_DIM*1+i) +:16]=xrow[1][i];
        end
        start2=1'b1; @(negedge clk); start2=1'b0;
        wait (done2==1'b1); @(negedge clk);
    end endtask

    task run_b4; begin
        @(negedge clk);
        for (rw=0; rw<4; rw=rw+1)
            for (i=0;i<MODEL_DIM;i=i+1)
                x4[16*(MODEL_DIM*rw+i) +:16]=xrow[rw][i];
        start4=1'b1; @(negedge clk); start4=1'b0;
        wait (done4==1'b1); @(negedge clk);
    end endtask

    // ---- bit-exact row compare ----
    integer b;
    task cmp_row; input [256*8-1:0] label; input integer row;
                  input [MODEL_DIM*16-1:0] got;
        begin
            for (b=0;b<MODEL_DIM*16;b=b+1)
                if (got[b]===1'bx || got[b]===1'bz) begin
                    $display("FAIL[%0s] row %0d: out bit %0d is X/Z", label, row, b);
                    errors=errors+1;
                end
            if (got !== ref_out[row]) begin
                $display("FAIL[%0s] row %0d: batched out %h != ref %h",
                         label, row, got, ref_out[row]);
                errors=errors+1;
            end
        end
    endtask

    integer p, s, band, k;
    task one_case; input integer seed0; input integer bnd; input integer pp; input integer ss;
        begin
            pos=pp[POSW-1:0]; s_len=ss[IDXW:0]; band=bnd;
            build_stimulus(seed0, bnd);
            // reference: 4 independent PE_M=1 runs (one per row's x)
            run_ref(0); run_ref(1); run_ref(2); run_ref(3);
            // batched runs
            run_b2();
            run_b4();
            // (1) BIT-EXACT per-row equivalence
            cmp_row("PEM2", 0, out2[16*MODEL_DIM*0 +: 16*MODEL_DIM]);
            cmp_row("PEM2", 1, out2[16*MODEL_DIM*1 +: 16*MODEL_DIM]);
            cmp_row("PEM4", 0, out4[16*MODEL_DIM*0 +: 16*MODEL_DIM]);
            cmp_row("PEM4", 1, out4[16*MODEL_DIM*1 +: 16*MODEL_DIM]);
            cmp_row("PEM4", 2, out4[16*MODEL_DIM*2 +: 16*MODEL_DIM]);
            cmp_row("PEM4", 3, out4[16*MODEL_DIM*3 +: 16*MODEL_DIM]);
            // (2) WEIGHT-FETCH amortization: batched w_req beat-count == ONE single run.
            //     (the per-run reference counts are all equal -> weight stream is
            //      data-independent and the batch shares the one fetch.)
            for (k=1;k<4;k=k+1)
                if (ref_cnt[k] !== ref_cnt[0]) begin
                    $display("FAIL[wreq]: ref run %0d count %0d != run0 %0d",
                             k, ref_cnt[k], ref_cnt[0]);
                    errors=errors+1;
                end
            if (cnt2 !== ref_cnt[0]) begin
                $display("FAIL[wreq]: PE_M=2 batch w_req beats %0d != single-run %0d",
                         cnt2, ref_cnt[0]); errors=errors+1;
            end
            if (cnt4 !== ref_cnt[0]) begin
                $display("FAIL[wreq]: PE_M=4 batch w_req beats %0d != single-run %0d",
                         cnt4, ref_cnt[0]); errors=errors+1;
            end
            test_count=test_count+1;
            if (errors==0)
                $display("  PASS[seed=%0d band=%0d pos=%0d S=%0d]  wreq: 1x=%0d  b2=%0d  b4=%0d  (B-runs would be %0dx)",
                         seed0, bnd, pp, ss, ref_cnt[0], cnt2, cnt4, 4);
        end
    endtask

    initial begin
        errors=0; test_count=0;
        start1=0; start2=0; start4=0; pos=0; s_len=0;
        x1=0; x2=0; x4=0; cnt1=0; cnt2=0; cnt4=0;
        @(negedge clk); rst=1'b0;

        one_case(  1, 0,    0, 1);   // S=1 pos=0 (RoPE identity)
        one_case(100, 0,    7, 1);   // S=1 pos>0
        one_case(200, 0,    0, 2);   // S=2 pos=0
        one_case(300, 0,    5, 2);   // S=2 pos>0
        one_case(400, 1, 4095, 2);   // wide-range, S=2, large pos
        one_case(700, 1,   42, 1);   // wide-range, S=1, pos>0

        if (errors==0) $display("ALL %0d TESTS PASSED", test_count);
        else begin
            $display("FAILED: %0d errors over %0d tests", errors, test_count);
            $fatal(1, "mla_attn_fp8 PE_M batching TB failed");
        end
        $finish;
    end

    initial begin
        #800000000;
        $display("FAIL: timeout"); $fatal(1, "timeout");
    end
endmodule
