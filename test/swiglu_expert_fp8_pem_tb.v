`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// swiglu_expert_fp8_pem_tb.v -- PE_M batching smoke test for swiglu_expert_fp8.
//
//  GOAL (per the build spec):
//   (1) ROW-INDEPENDENCE: a PE_M=B widened expert, fed B distinct token rows
//       through the SAME shared weight fetch, must produce for EACH row r an
//       output BIT-IDENTICAL to a PE_M=1 expert run on row r's own activation.
//   (2) ONE WEIGHT FETCH: the PE_M=B run must issue the SAME weight-request
//       stream length as a SINGLE PE_M=1 run (B rows -> 1 fetch, not B fetches).
//
//  We instantiate one PE_M=4 DUT and one PE_M=1 DUT sharing the SAME weight ROM
//  (combinational responder keyed on w_sel/w_grp/w_k -- identical for both, so
//  both see byte-identical weights).  We run the PE_M=1 DUT four times (one per
//  row) to build the reference, then run the PE_M=4 DUT once and compare its
//  per-row slices bit-exactly.  We count w_req pulses on each run.
//
//  X-AWARE; emits "ALL <N> TESTS PASSED" + $fatal on any mismatch / X.
//============================================================================
module swiglu_expert_fp8_pem_tb;

    localparam integer HIDDEN = 128;
    localparam integer INTER  = 64;     // MoE expert (INTER < HIDDEN)
    localparam integer TN     = 4;
    localparam integer KMAX   = 128;    // single K-block (NB=1) -- PE_M smoke
    localparam integer BLK    = 128;
    localparam integer NB     = (KMAX + BLK - 1) / BLK;   // = 1
    localparam integer KW     = $clog2(KMAX+1);
    localparam integer GW     = $clog2(HIDDEN/TN + 1);
    localparam integer B      = 4;      // PE_M batch under test

    `include "fp8_e4m3.vh"

    localparam [1:0] SEL_GATE = 2'd0, SEL_DOWN = 2'd2;

    reg clk = 1'b0, rst = 1'b1;
    always #5 clk = ~clk;

    integer pass_cnt = 0, fail_cnt = 0;

    // ---- shared weight ROM (E4M3 codes + bf16 block scales) ----
    reg [ 7:0] Wgate [0:HIDDEN-1][0:HIDDEN-1];
    reg [ 7:0] Wup   [0:HIDDEN-1][0:HIDDEN-1];
    reg [ 7:0] Wdown [0:HIDDEN-1][0:HIDDEN-1];
    reg [15:0] SCgate, SCup, SCdown;     // NB=1, single out-block -> one scale each

    // ---- B distinct token rows ----
    reg [15:0] XR [0:B-1][0:HIDDEN-1];

    // packed token vectors
    reg [16*HIDDEN*B-1:0] x_vec_b;       // PE_M=4 (rows 0..B-1)
    reg [16*HIDDEN-1:0]   x_vec_1;        // PE_M=1 (one row at a time)
    integer pr, pk;
    always @* begin
        for (pr = 0; pr < B; pr = pr + 1)
            for (pk = 0; pk < HIDDEN; pk = pk + 1)
                x_vec_b[16*(HIDDEN*pr + pk) +: 16] = XR[pr][pk];
    end

    //---------------------------------------------------------------------
    //  RNG helpers (bounded so products stay O(1), no NaN).
    //---------------------------------------------------------------------
    function [15:0] rnd_bf16(input integer lo_e, input integer hi_e);
        reg [7:0] e; reg [6:0] m; reg s;
        begin s=$random&1; e=lo_e+({$random}%(hi_e-lo_e+1)); m=$random; rnd_bf16={s,e,m}; end
    endfunction
    function [15:0] rnd_pos_bf16(input integer lo_e, input integer hi_e);
        reg [7:0] e; reg [6:0] m;
        begin e=lo_e+({$random}%(hi_e-lo_e+1)); m=$random; rnd_pos_bf16={1'b0,e,m}; end
    endfunction
    function [7:0] rnd_e4m3_b(input integer elo, input integer ehi);
        reg s; reg [3:0] e; reg [2:0] m;
        begin s=$random&1; e=elo+({$random}%(ehi-elo+1)); m=$random;
              if (e==4'hF && m==3'h7) m=3'h6; rnd_e4m3_b={s,e,m}; end
    endfunction

    integer i2,k2,r2;
    task gen_scenario;
        begin
            for (r2=0;r2<B;r2=r2+1)
                for (k2=0;k2<HIDDEN;k2=k2+1) XR[r2][k2]=rnd_bf16(125,129);
            for (i2=0;i2<HIDDEN;i2=i2+1)
                for (k2=0;k2<HIDDEN;k2=k2+1) begin
                    Wgate[i2][k2]=rnd_e4m3_b(4,9);
                    Wup  [i2][k2]=rnd_e4m3_b(4,9);
                    Wdown[i2][k2]=rnd_e4m3_b(4,9);
                end
            SCgate=rnd_pos_bf16(122,124);
            SCup  =rnd_pos_bf16(122,124);
            SCdown=rnd_pos_bf16(122,124);
        end
    endtask

    //======================================================================
    //  DUT A: PE_M = B (the widened expert)
    //======================================================================
    reg                   start_b;
    wire                  busy_b, done_b;
    wire                  wreq_b;
    wire [1:0]            wsel_b;
    wire [GW-1:0]         wgrp_b;
    wire [KW-1:0]         wk_b;
    reg  [8*TN-1:0]       wcol_b, wcolup_b;
    reg  [16*TN*NB-1:0]   wscG_b, wscU_b;
    wire [16*HIDDEN*B-1:0] y_b;

    swiglu_expert_fp8 #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN),
                        .KMAX(KMAX), .BLK(BLK), .PE_M(B)) dut_b (
        .clk(clk), .rst(rst), .start(start_b), .busy(busy_b), .done(done_b),
        .x_vec(x_vec_b),
        .w_req(wreq_b), .w_sel(wsel_b), .w_grp(wgrp_b), .w_k(wk_b),
        .w_col(wcol_b), .w_col_up(wcolup_b),
        .w_scale_g(wscG_b), .w_scale_u(wscU_b),
        .y_out(y_b)
    );

    //======================================================================
    //  DUT B: PE_M = 1 (the reference; run once per row)
    //======================================================================
    reg                   start_1;
    wire                  busy_1, done_1;
    wire                  wreq_1;
    wire [1:0]            wsel_1;
    wire [GW-1:0]         wgrp_1;
    wire [KW-1:0]         wk_1;
    reg  [8*TN-1:0]       wcol_1, wcolup_1;
    reg  [16*TN*NB-1:0]   wscG_1, wscU_1;
    wire [16*HIDDEN-1:0]  y_1;

    swiglu_expert_fp8 #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN),
                        .KMAX(KMAX), .BLK(BLK), .PE_M(1)) dut_1 (
        .clk(clk), .rst(rst), .start(start_1), .busy(busy_1), .done(done_1),
        .x_vec(x_vec_1),
        .w_req(wreq_1), .w_sel(wsel_1), .w_grp(wgrp_1), .w_k(wk_1),
        .w_col(wcol_1), .w_col_up(wcolup_1),
        .w_scale_g(wscG_1), .w_scale_u(wscU_1),
        .y_out(y_1)
    );

    // ---- shared combinational weight responders (SAME ROM, per-DUT request) ----
    integer rt;
    always @* begin
        for (rt=0; rt<TN; rt=rt+1) begin
            wcol_b  [8*rt +: 8] = (wsel_b==SEL_DOWN) ? Wdown[wgrp_b*TN+rt][wk_b]
                                                     : Wgate[wgrp_b*TN+rt][wk_b];
            wcolup_b[8*rt +: 8] = Wup[wgrp_b*TN+rt][wk_b];
            wscG_b[16*rt +: 16] = (wsel_b==SEL_DOWN) ? SCdown : SCgate;
            wscU_b[16*rt +: 16] = SCup;

            wcol_1  [8*rt +: 8] = (wsel_1==SEL_DOWN) ? Wdown[wgrp_1*TN+rt][wk_1]
                                                     : Wgate[wgrp_1*TN+rt][wk_1];
            wcolup_1[8*rt +: 8] = Wup[wgrp_1*TN+rt][wk_1];
            wscG_1[16*rt +: 16] = (wsel_1==SEL_DOWN) ? SCdown : SCgate;
            wscU_1[16*rt +: 16] = SCup;
        end
    end

    //----------------------------------------------------------------------
    //  Reference store: y for each row from the PE_M=1 run.
    //----------------------------------------------------------------------
    reg [16*HIDDEN-1:0] ref_y [0:B-1];
    integer wreq_count_1, wreq_count_b;   // weight-request beat counts
    integer wreq_one;                     // beats of exactly ONE PE_M=1 run (row 0)

    integer wd;
    // run the PE_M=1 DUT on row `row`, capture y into ref_y[row], count w_req.
    task run_ref(input integer row);
        begin
            @(negedge clk); start_1 = 1'b1;
            @(negedge clk); start_1 = 1'b0;
            wd = 0;
            while (done_1 !== 1'b1) begin
                @(negedge clk);
                if (wreq_1 === 1'b1) wreq_count_1 = wreq_count_1 + 1;
                wd = wd + 1;
                if (wd > 200000) begin $display("FATAL: ref row %0d timeout", row);
                                       $fatal(1,"timeout"); end
            end
            ref_y[row] = y_1;
            @(negedge clk);
        end
    endtask

    task run_batch;
        begin
            @(negedge clk); start_b = 1'b1;
            @(negedge clk); start_b = 1'b0;
            wd = 0;
            while (done_b !== 1'b1) begin
                @(negedge clk);
                if (wreq_b === 1'b1) wreq_count_b = wreq_count_b + 1;
                wd = wd + 1;
                if (wd > 200000) begin $display("FATAL: batch timeout");
                                       $fatal(1,"timeout"); end
            end
            @(negedge clk);
        end
    endtask

    integer row, o;
    reg [15:0] cb_b, cb_r;
    initial begin
        start_b=1'b0; start_1=1'b0; x_vec_1={16*HIDDEN{1'b0}};
        wreq_count_1=0; wreq_count_b=0;
        rst=1'b1; repeat(4) @(negedge clk); rst=1'b0; @(negedge clk);

        gen_scenario;

        // ---- reference: PE_M=1, once per row.  Count w_req beats of row 0's run
        //      alone (wreq_one) -- ONE single-row weight stream. ----
        for (row=0; row<B; row=row+1) begin
            for (o=0;o<HIDDEN;o=o+1) x_vec_1[16*o +: 16] = XR[row][o];
            @(negedge clk);  // let x_vec_1 settle into the combinational responder path
            wreq_count_1 = 0;
            run_ref(row);
            if (row==0) wreq_one = wreq_count_1;  // freeze one PE_M=1 run's beat count
        end

        // ---- widened: PE_M=B, ONE run, all rows share the weight stream ----
        wreq_count_b = 0;
        run_batch;

        // ---- (1) per-row bit-exact equality ----
        for (row=0; row<B; row=row+1) begin
            for (o=0; o<HIDDEN; o=o+1) begin
                cb_b = y_b[16*(HIDDEN*row + o) +: 16];
                cb_r = ref_y[row][16*o +: 16];
                if (^cb_b === 1'bx) begin
                    $display("FAIL: X in PE_M=%0d output row=%0d o=%0d", B, row, o);
                    fail_cnt = fail_cnt + 1;
                end else if (cb_b !== cb_r) begin
                    $display("FAIL: row=%0d o=%0d  batch=%h  ref=%h  (NOT bit-identical)",
                             row, o, cb_b, cb_r);
                    fail_cnt = fail_cnt + 1;
                end else begin
                    pass_cnt = pass_cnt + 1;
                end
            end
        end

        // ---- (2) ONE weight fetch: PE_M=B issues the SAME #beats as ONE PE_M=1 run ----
        if (wreq_count_b != wreq_one) begin
            $display("FAIL: weight fetch beats differ: PE_M=%0d=%0d  PE_M=1=%0d",
                     B, wreq_count_b, wreq_one);
            fail_cnt = fail_cnt + 1;
        end else begin
            $display("WEIGHT-SHARE OK: PE_M=%0d run issued %0d weight beats == ONE PE_M=1 run (%0d) -> %0d rows, 1 fetch stream",
                     B, wreq_count_b, wreq_one, B);
            pass_cnt = pass_cnt + 1;
        end

        if (fail_cnt != 0) begin
            $display("FAILED: %0d mismatch(es), %0d passed.", fail_cnt, pass_cnt);
            $fatal(1, "swiglu_expert_fp8 PE_M smoke FAILED");
        end else begin
            $display("ALL %0d TESTS PASSED", pass_cnt);
        end
        $finish;
    end
endmodule
