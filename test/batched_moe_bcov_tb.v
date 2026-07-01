`timescale 1ns/1ps
`include "glm_fp.vh"   // $unit-scope fp helpers (glm_fp_pipe.v ref modules use them)
//============================================================================
// batched_moe_bcov_tb.v  --  P1.3c: FULL B-COVERAGE for batched_moe.v
//----------------------------------------------------------------------------
// The committed batched_moe_tb.v proves the expert-grouped batched MoE at the
// single width B=PE_M=4 with ONE hand-picked overlap routing.  This TB closes
// the product B-coverage gap: it re-proves the SAME binding identity across a
// SWEEP of batch widths B in {1,2,3,5,8} and across FOUR routing regimes:
//     PATTERN 0 = heavy OVERLAP  (union << B*TOPK -- the amortization win)
//     PATTERN 1 = all DISTINCT   (max union, up to min(B*TOPK, N_EXPERT))
//     PATTERN 2 = all SAME expts (union == TOPK -- every row same experts)
//     PATTERN 3 = pseudo-RANDOM  (arbitrary TopK per row)
//
// BINDING (result identity, per row r):  batched_moe's bf16 y_out[r] must equal
//   -- BIT-EXACT (!==) -- the single-token reference
//        y_r = Σ_{e in TopK(r), ASCENDING e} gate_{r,e} * expert_e(x_r)
//   where expert_e is the SAME committed RTL (swiglu_expert_fp8, PE_M=1) run on
//   row r's activation with expert e's weights, and gate*accumulate uses the
//   SAME fp32 combine numerics (bf16->fp32, fp32_mul, fp32_add, ->bf16) in the
//   SAME ascending-expert order the DUT accumulates in.  ONE shared per-expert
//   weight ROM feeds both DUT and reference => the only legal output is
//   bit-identical.  Any X in a captured DUT bit is a hard failure.
//
// AMORTIZATION: for each case, assert the DUT fetches EXACTLY the union of the
//   selected experts, each expert's weight stream pulled ONCE (its w_req beat
//   count == one PE_M=1 run's beats) -- never a non-selected expert, never a
//   duplicate fetch.
//
// Each parametrized `bmoe_case` runs once when pulsed; the top sequences them
// (deterministic per-case $random reseed) and tallies.  Emits
// "ALL <N> TESTS PASSED"; $fatal on any mismatch / X / bad union / timeout.
//============================================================================

// ---------------------------------------------------------------------------
//  One self-contained (DUT + reference + checks) case at a fixed width B.
// ---------------------------------------------------------------------------
module bmoe_case #(
    parameter integer B        = 4,
    parameter integer PATTERN  = 0,       // 0=overlap 1=distinct 2=same 3=random
    parameter integer SEED     = 1,
    parameter integer HIDDEN   = 64,
    parameter integer INTER    = 32,
    parameter integer TN       = 4,
    parameter integer KMAX     = 64,
    parameter integer BLK      = 32,
    parameter integer N_EXPERT = 8,
    parameter integer TOPK     = 2,
    // derived
    parameter integer IDXW     = (N_EXPERT<=1)?1:$clog2(N_EXPERT),
    parameter integer NB       = (KMAX+BLK-1)/BLK,
    parameter integer NOB      = ((HIDDEN>INTER?HIDDEN:INTER)+BLK-1)/BLK,
    parameter integer MAXD     = (HIDDEN>INTER?HIDDEN:INTER),
    parameter integer KW       = $clog2(KMAX+1),
    parameter integer GW       = $clog2((INTER>HIDDEN?INTER:HIDDEN)/TN + 1)
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        go,          // 1-cycle pulse: run this case
    output reg         done_case,   // 1 when finished
    output reg [31:0]  fails,       // #failures in this case
    output reg [31:0]  checks       // #bit-exact element comparisons made
);
    localparam [1:0] SEL_GATE = 2'd0, SEL_DOWN = 2'd2;
    `include "glm_fp.vh"

    // -------------------- per-expert weight ROM --------------------
    reg [ 7:0] Wgate [0:N_EXPERT-1][0:MAXD-1][0:MAXD-1];
    reg [ 7:0] Wup   [0:N_EXPERT-1][0:MAXD-1][0:MAXD-1];
    reg [ 7:0] Wdown [0:N_EXPERT-1][0:MAXD-1][0:MAXD-1];
    reg [15:0] SCgate[0:N_EXPERT-1][0:NOB-1][0:NB-1];
    reg [15:0] SCup  [0:N_EXPERT-1][0:NOB-1][0:NB-1];
    reg [15:0] SCdown[0:N_EXPERT-1][0:NOB-1][0:NB-1];

    reg [15:0]     XR      [0:B-1][0:HIDDEN-1];
    reg [IDXW-1:0] tok_idx [0:B-1][0:TOPK-1];
    reg [15:0]     tok_gat [0:B-1][0:TOPK-1];

    // -------------------- RNG (per-case seeded) --------------------
    integer seed;
    function [15:0] rnd_bf16(input integer lo_e, input integer hi_e);
        reg [7:0] e; reg [6:0] m; reg s;
        begin s=$random(seed)&1; e=lo_e+({$random(seed)}%(hi_e-lo_e+1)); m=$random(seed);
              rnd_bf16={s,e,m}; end
    endfunction
    function [15:0] rnd_pos_bf16(input integer lo_e, input integer hi_e);
        reg [7:0] e; reg [6:0] m;
        begin e=lo_e+({$random(seed)}%(hi_e-lo_e+1)); m=$random(seed); rnd_pos_bf16={1'b0,e,m}; end
    endfunction
    function [7:0] rnd_e4m3_b(input integer elo, input integer ehi);
        reg s; reg [3:0] e; reg [2:0] m;
        begin s=$random(seed)&1; e=elo+({$random(seed)}%(ehi-elo+1)); m=$random(seed);
              if (e==4'hF && m==3'h7) m=3'h6; rnd_e4m3_b={s,e,m}; end
    endfunction

    // distinct TopK experts for row r under the chosen PATTERN (ascending idx)
    integer ge, gi, gk, gb, gob, rt2;
    reg [IDXW-1:0] used;
    task pick_row;                       // fills tok_idx[r][*] ASCENDING, distinct
        input integer r; integer a, b_, tmp, cand, ok, tt;
        begin
            case (PATTERN)
            0: begin                     // overlap: rotate through a small pool {1,3,5}
                for (a=0;a<TOPK;a=a+1) tok_idx[r][a] = ((1 + 2*((r+a)%3)) % N_EXPERT);
            end
            2: begin                     // same: every row -> experts {1,3,5,...}
                for (a=0;a<TOPK;a=a+1) tok_idx[r][a] = (1 + 2*a) % N_EXPERT;
            end
            1: begin                     // distinct across rows where possible
                for (a=0;a<TOPK;a=a+1) tok_idx[r][a] = (r*TOPK + a) % N_EXPERT;
            end
            default: begin               // 3: pseudo-random distinct within the row
                for (a=0;a<TOPK;a=a+1) begin
                    ok=0;
                    while (!ok) begin
                        cand = {$random(seed)} % N_EXPERT;
                        ok=1; for (tt=0;tt<a;tt=tt+1) if (tok_idx[r][tt]==cand[IDXW-1:0]) ok=0;
                    end
                    tok_idx[r][a] = cand[IDXW-1:0];
                end
            end
            endcase
            // sort ascending (TopK is small) -- DUT/ref combine in ascending e
            for (a=0;a<TOPK;a=a+1) for (b_=a+1;b_<TOPK;b_=b_+1)
                if (tok_idx[r][b_] < tok_idx[r][a]) begin
                    tmp=tok_idx[r][a]; tok_idx[r][a]=tok_idx[r][b_]; tok_idx[r][b_]=tmp; end
        end
    endtask

    task gen_scenario;
        begin
            for (ge=0; ge<N_EXPERT; ge=ge+1) begin
                for (gi=0; gi<MAXD; gi=gi+1)
                    for (gk=0; gk<MAXD; gk=gk+1) begin
                        Wgate[ge][gi][gk]=rnd_e4m3_b(4,9);
                        Wup  [ge][gi][gk]=rnd_e4m3_b(4,9);
                        Wdown[ge][gi][gk]=rnd_e4m3_b(4,9);
                    end
                for (gob=0; gob<NOB; gob=gob+1)
                    for (gb=0; gb<NB; gb=gb+1) begin
                        SCgate[ge][gob][gb]=rnd_pos_bf16(122,124);
                        SCup  [ge][gob][gb]=rnd_pos_bf16(122,124);
                        SCdown[ge][gob][gb]=rnd_pos_bf16(122,124);
                    end
            end
            for (gi=0; gi<B; gi=gi+1)
                for (gk=0; gk<HIDDEN; gk=gk+1) XR[gi][gk]=rnd_bf16(125,129);
            for (gi=0; gi<B; gi=gi+1) begin
                pick_row(gi);
                for (rt2=0; rt2<TOPK; rt2=rt2+1) tok_gat[gi][rt2]=rnd_pos_bf16(126,128);
            end
        end
    endtask

    // -------------------- DUT : batched_moe @ PE_M=B --------------------
    reg                    start;
    wire                   busy, done;
    reg  [16*HIDDEN*B-1:0] x_vec;
    reg  [TOPK*IDXW*B-1:0] sel_idx;
    reg  [TOPK*16*B-1:0]   sel_weight;
    wire [IDXW-1:0]        cur_expert;
    wire                   w_req;
    wire [1:0]             w_sel;
    wire [GW-1:0]          w_grp;
    wire [KW-1:0]          w_k;
    reg  [8*TN-1:0]        w_col, w_col_up;
    reg  [16*TN*NB-1:0]    w_scale_g, w_scale_u;
    wire [16*HIDDEN*B-1:0] y_out;

    batched_moe #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN), .KMAX(KMAX), .BLK(BLK),
                  .N_EXPERT(N_EXPERT), .TOPK(TOPK), .PE_M(B)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .x_vec(x_vec), .sel_idx(sel_idx), .sel_weight(sel_weight),
        .cur_expert(cur_expert), .w_req(w_req), .w_sel(w_sel),
        .w_grp(w_grp), .w_k(w_k), .w_col(w_col), .w_col_up(w_col_up),
        .w_scale_g(w_scale_g), .w_scale_u(w_scale_u), .y_out(y_out) );

    // -------------------- REFERENCE : swiglu_expert_fp8 @ PE_M=1 --------------------
    reg                  start_ref;
    wire                 busy_ref, done_ref, wreq_ref;
    wire [1:0]           wsel_ref;
    wire [GW-1:0]        wgrp_ref;
    wire [KW-1:0]        wk_ref;
    reg  [8*TN-1:0]      wcol_ref, wcolup_ref;
    reg  [16*TN*NB-1:0]  wscG_ref, wscU_ref;
    reg  [16*HIDDEN-1:0] x_vec_1;
    wire [16*HIDDEN-1:0] y_ref;
    reg  [IDXW-1:0]      ref_eid;

    swiglu_expert_fp8 #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN),
                        .KMAX(KMAX), .BLK(BLK), .PE_M(1)) dut_ref (
        .clk(clk), .rst(rst), .start(start_ref), .busy(busy_ref), .done(done_ref),
        .x_vec(x_vec_1),
        .w_req(wreq_ref), .w_sel(wsel_ref), .w_grp(wgrp_ref), .w_k(wk_ref),
        .w_col(wcol_ref), .w_col_up(wcolup_ref),
        .w_scale_g(wscG_ref), .w_scale_u(wscU_ref), .y_out(y_ref) );

    // -------------------- shared weight responder --------------------
    integer rt, rb;
    `define FILL_E(EID,WSEL,WGRP,WK,WCOL,WCOLUP,WSCG,WSCU)                          \
        for (rt=0; rt<TN; rt=rt+1) begin                                           \
            WCOL  [8*rt +: 8] = (WSEL==SEL_DOWN) ? Wdown[EID][WGRP*TN+rt][WK]       \
                                                 : Wgate[EID][WGRP*TN+rt][WK];      \
            WCOLUP[8*rt +: 8] = Wup[EID][WGRP*TN+rt][WK];                           \
            for (rb=0; rb<NB; rb=rb+1) begin                                        \
                WSCG[16*(rb*TN+rt) +: 16] = (WSEL==SEL_DOWN)                        \
                    ? SCdown[EID][(WGRP*TN+rt)/BLK][rb]                             \
                    : SCgate[EID][(WGRP*TN+rt)/BLK][rb];                            \
                WSCU[16*(rb*TN+rt) +: 16] = SCup[EID][(WGRP*TN+rt)/BLK][rb];        \
            end                                                                     \
        end
    always @* begin
        `FILL_E(cur_expert, w_sel,    w_grp,    w_k,    w_col,    w_col_up,  w_scale_g, w_scale_u)
        `FILL_E(ref_eid,    wsel_ref, wgrp_ref, wk_ref, wcol_ref, wcolup_ref, wscG_ref, wscU_ref)
    end

    // -------------------- golden combine (fp32, ascending expert) --------------------
    reg [31:0]          facc_ref [0:B-1][0:HIDDEN-1];
    reg [16*HIDDEN-1:0] ref_y    [0:B-1];
    integer WBEATS_ONE; reg first_ref;

    function [15:0] gate_of; input integer r; input integer e; integer t;
        begin gate_of=16'h0000;
            for (t=0;t<TOPK;t=t+1) if (tok_idx[r][t]==e[IDXW-1:0]) gate_of=tok_gat[r][t]; end
    endfunction
    function has_expert; input integer r; input integer e; integer t;
        begin has_expert=1'b0;
            for (t=0;t<TOPK;t=t+1) if (tok_idx[r][t]==e[IDXW-1:0]) has_expert=1'b1; end
    endfunction

    integer wd, o, beats;
    task run_ref; input integer rr_i; input integer ee_i;
        begin
            for (o=0;o<HIDDEN;o=o+1) x_vec_1[16*o +: 16]=XR[rr_i][o];
            ref_eid=ee_i[IDXW-1:0];
            @(negedge clk); @(negedge clk); start_ref=1'b1;
            @(negedge clk); start_ref=1'b0;
            wd=0; beats=0;
            while (done_ref!==1'b1) begin
                @(negedge clk);
                if (wreq_ref===1'b1) beats=beats+1;
                wd=wd+1; if (wd>600000) begin $display("FATAL: ref timeout B=%0d",B); $fatal(1,"to"); end
            end
            if (first_ref) begin WBEATS_ONE=beats; first_ref=1'b0; end
            @(negedge clk);
        end
    endtask

    integer dut_beats [0:N_EXPERT-1];
    reg     seen      [0:N_EXPERT-1];
    integer t, d, e, r, union_cnt, exp_union;
    reg [15:0] gv, dv, rv;

    // -------------------- run once on `go` --------------------
    initial begin
        done_case=1'b0; fails=32'd0; checks=32'd0;
        start=1'b0; start_ref=1'b0; ref_eid={IDXW{1'b0}};
        x_vec={16*HIDDEN*B{1'b0}}; sel_idx={TOPK*IDXW*B{1'b0}};
        sel_weight={TOPK*16*B{1'b0}}; x_vec_1={16*HIDDEN{1'b0}};
        first_ref=1'b1; WBEATS_ONE=0; seed=SEED;
        for (e=0;e<N_EXPERT;e=e+1) begin dut_beats[e]=0; seen[e]=1'b0; end

        wait (go === 1'b1);
        @(negedge clk);
        gen_scenario;

        // packed DUT routing buses
        for (r=0;r<B;r=r+1) for (t=0;t<TOPK;t=t+1) begin
            sel_idx[IDXW*(TOPK*r+t) +: IDXW]=tok_idx[r][t];
            sel_weight[16*(TOPK*r+t) +: 16] =tok_gat[r][t];
        end
        for (r=0;r<B;r=r+1) for (d=0;d<HIDDEN;d=d+1)
            x_vec[16*(HIDDEN*r+d) +: 16]=XR[r][d];

        // golden: per token, ascending expert index (== DUT accumulate order)
        for (r=0;r<B;r=r+1) for (d=0;d<HIDDEN;d=d+1) facc_ref[r][d]=32'h0;
        for (e=0;e<N_EXPERT;e=e+1) for (r=0;r<B;r=r+1)
            if (has_expert(r,e)) begin
                run_ref(r,e); gv=gate_of(r,e);
                for (d=0;d<HIDDEN;d=d+1)
                    facc_ref[r][d]=fp32_add(facc_ref[r][d],
                        fp32_mul(bf16_to_fp32(gv), bf16_to_fp32(y_ref[16*d +: 16])));
            end
        for (r=0;r<B;r=r+1) for (d=0;d<HIDDEN;d=d+1)
            ref_y[r][16*d +: 16]=fp32_to_bf16(facc_ref[r][d]);

        // DUT: one batched pass
        @(negedge clk); start=1'b1; @(negedge clk); start=1'b0;
        wd=0;
        while (done!==1'b1) begin
            @(negedge clk);
            if (w_req===1'b1) begin dut_beats[cur_expert]=dut_beats[cur_expert]+1; seen[cur_expert]=1'b1; end
            wd=wd+1; if (wd>2000000) begin $display("FATAL: dut timeout B=%0d",B); $fatal(1,"to"); end
        end
        @(negedge clk);

        // CHECK 1: per-token bit-exact output
        for (r=0;r<B;r=r+1) for (d=0;d<HIDDEN;d=d+1) begin
            dv=y_out[16*(HIDDEN*r+d) +: 16]; rv=ref_y[r][16*d +: 16];
            checks=checks+1;
            if (^dv===1'bx) begin
                $display("FAIL[B=%0d P=%0d] row %0d elt %0d: DUT X (%h)",B,PATTERN,r,d,dv); fails=fails+1;
            end else if (dv!==rv) begin
                $display("FAIL[B=%0d P=%0d] row %0d elt %0d: DUT %h != REF %h",B,PATTERN,r,d,dv,rv); fails=fails+1;
            end
        end

        // CHECK 2: amortization -- fetched experts == the union, each ONCE
        union_cnt=0; exp_union=0;
        for (e=0;e<N_EXPERT;e=e+1) begin
            exp_union=0; for (r=0;r<B;r=r+1) if (has_expert(r,e)) exp_union=1;
            if (exp_union) begin
                union_cnt=union_cnt+1;
                if (!seen[e]) begin
                    $display("FAIL[B=%0d P=%0d]: selected expert %0d never fetched",B,PATTERN,e); fails=fails+1;
                end else if (dut_beats[e]!==WBEATS_ONE) begin
                    $display("FAIL[B=%0d P=%0d]: expert %0d fetched %0d beats != one-run %0d",
                             B,PATTERN,e,dut_beats[e],WBEATS_ONE); fails=fails+1;
                end
            end else if (seen[e]) begin
                $display("FAIL[B=%0d P=%0d]: NON-selected expert %0d was fetched",B,PATTERN,e); fails=fails+1;
            end
        end
        $display("  ok[B=%0d P=%0d] %0d rows, union=%0d experts, %0d elts bit-exact, each expert 1x fetch",
                 B, PATTERN, B, union_cnt, B*HIDDEN);

        done_case=1'b1;
    end
endmodule

// ---------------------------------------------------------------------------
//  Top: ONE (B, PATTERN) case, selected at COMPILE TIME via +define.  A single
//  batched_moe+swiglu pair keeps iverilog elaboration fast (multiple heavy MoE
//  instances in one TB blow up iverilog-12 elaboration).  The B-SWEEP is done by
//  the Makefile/driver recompiling this TB per (B,PATTERN):
//     iverilog -DBCOV_B=8 -DBCOV_PAT=3 -DBCOV_SEED=44 ...
//  Sweep (with the committed batched_moe_tb.v B=4 overlap) covers widths
//  {1,2,3,4,8} and patterns {same, distinct, random, overlap}.
// ---------------------------------------------------------------------------
`ifndef BCOV_B
 `define BCOV_B 4
`endif
`ifndef BCOV_PAT
 `define BCOV_PAT 0
`endif
`ifndef BCOV_SEED
 `define BCOV_SEED 1
`endif
module batched_moe_bcov_tb;
    reg clk=1'b0, rst=1'b1; always #5 clk=~clk;

    reg         go;
    wire        done_case;
    wire [31:0] fails, checks;

    bmoe_case #(.B(`BCOV_B), .PATTERN(`BCOV_PAT), .SEED(`BCOV_SEED)) c (
        .clk(clk), .rst(rst), .go(go),
        .done_case(done_case), .fails(fails), .checks(checks) );

    initial begin
        go = 1'b0;
        repeat (4) @(negedge clk); rst=1'b0; @(negedge clk);
        @(negedge clk); go=1'b1; @(negedge clk); go=1'b0;
        wait (done_case===1'b1);
        $display("--------------------------------------------------------------");
        if (fails==0)
            $display("ALL %0d TESTS PASSED  (batched_moe B=%0d PATTERN=%0d: %0d bit-exact elts)",
                     1, `BCOV_B, `BCOV_PAT, checks);
        else begin
            $display("%0d FAILURE(S)  (batched_moe B=%0d PATTERN=%0d)", fails, `BCOV_B, `BCOV_PAT);
            $fatal(1, "batched_moe B-coverage case failed");
        end
        $finish;
    end
endmodule
