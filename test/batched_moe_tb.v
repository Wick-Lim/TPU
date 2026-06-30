`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// batched_moe_tb.v
//   SMOKE + BINDING check for batched_moe.v (ULTRA_PERF#1 expert-grouped MoE).
//
//   SCENARIO: B = PE_M = 4 token rows route (TOPK=2 of N_EXPERT=8) with HEAVY
//   OVERLAP so the UNION of selected experts {1,3,5} is far smaller than the
//   naive B*TOPK = 8 per-token fetches:
//        token0 -> {1,3}   token1 -> {1,5}   token2 -> {3,5}   token3 -> {1,3}
//
//   BINDING (result identity): for EACH token r, batched_moe's bf16 output must
//   equal -- BIT-EXACT (!==) -- the single-token reference
//        y_r = Σ_{e in TopK(r), ASCENDING e} gate_{r,e} * expert_e(x_r),
//   where expert_e is the SAME committed RTL run at PE_M=1 on row r's activation
//   against expert e's weights, and the gate*accumulate uses the SAME fp32
//   combine numerics as the DUT (bf16->fp32, fp32_mul gate, fp32_add, ->bf16).
//   Both DUT and reference read ONE shared per-expert weight ROM (responder keyed
//   on the DUT's cur_expert / the TB's ref_eid) -> byte-identical weights, so the
//   only legal output is bit-identical.
//
//   AMORTIZATION (the whole point): assert the DUT fetches the UNION ONCE EACH --
//   exactly the experts {1,3,5}, never a non-selected one, and each union expert
//   pulls EXACTLY one single-expert weight stream (its w_req beat count == ONE
//   PE_M=1 run's beat count).  Total = 3 fetch-streams, not the naive 8.
//
//   X-AWARE: any X in a captured DUT output bit is a hard failure.
//   Emits "ALL <N> TESTS PASSED" and $fatal on any mismatch / X / bad union.
//============================================================================
module batched_moe_tb;

    // ---- dimensions (MoE expert: INTER < HIDDEN) ----
    localparam integer HIDDEN   = 64;
    localparam integer INTER    = 32;
    localparam integer TN       = 4;
    localparam integer KMAX     = 64;
    localparam integer BLK      = 32;
    localparam integer N_EXPERT = 8;
    localparam integer TOPK     = 2;
    localparam integer B        = 4;     // PE_M batch
    localparam integer IDXW     = (N_EXPERT<=1)?1:$clog2(N_EXPERT);
    localparam integer NB       = (KMAX+BLK-1)/BLK;             // K-blocks
    localparam integer NOB      = ((HIDDEN>INTER?HIDDEN:INTER)+BLK-1)/BLK; // out-blocks
    localparam integer MAXD     = (HIDDEN>INTER?HIDDEN:INTER);
    localparam integer KW       = $clog2(KMAX+1);
    localparam integer GW       = $clog2((INTER>HIDDEN?INTER:HIDDEN)/TN + 1);
    localparam [1:0]   SEL_GATE = 2'd0, SEL_DOWN = 2'd2;

    `include "glm_fp.vh"

    reg clk = 1'b0, rst = 1'b1;
    always #5 clk = ~clk;

    integer pass_cnt = 0, fail_cnt = 0;

    // =====================================================================
    //  Per-expert weight ROM (E4M3 codes + bf16 [128,128] block scales)
    // =====================================================================
    reg [ 7:0] Wgate [0:N_EXPERT-1][0:MAXD-1][0:MAXD-1];  // [expert][out][k]
    reg [ 7:0] Wup   [0:N_EXPERT-1][0:MAXD-1][0:MAXD-1];
    reg [ 7:0] Wdown [0:N_EXPERT-1][0:MAXD-1][0:MAXD-1];
    reg [15:0] SCgate[0:N_EXPERT-1][0:NOB-1][0:NB-1];     // [expert][out-blk][k-blk]
    reg [15:0] SCup  [0:N_EXPERT-1][0:NOB-1][0:NB-1];
    reg [15:0] SCdown[0:N_EXPERT-1][0:NOB-1][0:NB-1];

    // token rows + routing
    reg [15:0] XR     [0:B-1][0:HIDDEN-1];
    reg [IDXW-1:0] tok_idx [0:B-1][0:TOPK-1];   // selected expert ids
    reg [15:0]     tok_gat [0:B-1][0:TOPK-1];   // bf16 routed gates

    // =====================================================================
    //  RNG helpers (bounded, no NaN) -- same recipe as the committed fp8 TBs.
    // =====================================================================
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

    integer ge, gi, gk, gb, gob;
    task gen_scenario;
        begin
            for (ge=0; ge<N_EXPERT; ge=ge+1) begin
                for (gi=0; gi<MAXD; gi=gi+1)
                    for (gk=0; gk<MAXD; gk=gk+1) begin
                        Wgate[ge][gi][gk] = rnd_e4m3_b(4,9);
                        Wup  [ge][gi][gk] = rnd_e4m3_b(4,9);
                        Wdown[ge][gi][gk] = rnd_e4m3_b(4,9);
                    end
                for (gob=0; gob<NOB; gob=gob+1)
                    for (gb=0; gb<NB; gb=gb+1) begin
                        SCgate[ge][gob][gb] = rnd_pos_bf16(122,124);
                        SCup  [ge][gob][gb] = rnd_pos_bf16(122,124);
                        SCdown[ge][gob][gb] = rnd_pos_bf16(122,124);
                    end
            end
            for (gi=0; gi<B; gi=gi+1)
                for (gk=0; gk<HIDDEN; gk=gk+1) XR[gi][gk] = rnd_bf16(125,129);

            // OVERLAPPING routing -> union {1,3,5}; per-token TopK ascending in idx.
            tok_idx[0][0]=3'd1; tok_idx[0][1]=3'd3;
            tok_idx[1][0]=3'd1; tok_idx[1][1]=3'd5;
            tok_idx[2][0]=3'd3; tok_idx[2][1]=3'd5;
            tok_idx[3][0]=3'd1; tok_idx[3][1]=3'd3;
            for (gi=0; gi<B; gi=gi+1)
                for (gk=0; gk<TOPK; gk=gk+1)
                    tok_gat[gi][gk] = rnd_pos_bf16(126,128);   // gates ~0.5..2
        end
    endtask

    // =====================================================================
    //  DUT : batched_moe at PE_M = B
    // =====================================================================
    reg                        start;
    wire                       busy, done;
    reg  [16*HIDDEN*B-1:0]     x_vec;
    reg  [TOPK*IDXW*B-1:0]     sel_idx;
    reg  [TOPK*16*B-1:0]       sel_weight;
    wire [IDXW-1:0]            cur_expert;
    wire                       w_req;
    wire [1:0]                 w_sel;
    wire [GW-1:0]              w_grp;
    wire [KW-1:0]              w_k;
    reg  [8*TN-1:0]            w_col, w_col_up;
    reg  [16*TN*NB-1:0]        w_scale_g, w_scale_u;
    wire [16*HIDDEN*B-1:0]     y_out;

    batched_moe #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN), .KMAX(KMAX), .BLK(BLK),
                  .N_EXPERT(N_EXPERT), .TOPK(TOPK), .PE_M(B)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .x_vec(x_vec), .sel_idx(sel_idx), .sel_weight(sel_weight),
        .cur_expert(cur_expert), .w_req(w_req), .w_sel(w_sel),
        .w_grp(w_grp), .w_k(w_k),
        .w_col(w_col), .w_col_up(w_col_up),
        .w_scale_g(w_scale_g), .w_scale_u(w_scale_u),
        .y_out(y_out) );

    // =====================================================================
    //  REFERENCE : ONE PE_M=1 swiglu_expert_fp8 (the SAME committed RTL).
    //  Driven per (token, expert) from x_vec_1 with expert ref_eid's weights.
    // =====================================================================
    reg                  start_ref;
    wire                 busy_ref, done_ref, wreq_ref;
    wire [1:0]           wsel_ref;
    wire [GW-1:0]        wgrp_ref;
    wire [KW-1:0]        wk_ref;
    reg  [8*TN-1:0]      wcol_ref, wcolup_ref;
    reg  [16*TN*NB-1:0]  wscG_ref, wscU_ref;
    reg  [16*HIDDEN-1:0] x_vec_1;
    wire [16*HIDDEN-1:0] y_ref;
    reg  [IDXW-1:0]      ref_eid;          // which expert the reference streams

    swiglu_expert_fp8 #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN),
                        .KMAX(KMAX), .BLK(BLK), .PE_M(1)) dut_ref (
        .clk(clk), .rst(rst), .start(start_ref), .busy(busy_ref), .done(done_ref),
        .x_vec(x_vec_1),
        .w_req(wreq_ref), .w_sel(wsel_ref), .w_grp(wgrp_ref), .w_k(wk_ref),
        .w_col(wcol_ref), .w_col_up(wcolup_ref),
        .w_scale_g(wscG_ref), .w_scale_u(wscU_ref), .y_out(y_ref) );

    // =====================================================================
    //  ONE combinational weight responder: serves BOTH the DUT (keyed on the
    //  DUT's cur_expert) and the reference (keyed on ref_eid) from the shared
    //  per-expert ROM -> byte-identical weights.  WGRP*TN+rt is the output row;
    //  its [128,128]-block out-index is (WGRP*TN+rt)/BLK, K-block is rb.
    // =====================================================================
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

    // =====================================================================
    //  Reference store + golden combine (fp32, ascending expert index).
    // =====================================================================
    reg [31:0]           facc_ref [0:B-1][0:HIDDEN-1];  // per-token fp32 accumulator
    reg [16*HIDDEN-1:0]  ref_y    [0:B-1];              // finalized bf16 golden
    integer WBEATS_ONE;                                  // w_req beats of ONE ref run

    // gate of token r for expert e (0 if not selected)
    function [15:0] gate_of;
        input integer r; input integer e;
        integer t;
        begin
            gate_of = 16'h0000;
            for (t=0; t<TOPK; t=t+1)
                if (tok_idx[r][t] == e[IDXW-1:0]) gate_of = tok_gat[r][t];
        end
    endfunction
    function has_expert;
        input integer r; input integer e;
        integer t;
        begin
            has_expert = 1'b0;
            for (t=0; t<TOPK; t=t+1) if (tok_idx[r][t] == e[IDXW-1:0]) has_expert = 1'b1;
        end
    endfunction

    // =====================================================================
    //  Drive a single PE_M=1 reference run on row `rr` with expert `ee`.
    //  Captures y_ref and (first time) freezes WBEATS_ONE.
    // =====================================================================
    integer wd, o, rr, ee, beats;
    reg first_ref;
    task run_ref;
        input integer rr_i; input integer ee_i;
        begin
            for (o=0;o<HIDDEN;o=o+1) x_vec_1[16*o +: 16] = XR[rr_i][o];
            ref_eid = ee_i[IDXW-1:0];
            @(negedge clk);                       // settle responder
            @(negedge clk); start_ref = 1'b1;
            @(negedge clk); start_ref = 1'b0;
            wd=0; beats=0;
            while (done_ref !== 1'b1) begin
                @(negedge clk);
                if (wreq_ref === 1'b1) beats = beats + 1;
                wd = wd + 1;
                if (wd > 600000) begin $display("FATAL: ref timeout"); $fatal(1,"to"); end
            end
            if (first_ref) begin WBEATS_ONE = beats; first_ref = 1'b0; end
            @(negedge clk);
        end
    endtask

    // =====================================================================
    //  DUT run bookkeeping: per-expert w_req beat counter + union set.
    // =====================================================================
    integer dut_beats [0:N_EXPERT-1];
    reg     seen      [0:N_EXPERT-1];

    integer t, d, e, r, total_dut_beats, union_cnt;
    reg [15:0] gv, dv, rv;

    initial begin
        // ---------------- reset ----------------
        start = 1'b0; start_ref = 1'b0; ref_eid = '0;
        x_vec = '0; sel_idx = '0; sel_weight = '0; x_vec_1 = '0;
        first_ref = 1'b1; WBEATS_ONE = 0;
        for (e=0;e<N_EXPERT;e=e+1) begin dut_beats[e]=0; seen[e]=1'b0; end
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        gen_scenario;

        // ---------------- build the packed DUT routing buses ----------------
        for (r=0;r<B;r=r+1)
            for (t=0;t<TOPK;t=t+1) begin
                sel_idx[IDXW*(TOPK*r+t) +: IDXW] = tok_idx[r][t];
                sel_weight[16*(TOPK*r+t) +: 16]  = tok_gat[r][t];
            end
        for (r=0;r<B;r=r+1)
            for (d=0;d<HIDDEN;d=d+1) x_vec[16*(HIDDEN*r+d) +: 16] = XR[r][d];

        // ---------------- GOLDEN: per token, ascending expert index ----------
        for (r=0;r<B;r=r+1)
            for (d=0;d<HIDDEN;d=d+1) facc_ref[r][d] = 32'h0000_0000;
        for (e=0;e<N_EXPERT;e=e+1)           // ASCENDING expert index (== DUT order)
            for (r=0;r<B;r=r+1)
                if (has_expert(r,e)) begin
                    run_ref(r, e);           // y_ref = expert_e(x_r), PE_M=1 RTL
                    gv = gate_of(r,e);
                    for (d=0;d<HIDDEN;d=d+1)
                        facc_ref[r][d] = fp32_add(
                            facc_ref[r][d],
                            fp32_mul(bf16_to_fp32(gv),
                                     bf16_to_fp32(y_ref[16*d +: 16])));
                end
        for (r=0;r<B;r=r+1)
            for (d=0;d<HIDDEN;d=d+1)
                ref_y[r][16*d +: 16] = fp32_to_bf16(facc_ref[r][d]);

        // ---------------- DUT: ONE batched MoE pass over B rows --------------
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;
        wd=0;
        while (done !== 1'b1) begin
            @(negedge clk);
            if (w_req === 1'b1) begin
                dut_beats[cur_expert] = dut_beats[cur_expert] + 1;
                seen[cur_expert] = 1'b1;
            end
            wd = wd + 1;
            if (wd > 2000000) begin $display("FATAL: dut timeout"); $fatal(1,"to"); end
        end
        @(negedge clk);

        // ---------------- CHECK 1: per-token bit-exact output -----------------
        for (r=0;r<B;r=r+1)
            for (d=0;d<HIDDEN;d=d+1) begin
                dv = y_out[16*(HIDDEN*r+d) +: 16];
                rv = ref_y[r][16*d +: 16];
                if (^dv === 1'bx) begin
                    $display("FAIL row %0d elt %0d: DUT output is X (%h)", r, d, dv);
                    fail_cnt = fail_cnt + 1;
                end else if (dv !== rv) begin
                    $display("FAIL row %0d elt %0d: DUT %h != REF %h", r, d, dv, rv);
                    fail_cnt = fail_cnt + 1;
                end else
                    pass_cnt = pass_cnt + 1;
            end

        // ---------------- CHECK 2: UNION fetched ONCE EACH --------------------
        // exactly {1,3,5} fetched; none else; each == one single-expert stream.
        union_cnt = 0; total_dut_beats = 0;
        for (e=0;e<N_EXPERT;e=e+1) begin
            if (seen[e]) begin
                union_cnt = union_cnt + 1;
                total_dut_beats = total_dut_beats + dut_beats[e];
                // union membership must be a real selection
                if (!(e==1 || e==3 || e==5)) begin
                    $display("FAIL: fetched non-union expert %0d", e);
                    fail_cnt = fail_cnt + 1;
                end else if (dut_beats[e] != WBEATS_ONE) begin
                    $display("FAIL: expert %0d fetched %0d beats != one-stream %0d (refetched?)",
                             e, dut_beats[e], WBEATS_ONE);
                    fail_cnt = fail_cnt + 1;
                end else
                    pass_cnt = pass_cnt + 1;
            end else if (e==1 || e==3 || e==5) begin
                $display("FAIL: union expert %0d was never fetched", e);
                fail_cnt = fail_cnt + 1;
            end
        end
        if (union_cnt == 3) pass_cnt = pass_cnt + 1;
        else begin
            $display("FAIL: union size %0d != 3", union_cnt);
            fail_cnt = fail_cnt + 1;
        end

        $display("----------------------------------------------------------------");
        $display("UNION = {1,3,5} (%0d experts); naive per-token fetches = B*TOPK = %0d",
                 union_cnt, B*TOPK);
        $display("weight-pull beats: batched total = %0d  vs  naive = %0d  (one stream = %0d)",
                 total_dut_beats, (B*TOPK)*WBEATS_ONE, WBEATS_ONE);
        $display("----------------------------------------------------------------");

        if (fail_cnt == 0)
            $display("ALL %0d TESTS PASSED", pass_cnt);
        else begin
            $display("FAILED: %0d of %0d checks failed", fail_cnt, pass_cnt+fail_cnt);
            $fatal(1, "batched_moe_tb mismatch");
        end
        $finish;
    end
endmodule
