`timescale 1ns/1ps
//============================================================================
// moe_router_fp8_pem_tb.v  --  PE_M BATCHING equivalence + weight-share TB for
//                              moe_router_fp8.v   (ACCEL_GLM52 §5/§6)
//----------------------------------------------------------------------------
// VERIFY STANCE  (RTL-vs-RTL, BIT-EXACT)
//   The committed moe_router_fp8_tb.v already pins the PE_M=1 datapath against a
//   FAITHFUL fp8 fp64 golden.  This TB instead pins the PE_M>1 WIDENING: it
//   asserts the contract that makes batching legal --
//
//     "row r of a PE_M=B run is BYTE-IDENTICAL to a standalone PE_M=1 run on
//      row r's token, AND the W_g weight stream is fetched ONCE for all B rows."
//
//   So it drives B token rows two ways and compares:
//     (a) REFERENCE  : B independent PE_M=1 instantiations, one token each.
//     (b) BATCHED    : one PE_M=B instantiation, all B tokens at once.
//   Both share the SAME W_g (E4M3 codes) + block scales (the system answers the
//   pull combinationally from one Wcode store, identical for both DUTs).
//
//   BINDING CHECKS (per row r, 0..B-1):
//     * sel_idx   [row r] === reference sel_idx   (EXACT top-K indices, X-aware)
//     * sel_weight[row r] === reference sel_weight (BIT-EXACT bf16 weights)
//   Because both sides are the SAME RTL fed the SAME data, the match is EXACT
//   (===), not tolerance-based: any per-row cross-talk, a_shift bleed, or shared-
//   weight desync would break it.
//
//   WEIGHT-SHARE CHECK (the whole point of PE_M):
//     Count the w_req pulses (each = one W_g column fetched from the system).
//     A PE_M=1 run streams K=HIDDEN beats -> HIDDEN fetches.  The PE_M=B batched
//     run MUST issue the SAME HIDDEN fetches (one shared stream feeds all B rows),
//     NOT B*HIDDEN.  We assert wreq_batched == wreq_single == HIDDEN, i.e. the
//     weight bandwidth is amortized B -> 1.
//
//   X-AWARE: every routed output is checked for X before the equality compare.
//============================================================================
module moe_router_fp8_pem_tb;

    integer total_tests = 0;
    integer total_fail  = 0;

    // shapes: HIDDEN<=128 so BLK=128 => exactly ONE K-block (NB=1).  Mix of B=2/4,
    // varied N_EXPERT/TOPK incl. ragged HIDDEN and TOPK==N_EXPERT (dense fallback).
    mrpem_harness #(.HIDDEN(8),  .N_EXPERT(8),  .TOPK(2), .B(2), .NRAND(40), .SEED(32'h0BAD_0001)) hA();
    mrpem_harness #(.HIDDEN(16), .N_EXPERT(16), .TOPK(4), .B(4), .NRAND(40), .SEED(32'h0BAD_0002)) hB();
    mrpem_harness #(.HIDDEN(32), .N_EXPERT(32), .TOPK(8), .B(2), .NRAND(25), .SEED(32'h0BAD_0003)) hC();
    mrpem_harness #(.HIDDEN(8),  .N_EXPERT(8),  .TOPK(8), .B(4), .NRAND(25), .SEED(32'h0BAD_0004)) hD();
    mrpem_harness #(.HIDDEN(64), .N_EXPERT(8),  .TOPK(2), .B(4), .NRAND(25), .SEED(32'h0BAD_0005)) hE();
    mrpem_harness #(.HIDDEN(7),  .N_EXPERT(5),  .TOPK(3), .B(3), .NRAND(25), .SEED(32'h0BAD_0006)) hF();

    initial begin
        hA.run; total_tests = total_tests + hA.n_tests; total_fail = total_fail + hA.n_fail;
        hB.run; total_tests = total_tests + hB.n_tests; total_fail = total_fail + hB.n_fail;
        hC.run; total_tests = total_tests + hC.n_tests; total_fail = total_fail + hC.n_fail;
        hD.run; total_tests = total_tests + hD.n_tests; total_fail = total_fail + hD.n_fail;
        hE.run; total_tests = total_tests + hE.n_tests; total_fail = total_fail + hE.n_fail;
        hF.run; total_tests = total_tests + hF.n_tests; total_fail = total_fail + hF.n_fail;

        if (total_fail != 0) begin
            $display("FAILED: %0d of %0d tests mismatched", total_fail, total_tests);
            $fatal(1, "moe_router_fp8_pem_tb: MISMATCH");
        end
        $display("ALL %0d TESTS PASSED", total_tests);
        $finish;
    end
endmodule

//============================================================================
// mrpem_harness : a PE_M=B batched DUT + a PE_M=1 reference DUT of a given shape,
//   the random/directed driver, and the bit-exact + weight-share checks.
//============================================================================
module mrpem_harness #(
    parameter integer HIDDEN   = 8,
    parameter integer N_EXPERT = 8,
    parameter integer TOPK     = 2,
    parameter integer B        = 2,     // batch rows for the widened DUT
    parameter integer NRAND    = 40,
    parameter [31:0]  SEED     = 32'hDEAD_BEEF
);
    localparam integer IDXW  = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT);
    localparam [31:0]  SCALE = 32'h40200000;          // 2.5 fp32 (== DUT default)
    localparam integer KMAX  = 128;                   // >= HIDDEN; BLK=128 => NB=1
    localparam integer BLK   = 128;
    localparam integer NB    = (KMAX + BLK - 1) / BLK;   // = 1
    localparam integer KW    = $clog2(KMAX+1);
    localparam integer ROWW  = (HIDDEN <= 1) ? 1 : $clog2(HIDDEN);

    integer n_tests; integer n_fail;

    // ---- clock / reset ----
    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst;

    // ================= shared model storage =================
    reg [7:0]  Wcode    [0:HIDDEN-1][0:N_EXPERT-1];   // E4M3 codes (W_g) -- shared
    reg [15:0] xb       [0:B-1][0:HIDDEN-1];          // B bf16 tokens
    reg [15:0] wscale_bf[0:N_EXPERT-1];               // bf16 block-0 scale per expert

    // ================= REFERENCE DUT (PE_M=1) =================
    reg                       start_s;
    wire                      busy_s, done_s;
    reg  [16*HIDDEN-1:0]      x_vec_s;
    wire                      w_req_s;
    wire [KW-1:0]             w_k_s;
    reg  [8*N_EXPERT-1:0]     w_col_s;
    reg  [16*N_EXPERT*NB-1:0] w_scale_s;
    wire [TOPK*IDXW-1:0]      sel_idx_s;
    wire [TOPK*16-1:0]        sel_weight_s;

    integer ce_s;
    always @* begin
        for (ce_s = 0; ce_s < N_EXPERT; ce_s = ce_s + 1)
            w_col_s[8*ce_s +: 8] = Wcode[w_k_s[ROWW-1:0]][ce_s];
    end

    moe_router_fp8 #(.HIDDEN(HIDDEN), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
                     .SCALE(SCALE), .KMAX(KMAX), .BLK(BLK), .PE_M(1)) dut_s (
        .clk(clk), .rst(rst),
        .start(start_s), .busy(busy_s), .done(done_s),
        .x_vec(x_vec_s),
        .w_req(w_req_s), .w_k(w_k_s), .w_col(w_col_s), .w_scale(w_scale_s),
        .sel_idx(sel_idx_s), .sel_weight(sel_weight_s)
    );

    // ================= BATCHED DUT (PE_M=B) =================
    reg                        start_w;
    wire                       busy_w, done_w;
    reg  [16*HIDDEN*B-1:0]     x_vec_w;
    wire                       w_req_w;
    wire [KW-1:0]              w_k_w;
    reg  [8*N_EXPERT-1:0]      w_col_w;
    reg  [16*N_EXPERT*NB-1:0]  w_scale_w;
    wire [TOPK*IDXW*B-1:0]     sel_idx_w;
    wire [TOPK*16*B-1:0]       sel_weight_w;

    integer ce_w;
    always @* begin
        for (ce_w = 0; ce_w < N_EXPERT; ce_w = ce_w + 1)
            w_col_w[8*ce_w +: 8] = Wcode[w_k_w[ROWW-1:0]][ce_w];
    end

    moe_router_fp8 #(.HIDDEN(HIDDEN), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
                     .SCALE(SCALE), .KMAX(KMAX), .BLK(BLK), .PE_M(B)) dut_w (
        .clk(clk), .rst(rst),
        .start(start_w), .busy(busy_w), .done(done_w),
        .x_vec(x_vec_w),
        .w_req(w_req_w), .w_k(w_k_w), .w_col(w_col_w), .w_scale(w_scale_w),
        .sel_idx(sel_idx_w), .sel_weight(sel_weight_w)
    );

    // captured reference results (per row) + the single-run weight-fetch count
    reg [TOPK*IDXW-1:0] ref_idx    [0:B-1];
    reg [TOPK*16-1:0]   ref_weight [0:B-1];
    integer             wreq_single;       // w_req pulses in ONE PE_M=1 run (== HIDDEN)

    // ===================================================================
    //  random model generators (same distributions as the committed TB)
    // ===================================================================
    reg [31:0] lfsr;
    function automatic [31:0] lfsr_step;
        begin
            lfsr = (lfsr >> 1) ^ ({32{lfsr[0]}} & 32'hD0000001);
            lfsr_step = lfsr;
        end
    endfunction
    // x bf16: exponent in [122,128]; sign random.
    function automatic [15:0] rnd_x_bf16;
        reg [31:0] r; reg [7:0] e;
        begin r = lfsr_step(); e = 8'd122 + (r[10:8] % 8'd7); rnd_x_bf16 = {r[15], e, r[6:0]}; end
    endfunction
    // x bf16 with a controllable exponent base (to force big per-row magnitude spread).
    function automatic [15:0] rnd_x_bf16_e(input [7:0] ebase);
        reg [31:0] r;
        begin r = lfsr_step(); rnd_x_bf16_e = {r[15], ebase + (r[9:8] % 8'd3), r[6:0]}; end
    endfunction
    // W_g E4M3 code: exp field in [0,7]; sign random; no NaN.
    function automatic [7:0] rnd_w_e4m3;
        reg [31:0] r; reg [3:0] e; reg [2:0] m;
        begin r = lfsr_step(); e = {1'b0, r[6:4]}; m = r[2:0]; rnd_w_e4m3 = {r[15], e, m}; end
    endfunction
    // block scale bf16: positive, exp in [125,129].
    function automatic [15:0] rnd_scale_bf16;
        reg [31:0] r; reg [7:0] e;
        begin r = lfsr_step(); e = 8'd125 + (r[18:16] % 8'd5); rnd_scale_bf16 = {1'b0, e, r[6:0]}; end
    endfunction

    // fill W_g + block scales (shared); fill each of the B rows with random x.
    task fill_random;
        integer k, e, b;
        begin
            for (k = 0; k < HIDDEN; k = k + 1)
                for (e = 0; e < N_EXPERT; e = e + 1) Wcode[k][e] = rnd_w_e4m3();
            for (e = 0; e < N_EXPERT; e = e + 1) wscale_bf[e] = rnd_scale_bf16();
            for (b = 0; b < B; b = b + 1)
                for (k = 0; k < HIDDEN; k = k + 1) xb[b][k] = rnd_x_bf16();
        end
    endtask

    // fill rows with WILDLY different per-row magnitudes (each row b an exponent
    // base far from the others) -> stresses the INDEPENDENT per-row a_shift (xsh).
    task fill_spread;
        integer k, e, b; reg [7:0] eb;
        begin
            for (k = 0; k < HIDDEN; k = k + 1)
                for (e = 0; e < N_EXPERT; e = e + 1) Wcode[k][e] = rnd_w_e4m3();
            for (e = 0; e < N_EXPERT; e = e + 1) wscale_bf[e] = rnd_scale_bf16();
            for (b = 0; b < B; b = b + 1) begin
                eb = 8'd112 + (b * 8'd6);                 // rows span ~6 binades apart
                for (k = 0; k < HIDDEN; k = k + 1) xb[b][k] = rnd_x_bf16_e(eb);
            end
        end
    endtask

    // fill with one ALL-ZERO row (emax=0 -> xsh=0 path) interleaved with live rows.
    task fill_zero_mix;
        integer k, e, b;
        begin
            for (k = 0; k < HIDDEN; k = k + 1)
                for (e = 0; e < N_EXPERT; e = e + 1) Wcode[k][e] = rnd_w_e4m3();
            for (e = 0; e < N_EXPERT; e = e + 1) wscale_bf[e] = rnd_scale_bf16();
            for (b = 0; b < B; b = b + 1)
                for (k = 0; k < HIDDEN; k = k + 1)
                    xb[b][k] = (b == 0) ? 16'h0000 : rnd_x_bf16();
        end
    endtask

    // ===================================================================
    //  run ONE reference PE_M=1 pass for row `b`; capture idx/weight, and on
    //  row 0 also tally the single-run weight-fetch count.
    // ===================================================================
    task run_ref_row;
        input integer b;
        integer t, i, cnt;
        begin
            for (t = 0; t < HIDDEN;   t = t + 1) x_vec_s[16*t +: 16] = xb[b][t];
            for (t = 0; t < N_EXPERT; t = t + 1) w_scale_s[16*t +: 16] = wscale_bf[t];

            @(negedge clk); start_s = 1'b1;
            @(negedge clk); start_s = 1'b0;
            i = 0; cnt = 0;
            while (!done_s && i < 20000) begin
                @(negedge clk);
                if (w_req_s) cnt = cnt + 1;     // count W_g column fetches
                i = i + 1;
            end
            @(negedge clk);                     // sample held outputs
            ref_idx[b]    = sel_idx_s;
            ref_weight[b] = sel_weight_s;
            if (b == 0) wreq_single = cnt;
        end
    endtask

    // ===================================================================
    //  run the BATCHED PE_M=B pass; count its weight fetches; CHECK each row vs
    //  the captured reference + assert the weight stream is shared (B -> 1).
    // ===================================================================
    task run_batched_and_check;
        input [255:0] label;
        integer b, t, i, cnt; reg fail_this, sawx;
        reg [TOPK*IDXW-1:0] gi_w; reg [TOPK*16-1:0] gw_w;
        begin
            n_tests = n_tests + 1;
            for (b = 0; b < B; b = b + 1)
                for (t = 0; t < HIDDEN; t = t + 1)
                    x_vec_w[16*(HIDDEN*b + t) +: 16] = xb[b][t];
            for (t = 0; t < N_EXPERT; t = t + 1) w_scale_w[16*t +: 16] = wscale_bf[t];

            @(negedge clk); start_w = 1'b1;
            @(negedge clk); start_w = 1'b0;
            i = 0; cnt = 0;
            while (!done_w && i < 20000) begin
                @(negedge clk);
                if (w_req_w) cnt = cnt + 1;
                i = i + 1;
            end
            fail_this = 1'b0;
            if (!done_w) begin
                $display("FAIL %0s [%0dx%0d k%0d B%0d]: batched TIMEOUT",
                         label, HIDDEN, N_EXPERT, TOPK, B);
                n_fail = n_fail + 1; disable run_batched_and_check;
            end
            @(negedge clk);                     // sample held outputs

            // ---- per-row BIT-EXACT equivalence to the PE_M=1 reference ----
            for (b = 0; b < B; b = b + 1) begin
                gi_w = sel_idx_w   [TOPK*IDXW*b +: TOPK*IDXW];
                gw_w = sel_weight_w[TOPK*16*b   +: TOPK*16];
                sawx = (^gi_w === 1'bx) || (^gw_w === 1'bx);
                if (sawx) begin
                    $display("FAIL %0s [%0dx%0d k%0d B%0d] row %0d: X in batched outputs idx=%b w=%h",
                             label, HIDDEN, N_EXPERT, TOPK, B, b, gi_w, gw_w);
                    fail_this = 1'b1;
                end else begin
                    if (gi_w !== ref_idx[b]) begin
                        $display("FAIL %0s [%0dx%0d k%0d B%0d] row %0d: sel_idx %h != ref %h",
                                 label, HIDDEN, N_EXPERT, TOPK, B, b, gi_w, ref_idx[b]);
                        fail_this = 1'b1;
                    end
                    if (gw_w !== ref_weight[b]) begin
                        $display("FAIL %0s [%0dx%0d k%0d B%0d] row %0d: sel_weight %h != ref %h",
                                 label, HIDDEN, N_EXPERT, TOPK, B, b, gw_w, ref_weight[b]);
                        fail_this = 1'b1;
                    end
                end
            end

            // ---- WEIGHT-SHARE: batched fetch count == single-run count (== HIDDEN),
            //      NOT B*HIDDEN.  One shared W_g stream feeds all B rows. ----
            if (cnt != wreq_single) begin
                $display("FAIL %0s [%0dx%0d k%0d B%0d]: batched w_req=%0d != single %0d (weight stream NOT shared)",
                         label, HIDDEN, N_EXPERT, TOPK, B, cnt, wreq_single);
                fail_this = 1'b1;
            end
            if (wreq_single != HIDDEN) begin
                $display("FAIL %0s [%0dx%0d k%0d B%0d]: single w_req=%0d != HIDDEN %0d",
                         label, HIDDEN, N_EXPERT, TOPK, B, wreq_single, HIDDEN);
                fail_this = 1'b1;
            end

            if (fail_this) n_fail = n_fail + 1;
            else $display("ok   %0s [%0dx%0d k%0d B%0d]  w_req batched=%0d single=%0d (B*single=%0d amortized->%0d)",
                          label, HIDDEN, N_EXPERT, TOPK, B, cnt, wreq_single, B*wreq_single, cnt);
        end
    endtask

    // run all B reference rows, then the batched run + checks.
    task do_case;
        input [255:0] label;
        integer b;
        begin
            for (b = 0; b < B; b = b + 1) run_ref_row(b);
            run_batched_and_check(label);
        end
    endtask

    // ===================================================================
    //  RUN
    // ===================================================================
    task run;
        integer r;
        begin
            n_tests = 0; n_fail = 0;
            lfsr = SEED;
            start_s = 1'b0; start_w = 1'b0;
            x_vec_s = {16*HIDDEN{1'b0}}; x_vec_w = {16*HIDDEN*B{1'b0}};
            w_scale_s = {16*N_EXPERT*NB{1'b0}}; w_scale_w = {16*N_EXPERT*NB{1'b0}};
            rst = 1'b1;
            @(negedge clk); @(negedge clk);
            rst = 1'b0;
            @(negedge clk);

            // directed: per-row magnitude spread (independent xsh stress)
            fill_spread;   do_case("spread");
            // directed: an all-zero row mixed with live rows (xsh=0 path)
            fill_zero_mix; do_case("zero-mix");

            // random batches
            for (r = 0; r < NRAND; r = r + 1) begin
                fill_random;
                do_case("random");
            end
        end
    endtask
endmodule
