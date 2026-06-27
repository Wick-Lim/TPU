`timescale 1ns/1ps
//============================================================================
// moe_router_fp8_tb.v  --  smoke TB for moe_router_fp8
//   Drives a known small FP8 W_g (E4M3 codes + [128,128] bf16 block scale=1.0)
//   and bf16 x; checks the routed TOP-K indices (EXACT) and the renormalized
//   x2.5 weights against an INDEPENDENT fp64 faithful-fp8 golden.  X-aware.
//   The W_g GEMV is FP8; sigmoid/topk/renorm stay bf16 (handled in the DUT).
//============================================================================
module moe_router_fp8_tb;
    localparam integer HIDDEN   = 128;
    localparam integer N_EXPERT = 8;
    localparam integer TOPK     = 2;
    localparam integer KMAX     = 128;
    localparam integer BLK      = 128;
    localparam integer NB       = (KMAX + BLK - 1) / BLK;   // = 1
    localparam integer IDXW     = 3;                        // $clog2(8)
    localparam [31:0]  SCALE    = 32'h40200000;             // 2.5

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst, start;

    reg  [16*HIDDEN-1:0] x_vec;
    wire                 w_req;
    wire [$clog2(KMAX+1)-1:0] w_k;
    reg  [8*N_EXPERT-1:0]     w_col;
    reg  [16*N_EXPERT*NB-1:0] w_scale;
    wire                 busy, done;
    wire [TOPK*IDXW-1:0] sel_idx;
    wire [TOPK*16-1:0]   sel_weight;

    moe_router_fp8 #(
        .HIDDEN(HIDDEN), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .SCALE(SCALE), .KMAX(KMAX), .BLK(BLK)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .x_vec(x_vec),
        .w_req(w_req), .w_k(w_k), .w_col(w_col), .w_scale(w_scale),
        .sel_idx(sel_idx), .sel_weight(sel_weight)
    );

    // ---- the "system": FP8 W_g store + combinational column response ----
    reg [7:0] Wcode [0:HIDDEN-1][0:N_EXPERT-1];   // E4M3 codes
    integer ce;
    always @* begin
        for (ce = 0; ce < N_EXPERT; ce = ce + 1)
            w_col[8*ce +: 8] = Wcode[w_k][ce];
    end

    // ===== helpers =====================================================
    // bf16 of a small exactly-representable real (sign exp mant7).
    function [15:0] bf16;
        input real v;
        reg sgn; integer e; real a, m; reg [7:0] eb; reg [6:0] mb;
        begin
            if (v == 0.0) bf16 = 16'h0000;
            else begin
                sgn = (v < 0.0); a = sgn ? -v : v;
                e = 0;
                while (a >= 2.0) begin a = a/2.0; e = e+1; end
                while (a <  1.0) begin a = a*2.0; e = e-1; end
                m  = (a - 1.0) * 128.0;            // 7 mantissa bits
                eb = (e + 127);
                mb = $rtoi(m + 0.5);
                bf16 = {sgn, eb, mb};
            end
        end
    endfunction

    // E4M3 decode -> real (bias 7; exp==0 subnormal; 0x7F/0xFF NaN unused here).
    function real e4m3r;
        input [7:0] c;
        reg sgn; integer e, m; real v;
        begin
            sgn = c[7]; e = c[6:3]; m = c[2:0];
            if (e == 0)       v = (m/8.0) * (2.0 ** (-6));
            else              v = (1.0 + m/8.0) * (2.0 ** (e - 7));
            e4m3r = sgn ? -v : v;
        end
    endfunction

    // bf16 bits -> real
    function real bf16r;
        input [15:0] b;
        reg sgn; integer e, m; real v;
        begin
            sgn = b[15]; e = b[14:7]; m = b[6:0];
            if (e == 0) v = (m/128.0) * (2.0 ** (-126));
            else        v = (1.0 + m/128.0) * (2.0 ** (e - 127));
            bf16r = sgn ? -v : v;
        end
    endfunction

    real xr [0:HIDDEN-1];          // golden x as real
    real logit_g [0:N_EXPERT-1];
    real sig_g   [0:N_EXPERT-1];

    integer errors = 0;
    integer tests  = 0;

    task clear_w;
        integer k, e;
        begin
            for (k = 0; k < HIDDEN; k = k + 1)
                for (e = 0; e < N_EXPERT; e = e + 1) Wcode[k][e] = 8'h00;
        end
    endtask

    // golden: faithful fp8 logits (our chosen acts/weights are exactly E4M3 so
    // the per-term products are exact -> order-independent), then sigmoid + topk
    // + renorm*2.5.  Returns expected idx[0..1] and weights w[0..1].
    integer exp_i0, exp_i1;
    real    exp_w0, exp_w1, exp_sum;

    task golden;
        integer k, e, b0, b1; real s, best, second;
        begin
            for (e = 0; e < N_EXPERT; e = e + 1) begin
                s = 0.0;
                for (k = 0; k < HIDDEN; k = k + 1)
                    s = s + xr[k] * e4m3r(Wcode[k][e]);   // block scale 1.0
                logit_g[e] = s;
                sig_g[e]   = 1.0 / (1.0 + $exp(-s));
            end
            // top-2 by sigmoid (== by logit), lower-index tie-break.
            best = -1.0e30; b0 = 0;
            for (e = 0; e < N_EXPERT; e = e + 1)
                if (sig_g[e] > best) begin best = sig_g[e]; b0 = e; end
            second = -1.0e30; b1 = 0;
            for (e = 0; e < N_EXPERT; e = e + 1)
                if (e != b0 && sig_g[e] > second) begin second = sig_g[e]; b1 = e; end
            exp_i0  = b0; exp_i1 = b1;
            exp_sum = sig_g[b0] + sig_g[b1];
            exp_w0  = sig_g[b0] / exp_sum * 2.5;
            exp_w1  = sig_g[b1] / exp_sum * 2.5;
        end
    endtask

    // run one token: load x_vec, pulse start, wait done, check.
    task run_token;
        input [255:0] name;
        integer i; real gw0, gw1, gsum; integer gi0, gi1;
        reg [IDXW-1:0] i0, i1; real w0r, w1r;
        begin
            tests = tests + 1;
            golden;  gi0 = exp_i0; gi1 = exp_i1; gw0 = exp_w0; gw1 = exp_w1;
            gsum = gw0 + gw1;

            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;
            // wait for done
            i = 0;
            while (!done && i < 5000) begin @(negedge clk); i = i + 1; end
            if (!done) begin
                $display("  [%0s] TIMEOUT waiting for done", name);
                errors = errors + 1;
            end else begin
                // X-aware: outputs must be fully defined
                if (^sel_idx === 1'bx || ^sel_weight === 1'bx) begin
                    $display("  [%0s] X in outputs sel_idx=%b sel_weight=%h",
                             name, sel_idx, sel_weight);
                    errors = errors + 1;
                end
                i0 = sel_idx[0*IDXW +: IDXW];
                i1 = sel_idx[1*IDXW +: IDXW];
                w0r = bf16r(sel_weight[0*16 +: 16]);
                w1r = bf16r(sel_weight[1*16 +: 16]);
                $display("  [%0s] idx={%0d,%0d} exp={%0d,%0d}  w={%0.4f,%0.4f} exp={%0.4f,%0.4f} sum=%0.4f",
                         name, i0, i1, gi0, gi1, w0r, w1r, gw0, gw1, w0r+w1r);
                // indices EXACT
                if (i0 !== gi0[IDXW-1:0] || i1 !== gi1[IDXW-1:0]) begin
                    $display("  [%0s] INDEX MISMATCH", name);
                    errors = errors + 1;
                end
                // weights within fp8 tol of golden
                if ((w0r-gw0 > 0.10) || (gw0-w0r > 0.10) ||
                    (w1r-gw1 > 0.10) || (gw1-w1r > 0.10)) begin
                    $display("  [%0s] WEIGHT MISMATCH", name);
                    errors = errors + 1;
                end
                // renorm-then-scale invariant: selected weights sum to 2.5
                if ((w0r+w1r) - 2.5 > 0.03 || 2.5 - (w0r+w1r) > 0.03) begin
                    $display("  [%0s] SUM!=2.5 (renorm/scale broken)", name);
                    errors = errors + 1;
                end
                // ordering: slot0 (top score) weight >= slot1 weight
                if (w1r - w0r > 0.001) begin
                    $display("  [%0s] ORDER MISMATCH w0<w1", name);
                    errors = errors + 1;
                end
            end
        end
    endtask

    integer kk;
    initial begin
        // ---- block scales: bf16(1.0) for every expert, block 0 ----
        w_scale = {16*N_EXPERT*NB{1'b0}};
        for (kk = 0; kk < N_EXPERT; kk = kk + 1)
            w_scale[16*(0*N_EXPERT + kk) +: 16] = 16'h3F80;   // 1.0

        // ---- weights (E4M3 codes) ----
        clear_w;
        // token-A active rows (x[0],x[1]) : experts 0,1 win
        Wcode[0][0] = 8'h40; Wcode[1][0] = 8'h40;   // 2.0,2.0 -> logit 4.0  (expert0)
        Wcode[0][1] = 8'h3C; Wcode[1][1] = 8'h3C;   // 1.5,1.5 -> logit 3.0  (expert1)
        // token-B active rows (x[2],x[3]) : experts 5,6 win
        Wcode[2][5] = 8'h40; Wcode[3][5] = 8'h38;   // 2.0,1.0 -> logit 3.0  (expert5)
        Wcode[2][6] = 8'h38; Wcode[3][6] = 8'h38;   // 1.0,1.0 -> logit 2.0  (expert6)

        rst = 1'b1; start = 1'b0; x_vec = {16*HIDDEN{1'b0}};
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ===== Token A : x[0]=x[1]=1.0 -> winners {0,1} =====
        for (kk = 0; kk < HIDDEN; kk = kk + 1) begin
            xr[kk] = 0.0; x_vec[16*kk +: 16] = 16'h0000;
        end
        xr[0] = 1.0; x_vec[16*0 +: 16] = bf16(1.0);
        xr[1] = 1.0; x_vec[16*1 +: 16] = bf16(1.0);
        run_token("token A");

        @(negedge clk);

        // ===== Token B : x[2]=x[3]=1.0 -> winners {5,6} =====
        for (kk = 0; kk < HIDDEN; kk = kk + 1) begin
            xr[kk] = 0.0; x_vec[16*kk +: 16] = 16'h0000;
        end
        xr[2] = 1.0; x_vec[16*2 +: 16] = bf16(1.0);
        xr[3] = 1.0; x_vec[16*3 +: 16] = bf16(1.0);
        run_token("token B");

        if (errors == 0)
            $display("ALL %0d TESTS PASSED", tests);
        else begin
            $display("FAILED: %0d error(s) across %0d tests", errors, tests);
            $fatal(1, "moe_router_fp8 smoke test failed");
        end
        $finish;
    end
endmodule
