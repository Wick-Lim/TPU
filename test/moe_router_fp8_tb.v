`timescale 1ns/1ps
//============================================================================
// moe_router_fp8_tb.v  --  INDEPENDENT fp64 FAITHFUL-FP8 golden TB for
//                          moe_router_fp8.v   (ACCEL_GLM52 §5/§6)
//----------------------------------------------------------------------------
// VERIFY STANCE (per the GLM-5.2-FP8 contract): we check the DUT against a
// FAITHFUL fp8 reference that performs the SAME fp8 math the hardware does --
// NOT an fp32/fp64 "ideal".  The golden is written from the spec, NOT
// transcribed from the RTL:
//
//   DYNAMIC ACTIVATION QUANT (per-token pow2 scale, activation_scheme=dynamic):
//     emax  = max bf16-exponent field over x        (pure exponent max)
//     ash   = (emax==0) ? 0 : clamp_{[-128,127]}(134 - emax)
//   GATE GEMV (logits = x @ W_g), the FP8-NATIVE part, modelled EXACTLY:
//     a_q[k] = E4M3( bf16(x[k]) * 2^ash )            (RNE + saturate to +/-448)
//     prod   = e4m3_real(a_q[k]) * e4m3_real(W_g[k,e])   (EXACT fp8 product)
//     blkacc = SUM_k prod                            (single [128,128] K-block)
//     logit  = bf16( blkacc * w_scale[e] * 2^-ash )  (block dequant + undo ash)
//   bf16 TAIL (NOT weight matmuls -> NOT quantized; same as moe_router):
//     gate_e = bf16( sigmoid(logit_e) )
//     (idx,g)= TOP-K(gate_e), LOWER-INDEX tie-break  (== topk_select)
//     s      = SUM_{j in TopK} gate_j
//     w_j    = bf16( (gate_j / s) * 2.5 )            (renorm THEN x2.5)
//
//   All E4M3 encode/decode, the bf16 grid, the activation pow2 scale, and the
//   block dequant are modelled to the BIT; the only difference from the DUT is
//   that the fp8 products are reduced in fp64 here (the DUT reduces in fp32 and
//   uses a Quake-seed rsqrt reciprocal).  So the residual DUT-vs-golden error is
//   ONLY fp32-accumulation-order + rsqrt + the final bf16 narrow -- never the
//   fp8 quantization itself (which the golden reproduces exactly).  The fp8
//   logit error is therefore NOT hidden by a loose tolerance.
//
// BINDING CHECKS
//   * SELECTED INDICES : EXACT match of all TOPK slots (DUT's emitted order),
//     with a principled exception ONLY for routing-INDISTINGUISHABLE pairs
//     (gates EQUAL on the bf16 grid -- a true tie topk breaks by lower index --
//     or BOTH gates below a negligible floor GTOL=2^-12).  A swap of two
//     non-equal, non-negligible gates still FAILS.
//   * WEIGHTS          : |w_dut - w_gold| <= WTOL_REL*|w_gold| + WTOL_ABS,
//     WTOL_REL=2^-7 (one bf16 ULP for the final narrow + a half-ULP-class margin
//     for the rsqrt reciprocal and fp32-vs-fp64 accumulation order), WTOL_ABS=
//     2^-9 (an absolute floor so negligible tail weights are not judged by a
//     meaningless relative error).
//   * INVARIANT        : the TOPK selected weights sum to SCALE=2.5.
//
// X-AWARE: every routed output is checked for X before use.
//============================================================================
module moe_router_fp8_tb;

    integer total_tests = 0;
    integer total_fail  = 0;
    real    g_worst_ratio; initial g_worst_ratio = 0.0;

    // ---- shapes (HIDDEN<=128 so BLK=128 => exactly ONE K-block, NB=1) ----
    mr8_harness #(.HIDDEN(8),  .N_EXPERT(8),  .TOPK(2), .NRAND(40), .SEED(32'h1111_0001)) hA();
    mr8_harness #(.HIDDEN(16), .N_EXPERT(16), .TOPK(4), .NRAND(40), .SEED(32'h2222_0002)) hB();
    mr8_harness #(.HIDDEN(32), .N_EXPERT(32), .TOPK(8), .NRAND(30), .SEED(32'h3333_0003)) hC();
    mr8_harness #(.HIDDEN(8),  .N_EXPERT(8),  .TOPK(8), .NRAND(20), .SEED(32'h4444_0004)) hD();
    mr8_harness #(.HIDDEN(64), .N_EXPERT(8),  .TOPK(2), .NRAND(30), .SEED(32'h5555_0005)) hE();

    initial begin
        hA.run; total_tests = total_tests + hA.n_tests; total_fail = total_fail + hA.n_fail;
                if (hA.worst_ratio > g_worst_ratio) g_worst_ratio = hA.worst_ratio;
        hB.run; total_tests = total_tests + hB.n_tests; total_fail = total_fail + hB.n_fail;
                if (hB.worst_ratio > g_worst_ratio) g_worst_ratio = hB.worst_ratio;
        hC.run; total_tests = total_tests + hC.n_tests; total_fail = total_fail + hC.n_fail;
                if (hC.worst_ratio > g_worst_ratio) g_worst_ratio = hC.worst_ratio;
        hD.run; total_tests = total_tests + hD.n_tests; total_fail = total_fail + hD.n_fail;
                if (hD.worst_ratio > g_worst_ratio) g_worst_ratio = hD.worst_ratio;
        hE.run; total_tests = total_tests + hE.n_tests; total_fail = total_fail + hE.n_fail;
                if (hE.worst_ratio > g_worst_ratio) g_worst_ratio = hE.worst_ratio;

        $display("WORST weight err/tol ratio across all tests = %0.4f", g_worst_ratio);
        if (total_fail != 0) begin
            $display("FAILED: %0d of %0d tests mismatched", total_fail, total_tests);
            $fatal(1, "moe_router_fp8_tb: MISMATCH");
        end
        $display("ALL %0d TESTS PASSED", total_tests);
        $finish;
    end
endmodule

//============================================================================
// mr8_harness : one moe_router_fp8 instance of a given shape + the faithful
//   fp8 golden + a directed/random driver.
//============================================================================
module mr8_harness #(
    parameter integer HIDDEN   = 8,
    parameter integer N_EXPERT = 8,
    parameter integer TOPK     = 2,
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

    // tolerances (see header)
    real WTOL_REL; initial WTOL_REL = 1.0 / 128.0;    // 2^-7
    real WTOL_ABS; initial WTOL_ABS = 1.0 / 512.0;    // 2^-9
    real GTOL;     initial GTOL     = 1.0 / 4096.0;   // 2^-12 negligible-gate floor

    integer n_tests; integer n_fail;
    real    worst_ratio, worst_aerr, worst_tol;

    // ---- clock / reset ----
    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst;

    // ---- DUT control ----
    reg                       start;
    wire                      busy, done;
    reg  [16*HIDDEN-1:0]      x_vec;
    wire                      w_req;
    wire [KW-1:0]             w_k;
    reg  [8*N_EXPERT-1:0]     w_col;
    reg  [16*N_EXPERT*NB-1:0] w_scale;
    wire [TOPK*IDXW-1:0]      sel_idx;
    wire [TOPK*16-1:0]        sel_weight;

    // ---- model storage (golden + DUT share the SAME data) ----
    reg [7:0]  Wcode    [0:HIDDEN-1][0:N_EXPERT-1];   // E4M3 codes (W_g)
    reg [15:0] xb       [0:HIDDEN-1];                 // bf16 token
    reg [15:0] wscale_bf[0:N_EXPERT-1];               // bf16 block-0 scale per expert

    // combinational FP8 weight-pull answer: w_col = W_g[w_k, *] (E4M3 lanes).
    integer ce;
    always @* begin
        for (ce = 0; ce < N_EXPERT; ce = ce + 1)
            w_col[8*ce +: 8] = Wcode[w_k[ROWW-1:0]][ce];
    end

    // ---- DUT ----
    moe_router_fp8 #(.HIDDEN(HIDDEN), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
                     .SCALE(SCALE), .KMAX(KMAX), .BLK(BLK)) dut (
        .clk(clk), .rst(rst),
        .start(start), .busy(busy), .done(done),
        .x_vec(x_vec),
        .w_req(w_req), .w_k(w_k), .w_col(w_col), .w_scale(w_scale),
        .sel_idx(sel_idx), .sel_weight(sel_weight)
    );

    // ===================================================================
    //  fp helpers (INDEPENDENT fp64 reference)
    // ===================================================================
    // 2^n for integer n (any sign), in fp64.
    function automatic real pow2(input integer n);
        real r; integer i;
        begin
            r = 1.0;
            if (n >= 0) for (i = 0; i < n;  i = i + 1) r = r * 2.0;
            else        for (i = 0; i < -n; i = i + 1) r = r / 2.0;
            pow2 = r;
        end
    endfunction

    // bf16 bits -> real (FTZ subnormals/zero, matches glm_fp).
    function automatic real bf16_to_real(input [15:0] b);
        reg s; reg [7:0] e; reg [6:0] m; real mant;
        begin
            s = b[15]; e = b[14:7]; m = b[6:0];
            if (e == 8'h00)      bf16_to_real = 0.0;             // zero / FTZ
            else if (e == 8'hFF) bf16_to_real = 0.0;             // inf/nan: unused
            else begin
                mant = 1.0;
                if (m[6]) mant = mant + 0.5;
                if (m[5]) mant = mant + 0.25;
                if (m[4]) mant = mant + 0.125;
                if (m[3]) mant = mant + 0.0625;
                if (m[2]) mant = mant + 0.03125;
                if (m[1]) mant = mant + 0.015625;
                if (m[0]) mant = mant + 0.0078125;
                bf16_to_real = (s ? -mant : mant) * pow2($signed({1'b0,e}) - 127);
            end
        end
    endfunction

    // real -> bf16 (RNE), matching glm_fp.vh fp32_to_bf16 on the bf16 grid.
    function automatic [15:0] real_to_bf16(input real v);
        reg s; real a; integer e; real mant, scaled, half; integer fi; reg [6:0] frac; reg [7:0] ebias;
        begin
            if (v == 0.0) real_to_bf16 = 16'h0000;
            else begin
                s = (v < 0.0); a = s ? -v : v;
                e = 0;
                if (a >= 1.0) while (a / pow2(e) >= 2.0) e = e + 1;
                else          while (a / pow2(e) <  1.0) e = e - 1;
                mant   = a / pow2(e);              // [1,2)
                scaled = (mant - 1.0) * 128.0;     // 7 mantissa bits
                fi     = $rtoi(scaled);
                half   = scaled - fi;
                if (half > 0.5)       fi = fi + 1;
                else if (half == 0.5) fi = fi + (fi[0] ? 1 : 0);   // ties to even
                if (fi == 128) begin fi = 0; e = e + 1; end
                frac = fi[6:0];
                if ((e + 127) >= 255)   real_to_bf16 = {s, 8'hFE, 7'h7F};
                else if ((e + 127) <= 0) real_to_bf16 = {s, 8'h00, 7'h00};
                else begin ebias = (e + 127); real_to_bf16 = {s, ebias, frac}; end
            end
        end
    endfunction

    // E4M3 code -> real (decode; matches fp8e4m3_to_fp32).  NaN unused -> 0.
    function automatic real e4m3_to_real(input [7:0] c);
        reg s; reg [3:0] e; reg [2:0] m; real v;
        begin
            s = c[7]; e = c[6:3]; m = c[2:0];
            if (e == 4'hF && m == 3'h7)       v = 0.0;                       // NaN (unused)
            else if (e == 4'h0 && m == 3'h0)  v = 0.0;                       // zero
            else if (e == 4'h0)               v = (m * 1.0) * pow2(-9);      // subnormal m*2^-9
            else                              v = (1.0 + (m * 1.0)/8.0) * pow2($signed({1'b0,e}) - 7);
            e4m3_to_real = s ? -v : v;
        end
    endfunction

    // real -> E4M3 code (RNE + saturate to +/-448), matching fp32_to_fp8e4m3.
    // Inputs here are exactly (bf16 * 2^ash), so this is an exact fp8 grid round.
    function automatic [7:0] real_to_e4m3(input real v);
        reg s; real a, sig, scaled, frac, qf, qfrac; integer E, fi, m, qi;
        reg [3:0] ef; reg [2:0] mm;
        begin
            if (v == 0.0) real_to_e4m3 = 8'h00;
            else begin
                s = (v < 0.0); a = s ? -v : v;
                E = 0;
                if (a >= 1.0) while (a / pow2(E) >= 2.0) E = E + 1;
                else          while (a / pow2(E) <  1.0) E = E - 1;
                if (E >= -6) begin
                    // ---- NORMAL E4M3 (exp field E+7 in [1..15]) ----
                    sig    = a / pow2(E);          // [1,2)
                    scaled = (sig - 1.0) * 8.0;    // [0,8)  -> 3 mantissa bits
                    fi     = $rtoi(scaled);
                    frac   = scaled - fi;
                    if (frac > 0.5)       fi = fi + 1;
                    else if (frac == 0.5) fi = fi + (fi[0] ? 1 : 0);  // ties to even
                    m = fi;
                    if (m == 8) begin m = 0; E = E + 1; end           // mantissa carry
                    if (E > 8 || (E == 8 && m == 7))                  // overflow / NaN-slot
                        real_to_e4m3 = {s, 4'b1111, 3'b110};          // saturate to 448
                    else begin
                        ef = E + 7; mm = m[2:0];
                        real_to_e4m3 = {s, ef, mm};
                    end
                end else begin
                    // ---- SUBNORMAL E4M3 : round a to nearest multiple of 2^-9 ----
                    qf    = a * 512.0;             // a * 2^9
                    qi    = $rtoi(qf);
                    qfrac = qf - qi;
                    if (qfrac > 0.5)       qi = qi + 1;
                    else if (qfrac == 0.5) qi = qi + (qi[0] ? 1 : 0);  // ties to even
                    if (qi <= 0)      real_to_e4m3 = {s, 7'b0000000};      // signed zero
                    else if (qi >= 8) real_to_e4m3 = {s, 4'b0001, 3'b000};// up to smallest normal
                    else begin mm = qi[2:0]; real_to_e4m3 = {s, 4'b0000, mm}; end
                end
            end
        end
    endfunction

    // exp(x) in fp64 (range-reduced Taylor; |logit| bounded in these tests).
    function automatic real exp_real(input real x);
        real term, sum, xx; integer i, k;
        begin
            xx = x; k = 0;
            while ((xx > 0.5) || (xx < -0.5)) begin xx = xx / 2.0; k = k + 1; end
            sum = 1.0; term = 1.0;
            for (i = 1; i < 30; i = i + 1) begin term = term * xx / i; sum = sum + term; end
            for (i = 0; i < k; i = i + 1) sum = sum * sum;
            exp_real = sum;
        end
    endfunction

    // dynamic per-token pow2 activation scale (== DUT dyn_shift).
    function automatic integer dyn_shift(input [7:0] emax);
        integer sh;
        begin
            if (emax == 8'd0) dyn_shift = 0;
            else begin
                sh = 134 - emax;
                if (sh >  127) sh =  127;
                if (sh < -128) sh = -128;
                dyn_shift = sh;
            end
        end
    endfunction

    // ===================================================================
    //  GOLDEN (faithful fp8) for the CURRENT xb / Wcode / wscale_bf.
    // ===================================================================
    real    g_logit  [0:N_EXPERT-1];
    reg [15:0] g_gate_bf [0:N_EXPERT-1];   // bf16 gate (what topk sees)
    real    g_gate   [0:N_EXPERT-1];       // fp64 value of that bf16 gate
    integer g_idx    [0:TOPK-1];
    real    g_weight [0:TOPK-1];           // fp64 ideal renormalized weight
    reg [15:0] g_w_bf [0:TOPK-1];          // bf16 of that weight

    task golden;
        integer e, k, t, j, best, ash; reg [7:0] emax;
        real acc, xr, xs, av, wv, blk, lg, gv, s, bestv;
        reg used [0:N_EXPERT-1];
        begin
            // dynamic activation pow2 scale: max bf16 exponent field over xb.
            emax = 8'd0;
            for (k = 0; k < HIDDEN; k = k + 1)
                if (xb[k][14:7] > emax) emax = xb[k][14:7];
            ash = dyn_shift(emax);

            // logits = x @ W_g, faithful fp8 (single [128,128] K-block).
            for (e = 0; e < N_EXPERT; e = e + 1) begin
                acc = 0.0;
                for (k = 0; k < HIDDEN; k = k + 1) begin
                    xr = bf16_to_real(xb[k]);
                    xs = xr * pow2(ash);                 // pre-scale (exact pow2)
                    av = e4m3_to_real(real_to_e4m3(xs)); // E4M3-quantized activation
                    wv = e4m3_to_real(Wcode[k][e]);      // E4M3 weight
                    acc = acc + av * wv;                 // EXACT fp8 product, fp64 reduce
                end
                blk = bf16_to_real(wscale_bf[e]);        // [128,128] block dequant scale
                lg  = acc * blk * pow2(-ash);            // block scale, undo act pre-scale
                g_logit[e]   = bf16_to_real(real_to_bf16(lg));   // GEMV emits bf16 logit
                gv           = 1.0 / (1.0 + exp_real(-g_logit[e]));
                g_gate_bf[e] = real_to_bf16(gv);                 // glm_act emits bf16 gate
                g_gate[e]    = bf16_to_real(g_gate_bf[e]);
            end

            // top-K on the bf16 gate, LOWER-INDEX tie-break (== topk_select).
            for (e = 0; e < N_EXPERT; e = e + 1) used[e] = 1'b0;
            for (t = 0; t < TOPK; t = t + 1) begin
                best = -1; bestv = 0.0;
                for (j = 0; j < N_EXPERT; j = j + 1) begin
                    if (!used[j]) begin
                        if (best == -1) begin best = j; bestv = g_gate[j]; end
                        else if (g_gate[j] > bestv) begin best = j; bestv = g_gate[j]; end
                    end
                end
                g_idx[t] = best; used[best] = 1'b1;
            end

            // renorm-THEN-scale: s = SUM selected gate ; w = (g/s)*2.5 ; bf16.
            s = 0.0;
            for (t = 0; t < TOPK; t = t + 1) s = s + g_gate[g_idx[t]];
            for (t = 0; t < TOPK; t = t + 1) begin
                g_weight[t] = (g_gate[g_idx[t]] / s) * 2.5;
                g_w_bf[t]   = real_to_bf16(g_weight[t]);
            end
        end
    endtask

    // ===================================================================
    //  DRIVE one token through the DUT and CHECK vs the faithful fp8 golden.
    // ===================================================================
    task run_one;
        input [255:0] label;
        integer t, gi, di, i; real gw, dw, aerr, tol, ratio, wsum;
        reg [IDXW-1:0] didx; reg fail_this, idx_ok, sawx;
        begin
            n_tests = n_tests + 1;
            // pack x_vec / w_scale from the model
            for (t = 0; t < HIDDEN;   t = t + 1) x_vec[16*t +: 16] = xb[t];
            for (t = 0; t < N_EXPERT; t = t + 1) w_scale[16*t +: 16] = wscale_bf[t]; // bj=0

            golden;   // faithful fp8 reference for this token

            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;
            i = 0;
            while (!done && i < 20000) begin @(negedge clk); i = i + 1; end
            fail_this = 1'b0;
            if (!done) begin
                $display("FAIL %0s [%0dx%0d k%0d]: TIMEOUT waiting for done",
                         label, HIDDEN, N_EXPERT, TOPK);
                n_fail = n_fail + 1;
                disable run_one;
            end
            @(negedge clk);   // sample held outputs

            // X-AWARE: every routed output bit must be defined.
            sawx = (^sel_idx === 1'bx) || (^sel_weight === 1'bx);
            if (sawx) begin
                $display("FAIL %0s [%0dx%0d k%0d]: X in outputs sel_idx=%b sel_weight=%h",
                         label, HIDDEN, N_EXPERT, TOPK, sel_idx, sel_weight);
                fail_this = 1'b1;
            end

            wsum = 0.0;
            for (t = 0; t < TOPK; t = t + 1) begin
                gi   = g_idx[t];
                didx = sel_idx[IDXW*t +: IDXW];
                di   = didx;
                gw   = g_weight[t];
                dw   = bf16_to_real(sel_weight[16*t +: 16]);
                wsum = wsum + dw;
                aerr = (dw - gw) < 0.0 ? (gw - dw) : (dw - gw);
                tol  = WTOL_REL * (gw < 0.0 ? -gw : gw) + WTOL_ABS;

                // INDEX exact, with the principled tie / negligible-tail exception.
                idx_ok = (di === gi)
                      || (g_gate_bf[di] === g_gate_bf[gi])
                      || ((g_gate[di] < GTOL) && (g_gate[gi] < GTOL));

                if (!sawx && !idx_ok) begin
                    $display("FAIL %0s [%0dx%0d k%0d] slot %0d: idx got %0d (g=%g) exp %0d (g=%g)",
                             label, HIDDEN, N_EXPERT, TOPK, t, di, g_gate[di], gi, g_gate[gi]);
                    fail_this = 1'b1;
                end else if (!sawx && (aerr > tol)) begin
                    $display("FAIL %0s [%0dx%0d k%0d] slot %0d: idx %0d w got %04h (%g) exp %04h (%g) aerr %g > tol %g",
                             label, HIDDEN, N_EXPERT, TOPK, t, di,
                             sel_weight[16*t +: 16], dw, g_w_bf[t], gw, aerr, tol);
                    fail_this = 1'b1;
                end

                // track worst err/tol ratio (only where the weight check applies)
                if (!sawx && idx_ok) begin
                    ratio = aerr / tol;
                    if (ratio > worst_ratio) begin
                        worst_ratio = ratio; worst_aerr = aerr; worst_tol = tol;
                    end
                end
            end

            // INVARIANT: the TOPK selected weights sum to SCALE = 2.5.
            if (!sawx && (((wsum - 2.5) > 0.03) || ((2.5 - wsum) > 0.03))) begin
                $display("FAIL %0s [%0dx%0d k%0d]: weight sum %0.4f != 2.5",
                         label, HIDDEN, N_EXPERT, TOPK, wsum);
                fail_this = 1'b1;
            end

            if (fail_this) n_fail = n_fail + 1;
            else $display("ok   %0s [%0dx%0d k%0d]  idx0 %0d  w0 %04h  sum %0.4f",
                          label, HIDDEN, N_EXPERT, TOPK, sel_idx[0 +: IDXW],
                          sel_weight[0 +: 16], wsum);
        end
    endtask

    // ===================================================================
    //  random model generators
    // ===================================================================
    reg [31:0] lfsr;
    function automatic [31:0] lfsr_step;
        begin
            lfsr = (lfsr >> 1) ^ ({32{lfsr[0]}} & 32'hD0000001);
            lfsr_step = lfsr;
        end
    endfunction

    // x bf16: exponent in [122,128] (|val| ~ [2^-5, ~2]); sign random.
    function automatic [15:0] rnd_x_bf16;
        reg [31:0] r; reg [7:0] e;
        begin
            r = lfsr_step();
            e = 8'd122 + (r[10:8] % 8'd7);
            rnd_x_bf16 = {r[15], e, r[6:0]};
        end
    endfunction

    // W_g E4M3 code: exp field in [0,7] (subnormals..~1.875), sign random; no NaN.
    function automatic [7:0] rnd_w_e4m3;
        reg [31:0] r; reg [3:0] e; reg [2:0] m;
        begin
            r = lfsr_step();
            e = {1'b0, r[6:4]};      // 0..7  (never 15 -> never NaN)
            m = r[2:0];
            rnd_w_e4m3 = {r[15], e, m};
        end
    endfunction

    // block scale bf16: positive, exp in [125,129] (~[0.25,4]).
    function automatic [15:0] rnd_scale_bf16;
        reg [31:0] r; reg [7:0] e;
        begin
            r = lfsr_step();
            e = 8'd125 + (r[18:16] % 8'd5);
            rnd_scale_bf16 = {1'b0, e, r[6:0]};
        end
    endfunction

    task fill_random;
        integer k, e;
        begin
            for (k = 0; k < HIDDEN; k = k + 1) begin
                xb[k] = rnd_x_bf16();
                for (e = 0; e < N_EXPERT; e = e + 1) Wcode[k][e] = rnd_w_e4m3();
            end
            for (e = 0; e < N_EXPERT; e = e + 1) wscale_bf[e] = rnd_scale_bf16();
        end
    endtask

    task fill_const; input [15:0] xc; input [7:0] wc; input [15:0] sc;
        integer k, e;
        begin
            for (k = 0; k < HIDDEN; k = k + 1) begin
                xb[k] = xc;
                for (e = 0; e < N_EXPERT; e = e + 1) Wcode[k][e] = wc;
            end
            for (e = 0; e < N_EXPERT; e = e + 1) wscale_bf[e] = sc;
        end
    endtask

    // ===================================================================
    //  RUN : reset, directed edge cases, then NRAND random tokens.
    // ===================================================================
    task run;
        integer r, k, e;
        begin
            n_tests = 0; n_fail = 0;
            worst_ratio = 0.0; worst_aerr = 0.0; worst_tol = 0.0;
            lfsr = SEED;
            start = 1'b0; x_vec = {16*HIDDEN{1'b0}}; w_scale = {16*N_EXPERT*NB{1'b0}};
            rst = 1'b1;
            @(negedge clk); @(negedge clk);
            rst = 1'b0;
            @(negedge clk);

            // ---- directed 1: a CLEAR DOMINANT expert (deterministic) ----
            // x=1.0, block scale 1.0, all W=0 except expert N/2's column = 1.0(E4M3
            // 0x38) -> that expert gets a big positive logit; others logit 0.
            fill_const(16'h3F80 /*1.0*/, 8'h00 /*E4M3 0*/, 16'h3F80 /*1.0*/);
            for (k = 0; k < HIDDEN; k = k + 1) Wcode[k][N_EXPERT/2] = 8'h38; // 1.0
            run_one("dominant");

            // ---- directed 2: ALL-EQUAL logits (pure lower-index tie-break) ----
            // x=1.0, W=0.5(E4M3 0x30), scale 1.0 -> every logit identical.
            fill_const(16'h3F80 /*1.0*/, 8'h30 /*0.5*/, 16'h3F80 /*1.0*/);
            run_one("all-equal");

            // ---- directed 3: TRUE TIE between experts 0 and 1 (tie-break) ----
            // small base everywhere; columns 0 and 1 identical & larger than rest.
            fill_const(16'h3F80, 8'h28 /*0.25*/, 16'h3F80);
            for (k = 0; k < HIDDEN; k = k + 1) begin
                Wcode[k][0] = 8'h38;   // 1.0
                Wcode[k][1] = 8'h38;   // 1.0  (exactly equal -> tie -> idx 0 first)
            end
            run_one("true-tie");

            // ---- directed 4: ALL-ZERO x (emax=0 -> ash=0 -> all logits 0) ----
            fill_const(16'h0000 /*0*/, 8'h30 /*0.5*/, 16'h3F80 /*1.0*/);
            run_one("zero-x");

            // ---- directed 5: a random token with NON-UNIT block scales ----
            fill_random;
            run_one("dir-rand");

            // ---- NRAND random tokens (random x, E4M3 W_g, bf16 block scales) ----
            for (r = 0; r < NRAND; r = r + 1) begin
                fill_random;
                run_one("random");
            end
        end
    endtask
endmodule
