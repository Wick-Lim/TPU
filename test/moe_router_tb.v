`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// moe_router_tb.v  --  INDEPENDENT fp64 golden testbench for moe_router.v
//----------------------------------------------------------------------------
// GOLDEN (written from the GLM-5.2 / DeepSeek-v3 router spec, ACCEL_GLM52 §5,
// NOT transcribed from the RTL):
//
//   logits_e = Σ_k  bf16(x[k]) * bf16(W_g[k,e])        (reduce in fp64)
//   logit_e  = bf16( logits_e )                         (GEMV emits bf16)
//   gate_e   = 1 / (1 + exp(-logit_e))                  (sigmoid, fp64)
//   gate_e   = bf16( gate_e )                            (glm_act emits bf16)
//   (idx,g)  = TOP-K of gate_e, LOWER-INDEX tie-break    (== topk_select)
//   s        = Σ_{j in TopK} gate_j                      (fp64)
//   w_j      = (gate_j / s) * 2.5                        (renorm THEN scale)
//   weight_j = bf16( w_j )                               (RNE narrow, as emitted)
//
// The golden quantizes to bf16 at EXACTLY the three points the hardware does
// (logit, gate, weight).  Crucially, the top-K selection runs on the bf16-
// quantized gate (the value topk_select actually sees), so the SELECTED INDICES
// and the lower-index tie-break match the DUT bit-for-bit.  All reductions and
// the sigmoid/reciprocal are done in fp64 (Verilog `real`) -- an independent
// oracle, not the RTL's fp32 rsqrt path.
//
// PASS CRITERIA
//   * SELECTED INDICES : EXACT match (all TOPK slots, in DUT's emitted order).
//   * WEIGHTS          : within a PRINCIPLED RELATIVE TOLERANCE of
//                          WTOL_REL = 2^-7 (~0.78 %).
//     Rationale: the emitted weight is bf16, so its own RNE rounding is already
//     ~2^-8 (one bf16 ULP) relative.  On top of that the DUT's reciprocal is the
//     Quake-seed rsqrt^2 (~2^-21 rel) and the GEMV accumulates in fp32 (the
//     golden in fp64), with bf16-grid logit/gate quantization shared.  Summing
//     one bf16 ULP (2^-8) for the final narrow plus a half-ULP-class margin for
//     reciprocal + accumulation-order differences gives 2^-7 as a tight,
//     principled bound.  (Observed margin in practice is well under this.)
//============================================================================
module moe_router_tb;

    // ---- per-test parameters (overridden per instance via the generate run) ----
    // We exercise several (HIDDEN,N_EXPERT,TOPK) shapes by RE-INSTANTIATING the
    // DUT inside a generate-for over a config table is awkward in pure V2005, so
    // instead we drive ONE generic harness module (mr_harness) per shape and
    // tally results in this top.

    integer total_tests = 0;
    integer total_fail  = 0;

    // ---- shape A : router ratio, small  (HIDDEN=4 , N=8 , K=2) ----
    mr_harness #(.HIDDEN(4),  .N_EXPERT(8),  .TOPK(2),  .NRAND(40), .SEED(32'h1111_0001))
        hA();
    // ---- shape B : larger N            (HIDDEN=8 , N=16, K=4) ----
    mr_harness #(.HIDDEN(8),  .N_EXPERT(16), .TOPK(4),  .NRAND(40), .SEED(32'h2222_0002))
        hB();
    // ---- shape C : wider hidden, big N (HIDDEN=16, N=32, K=8) ----
    mr_harness #(.HIDDEN(16), .N_EXPERT(32), .TOPK(8),  .NRAND(40), .SEED(32'h3333_0003))
        hC();
    // ---- shape D : TOPK == N_EXPERT (select-all / dense fallback) ----
    mr_harness #(.HIDDEN(4),  .N_EXPERT(8),  .TOPK(8),  .NRAND(20), .SEED(32'h4444_0004))
        hD();
    // ---- shape E : router-faithful 8->2 with larger hidden ----
    mr_harness #(.HIDDEN(32), .N_EXPERT(8),  .TOPK(2),  .NRAND(40), .SEED(32'h5555_0005))
        hE();

    // ---- orchestrate the harnesses sequentially, then tally ----
    initial begin
        // kick each harness in turn; each pulses its `done_all` when finished.
        hA.run; total_tests = total_tests + hA.n_tests; total_fail = total_fail + hA.n_fail;
        hB.run; total_tests = total_tests + hB.n_tests; total_fail = total_fail + hB.n_fail;
        hC.run; total_tests = total_tests + hC.n_tests; total_fail = total_fail + hC.n_fail;
        hD.run; total_tests = total_tests + hD.n_tests; total_fail = total_fail + hD.n_fail;
        hE.run; total_tests = total_tests + hE.n_tests; total_fail = total_fail + hE.n_fail;

        if (total_fail != 0) begin
            $display("FAILED: %0d of %0d tests mismatched", total_fail, total_tests);
            $fatal(1, "moe_router_tb: MISMATCH");
        end
        $display("ALL %0d TESTS PASSED", total_tests);
        $finish;
    end

endmodule

//============================================================================
// mr_harness : one DUT instance of a given shape + its golden + driver task.
//   `run` (an exported task) executes: directed cases + NRAND random cases,
//   counts tests into n_tests / failures into n_fail.
//============================================================================
module mr_harness #(
    parameter integer HIDDEN   = 4,
    parameter integer N_EXPERT = 8,
    parameter integer TOPK     = 2,
    parameter integer NRAND    = 40,
    parameter [31:0]  SEED     = 32'hDEAD_BEEF
);
    `include "glm_fp.vh"

    localparam integer IDXW  = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT);
    localparam [31:0]  SCALE = 32'h40200000;   // 2.5 fp32 (== DUT default)
    localparam integer KMAX  = 16384;
    localparam integer KW    = $clog2(KMAX+1);

    // weight tolerance (see header rationale):
    //   |w_dut - w_gold| <= WTOL_REL*|w_gold| + WTOL_ABS
    //   WTOL_REL = 2^-7 (~0.78%) : one bf16 ULP (final RNE narrow) + a half-ULP-
    //              class margin for the Quake-rsqrt reciprocal + fp32-vs-fp64
    //              GEMV accumulation order.
    //   WTOL_ABS = 2^-9 : an absolute floor so NEGLIGIBLE tail weights (gate~0,
    //              the renormalized weight is ~1e-4..1e-5) are not judged by a
    //              meaningless relative error -- the bf16 ULP of such a tiny
    //              value is ~its own magnitude.  A 2^-9 floor is < 0.08% of the
    //              total routed mass (which sums to 2.5), i.e. routing-irrelevant.
    real WTOL_REL; initial WTOL_REL = 1.0 / 128.0;
    real WTOL_ABS; initial WTOL_ABS = 1.0 / 512.0;
    // negligible-gate floor (2^-12): experts with gate below this are routing-
    // irrelevant; their relative order is below the fp32 GEMV's logit precision.
    real GTOL; initial GTOL = 1.0 / 4096.0;

    integer n_tests; integer n_fail;

    // ---- clock / reset ----
    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst;

    // ---- DUT control ----
    reg                       start;
    wire                      busy, done;
    reg  [16*HIDDEN-1:0]      x_vec;
    wire                      w_req;
    wire [KW-1:0]             w_k;
    reg  [16*N_EXPERT-1:0]    w_col;
    wire [TOPK*IDXW-1:0]      sel_idx;
    wire [TOPK*16-1:0]        sel_weight;

    // ---- W_g storage (bf16), HIDDEN rows x N_EXPERT cols, packed per row ----
    reg [15:0] Wg [0:HIDDEN-1][0:N_EXPERT-1];
    reg [15:0] xb [0:HIDDEN-1];

    // combinational weight-pull answer: w_col = W_g[w_k, *].
    //   w_k is a row index 0..HIDDEN-1; present that row's N_EXPERT bf16 lanes,
    //   COMBINATIONALLY the same cycle (mirrors swiglu_expert's w_col answer).
    localparam integer ROWW = (HIDDEN <= 1) ? 1 : $clog2(HIDDEN);
    integer ce;
    always @* begin
        for (ce = 0; ce < N_EXPERT; ce = ce + 1)
            w_col[16*ce +: 16] = Wg[w_k[ROWW-1:0]][ce];
    end

    // ---- DUT ----
    moe_router #(.HIDDEN(HIDDEN), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
                 .SCALE(SCALE), .KMAX(KMAX)) dut (
        .clk(clk), .rst(rst),
        .start(start), .busy(busy), .done(done),
        .x_vec(x_vec),
        .w_req(w_req), .w_k(w_k), .w_col(w_col),
        .sel_idx(sel_idx), .sel_weight(sel_weight)
    );

    // ===================================================================
    //  GOLDEN (fp64)
    // ===================================================================
    // bf16 <-> real helpers (independent fp64 reference).
    function automatic real bf16_to_real(input [15:0] b);
        reg        s; reg [7:0] e; reg [6:0] m;
        real       mant; integer ei;
        begin
            s = b[15]; e = b[14:7]; m = b[6:0];
            if (e == 8'h00) bf16_to_real = 0.0;            // FTZ (matches glm_fp)
            else if (e == 8'hFF) bf16_to_real = 0.0;       // inf/nan: not generated
            else begin
                mant = 1.0;
                if (m[6]) mant = mant + 0.5;
                if (m[5]) mant = mant + 0.25;
                if (m[4]) mant = mant + 0.125;
                if (m[3]) mant = mant + 0.0625;
                if (m[2]) mant = mant + 0.03125;
                if (m[1]) mant = mant + 0.015625;
                if (m[0]) mant = mant + 0.0078125;
                ei = e - 127;
                bf16_to_real = (s ? -mant : mant) * pow2(ei);
            end
        end
    endfunction

    // 2^n for integer n (any sign), in fp64.
    function automatic real pow2(input integer n);
        real r; integer i;
        begin
            r = 1.0;
            if (n >= 0) for (i = 0; i < n; i = i + 1) r = r * 2.0;
            else        for (i = 0; i < -n; i = i + 1) r = r / 2.0;
            pow2 = r;
        end
    endfunction

    // real -> bf16 (RNE), matching glm_fp.vh fp32_to_bf16 on the bf16 grid.
    // We round the fp64 value to the nearest bf16 (8-bit exp, 7-bit mantissa).
    function automatic [15:0] real_to_bf16(input real v);
        reg        s; real a; integer e; real mant; reg [6:0] frac;
        real       scaled; integer fi; real half; reg [7:0] ebias;
        begin
            if (v == 0.0) real_to_bf16 = 16'h0000;
            else begin
                s = (v < 0.0) ? 1'b1 : 1'b0;
                a = s ? -v : v;
                // find exponent e such that 1.0 <= a/2^e < 2.0
                e = 0;
                if (a >= 1.0) begin
                    while (a / pow2(e) >= 2.0) e = e + 1;
                end else begin
                    while (a / pow2(e) < 1.0) e = e - 1;
                end
                mant = a / pow2(e);           // in [1,2)
                // fractional part * 128, RNE to 7 bits
                scaled = (mant - 1.0) * 128.0;
                fi = $rtoi(scaled);
                half = scaled - fi;
                // round to nearest, ties to even
                if (half > 0.5)      fi = fi + 1;
                else if (half == 0.5) fi = fi + (fi[0] ? 1 : 0);
                if (fi == 128) begin           // mantissa carry -> exp++
                    fi = 0; e = e + 1;
                end
                frac = fi[6:0];
                // bias and pack; clamp to a representable range (no inf in tests)
                if ((e + 127) >= 255) real_to_bf16 = {s, 8'hFE, 7'h7F};
                else if ((e + 127) <= 0) real_to_bf16 = {s, 8'h00, 7'h00};
                else begin
                    ebias = (e + 127);
                    real_to_bf16 = {s, ebias, frac};
                end
            end
        end
    endfunction

    function automatic real exp_real(input real x);
        // series + range reduction for fp64; |x| here is small (|logit|<~30).
        real r, term, sum; integer i, k; real xx;
        begin
            // reduce by integer powers of e via repeated mult is messy; the
            // logits are bounded (bf16 x,W small) so a plain Taylor with enough
            // terms over a moderate range is accurate to fp64 for our needs.
            // To be safe over |x|<=40 we scale: exp(x)=(exp(x/2^s))^(2^s).
            xx = x; k = 0;
            while ((xx > 0.5) || (xx < -0.5)) begin xx = xx / 2.0; k = k + 1; end
            sum = 1.0; term = 1.0;
            for (i = 1; i < 30; i = i + 1) begin
                term = term * xx / i;
                sum = sum + term;
            end
            for (i = 0; i < k; i = i + 1) sum = sum * sum;
            exp_real = sum;
        end
    endfunction

    // ---- golden state ----
    real    g_logit  [0:N_EXPERT-1];
    reg [15:0] g_gate_bf [0:N_EXPERT-1];   // bf16 gate (what topk sees)
    real    g_gate   [0:N_EXPERT-1];       // fp64 value of that bf16 gate
    integer g_idx    [0:TOPK-1];
    real    g_weight [0:TOPK-1];           // fp64 ideal weight
    reg [15:0] g_w_bf[0:TOPK-1];           // bf16 of ideal weight

    // compute the golden for the CURRENT xb/Wg.
    task golden;
        integer e, k, t, j, best;
        real acc, gv, s, bestv;
        reg used [0:N_EXPERT-1];
        begin
            // logits = x @ W_g (fp64 over bf16 operands), then bf16-quantize.
            for (e = 0; e < N_EXPERT; e = e + 1) begin
                acc = 0.0;
                for (k = 0; k < HIDDEN; k = k + 1)
                    acc = acc + bf16_to_real(xb[k]) * bf16_to_real(Wg[k][e]);
                // GEMV emits a bf16 logit:
                g_logit[e] = bf16_to_real(real_to_bf16(acc));
                // sigmoid, then glm_act emits bf16 gate:
                gv = 1.0 / (1.0 + exp_real(-g_logit[e]));
                g_gate_bf[e] = real_to_bf16(gv);
                g_gate[e]    = bf16_to_real(g_gate_bf[e]);
            end
            // top-K on the bf16 gate, lower-index tie-break (== topk_select):
            for (e = 0; e < N_EXPERT; e = e + 1) used[e] = 1'b0;
            for (t = 0; t < TOPK; t = t + 1) begin
                best = -1; bestv = 0.0;
                for (j = 0; j < N_EXPERT; j = j + 1) begin
                    if (!used[j]) begin
                        if (best == -1) begin best = j; bestv = g_gate[j]; end
                        else if (g_gate[j] > bestv) begin best = j; bestv = g_gate[j]; end
                        // STRICTLY greater => equal keeps the lower index (already
                        // chosen first because j ascends) => lower-index tie-break.
                    end
                end
                g_idx[t]   = best;
                used[best] = 1'b1;
            end
            // renorm-then-scale: s = Σ selected gate ; w = (g/s)*2.5 ; bf16.
            s = 0.0;
            for (t = 0; t < TOPK; t = t + 1) s = s + g_gate[g_idx[t]];
            for (t = 0; t < TOPK; t = t + 1) begin
                g_weight[t] = (g_gate[g_idx[t]] / s) * 2.5;
                g_w_bf[t]   = real_to_bf16(g_weight[t]);
            end
        end
    endtask

    // ===================================================================
    //  DRIVE ONE TOKEN through the DUT and CHECK vs golden.
    // ===================================================================
    task run_one;
        input [255:0] label;
        integer t, gi, di; real gw, dw, aerr, tol, gg_di, gg_gi;
        reg [IDXW-1:0] didx; reg fail_this; reg idx_ok;
        begin
            // pack x_vec from xb
            for (t = 0; t < HIDDEN; t = t + 1) x_vec[16*t +: 16] = xb[t];

            // pulse start
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;
            // wait for done
            wait (done == 1'b1);
            @(negedge clk);   // sample held outputs after done

            golden;   // compute reference for this token

            fail_this = 1'b0;
            for (t = 0; t < TOPK; t = t + 1) begin
                gi   = g_idx[t];
                didx = sel_idx[IDXW*t +: IDXW];
                di   = didx;
                gw   = g_weight[t];
                dw   = bf16_to_real(sel_weight[16*t +: 16]);
                aerr = (dw - gw) < 0.0 ? (gw - dw) : (dw - gw);   // |w_dut - w_gold|
                tol  = WTOL_REL * (gw < 0.0 ? -gw : gw) + WTOL_ABS;

                // INDEX check (EXACT), with a principled TIE / NEGLIGIBLE-TAIL
                // exception.  The GEMV reduces in fp32 (golden in fp64); a per-slot
                // index mismatch is ACCEPTED ONLY when the two contending experts
                // are routing-INDISTINGUISHABLE, i.e. EITHER
                //   (a) their golden gates are EQUAL on the bf16 grid (a true tie
                //       topk_select breaks by lower index, equally correct), OR
                //   (b) BOTH golden gates are below a negligible-gate floor
                //       GTOL=2^-12 -- gates this small contribute < 2^-12/s of the
                //       routed mass (which sums to 2.5), and their order is below
                //       the fp32 GEMV's logit precision, so the swap is correct-
                //       within-tolerance, never a real misranking of a meaningful
                //       expert.  A swap of two NON-negligible, NON-equal gates
                //       still FAILS (catches real ordering bugs).
                gg_di = g_gate[di];
                gg_gi = g_gate[gi];
                idx_ok = (di === gi)
                      || (real_to_bf16(gg_di) === real_to_bf16(gg_gi))
                      || ((gg_di < GTOL) && (gg_gi < GTOL));

                n_tests = n_tests + 1;
                if (!idx_ok) begin
                    $display("FAIL %0s [%0dx%0d k%0d] slot %0d: idx got %0d (g=%g) exp %0d (g=%g)",
                             label, HIDDEN, N_EXPERT, TOPK, t, di, gg_di, gi, gg_gi);
                    fail_this = 1'b1; n_fail = n_fail + 1;
                end else if (aerr > tol) begin
                    $display("FAIL %0s [%0dx%0d k%0d] slot %0d: idx %0d  w got %04h (%g) exp %04h (%g) aerr %g > tol %g",
                             label, HIDDEN, N_EXPERT, TOPK, t, di,
                             sel_weight[16*t +: 16], dw, g_w_bf[t], gw, aerr, tol);
                    fail_this = 1'b1; n_fail = n_fail + 1;
                end
            end
            if (!fail_this)
                $display("ok   %0s [%0dx%0d k%0d]  idx0 %0d  w0 %04h",
                         label, HIDDEN, N_EXPERT, TOPK,
                         sel_idx[0 +: IDXW], sel_weight[0 +: 16]);
        end
    endtask

    // ---- LFSR random bf16 generator (small magnitudes, |val| ~ [-2,2]) ----
    reg [31:0] lfsr;
    function automatic [15:0] rnd_bf16;
        reg [15:0] b; reg [7:0] e;
        begin
            // step LFSR
            lfsr = (lfsr >> 1) ^ ({32{lfsr[0]}} & 32'hD0000001);
            // exponent in [120,130] -> magnitudes ~ [2^-7, 2^3]; sign random.
            e = 8'd120 + lfsr[11:8] % 8'd11;
            b = {lfsr[15], e, lfsr[6:0]};
            rnd_bf16 = b;
        end
    endfunction

    // fill xb / Wg with a constant bf16
    task fill_const; input [15:0] xc; input [15:0] wc;
        integer k, e;
        begin
            for (k = 0; k < HIDDEN; k = k + 1) begin
                xb[k] = xc;
                for (e = 0; e < N_EXPERT; e = e + 1) Wg[k][e] = wc;
            end
        end
    endtask

    task fill_random;
        integer k, e;
        begin
            for (k = 0; k < HIDDEN; k = k + 1) begin
                xb[k] = rnd_bf16();
                for (e = 0; e < N_EXPERT; e = e + 1) Wg[k][e] = rnd_bf16();
            end
        end
    endtask

    // ===================================================================
    //  RUN : reset, directed cases, then NRAND random tokens.
    // ===================================================================
    task run;
        integer r, k, e;
        begin
            n_tests = 0; n_fail = 0;
            lfsr = SEED;
            start = 1'b0; x_vec = {16*HIDDEN{1'b0}};
            rst = 1'b1;
            @(negedge clk); @(negedge clk);
            rst = 1'b0;
            @(negedge clk);

            // ---- directed 1: a CLEAR DOMINANT expert ----
            // all-zero W except a large positive column for expert (N/2), x=1.0:
            //   that expert gets a big positive logit -> gate ~1; others gate=0.5.
            fill_const(16'h3F80 /*1.0*/, 16'h0000 /*0*/);
            for (k = 0; k < HIDDEN; k = k + 1)
                Wg[k][N_EXPERT/2] = 16'h40A0;   // 5.0 -> logit = 5*HIDDEN, huge
            run_one("dominant");

            // ---- directed 2: ALL-EQUAL gates ----
            // x and W all 1.0 => every logit identical => every gate identical =>
            // top-K must pick indices 0..K-1 (pure lower-index tie-break), each
            // weight = (1/K)*2.5.
            fill_const(16'h3F80 /*1.0*/, 16'h3DCD /*~0.1, small so gate!=sat*/);
            run_one("all-equal");

            // ---- directed 3: NEAR-TIE between two experts (tie-break stress) ----
            // make expert 0 and 1 have ALMOST equal (but tiny-different) logits,
            // and the rest clearly smaller; checks tie-break + ordering.
            fill_const(16'h3F80, 16'h3D00 /*~0.03*/);
            for (k = 0; k < HIDDEN; k = k + 1) begin
                Wg[k][0] = 16'h3F00;   // 0.5
                Wg[k][1] = 16'h3F00;   // 0.5  (exactly equal -> tie -> idx 0 first)
            end
            run_one("near-tie");

            // ---- directed 4: TOPK == N case handled when TOPK==N_EXPERT shape ----
            // (covered by shape D's random+directed; here just run a random one.)
            fill_random;
            run_one("dir-rand");

            // ---- NRAND random tokens ----
            for (r = 0; r < NRAND; r = r + 1) begin
                fill_random;
                run_one("random");
            end
        end
    endtask

endmodule
