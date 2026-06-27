`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// sampler_tb.v  --  self-checking TB for sampler (GLM-5.2 token sampler)
//                                                            (ACCEL_GLM52 §3)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL
//   The DUT pipeline is: temperature-scale (fp32 mul by 1/T) -> top-k filter
//   (topk_select) -> softmax (glm_softmax, bf16/fp32) -> multinomial draw from
//   an LFSR.  This golden recomputes the SAME decision a COMPLETELY DIFFERENT
//   way, in Verilog `real` (IEEE double):
//     * widen each stored bf16 logit to its EXACT real value (bf16->real),
//     * scale by 1/T in fp64 (real divide-equivalent multiply),
//     * find the TOPK highest by a plain descending selection (lower-index
//       tie-break, matching topk_select's deterministic order),
//     * softmax over the kept scores in fp64 ($exp + true fp64 divide -- NOT
//       the DUT's range-reduced Taylor exp nor its rsqrt^2 reciprocal),
//     * MULTINOMIAL: the LFSR is the unit's DEFINED PRNG (the spec), so the
//       golden steps the SAME Galois LFSR and forms the SAME u in [0,1); then
//       it walks the fp64 cumulative probability (scaled by the kept mass) and
//       picks the first kept token whose cumsum >= u.
//   Everything that produces a *value* (the scaling, the exp, the reduction,
//   the reciprocal) differs from the DUT; only the FORMAT contracts (bf16 grid,
//   the LFSR recurrence, the >= tie rule) are shared, exactly as the unit
//   defines them.  So the golden catches DUT arithmetic/ordering bugs.
//
//   ROBUST VECTORS: directed logit sets are chosen so the WINNING token's
//   probability is well separated from its neighbours' (a wide cumsum margin
//   around u), so the fp64-golden token and the bf16/fp32-DUT token must agree
//   despite the rounding-format difference -- a real bug (wrong scaling, lost
//   max-subtract, mis-ordered top-k, broken LFSR/walk) moves the boundary by
//   >> the margin and flips the token, failing the check.
//
//----------------------------------------------------------------------------
// CHECKS  (a "TEST" = one resolved sampler draw whose token is asserted)
//   * GREEDY (T->0): argmax of the scaled logits, for several rows.
//   * MULTINOMIAL with a fixed seed: hand-/golden-computed cumulative-walk
//     token, for several seeds/rows.
//   * TOP-K EXCLUSION: a low-prob token (outside the TOPK) is NEVER sampled,
//     swept over MANY LFSR draws (every u maps into the kept set only).
//   * TOP-P (nucleus): with a tight p the tail is excluded from the draw.
//   X-AWARE: any X/Z bit on token_o/done is an immediate FAIL.
//
//   On all pass: prints "ALL <N> TESTS PASSED"; any mismatch -> $fatal.
//============================================================================
module sampler_tb;
    // ----------------- parameters under test -----------------
    localparam integer VOCAB  = 16;
    localparam integer TOPK   = 4;
    localparam [31:0]  SEED   = 32'hACE1_2345;
    localparam integer LFSR_W = 32;
    localparam integer IDXW   = 4;          // $clog2(16)
    localparam integer KEFF   = (TOPK < VOCAB) ? TOPK : VOCAB;

    localparam [31:0] TAP32   = 32'h80200003;
    localparam [15:0] BF16_ONE = 16'h3F80;  // 1.0
    localparam [15:0] BF16_INF = 16'h7F80;  // +inf  -> greedy
    localparam [15:0] BF16_HALF= 16'h3F00;  // 0.5

    // ----------------- DUT I/O -----------------
    reg                 clk, rst, start, greedy;
    reg  [15:0]         inv_temp, topp;
    wire                load_req;
    reg  [15:0]         logit_in;
    reg                 logit_valid;
    wire [IDXW-1:0]     token_o;
    wire                done, busy;

    // logit source memory (bf16), vocab order
    reg  [15:0]         logits [0:VOCAB-1];

    sampler #(
        .VOCAB(VOCAB), .TOPK(TOPK), .SEED(SEED), .LFSR_W(LFSR_W)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .greedy(greedy),
        .inv_temp(inv_temp), .topp(topp),
        .load_req(load_req), .logit_in(logit_in), .logit_valid(logit_valid),
        .token_o(token_o), .done(done), .busy(busy)
    );

    // ----------------- clock -----------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    integer tests = 0;
    integer errors = 0;

    // ===================================================================
    //  bf16 -> real (exact widen; same value the DUT sees)
    // ===================================================================
    function automatic real bf16_to_real(input [15:0] b);
        reg        s; reg [7:0] e; reg [6:0] m;
        real       mant; integer ei;
        begin
            s = b[15]; e = b[14:7]; m = b[6:0];
            if (e == 8'h00) begin
                bf16_to_real = 0.0;                 // FTZ (subnormal/zero -> 0)
            end else if (e == 8'hFF) begin
                bf16_to_real = (m==0) ? (s ? -1.0e38 : 1.0e38) : 0.0; // inf-ish
            end else begin
                mant = 1.0 + m / 128.0;
                ei   = e - 127;
                bf16_to_real = mant * (2.0 ** ei) * (s ? -1.0 : 1.0);
            end
        end
    endfunction

    // ===================================================================
    //  GOLDEN: compute the sampled token for a given config.
    //  greedy_g: 1 => argmax.   itemp_r = 1/T (real).  topp_r = p (real).
    //  draws the SAME LFSR sequence the DUT uses, advancing `lfsr_state`.
    // ===================================================================
    integer lfsr_state;   // mirrors the DUT LFSR across draws (32-bit in int)

    // step the 32-bit Galois LFSR once (matches the DUT's lfsr_next)
    function automatic [31:0] lfsr_step(input [31:0] s);
        begin
            if (s[0]) lfsr_step = (s >> 1) ^ TAP32;
            else      lfsr_step = (s >> 1);
        end
    endfunction

    // u in [0,1) from LFSR state, EXACTLY as the DUT forms it: top 23 bits as
    // an fp32 fraction of 1.frac, minus 1.0.  We reproduce in real.
    function automatic real lfsr_u(input [31:0] s);
        reg [22:0] frac; real f;
        begin
            frac = s[31:9];                 // top 23 bits
            f    = frac / (2.0**23);        // in [0,1)
            lfsr_u = f;
        end
    endfunction

    // golden token.  Uses `golden_idx`/`golden_sc` scratch arrays.
    integer gk_idx [0:KEFF-1];     // kept vocab indices (descending score)
    real    gk_sc  [0:KEFF-1];     // kept scaled scores
    real    gk_p   [0:KEFF-1];     // kept softmax probs
    integer golden_token;

    task automatic golden_compute(input integer greedy_g,
                                  input real    itemp_r,
                                  input real    topp_r,
                                  input integer use_lfsr);  // 1: consume a draw
        real    sc [0:VOCAB-1];
        integer used [0:VOCAB-1];
        integer i, j, best, kk;
        real    bestv, mx, denom, cum, u, mass, target;
        integer nuc_last;
        begin
            // scaled scores
            for (i = 0; i < VOCAB; i = i + 1) begin
                sc[i]   = bf16_to_real(logits[i]) * itemp_r;
                used[i] = 0;
            end
            // descending top-KEFF selection, lower-index tie-break
            for (kk = 0; kk < KEFF; kk = kk + 1) begin
                best = -1; bestv = -1.0e60;
                for (i = 0; i < VOCAB; i = i + 1)
                    if (!used[i] && (sc[i] > bestv)) begin
                        bestv = sc[i]; best = i;
                    end
                used[best]  = 1;
                gk_idx[kk]  = best;
                gk_sc[kk]   = sc[best];
            end
            if (greedy_g) begin
                golden_token = gk_idx[0];
            end else begin
                // softmax over kept scores (fp64, stable)
                mx = gk_sc[0];
                for (kk = 1; kk < KEFF; kk = kk + 1)
                    if (gk_sc[kk] > mx) mx = gk_sc[kk];
                denom = 0.0;
                for (kk = 0; kk < KEFF; kk = kk + 1) begin
                    gk_p[kk] = $exp(gk_sc[kk] - mx);
                    denom    = denom + gk_p[kk];
                end
                for (kk = 0; kk < KEFF; kk = kk + 1)
                    gk_p[kk] = gk_p[kk] / denom;
                // nucleus cutoff (descending): first prefix >= topp_r
                nuc_last = KEFF-1;
                cum = 0.0;
                for (kk = 0; kk < KEFF; kk = kk + 1) begin
                    cum = cum + gk_p[kk];
                    if (topp_r < 1.0 && cum >= topp_r && nuc_last == KEFF-1
                        && kk < KEFF-1)
                        nuc_last = kk;
                end
                // nucleus mass
                mass = 0.0;
                for (kk = 0; kk <= nuc_last; kk = kk + 1) mass = mass + gk_p[kk];
                // draw u (consume one LFSR step like the DUT)
                if (use_lfsr) begin
                    u          = lfsr_u(lfsr_state);
                    lfsr_state = lfsr_step(lfsr_state);
                end else u = 0.0;
                target = u * mass;
                // walk prefix cumsum, first >= target
                cum = 0.0; golden_token = gk_idx[nuc_last];
                for (kk = 0; kk <= nuc_last; kk = kk + 1) begin
                    cum = cum + gk_p[kk];
                    if (cum >= target) begin
                        golden_token = gk_idx[kk];
                        kk = nuc_last + 1;   // break
                    end
                end
            end
        end
    endtask

    // ===================================================================
    //  drive one sampler transaction and capture token_o
    // ===================================================================
    integer fb;
    reg [IDXW-1:0] cap_token;

    task automatic run_draw(input integer greedy_g,
                            input [15:0] itemp_b,
                            input [15:0] topp_b);
        begin
            @(negedge clk);
            greedy      <= greedy_g[0];
            inv_temp    <= itemp_b;
            topp        <= topp_b;
            start       <= 1'b1;
            @(negedge clk);
            start       <= 1'b0;
            // feed logits whenever load_req is high
            fb = 0;
            while (fb < VOCAB) begin
                @(negedge clk);
                if (load_req) begin
                    logit_in    <= logits[fb];
                    logit_valid <= 1'b1;
                    fb = fb + 1;
                end else begin
                    logit_valid <= 1'b0;
                end
            end
            @(negedge clk);
            logit_valid <= 1'b0;
            // wait for done
            while (!done) @(negedge clk);
            cap_token = token_o;
            // X-AWARE: token/done must be fully driven
            if (^{token_o, done} === 1'bx) begin
                $display("FAIL: X/Z on sampler outputs (token=%b done=%b)",
                         token_o, done);
                errors = errors + 1;
            end
        end
    endtask

    // ===================================================================
    //  one checked test: drive DUT + golden, compare tokens
    // ===================================================================
    task automatic check_draw(input [127:0] name,
                              input integer greedy_g,
                              input [15:0]  itemp_b,
                              input real    itemp_r,
                              input [15:0]  topp_b,
                              input real    topp_r);
        begin
            golden_compute(greedy_g, itemp_r, topp_r, greedy_g ? 0 : 1);
            run_draw(greedy_g, itemp_b, topp_b);
            tests = tests + 1;
            if (cap_token !== golden_token[IDXW-1:0]) begin
                $display("FAIL [%0s]: dut_token=%0d golden_token=%0d",
                         name, cap_token, golden_token);
                errors = errors + 1;
            end else begin
                $display("PASS [%0s]: token=%0d", name, cap_token);
            end
        end
    endtask

    // load a logit set into `logits`
    task automatic set_logits6(input [15:0] a0,a1,a2,a3,a4,a5);
        integer i;
        begin
            logits[0]=a0; logits[1]=a1; logits[2]=a2; logits[3]=a3;
            logits[4]=a4; logits[5]=a5;
            for (i=6;i<VOCAB;i=i+1) logits[i]=16'hC100; // -8.0 : low, excluded
        end
    endtask

    // ===================================================================
    //  TOP-K EXCLUSION sweep: over MANY LFSR draws, the sampled token is
    //  ALWAYS in the kept (top-k) set and NEVER a masked-out low token.
    // ===================================================================
    integer d, s, in_kept;
    task automatic check_topk_exclusion(input integer ndraws);
        begin
            for (d = 0; d < ndraws; d = d + 1) begin
                run_draw(0, BF16_ONE, BF16_ONE);   // T=1, no top-p
                lfsr_state = lfsr_step(lfsr_state);// keep mirror in lockstep
                // recompute the kept set for THIS config (deterministic per row)
                golden_compute(0, 1.0, 1.0, 0);    // fills gk_idx (no LFSR use)
                in_kept = 0;
                for (s = 0; s < KEFF; s = s + 1)
                    if (cap_token == gk_idx[s][IDXW-1:0]) in_kept = 1;
                tests = tests + 1;
                if (!in_kept) begin
                    $display("FAIL [topk-excl]: sampled token %0d not in top-%0d",
                             cap_token, KEFF);
                    errors = errors + 1;
                end
            end
            $display("PASS [topk-excl]: %0d draws all inside top-%0d set",
                     ndraws, KEFF);
        end
    endtask

    integer seedrun;
    // ===================================================================
    //  MAIN
    // ===================================================================
    initial begin
        rst         = 1'b1;
        start       = 1'b0;
        greedy      = 1'b0;
        inv_temp    = BF16_ONE;
        topp        = BF16_ONE;
        logit_in    = 16'b0;
        logit_valid = 1'b0;
        lfsr_state  = SEED;          // golden LFSR mirror starts at the seed
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ---- ROW A : well-separated logits ----
        //  idx2 highest, idx0 next, then 4,1 ; others very low.
        //  bf16: 4.0=0x4080 3.0=0x4040 2.0=0x4000 1.0=0x3F80 0.5=0x3F00 -2=0xC000
        set_logits6(16'h4040 /*3*/, 16'h4000 /*2*/, 16'h4080 /*4*/,
                    16'hC000 /*-2*/, 16'h3F80 /*1*/, 16'h3F00 /*0.5*/);

        // GREEDY: argmax must be idx 2 (4.0 highest)
        check_draw("greedyA", 1, BF16_INF, 1.0e38, BF16_ONE, 1.0);

        // GREEDY via tiny temperature (inv_temp huge) also argmax.  inv_temp
        // = 64.0 (0x4280) sharpens; greedy flag still selects argmax exactly.
        check_draw("greedyA2", 1, 16'h4280, 64.0, BF16_ONE, 1.0);

        // MULTINOMIAL draws (T=1), several seeds via successive LFSR steps.
        for (seedrun = 0; seedrun < 6; seedrun = seedrun + 1)
            check_draw("multiA", 0, BF16_ONE, 1.0, BF16_ONE, 1.0);

        // sharper (inv_temp=2 => T=0.5): mass concentrates on idx2; still
        // golden-matched cumulative walk.
        for (seedrun = 0; seedrun < 4; seedrun = seedrun + 1)
            check_draw("multiA_T0.5", 0, 16'h4000 /*2.0*/, 2.0,
                       BF16_ONE, 1.0);

        // TOP-P nucleus: p=0.5 keeps only the top mass; tail excluded.
        for (seedrun = 0; seedrun < 4; seedrun = seedrun + 1)
            check_draw("nucleusA", 0, BF16_ONE, 1.0, BF16_HALF, 0.5);

        // ---- ROW B : different ranking, idx5 dominant ----
        set_logits6(16'h3F00 /*0.5*/, 16'h4000 /*2*/, 16'h3F80 /*1*/,
                    16'h4040 /*3*/, 16'hC000 /*-2*/, 16'h4100 /*8*/);
        check_draw("greedyB", 1, BF16_INF, 1.0e38, BF16_ONE, 1.0);
        for (seedrun = 0; seedrun < 6; seedrun = seedrun + 1)
            check_draw("multiB", 0, BF16_ONE, 1.0, BF16_ONE, 1.0);

        // TOP-K EXCLUSION sweep on ROW B (low idx>=6 = -8.0 never sampled).
        check_topk_exclusion(16);

        // ---- ROW C : ties exercise the lower-index tie-break in top-k ----
        set_logits6(16'h4080 /*4*/, 16'h4080 /*4*/, 16'h4000 /*2*/,
                    16'h4000 /*2*/, 16'h3F80 /*1*/, 16'h3F80 /*1*/);
        check_draw("greedyC_tie", 1, BF16_INF, 1.0e38, BF16_ONE, 1.0);
        for (seedrun = 0; seedrun < 4; seedrun = seedrun + 1)
            check_draw("multiC", 0, BF16_ONE, 1.0, BF16_ONE, 1.0);

        // ---- summary ----
        if (errors == 0)
            $display("ALL %0d TESTS PASSED", tests);
        else begin
            $display("TESTS FAILED: %0d of %0d", errors, tests);
            $fatal(1, "sampler_tb: %0d failures", errors);
        end
        $finish;
    end

    // global timeout guard
    initial begin
        #2_000_000;
        $display("FAIL: global timeout");
        $fatal(1, "sampler_tb timeout");
    end
endmodule
