`timescale 1ns/1ps
//============================================================================
// topk_select_tb.v  --  INDEPENDENT golden proof for the shared TOP-K selector
//                                                       (ACCEL_GLM52 §4.2 / §5)
//----------------------------------------------------------------------------
// WHAT IS PROVEN
//   topk_select selects the K LARGEST of N fp32 scores, returning their
//   INDICES + scores in a deterministic order:  slot 0 = largest, and on EQUAL
//   scores the LOWER index ranks first -- i.e. the unique order of a stable
//   sort by (-score, +index).  This TB builds that golden ENTIRELY in the TB,
//   sharing NO arithmetic with the DUT, and checks per-rank index + score AND
//   the selected SET (mask_o) exactly.
//
// INDEPENDENT GOLDEN
//   * fp32 ordering is done with the TB's OWN `tb_gt` (sign+magnitude aware,
//     NaN = smallest, +0==-0) -- a from-scratch reimplementation, not the
//     DUT's fp32_gt.  A simple selection sort over the N scores with the SAME
//     tie-break (strictly-greater displaces; equal keeps lower index) yields a
//     permutation; the top KEFF entries are the golden ranks.  KEFF=min(K,N).
//   * For each rank s < KEFF:  golden index, golden score (== that candidate's
//     stored score), and sel_valid_o[s]==1 are checked EXACTLY.
//   * Surplus slots s in [KEFF,K) (only when K>N) must report sel_valid_o[s]==0.
//   * mask_o must equal the OR of the KEFF golden indices, EXACTLY.
//
// COVERAGE  (representative GLM sizes + directed edge cases + heavy random)
//   harness sizes:
//     (N=256,K=8)   router-shaped                  LANES_IN=1
//     (N=32, K=8)   small DSA                       LANES_IN=1
//     (N=64, K=16)  larger DSA pattern              LANES_IN=4 (multi-lane load)
//     (N=128,K=12)  mid DSA                         LANES_IN=8
//     (N=16, K=16)  K==N dense fallback             LANES_IN=2
//     (N=200,K=200) K>=N dense (K==N here)          LANES_IN=1
//     (N=64, K=1)   pure argmax                     LANES_IN=1
//   each harness runs DIRECTED edge vectors + many constrained-random fp32
//   vectors (wide dynamic range, mixed sign, sigmoid-gate [0,1], all-equal,
//   single dominant outlier, duplicate maxima, negatives).
//
//   PRINTS "ALL <N> TESTS PASSED"; $fatal on the first mismatch.
//============================================================================

// ----------------------------------------------------------------------------
// Reusable size-generic harness: instantiates a topk_select #(N,K,LANES_IN),
// drives the pull-load handshake, builds an independent golden, and checks.
// ----------------------------------------------------------------------------
module tk_harness #(
    parameter integer N        = 256,
    parameter integer K        = 8,
    parameter integer LANES_IN = 1,
    parameter integer NVEC     = 60   // random vectors after the directed set
)(
    input  wire    clk,
    input  wire    go,
    output reg     done_all,
    output integer pass,
    output integer fail
);
    localparam integer SCORE_W = 32;
    localparam integer IDXW    = (N <= 1) ? 1 : $clog2(N);
    localparam integer NBEATS  = N / LANES_IN;
    localparam integer KEFF    = (K < N) ? K : N;

    // ---- DUT I/O ----
    reg                         rst;
    reg                         start;
    wire                        load_req;
    reg  [LANES_IN*SCORE_W-1:0] score_in;
    reg                         score_valid;
    wire [K*IDXW-1:0]           sel_idx_o;
    wire [K*SCORE_W-1:0]        sel_score_o;
    wire [K-1:0]                sel_valid_o;
    wire [N-1:0]                mask_o;
    wire                        busy;
    wire                        done;

    topk_select #(
        .N(N), .K(K), .SCORE_W(SCORE_W), .LANES_IN(LANES_IN)
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .load_req(load_req), .score_in(score_in), .score_valid(score_valid),
        .sel_idx_o(sel_idx_o), .sel_score_o(sel_score_o),
        .sel_valid_o(sel_valid_o), .mask_o(mask_o),
        .busy(busy), .done(done)
    );

    // ---- per-test data ----
    reg  [SCORE_W-1:0] scores [0:N-1];   // current test vector
    // golden ranking (a permutation of 0..N-1 by (-score,+index))
    integer perm [0:N-1];
    reg [SCORE_W-1:0] gscore [0:N-1];    // score at each rank

    integer i, j, b, lane, s;
    integer tmp;
    reg [SCORE_W-1:0] stmp;

    //------------------------------------------------------------------------
    // INDEPENDENT fp32 "strictly greater than" (NaN = smallest, +0==-0).
    // Written from scratch -- does NOT reuse the DUT's comparator.
    //------------------------------------------------------------------------
    function automatic tb_gt(input [31:0] a, input [31:0] b);
        reg sa, sb, anan, bnan, az, bz;
        reg [30:0] ma, mb;
        begin
            sa = a[31];  sb = b[31];
            ma = a[30:0];  mb = b[30:0];
            anan = (a[30:23] == 8'hFF) && (a[22:0] != 0);
            bnan = (b[30:23] == 8'hFF) && (b[22:0] != 0);
            az = (ma == 0);
            bz = (mb == 0);
            if (anan)            tb_gt = 1'b0;             // NaN never greater
            else if (bnan)       tb_gt = 1'b1;             // anything > NaN
            else if (az && bz)   tb_gt = 1'b0;             // +0 == -0
            else if (sa != sb)   tb_gt = (sa == 1'b0);     // positive wins
            else if (sa == 1'b0) tb_gt = (ma > mb);        // both >=0
            else                 tb_gt = (ma < mb);        // both <0
        end
    endfunction

    //------------------------------------------------------------------------
    // Build the golden permutation: selection sort by (-score, +index).
    // For each output rank, scan ALL candidates and pick the one that is
    // strictly-greater than the running best, OR (tie) has a LOWER index.
    // This is the same total order the DUT realizes, computed independently.
    //------------------------------------------------------------------------
    task build_golden;
        integer r, c, best;
        reg [N-1:0] used;
        begin
            used = {N{1'b0}};
            for (r = 0; r < N; r = r + 1) begin
                best = -1;
                for (c = 0; c < N; c = c + 1) begin
                    if (!used[c]) begin
                        if (best == -1) best = c;
                        // c beats current best iff strictly greater; on a tie the
                        // already-chosen `best` is the lower index (we scan c
                        // ascending, so best is always <= c on ties) -> keep best.
                        else if (tb_gt(scores[c], scores[best])) best = c;
                    end
                end
                perm[r]   = best;
                gscore[r] = scores[best];
                used[best] = 1'b1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Drive one full transaction: pulse start, service the pull-load handshake
    // for NBEATS beats (LANES_IN scores/beat), wait for done, then CHECK.
    //------------------------------------------------------------------------
    task run_one;
        integer beat;
        reg [SCORE_W-1:0] g_idx, d_idx;
        reg [SCORE_W-1:0] d_score;
        reg [N-1:0] exp_mask;
        integer wd;
        begin
            build_golden;

            // pulse start
            @(posedge clk); start <= 1'b1;
            @(posedge clk); start <= 1'b0;

            // service the load: feed beats when load_req is high.
            beat = 0;
            score_valid <= 1'b0;
            while (beat < NBEATS) begin
                @(posedge clk);
                if (load_req) begin
                    for (lane = 0; lane < LANES_IN; lane = lane + 1)
                        score_in[SCORE_W*lane +: SCORE_W] <= scores[beat*LANES_IN + lane];
                    score_valid <= 1'b1;
                    beat = beat + 1;
                end else begin
                    score_valid <= 1'b0;
                end
            end
            @(posedge clk);
            score_valid <= 1'b0;

            // wait for done (bounded)
            wd = 0;
            while (done !== 1'b1 && wd < (N + 4*K + 64)) begin
                @(posedge clk); wd = wd + 1;
            end
            if (done !== 1'b1) begin
                $display("FAIL [N=%0d K=%0d L=%0d]: done never asserted", N, K, LANES_IN);
                fail = fail + 1;
                disable run_one;
            end

            // ---- CHECK per-rank index + score, and sel_valid ----
            exp_mask = {N{1'b0}};
            for (s = 0; s < KEFF; s = s + 1) begin
                g_idx   = perm[s];
                d_idx   = {{(32-IDXW){1'b0}}, sel_idx_o[s*IDXW +: IDXW]};
                d_score = sel_score_o[s*SCORE_W +: SCORE_W];
                exp_mask[perm[s]] = 1'b1;

                if (d_idx !== g_idx) begin
                    $display("FAIL [N=%0d K=%0d L=%0d] rank %0d: idx dut=%0d gold=%0d (score gold=%h)",
                             N, K, LANES_IN, s, d_idx, g_idx, gscore[s]);
                    fail = fail + 1;
                    disable run_one;
                end
                if (d_score !== gscore[s]) begin
                    $display("FAIL [N=%0d K=%0d L=%0d] rank %0d idx=%0d: score dut=%h gold=%h",
                             N, K, LANES_IN, s, g_idx, d_score, gscore[s]);
                    fail = fail + 1;
                    disable run_one;
                end
                if (sel_valid_o[s] !== 1'b1) begin
                    $display("FAIL [N=%0d K=%0d L=%0d] rank %0d: sel_valid=0 (should be valid)",
                             N, K, LANES_IN, s);
                    fail = fail + 1;
                    disable run_one;
                end
            end
            // surplus slots (K>N) must be invalid
            for (s = KEFF; s < K; s = s + 1) begin
                if (sel_valid_o[s] !== 1'b0) begin
                    $display("FAIL [N=%0d K=%0d L=%0d] surplus slot %0d: sel_valid=1 (should be 0)",
                             N, K, LANES_IN, s);
                    fail = fail + 1;
                    disable run_one;
                end
            end
            // mask must equal the union of the KEFF selected indices
            if (mask_o !== exp_mask) begin
                $display("FAIL [N=%0d K=%0d L=%0d]: mask dut=%h gold=%h", N, K, LANES_IN, mask_o, exp_mask);
                fail = fail + 1;
                disable run_one;
            end

            pass = pass + 1;
        end
    endtask

    //------------------------------------------------------------------------
    // random fp32 generators
    //------------------------------------------------------------------------
    reg [63:0] rng;
    function automatic [31:0] rnd32;
        begin
            // xorshift64
            rng = rng ^ (rng << 13);
            rng = rng ^ (rng >> 7);
            rng = rng ^ (rng << 17);
            rnd32 = rng[31:0];
        end
    endfunction

    // a "tame" fp32 with bounded exponent (avoid inf/nan unless asked) and
    // mixed sign: exponent in [0x60,0x9E] gives ~[1e-9,1e9] magnitudes.
    function automatic [31:0] rnd_fp_wide;
        reg [31:0] r; reg [7:0] e;
        begin
            r = rnd32();
            e = 8'h60 + (r[14:8] % 8'h3F);   // 0x60..0x9E
            rnd_fp_wide = {r[31], e, r[22:0]};
        end
    endfunction

    // a sigmoid-gate-like value in [0,1): exponent <= 0x7F (>=0.5 region down).
    function automatic [31:0] rnd_fp_unit;
        reg [31:0] r; reg [7:0] e;
        begin
            r = rnd32();
            e = 8'h6F + (r[12:8] % 8'h11);   // 0x6F..0x7F  -> ~[~6e-5, ~2)
            if (e > 8'h7E) e = 8'h7E;        // keep < 1.0-ish
            rnd_fp_unit = {1'b0, e, r[22:0]};
        end
    endfunction

    //------------------------------------------------------------------------
    // the suite
    //------------------------------------------------------------------------
    integer v;
    initial begin
        pass = 0; fail = 0; done_all = 1'b0;
        start = 1'b0; score_valid = 1'b0; score_in = {LANES_IN*SCORE_W{1'b0}};
        rng = 64'hDEADBEEF_00C0FFEE ^ (N*K*LANES_IN + 64'h1234567);
        rst = 1'b1;
        @(posedge go);                    // wait for the top to release us
        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        // ---- DIRECTED EDGE CASES ----

        // E1: distinct ascending magnitudes (largest index = largest score).
        //   Build by MANTISSA only so the exponent is fixed (0x80) and the value
        //   is always a finite normal -- never accidentally exp=0xFF (NaN/inf),
        //   which for large N could otherwise land in a valid K==N slot.
        for (i = 0; i < N; i = i + 1)
            scores[i] = {1'b0, 8'h80, i[22:0]};   // 2.0 * (1 + i*2^-23), strictly increasing
        run_one;

        // E2: all-equal scores -> tie-break determinism (golden = ascending idx)
        for (i = 0; i < N; i = i + 1) scores[i] = 32'h3F800000; // 1.0
        run_one;

        // E3: single dominant outlier at a middle index
        for (i = 0; i < N; i = i + 1) scores[i] = 32'h3DCCCCCD;  // ~0.1
        scores[N/2] = 32'h42C80000;                              // 100.0
        run_one;

        // E4: all negative (smaller magnitude == larger value)
        for (i = 0; i < N; i = i + 1)
            scores[i] = {1'b1, 8'h7F, i[6:0], 16'h0000};         // -1.xxx, varied
        run_one;

        // E5: duplicate maxima (several equal top values) + filler below
        for (i = 0; i < N; i = i + 1) scores[i] = 32'h3F000000;  // 0.5 filler
        for (i = 0; i < (KEFF+2 < N ? KEFF+2 : N); i = i + 1)
            scores[(i*3) % N] = 32'h40000000;                    // 2.0 dups
        run_one;

        // E6: signed zeros mixed with tiny values (+0/-0 must compare equal)
        for (i = 0; i < N; i = i + 1) scores[i] = (i[0]) ? 32'h00000000 : 32'h80000000;
        if (N > 2) scores[N-1] = 32'h3F800000;                   // one real max
        run_one;

        // E7: NaNs are STRUCTURALLY EXCLUDED garbage (DUT contract: a NaN-fed
        //   lane is masked-off and is the SMALLEST value, so it must never be
        //   SELECTED).  We therefore exercise NaNs only when at least 2 slots
        //   sit BELOW the selection cutoff (KEFF+2 <= N) -- i.e. the NaNs land
        //   in the non-selected tail, exactly their documented use (the K>=N
        //   dense harnesses, where every slot is returned, never feed NaN).
        //   This proves NaNs never beat a real candidate; the all-NaN-remaining
        //   case (selecting a NaN) is outside the unit's contract.
        if (KEFF + 2 <= N) begin
            for (i = 0; i < N; i = i + 1)
                scores[i] = {1'b0, 8'h80, 23'h001000 + i[10:0]}; // small positives
            // put the NaNs at the highest indices so they are guaranteed to be
            // in the non-selected tail (lowest "value" -> last in the order).
            scores[N-1] = 32'h7FC00000;                          // +NaN
            scores[N-2] = 32'hFFC00001;                          // -NaN
            run_one;
        end

        // E8: sigmoid gates in [0,1)
        for (i = 0; i < N; i = i + 1) scores[i] = rnd_fp_unit();
        run_one;

        // ---- RANDOM VECTORS (wide dynamic range, mixed sign) ----
        for (v = 0; v < NVEC; v = v + 1) begin
            for (i = 0; i < N; i = i + 1) begin
                // mix: 75% wide-range signed, 25% unit-interval gates
                if ((rnd32() & 32'h3) == 32'h0) scores[i] = rnd_fp_unit();
                else                            scores[i] = rnd_fp_wide();
            end
            run_one;
        end

        done_all = 1'b1;
    end
endmodule

// ----------------------------------------------------------------------------
// TOP: run all harness sizes in parallel, gather pass/fail, print verdict.
// ----------------------------------------------------------------------------
module topk_select_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg go = 1'b0;

    // verdict accumulators
    integer p0,f0, p1,f1, p2,f2, p3,f3, p4,f4, p5,f5, p6,f6;
    wire d0,d1,d2,d3,d4,d5,d6;

    // (N=256,K=8) router        LANES_IN=1
    tk_harness #(.N(256), .K(8),   .LANES_IN(1), .NVEC(40)) h0 (clk, go, d0, p0, f0);
    // (N=32, K=8) small DSA      LANES_IN=1
    tk_harness #(.N(32),  .K(8),   .LANES_IN(1), .NVEC(80)) h1 (clk, go, d1, p1, f1);
    // (N=64, K=16) larger DSA    LANES_IN=4
    tk_harness #(.N(64),  .K(16),  .LANES_IN(4), .NVEC(60)) h2 (clk, go, d2, p2, f2);
    // (N=128,K=12) mid DSA       LANES_IN=8
    tk_harness #(.N(128), .K(12),  .LANES_IN(8), .NVEC(40)) h3 (clk, go, d3, p3, f3);
    // (N=16, K=16) K==N dense    LANES_IN=2
    tk_harness #(.N(16),  .K(16),  .LANES_IN(2), .NVEC(80)) h4 (clk, go, d4, p4, f4);
    // (N=200,K=200) K>=N dense   LANES_IN=1
    tk_harness #(.N(200), .K(200), .LANES_IN(1), .NVEC(8)) h5 (clk, go, d5, p5, f5);
    // (N=64, K=1) pure argmax    LANES_IN=1
    tk_harness #(.N(64),  .K(1),   .LANES_IN(1), .NVEC(80)) h6 (clk, go, d6, p6, f6);

    integer total_pass, total_fail;
    integer guard;

    initial begin
        @(posedge clk);
        go = 1'b1;            // release all harnesses
        @(posedge clk);
        go = 1'b0;

        // wait for every harness to finish (bounded)
        guard = 0;
        while (!(d0 && d1 && d2 && d3 && d4 && d5 && d6) && guard < 4_000_000) begin
            @(posedge clk); guard = guard + 1;
        end

        if (!(d0 && d1 && d2 && d3 && d4 && d5 && d6)) begin
            $display("FAIL: timeout, not all harnesses completed (guard=%0d)", guard);
            $fatal(1, "topk_select_tb timeout");
        end

        total_pass = p0+p1+p2+p3+p4+p5+p6;
        total_fail = f0+f1+f2+f3+f4+f5+f6;

        $display("------------------------------------------------------------");
        $display(" topk_select_tb results by configuration:");
        $display("   N=256 K=8   L=1 : pass=%0d fail=%0d", p0, f0);
        $display("   N=32  K=8   L=1 : pass=%0d fail=%0d", p1, f1);
        $display("   N=64  K=16  L=4 : pass=%0d fail=%0d", p2, f2);
        $display("   N=128 K=12  L=8 : pass=%0d fail=%0d", p3, f3);
        $display("   N=16  K=16  L=2 : pass=%0d fail=%0d", p4, f4);
        $display("   N=200 K=200 L=1 : pass=%0d fail=%0d", p5, f5);
        $display("   N=64  K=1   L=1 : pass=%0d fail=%0d", p6, f6);
        $display("------------------------------------------------------------");

        if (total_fail != 0)
            $fatal(1, "topk_select_tb: %0d MISMATCH(es)", total_fail);

        $display("ALL %0d TESTS PASSED", total_pass);
        $finish;
    end
endmodule
