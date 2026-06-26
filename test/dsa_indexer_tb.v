`timescale 1ns/1ps
//============================================================================
// dsa_indexer_tb.v  --  self-checking TB for dsa_indexer (GLM-5.2 DSA indexer)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN
//   For each key j we compute score_j = dot(q_idx, k_idx[j]) over IDX_DIM in
//   Verilog `real` (fp64), widening each bf16 operand to its exact real value
//   (bf16 = top 16 bits of fp32, so the value is exact in fp64).  We then take
//   the top-TOPK key indices by DESCENDING score with a LOWER-INDEX tie-break
//   -- exactly topk_select's order (strict-greater displacement keeps the lower
//   index on equal scores).  The DUT's selected index SET is compared slot-by-
//   slot (which, for distinct scores, is an exact ordered match; for the
//   all-equal directed case the golden replicates the lower-index ordering),
//   plus the valid count.  No DUT arithmetic is shared.
//
//   SPARSE (S>TOPK)  : compare ranks 0..TOPK-1 against gperm[], count == TOPK.
//   DENSE  (S<=TOPK) : indexer is a NO-OP, expect indices 0..S-1, count == S.
//
//   X-AWARE: any X bit in sel_count or in any compared sel_idx slot -> FAIL.
//
// COVERAGE
//   Multiple (IDX_DIM,S_MAX,TOPK) harness shapes.  Directed: one clearly-best
//   key, near-tie keys (tie-break stress), ALL-EQUAL scores (pure index order),
//   S==TOPK boundary (dense), S==TOPK+1 (smallest sparse), S==1, S==S_MAX.
//   Randomized: wide-range MIXED-SIGN q/k bf16 vectors with per-key distinct
//   magnitudes so the top-K ordering is unambiguous under bf16/fp32 rounding.
//============================================================================
module dsa_harness #(
    parameter integer IDX_DIM = 4,
    parameter integer S_MAX   = 16,
    parameter integer TOPK    = 4,
    parameter integer NVEC    = 48
)(
    input  wire    clk,
    input  wire    go,
    output reg     done_all,
    output integer pass,
    output integer fail
);
    localparam integer IDXW = (S_MAX <= 1) ? 1 : $clog2(S_MAX);

    reg                      rst;
    reg                      start;
    reg  [IDX_DIM*16-1:0]    q_idx;
    reg  [IDXW:0]            s_len;
    wire                     key_req;
    wire [IDXW-1:0]          key_idx;
    reg  [IDX_DIM*16-1:0]    k_idx;
    reg                      key_valid;
    wire [TOPK*IDXW-1:0]     sel_idx;
    wire [IDXW:0]            sel_count;
    wire                     busy;
    wire                     done;

    dsa_indexer #(.IDX_DIM(IDX_DIM), .S_MAX(S_MAX), .TOPK(TOPK)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .q_idx(q_idx), .s_len(s_len),
        .key_req(key_req), .key_idx(key_idx), .k_idx(k_idx), .key_valid(key_valid),
        .sel_idx(sel_idx), .sel_count(sel_count)
    );

    // ---- test data ----
    reg  [15:0] qv [0:IDX_DIM-1];
    reg  [15:0] kv [0:S_MAX-1][0:IDX_DIM-1];
    real        score [0:S_MAX-1];          // golden real dot product per key
    integer     gperm [0:S_MAX-1];          // golden ranking (descending score)
    integer     S;

    integer i, j, d;

    // bf16 (top 16 bits of fp32) -> EXACT real, for the golden dot product.
    function automatic real bf16_to_real(input [15:0] b);
        reg [31:0] f; integer e, mi; real m, v;
        begin
            f = {b, 16'h0000};
            if (f[30:23] == 8'h00) v = 0.0;          // subnormal/zero -> FTZ
            else if (f[30:23] == 8'hFF) v = 0.0;     // inf/nan: not used in vectors
            else begin
                e = f[30:23] - 127;
                m = 1.0;
                for (mi = 0; mi < 23; mi = mi + 1)
                    if (f[mi]) m = m + (2.0 ** (mi - 23));
                v = m * (2.0 ** e);
                if (f[31]) v = -v;
            end
            bf16_to_real = v;
        end
    endfunction

    // Golden ranking: descending score, lower-index tie-break (strict >).
    task build_golden;
        integer r, c, best; reg [S_MAX-1:0] used;
        begin
            for (j = 0; j < S; j = j + 1) begin
                score[j] = 0.0;
                for (d = 0; d < IDX_DIM; d = d + 1)
                    score[j] = score[j] + bf16_to_real(qv[d]) * bf16_to_real(kv[j][d]);
            end
            used = {S_MAX{1'b0}};
            for (r = 0; r < S; r = r + 1) begin
                best = -1;
                for (c = 0; c < S; c = c + 1)
                    if (!used[c]) begin
                        if (best == -1) best = c;
                        else if (score[c] > score[best]) best = c;  // strict: lower idx on tie
                    end
                gperm[r]   = best;
                used[best] = 1'b1;
            end
        end
    endtask

    integer kff;
    task run_one;
        integer s, wd;
        reg [IDXW-1:0] g, dsel;
        begin
            build_golden;
            kff = (S < TOPK) ? S : TOPK;

            for (d = 0; d < IDX_DIM; d = d + 1) q_idx[16*d +: 16] = qv[d];
            s_len = S[IDXW:0];

            @(posedge clk); start <= 1'b1;
            @(posedge clk); start <= 1'b0;

            // service the unit's key pulls
            wd = 0;
            key_valid <= 1'b0;
            while (done !== 1'b1 && wd < (S*IDX_DIM*20 + S_MAX*8 + 600)) begin
                @(posedge clk);
                if (key_req) begin
                    for (d = 0; d < IDX_DIM; d = d + 1)
                        k_idx[16*d +: 16] <= kv[key_idx][d];
                    key_valid <= 1'b1;
                end else begin
                    key_valid <= 1'b0;
                end
                wd = wd + 1;
            end

            if (done !== 1'b1) begin
                $display("FAIL [DIM=%0d Smax=%0d K=%0d S=%0d]: done never asserted",
                         IDX_DIM, S_MAX, TOPK, S);
                fail = fail + 1; disable run_one;
            end

            // ---- X-AWARE count check ----
            if (^sel_count === 1'bx) begin
                $display("FAIL [DIM=%0d Smax=%0d K=%0d S=%0d]: sel_count has X (%b)",
                         IDX_DIM, S_MAX, TOPK, S, sel_count);
                fail = fail + 1; disable run_one;
            end
            if (sel_count !== kff[IDXW:0]) begin
                $display("FAIL [DIM=%0d Smax=%0d K=%0d S=%0d]: count dut=%0d gold=%0d",
                         IDX_DIM, S_MAX, TOPK, S, sel_count, kff);
                fail = fail + 1; disable run_one;
            end

            if (S <= TOPK) begin
                // DENSE no-op: indices 0..S-1 in order.
                for (s = 0; s < kff; s = s + 1) begin
                    dsel = sel_idx[s*IDXW +: IDXW];
                    if (^dsel === 1'bx) begin
                        $display("FAIL [DIM=%0d Smax=%0d K=%0d S=%0d] dense slot %0d: idx X",
                                 IDX_DIM, S_MAX, TOPK, S, s);
                        fail = fail + 1; disable run_one;
                    end
                    if (dsel !== s[IDXW-1:0]) begin
                        $display("FAIL [DIM=%0d Smax=%0d K=%0d S=%0d] dense slot %0d: dut=%0d expect=%0d",
                                 IDX_DIM, S_MAX, TOPK, S, s, dsel, s);
                        fail = fail + 1; disable run_one;
                    end
                end
            end else begin
                // SPARSE: top-K by score, descending, lower-index tie-break.
                for (s = 0; s < kff; s = s + 1) begin
                    g    = gperm[s][IDXW-1:0];
                    dsel = sel_idx[s*IDXW +: IDXW];
                    if (^dsel === 1'bx) begin
                        $display("FAIL [DIM=%0d Smax=%0d K=%0d S=%0d] rank %0d: idx X",
                                 IDX_DIM, S_MAX, TOPK, S, s);
                        fail = fail + 1; disable run_one;
                    end
                    if (dsel !== g) begin
                        $display("FAIL [DIM=%0d Smax=%0d K=%0d S=%0d] rank %0d: dut=%0d gold=%0d (score=%f)",
                                 IDX_DIM, S_MAX, TOPK, S, s, dsel, g, score[gperm[s]]);
                        fail = fail + 1; disable run_one;
                    end
                end
            end
            pass = pass + 1;
        end
    endtask

    // ---- bf16 generators ----
    reg [63:0] rng;
    function automatic [31:0] rnd32;
        begin
            rng = rng ^ (rng << 13); rng = rng ^ (rng >> 7); rng = rng ^ (rng << 17);
            rnd32 = rng[31:0];
        end
    endfunction
    // positive integer -> bf16 (exact for small ints).
    function automatic [15:0] bf_from_int(input integer iv);
        reg [31:0] f; integer e, msb, m, k;
        begin
            if (iv == 0) bf_from_int = 16'h0000;
            else begin
                msb = 0;
                for (k = 0; k < 24; k = k + 1) if ((iv >> k) & 1) msb = k;
                e = 127 + msb;
                m = (iv - (1 << msb));
                f = {1'b0, e[7:0], 23'b0};
                f = f | (m << (23 - msb));
                bf_from_int = f[31:16];
            end
        end
    endfunction
    // MIXED-SIGN bf16 over a wide range: exp 0x74..0x84 (~[0.004,16)), random sign.
    function automatic [15:0] rnd_bf_signed;
        reg [31:0] r; reg [7:0] e;
        begin
            r = rnd32();
            e = 8'h74 + (r[12:8] % 8'h11);   // 0x74..0x84
            rnd_bf_signed = {r[31], e, r[22:16]};
        end
    endfunction

    integer v;
    reg [7:0] kbias;
    initial begin
        pass = 0; fail = 0; done_all = 1'b0;
        start = 1'b0; key_valid = 1'b0; q_idx = 0; k_idx = 0; s_len = 0;
        rng = 64'hDA7A_5EED_1234567 ^ (IDX_DIM*S_MAX*131 + TOPK*7919 + 64'h9E3779B97F4A7C15);
        rst = 1'b1;
        @(posedge go);
        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        // ---- DIRECTED ----

        // D1 SPARSE: one CLEARLY-BEST key in the middle (huge), rest tiny.
        for (d = 0; d < IDX_DIM; d = d + 1) qv[d] = 16'h3F80;            // 1.0
        for (j = 0; j < S_MAX; j = j + 1)
            for (d = 0; d < IDX_DIM; d = d + 1) kv[j][d] = 16'h3DCC;     // ~0.1
        for (d = 0; d < IDX_DIM; d = d + 1) kv[S_MAX/2][d] = 16'h4248;   // ~50 dominant
        S = S_MAX; run_one;

        // D2 SPARSE: monotonically increasing per-key scores (well separated)
        //   -> top ranks are the highest indices, descending order.
        for (d = 0; d < IDX_DIM; d = d + 1) qv[d] = 16'h3F80;            // 1.0
        for (j = 0; j < S_MAX; j = j + 1)
            for (d = 0; d < IDX_DIM; d = d + 1) kv[j][d] = bf_from_int(j + 1);
        S = S_MAX; run_one;

        // D3 NEAR-TIE / TIE-BREAK stress: several keys with IDENTICAL scores
        //   plus a couple distinct ones.  Equal-score keys must come out in
        //   ASCENDING index order (lower-index tie-break).
        for (d = 0; d < IDX_DIM; d = d + 1) qv[d] = 16'h3F80;            // 1.0
        for (j = 0; j < S_MAX; j = j + 1)
            for (d = 0; d < IDX_DIM; d = d + 1) kv[j][d] = 16'h4000;     // 2.0 (all equal)
        // make two keys distinctly larger so they rank above the tie block
        for (d = 0; d < IDX_DIM; d = d + 1) begin
            kv[S_MAX-1][d] = 16'h4080;                                   // 4.0 (best)
            kv[1][d]       = 16'h4040;                                   // 3.0 (2nd)
        end
        S = S_MAX; run_one;

        // D4 ALL-EQUAL scores: pure index order, ranks 0..TOPK-1 = keys 0..TOPK-1.
        for (d = 0; d < IDX_DIM; d = d + 1) qv[d] = 16'h3F80;
        for (j = 0; j < S_MAX; j = j + 1)
            for (d = 0; d < IDX_DIM; d = d + 1) kv[j][d] = 16'h3F80;     // all 1.0
        S = S_MAX; run_one;

        // D5 SPARSE smallest case: S == TOPK+1.
        if (TOPK + 1 <= S_MAX) begin
            for (d = 0; d < IDX_DIM; d = d + 1) qv[d] = 16'h3F80;
            for (j = 0; j < S_MAX; j = j + 1)
                for (d = 0; d < IDX_DIM; d = d + 1)
                    kv[j][d] = bf_from_int(((j*7 + 3) % 17) + 1);
            S = TOPK + 1; run_one;
        end

        // D6 DENSE boundary: S == TOPK -> emit 0..TOPK-1.
        for (d = 0; d < IDX_DIM; d = d + 1) qv[d] = 16'h3F80;
        for (j = 0; j < S_MAX; j = j + 1)
            for (d = 0; d < IDX_DIM; d = d + 1) kv[j][d] = rnd_bf_signed();
        S = TOPK; run_one;

        // D7 DENSE: S < TOPK.
        S = (TOPK > 1) ? (TOPK - 1) : 1; run_one;

        // D8 DENSE: S == 1 (single causal key).
        S = 1; run_one;

        // ---- RANDOM (wide-range MIXED-SIGN, per-key distinct magnitude) ----
        for (v = 0; v < NVEC; v = v + 1) begin
            for (d = 0; d < IDX_DIM; d = d + 1) qv[d] = rnd_bf_signed();
            // per-key DISTINCT integer scale * mixed-sign random -> separated sums.
            for (j = 0; j < S_MAX; j = j + 1)
                for (d = 0; d < IDX_DIM; d = d + 1) begin
                    kv[j][d] = rnd_bf_signed();
                    // bias the magnitude by a per-key distinct exponent factor
                    kbias = 8'h76 + ((j*5 + 1) % 13);   // 0x76..0x82
                    kv[j][d][14:7] = kbias;
                end
            case (v % 4)
                0: S = S_MAX;
                1: S = (TOPK + 1 <= S_MAX) ? TOPK + 1 : S_MAX;
                2: S = TOPK;
                default: S = (TOPK > 1) ? (TOPK - 1) : 1;
            endcase
            run_one;
        end

        done_all = 1'b1;
    end
endmodule

//----------------------------------------------------------------------------
module dsa_indexer_tb;
    reg clk = 1'b0; always #5 clk = ~clk;
    reg go = 1'b0;

    integer p0,f0, p1,f1, p2,f2, p3,f3;
    wire d0,d1,d2,d3;

    // (IDX_DIM, S_MAX, TOPK)
    dsa_harness #(.IDX_DIM(4),  .S_MAX(16), .TOPK(4), .NVEC(48)) h0 (clk, go, d0, p0, f0);
    dsa_harness #(.IDX_DIM(8),  .S_MAX(12), .TOPK(3), .NVEC(48)) h1 (clk, go, d1, p1, f1);
    dsa_harness #(.IDX_DIM(2),  .S_MAX(8),  .TOPK(1), .NVEC(40)) h2 (clk, go, d2, p2, f2);
    dsa_harness #(.IDX_DIM(16), .S_MAX(32), .TOPK(8), .NVEC(32)) h3 (clk, go, d3, p3, f3);

    integer total_pass, total_fail, guard;
    initial begin
        @(posedge clk); go = 1'b1;
        @(posedge clk); go = 1'b0;
        guard = 0;
        while (!(d0 && d1 && d2 && d3) && guard < 12_000_000) begin
            @(posedge clk); guard = guard + 1;
        end
        if (!(d0 && d1 && d2 && d3))
            $fatal(1, "dsa_indexer_tb timeout (guard=%0d)", guard);

        total_pass = p0+p1+p2+p3;
        total_fail = f0+f1+f2+f3;
        $display("------------------------------------------------------------");
        $display("  DIM=4  Smax=16 K=4 : pass=%0d fail=%0d", p0, f0);
        $display("  DIM=8  Smax=12 K=3 : pass=%0d fail=%0d", p1, f1);
        $display("  DIM=2  Smax=8  K=1 : pass=%0d fail=%0d", p2, f2);
        $display("  DIM=16 Smax=32 K=8 : pass=%0d fail=%0d", p3, f3);
        $display("------------------------------------------------------------");
        if (total_fail != 0) $fatal(1, "dsa_indexer_tb: %0d MISMATCH(es)", total_fail);
        $display("ALL %0d TESTS PASSED", total_pass);
        $finish;
    end
endmodule
