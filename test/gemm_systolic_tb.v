`include "tpu_defs.vh"
//============================================================================
// gemm_systolic_tb  --  self-checking unit TB for gemm_systolic  (SPEC §5.1)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL
//   The DUT computes C = A.B with fixed-point integer MACs in a skewed 4x4
//   systolic mesh, a 48-bit Q15.16 accumulator, and a round-half-up+saturate
//   narrowing to Q7.8.  The golden model here is COMPUTED A DIFFERENT WAY:
//   it converts each Q7.8 input element to a `real` (val/256.0), performs the
//   4x4 matmul ENTIRELY in floating point (real triple-nested loop, real
//   accumulation), and quantizes to Q7.8 ONLY at the boundary by scaling back
//   by 256.0, applying round-half-up, and saturating to [-32768, 32767].
//   The reference never uses the DUT's integer accumulation path, so the two
//   cannot share an arithmetic bug.
//
//   COMPARISON.  GEMM with round+saturate is an EXACT op (the reference applies
//   round/sat as a *separate final quantization step*, not as the accumulation
//   method, per SPEC §6), so the 16 C elements are compared BIT-EXACT (no
//   tolerance).  The reference's real accumulation of four Q14.16-scale
//   products is exactly representable in double precision (|product| < 2^31,
//   sum of 4 < 2^33, well within a 53-bit mantissa), so the only rounding is
//   the single boundary quantization -- identical to what the DUT performs on
//   its exact integer accumulator.  Hence bit-exact equality is the correct,
//   strict check.  `sat` is asserted to fire EXACTLY when the real (unrounded)
//   result of any element falls outside the Q7.8 representable range.
//
// MODELLED EXTERNAL MEMORY
//   A tiny behavioral tile memory lives INSIDE this TB (32 x 128b, combinational
//   read on two ports, synchronous write, synchronous reset).  It is wired to
//   the DUT's TM access ports.  No src/ module is instantiated except the DUT.
//
// COVERAGE
//   T1  identity:  A = I (Q7.8 1.0 on diagonal) => C == B            (bit-exact)
//   T2  zeros:     A = 0 => C == 0
//   T3  negatives: signed A,B with negative entries
//   T4  impulse/one-hot A picks single B rows
//   T5  all-equal small matrices
//   T6  all-max saturation: A,B = +max => product overflows Q7.8 => sat=1,
//       every C element clamps to +max; also a -max case clamping to -max.
//   T7  base-offset independence: run a GEMM at non-zero a/b/c bases.
//   T8  busy/done LATENCY assertion: start->done == 11 cycles (SPEC §3.3),
//       busy high across the op, done a 1-cycle pulse.
//   T9  constrained-random A,B (seeded $random) >= 200 vectors, bit-exact C,
//       and sat-flag-exactness checked against the real reference every vector.
//
// GATES: prints "ALL <N> TESTS PASSED"; $fatal on ANY mismatch.
//============================================================================
module gemm_systolic_tb;

    localparam integer N        = `GEMM_N;       // 4
    localparam integer LAT      = 11;            // SPEC §3.3 start->done target
    localparam integer NRAND    = 250;           // constrained-random vectors

    // ---- clock / reset ----
    reg clk;
    reg rst;

    // ---- DUT control ----
    reg                  start;
    reg  [`TM_IDX_W-1:0] a_base, b_base, c_base;
    wire                 busy, done, sat;

    // ---- DUT <-> modelled TM ----
    wire [`TM_IDX_W-1:0] tm_raddr1, tm_raddr2, tm_waddr;
    wire                 tm_we;
    wire [`LINE_W-1:0]   tm_rdata1, tm_rdata2, tm_wdata;

    // ============================ DUT ============================
    gemm_systolic dut (
        .clk(clk), .rst(rst),
        .start(start), .a_base(a_base), .b_base(b_base), .c_base(c_base),
        .busy(busy), .done(done), .sat(sat),
        .tm_raddr1(tm_raddr1), .tm_rdata1(tm_rdata1),
        .tm_raddr2(tm_raddr2), .tm_rdata2(tm_rdata2),
        .tm_we(tm_we), .tm_waddr(tm_waddr), .tm_wdata(tm_wdata)
    );

    // ==================== modelled tile memory ====================
    // 32 x 128b; combinational dual read; synchronous write; sync reset.
    reg [`LINE_W-1:0] tm [0:`TM_LINES-1];
    integer tmi;
    assign tm_rdata1 = tm[tm_raddr1];
    assign tm_rdata2 = tm[tm_raddr2];
    always @(posedge clk) begin
        if (rst) begin
            for (tmi = 0; tmi < `TM_LINES; tmi = tmi + 1)
                tm[tmi] <= {`LINE_W{1'b0}};
        end else if (tm_we) begin
            tm[tm_waddr] <= tm_wdata;
        end
    end

    // ---- bookkeeping ----
    integer pass, fail, t;
    integer seed;
    integer cyc;
    integer cov_sat;    // # checked cases where the golden saturated (sat==1)
    integer cov_nosat;  // # checked cases where it did not          (sat==0)

    // ======================================================================
    // SECOND-SIZE PROOF (parameter exercise):  a SECOND gemm_systolic
    // instantiated at N2=2 with its OWN modelled tile memory and an
    // INDEPENDENT 2x2 real-domain golden (computed below in the real/integer
    // domain, NOT mirrored from the DUT).  This actually exercises the N
    // parameter -- the N=2 unit has a different wavefront depth (LAST_T=4,
    // latency 5), different counter/index widths, and the parameter-correct
    // arr_k / wr_row paths.  The primary N=4 coverage above is left intact.
    // ======================================================================
    localparam integer N2   = 2;            // second instantiated GEMM size
    localparam integer LAT2 = 3*N2 - 2 + 1; // SPEC §3.3 latency = 2N-1 + N = 5

    // ---- DUT2 control ----
    reg                  start2;
    reg  [`TM_IDX_W-1:0] a_base2, b_base2, c_base2;
    wire                 busy2, done2, sat2;

    // ---- DUT2 <-> modelled TM2 ----
    wire [`TM_IDX_W-1:0] tm2_raddr1, tm2_raddr2, tm2_waddr;
    wire                 tm2_we;
    wire [`LINE_W-1:0]   tm2_rdata1, tm2_rdata2, tm2_wdata;

    gemm_systolic #(.N(N2)) dut2 (
        .clk(clk), .rst(rst),
        .start(start2), .a_base(a_base2), .b_base(b_base2), .c_base(c_base2),
        .busy(busy2), .done(done2), .sat(sat2),
        .tm_raddr1(tm2_raddr1), .tm_rdata1(tm2_rdata1),
        .tm_raddr2(tm2_raddr2), .tm_rdata2(tm2_rdata2),
        .tm_we(tm2_we), .tm_waddr(tm2_waddr), .tm_wdata(tm2_wdata)
    );

    // Second modelled tile memory (independent storage).
    reg [`LINE_W-1:0] tm2 [0:`TM_LINES-1];
    integer tm2i;
    assign tm2_rdata1 = tm2[tm2_raddr1];
    assign tm2_rdata2 = tm2[tm2_raddr2];
    always @(posedge clk) begin
        if (rst) begin
            for (tm2i = 0; tm2i < `TM_LINES; tm2i = tm2i + 1)
                tm2[tm2i] <= {`LINE_W{1'b0}};
        end else if (tm2_we) begin
            tm2[tm2_waddr] <= tm2_wdata;
        end
    end

    // N2-sized matrices + golden (separate storage from the N=4 set).
    reg signed [`ELEM_W-1:0] A2 [0:N2-1][0:N2-1];
    reg signed [`ELEM_W-1:0] B2 [0:N2-1][0:N2-1];
    reg signed [`ELEM_W-1:0] C2gold [0:N2-1][0:N2-1];
    reg                      sat2_gold;
    integer h2i, h2j, h2k;
    real    racc2;

    // ======================================================================
    // THIRD-SIZE PROOF (N3=3) -- the CRITICAL NON-POWER-OF-TWO case.  The
    // ORIGINAL "arr_k = tcnt[1:0]-i-j" mod-4 BIT-TRUNCATION is WRONG for N=3
    // (it pre-narrows tcnt to 2 bits = mod 4, not mod N).  The parameterized
    // unit uses a TRUE full-width subtraction, so a 3x3 GEMM is exact.  This
    // instance, checked against an INDEPENDENT 3x3 real golden, proves the fix
    // handles non-power-of-two N -- not just powers of two.
    // ======================================================================
    localparam integer N3   = 3;            // third instantiated GEMM size
    localparam integer LAT3 = 3*N3 - 2 + 1; // latency = 2N-1 + N = 8

    reg                  start3;
    reg  [`TM_IDX_W-1:0] a_base3, b_base3, c_base3;
    wire                 busy3, done3, sat3;
    wire [`TM_IDX_W-1:0] tm3_raddr1, tm3_raddr2, tm3_waddr;
    wire                 tm3_we;
    wire [`LINE_W-1:0]   tm3_rdata1, tm3_rdata2, tm3_wdata;

    gemm_systolic #(.N(N3)) dut3 (
        .clk(clk), .rst(rst),
        .start(start3), .a_base(a_base3), .b_base(b_base3), .c_base(c_base3),
        .busy(busy3), .done(done3), .sat(sat3),
        .tm_raddr1(tm3_raddr1), .tm_rdata1(tm3_rdata1),
        .tm_raddr2(tm3_raddr2), .tm_rdata2(tm3_rdata2),
        .tm_we(tm3_we), .tm_waddr(tm3_waddr), .tm_wdata(tm3_wdata)
    );

    reg [`LINE_W-1:0] tm3 [0:`TM_LINES-1];
    integer tm3i;
    assign tm3_rdata1 = tm3[tm3_raddr1];
    assign tm3_rdata2 = tm3[tm3_raddr2];
    always @(posedge clk) begin
        if (rst) begin
            for (tm3i = 0; tm3i < `TM_LINES; tm3i = tm3i + 1)
                tm3[tm3i] <= {`LINE_W{1'b0}};
        end else if (tm3_we) begin
            tm3[tm3_waddr] <= tm3_wdata;
        end
    end

    reg signed [`ELEM_W-1:0] A3 [0:N3-1][0:N3-1];
    reg signed [`ELEM_W-1:0] B3 [0:N3-1][0:N3-1];
    reg signed [`ELEM_W-1:0] C3gold [0:N3-1][0:N3-1];
    reg                      sat3_gold;
    integer h3i, h3j, h3k;
    real    racc3;

    // 10ns clock.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ----------------------------------------------------------------------
    // Q7.8 conversion helpers (TB side).
    // ----------------------------------------------------------------------
    // 16-bit signed Q7.8 -> real value.
    function real q78_to_real;
        input signed [`ELEM_W-1:0] q;
        begin
            q78_to_real = $itor(q) / 256.0;
        end
    endfunction

    // real -> Q7.8 with round-half-up + saturate (the BOUNDARY quantization).
    // Returns the 16-bit signed value; sets sat_out=1 iff clamped.
    // round-half-up: floor(x + 0.5).  This matches the DUT's
    //   (acc + (1<<(FRAC-1))) >>> FRAC  on the EXACT integer accumulator.
    function signed [`ELEM_W-1:0] real_to_q78;
        input real x;          // real result value (already in "units")
        input integer do_sat;  // 1 to saturate, 0 caller handles
        real scaled;
        real rnd;
        begin
            scaled = x * 256.0;                 // back to Q7.8 integer scale
            // round-half-up (ties toward +inf): floor(scaled + 0.5)
            rnd = $rtoi(floor_r(scaled + 0.5));
            if (rnd > 32767.0)
                real_to_q78 = `Q78_MAX;
            else if (rnd < -32768.0)
                real_to_q78 = `Q78_MIN;
            else
                real_to_q78 = $rtoi(rnd);
            // do_sat unused for value (always clamp); kept for API symmetry.
            if (do_sat == 0) ; // no-op
        end
    endfunction

    // real floor (iverilog: $floor available; provide a wrapper for clarity).
    function real floor_r;
        input real x;
        begin
            floor_r = $floor(x);
        end
    endfunction

    // ----------------------------------------------------------------------
    // Pack/unpack a TM line of four Q7.8 lanes (sign-extended into 32 bits).
    // ----------------------------------------------------------------------
    function [`LINE_W-1:0] pack_row;
        input signed [`ELEM_W-1:0] e0, e1, e2, e3;   // lane 0..3
        reg   [`LANE_W-1:0] l0, l1, l2, l3;
        begin
            l0 = {{(`LANE_W-`ELEM_W){e0[`ELEM_W-1]}}, e0};
            l1 = {{(`LANE_W-`ELEM_W){e1[`ELEM_W-1]}}, e1};
            l2 = {{(`LANE_W-`ELEM_W){e2[`ELEM_W-1]}}, e2};
            l3 = {{(`LANE_W-`ELEM_W){e3[`ELEM_W-1]}}, e3};
            pack_row = {l3, l2, l1, l0};
        end
    endfunction

    // Extract Q7.8 lane (low 16 bits) of a 128-bit line.
    function signed [`ELEM_W-1:0] lane_of;
        input [`LINE_W-1:0] line;
        input integer       lane;     // 0..3
        reg   [`LANE_W-1:0] raw;
        begin
            raw     = line[lane*`LANE_W +: `LANE_W];
            lane_of = raw[`ELEM_W-1:0];
        end
    endfunction

    // ----------------------------------------------------------------------
    // Test matrices (Q7.8 16-bit), the golden result, and golden sat flag.
    // ----------------------------------------------------------------------
    reg signed [`ELEM_W-1:0] A [0:N-1][0:N-1];
    reg signed [`ELEM_W-1:0] B [0:N-1][0:N-1];
    reg signed [`ELEM_W-1:0] Cgold [0:N-1][0:N-1];
    reg                      sat_gold;

    integer gi, gj, gk;
    real    racc;
    integer overflow_any;

    // Compute the INDEPENDENT real-typed golden product into Cgold + sat_gold.
    task compute_golden;
        begin
            sat_gold = 1'b0;
            for (gi = 0; gi < N; gi = gi + 1)
                for (gj = 0; gj < N; gj = gj + 1) begin
                    racc = 0.0;
                    for (gk = 0; gk < N; gk = gk + 1)
                        racc = racc + q78_to_real(A[gi][gk]) *
                                      q78_to_real(B[gk][gj]);
                    // Saturation fires iff the round-half-up Q7.8 integer
                    //   rnd = floor(racc*256 + 0.5)
                    // falls outside the signed 16-bit range [-32768, 32767].
                    // Since floor(x) >= K  <=>  x >= K  and  floor(x) < K <=> x < K
                    // for integer K, the real comparisons below are exact.
                    if ((racc * 256.0 + 0.5) >=  32768.0) sat_gold = 1'b1;
                    if ((racc * 256.0 + 0.5) <  -32768.0) sat_gold = 1'b1;
                    Cgold[gi][gj] = real_to_q78(racc, 1);
                end
        end
    endtask

    // ----------------------------------------------------------------------
    // Load A at line base `ab`, B at base `bb` into the modelled TM.
    // (Direct array poke is the TB modelling external memory, not the DUT.)
    // ----------------------------------------------------------------------
    task load_inputs;
        input [`TM_IDX_W-1:0] ab, bb;
        integer r;
        begin
            for (r = 0; r < N; r = r + 1) begin
                tm[ab + r[`TM_IDX_W-1:0]] =
                    pack_row(A[r][0], A[r][1], A[r][2], A[r][3]);
                tm[bb + r[`TM_IDX_W-1:0]] =
                    pack_row(B[r][0], B[r][1], B[r][2], B[r][3]);
            end
        end
    endtask

    // ----------------------------------------------------------------------
    // Run one GEMM: pulse start, wait for done, MEASURE start->done latency.
    // Returns measured latency in `meas_lat`; checks busy/done timing.
    // ----------------------------------------------------------------------
    integer meas_lat;
    reg     busy_seen;
    task run_gemm;
        input [`TM_IDX_W-1:0] ab, bb, cb;
        input [255:0]         tag;
        begin
            // Drive bases + a single-cycle start pulse.  We set inputs on a
            // negedge so they are stable for the next posedge; the posedge that
            // samples start=1 is "cycle 0" of the latency count.
            @(negedge clk);
            a_base = ab; b_base = bb; c_base = cb;
            start  = 1'b1;
            @(posedge clk);            // <-- P0: start sampled (IDLE -> RUN)
            #1 start = 1'b0;           // deassert start AFTER P0
            meas_lat  = 0;             // P0 already happened
            busy_seen = 1'b0;
            // Count posedges P1, P2, ... until `done` is observed high.
            while (done !== 1'b1) begin
                @(posedge clk);
                meas_lat = meas_lat + 1;
                if (busy === 1'b1) busy_seen = 1'b1;
                if (meas_lat > 100) begin
                    $display("FAIL[%0s] done never asserted (hang)", tag);
                    fail = fail + 1;
                    $fatal(1, "gemm_systolic hang");
                end
            end
            // `done` high now: meas_lat counts P0->done edges (== LAT == 11).
            if (meas_lat != LAT) begin
                $display("FAIL[%0s] latency=%0d expected=%0d",
                         tag, meas_lat, LAT);
                fail = fail + 1;
                $fatal(1, "gemm_systolic latency mismatch");
            end else begin
                pass = pass + 1;
            end
            if (!busy_seen) begin
                $display("FAIL[%0s] busy never asserted during op", tag);
                fail = fail + 1;
                $fatal(1, "gemm_systolic busy never high");
            end else begin
                pass = pass + 1;
            end
            // The LAST C row write is SYNCHRONOUS and commits on the posedge
            // AFTER `done`; wait one cycle so all 4 C rows are readable.
            @(posedge clk);
            #1;  // settle past the edge
        end
    endtask

    // ----------------------------------------------------------------------
    // After done: compare DUT C (read from modelled TM at cb) vs golden,
    // bit-exact, and check the sat flag against sat_gold.
    // ----------------------------------------------------------------------
    task check_result;
        input [`TM_IDX_W-1:0] cb;
        input [255:0]         tag;
        integer r, c;
        reg signed [`ELEM_W-1:0] got;
        begin
            for (r = 0; r < N; r = r + 1)
                for (c = 0; c < N; c = c + 1) begin
                    got = lane_of(tm[cb + r[`TM_IDX_W-1:0]], c);
                    if (got !== Cgold[r][c]) begin
                        $display("FAIL[%0s] C[%0d][%0d] got=%0d (0x%04x) exp=%0d (0x%04x)",
                                 tag, r, c, got, got, Cgold[r][c], Cgold[r][c]);
                        fail = fail + 1;
                        $fatal(1, "gemm_systolic C mismatch");
                    end else begin
                        pass = pass + 1;
                    end
                end
            // sat-flag exactness (DUT sat must EXACTLY equal the real-model
            // saturation prediction), and record sat coverage in both polarities.
            if (sat !== sat_gold) begin
                $display("FAIL[%0s] sat=%b expected=%b", tag, sat, sat_gold);
                fail = fail + 1;
                $fatal(1, "gemm_systolic sat mismatch");
            end else begin
                pass = pass + 1;
            end
            if (sat_gold) cov_sat   = cov_sat   + 1;
            else          cov_nosat = cov_nosat + 1;
        end
    endtask

    // Convenience: set A and B from a fill function, run, check at given bases.
    task do_case;
        input [`TM_IDX_W-1:0] ab, bb, cb;
        input [255:0]         tag;
        begin
            compute_golden;
            load_inputs(ab, bb);
            run_gemm(ab, bb, cb, tag);
            check_result(cb, tag);
        end
    endtask

    // ======================================================================
    // SECOND-SIZE (N2=2) helpers.  INDEPENDENT 2x2 real-domain golden, a
    // separate TM2, and a self-checking driver targeting the N2-parameterized
    // DUT2.  All comparisons are bit-exact (round/sat is the single boundary
    // quantization, as for N=4); the sat flag is checked against the real
    // prediction.  These feed the SAME pass/fail/coverage counters.
    // ======================================================================
    // Compute the independent real-typed 2x2 golden into C2gold + sat2_gold.
    task compute_golden2;
        begin
            sat2_gold = 1'b0;
            for (h2i = 0; h2i < N2; h2i = h2i + 1)
                for (h2j = 0; h2j < N2; h2j = h2j + 1) begin
                    racc2 = 0.0;
                    for (h2k = 0; h2k < N2; h2k = h2k + 1)
                        racc2 = racc2 + q78_to_real(A2[h2i][h2k]) *
                                        q78_to_real(B2[h2k][h2j]);
                    if ((racc2 * 256.0 + 0.5) >=  32768.0) sat2_gold = 1'b1;
                    if ((racc2 * 256.0 + 0.5) <  -32768.0) sat2_gold = 1'b1;
                    C2gold[h2i][h2j] = real_to_q78(racc2, 1);
                end
        end
    endtask

    // Load A2/B2 into TM2 (only the low N2 lanes carry data; the DUT reads
    // exactly N2 lanes per row, so the upper lanes are don't-care = 0 here).
    task load_inputs2;
        input [`TM_IDX_W-1:0] ab, bb;
        integer r;
        begin
            for (r = 0; r < N2; r = r + 1) begin
                tm2[ab + r[`TM_IDX_W-1:0]] =
                    pack_row(A2[r][0], A2[r][1], 16'sd0, 16'sd0);
                tm2[bb + r[`TM_IDX_W-1:0]] =
                    pack_row(B2[r][0], B2[r][1], 16'sd0, 16'sd0);
            end
        end
    endtask

    // Run one N2 GEMM on DUT2; measure start->done latency == LAT2 (==5),
    // assert busy is seen.  Mirrors run_gemm but targets the second DUT.
    integer meas_lat2;
    reg     busy_seen2;
    task run_gemm2;
        input [`TM_IDX_W-1:0] ab, bb, cb;
        input [255:0]         tag;
        begin
            @(negedge clk);
            a_base2 = ab; b_base2 = bb; c_base2 = cb;
            start2  = 1'b1;
            @(posedge clk);            // P0: start sampled
            #1 start2 = 1'b0;
            meas_lat2  = 0;
            busy_seen2 = 1'b0;
            while (done2 !== 1'b1) begin
                @(posedge clk);
                meas_lat2 = meas_lat2 + 1;
                if (busy2 === 1'b1) busy_seen2 = 1'b1;
                if (meas_lat2 > 100) begin
                    $display("FAIL[%0s] done2 never asserted (hang)", tag);
                    fail = fail + 1;
                    $fatal(1, "gemm_systolic N2 hang");
                end
            end
            if (meas_lat2 != LAT2) begin
                $display("FAIL[%0s] N2 latency=%0d expected=%0d",
                         tag, meas_lat2, LAT2);
                fail = fail + 1;
                $fatal(1, "gemm_systolic N2 latency mismatch");
            end else begin
                pass = pass + 1;
            end
            if (!busy_seen2) begin
                $display("FAIL[%0s] busy2 never asserted during op", tag);
                fail = fail + 1;
                $fatal(1, "gemm_systolic N2 busy never high");
            end else begin
                pass = pass + 1;
            end
            @(posedge clk);    // last C row commits the cycle after done2
            #1;
        end
    endtask

    // Compare DUT2 C (read from TM2 at cb) vs the 2x2 golden, bit-exact, and
    // check sat2 against sat2_gold.  Feeds the shared coverage counters.
    task check_result2;
        input [`TM_IDX_W-1:0] cb;
        input [255:0]         tag;
        integer r, c;
        reg signed [`ELEM_W-1:0] got;
        begin
            for (r = 0; r < N2; r = r + 1)
                for (c = 0; c < N2; c = c + 1) begin
                    got = lane_of(tm2[cb + r[`TM_IDX_W-1:0]], c);
                    if (got !== C2gold[r][c]) begin
                        $display("FAIL[%0s] N2 C[%0d][%0d] got=%0d (0x%04x) exp=%0d (0x%04x)",
                                 tag, r, c, got, got, C2gold[r][c], C2gold[r][c]);
                        fail = fail + 1;
                        $fatal(1, "gemm_systolic N2 C mismatch");
                    end else begin
                        pass = pass + 1;
                    end
                end
            if (sat2 !== sat2_gold) begin
                $display("FAIL[%0s] N2 sat=%b expected=%b", tag, sat2, sat2_gold);
                fail = fail + 1;
                $fatal(1, "gemm_systolic N2 sat mismatch");
            end else begin
                pass = pass + 1;
            end
            if (sat2_gold) cov_sat   = cov_sat   + 1;
            else           cov_nosat = cov_nosat + 1;
        end
    endtask

    task do_case2;
        input [`TM_IDX_W-1:0] ab, bb, cb;
        input [255:0]         tag;
        begin
            compute_golden2;
            load_inputs2(ab, bb);
            run_gemm2(ab, bb, cb, tag);
            check_result2(cb, tag);
        end
    endtask

    // Constrained-random fill for the N2 matrices (independent seed stream).
    integer f2i, f2j;
    task fill_rand2; input integer which; input integer lo; input integer hi;
        integer rv, span;
        begin
            span = hi - lo + 1;
            for (f2i=0; f2i<N2; f2i=f2i+1) for (f2j=0; f2j<N2; f2j=f2j+1) begin
                rv = lo + ({$random(seed)} % span);
                if (which==0) A2[f2i][f2j]=rv[`ELEM_W-1:0];
                else          B2[f2i][f2j]=rv[`ELEM_W-1:0];
            end
        end
    endtask

    // ======================================================================
    // THIRD-SIZE (N3=3) helpers.  INDEPENDENT 3x3 real golden, separate TM3,
    // self-checking driver on DUT3.  Bit-exact + sat-checked; feeds the shared
    // counters.  This is the NON-POWER-OF-TWO size that distinguishes the new
    // true-subtraction arr_k from the original mod-4 bit-truncation.
    // ======================================================================
    task compute_golden3;
        begin
            sat3_gold = 1'b0;
            for (h3i = 0; h3i < N3; h3i = h3i + 1)
                for (h3j = 0; h3j < N3; h3j = h3j + 1) begin
                    racc3 = 0.0;
                    for (h3k = 0; h3k < N3; h3k = h3k + 1)
                        racc3 = racc3 + q78_to_real(A3[h3i][h3k]) *
                                        q78_to_real(B3[h3k][h3j]);
                    if ((racc3 * 256.0 + 0.5) >=  32768.0) sat3_gold = 1'b1;
                    if ((racc3 * 256.0 + 0.5) <  -32768.0) sat3_gold = 1'b1;
                    C3gold[h3i][h3j] = real_to_q78(racc3, 1);
                end
        end
    endtask

    // Load A3/B3 into TM3 (low N3 lanes carry data; lane 3 is don't-care = 0).
    task load_inputs3;
        input [`TM_IDX_W-1:0] ab, bb;
        integer r;
        begin
            for (r = 0; r < N3; r = r + 1) begin
                tm3[ab + r[`TM_IDX_W-1:0]] =
                    pack_row(A3[r][0], A3[r][1], A3[r][2], 16'sd0);
                tm3[bb + r[`TM_IDX_W-1:0]] =
                    pack_row(B3[r][0], B3[r][1], B3[r][2], 16'sd0);
            end
        end
    endtask

    integer meas_lat3;
    reg     busy_seen3;
    task run_gemm3;
        input [`TM_IDX_W-1:0] ab, bb, cb;
        input [255:0]         tag;
        begin
            @(negedge clk);
            a_base3 = ab; b_base3 = bb; c_base3 = cb;
            start3  = 1'b1;
            @(posedge clk);
            #1 start3 = 1'b0;
            meas_lat3  = 0;
            busy_seen3 = 1'b0;
            while (done3 !== 1'b1) begin
                @(posedge clk);
                meas_lat3 = meas_lat3 + 1;
                if (busy3 === 1'b1) busy_seen3 = 1'b1;
                if (meas_lat3 > 100) begin
                    $display("FAIL[%0s] done3 never asserted (hang)", tag);
                    fail = fail + 1;
                    $fatal(1, "gemm_systolic N3 hang");
                end
            end
            if (meas_lat3 != LAT3) begin
                $display("FAIL[%0s] N3 latency=%0d expected=%0d",
                         tag, meas_lat3, LAT3);
                fail = fail + 1;
                $fatal(1, "gemm_systolic N3 latency mismatch");
            end else begin
                pass = pass + 1;
            end
            if (!busy_seen3) begin
                $display("FAIL[%0s] busy3 never asserted during op", tag);
                fail = fail + 1;
                $fatal(1, "gemm_systolic N3 busy never high");
            end else begin
                pass = pass + 1;
            end
            @(posedge clk);
            #1;
        end
    endtask

    task check_result3;
        input [`TM_IDX_W-1:0] cb;
        input [255:0]         tag;
        integer r, c;
        reg signed [`ELEM_W-1:0] got;
        begin
            for (r = 0; r < N3; r = r + 1)
                for (c = 0; c < N3; c = c + 1) begin
                    got = lane_of(tm3[cb + r[`TM_IDX_W-1:0]], c);
                    if (got !== C3gold[r][c]) begin
                        $display("FAIL[%0s] N3 C[%0d][%0d] got=%0d (0x%04x) exp=%0d (0x%04x)",
                                 tag, r, c, got, got, C3gold[r][c], C3gold[r][c]);
                        fail = fail + 1;
                        $fatal(1, "gemm_systolic N3 C mismatch");
                    end else begin
                        pass = pass + 1;
                    end
                end
            if (sat3 !== sat3_gold) begin
                $display("FAIL[%0s] N3 sat=%b expected=%b", tag, sat3, sat3_gold);
                fail = fail + 1;
                $fatal(1, "gemm_systolic N3 sat mismatch");
            end else begin
                pass = pass + 1;
            end
            if (sat3_gold) cov_sat   = cov_sat   + 1;
            else           cov_nosat = cov_nosat + 1;
        end
    endtask

    task do_case3;
        input [`TM_IDX_W-1:0] ab, bb, cb;
        input [255:0]         tag;
        begin
            compute_golden3;
            load_inputs3(ab, bb);
            run_gemm3(ab, bb, cb, tag);
            check_result3(cb, tag);
        end
    endtask

    integer f3i, f3j;
    task fill_rand3; input integer which; input integer lo; input integer hi;
        integer rv, span;
        begin
            span = hi - lo + 1;
            for (f3i=0; f3i<N3; f3i=f3i+1) for (f3j=0; f3j<N3; f3j=f3j+1) begin
                rv = lo + ({$random(seed)} % span);
                if (which==0) A3[f3i][f3j]=rv[`ELEM_W-1:0];
                else          B3[f3i][f3j]=rv[`ELEM_W-1:0];
            end
        end
    endtask

    // ----------------------------------------------------------------------
    // Matrix fill helpers (directed).
    // ----------------------------------------------------------------------
    integer fi, fj;
    task fill_zero;   input integer which; // 0=A,1=B
        begin
            for (fi=0; fi<N; fi=fi+1) for (fj=0; fj<N; fj=fj+1)
                if (which==0) A[fi][fj]=16'sd0; else B[fi][fj]=16'sd0;
        end
    endtask
    task fill_ident;  input integer which; // 1.0 == 256 in Q7.8 on diagonal
        begin
            for (fi=0; fi<N; fi=fi+1) for (fj=0; fj<N; fj=fj+1) begin
                if (which==0) A[fi][fj] = (fi==fj) ? 16'sd256 : 16'sd0;
                else          B[fi][fj] = (fi==fj) ? 16'sd256 : 16'sd0;
            end
        end
    endtask
    task fill_const;  input integer which; input signed [`ELEM_W-1:0] v;
        begin
            for (fi=0; fi<N; fi=fi+1) for (fj=0; fj<N; fj=fj+1)
                if (which==0) A[fi][fj]=v; else B[fi][fj]=v;
        end
    endtask
    task fill_rand;   input integer which; input integer lo; input integer hi;
        integer rv, span;
        begin
            span = hi - lo + 1;
            for (fi=0; fi<N; fi=fi+1) for (fj=0; fj<N; fj=fj+1) begin
                rv = lo + ({$random(seed)} % span);
                if (which==0) A[fi][fj]=rv[`ELEM_W-1:0];
                else          B[fi][fj]=rv[`ELEM_W-1:0];
            end
        end
    endtask

    integer v;

    initial begin
        pass = 0; fail = 0; seed = 32'h1234_ABCD;
        cov_sat = 0; cov_nosat = 0;
        start = 1'b0; a_base=0; b_base=0; c_base=0;
        start2 = 1'b0; a_base2=0; b_base2=0; c_base2=0;
        start3 = 1'b0; a_base3=0; b_base3=0; c_base3=0;

        // ---- synchronous reset ----
        rst = 1'b1;
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ===============================================================
        // T1  identity: A = I  => C = B (small signed B).
        // ===============================================================
        fill_ident(0);
        B[0][0]= 16'sd100;  B[0][1]= -16'sd50; B[0][2]= 16'sd300;  B[0][3]= 16'sd0;
        B[1][0]= 16'sd7;    B[1][1]= 16'sd8;   B[1][2]= -16'sd9;   B[1][3]= 16'sd256;
        B[2][0]= -16'sd256; B[2][1]= 16'sd1;   B[2][2]= 16'sd2;    B[2][3]= -16'sd3;
        B[3][0]= 16'sd512;  B[3][1]= -16'sd512;B[3][2]= 16'sd64;   B[3][3]= 16'sd128;
        do_case(0, 4, 8, "T1-identity");

        // ===============================================================
        // T2  zeros: A = 0 => C = 0 (B arbitrary).
        // ===============================================================
        fill_zero(0);
        fill_const(1, 16'sd256);
        do_case(0, 4, 8, "T2-zeroA");

        // ===============================================================
        // T3  negatives: signed A,B with negative entries (modest magnitude).
        // ===============================================================
        A[0][0]=-16'sd256; A[0][1]= 16'sd128; A[0][2]=-16'sd64;  A[0][3]= 16'sd32;
        A[1][0]= 16'sd16;  A[1][1]=-16'sd16;  A[1][2]= 16'sd8;   A[1][3]=-16'sd8;
        A[2][0]= 16'sd256; A[2][1]= 16'sd256; A[2][2]=-16'sd256; A[2][3]=-16'sd256;
        A[3][0]=-16'sd100; A[3][1]= 16'sd50;  A[3][2]=-16'sd25;  A[3][3]= 16'sd12;
        B[0][0]= 16'sd64;  B[0][1]=-16'sd64;  B[0][2]= 16'sd32;  B[0][3]=-16'sd32;
        B[1][0]=-16'sd128; B[1][1]= 16'sd64;  B[1][2]=-16'sd32;  B[1][3]= 16'sd16;
        B[2][0]= 16'sd256; B[2][1]=-16'sd128; B[2][2]= 16'sd64;  B[2][3]=-16'sd32;
        B[3][0]=-16'sd16;  B[3][1]= 16'sd8;   B[3][2]=-16'sd4;   B[3][3]= 16'sd2;
        do_case(2, 6, 10, "T3-negatives");

        // ===============================================================
        // T4  impulse / one-hot A: A row i selects exactly B row (perm).
        //     A = a permutation-like one-hot picks single B rows.
        // ===============================================================
        fill_zero(0);
        A[0][2]=16'sd256; A[1][0]=16'sd256; A[2][3]=16'sd256; A[3][1]=16'sd256;
        B[0][0]= 16'sd1;  B[0][1]= 16'sd2;  B[0][2]= 16'sd3;  B[0][3]= 16'sd4;
        B[1][0]= 16'sd5;  B[1][1]= 16'sd6;  B[1][2]= 16'sd7;  B[1][3]= 16'sd8;
        B[2][0]= 16'sd9;  B[2][1]= 16'sd10; B[2][2]= 16'sd11; B[2][3]= 16'sd12;
        B[3][0]= 16'sd13; B[3][1]= 16'sd14; B[3][2]= 16'sd15; B[3][3]= 16'sd16;
        do_case(0, 4, 8, "T4-onehotA");

        // ===============================================================
        // T5  all-equal small matrices (0.5 == 128 in Q7.8).
        // ===============================================================
        fill_const(0, 16'sd128);
        fill_const(1, 16'sd128);
        do_case(0, 4, 8, "T5-allequal");

        // ===============================================================
        // T6a  all-max saturation: A=B=+max(0x7FFF) => huge positive => sat,
        //      every C element clamps to +32767.
        // ===============================================================
        fill_const(0, `Q78_MAX);
        fill_const(1, `Q78_MAX);
        do_case(0, 4, 8, "T6a-maxsat");
        if (sat_gold !== 1'b1) begin
            $display("FAIL[T6a] expected golden saturation"); fail=fail+1;
            $fatal(1, "T6a golden not saturating");
        end else pass = pass + 1;

        // ===============================================================
        // T6b  negative saturation: A=+max, B=-max => huge negative => sat,
        //      clamps to -32768.
        // ===============================================================
        fill_const(0, `Q78_MAX);
        fill_const(1, `Q78_MIN);
        do_case(0, 4, 8, "T6b-minsat");
        if (sat_gold !== 1'b1) begin
            $display("FAIL[T6b] expected golden saturation"); fail=fail+1;
            $fatal(1, "T6b golden not saturating");
        end else pass = pass + 1;

        // ===============================================================
        // T7  base-offset independence: run at non-zero bases.
        // ===============================================================
        fill_ident(0);
        fill_rand(1, -300, 300);
        do_case(16, 20, 24, "T7-bases");

        // ===============================================================
        // T8  explicit latency / busy / done-pulse re-check on a fresh op.
        //     (run_gemm already asserts LAT==11 and busy; here confirm `done`
        //      is exactly a 1-cycle pulse and the unit returns to idle.)
        // ===============================================================
        fill_const(0, 16'sd256);   // identity-ish scale 1.0
        fill_const(1, 16'sd1);
        compute_golden;
        load_inputs(0, 4);
        run_gemm(0, 4, 8, "T8-latency");
        check_result(8, "T8-latency");
        @(posedge clk);
        if (done === 1'b1) begin
            $display("FAIL[T8] done stayed high >1 cycle"); fail=fail+1;
            $fatal(1, "done not a single-cycle pulse");
        end else pass = pass + 1;
        if (busy === 1'b1) begin
            $display("FAIL[T8] busy still high after done"); fail=fail+1;
            $fatal(1, "busy not deasserted at idle");
        end else pass = pass + 1;

        // ===============================================================
        // T9  constrained-random A,B: >=200 vectors, bit-exact C + sat check.
        //     Magnitude range chosen to exercise BOTH in-range and saturating
        //     results so the sat flag is checked in both directions.
        // ===============================================================
        // Raw Q7.8 ranges chosen so BOTH polarities of the sat flag occur:
        //   * LARGE operands (|val| up to ~8000 raw = ~31.2): the length-4 dot
        //     product frequently exceeds the Q7.8 range (+-128.0) -> sat==1.
        //   * SMALL operands (|val| up to ~300 raw = ~1.17): products are tiny,
        //     results stay in range -> sat==0.
        // Both branches are bit-exact compared and have their sat flag checked.
        for (v = 0; v < NRAND; v = v + 1) begin
            if (v[0]) begin
                fill_rand(0, -8000, 8000);
                fill_rand(1, -8000, 8000);
            end else begin
                fill_rand(0, -300, 300);
                fill_rand(1, -300, 300);
            end
            do_case(5'd0, 5'd4, 5'd8, "T9-rand");
        end

        // ===============================================================
        // T10  SECOND-SIZE PROOF (N2=2):  exercise the gemm_systolic N
        //      parameter on a 2x2 instance with an INDEPENDENT 2x2 golden.
        //      Directed (identity / negatives / both-polarity saturation) +
        //      a constrained-random sweep, all bit-exact + sat-checked, plus
        //      the N2 latency (==5) and busy assertions inside run_gemm2.
        // ===============================================================
        // T10a  identity: A2 = I => C2 == B2 (small signed B2).
        A2[0][0]=16'sd256; A2[0][1]=16'sd0;
        A2[1][0]=16'sd0;   A2[1][1]=16'sd256;
        B2[0][0]=16'sd100; B2[0][1]=-16'sd50;
        B2[1][0]=16'sd512; B2[1][1]=16'sd7;
        do_case2(5'd0, 5'd2, 5'd4, "T10a-id2");

        // T10b  negatives / mixed signs.
        A2[0][0]=-16'sd256; A2[0][1]= 16'sd128;
        A2[1][0]= 16'sd64;  A2[1][1]=-16'sd32;
        B2[0][0]= 16'sd200; B2[0][1]=-16'sd300;
        B2[1][0]=-16'sd128; B2[1][1]= 16'sd256;
        do_case2(5'd6, 5'd8, 5'd10, "T10b-neg2");

        // T10c  positive saturation: A2=B2=+max => sat=1, clamps to +max.
        A2[0][0]=`Q78_MAX; A2[0][1]=`Q78_MAX;
        A2[1][0]=`Q78_MAX; A2[1][1]=`Q78_MAX;
        B2[0][0]=`Q78_MAX; B2[0][1]=`Q78_MAX;
        B2[1][0]=`Q78_MAX; B2[1][1]=`Q78_MAX;
        do_case2(5'd0, 5'd2, 5'd4, "T10c-maxsat2");
        if (sat2_gold !== 1'b1) begin
            $display("FAIL[T10c] expected golden saturation"); fail=fail+1;
            $fatal(1, "T10c golden not saturating");
        end else pass = pass + 1;

        // T10d  negative saturation: A2=+max, B2=-max => sat=1, clamps to -max.
        A2[0][0]=`Q78_MAX; A2[0][1]=`Q78_MAX;
        A2[1][0]=`Q78_MAX; A2[1][1]=`Q78_MAX;
        B2[0][0]=`Q78_MIN; B2[0][1]=`Q78_MIN;
        B2[1][0]=`Q78_MIN; B2[1][1]=`Q78_MIN;
        do_case2(5'd0, 5'd2, 5'd4, "T10d-minsat2");
        if (sat2_gold !== 1'b1) begin
            $display("FAIL[T10d] expected golden saturation"); fail=fail+1;
            $fatal(1, "T10d golden not saturating");
        end else pass = pass + 1;

        // T10e  constrained-random 2x2 sweep (both magnitude regimes so the
        //       sat flag is exercised in both polarities), bit-exact + sat.
        for (v = 0; v < NRAND; v = v + 1) begin
            if (v[0]) begin
                fill_rand2(0, -9000, 9000);
                fill_rand2(1, -9000, 9000);
            end else begin
                fill_rand2(0, -300, 300);
                fill_rand2(1, -300, 300);
            end
            do_case2(5'd0, 5'd2, 5'd4, "T10e-rand2");
        end

        // ===============================================================
        // T11  THIRD-SIZE PROOF (N3=3) -- NON-POWER-OF-TWO.  This is the size
        //      for which the original mod-4 bit-truncation arr_k would be
        //      INCORRECT.  Checked bit-exact against an independent 3x3 golden,
        //      plus N3 latency (==8) + busy assertions.
        // ===============================================================
        // T11a  identity: A3 = I => C3 == B3.
        A3[0][0]=16'sd256; A3[0][1]=16'sd0;   A3[0][2]=16'sd0;
        A3[1][0]=16'sd0;   A3[1][1]=16'sd256; A3[1][2]=16'sd0;
        A3[2][0]=16'sd0;   A3[2][1]=16'sd0;   A3[2][2]=16'sd256;
        B3[0][0]=16'sd100; B3[0][1]=-16'sd50; B3[0][2]=16'sd300;
        B3[1][0]=16'sd7;   B3[1][1]=16'sd8;   B3[1][2]=-16'sd9;
        B3[2][0]=-16'sd256;B3[2][1]=16'sd1;   B3[2][2]=16'sd2;
        do_case3(5'd0, 5'd3, 5'd6, "T11a-id3");

        // T11b  negatives / mixed signs (exercises arr_k across the 3x3 mesh).
        A3[0][0]=-16'sd256;A3[0][1]= 16'sd128;A3[0][2]=-16'sd64;
        A3[1][0]= 16'sd64; A3[1][1]=-16'sd32; A3[1][2]= 16'sd16;
        A3[2][0]= 16'sd200;A3[2][1]= 16'sd50; A3[2][2]=-16'sd100;
        B3[0][0]= 16'sd64; B3[0][1]=-16'sd64; B3[0][2]= 16'sd32;
        B3[1][0]=-16'sd128;B3[1][1]= 16'sd64; B3[1][2]=-16'sd32;
        B3[2][0]= 16'sd256;B3[2][1]=-16'sd128;B3[2][2]= 16'sd64;
        do_case3(5'd9, 5'd12, 5'd15, "T11b-neg3");

        // T11c  positive saturation: A3=B3=+max => sat=1.
        fill_rand3(0, 0, 0); fill_rand3(1, 0, 0);   // clear (lo==hi==0)
        for (gi=0; gi<N3; gi=gi+1) for (gj=0; gj<N3; gj=gj+1) begin
            A3[gi][gj]=`Q78_MAX; B3[gi][gj]=`Q78_MAX;
        end
        do_case3(5'd0, 5'd3, 5'd6, "T11c-maxsat3");
        if (sat3_gold !== 1'b1) begin
            $display("FAIL[T11c] expected golden saturation"); fail=fail+1;
            $fatal(1, "T11c golden not saturating");
        end else pass = pass + 1;

        // T11d  constrained-random 3x3 sweep (both magnitude regimes).
        for (v = 0; v < NRAND; v = v + 1) begin
            if (v[0]) begin
                fill_rand3(0, -7000, 7000);
                fill_rand3(1, -7000, 7000);
            end else begin
                fill_rand3(0, -250, 250);
                fill_rand3(1, -250, 250);
            end
            do_case3(5'd0, 5'd3, 5'd6, "T11d-rand3");
        end

        // ===============================================================
        // Coverage gate: the sat flag must have been exercised AND verified in
        // BOTH polarities (some checked cases saturated, some did not).  This
        // guarantees the constrained-random + directed mix is not one-sided.
        // ===============================================================
        if (cov_sat == 0) begin
            $display("FAIL coverage: no saturating case was checked");
            fail = fail + 1;
            $fatal(1, "gemm_systolic_tb sat coverage (sat==1) empty");
        end
        if (cov_nosat == 0) begin
            $display("FAIL coverage: no non-saturating case was checked");
            fail = fail + 1;
            $fatal(1, "gemm_systolic_tb sat coverage (sat==0) empty");
        end
        $display("INFO sat coverage: saturating=%0d non-saturating=%0d",
                 cov_sat, cov_nosat);

        // ===============================================================
        if (fail != 0) begin
            $display("GEMM_SYSTOLIC TB: %0d FAILURES", fail);
            $fatal(1, "gemm_systolic_tb FAILED");
        end
        $display("ALL %0d TESTS PASSED", pass);
        $finish;
    end

    // hard watchdog against any hang.
    initial begin
        #2000000;
        $display("FAIL: global timeout");
        $fatal(1, "gemm_systolic_tb global timeout");
    end
endmodule
