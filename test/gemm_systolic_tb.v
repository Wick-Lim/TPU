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
