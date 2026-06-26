`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_matmul_pipe_tb.v  --  thorough self-checking TB for glm_matmul_pipe
//                           (the ULTRA-HIGH-FMAX pipelined systolic GEMM)
//----------------------------------------------------------------------------
// FUNCTION UNDER TEST
//   glm_matmul_pipe computes  C[M,N] = A[M,K] x W[K,N]  with BF16 operands,
//   an FP32 accumulation across the K reduction, and a single BF16 result
//   rounded (RNE) only at the very end.  The tile is the whole PE_M x PE_N
//   output, resident in the array (output-stationary).  Operands stream in as
//   outer-product K-beats: on each accepted beat (in_valid) the producer
//   presents column k of A (PE_M bf16 lanes, lane pi = A[pi][k], packed in
//   a_col) and row k of W (PE_N bf16 lanes, lane pj = W[k][pj], in w_row).
//   After all k_len beats drain, out_valid pulses for one cycle with the full
//   PE_M x PE_N bf16 C tile row-major-packed in c_out.
//
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL  (shares NONE of the DUT's fp32 arithmetic)
//   The DUT reduces in the project's bit-accurate pipelined fp32 primitives
//   (fp32_mac_pipe + an fp32_add_pipe tree, in an L=5-way interleaved order).
//   The golden recomputes C a COMPLETELY DIFFERENT way, in Verilog `real`
//   (IEEE double, fp64):
//     * widen each stored bf16 A[i][k], W[k][j] to its EXACT real value
//       (bf16->real is lossless: bf16 is the high 16 bits of fp32),
//     * accumulate  C[i][j] = sum_k A_real[i][k]*W_real[k][j]  in fp64 `real`
//       -- a true double-precision dot product, NOT the fp32 mac/add tree and
//       NOT the DUT's L-way grouping,
//     * quantize C[i][j] to bf16 the SAME way the unit emits bf16: real->fp32
//       bits ($shortrealtobits) then fp32_to_bf16 RNE (the unit's defined
//       output format; both DUT and golden must land on the same bf16 grid).
//   Everything that produces the *value* differs (DUT: K-long fp32 sum in a
//   5-way-interleaved + binary-tree order; golden: a single fp64 sum), so the
//   golden catches DUT arithmetic bugs -- a dropped term, a wrong product, a
//   stale/mis-ordered accumulate, a lane collision -- instead of mirroring
//   them.  The DUT is fed the SAME bf16 bit patterns the golden widens, so any
//   discrepancy is the unit's bf16*bf16->fp32-accumulate path, not input skew.
//
//----------------------------------------------------------------------------
// TOLERANCE  (STATED + JUSTIFIED, K-dependent)
//   Per element we compare the DUT bf16 output to the fp64-golden value
//   (re-widened from its bf16 quantization) as a RELATIVE error:
//        relerr = |c_dut_real - c_gold_real| / max(|c_gold_real|, TINY)
//   and require  relerr <= REL_TOL(K).
//
//   bf16 carries 8 significand bits (7 stored + implicit), so 1 bf16 ULP is
//   2^-8 of the value; the final round-to-bf16 alone costs up to 0.5 ULP.
//   The dominant extra effect: the DUT and golden can round the SAME real dot
//   product to ADJACENT bf16 values near a rounding boundary -- the golden
//   rounds the exact fp64 sum, the DUT rounds an fp32 running sum whose own
//   per-step rounding (and the L-way grouping / tree-order non-associativity)
//   can tip it across the halfway point.  That double-rounding gap is ~1 bf16
//   ULP = 2^-8.  The fp32 running-sum error grows mildly with K (each of the
//   ~K adds is ~2^-23 relative, and catastrophic cancellation in a near-zero
//   sum can inflate the *relative* error of the tiny result), so we let the
//   tolerance scale gently with K:
//        REL_TOL(K) = 2^-6 + (K/8)*2^-9   ( = 1/64 + K*2^-12 )
//   a 4-ULP floor (2^-6) plus ~0.5 bf16-ULP per 8 reduction terms.  For the
//   tested K (up to 200) this is a tight 4..~50 bf16-ULP band: it passes the
//   correct fp32-accumulate datapath with margin for the double-rounding case,
//   yet fails on a real arithmetic bug (a wrong product / dropped term moves an
//   element by >> a handful of ULP, typically a whole missing reduction term).
//
//   For elements whose golden magnitude is below TINY (true near-zero from an
//   exact/near-exact cancellation, where relative error is meaningless) we
//   instead require the DUT magnitude to be within ABS_TOL of zero.
//   We track and PRINT the worst observed rel-err across the run.
//
//----------------------------------------------------------------------------
// COVERAGE  (every output element checked; $fatal on any miss / nan / inf)
//   Four (M,N) shapes (separate DUT instances; PE_M/PE_N are compile-time)
//   exercise the array geometry, incl. NON-SQUARE:
//        (M,N) = (4,4)   square default
//        (M,N) = (2,5)   non-square, wide
//        (M,N) = (6,3)   non-square, tall
//        (M,N) = (1,1)   degenerate single-PE
//   K is a RUNTIME input; per shape we sweep several K including a K that
//   exercises the accumulation-pipeline DEPTH (K >> MAC latency L=5, so the
//   L-way interleave / lane-wrap / forwarding is stressed):
//        K in {1, 4, 5, 6, 7, 16, 64, 200}
//          (K=1: below L; K=5: exactly L lanes; K=6,7: lane-wrap; K=200: deep)
//   Directed operand patterns per (shape,K):
//        zero A,W                   -> C == 0
//        all +1 A and +1 W          -> every C element == K exactly
//        identity-like W (K>=N)     -> C[:,j] == A[:,j]
//        single large A value       -> one element dominates each dot product
//        alternating-sign K terms   -> cancellation / catastrophic borrow in
//                                       the fp32 sum (probes accumulate order)
//        random wide-dynamic-range  -> |.| ~ 1e-3..1e3, mixed sign (x8)
//   THROUGHPUT: one random case PER K is timed.  out_valid must land at a
//   FIXED, deterministic latency that is AFFINE in K with slope exactly 1 --
//   i.e. latency(K) = K + LAT_CONST -- proving 1 K-beat/cycle streaming (no
//   per-beat stall): adding one reduction term costs exactly one cycle.  The
//   constant matches the DUT's documented drain K + L + TREE_LAT + 1 plus the
//   TB's fixed measurement offset (start pulse + issue/writeback edge); we pin
//   the whole affine relation, so a stall or dropped pipeline beat fails it.
//   BACK-PRESSURE / BUBBLES: one random case is re-run while randomly
//   DE-asserting in_valid (injecting bubbles between beats); the result must be
//   bit-identical to the no-bubble run -- proving bubbles don't corrupt the
//   L-way accumulation (the lane shift-pipe must track issued lanes, not raw
//   cycles).
//
//   GATES: prints "ALL <N> TESTS PASSED"; $fatal on any element mismatch, any
//   nan/inf in the output, or any latency/protocol violation.
//============================================================================
module glm_matmul_pipe_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ---- shapes (compile-time per DUT) ----
    localparam integer M0 = 4, N0 = 4;   // square
    localparam integer M1 = 2, N1 = 5;   // non-square wide
    localparam integer M2 = 6, N2 = 3;   // non-square tall
    localparam integer M3 = 1, N3 = 1;   // single PE

    localparam integer MMAX = 6;
    localparam integer NMAX = 5;
    localparam integer KMAX = 256;       // DUT counter width

    localparam integer L        = 5;     // fp32_mac_pipe latency (DUT crux)
    localparam integer TREE_LAT = 9;     // 3 levels * fp32_add_pipe LAT(3)

    // ---- tolerances ----
    localparam real TINY    = 1.0e-9;
    localparam real ABS_TOL = 1.0/64.0;

    // ===================================================================
    //  Operand storage (sized to the largest shape / longest K).
    //  Abuf[i][k] = A[i][k] (bf16), Wbuf[k][j] = W[k][j] (bf16).
    // ===================================================================
    reg [15:0] Abuf [0:MMAX-1][0:KMAX-1];
    reg [15:0] Wbuf [0:KMAX-1][0:NMAX-1];

    // golden + captured DUT output
    reg [15:0] cref [0:MMAX-1][0:NMAX-1];
    reg [15:0] cdut [0:MMAX-1][0:NMAX-1];

    // ===================================================================
    //  DUT instances (one per shape).  Wide packed operand ports.
    // ===================================================================
    // common driven beat regs (max-width; sliced per DUT)
    reg                  start;
    reg [$clog2(KMAX+1)-1:0] k_len;
    reg                  in_valid;
    reg [16*MMAX-1:0]    a_col_max;
    reg [16*NMAX-1:0]    w_row_max;

    // -- DUT0 (4,4) --
    wire d0_busy, d0_ov; wire [16*M0*N0-1:0] d0_c;
    glm_matmul_pipe #(.PE_M(M0), .PE_N(N0), .KMAX(KMAX)) dut0 (
        .clk(clk), .rst(rst), .start(start), .k_len(k_len),
        .in_valid(in_valid), .a_col(a_col_max[16*M0-1:0]),
        .w_row(w_row_max[16*N0-1:0]),
        .busy(d0_busy), .out_valid(d0_ov), .c_out(d0_c));

    // -- DUT1 (2,5) --
    wire d1_busy, d1_ov; wire [16*M1*N1-1:0] d1_c;
    glm_matmul_pipe #(.PE_M(M1), .PE_N(N1), .KMAX(KMAX)) dut1 (
        .clk(clk), .rst(rst), .start(start), .k_len(k_len),
        .in_valid(in_valid), .a_col(a_col_max[16*M1-1:0]),
        .w_row(w_row_max[16*N1-1:0]),
        .busy(d1_busy), .out_valid(d1_ov), .c_out(d1_c));

    // -- DUT2 (6,3) --
    wire d2_busy, d2_ov; wire [16*M2*N2-1:0] d2_c;
    glm_matmul_pipe #(.PE_M(M2), .PE_N(N2), .KMAX(KMAX)) dut2 (
        .clk(clk), .rst(rst), .start(start), .k_len(k_len),
        .in_valid(in_valid), .a_col(a_col_max[16*M2-1:0]),
        .w_row(w_row_max[16*N2-1:0]),
        .busy(d2_busy), .out_valid(d2_ov), .c_out(d2_c));

    // -- DUT3 (1,1) --
    wire d3_busy, d3_ov; wire [16*M3*N3-1:0] d3_c;
    glm_matmul_pipe #(.PE_M(M3), .PE_N(N3), .KMAX(KMAX)) dut3 (
        .clk(clk), .rst(rst), .start(start), .k_len(k_len),
        .in_valid(in_valid), .a_col(a_col_max[16*M3-1:0]),
        .w_row(w_row_max[16*N3-1:0]),
        .busy(d3_busy), .out_valid(d3_ov), .c_out(d3_c));

    // ===================================================================
    //  bf16 <-> real helpers (independent of the DUT's fp32 path)
    // ===================================================================
    function automatic real bf16_to_real(input [15:0] b);
        reg [31:0] f;
        begin
            f = {b, 16'h0000};                 // exact bf16->fp32 (lossless)
            bf16_to_real = $bitstoshortreal(f);
        end
    endfunction

    function automatic [15:0] real_to_bf16(input real r);
        reg [31:0] f;
        begin
            f = $shortrealtobits(r);
            real_to_bf16 = fp32_to_bf16(f);
        end
    endfunction

    // ===================================================================
    //  Test bookkeeping
    // ===================================================================
    integer test_count = 0;
    integer errors     = 0;
    real    worst_relerr = 0.0;

    // ---- random real in [lo,hi) with optional random sign ----
    function automatic real rand_real(input real lo, input real hi,
                                      input integer signd);
        real u; integer r;
        begin
            r = $random;
            u = (r & 32'h7fffffff) / 2147483647.0;   // [0,1]
            rand_real = lo + u*(hi-lo);
            if (signd && ($random & 1)) rand_real = -rand_real;
        end
    endfunction

    // ===================================================================
    //  COMPUTE GOLDEN for current Abuf/Wbuf and a given (M,N,K) in fp64.
    // ===================================================================
    task compute_golden(input integer MM, input integer NN, input integer KK);
        integer i, j, k;
        real    acc;
        begin
            for (i = 0; i < MM; i = i + 1)
                for (j = 0; j < NN; j = j + 1) begin
                    acc = 0.0;
                    for (k = 0; k < KK; k = k + 1)
                        acc = acc + bf16_to_real(Abuf[i][k])
                                  * bf16_to_real(Wbuf[k][j]);
                    cref[i][j] = real_to_bf16(acc);
                end
        end
    endtask

    // ===================================================================
    //  CHECK one finished tile vs golden, K-scaled relative tolerance.
    // ===================================================================
    task check_tile(input integer MM, input integer NN, input integer KK,
                    input [255:0] tag);
        integer i, j;
        real    gold, got, diff, denom, relerr, rel_tol;
        reg [15:0] db;
        begin
            rel_tol = (1.0/64.0) + (KK * (1.0/4096.0));  // 2^-6 + K*2^-12
            for (i = 0; i < MM; i = i + 1)
                for (j = 0; j < NN; j = j + 1) begin
                    db = cdut[i][j];
                    if (db[14:7] == 8'hFF) begin
                        $display("FAIL [%0s] C[%0d][%0d] is nan/inf (bf16=%04h)",
                                 tag, i, j, db);
                        errors = errors + 1;
                    end else begin
                        gold  = bf16_to_real(cref[i][j]);
                        got   = bf16_to_real(db);
                        diff  = got - gold; if (diff < 0.0) diff = -diff;
                        denom = (gold < 0.0) ? -gold : gold;
                        if (denom < TINY) begin
                            if (diff > ABS_TOL) begin
                                $display("FAIL [%0s] C[%0d][%0d] near0 gold=%g got=%g |d|=%g > %g",
                                         tag, i, j, gold, got, diff, ABS_TOL);
                                errors = errors + 1;
                            end
                        end else begin
                            relerr = diff/denom;
                            if (relerr > worst_relerr) worst_relerr = relerr;
                            if (relerr > rel_tol) begin
                                $display("FAIL [%0s] C[%0d][%0d] gold=%g got=%g relerr=%g > tol=%g (K=%0d)",
                                         tag, i, j, gold, got, relerr, rel_tol, KK);
                                errors = errors + 1;
                            end
                        end
                    end
                end
            test_count = test_count + 1;
        end
    endtask

    // ===================================================================
    //  capture helper: unpack a DUT c_out vector into cdut[][]
    //  (c_out is row-major: element (i,j) at bit 16*(i*NN+j))
    // ===================================================================
    task capture(input integer which, input integer MM, input integer NN);
        integer i, j, idx;
        begin
            for (i = 0; i < MM; i = i + 1)
                for (j = 0; j < NN; j = j + 1) begin
                    idx = i*NN + j;
                    case (which)
                      0: cdut[i][j] = d0_c[16*idx +: 16];
                      1: cdut[i][j] = d1_c[16*idx +: 16];
                      2: cdut[i][j] = d2_c[16*idx +: 16];
                      default: cdut[i][j] = d3_c[16*idx +: 16];
                    endcase
                end
        end
    endtask

    // pick the right busy/out_valid for a DUT index
    function automatic ov_of(input integer which);
        ov_of = (which==0) ? d0_ov : (which==1) ? d1_ov :
                (which==2) ? d2_ov : d3_ov;
    endfunction

    // ===================================================================
    //  DRIVE one tile.  bubble_en: randomly de-assert in_valid between beats.
    //  check_lat: if set, assert out_valid arrives exactly at the documented
    //  latency K + L + TREE_LAT + 1 measured from the first accepted beat.
    // ===================================================================
    task run_tile(input integer which, input integer MM, input integer NN,
                  input integer KK, input integer bubble_en,
                  input integer check_lat);
        integer k, j, guard;
        integer first_beat_t, beat_cnt, exp_lat, got_lat;
        reg     saw_first;
        begin
            @(negedge clk);
            start    <= 1'b1;
            k_len    <= KK[$clog2(KMAX+1)-1:0];
            in_valid <= 1'b0;
            @(negedge clk);
            start <= 1'b0;

            k = 0; guard = 0; beat_cnt = 0; saw_first = 1'b0;
            first_beat_t = 0; got_lat = -1;
            // Deterministic, affine-in-K with slope 1: the DUT's documented
            // drain (K + L + TREE_LAT + 1) measured from this TB's reference
            // (first-beat-presented negedge) lands +2 cycles later (the start
            // pulse cycle + the issue/writeback edge).  Pinning this whole
            // affine relation proves 1 K-beat/cycle (no per-beat stall).
            exp_lat = KK + L + TREE_LAT + 3;

            // stream KK beats, optionally with bubbles, watch for out_valid
            while (got_lat < 0 && guard < 100000) begin
                if (k < KK) begin
                    if (bubble_en && (($random & 3) == 0)) begin
                        in_valid <= 1'b0;            // inject a bubble
                    end else begin
                        for (j = 0; j < MM; j = j + 1)
                            a_col_max[16*j +: 16] <= Abuf[j][k];
                        for (j = 0; j < NN; j = j + 1)
                            w_row_max[16*j +: 16] <= Wbuf[k][j];
                        in_valid <= 1'b1;
                        if (!saw_first) begin
                            saw_first = 1'b1;
                            // beat k=0 is accepted on the NEXT posedge; record
                            // the cycle index of that acceptance
                            first_beat_t = beat_cnt + 1;
                        end
                        k = k + 1;
                    end
                end else begin
                    in_valid <= 1'b0;
                end

                @(posedge clk);                      // sample outputs here
                if (saw_first) beat_cnt = beat_cnt + 1;
                if (ov_of(which)) begin
                    capture(which, MM, NN);
                    got_lat = beat_cnt - first_beat_t; // cycles after 1st beat
                end
                @(negedge clk);
                guard = guard + 1;
            end
            in_valid <= 1'b0;

            if (guard >= 100000) begin
                $display("FAIL DUT%0d timeout (which=%0d K=%0d)", which, which, KK);
                errors = errors + 1;
            end else if (check_lat && !bubble_en) begin
                // no-bubble run: latency must be exactly K + L + TREE_LAT + 1
                if (got_lat != exp_lat) begin
                    $display("FAIL latency which=%0d K=%0d: got %0d expected %0d",
                             which, KK, got_lat, exp_lat);
                    errors = errors + 1;
                end else begin
                    $display("  throughput OK which=%0d K=%0d latency=%0d (affine in K, slope 1)",
                             which, KK, got_lat);
                end
            end
        end
    endtask

    // ===================================================================
    //  Operand fill helpers
    // ===================================================================
    task fill_const(input integer MM, input integer NN, input integer KK,
                    input real av, input real wv);
        integer i, k;
        begin
            for (i = 0; i < MM; i = i + 1)
                for (k = 0; k < KK; k = k + 1) Abuf[i][k] = real_to_bf16(av);
            for (k = 0; k < KK; k = k + 1)
                for (i = 0; i < NN; i = i + 1) Wbuf[k][i] = real_to_bf16(wv);
        end
    endtask

    task fill_random(input integer MM, input integer NN, input integer KK);
        integer i, k;
        begin
            for (i = 0; i < MM; i = i + 1)
                for (k = 0; k < KK; k = k + 1)
                    Abuf[i][k] = real_to_bf16(rand_real(1.0e-3, 1.0e3, 1));
            for (k = 0; k < KK; k = k + 1)
                for (i = 0; i < NN; i = i + 1)
                    Wbuf[k][i] = real_to_bf16(rand_real(1.0e-3, 1.0e3, 1));
        end
    endtask

    // W = identity (padded): W[k][j] = (k==j)?1:0 -> C[:,j] = A[:,j]
    task fill_identityW(input integer MM, input integer NN, input integer KK);
        integer i, k;
        begin
            for (i = 0; i < MM; i = i + 1)
                for (k = 0; k < KK; k = k + 1)
                    Abuf[i][k] = real_to_bf16(rand_real(1.0e-2, 1.0e2, 1));
            for (k = 0; k < KK; k = k + 1)
                for (i = 0; i < NN; i = i + 1)
                    Wbuf[k][i] = (k == i) ? real_to_bf16(1.0)
                                          : real_to_bf16(0.0);
        end
    endtask

    // alternating-sign K terms (cancellation)
    task fill_altsign(input integer MM, input integer NN, input integer KK);
        integer i, k; real v;
        begin
            for (i = 0; i < MM; i = i + 1)
                for (k = 0; k < KK; k = k + 1) begin
                    v = 100.0;
                    Abuf[i][k] = real_to_bf16((k & 1) ? -v : v);
                end
            for (k = 0; k < KK; k = k + 1)
                for (i = 0; i < NN; i = i + 1) Wbuf[k][i] = real_to_bf16(1.0);
        end
    endtask

    // single large A value
    task fill_single_large(input integer MM, input integer NN, input integer KK);
        integer i, k;
        begin
            for (i = 0; i < MM; i = i + 1)
                for (k = 0; k < KK; k = k + 1)
                    Abuf[i][k] = real_to_bf16(1.0e-2);
            Abuf[0][0] = real_to_bf16(500.0);
            for (k = 0; k < KK; k = k + 1)
                for (i = 0; i < NN; i = i + 1)
                    Wbuf[k][i] = real_to_bf16(rand_real(0.1, 2.0, 1));
        end
    endtask

    // ===================================================================
    //  Per-(shape,K) exercise: directed + random + throughput + bubbles.
    // ===================================================================
    task exercise_k(input integer which,
                    input integer MM, input integer NN, input integer KK);
        integer r;
        reg [15:0] cnobub [0:MMAX-1][0:NMAX-1];
        integer i, j;
        begin
            // zero
            fill_const(MM, NN, KK, 0.0, 0.0);
            compute_golden(MM, NN, KK);
            run_tile(which, MM, NN, KK, 0, 0); check_tile(MM, NN, KK, "zero");

            // all ones -> C == K
            fill_const(MM, NN, KK, 1.0, 1.0);
            compute_golden(MM, NN, KK);
            run_tile(which, MM, NN, KK, 0, 0); check_tile(MM, NN, KK, "ones");

            // identity W (only meaningful when K>=N; harmless otherwise)
            fill_identityW(MM, NN, KK);
            compute_golden(MM, NN, KK);
            run_tile(which, MM, NN, KK, 0, 0); check_tile(MM, NN, KK, "identW");

            // single large
            fill_single_large(MM, NN, KK);
            compute_golden(MM, NN, KK);
            run_tile(which, MM, NN, KK, 0, 0); check_tile(MM, NN, KK, "single");

            // alternating sign (cancellation)
            fill_altsign(MM, NN, KK);
            compute_golden(MM, NN, KK);
            run_tile(which, MM, NN, KK, 0, 0); check_tile(MM, NN, KK, "altsign");

            // random (x8); time the FIRST one for throughput/latency
            for (r = 0; r < 8; r = r + 1) begin
                fill_random(MM, NN, KK);
                compute_golden(MM, NN, KK);
                run_tile(which, MM, NN, KK, 0, (r == 0) ? 1 : 0);
                check_tile(MM, NN, KK, "rand");
            end

            // bubble / back-pressure: run last random WITHOUT bubbles, capture,
            // then WITH bubbles, and require bit-identical output (and both pass
            // golden).  Proves bubbles don't corrupt the L-way accumulation.
            fill_random(MM, NN, KK);
            compute_golden(MM, NN, KK);
            run_tile(which, MM, NN, KK, 0, 0);
            for (i = 0; i < MM; i = i + 1)
                for (j = 0; j < NN; j = j + 1) cnobub[i][j] = cdut[i][j];
            check_tile(MM, NN, KK, "nobub");
            run_tile(which, MM, NN, KK, 1, 0);    // same data, with bubbles
            for (i = 0; i < MM; i = i + 1)
                for (j = 0; j < NN; j = j + 1)
                    if (cdut[i][j] !== cnobub[i][j]) begin
                        $display("FAIL bubble-corrupt which=%0d K=%0d C[%0d][%0d] nobub=%04h bub=%04h",
                                 which, KK, i, j, cnobub[i][j], cdut[i][j]);
                        errors = errors + 1;
                    end
            check_tile(MM, NN, KK, "bubble");
        end
    endtask

    // sweep K set for one shape
    task exercise_shape(input integer which,
                        input integer MM, input integer NN);
        begin
            exercise_k(which, MM, NN, 1);
            exercise_k(which, MM, NN, 4);
            exercise_k(which, MM, NN, 5);    // == L lanes
            exercise_k(which, MM, NN, 6);    // lane wrap
            exercise_k(which, MM, NN, 7);    // lane wrap
            exercise_k(which, MM, NN, 16);
            exercise_k(which, MM, NN, 64);
            exercise_k(which, MM, NN, 200);  // K >> L : accumulation DEPTH
        end
    endtask

    // ===================================================================
    //  MAIN
    // ===================================================================
    initial begin
        rst = 1'b1;
        start = 1'b0; in_valid = 1'b0; k_len = 0;
        a_col_max = 0; w_row_max = 0;
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        exercise_shape(0, M0, N0);   // (4,4)
        exercise_shape(1, M1, N1);   // (2,5)
        exercise_shape(2, M2, N2);   // (6,3)
        exercise_shape(3, M3, N3);   // (1,1)

        if (errors != 0) begin
            $display("FAILED: %0d mismatch(es) over %0d tiles (worst relerr=%g)",
                     errors, test_count, worst_relerr);
            $fatal(1, "glm_matmul_pipe_tb FAILED");
        end else begin
            $display("worst observed relerr = %g", worst_relerr);
            $display("ALL %0d TESTS PASSED", test_count);
        end
        $finish;
    end

    // global watchdog
    initial begin
        #20_000_000;
        $display("FAIL global timeout");
        $fatal(1, "glm_matmul_pipe_tb global timeout");
    end

endmodule
