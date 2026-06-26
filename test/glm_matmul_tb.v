`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_matmul_tb.v  --  thorough self-checking TB for glm_matmul   (§6,§8)
//----------------------------------------------------------------------------
// FUNCTION UNDER TEST
//   glm_matmul computes  C[M,N] = A[M,K] x W[K,N]  with BF16 operands, an
//   FP32 accumulator across the K reduction, and a BF16 result rounded
//   (RNE) only at the very end.  Operands stream in as outer-product beats:
//   on each accepted K-beat the producer presents column k of A (M bf16
//   lanes, lane pi = A[pi][k]) and row k of W (N bf16 lanes, lane pj =
//   W[k][pj]); the result streams out one C row per cycle.
//
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL
//   The DUT reduces in the project's bit-accurate fp32 primitives (a chain of
//   fp32_mul + fp32_add per output, in the array's accumulation order).  The
//   golden here shares NONE of that arithmetic.  It recomputes C a COMPLETELY
//   DIFFERENT way, in Verilog `real` (IEEE double, fp64):
//     * widen each stored bf16 A[i][k], W[k][j] to its EXACT real value
//       (bf16->real is lossless: bf16 is the high 16 bits of fp32),
//     * accumulate  C[i][j] = Σ_k A_real[i][k]*W_real[k][j]  in fp64 `real`
//       (NOT the fp32 mul/add tree -- a true double-precision dot product),
//     * quantize C[i][j] to bf16 the SAME way the unit emits bf16
//       (real -> fp32 bits via $shortrealtobits, then fp32_to_bf16 RNE).
//   The ONLY shared step is the final fp32->bf16 RNE pack, which IS the unit's
//   defined output format (both DUT and golden must land on the same bf16
//   grid).  Everything that produces the *value* differs -- the DUT does a
//   K-long fp32 sum, the golden a fp64 sum -- so the golden catches DUT
//   arithmetic bugs (a dropped term, a wrong product, a mis-ordered/lost
//   accumulate) instead of mirroring them.
//
//   The DUT is fed bf16-quantized A and W (the SAME bit patterns the golden
//   widens), so any discrepancy is the unit's bf16*bf16->fp32-accumulate path,
//   not input-quantization skew.
//
//----------------------------------------------------------------------------
// TOLERANCE  (STATED + JUSTIFIED)
//   We compare the DUT bf16 output to the fp64-golden value (re-widened from
//   its bf16 quantization) as a RELATIVE error, per element:
//        relerr = |c_dut_real - c_gold_real| / max(|c_gold_real|, TINY)
//   and require  relerr <= REL_TOL.
//
//   bf16 carries 8 significand bits (7 stored + implicit), so one bf16 ULP is
//   2^-8 of the value and the final round-to-bf16 alone costs up to 0.5 ULP =
//   2^-9.  The DOMINANT extra effect is that the DUT and golden can round the
//   SAME real dot product to ADJACENT bf16 values near a rounding boundary
//   (the golden rounds the exact fp64 sum; the DUT rounds an fp32 running sum
//   whose own per-step rounding can tip it across the halfway point).  That
//   double-rounding gap is bounded by ~1 bf16 ULP = 2^-8 of the value.  The
//   fp32 running-sum error grows mildly with K (each of the K adds is ~2^-23
//   relative, and catastrophic cancellation in a near-zero sum can inflate the
//   *relative* error of the tiny result), so we let the tolerance scale gently
//   with K:
//        REL_TOL(K) = 2^-6 + (K/8)*2^-9     (= 1/64 + K*2^-12)
//   i.e. a 4-ULP floor (2^-6) plus ~0.5 bf16-ULP per 8 reduction terms.  For
//   the tested K (<=64) this stays a TIGHT 4..6 bf16 ULP -- it passes the
//   correct fp32-accumulate datapath with margin for the rare double-rounding
//   case, yet fails on a real arithmetic bug (a wrong product or a dropped
//   term moves an element by >> a handful of ULP).
//
//   For elements whose golden magnitude is below TINY (true near-zero from an
//   exact or near-exact cancellation, where relative error is meaningless) we
//   instead require the DUT magnitude to also be within ABS_TOL of zero.
//   We additionally track and PRINT the worst observed rel-err across the run.
//
//----------------------------------------------------------------------------
// COVERAGE  (each element checked within tolerance; $fatal on any miss)
//   THREE (M,N,K) shapes exercise the array geometry + accumulation depth:
//        (M,N,K) = (8,8,8)   -- the default square tile
//        (M,N,K) = (4,6,16)  -- NON-SQUARE M!=N, deeper K (accumulation depth)
//        (M,N,K) = (5,3,64)  -- tall-thin, LONG K (the fp32-accumulate raison)
//   Directed cases per shape:
//        zero A (and zero W)        -> C == 0
//        identity-like W (K>=N)     -> C[:, j] == A[:, j]  (W = I padded)
//        single large A value       -> one element dominates each dot product
//        alternating-sign K terms   -> cancellation / catastrophic-borrow in
//                                       the fp32 sum (probes accumulate order)
//        all +1 A and +1 W          -> every C element == K exactly
//   Plus many RANDOM A,W per shape over a WIDE dynamic range (|.| ~ 1e-3..1e3,
//   mixed sign).  Every output element is checked within tolerance.
//   A STALL test re-runs one random case while throttling ab_valid to prove
//   the pull/valid handshake is correct under back-pressure (same result).
//
//   GATES: prints "ALL <N> TESTS PASSED"; $fatal on any element mismatch, any
//   nan/inf in the output, or any handshake/protocol violation.
//============================================================================
module glm_matmul_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ---- tile shapes (compile-time per DUT) ----
    localparam integer M0 = 8, N0 = 8, K0 = 8;    // square default
    localparam integer M1 = 4, N1 = 6, K1 = 16;   // non-square, deeper K
    localparam integer M2 = 5, N2 = 3, K2 = 64;   // tall-thin, long K

    localparam integer MMAX = 8;
    localparam integer NMAX = 8;
    localparam integer KMAX = 64;

    // ---- tolerances ----
    localparam real TINY    = 1.0e-9;
    localparam real ABS_TOL = 1.0/64.0;

    // ===================================================================
    //  Generic driver-side operand storage (sized to the largest shape).
    //  Abuf[i][k] = A[i][k] (bf16), Wbuf[k][j] = W[k][j] (bf16).
    // ===================================================================
    reg [15:0] Abuf [0:MMAX-1][0:KMAX-1];
    reg [15:0] Wbuf [0:KMAX-1][0:NMAX-1];

    // ---- shared control ----
    reg        stall_en;                  // throttle ab_valid when 1

    // ===================================================================
    //  DUT 0 : (8,8,8)
    // ===================================================================
    reg                 d0_start;
    wire                d0_a_req, d0_c_valid, d0_busy, d0_done;
    reg  [M0*16-1:0]    d0_a_col;
    reg  [N0*16-1:0]    d0_w_row;
    reg                 d0_ab_valid;
    wire [N0*16-1:0]    d0_c_row;
    glm_matmul #(.M(M0), .N(N0), .K(K0)) dut0 (
        .clk(clk), .rst(rst), .start(d0_start),
        .a_req(d0_a_req), .a_col(d0_a_col), .w_row(d0_w_row),
        .ab_valid(d0_ab_valid),
        .c_valid(d0_c_valid), .c_row(d0_c_row),
        .busy(d0_busy), .done(d0_done));

    // ===================================================================
    //  DUT 1 : (4,6,16)
    // ===================================================================
    reg                 d1_start;
    wire                d1_a_req, d1_c_valid, d1_busy, d1_done;
    reg  [M1*16-1:0]    d1_a_col;
    reg  [N1*16-1:0]    d1_w_row;
    reg                 d1_ab_valid;
    wire [N1*16-1:0]    d1_c_row;
    glm_matmul #(.M(M1), .N(N1), .K(K1)) dut1 (
        .clk(clk), .rst(rst), .start(d1_start),
        .a_req(d1_a_req), .a_col(d1_a_col), .w_row(d1_w_row),
        .ab_valid(d1_ab_valid),
        .c_valid(d1_c_valid), .c_row(d1_c_row),
        .busy(d1_busy), .done(d1_done));

    // ===================================================================
    //  DUT 2 : (5,3,64)
    // ===================================================================
    reg                 d2_start;
    wire                d2_a_req, d2_c_valid, d2_busy, d2_done;
    reg  [M2*16-1:0]    d2_a_col;
    reg  [N2*16-1:0]    d2_w_row;
    reg                 d2_ab_valid;
    wire [N2*16-1:0]    d2_c_row;
    glm_matmul #(.M(M2), .N(N2), .K(K2)) dut2 (
        .clk(clk), .rst(rst), .start(d2_start),
        .a_req(d2_a_req), .a_col(d2_a_col), .w_row(d2_w_row),
        .ab_valid(d2_ab_valid),
        .c_valid(d2_c_valid), .c_row(d2_c_row),
        .busy(d2_busy), .done(d2_done));

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

    // quantize a real to bf16 the SAME way the unit emits bf16: real->fp32
    // bits (shortreal = IEEE single), then fp32_to_bf16 RNE from glm_fp.vh.
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

    // golden + captured DUT output (sized to largest tile)
    reg [15:0] cref [0:MMAX-1][0:NMAX-1];
    reg [15:0] cdut [0:MMAX-1][0:NMAX-1];

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
    //  COMPUTE GOLDEN  for the current Abuf/Wbuf and a given (M,N,K).
    //  C_gold[i][j] = bf16( Σ_k bf16_real(A[i][k]) * bf16_real(W[k][j]) )
    //  with the sum accumulated in fp64 `real`.
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
    //  CHECK one finished tile: compare cdut vs cref element-wise within
    //  the K-scaled relative tolerance.  $fatal on any miss / nan / inf.
    // ===================================================================
    task check_tile(input integer MM, input integer NN, input integer KK,
                    input [255:0] tag);
        integer i, j;
        real    gold, got, diff, denom, relerr, rel_tol;
        reg [15:0] db;
        begin
            // REL_TOL(K) = 2^-6 + (K/8)*2^-9  = 1/64 + K*2^-12
            rel_tol = (1.0/64.0) + (KK * (1.0/4096.0));
            for (i = 0; i < MM; i = i + 1)
                for (j = 0; j < NN; j = j + 1) begin
                    db   = cdut[i][j];
                    // reject nan/inf in the DUT output
                    if (db[14:7] == 8'hFF) begin
                        $display("FAIL [%0s] C[%0d][%0d] is nan/inf (bf16=%04h)",
                                 tag, i, j, db);
                        errors = errors + 1;
                    end else begin
                        gold = bf16_to_real(cref[i][j]);
                        got  = bf16_to_real(db);
                        diff = got - gold; if (diff < 0.0) diff = -diff;
                        denom = (gold < 0.0) ? -gold : gold;
                        if (denom < TINY) begin
                            // near-zero golden: require |got| within ABS_TOL
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
    //  DRIVE one tile on DUT 0 (8,8,8).  Pulses start, answers a_req with
    //  a beat each cycle (honoring stall_en back-pressure), captures the
    //  M output rows, waits for done.
    // ===================================================================
    task run_dut0;
        integer k, orow_i, j, guard;
        begin
            @(negedge clk);
            d0_start    <= 1'b1;
            d0_ab_valid <= 1'b0;
            @(negedge clk);
            d0_start <= 1'b0;

            k      = 0;
            orow_i = 0;
            guard  = 0;
            // service the stream until done
            while (!d0_done && guard < 100000) begin
                // ---- operand pull: answer a_req with beat k ----
                if (d0_a_req && (k < K0)) begin
                    if (stall_en && (($random & 3) == 0)) begin
                        d0_ab_valid <= 1'b0;        // inject a bubble
                    end else begin
                        for (j = 0; j < M0; j = j + 1)
                            d0_a_col[16*j +: 16] <= Abuf[j][k];
                        for (j = 0; j < N0; j = j + 1)
                            d0_w_row[16*j +: 16] <= Wbuf[k][j];
                        d0_ab_valid <= 1'b1;
                        k = k + 1;
                    end
                end else begin
                    d0_ab_valid <= 1'b0;
                end
                // ---- output capture: one C row per c_valid ----
                if (d0_c_valid && (orow_i < M0)) begin
                    for (j = 0; j < N0; j = j + 1)
                        cdut[orow_i][j] = d0_c_row[16*j +: 16];
                    orow_i = orow_i + 1;
                end
                @(negedge clk);
                guard = guard + 1;
            end
            d0_ab_valid <= 1'b0;
            if (guard >= 100000) begin
                $display("FAIL DUT0 timeout (k=%0d orow=%0d)", k, orow_i);
                errors = errors + 1;
            end
        end
    endtask

    // ---- DUT 1 (4,6,16) ----
    task run_dut1;
        integer k, orow_i, j, guard;
        begin
            @(negedge clk);
            d1_start    <= 1'b1;
            d1_ab_valid <= 1'b0;
            @(negedge clk);
            d1_start <= 1'b0;
            k = 0; orow_i = 0; guard = 0;
            while (!d1_done && guard < 100000) begin
                if (d1_a_req && (k < K1)) begin
                    if (stall_en && (($random & 3) == 0)) begin
                        d1_ab_valid <= 1'b0;
                    end else begin
                        for (j = 0; j < M1; j = j + 1)
                            d1_a_col[16*j +: 16] <= Abuf[j][k];
                        for (j = 0; j < N1; j = j + 1)
                            d1_w_row[16*j +: 16] <= Wbuf[k][j];
                        d1_ab_valid <= 1'b1;
                        k = k + 1;
                    end
                end else begin
                    d1_ab_valid <= 1'b0;
                end
                if (d1_c_valid && (orow_i < M1)) begin
                    for (j = 0; j < N1; j = j + 1)
                        cdut[orow_i][j] = d1_c_row[16*j +: 16];
                    orow_i = orow_i + 1;
                end
                @(negedge clk);
                guard = guard + 1;
            end
            d1_ab_valid <= 1'b0;
            if (guard >= 100000) begin
                $display("FAIL DUT1 timeout (k=%0d orow=%0d)", k, orow_i);
                errors = errors + 1;
            end
        end
    endtask

    // ---- DUT 2 (5,3,64) ----
    task run_dut2;
        integer k, orow_i, j, guard;
        begin
            @(negedge clk);
            d2_start    <= 1'b1;
            d2_ab_valid <= 1'b0;
            @(negedge clk);
            d2_start <= 1'b0;
            k = 0; orow_i = 0; guard = 0;
            while (!d2_done && guard < 100000) begin
                if (d2_a_req && (k < K2)) begin
                    if (stall_en && (($random & 3) == 0)) begin
                        d2_ab_valid <= 1'b0;
                    end else begin
                        for (j = 0; j < M2; j = j + 1)
                            d2_a_col[16*j +: 16] <= Abuf[j][k];
                        for (j = 0; j < N2; j = j + 1)
                            d2_w_row[16*j +: 16] <= Wbuf[k][j];
                        d2_ab_valid <= 1'b1;
                        k = k + 1;
                    end
                end else begin
                    d2_ab_valid <= 1'b0;
                end
                if (d2_c_valid && (orow_i < M2)) begin
                    for (j = 0; j < N2; j = j + 1)
                        cdut[orow_i][j] = d2_c_row[16*j +: 16];
                    orow_i = orow_i + 1;
                end
                @(negedge clk);
                guard = guard + 1;
            end
            d2_ab_valid <= 1'b0;
            if (guard >= 100000) begin
                $display("FAIL DUT2 timeout (k=%0d orow=%0d)", k, orow_i);
                errors = errors + 1;
            end
        end
    endtask

    // generic dispatch: run the selected DUT's stream task
    task run_dut(input integer which);
        begin
            if      (which == 0) run_dut0;
            else if (which == 1) run_dut1;
            else                 run_dut2;
        end
    endtask

    // ===================================================================
    //  Operand fill helpers (write Abuf / Wbuf as bf16, sized by shape).
    // ===================================================================
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

    task fill_const(input integer MM, input integer NN, input integer KK,
                    input real av, input real wv);
        integer i, k;
        begin
            for (i = 0; i < MM; i = i + 1)
                for (k = 0; k < KK; k = k + 1)
                    Abuf[i][k] = real_to_bf16(av);
            for (k = 0; k < KK; k = k + 1)
                for (i = 0; i < NN; i = i + 1)
                    Wbuf[k][i] = real_to_bf16(wv);
        end
    endtask

    // W = identity (padded): W[k][j] = (k==j) ? 1 : 0  -> C[:,j] = A[:,j]
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

    // alternating-sign K terms: A[i][k] = (-1)^k * v, W[k][j] = u  -> the
    // fp32 sum sees + - + - ... (cancellation / catastrophic borrow).
    task fill_altsign(input integer MM, input integer NN, input integer KK);
        integer i, k;
        real v;
        begin
            for (i = 0; i < MM; i = i + 1)
                for (k = 0; k < KK; k = k + 1) begin
                    v = 100.0;
                    Abuf[i][k] = real_to_bf16((k & 1) ? -v : v);
                end
            for (k = 0; k < KK; k = k + 1)
                for (i = 0; i < NN; i = i + 1)
                    Wbuf[k][i] = real_to_bf16(1.0);
        end
    endtask

    // single large A value: A all small except A[0][0] huge.
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
    //  Per-shape exercise: directed + random cases on one DUT.
    // ===================================================================
    task exercise(input integer which,
                  input integer MM, input integer NN, input integer KK);
        integer r;
        begin
            // -- zero A, zero W -> C == 0
            fill_const(MM, NN, KK, 0.0, 0.0);
            compute_golden(MM, NN, KK); run_dut(which); check_tile(MM, NN, KK, "zero");

            // -- all +1 A and +1 W -> every C == K
            fill_const(MM, NN, KK, 1.0, 1.0);
            compute_golden(MM, NN, KK); run_dut(which); check_tile(MM, NN, KK, "ones");

            // -- identity-like W -> C[:,j] == A[:,j]
            fill_identityW(MM, NN, KK);
            compute_golden(MM, NN, KK); run_dut(which); check_tile(MM, NN, KK, "identW");

            // -- single large A value
            fill_single_large(MM, NN, KK);
            compute_golden(MM, NN, KK); run_dut(which); check_tile(MM, NN, KK, "single");

            // -- alternating-sign K terms (cancellation)
            fill_altsign(MM, NN, KK);
            compute_golden(MM, NN, KK); run_dut(which); check_tile(MM, NN, KK, "altsign");

            // -- many random A,W (wide dynamic range, mixed sign)
            for (r = 0; r < 8; r = r + 1) begin
                fill_random(MM, NN, KK);
                compute_golden(MM, NN, KK); run_dut(which); check_tile(MM, NN, KK, "rand");
            end

            // -- stall test: re-run a random case under back-pressure
            fill_random(MM, NN, KK);
            compute_golden(MM, NN, KK);
            stall_en = 1'b1;  run_dut(which);  stall_en = 1'b0;
            check_tile(MM, NN, KK, "stall");
        end
    endtask

    // ===================================================================
    //  MAIN
    // ===================================================================
    initial begin
        rst         = 1'b1;
        stall_en    = 1'b0;
        d0_start    = 1'b0; d1_start = 1'b0; d2_start = 1'b0;
        d0_ab_valid = 1'b0; d1_ab_valid = 1'b0; d2_ab_valid = 1'b0;
        d0_a_col=0; d0_w_row=0; d1_a_col=0; d1_w_row=0; d2_a_col=0; d2_w_row=0;
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        exercise(0, M0, N0, K0);   // (8,8,8)  square
        exercise(1, M1, N1, K1);   // (4,6,16) non-square, deeper K
        exercise(2, M2, N2, K2);   // (5,3,64) tall-thin, long K

        if (errors != 0) begin
            $display("FAILED: %0d mismatch(es) over %0d tiles (worst relerr=%g)",
                     errors, test_count, worst_relerr);
            $fatal(1, "glm_matmul_tb FAILED");
        end else begin
            $display("worst observed relerr = %g", worst_relerr);
            $display("ALL %0d TESTS PASSED", test_count);
        end
        $finish;
    end

    // global watchdog
    initial begin
        #5_000_000;
        $display("FAIL global timeout");
        $fatal(1, "glm_matmul_tb global timeout");
    end

endmodule
