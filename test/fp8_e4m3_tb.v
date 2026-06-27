`timescale 1ns/1ps
//============================================================================
// fp8_e4m3_tb.v  --  EXHAUSTIVE, INDEPENDENT fp64-golden TB for fp8_e4m3.vh
//                                                            (ACCEL_GLM52 §6)
//----------------------------------------------------------------------------
// WHAT THIS PROVES
//   src/fp8_e4m3.vh exposes three pure combinational primitives:
//        fp8e4m3_to_fp32(x)   decode E4M3 -> fp32
//        fp32_to_fp8e4m3(f)   encode fp32 -> E4M3 (RNE + saturation)
//        fp8_mul(a,b)         E4M3*E4M3 -> EXACT fp32 (4x4 mantissa multiply)
//
//   FP8 E4M3 has only 256 codes, so the whole space is checked EXHAUSTIVELY:
//
//   (1) DECODE  -- for ALL 256 codes x:
//         * X-aware: the decode output carries no X bits.
//         * NaN code (S.1111.111) decodes to an fp32 NaN.
//         * every other code decodes to EXACTLY the real E4M3 value, computed
//           by an INDEPENDENT fp64 (Verilog `real`) golden (e4m3_real) built
//           straight from the E4M3 field definition.  The DUT's fp32 result is
//           widened to fp64 (f32_to_real) and compared for EXACT equality
//           (every E4M3 value is exactly representable in fp32, hence in fp64).
//
//   (2) ENCODE / ROUND-TRIP  -- for ALL 256 codes x:
//         fp32_to_fp8e4m3(fp8e4m3_to_fp32(x)) == x   (bit-exact, incl. signed
//         zero and the NaN code).  Plus directed RNE-tie and saturation cases
//         driven with hand-built fp32 bit patterns (ties-to-even up & down,
//         subnormal ties, round-up-into-smallest-normal, 448/464/480/Inf/NaN
//         saturation), X-aware.
//
//   (3) MULTIPLY  -- for ALL 256x256 = 65536 ordered pairs (a,b):
//         * X-aware: the product carries no X bits.
//         * any NaN operand -> the product is an fp32 NaN (propagation).
//         * otherwise fp8_mul(a,b) decoded to fp64 EXACTLY equals the fp64
//           product e4m3_real(a)*e4m3_real(b)  (the fp8 product is exact in
//           fp32).  The golden multiply is fp64 and lives OUTSIDE the DUT.
//
//   On ANY mismatch the offending case is printed and the TB $fatal-s.
//   On success it prints "ALL <N> TESTS PASSED".
//============================================================================
module fp8_e4m3_tb;

    // the combinational contract under test (golden = independent fp64 below)
    `include "fp8_e4m3.vh"

    integer test_count = 0;
    integer errors     = 0;

    // ------------------------------------------------------------------------
    // INDEPENDENT fp64 GOLDEN MACHINERY  (no DUT code reuse)
    // ------------------------------------------------------------------------
    // pow2(p) = 2.0^p, exact for the small integer exponents used here.
    function real pow2(input integer p);
        integer i; real r;
        begin
            r = 1.0;
            if (p >= 0) for (i = 0; i < p;  i = i + 1) r = r * 2.0;
            else        for (i = 0; i < -p; i = i + 1) r = r / 2.0;
            pow2 = r;
        end
    endfunction

    // is this E4M3 code the single NaN pattern S.1111.111 ?
    function is_e4m3_nan(input [7:0] x);
        is_e4m3_nan = (x[6:3] == 4'hF) && (x[2:0] == 3'h7);
    endfunction

    // GOLDEN real value of a (non-NaN) E4M3 code, from the field definition.
    //   subnormal/zero (e==0): v = m * 2^-9
    //   normal:                v = (8 + m) * 2^(e-10)
    function real e4m3_real(input [7:0] x);
        reg s; reg [3:0] e; reg [2:0] m; real v;
        begin
            s = x[7]; e = x[6:3]; m = x[2:0];
            if (e == 4'h0) v = m * pow2(-9);
            else           v = (8 + m) * pow2($signed({1'b0, e}) - 10);
            if (s) v = -v;
            e4m3_real = v;
        end
    endfunction

    // widen an fp32 bit pattern to fp64 (DUT decode/product is always a normal
    // fp32 or a signed zero in the value-compare paths; inf/nan never compared
    // here -- those go through the NaN branch).
    function real f32_to_real(input [31:0] b);
        reg [7:0] ex; reg [23:0] sig; real v;
        begin
            ex = b[30:23];
            if (ex == 8'hFF || ex == 8'h00) begin
                v = 0.0;                                   // not used in value paths
            end else begin
                sig = {1'b1, b[22:0]};
                v = sig * pow2($signed({1'b0, ex}) - 127 - 23);
            end
            if (b[31]) v = -v;
            f32_to_real = v;
        end
    endfunction

    function is_fp32_nan(input [31:0] b);
        is_fp32_nan = (b[30:23] == 8'hFF) && (b[22:0] != 23'b0);
    endfunction

    // ------------------------------------------------------------------------
    // working variables
    // ------------------------------------------------------------------------
    integer        ia, ib;
    reg  [7:0]     a8, b8, rt;
    reg  [31:0]    dec, prod;
    real           gold, got;

    // directed ENCODE check (hand-built fp32 -> expected E4M3 code)
    task chk_enc(input [31:0] f, input [7:0] exp_code, input [255:0] name);
        reg [7:0] got_code;
        begin
            got_code = fp32_to_fp8e4m3(f);
            if (^got_code === 1'bx) begin
                $display("X in encode(%s): f=%h -> %b", name, f, got_code);
                errors = errors + 1;
            end else if (got_code !== exp_code) begin
                $display("ENCODE MISMATCH %s: f=%h got=%b exp=%b",
                         name, f, got_code, exp_code);
                errors = errors + 1;
            end
            test_count = test_count + 1;
        end
    endtask

    initial begin
        // ===================== (1) DECODE, all 256 =====================
        for (ia = 0; ia < 256; ia = ia + 1) begin
            a8  = ia[7:0];
            dec = fp8e4m3_to_fp32(a8);
            if (^dec === 1'bx) begin
                $display("X in decode(%b) -> %h", a8, dec);
                errors = errors + 1;
            end else if (is_e4m3_nan(a8)) begin
                if (!is_fp32_nan(dec)) begin
                    $display("DECODE NaN MISMATCH: code=%b -> %h (not fp32 NaN)", a8, dec);
                    errors = errors + 1;
                end
            end else begin
                gold = e4m3_real(a8);
                got  = f32_to_real(dec);
                if (got != gold) begin
                    $display("DECODE MISMATCH: code=%b -> %h  got=%0.12g exp=%0.12g",
                             a8, dec, got, gold);
                    errors = errors + 1;
                end
            end
            test_count = test_count + 1;
        end

        // ================ (2) ENCODE round-trip, all 256 ===============
        for (ia = 0; ia < 256; ia = ia + 1) begin
            a8 = ia[7:0];
            rt = fp32_to_fp8e4m3(fp8e4m3_to_fp32(a8));
            if (^rt === 1'bx) begin
                $display("X in round-trip(%b) -> %b", a8, rt);
                errors = errors + 1;
            end else if (rt !== a8) begin
                $display("ROUND-TRIP MISMATCH: code=%b -> %b", a8, rt);
                errors = errors + 1;
            end
            test_count = test_count + 1;
        end

        // ----------------- directed RNE / saturation -------------------
        // saturation tails
        chk_enc(32'h43E00000, 8'b0_1111_110, "448");        //  448 exact -> 448
        chk_enc(32'h43E80000, 8'b0_1111_110, "464");        //  464 (>=mid) -> 448 (NaN slot saturates)
        chk_enc(32'h43F00000, 8'b0_1111_110, "480");        //  480 -> saturate 448
        chk_enc(32'h7F7FFFFF, 8'b0_1111_110, "fmax");       //  ~3.4e38 -> 448
        chk_enc(32'hC3E00000, 8'b1_1111_110, "-448");       // -448 -> -448
        chk_enc(32'h7F800000, 8'b0_1111_110, "+Inf");       // +Inf -> +448 (satfinite)
        chk_enc(32'hFF800000, 8'b1_1111_110, "-Inf");       // -Inf -> -448
        chk_enc(32'h7FC00000, 8'b0_1111_111, "+NaN");       // NaN  -> NaN code
        chk_enc(32'hFFC00000, 8'b1_1111_111, "-NaN");       // -NaN -> NaN code (sign kept)
        // normal RNE ties (tie-to-even)
        chk_enc(32'h3F880000, 8'b0_0111_000, "tie 1.0625->1.0");   // halfway 1.0/1.125 -> even 1.0
        chk_enc(32'h3F980000, 8'b0_0111_010, "tie 1.1875->1.25");  // halfway 1.125/1.25 -> even 1.25
        // subnormal RNE ties
        chk_enc(32'h3A800000, 8'b0_0000_000, "tie 2^-10->0");      // 0.5 step -> even 0
        chk_enc(32'h3B400000, 8'b0_0000_010, "tie 3*2^-10->2");    // 1.5 step -> even 2
        chk_enc(32'h3BA00000, 8'b0_0000_010, "tie 5*2^-10->2");    // 2.5 step -> even 2
        chk_enc(32'h3BE00000, 8'b0_0000_100, "tie 7*2^-10->4");    // 3.5 step -> even 4
        // round subnormal UP into the smallest normal
        chk_enc(32'h3C780000, 8'b0_0001_000, "7.75*2^-9->minnorm");
        // far below the grid -> zero, signs preserved
        chk_enc(32'h3A000000, 8'b0_0000_000, "2^-11->+0");
        chk_enc(32'hBA000000, 8'b1_0000_000, "-2^-11->-0");
        chk_enc(32'h00000000, 8'b0_0000_000, "+0");
        chk_enc(32'h80000000, 8'b1_0000_000, "-0");
        chk_enc(32'h00400000, 8'b0_0000_000, "fp32 subnorm->+0");  // tiny fp32 subnormal

        // ================= (3) MULTIPLY, all 65536 =====================
        for (ia = 0; ia < 256; ia = ia + 1) begin
            for (ib = 0; ib < 256; ib = ib + 1) begin
                a8   = ia[7:0];
                b8   = ib[7:0];
                prod = fp8_mul(a8, b8);
                if (^prod === 1'bx) begin
                    $display("X in fp8_mul(%b,%b) -> %h", a8, b8, prod);
                    errors = errors + 1;
                end else if (is_e4m3_nan(a8) || is_e4m3_nan(b8)) begin
                    if (!is_fp32_nan(prod)) begin
                        $display("MUL NaN-PROP MISMATCH: a=%b b=%b -> %h", a8, b8, prod);
                        errors = errors + 1;
                    end
                end else begin
                    gold = e4m3_real(a8) * e4m3_real(b8);
                    got  = f32_to_real(prod);
                    if (got != gold) begin
                        $display("MUL MISMATCH: a=%b b=%b -> %h  got=%0.12g exp=%0.12g",
                                 a8, b8, prod, got, gold);
                        errors = errors + 1;
                    end
                end
                test_count = test_count + 1;
            end
        end

        // ========================= verdict =============================
        if (errors != 0) begin
            $display("FAILED: %0d error(s) over %0d checks.", errors, test_count);
            $fatal(1, "fp8_e4m3_tb: MISMATCH");
        end else begin
            $display("ALL %0d TESTS PASSED", test_count);
        end
        $finish;
    end

endmodule
