`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// glm_matmul_fp8_tb.v  --  self-checking TB for the FP8-native GEMM.
//
//   GOLDEN MODEL: an INDEPENDENT fp64 GEMM whose inputs are first quantized to
//   E4M3 exactly as the DUT does (activation: bf16 -> *2^a_shift -> E4M3 encode;
//   weight: the given E4M3 code), and whose REDUCTION uses the SAME [128,128]
//   BLOCK-SCALED scheme as the DUT: fp8 products are inner-accumulated per
//   128-wide K-block, each block partial is multiplied by that block's bf16
//   weight scale, the scaled block partials are outer-accumulated, and finally
//   the per-token 2^-a_shift is undone.  The fp8 per-element quantization error
//   is therefore INSIDE the golden; the only residual DUT/golden gap is fp32
//   accumulation (vs fp64) + the fp32 dequant multiply + the final bf16
//   rounding.  The tolerance is built from those (scales with K for the
//   accumulation term), NOT from the raw 2^-3 E4M3 ulp.
//
//   X-AWARE: any X in a captured output bit is a hard failure.
//   Emits "ALL <N> TESTS PASSED" and $fatal on any mismatch / X.
//============================================================================
module glm_matmul_fp8_tb;

    localparam integer PE_M = 4;
    localparam integer PE_N = 4;
    localparam integer KMAX = 256;
    localparam integer BLK  = 128;
    localparam integer NB   = (KMAX + BLK - 1) / BLK;   // # K-blocks (= 2)
    localparam integer KW   = $clog2(KMAX+1);

    // ---- include the contract functions for the encode step of the golden ----
    `include "glm_fp.vh"
    `include "fp8_e4m3.vh"

    reg                       clk = 1'b0;
    reg                       rst = 1'b1;
    reg                       start = 1'b0;
    reg  [KW-1:0]             k_len = 0;
    reg                       in_valid = 1'b0;
    reg  [16*PE_M-1:0]        a_col = 0;
    reg  [ 8*PE_N-1:0]        w_row = 0;
    reg  [ 8*PE_M-1:0]        a_shift = 0;
    reg  [16*PE_N*NB-1:0]     w_scale = 0;          // bf16 scale per (col, K-block)
    wire                      busy;
    wire                      out_valid;
    wire [16*PE_M*PE_N-1:0]   c_out;

    glm_matmul_fp8 #(.PE_M(PE_M), .PE_N(PE_N), .KMAX(KMAX), .BLK(BLK)) dut (
        .clk(clk), .rst(rst), .start(start), .k_len(k_len),
        .in_valid(in_valid), .a_col(a_col), .w_row(w_row),
        .a_shift(a_shift), .w_scale(w_scale),
        .busy(busy), .out_valid(out_valid), .c_out(c_out)
    );

    always #5 clk = ~clk;

    integer pass_cnt = 0;
    integer fail_cnt = 0;

    //----------------------------------------------------------------------
    // Independent fp64 decoders (NOT the DUT's fp32 functions).
    //----------------------------------------------------------------------
    function real e4m3_real(input [7:0] x);
        reg       s; reg [3:0] e; reg [2:0] m; real v;
        begin
            s = x[7]; e = x[6:3]; m = x[2:0];
            if (e == 4'hF && m == 3'h7)      v = 0.0;          // NaN: never used
            else if (e == 4'h0)              v = m * (2.0**(-9)); // zero/subnormal
            else                             v = (1.0 + m/8.0) * (2.0**(real'($signed({1'b0,e}))-7.0));
            e4m3_real = s ? -v : v;
        end
    endfunction

    function real bf16_real(input [15:0] b);
        reg       s; reg [7:0] e; reg [6:0] m; real v;
        begin
            s = b[15]; e = b[14:7]; m = b[6:0];
            if (e == 8'hFF)      v = 0.0;                       // inf/nan: never used
            else if (e == 8'h00) v = m * (2.0**(-133));         // subnormal (we avoid)
            else                 v = (1.0 + m/128.0) * (2.0**(real'($signed({1'b0,e}))-127.0));
            bf16_real = s ? -v : v;
        end
    endfunction

    // Exact fp32 word of bf16_value * 2^k (the DUT's fp32_scale_pow2), so the
    // golden's E4M3 encode sees the SAME fp32 the DUT encodes.
    function [31:0] scaled_fp32(input [15:0] b, input signed [9:0] k);
        reg               s; reg [7:0] e; reg [22:0] m; reg signed [10:0] ne;
        begin
            s = b[15]; e = b[14:7]; m = {b[6:0], 16'b0};
            if (e == 8'hFF)      scaled_fp32 = {s, e, m};
            else if (e == 8'h00) scaled_fp32 = {s, 31'b0};
            else begin
                ne = $signed({3'b0, e}) + k;
                if (ne >= 11'sd255)      scaled_fp32 = {s, 8'hFF, 23'b0};
                else if (ne <= 11'sd0)   scaled_fp32 = {s, 31'b0};
                else                     scaled_fp32 = {s, ne[7:0], m};
            end
        end
    endfunction

    //----------------------------------------------------------------------
    // Test storage
    //----------------------------------------------------------------------
    reg [15:0] A  [0:PE_M-1][0:KMAX-1];   // bf16 activations
    reg [ 7:0] W  [0:KMAX-1][0:PE_N-1];   // E4M3 weights
    reg [ 7:0] ash_v [0:PE_M-1];          // per-row activation pow2 shift (signed)
    reg [15:0] wsc_v [0:PE_N-1][0:NB-1];  // bf16 weight BLOCK scale per (col, K-block)

    integer kk;

    // generate a "safe" normal bf16: exp in a moderate range, random sign/mant.
    function [15:0] rnd_bf16(input integer lo_e, input integer hi_e);
        reg [7:0] e; reg [6:0] m; reg s;
        begin
            s = $random & 1;
            e = lo_e + ({$random} % (hi_e - lo_e + 1));
            m = $random;
            rnd_bf16 = {s, e, m};
        end
    endfunction

    // generate a non-NaN E4M3 code.
    function [7:0] rnd_e4m3;
        reg [7:0] x;
        begin
            x = $random;
            if (x[6:0] == 7'b1111111) x[0] = 1'b0;   // dodge the NaN pattern
            rnd_e4m3 = x;
        end
    endfunction

    real max_ratio;   // worst (err/tol) seen, for reporting headroom

    //----------------------------------------------------------------------
    // Drive one tile and check every output element vs the fp64 golden.
    //----------------------------------------------------------------------
    task run_tile(input integer K, input [127:0] tag);
        integer pi, pj, k, t, b, kstart, kend;
        real    golden, sum_abs, dutr, err, tol;
        real    bsum, babs, wsr, ashf;
        reg [31:0] sfp;
        reg [ 7:0] aq;
        real    aqr, wr;
        reg [15:0] cbits;
        reg signed [9:0] ksh;
        begin
            // ---- latch scales + dims via start ----
            @(negedge clk);
            for (pi = 0; pi < PE_M; pi = pi + 1)
                a_shift[8*pi +: 8] = ash_v[pi];
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (b = 0; b < NB; b = b + 1)
                    w_scale[16*(b*PE_N + pj) +: 16] = wsc_v[pj][b];
            k_len    = K[KW-1:0];
            start    = 1'b1;
            in_valid = 1'b0;
            @(negedge clk);
            start = 1'b0;

            // ---- stream K beats ----
            for (k = 0; k < K; k = k + 1) begin
                in_valid = 1'b1;
                for (pi = 0; pi < PE_M; pi = pi + 1)
                    a_col[16*pi +: 16] = A[pi][k];
                for (pj = 0; pj < PE_N; pj = pj + 1)
                    w_row[8*pj +: 8] = W[k][pj];
                @(negedge clk);
            end
            in_valid = 1'b0;
            a_col = 0; w_row = 0;

            // ---- wait for out_valid ----
            t = 0;
            while (out_valid !== 1'b1) begin
                @(negedge clk);
                t = t + 1;
                if (t > KMAX + 200) begin
                    $display("FATAL [%0s]: out_valid never asserted", tag);
                    $fatal(1, "timeout");
                end
            end

            // ---- check every output element vs the [128,128] block-scaled golden ----
            for (pi = 0; pi < PE_M; pi = pi + 1) begin
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    ksh  = $signed(ash_v[pi]);
                    ashf = 2.0 ** real'(ksh);
                    golden = 0.0; sum_abs = 0.0;
                    // outer loop over 128-wide K-blocks; inner accumulate then scale.
                    for (b = 0; b < NB; b = b + 1) begin
                        kstart = b * BLK;
                        kend   = (b + 1) * BLK; if (kend > K) kend = K;
                        bsum = 0.0; babs = 0.0;
                        for (k = kstart; k < kend; k = k + 1) begin
                            sfp = scaled_fp32(A[pi][k], ksh);      // bf16*2^ash as fp32
                            aq  = fp32_to_fp8e4m3(sfp);            // DUT-identical encode
                            aqr = e4m3_real(aq);                   // independent fp64 decode
                            wr  = e4m3_real(W[k][pj]);
                            bsum = bsum + aqr * wr;
                            babs = babs + (aqr*wr >= 0.0 ? aqr*wr : -(aqr*wr));
                        end
                        wsr      = bf16_real(wsc_v[pj][b]);        // this block's weight scale
                        golden   = golden  + bsum * wsr;
                        sum_abs  = sum_abs + babs * (wsr >= 0.0 ? wsr : -wsr);
                    end
                    // undo the per-token pow2 prescale.
                    golden  = golden  / ashf;
                    sum_abs = sum_abs / ashf;

                    cbits = c_out[16*(pi*PE_N + pj) +: 16];
                    // X-aware: any X bit is a failure.
                    if (^cbits === 1'bx) begin
                        $display("FAIL [%0s] (%0d,%0d): X in output = %b", tag, pi, pj, cbits);
                        fail_cnt = fail_cnt + 1;
                    end else begin
                        dutr = bf16_real(cbits);
                        err  = dutr - golden; if (err < 0.0) err = -err;
                        // principled tolerance: fp32 accumulation (~K ulp, scaled
                        // by the sum of |terms| to be safe under cancellation) +
                        // bf16 output rounding (~2^-7 of |golden|) + a tiny floor.
                        tol  = sum_abs * (real'(K) * (2.0**(-22)))
                             + (golden >= 0.0 ? golden : -golden) * (2.0**(-7))
                             + (2.0**(-18));
                        if (err > tol) begin
                            $display("FAIL [%0s] (%0d,%0d): dut=%g golden=%g err=%g tol=%g (K=%0d)",
                                     tag, pi, pj, dutr, golden, err, tol, K);
                            fail_cnt = fail_cnt + 1;
                        end else begin
                            pass_cnt = pass_cnt + 1;
                            if (tol > 0.0 && (err/tol) > max_ratio) max_ratio = err/tol;
                        end
                    end
                end
            end
            @(negedge clk);
        end
    endtask

    // helper: set ALL K-blocks of every column to 1.0 (for single-block tests).
    task set_unit_scales;
        integer pj2, b2;
        begin
            for (pj2 = 0; pj2 < PE_N; pj2 = pj2 + 1)
                for (b2 = 0; b2 < NB; b2 = b2 + 1)
                    wsc_v[pj2][b2] = 16'h3F80;  // 1.0
        end
    endtask

    integer ti, pi2, pj2, ki2, b2;

    initial begin
        max_ratio = 0.0;
        // reset
        rst = 1'b1;
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        //==================================================================
        // TEST 1: small K, all a_shift=0, all block scales 1.0, simple values.
        //==================================================================
        for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1) ash_v[pi2] = 8'sd0;
        set_unit_scales;
        for (ki2 = 0; ki2 < 8; ki2 = ki2 + 1) begin
            for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1) A[pi2][ki2] = rnd_bf16(124, 129);
            for (pj2 = 0; pj2 < PE_N; pj2 = pj2 + 1) W[ki2][pj2] = rnd_e4m3();
        end
        run_tile(8, "K8");

        //==================================================================
        // TEST 2: K=1 edge (single term).
        //==================================================================
        for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1) A[pi2][0] = rnd_bf16(125, 128);
        for (pj2 = 0; pj2 < PE_N; pj2 = pj2 + 1) W[0][pj2] = rnd_e4m3();
        run_tile(1, "K1");

        //==================================================================
        // TEST 3: zeros (E4M3 +0 weights -> exact 0 output).
        //==================================================================
        for (ki2 = 0; ki2 < 16; ki2 = ki2 + 1) begin
            for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1) A[pi2][ki2] = rnd_bf16(124, 129);
            for (pj2 = 0; pj2 < PE_N; pj2 = pj2 + 1) W[ki2][pj2] = 8'h00; // +0
        end
        run_tile(16, "ZERO");

        //==================================================================
        // TEST 4: per-row activation shifts + per-col weight scales (single block).
        //==================================================================
        ash_v[0] = 8'sd3; ash_v[1] = -8'sd2; ash_v[2] = 8'sd0; ash_v[3] = 8'sd5;
        set_unit_scales;
        wsc_v[0][0] = 16'h3F80; // 1.0
        wsc_v[1][0] = 16'h4000; // 2.0
        wsc_v[2][0] = 16'h3E80; // 0.25
        wsc_v[3][0] = 16'h40A0; // 5.0
        for (ki2 = 0; ki2 < 64; ki2 = ki2 + 1) begin
            // smaller activations so the +shift lands them inside E4M3 range
            for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1) A[pi2][ki2] = rnd_bf16(118, 126);
            for (pj2 = 0; pj2 < PE_N; pj2 = pj2 + 1) W[ki2][pj2] = rnd_e4m3();
        end
        run_tile(64, "SCALES");

        //==================================================================
        // TEST 5: full KMAX reduction -> TWO 128-wide K-blocks with DISTINCT
        //         per-block weight scales (exercises block-scaled accumulation).
        //==================================================================
        for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1) ash_v[pi2] = 8'sd0;
        // K-block 0 scales
        wsc_v[0][0] = 16'h3F00; // 0.5
        wsc_v[1][0] = 16'h3F80; // 1.0
        wsc_v[2][0] = 16'h3FC0; // 1.5
        wsc_v[3][0] = 16'h4040; // 3.0
        // K-block 1 scales (DIFFERENT from block 0)
        wsc_v[0][1] = 16'h4000; // 2.0
        wsc_v[1][1] = 16'h3E80; // 0.25
        wsc_v[2][1] = 16'h40A0; // 5.0
        wsc_v[3][1] = 16'h3F40; // 0.75
        for (ki2 = 0; ki2 < KMAX; ki2 = ki2 + 1) begin
            for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1) A[pi2][ki2] = rnd_bf16(123, 130);
            for (pj2 = 0; pj2 < PE_N; pj2 = pj2 + 1) W[ki2][pj2] = rnd_e4m3();
        end
        run_tile(KMAX, "KMAX2BLK");

        //==================================================================
        // TEST 6: K=200 -> a FULL block + a PARTIAL second block, distinct scales.
        //==================================================================
        ash_v[0] = 8'sd1; ash_v[1] = 8'sd0; ash_v[2] = -8'sd1; ash_v[3] = 8'sd2;
        wsc_v[0][0] = 16'h3F80; wsc_v[0][1] = 16'h3FC0; // 1.0 / 1.5
        wsc_v[1][0] = 16'h4000; wsc_v[1][1] = 16'h3F00; // 2.0 / 0.5
        wsc_v[2][0] = 16'h3F40; wsc_v[2][1] = 16'h4040; // 0.75 / 3.0
        wsc_v[3][0] = 16'h3E80; wsc_v[3][1] = 16'h3F80; // 0.25 / 1.0
        for (ki2 = 0; ki2 < 200; ki2 = ki2 + 1) begin
            for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1) A[pi2][ki2] = rnd_bf16(120, 127);
            for (pj2 = 0; pj2 < PE_N; pj2 = pj2 + 1) W[ki2][pj2] = rnd_e4m3();
        end
        run_tile(200, "K200");

        //==================================================================
        // TESTS 7..14: random matrices, random shifts/scales, varied K
        //              (alternating single-block 100 and two-block 200).
        //==================================================================
        for (ti = 0; ti < 8; ti = ti + 1) begin
            for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1)
                ash_v[pi2] = ($random % 5);             // small +/- shift
            for (pj2 = 0; pj2 < PE_N; pj2 = pj2 + 1)
                for (b2 = 0; b2 < NB; b2 = b2 + 1)
                    wsc_v[pj2][b2] = rnd_bf16(124, 130); // random positive-ish scale per block
            for (ki2 = 0; ki2 < 200; ki2 = ki2 + 1) begin
                for (pi2 = 0; pi2 < PE_M; pi2 = pi2 + 1) A[pi2][ki2] = rnd_bf16(121, 130);
                for (pj2 = 0; pj2 < PE_N; pj2 = pj2 + 1) W[ki2][pj2] = rnd_e4m3();
            end
            run_tile((ti & 1) ? 200 : 100, "RND");
        end

        //==================================================================
        if (fail_cnt != 0) begin
            $display("FAILED: %0d mismatch(es), %0d passed.", fail_cnt, pass_cnt);
            $fatal(1, "glm_matmul_fp8 verification FAILED");
        end else begin
            $display("ALL %0d TESTS PASSED  (worst err/tol = %.4f)", pass_cnt, max_ratio);
        end
        $finish;
    end
endmodule
