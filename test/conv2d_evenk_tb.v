`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// conv2d_evenk_tb  --  2nd-SIZE proof for the parameterized conv2d_unit at an
//                      EVEN kernel (K=2), the exact case the counter-width fix
//                      repairs.
//----------------------------------------------------------------------------
// PURPOSE / WHAT THIS PROVES
//   conv2d_unit sizes its output-pixel counter from NPIX_MAX = ODH_MAX*ODW_MAX
//   with ODH_MAX = OH + 2*PAD_MAX (OH = IMG-K+1, PAD_MAX = 1).  A FIXED bug
//   sized NPIX_MAX as IMG_H*IMG_W, which is only correct for K=3 (where pad=1
//   grows the output back to exactly IMG).  For an EVEN/small K (here K=2) with
//   pad=1, the true output dim is OH + 2*pad = IMG - K + 3 > IMG, so the old
//   IMG_H*IMG_W basis UNDER-sized the PW-bit pixel counter (and npix), which
//   wrapped, truncating the COMPUTE run and pulsing 'done' early -- only a
//   handful of the real output pixels were ever produced.
//
//   2nd SIZE (DIFFERENT from the default 8/8/3 AND from conv2d_param_tb's 6/6/3):
//     IMG_H = 4, IMG_W = 6, K = 2   (EVEN kernel).
//       * VALID-pad output (pad=0,stride=1) : OH=3, OW=5  => 3x5 = 15 px
//       * pad=1,stride=1                     : 5x7       = 35 px  <-- the case
//         that OVERFLOWED before the fix.  IMG_H*IMG_W = 24 < 35, so the buggy
//         counter wrapped at ~3 of the 35 pixels and signalled 'done' early.
//         The fixed sizing is ODH_MAX*ODW_MAX = (3+2)*(5+2) = 5*7 = 35.
//       * pad=0,stride=2                     : 2x3       =  6 px
//       * pad=1,stride=2                     : 3x4       = 12 px
//   This sits strictly inside the architectural envelope (IMG_W=6 <= 8 cols/TM
//   line; K=2 <= 8; K <= IMG_{H,W}).  A non-square map AND an even kernel are
//   both exercised, so size-literals / K=3-only assumptions are caught.
//
// INDEPENDENT GOLDEN
//   A direct 4-nested-loop convolution carried out in Verilog `real`
//   (floating-point), with ONLY the final pixel quantized back to Q7.8
//   (round-half-up + saturate).  This shares NO fixed-point accumulation path
//   with the DUT (which MACs in a 48-bit integer accumulator) and is NOT
//   mirrored from the DUT -- it is computed straight from img[]/ker[] using the
//   SAME stride/pad sanitization the DUT documents.  Each Q7.8*Q7.8 product is
//   an exact integer and the K*K sum fits a 52-bit real mantissa, so the float
//   sum is EXACT and the comparison is BIT-EXACT.
//
// COVERAGE (all four stride/pad combinations + directed + >=40 random)
//   P1 impulse kernel -> identity-ish (valid 3x5 crop of the 4x6 input)
//   P2 negative input & kernel (signed MAC)
//   P3 positive & negative saturation (sat flag)
//   P4 the three remaining (stride,pad) geometries, INCLUDING pad=1 -> 5x7=35
//      px (the repaired case): the PIXEL COUNT, done-latency and every output
//      value are all asserted against the golden.
//   PR >=40 constrained-random runs across all (stride,pad), bit-exact + sat.
//   Every run also checks the (IMG_H+K)+ODH*ODW+1 latency closed form and the
//   busy/done handshake, exactly as conv2d_param_tb does.  Because the latency
//   closed form includes ODH*ODW, the pad=1 35-pixel run cannot pass unless the
//   counter actually walks all 35 pixels -- the pre-fix unit fails here.
//
// GATE: prints "ALL <N> TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module conv2d_evenk_tb;

    // ---- 2nd-size geometry: EVEN K, non-square map (DIFFERENT from 8/8/3) ----
    localparam integer H    = 4;           // IMG_H
    localparam integer W    = 6;           // IMG_W
    localparam integer K    = 2;           // EVEN kernel
    localparam integer ELEM = `ELEM_W;     // 16

    // ---- clock / reset ----
    reg clk;
    reg rst;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ---- DUT control ports ----
    reg                  start;
    reg  [`TM_IDX_W-1:0] in_base, k_base, out_base;
    reg  [1:0]           stride, pad;
    wire                 busy, done, sat;

    // ---- DUT <-> TM access ports ----
    wire [`TM_IDX_W-1:0] rd_addr;
    reg  [`LINE_W-1:0]   rd_data;
    wire                 tm_we;
    wire [`TM_IDX_W-1:0] tm_waddr;
    wire [`LINE_W-1:0]   tm_wdata;

    // ---- behavioral tile memory owned by the TB (models TM) ----
    reg [`LINE_W-1:0] tm [0:`TM_LINES-1];
    always @(*) rd_data = tm[rd_addr];
    always @(posedge clk) begin
        if (tm_we) tm[tm_waddr] <= tm_wdata;
    end

    // ---- DUT instantiated at the EVEN-K size via parameter override ----
    conv2d_unit #(.IMG_H(H), .IMG_W(W), .K(K)) dut (
        .clk      (clk),
        .rst      (rst),
        .start    (start),
        .in_base  (in_base),
        .k_base   (k_base),
        .out_base (out_base),
        .stride   (stride),
        .pad      (pad),
        .busy     (busy),
        .done     (done),
        .sat      (sat),
        .rd_addr  (rd_addr),
        .rd_data  (rd_data),
        .tm_we    (tm_we),
        .tm_waddr (tm_waddr),
        .tm_wdata (tm_wdata)
    );

    integer pass;
    integer fail;
    integer seed;

    // ---- TM layout: 4 image rows at 0..3, 2 kernel rows at 4..5, out from 6 ----
    //   pad=1 output is 5x7 = 35 px => ceil(35/4) = 9 lines (6..14): fits the 32
    //   line TM with room to spare.  Mirrors conv2d_param_tb's packing/layout.
    localparam integer IN_BASE  = 0;
    localparam integer K_BASE   = H;            // 4
    localparam integer OUT_BASE = H + K;        // 6

    // ---- TB-side image / kernel storage as plain signed integers (Q7.8) ----
    integer img [0:H*W-1];
    integer ker [0:K*K-1];

    // ---- golden output buffer.  Max pixels at pad=1,stride=1 is 5*7 = 35. ----
    localparam integer GMAX = (H - K + 1 + 2) * (W - K + 1 + 2); // 5*7 = 35
    integer gold_out [0:GMAX-1];
    integer gold_sat;

    //========================================================================
    // Helpers  (TM packing mirrors conv2d_param_tb exactly)
    //========================================================================

    // Pack the W image columns of row r into a 128-bit TM line and store it.
    task load_image_to_tm;
        integer r, c;
        reg [`LINE_W-1:0] line;
        begin
            for (r = 0; r < H; r = r + 1) begin
                line = {`LINE_W{1'b0}};
                for (c = 0; c < W; c = c + 1)
                    line[c*ELEM +: ELEM] = img[r*W + c][ELEM-1:0];
                tm[IN_BASE + r] = line;
            end
        end
    endtask

    // Pack the K kernel columns of row kr into the low K*16 bits of a TM line.
    task load_kernel_to_tm;
        integer kr, kc;
        reg [`LINE_W-1:0] line;
        begin
            for (kr = 0; kr < K; kr = kr + 1) begin
                line = {`LINE_W{1'b0}};
                for (kc = 0; kc < K; kc = kc + 1)
                    line[kc*ELEM +: ELEM] = ker[kr*K + kc][ELEM-1:0];
                tm[K_BASE + kr] = line;
            end
        end
    endtask

    // INDEPENDENT golden: real-valued 4-nested-loop conv, quantized at the end.
    // Uses the SAME stride/pad sanitization the DUT documents.  Output dims are
    // computed PER AXIS (the unit supports rectangular maps; here H!=W, K even).
    task golden_conv;
        input  [1:0] s_in, p_in;
        output integer odh, odw;
        integer ss, pp, oy, ox, wr, wc, iy, ix;
        real    accr, biasr, shifted;
        integer q;
        begin
            ss  = (s_in == 2'd2) ? 2 : 1;
            pp  = (p_in != 2'd0) ? 1 : 0;
            odh = ((H + 2*pp - K) / ss) + 1;
            odw = ((W + 2*pp - K) / ss) + 1;
            gold_sat = 0;
            for (oy = 0; oy < odh; oy = oy + 1) begin
                for (ox = 0; ox < odw; ox = ox + 1) begin
                    accr = 0.0;
                    for (wr = 0; wr < K; wr = wr + 1) begin
                        for (wc = 0; wc < K; wc = wc + 1) begin
                            iy = oy*ss - pp + wr;
                            ix = ox*ss - pp + wc;
                            if ((iy >= 0) && (iy < H) &&
                                (ix >= 0) && (ix < W)) begin
                                accr = accr +
                                    (img[iy*W + ix] * 1.0) * (ker[wr*K + wc] * 1.0);
                            end
                        end
                    end
                    // Boundary quantize: round-half-up of acc/2^8, then clamp.
                    biasr   = accr + 128.0;
                    shifted = $floor(biasr / 256.0);
                    q = shifted;
                    if (q > `Q78_MAX_VAL) begin
                        q = `Q78_MAX_VAL; gold_sat = 1;
                    end else if (q < `Q78_MIN_VAL) begin
                        q = `Q78_MIN_VAL; gold_sat = 1;
                    end
                    gold_out[oy*odw + ox] = q;
                end
            end
        end
    endtask

    // Read back one valid output pixel `p` (raster index) from the DUT's TM:
    // LINE_LANES px/line packed low-first starting at OUT_BASE.
    function integer read_dut_out;
        input integer p;
        reg [`LINE_W-1:0] line;
        reg [ELEM-1:0]    raw;
        integer ln, lane;
        begin
            ln   = OUT_BASE + (p / `LINE_LANES);
            lane = p % `LINE_LANES;
            line = tm[ln];
            raw  = line[lane*ELEM +: ELEM];
            read_dut_out = $signed(raw);
        end
    endfunction

    // Pulse start, wait for done, assert the latency closed form + handshake.
    // exp_npix is the GOLDEN pixel count; the closed form below FAILS for the
    // pre-fix unit on pad=1 because its truncated walk produces too few pixels
    // (done fires early) -- this is the assertion that catches the bug.
    task run_and_time;
        input [1:0] s_in, p_in;
        input integer exp_npix;
        integer cyc, exp_cyc;
        begin
            @(negedge clk);
            start  = 1'b1;
            stride = s_in;
            pad    = p_in;
            @(posedge clk);                // edge that samples start
            @(negedge clk);
            start  = 1'b0;
            if (busy !== 1'b1) begin
                $display("FAIL: busy not asserted after start (s=%0d p=%0d)",
                         s_in, p_in);
                fail = fail + 1; $fatal(1, "busy handshake");
            end
            cyc = 0;
            while (done !== 1'b1) begin
                @(posedge clk);
                cyc = cyc + 1;
                if (cyc > 1000) begin
                    $display("FAIL: done never asserted");
                    fail = fail + 1; $fatal(1, "done timeout");
                end
            end
            // Documented latency: (IMG_H+K) load + npix compute + 1 done.
            exp_cyc = (H + K) + exp_npix + 1;
            if (cyc !== exp_cyc) begin
                $display("FAIL: latency mismatch s=%0d p=%0d got=%0d exp=%0d",
                         s_in, p_in, cyc, exp_cyc);
                fail = fail + 1; $fatal(1, "latency");
            end else pass = pass + 1;
            if (busy !== 1'b0) begin
                $display("FAIL: busy still high at done");
                fail = fail + 1; $fatal(1, "busy clear");
            end else pass = pass + 1;
            @(negedge clk);                // let the final write settle in tm[]
        end
    endtask

    // Full run: program TM, launch DUT, compare every valid output bit-exact
    // and the sat flag against the golden.
    task do_run;
        input [255:0] tag;
        input [1:0]   s_in, p_in;
        integer odh, odw, p, npix, dval, gval;
        begin
            load_image_to_tm;
            load_kernel_to_tm;
            golden_conv(s_in, p_in, odh, odw);
            npix = odh * odw;
            run_and_time(s_in, p_in, npix);
            for (p = 0; p < npix; p = p + 1) begin
                dval = read_dut_out(p);
                gval = gold_out[p];
                if (dval !== gval) begin
                    $display("FAIL[%0s] px=%0d (oy=%0d ox=%0d) dut=%0d gold=%0d",
                             tag, p, p/odw, p%odw, dval, gval);
                    fail = fail + 1; $fatal(1, "conv output mismatch");
                end else pass = pass + 1;
            end
            if (sat !== gold_sat[0]) begin
                $display("FAIL[%0s] sat flag dut=%b gold=%b", tag, sat, gold_sat[0]);
                fail = fail + 1; $fatal(1, "sat flag mismatch");
            end else pass = pass + 1;
        end
    endtask

    task set_img_const; input integer v; integer i;
        begin for (i=0;i<H*W;i=i+1) img[i]=v; end
    endtask
    task set_ker_const; input integer v; integer i;
        begin for (i=0;i<K*K;i=i+1) ker[i]=v; end
    endtask

    function integer rnd_q78;
        input integer dummy;
        reg [15:0] r;
        begin
            r = $random(seed);
            rnd_q78 = $signed(r);
        end
    endfunction

    integer i, j, r, run, sc, pc;
    reg [1:0] svals [0:1];
    reg [1:0] pvals [0:1];

    //========================================================================
    // Main
    //========================================================================
    initial begin
        pass = 0;
        fail = 0;
        seed = 32'h0E5E_8B2D;          // distinct seed for the even-K sweep

        start    = 1'b0;
        in_base  = IN_BASE[`TM_IDX_W-1:0];
        k_base   = K_BASE[`TM_IDX_W-1:0];
        out_base = OUT_BASE[`TM_IDX_W-1:0];
        stride   = 2'd1;
        pad      = 2'd0;

        for (i = 0; i < `TM_LINES; i = i + 1) tm[i] = {`LINE_W{1'b0}};

        rst = 1'b1;
        @(posedge clk);
        @(posedge clk);
        @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        //--------------------------------------------------------------------
        // P1: impulse kernel (top-left tap = 1.0) -> valid 3x5 crop of input.
        //   For an EVEN K=2 cross-correlation, the K*K window at (oy,ox) covers
        //   input rows oy..oy+1, cols ox..ox+1; a top-left impulse selects
        //   in(oy,ox).  So out(oy,ox) = in(oy,ox) over the valid 3x5 grid.
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = (i*5) - 40;
        set_ker_const(0);
        ker[0] = 256;                      // top-left tap = 1.0 in Q7.8
        do_run("P1-impulse", 2'd1, 2'd0);  // 3x5
        for (r = 0; r < (H-K+1); r = r + 1)
            for (j = 0; j < (W-K+1); j = j + 1)
                if (read_dut_out(r*(W-K+1) + j) !== img[r*W + j]) begin
                    $display("FAIL[P1-cropcheck] (%0d,%0d) got=%0d want=%0d",
                             r, j, read_dut_out(r*(W-K+1)+j), img[r*W+j]);
                    fail = fail + 1; $fatal(1, "identity crop");
                end
        pass = pass + 1;

        //--------------------------------------------------------------------
        // P2: negative input & kernel (signed MAC correctness).
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = -((i % 7) * 40 + 9);
        for (i = 0; i < K*K; i = i + 1) ker[i] = (i % 2) ? -72 : 96;
        do_run("P2-negative", 2'd1, 2'd0);

        //--------------------------------------------------------------------
        // P3a/b: positive and negative saturation.
        //--------------------------------------------------------------------
        set_img_const(100 * 256);          // 100.0 in Q7.8
        set_ker_const(256);                // 1.0
        do_run("P3a-satpos", 2'd1, 2'd0);
        if (sat !== 1'b1) begin
            $display("FAIL[P3a] expected sat=1"); fail=fail+1; $fatal(1,"sat+");
        end else pass = pass + 1;
        set_img_const(-100 * 256);
        set_ker_const(256);
        do_run("P3b-satneg", 2'd1, 2'd0);
        if (sat !== 1'b1) begin
            $display("FAIL[P3b] expected sat=1"); fail=fail+1; $fatal(1,"sat-");
        end else pass = pass + 1;

        //--------------------------------------------------------------------
        // P4: the three remaining (stride,pad) geometries.
        //   stride=2,pad=0 -> 2x3 (6 px)
        //   pad=1,stride=1 -> 5x7 (35 px)  <-- THE REPAIRED CASE.  The latency
        //     closed form in run_and_time asserts the unit walks all 35 pixels
        //     (done fires at exactly (H+K)+35+1); do_run then checks all 35
        //     output values + the sat flag.  Pre-fix, npix wrapped (24-line
        //     counter) and only ~3 px were produced before an early 'done', so
        //     BOTH the latency assertion and the per-pixel checks fail.
        //   pad=1,stride=2 -> 3x4 (12 px)
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = ((i*17) % 200) - 90;
        for (i = 0; i < K*K; i = i + 1) ker[i] = (i - 2) * 50;
        do_run("P4-stride2",   2'd2, 2'd0);   // 2x3  =  6 px
        do_run("P4-pad1",      2'd1, 2'd1);   // 5x7  = 35 px  (overflow case)
        do_run("P4-pad1str2",  2'd2, 2'd1);   // 3x4  = 12 px

        // Extra DIRECTED pad=1 run with a full random map + explicit pixel-count
        // assertion, hammering the repaired 35-pixel path once more.
        for (i = 0; i < H*W; i = i + 1) img[i] = rnd_q78(0) % 1024;
        for (i = 0; i < K*K; i = i + 1) ker[i] = rnd_q78(0) % 512;
        begin : pad1_count
            integer odh4, odw4;
            golden_conv(2'd1, 2'd1, odh4, odw4);
            if ((odh4 != 5) || (odw4 != 7) || (odh4*odw4 != 35)) begin
                $display("FAIL[P4-pad1-geom] golden dims %0dx%0d (exp 5x7)",
                         odh4, odw4);
                fail = fail + 1; $fatal(1, "pad1 geometry");
            end else pass = pass + 1;
        end
        do_run("P4-pad1-rand", 2'd1, 2'd1);   // 35 px, full bit-exact recheck

        //--------------------------------------------------------------------
        // PR: constrained-random sweep across all (stride,pad).  >=40 runs;
        //   run%2 selects stride, (run/2)%2 selects pad, so every 4 runs covers
        //   all four (stride,pad) combos -- the pad=1 35-px path is hit ~1/4 of
        //   the time, each time bit-exact + latency-checked.
        //--------------------------------------------------------------------
        svals[0] = 2'd1; svals[1] = 2'd2;
        pvals[0] = 2'd0; pvals[1] = 2'd1;
        for (run = 0; run < 48; run = run + 1) begin
            for (i = 0; i < H*W; i = i + 1) begin
                if ((run % 8) == 0) img[i] = rnd_q78(0);
                else                img[i] = (rnd_q78(0) % 2048);
            end
            for (i = 0; i < K*K; i = i + 1) begin
                if ((run % 8) == 0) ker[i] = rnd_q78(0);
                else                ker[i] = (rnd_q78(0) % 768);
            end
            sc = run % 2;
            pc = (run / 2) % 2;
            do_run("PR-rand", svals[sc], pvals[pc]);
        end

        //--------------------------------------------------------------------
        if (fail != 0) begin
            $display("CONV2D_EVENK TB: %0d FAILURES", fail);
            $fatal(1, "conv2d_evenk_tb FAILED");
        end
        $display("ALL %0d TESTS PASSED", pass);
        $finish;
    end

endmodule
