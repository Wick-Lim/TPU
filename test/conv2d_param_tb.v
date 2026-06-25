`include "tpu_defs.vh"
//============================================================================
// conv2d_param_tb  --  2nd-SIZE proof for the parameterized conv2d_unit
//----------------------------------------------------------------------------
// PURPOSE
//   conv2d_unit was parameterized over IMG_H / IMG_W / K (defaults `CONV_H,
//   `CONV_W, `CONV_K = 8,8,3).  conv2d_unit_tb exercises the DEFAULT size
//   exhaustively and must stay byte-identical / same assertion count.  This
//   SEPARATE testbench instantiates the unit at a DIFFERENT, in-range size to
//   prove the parameterization is structurally correct (counters, line-buffer
//   depths, packing indices and the raster walk all track the parameters).
//
//   2nd SIZE:  IMG_H = IMG_W = 6, K = 3.
//     * VALID-pad output (pad=0,stride=1) : OH = OW = 6-3+1 = 4  (16 px)
//     * pad=1,stride=1                    : 6x6 = 36 px
//     * pad=0,stride=2                    : 2x2 =  4 px
//     * pad=1,stride=2                    : 3x3 =  9 px
//   This sits strictly inside the architectural envelope (IMG_W=6 <= 8 cols/TM
//   line; K=3 <= 8; K <= IMG_{H,W}).  Both the OUTPUT 4-lane packing and the
//   per-row image/kernel packing differ from the default (rows are 6 wide, not
//   8), so a fixed-8 implementation would mis-address; only a correctly
//   parameter-derived unit passes.
//
// INDEPENDENT GOLDEN
//   A direct 4-nested-loop convolution carried out in Verilog `real`
//   (floating-point), with ONLY the final pixel quantized back to Q7.8
//   (round-half-up + saturate).  This shares NO fixed-point accumulation path
//   with the DUT (which MACs in a 48-bit integer accumulator), and is NOT
//   mirrored from the DUT -- it is computed straight from img[]/ker[].  Because
//   each Q7.8*Q7.8 product is an exact integer and the K*K sum fits a 52-bit
//   real mantissa, the float sum is EXACT and the comparison is BIT-EXACT.
//
// COVERAGE (all four stride/pad combinations + directed + random)
//   P1 impulse kernel -> identity (valid 4x4 center crop of the 6x6 input)
//   P2 negative input & kernel
//   P3 positive & negative saturation (sat flag)
//   P4 stride=2 / pad=1 geometry (2x2, 6x6, 3x3 outputs)
//   PR >=120 constrained-random runs across all (stride,pad), bit-exact + sat
//   Every run also checks the (IMG_H+K)+ODH*ODW+1 latency closed form and the
//   busy/done handshake, exactly as the default TB does.
//
// GATE: prints "ALL <N> TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module conv2d_param_tb;

    // ---- 2nd-size geometry (DIFFERENT from the default 8/8/3) ----
    localparam integer H    = 6;           // IMG_H
    localparam integer W    = 6;           // IMG_W
    localparam integer K    = 3;           // kernel
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

    // ---- DUT instantiated at the 2nd size via parameter override ----
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

    // ---- TM layout: 6 image rows at 0..5, 3 kernel rows at 6..8, out from 9 ----
    localparam integer IN_BASE  = 0;
    localparam integer K_BASE   = H;            // 6
    localparam integer OUT_BASE = H + K;        // 9

    // ---- TB-side image / kernel storage as plain signed integers (Q7.8) ----
    integer img [0:H*W-1];
    integer ker [0:K*K-1];

    // ---- golden output buffer (up to 6x6 = 36 valid outputs) ----
    integer gold_out [0:H*W-1];
    integer gold_sat;

    //========================================================================
    // Helpers
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
    // computed PER AXIS (the unit supports rectangular maps; here square).
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
        seed = 32'h0C0FFEE1;

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
        // P1: impulse (identity) kernel -> valid 4x4 center crop of the input.
        //   center tap = 1.0 (Q7.8 = 256), others 0. out(oy,ox)=in(oy+1,ox+1)
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = (i*5) - 40;
        set_ker_const(0);
        ker[K*K/2] = 256;                  // center = 1.0 in Q7.8
        do_run("P1-impulse", 2'd1, 2'd0);  // 4x4
        for (r = 0; r < (H-K+1); r = r + 1)
            for (j = 0; j < (W-K+1); j = j + 1)
                if (read_dut_out(r*(W-K+1) + j) !== img[(r+1)*W + (j+1)]) begin
                    $display("FAIL[P1-cropcheck] (%0d,%0d) got=%0d want=%0d",
                             r, j, read_dut_out(r*(W-K+1)+j), img[(r+1)*W+(j+1)]);
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
        //   stride=2,pad=0 -> 2x2 ; pad=1,stride=1 -> 6x6 ; pad=1,stride=2 -> 3x3
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = ((i*17) % 200) - 90;
        for (i = 0; i < K*K; i = i + 1) ker[i] = (i - 4) * 50;
        do_run("P4-stride2",   2'd2, 2'd0);   // 2x2
        do_run("P4-pad1",      2'd1, 2'd1);   // 6x6
        do_run("P4-pad1str2",  2'd2, 2'd1);   // 3x3

        //--------------------------------------------------------------------
        // PR: constrained-random sweep across all (stride,pad).
        //--------------------------------------------------------------------
        svals[0] = 2'd1; svals[1] = 2'd2;
        pvals[0] = 2'd0; pvals[1] = 2'd1;
        for (run = 0; run < 160; run = run + 1) begin
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
            $display("CONV2D_PARAM TB: %0d FAILURES", fail);
            $fatal(1, "conv2d_param_tb FAILED");
        end
        $display("ALL %0d TESTS PASSED", pass);
        $finish;
    end

endmodule
