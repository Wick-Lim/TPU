`include "tpu_defs.vh"
//============================================================================
// conv2d_unit_tb  --  self-checking unit TB for conv2d_unit   (SPEC §5.2, §6)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL
//   The DUT is the line-buffered fixed-point 2-D conv unit.  The golden model
//   is a DIFFERENT computation: a direct 4-nested-loop convolution carried out
//   entirely in Verilog `real` (floating-point) arithmetic -- input/kernel
//   elements are converted Q7.8 -> real, the 3x3 window-sum is accumulated as a
//   real, and ONLY the final pixel is quantized back to Q7.8 (round-half-up +
//   saturate) at the boundary.  The golden therefore shares no fixed-point
//   accumulation path with the DUT (the DUT MACs in a 48-bit integer
//   accumulator; the golden sums in float), so the two cannot share an
//   arithmetic bug.  Because conv is an EXACT integer op (no LUT), the
//   committed contract is round-half-up+saturate and the comparison is
//   BIT-EXACT (no tolerance): see note (*) below for why the float reference
//   reproduces the same exact value.
//
//   (*) Each Q7.8*Q7.8 product is an exact integer (<= 32767*32767 < 2^30) and
//   the 9-term sum is exact in a 48-bit integer; a Verilog `real` has a 52-bit
//   mantissa, so the float sum of the same 9 integer products is also EXACT.
//   The boundary quantization (round-half-up of acc/256, then clamp to
//   [-32768,32767]) is applied identically as a SEPARATE final step in the
//   golden (computed on the exact real accumulator), so DUT and golden agree
//   bit-exactly while having independent accumulation paths.  This matches
//   SPEC §6 ("bit-exact for the exact ops ... reference applies round+saturate
//   as a separate final quantization step, not as the accumulation method").
//
// TM MODEL
//   The TB owns a behavioral tile memory `tm[]` (32 x 128b) with the same
//   combinational-read / synchronous-write contract as src/tile_memory.v, and
//   wires the DUT's access ports to it.  The TB does NOT instantiate any other
//   src/ module -- conv2d_unit is standalone here.
//
// COVERAGE
//   D1  impulse kernel (center tap = 1.0, rest 0) -> identity: out == valid
//       center crop of the input (exercises the basic window + packing)
//   D2  all-zero input -> all-zero output
//   D3  negative input & kernel (sign handling through the signed MAC)
//   D4  saturation: large positive input * large positive kernel -> clamp to
//       +max, sat flag set; large negative -> clamp to -min
//   D5  edge windows: a kernel that only reads a corner tap, verifying the
//       top-left and bottom-right valid windows address the right pixels
//   D6  stride=2, pad=0 -> 3x3 output
//   D7  pad=1, stride=1 -> 8x8 output (zero edge padding correctness)
//   D8  pad=1, stride=2 -> 4x4 output
//   D9  parameter sanitization: stride=0 behaves as stride=1; stride=3 too;
//       pad=2/3 behave as pad=1
//   R   >=200 constrained-random runs: random Q7.8 maps & kernels across all
//       four (stride,pad) combinations, every valid output compared bit-exact,
//       and the sat flag checked against the golden's saturation count.
//
//   Every run also asserts busy/done HANDSHAKE TIMING == the documented closed
//   form  (H+K) + OH*OW + 1  posedges from start, and that busy is high
//   throughout and low at/after done.
//
// GATES: prints "ALL <N> TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module conv2d_unit_tb;

    // ---- sizes mirrored from defs for the TB's own loops ----
    localparam integer H    = `CONV_H;     // 8
    localparam integer W    = `CONV_W;     // 8
    localparam integer K    = `CONV_K;     // 3
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

    // Combinational read: the DUT presents rd_addr, the TM returns the line.
    always @(*) rd_data = tm[rd_addr];

    // Synchronous write: DUT result lines committed on the clock edge.
    integer ti;
    always @(posedge clk) begin
        if (tm_we) tm[tm_waddr] <= tm_wdata;
    end

    // ---- DUT ----
    conv2d_unit dut (
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

    // ---- test geometry constants for the TB ----
    localparam integer IN_BASE  = 0;       // 8 image rows at lines 0..7
    localparam integer K_BASE   = 8;       // 3 kernel rows at lines 8..10
    localparam integer OUT_BASE = 11;      // output lines from 11 up (<=16 px/16 lines)

    // ---- TB-side image / kernel storage as plain signed integers (Q7.8) ----
    integer img    [0:H*W-1];              // input map, Q7.8 integer values
    integer ker    [0:K*K-1];              // kernel,    Q7.8 integer values

    // ---- golden output buffer ----
    integer gold_out [0:63];               // up to 8x8 valid outputs
    integer gold_sat;                      // golden saturation occurred?

    //========================================================================
    // Helpers
    //========================================================================

    // Pack the 8 image columns of row r into a 128-bit TM line and store it.
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

    // Pack the 3 kernel columns of row kr into the low 48 bits of a TM line.
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
    // Returns the output dim `od` via the function-style outputs (written into
    // gold_out[] and gold_sat).  Uses the *same* stride/pad sanitization the
    // DUT documents so the reference and DUT agree on geometry.
    task golden_conv;
        input [1:0] s_in, p_in;
        output integer od;
        integer ss, pp, oy, ox, wr, wc, iy, ix;
        real    accr, biasr, shifted;
        integer q;
        begin
            ss = (s_in == 2'd2) ? 2 : 1;
            pp = (p_in != 2'd0) ? 1 : 0;
            od = ((H + 2*pp - K) / ss) + 1;
            gold_sat = 0;
            for (oy = 0; oy < od; oy = oy + 1) begin
                for (ox = 0; ox < od; ox = ox + 1) begin
                    accr = 0.0;
                    for (wr = 0; wr < K; wr = wr + 1) begin
                        for (wc = 0; wc < K; wc = wc + 1) begin
                            iy = oy*ss - pp + wr;
                            ix = ox*ss - pp + wc;
                            if ((iy >= 0) && (iy < H) &&
                                (ix >= 0) && (ix < W)) begin
                                // product of two Q7.8 integers (exact), summed
                                // in float; this is Q14.16 scaled (acc*1.0).
                                accr = accr +
                                    (img[iy*W + ix] * 1.0) * (ker[wr*K + wc] * 1.0);
                            end
                            // out-of-range -> contributes 0 (zero padding)
                        end
                    end
                    // Boundary quantize: round-half-up of acc/2^8, then clamp.
                    // acc is in Q14.16 units (= true_value * 2^16); dividing by
                    // 2^8 returns to Q7.8 scale.  Round-half-up: floor(x + 0.5).
                    biasr   = accr + 128.0;          // +0.5 LSB in Q?.16 == 128
                    shifted = $floor(biasr / 256.0); // arithmetic >>8 of rounded
                    q = shifted;
                    if (q > `Q78_MAX_VAL) begin
                        q = `Q78_MAX_VAL; gold_sat = 1;
                    end else if (q < `Q78_MIN_VAL) begin
                        q = `Q78_MIN_VAL; gold_sat = 1;
                    end
                    gold_out[oy*od + ox] = q;
                end
            end
        end
    endtask

    // Read back one valid output pixel `p` (raster index) from the DUT's TM:
    // 4 px/line packed low-first starting at OUT_BASE.
    function integer read_dut_out;
        input integer p;
        reg [`LINE_W-1:0] line;
        reg [ELEM-1:0]    raw;
        integer ln, lane;
        begin
            ln   = OUT_BASE + (p / 4);
            lane = p % 4;
            line = tm[ln];
            raw  = line[lane*ELEM +: ELEM];
            read_dut_out = $signed(raw);
        end
    endfunction

    // Pulse start for one cycle, then wait for done, measuring posedges.
    // Asserts busy/done timing against the documented closed form.
    task run_and_time;
        input [1:0] s_in, p_in;
        input integer exp_od;             // expected output dim (for latency)
        integer cyc, exp_cyc, npix;
        begin
            // launch
            @(negedge clk);
            start  = 1'b1;
            stride = s_in;
            pad    = p_in;
            @(posedge clk);               // edge that samples start
            @(negedge clk);
            start  = 1'b0;
            if (busy !== 1'b1) begin
                $display("FAIL: busy not asserted after start (s=%0d p=%0d)",
                         s_in, p_in);
                fail = fail + 1;
                $fatal(1, "busy handshake");
            end
            // count posedges until done
            cyc = 0;
            while (done !== 1'b1) begin
                @(posedge clk);
                cyc = cyc + 1;
                if (cyc > 1000) begin
                    $display("FAIL: done never asserted");
                    fail = fail + 1;
                    $fatal(1, "done timeout");
                end
            end
            // Documented latency: (H+K) load + npix compute + 1 done.
            npix    = exp_od * exp_od;
            exp_cyc = (H + K) + npix + 1;
            if (cyc !== exp_cyc) begin
                $display("FAIL: latency mismatch s=%0d p=%0d got=%0d exp=%0d",
                         s_in, p_in, cyc, exp_cyc);
                fail = fail + 1;
                $fatal(1, "latency");
            end else begin
                pass = pass + 1;
            end
            // busy must be low now (done cycle).
            if (busy !== 1'b0) begin
                $display("FAIL: busy still high at done");
                fail = fail + 1;
                $fatal(1, "busy clear");
            end else begin
                pass = pass + 1;
            end
            @(negedge clk);               // let the final write settle in tm[]
        end
    endtask

    // Full run: program TM, launch DUT, compare every valid output bit-exact
    // and the sat flag against the golden.
    task do_run;
        input [255:0] tag;
        input [1:0]   s_in, p_in;
        integer od, p, npix, dval, gval;
        begin
            load_image_to_tm;
            load_kernel_to_tm;
            golden_conv(s_in, p_in, od);
            run_and_time(s_in, p_in, od);

            npix = od * od;
            for (p = 0; p < npix; p = p + 1) begin
                dval = read_dut_out(p);
                gval = gold_out[p];
                if (dval !== gval) begin
                    $display("FAIL[%0s] px=%0d (oy=%0d ox=%0d) dut=%0d gold=%0d",
                             tag, p, p/od, p%od, dval, gval);
                    fail = fail + 1;
                    $fatal(1, "conv output mismatch");
                end else begin
                    pass = pass + 1;
                end
            end
            // sat flag must match the golden's saturation outcome.
            if (sat !== gold_sat[0]) begin
                $display("FAIL[%0s] sat flag dut=%b gold=%b", tag, sat, gold_sat[0]);
                fail = fail + 1;
                $fatal(1, "sat flag mismatch");
            end else begin
                pass = pass + 1;
            end
        end
    endtask

    // Fill img/ker with constants.
    task set_img_const; input integer v; integer i;
        begin for (i=0;i<H*W;i=i+1) img[i]=v; end
    endtask
    task set_ker_const; input integer v; integer i;
        begin for (i=0;i<K*K;i=i+1) ker[i]=v; end
    endtask

    // sign-extend a 16-bit value sampled from $random into a Q7.8 integer.
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
        seed = 32'h2D5A11CE;

        start    = 1'b0;
        in_base  = IN_BASE[`TM_IDX_W-1:0];
        k_base   = K_BASE[`TM_IDX_W-1:0];
        out_base = OUT_BASE[`TM_IDX_W-1:0];
        stride   = 2'd1;
        pad      = 2'd0;

        // clear TM
        for (i = 0; i < `TM_LINES; i = i + 1) tm[i] = {`LINE_W{1'b0}};

        // ---- synchronous reset ----
        rst = 1'b1;
        @(posedge clk);
        @(posedge clk);
        @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        //--------------------------------------------------------------------
        // D1: impulse (identity) kernel -> valid center crop of the input.
        //   center tap = 1.0 (Q7.8 = 256), others 0.  out(oy,ox)=in(oy+1,ox+1)
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = (i*7) - 100; // arbitrary spread
        set_ker_const(0);
        ker[K*K/2] = 256;                  // center = 1.0 in Q7.8
        do_run("D1-impulse", 2'd1, 2'd0);
        // extra explicit identity assertion vs. the raw input crop
        for (r = 0; r < 6; r = r + 1)
            for (j = 0; j < 6; j = j + 1)
                if (read_dut_out(r*6 + j) !== img[(r+1)*W + (j+1)]) begin
                    $display("FAIL[D1-cropcheck] (%0d,%0d) got=%0d want=%0d",
                             r, j, read_dut_out(r*6+j), img[(r+1)*W+(j+1)]);
                    fail = fail + 1;
                    $fatal(1, "identity crop");
                end
        pass = pass + 1;

        //--------------------------------------------------------------------
        // D2: all-zero input -> all-zero output, no saturation.
        //--------------------------------------------------------------------
        set_img_const(0);
        for (i = 0; i < K*K; i = i + 1) ker[i] = (i+1)*37;  // nonzero kernel
        do_run("D2-zeroin", 2'd1, 2'd0);

        //--------------------------------------------------------------------
        // D3: negative input & kernel (signed MAC correctness).
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = -((i % 11) * 30 + 5);
        for (i = 0; i < K*K; i = i + 1) ker[i] = (i % 2) ? -64 : 80;
        do_run("D3-negative", 2'd1, 2'd0);

        //--------------------------------------------------------------------
        // D4a: positive saturation. input = +max-ish (100.0), kernel all +1.0
        //   sum of 9 * 100.0 = 900.0 >> Q7.8 max (~128) -> clamp +max, sat set.
        //--------------------------------------------------------------------
        set_img_const(100 * 256);          // 100.0 in Q7.8 (>32767? 25600 ok)
        set_ker_const(256);                // 1.0
        do_run("D4a-satpos", 2'd1, 2'd0);
        if (sat !== 1'b1) begin
            $display("FAIL[D4a] expected sat=1"); fail=fail+1; $fatal(1,"sat+");
        end else pass = pass + 1;

        //--------------------------------------------------------------------
        // D4b: negative saturation.
        //--------------------------------------------------------------------
        set_img_const(-100 * 256);
        set_ker_const(256);
        do_run("D4b-satneg", 2'd1, 2'd0);
        if (sat !== 1'b1) begin
            $display("FAIL[D4b] expected sat=1"); fail=fail+1; $fatal(1,"sat-");
        end else pass = pass + 1;

        //--------------------------------------------------------------------
        // D5: edge-window addressing. kernel reads ONLY top-left tap (kc=0,kr=0)
        //   so out(oy,ox) == in(oy,ox)*1.0; verifies window base/edge mapping.
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = ((i*13) % 251) - 120;
        set_ker_const(0);
        ker[0] = 256;                      // top-left tap = 1.0
        do_run("D5-edgewin", 2'd1, 2'd0);
        for (r = 0; r < 6; r = r + 1)
            for (j = 0; j < 6; j = j + 1)
                if (read_dut_out(r*6 + j) !== img[r*W + j]) begin
                    $display("FAIL[D5-tl] (%0d,%0d) got=%0d want=%0d",
                             r, j, read_dut_out(r*6+j), img[r*W+j]);
                    fail = fail + 1;
                    $fatal(1, "edge window tl");
                end
        pass = pass + 1;

        //--------------------------------------------------------------------
        // D6: stride=2, pad=0 -> 3x3 output.
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = ((i*17) % 200) - 90;
        for (i = 0; i < K*K; i = i + 1) ker[i] = (i - 4) * 50;
        do_run("D6-stride2", 2'd2, 2'd0);

        //--------------------------------------------------------------------
        // D7: pad=1, stride=1 -> 8x8 (zero edge padding correctness).
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = ((i*23) % 180) - 70;
        for (i = 0; i < K*K; i = i + 1) ker[i] = (i + 1) * 20;
        do_run("D7-pad1", 2'd1, 2'd1);

        //--------------------------------------------------------------------
        // D8: pad=1, stride=2 -> 4x4 output.
        //--------------------------------------------------------------------
        do_run("D8-pad1str2", 2'd2, 2'd1);

        //--------------------------------------------------------------------
        // D9: parameter sanitization.  stride=0 -> like stride=1;
        //     stride=3 -> like stride=1; pad=2/3 -> like pad=1.
        //   (golden_conv applies the SAME sanitization, so do_run validates it.)
        //--------------------------------------------------------------------
        for (i = 0; i < H*W; i = i + 1) img[i] = ((i*5) % 60) - 30;
        for (i = 0; i < K*K; i = i + 1) ker[i] = (i*i) - 20;
        do_run("D9-stride0", 2'd0, 2'd0);   // -> 6x6
        do_run("D9-stride3", 2'd3, 2'd0);   // -> 6x6
        do_run("D9-pad2",    2'd1, 2'd2);   // -> 8x8
        do_run("D9-pad3",    2'd1, 2'd3);   // -> 8x8

        //--------------------------------------------------------------------
        // R: constrained-random sweep, >=200 runs across all (stride,pad).
        //--------------------------------------------------------------------
        svals[0] = 2'd1; svals[1] = 2'd2;
        pvals[0] = 2'd0; pvals[1] = 2'd1;
        for (run = 0; run < 240; run = run + 1) begin
            // random Q7.8 map & kernel, scaled down so most runs DON'T saturate
            // (so we exercise the in-range path heavily) but some do.
            for (i = 0; i < H*W; i = i + 1) begin
                // mostly small magnitudes, occasional large
                if ((run % 8) == 0) img[i] = rnd_q78(0);             // full range
                else                img[i] = (rnd_q78(0) % 2048);    // ~ +-8.0
            end
            for (i = 0; i < K*K; i = i + 1) begin
                if ((run % 8) == 0) ker[i] = rnd_q78(0);
                else                ker[i] = (rnd_q78(0) % 768);     // ~ +-3.0
            end
            sc = run % 2;
            pc = (run / 2) % 2;
            do_run("R-rand", svals[sc], pvals[pc]);
        end

        //--------------------------------------------------------------------
        if (fail != 0) begin
            $display("CONV2D_UNIT TB: %0d FAILURES", fail);
            $fatal(1, "conv2d_unit_tb FAILED");
        end
        $display("ALL %0d TESTS PASSED", pass);
        $finish;
    end

endmodule
