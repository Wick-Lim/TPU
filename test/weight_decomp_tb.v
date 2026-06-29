`timescale 1ns/1ps
//============================================================================
// weight_decomp_tb.v  --  ROUND-TRIP TB for the FP8 weight decompressor
//                                          (IMPROVEMENT_PLAN.md P2.1, §P2)
//----------------------------------------------------------------------------
// WHAT THIS PROVES
//   1. LOSSLESS: for every block (representative trained-weight FP8 streams +
//      edge cases) the bytes that come OUT of weight_decomp are BIT-EXACT the
//      FP8 bytes the offline compressor started from.  $fatal on any mismatch,
//      any X, or any length mismatch.
//   2. FP8 STREAM IS THE REPO ENCODE: the golden the DUT output is checked
//      against is recomputed HERE by the repo's own quantizer fp32_to_fp8e4m3
//      (fp8_e4m3.vh) run on the raw fp32 weight samples.  The compressed image
//      was built (offline, tools/fp8_gen.py) by a python MIRROR of that encode;
//      because the DUT output is checked against the RTL encode, any divergence
//      between the mirror and the hardware encode is caught here as a $fatal.
//   3. RATIO: measures + prints the real compression ratio (original FP8 bytes
//      / compressed bytes) on the representative distribution -- the effective
//      Flash-BW / energy multiplier.
//
// FLOW (one prep + one sim, no $system):
//   prep:  python3 tools/fp8_gen.py gen scratchpad/wd_vec.txt
//   sim :  per case read {rep,nsamp,ncodes,ncomp}, the canonical count table,
//          the symbol order, the compressed bytes, and the fp32 stim; load the
//          table into the DUT, stream the compressed bytes, capture the decoded
//          FP8, and compare to fp32_to_fp8e4m3(fp32 stim).
//
//   Back-pressure on BOTH ports is exercised (input in_valid bubbles + output
//   out_ready stalls) on a subset of cases via LFSR-driven gating.
//============================================================================
module weight_decomp_tb;

    `include "fp8_e4m3.vh"

    // ---- DUT params (match weight_decomp defaults) ----
    localparam integer MAXLEN  = 15;
    localparam integer SYMW    = 9;
    localparam integer COUNTW  = 10;
    localparam integer AW      = 9;
    localparam integer BUFW    = 32;
    localparam integer EOB_SYM = 256;

    // ---- clock / reset ----
    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    // ---- table-load port ----
    reg              tbl_we    = 1'b0;
    reg              tbl_sel   = 1'b0;
    reg [AW-1:0]     tbl_addr  = {AW{1'b0}};
    reg [COUNTW-1:0] tbl_wdata = {COUNTW{1'b0}};

    // ---- control / data ports ----
    reg              start  = 1'b0;
    reg  [7:0]       in_byte;
    reg              in_valid;
    wire             in_ready;
    wire [7:0]       out_byte;
    wire             out_valid;
    reg              out_ready;
    wire             eob;

    // ---- DUT ----
    weight_decomp #(
        .MAXLEN(MAXLEN), .SYMW(SYMW), .COUNTW(COUNTW),
        .AW(AW), .BUFW(BUFW), .EOB_SYM(EOB_SYM)
    ) dut (
        .clk(clk), .rst(rst),
        .tbl_we(tbl_we), .tbl_sel(tbl_sel), .tbl_addr(tbl_addr), .tbl_wdata(tbl_wdata),
        .start(start),
        .in_byte(in_byte), .in_valid(in_valid), .in_ready(in_ready),
        .out_byte(out_byte), .out_valid(out_valid), .out_ready(out_ready), .eob(eob)
    );

    // ---- storage ----
    reg  [7:0]  orig_fp8   [0:8191];     // golden FP8 (repo encode of fp32 stim)
    reg  [7:0]  comp_bytes [0:16383];    // compressed stream for the current block
    reg  [7:0]  got        [0:8191];     // bytes captured from the DUT
    integer     cnts       [0:15];       // canonical per-length counts (idx 1..15)
    reg  [SYMW-1:0] order_arr [0:511];   // canonical symbol order

    // ---- producer / consumer index state ----
    integer pi;        // producer index into comp_bytes
    integer gi;        // consumer index into got
    integer ncomp;     // # compressed bytes to feed this block
    reg     run;       // streaming active
    reg     prep;      // hold producer/consumer indices at 0

    // ---- back-pressure LFSR + per-case modes ----
    reg [7:0] lfsr = 8'hA5;
    reg       in_stall_mode;   // gate in_valid with lfsr
    reg       out_stall_mode;  // gate out_ready with lfsr
    always @(posedge clk) lfsr <= {lfsr[6:0], lfsr[7]^lfsr[5]^lfsr[4]^lfsr[3]};

    // ---- producer: present comp_bytes[pi] while pi<ncomp; advance on accept ----
    wire prod_active = run && (pi < ncomp);
    always @* begin
        in_byte  = comp_bytes[pi];
        in_valid = prod_active && (in_stall_mode ? lfsr[1] : 1'b1);
    end
    always @(posedge clk) begin
        if (prep)                       pi <= 0;
        else if (in_valid && in_ready)  pi <= pi + 1;
    end

    // ---- consumer: capture out_byte on accept ----
    always @* out_ready = (!run) ? 1'b1 : (out_stall_mode ? lfsr[0] : 1'b1);
    wire cons_take = out_valid && out_ready;
    always @(posedge clk) begin
        if (prep) gi <= 0;
        else if (cons_take) begin
            if (^out_byte === 1'bx) begin
                $display("FATAL: X on out_byte at gi=%0d", gi);
                $fatal;
            end
            got[gi] <= out_byte;
            gi <= gi + 1;
        end
    end

    // ---- watchdog ----
    initial begin
        #80_000_000;
        $display("FATAL: simulation TIMEOUT (decode hung)");
        $fatal;
    end

    // ---- main flow ----
    integer fd;
    integer ncases, c, i, rep;
    integer nsamp, ncodes;
    reg [31:0] fp32word;
    integer passes;
    integer rep_o, rep_c, tot_o, tot_c;
    real    ratio;
    integer rc;

    initial begin
        in_byte = 8'h00; in_valid = 1'b0; out_ready = 1'b1;
        run = 1'b0; prep = 1'b1;
        in_stall_mode = 1'b0; out_stall_mode = 1'b0;

        // global reset
        rst = 1'b1;
        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        fd = $fopen("scratchpad/wd_vec.txt", "r");
        if (fd == 0) begin
            $display("FATAL: cannot open scratchpad/wd_vec.txt -- run:");
            $display("       python3 tools/fp8_gen.py gen scratchpad/wd_vec.txt");
            $fatal;
        end
        rc = $fscanf(fd, "%d", ncases);

        passes = 0;
        rep_o = 0; rep_c = 0; tot_o = 0; tot_c = 0;

        for (c = 0; c < ncases; c = c + 1) begin
            // ---- 1) read the per-case vector ----
            rc = $fscanf(fd, "%d", rep);
            rc = $fscanf(fd, "%d", nsamp);
            rc = $fscanf(fd, "%d", ncodes);
            rc = $fscanf(fd, "%d", ncomp);
            for (i = 1; i <= 15; i = i + 1)      rc = $fscanf(fd, "%d", cnts[i]);
            for (i = 0; i < ncodes; i = i + 1)   rc = $fscanf(fd, "%d", order_arr[i]);
            for (i = 0; i < ncomp;  i = i + 1)   rc = $fscanf(fd, "%d", comp_bytes[i]);
            // fp32 stim -> golden FP8 via the REPO encode (fp8_e4m3.vh)
            for (i = 0; i < nsamp;  i = i + 1) begin
                rc = $fscanf(fd, "%d", fp32word);
                orig_fp8[i] = fp32_to_fp8e4m3(fp32word);
            end

            // no pathological expansion: at most ~9 bits/symbol + EOB + pad
            if (ncomp > (nsamp*9 + 7)/8 + 4) begin
                $display("FATAL: case %0d pathological expansion: %0d -> %0d bytes",
                         c, nsamp, ncomp);
                $fatal;
            end

            // ---- 2) per-case back-pressure modes ----
            in_stall_mode  = c[0];     // bubble the input on odd cases
            out_stall_mode = c[1];     // stall the output on cases 2,3,6,7

            // ---- 3) start the block + load the table (held in prep: pi=gi=0) ----
            // NOTE: all DUT-facing signals are driven on negedge so they are
            // stable across the sampling posedge (avoids a TB-side race where a
            // posedge-driven `start` would be cleared in the same delta as the
            // clock edge and missed by the DUT's posedge logic).
            prep = 1'b1; run = 1'b0;
            @(negedge clk); start = 1'b1;       // start spans one full posedge
            @(negedge clk); start = 1'b0;       // ... then deassert at negedge

            tbl_we = 1'b1;
            tbl_sel = 1'b0;
            tbl_addr = 0; tbl_wdata = 0; @(negedge clk);          // count_table[0]=0
            for (i = 1; i <= 15; i = i + 1) begin
                tbl_addr  = i[AW-1:0];
                tbl_wdata = cnts[i][COUNTW-1:0];
                @(negedge clk);
            end
            tbl_sel = 1'b1;
            for (i = 0; i < ncodes; i = i + 1) begin
                tbl_addr  = i[AW-1:0];
                tbl_wdata = {{(COUNTW-SYMW){1'b0}}, order_arr[i]};
                @(negedge clk);
            end
            tbl_we = 1'b0;
            @(negedge clk);

            // ---- 4) stream it ----
            prep = 1'b0;
            run  = 1'b1;
            wait (eob && !out_valid);
            @(negedge clk);
            run = 1'b0;
            @(negedge clk);

            // ---- 5) check lossless + length ----
            if (gi != nsamp) begin
                $display("FATAL: case %0d emitted %0d bytes, expected %0d", c, gi, nsamp);
                $fatal;
            end
            for (i = 0; i < nsamp; i = i + 1) begin
                if (got[i] !== orig_fp8[i]) begin
                    $display("FATAL: case %0d MISMATCH at byte %0d: got %02h exp %02h",
                             c, i, got[i], orig_fp8[i]);
                    $fatal;
                end
            end

            // ---- 6) accumulate + report ratio ----
            ratio = (ncomp != 0) ? (nsamp * 1.0) / ncomp : 0.0;
            $display("PASS case %0d (%0s): %5d FP8 bytes -> %5d comp bytes  ratio %0.2fx  (%0.2f bits/sym)%0s",
                     c, (rep != 0) ? "trained-weight" : "edge/worst",
                     nsamp, ncomp, ratio, (8.0*ncomp)/nsamp,
                     (out_stall_mode||in_stall_mode) ? "  [back-pressure]" : "");
            tot_o = tot_o + nsamp; tot_c = tot_c + ncomp;
            if (rep != 0) begin rep_o = rep_o + nsamp; rep_c = rep_c + ncomp; end
            passes = passes + 1;
        end
        $fclose(fd);

        $display("----------------------------------------------------------------");
        $display("REPRESENTATIVE FP8 weight stream: %0d -> %0d bytes  RATIO %0.2fx (%0.2f bits/sym)",
                 rep_o, rep_c, (rep_o*1.0)/rep_c, (8.0*rep_c)/rep_o);
        $display("ALL CASES (incl. edge/worst):    %0d -> %0d bytes  RATIO %0.2fx",
                 tot_o, tot_c, (tot_o*1.0)/tot_c);
        $display("----------------------------------------------------------------");
        $display("ALL %0d TESTS PASSED", passes);
        $finish;
    end

endmodule
