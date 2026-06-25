`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// dma_controller_tb.v  --  unit testbench for dma_controller   (SPEC §2, §5.5)
//----------------------------------------------------------------------------
// WHAT IT CHECKS
//   * Functional correctness of the multi-word DMEM block copy
//        DMEM[dst+i] = DMEM[src+i], i = 0..len-1
//     against an INDEPENDENT golden model.  The golden is computed a DIFFERENT
//     way than the DUT: the TB snapshots the ENTIRE pre-copy DMEM into a
//     separate array `gold`, then performs the ascending copy on that array
//     with a plain software loop (no handshake, no pipeline, no DUT signals),
//     and compares the DUT's post-copy DMEM word-by-word against `gold`.  The
//     reference therefore shares no logic with the DUT's FSM/pipeline.
//   * Exact (bit-exact) comparison -- this is an integer block move, no
//     tolerance.  $fatal on ANY mismatch.
//   * Handshake/latency: busy asserts during the transfer; done is a 1-cycle
//     pulse; the start->done latency equals the documented formula
//        len == 0 : 1 cycle      (no-op acknowledge)
//        len >= 1 : len + 1      (1-deep read->write pipeline, start cycle
//                                 overlaps the first read)
//     The DUT spec target is "len(+2)"; the measured exact value is len+1 for
//     len>=1 (see caveat in the report) and the TB asserts that exact value.
//   * `words` output == latched len (for rC write-back).
//   * No spurious DMEM writes for len==0.
//
// SELF-CONTAINED: the TB models DMEM itself (a behavioral array with
//   combinational read on the DUT's rd_addr and synchronous write on the DUT's
//   wr_en/wr_addr/wr_data) and drives the DUT's ports.  It instantiates NO other
//   src/ module.
//
// CORNER CASES: zero-length (no-op), len=1, full small run, max-address run,
//   forward-overlap (dst>src), back-overlap (dst<src), src==dst aliasing,
//   adjacent/disjoint regions, plus >=200 constrained-random descriptors with a
//   seeded $random.
//============================================================================
module dma_controller_tb;

    // ---------------- parameters / sizes ----------------------------------
    localparam integer AW    = `DMEM_ADDR_W;   // 8
    localparam integer DEPTH = `DMEM_DEPTH;    // 256
    localparam integer XW    = `XLEN;          // 32

    // ---------------- clock / reset ---------------------------------------
    reg clk = 1'b0;
    reg rst;
    always #5 clk = ~clk;                       // 100 MHz

    // ---------------- DUT descriptor / handshake --------------------------
    reg               start;
    reg  [AW-1:0]     src_addr;
    reg  [AW-1:0]     dst_addr;
    reg  [AW:0]       len;
    wire              busy;
    wire              done;
    wire [AW:0]       words;

    // ---------------- DUT <-> DMEM access ports ---------------------------
    wire [AW-1:0]     rd_addr;
    wire [XW-1:0]     rd_data;     // combinational read of the TB DMEM model
    wire              wr_en;
    wire [AW-1:0]     wr_addr;
    wire [XW-1:0]     wr_data;

    // ---------------- behavioral DMEM model (lives in the TB) -------------
    reg  [XW-1:0]     dmem [0:DEPTH-1];
    // golden result array -- the INDEPENDENT reference
    reg  [XW-1:0]     gold [0:DEPTH-1];

    integer i;
    integer errors;
    integer tests;

    // Combinational read port the DUT drives (mirrors a real comb-read DMEM).
    // Continuous assign on a memory word -> reacts to rd_addr and to writes.
    assign rd_data = dmem[rd_addr];

    // Synchronous write port the DUT drives.
    always @(posedge clk) begin
        if (wr_en) dmem[wr_addr] <= wr_data;
    end

    // ---------------- DUT -------------------------------------------------
    dma_controller dut (
        .clk      (clk),
        .rst      (rst),
        .start    (start),
        .src_addr (src_addr),
        .dst_addr (dst_addr),
        .len      (len),
        .busy     (busy),
        .done     (done),
        .words    (words),
        .rd_addr  (rd_addr),
        .rd_data  (rd_data),
        .wr_en    (wr_en),
        .wr_addr  (wr_addr),
        .wr_data  (wr_data)
    );

    // ---------------- helpers ---------------------------------------------
    // Fill DMEM with a deterministic, distinct pattern so a wrong/missing/extra
    // copy is always visible.
    task seed_dmem;
        integer k;
        begin
            for (k = 0; k < DEPTH; k = k + 1)
                dmem[k] = (32'hA5A5_0000 ^ (k * 32'h0001_0193)) + k[31:0];
        end
    endtask

    // Compute the independent golden.  The DUT performs the spec's literal C
    // loop, reading LIVE memory each iteration:
    //      for (i=0;i<n;i++) DMEM[dst+i] = DMEM[src+i];
    // so for overlapping forward regions (dst just above src) an earlier write
    // is visible to a later read (memmove-ascending, NOT memcpy).  The golden
    // reproduces that SAME in-place ascending semantics on its own array `gold`
    // (which it seeds from the current dmem snapshot) using a plain SW loop --
    // it shares no logic with the DUT's FSM/pipeline, only the documented
    // ordering.  Reading and writing the same array models the live read.
    task compute_golden;
        input [AW-1:0] s;
        input [AW-1:0] d;
        input [AW:0]   n;
        integer k;
        begin
            for (k = 0; k < DEPTH; k = k + 1)
                gold[k] = dmem[k];
            // in-place ascending copy (live read of the same array)
            for (k = 0; k < n; k = k + 1) begin
                // address may wrap mod DEPTH; AW-bit add models that exactly.
                gold[(d + k[AW-1:0]) % DEPTH] = gold[(s + k[AW-1:0]) % DEPTH];
            end
        end
    endtask

    // Drive one transfer and check function + handshake/latency.
    //
    // LATENCY CONVENTION: `lat` = number of clock posedges from the posedge that
    // SAMPLES the start pulse up to and including the posedge at which `done`
    // first goes high (i.e. how many cycles after start the done pulse lands).
    // Measured exact values (see header / report caveats):
    //      len == 0 : lat == 0   (done set the same cycle start is sampled,
    //                             observed in the cycle right after -> a 1-cycle
    //                             no-op acknowledge; busy never asserts)
    //      len >= 1 : lat == len (one word committed per cycle; the write of the
    //                             final word and `done` coincide)
    // We sample DUT registered outputs at #1 AFTER each posedge so the values
    // reflect that edge's update.
    task run_copy;
        input [AW-1:0]    s;
        input [AW-1:0]    d;
        input [AW:0]      n;
        input [8*40-1:0]  name;            // label for messages
        integer           cyc;
        integer           lat;
        integer           done_cnt;
        integer           exp_lat;
        reg               saw_busy;
        begin
            tests = tests + 1;

            // Golden computed from CURRENT dmem contents (before any DUT write).
            compute_golden(s, d, n);

            // Present descriptor and assert start, aligned to the posedge that
            // samples it.  Drive on negedge so values are stable at the posedge.
            @(negedge clk);
            src_addr = s;
            dst_addr = d;
            len      = n;
            start    = 1'b1;
            @(posedge clk);          // <-- this posedge SAMPLES start (cyc 0)
            #1 start = 1'b0;         // 1-cycle start pulse; drop right after edge

            // Now scan subsequent posedges, counting until done is high.
            cyc      = 0;
            lat      = -1;
            done_cnt = 0;
            saw_busy = 1'b0;
            if (busy) saw_busy = 1'b1;    // sample busy set by the start posedge
            if (done) begin done_cnt = 1; lat = 0; end  // len==0 acknowledge
            while (done_cnt == 0 && cyc < (DEPTH + 16)) begin
                cyc = cyc + 1;
                @(posedge clk);
                #1;
                if (busy) saw_busy = 1'b1;
                if (done) begin
                    done_cnt = done_cnt + 1;
                    lat      = cyc;
                end
            end

            // ---- handshake / latency checks ----
            if (done_cnt != 1) begin
                $display("FAIL[%0s]: expected exactly 1 done pulse, saw %0d (src=%0d dst=%0d len=%0d)",
                         name, done_cnt, s, d, n);
                errors = errors + 1;
            end

            // advance one more cycle; done must be a 1-cycle pulse, busy clear
            @(posedge clk); #1;
            if (done) begin
                $display("FAIL[%0s]: done not a 1-cycle pulse (still high)", name);
                errors = errors + 1;
            end
            if (busy) begin
                $display("FAIL[%0s]: busy still high after done", name);
                errors = errors + 1;
            end

            // expected latency formula (measured/exact)
            exp_lat = (n == 0) ? 0 : n;
            if (lat != exp_lat) begin
                $display("FAIL[%0s]: latency=%0d expected=%0d (src=%0d dst=%0d len=%0d)",
                         name, lat, exp_lat, s, d, n);
                errors = errors + 1;
            end

            // busy must be high for >=1 cycle whenever len>=1, and never for len==0.
            if (n == 0 && saw_busy) begin
                $display("FAIL[%0s]: busy asserted for len==0 no-op", name);
                errors = errors + 1;
            end
            if (n >= 1 && !saw_busy) begin
                $display("FAIL[%0s]: busy never asserted for len=%0d", name, n);
                errors = errors + 1;
            end

            // ---- result word check (#words) ----
            if (words !== n) begin
                $display("FAIL[%0s]: words=%0d expected=%0d", name, words, n);
                errors = errors + 1;
            end

            // ---- functional check: full DMEM vs golden ----
            for (i = 0; i < DEPTH; i = i + 1) begin
                if (dmem[i] !== gold[i]) begin
                    $display("FAIL[%0s]: DMEM[%0d]=%h expected %h (src=%0d dst=%0d len=%0d)",
                             name, i, dmem[i], gold[i], s, d, n);
                    errors = errors + 1;
                end
            end
        end
    endtask

    // ---------------- stimulus --------------------------------------------
    integer t;
    reg [AW-1:0] rs, rd;
    reg [AW:0]   rl;
    integer      maxlen;

    initial begin
        errors = 0;
        tests  = 0;
        start  = 1'b0;
        src_addr = {AW{1'b0}};
        dst_addr = {AW{1'b0}};
        len      = {(AW+1){1'b0}};

        // synchronous reset
        rst = 1'b1;
        seed_dmem();
        repeat (3) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ----- directed corner cases -----
        // len == 0 : no-op (must not touch DMEM, done in 1 cycle, no busy)
        seed_dmem(); run_copy(8'd10, 8'd40, 9'd0,  "len0_noop");

        // len == 1 : single word
        seed_dmem(); run_copy(8'd5,  8'd20, 9'd1,  "len1_single");

        // small disjoint run
        seed_dmem(); run_copy(8'd0,  8'd64, 9'd16, "disjoint16");

        // a longer run
        seed_dmem(); run_copy(8'd0,  8'd128,9'd64, "run64");

        // max-address tail run (src/dst end at the last word)
        seed_dmem(); run_copy(8'd248,8'd120,9'd8,  "src_tail8");
        seed_dmem(); run_copy(8'd60, 8'd248,9'd8,  "dst_tail8");

        // src == dst aliasing (copy region onto itself -> DMEM unchanged)
        seed_dmem(); run_copy(8'd33, 8'd33, 9'd20, "alias_same");

        // forward overlap: dst just above src (dst > src, regions overlap).
        // ascending copy with dst>src would smear in a memmove sense; the
        // golden models the SAME ascending order so the comparison is exact
        // and pins the DUT's documented ordering.
        seed_dmem(); run_copy(8'd30, 8'd33, 9'd12, "fwd_overlap");

        // backward overlap: dst below src (dst < src), regions overlap.
        seed_dmem(); run_copy(8'd40, 8'd36, 9'd12, "bwd_overlap");

        // adjacent regions (dst = src + len, no overlap)
        seed_dmem(); run_copy(8'd10, 8'd22, 9'd12, "adjacent");

        // full-depth copy is not safe (dst would wrap onto src); use a big but
        // bounded run from low to high half.
        seed_dmem(); run_copy(8'd0,  8'd100,9'd100,"big100");

        // ----- two back-to-back transfers (engine returns to IDLE cleanly) --
        seed_dmem();
        run_copy(8'd0, 8'd50, 9'd10, "b2b_a");
        run_copy(8'd60,8'd200,9'd10, "b2b_b");

        // ----- constrained-random: >=200 vectors, seeded -----
        // Each picks src,dst,len so that BOTH [src,src+len) and [dst,dst+len)
        // stay within [0,DEPTH) (no wrap) -- the common DMA case.  The golden
        // already handles wrap, but bounding keeps regions interpretable.
        for (t = 0; t < 230; t = t + 1) begin
            // length 0..32
            rl = $random;            // wide
            rl = rl % 33;            // 0..32
            maxlen = DEPTH - rl;     // ensures src,dst+len <= DEPTH
            if (maxlen < 1) maxlen = 1;
            rs = $random % maxlen;
            rd = $random % maxlen;
            if (rs[AW-1] === 1'bx) rs = 0;  // guard against X on first iters
            if (rd[AW-1] === 1'bx) rd = 0;
            seed_dmem();
            run_copy(rs, rd, rl, "rand");
        end

        // ----- final tally -----
        if (errors == 0) begin
            $display("ALL %0d TESTS PASSED", tests);
            $finish;
        end else begin
            $display("FAILED: %0d error(s) across %0d tests", errors, tests);
            $fatal(1, "dma_controller_tb: %0d mismatch(es)", errors);
        end
    end

    // global safety timeout
    initial begin
        #5_000_000;
        $display("FAIL: global timeout");
        $fatal(1, "dma_controller_tb: timeout");
    end

endmodule
