`timescale 1ns/1ps
//============================================================================
// cdc_async_fifo_tb.v  --  unit testbench for cdc_async_fifo.v
//----------------------------------------------------------------------------
// SELF-CONTAINED: instantiates ONLY the DUT (cdc_async_fifo).
//
// METHOD
//   The write side and the read side run as TWO INDEPENDENT processes driven by
//   TWO ASYNCHRONOUS clocks (wclk = 7ns period, rclk = 11ns period; their ratio
//   makes the edges drift across one another throughout the run, exercising the
//   CDC synchronizers under arbitrary phase relationships).
//
//   A long, known, deterministic sequence of NWORDS (>=200) words is produced by
//   gen_word(i).  The WRITER pushes word i whenever the FIFO is not full; the
//   READER pops whenever the FIFO is not empty.
//
//   INDEPENDENT GOLDEN MODEL: the TB keeps its OWN linear software record
//   `gbuf[]` of every word the writer ACCEPTS, indexed by producer order.  It
//   NEVER inspects any DUT internal pointer or memory.  On each ACCEPTED read
//   the dequeued (expected) value gbuf[rd_count] must BIT-EXACTLY equal the
//   DUT's rd_data.  Because a FIFO is strictly order-preserving, this single
//   in-order comparison catches any loss, duplication, reordering, or
//   corruption.
//
//   READ TIMING CONTRACT (matches the DUT's registered-output RAM with
//   registered full/empty): on the rclk posedge where (rd_en & ~empty) holds,
//   the head word is consumed and appears on rd_data immediately after that
//   edge.  The reader drives rd_en combinationally from `empty` in the negedge
//   half, then -- one #1 after the posedge -- samples rd_data and compares it to
//   the golden, advancing rd_count.  (Verified by waveform during bring-up.)
//
//   SAFETY ASSERTIONS (continuously checked):
//     * the writer NEVER drives wr_en while `full`   (no overflow);
//     * the reader NEVER drives rd_en while `empty`  (no underflow).
//   These hold by construction in the drivers AND are independently re-checked
//   by always-blocks on each clock.
//
//   COVERAGE includes a directed BURST: hold the reader off and fill the FIFO
//   until `full` (DEPTH words resident per the golden occupancy), then hold the
//   writer off and drain until `empty`, confirming the full<->empty boundary --
//   before returning to free-running concurrent asynchronous streaming of the
//   remaining words.
//
//   Counts checks, prints "ALL <N> TESTS PASSED", $fatal on any mismatch.
//============================================================================
module cdc_async_fifo_tb;

    localparam integer DATA_W = 32;
    localparam integer ADDR_W = 4;            // depth 16
    localparam integer DEPTH  = (1 << ADDR_W);
    localparam integer NWORDS = 300;          // >=200 streamed words

    // ---- DUT write-domain I/O ----
    reg                   wclk;
    reg                   wrst_n;
    reg                   wr_en;
    reg  [DATA_W-1:0]     wr_data;
    wire                  full;

    // ---- DUT read-domain I/O ----
    reg                   rclk;
    reg                   rrst_n;
    reg                   rd_en;
    wire [DATA_W-1:0]     rd_data;
    wire                  empty;

    // ---- bookkeeping ----
    integer test_count;
    integer fail_count;

    // ---- INDEPENDENT golden record (linear; never peeks DUT internals) ----
    reg [DATA_W-1:0] gbuf [0:NWORDS+DEPTH];
    integer wr_count;   // words ACCEPTED by the writer so far (producer index)
    integer rd_count;   // words ACCEPTED/captured by the reader so far

    // deterministic known sequence (no relation to DUT internals)
    function [DATA_W-1:0] gen_word;
        input integer i;
        begin
            gen_word = (32'h9E37_79B9 * (i + 1)) ^ {i[15:0], ~i[15:0]};
        end
    endfunction

    // ---- DUT ----
    cdc_async_fifo #(
        .DATA_W (DATA_W),
        .ADDR_W (ADDR_W)
    ) dut (
        .wclk    (wclk),
        .wrst_n  (wrst_n),
        .wr_en   (wr_en),
        .wr_data (wr_data),
        .full    (full),
        .rclk    (rclk),
        .rrst_n  (rrst_n),
        .rd_en   (rd_en),
        .rd_data (rd_data),
        .empty   (empty)
    );

    // ---- asynchronous clocks: 7ns vs 11ns ----
    initial wclk = 1'b0;
    always #3.5 wclk = ~wclk;   // 7ns period
    initial rclk = 1'b0;
    always #5.5 rclk = ~rclk;   // 11ns period

    // ---- phase controls for writer / reader ----
    reg run_writer;     // master enable for writer
    reg run_reader;     // master enable for reader
    reg writer_paused;  // when 1, writer holds wr_en low
    reg reader_paused;  // when 1, reader holds rd_en low

    // ==================================================================
    // SAFETY: never drive wr_en@full / rd_en@empty (independent re-check)
    // ==================================================================
    always @(posedge wclk) begin
        if (wrst_n && wr_en && full) begin
            fail_count = fail_count + 1;
            $display("FAIL: wr_en asserted while FULL @t=%0t", $time);
        end
    end
    always @(posedge rclk) begin
        if (rrst_n && rd_en && empty) begin
            fail_count = fail_count + 1;
            $display("FAIL: rd_en asserted while EMPTY @t=%0t", $time);
        end
    end

    // ==================================================================
    // WRITER (write clock domain)
    //   Drive wr_en/wr_data in the negedge half so they are stable around the
    //   capturing posedge.  Account an ACCEPTED write -- and record it into the
    //   golden -- on the posedge where (wr_en & ~full) holds.
    // ==================================================================
    initial begin
        wr_en   = 1'b0;
        wr_data = {DATA_W{1'b0}};
    end
    always @(negedge wclk) begin
        if (!wrst_n) begin
            wr_en   = 1'b0;
            wr_data = {DATA_W{1'b0}};
        end else if (run_writer && !writer_paused && (wr_count < NWORDS) && !full) begin
            wr_en   = 1'b1;
            wr_data = gen_word(wr_count);
        end else begin
            wr_en   = 1'b0;
            wr_data = {DATA_W{1'b0}};
        end
    end
    always @(posedge wclk) begin
        if (wrst_n && wr_en && !full) begin
            // independent golden record, in producer order
            gbuf[wr_count] = gen_word(wr_count);
            wr_count       = wr_count + 1;
        end
    end

    // ==================================================================
    // READER (read clock domain)
    //   Drive rd_en in the negedge half from `empty`.  On the posedge where
    //   (rd_en & ~empty) holds, the head word is now on rd_data: capture and
    //   compare it against the golden in order, then advance rd_count.
    // ==================================================================
    initial begin
        rd_en = 1'b0;
    end
    always @(negedge rclk) begin
        if (!rrst_n) begin
            rd_en = 1'b0;
        end else if (run_reader && !reader_paused && !empty) begin
            rd_en = 1'b1;
        end else begin
            rd_en = 1'b0;
        end
    end
    always @(posedge rclk) begin
        if (rrst_n && rd_en && !empty) begin
            #1;  // let the registered rd_data settle after the edge
            test_count = test_count + 1;
            if (rd_data !== gbuf[rd_count]) begin
                fail_count = fail_count + 1;
                $display("FAIL: read[%0d] got=%h exp=%h @t=%0t",
                         rd_count, rd_data, gbuf[rd_count], $time);
            end
            rd_count = rd_count + 1;
        end
    end

    // golden occupancy (TB's own counters only; never DUT pointers)
    function integer occ_now;
        input dummy;
        begin
            occ_now = wr_count - rd_count;
        end
    endfunction

    // ==================================================================
    // MAIN SEQUENCER
    // ==================================================================
    integer guard;
    initial begin
        test_count = 0;
        fail_count = 0;
        wr_count   = 0;
        rd_count   = 0;

        run_writer    = 1'b0;
        run_reader    = 1'b0;
        writer_paused = 1'b1;
        reader_paused = 1'b1;

        // ---- synchronous active-low reset on both domains ----
        wrst_n = 1'b0;
        rrst_n = 1'b0;
        repeat (4) @(posedge wclk);
        repeat (4) @(posedge rclk);
        @(negedge wclk) wrst_n = 1'b1;
        @(negedge rclk) rrst_n = 1'b1;

        // After reset: empty asserted, full deasserted.
        @(posedge rclk); #1;
        test_count = test_count + 1;
        if (empty !== 1'b1) begin
            fail_count = fail_count + 1;
            $display("FAIL: not EMPTY after reset @t=%0t", $time);
        end
        @(posedge wclk); #1;
        test_count = test_count + 1;
        if (full !== 1'b0) begin
            fail_count = fail_count + 1;
            $display("FAIL: FULL after reset @t=%0t", $time);
        end

        //------------------------------------------------------------------
        // DIRECTED BURST -- Phase A: fill to FULL with the reader held off.
        //------------------------------------------------------------------
        reader_paused = 1'b1;
        writer_paused = 1'b0;
        run_writer    = 1'b1;
        run_reader    = 1'b1;   // reader active but paused -> rd_en stays low

        guard = 0;
        while (!full && guard < 100000) begin
            @(posedge wclk);
            guard = guard + 1;
        end
        test_count = test_count + 1;
        if (!full) begin
            fail_count = fail_count + 1;
            $display("FAIL: FIFO never reached FULL in burst @t=%0t", $time);
        end
        // With full asserted the writer holds wr_en low; let it settle and check
        // the golden occupancy equals DEPTH (all DEPTH slots resident).
        repeat (2) @(posedge wclk); #1;
        test_count = test_count + 1;
        if (occ_now(0) !== DEPTH) begin
            fail_count = fail_count + 1;
            $display("FAIL: occupancy at FULL = %0d, expected %0d @t=%0t",
                     occ_now(0), DEPTH, $time);
        end

        //------------------------------------------------------------------
        // DIRECTED BURST -- Phase B: drain to EMPTY with the writer held off.
        //------------------------------------------------------------------
        writer_paused = 1'b1;
        reader_paused = 1'b0;
        guard = 0;
        while (!empty && guard < 100000) begin
            @(posedge rclk);
            guard = guard + 1;
        end
        repeat (2) @(posedge rclk); #1;
        test_count = test_count + 1;
        if (!empty) begin
            fail_count = fail_count + 1;
            $display("FAIL: FIFO never reached EMPTY in drain @t=%0t", $time);
        end
        test_count = test_count + 1;
        if (occ_now(0) !== 0) begin
            fail_count = fail_count + 1;
            $display("FAIL: occupancy at EMPTY = %0d, expected 0 @t=%0t",
                     occ_now(0), $time);
        end

        //------------------------------------------------------------------
        // CONCURRENT STREAMING: both sides free-running asynchronously until
        // every one of the NWORDS words has been written AND read back.
        //------------------------------------------------------------------
        writer_paused = 1'b0;
        reader_paused = 1'b0;
        guard = 0;
        while ((rd_count < NWORDS) && (guard < 1000000)) begin
            @(posedge wclk);
            guard = guard + 1;
        end
        // drain any final in-flight reads
        repeat (8) @(posedge rclk); #1;

        //------------------------------------------------------------------
        // FINAL CHECKS
        //------------------------------------------------------------------
        test_count = test_count + 1;
        if (wr_count !== NWORDS) begin
            fail_count = fail_count + 1;
            $display("FAIL: writer accepted %0d words, expected %0d",
                     wr_count, NWORDS);
        end
        test_count = test_count + 1;
        if (rd_count !== NWORDS) begin
            fail_count = fail_count + 1;
            $display("FAIL: reader captured %0d words, expected %0d (timeout?)",
                     rd_count, NWORDS);
        end
        test_count = test_count + 1;
        if (empty !== 1'b1) begin
            fail_count = fail_count + 1;
            $display("FAIL: FIFO not EMPTY after all words drained @t=%0t",
                     $time);
        end

        //------------------------------------------------------------------
        // SUMMARY
        //------------------------------------------------------------------
        if (fail_count != 0) begin
            $display("CDC_ASYNC_FIFO: %0d/%0d checks FAILED",
                     fail_count, test_count);
            $fatal(1, "cdc_async_fifo_tb: %0d mismatches", fail_count);
        end else begin
            $display("ALL %0d TESTS PASSED", test_count);
        end
        $finish;
    end

    // global watchdog so a hang never runs forever
    initial begin
        #200000;
        $display("CDC_ASYNC_FIFO: WATCHDOG TIMEOUT (rd_count=%0d wr_count=%0d)",
                 rd_count, wr_count);
        $fatal(1, "cdc_async_fifo_tb: watchdog timeout");
    end

endmodule
