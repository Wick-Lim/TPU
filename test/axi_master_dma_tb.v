`timescale 1ns/1ps
//============================================================================
// axi_master_dma_tb.v  --  unit testbench for the AXI4-Lite MASTER DMA engine
//----------------------------------------------------------------------------
// WHAT IT CHECKS
//   The DUT (src/axi_master_dma.v) is an AXI4-Lite *master*.  This TB provides
//   the missing *slave*: a behavioral AXI4-Lite SLAVE-MEMORY BFM -- a 64x32-bit
//   reg array that fully implements the slave side of AW/W/B and AR/R with
//   realistic, INDEPENDENTLY-handshaked VALID/READY (including injected wait
//   states), an OKAY response for in-range words, and a SLVERR response for any
//   word index >= the array depth (the out-of-range / error path).
//
//   It also models the DMA's small INTERNAL streaming port:
//     * READ  dir: a "sink" array captured from {wr_en,wr_idx,wr_data}.
//     * WRITE dir: a "source" array returned combinationally on {rd_req,rd_idx}.
//
//   TESTS
//     1. READ multi-word : preload the BFM with a known pattern, command a READ
//        of N words into the sink, check every sink word == INDEPENDENT golden,
//        in order, with no error.
//     2. WRITE multi-word: load a known source pattern, command a WRITE of N
//        words, check the BFM memory now holds exactly that pattern (golden),
//        no error.
//     3. READ  len = 1   : single-word read, exact value, no error.
//     4. WRITE len = 1   : single-word write, BFM word updated, no error.
//     5. ERROR (read)    : command a READ whose base address is out of the BFM's
//        range -> the slave returns SLVERR -> the DUT must raise `err`, abort,
//        and pulse `done`; the sink must be untouched past the faulting beat.
//     6. ERROR (write)   : command a WRITE to an out-of-range address -> SLVERR
//        -> DUT raises `err`, aborts, pulses `done`; the BFM memory is unchanged.
//
//   The golden is INDEPENDENT: it is the TB's own copy of the patterns computed
//   with plain software loops; it shares no logic with the DUT FSM or pipeline.
//
//   Prints "ALL <N> TESTS PASSED" and $fatal on the FIRST mismatch.
//============================================================================
module axi_master_dma_tb;

    // ---------------- parameters / sizes ----------------------------------
    localparam integer ADDR_W    = 32;
    localparam integer LENW      = 8;
    localparam integer MEM_WORDS = 64;          // BFM depth (words)
    localparam integer MEM_AW    = 6;           // log2(64) word-index bits

    localparam [1:0] RESP_OKAY   = 2'b00;
    localparam [1:0] RESP_SLVERR = 2'b10;

    // ---------------- clock / reset ---------------------------------------
    reg ACLK = 1'b0;
    reg ARESETn;
    always #5 ACLK = ~ACLK;                      // 100 MHz

    // ---------------- command / control interface -------------------------
    reg               start;
    reg  [ADDR_W-1:0] ext_addr;
    reg  [LENW-1:0]   len;
    reg               dir;
    wire              busy;
    wire              done;
    wire              err;

    // ---------------- internal streaming sink (READ dir) ------------------
    wire              wr_en;
    wire [LENW-1:0]   wr_idx;
    wire [31:0]       wr_data;

    // ---------------- internal streaming source (WRITE dir) ---------------
    wire              rd_req;
    wire [LENW-1:0]   rd_idx;
    reg  [31:0]       rd_data;

    // ---------------- AXI4-Lite master channels (DUT drives) --------------
    wire [ADDR_W-1:0] AWADDR;
    wire [2:0]        AWPROT;
    wire              AWVALID;
    reg               AWREADY;
    wire [31:0]       WDATA;
    wire [3:0]        WSTRB;
    wire              WVALID;
    reg               WREADY;
    reg  [1:0]        BRESP;
    reg               BVALID;
    wire              BREADY;
    wire [ADDR_W-1:0] ARADDR;
    wire [2:0]        ARPROT;
    wire              ARVALID;
    reg               ARREADY;
    reg  [31:0]       RDATA;
    reg  [1:0]        RRESP;
    reg               RVALID;
    wire              RREADY;

    // unused-but-driven sideband (silence lint in the TB scope)
    /* verilator lint_off UNUSEDSIGNAL */
    wire [2:0] unused_awprot = AWPROT;
    wire [2:0] unused_arprot = ARPROT;
    wire [3:0] unused_wstrb  = WSTRB;
    /* verilator lint_on UNUSEDSIGNAL */

    // ---------------- DUT -------------------------------------------------
    axi_master_dma #(.ADDR_W(ADDR_W), .LENW(LENW)) dut (
        .ACLK     (ACLK),
        .ARESETn  (ARESETn),
        .start    (start),
        .ext_addr (ext_addr),
        .len      (len),
        .dir      (dir),
        .busy     (busy),
        .done     (done),
        .err      (err),
        .wr_en    (wr_en),
        .wr_idx   (wr_idx),
        .wr_data  (wr_data),
        .rd_req   (rd_req),
        .rd_idx   (rd_idx),
        .rd_data  (rd_data),
        .AWADDR   (AWADDR),
        .AWPROT   (AWPROT),
        .AWVALID  (AWVALID),
        .AWREADY  (AWREADY),
        .WDATA    (WDATA),
        .WSTRB    (WSTRB),
        .WVALID   (WVALID),
        .WREADY   (WREADY),
        .BRESP    (BRESP),
        .BVALID   (BVALID),
        .BREADY   (BREADY),
        .ARADDR   (ARADDR),
        .ARPROT   (ARPROT),
        .ARVALID  (ARVALID),
        .ARREADY  (ARREADY),
        .RDATA    (RDATA),
        .RRESP    (RRESP),
        .RVALID   (RVALID),
        .RREADY   (RREADY)
    );

    // ======================================================================
    // Behavioral storage.
    //   mem   : the BFM's slave memory (what the DUT reads/writes over AXI).
    //   sink  : captures the DUT's internal write-stream (READ dir results).
    //   src   : the DUT's internal source (WRITE dir input words).
    //   gold  : the INDEPENDENT reference array.
    // ======================================================================
    reg [31:0] mem  [0:MEM_WORDS-1];
    reg [31:0] sink [0:255];
    reg [31:0] src  [0:255];
    reg [31:0] gold [0:255];

    integer i;
    integer tests;
    integer sink_count;     // # of sink words captured in the current run

    // ----------------------------------------------------------------------
    // Internal SOURCE port (WRITE dir): when the DUT pulses rd_req for rd_idx,
    // return src[rd_idx] combinationally THIS cycle so the DUT registers it.
    // Default to X-ish 0 when no request is active (DUT ignores it then).
    // ----------------------------------------------------------------------
    always @(*) begin
        if (rd_req) rd_data = src[rd_idx];
        else        rd_data = 32'hDEAD_BEEF;   // poison: must never be sampled
    end

    // ----------------------------------------------------------------------
    // Internal SINK port (READ dir): capture each returned word in order.
    // ----------------------------------------------------------------------
    always @(posedge ACLK) begin
        if (wr_en) begin
            sink[wr_idx] <= wr_data;
            sink_count   <= sink_count + 1;
        end
    end

    // ======================================================================
    // AXI4-Lite SLAVE-MEMORY BFM
    //----------------------------------------------------------------------
    // Implements the slave side of all five channels with registered, single-
    // outstanding handshakes and a few injected wait states (the *READY pulses
    // are delayed by a small per-channel counter so the master's VALID-hold and
    // back-pressure handling are exercised).  Word index = ADDR[ADDR_W-1:2];
    // any index >= MEM_WORDS returns SLVERR (the out-of-range error path).
    //
    // The BFM never combinationally drives a READY from the matching VALID in a
    // way that loops: each READY is a REGISTERED pulse produced from a counter.
    // ======================================================================

    // ---- write address channel ----
    reg [MEM_AW+1:0] aw_word;     // captured word index (+2 guard bits for range)
    reg              aw_have;     // a write address has been captured, awaiting data
    reg [3:0]        aw_wait;     // wait-state countdown before asserting AWREADY

    // ---- write data channel ----
    reg [31:0]       w_data_lat;
    reg              w_have;      // write data captured, awaiting address (or commit)
    reg [3:0]        w_wait;

    // ---- read address channel ----
    reg [MEM_AW+1:0] ar_word;
    reg              ar_have;
    reg [3:0]        ar_wait;

    // range check on captured indices (full ADDR width compare via word index)
    wire aw_oor = (aw_word >= MEM_WORDS[MEM_AW+1:0]);
    wire ar_oor = (ar_word >= MEM_WORDS[MEM_AW+1:0]);

    // Number of ACLK wait states to insert before each channel's READY.  A small
    // rotating pattern (0..2) so different beats see different back-pressure.
    reg [1:0] ws_rot;

    always @(posedge ACLK) begin
        if (!ARESETn) begin
            AWREADY    <= 1'b0;
            WREADY     <= 1'b0;
            BVALID     <= 1'b0;
            BRESP      <= RESP_OKAY;
            ARREADY    <= 1'b0;
            RVALID     <= 1'b0;
            RRESP      <= RESP_OKAY;
            RDATA      <= 32'b0;
            aw_word    <= {(MEM_AW+2){1'b0}};
            aw_have    <= 1'b0;
            aw_wait    <= 4'd0;
            w_data_lat <= 32'b0;
            w_have     <= 1'b0;
            w_wait     <= 4'd0;
            ar_word    <= {(MEM_AW+2){1'b0}};
            ar_have    <= 1'b0;
            ar_wait    <= 4'd0;
            ws_rot     <= 2'd0;
        end else begin
            // ---------------- WRITE ADDRESS (AW) ----------------
            // Insert aw_wait wait states, then pulse AWREADY for one cycle.
            if (AWREADY) begin
                AWREADY <= 1'b0;                       // single-cycle pulse
            end else if (AWVALID && !aw_have) begin
                if (aw_wait == 4'd0) begin
                    AWREADY <= 1'b1;                   // accept this cycle
                    aw_word <= AWADDR[MEM_AW+3:2];     // capture word index
                    aw_have <= 1'b1;
                    aw_wait <= {2'b0, ws_rot};         // re-arm for next beat
                end else begin
                    aw_wait <= aw_wait - 4'd1;         // burn a wait state
                end
            end

            // ---------------- WRITE DATA (W) ----------------
            if (WREADY) begin
                WREADY <= 1'b0;
            end else if (WVALID && !w_have) begin
                if (w_wait == 4'd0) begin
                    WREADY     <= 1'b1;
                    w_data_lat <= WDATA;
                    w_have     <= 1'b1;
                    w_wait     <= {2'b0, ~ws_rot};     // different phase than AW
                end else begin
                    w_wait <= w_wait - 4'd1;
                end
            end

            // ---------------- WRITE COMMIT + RESPONSE (B) ----------------
            // When both address and data are captured and no B is in flight,
            // perform the memory update (if in range) and raise BVALID.
            if (aw_have && w_have && !BVALID) begin
                if (aw_oor) begin
                    BRESP <= RESP_SLVERR;              // out-of-range -> error
                end else begin
                    mem[aw_word[MEM_AW-1:0]] <= w_data_lat;
                    BRESP <= RESP_OKAY;
                end
                BVALID <= 1'b1;
            end
            if (BVALID && BREADY) begin
                BVALID  <= 1'b0;                       // response accepted
                aw_have <= 1'b0;                       // free the slave for next
                w_have  <= 1'b0;
                ws_rot  <= ws_rot + 2'd1;              // rotate back-pressure
            end

            // ---------------- READ ADDRESS (AR) ----------------
            if (ARREADY) begin
                ARREADY <= 1'b0;
            end else if (ARVALID && !ar_have) begin
                if (ar_wait == 4'd0) begin
                    ARREADY <= 1'b1;
                    ar_word <= ARADDR[MEM_AW+3:2];
                    ar_have <= 1'b1;
                    ar_wait <= {2'b0, ws_rot};
                end else begin
                    ar_wait <= ar_wait - 4'd1;
                end
            end

            // ---------------- READ DATA + RESPONSE (R) ----------------
            if (ar_have && !RVALID) begin
                if (ar_oor) begin
                    RRESP <= RESP_SLVERR;
                    RDATA <= 32'hBAD0_BAD0;            // garbage on error
                end else begin
                    RRESP <= RESP_OKAY;
                    RDATA <= mem[ar_word[MEM_AW-1:0]];
                end
                RVALID <= 1'b1;
            end
            if (RVALID && RREADY) begin
                RVALID  <= 1'b0;
                ar_have <= 1'b0;
                ws_rot  <= ws_rot + 2'd1;
            end
        end
    end

    // ======================================================================
    // Test sequencing helpers.
    // ======================================================================

    // Issue a 1-cycle start pulse with the given descriptor, then wait for the
    // done pulse (with a watchdog).  All command signals are driven on negedge
    // so they are stable around the DUT's posedge sample.
    task run_dma(input [ADDR_W-1:0] addr, input [LENW-1:0] n, input d);
        integer guard;
        begin
            @(negedge ACLK);
            ext_addr = addr;
            len      = n;
            dir      = d;
            start    = 1'b1;
            @(negedge ACLK);
            start = 1'b0;
            // wait for done
            guard = 0;
            while (!done) begin
                @(negedge ACLK);
                guard = guard + 1;
                if (guard > 5000) begin
                    $display("FAIL: DMA timeout (no done) addr=%h n=%0d dir=%0d",
                             addr, n, d);
                    $fatal;
                end
            end
            @(negedge ACLK);   // settle one cycle past done
        end
    endtask

    // ======================================================================
    // Stimulus / checks.
    // ======================================================================
    integer k;
    integer base_word;
    integer errcap;

    initial begin
        // init
        start    = 1'b0;
        ext_addr = {ADDR_W{1'b0}};
        len      = {LENW{1'b0}};
        dir      = 1'b0;
        tests    = 0;
        sink_count = 0;
        for (i = 0; i < 256; i = i + 1) begin
            sink[i] = 32'h0;
            src[i]  = 32'h0;
            gold[i] = 32'h0;
        end
        for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = 32'h0;

        // reset (active-low) for a few cycles
        ARESETn = 1'b0;
        repeat (4) @(negedge ACLK);
        ARESETn = 1'b1;
        @(negedge ACLK);

        // ------------------------------------------------------------------
        // TEST 1: READ multi-word (N=10) from base word 3.
        // ------------------------------------------------------------------
        base_word = 3;
        for (i = 0; i < MEM_WORDS; i = i + 1)
            mem[i] = 32'hA000_0000 ^ (i * 32'h0001_0001) ^ {i[7:0], 8'h5A, i[7:0]};
        // INDEPENDENT golden: the 10 words the DUT should deliver, in order.
        for (k = 0; k < 10; k = k + 1)
            gold[k] = mem[base_word + k];
        // clear sink
        for (i = 0; i < 256; i = i + 1) sink[i] = 32'h0;
        sink_count = 0;

        run_dma(base_word * 4, 8'd10, 1'b0);

        if (err) begin
            $display("FAIL T1: unexpected err on in-range READ"); $fatal;
        end
        if (sink_count !== 10) begin
            $display("FAIL T1: sink_count=%0d expected 10", sink_count); $fatal;
        end
        for (k = 0; k < 10; k = k + 1) begin
            if (sink[k] !== gold[k]) begin
                $display("FAIL T1: sink[%0d]=%h expected %h", k, sink[k], gold[k]);
                $fatal;
            end
        end
        tests = tests + 1;
        $display("PASS T1: READ 10 words match golden, in order, no err");

        // ------------------------------------------------------------------
        // TEST 2: WRITE multi-word (N=12) to base word 8.
        // ------------------------------------------------------------------
        base_word = 8;
        // known source pattern + independent golden of what mem should become.
        for (k = 0; k < 12; k = k + 1) begin
            src[k]  = 32'hC0DE_0000 + (k * 32'h0011_0007) + 32'h0000_1000;
            gold[k] = src[k];
        end
        // pre-fill the target region with a sentinel so we know it changed.
        for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = 32'hFFFF_FFFF;

        run_dma(base_word * 4, 8'd12, 1'b1);

        if (err) begin
            $display("FAIL T2: unexpected err on in-range WRITE"); $fatal;
        end
        for (k = 0; k < 12; k = k + 1) begin
            if (mem[base_word + k] !== gold[k]) begin
                $display("FAIL T2: mem[%0d]=%h expected %h",
                         base_word + k, mem[base_word + k], gold[k]);
                $fatal;
            end
        end
        // words outside the written range must be untouched (still sentinel).
        for (i = 0; i < base_word; i = i + 1)
            if (mem[i] !== 32'hFFFF_FFFF) begin
                $display("FAIL T2: mem[%0d] under-run corrupted = %h", i, mem[i]);
                $fatal;
            end
        for (i = base_word + 12; i < MEM_WORDS; i = i + 1)
            if (mem[i] !== 32'hFFFF_FFFF) begin
                $display("FAIL T2: mem[%0d] over-run corrupted = %h", i, mem[i]);
                $fatal;
            end
        tests = tests + 1;
        $display("PASS T2: WRITE 12 words landed in BFM mem, no over/under-run, no err");

        // ------------------------------------------------------------------
        // TEST 3: READ len=1 from base word 20.
        // ------------------------------------------------------------------
        base_word = 20;
        for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = 32'h1234_0000 + i;
        gold[0] = mem[base_word];
        for (i = 0; i < 256; i = i + 1) sink[i] = 32'h0;
        sink_count = 0;

        run_dma(base_word * 4, 8'd1, 1'b0);

        if (err) begin $display("FAIL T3: unexpected err"); $fatal; end
        if (sink_count !== 1) begin
            $display("FAIL T3: sink_count=%0d expected 1", sink_count); $fatal;
        end
        if (sink[0] !== gold[0]) begin
            $display("FAIL T3: sink[0]=%h expected %h", sink[0], gold[0]); $fatal;
        end
        tests = tests + 1;
        $display("PASS T3: READ len=1 word matches golden, no err");

        // ------------------------------------------------------------------
        // TEST 4: WRITE len=1 to base word 31.
        // ------------------------------------------------------------------
        base_word = 31;
        src[0]  = 32'hBEEF_F00D;
        gold[0] = src[0];
        for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = 32'h0000_0000;

        run_dma(base_word * 4, 8'd1, 1'b1);

        if (err) begin $display("FAIL T4: unexpected err"); $fatal; end
        if (mem[base_word] !== gold[0]) begin
            $display("FAIL T4: mem[%0d]=%h expected %h",
                     base_word, mem[base_word], gold[0]); $fatal;
        end
        // neighbour words must be untouched
        if (mem[base_word-1] !== 32'h0) begin
            $display("FAIL T4: neighbour mem[%0d] corrupted=%h",
                     base_word-1, mem[base_word-1]); $fatal;
        end
        tests = tests + 1;
        $display("PASS T4: WRITE len=1 word landed, neighbour intact, no err");

        // ------------------------------------------------------------------
        // TEST 5: ERROR on READ -- base address is out of the BFM range.
        //   base word 70 (> MEM_WORDS=64) -> slave returns SLVERR on the first
        //   R beat -> DUT must raise err, abort, and pulse done.  Sink stays
        //   empty (no word should be committed for an errored beat).
        // ------------------------------------------------------------------
        for (i = 0; i < 256; i = i + 1) sink[i] = 32'h0;
        sink_count = 0;

        run_dma(70 * 4, 8'd4, 1'b0);

        if (!err) begin
            $display("FAIL T5: err NOT raised on out-of-range READ"); $fatal;
        end
        if (sink_count !== 0) begin
            $display("FAIL T5: sink got %0d words on errored READ (expected 0)",
                     sink_count); $fatal;
        end
        tests = tests + 1;
        $display("PASS T5: out-of-range READ -> err raised, run aborted, no sink writes");

        // ------------------------------------------------------------------
        // TEST 6: ERROR on WRITE -- out-of-range address -> SLVERR -> err.
        //   The BFM memory must be unchanged (no in-range word was addressed).
        // ------------------------------------------------------------------
        for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = 32'hAAAA_AAAA;
        src[0] = 32'h5555_5555;

        run_dma(70 * 4, 8'd3, 1'b1);

        if (!err) begin
            $display("FAIL T6: err NOT raised on out-of-range WRITE"); $fatal;
        end
        for (i = 0; i < MEM_WORDS; i = i + 1)
            if (mem[i] !== 32'hAAAA_AAAA) begin
                $display("FAIL T6: BFM mem[%0d] changed=%h on errored WRITE",
                         i, mem[i]); $fatal;
            end
        tests = tests + 1;
        $display("PASS T6: out-of-range WRITE -> err raised, BFM mem untouched");

        // ------------------------------------------------------------------
        // Sanity: after a clean run following an error, err must clear (start
        // clears the sticky flag).  Re-run T3-style read, expect no err.
        // ------------------------------------------------------------------
        base_word = 5;
        for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = 32'h9000_0000 + i;
        gold[0] = mem[base_word];
        gold[1] = mem[base_word+1];
        for (i = 0; i < 256; i = i + 1) sink[i] = 32'h0;
        sink_count = 0;

        run_dma(base_word * 4, 8'd2, 1'b0);

        if (err) begin
            $display("FAIL T7: err did NOT clear on a fresh start after an error");
            $fatal;
        end
        if (sink[0] !== gold[0] || sink[1] !== gold[1]) begin
            $display("FAIL T7: post-error READ data wrong"); $fatal;
        end
        tests = tests + 1;
        $display("PASS T7: err cleared by new start; subsequent clean READ correct");

        // ------------------------------------------------------------------
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end

    // Global watchdog so a hang fails loudly instead of running forever.
    initial begin
        #500000;
        $display("FAIL: global timeout");
        $fatal;
    end

endmodule
