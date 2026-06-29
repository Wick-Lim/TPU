`timescale 1ns/1ps
//============================================================================
// boot_loader_tb.v -- power-up resident-set DMA correctness testbench
//----------------------------------------------------------------------------
// Proves the GLM-5.2-FP8 "model load" is correct: boot_loader copies the
// (non-contiguous, multi-segment) HOT/RESIDENT set out of Flash into DDR5 at
// power-up, word-for-word, and raises its single `done` gate exactly once.
//
// TB stubs (real ONFI/NVMe + DDR5 controllers are vendor IP):
//   * Flash READ port  : a preloaded memory + a FLASH_LAT-deep, IN-ORDER
//     latency pipeline.  flash_ready gates acceptance (LFSR stalls);
//     flash_rvalid/flash_rdata return FLASH_LAT cycles after the issue.
//   * DDR5 WRITE port   : an initially-SENTINEL memory with back-pressure
//     (ddr_ready LFSR stalls -> exercises the skid FIFO fill / read stall).
//
// INDEPENDENT golden: a flat res_flag[]/res_src[] map of the resident DDR5
// image built straight from the descriptor + Flash source words.  Checks are
// X-aware (===).  Asserts, per scenario:
//   (1) done rises EXACTLY once, after the copy, and is a steady level (busy
//       drops, no post-done writes);
//   (2) EVERY resident DDR5 address holds EXACTLY its Flash source word;
//   (3) NOTHING outside the resident region was written (sentinel intact +
//       every observed write address was in-region);
//   (4) words_done == tb_writes == total resident words.
// Plus a continuous no-X monitor on the live control/datapath signals.
// Reports cycles-to-load (boot latency) per scenario.
//============================================================================
module boot_loader_tb;

    // ---- DUT geometry (defaults) ----
    localparam integer ADDR_W  = 32;
    localparam integer DATA_W  = 64;
    localparam integer SEG_MAX = 4;
    localparam integer BURST   = 8;
    localparam integer LEN_W   = 16;
    localparam integer SEGW    = 3;          // clog2(SEG_MAX+1)
    localparam integer PROG_W  = LEN_W+SEGW; // 19

    // ---- TB stub geometry ----
    localparam integer FLASH_LAT = 5;        // Flash read latency (cycles)
    localparam integer FSIZE     = 1024;     // modeled Flash words
    localparam integer DSIZE     = 1024;     // modeled DDR5 words
    localparam [DATA_W-1:0] SENT = 64'hDEAD_BEEF_DEAD_BEEF; // untouched marker
    localparam integer MAXCYC    = 200000;

    integer pass_count = 0;
    integer errors     = 0;

    task chk(input cond, input [1023:0] msg);
        begin
            if (cond) pass_count = pass_count + 1;
            else begin
                errors = errors + 1;
                $display("ASSERT FAIL: %0s  (t=%0t)", msg, $time);
                $fatal(1, "boot_loader_tb assertion failed");
            end
        end
    endtask

    // ------------------------------------------------------------------
    // clock / reset
    // ------------------------------------------------------------------
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ------------------------------------------------------------------
    // DUT I/O
    // ------------------------------------------------------------------
    reg                        start;
    reg  [SEGW-1:0]            seg_count;
    reg  [SEG_MAX*ADDR_W-1:0]  seg_flash_base;
    reg  [SEG_MAX*ADDR_W-1:0]  seg_ddr_base;
    reg  [SEG_MAX*LEN_W-1:0]   seg_len;

    wire                       flash_req;
    wire [ADDR_W-1:0]          flash_addr;
    reg                        flash_ready;
    wire                       flash_rvalid;
    wire [DATA_W-1:0]          flash_rdata;

    wire                       ddr_we;
    wire [ADDR_W-1:0]          ddr_addr;
    wire [DATA_W-1:0]          ddr_wdata;
    reg                        ddr_ready;

    wire                       busy;
    wire                       done;
    wire [PROG_W-1:0]          words_done;

    boot_loader #(
        .ADDR_W (ADDR_W), .DATA_W (DATA_W), .SEG_MAX(SEG_MAX),
        .BURST  (BURST),  .LEN_W  (LEN_W)
    ) dut (
        .clk(clk), .rst(rst),
        .start(start), .seg_count(seg_count),
        .seg_flash_base(seg_flash_base), .seg_ddr_base(seg_ddr_base),
        .seg_len(seg_len),
        .flash_req(flash_req), .flash_addr(flash_addr),
        .flash_ready(flash_ready), .flash_rvalid(flash_rvalid),
        .flash_rdata(flash_rdata),
        .ddr_we(ddr_we), .ddr_addr(ddr_addr), .ddr_wdata(ddr_wdata),
        .ddr_ready(ddr_ready),
        .busy(busy), .done(done), .words_done(words_done)
    );

    // ------------------------------------------------------------------
    // Flash memory + IN-ORDER FLASH_LAT latency pipeline (TB stub)
    // ------------------------------------------------------------------
    reg [DATA_W-1:0] FlashMem [0:FSIZE-1];

    wire issue_fire_tb = flash_req & flash_ready;
    wire [DATA_W-1:0] flash_src_word = FlashMem[flash_addr % FSIZE];

    reg               lat_val [0:FLASH_LAT-1];
    reg [DATA_W-1:0]  lat_dat [0:FLASH_LAT-1];
    integer p;
    always @(posedge clk) begin
        if (rst) begin
            for (p = 0; p < FLASH_LAT; p = p + 1) begin
                lat_val[p] <= 1'b0;
                lat_dat[p] <= {DATA_W{1'b0}};
            end
        end else begin
            lat_val[0] <= issue_fire_tb;
            lat_dat[0] <= flash_src_word;
            for (p = 1; p < FLASH_LAT; p = p + 1) begin
                lat_val[p] <= lat_val[p-1];
                lat_dat[p] <= lat_dat[p-1];
            end
        end
    end
    assign flash_rvalid = lat_val[FLASH_LAT-1];
    assign flash_rdata  = lat_dat[FLASH_LAT-1];

    // ------------------------------------------------------------------
    // DDR5 memory (TB stub) + write capture & spurious-write detector
    // ------------------------------------------------------------------
    reg [DATA_W-1:0] DDRMem  [0:DSIZE-1];
    reg              wrote   [0:DSIZE-1];   // was this DDR addr written this run?
    reg              res_flag[0:DSIZE-1];   // golden: addr in resident set?
    reg [DATA_W-1:0] res_src [0:DSIZE-1];   // golden: expected source word

    integer tb_writes;
    wire    write_fire_tb = ddr_we & ddr_ready;
    integer wa;
    always @(posedge clk) begin
        if (!rst && write_fire_tb) begin
            wa = ddr_addr;  // integer copy
            // (3a) every observed write must land inside the resident region
            chk((wa < DSIZE) && res_flag[wa],
                "spurious write outside resident DDR5 region");
            if (wa < DSIZE) begin
                DDRMem[wa] <= ddr_wdata;
                wrote[wa]  <= 1'b1;
            end
            tb_writes = tb_writes + 1;
        end
    end

    // ------------------------------------------------------------------
    // back-pressure LFSRs (maximal -> never stuck low: progress guaranteed)
    //   flash_ready ~ high most cycles, occasional stall
    //   ddr_ready   ~ frequent stalls -> fills skid FIFO, stalls read side
    // ------------------------------------------------------------------
    reg [15:0] lf1, lf2;
    always @(posedge clk) begin
        if (rst) begin
            lf1 <= 16'hACE1;
            lf2 <= 16'hBEEF;
        end else begin
            lf1 <= {lf1[14:0], lf1[15]^lf1[13]^lf1[12]^lf1[10]};
            lf2 <= {lf2[14:0], lf2[15]^lf2[13]^lf2[12]^lf2[10]};
        end
    end
    always @* flash_ready = lf1[0] | lf1[1] | lf1[2];  // ~7/8 accept
    always @* ddr_ready   = lf2[0] | lf2[1];           // ~3/4 accept

    // ------------------------------------------------------------------
    // continuous no-X monitor on live control/datapath
    // ------------------------------------------------------------------
    reg mon_en;
    always @(posedge clk) begin
        if (mon_en && !rst) begin
            chk(^{busy,done} !== 1'bx,            "X on busy/done");
            chk(^words_done   !== 1'bx,            "X on words_done");
            chk(flash_req     !== 1'bx,            "X on flash_req");
            chk(ddr_we        !== 1'bx,            "X on ddr_we");
            if (flash_req) chk(^flash_addr !== 1'bx, "X on flash_addr while req");
            if (ddr_we) begin
                chk(^ddr_addr  !== 1'bx,           "X on ddr_addr while we");
                chk(^ddr_wdata !== 1'bx,           "X on ddr_wdata while we");
            end
        end
    end

    // ------------------------------------------------------------------
    // done rising-edge counter (proves "exactly once" per run, level-style)
    // ------------------------------------------------------------------
    reg done_prev;
    integer done_rises;
    always @(posedge clk) begin
        if (rst) begin
            done_prev <= 1'b0;
            done_rises <= 0;
        end else begin
            if (done & ~done_prev) done_rises = done_rises + 1;
            done_prev <= done;
        end
    end

    // ------------------------------------------------------------------
    // scenario descriptor (TB-side, unpacked) + golden builder
    // ------------------------------------------------------------------
    integer fbase [0:SEG_MAX-1];
    integer dbase [0:SEG_MAX-1];
    integer len_  [0:SEG_MAX-1];
    integer segc;
    integer total_words;

    integer k, w, a;

    task build_descriptor;
        begin
            // pack the three flat buses + count
            seg_flash_base = {(SEG_MAX*ADDR_W){1'b0}};
            seg_ddr_base   = {(SEG_MAX*ADDR_W){1'b0}};
            seg_len        = {(SEG_MAX*LEN_W){1'b0}};
            for (k = 0; k < SEG_MAX; k = k + 1) begin
                seg_flash_base[k*ADDR_W +: ADDR_W] = fbase[k][ADDR_W-1:0];
                seg_ddr_base  [k*ADDR_W +: ADDR_W] = dbase[k][ADDR_W-1:0];
                seg_len       [k*LEN_W  +: LEN_W ] = len_[k][LEN_W-1:0];
            end
            seg_count = segc[SEGW-1:0];

            // independent golden map of the resident DDR5 image
            for (a = 0; a < DSIZE; a = a + 1) begin
                res_flag[a] = 1'b0;
                res_src[a]  = {DATA_W{1'b0}};
            end
            total_words = 0;
            for (k = 0; k < segc; k = k + 1) begin
                for (w = 0; w < len_[k]; w = w + 1) begin
                    a = dbase[k] + w;
                    res_flag[a] = 1'b1;
                    res_src[a]  = FlashMem[(fbase[k] + w) % FSIZE];
                    total_words = total_words + 1;
                end
            end
        end
    endtask

    // ------------------------------------------------------------------
    // run one scenario start->done; verify the whole DDR5 image
    // ------------------------------------------------------------------
    integer boot_cyc;
    integer base_rises;
    integer wr_snapshot;
    task run_scenario(input [511:0] name);
        begin
            build_descriptor;

            // fresh DDR5 + write shadow
            for (a = 0; a < DSIZE; a = a + 1) begin
                DDRMem[a] = SENT;
                wrote[a]  = 1'b0;
            end
            tb_writes  = 0;
            base_rises = done_rises;

            // power-on: 1-cycle start pulse (inputs may change right after)
            @(negedge clk);
            start = 1'b1;
            @(posedge clk);            // descriptor latched here
            @(negedge clk);
            start = 1'b0;
            // scramble the inputs to prove they were latched
            seg_flash_base = {(SEG_MAX*ADDR_W){1'b1}};
            seg_ddr_base   = {(SEG_MAX*ADDR_W){1'b1}};
            seg_len        = {(SEG_MAX*LEN_W){1'b1}};
            seg_count      = {SEGW{1'b1}};

            // count cycles to load
            boot_cyc = 0;
            while (!done) begin
                @(posedge clk);
                boot_cyc = boot_cyc + 1;
                if (boot_cyc > MAXCYC)
                    chk(1'b0, "TIMEOUT waiting for done");
            end

            // ---- (1) done is a steady level; busy dropped; exactly one rise --
            chk(busy === 1'b0, "busy must drop when done");
            chk((done_rises - base_rises) == 1, "done must rise exactly once");
            chk(done === 1'b1, "done must be high after copy");

            // ---- (4) progress count ----
            chk(words_done === total_words[PROG_W-1:0],
                "words_done must equal total resident words");
            chk(tb_writes == total_words,
                "observed DDR5 writes must equal total resident words");

            // ---- (2)+(3) full DDR5 image equality, X-aware ----
            for (a = 0; a < DSIZE; a = a + 1) begin
                if (res_flag[a]) begin
                    chk(DDRMem[a] === res_src[a],
                        "resident DDR5 word must equal Flash source");
                    chk(wrote[a] === 1'b1,
                        "every resident word must have been written once");
                end else begin
                    chk(DDRMem[a] === SENT,
                        "non-resident DDR5 word must stay untouched");
                    chk(wrote[a] === 1'b0,
                        "no write may land outside the resident region");
                end
            end

            // ---- (1b) level stays high + NO post-done writes when idle ----
            wr_snapshot = tb_writes;
            for (k = 0; k < 8; k = k + 1) @(posedge clk);
            chk(done === 1'b1, "done must remain a steady level while idle");
            chk(busy === 1'b0, "busy must stay low while idle");
            chk(tb_writes == wr_snapshot, "no spurious writes after done");

            $display("  [%0s] PASS  segs=%0d  words=%0d  boot_latency=%0d cycles",
                     name, segc, total_words, boot_cyc);
        end
    endtask

    // ------------------------------------------------------------------
    // stimulus
    // ------------------------------------------------------------------
    integer si;
    initial begin
        // preload Flash with known random image
        for (si = 0; si < FSIZE; si = si + 1)
            FlashMem[si] = {$random, $random};

        start = 1'b0; mon_en = 1'b0;
        seg_count = {SEGW{1'b0}};
        seg_flash_base = {(SEG_MAX*ADDR_W){1'b0}};
        seg_ddr_base   = {(SEG_MAX*ADDR_W){1'b0}};
        seg_len        = {(SEG_MAX*LEN_W){1'b0}};

        // synchronous active-high reset
        rst = 1'b1;
        repeat (4) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;
        mon_en = 1'b1;
        @(posedge clk);

        // post-reset sanity
        chk(busy === 1'b0 && done === 1'b0, "reset clears busy/done");
        chk(words_done === {PROG_W{1'b0}},  "reset clears words_done");

        // ---- Scenario A: 4 segments, non-contiguous, with edges ----
        //   seg0 len=13 (NOT a multiple of BURST=8)
        //   seg1 len=0  (zero-length MIDDLE segment -> skipped)
        //   seg2 len=8  (EXACTLY BURST)
        //   seg3 len=20 (NOT a multiple of BURST; > BURST)
        fbase[0]=32'h0010; dbase[0]=32'h0100; len_[0]=13;
        fbase[1]=32'h0030; dbase[1]=32'h0180; len_[1]=0;
        fbase[2]=32'h0040; dbase[2]=32'h0200; len_[2]=8;
        fbase[3]=32'h0080; dbase[3]=32'h0300; len_[3]=20;
        segc = 4;
        run_scenario("A: 4seg non-contig +zero-mid +nonmult-BURST");

        // ---- Scenario B: seg_count==0 -> nothing to copy, done immediate ----
        fbase[0]=0; dbase[0]=0; len_[0]=0;
        fbase[1]=0; dbase[1]=0; len_[1]=0;
        fbase[2]=0; dbase[2]=0; len_[2]=0;
        fbase[3]=0; dbase[3]=0; len_[3]=0;
        segc = 0;
        run_scenario("B: seg_count=0 (nothing to copy)");
        chk(boot_cyc <= 4, "empty resident set must finish near-immediately");

        // ---- Scenario C: re-run after A/B proves done/busy reusable; also
        //      a single long segment whose len is NOT a multiple of BURST ----
        fbase[0]=32'h0200; dbase[0]=32'h0040; len_[0]=37;
        fbase[1]=0; dbase[1]=0; len_[1]=0;
        fbase[2]=0; dbase[2]=0; len_[2]=0;
        fbase[3]=0; dbase[3]=0; len_[3]=0;
        segc = 1;
        run_scenario("C: re-run, single 37-word segment");

        // ---- Scenario D: all-zero lengths but seg_count>0 -> nothing copied -
        fbase[0]=32'h0001; dbase[0]=32'h0001; len_[0]=0;
        fbase[1]=32'h0002; dbase[1]=32'h0002; len_[1]=0;
        fbase[2]=32'h0003; dbase[2]=32'h0003; len_[2]=0;
        fbase[3]=32'h0004; dbase[3]=32'h0004; len_[3]=0;
        segc = 4;
        run_scenario("D: seg_count=4 but all len==0");

        if (errors == 0)
            $display("ALL %0d TESTS PASSED", pass_count);
        else
            $fatal(1, "%0d ERRORS", errors);
        $finish;
    end

    // global watchdog
    initial begin
        #(20*MAXCYC);
        $fatal(1, "GLOBAL TIMEOUT");
    end

endmodule
