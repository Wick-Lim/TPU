`timescale 1ns/1ps
//============================================================================
// ddr5_xbar_tb.v  --  INDEPENDENT self-checking TB for the banked DDR5 READ
//                     fabric (src/ddr5_xbar.v).
//----------------------------------------------------------------------------
// WHAT IT PROVES  (two independent claims)
//
//  (1) CORRECTNESS.  Each per-channel DDR5 stub returns data = gdata(addr), a
//      deterministic function of the *address* it was handed, plus the carried
//      tag.  The fabric must route every request to the addr-banked channel,
//      back-pressure correctly, and deliver EXACTLY ONE response per request
//      whose data == gdata(addr-that-tag-was-issued-with) and whose tag was one
//      we issued and never duplicated, never X/Z, none lost.  The expected data
//      is recomputed in the CHECKER purely from the address the TB chose for
//      that tag (recorded in tag_addr[]) -- the fabric is on NONE of the golden
//      path, and it never transforms data.  Two address regimes are exercised:
//        - STRIPED  (addr = 0,1,2,...)      perfect round-robin over all N_CH.
//        - RANDOMISH (addr = hash(index))   irregular banking, hot/cold channels,
//                                           heavy out-of-order return.
//
//  (2) THROUGHPUT SCALING -- the whole DDR5-multichannel premise.
//      A single real DDR5 channel does NOT sustain one read per fabric cycle;
//      command/bank turnaround limits it to ~1 read every CH_GAP cycles.  The
//      channel STUB models exactly that: after accepting a read it drops
//      mem_req_ready for CH_GAP cycles (data still returns ROW_LAT later, so up
//      to ceil(ROW_LAT/CH_GAP) reads stay in flight per channel).  Aggregate
//      issue rate is therefore  N_CH / CH_GAP  reads/cycle, capped by the single
//      fabric req/resp port at 1 read/cycle.  With CH_GAP = 8:
//          N_CH=1  ->  ~1 read / 8 cycles   ->  ~8*M cycles for M reads
//          N_CH=8  ->  ~1 read / 1 cycle    ->  ~M   cycles for M reads
//      so striping M reads over 8 channels must run ~8x (==N) faster.  The TB
//      runs the SAME M on an N_CH=1 rig and an N_CH=8 rig (identical channel
//      hardware: same CH_GAP, ROW_LAT, RESP_QD), measures the cycle count of
//      each, and asserts the 8-channel run is ~Nx faster.  If the channel stub
//      were a naive 1/cycle pipe instead, the single fabric port would hide all
//      scaling -- which is exactly why the stub models per-channel BW.
//
//   NOTE on the N_CH=1 rig: with N_CH=1 the fabric still derives a 1-bit
//   channel-select field from req_addr; to keep that field 0 (only channel 0
//   exists) the N_CH=1 rig issues EVEN addresses (addr = index<<1).  Throughput
//   is channel-BW-limited, not address-limited, so this does not perturb the
//   cycle comparison; both rigs perform the same M reads.
//
//   Prints "ALL <N> TESTS PASSED"; $fatal on any mismatch / X / dup / loss /
//   timeout / sub-linear scaling.
//============================================================================

// ===========================================================================
// ddr5_xbar_rig : one full instance under test --
//   DUT + N_CH realistic channel stubs + self-contained driver + checker.
//   Runs M=NREQ reads after `start`, asserts full correctness, and reports
//   the elapsed cycle count on `cycles` when `done` rises.
// ===========================================================================
module ddr5_xbar_rig #(
    parameter integer N_CH    = 8,
    parameter integer ADDR_W  = 16,
    parameter integer DATA_W  = 256,
    parameter integer TAG_W   = 16,
    parameter integer ROW_LAT = 11,    // per-channel read latency (cycles)
    parameter integer RESP_QD = 4,
    parameter integer BANK_LSB= 0,
    parameter integer CH_GAP  = 8,     // per-channel command gap: 1 read / CH_GAP cyc
    parameter integer NREQ    = 1024,  // M reads
    parameter integer MODE    = 0      // 0=striped(N>=2), 1=even-striped(N==1), 2=random
)(
    input  wire        clk,
    input  wire        rst,            // synchronous, active-high (global)
    input  wire        start,          // 1-cycle pulse to launch this rig
    output reg         done,
    output reg [31:0]  cycles
);
    // -------- golden data: deterministic function of ADDRESS (DUT-independent) --
    function [31:0] gword;
        input [ADDR_W-1:0] a;
        begin
            gword = ({{(32-ADDR_W){1'b0}}, a} ^ 32'hDEAD_0000)
                    + (({{(32-ADDR_W){1'b0}}, a}) << 7);
        end
    endfunction
    function [DATA_W-1:0] gdata;
        input [ADDR_W-1:0] a;
        integer w;
        begin
            gdata = {DATA_W{1'b0}};
            for (w = 0; w < DATA_W/32; w = w + 1)
                gdata[w*32 +: 32] = gword(a) ^ {26'h0, w[5:0]};
        end
    endfunction

    // -------- address generator for request #ix (regime selected by MODE) -------
    function [ADDR_W-1:0] mkaddr;
        input integer ix;
        reg [31:0] h;
        begin
            if (MODE == 1) begin
                mkaddr = ix[ADDR_W-1:0] << 1;          // N_CH==1: keep select bit 0
            end else if (MODE == 2) begin
                h = ix * 32'h9E37_79B1;                // mix so low bits (channel) spread
                h = h ^ (h >> 15);
                h = h * 32'h85EB_CA77;
                h = h ^ (h >> 13);
                mkaddr = h[ADDR_W-1:0];
            end else begin
                mkaddr = ix[ADDR_W-1:0];               // striped 0,1,2,...
            end
        end
    endfunction

    // ======================= DUT wiring =======================
    wire                    req_valid;
    wire                    req_ready;
    wire [ADDR_W-1:0]       req_addr;
    wire [TAG_W-1:0]        req_tag;

    wire [N_CH-1:0]         mem_req_valid;
    wire [N_CH-1:0]         mem_req_ready;
    wire [N_CH*ADDR_W-1:0]  mem_req_addr;
    wire [N_CH*TAG_W-1:0]   mem_req_tag;

    wire [N_CH-1:0]         mem_resp_valid;
    wire [N_CH-1:0]         mem_resp_ready;
    wire [N_CH*DATA_W-1:0]  mem_resp_data;
    wire [N_CH*TAG_W-1:0]   mem_resp_tag;

    wire                    resp_valid;
    wire                    resp_ready;
    wire [DATA_W-1:0]       resp_data;
    wire [TAG_W-1:0]        resp_tag;

    ddr5_xbar #(
        .N_CH(N_CH), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
        .ROW_LAT(ROW_LAT), .RESP_QD(RESP_QD), .BANK_LSB(BANK_LSB)
    ) dut (
        .clk(clk), .rst(rst),
        .req_valid(req_valid), .req_ready(req_ready),
        .req_addr(req_addr), .req_tag(req_tag),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr),   .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data),   .mem_resp_tag(mem_resp_tag),
        .resp_valid(resp_valid), .resp_ready(resp_ready),
        .resp_data(resp_data),   .resp_tag(resp_tag)
    );

    // ============== PER-CHANNEL DDR5 STUB (realistic BW model) ==============
    //   * ROW_LAT-deep read-data delay line (data = gdata(addr)).
    //   * Command-gap throttle: after accepting a read, mem_req_ready[c] stays
    //     low for CH_GAP cycles  -> at most one new read accepted / CH_GAP cyc.
    //   * Honours mem_resp_ready back-pressure: when the last data stage is held,
    //     the whole channel freezes (pipe + gap timer), and req_ready drops.
    genvar gc;
    generate
        for (gc = 0; gc < N_CH; gc = gc + 1) begin : g_stub
            reg              pv  [0:ROW_LAT-1];
            reg [ADDR_W-1:0] pa  [0:ROW_LAT-1];
            reg [TAG_W-1:0]  pt  [0:ROW_LAT-1];
            reg [15:0]       busy;                 // cycles until ready again
            integer s;

            wire out_v  = pv[ROW_LAT-1];
            wire stall  = out_v && !mem_resp_ready[gc];   // data stage blocked
            wire ch_rdy = (busy == 16'd0) && !stall;      // gap elapsed & not blocked
            wire accept = mem_req_valid[gc] && ch_rdy;

            assign mem_req_ready[gc]                   = ch_rdy;
            assign mem_resp_valid[gc]                  = out_v;
            assign mem_resp_data[gc*DATA_W +: DATA_W]  = gdata(pa[ROW_LAT-1]);
            assign mem_resp_tag [gc*TAG_W  +: TAG_W ]  = pt[ROW_LAT-1];

            always @(posedge clk) begin
                if (rst) begin
                    busy <= 16'd0;
                    for (s = 0; s < ROW_LAT; s = s + 1) begin
                        pv[s] <= 1'b0;
                        pa[s] <= {ADDR_W{1'b0}};
                        pt[s] <= {TAG_W{1'b0}};
                    end
                end else if (!stall) begin
                    // advance read-data delay line
                    for (s = ROW_LAT-1; s > 0; s = s - 1) begin
                        pv[s] <= pv[s-1];
                        pa[s] <= pa[s-1];
                        pt[s] <= pt[s-1];
                    end
                    pv[0] <= accept;
                    pa[0] <= mem_req_addr[gc*ADDR_W +: ADDR_W];
                    pt[0] <= mem_req_tag [gc*TAG_W  +: TAG_W ];
                    // command-gap throttle
                    if (accept)            busy <= CH_GAP[15:0];
                    else if (busy != 16'd0) busy <= busy - 16'd1;
                end
                // stall: freeze everything (pipe held, busy held, req_ready low)
            end
        end
    endgenerate

    // ======================= DRIVER + CHECKER =======================
    reg              run;
    reg  [31:0]      issued;
    reg  [31:0]      got;
    reg  [31:0]      cyc;
    reg              checked;
    reg              seen     [0:NREQ-1];
    reg  [ADDR_W-1:0] tag_addr[0:NREQ-1];     // address issued for each tag
    integer          i;
    reg  [DATA_W-1:0] exp;

    // request port: drive straight from issue state (feed-forward)
    assign req_valid = run && (issued < NREQ);
    assign req_addr  = mkaddr(issued[31:0] & 32'h7FFF_FFFF);
    assign req_tag   = issued[TAG_W-1:0];
    assign resp_ready = run;                  // always ready to accept responses

    wire req_fire = run && req_valid && req_ready;
    wire cap_fire = run && resp_valid && resp_ready && !done;

    always @(posedge clk) begin
        if (rst) begin
            run     <= 1'b0;
            done    <= 1'b0;
            checked <= 1'b0;
            issued  <= 32'd0;
            got     <= 32'd0;
            cyc     <= 32'd0;
            cycles  <= 32'd0;
            for (i = 0; i < NREQ; i = i + 1) begin
                seen[i]     <= 1'b0;
                tag_addr[i] <= {ADDR_W{1'b0}};
            end
        end else begin
            if (start) run <= 1'b1;
            if (run && !done) cyc <= cyc + 32'd1;     // elapsed-cycle counter

            // ---- issue ----
            if (req_fire) begin
                tag_addr[issued[$clog2(NREQ)-1:0]] <= req_addr;
                issued <= issued + 32'd1;
            end

            // ---- capture + check ----
            if (cap_fire) begin
                if (^{resp_data, resp_tag} === 1'bx) begin
                    $display("FAIL[N_CH=%0d MODE=%0d]: X/Z on response (tag=%0d)",
                             N_CH, MODE, resp_tag);
                    $fatal;
                end
                if (resp_tag >= NREQ[TAG_W-1:0]) begin
                    $display("FAIL[N_CH=%0d MODE=%0d]: tag %0d out of range",
                             N_CH, MODE, resp_tag);
                    $fatal;
                end
                if (seen[resp_tag]) begin
                    $display("FAIL[N_CH=%0d MODE=%0d]: duplicate response tag %0d",
                             N_CH, MODE, resp_tag);
                    $fatal;
                end
                exp = gdata(tag_addr[resp_tag]);
                if (resp_data !== exp) begin
                    $display("FAIL[N_CH=%0d MODE=%0d]: tag %0d data mismatch (addr=%0d)",
                             N_CH, MODE, resp_tag, tag_addr[resp_tag]);
                    $display("   exp = %h", exp);
                    $display("   got = %h", resp_data);
                    $fatal;
                end
                seen[resp_tag] <= 1'b1;
                got <= got + 32'd1;
                if (got == (NREQ-1)) begin     // this is the last response
                    done   <= 1'b1;
                    cycles <= cyc;             // freeze elapsed count
                end
            end

            // ---- after done: confirm none lost (all tags seen exactly once) ----
            if (done && !checked) begin
                checked <= 1'b1;
                for (i = 0; i < NREQ; i = i + 1)
                    if (!seen[i]) begin
                        $display("FAIL[N_CH=%0d MODE=%0d]: missing response tag %0d",
                                 N_CH, MODE, i);
                        $fatal;
                    end
            end
        end
    end
endmodule


// ===========================================================================
// TOP TB : orchestrates 3 rigs, checks correctness everywhere, and compares
//          N_CH=1 vs N_CH=8 cycle counts for the throughput-scaling claim.
// ===========================================================================
module ddr5_xbar_tb;
    localparam integer N_BIG   = 8;
    localparam integer ADDR_W  = 16;
    localparam integer DATA_W  = 256;
    localparam integer TAG_W   = 16;
    localparam integer ROW_LAT = 11;     // odd, non-trivial channel read latency
    localparam integer RESP_QD = 4;
    localparam integer CH_GAP  = 8;      // per-channel: 1 read / 8 cycles
    localparam integer NREQ    = 1024;   // M reads per rig

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;
    reg start;

    // --- rig A: N_CH=8, striped  (scaling fast leg + correctness) ---
    wire        doneA;  wire [31:0] cycA;
    ddr5_xbar_rig #(.N_CH(N_BIG), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
                    .ROW_LAT(ROW_LAT), .RESP_QD(RESP_QD), .CH_GAP(CH_GAP),
                    .NREQ(NREQ), .MODE(0))
        rigA (.clk(clk), .rst(rst), .start(start), .done(doneA), .cycles(cycA));

    // --- rig B: N_CH=1, even-striped  (scaling slow leg + correctness) ---
    wire        doneB;  wire [31:0] cycB;
    ddr5_xbar_rig #(.N_CH(1), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
                    .ROW_LAT(ROW_LAT), .RESP_QD(RESP_QD), .CH_GAP(CH_GAP),
                    .NREQ(NREQ), .MODE(1))
        rigB (.clk(clk), .rst(rst), .start(start), .done(doneB), .cycles(cycB));

    // --- rig C: N_CH=8, randomish addresses  (irregular banking correctness) ---
    wire        doneC;  wire [31:0] cycC;
    ddr5_xbar_rig #(.N_CH(N_BIG), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
                    .ROW_LAT(ROW_LAT), .RESP_QD(RESP_QD), .CH_GAP(CH_GAP),
                    .NREQ(NREQ), .MODE(2))
        rigC (.clk(clk), .rst(rst), .start(start), .done(doneC), .cycles(cycC));

    // ---- sequencing ----
    real spd;
    real ideal_spd;
    integer total_tests;
    initial begin
        rst   = 1'b1;
        start = 1'b0;
        repeat (4) @(posedge clk);
        @(negedge clk); rst = 1'b0;
        @(negedge clk); start = 1'b1;          // launch all three rigs together
        @(negedge clk); start = 1'b0;

        // wait for all rigs to finish
        wait (doneA && doneB && doneC);
        @(posedge clk);                         // let post-done all-seen scan run

        // ---------- THROUGHPUT-SCALING evaluation ----------
        spd       = $itor(cycB) / $itor(cycA);
        ideal_spd = $itor(N_BIG);
        $display("----------------------------------------------------------");
        $display("CORRECTNESS: rigA(N_CH=8 striped), rigB(N_CH=1), rigC(N_CH=8 random)");
        $display("             all %0d reads each verified: data+tag exact, no dup,",
                 NREQ);
        $display("             no loss, no X  (golden recomputed from issued addr).");
        $display("----------------------------------------------------------");
        $display("THROUGHPUT SCALING (same M=%0d reads, identical channel HW):", NREQ);
        $display("   N_CH=1  : %0d cycles", cycB);
        $display("   N_CH=8  : %0d cycles", cycA);
        $display("   speedup : %0.2fx   (ideal N = %0d, CH_GAP = %0d)",
                 spd, N_BIG, CH_GAP);
        if (spd >= 6.5)
            $display("   -> scales ~Nx: 8 channels striped saturate the 1-read/cycle");
        else
            $display("   -> speedup below N (arbitration/back-pressure/latency overhead)");
        $display("----------------------------------------------------------");

        // ---------- scaling assertion ----------
        // ideal is N (=8); allow margin for pipeline-fill + drain latency.
        if (!(cycA < cycB)) begin
            $display("FAIL: N_CH=8 (%0d) not faster than N_CH=1 (%0d)", cycA, cycB);
            $fatal;
        end
        if (spd < 6.0) begin   // require >= 0.75*N so the multichannel premise holds
            $display("FAIL: scaling %0.2fx < 6.0x (expected ~%0dx)", spd, N_BIG);
            $fatal;
        end

        // total verified: 3 rigs * NREQ responses + 1 scaling test
        total_tests = 3*NREQ + 1;
        $display("ALL %0d TESTS PASSED", total_tests);
        $finish;
    end

    // ---- timeout watchdog ----
    initial begin
        #5_000_000;   // 5 ms >> worst rig (~1024*8 cycles ~= 82 us)
        $display("FAIL: timeout (doneA=%0b doneB=%0b doneC=%0b)", doneA, doneB, doneC);
        $fatal;
    end
endmodule
