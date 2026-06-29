`timescale 1ns/1ps
//============================================================================
// flash_xbar_tb.v  --  INDEPENDENT self-checking TB for the banked, deep-
//                      outstanding FLASH READ fabric (src/flash_xbar.v).
//----------------------------------------------------------------------------
// WHAT IT PROVES  (two independent claims)
//
//  (1) CORRECTNESS.  Each per-channel FLASH stub returns data = gdata(addr), a
//      deterministic function of the *address* it was handed, plus the carried
//      tag.  The fabric must route every request to the addr-banked channel,
//      cap in-flight reads per channel at QDEPTH, back-pressure correctly, and
//      deliver EXACTLY ONE response per request whose data == gdata(addr issued
//      for that tag) and whose tag was one we issued and never duplicated, never
//      X/Z, none lost.  Expected data is recomputed in the CHECKER purely from
//      the address the TB chose for that tag (tag_addr[]) -- the fabric is on
//      NONE of the golden path and never transforms data.  Two address regimes:
//        - STRIPED  (addr = 0,1,2,...)   perfect round-robin over all N_CH dies.
//        - RANDOMISH (addr = hash(index)) irregular banking, hot/cold channels,
//                                         heavy out-of-order return.
//
//  (2) DEEP-OUTSTANDING LATENCY HIDING -- the whole high-latency-Flash premise.
//      A NAND read takes FLASH_LAT cycles (here a tractable stand-in for the
//      real ~thousands).  The channel STUB is a FLASH_LAT-deep delay line that
//      can accept a new read EVERY cycle while older reads are still in flight
//      (NAND completes in issue order on a die).  By Little's law a single
//      channel then sustains  min(1/cycle, QDEPTH/FLASH_LAT)  reads/cycle, so:
//          QDEPTH=1 : ~1 read every FLASH_LAT cycles  -> ~M*FLASH_LAT cycles
//          QDEPTH=K : ~K reads every FLASH_LAT cycles -> ~M*FLASH_LAT/K cycles
//      The TB runs the SAME M reads on a SINGLE-channel rig with QDEPTH=1 and a
//      SINGLE-channel rig with QDEPTH=DEEP (identical channel HW: same FLASH_LAT),
//      measures cycles, and asserts the deep-outstanding run is ~DEEPx faster.
//      If the fabric issued only one outstanding read/channel (like a naive
//      ddr5-style router), the deep rig would show NO speedup -- which is exactly
//      why QDEPTH is the new knob.  A third rig (N_CH=8) shows banking scaling
//      stacks on top.
//
//   NOTE on the FLASH_LAT value: the fabric tracks reads by TAG / outstanding
//   COUNT, never by counting cycles, so correctness is latency-agnostic; a
//   modest FLASH_LAT here keeps sim fast while QDEPTH<<FLASH_LAT keeps us firmly
//   in the latency-hiding regime the real device lives in.
//
//   Prints "ALL <N> TESTS PASSED"; $fatal on any mismatch / X / dup / loss /
//   timeout / sub-linear scaling.
//============================================================================

// ===========================================================================
// flash_xbar_rig : DUT + N_CH high-latency FLASH stubs + driver + checker.
//   Runs M=NREQ reads after `start`, asserts full correctness, reports elapsed
//   cycle count on `cycles` when `done` rises.
// ===========================================================================
module flash_xbar_rig #(
    parameter integer N_CH      = 8,
    parameter integer ADDR_W    = 16,
    parameter integer DATA_W    = 256,
    parameter integer TAG_W     = 16,
    parameter integer QDEPTH    = 8,
    parameter integer FLASH_LAT = 64,    // high per-read latency (cycles)
    parameter integer BANK_LSB  = 0,
    parameter integer NREQ      = 512,   // M reads
    parameter integer MODE      = 0      // 0=striped(N>=2), 1=even-striped(N==1), 2=random
)(
    input  wire        clk,
    input  wire        rst,              // synchronous, active-high (global)
    input  wire        start,            // 1-cycle pulse to launch this rig
    output reg         done,
    output reg [31:0]  cycles
);
    // -------- golden data: deterministic function of ADDRESS (DUT-independent) --
    function [31:0] gword;
        input [ADDR_W-1:0] a;
        begin
            gword = ({{(32-ADDR_W){1'b0}}, a} ^ 32'hF1A5_0000)
                    + (({{(32-ADDR_W){1'b0}}, a}) << 9);
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
                h = ix * 32'h9E37_79B1;
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

    flash_xbar #(
        .N_CH(N_CH), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
        .QDEPTH(QDEPTH), .FLASH_LAT(FLASH_LAT), .BANK_LSB(BANK_LSB)
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

    // ============== PER-CHANNEL FLASH STUB (high-latency, deep-pipelined) ======
    //   * FLASH_LAT-deep read-data delay line: a read accepted at cycle t emerges
    //     FLASH_LAT cycles later (data = gdata(addr), tag carried).  The die
    //     accepts a NEW read EVERY cycle while older reads are still in flight ->
    //     up to FLASH_LAT reads in flight at once (the fabric's QDEPTH caps how
    //     many the fabric actually issues).  Completions stay in issue order.
    //   * Honours mem_resp_ready back-pressure: when the emerging data stage is
    //     held, the whole channel freezes (so no completed read is dropped).
    genvar gc;
    generate
        for (gc = 0; gc < N_CH; gc = gc + 1) begin : g_stub
            reg              pv  [0:FLASH_LAT-1];
            reg [ADDR_W-1:0] pa  [0:FLASH_LAT-1];
            reg [TAG_W-1:0]  pt  [0:FLASH_LAT-1];
            integer s;

            wire out_v  = pv[FLASH_LAT-1];
            wire stall  = out_v && !mem_resp_ready[gc];   // emerging data blocked
            wire accept = mem_req_valid[gc] && !stall;    // 1 new read/cycle

            assign mem_req_ready[gc]                   = !stall;
            assign mem_resp_valid[gc]                  = out_v;
            assign mem_resp_data[gc*DATA_W +: DATA_W]  = gdata(pa[FLASH_LAT-1]);
            assign mem_resp_tag [gc*TAG_W  +: TAG_W ]  = pt[FLASH_LAT-1];

            always @(posedge clk) begin
                if (rst) begin
                    for (s = 0; s < FLASH_LAT; s = s + 1) begin
                        pv[s] <= 1'b0;
                        pa[s] <= {ADDR_W{1'b0}};
                        pt[s] <= {TAG_W{1'b0}};
                    end
                end else if (!stall) begin
                    for (s = FLASH_LAT-1; s > 0; s = s - 1) begin
                        pv[s] <= pv[s-1];
                        pa[s] <= pa[s-1];
                        pt[s] <= pt[s-1];
                    end
                    pv[0] <= accept;
                    pa[0] <= mem_req_addr[gc*ADDR_W +: ADDR_W];
                    pt[0] <= mem_req_tag [gc*TAG_W  +: TAG_W ];
                end
                // stall: freeze the whole pipe (no read lost)
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

    assign req_valid  = run && (issued < NREQ);
    assign req_addr   = mkaddr(issued[31:0] & 32'h7FFF_FFFF);
    assign req_tag    = issued[TAG_W-1:0];
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
                    $display("FAIL[N_CH=%0d Q=%0d MODE=%0d]: X/Z on response (tag=%0d)",
                             N_CH, QDEPTH, MODE, resp_tag);
                    $fatal;
                end
                if (resp_tag >= NREQ[TAG_W-1:0]) begin
                    $display("FAIL[N_CH=%0d Q=%0d MODE=%0d]: tag %0d out of range",
                             N_CH, QDEPTH, MODE, resp_tag);
                    $fatal;
                end
                if (seen[resp_tag]) begin
                    $display("FAIL[N_CH=%0d Q=%0d MODE=%0d]: duplicate response tag %0d",
                             N_CH, QDEPTH, MODE, resp_tag);
                    $fatal;
                end
                exp = gdata(tag_addr[resp_tag]);
                if (resp_data !== exp) begin
                    $display("FAIL[N_CH=%0d Q=%0d MODE=%0d]: tag %0d data mismatch (addr=%0d)",
                             N_CH, QDEPTH, MODE, resp_tag, tag_addr[resp_tag]);
                    $display("   exp = %h", exp);
                    $display("   got = %h", resp_data);
                    $fatal;
                end
                seen[resp_tag] <= 1'b1;
                got <= got + 32'd1;
                if (got == (NREQ-1)) begin     // this is the last response
                    done   <= 1'b1;
                    cycles <= cyc;
                end
            end

            // ---- after done: confirm none lost (all tags seen exactly once) ----
            if (done && !checked) begin
                checked <= 1'b1;
                for (i = 0; i < NREQ; i = i + 1)
                    if (!seen[i]) begin
                        $display("FAIL[N_CH=%0d Q=%0d MODE=%0d]: missing response tag %0d",
                                 N_CH, QDEPTH, MODE, i);
                        $fatal;
                    end
            end
        end
    end
endmodule


// ===========================================================================
// TOP TB : orchestrates 4 rigs, checks correctness everywhere, and compares
//          QDEPTH=1 vs QDEPTH=DEEP single-channel cycle counts (latency hiding).
// ===========================================================================
module flash_xbar_tb;
    localparam integer N_BIG     = 8;
    localparam integer ADDR_W    = 16;
    localparam integer DATA_W    = 256;
    localparam integer TAG_W     = 16;
    localparam integer FLASH_LAT = 64;     // high per-read latency (stand-in)
    localparam integer QDEEP     = 8;      // deep outstanding budget per channel
    localparam integer NREQ      = 512;    // M reads per rig

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;
    reg start;

    // --- rig A: N_CH=8, QDEPTH=8, striped  (banking+depth fast + correctness) ---
    wire        doneA;  wire [31:0] cycA;
    flash_xbar_rig #(.N_CH(N_BIG), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
                     .QDEPTH(QDEEP), .FLASH_LAT(FLASH_LAT), .NREQ(NREQ), .MODE(0))
        rigA (.clk(clk), .rst(rst), .start(start), .done(doneA), .cycles(cycA));

    // --- rig B: N_CH=1, QDEPTH=1  (latency-hiding SLOW leg: 1 in flight) ---
    wire        doneB;  wire [31:0] cycB;
    flash_xbar_rig #(.N_CH(1), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
                     .QDEPTH(1), .FLASH_LAT(FLASH_LAT), .NREQ(NREQ), .MODE(1))
        rigB (.clk(clk), .rst(rst), .start(start), .done(doneB), .cycles(cycB));

    // --- rig C: N_CH=1, QDEPTH=8  (latency-hiding FAST leg: 8 in flight) ---
    wire        doneC;  wire [31:0] cycC;
    flash_xbar_rig #(.N_CH(1), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
                     .QDEPTH(QDEEP), .FLASH_LAT(FLASH_LAT), .NREQ(NREQ), .MODE(1))
        rigC (.clk(clk), .rst(rst), .start(start), .done(doneC), .cycles(cycC));

    // --- rig D: N_CH=8, QDEPTH=8, randomish addresses  (irregular banking) ---
    wire        doneD;  wire [31:0] cycD;
    flash_xbar_rig #(.N_CH(N_BIG), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
                     .QDEPTH(QDEEP), .FLASH_LAT(FLASH_LAT), .NREQ(NREQ), .MODE(2))
        rigD (.clk(clk), .rst(rst), .start(start), .done(doneD), .cycles(cycD));

    real spd;
    integer total_tests;
    initial begin
        rst   = 1'b1;
        start = 1'b0;
        repeat (4) @(posedge clk);
        @(negedge clk); rst = 1'b0;
        @(negedge clk); start = 1'b1;          // launch all four rigs together
        @(negedge clk); start = 1'b0;

        wait (doneA && doneB && doneC && doneD);
        @(posedge clk);                         // let post-done all-seen scan run

        // ---------- DEEP-OUTSTANDING (latency-hiding) evaluation ----------
        spd = $itor(cycB) / $itor(cycC);
        $display("----------------------------------------------------------");
        $display("CORRECTNESS: rigA(N=8 Q=8 striped), rigB(N=1 Q=1), rigC(N=1 Q=8),");
        $display("             rigD(N=8 Q=8 random) -- all %0d reads each verified:",
                 NREQ);
        $display("             data+tag exact, no dup, no loss, no X.");
        $display("----------------------------------------------------------");
        $display("LATENCY HIDING (same M=%0d reads, FLASH_LAT=%0d, single channel):",
                 NREQ, FLASH_LAT);
        $display("   QDEPTH=1 : %0d cycles  (~1 read in flight)", cycB);
        $display("   QDEPTH=8 : %0d cycles  (~8 reads in flight)", cycC);
        $display("   speedup  : %0.2fx   (ideal QDEPTH = %0d)", spd, QDEEP);
        $display("   N=8,Q=8  : %0d cycles  (banking x depth, streaming)", cycA);
        if (spd >= 6.5)
            $display("   -> deep outstanding hides NAND latency: ~QDEPTHx more BW");
        else
            $display("   -> below QDEPTH (fill/drain overhead)");
        $display("----------------------------------------------------------");

        // ---------- scaling assertions ----------
        if (!(cycC < cycB)) begin
            $display("FAIL: QDEPTH=8 (%0d) not faster than QDEPTH=1 (%0d)", cycC, cycB);
            $fatal;
        end
        if (spd < 6.0) begin   // require >= 0.75*QDEPTH so latency-hiding holds
            $display("FAIL: latency-hiding %0.2fx < 6.0x (expected ~%0dx)", spd, QDEEP);
            $fatal;
        end

        // total verified: 4 rigs * NREQ responses + 1 latency-hiding test
        total_tests = 4*NREQ + 1;
        $display("ALL %0d TESTS PASSED", total_tests);
        $finish;
    end

    // ---- timeout watchdog ----
    initial begin
        #20_000_000;  // 20 ms >> worst rig (~512*64 = 32768 cycles ~= 0.33 ms)
        $display("FAIL: timeout (A=%0b B=%0b C=%0b D=%0b)", doneA, doneB, doneC, doneD);
        $fatal;
    end
endmodule
