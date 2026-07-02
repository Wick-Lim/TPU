`timescale 1ns/1ps
//============================================================================
// clk_gate_cluster_tb.v  --  self-checking TB for the C7 clock-gating wrapper
//----------------------------------------------------------------------------
// Proves, against a FREE-RUNNING reference register_file clocked by the ungated
// clk and driven with identical stimulus:
//   (1) EQUIVALENCE : while active (req=1) the gated-clock leaf is BIT-IDENTICAL
//                     to the free-running leaf on the same stimulus.
//   (2) GATING      : when idle (req=0), once gated the gated_clk does NOT
//                     toggle across an idle window (held constant) -> power saved.
//   (3) SCAN        : scan_enable=1 forces gated_clk == clk (full toggling),
//                     regardless of enable / idle.
//   (4) SAFETY      : a req pulse forces clk_en(=ICG enable)=1 that SAME cycle,
//                     even from a gated state (pending work is never gated).
// Prints "ALL N TESTS PASSED"; $fatal on any mismatch / X.
//============================================================================
module clk_gate_cluster_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;   // 10ns period

    reg        rst;
    reg        scan_enable;
    reg        req;

    // ---- shared register_file stimulus ----
    reg  [3:0]  read_addr1, read_addr2, read_addr3;
    reg  [3:0]  write_addr;
    reg  [31:0] write_data;
    reg         write_enable;

    // ---- DUT (gated) outputs ----
    wire [31:0] g_rd1, g_rd2, g_rd3;
    wire        clk_en;
    wire        gated_clk;

    // ---- reference (free-running) outputs ----
    wire [31:0] r_rd1, r_rd2, r_rd3;

    integer tests = 0;

    //------------------------------------------------------------------
    // DUT: clock-gating cluster (register_file on the GATED clock)
    //------------------------------------------------------------------
    clk_gate_cluster #(.HOLD(1)) dut (
        .clk          (clk),
        .rst          (rst),
        .scan_enable  (scan_enable),
        .req          (req),
        .read_addr1   (read_addr1),
        .read_addr2   (read_addr2),
        .read_addr3   (read_addr3),
        .write_addr   (write_addr),
        .write_data   (write_data),
        .write_enable (write_enable),
        .read_data1   (g_rd1),
        .read_data2   (g_rd2),
        .read_data3   (g_rd3),
        .clk_en       (clk_en),
        .gated_clk    (gated_clk)
    );

    //------------------------------------------------------------------
    // REFERENCE: identical leaf on the FREE-RUNNING clk, same stimulus
    //------------------------------------------------------------------
    register_file ref_rf (
        .clk          (clk),          // ungated
        .rst          (rst),
        .read_addr1   (read_addr1),
        .read_addr2   (read_addr2),
        .read_addr3   (read_addr3),
        .write_addr   (write_addr),
        .write_data   (write_data),
        .write_enable (write_enable),
        .read_data1   (r_rd1),
        .read_data2   (r_rd2),
        .read_data3   (r_rd3)
    );

    //------------------------------------------------------------------
    // helpers
    //------------------------------------------------------------------
    task pass(input [255:0] name);
        begin
            tests = tests + 1;
            $display("  PASS [%0d]: %0s", tests, name);
        end
    endtask

    // check gated leaf == reference leaf on all 3 read ports (no X allowed)
    task check_equiv(input [255:0] name);
        begin
            if (^{g_rd1,g_rd2,g_rd3,r_rd1,r_rd2,r_rd3} === 1'bx) begin
                $display("FAIL %0s: X on read data (g=%h/%h/%h r=%h/%h/%h)",
                         name, g_rd1,g_rd2,g_rd3, r_rd1,r_rd2,r_rd3);
                $fatal;
            end
            if ((g_rd1 !== r_rd1) || (g_rd2 !== r_rd2) || (g_rd3 !== r_rd3)) begin
                $display("FAIL %0s: mismatch gated(%h,%h,%h) != ref(%h,%h,%h)",
                         name, g_rd1,g_rd2,g_rd3, r_rd1,r_rd2,r_rd3);
                $fatal;
            end
            pass(name);
        end
    endtask

    // drive one write on the shared stimulus and step one clock
    task drive_write(input [3:0] a, input [31:0] d);
        begin
            @(negedge clk);
            write_addr   = a;
            write_data   = d;
            write_enable = 1'b1;
            @(posedge clk);
            #1;
        end
    endtask

    integer i;
    integer edges;
    reg gclk_prev;

    initial begin
        // ---- init ----
        rst          = 1'b1;
        scan_enable  = 1'b0;
        req          = 1'b1;      // keep clock alive THROUGH reset
        read_addr1   = 4'd0; read_addr2 = 4'd0; read_addr3 = 4'd0;
        write_addr   = 4'd0; write_data = 32'd0; write_enable = 1'b0;

        // hold reset for a few gated edges
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;
        @(posedge clk); #1;

        //==============================================================
        // (1) EQUIVALENCE while active (req=1 -> gated_clk == clk)
        //==============================================================
        // after reset both leaves read 0 (defined, not X)
        read_addr1 = 4'd1; read_addr2 = 4'd2; read_addr3 = 4'd3;
        #1;
        check_equiv("equiv-post-reset-zero");

        // stream of writes; both leaves clock identically while active
        drive_write(4'd1,  32'hA5A5_0001);
        read_addr1 = 4'd1; #1;
        check_equiv("equiv-write-r1");

        drive_write(4'd2,  32'hDEAD_BEEF);
        read_addr2 = 4'd2; #1;
        check_equiv("equiv-write-r2");

        drive_write(4'd15, 32'h1234_5678);
        read_addr3 = 4'd15; #1;
        check_equiv("equiv-write-r15");

        // r0 hardwired-zero must hold on both
        drive_write(4'd0,  32'hFFFF_FFFF);   // ignored write to r0
        read_addr1 = 4'd0; #1;
        check_equiv("equiv-r0-hardwired-zero");

        // several more writes across the file, checking every cycle
        for (i = 1; i < 8; i = i + 1) begin
            drive_write(i[3:0], (32'hC0DE_0000 | i));
            read_addr1 = i[3:0];
            read_addr2 = (i-1) & 4'hF;
            read_addr3 = (i+1) & 4'hF;
            #1;
            check_equiv("equiv-stream");
        end

        //==============================================================
        // (2) GATING while idle: gated_clk held constant (no toggle)
        //==============================================================
        write_enable = 1'b0;
        req          = 1'b0;         // go idle
        scan_enable  = 1'b0;
        // wait for the hysteresis (HOLD=1) to expire and the gate to close
        for (i = 0; (i < 10) && (clk_en !== 1'b0); i = i + 1)
            @(posedge clk);
        if (clk_en !== 1'b0) begin
            $display("FAIL gating: clk_en never dropped (still %b)", clk_en);
            $fatal;
        end
        // now monitor gated_clk across a multi-cycle idle window: must not move
        @(negedge clk);              // align; gate closed => gated_clk should be 0
        #1;
        if (gated_clk === 1'bx) begin
            $display("FAIL gating: gated_clk is X"); $fatal;
        end
        gclk_prev = gated_clk;
        edges = 0;
        // sample every 1ns across ~4 clk periods
        for (i = 0; i < 40; i = i + 1) begin
            #1;
            if (gated_clk !== gclk_prev) edges = edges + 1;
            gclk_prev = gated_clk;
        end
        if (edges != 0) begin
            $display("FAIL gating: gated_clk toggled %0d times while idle", edges);
            $fatal;
        end
        pass("gating-gated_clk-frozen-idle");

        //==============================================================
        // (3) SCAN: scan_enable=1 => gated_clk == clk (full toggling),
        //     even while idle (req=0, functional enable would gate).
        //==============================================================
        req         = 1'b0;          // still idle functionally
        scan_enable = 1'b1;          // DFT force-transparent
        @(negedge clk); #1;          // let the low-phase latch capture test_en
        edges = 0;
        gclk_prev = gated_clk;       // low phase -> should be 0
        // sample INSIDE each phase (1ns after the edge, away from the edge race):
        // gated_clk must be HIGH during clk-high and LOW during clk-low == follows clk.
        for (i = 0; i < 8; i = i + 1) begin
            @(posedge clk); #1;
            if (gated_clk !== 1'b1) begin
                $display("FAIL scan: gated_clk(%b) != clk(1) in high phase at t=%0t",
                         gated_clk, $time);
                $fatal;
            end
            if (gated_clk !== gclk_prev) edges = edges + 1;
            gclk_prev = gated_clk;

            @(negedge clk); #1;
            if (gated_clk !== 1'b0) begin
                $display("FAIL scan: gated_clk(%b) != clk(0) in low phase at t=%0t",
                         gated_clk, $time);
                $fatal;
            end
            if (gated_clk !== gclk_prev) edges = edges + 1;
            gclk_prev = gated_clk;
        end
        if (edges < 4) begin
            $display("FAIL scan: gated_clk only toggled %0d times (expected free-running)", edges);
            $fatal;
        end
        pass("scan-gated_clk-follows-clk");
        scan_enable = 1'b0;

        //==============================================================
        // (4) SAFETY: a req pulse forces clk_en(=enable)=1 the SAME cycle,
        //     even starting from a fully gated state.
        //==============================================================
        // ensure we are gated first
        req = 1'b0; scan_enable = 1'b0;
        for (i = 0; (i < 10) && (clk_en !== 1'b0); i = i + 1)
            @(posedge clk);
        if (clk_en !== 1'b0) begin
            $display("FAIL safety: could not reach gated state"); $fatal;
        end
        // assert req mid-cycle; clk_en is combinational => must rise immediately
        @(negedge clk);
        req = 1'b1;
        #1;
        if (clk_en !== 1'b1) begin
            $display("FAIL safety: req=1 but clk_en=%b (pending work gated!)", clk_en);
            $fatal;
        end
        pass("safety-req-forces-enable-same-cycle");

        // and it stays satisfied across the whole active cycle (req >= enable)
        @(posedge clk); #1;
        if (req && (clk_en !== 1'b1)) begin
            $display("FAIL safety: clk_en dropped below req during active cycle");
            $fatal;
        end
        pass("safety-invariant-holds-active-edge");

        //==============================================================
        // done
        //==============================================================
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end

    // global watchdog
    initial begin
        #20000;
        $display("FAIL: timeout");
        $fatal;
    end

endmodule
