`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// register_file_tb.v  --  unit testbench for register_file.v  (SPEC.md §6.1)
//----------------------------------------------------------------------------
// SELF-CONTAINED: instantiates ONLY the DUT (register_file).  No other src/
// modules are pulled in; there is no external memory to model for this unit.
//
// INDEPENDENT GOLDEN MODEL
//   The reference is a behavioral `gold[]` array maintained by the TB with a
//   DIFFERENT update rule than the DUT's clocked process:
//     * the TB applies the SAME write contract (write_enable & addr!=0 commits)
//       but as a plain procedural array assignment timed by the TB clock, not
//       by replicating the DUT's RTL;
//     * r0 is enforced on the READ side by the golden's `rd_gold` function,
//       which returns 0 for index 0 unconditionally -- so even a DUT that
//       managed to store a non-zero value into cell 0 would be caught.
//   Comparison is BIT-EXACT (this is an exact integer storage unit; no
//   fixed-point approximation is involved -> no tolerance).
//
// COVERAGE
//   Directed:
//     - synchronous reset clears all 16 registers to 0 (read back all ports)
//     - write then read-back every register r1..r15 (distinct payloads)
//     - write to r0 is ignored; r0 reads 0 on all three ports
//     - max-value payload (0xFFFFFFFF) round-trips through r1..r15
//     - simultaneous independent reads: the 3 ports return 3 different regs
//     - write_enable low => no state change (hold)
//     - same-cycle write+read of a different register (port independence)
//     - reset re-asserted mid-stream clears everything again
//   Constrained-random:
//     - >=200 seeded-$random vectors driving {addr1,addr2,addr3,waddr,wdata,we,
//       rst}; every cycle all three read ports are checked vs the golden.
//
// Counts pass/fail, prints "ALL <N> TESTS PASSED", $fatal on any mismatch.
//============================================================================
module register_file_tb;

    localparam integer REGS   = `RF_REGS;   // 16
    localparam integer IDX_W  = `RF_IDX_W;  // 4
    localparam integer WORD_W = `XLEN;      // 32
    localparam integer NRAND  = 400;        // constrained-random vectors (>=200)

    // ---- DUT I/O ----
    reg                  clk;
    reg                  rst;
    reg  [IDX_W-1:0]     read_addr1, read_addr2, read_addr3;
    reg  [IDX_W-1:0]     write_addr;
    reg  [WORD_W-1:0]    write_data;
    reg                  write_enable;
    wire [WORD_W-1:0]    read_data1, read_data2, read_data3;

    // ---- bookkeeping ----
    integer test_count;
    integer fail_count;
    integer k;
    integer seed;

    // ---- INDEPENDENT golden state ----
    reg [WORD_W-1:0] gold [0:REGS-1];

    // golden read with hardwired-zero r0 enforced independently of storage
    function [WORD_W-1:0] rd_gold;
        input [IDX_W-1:0] a;
        begin
            if (a == {IDX_W{1'b0}})
                rd_gold = {WORD_W{1'b0}};
            else
                rd_gold = gold[a];
        end
    endfunction

    // ---- DUT ----
    register_file dut (
        .clk          (clk),
        .rst          (rst),
        .read_addr1   (read_addr1),
        .read_addr2   (read_addr2),
        .read_addr3   (read_addr3),
        .write_addr   (write_addr),
        .write_data   (write_data),
        .write_enable (write_enable),
        .read_data1   (read_data1),
        .read_data2   (read_data2),
        .read_data3   (read_data3)
    );

    // ---- clock: 10ns period ----
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ---- golden update: same write contract as DUT, modeled independently ----
    // Timed off the same posedge; uses the CURRENT inputs (which are held stable
    // by the TB across the edge).  This is a behavioral mirror of the CONTRACT,
    // not of the RTL.
    integer g;
    always @(posedge clk) begin
        if (rst) begin
            for (g = 0; g < REGS; g = g + 1)
                gold[g] <= {WORD_W{1'b0}};
        end else if (write_enable && (write_addr != {IDX_W{1'b0}})) begin
            gold[write_addr] <= write_data;
        end
    end

    // ---- combinational read checker ----
    // Compare every read port against the golden.  Reads are combinational, so
    // we sample after inputs/state have settled (called from the directed and
    // random sequences just before the next edge).
    task check_reads;
        input [127:0] tag;  // up to 16 ASCII chars label
        reg [WORD_W-1:0] e1, e2, e3;
        begin
            e1 = rd_gold(read_addr1);
            e2 = rd_gold(read_addr2);
            e3 = rd_gold(read_addr3);
            test_count = test_count + 3;
            if (read_data1 !== e1) begin
                fail_count = fail_count + 1;
                $display("FAIL [%0s] port1 addr=%0d got=%h exp=%h @t=%0t",
                         tag, read_addr1, read_data1, e1, $time);
            end
            if (read_data2 !== e2) begin
                fail_count = fail_count + 1;
                $display("FAIL [%0s] port2 addr=%0d got=%h exp=%h @t=%0t",
                         tag, read_addr2, read_data2, e2, $time);
            end
            if (read_data3 !== e3) begin
                fail_count = fail_count + 1;
                $display("FAIL [%0s] port3 addr=%0d got=%h exp=%h @t=%0t",
                         tag, read_addr3, read_data3, e3, $time);
            end
        end
    endtask

    // Apply one write vector and advance one clock.  Inputs are set before the
    // edge and held; both DUT and golden commit on the same posedge.
    task do_write;
        input [IDX_W-1:0]  wa;
        input [WORD_W-1:0] wd;
        input              we;
        begin
            @(negedge clk);
            write_addr   = wa;
            write_data   = wd;
            write_enable = we;
            @(posedge clk);   // DUT + golden both commit here
            #1;               // let combinational reads settle
        end
    endtask

    // Drive read addresses (combinational; no clock needed) then check.
    task read_and_check;
        input [IDX_W-1:0] a1;
        input [IDX_W-1:0] a2;
        input [IDX_W-1:0] a3;
        input [127:0]     tag;
        begin
            read_addr1 = a1;
            read_addr2 = a2;
            read_addr3 = a3;
            #1;  // settle combinational reads
            check_reads(tag);
        end
    endtask

    integer rv;  // scratch for random
    reg [IDX_W-1:0] ra1, ra2, ra3, wa_r;
    reg [WORD_W-1:0] wd_r;
    reg              we_r, rst_r;

    initial begin
        test_count = 0;
        fail_count = 0;
        seed       = 32'hDEAD_BEEF;

        // init inputs
        rst          = 1'b1;
        read_addr1   = {IDX_W{1'b0}};
        read_addr2   = {IDX_W{1'b0}};
        read_addr3   = {IDX_W{1'b0}};
        write_addr   = {IDX_W{1'b0}};
        write_data   = {WORD_W{1'b0}};
        write_enable = 1'b0;

        //------------------------------------------------------------------
        // DIRECTED 1: synchronous reset clears all registers to 0
        //------------------------------------------------------------------
        @(negedge clk);
        rst = 1'b1;
        @(posedge clk); #1;   // reset commits
        @(negedge clk);
        rst = 1'b0;
        #1;
        // read back every register on all three ports -> must be 0
        for (k = 0; k < REGS; k = k + 1) begin
            read_and_check(k[IDX_W-1:0], k[IDX_W-1:0], k[IDX_W-1:0], "reset0");
        end

        //------------------------------------------------------------------
        // DIRECTED 2: write then read-back each register r1..r15
        //------------------------------------------------------------------
        for (k = 1; k < REGS; k = k + 1) begin
            // distinct, non-trivial payload per register
            do_write(k[IDX_W-1:0], (32'hA5A5_0000 | k[WORD_W-1:0]), 1'b1);
            read_and_check(k[IDX_W-1:0], {IDX_W{1'b0}}, k[IDX_W-1:0], "wr_rd");
        end

        //------------------------------------------------------------------
        // DIRECTED 3: write to r0 is IGNORED; r0 reads 0 on all ports
        //------------------------------------------------------------------
        do_write({IDX_W{1'b0}}, 32'hFFFF_FFFF, 1'b1);  // attempt write r0
        read_and_check({IDX_W{1'b0}}, {IDX_W{1'b0}}, {IDX_W{1'b0}}, "r0_wr_ign");
        // and r0 stays 0 even when read alongside a real reg
        read_and_check({IDX_W{1'b0}}, 4'd1, {IDX_W{1'b0}}, "r0_mixed");

        //------------------------------------------------------------------
        // DIRECTED 4: max-value payload round-trips through r1..r15
        //------------------------------------------------------------------
        for (k = 1; k < REGS; k = k + 1) begin
            do_write(k[IDX_W-1:0], 32'hFFFF_FFFF, 1'b1);
            read_and_check(k[IDX_W-1:0], k[IDX_W-1:0], {IDX_W{1'b0}}, "maxval");
        end

        //------------------------------------------------------------------
        // DIRECTED 5: simultaneous read-port independence
        //   load r3,r7,r11 with distinct values, read all three at once
        //------------------------------------------------------------------
        do_write(4'd3,  32'h1111_1111, 1'b1);
        do_write(4'd7,  32'h2222_2222, 1'b1);
        do_write(4'd11, 32'h3333_3333, 1'b1);
        read_and_check(4'd3, 4'd7, 4'd11, "3port_indep");
        // permute the ports -> still independent
        read_and_check(4'd11, 4'd3, 4'd7, "3port_perm");

        //------------------------------------------------------------------
        // DIRECTED 6: write_enable low => HOLD (no state change)
        //------------------------------------------------------------------
        do_write(4'd5, 32'hCAFE_BABE, 1'b1);   // set r5
        do_write(4'd5, 32'h0000_0000, 1'b0);   // we=0: must NOT overwrite
        read_and_check(4'd5, 4'd5, 4'd5, "we_low_hold");

        //------------------------------------------------------------------
        // DIRECTED 7: same-cycle write to one reg, read of a DIFFERENT reg
        //   (port independence: reads are combinational on current state)
        //------------------------------------------------------------------
        do_write(4'd9, 32'hDEAD_0009, 1'b1);   // r9 known
        @(negedge clk);
        write_addr   = 4'd2;                    // writing r2 this cycle
        write_data   = 32'h0BAD_0002;
        write_enable = 1'b1;
        read_addr1   = 4'd9;                    // read r9 (old, stable) ...
        read_addr2   = 4'd0;                    // ... and r0 ...
        read_addr3   = 4'd9;                    // ... and r9 again
        #1;
        check_reads("wr_rd_diff");              // r9/r0 unaffected by r2 write
        @(posedge clk); #1;                     // r2 write commits

        //------------------------------------------------------------------
        // DIRECTED 8: reset re-asserted mid-stream clears everything
        //------------------------------------------------------------------
        @(negedge clk);
        rst = 1'b1;
        @(posedge clk); #1;
        @(negedge clk);
        rst = 1'b0;
        #1;
        for (k = 0; k < REGS; k = k + 1) begin
            read_and_check(k[IDX_W-1:0], k[IDX_W-1:0], k[IDX_W-1:0], "reset_mid");
        end

        //------------------------------------------------------------------
        // CONSTRAINED-RANDOM: >=200 seeded vectors
        //   Each cycle: random rst (rare), random write {addr,data,we}, random
        //   read addresses on all 3 ports.  Golden and DUT commit on the same
        //   edge; all three reads checked after settle.
        //------------------------------------------------------------------
        for (rv = 0; rv < NRAND; rv = rv + 1) begin
            // ~1/16 chance of a reset pulse to exercise the clear path randomly
            rst_r = ($random(seed) % 16 == 0) ? 1'b1 : 1'b0;
            wa_r  = $random(seed);
            wd_r  = $random(seed);
            we_r  = $random(seed);
            ra1   = $random(seed);
            ra2   = $random(seed);
            ra3   = $random(seed);

            @(negedge clk);
            rst          = rst_r;
            write_addr   = wa_r;
            write_data   = wd_r;
            write_enable = we_r;
            @(posedge clk);   // DUT + golden commit (reset or gated write)
            #1;
            rst = 1'b0;       // deassert before reads so we test stored state

            // combinational reads on the resulting state
            read_addr1 = ra1;
            read_addr2 = ra2;
            read_addr3 = ra3;
            #1;
            check_reads("rand");
        end

        //------------------------------------------------------------------
        // SUMMARY
        //------------------------------------------------------------------
        if (fail_count != 0) begin
            $display("REGISTER_FILE: %0d/%0d checks FAILED", fail_count, test_count);
            $fatal(1, "register_file_tb: %0d mismatches", fail_count);
        end else begin
            $display("ALL %0d TESTS PASSED", test_count);
        end
        $finish;
    end

endmodule
