`timescale 1ns/1ps
//============================================================================
// mbist_ctrl_tb.v  --  correctness TB for the March C- memory BIST controller
//----------------------------------------------------------------------------
// Proves src/mbist_ctrl.v actually implements March C- against a behavioral
// single-port SRAM, in the two mandated scenarios:
//
//   (1) GOOD SRAM        : a fault-free single-port RAM -> the march runs to
//                          completion with done==1 and fail==0, and the RAM
//                          ends up all-zeros (the final w0 sweep), i.e. every
//                          March read matched.
//   (2) STUCK-AT-0 CELL  : one chosen cell has bit 0 permanently stuck at 0
//                          (writes to it never set that bit).  March C- MUST
//                          detect it: done==1, fail==1, and fail_addr equal to
//                          the stuck cell's address.  The stuck cell is caught
//                          the first time the march reads back a '1' it wrote
//                          (element M1: up r0,w1 -> M2: up r1 ... reads the
//                          bad '1' as '0').
//
// The behavioral RAM models the SAME single-port contract the controller drives
// against (the project's memory.v style): one access per cycle, synchronous
// write on the clock edge, and a COMBINATIONAL read (rdata = RAM[addr]).  Since
// the controller's `addr` is registered, an address driven at edge T is stable
// through cycle T+1, so rdata is valid the cycle after addr -- exactly the
// 1-deep pending-read compare the controller performs.  `fault_en`/`fault_addr`
// force one cell's bit 0 stuck-at-0.  X-aware compares (===) throughout.  Emits
// "ALL N TESTS PASSED" on success; $fatal on any mismatch.
//============================================================================
module mbist_ctrl_tb;

    // ---- DUT geometry ----
    localparam integer DEPTH = 16;
    localparam integer WIDTH = 8;
    localparam integer AW    = 4;            // $clog2(16)
    localparam integer MAXCYC = 10000;

    integer pass_count = 0;
    integer errors     = 0;

    task chk(input cond, input [1023:0] msg);
        begin
            if (cond) pass_count = pass_count + 1;
            else begin
                errors = errors + 1;
                $display("ASSERT FAIL: %0s  (t=%0t)", msg, $time);
                $fatal(1, "mbist_ctrl_tb assertion failed");
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
    reg              start;
    wire             busy;
    wire             done;
    wire             fail;
    wire [AW-1:0]    fail_addr;
    wire [AW-1:0]    addr;
    wire             we;
    wire [WIDTH-1:0] wdata;
    wire [WIDTH-1:0] rdata;                   // driven by the behavioral RAM

    mbist_ctrl #(.DEPTH(DEPTH), .WIDTH(WIDTH)) dut (
        .clk(clk), .rst(rst),
        .start(start), .busy(busy), .done(done),
        .fail(fail), .fail_addr(fail_addr),
        .addr(addr), .we(we), .wdata(wdata), .rdata(rdata)
    );

    // ------------------------------------------------------------------
    // Behavioral single-port SRAM (COMBINATIONAL read, synchronous write) --
    // the project's memory.v contract (assign data_out = sram[addr]):
    //   * WRITE  : synchronous, on the clock edge, when we==1.
    //   * READ   : rdata = RAM[addr] combinationally.  Because the controller's
    //              `addr` is a registered output, a read address driven at edge
    //              T is stable through cycle T+1, so RAM[addr] (hence rdata) is
    //              valid during cycle T+1 -- exactly the 1-cycle-later compare
    //              the controller performs against its 1-deep pending read.
    //   * FAULT  : if fault_en, cell fault_addr has bit 0 stuck-at-0 -- any word
    //              written there is stored with bit 0 forced to 0.
    // The RAM is preloaded to a non-zero garbage pattern so a MISSED write (a
    // cell the march never zeroed) would surface as a mismatch, not read as a
    // convenient 0.
    // ------------------------------------------------------------------
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg             fault_en;
    reg [AW-1:0]    fault_addr;

    reg [WIDTH-1:0] wr_store;                 // word actually stored (post-fault)
    integer         mi;

    // combinational read (matches memory.v: assign data_out = sram[addr])
    assign rdata = mem[addr];

    always @(posedge clk) begin
        if (we) begin
            wr_store = wdata;
            if (fault_en && (addr == fault_addr))
                wr_store[0] = 1'b0;           // stuck-at-0 on bit 0 of this cell
            mem[addr] <= wr_store;
        end
    end

    // preload the RAM to a non-zero garbage pattern for a fresh scenario
    task preload_ram;
        begin
            for (mi = 0; mi < DEPTH; mi = mi + 1)
                mem[mi] = 8'hA5 ^ mi[7:0];    // arbitrary non-zero seed
        end
    endtask

    // ------------------------------------------------------------------
    // run one march to completion (bounded), return the observed result
    // ------------------------------------------------------------------
    integer cyc;
    task run_march;
        output        o_fail;
        output [AW-1:0] o_fail_addr;
        begin
            // pulse start for exactly one cycle
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;
            // wait for done (level), bounded
            cyc = 0;
            while (!done && cyc < MAXCYC) begin
                @(negedge clk); cyc = cyc + 1;
            end
            o_fail      = fail;
            o_fail_addr = fail_addr;
        end
    endtask

    // ------------------------------------------------------------------
    // stimulus / checks
    // ------------------------------------------------------------------
    reg              r_fail;
    reg [AW-1:0]     r_fail_addr;
    integer          j;
    reg              all_zero;

    initial begin
        rst      = 1'b1;
        start    = 1'b0;
        fault_en = 1'b0;
        fault_addr = {AW{1'b0}};
        preload_ram;

        // synchronous reset for a few cycles
        repeat (3) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        //==============================================================
        // SCENARIO 1 : GOOD SRAM  -> done, fail == 0, RAM ends all-zero
        //==============================================================
        fault_en = 1'b0;
        preload_ram;
        run_march(r_fail, r_fail_addr);

        chk(done  === 1'b1, "good: done asserted");
        chk(r_fail === 1'b0, "good: fail deasserted (no fault)");
        chk(busy  === 1'b0, "good: busy dropped at completion");
        // The final March C- element is a DOWN w0 sweep only after M4 (up r1,w0
        // -> down r0 w1 -> down r1 w0 -> down r0); the RAM's last write pass is
        // w0, so every cell must be 0 at the end.
        all_zero = 1'b1;
        for (j = 0; j < DEPTH; j = j + 1)
            if (mem[j] !== {WIDTH{1'b0}}) all_zero = 1'b0;
        chk(all_zero, "good: RAM all-zero after the march's final w0 sweep");

        //==============================================================
        // SCENARIO 2 : STUCK-AT-0 CELL -> fail==1 with correct fail_addr
        //==============================================================
        // pick a representative interior cell
        fault_addr = 4'd10;
        fault_en   = 1'b1;
        preload_ram;
        run_march(r_fail, r_fail_addr);

        chk(done   === 1'b1, "stuck0: done asserted");
        chk(r_fail === 1'b1, "stuck0: fail asserted (defect present)");
        chk(r_fail_addr === fault_addr,
            "stuck0: fail_addr == stuck cell address");

        //==============================================================
        // SCENARIO 2b : a DIFFERENT stuck cell (address 0, the boundary) ->
        // fail_addr must track the actual defect, not a fixed value.
        //==============================================================
        rst = 1'b1; @(negedge clk); rst = 1'b0; @(negedge clk);
        fault_addr = 4'd0;
        fault_en   = 1'b1;
        preload_ram;
        run_march(r_fail, r_fail_addr);

        chk(done   === 1'b1, "stuck0@0: done asserted");
        chk(r_fail === 1'b1, "stuck0@0: fail asserted");
        chk(r_fail_addr === 4'd0, "stuck0@0: fail_addr == 0 (boundary cell)");

        //==============================================================
        // SCENARIO 3 : re-run GOOD after a fault run -> engine recovers, no
        // sticky fail (fail must clear on the next start).
        //==============================================================
        fault_en = 1'b0;
        preload_ram;
        run_march(r_fail, r_fail_addr);
        chk(done   === 1'b1, "recover: done asserted");
        chk(r_fail === 1'b0, "recover: fail cleared on new run");

        //==============================================================
        // verdict
        //==============================================================
        if (errors == 0)
            $display("ALL %0d TESTS PASSED", pass_count);
        else begin
            $display("%0d of %0d CHECKS FAILED", errors, pass_count + errors);
            $fatal;
        end
        $finish;
    end

    // global watchdog
    initial begin
        #(MAXCYC*20);
        $display("TIMEOUT: march did not complete");
        $fatal;
    end

endmodule
