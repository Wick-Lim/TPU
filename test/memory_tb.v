`include "tpu_defs.vh"
`timescale 1ns/1ps
//============================================================================
// memory_tb  --  self-checking unit testbench for memory.v (DMEM 256 x 32)
//----------------------------------------------------------------------------
// Verifies the v2.0 DMEM contract (SPEC §2, §5.5):
//   * 256 x 32-bit, 8-bit address.
//   * COMBINATIONAL read on the primary port AND a second (DMA-source) read
//     port -- two independent reads in the same cycle.
//   * SYNCHRONOUS single-port write.
//   * SYNCHRONOUS reset clears every word to 0.
//   * Boundary addresses 0 and 255 behave like any other word.
//
// INDEPENDENT GOLDEN MODEL:
//   The golden is a separate associative-array-free flat `reg`-array shadow
//   ("expected[]") maintained in the TB by a DIFFERENT mechanism than the DUT:
//   the TB applies the SAME architectural write/reset events to `expected[]`
//   one statement at a time in straight-line procedural code, with NO sharing
//   of the DUT's port/array logic.  Reads from the DUT are compared bit-exact
//   against `expected[]`.  Because DMEM is an exact integer store (no Q-format,
//   no LUT), the comparison is bit-exact (tolerance = 0 LSB).
//
// All checks are bit-exact; $fatal on any mismatch; prints "ALL <N> TESTS
// PASSED" at the end with zero failures.
//============================================================================
module memory_tb;

    // ---- DUT parameters mirrored locally (read-only from tpu_defs) ----
    localparam integer DEPTH  = `DMEM_DEPTH;     // 256
    localparam integer AW     = `DMEM_ADDR_W;    // 8
    localparam integer DW     = `XLEN;           // 32
    localparam integer NRAND  = 300;             // constrained-random vectors

    // ---- DUT ports ----
    reg              clk;
    reg              rst;
    reg  [AW-1:0]    addr;
    reg  [DW-1:0]    data_in;
    reg              write_enable;
    wire [DW-1:0]    data_out;
    reg  [AW-1:0]    raddr2;
    wire [DW-1:0]    rdata2;

    // ---- DUT ----
    memory dut (
        .clk          (clk),
        .rst          (rst),
        .addr         (addr),
        .data_in      (data_in),
        .write_enable (write_enable),
        .data_out     (data_out),
        .raddr2       (raddr2),
        .rdata2       (rdata2)
    );

    // ---- Independent golden shadow store ----
    reg [DW-1:0] expected [0:DEPTH-1];

    // ---- Bookkeeping ----
    integer tests;     // number of assertion checks performed
    integer errors;
    integer i;
    integer k;
    integer mirror;

    // ---- 10ns clock ----
    initial clk = 1'b0;
    always #5 clk = ~clk;

    //------------------------------------------------------------------
    // Helpers
    //------------------------------------------------------------------

    // Drive a synchronous write to BOTH the DUT and the golden shadow.
    // The shadow is updated by an INDEPENDENT plain array store -- it does not
    // touch the DUT's internals.
    task do_write(input [AW-1:0] a, input [DW-1:0] d);
        begin
            @(negedge clk);
            addr         = a;
            data_in      = d;
            write_enable = 1'b1;
            @(posedge clk);             // DUT latches here
            #1;                         // settle past the edge
            expected[a]  = d;           // golden: same architectural effect
            @(negedge clk);
            write_enable = 1'b0;
        end
    endtask

    // Combinational read check on the PRIMARY port, bit-exact vs golden.
    task check_read1(input [AW-1:0] a);
        begin
            addr = a;
            #1;                         // allow combinational settle
            tests = tests + 1;
            if (data_out !== expected[a]) begin
                errors = errors + 1;
                $display("FAIL[read1] addr=%0d got=%h exp=%h",
                         a, data_out, expected[a]);
                $fatal(1, "primary-port read mismatch");
            end
        end
    endtask

    // Combinational read check on the SECONDARY (DMA-source) port.
    task check_read2(input [AW-1:0] a);
        begin
            raddr2 = a;
            #1;
            tests = tests + 1;
            if (rdata2 !== expected[a]) begin
                errors = errors + 1;
                $display("FAIL[read2] addr=%0d got=%h exp=%h",
                         a, rdata2, expected[a]);
                $fatal(1, "secondary-port read mismatch");
            end
        end
    endtask

    // Dual-port INDEPENDENT read: drive two DIFFERENT addresses and confirm the
    // two ports return their own words simultaneously (no cross-port coupling).
    task check_dual(input [AW-1:0] a1, input [AW-1:0] a2);
        begin
            addr   = a1;
            raddr2 = a2;
            #1;
            tests = tests + 1;
            if (data_out !== expected[a1]) begin
                errors = errors + 1;
                $display("FAIL[dual.p1] a1=%0d got=%h exp=%h",
                         a1, data_out, expected[a1]);
                $fatal(1, "dual-port primary mismatch");
            end
            tests = tests + 1;
            if (rdata2 !== expected[a2]) begin
                errors = errors + 1;
                $display("FAIL[dual.p2] a2=%0d got=%h exp=%h",
                         a2, rdata2, expected[a2]);
                $fatal(1, "dual-port secondary mismatch");
            end
        end
    endtask

    // Synchronous reset: assert rst across an edge, then confirm DUT==0 and zero
    // the golden shadow the same way.
    task do_reset;
        begin
            @(negedge clk);
            rst          = 1'b1;
            write_enable = 1'b0;
            @(posedge clk);
            #1;
            for (k = 0; k < DEPTH; k = k + 1)
                expected[k] = {DW{1'b0}};
            @(negedge clk);
            rst = 1'b0;
        end
    endtask

    //------------------------------------------------------------------
    // Stimulus
    //------------------------------------------------------------------
    integer seed;
    reg [AW-1:0] ra;
    reg [AW-1:0] rb;
    reg [DW-1:0] rd;

    initial begin
        tests        = 0;
        errors       = 0;
        addr         = {AW{1'b0}};
        raddr2       = {AW{1'b0}};
        data_in      = {DW{1'b0}};
        write_enable = 1'b0;
        rst          = 1'b0;
        seed         = 32'hC0FFEE01;
        for (k = 0; k < DEPTH; k = k + 1)
            expected[k] = {DW{1'b0}};

        // --- 1. Reset clears every word to 0 (and golden agrees) ----------
        do_reset;
        for (i = 0; i < DEPTH; i = i + 1)
            check_read1(i[AW-1:0]);
        // confirm secondary port also reads all zeros after reset
        for (i = 0; i < DEPTH; i = i + 1)
            check_read2(i[AW-1:0]);

        // --- 2. Boundary addresses 0 and 255: write distinct, read back ----
        do_write({AW{1'b0}},          32'hDEAD_BEEF);   // addr 0
        do_write({AW{1'b1}},          32'h1234_5678);   // addr 255
        check_read1({AW{1'b0}});
        check_read1({AW{1'b1}});
        // verify they did not bleed into neighbours 1 and 254
        check_read1(8'd1);
        check_read1(8'd254);

        // --- 3. Address-sweep write then full read-back (every word) -------
        // Write a unique value to each address (value derived from address so
        // any address-decode/aliasing bug surfaces).
        for (i = 0; i < DEPTH; i = i + 1)
            do_write(i[AW-1:0], (32'hA5A5_0000 | i[DW-1:0]) ^ {i[7:0], 24'd0});
        for (i = 0; i < DEPTH; i = i + 1)
            check_read1(i[AW-1:0]);
        for (i = 0; i < DEPTH; i = i + 1)
            check_read2(i[AW-1:0]);

        // --- 4. Dual-port INDEPENDENT reads (two different addresses) ------
        // primary reads ascending, secondary reads the mirror address.
        for (i = 0; i < DEPTH; i = i + 1) begin
            mirror = DEPTH - 1 - i;
            check_dual(i[AW-1:0], mirror[AW-1:0]);
        end
        // explicit boundary pairing 0 <-> 255 in BOTH orderings
        check_dual({AW{1'b0}}, {AW{1'b1}});
        check_dual({AW{1'b1}}, {AW{1'b0}});

        // --- 5. Same-address both ports (no port should disturb the other) -
        for (i = 0; i < 8; i = i + 1)
            check_dual(i[AW-1:0], i[AW-1:0]);

        // --- 6. Overwrite test: write a word, read; rewrite, read ----------
        do_write(8'd100, 32'hFFFF_FFFF);
        check_read1(8'd100);
        do_write(8'd100, 32'h0000_0000);
        check_read1(8'd100);
        do_write(8'd100, 32'h8000_0001);   // sign-ish / corner bit pattern
        check_read1(8'd100);

        // --- 7. write_enable=0 must NOT modify memory ----------------------
        // Park a known value, then drive data_in/addr with we=0 across an edge.
        do_write(8'd42, 32'hCAFE_F00D);
        @(negedge clk);
        addr         = 8'd42;
        data_in      = 32'h0BAD_0BAD;   // bogus data
        write_enable = 1'b0;            // disabled
        @(posedge clk);
        #1;
        check_read1(8'd42);             // golden still CAFEF00D -> must match

        // --- 8. Constrained-random read/write (seeded) ---------------------
        for (i = 0; i < NRAND; i = i + 1) begin
            ra = $random(seed);          // pseudo-random 8-bit address
            rd = $random(seed);          // pseudo-random 32-bit data
            do_write(ra, rd);
            check_read1(ra);
            // random secondary-port read of an independent address
            rb = $random(seed);
            check_read2(rb);
            // random dual read of two independent addresses
            ra = $random(seed);
            rb = $random(seed);
            check_dual(ra, rb);
        end

        // --- 9. Reset again mid-stream clears state (sticky-free) ----------
        do_reset;
        for (i = 0; i < DEPTH; i = i + 1)
            check_read1(i[AW-1:0]);

        // --- Final tally ---------------------------------------------------
        if (errors != 0) begin
            $display("FAILED: %0d errors out of %0d checks", errors, tests);
            $fatal(1, "memory_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end

    // Safety net: never hang.
    initial begin
        #5_000_000;
        $fatal(1, "TIMEOUT: memory_tb did not finish");
    end

endmodule
