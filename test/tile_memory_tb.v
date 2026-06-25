`include "tpu_defs.vh"
//============================================================================
// tile_memory_tb  --  self-checking unit TB for tile_memory  (SPEC §1.1,§2,§5.5)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL
//   The DUT is the synthesizable 32x128 line memory.  The golden model is a
//   plain behavioral associative-style array `gold[]` maintained ENTIRELY in
//   the TB by a different mechanism than the DUT (the TB updates `gold` from
//   the *intended* transaction semantics -- a value is considered stored the
//   cycle AFTER a posedge with we=1 and no reset -- it never peeks at the DUT's
//   internal `lines[]`).  We then read back through the DUT's real ports and
//   compare bit-exact.  This is an exact-integer storage op, so comparison is
//   bit-exact (no tolerance).
//
// COVERAGE
//   T1  reset clears every line to 0 (read back all 32 lines == 0)
//   T2  write-then-read EVERY line with a distinct directed pattern
//   T3  directed line patterns: all-zero, all-ones, alternating, one-hot,
//       sign-pattern, max/min Q7.8 lanes packed
//   T4  dual-port read INDEPENDENCE: two different addresses read the two
//       distinct lines simultaneously on the same cycle; same address on both
//       ports returns identical data
//   T5  write priority / hold: a non-written line keeps its value across an
//       unrelated write; we=0 holds (no spurious write)
//   T6  mid-stream synchronous reset clears a populated memory back to 0
//   T7  constrained-random addr/data sweep (seeded $random, >=200 vectors)
//       interleaving random writes and dual random reads, checked against gold
//
// GATES: prints "ALL <N> TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module tile_memory_tb;

    // ---- clock / reset ----
    reg clk;
    reg rst;

    // ---- DUT ports ----
    reg  [`TM_IDX_W-1:0] raddr1, raddr2, waddr;
    reg                  we;
    reg  [`LINE_W-1:0]   wdata;
    wire [`LINE_W-1:0]   rdata1, rdata2;

    // ---- DUT ----
    tile_memory dut (
        .clk    (clk),
        .rst    (rst),
        .raddr1 (raddr1),
        .rdata1 (rdata1),
        .raddr2 (raddr2),
        .rdata2 (rdata2),
        .we     (we),
        .waddr  (waddr),
        .wdata  (wdata)
    );

    // ---- INDEPENDENT golden model (TB-maintained, never reads dut.lines) ----
    reg [`LINE_W-1:0] gold [0:`TM_LINES-1];

    integer pass;
    integer fail;
    integer i;
    integer k;

    // 10ns clock.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // Advance one clock: apply the current control inputs across a posedge,
    // then update the golden model with the SAME intended semantics the DUT
    // must implement (reset dominates, else we writes), independently.
    task step;
        begin
            @(posedge clk);
            if (rst) begin
                for (k = 0; k < `TM_LINES; k = k + 1)
                    gold[k] = {`LINE_W{1'b0}};
            end else if (we) begin
                gold[waddr] = wdata;
            end
            #1; // settle past the edge so combinational rdata is stable
        end
    endtask

    // Combinational read check on BOTH ports against the golden model.
    task check_read;
        input [`TM_IDX_W-1:0] a1;
        input [`TM_IDX_W-1:0] a2;
        input [255:0]         tag;
        begin
            raddr1 = a1;
            raddr2 = a2;
            #1; // let combinational reads settle
            if (rdata1 !== gold[a1]) begin
                $display("FAIL[%0s] port1 addr=%0d exp=%032x got=%032x",
                         tag, a1, gold[a1], rdata1);
                fail = fail + 1;
                $fatal(1, "tile_memory read mismatch (port1)");
            end else begin
                pass = pass + 1;
            end
            if (rdata2 !== gold[a2]) begin
                $display("FAIL[%0s] port2 addr=%0d exp=%032x got=%032x",
                         tag, a2, gold[a2], rdata2);
                fail = fail + 1;
                $fatal(1, "tile_memory read mismatch (port2)");
            end else begin
                pass = pass + 1;
            end
        end
    endtask

    // Drive a synchronous write of `d` to line `a`, advancing one clock.
    task do_write;
        input [`TM_IDX_W-1:0] a;
        input [`LINE_W-1:0]   d;
        begin
            we    = 1'b1;
            waddr = a;
            wdata = d;
            step;
            we    = 1'b0;
        end
    endtask

    // Build a 128-bit line of four packed 32-bit lanes.
    function [`LINE_W-1:0] pack4;
        input [31:0] l3, l2, l1, l0;
        begin
            pack4 = {l3, l2, l1, l0};
        end
    endfunction

    integer seed;
    reg [`LINE_W-1:0] rnd;
    reg [`TM_IDX_W-1:0] ra, rb, rw;

    initial begin
        pass = 0;
        fail = 0;
        seed = 32'hC0FFEE01;

        // Idle defaults.
        we     = 1'b0;
        waddr  = {`TM_IDX_W{1'b0}};
        wdata  = {`LINE_W{1'b0}};
        raddr1 = {`TM_IDX_W{1'b0}};
        raddr2 = {`TM_IDX_W{1'b0}};

        // ---------------------------------------------------------------
        // T1: synchronous reset clears all lines to 0.
        // ---------------------------------------------------------------
        rst = 1'b1;
        step;            // one reset edge clears DUT + gold
        step;            // hold reset a second cycle for good measure
        rst = 1'b0;
        for (i = 0; i < `TM_LINES; i = i + 1)
            check_read(i[`TM_IDX_W-1:0], i[`TM_IDX_W-1:0], "T1-reset0");

        // ---------------------------------------------------------------
        // T2: write-then-read EVERY line with a distinct pattern.
        //     Use a per-line value that depends on the index so a swapped
        //     address would be caught.
        // ---------------------------------------------------------------
        for (i = 0; i < `TM_LINES; i = i + 1) begin
            do_write(i[`TM_IDX_W-1:0],
                     pack4(32'hA5A50000 + i, 32'h0BAD0000 + i,
                           32'hDEAD0000 + i, 32'hBEEF0000 + i));
        end
        // Read every line back (dual-ported: port2 reads the mirror-down index).
        for (i = 0; i < `TM_LINES; i = i + 1)
            check_read(i[`TM_IDX_W-1:0],
                       (`TM_LINES-1-i),
                       "T2-wreveryline");

        // ---------------------------------------------------------------
        // T3: directed corner-case line patterns at a fixed line.
        // ---------------------------------------------------------------
        do_write(5'd7, {`LINE_W{1'b0}});                 // all-zero
        check_read(5'd7, 5'd7, "T3-zero");
        do_write(5'd7, {`LINE_W{1'b1}});                 // all-ones
        check_read(5'd7, 5'd7, "T3-ones");
        do_write(5'd7, {(`LINE_W/2){2'b10}});            // alternating 1010..
        check_read(5'd7, 5'd7, "T3-alt10");
        do_write(5'd7, {(`LINE_W/2){2'b01}});            // alternating 0101..
        check_read(5'd7, 5'd7, "T3-alt01");
        do_write(5'd7, {{(`LINE_W-1){1'b0}}, 1'b1});     // one-hot LSB
        check_read(5'd7, 5'd7, "T3-onehotlsb");
        do_write(5'd7, {1'b1, {(`LINE_W-1){1'b0}}});     // one-hot MSB
        check_read(5'd7, 5'd7, "T3-onehotmsb");
        // Max/min Q7.8 lanes packed (sign extremes interpreted as data).
        do_write(5'd7, pack4(32'sh00007FFF, 32'shFFFF8000,
                             32'h00000000, 32'shFFFFFFFF));
        check_read(5'd7, 5'd7, "T3-q78extremes");

        // ---------------------------------------------------------------
        // T4: dual-port read independence.
        //   Populate two lines with clearly different data, read them on the
        //   two ports simultaneously and confirm each port returns ITS line,
        //   not the other.  Then alias both ports to one address.
        // ---------------------------------------------------------------
        do_write(5'd3,  pack4(32'h11111111, 32'h22222222,
                              32'h33333333, 32'h44444444));
        do_write(5'd28, pack4(32'h55555555, 32'h66666666,
                              32'h77777777, 32'h88888888));
        check_read(5'd3, 5'd28, "T4-indep");       // port1!=port2 data
        check_read(5'd28, 5'd3, "T4-swap");        // swap ports, still independent
        check_read(5'd3, 5'd3, "T4-alias");        // same addr -> identical
        if (rdata1 !== rdata2) begin
            $display("FAIL[T4-aliasdata] aliased ports differ %032x %032x",
                     rdata1, rdata2);
            fail = fail + 1;
            $fatal(1, "tile_memory aliased dual-read mismatch");
        end else begin
            pass = pass + 1;
        end

        // ---------------------------------------------------------------
        // T5: hold (we=0) and write isolation.
        //   With we=0, an unrelated address change must NOT alter storage.
        //   A write to one line must NOT disturb its neighbors.
        // ---------------------------------------------------------------
        we    = 1'b0;
        waddr = 5'd3;            // dangle the write addr while we=0
        wdata = {`LINE_W{1'b1}};
        step;                    // we=0 -> nothing stored
        check_read(5'd3, 5'd28, "T5-hold");        // both lines unchanged
        do_write(5'd4, pack4(32'hCAFEBABE, 32'hFEEDFACE,
                             32'h8BADF00D, 32'hABAD1DEA));
        check_read(5'd3, 5'd4, "T5-neighbor3");    // line3 intact, line4 new
        check_read(5'd5, 5'd28, "T5-neighbor5");   // line5 (untouched) intact

        // ---------------------------------------------------------------
        // T6: mid-stream synchronous reset clears a populated memory.
        // ---------------------------------------------------------------
        rst = 1'b1;
        step;
        rst = 1'b0;
        for (i = 0; i < `TM_LINES; i = i + 1)
            check_read(i[`TM_IDX_W-1:0], i[`TM_IDX_W-1:0], "T6-reset0");

        // ---------------------------------------------------------------
        // T7: constrained-random addr/data sweep (>=200 vectors).
        //   Each iteration: optionally write a random line with random data,
        //   then read two random lines on the two ports.  Golden tracks every
        //   committed write; comparison is bit-exact.
        // ---------------------------------------------------------------
        for (k = 0; k < 300; k = k + 1) begin
            // random 128-bit data from four $random words
            rnd = { $random(seed), $random(seed),
                    $random(seed), $random(seed) };
            rw  = $random(seed);
            ra  = $random(seed);
            rb  = $random(seed);

            // ~75% of iterations perform a write, the rest are read-only.
            if ((k % 4) != 0) begin
                do_write(rw, rnd);
            end
            check_read(ra, rb, "T7-rand");
        end

        // ---------------------------------------------------------------
        if (fail != 0) begin
            $display("TILE_MEMORY TB: %0d FAILURES", fail);
            $fatal(1, "tile_memory_tb FAILED");
        end
        $display("ALL %0d TESTS PASSED", pass);
        $finish;
    end
endmodule
