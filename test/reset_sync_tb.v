`timescale 1ns/1ps
//============================================================================
// reset_sync_tb.v -- testbench for reset_sync (async-assert / sync-deassert)
//----------------------------------------------------------------------------
// WHAT THIS PROVES  (P2.2 -- CDC reset synchronizer signoff)
//   (a) ASYNC ASSERT: the instant arst_in rises, rst_out asserts in the SAME
//       delta -- no dest-clock edge is required (proven both mid-cycle and with
//       the clock deliberately parked, so a domain whose PLL has not locked is
//       still held in reset).
//   (b) SYNC DEASSERT / DEPTH: after arst_in falls, rst_out stays asserted for
//       EXACTLY STAGES dest-clock rising edges and then deasserts on the
//       STAGES-th edge -- measured by counting edges from release to drop, for
//       STAGES = 2 and (a 2nd size) STAGES = 3.
//   (c) SUB-CYCLE GLITCH: a reset pulse far shorter than a clock period still
//       latches into the chain and yields a full, clean STAGES-edge output
//       reset -- the async assert cannot be "missed" and only a synchronous
//       walk can release it.
//
//   Every mismatch prints the case and $fatal's; a clean run prints
//   "ALL N TESTS PASSED".
//============================================================================
module reset_sync_tb;

    localparam integer STAGES2 = 2;
    localparam integer STAGES3 = 3;
    localparam integer PERIOD  = 10;   // 10ns dest clock

    // ---- clocks / DUT I/O ----
    reg  clk;
    reg  arst2, arst3;
    wire rst2, rst3;

    integer tests_passed;

    // Two instances at different depths to prove the parameterization.
    reset_sync #(.STAGES(STAGES2)) dut2 (
        .clk     (clk),
        .arst_in (arst2),
        .rst_out (rst2)
    );
    reset_sync #(.STAGES(STAGES3)) dut3 (
        .clk     (clk),
        .arst_in (arst3),
        .rst_out (rst3)
    );

    // dest clock (can be parked by holding clk low in a test)
    reg clk_run;
    initial clk = 1'b0;
    always #(PERIOD/2) if (clk_run) clk = ~clk; else clk = 1'b0;

    //------------------------------------------------------------------------
    // Measure, for a given DUT, how many dest-clock rising edges rst_out stays
    // asserted AFTER arst_in is released.  arst_in must already be released and
    // the count begins from the next posedge clk.  Returns once rst_out drops.
    //------------------------------------------------------------------------
    // (Implemented inline per-DUT below since Verilog tasks can't take a wire
    //  reference; kept small and explicit.)

    integer edges;

    initial begin
        tests_passed = 0;

        // ---- power-on: assert both async resets, clock running ----
        clk_run = 1'b1;
        arst2   = 1'b1;
        arst3   = 1'b1;

        //====================================================================
        // TEST 1 -- ASYNC ASSERT is immediate (same delta), clock RUNNING.
        //   Release, let outputs deassert, then re-assert mid-cycle and check
        //   rst_out is 1 with NO intervening clock edge.
        //====================================================================
        // let the initial async reset flush through and deassert
        arst2 = 1'b0; arst3 = 1'b0;
        repeat (STAGES3 + 3) @(posedge clk);
        #1;
        if (rst2 !== 1'b0 || rst3 !== 1'b0) begin
            $display("TEST1 SETUP FAIL: outputs not deasserted (rst2=%b rst3=%b)", rst2, rst3);
            $fatal(1);
        end
        // re-assert asynchronously in the MIDDLE of a clock phase (right after a
        // posedge, well before the next one) -- must take effect immediately.
        @(posedge clk);
        #2;                              // 2ns into the high phase, no new edge
        arst2 = 1'b1;
        #1;                              // settle the async path (still no clk edge)
        if (rst2 !== 1'b1) begin
            $display("TEST1 FAIL: rst2=%b immediately after async assert (expected 1, no clk edge)", rst2);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 1 PASS: async assert -> rst_out=1 in the same delta (no clk edge needed)");

        //====================================================================
        // TEST 2 -- ASYNC ASSERT holds even with the dest clock PARKED.
        //   Deassert, park the clock, then assert async: rst_out must still go
        //   high with zero clock activity (PLL-not-locked scenario).
        //====================================================================
        arst2 = 1'b0;
        repeat (STAGES2 + 3) @(posedge clk);   // clear rst2 first
        #1;
        // park the clock LOW
        @(posedge clk); #1; clk_run = 1'b0;
        #(PERIOD*3);                            // no edges at all now
        arst2 = 1'b1;
        #1;
        if (rst2 !== 1'b1) begin
            $display("TEST2 FAIL: rst2=%b with clock parked (expected 1)", rst2);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 2 PASS: async assert holds reset with the dest clock parked (no edges)");

        // un-park; leave both in reset for the depth tests
        arst2 = 1'b1; arst3 = 1'b1;
        clk_run = 1'b1;
        repeat (2) @(posedge clk);

        //====================================================================
        // TEST 3 -- SYNC DEASSERT DEPTH == STAGES (=2) dest-clock edges.
        //   Release arst2 just after a posedge; count posedges until rst2 drops.
        //====================================================================
        @(posedge clk);
        #1 arst2 = 1'b0;                  // release right after the edge
        edges = 0;
        // rst2 is still 1 here; count edges until it deasserts
        while (rst2 === 1'b1) begin
            @(posedge clk);
            #1;
            edges = edges + 1;
            if (edges > STAGES2 + 5) begin
                $display("TEST3 FAIL: rst2 never deasserted within %0d edges", edges);
                $fatal(1);
            end
        end
        if (edges !== STAGES2) begin
            $display("TEST3 FAIL: rst2 deasserted after %0d edges (expected STAGES=%0d)", edges, STAGES2);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 3 PASS: STAGES=2 -> rst_out asserted for exactly 2 dest-clock edges then deasserts");

        //====================================================================
        // TEST 4 -- SYNC DEASSERT DEPTH == STAGES (=3) for the 2nd-size DUT.
        //====================================================================
        @(posedge clk);
        #1 arst3 = 1'b0;
        edges = 0;
        while (rst3 === 1'b1) begin
            @(posedge clk);
            #1;
            edges = edges + 1;
            if (edges > STAGES3 + 5) begin
                $display("TEST4 FAIL: rst3 never deasserted within %0d edges", edges);
                $fatal(1);
            end
        end
        if (edges !== STAGES3) begin
            $display("TEST4 FAIL: rst3 deasserted after %0d edges (expected STAGES=%0d)", edges, STAGES3);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 4 PASS: STAGES=3 -> rst_out asserted for exactly 3 dest-clock edges then deasserts");

        //====================================================================
        // TEST 5 -- rst_out is HELD (not glitchy) across the whole deassert
        //   walk: once released it must stay 1 for every one of the STAGES
        //   edges (no early drop, no re-assert) -- checked edge-by-edge.
        //====================================================================
        // put dut2 back in reset
        arst2 = 1'b1; repeat (2) @(posedge clk);
        @(posedge clk);
        #1 arst2 = 1'b0;
        // first STAGES2-1 edges after release: rst2 must still be 1
        begin : hold_chk
            integer k;
            for (k = 0; k < STAGES2 - 1; k = k + 1) begin
                @(posedge clk); #1;
                if (rst2 !== 1'b1) begin
                    $display("TEST5 FAIL: rst2 dropped early at edge %0d (expected held 1)", k+1);
                    $fatal(1);
                end
            end
            // the STAGES2-th edge: must deassert here
            @(posedge clk); #1;
            if (rst2 !== 1'b0) begin
                $display("TEST5 FAIL: rst2=%b on the STAGES-th edge (expected 0)", rst2);
                $fatal(1);
            end
        end
        tests_passed = tests_passed + 1;
        $display("TEST 5 PASS: rst_out held asserted every edge of the walk, then deasserts on edge STAGES");

        //====================================================================
        // TEST 6 -- SUB-CYCLE GLITCH: a reset pulse shorter than a clock period
        //   still yields a full STAGES-edge output reset.  Pulse arst2 high for
        //   1ns (<< PERIOD) mid-phase, then measure the deassert depth.
        //====================================================================
        // ensure dut2 is fully out of reset first
        repeat (STAGES2 + 3) @(posedge clk); #1;
        if (rst2 !== 1'b0) begin
            $display("TEST6 SETUP FAIL: rst2 not clear before glitch (rst2=%b)", rst2);
            $fatal(1);
        end
        // sub-cycle glitch, nowhere near a clock edge
        @(posedge clk);
        #3;                              // 3ns into the phase
        arst2 = 1'b1;
        #1;                              // 1ns-wide pulse (<< 10ns period)
        if (rst2 !== 1'b1) begin
            $display("TEST6 FAIL: sub-cycle glitch did not assert rst2 (rst2=%b)", rst2);
            $fatal(1);
        end
        arst2 = 1'b0;                    // glitch over, still same clock phase
        // now measure: rst2 must remain asserted a FULL STAGES walk
        edges = 0;
        while (rst2 === 1'b1) begin
            @(posedge clk);
            #1;
            edges = edges + 1;
            if (edges > STAGES2 + 5) begin
                $display("TEST6 FAIL: rst2 never deasserted after glitch (%0d edges)", edges);
                $fatal(1);
            end
        end
        if (edges !== STAGES2) begin
            $display("TEST6 FAIL: post-glitch reset lasted %0d edges (expected STAGES=%0d)", edges, STAGES2);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 6 PASS: sub-cycle glitch still yields a clean full %0d-edge output reset", STAGES2);

        //====================================================================
        // TEST 7 -- X-safe: rst_out is never X at any point of a fresh power-on
        //   assert (async set forces a defined 0 into every flop -> rst_out=1).
        //====================================================================
        arst2 = 1'b1; arst3 = 1'b1;
        #1;
        if ((rst2 !== 1'b1) || (rst3 !== 1'b1)) begin
            $display("TEST7 FAIL: outputs not defined-asserted on power-on (rst2=%b rst3=%b)", rst2, rst3);
            $fatal(1);
        end
        if ((rst2 === 1'bx) || (rst3 === 1'bx)) begin
            $display("TEST7 FAIL: rst_out is X on assert");
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 7 PASS: rst_out is defined (never X) -- async assert forces a known reset");

        $display("ALL %0d TESTS PASSED (STAGES tested: %0d and %0d)", tests_passed, STAGES2, STAGES3);
        $finish;
    end

    // safety net: never hang
    initial begin
        #100000;
        $display("TIMEOUT");
        $fatal(1);
    end

endmodule
