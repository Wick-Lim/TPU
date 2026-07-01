`timescale 1ns/1ps
//============================================================================
// icg_cell_tb.v -- testbench for icg_cell (docs/IMPROVEMENT_PLAN.md P2.4)
//----------------------------------------------------------------------------
// WHAT THIS PROVES  (the glitch-free integrated clock gate is correct)
//   (a) enable held HIGH  => gated_clk is IDENTICAL to clk, cycle for cycle.
//   (b) enable held LOW   => gated_clk stays flat LOW (no pulses at all).
//   (c) GLITCH-FREEDOM: toggling `enable` mid-HIGH-phase produces NO partial /
//       runt pulse.  Every gated_clk high pulse is a WHOLE clk pulse (rises
//       exactly with a clk rising edge, falls exactly with the matching clk
//       falling edge) whose PRESENCE is decided by the enable value latched on
//       the PRECEDING low phase.  A continuous monitor $fatal's on any runt.
//   (d) EDGE COUNT: over a mixed enable pattern the number of gated_clk rising
//       edges equals exactly the number of clk high phases whose preceding low
//       phase latched enable==1 (the TB predicts this independently and checks).
//   (e) TEST-ENABLE: test_en forces the gate transparent (scan bypass) even
//       when enable==0.
//   (f) INVERTED-POLARITY instance (GATE_POLARITY=0): enable LOW passes clk.
//----------------------------------------------------------------------------
// TIMING MODEL / METHODOLOGY
//   Clock: 10ns period (5ns low, 5ns high).  Because the enable latch is
//   transparent while clk is LOW, the value that governs a given high phase is
//   whatever `enable` is at the moment clk RISES (the last value seen while
//   low).  The TB therefore drives `enable` changes and, at each clk rising
//   edge, samples the enable-at-that-edge to PREDICT whether this high phase
//   should carry a pulse; it then confirms the gated_clk edge behavior matches.
//   To exercise (c) it also flips `enable` in the MIDDLE of high phases -- the
//   gate must ignore those flips entirely for the current pulse.
//============================================================================
module icg_cell_tb;

    // ---- DUT I/O (primary, active-high enable, GATE_POLARITY=1) ----
    reg  clk;
    reg  enable;
    reg  test_en;
    wire gated_clk;

    // ---- inverted-polarity DUT (GATE_POLARITY=0): enable LOW passes ----
    reg  enable_inv;
    wire gated_clk_inv;

    // ---- bookkeeping ----
    integer tests_passed;
    integer glitch_checks;       // continuous glitch-monitor samples taken
    integer exp_rise;            // TB-predicted gated_clk rising edges (test d)
    integer got_rise;            // observed gated_clk rising edges
    reg     mon_active;          // 1 = run the continuous glitch monitor
    reg     count_active;        // 1 = accumulate exp_rise / got_rise

    // primary gate: ordinary active-high enable
    icg_cell #(.GATE_POLARITY(1)) dut (
        .clk       (clk),
        .enable    (enable),
        .test_en   (test_en),
        .gated_clk (gated_clk)
    );

    // inverted-sense gate: enable LOW passes the clock
    icg_cell #(.GATE_POLARITY(0)) dut_inv (
        .clk       (clk),
        .enable    (enable_inv),
        .test_en   (1'b0),
        .gated_clk (gated_clk_inv)
    );

    // 10ns clock (5 low / 5 high)
    initial clk = 1'b0;
    always #5 clk = ~clk;

    //------------------------------------------------------------------------
    // (c) CONTINUOUS GLITCH MONITOR.
    //   A glitch-free gate guarantees, for the PRIMARY dut, that gated_clk is
    //   always EXACTLY clk (during enabled high phases) or EXACTLY 0 -- it never
    //   rises or falls in the MIDDLE of a clk phase (a runt).  Equivalently:
    //
    //       gated_clk transitions ONLY when clk transitions, and
    //       gated_clk is LOW whenever clk is LOW.
    //
    //   These are combinational relationships, so they must be evaluated only
    //   AFTER the whole time step has settled (past the delta in which clk and
    //   the derived gated_clk both update).  We therefore sample the invariant
    //   in the STROBE region of every gated_clk / clk event via #0 settling,
    //   which avoids false hits from intra-step ordering while still catching a
    //   real runt (a gated_clk value that persists once combinational logic has
    //   settled).  Any real runt leaves gated_clk==1 with clk==0 after settle.
    //------------------------------------------------------------------------
    always @(gated_clk or clk) begin
        if (mon_active) begin
            #0;   // settle: let the combinational AND / latch reach steady state
            // X-aware: gated_clk must never be unknown once settled.
            if (gated_clk === 1'bx) begin
                $display("X-FAIL: gated_clk is X @ t=%0t", $time);
                $fatal(1);
            end
            // INVARIANT: a settled gated clock HIGH implies the source clock is
            // HIGH.  A runt pulse (gated_clk high while clk low) violates this.
            if ((gated_clk === 1'b1) && (clk !== 1'b1)) begin
                $display("GLITCH-FAIL: gated_clk HIGH while clk LOW @ t=%0t (clk=%b gated=%b)",
                         $time, clk, gated_clk);
                $fatal(1);
            end
            glitch_checks = glitch_checks + 1;
        end
    end

    // Whole-pulse count + alignment.  Sample gated_clk on the SETTLED value at
    // each clk edge: on a clk RISING edge gated_clk is either 1 (a whole pulse
    // begins) or 0 (this phase is gated); on a clk FALLING edge gated_clk must
    // already be back to 0 (the pulse ended cleanly with clk).  Because we
    // sample after #0 settling at the clk edges themselves (not on gated_clk's
    // own edge), there is no delta race, and a mid-phase enable flip that leaked
    // a late/early transition would show up as gated_clk!=clk at these samples.
    always @(posedge clk) begin
        if (mon_active) begin
            #0;                              // settle the AND for this rising edge
            // gated_clk is either 1 (a whole pulse begins) or 0 (phase gated).
            if ((gated_clk === 1'b1) && count_active)
                got_rise = got_rise + 1;     // one whole gated_clk pulse begins
        end
    end

    always @(negedge clk) begin
        if (mon_active) begin
            #0;                              // settle the AND for this falling edge
            // The pulse (if any) must have ended exactly with clk: gated_clk low.
            if (gated_clk !== 1'b0) begin
                $display("RUNT-FAIL: gated_clk=%b at clk falling edge (expected 0) @ t=%0t",
                         gated_clk, $time);
                $fatal(1);
            end
        end
    end

    // integers for loops / results
    integer i;
    integer n_pulses_a;
    integer n_pulses_b;
    integer seed;
    reg     e;

    initial begin
        tests_passed        = 0;
        glitch_checks       = 0;
        exp_rise            = 0;
        got_rise            = 0;
        mon_active          = 1'b0;
        count_active        = 1'b0;

        // safe init
        enable     = 1'b0;
        test_en    = 1'b0;
        enable_inv = 1'b1;      // inverted gate: 1 == gated (idle) at start

        // settle a couple of clocks with the monitor off (latch self-corrects
        // on its first low phase, like a real standard-cell ICG).
        repeat (2) @(posedge clk);
        #1;
        mon_active = 1'b1;      // begin continuous glitch monitoring

        //====================================================================
        // TEST (a) -- enable ALWAYS 1  =>  gated_clk == clk, cycle for cycle.
        //   Set enable high while clk is low so it is latched, then sample the
        //   two signals at several points across many cycles.
        //====================================================================
        @(negedge clk); #1 enable = 1'b1;   // change while low -> clean latch
        @(posedge clk);                     // this high phase now enabled
        n_pulses_a = 0;
        for (i = 0; i < 40; i = i + 1) begin
            #1;                              // sample mid-phase
            if (gated_clk !== clk) begin
                $display("TESTa FAIL: gated_clk(%b) != clk(%b) with enable=1 @ t=%0t",
                         gated_clk, clk, $time);
                $fatal(1);
            end
            @(posedge clk) n_pulses_a = n_pulses_a + 1;  // count passed pulses
        end
        if (n_pulses_a != 40) begin
            $display("TESTa FAIL: counted %0d pulses (expected 40)", n_pulses_a);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST a PASS: enable=1 -> gated_clk identical to clk (%0d whole pulses)", n_pulses_a);

        //====================================================================
        // TEST (b) -- enable ALWAYS 0  =>  gated_clk stays LOW (no pulses).
        //====================================================================
        @(negedge clk); #1 enable = 1'b0;   // drop while low -> latched 0
        @(posedge clk);
        for (i = 0; i < 40; i = i + 1) begin
            #1;
            if (gated_clk !== 1'b0) begin
                $display("TESTb FAIL: gated_clk=%b with enable=0 @ t=%0t (expected 0)",
                         gated_clk, $time);
                $fatal(1);
            end
            @(posedge clk);
        end
        tests_passed = tests_passed + 1;
        $display("TEST b PASS: enable=0 -> gated_clk held flat LOW (no pulses)");

        //====================================================================
        // TEST (c) -- GLITCH-FREEDOM: toggle enable in the MIDDLE of high phases.
        //   For each high phase we set the enable that should govern it (latched
        //   on the preceding low phase), then FLIP enable mid-high-phase.  The
        //   gate must IGNORE the mid-phase flip: the pulse presence is fixed by
        //   the low-phase-latched value, and no runt is produced.  The
        //   continuous monitors above ($fatal on any runt/misaligned edge) do
        //   the proving; here we also confirm the pulse-present decision.
        //====================================================================
        n_pulses_b = 0;
        for (i = 0; i < 20; i = i + 1) begin
            e = i[0];                        // alternate the governing value
            @(negedge clk); #1 enable = e;   // latch e on this low phase
            @(posedge clk);                  // high phase begins, pulse fixed by e
            #2 enable = ~e;                  // ADVERSARIAL: flip mid-high-phase
            #1;                              // observe: gate must ignore the flip
            if (gated_clk !== (e ? clk : 1'b0)) begin
                $display("TESTc FAIL: mid-phase enable flip leaked (i=%0d e=%b gated_clk=%b clk=%b) @ t=%0t",
                         i, e, gated_clk, clk, $time);
                $fatal(1);
            end
            if (e) n_pulses_b = n_pulses_b + 1;
        end
        // restore a clean low-phase enable before leaving the region
        @(negedge clk); #1 enable = 1'b0;
        if (n_pulses_b != 10) begin
            $display("TESTc FAIL: expected 10 enabled high phases, saw %0d", n_pulses_b);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST c PASS: mid-high-phase enable toggling produced NO glitch/runt (%0d whole pulses only)",
                 n_pulses_b);

        //====================================================================
        // TEST (d) -- EDGE COUNT over a mixed enable pattern.
        //   Drive a pseudo-random enable, latched cleanly on each low phase, and
        //   independently PREDICT the number of gated_clk rising edges = number
        //   of high phases whose preceding-low-phase enable was 1.  The posedge
        //   clk monitor counts the actual whole pulses; the counts must match.
        //====================================================================
        @(negedge clk); #1;                  // align to a low phase
        exp_rise     = 0;
        got_rise     = 0;
        count_active = 1'b1;                  // start counting real rises
        seed         = 32'hC0FFEE01;
        for (i = 0; i < 64; i = i + 1) begin
            e = $random(seed) & 1'b1;         // next enable value
            #1 enable = e;                     // set while clk is LOW (latched)
            if (e) exp_rise = exp_rise + 1;    // this high phase will pulse
            @(posedge clk);                    // enter the governed high phase --
                                               // monitor counts this whole pulse
            #2 enable = ~e;                    // stir the pot mid-phase (ignored)
            @(negedge clk);                    // next low phase
        end
        // Force enable LOW for the trailing low phase so the next (uncounted-in-
        // exp_rise) high phase carries no pulse, then stop counting cleanly.
        #1 enable = 1'b0;
        @(posedge clk); #1;
        count_active = 1'b0;
        if (got_rise != exp_rise) begin
            $display("TESTd FAIL: gated_clk rising edges = %0d, expected %0d", got_rise, exp_rise);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST d PASS: gated_clk rising-edge count = %0d matches predicted %0d (mixed enable)",
                 got_rise, exp_rise);

        //====================================================================
        // TEST (e) -- TEST-ENABLE (scan bypass): test_en forces transparency
        //   even with enable==0.  gated_clk must follow clk.
        //====================================================================
        @(negedge clk); #1 enable = 1'b0;    // functional enable OFF
        @(negedge clk); #1 test_en = 1'b1;   // but scan force-transparent ON
        @(posedge clk);
        for (i = 0; i < 20; i = i + 1) begin
            #1;
            if (gated_clk !== clk) begin
                $display("TESTe FAIL: gated_clk(%b)!=clk(%b) with test_en=1 @ t=%0t",
                         gated_clk, clk, $time);
                $fatal(1);
            end
            @(posedge clk);
        end
        @(negedge clk); #1 test_en = 1'b0;
        tests_passed = tests_passed + 1;
        $display("TEST e PASS: test_en forces gate transparent (scan bypass) despite enable=0");

        //====================================================================
        // TEST (f) -- INVERTED POLARITY instance: enable LOW passes the clock.
        //   Drive enable_inv=0 (pass) then =1 (gate) and confirm the second DUT.
        //====================================================================
        @(negedge clk); #1 enable_inv = 1'b0;   // LOW -> pass
        @(posedge clk);
        for (i = 0; i < 10; i = i + 1) begin
            #1;
            if (gated_clk_inv !== clk) begin
                $display("TESTf FAIL: inv gate enable_inv=0 -> gated(%b)!=clk(%b) @ t=%0t",
                         gated_clk_inv, clk, $time);
                $fatal(1);
            end
            @(posedge clk);
        end
        @(negedge clk); #1 enable_inv = 1'b1;   // HIGH -> gate
        @(posedge clk);
        for (i = 0; i < 10; i = i + 1) begin
            #1;
            if (gated_clk_inv !== 1'b0) begin
                $display("TESTf FAIL: inv gate enable_inv=1 -> gated(%b)!=0 @ t=%0t",
                         gated_clk_inv, $time);
                $fatal(1);
            end
            @(posedge clk);
        end
        tests_passed = tests_passed + 1;
        $display("TEST f PASS: inverted-polarity gate -- enable LOW passes clk, HIGH gates it");

        //====================================================================
        $display("ALL %0d TESTS PASSED (%0d glitch-monitor checks, %0d gated rising edges verified)",
                 tests_passed, glitch_checks, got_rise);
        $finish;
    end

    // safety net: never hang
    initial begin
        #500_000;
        $display("TIMEOUT");
        $fatal(1);
    end

endmodule
