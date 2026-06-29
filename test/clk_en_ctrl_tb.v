`timescale 1ns/1ps
//============================================================================
// clk_en_ctrl_tb.v -- testbench for clk_en_ctrl (docs/IMPROVEMENT_PLAN.md P3.2)
//----------------------------------------------------------------------------
// WHAT THIS PROVES
//   (1) SAFETY (the binding property): on EVERY cycle, for EVERY cluster, an
//       ADVANCING cluster -- one whose work bit or input handshake fires while
//       the die is not in a global no-advance window -- has clk_en HIGH. Work
//       is therefore never clock-gated / dropped / corrupted. Conversely, in a
//       genuine global idle window (boot / Flash stall) clk_en is forced LOW
//       (where the idle dynamic power is saved). Checked continuously, X-aware,
//       $fatal on any violation.
//   (2) WAKE MARGIN: the cycle a producer first raises input_valid on a gated
//       cluster, clk_en rises that SAME cycle (combinational) -- one cycle
//       before the sampling edge, so the first active edge is never lost. The
//       hysteresis overhead is exactly HOLD+1 clocked-but-idle cycles per burst.
//   (3) MEASUREMENT: the gated-cycle FRACTION (= idle dynamic power saved) is
//       reported across three duty cycles -- 25% active (the Flash-bound die),
//       50%, and 90%. At ~25% active the gated fraction approaches ~74% (the
//       ~75% idle minus the small wake-margin overhead).
//----------------------------------------------------------------------------
module clk_en_ctrl_tb;

    localparam integer N_CLUSTER = 4;
    localparam integer HOLD      = 3;
    localparam integer CNT_W     = 32;
    localparam integer WAKE_OVH  = HOLD + 1;   // clocked-but-idle cycles / burst

    // ---- DUT I/O ----
    reg                         clk;
    reg                         rst;
    reg                         boot_active;
    reg                         stall;
    reg  [N_CLUSTER-1:0]        has_pending_work;
    reg  [N_CLUSTER-1:0]        input_valid;
    reg  [N_CLUSTER-1:0]        output_ready_downstream;
    wire [N_CLUSTER-1:0]        clk_en;
    /* verilator lint_off UNUSEDSIGNAL */  // only cluster-0 counter is measured
    wire [N_CLUSTER*CNT_W-1:0]  gated_cycles;
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- bookkeeping ----
    integer tests_passed;
    integer cycle_checks;     // per-cluster per-cycle safety checks performed
    integer my_gcnt0;         // mirror of cluster-0 gateable-cycle count
    reg     chk_active;       // 1 = run the continuous safety monitor

    clk_en_ctrl #(
        .N_CLUSTER (N_CLUSTER),
        .HOLD      (HOLD),
        .CNT_W     (CNT_W)
    ) dut (
        .clk                     (clk),
        .rst                     (rst),
        .boot_active             (boot_active),
        .stall                   (stall),
        .has_pending_work        (has_pending_work),
        .input_valid             (input_valid),
        .output_ready_downstream (output_ready_downstream),
        .clk_en                  (clk_en),
        .gated_cycles            (gated_cycles)
    );

    // 10ns clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    //------------------------------------------------------------------------
    // Continuous SAFETY monitor + X-aware check. Called once per driven cycle,
    // after inputs have settled and while clk_en is valid for this cycle.
    //------------------------------------------------------------------------
    task check_safety;
        integer c;
        reg adv_c, blk;
        begin
            if (chk_active) begin
                blk = boot_active | stall;
                for (c = 0; c < N_CLUSTER; c = c + 1) begin
                    // adv uses the two real activity hints (the output-handshake
                    // term is logically covered by has_pending_work).
                    adv_c = has_pending_work[c] | input_valid[c];

                    // X-aware: clk_en must never be unknown.
                    if (clk_en[c] === 1'bx) begin
                        $display("X-FAIL: clk_en[%0d] is X @ t=%0t", c, $time);
                        $fatal(1);
                    end

                    // BINDING PROPERTY: advancing & not-blocked => clocked.
                    if ((adv_c & ~blk) && (clk_en[c] !== 1'b1)) begin
                        $display("SAFETY-FAIL: cluster %0d advancing but GATED @ t=%0t (work=%b ivld=%b boot=%b stall=%b clk_en=%b)",
                                 c, $time, has_pending_work[c], input_valid[c],
                                 boot_active, stall, clk_en[c]);
                        $fatal(1);
                    end

                    // blocked window => clock forced low (idle power saved).
                    if (blk && (clk_en[c] !== 1'b0)) begin
                        $display("BLOCKED-FAIL: cluster %0d clocked during boot/stall @ t=%0t (clk_en=%b)",
                                 c, $time, clk_en[c]);
                        $fatal(1);
                    end

                    cycle_checks = cycle_checks + 1;
                end
                // mirror the cluster-0 gateable-cycle count
                if (clk_en[0] === 1'b0)
                    my_gcnt0 = my_gcnt0 + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Drive one cycle: apply inputs just after a posedge, hold them stable
    // across the next sampling posedge, and run the safety check in between.
    //------------------------------------------------------------------------
    task do_cycle(input [N_CLUSTER-1:0] work,
                  input [N_CLUSTER-1:0] ivld,
                  input [N_CLUSTER-1:0] ordy,
                  input                 boot,
                  input                 stl);
        begin
            @(posedge clk);
            #1;
            has_pending_work        = work;
            input_valid             = ivld;
            output_ready_downstream = ordy;
            boot_active             = boot;
            stall                   = stl;
            #1;                       // let combinational clk_en settle
            check_safety;             // clk_en valid until next posedge
        end
    endtask

    // convenience all-ones / all-zeros cluster vectors
    localparam [N_CLUSTER-1:0] ALLW = {N_CLUSTER{1'b1}};
    localparam [N_CLUSTER-1:0] NONE = {N_CLUSTER{1'b0}};

    //------------------------------------------------------------------------
    // Measure one duty cycle: P-cycle period, A active cycles, NPER periods,
    // genuine idle (no stall) so the wake-margin overhead is exercised.
    // Returns gated cluster-0 cycles and total cycles.
    //------------------------------------------------------------------------
    task run_duty(input integer P, input integer A, input integer NPER,
                  output integer gated, output integer total);
        integer p, i;
        begin
            gated = 0; total = 0;
            for (p = 0; p < NPER; p = p + 1) begin
                for (i = 0; i < A; i = i + 1) begin            // active burst
                    do_cycle(ALLW, ALLW, ALLW, 1'b0, 1'b0);
                    if (clk_en[0] === 1'b0) gated = gated + 1;
                    total = total + 1;
                end
                for (i = 0; i < (P - A); i = i + 1) begin       // genuine idle gap
                    do_cycle(NONE, NONE, NONE, 1'b0, 1'b0);
                    if (clk_en[0] === 1'b0) gated = gated + 1;
                    total = total + 1;
                end
            end
        end
    endtask

    // report helper: print a basis-point (xx.xx%) fraction
    task report_frac(input [255:0] label, input integer gated, input integer total);
        integer bp;
        begin
            bp = (gated * 10000) / total;
            $display("  %0s gated=%0d/%0d  -> gated fraction = %0d.%02d%%",
                     label, gated, total, bp/100, bp%100);
        end
    endtask

    //------------------------------------------------------------------------
    integer g25, t25, g50, t50, g90, t90;
    integer i, wake_clocked;
    reg     seen_gate;
    integer mod_g0;

    initial begin
        tests_passed = 0;
        cycle_checks = 0;
        my_gcnt0     = 0;
        chk_active   = 0;

        // safe init
        rst                     = 1'b1;
        boot_active             = 1'b0;
        stall                   = 1'b0;
        has_pending_work        = NONE;
        input_valid             = NONE;
        output_ready_downstream = NONE;

        // ---- reset (synchronous, active-high) ----
        repeat (3) @(posedge clk);
        #1 rst = 1'b0;
        @(posedge clk);
        chk_active = 1;          // begin continuous safety monitoring

        //====================================================================
        // TEST 1 -- BOOT phase: whole die idle, every cluster clock forced low.
        //====================================================================
        for (i = 0; i < 20; i = i + 1)
            do_cycle(ALLW, ALLW, ALLW, 1'b1, 1'b0);   // boot_active=1 even w/ work
        if (clk_en !== NONE) begin
            $display("TEST1 FAIL: clk_en=%b during boot (expected all 0)", clk_en);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 1 PASS: boot phase gates all clusters (clk_en==0) despite asserted work");

        //====================================================================
        // TEST 2 -- STALL gates immediately (Flash fetch in flight).
        //====================================================================
        for (i = 0; i < 15; i = i + 1)
            do_cycle(ALLW, ALLW, ALLW, 1'b0, 1'b1);   // stall=1 even w/ work
        if (clk_en !== NONE) begin
            $display("TEST2 FAIL: clk_en=%b during stall (expected all 0)", clk_en);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 2 PASS: Flash-stall window gates all clusters immediately");

        //====================================================================
        // TEST 3 -- WAKE MARGIN: first input_valid raises clk_en the SAME cycle.
        //   First drive the die idle long enough to gate cluster 0, then assert
        //   input_valid and confirm clk_en[0] is HIGH on that very cycle.
        //====================================================================
        for (i = 0; i < (WAKE_OVH + 5); i = i + 1)            // idle -> gate it
            do_cycle(NONE, NONE, NONE, 1'b0, 1'b0);
        if (clk_en[0] !== 1'b0) begin
            $display("TEST3 SETUP FAIL: cluster 0 not gated after idle (clk_en[0]=%b)", clk_en[0]);
            $fatal(1);
        end
        // the wake cycle: raise input_valid[0]; clk_en[0] must rise this cycle.
        do_cycle({{(N_CLUSTER-1){1'b0}}, 1'b1}, {{(N_CLUSTER-1){1'b0}}, 1'b1}, NONE, 1'b0, 1'b0);
        if (clk_en[0] !== 1'b1) begin
            $display("TEST3 FAIL: clk_en[0]=%b on wake cycle (input_valid just rose)", clk_en[0]);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 3 PASS: wake margin -- clk_en rises the same cycle input_valid first asserts");

        //====================================================================
        // TEST 4 -- WAKE-MARGIN OVERHEAD == HOLD+1 clocked-but-idle cycles.
        //   One active cycle to arm the hysteresis, then count clocked idle
        //   cycles before the gate engages.
        //====================================================================
        do_cycle(ALLW, ALLW, ALLW, 1'b0, 1'b0);           // arm: req high
        wake_clocked = 0;
        seen_gate    = 1'b0;
        for (i = 0; i < (WAKE_OVH + 6); i = i + 1) begin
            do_cycle(NONE, NONE, NONE, 1'b0, 1'b0);
            if (!seen_gate) begin
                if (clk_en[0] === 1'b1) wake_clocked = wake_clocked + 1;
                else                    seen_gate    = 1'b1;
            end
        end
        if (wake_clocked !== WAKE_OVH) begin
            $display("TEST4 FAIL: wake overhead = %0d clocked idle cycles (expected %0d)",
                     wake_clocked, WAKE_OVH);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 4 PASS: wake-margin overhead = %0d clocked-but-idle cycles per burst (= HOLD+1)",
                 wake_clocked);

        //====================================================================
        // TEST 5 -- the Flash-bound burst:stall pattern (80 active : 240 stall),
        //   safety monitored continuously (no $fatal == never gated active work).
        //====================================================================
        for (i = 0; i < 80; i = i + 1)
            do_cycle(ALLW, ALLW, ALLW, 1'b0, 1'b0);       // active burst
        for (i = 0; i < 240; i = i + 1)
            do_cycle(ALLW, ALLW, ALLW, 1'b0, 1'b1);       // 240-cycle Flash stall
        tests_passed = tests_passed + 1;
        $display("TEST 5 PASS: 80-active : 240-stall Flash-bound pattern -- safety held every cycle");

        //====================================================================
        // TEST 6 -- per-cluster INDEPENDENCE: drive cluster 0 active while the
        //   rest go idle; cluster 0 stays clocked, the idle ones gate.
        //====================================================================
        for (i = 0; i < (WAKE_OVH + 6); i = i + 1)
            do_cycle({{(N_CLUSTER-1){1'b0}}, 1'b1},        // only cluster 0 has work
                     {{(N_CLUSTER-1){1'b0}}, 1'b1},
                     NONE, 1'b0, 1'b0);
        if (clk_en[0] !== 1'b1) begin
            $display("TEST6 FAIL: active cluster 0 gated (clk_en[0]=%b)", clk_en[0]);
            $fatal(1);
        end
        if (clk_en[N_CLUSTER-1:1] !== {(N_CLUSTER-1){1'b0}}) begin
            $display("TEST6 FAIL: idle clusters not gated (clk_en=%b)", clk_en);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 6 PASS: per-cluster independence -- active cluster clocked, idle clusters gated");

        //====================================================================
        // TESTS 7-9 -- gated-cycle FRACTION across duty cycles (power saved).
        //   period P=320, NPER=3.  Idle is genuine (no stall) so the wake-margin
        //   overhead is included in the measured number.
        //====================================================================
        $display("--- gated-cycle fraction (idle dynamic power saved) ---");

        run_duty(320,  80, 3, g25, t25);                  // 25% active (Flash-bound)
        report_frac("25%% active (Flash-bound):", g25, t25);
        // expect exactly NPER*(I - (HOLD+1)) gated, I = P-A
        if (g25 !== 3*((320-80) - WAKE_OVH)) begin
            $display("TEST7 FAIL: 25%% gated=%0d expected=%0d", g25, 3*((320-80)-WAKE_OVH));
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 7 PASS: 25%% duty gated fraction matches expected (~74%%)");

        run_duty(320, 160, 3, g50, t50);                  // 50% active
        report_frac("50%% active:               ", g50, t50);
        if (g50 !== 3*((320-160) - WAKE_OVH)) begin
            $display("TEST8 FAIL: 50%% gated=%0d expected=%0d", g50, 3*((320-160)-WAKE_OVH));
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 8 PASS: 50%% duty gated fraction matches expected (~49%%)");

        run_duty(320, 288, 3, g90, t90);                  // 90% active
        report_frac("90%% active:               ", g90, t90);
        if (g90 !== 3*((320-288) - WAKE_OVH)) begin
            $display("TEST9 FAIL: 90%% gated=%0d expected=%0d", g90, 3*((320-288)-WAKE_OVH));
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 9 PASS: 90%% duty gated fraction matches expected (~9%%)");

        //====================================================================
        // TEST 10 -- monotonicity: lower duty -> more gating -> more power saved.
        //====================================================================
        if (!((g25*t50 > g50*t25) && (g50*t90 > g90*t50))) begin
            $display("TEST10 FAIL: gated fraction not monotonic across duty (25:%0d/%0d 50:%0d/%0d 90:%0d/%0d)",
                     g25,t25,g50,t50,g90,t90);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 10 PASS: gated fraction rises as duty falls (more idle -> more power saved)");

        //====================================================================
        // TEST 11 -- the module's own saturating gated_cycles[0] counter tracks
        //   the gateable cycles (matches the TB mirror within counting latency).
        //====================================================================
        mod_g0 = gated_cycles[0*CNT_W +: CNT_W];
        if (mod_g0 == 0) begin
            $display("TEST11 FAIL: module gated_cycles[0]=0 (expected > 0)");
            $fatal(1);
        end
        if ((mod_g0 > my_gcnt0 + 2) || (mod_g0 + 2 < my_gcnt0)) begin
            $display("TEST11 FAIL: gated_cycles[0]=%0d vs TB mirror=%0d (drift > 1 cycle)",
                     mod_g0, my_gcnt0);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 11 PASS: module gated_cycles[0]=%0d tracks TB mirror=%0d", mod_g0, my_gcnt0);

        //====================================================================
        // TEST 12 -- X-aware: clk_en was never X across the whole run (the
        //   continuous monitor would have $fatal'd otherwise); confirm valid now.
        //====================================================================
        if (^clk_en === 1'bx) begin
            $display("TEST12 FAIL: clk_en has X bits (%b)", clk_en);
            $fatal(1);
        end
        tests_passed = tests_passed + 1;
        $display("TEST 12 PASS: clk_en stayed driven/X-free for all %0d cycle-checks", cycle_checks);

        //====================================================================
        $display("ALL %0d TESTS PASSED (%0d cycle-checks, gated25=%0d/%0d, gated_cycles[0]=%0d, wake_ovh=%0d)",
                 tests_passed, cycle_checks, g25, t25, mod_g0, WAKE_OVH);
        $finish;
    end

    // safety net: never hang
    initial begin
        #5_000_000;
        $display("TIMEOUT");
        $fatal(1);
    end

endmodule
