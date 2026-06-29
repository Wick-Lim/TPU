`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// clk_en_ctrl.v  --  WORK-ACTIVITY CLOCK-ENABLE CONTROLLER for the FP8 die
//                                              (docs/IMPROVEMENT_PLAN.md P3.2)
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   The FP8 compute die is Flash-bandwidth-bound: it idles ~75% of the time
//   waiting on weight fetches, and it sits fully idle during the boot model
//   load.  Clock-gating the compute clusters during those idle windows removes
//   idle DYNAMIC power with ZERO throughput change.  In synthesizable RTL we do
//   NOT hand-write a gate; we emit a per-cluster CLOCK-ENABLE that the synthesis
//   tool maps to an integrated clock-gate (ICG) cell + register enables.  This
//   block is pure integer/control logic: it decides, per cluster and per cycle,
//   whether that cluster's clock may be gated.
//
//   The committed datapath is handshake-driven (valid/ready), so "no work" is
//   visible structurally: a cluster is idle when it holds no pending op AND no
//   producer is presenting an input.  Two GLOBAL windows force the whole die
//   idle: boot_active (model still loading) and stall (blocked on a Flash
//   fetch); in both the clusters cannot advance, so all clocks are gateable.
//
//----------------------------------------------------------------------------
// ACTIVITY -> ENABLE  (per cluster c)
//   Per-cluster hints (one bit each, packed [N_CLUSTER-1:0]):
//       has_pending_work[c]        : an in-flight op whose FSM/pipe steps every
//                                    cycle (internal compute -> a state advance)
//       input_valid[c]             : a producer is presenting data this cycle;
//                                    it is SAMPLED on the next active edge
//       output_ready_downstream[c] : the sink can accept this cycle (a held
//                                    result + ready sink == an output handshake)
//   Global:  boot_active, stall.
//
//   A cluster may ADVANCE STATE this cycle iff it has a reason to AND no global
//   window blocks it:
//       adv[c]     = has_pending_work[c] | input_valid[c]
//                  | (has_pending_work[c] & output_ready_downstream[c]);
//       blocked    = boot_active | stall;            // whole-die no-advance
//       req[c]     = adv[c] & ~blocked;              // wants the clock NOW
//   (the output-handshake term is logically covered by has_pending_work; it is
//    spelled out so all three hints participate and the intent is explicit.)
//
//----------------------------------------------------------------------------
// WAKE MARGIN  (never lose the first active edge)
//   clk_en is driven COMBINATIONALLY from req plus a registered hysteresis bit:
//       clk_en[c] = ~blocked & (en_reg[c] | adv[c]);
//   So the cycle a producer first raises input_valid[c] (or work appears),
//   adv[c] is already 1 and clk_en[c] rises in the SAME cycle -- one cycle
//   BEFORE the edge that samples that input.  The first active edge is always
//   clocked; no work is ever lost on wake-up.
//
// HYSTERESIS / HOLD  (don't toggle the gate on bursty input)
//   A per-cluster down-counter hold_cnt holds the clock on for HOLD extra idle
//   cycles after activity ceases (reloaded to HOLD whenever req[c] is high).
//   Gaps shorter than the margin never gate -> no per-cycle gate toggling
//   (gating itself costs energy).  hold_cnt is frozen during blocked windows so
//   a long Flash stall does not burn the wake margin.
//
//----------------------------------------------------------------------------
// SAFETY PROPERTY  (the one that must never break)
//   A cluster advances state ONLY when req[c] is high.  By construction
//       req[c]=1  =>  blocked=0 AND adv[c]=1  =>  clk_en[c] = 1 & (en_reg|1) = 1.
//   Therefore clk_en[c] >= req[c] EVERY cycle: an advancing cluster is NEVER
//   gated, so work can never be dropped or corrupted.  Conversely, during a
//   blocked window (boot/stall) the cluster does NOT advance, so forcing
//   clk_en[c]=0 there is safe and is exactly where the idle power is saved.
//
//----------------------------------------------------------------------------
// MEASUREMENT
//   gated_cycles[c] : saturating count of cycles this cluster's clock was
//   gateable (clk_en[c]==0) -- the realized idle-power-saving opportunity.
//
//----------------------------------------------------------------------------
// RESET / STRUCTURE
//   Synchronous, ACTIVE-HIGH reset.  clk_en defaults SAFE = ENABLED at reset
//   (en_reg=1, hold_cnt=HOLD) so nothing is gated until proven idle.  No latch,
//   no combinational loop (clk_en is a pure function of inputs + one register),
//   X-aware (all state reset).
//============================================================================
module clk_en_ctrl #(
    parameter integer N_CLUSTER = 4,    // number of clock-gateable compute clusters
    parameter integer HOLD      = 3,    // hysteresis: extra idle cycles kept clocked
    parameter integer CNT_W     = 32,   // width of the per-cluster gated-cycle counter
    // Derived (do NOT override) -- guard the degenerate HOLD<1 case.
    parameter integer HOLD_W = (HOLD < 1) ? 1 : $clog2(HOLD + 1)
)(
    input  wire                       clk,
    input  wire                       rst,   // synchronous, ACTIVE-HIGH

    // ---- global no-advance windows (whole die) ----
    input  wire                       boot_active,  // model still loading -> idle
    input  wire                       stall,        // blocked on a Flash fetch -> idle

    // ---- per-cluster activity hints ----
    input  wire [N_CLUSTER-1:0]       has_pending_work,
    input  wire [N_CLUSTER-1:0]       input_valid,
    input  wire [N_CLUSTER-1:0]       output_ready_downstream,

    // ---- per-cluster clock enable (HIGH = clocked, LOW = gateable) ----
    output wire [N_CLUSTER-1:0]       clk_en,

    // ---- measurement: saturating gated-cycle count per cluster ----
    output wire [N_CLUSTER*CNT_W-1:0] gated_cycles
);

    // global blocked window: the whole die cannot advance
    wire blocked = boot_active | stall;

    // per-cluster hysteresis state + gated-cycle counters
    reg              en_reg   [0:N_CLUSTER-1];
    reg [HOLD_W-1:0] hold_cnt [0:N_CLUSTER-1];
    reg [CNT_W-1:0]  gcnt     [0:N_CLUSTER-1];

    genvar g;
    generate
        for (g = 0; g < N_CLUSTER; g = g + 1) begin : CLUSTER
            // ---- activity -> enable (combinational) ----
            // a reason the cluster may advance state this cycle
            wire adv = has_pending_work[g] | input_valid[g]
                     | (has_pending_work[g] & output_ready_downstream[g]);
            // wants the clock this cycle (also the safety lower bound on clk_en)
            wire req = adv & ~blocked;

            // WAKE MARGIN: adv lifts clk_en the same cycle work appears (one
            // cycle before the sampling edge).  SAFETY: req=1 => clk_en=1.
            assign clk_en[g] = ~blocked & (en_reg[g] | adv);

            // expose the saturating gated-cycle counter
            assign gated_cycles[g*CNT_W +: CNT_W] = gcnt[g];

            // ---- hysteresis / HOLD + measurement (sequential) ----
            always @(posedge clk) begin
                if (rst) begin
                    en_reg[g]   <= 1'b1;            // SAFE default: enabled
                    hold_cnt[g] <= HOLD[HOLD_W-1:0];
                    gcnt[g]     <= {CNT_W{1'b0}};
                end else begin
                    if (req) begin
                        // active now -> clock on, reload the wake margin
                        en_reg[g]   <= 1'b1;
                        hold_cnt[g] <= HOLD[HOLD_W-1:0];
                    end else if (blocked) begin
                        // stalled / booting: output already gated; freeze the
                        // margin so a long stall does not consume it
                        en_reg[g]   <= en_reg[g];
                        hold_cnt[g] <= hold_cnt[g];
                    end else if (hold_cnt[g] != {HOLD_W{1'b0}}) begin
                        // idle but inside the hysteresis window -> stay clocked
                        en_reg[g]   <= 1'b1;
                        hold_cnt[g] <= hold_cnt[g] - 1'b1;
                    end else begin
                        // idle and margin exhausted -> gateable
                        en_reg[g]   <= 1'b0;
                    end

                    // measurement: count cycles the clock was gateable
                    if (~clk_en[g] && (gcnt[g] != {CNT_W{1'b1}}))
                        gcnt[g] <= gcnt[g] + 1'b1;
                end
            end
        end
    endgenerate

endmodule
/* verilator lint_on DECLFILENAME */
