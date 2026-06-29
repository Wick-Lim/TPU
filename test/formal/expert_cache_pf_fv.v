`timescale 1ns/1ps
//============================================================================
// expert_cache_pf_fv.v -- FORMAL HARNESS (bounded model checking) for the
//                         committed DUT src/expert_cache_pf.v
//----------------------------------------------------------------------------
// NEW formal-only harness.  It does NOT modify the committed module: it
// instantiates expert_cache_pf as `dut`, drives every input from free formal
// stimulus constrained (by construction) to legal protocol, models the shared
// Flash DMA channel with a small fixed latency, and proves the target safety
// properties using ONLY the DUT's PORTS plus an independent reference model.
//
// WHY PORT-ONLY:  this yosys build (0.66) cannot read a sub-module's internal
// signals -- cross-module hierarchical references read stale-constant and
// `bind` is silently dropped.  So the directory invariants are checked against
// an independent harness-side SHADOW DIRECTORY that mirrors the DUT exactly.
//
// SHADOW DIRECTORY (exact mirror when PF_ENABLE==0):
//   With prefetch disabled the DUT changes its directory ONLY on a demand
//   response: a MISS response (resp_valid & !hit) installs the requested expert
//   into resp_slot; a HIT response changes no slot contents.  The harness snoops
//   these responses and maintains shadow_valid[]/shadow_tag[] identically, so
//   the shadow IS the DUT directory.  (The DUT's documented invariant is that
//   with pf off its demand behaviour is bit-identical to expert_cache_ctrl.)
//
// PROPERTIES
//   P1  HIT-RETURNS-RIGHT-SLOT (PF_ENABLE==0): on every HIT response the slot
//       the DUT returns is, per the independently-observed install history,
//       VALID and holds exactly the requested expert id.  A wrong-slot / wrong-id
//       hit (an LRU/rank or lookup bug) fails this.
//   P2a DIRECTORY UNIQUENESS via NO-DUP-INSTALL (PF_ENABLE==0): a MISS only ever
//       occurs for an expert that is NOT already resident -- so the DUT never
//       installs a second copy of an id it already holds (the mechanism by which
//       a duplicate would ever arise).  Plus P2b: the shadow (== DUT directory)
//       never has one id in two valid slots.
//   P3  BOUNDED RESP LIVENESS (PF_ENABLE==1, prefetch free/adversarial): once a
//       demand request is outstanding it is answered within LIVE_BOUND cycles,
//       even while best-effort prefetch contends for the shared Flash channel.
//       Encoded as a safety watchdog wait_cnt <= LIVE_BOUND.
//----------------------------------------------------------------------------
module expert_cache_pf_fv #(
    parameter integer SLOTS      = 2,
    parameter integer N_EXPERT   = 4,
    parameter integer ID_W       = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer SLOT_W     = (SLOTS    <= 1) ? 1 : $clog2(SLOTS),
    parameter integer FLAT       = 2,    // modelled Flash latency (cycles flash_req held)
    parameter integer LIVE_BOUND = 12,   // bounded-liveness watchdog ceiling
    parameter integer PF_ENABLE  = 0     // 0: prefetch off (P1/P2 shadow); 1: pf adversarial (P3)
)(
    input  wire                clk,
    // free formal stimulus (smtbmc drives these arbitrarily each step)
    input  wire                f_start,        // begin a new demand txn when idle
    input  wire [ID_W-1:0]     f_id,           // its expert id
    input  wire                f_pf_valid,      // prefetch hint valid (only used if PF_ENABLE)
    input  wire [ID_W-1:0]     f_pf_id          // prefetch hint id
);
    //------------------------------------------------------------------------
    // RESET GENERATOR -- rst high for the first 3 cycles, then low forever.
    // The initialiser (=0) is honoured by yosys -formal as the initial state
    // (verified separately), so the reset is deterministic.
    //------------------------------------------------------------------------
    reg  [2:0] rst_cnt = 3'd0;
    always @(posedge clk) if (rst_cnt != 3'd7) rst_cnt <= rst_cnt + 3'd1;
    wire rst = (rst_cnt < 3'd3);

    // CHECKS-ENABLED: properties evaluate only after we have passed through
    // reset (initialised 0; set when !rst).  Guarantees no property is sampled
    // against pre-reset (arbitrary anyinit) state.
    reg checks_en = 1'b0;
    always @(posedge clk) checks_en <= ~rst;

    //------------------------------------------------------------------------
    // DEMAND DRIVER -- legal router protocol: hold req_valid + a STABLE
    // req_expert_id from the cycle a txn starts until resp_valid is seen.
    //------------------------------------------------------------------------
    reg              pending;
    reg  [ID_W-1:0]  req_id_reg;
    wire             req_valid     = pending;
    wire [ID_W-1:0]  req_expert_id = req_id_reg;

    always @(posedge clk) begin
        if (rst) begin
            pending    <= 1'b0;
            req_id_reg <= {ID_W{1'b0}};
        end else if (resp_valid) begin
            pending <= 1'b0;                      // txn answered
        end else if (!pending && f_start) begin
            pending    <= 1'b1;                   // start txn, latch a STABLE id
            req_id_reg <= f_id;
        end
    end

    //------------------------------------------------------------------------
    // PREFETCH DRIVE (best-effort).  Off entirely when PF_ENABLE==0.
    //------------------------------------------------------------------------
    wire             pf_valid     = (PF_ENABLE != 0) ? f_pf_valid : 1'b0;
    wire [ID_W-1:0]  pf_expert_id = f_pf_id;

    //------------------------------------------------------------------------
    // SHARED FLASH DMA MODEL -- fixed latency.  flash_req is a DUT output; we
    // count cycles it is held and raise flash_done after FLAT cycles.  Used by
    // both demand misses and prefetch fetches (same channel).
    //------------------------------------------------------------------------
    reg  [3:0] fcnt;
    always @(posedge clk) begin
        if (rst)            fcnt <= 4'd0;
        else if (flash_req) fcnt <= fcnt + 4'd1;
        else                fcnt <= 4'd0;
    end
    wire flash_done = flash_req && (fcnt >= (FLAT-1));

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    wire               resp_valid;
    wire               hit;
    wire [SLOT_W-1:0]  resp_slot;
    wire               busy;
    wire               pf_ready;
    wire               flash_req;
    wire [ID_W-1:0]    flash_expert_id;
    wire [31:0]        hit_count, miss_count, demand_stall_cycles, pf_issued, pf_hit;

    expert_cache_pf #(
        .SLOTS(SLOTS), .N_EXPERT(N_EXPERT), .CACHE_HIT_LAT(0), .REPL_POLICY(0)
    ) dut (
        .clk(clk), .rst(rst),
        .req_valid(req_valid), .req_expert_id(req_expert_id),
        .resp_valid(resp_valid), .hit(hit), .resp_slot(resp_slot), .busy(busy),
        .pf_valid(pf_valid), .pf_expert_id(pf_expert_id), .pf_ready(pf_ready),
        .flash_req(flash_req), .flash_expert_id(flash_expert_id), .flash_done(flash_done),
        .hit_count(hit_count), .miss_count(miss_count),
        .demand_stall_cycles(demand_stall_cycles),
        .pf_issued(pf_issued), .pf_hit(pf_hit)
    );

    //------------------------------------------------------------------------
    // LIVENESS WATCHDOG (port-only) : cycles a demand stays outstanding.
    //------------------------------------------------------------------------
    reg [7:0] wait_cnt;
    always @(posedge clk) begin
        if (rst)             wait_cnt <= 8'd0;
        else if (!req_valid) wait_cnt <= 8'd0;   // nothing outstanding
        else if (resp_valid) wait_cnt <= 8'd0;   // answered this cycle
        else                 wait_cnt <= wait_cnt + 8'd1;
    end

    always @(posedge clk)
        if (checks_en)
            a_liveness : assert (wait_cnt <= LIVE_BOUND[7:0]);

    //------------------------------------------------------------------------
    // SHADOW DIRECTORY + P1/P2 (only meaningful with prefetch OFF, where the
    // shadow mirrors the DUT directory exactly).  SLOTS=2 hand-unrolled.
    //------------------------------------------------------------------------
    generate
    if (PF_ENABLE == 0) begin : g_shadow
        reg              sv0, sv1;          // shadow valid bits
        reg [ID_W-1:0]   st0, st1;          // shadow resident tags

        // requested id resident in the shadow (== resident in the DUT) ?
        wire res0 = sv0 && (st0 == req_id_reg);
        wire res1 = sv1 && (st1 == req_id_reg);

        always @(posedge clk) begin
            if (rst) begin
                sv0 <= 1'b0; sv1 <= 1'b0;
                st0 <= {ID_W{1'b0}}; st1 <= {ID_W{1'b0}};
            end else if (resp_valid && !hit) begin
                // MISS install: DUT placed req_id_reg into resp_slot.
                if (resp_slot == 1'b0) begin sv0 <= 1'b1; st0 <= req_id_reg; end
                else                   begin sv1 <= 1'b1; st1 <= req_id_reg; end
            end
            // HIT response: no slot-content change (LRU recency only).
        end

        always @(posedge clk) begin
            if (checks_en) begin
                // P1: a HIT returns a slot that (independently) holds the req id.
                if (resp_valid && hit) begin
                    if (resp_slot == 1'b0) begin
                        a_hit_slot0_valid : assert (sv0);
                        a_hit_slot0_tag   : assert (st0 == req_id_reg);
                    end else begin
                        a_hit_slot1_valid : assert (sv1);
                        a_hit_slot1_tag   : assert (st1 == req_id_reg);
                    end
                end
                // P2a: a MISS only happens for a NON-resident expert -> the DUT
                // never installs a duplicate of an id it already holds.
                if (resp_valid && !hit) begin
                    a_miss_not_resident : assert (!res0 && !res1);
                end
                // P2b: directory uniqueness invariant (== DUT directory).
                a_unique : assert (!(sv0 && sv1 && (st0 == st1)));
            end
        end
    end
    endgenerate
endmodule
