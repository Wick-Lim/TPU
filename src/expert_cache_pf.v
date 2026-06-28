`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// expert_cache_pf.v  --  GLM-5.2-FP8 MoE EXPERT-WEIGHT HBM CACHE + PREFETCH
//                        (expert_cache_ctrl PLUS a best-effort prefetch engine)
//----------------------------------------------------------------------------
// This is expert_cache_ctrl (src/expert_cache_ctrl.v) -- tag lookup + EXACT
// move-to-front LRU + miss->Flash-DMA->refill FSM + hit/miss counters -- with
// the DEMAND PATH KEPT BIT-IDENTICAL, plus a PREFETCH hint port that brings an
// expert into HBM in the background over the SAME Flash DMA channel.
//
//   INVARIANT: with pf_valid tied low this module's demand hit/miss/timing
//   behaviour is IDENTICAL to expert_cache_ctrl (same LRU, same FSM, same
//   resp/busy timing).  Prefetch only changes WHEN experts are resident; it
//   never alters the demand FSM's per-request decisions when pf is off.
//
//----------------------------------------------------------------------------
// PREFETCH INTERFACE (best-effort, NO response, never stalls the demand path)
//   * pf_valid / pf_expert_id : a hint "please bring this expert into HBM".
//   * pf_ready : combinational accept.  We accept a hint ONLY when the demand
//       path is idle and no demand request is presented this cycle
//       (pf_ready = (state==IDLE) && !req_valid) -- DEMAND-FIRST arbitration.
//   * On accept, if the expert is already resident we SKIP (no-op, no Flash).
//       Otherwise we issue a background Flash fetch into the current LRU-victim
//       slot and install it when Flash completes.
//
// SHARED FLASH DMA, DEMAND-FIRST ARBITRATION
//   One Flash channel (flash_req/flash_expert_id/flash_done) is shared.
//   * A demand miss ALWAYS gets the channel: a prefetch fetch is *started* only
//     from IDLE with no demand pending/in-flight ("prefetch fetches run only
//     when no demand fetch is in flight and no demand request is pending").
//   * An already-running best-effort prefetch is NON-preemptive (the TB Flash
//     model is non-abortable): while it drains, a demand HIT is still served
//     immediately with ZERO stall (hits need no Flash), and a demand MISS is
//     latched and serviced the instant the prefetch frees the channel.
//
// NEW STATS
//   * demand_stall_cycles : cycles 'busy' is high (busy is raised ONLY for a
//       demand miss / a demand miss queued behind a prefetch) -- the cycles the
//       compute die actually waits.
//   * pf_issued : prefetch Flash fetches actually started.
//   * pf_hit    : prefetched experts that were later DEMANDED while still
//       resident (counted once per prefetch, at the first demand hit).
//----------------------------------------------------------------------------
module expert_cache_pf #(
    parameter integer SLOTS     = 8,    // HBM cache slots
    parameter integer N_EXPERT  = 64,   // total distinct expert ids
    /* verilator lint_off UNUSEDPARAM */
    parameter integer FLASH_LAT = 20,   // miss fetch latency (doc only; TB models it)
    /* verilator lint_on UNUSEDPARAM */
    // GDDR6 read latency (cycles) to DELIVER an expert that is RESIDENT in the
    // cache.  A demand HIT now waits CACHE_HIT_LAT extra cycles before resp_valid
    // and those cycles are counted in demand_stall_cycles (the compute die waits
    // on the GDDR6 read).  CACHE_HIT_LAT==0 => the wait state is skipped entirely
    // and behaviour is BIT-IDENTICAL to the committed (no-hit-latency) module.
    parameter integer CACHE_HIT_LAT = 0,
    // Derived widths (do NOT override) -- guard the degenerate ==1 cases.
    parameter integer ID_W   = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer SLOT_W = (SLOTS    <= 1) ? 1 : $clog2(SLOTS)
)(
    input  wire                clk,
    input  wire                rst,         // synchronous, ACTIVE-HIGH

    // ---- router demand request ----
    input  wire                req_valid,
    input  wire [ID_W-1:0]     req_expert_id,

    // ---- demand response: here it is (which HBM slot) ----
    output reg                 resp_valid,
    output reg                 hit,
    output reg  [SLOT_W-1:0]   resp_slot,
    output reg                 busy,        // high while a DEMAND miss is serviced (stall)

    // ---- prefetch hint (best-effort, no response) ----
    input  wire                pf_valid,
    input  wire [ID_W-1:0]     pf_expert_id,
    output reg                 pf_ready,    // combinational accept

    // ---- Flash DMA handshake (TB models the backing store + latency) ----
    output reg                 flash_req,
    output reg  [ID_W-1:0]     flash_expert_id,
    input  wire                flash_done,

    // ---- stats (readable) ----
    output reg  [31:0]         hit_count,
    output reg  [31:0]         miss_count,
    output reg  [31:0]         demand_stall_cycles,
    output reg  [31:0]         pf_issued,
    output reg  [31:0]         pf_hit
);
    // ---- cache directory ----
    reg               valid_arr [0:SLOTS-1];   // per-slot valid bit
    reg  [ID_W-1:0]   tag_arr   [0:SLOTS-1];   // resident expert id
    reg  [SLOT_W-1:0] rank      [0:SLOTS-1];   // recency position (0=MRU)
    reg               pf_flag   [0:SLOTS-1];   // slot was installed by a prefetch,
                                               // not yet demanded (for pf_hit)

    // ---- demand-miss bookkeeping (identical role to expert_cache_ctrl) ----
    reg  [SLOT_W-1:0] victim_q;                // latched victim slot (demand miss)
    reg  [ID_W-1:0]   req_id_q;                // latched requested id (demand miss)

    // ---- prefetch bookkeeping ----
    reg  [ID_W-1:0]   pf_id_q;                 // id of the in-flight prefetch

    // ---- GDDR6 hit-read wait state (active ONLY when CACHE_HIT_LAT>0) ----
    //   On a demand HIT we latch hit/resp_slot and update the LRU/counters
    //   immediately (exactly as before), but hold resp_valid low for
    //   CACHE_HIT_LAT cycles while this down-counter drains.  busy is held high
    //   during the wait so the cycles land in demand_stall_cycles.  When
    //   CACHE_HIT_LAT==0 the counter is never loaded -> every (hit_wait != 0)
    //   branch is dead and timing/counters are byte-for-byte the committed core.
    // hit_wait only ever holds 0..CACHE_HIT_LAT, so size it to exactly that
    // range (1 bit when CACHE_HIT_LAT==0).  This drops the 32-bit down-counter
    // subtractor+compare to an HW_W-bit one (a single dead 1-bit reg in the
    // default build); behaviour is unchanged (same values, same compares).
    localparam integer HW_W = (CACHE_HIT_LAT <= 0) ? 1 : $clog2(CACHE_HIT_LAT+1);
    reg  [HW_W-1:0]     hit_wait;
    localparam [HW_W-1:0] HIT_LAT_C = CACHE_HIT_LAT[HW_W-1:0];

    // ---- demand queued behind an in-flight prefetch ----
    reg               dmd_pending;             // a demand miss waits for the Flash channel
    reg  [ID_W-1:0]   dmd_pending_id;          // its expert id

    // ---- FSM ----
    localparam [1:0] S_IDLE     = 2'd0,  // demand idle (may start a prefetch)
                     S_FETCH    = 2'd1,  // demand miss Flash in flight (busy=1)
                     S_PF_FETCH = 2'd2,  // prefetch Flash in flight (busy=0 unless dmd queued)
                     S_DMD_REST = 2'd3;  // re-evaluate a demand queued behind a prefetch
    reg [1:0] state;

    integer k;

    // recency position of the LRU slot (width-matched constant for compares)
    localparam [SLOT_W-1:0] LRU_POS = SLOT_W'(SLOTS-1);

    //------------------------------------------------------------------------
    // COMBINATIONAL TAG LOOKUP  (parallel associative compare across SLOTS)
    //   query_id is req_expert_id for every demand path that exists when pf is
    //   off; it only switches to dmd_pending_id in S_DMD_REST, a state that is
    //   never reached with pf off -> lookup is byte-for-byte the ctrl lookup.
    //------------------------------------------------------------------------
    wire [ID_W-1:0] query_id = (state == S_DMD_REST) ? dmd_pending_id : req_expert_id;

    reg               lookup_hit;
    reg  [SLOT_W-1:0] lookup_slot;
    always @* begin
        lookup_hit  = 1'b0;
        lookup_slot = {SLOT_W{1'b0}};
        for (k = 0; k < SLOTS; k = k + 1)
            if (valid_arr[k] && (tag_arr[k] == query_id)) begin
                lookup_hit  = 1'b1;
                lookup_slot = k[SLOT_W-1:0];
            end
    end

    // residency check for an incoming prefetch hint id
    reg pf_in_resident;
    always @* begin
        pf_in_resident = 1'b0;
        for (k = 0; k < SLOTS; k = k + 1)
            if (valid_arr[k] && (tag_arr[k] == pf_expert_id))
                pf_in_resident = 1'b1;
    end

    // residency check for the in-flight prefetch id (recheck at install time)
    reg pf_q_resident;
    always @* begin
        pf_q_resident = 1'b0;
        for (k = 0; k < SLOTS; k = k + 1)
            if (valid_arr[k] && (tag_arr[k] == pf_id_q))
                pf_q_resident = 1'b1;
    end

    //------------------------------------------------------------------------
    // COMBINATIONAL VICTIM SELECT  (lowest invalid first, else EXACT-LRU)
    //   Shared by the demand path (S_IDLE / S_DMD_REST) and the prefetch
    //   install -- each uses it at its own cycle against the settled directory.
    //------------------------------------------------------------------------
    reg               have_invalid;
    reg  [SLOT_W-1:0] invalid_slot;
    reg  [SLOT_W-1:0] lru_slot;
    always @* begin
        have_invalid = 1'b0;
        invalid_slot = {SLOT_W{1'b0}};
        lru_slot     = {SLOT_W{1'b0}};
        // scan high->low so the LOWEST-index invalid slot wins (last assignment)
        for (k = SLOTS-1; k >= 0; k = k - 1)
            if (!valid_arr[k]) begin
                have_invalid = 1'b1;
                invalid_slot = k[SLOT_W-1:0];
            end
        // LRU = the slot at recency position SLOTS-1
        for (k = 0; k < SLOTS; k = k + 1)
            if (rank[k] == LRU_POS)
                lru_slot = k[SLOT_W-1:0];
    end

    wire [SLOT_W-1:0] victim_slot = have_invalid ? invalid_slot : lru_slot;

    //------------------------------------------------------------------------
    // COMBINATIONAL prefetch accept (DEMAND-FIRST): only from idle, only when
    // no demand request is presented this cycle.
    //------------------------------------------------------------------------
    //   (hit_wait != 0) only ever holds when CACHE_HIT_LAT>0 -> with hit latency
    //   off this is identical to the committed accept condition.
    always @* begin
        pf_ready = (state == S_IDLE) && !req_valid && (hit_wait == {HW_W{1'b0}});
    end

    //------------------------------------------------------------------------
    // SEQUENTIAL FSM + directory update
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            resp_valid          <= 1'b0;
            hit                 <= 1'b0;
            resp_slot           <= {SLOT_W{1'b0}};
            busy                <= 1'b0;
            flash_req           <= 1'b0;
            flash_expert_id     <= {ID_W{1'b0}};
            hit_count           <= 32'd0;
            miss_count          <= 32'd0;
            demand_stall_cycles <= 32'd0;
            pf_issued           <= 32'd0;
            pf_hit              <= 32'd0;
            victim_q            <= {SLOT_W{1'b0}};
            req_id_q            <= {ID_W{1'b0}};
            pf_id_q             <= {ID_W{1'b0}};
            dmd_pending         <= 1'b0;
            dmd_pending_id      <= {ID_W{1'b0}};
            hit_wait            <= {HW_W{1'b0}};
            state               <= S_IDLE;
            for (k = 0; k < SLOTS; k = k + 1) begin
                valid_arr[k] <= 1'b0;
                tag_arr[k]   <= {ID_W{1'b0}};
                rank[k]      <= k[SLOT_W-1:0];
                pf_flag[k]   <= 1'b0;
            end
        end else begin
            resp_valid <= 1'b0;   // default: response is a 1-cycle pulse

            // demand_stall: busy is ONLY ever high for a demand miss (in flight
            // or queued behind a prefetch) -> these are the compute-die waits.
            if (busy)
                demand_stall_cycles <= demand_stall_cycles + 32'd1;

            case (state)
                //------------------------------------------------------------
                S_IDLE: begin
                    if (hit_wait != {HW_W{1'b0}}) begin
                        // ---- GDDR6 hit read in flight (CACHE_HIT_LAT>0 only) ----
                        // hit/resp_slot/LRU/counters were committed at the hit
                        // cycle; deliver resp_valid when the read finishes.  busy
                        // stays high -> these are counted as demand_stall_cycles.
                        if (hit_wait == {{(HW_W-1){1'b0}},1'b1}) begin
                            resp_valid <= 1'b1;
                            busy       <= 1'b0;
                            hit_wait   <= {HW_W{1'b0}};
                        end else begin
                            hit_wait <= hit_wait - 1'b1;
                        end
                    end else if (req_valid) begin
                        // ---- DEMAND first (identical to expert_cache_ctrl) ----
                        if (lookup_hit) begin
                            hit        <= 1'b1;
                            resp_slot  <= lookup_slot;
                            hit_count  <= hit_count + 32'd1;
                            if (pf_flag[lookup_slot]) begin
                                pf_hit            <= pf_hit + 32'd1;
                                pf_flag[lookup_slot] <= 1'b0;   // count once
                            end
                            for (k = 0; k < SLOTS; k = k + 1)
                                if (rank[k] < rank[lookup_slot])
                                    rank[k] <= rank[k] + 1'b1;
                            rank[lookup_slot] <= {SLOT_W{1'b0}};
                            if (CACHE_HIT_LAT == 0) begin
                                resp_valid <= 1'b1;   // committed timing (no wait)
                            end else begin
                                // hold resp_valid; wait CACHE_HIT_LAT GDDR6 cycles
                                busy     <= 1'b1;
                                hit_wait <= HIT_LAT_C;
                            end
                        end else begin
                            busy            <= 1'b1;
                            flash_req       <= 1'b1;
                            flash_expert_id <= req_expert_id;
                            req_id_q        <= req_expert_id;
                            victim_q        <= victim_slot;
                            state           <= S_FETCH;
                        end
                    end else if (pf_valid) begin
                        // ---- accept a PREFETCH hint (pf_ready high here) ----
                        pf_id_q <= pf_expert_id;
                        if (!pf_in_resident) begin
                            flash_req       <= 1'b1;
                            flash_expert_id <= pf_expert_id;
                            pf_issued       <= pf_issued + 32'd1;
                            state           <= S_PF_FETCH;
                        end
                        // already resident -> skip (no-op, no Flash, stay IDLE)
                    end
                end

                //------------------------------------------------------------
                S_FETCH: begin   // DEMAND miss -- identical to expert_cache_ctrl
                    if (flash_done) begin
                        flash_req <= 1'b0;
                        valid_arr[victim_q] <= 1'b1;
                        tag_arr[victim_q]   <= req_id_q;
                        pf_flag[victim_q]   <= 1'b0;   // demand-installed, not a prefetch
                        for (k = 0; k < SLOTS; k = k + 1)
                            if (rank[k] < rank[victim_q])
                                rank[k] <= rank[k] + 1'b1;
                        rank[victim_q] <= {SLOT_W{1'b0}};
                        resp_valid <= 1'b1;
                        hit        <= 1'b0;
                        resp_slot  <= victim_q;
                        miss_count <= miss_count + 32'd1;
                        busy       <= 1'b0;
                        state      <= S_IDLE;
                    end
                end

                //------------------------------------------------------------
                S_PF_FETCH: begin   // background prefetch Flash in flight
                    if (flash_done) begin
                        // ---- install the prefetched expert (skip if resident) ----
                        flash_req <= 1'b0;
                        if (!pf_q_resident) begin
                            valid_arr[victim_slot] <= 1'b1;
                            tag_arr[victim_slot]   <= pf_id_q;
                            pf_flag[victim_slot]   <= 1'b1;   // mark as prefetched
                            for (k = 0; k < SLOTS; k = k + 1)
                                if (rank[k] < rank[victim_slot])
                                    rank[k] <= rank[k] + 1'b1;
                            rank[victim_slot] <= {SLOT_W{1'b0}};
                        end
                        // a demand arriving on this exact cycle (or already queued)
                        // is deferred to S_DMD_REST to avoid a same-cycle directory
                        // write conflict with the install above.
                        if (hit_wait != {HW_W{1'b0}}) begin
                            // a GDDR6 hit read is mid-flight (CACHE_HIT_LAT>0 only):
                            // finish the prefetch install but keep delivering that
                            // hit from S_IDLE (busy/hit_wait preserved).  The router
                            // is stalled (busy) so no new demand can race in here.
                            state <= S_IDLE;
                        end else if (req_valid && !dmd_pending) begin
                            dmd_pending    <= 1'b1;
                            dmd_pending_id <= req_expert_id;
                            busy           <= 1'b1;
                            state          <= S_DMD_REST;
                        end else if (dmd_pending) begin
                            state <= S_DMD_REST;          // busy already high
                        end else begin
                            state <= S_IDLE;
                        end
                    end else if (hit_wait != {HW_W{1'b0}}) begin
                        // GDDR6 hit read draining concurrently with the prefetch
                        // (CACHE_HIT_LAT>0 only).  Count down; prefetch keeps going.
                        if (hit_wait == {{(HW_W-1){1'b0}},1'b1}) begin
                            resp_valid <= 1'b1;
                            busy       <= 1'b0;
                            hit_wait   <= {HW_W{1'b0}};
                        end else begin
                            hit_wait <= hit_wait - 1'b1;
                        end
                    end else begin
                        // prefetch still draining: serve demand traffic.
                        if (req_valid && !dmd_pending) begin
                            if (lookup_hit) begin
                                // HIT needs no Flash.  With CACHE_HIT_LAT==0 it is
                                // served immediately, ZERO stall (committed
                                // behaviour); otherwise it pays the GDDR6 read.
                                hit        <= 1'b1;
                                resp_slot  <= lookup_slot;
                                hit_count  <= hit_count + 32'd1;
                                if (pf_flag[lookup_slot]) begin
                                    pf_hit            <= pf_hit + 32'd1;
                                    pf_flag[lookup_slot] <= 1'b0;
                                end
                                for (k = 0; k < SLOTS; k = k + 1)
                                    if (rank[k] < rank[lookup_slot])
                                        rank[k] <= rank[k] + 1'b1;
                                rank[lookup_slot] <= {SLOT_W{1'b0}};
                                if (CACHE_HIT_LAT == 0) begin
                                    resp_valid <= 1'b1;
                                end else begin
                                    busy     <= 1'b1;
                                    hit_wait <= HIT_LAT_C;
                                end
                            end else begin
                                // MISS needs Flash (busy with prefetch) -> queue it
                                dmd_pending    <= 1'b1;
                                dmd_pending_id <= req_expert_id;
                                busy           <= 1'b1;
                            end
                        end
                    end
                end

                //------------------------------------------------------------
                S_DMD_REST: begin   // re-evaluate the queued demand (directory settled)
                    if (lookup_hit) begin
                        // became resident (e.g. the prefetch we waited on) -> HIT
                        hit        <= 1'b1;
                        resp_slot  <= lookup_slot;
                        hit_count  <= hit_count + 32'd1;
                        if (pf_flag[lookup_slot]) begin
                            pf_hit            <= pf_hit + 32'd1;
                            pf_flag[lookup_slot] <= 1'b0;
                        end
                        for (k = 0; k < SLOTS; k = k + 1)
                            if (rank[k] < rank[lookup_slot])
                                rank[k] <= rank[k] + 1'b1;
                        rank[lookup_slot] <= {SLOT_W{1'b0}};
                        dmd_pending <= 1'b0;
                        if (CACHE_HIT_LAT == 0) begin
                            resp_valid <= 1'b1;   // committed timing (no wait)
                            busy       <= 1'b0;
                            state      <= S_IDLE;
                        end else begin
                            // pay the GDDR6 read in S_IDLE (busy already high).
                            busy     <= 1'b1;
                            hit_wait <= HIT_LAT_C;
                            state    <= S_IDLE;
                        end
                    end else begin
                        // genuine miss -> now own the Flash channel
                        flash_req       <= 1'b1;
                        flash_expert_id <= dmd_pending_id;
                        req_id_q        <= dmd_pending_id;
                        victim_q        <= victim_slot;
                        busy            <= 1'b1;
                        dmd_pending     <= 1'b0;
                        state           <= S_FETCH;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
