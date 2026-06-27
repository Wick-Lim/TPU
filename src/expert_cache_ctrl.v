`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// expert_cache_ctrl.v  --  GLM-5.2-FP8 MoE EXPERT-WEIGHT HBM CACHE CONTROLLER
//                          (the demand cache between moe_router and the experts)
//----------------------------------------------------------------------------
// FUNCTION  (the performance-critical block of the single-package MoE layer)
//   In an MoE layer the router (src/moe_router_fp8.v) picks the top-8 of 256
//   experts for each token.  Each expert's ~37 MB of FP8 weights lives in the
//   725 GB Flash COLD pool, but an expert unit (src/swiglu_expert_fp8.v) can
//   only READ weights that are resident in HBM.  This controller sits BETWEEN
//   them: given a requested expert id it answers with the HBM SLOT that holds
//   that expert's weights, fetching from Flash on a miss.  Achieved tokens/s is
//   governed by this cache's HIT RATE, so the policy is EXACT (textbook) LRU.
//
//   This is a PoC of the pure integer/control logic at small scale; the real
//   HBM/Flash backing stores are modeled in the TB as latency memories.  There
//   is NO floating point here -- only tags, valid bits, recency, and an FSM.
//
//----------------------------------------------------------------------------
// POLICY  (serialize one request at a time -- simplest correct PoC)
//   * TAG LOOKUP : on req_valid, a parallel associative compare across all SLOTS
//       (tag = resident expert id, qualified by a per-slot VALID bit).  An id is
//       resident in AT MOST one slot (installs only happen on a miss, so an id
//       is never duplicated), so the match is unambiguous.
//   * HIT  : touch the slot's recency -> MRU, drive hit=1, resp_slot=slot,
//       resp_valid (next cycle, fast -- busy stays low), bump hit_count.
//   * MISS : pick a VICTIM -- prefer the lowest-index INVALID (cold/empty) slot;
//       if every slot is valid, evict the EXACT-LRU slot.  Issue flash_req +
//       flash_expert_id, raise busy, and WAIT for flash_done (the TB returns it
//       after FLASH_LAT cycles).  Then install req_expert_id into the victim
//       (valid=1), make it MRU, drive hit=0, resp_slot=victim, resp_valid, and
//       bump miss_count.  busy drops with the response.
//
// EXACT LRU  (rank permutation -- a "move-to-front" recency stack)
//   rank[s] in 0..SLOTS-1 is slot s's recency POSITION: 0 == MRU, SLOTS-1 == LRU.
//   {rank[0..SLOTS-1]} is always a PERMUTATION of 0..SLOTS-1.  Touching slot s
//   moves it to the front: every slot strictly more-recent than s (rank<rank[s])
//   ages by one, and rank[s] becomes 0.  This is identical to a textbook
//   move-to-front LRU stack; the victim when full is the slot with rank==SLOTS-1.
//   On reset all slots are invalid and rank[s]=s.
//
//----------------------------------------------------------------------------
// TIMING  (deterministic, synchronous, ACTIVE-HIGH reset)
//   HIT  : req_valid (cycle t) -> resp_valid pulse (cycle t+1), busy stays 0.
//   MISS : req_valid (cycle t) -> busy=1 + flash_req=1 (cycle t+1) held until
//          flash_done; the cycle flash_done is seen the victim is installed and
//          resp_valid pulses (hit=0) while busy drops.  resp_valid is a 1-cycle
//          pulse.  New requests are accepted only while idle (busy low); the
//          caller stalls on busy.
//----------------------------------------------------------------------------
module expert_cache_ctrl #(
    parameter integer SLOTS     = 8,    // HBM cache slots
    parameter integer N_EXPERT  = 64,   // total distinct expert ids
    /* verilator lint_off UNUSEDPARAM */
    parameter integer FLASH_LAT = 20,   // miss fetch latency (doc only; TB models it)
    /* verilator lint_on UNUSEDPARAM */
    // Derived widths (do NOT override) -- guard the degenerate ==1 cases.
    parameter integer ID_W   = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer SLOT_W = (SLOTS    <= 1) ? 1 : $clog2(SLOTS)
)(
    input  wire                clk,
    input  wire                rst,         // synchronous, ACTIVE-HIGH

    // ---- router request ----
    input  wire                req_valid,
    input  wire [ID_W-1:0]     req_expert_id,

    // ---- response: here it is (which HBM slot) ----
    output reg                 resp_valid,
    output reg                 hit,
    output reg  [SLOT_W-1:0]   resp_slot,
    output reg                 busy,        // high while a miss is serviced (stall)

    // ---- Flash DMA handshake (TB models the backing store + latency) ----
    output reg                 flash_req,
    output reg  [ID_W-1:0]     flash_expert_id,
    input  wire                flash_done,

    // ---- stats (readable) ----
    output reg  [31:0]         hit_count,
    output reg  [31:0]         miss_count
);
    // ---- cache directory ----
    reg               valid_arr [0:SLOTS-1];   // per-slot valid bit
    reg  [ID_W-1:0]   tag_arr   [0:SLOTS-1];   // resident expert id
    reg  [SLOT_W-1:0] rank      [0:SLOTS-1];   // recency position (0=MRU)

    // ---- miss bookkeeping ----
    reg  [SLOT_W-1:0] victim_q;                // latched victim slot
    reg  [ID_W-1:0]   req_id_q;                // latched requested id

    // ---- FSM ----
    localparam [0:0] S_IDLE  = 1'b0,
                     S_FETCH = 1'b1;
    reg state;

    integer k;

    // recency position of the LRU slot (width-matched constant for compares)
    localparam [SLOT_W-1:0] LRU_POS = SLOT_W'(SLOTS-1);

    //------------------------------------------------------------------------
    // COMBINATIONAL TAG LOOKUP  (parallel associative compare across SLOTS)
    //------------------------------------------------------------------------
    reg               lookup_hit;
    reg  [SLOT_W-1:0] lookup_slot;
    always @* begin
        lookup_hit  = 1'b0;
        lookup_slot = {SLOT_W{1'b0}};
        for (k = 0; k < SLOTS; k = k + 1)
            if (valid_arr[k] && (tag_arr[k] == req_expert_id)) begin
                lookup_hit  = 1'b1;
                lookup_slot = k[SLOT_W-1:0];
            end
    end

    //------------------------------------------------------------------------
    // COMBINATIONAL VICTIM SELECT  (lowest invalid first, else EXACT-LRU)
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
    // SEQUENTIAL FSM + directory update
    //------------------------------------------------------------------------
    // NOTE: `rank` is updated via "move-to-front" -- the touched slot becomes 0
    // and every strictly-more-recent slot ages by one.  Reads use the registered
    // (old) values (non-blocking), so the permutation is preserved each touch.
    always @(posedge clk) begin
        if (rst) begin
            resp_valid      <= 1'b0;
            hit             <= 1'b0;
            resp_slot       <= {SLOT_W{1'b0}};
            busy            <= 1'b0;
            flash_req       <= 1'b0;
            flash_expert_id <= {ID_W{1'b0}};
            hit_count       <= 32'd0;
            miss_count      <= 32'd0;
            victim_q        <= {SLOT_W{1'b0}};
            req_id_q        <= {ID_W{1'b0}};
            state           <= S_IDLE;
            for (k = 0; k < SLOTS; k = k + 1) begin
                valid_arr[k] <= 1'b0;
                tag_arr[k]   <= {ID_W{1'b0}};
                rank[k]      <= k[SLOT_W-1:0];
            end
        end else begin
            resp_valid <= 1'b0;   // default: response is a 1-cycle pulse

            case (state)
                //------------------------------------------------------------
                S_IDLE: begin
                    if (req_valid) begin
                        if (lookup_hit) begin
                            // ---- HIT : answer fast, touch recency ----
                            resp_valid <= 1'b1;
                            hit        <= 1'b1;
                            resp_slot  <= lookup_slot;
                            hit_count  <= hit_count + 32'd1;
                            for (k = 0; k < SLOTS; k = k + 1)
                                if (rank[k] < rank[lookup_slot])
                                    rank[k] <= rank[k] + 1'b1;
                            rank[lookup_slot] <= {SLOT_W{1'b0}};
                            // stay in S_IDLE, busy stays low
                        end else begin
                            // ---- MISS : start the Flash fetch ----
                            busy            <= 1'b1;
                            flash_req       <= 1'b1;
                            flash_expert_id <= req_expert_id;
                            req_id_q        <= req_expert_id;
                            victim_q        <= victim_slot;   // snapshot the victim
                            state           <= S_FETCH;
                        end
                    end
                end

                //------------------------------------------------------------
                S_FETCH: begin
                    // hold flash_req + busy until the Flash DMA completes
                    if (flash_done) begin
                        flash_req <= 1'b0;
                        // install the fetched expert into the victim slot
                        valid_arr[victim_q] <= 1'b1;
                        tag_arr[victim_q]   <= req_id_q;
                        // make the victim MRU
                        for (k = 0; k < SLOTS; k = k + 1)
                            if (rank[k] < rank[victim_q])
                                rank[k] <= rank[k] + 1'b1;
                        rank[victim_q] <= {SLOT_W{1'b0}};
                        // respond
                        resp_valid <= 1'b1;
                        hit        <= 1'b0;
                        resp_slot  <= victim_q;
                        miss_count <= miss_count + 32'd1;
                        busy       <= 1'b0;
                        state      <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
