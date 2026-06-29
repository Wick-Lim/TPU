`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// expert_prefetch_top.v -- GLM-5.2-FP8 MoE PREDICTIVE-PREFETCH INTEGRATION
//----------------------------------------------------------------------------
// WIRING ONLY (the cache + predictor logic are unchanged; this is the knobs).
//
// Connects expert_predictor's registered prediction HINT output to
// expert_cache_pf's PREFETCH input so the predictor drives a DEEPER-than-one-
// layer prefetch: it predicts the experts for pred_layer = (current + LOOKAHEAD)
// while the SAME incoming routing stream feeds (a) the predictor's UPDATE (the
// actual routing as it streams) and (b) the cache's DEMAND (the experts needed
// THIS layer).  The predictor therefore runs LOOKAHEAD layers ahead of demand,
// and its top-1 high-confidence hint is injected as the cache prefetch hint.
//
//   incoming routing stream (in_valid, in_layer, in_expert-within-layer)
//      |                                         |
//      |  upd_*  (history update, this layer)    |  req_*  (demand, this layer,
//      v                                         v          GLOBAL expert id)
//   +-----------------+   pf_hint_valid /      +------------------+
//   | expert_predictor|   pf_hint_expert  -->  |  expert_cache_pf |
//   |  pred_layer =   |   (top-1 hint for      |   pf_valid /     |
//   | current+LOOKAHEAD)  layer current+LA)    |   pf_expert_id   |
//   +-----------------+                        +------------------+
//
// Demand-first arbitration inside the cache means a prefetch hint is only
// ACCEPTED when the demand path is idle (gaps in the stream) -- best-effort,
// never stalls demand.  The cache's hit/miss/demand-stall + pf counters are
// surfaced at the top.
//
// IDs: predictor predicted ids are GLOBAL (layer*N_EXPERT + expert), which is
// exactly the cache's expert-id space (cache N_EXPERT = N_LAYER*N_EXPERT).
//
// Pure integer / control.  Sync active-high reset.  No latch, no comb loop.
//============================================================================
module expert_prefetch_top #(
    parameter integer N_EXPERT    = 16,   // experts PER LAYER
    parameter integer N_LAYER     = 4,    // MoE layers
    parameter integer SLOTS       = 8,    // HBM cache slots
    parameter integer LOOKAHEAD   = 2,    // predict experts this many layers AHEAD of demand
    parameter integer CONF_THRESH = 2,    // predictor: emit a prefetch hint only when conf >= this
    parameter integer TOPK        = 4,    // experts selected per layer (predictor TOP_P)
    parameter integer HIST_DEPTH  = 32,   // predictor aging window
    parameter integer FREQ_W      = 4,    // predictor counter width
    parameter integer FLASH_LAT   = 20,   // cache miss fetch latency (doc; TB models it)
    // Derived widths (do NOT override).
    parameter integer TOTAL  = N_LAYER*N_EXPERT,           // total distinct expert ids
    parameter integer EID_W  = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer LAY_W  = (N_LAYER  <= 1) ? 1 : $clog2(N_LAYER),
    parameter integer ID_W   = (TOTAL    <= 1) ? 1 : $clog2(TOTAL),   // == predictor GID_W
    parameter integer SLOT_W = (SLOTS    <= 1) ? 1 : $clog2(SLOTS)
)(
    input  wire                clk,
    input  wire                rst,          // synchronous, ACTIVE-HIGH

    // ---- incoming routing stream (one routed expert per beat) ----
    input  wire                in_valid,     // a routed (layer,expert) this beat
    input  wire [LAY_W-1:0]    in_layer,     // layer of this routed expert
    input  wire [EID_W-1:0]    in_expert,    // expert WITHIN the layer

    // ---- cache demand response ----
    output wire                resp_valid,
    output wire                hit,
    output wire [SLOT_W-1:0]   resp_slot,
    output wire                busy,

    // ---- Flash DMA handshake (TB models the backing store + latency) ----
    output wire                flash_req,
    output wire [ID_W-1:0]     flash_expert_id,
    input  wire                flash_done,

    // ---- prefetch hint visibility ----
    output wire                pf_ready,        // cache accepted (idle) this cycle
    output wire                pf_hint_valid,   // predictor issued a hint this cycle
    output wire [ID_W-1:0]     pf_hint_expert,  // its (global) expert id

    // ---- cache stats (surfaced) ----
    output wire [31:0]         hit_count,
    output wire [31:0]         miss_count,
    output wire [31:0]         demand_stall_cycles,
    output wire [31:0]         pf_issued,
    output wire [31:0]         pf_hit
);
    //------------------------------------------------------------------------
    // DEMAND global id: this layer's routed expert, on the cache's id space.
    //------------------------------------------------------------------------
    /* verilator lint_off UNUSEDSIGNAL */  // only the low ID_W/LAY_W bits matter
    wire [31:0] dmd_global = (in_layer * N_EXPERT) + {{(32-EID_W){1'b0}}, in_expert};
    //------------------------------------------------------------------------
    // LOOKAHEAD: query the predictor for the layer LOOKAHEAD layers ahead of the
    // demand stream (wrapped into [0,N_LAYER)).  LA_MOD is a compile-time const
    // so this is a single conditional subtract, never a runtime modulo.
    //------------------------------------------------------------------------
    localparam integer LA_MOD = (N_LAYER <= 1) ? 0 : (LOOKAHEAD % N_LAYER);
    wire [31:0] pl_sum  = {{(32-LAY_W){1'b0}}, in_layer} + LA_MOD[31:0];
    wire [31:0] pl_wrap = (pl_sum >= N_LAYER[31:0]) ? (pl_sum - N_LAYER[31:0]) : pl_sum;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [ID_W-1:0]  req_expert_id = dmd_global[ID_W-1:0];
    wire [LAY_W-1:0] pred_layer    = pl_wrap[LAY_W-1:0];

    // unused predictor outputs (only the registered top-1 hint is wired to the
    // cache); named here so the instance has no empty pin connections.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [TOPK*ID_W-1:0]   u_pred_id_flat;
    wire [TOPK*FREQ_W-1:0] u_pred_conf_flat;
    wire [TOPK-1:0]        u_pred_valid_mask;
    wire [TOPK-1:0]        u_pred_hint_mask;
    wire [$clog2(TOPK+1)-1:0] u_pred_hint_n;
    /* verilator lint_on UNUSEDSIGNAL */

    //------------------------------------------------------------------------
    // PREDICTOR : UPDATE from the streaming routing, PREDICT LOOKAHEAD ahead.
    //   Its registered single-best hint (pf_hint_valid/pf_hint_expert) is the
    //   prefetch hint that drives the cache.
    //------------------------------------------------------------------------
    expert_predictor #(
        .N_EXPERT   (N_EXPERT),
        .TOPK       (TOPK),
        .N_LAYER    (N_LAYER),
        .HIST_DEPTH (HIST_DEPTH),
        .CONF_THRESH(CONF_THRESH),
        .TOP_P      (TOPK),
        .FREQ_W     (FREQ_W)
    ) u_pred (
        .clk            (clk),
        .rst            (rst),
        // observe the actual routing as it streams
        .upd_valid      (in_valid),
        .upd_layer      (in_layer),
        .upd_expert     (in_expert),
        // predict LOOKAHEAD layers ahead of the demand
        .pred_layer     (pred_layer),
        .pred_id_flat   (u_pred_id_flat),
        .pred_conf_flat (u_pred_conf_flat),
        .pred_valid_mask(u_pred_valid_mask),
        .pred_hint_mask (u_pred_hint_mask),
        .pred_hint_n    (u_pred_hint_n),
        // registered top-1 hint -> cache prefetch port
        .pf_hint_valid  (pf_hint_valid),
        .pf_hint_expert (pf_hint_expert)
    );

    //------------------------------------------------------------------------
    // CACHE : DEMAND from the streaming routing (this layer), PREFETCH from the
    //   predictor's deeper-than-one-layer hint.  Demand-first arbitration and
    //   all counters are the committed cache, unchanged.
    //------------------------------------------------------------------------
    expert_cache_pf #(
        .SLOTS    (SLOTS),
        .N_EXPERT (TOTAL),      // global expert id space = N_LAYER*N_EXPERT
        .FLASH_LAT(FLASH_LAT)
    ) u_cache (
        .clk                (clk),
        .rst                (rst),
        // demand: experts needed THIS layer
        .req_valid          (in_valid),
        .req_expert_id      (req_expert_id),
        .resp_valid         (resp_valid),
        .hit                (hit),
        .resp_slot          (resp_slot),
        .busy               (busy),
        // prefetch hint: predictor, LOOKAHEAD layers ahead
        .pf_valid           (pf_hint_valid),
        .pf_expert_id       (pf_hint_expert),
        .pf_ready           (pf_ready),
        // Flash DMA
        .flash_req          (flash_req),
        .flash_expert_id    (flash_expert_id),
        .flash_done         (flash_done),
        // stats
        .hit_count          (hit_count),
        .miss_count         (miss_count),
        .demand_stall_cycles(demand_stall_cycles),
        .pf_issued          (pf_issued),
        .pf_hit             (pf_hit)
    );
endmodule
/* verilator lint_on DECLFILENAME */
