`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// expert_predictor.v -- GLM-5.2-FP8 MoE EXPERT PREFETCH PREDICTOR
//----------------------------------------------------------------------------
// Feeds the cache's prefetch hint port (expert_cache_pf pf_valid/pf_expert_id).
// Given the routing HISTORY it predicts which experts a layer will need on its
// NEXT occurrence (next token, same layer), so they can be brought into HBM
// EARLIER than the one-layer-ahead a demand cache can manage.
//
// SCHEME (tractable HW, NOT an MLP): a per-(layer,expert) recent-routing
// FREQUENCY / LOCALITY table.
//   * Each layer owns a row of N_EXPERT small SATURATING counters freq[].
//   * UPDATE: when a layer actually selects expert e (observed in the routing
//     stream) its counter is saturating-incremented.  Every HIST_DEPTH updates
//     to a layer the whole row is HALVED (aging) -- a sliding popularity window
//     that tracks the trained router's mild popularity skew + weak temporal
//     locality without locking onto a stale burst.
//   * PREDICT (combinational, for pred_layer): emit the TOP_P experts of that
//     layer's row by counter value (ties -> lowest expert id).  Each carries a
//     CONFIDENCE = its (aged, saturating) counter.  A prefetch HINT is asserted
//     only for predictions whose confidence >= CONF_THRESH.
//
// Fine-grained-MoE caveat (256 experts / top-8, weak locality): we deliberately
// OVER-predict the high-confidence experts and accept lower precision -- the
// research target is "at-least-one-correct" recall, not Mixtral-class precision.
//
// IDs: the table is addressed by (layer, expert-within-layer); predicted ids
// are GLOBAL (layer*N_EXPERT + expert) so they drop straight onto the cache's
// expert-id space (expert_cache_pf ID_W = clog2(N_LAYER*N_EXPERT)).
//
// Pure integer / control logic.  Sync active-high reset.  No latch, no comb
// loop, no floating point.
//============================================================================
module expert_predictor #(
    parameter integer N_EXPERT    = 16,   // experts per layer
    parameter integer TOPK        = 4,    // experts selected per layer (doc; default TOP_P)
    parameter integer N_LAYER     = 4,    // MoE layers
    parameter integer HIST_DEPTH  = 32,   // halve a layer's row every HIST_DEPTH updates (aging window)
    parameter integer CONF_THRESH = 2,    // emit a prefetch hint only when conf >= this
    parameter integer TOP_P       = TOPK, // predictions emitted per query
    parameter integer FREQ_W      = 4,    // counter width (saturates at 2^FREQ_W-1)
    // Derived widths (do NOT override) -- guard degenerate ==1 cases.
    parameter integer EID_W = (N_EXPERT          <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer LAY_W = (N_LAYER           <= 1) ? 1 : $clog2(N_LAYER),
    parameter integer GID_W = (N_LAYER*N_EXPERT  <= 1) ? 1 : $clog2(N_LAYER*N_EXPERT),
    parameter integer PCNT_W= (TOP_P             <  1) ? 1 : $clog2(TOP_P+1)
)(
    input  wire                       clk,
    input  wire                       rst,          // synchronous, ACTIVE-HIGH

    // ---- observe: a layer actually selected this expert (update history) ----
    input  wire                       upd_valid,
    input  wire [LAY_W-1:0]           upd_layer,
    input  wire [EID_W-1:0]           upd_expert,   // expert WITHIN the layer

    // ---- predict: top-P experts for pred_layer's NEXT occurrence (combinational)
    input  wire [LAY_W-1:0]           pred_layer,
    output reg  [TOP_P*GID_W-1:0]     pred_id_flat,   // global ids,   slice [p*GID_W +: GID_W]
    output reg  [TOP_P*FREQ_W-1:0]    pred_conf_flat, // confidences,  slice [p*FREQ_W +: FREQ_W]
    output reg  [TOP_P-1:0]           pred_valid_mask,// prediction p has freq>0
    output reg  [TOP_P-1:0]           pred_hint_mask, // prediction p is a prefetch hint (conf>=THRESH)
    output reg  [PCNT_W-1:0]          pred_hint_n,    // popcount(pred_hint_mask)

    // ---- registered single best (highest-confidence) hint -> cache pf port ----
    output reg                        pf_hint_valid,  // 1-cycle-delayed top-1 hint for pred_layer
    output reg  [GID_W-1:0]           pf_hint_expert
);
    localparam integer TBL = N_LAYER*N_EXPERT;
    localparam [FREQ_W-1:0] FREQ_MAXV = {FREQ_W{1'b1}};
    localparam integer AGE_W = (HIST_DEPTH <= 2) ? 1 : $clog2(HIST_DEPTH);
    localparam [AGE_W-1:0] AGE_LAST = AGE_W'(HIST_DEPTH-1);

    // per-(layer,expert) saturating recency-frequency counters, flattened.
    reg [FREQ_W-1:0] freq    [0:TBL-1];
    // per-layer aging tick counter.
    reg [AGE_W-1:0]  age_ctr [0:N_LAYER-1];

    integer k;

    //------------------------------------------------------------------------
    // COMBINATIONAL TOP-P SELECTION over pred_layer's row.
    //   P iterative max passes; a `taken` mask excludes already-chosen experts.
    //   Strict-> replacement scanning low->high id => ties break to lowest id.
    //   A prediction is VALID only if its counter > 0; a HINT only if >= THRESH.
    //------------------------------------------------------------------------
    integer ip, ie, base, best_eid;
    reg [N_EXPERT-1:0] taken;
    reg                best_has;
    reg [FREQ_W-1:0]   best_fr;
    reg [FREQ_W-1:0]   cur;
    reg                vld, hnt;
    always @* begin
        base            = pred_layer * N_EXPERT;
        taken           = {N_EXPERT{1'b0}};
        pred_id_flat    = {(TOP_P*GID_W){1'b0}};
        pred_conf_flat  = {(TOP_P*FREQ_W){1'b0}};
        pred_valid_mask = {TOP_P{1'b0}};
        pred_hint_mask  = {TOP_P{1'b0}};
        pred_hint_n     = {PCNT_W{1'b0}};
        for (ip = 0; ip < TOP_P; ip = ip + 1) begin
            best_has = 1'b0;
            best_fr  = {FREQ_W{1'b0}};
            best_eid = 0;
            for (ie = 0; ie < N_EXPERT; ie = ie + 1) begin
                cur = freq[base + ie];
                if (!taken[ie] && (!best_has || (cur > best_fr))) begin
                    best_has = 1'b1;
                    best_fr  = cur;
                    best_eid = ie;
                end
            end
            if (best_has) taken[best_eid] = 1'b1;
            vld = best_has && (best_fr != {FREQ_W{1'b0}});
            // confidence threshold compare (zero-extended to a wide compare)
            hnt = vld && ({{(32-FREQ_W){1'b0}}, best_fr} >= CONF_THRESH[31:0] ? 1'b1 : 1'b0);
            pred_id_flat  [ip*GID_W +: GID_W]  = GID_W'(base + best_eid);
            pred_conf_flat[ip*FREQ_W +: FREQ_W] = best_fr;
            pred_valid_mask[ip] = vld;
            pred_hint_mask[ip]  = hnt;
            if (hnt) pred_hint_n = pred_hint_n + 1'b1;
        end
    end

    //------------------------------------------------------------------------
    // SEQUENTIAL: history UPDATE (saturating increment + per-layer aging) and
    // the registered single-best hint snapshot.  Index math + the pre-increment
    // (aged) value are computed combinationally so the clocked body is purely
    // nonblocking (no comb loop -- these wires depend only on inputs/state).
    //------------------------------------------------------------------------
    wire [31:0] ub    = upd_layer*N_EXPERT;                // base index of upd_layer's row
    wire [31:0] uidx  = ub + {{(32-EID_W){1'b0}}, upd_expert}; // touched flat index
    wire        atick = (age_ctr[upd_layer] == AGE_LAST);  // aging this update?
    wire [FREQ_W-1:0] aged = atick ? (freq[uidx] >> 1) : freq[uidx];
    always @(posedge clk) begin
        if (rst) begin
            pf_hint_valid  <= 1'b0;
            pf_hint_expert <= {GID_W{1'b0}};
            for (k = 0; k < TBL; k = k + 1)
                freq[k] <= {FREQ_W{1'b0}};
            for (k = 0; k < N_LAYER; k = k + 1)
                age_ctr[k] <= {AGE_W{1'b0}};
        end else begin
            // register the highest-confidence prediction for the queried layer
            pf_hint_valid  <= pred_hint_mask[0];
            pf_hint_expert <= pred_id_flat[0 +: GID_W];

            if (upd_valid) begin
                // aging tick: every HIST_DEPTH updates to this layer, halve its row.
                if (atick) begin
                    age_ctr[upd_layer] <= {AGE_W{1'b0}};
                    for (k = 0; k < N_EXPERT; k = k + 1)
                        if ((ub + k) != uidx)
                            freq[ub + k] <= freq[ub + k] >> 1;     // other slots: halve
                end else begin
                    age_ctr[upd_layer] <= age_ctr[upd_layer] + 1'b1;
                end
                // touched expert: saturating increment (overrides any halve above)
                freq[uidx] <= (aged == FREQ_MAXV) ? FREQ_MAXV : (aged + 1'b1);
            end
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
