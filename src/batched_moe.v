`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// batched_moe.v  --  GLM-5.2-FP8 EXPERT-GROUPED BATCHED MoE LAYER  (ULTRA_PERF#1)
//----------------------------------------------------------------------------
// WHAT / WHY  (the biggest AGGREGATE-throughput lever -- docs/ULTRA_PERF.md #1)
//   A batch of B = PE_M token rows each routed (by moe_router_fp8) to TOPK of
//   N_EXPERT experts.  The NAIVE schedule fetches each token's TOPK experts
//   independently: B*TOPK expert-weight fetches from Flash -- the MoE bottleneck.
//
//   This module instead processes the batch EXPERT-GROUPED: it forms the UNION
//   of all experts any of the B rows routed to, and visits each UNION expert
//   ONCE.  Per union-expert e it fetches e's weights ONE TIME (one swiglu weight-
//   pull stream) and runs swiglu_expert_fp8 at PE_M over ALL B rows against that
//   single fetch, then accumulates gate_{r,e} * expert_e(x_r) into row r's MoE
//   output ONLY for the rows r that actually selected e (a per-expert row-gather
//   + per-row gate scale).  Because the router is load-balanced over N_EXPERT,
//   E[distinct experts for B tokens] = N_EXPERT*(1-(1-TOPK/N_EXPERT)^B) grows far
//   slower than B*TOPK -> the per-token expert-fetch footprint shrinks with B.
//
//   RESULT IDENTITY:  y_r = Σ_{e in TopK(r)} gate_{r,e} * expert_e(x_r), byte-for-
//   byte what you'd get processing token r ALONE -- but each expert's weights are
//   fetched once for the WHOLE batch instead of once per selecting token.
//
//----------------------------------------------------------------------------
// THE UNION + PER-EXPERT ROW-GATHER  (this module's whole job)
//   The union is enumerated implicitly by SCANNING e = 0,1,...,N_EXPERT-1 and
//   processing e iff at least one of the B rows selected it (any_has).  Experts
//   no row routed to are SKIPPED -> never fetched.  This visits each distinct
//   expert exactly once, in ASCENDING INDEX ORDER.
//
//   For the current expert e, membership is a pure combinational gather over the
//   router result: row r "has" e iff one of its TOPK sel_idx slots == e, and then
//   row r's gate for e is that slot's bf16 sel_weight.  Indices within a row's
//   TopK are distinct, so at most one slot matches per row.
//
//----------------------------------------------------------------------------
// HOW IT REUSES THE PE_M SWIGLU  (zero extra weight bandwidth)
//   swiglu_expert_fp8 is already PE_M-batched: it streams PE_M activation lanes
//   against ONE shared weight stream (w_req/w_sel/w_grp/w_k -> w_col/.../w_scale_*)
//   and emits PE_M independent rows.  We feed it ALL B token rows (xbuf) for every
//   union expert, so ONE weight fetch covers the whole batch -- the w_* request
//   stream is identical to a single-token run.  The OUTSIDE system disambiguates
//   WHICH expert's ROM/DMA to answer via the extra `cur_expert` qualifier this
//   module drives alongside swiglu's own w_sel/w_grp/w_k (mirrors how
//   glm_decoder_block qualifies its expert weight pull with fw_eidx).
//
//   The swiglu computes expert_e(x_r) for EVERY row r (it has no notion of which
//   rows selected e); the row-gather happens at the COMBINE: rows that did not
//   select e are simply not accumulated (their gate would be 0 anyway).
//
//----------------------------------------------------------------------------
// THE GATE / COMBINE NUMERICS  (§6 contract; matches glm_decoder_block)
//   Per selecting row r, per output element d:
//       facc[r][d] += bf16_to_fp32(gate_{r,e}) * bf16_to_fp32(expert_e(x_r)[d])
//   accumulated in fp32 (one fp32_mul + one fp32_add per (row,d) lane), then the
//   final y_out[r][d] = fp32_to_bf16(facc[r][d]).  gate_{r,e} is the router's
//   already-renormalized-and-scaled bf16 routed weight (sel_weight); the shared
//   expert is handled OUTSIDE this unit, exactly as in the datapath.
//
//   ORDERING == BIT-EXACTNESS:  fp32 add is NOT associative, so the per-token sum
//   is only well-defined once the summation ORDER is fixed.  Both this batched
//   schedule AND the single-token reference add a row's selected experts in
//   ASCENDING EXPERT-INDEX order (this module visits the union ascending; a
//   non-selecting expert contributes nothing, and fp32_add of a real partial sum
//   with the next term is identical whether or not 0-terms were interleaved).  So
//   row r's accumulator is bit-identical to summing ONLY r's TopK experts alone,
//   ascending by index.  (A reference that adds in TopK-gate order would need that
//   same order here -- index order is the canonical, batch-invariant choice.)
//
//----------------------------------------------------------------------------
// STYLE: synchronous ACTIVE-HIGH reset; NO latch (every reg assigned on all
//   paths); NO combinational loop (the only feedback is the registered fp32
//   accumulator and the swiglu pipeline).  Reuses swiglu_expert_fp8 + glm_fp
//   UNCHANGED.  At PE_M=1 this is just "evaluate this one token's TopK experts".
//============================================================================
module batched_moe #(
    parameter integer HIDDEN   = 128,   // model hidden size
    parameter integer INTER    = 64,    // MoE expert FFN inter size
    parameter integer TN       = 4,     // swiglu output-tile width (= matmul PE_N)
    parameter integer KMAX     = 16384, // >= max(HIDDEN,INTER) for matmul counter
    parameter integer BLK      = 128,   // weight block size -- [128,128]
    parameter integer N_EXPERT = 8,     // routed experts (real 256)
    parameter integer TOPK     = 2,     // experts per token (real 8)
    parameter integer PE_M     = 1,     // token ROWS (batch B) sharing each fetch
    // index width for an expert id -- DERIVED, matches moe_router_fp8.  Do NOT override.
    parameter integer IDXW     = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT)
)(
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    // ---- control handshake ----
    input  wire                       start,      // begin one MoE layer over B rows
    output reg                        busy,
    output reg                        done,       // 1-cycle pulse when y_out valid

    // ---- token input (PE_M rows, row-major packed) ----
    //   row r element k = x_vec[16*(HIDDEN*r + k) +: 16]
    input  wire [16*HIDDEN*PE_M-1:0]  x_vec,      // B bf16 token hidden vectors

    // ---- routing result (from moe_router_fp8; PE_M rows, row-major packed) ----
    //   row r slot t index  = sel_idx[IDXW*(TOPK*r + t) +: IDXW]
    //   row r slot t weight = sel_weight[16*(TOPK*r + t) +: 16]   (bf16 routed gate)
    input  wire [TOPK*IDXW*PE_M-1:0]  sel_idx,    // per-token top-K expert indices
    input  wire [TOPK*16*PE_M-1:0]    sel_weight, // per-token top-K bf16 gates

    // ---- expert-weight pull (swiglu's stream, QUALIFIED by which union expert) --
    //   The surrounding system answers w_col/.../w_scale_* combinationally for the
    //   FP8 weights of expert `cur_expert`, tile (w_sel,w_grp), reduction beat w_k.
    output wire [IDXW-1:0]            cur_expert, // union expert currently streaming
    output wire                       w_req,      // need weight codes this cycle
    output wire [1:0]                 w_sel,      // 0=GATE(+UP), 2=DOWN (registered)
    output wire [$clog2((INTER>HIDDEN?INTER:HIDDEN)/TN+1)-1:0] w_grp, // tile-group
    output wire [$clog2(KMAX+1)-1:0]  w_k,        // reduction index of this beat
    input  wire [8*TN-1:0]            w_col,      // E4M3 W_{gate|down} lanes
    input  wire [8*TN-1:0]            w_col_up,   // E4M3 W_up lanes (GATE/UP pass)
    input  wire [16*TN*((KMAX+BLK-1)/BLK)-1:0] w_scale_g, // gate/down block scales
    input  wire [16*TN*((KMAX+BLK-1)/BLK)-1:0] w_scale_u, // up block scales

    // ---- result (PE_M rows, row-major packed; held from done until next start) --
    //   row r output d = y_out[16*(HIDDEN*r + d) +: 16]
    output reg  [16*HIDDEN*PE_M-1:0]  y_out       // B bf16 MoE outputs
);
    `include "glm_fp.vh"

    // ---------------- derived ----------------
    // Expert SCAN counter must reach N_EXPERT (one past the last id) to terminate.
    localparam integer ECW = $clog2(N_EXPERT + 1);

    // ===================================================================
    //  Latched inputs (stable for the whole layer evaluation)
    // ===================================================================
    reg [16*HIDDEN*PE_M-1:0]  xbuf;        // B token rows -> fed to swiglu each pass
    reg [TOPK*IDXW*PE_M-1:0]  sel_idx_q;   // latched routing indices
    reg [TOPK*16*PE_M-1:0]    sel_weight_q;// latched routing gates

    // fp32 per-(row,element) combine accumulator: facc[r][d]
    reg [31:0] facc [0:PE_M-1][0:HIDDEN-1];

    // ===================================================================
    //  The shared PE_M swiglu expert: ONE weight fetch -> all B rows.
    //  Its w_* request stream is forwarded out, qualified by `cur_expert`.
    // ===================================================================
    reg                       sw_start;
    wire                      sw_busy;     // unused (FSM gates on sw_done)
    wire                      sw_done;
    wire [16*HIDDEN*PE_M-1:0] sw_y;        // expert_e(x_r) for every row r

    /* verilator lint_off UNUSEDSIGNAL */
    wire sw_busy_w = sw_busy;
    /* verilator lint_on UNUSEDSIGNAL */

    swiglu_expert_fp8 #(
        .HIDDEN(HIDDEN), .INTER(INTER), .TN(TN),
        .KMAX(KMAX), .BLK(BLK), .PE_M(PE_M)
    ) u_expert (
        .clk(clk), .rst(rst),
        .start(sw_start), .busy(sw_busy), .done(sw_done),
        .x_vec(xbuf),
        .w_req(w_req), .w_sel(w_sel), .w_grp(w_grp), .w_k(w_k),
        .w_col(w_col), .w_col_up(w_col_up),
        .w_scale_g(w_scale_g), .w_scale_u(w_scale_u),
        .y_out(sw_y)
    );

    // expert output captured at sw_done, scaled+accumulated the next cycle
    reg [16*HIDDEN*PE_M-1:0]  sw_y_q;

    // ===================================================================
    //  FSM
    // ===================================================================
    localparam [2:0] S_IDLE = 3'd0,
                     S_SCAN = 3'd1,   // advance e until a union member (or end)
                     S_RUN  = 3'd2,   // wait swiglu drain for expert cur_expert
                     S_ACC  = 3'd3,   // scale+accumulate this expert into facc
                     S_FIN  = 3'd4,   // narrow fp32 acc -> bf16 y_out
                     S_DONE = 3'd5;
    reg [2:0]     state;
    reg [ECW-1:0] escan;             // current expert index under consideration
    assign cur_expert = escan[IDXW-1:0];   // valid id whenever escan < N_EXPERT

    // ===================================================================
    //  COMBINATIONAL per-expert ROW-GATHER for `escan`.
    //    row_has[r] : does row r route to expert escan?
    //    row_gate[r]: that row's bf16 routed gate for escan (else 0)
    //    any_has    : is escan in the UNION (some row selected it)?
    // ===================================================================
    reg [15:0] row_gate [0:PE_M-1];
    reg        row_has  [0:PE_M-1];
    reg        any_has;
    integer    gr, gt;
    always @* begin
        any_has = 1'b0;
        for (gr = 0; gr < PE_M; gr = gr + 1) begin
            row_has[gr]  = 1'b0;
            row_gate[gr] = 16'b0;
            for (gt = 0; gt < TOPK; gt = gt + 1) begin
                // escan < N_EXPERT here whenever we test membership (S_SCAN guard);
                // upper escan bits beyond IDXW are 0 in range, so a low-bit compare
                // is the exact expert-id match.
                if ((escan < N_EXPERT[ECW-1:0]) &&
                    (sel_idx_q[IDXW*(TOPK*gr + gt) +: IDXW] == escan[IDXW-1:0])) begin
                    row_has[gr]  = 1'b1;
                    row_gate[gr] = sel_weight_q[16*(TOPK*gr + gt) +: 16];
                end
            end
            if (row_has[gr]) any_has = 1'b1;
        end
    end

    // ===================================================================
    //  Main control + datapath
    // ===================================================================
    integer ar, ad;
    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            sw_start     <= 1'b0;
            escan        <= {ECW{1'b0}};
            xbuf         <= {16*HIDDEN*PE_M{1'b0}};
            sel_idx_q    <= {TOPK*IDXW*PE_M{1'b0}};
            sel_weight_q <= {TOPK*16*PE_M{1'b0}};
            sw_y_q       <= {16*HIDDEN*PE_M{1'b0}};
            y_out        <= {16*HIDDEN*PE_M{1'b0}};
            for (ar = 0; ar < PE_M; ar = ar + 1)
                for (ad = 0; ad < HIDDEN; ad = ad + 1)
                    facc[ar][ad] <= 32'b0;
        end else begin
            done     <= 1'b0;   // 1-cycle pulse
            sw_start <= 1'b0;   // 1-cycle pulse

            case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    busy         <= 1'b1;
                    xbuf         <= x_vec;        // latch B token rows
                    sel_idx_q    <= sel_idx;      // latch routing
                    sel_weight_q <= sel_weight;
                    escan        <= {ECW{1'b0}};
                    for (ar = 0; ar < PE_M; ar = ar + 1)
                        for (ad = 0; ad < HIDDEN; ad = ad + 1)
                            facc[ar][ad] <= 32'b0; // clear combine accumulator
                    state        <= S_SCAN;
                end
            end

            // ---- scan the expert axis; process each UNION member once ----
            S_SCAN: begin
                if (escan == N_EXPERT[ECW-1:0]) begin
                    state <= S_FIN;               // whole union processed
                end else if (any_has) begin
                    sw_start <= 1'b1;             // fetch+run expert `escan` over B rows
                    state    <= S_RUN;
                end else begin
                    escan <= escan + 1'b1;        // not in union -> skip (no fetch)
                end
            end

            // ---- wait the shared swiglu drain for expert `escan` ----
            S_RUN: begin
                if (sw_done) begin
                    sw_y_q <= sw_y;               // expert_e(x_r) for all rows
                    state  <= S_ACC;
                end
            end

            // ---- per-expert ROW-GATHER + gate scale -> fp32 accumulate ----
            //   facc[r][d] += gate_{r,e} * expert_e(x_r)[d]   (selecting rows only;
            //   non-selecting rows add nothing -> identical to a 0-gate term).
            S_ACC: begin
                for (ar = 0; ar < PE_M; ar = ar + 1)
                    if (row_has[ar])
                        for (ad = 0; ad < HIDDEN; ad = ad + 1)
                            facc[ar][ad] <= fp32_add(
                                facc[ar][ad],
                                fp32_mul(bf16_to_fp32(row_gate[ar]),
                                         bf16_to_fp32(sw_y_q[16*(HIDDEN*ar + ad) +: 16])));
                escan <= escan + 1'b1;            // advance to next union candidate
                state <= S_SCAN;
            end

            // ---- narrow the fp32 accumulators to bf16 outputs ----
            S_FIN: begin
                for (ar = 0; ar < PE_M; ar = ar + 1)
                    for (ad = 0; ad < HIDDEN; ad = ad + 1)
                        y_out[16*(HIDDEN*ar + ad) +: 16] <= fp32_to_bf16(facc[ar][ad]);
                state <= S_DONE;
            end

            // ---- done ----
            S_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
