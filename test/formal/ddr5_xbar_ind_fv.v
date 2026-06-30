`timescale 1ns/1ps
//============================================================================
// ddr5_xbar_ind_fv.v  --  k-INDUCTION harness for ddr5_xbar.v
//----------------------------------------------------------------------------
// NEW, formal-only harness (does NOT modify the committed DUT src/ddr5_xbar.v,
// nor the committed BMC harness test/formal/ddr5_xbar_fv.v). It instantiates the
// DUT read-only as `u_dut`, drives its inputs from FREE formal signals, and
// proves the REQUEST-PATH safety properties by TEMPORAL k-INDUCTION
// (yosys-smtbmc -i). With the base case this makes them UNBOUNDED -- they hold
// in EVERY reachable state, for all time, not just K cycles from reset.
//
//----------------------------------------------------------------------------
// WHAT IS (and is NOT) k-INDUCTIVELY PROVABLE HERE -- and WHY
//
//   The ddr5_xbar request path is PURE FEED-FORWARD COMBINATIONAL banking
//   (src/ddr5_xbar.v lines 136-151): the requester beat is routed to exactly the
//   banked channel  ch = req_addr[BANK_LSB +: CH_IDX_W]  with NO state. Its safety
//   properties (exclusive one-hot routing / no duplicated or phantom channel
//   request / correct banked-channel selection / ready coherence / payload
//   integrity) are functions of the CURRENT ports only, so they hold in every
//   state -- including the arbitrary pre-states of the induction step -- and are
//   therefore k-INDUCTIVELY (in fact 0-inductively) UNBOUNDED. They are proven
//   below (R1..R5), and they are NON-TRIVIAL: they certify the banking fabric
//   never sends one read to two channels (no double-fetch) and never fabricates a
//   channel request, for ALL input sequences of ANY length.
//
//   The RESPONSE-path FIFO properties (no-overflow / no-underflow / tag-issued,
//   proven BOUNDED in ddr5_xbar_fv.v) are NOT k-inductively provable in this
//   toolchain. Their strengthening invariant is the classic outstanding<=cap:
//       inflight == Sigma_c cnt[c]   with   cnt[c] <= RESP_QD,
//   which fundamentally references the DUT's INTERNAL per-channel occupancy
//   cnt[] / pointers head[],tail[] / round-robin rr. yosys 0.66 + write_smt2 (no
//   SymbiYosys, `bind` dropped) cannot reference those internal nets from a
//   read-only external harness:
//     * a hierarchical reference `u_dut.cnt[c]` does NOT resolve -- yosys emits
//       "Identifier '\u_dut.cnt' is implicitly declared" + "Range select out of
//       bounds ... undef" and binds it to a FREE 1-bit PHANTOM net. (Proof: the
//       harness assert  `u_dut.cnt[0] <= 5`  FAILS in BMC, though the real cnt is
//       2-bit with reachable max RESP_QD=2 -- the solver freely drove the phantom
//       above 5.) The real flattened nets are named `\u_dut.cnt[0]` /
//       `\u_dut.cnt[1]` (the brackets are part of the net name), which no Verilog
//       identifier can spell.
//     * reconstructing cnt[] from OBSERVABLE ports alone fails too: per-channel
//       drain attribution needs the round-robin pointer `rr` and grant `gnt`,
//       which are neither ports nor referenceable, so a shadow-equivalence
//       induction cannot be closed.
//   Hence the response-FIFO properties remain BOUNDED (BMC, ddr5_xbar_fv.v); an
//   unbounded proof needs internal observability (SymbiYosys with working `bind`,
//   or in-RTL assertions added to the DUT) -- out of scope for a read-only harness
//   here. See the coverage note returned with this work.
//
// PARAMS kept SMALL (same instance as the BMC harness): N_CH=2.
//============================================================================
module ddr5_xbar_ind_fv #(
    parameter integer N_CH    = 2,
    parameter integer ADDR_W  = 2,
    parameter integer DATA_W  = 1,
    parameter integer TAG_W   = 2,
    parameter integer RESP_QD = 2,
    parameter integer BANK_LSB = 0
)(
    input wire clk,
    // ---- FREE formal inputs (unconstrained -> $anyseq) ----
    input wire                     req_valid,
    input wire [ADDR_W-1:0]        req_addr,
    input wire [TAG_W-1:0]         req_tag,
    input wire [N_CH-1:0]          mem_req_ready,
    input wire [N_CH-1:0]          mem_resp_valid,
    input wire [N_CH*DATA_W-1:0]   mem_resp_data,
    input wire [N_CH*TAG_W-1:0]    mem_resp_tag,
    input wire                     resp_ready
);
    localparam integer CH_IDX_W = (N_CH <= 1) ? 1 : $clog2(N_CH);

    // ---------------- reset bookkeeping (request path has no state, so the
    // routing properties hold regardless; reset is wired only for the DUT) ------
    reg [1:0] rst_cnt = 2'd0;
    always @(posedge clk) if (rst_cnt != 2'd3) rst_cnt <= rst_cnt + 2'd1;
    wire rst = (rst_cnt == 2'd0);

    reg f_past_valid = 1'b0;
    always @(posedge clk) f_past_valid <= 1'b1;

    // ---------------- DUT outputs ----------------
    wire                  req_ready;
    wire [N_CH-1:0]       mem_req_valid;
    wire [N_CH*ADDR_W-1:0] mem_req_addr;
    wire [N_CH*TAG_W-1:0]  mem_req_tag;
    wire [N_CH-1:0]       mem_resp_ready;
    wire                  resp_valid;
    wire [DATA_W-1:0]     resp_data;
    wire [TAG_W-1:0]      resp_tag;

    // ---------------- DUT ----------------
    ddr5_xbar #(
        .N_CH(N_CH), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
        .ROW_LAT(4), .RESP_QD(RESP_QD), .BANK_LSB(BANK_LSB)
    ) u_dut (
        .clk(clk), .rst(rst),
        .req_valid(req_valid), .req_ready(req_ready),
        .req_addr(req_addr), .req_tag(req_tag),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .resp_valid(resp_valid), .resp_ready(resp_ready),
        .resp_data(resp_data), .resp_tag(resp_tag)
    );

    // =====================================================================
    // REFERENCE banking model -- recomputed from the SAME ports the DUT sees
    // (this is NOT a hierarchical reference into the DUT; it is the spec).
    // =====================================================================
    wire [CH_IDX_W-1:0] exp_ch = req_addr[BANK_LSB +: CH_IDX_W];

    // popcount of mem_req_valid (the DUT's channel-request one-hot)
    integer ci;
    reg [CH_IDX_W:0] mrv_pop;
    always @* begin
        mrv_pop = {(CH_IDX_W+1){1'b0}};
        for (ci = 0; ci < N_CH; ci = ci + 1)
            mrv_pop = mrv_pop + {{CH_IDX_W{1'b0}}, mem_req_valid[ci]};
    end

    // =====================================================================
    // REQUEST-PATH SAFETY PROPERTIES  (stateless -> k-inductive UNBOUNDED)
    // Asserted UNCONDITIONALLY: they are combinational identities that hold in
    // every state, so temporal induction may assume/discharge them everywhere.
    // =====================================================================
    genvar gc;
    generate
        for (gc = 0; gc < N_CH; gc = gc + 1) begin : g_req_props
            always @(posedge clk) begin
                // R4 : payload integrity -- the broadcast request word carried to
                //      every channel equals the requester's addr/tag (no corruption)
                assert(mem_req_addr[gc*ADDR_W +: ADDR_W] == req_addr);
                assert(mem_req_tag [gc*TAG_W  +: TAG_W ] == req_tag );
                // R5 : a channel sees a request ONLY if it is the banked channel
                //      AND the requester is offering one (no phantom per-channel req)
                assert(mem_req_valid[gc] ==
                       (req_valid && (exp_ch == gc[CH_IDX_W-1:0])));
            end
        end
    endgenerate

    always @(posedge clk) begin
        // R1 : EXCLUSIVE routing -- at most one channel request, and exactly one
        //      iff the requester is valid. (no read duplicated to two channels;
        //      no spurious channel request when the requester is idle.)
        assert(mrv_pop == {{CH_IDX_W{1'b0}}, req_valid});
        // R2 : the banked (selected) channel is the one that fires.
        if (req_valid) assert(mem_req_valid[exp_ch]);
        // R3 : READY coherence -- the requester is accepted iff ITS banked
        //      channel can accept (head-of-line back-pressure is faithful).
        assert(req_ready == mem_req_ready[exp_ch]);
    end

    // =====================================================================
    // NON-VACUITY PROBES (each a DELIBERATELY FALSE claim; selecting one with
    // -D<NAME> must yield a BMC counterexample -> the interesting state is
    // reachable / the assumptions are consistent / the properties are not vacuous).
    // =====================================================================
    always @(posedge clk) if (f_past_valid && !rst) begin
`ifdef PROBE_REQFIRE
        assert(mrv_pop == 0);                 // FAIL: a channel request can fire
`endif
`ifdef PROBE_BOTHCH
        // FAIL: BOTH channels are individually reachable as the routed target
        assert(!(req_valid && exp_ch == {CH_IDX_W{1'b1}}));
`endif
`ifdef PROBE_READY
        assert(!req_ready);                   // FAIL: a request can be accepted
`endif
    end
endmodule
