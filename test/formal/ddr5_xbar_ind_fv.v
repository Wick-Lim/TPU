`timescale 1ns/1ps
//============================================================================
// ddr5_xbar_ind_fv.v  --  k-INDUCTION (UNBOUNDED) harness for ddr5_xbar.v
//----------------------------------------------------------------------------
// UNBOUNDED companion to test/formal/ddr5_xbar_fv.v.  The bounded harness
// proves the response-FIFO no-overflow / no-underflow (conservation form) for
// K cycles from reset via plain BMC; here we lift the SAME properties to an
// UNBOUNDED proof by temporal k-INDUCTION (yosys-smtbmc -i), i.e. for ALL
// reachable states (no cycle bound).
//
// WHY A SEPARATE FILE / WHY STRENGTHENING INVARIANTS
//   Naive k-induction on the black-box shadow counter FAILS the induction
//   STEP: `inflight` (total accepted - total emitted) is a plain harness
//   register, so in an arbitrary (unreachable) inductive pre-state the solver
//   is free to pick inflight > CAP, or inflight==0 while a response fires, and
//   exhibit a 1-step "counterexample" that can never actually occur.
//
//   The fix is STRENGTHENING INVARIANTS that pin the reachable state space by
//   tying the harness shadow counter to the DUT's OWN registered per-channel
//   response-FIFO occupancy counters cnt[0..N_CH-1].  ddr5_xbar has NO
//   outstanding-read register (unlike flash_xbar); its only response-side
//   state is the per-channel FIFO occupancy cnt[c], so the conservation proof
//   closes on cnt[] alone:
//     S2  per-channel FIFO bound : u_dut.cnt[c] <= RESP_QD
//           -- self-inductive: an enq requires mem_resp_ready[c]==(cnt!=RESP_QD),
//              so cnt[c] can never step past RESP_QD (no FIFO overflow); a deq of
//              channel c requires the round-robin grant to select c, which the
//              combinational arbiter grants only when fifo_ne[c]==(cnt[c]!=0),
//              so cnt[c] never underflows either.
//     L_in global inflight linkage : inflight == Sum_c u_dut.cnt[c]
//           -- inductive BY CONSTRUCTION: identical reset (all 0) and identical
//              per-cycle update on both sides.  Harness enq_pop == Sum_c
//              enq_fire[c] (both count mem_resp_valid[c]&&mem_resp_ready[c]) and
//              harness deq_fire == Sum_c deq_fire[c] (exactly one channel is
//              granted per cycle: gnt_valid&&resp_ready).
//   With S2+L_in:  inflight = Sum cnt[c] <= N_CH*RESP_QD = CAP           (P1a)
//   With L_in    :  a firing deq => resp_valid => gnt_valid => some cnt[c]>=1
//                   => Sum>=1 => inflight>=1                             (P1b)
//   Each conjunct is 1-step inductive, so the conjunction closes (min k=2;
//   the target runs k=12 for margin, matching the BMC bound).
//
// HOW THE INTERNAL COUNTER IS REACHED (connect-bind, mirrors flash_xbar_ind_fv):
//   yosys 0.66 has NO hierarchical-reference support in the Verilog frontend
//   (`u_dut.cnt[0]` would parse as a fresh implicit flat wire, not the DUT
//   register).  So the cnt-probe wires below are declared UNDRIVEN and
//   `(* keep *)`, and are wired to the DUT's flattened internal registers
//   \u_dut.cnt[c] by `connect` in the build script -- the standalone yosys
//   command that mirrors the Makefile formal-ind `run_kind` recipe for an
//   *_ind_fv proof (analogous to FLASH_IND_CONN):
//       connect -set \dut_cnt0 \u_dut.cnt[0] <-- trailing SPACE required
//   The trailing space terminates the bracketed escaped id; without it the
//   `[0]` is parsed as a bit-select of a non-existent wire and connect fails.
//   The committed RTL is untouched.  cnt[c] = response-FIFO occupancy of
//   channel c (CNT_W = clog2(RESP_QD+1) = 2 bits for RESP_QD=2).
//
// SCOPE: small tractable instance N_CH=2, RESP_QD=2 (same as the BMC harness).
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
    localparam integer CAP = N_CH * RESP_QD;
    localparam integer IW  = 3;

    // ---------------- reset & past-valid bookkeeping ----------------
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

    integer ci;

    // =====================================================================
    // REQUEST-PATH ROUTING SAFETY  (stateless -> 0/k-inductive UNBOUNDED)
    //   Kept from the prior request-path harness so this file proves BOTH the
    //   feed-forward banking safety AND the response-FIFO conservation.  These
    //   are combinational identities over the CURRENT ports (the request path
    //   has no state), so they hold in every reachable state -- including the
    //   induction step's arbitrary pre-state -- and pass trivially.
    // =====================================================================
    wire [CH_IDX_W-1:0] exp_ch = req_addr[BANK_LSB +: CH_IDX_W];

    reg [CH_IDX_W:0] mrv_pop;    // popcount of the DUT's channel-request one-hot
    always @* begin
        mrv_pop = {(CH_IDX_W+1){1'b0}};
        for (ci = 0; ci < N_CH; ci = ci + 1)
            mrv_pop = mrv_pop + {{CH_IDX_W{1'b0}}, mem_req_valid[ci]};
    end

    genvar gc;
    generate
        for (gc = 0; gc < N_CH; gc = gc + 1) begin : g_req_props
            always @(posedge clk) begin
                // R4: payload integrity -- broadcast request word == requester's
                assert(mem_req_addr[gc*ADDR_W +: ADDR_W] == req_addr);
                assert(mem_req_tag [gc*TAG_W  +: TAG_W ] == req_tag );
                // R5: a channel fires only if it is the banked channel and valid
                assert(mem_req_valid[gc] ==
                       (req_valid && (exp_ch == gc[CH_IDX_W-1:0])));
            end
        end
    endgenerate

    always @(posedge clk) begin
        // R1: EXCLUSIVE routing -- exactly one channel req iff requester valid
        assert(mrv_pop == {{CH_IDX_W{1'b0}}, req_valid});
        // R2: the banked (selected) channel is the one that fires
        if (req_valid) assert(mem_req_valid[exp_ch]);
        // R3: READY coherence -- accepted iff ITS banked channel can accept
        assert(req_ready == mem_req_ready[exp_ch]);
    end

    // =====================================================================
    // ENVIRONMENT ASSUMPTIONS  (identical legal protocol to the BMC harness)
    // =====================================================================
    // (A) Requester producer stability.
    always @(posedge clk) if (f_past_valid && !$past(rst)) begin
        if ($past(req_valid) && !$past(req_ready)) begin
            assume(req_valid);
            assume(req_addr == $past(req_addr));
            assume(req_tag  == $past(req_tag));
        end
    end

    // (B) Channel response producer stability.
    genvar gb;
    generate
        for (gb = 0; gb < N_CH; gb = gb + 1) begin : g_stab
            always @(posedge clk) if (f_past_valid && !$past(rst)) begin
                if ($past(mem_resp_valid[gb]) && !$past(mem_resp_ready[gb])) begin
                    assume(mem_resp_valid[gb]);
                    assume(mem_resp_data[gb*DATA_W +: DATA_W] ==
                           $past(mem_resp_data[gb*DATA_W +: DATA_W]));
                    assume(mem_resp_tag[gb*TAG_W +: TAG_W] ==
                           $past(mem_resp_tag[gb*TAG_W +: TAG_W]));
                end
            end
        end
    endgenerate

    // =====================================================================
    // HARNESS SHADOW COUNTER  (black-box conservation form, same as BMC harness)
    //   inflight := (total channel responses ACCEPTED) - (total EMITTED)
    // =====================================================================
    reg [IW-1:0] enq_pop;
    always @* begin
        enq_pop = {IW{1'b0}};
        for (ci = 0; ci < N_CH; ci = ci + 1)
            if (mem_resp_valid[ci] && mem_resp_ready[ci])
                enq_pop = enq_pop + 1'b1;
    end
    wire deq_fire = resp_valid && resp_ready;

    reg [IW-1:0] inflight = {IW{1'b0}};
    always @(posedge clk) begin
        if (rst) inflight <= {IW{1'b0}};
        else     inflight <= inflight + enq_pop
                                      - {{(IW-1){1'b0}}, deq_fire};
    end

    // =====================================================================
    // DUT-INTERNAL PROBES  (per-channel registered FIFO-occupancy counters)
    //   Declared UNDRIVEN + (* keep *); wired to \u_dut.cnt[c] by `connect`
    //   in the build script (see header + the standalone yosys invocation).
    // =====================================================================
    (* keep *) wire [1:0] dut_cnt0;
    (* keep *) wire [1:0] dut_cnt1;

    // =====================================================================
    // STRENGTHENING INVARIANTS  (asserted on every operational state so they
    // also constrain the arbitrary inductive pre-state; gated only by !rst)
    // =====================================================================
    always @(posedge clk) if (!rst) begin
        // S2: per-channel response-FIFO occupancy bound (no FIFO overflow)
        assert(dut_cnt0 <= RESP_QD[1:0]);
        assert(dut_cnt1 <= RESP_QD[1:0]);
        // L_in: global inflight shadow == sum of per-channel DUT cnt[]
        assert(inflight == ({1'b0,dut_cnt0} + {1'b0,dut_cnt1}));
    end

    // =====================================================================
    // HEADLINE SAFETY PROPERTIES  (the ones we are proving UNBOUNDED)
    // =====================================================================
    always @(posedge clk) if (!rst) begin
        // P1a: response-FIFO no-overflow (total occupancy <= provisioned cap)
        assert(inflight <= CAP[IW-1:0]);
        // P1b: never emit a response that was not accepted (no underflow)
        if (deq_fire) assert(inflight >= {{(IW-1){1'b0}},1'b1});
    end

    // =====================================================================
    // NON-VACUITY PROBES  (deliberately FALSE; select with -D<NAME>; a
    // counterexample proves the interesting state is reachable -> not vacuous)
    // =====================================================================
    always @(posedge clk) if (f_past_valid && !rst) begin
`ifdef PROBE_RESP
        assert(!resp_valid);                 // expect FAIL: a response can fire
`endif
`ifdef PROBE_FULL
        assert(inflight != CAP[IW-1:0]);     // expect FAIL: FIFOs can reach FULL
`endif
`ifdef PROBE_ENQ
        assert(enq_pop == {IW{1'b0}});       // expect FAIL: a channel push happens
`endif
    end

endmodule
