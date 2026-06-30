`timescale 1ns/1ps
//============================================================================
// flash_xbar_ind_fv.v  --  k-INDUCTION (UNBOUNDED) harness for flash_xbar.v
//----------------------------------------------------------------------------
// This is the UNBOUNDED companion to test/formal/flash_xbar_fv.v.  The bounded
// harness (flash_xbar_fv.v) proves the safety properties for K cycles from
// reset via plain BMC; here we prove the SAME per-channel-queue no-overflow /
// outstanding<=N_CH*QDEPTH properties by k-INDUCTION (yosys-smtbmc -i), i.e.
// for ALL reachable states (no cycle bound).
//
// WHY A SEPARATE FILE / WHY STRENGTHENING INVARIANTS
//   Naive k-induction on the bounded harness FAILS the induction STEP: the
//   harness shadow counters (inflight / outstanding / die_inflight) are plain
//   registers, so in an arbitrary (unreachable) inductive pre-state the solver
//   is free to pick e.g. inflight > outstanding, or a per-channel count that
//   exceeds QDEPTH, and then exhibit a 1-step "counterexample" that can never
//   actually occur.  The committed BMC counterexample was at
//   `assert(inflight <= outstanding)` (flash_xbar_fv.v:261).
//
//   The fix is STRENGTHENING INVARIANTS that pin the reachable state space by
//   tying the harness shadow counters to the DUT's OWN registered counters
//   (u_dut.outst[c] = outstanding reads on channel c, u_dut.cnt[c] = response
//   FIFO occupancy of channel c).  yosys 0.66 CAN reference a flattened DUT
//   array element through a hierarchical name in a continuous assign
//   (`assign w = u_dut.outst[0];`), which is what makes these invariants
//   expressible from a read-only external harness without touching the RTL.
//
//   The strengthening set (all hold on every reachable state, and together are
//   1-step inductive, so the induction STEP goes through):
//     S1  per-channel outstanding bound : u_dut.outst[c] <= QDEPTH
//           -- THE fundamental acceptance-gate invariant.  Self-inductive: a
//              channel issue requires the DUT's !ch_full == (outst[c]!=QDEPTH),
//              so outst[c] can never step past QDEPTH.
//     S2  per-channel FIFO bound        : u_dut.cnt[c]   <= QDEPTH (=RESP_QD)
//           -- self-inductive: an enq requires mem_resp_ready[c]==(cnt!=RESP_QD).
//     S3  FIFO <= outstanding (per ch)  : u_dut.cnt[c]   <= u_dut.outst[c]
//           -- in-FIFO <= in-FIFO+in-die; closes with L_die + the die contract.
//     L_out global outstanding linkage  : outstanding == Sum_c u_dut.outst[c]
//     L_in  global inflight linkage     : inflight    == Sum_c u_dut.cnt[c]
//     L_die per-channel die linkage     : die_inflight[c] == outst[c]-cnt[c]
//   With S1+L_out:  outstanding = Sum outst[c] <= N_CH*QDEPTH = CAP   (P3)
//   With S2+L_in :  inflight    = Sum cnt[c]   <= N_CH*QDEPTH = CAP   (P1a)
//   With S3+L_in+L_out: inflight <= outstanding.
//   Each linkage is inductive BY CONSTRUCTION (identical reset + identical
//   per-cycle update on both sides), so the conjunction is 1-inductive.
//
// SCOPE: small tractable instance N_CH=2, QDEPTH=2 (same as the BMC harness).
//============================================================================
module flash_xbar_ind_fv #(
    parameter integer N_CH    = 2,
    parameter integer ADDR_W  = 2,
    parameter integer DATA_W  = 1,
    parameter integer TAG_W   = 2,
    parameter integer QDEPTH  = 2,
    parameter integer BANK_LSB = 0
)(
    input wire clk,
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
    localparam integer CAP = N_CH * QDEPTH;
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
    flash_xbar #(
        .N_CH(N_CH), .ADDR_W(ADDR_W), .DATA_W(DATA_W), .TAG_W(TAG_W),
        .QDEPTH(QDEPTH), .BANK_LSB(BANK_LSB)
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
    // ISSUED-TAG SCOREBOARD  (for P2)
    // =====================================================================
    reg [(1<<TAG_W)-1:0] issued = {(1<<TAG_W){1'b0}};
    always @(posedge clk) begin
        if (rst) issued <= {(1<<TAG_W){1'b0}};
        else if (req_valid && req_ready) issued[req_tag] <= 1'b1;
    end

    // Memory contract: a channel only returns a tag that was issued.
    always @* begin
        for (ci = 0; ci < N_CH; ci = ci + 1) begin
            if (mem_resp_valid[ci])
                assume(issued[ mem_resp_tag[ci*TAG_W +: TAG_W] ]);
        end
    end

    // Per-channel conservation: a die only completes a read it was given.
    reg [IW-1:0] die_inflight [0:N_CH-1];
    genvar gd;
    generate
        for (gd = 0; gd < N_CH; gd = gd + 1) begin : g_die
            wire iss_c = mem_req_valid[gd]  && mem_req_ready[gd];
            wire cmp_c = mem_resp_valid[gd] && mem_resp_ready[gd];
            always @(posedge clk) begin
                if (rst) die_inflight[gd] <= {IW{1'b0}};
                else     die_inflight[gd] <= die_inflight[gd]
                                           + {{(IW-1){1'b0}}, iss_c}
                                           - {{(IW-1){1'b0}}, cmp_c};
            end
            always @* if (mem_resp_valid[gd])
                assume(die_inflight[gd] != {IW{1'b0}});
        end
    endgenerate

    // =====================================================================
    // HARNESS SHADOW COUNTERS  (same as BMC harness)
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

    reg [IW-1:0] iss_pop;
    always @* begin
        iss_pop = {IW{1'b0}};
        for (ci = 0; ci < N_CH; ci = ci + 1)
            if (mem_req_valid[ci] && mem_req_ready[ci])
                iss_pop = iss_pop + 1'b1;
    end

    reg [IW-1:0] outstanding = {IW{1'b0}};
    always @(posedge clk) begin
        if (rst) outstanding <= {IW{1'b0}};
        else     outstanding <= outstanding + iss_pop
                                            - {{(IW-1){1'b0}}, deq_fire};
    end

    // =====================================================================
    // DUT-INTERNAL PROBES  (per-channel registered counters)
    //   yosys 0.66 has NO hierarchical-reference support in the Verilog
    //   frontend (`u_dut.outst[0]` would parse as a fresh implicit flat wire,
    //   not the DUT register).  So these probe wires are declared here UNDRIVEN
    //   and `(* keep *)`, and are wired to the DUT's flattened internal
    //   registers \u_dut.outst[c] / \u_dut.cnt[c] by `connect` in the build
    //   script (see the Makefile `formal-ind` target / docs/FORMAL.md):
    //       connect -set \dut_outst0 \u_dut.outst[0] <-- trailing SPACE required
    //   The trailing space terminates the bracketed escaped id; without it the
    //   `[0]` is parsed as a bit-select of a non-existent wire and the connect
    //   fails.  This keeps the committed RTL untouched while letting the
    //   strengthening invariants pin the DUT's own state.  outst[c] = reads
    //   outstanding on channel c (issued-minus-drained); cnt[c] = FIFO occupancy.
    // =====================================================================
    (* keep *) wire [1:0] dut_outst0;
    (* keep *) wire [1:0] dut_outst1;
    (* keep *) wire [1:0] dut_cnt0;
    (* keep *) wire [1:0] dut_cnt1;

    // =====================================================================
    // STRENGTHENING INVARIANTS  (asserted on every operational state so they
    // also constrain the arbitrary inductive pre-state; gated only by !rst)
    // =====================================================================
    always @(posedge clk) if (!rst) begin
        // S1: per-channel outstanding bound (the acceptance-gate invariant)
        assert(dut_outst0 <= QDEPTH[1:0]);
        assert(dut_outst1 <= QDEPTH[1:0]);
        // S2: per-channel response-FIFO occupancy bound (no FIFO overflow)
        assert(dut_cnt0 <= QDEPTH[1:0]);
        assert(dut_cnt1 <= QDEPTH[1:0]);
        // S3: FIFO occupancy never exceeds outstanding on the same channel
        assert(dut_cnt0 <= dut_outst0);
        assert(dut_cnt1 <= dut_outst1);
        // L_die: harness die counter == in-die reads = outstanding - in-FIFO
        assert(die_inflight[0] == {1'b0, (dut_outst0 - dut_cnt0)});
        assert(die_inflight[1] == {1'b0, (dut_outst1 - dut_cnt1)});
        // L_out: global outstanding shadow == sum of per-channel DUT outst[]
        assert(outstanding == ({1'b0,dut_outst0} + {1'b0,dut_outst1}));
        // L_in: global inflight shadow == sum of per-channel DUT cnt[]
        assert(inflight    == ({1'b0,dut_cnt0}   + {1'b0,dut_cnt1}));
    end

    // =====================================================================
    // HEADLINE SAFETY PROPERTIES  (the ones we are proving UNBOUNDED)
    // =====================================================================
    always @(posedge clk) if (!rst) begin
        // P3 : outstanding-budget honored (outstanding <= N_CH*QDEPTH)
        assert(outstanding <= CAP[IW-1:0]);
        // P1a: per-channel-queue no-overflow (total FIFO occupancy <= capacity)
        assert(inflight <= CAP[IW-1:0]);
        // FIFO occupancy can never exceed reads in flight on the dies
        assert(inflight <= outstanding);
        // P1b: never emit a response that was not accepted (no underflow)
        if (deq_fire) assert(inflight >= {{(IW-1){1'b0}},1'b1});
    end

`ifdef WITH_P2
    // ---- P2 : no spurious response / resp_tag was issued -----------------
    always @(posedge clk) if (!rst) begin
        if (resp_valid)
            assert(issued[resp_tag]);
    end
`endif

    // =====================================================================
    // NON-VACUITY PROBES  (deliberately FALSE; select with -D<NAME>; a
    // counterexample proves the interesting state is reachable -> not vacuous)
    // =====================================================================
    always @(posedge clk) if (f_past_valid && !rst) begin
`ifdef PROBE_RESP
        assert(!resp_valid);
`endif
`ifdef PROBE_FULL
        assert(inflight != CAP[IW-1:0]);
`endif
`ifdef PROBE_OUTST
        assert(outstanding != CAP[IW-1:0]);
`endif
    end

endmodule
