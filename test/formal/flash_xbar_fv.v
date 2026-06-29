`timescale 1ns/1ps
//============================================================================
// flash_xbar_fv.v  --  FORMAL HARNESS (bounded model checking) for flash_xbar.v
//----------------------------------------------------------------------------
// NEW, standalone formal-only harness.  It does NOT modify the committed DUT
// (src/flash_xbar.v); it instantiates it as `u_dut` and drives the DUT inputs
// from FREE formal signals (the harness top-level input ports are unconstrained
// = $anyseq under yosys -formal) that are CONSTRAINED with assume() to the legal
// valid/ready protocol, then states the safety properties as immediate assert().
//
// flash_xbar is the high-latency Flash analogue of ddr5_xbar: identical banked
// read fabric + per-channel response FIFO + rotate-mask drain arbiter, but with
// the NEW knob QDEPTH = MAX outstanding reads per channel (deep, to hide the
// thousand-cycle NAND latency).  The per-channel response FIFO is sized to
// RESP_QD = QDEPTH, so the per-channel queue capacity proven below is QDEPTH.
//
// PARAMS kept SMALL so BMC is tractable:
//   N_CH=2, ADDR_W=2, DATA_W=1, TAG_W=2, QDEPTH=2.   (FLASH_LAT is a TB-stub-only
//   parameter -- the fabric is latency-agnostic -- so its value is irrelevant to
//   the control/FIFO logic proven here; left at default.)
//
// Properties proven:
//   P1 (no per-channel-queue overflow / FIFO bounds): the fabric buffers every
//       response it ACCEPTS from a channel until it EMITS it to the requester;
//       the provisioned total FIFO storage is N_CH*QDEPTH.  P1a: buffered
//       (inflight) never exceeds that capacity (no overflow / lost response).
//       P1b: the fabric never emits a response it did not accept (no underflow /
//       no phantom DATA beat).
//   P2 (no spurious response / resp_tag was issued): whenever the fabric raises
//       resp_valid, resp_tag is a tag that was actually ISSUED earlier on the
//       requester port (covers "no phantom response" and "no corrupted tag":
//       a fabricated/garbage tag would have issued[tag]==0).  Environment
//       assumption: a Flash channel only returns responses for tags it was
//       actually asked for -- the memory contract, modeled by assume() below.
//   P3 (outstanding budget honored): the total reads in flight (issued to the
//       channels but not yet drained to the requester) never exceeds the total
//       provisioned outstanding budget N_CH*QDEPTH.  This is the flash-specific
//       deep-queue invariant: the fabric counts every issue and caps it, so the
//       per-channel response FIFO (also sized QDEPTH) can never be over-filled.
//============================================================================
module flash_xbar_fv #(
    parameter integer N_CH    = 2,
    parameter integer ADDR_W  = 2,
    parameter integer DATA_W  = 1,   // datapath width irrelevant to control/FIFO
    parameter integer TAG_W   = 2,   // >=2 so "wrong tag" is expressible
    parameter integer QDEPTH  = 2,   // per-channel outstanding budget (== RESP_QD)
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
    localparam integer CAP = N_CH * QDEPTH;   // total provisioned FIFO/outstanding
    localparam integer IW  = 3;               // 0..7: covers CAP(=4)+overflow probe

    // ---------------- reset & past-valid bookkeeping ----------------
    // rst is asserted ONLY during cycle 0 (>=1 cycle of synchronous reset),
    // so from cycle 1 onward the DUT state is the clean reset state.
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

    // =====================================================================
    // ENVIRONMENT ASSUMPTIONS  (legal protocol)
    // =====================================================================
    integer ci;

    // (A) Requester producer stability: if a request is offered but not
    //     accepted, it must be held stable until accepted (valid/ready rule).
    always @(posedge clk) if (f_past_valid && !$past(rst)) begin
        if ($past(req_valid) && !$past(req_ready)) begin
            assume(req_valid);
            assume(req_addr == $past(req_addr));
            assume(req_tag  == $past(req_tag));
        end
    end

    // (B) Channel response producer stability: if a channel offers a response
    //     but the fabric back-pressures it (mem_resp_ready low), the channel
    //     must hold that response (valid+data+tag) stable until accepted.
    //     (A real NAND die completes a read and holds the beat; it cannot drop
    //     a completed read -- which is exactly why the fabric sizes the FIFO so
    //     mem_resp_ready can never need to fall.)
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
    //   issued[t] := a request carrying tag t was accepted on the requester
    //   port at some earlier cycle.  Monotonic (only set, cleared on reset).
    // =====================================================================
    reg [(1<<TAG_W)-1:0] issued = {(1<<TAG_W){1'b0}};
    always @(posedge clk) begin
        if (rst) issued <= {(1<<TAG_W){1'b0}};
        else if (req_valid && req_ready) issued[req_tag] <= 1'b1;
    end

    // Memory contract: a channel only ever returns a response for a tag that
    // was actually issued to the fabric.  (Without this, "no phantom response"
    // is not a property of the FABRIC -- the phantom would be injected by the
    // memory model itself.)
    always @* begin
        for (ci = 0; ci < N_CH; ci = ci + 1) begin
            if (mem_resp_valid[ci])
                assume(issued[ mem_resp_tag[ci*TAG_W +: TAG_W] ]);
        end
    end

    // Memory contract (per channel, conservation): a Flash die can only COMPLETE
    // a read it was actually GIVEN -- it cannot manufacture completions out of
    // nothing.  Track the per-channel reads currently inside the die
    //   die_inflight[c] = (issued to c) - (completed by c)
    // (issue := mem_req_valid[c]&&mem_req_ready[c];  complete := the accepted
    //  enq mem_resp_valid[c]&&mem_resp_ready[c]) and assume the channel only
    // raises mem_resp_valid[c] when it has >=1 such read in flight.  A read
    // issued this cycle increments die_inflight only NEXT cycle, so a die can
    // never complete a read in the same cycle it was issued (latency >= 1),
    // matching real NAND.  This is the necessary companion to the tag contract:
    // without it the free environment could inject completions for reads never
    // issued to that channel.
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
    // SAFETY PROPERTIES
    // =====================================================================

    // ---- P2 : no spurious response / resp_tag was issued ----------------
    always @(posedge clk) if (f_past_valid && !rst) begin
        if (resp_valid)
            assert(issued[resp_tag]);                       // P2
    end

    // ---- P1 : per-channel-queue no-overflow (black-box conservation) ----
    //   The fabric buffers every response it ACCEPTS from a channel
    //   (enq := mem_resp_valid[c] && mem_resp_ready[c]) until it EMITS it to
    //   the requester (deq := resp_valid && resp_ready).  The per-channel
    //   response FIFOs (sized RESP_QD=QDEPTH each) give a TOTAL buffer capacity
    //   of N_CH*QDEPTH.  yosys 0.66 has no hierarchical-reference support, so
    //   the DUT's internal cnt[]/head[]/tail[] cannot be reached from a
    //   read-only external harness; instead we track the equivalent OBSERVABLE
    //   boundary invariant:
    //       inflight := (total accepted) - (total emitted)
    //   and prove
    //       P1a : inflight <= N_CH*QDEPTH    (never buffers beyond the
    //             provisioned FIFO capacity -> no overflow / lost response)
    //       P1b : the fabric never emits a response it did not accept
    //             (deq only when inflight >= 1 -> no underflow / no phantom
    //             DATA beat, complementing the tag-level P2).
    // number of channel responses accepted this cycle (popcount of enq)
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

    always @(posedge clk) if (f_past_valid && !rst) begin
        assert(inflight <= CAP[IW-1:0]);                    // P1a
        if (deq_fire) assert(inflight >= {{(IW-1){1'b0}},1'b1}); // P1b
    end

    // ---- P3 : outstanding-budget honored (the flash deep-queue invariant) ----
    //   in-flight reads = (reads ISSUED to the channels) - (responses DRAINED to
    //   the requester).  issue := mem_req_valid[c] && mem_req_ready[c] per chan;
    //   drain := resp_valid && resp_ready.  The fabric caps per-channel
    //   outstanding at QDEPTH (== RESP_QD), so the total can never exceed
    //   N_CH*QDEPTH.  Because outstanding = (in-die) + (in-FIFO) >= in-FIFO, this
    //   bound is what guarantees the QDEPTH-deep response FIFOs can never be
    //   over-filled -> structurally implies P1a.
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

    always @(posedge clk) if (f_past_valid && !rst) begin
        assert(outstanding <= CAP[IW-1:0]);                 // P3
        // accepted responses can never exceed reads in flight on the dies
        assert(inflight <= outstanding);                    // (FIFO <= outstanding)
    end

    // Reachability / non-vacuity probes: each is a DELIBERATELY FALSE claim.
    // BMC returning a counterexample for it proves the corresponding
    // interesting state IS reachable under our assumptions (-> not vacuous).
    // Select one at a time with -D<NAME>.
    always @(posedge clk) if (f_past_valid && !rst) begin
`ifdef PROBE_RESP
        assert(!resp_valid);                 // expect FAIL: a response can fire
`endif
`ifdef PROBE_ENQ
        assert(enq_pop == 0);                // expect FAIL: a channel push happens
`endif
`ifdef PROBE_FULL
        assert(inflight != CAP[IW-1:0]);     // expect FAIL: FIFOs can reach FULL
`endif
`ifdef PROBE_OUTST
        assert(outstanding != CAP[IW-1:0]);  // expect FAIL: outstanding can saturate
`endif
`ifdef PROBE_TAG
        assert(!(resp_valid && resp_tag != {TAG_W{1'b0}})); // FAIL: nonzero tag out
`endif
    end

endmodule
