`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// flash_xbar.v  --  N_CH-channel BANKED FLASH READ fabric with DEEP per-channel
//                   outstanding-request queues (channel-parallel + latency-hiding
//                   bandwidth).                       (docs/IMPROVEMENT_PLAN.md P1)
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   tokens/s is LINEAR in Flash read bandwidth.  A 1 TB on-module Flash is many
//   NAND dies; "10s of GB/s" is only reachable by reading those dies IN PARALLEL.
//   This fabric BANKS a stream of read requests across N_CH independent Flash
//   channels (one die / channel) so up to N_CH reads land on different dies at
//   once -> aggregate BW ~ N_CH x a single die.
//
//   THE NEW KNOB vs ddr5_xbar (QDEPTH).  A DDR5 read returns in ~tens of ns; a
//   NAND read returns in ~10-100 us = THOUSANDS of cycles (FLASH_LAT).  By
//   Little's law the bandwidth a channel sustains is
//        BW_channel = (outstanding reads in flight) / FLASH_LAT
//   so at one outstanding read per channel the channel delivers only
//   1/FLASH_LAT reads/cycle -- catastrophically low.  To keep a high-latency
//   channel BUSY you must have MANY reads in flight on it at once: issue read
//   N+1, N+2, ... while read N is still being serviced by the die.  This fabric
//   therefore gives EACH channel a deep outstanding-request budget QDEPTH: a
//   channel may have up to QDEPTH reads issued-but-not-yet-returned.  Then
//        BW_channel ~ min(1/cycle, QDEPTH/FLASH_LAT)
//   and a deep QDEPTH (QDEPTH ~ FLASH_LAT) hides the entire NAND latency,
//   restoring ~1 read/cycle/channel and ~N_CH reads/cycle aggregate.  No new
//   math: banking + arbitration + per-channel in-flight (outstanding) tracking.
//
//----------------------------------------------------------------------------
// BANKING SCHEME  (addr -> channel)   [identical policy to ddr5_xbar]
//   channel = req_addr[BANK_LSB +: CH_IDX_W].  With BANK_LSB=0 and block-granular
//   addresses, 0,1,2,3,... stripe round-robin over the channels so a streaming
//   access spreads perfectly across all N_CH dies.  N_CH MUST be a power of two.
//
//----------------------------------------------------------------------------
// OUTSTANDING-QUEUE / BACK-PRESSURE MODEL   [the high-latency engineering]
//   Each channel c owns a registered counter outst[c] = (reads ISSUED to channel
//   c) - (responses DRAINED from channel c to the requester).  i.e. the number
//   of reads currently "owned" by channel c that the requester has not yet seen
//   (still inside the die OR sitting in c's response FIFO).
//     * A request to channel c is ACCEPTED only when outst[c] < QDEPTH (room for
//       another in-flight read) AND the channel PHY can take a command this cycle
//       (mem_req_ready[c]).  Otherwise req_ready is held low (head-of-line stall,
//       same simple-router policy as ddr5_xbar).  This bounds in-flight reads per
//       channel to QDEPTH.
//     * Because outst[c] <= QDEPTH and outst[c] = (in-die) + (in-FIFO), the
//       response FIFO never needs to hold more than QDEPTH entries -- so we size
//       each per-channel response FIFO to QDEPTH (RESP_QD=QDEPTH) and it can
//       NEVER overflow.  mem_resp_ready[c] = (FIFO not full) is therefore always
//       high in practice, i.e. the fabric never has to back-pressure a completing
//       NAND read (NAND completions are not re-orderable on a die).
//
// ORDERING
//   WITHIN a channel, responses stay in issue order (per-channel FIFO is FIFO,
//   and a NAND die completes reads in issue order).  ACROSS channels responses
//   may return out of order (independent dies + round-robin drain), so resp_tag
//   is carried end-to-end for the requester to reorder.
//
//----------------------------------------------------------------------------
// INTERFACES
//   REQUESTER (single port):  req_valid/req_ready/req_addr/req_tag ->
//                             resp_valid/resp_ready/resp_data/resp_tag.
//   CHANNEL REQUEST (N_CH):   mem_req_valid/ready/addr/tag  (fabric -> Flash PHY)
//   CHANNEL RESPONSE (N_CH):  mem_resp_valid/ready/data/tag (Flash PHY -> fabric)
//   The real Flash controller/PHY is vendor IP; the per-channel memory and its
//   FLASH_LAT read latency are modeled by the TB stub.  FLASH_LAT is a parameter
//   only -- the fabric tracks reads by TAG/outstanding count, never by counting
//   cycles, so it is latency-agnostic.
//
// Sync active-high reset.  No latch, no comb loop (request routing is feed-forward
// from inputs + REGISTERED outst[]; the response arbiter reads REGISTERED FIFO
// occupancy only).
//============================================================================
module flash_xbar #(
    parameter integer N_CH    = 8,      // Flash channels / dies (power of two)
    parameter integer ADDR_W  = 32,     // block-address width
    parameter integer DATA_W  = 256,    // read-data width (one page beat)
    parameter integer TAG_W   = 8,      // requester tag width (in-flight id)
    parameter integer QDEPTH  = 8,      // MAX outstanding reads PER CHANNEL (knob)
    /* verilator lint_off UNUSEDPARAM */
    parameter integer FLASH_LAT = 1000, // per-read NAND latency (TB stub only)
    /* verilator lint_on UNUSEDPARAM */
    parameter integer BANK_LSB = 0      // channel-select field LSB in req_addr
)(
    input  wire                     clk,
    input  wire                     rst,        // synchronous, active-high

    // ---- requester request (single port) ----
    input  wire                     req_valid,
    output wire                     req_ready,
    input  wire [ADDR_W-1:0]        req_addr,
    input  wire [TAG_W-1:0]         req_tag,

    // ---- channel request (N_CH ports, to per-channel Flash PHY) ----
    output wire [N_CH-1:0]          mem_req_valid,
    input  wire [N_CH-1:0]          mem_req_ready,
    output wire [N_CH*ADDR_W-1:0]   mem_req_addr,
    output wire [N_CH*TAG_W-1:0]    mem_req_tag,

    // ---- channel response (N_CH ports, from per-channel Flash PHY) ----
    input  wire [N_CH-1:0]          mem_resp_valid,
    output wire [N_CH-1:0]          mem_resp_ready,
    input  wire [N_CH*DATA_W-1:0]   mem_resp_data,
    input  wire [N_CH*TAG_W-1:0]    mem_resp_tag,

    // ---- requester response (single port) ----
    output reg                      resp_valid,
    input  wire                     resp_ready,
    output reg  [DATA_W-1:0]        resp_data,
    output reg  [TAG_W-1:0]         resp_tag
);
    // ---------------- helpers ----------------
    function integer clog2;
        input integer v; integer i;
        begin
            clog2 = 0;
            for (i = v-1; i > 0; i = i >> 1) clog2 = clog2 + 1;
        end
    endfunction

    // The per-channel response FIFO is sized to QDEPTH: because outstanding is
    // capped at QDEPTH per channel and outstanding = in-die + in-FIFO, the FIFO
    // can never be asked to hold more than QDEPTH completed reads -> no overflow.
    localparam integer RESP_QD  = QDEPTH;
    localparam integer CH_IDX_W = (clog2(N_CH)  < 1) ? 1 : clog2(N_CH);
    localparam integer PTR_W    = (clog2(RESP_QD)< 1) ? 1 : clog2(RESP_QD);
    localparam integer CNT_W    = clog2(RESP_QD + 1);   // 0..RESP_QD  (FIFO occ)
    localparam integer OST_W    = clog2(QDEPTH  + 1);   // 0..QDEPTH   (outstanding)
    localparam integer PAY_W    = DATA_W + TAG_W;        // {tag, data}
    localparam [PTR_W-1:0]    QD_LAST = RESP_QD[PTR_W-1:0] - 1'b1;   // FIFO wrap
    localparam [CH_IDX_W-1:0] CH_LAST = N_CH[CH_IDX_W-1:0]  - 1'b1;  // rr wrap

    genvar c;
    integer k;

    // ===================================================================
    // OUTSTANDING-REQUEST COUNTERS  (the high-latency knob)
    //   outst[c] = issued-to-c minus drained-from-c.  Registered; the request
    //   path reads it combinationally to gate acceptance.
    // ===================================================================
    reg [OST_W-1:0] outst [0:N_CH-1];

    // ===================================================================
    // REQUEST PATH  --  feed-forward banking / routing, gated by outstanding room
    // ===================================================================
    wire [CH_IDX_W-1:0] req_ch   = req_addr[BANK_LSB +: CH_IDX_W];
    wire                ch_full  = (outst[req_ch] == QDEPTH[OST_W-1:0]);

    generate
        for (c = 0; c < N_CH; c = c + 1) begin : g_req
            // present a command to the banked channel ONLY when there is room
            // for another outstanding read on it (so the PHY never accepts a
            // command the fabric has not counted -> no lost/double issue).
            assign mem_req_valid[c] =
                req_valid && (req_ch == c[CH_IDX_W-1:0]) && !ch_full;
            assign mem_req_addr[c*ADDR_W +: ADDR_W] = req_addr;
            assign mem_req_tag [c*TAG_W  +: TAG_W ] = req_tag;
        end
    endgenerate

    // accepted iff banked channel has outstanding room AND can take the command
    assign req_ready = !ch_full && mem_req_ready[req_ch];

    // issue fire per channel (request accepted into channel c this cycle)
    wire [N_CH-1:0] iss_fire;
    generate
        for (c = 0; c < N_CH; c = c + 1) begin : g_iss
            assign iss_fire[c] = mem_req_valid[c] && mem_req_ready[c];
        end
    endgenerate

    // ===================================================================
    // RESPONSE PATH  --  per-channel FIFO + round-robin drain arbiter
    //                    (rotate-mask priority encoder, reused from ddr5_xbar)
    // ===================================================================
    reg [PAY_W-1:0] fifo [0:N_CH-1][0:RESP_QD-1];
    reg [PTR_W-1:0] head [0:N_CH-1];
    reg [PTR_W-1:0] tail [0:N_CH-1];
    reg [CNT_W-1:0] cnt  [0:N_CH-1];
    reg [CH_IDX_W-1:0] rr;   // round-robin pointer

    reg                gnt_valid;
    reg [CH_IDX_W-1:0] gnt;
    integer ai; integer idx;
    reg [N_CH-1:0]     rot;        // fifo_ne rotated into rr-relative order
    reg [CH_IDX_W-1:0] sel;        // first set position WITHIN rot (rr-relative)
    reg                found;      // any FIFO non-empty
    reg [CH_IDX_W:0]   gnt_idx;    // sel+rr before the single modulo fold

    wire [N_CH-1:0] fifo_ne;     // FIFO non-empty
    wire [N_CH-1:0] deq_fire;    // arbiter drains this channel this cycle
    generate
        for (c = 0; c < N_CH; c = c + 1) begin : g_occ
            // back-pressure the channel when its FIFO is full (cannot occur given
            // the QDEPTH invariant, but kept for a self-contained correct FIFO).
            assign mem_resp_ready[c] = (cnt[c] != RESP_QD[CNT_W-1:0]);
            assign fifo_ne[c]        = (cnt[c] != {CNT_W{1'b0}});
            assign deq_fire[c]       =
                gnt_valid && (gnt == c[CH_IDX_W-1:0]) && resp_ready;
        end
    endgenerate

    // ---- combinational round-robin grant (reads REGISTERED occupancy) ----
    always @* begin
        // (1) parallel barrel-rotate fifo_ne by rr (rot[0] is channel rr)
        rot = {N_CH{1'b0}};
        for (ai = 0; ai < N_CH; ai = ai + 1) begin
            idx = ai + {{(32-CH_IDX_W){1'b0}}, rr};
            if (idx >= N_CH) idx = idx - N_CH;
            rot[ai] = fifo_ne[idx[CH_IDX_W-1:0]];
        end
        // (2) priority-encode rot: lowest set position wins
        found = 1'b0;
        sel   = {CH_IDX_W{1'b0}};
        for (ai = N_CH-1; ai >= 0; ai = ai - 1) begin
            if (rot[ai[CH_IDX_W-1:0]]) begin
                found = 1'b1;
                sel   = ai[CH_IDX_W-1:0];
            end
        end
        // (3) single modulo fold back to the absolute channel index
        gnt_idx = {1'b0, sel} + {1'b0, rr};
        if (gnt_idx >= N_CH[CH_IDX_W:0]) gnt_idx = gnt_idx - N_CH[CH_IDX_W:0];
        gnt_valid = found;
        gnt       = gnt_idx[CH_IDX_W-1:0];
        // drive requester response port from the granted FIFO head
        resp_valid = gnt_valid;
        resp_data  = fifo[gnt][head[gnt]][DATA_W-1:0];
        resp_tag   = fifo[gnt][head[gnt]][DATA_W +: TAG_W];
    end

    // ---- sequential: FIFO update, outstanding update, rr advance ----
    wire [N_CH-1:0] enq_fire;     // channel pushes a completed read this cycle
    generate
        for (c = 0; c < N_CH; c = c + 1) begin : g_enq
            assign enq_fire[c] = mem_resp_valid[c] && mem_resp_ready[c];
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            rr <= {CH_IDX_W{1'b0}};
            for (k = 0; k < N_CH; k = k + 1) begin
                head [k] <= {PTR_W{1'b0}};
                tail [k] <= {PTR_W{1'b0}};
                cnt  [k] <= {CNT_W{1'b0}};
                outst[k] <= {OST_W{1'b0}};
            end
        end else begin
            for (k = 0; k < N_CH; k = k + 1) begin
                if (enq_fire[k]) begin
                    fifo[k][tail[k]] <=
                        { mem_resp_tag [k*TAG_W  +: TAG_W ],
                          mem_resp_data[k*DATA_W +: DATA_W] };
                    tail[k] <= (tail[k] == QD_LAST)
                                   ? {PTR_W{1'b0}} : tail[k] + 1'b1;
                end
                if (deq_fire[k]) begin
                    head[k] <= (head[k] == QD_LAST)
                                   ? {PTR_W{1'b0}} : head[k] + 1'b1;
                end
                // FIFO occupancy:  +enq  -deq
                cnt[k] <= cnt[k]
                          + {{(CNT_W-1){1'b0}}, enq_fire[k]}
                          - {{(CNT_W-1){1'b0}}, deq_fire[k]};
                // outstanding:     +issue -drain(=deq)
                outst[k] <= outst[k]
                          + {{(OST_W-1){1'b0}}, iss_fire[k]}
                          - {{(OST_W-1){1'b0}}, deq_fire[k]};
            end
            // advance round-robin past the channel we just drained
            if (gnt_valid && resp_ready)
                rr <= (gnt == CH_LAST)
                          ? {CH_IDX_W{1'b0}} : gnt + 1'b1;
        end
    end

endmodule
/* verilator lint_on DECLFILENAME */
