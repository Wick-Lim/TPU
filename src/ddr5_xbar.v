`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// ddr5_xbar.v  --  N_CH-channel BANKED DDR5 READ fabric (channel-parallel BW)
//                                                  (SYSTEM_SINGLE_PACKAGE.md §mem)
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   A single DDR5 channel delivers only ~51 GB/s; the single-package GLM-5.2-FP8
//   system needs ~400-600 GB/s.  That bandwidth is realized by running N_CH=8-12
//   channels IN PARALLEL.  Today the repo's fetch path (expert_cache_pf) is
//   effectively single-channel.  This module is the RTL fabric that makes the
//   channel-parallelism REAL: it BANKS a stream of read requests across N_CH
//   independent channel pipelines so that up to N_CH reads are in flight at once
//   -> aggregate read throughput ~ N_CH x a single channel.  No new math: this
//   is banking + arbitration + in-flight tag tracking only.
//
//----------------------------------------------------------------------------
// BANKING SCHEME  (addr -> channel)
//   The target channel is taken from CH_IDX_W low bits of the *block* address:
//        channel = req_addr[BANK_LSB +: CH_IDX_W]
//   With BANK_LSB=0 and block-granular addresses, consecutive addresses
//   0,1,2,3,... stripe round-robin across channel 0,1,2,3,... so a streaming /
//   sequential access pattern spreads PERFECTLY over all N_CH channels (one new
//   read issued per cycle, each landing on a different channel).  BANK_LSB lets
//   the caller place the channel-select field higher (e.g. interleave at a
//   coarser block size).  N_CH MUST be a power of two so the field is a clean
//   modulo-N_CH index (default 8).
//
//----------------------------------------------------------------------------
// INTERFACES
//   REQUESTER (single port, into the fabric):
//     req_valid / req_ready / req_addr[ADDR_W] / req_tag[TAG_W]
//       Standard valid/ready.  req_tag is requester-defined and is carried
//       end-to-end so the requester can match (and reorder) responses.
//   CHANNEL REQUEST (N_CH ports, fabric -> per-channel DDR5 PHY/stub):
//     mem_req_valid[N_CH] / mem_req_ready[N_CH] /
//     mem_req_addr[N_CH*ADDR_W] / mem_req_tag[N_CH*TAG_W]
//       The fabric ROUTES the requester's beat to exactly the banked channel.
//   CHANNEL RESPONSE (N_CH ports, per-channel DDR5 PHY/stub -> fabric):
//     mem_resp_valid[N_CH] / mem_resp_ready[N_CH] /
//     mem_resp_data[N_CH*DATA_W] / mem_resp_tag[N_CH*TAG_W]
//       Each channel returns its read ROW_LAT cycles later, tagged.  The fabric
//       captures it into a small per-channel response FIFO (back-pressuring the
//       channel via mem_resp_ready when that FIFO is full).
//   REQUESTER RESPONSE (single port, out of the fabric):
//     resp_valid / resp_ready / resp_data[DATA_W] / resp_tag[TAG_W]
//
//   NOTE: the real DDR5 PHY is vendor IP.  Here the per-channel memory and its
//   ROW_LAT read latency are modeled by the TB stub; ROW_LAT is a parameter only
//   so the fabric's in-flight tracking is latency-agnostic (it tracks by TAG,
//   not by counting cycles).
//
//----------------------------------------------------------------------------
// BACK-PRESSURE POLICY  (documented choice)
//   REQUEST side: the requester is banked COMBINATIONALLY to one channel and
//   req_ready = mem_req_ready[target_channel].  i.e. a request is accepted iff
//   ITS banked channel can accept it.  This is simple head-of-line stalling: a
//   request to a momentarily-busy channel stalls the requester even if other
//   channels are idle.  For the intended STRIPED/streaming pattern that never
//   bites (each cycle targets the next channel), and it keeps the fabric a thin,
//   provably-correct router.  A reorder/skid front-end could relax it but is out
//   of scope here.
//   RESPONSE side: each channel has an independent RESP_QD-deep response FIFO.
//   mem_resp_ready[c] is high whenever channel c's FIFO is not full, so a
//   channel may hold a small queue of completed reads.  A round-robin arbiter
//   drains ONE channel FIFO per cycle onto the single requester resp port
//   (gated by resp_ready).
//
// ORDERING
//   Responses MAY return OUT OF ORDER across channels (different channels finish
//   independently and the arbiter is round-robin), so resp_tag is exposed for
//   the requester to reorder.  WITHIN a single channel responses stay in issue
//   order (per-channel FIFO is in-order).
//
//----------------------------------------------------------------------------
// Sync active-high reset.  No latch, no comb loop (req routing is feed-forward
// from inputs; the response arbiter reads REGISTERED FIFO occupancy only).
//============================================================================
module ddr5_xbar #(
    parameter integer N_CH    = 8,      // number of DDR5 channels (power of two)
    parameter integer ADDR_W  = 32,     // block-address width
    parameter integer DATA_W  = 256,    // read-data width (one burst beat)
    parameter integer TAG_W   = 8,      // requester tag width (in-flight id)
    /* verilator lint_off UNUSEDPARAM */
    parameter integer ROW_LAT = 10,     // per-channel read latency (TB stub only)
    /* verilator lint_on UNUSEDPARAM */
    parameter integer RESP_QD = 4,      // per-channel response FIFO depth
    parameter integer BANK_LSB = 0      // channel-select field LSB in req_addr
)(
    input  wire                     clk,
    input  wire                     rst,        // synchronous, active-high

    // ---- requester request (single port) ----
    input  wire                     req_valid,
    output wire                     req_ready,
    input  wire [ADDR_W-1:0]        req_addr,
    input  wire [TAG_W-1:0]         req_tag,

    // ---- channel request (N_CH ports, to per-channel DDR5) ----
    output wire [N_CH-1:0]          mem_req_valid,
    input  wire [N_CH-1:0]          mem_req_ready,
    output wire [N_CH*ADDR_W-1:0]   mem_req_addr,
    output wire [N_CH*TAG_W-1:0]    mem_req_tag,

    // ---- channel response (N_CH ports, from per-channel DDR5) ----
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

    localparam integer CH_IDX_W = (clog2(N_CH) < 1) ? 1 : clog2(N_CH);
    localparam integer PTR_W    = (clog2(RESP_QD) < 1) ? 1 : clog2(RESP_QD);
    localparam integer CNT_W    = clog2(RESP_QD + 1);   // 0..RESP_QD
    localparam integer PAY_W    = DATA_W + TAG_W;        // {tag, data}
    localparam [PTR_W-1:0]    QD_LAST = RESP_QD[PTR_W-1:0] - 1'b1;   // FIFO wrap
    localparam [CH_IDX_W-1:0] CH_LAST = N_CH[CH_IDX_W-1:0]  - 1'b1;  // rr wrap

    genvar c;
    integer k;

    // ===================================================================
    // REQUEST PATH  --  pure feed-forward banking / routing (no state)
    // ===================================================================
    wire [CH_IDX_W-1:0] req_ch = req_addr[BANK_LSB +: CH_IDX_W];

    generate
        for (c = 0; c < N_CH; c = c + 1) begin : g_req
            assign mem_req_valid[c] =
                req_valid && (req_ch == c[CH_IDX_W-1:0]);
            assign mem_req_addr[c*ADDR_W +: ADDR_W] = req_addr;
            assign mem_req_tag [c*TAG_W  +: TAG_W ] = req_tag;
        end
    endgenerate

    // accepted iff the banked channel can take it (head-of-line policy)
    assign req_ready = mem_req_ready[req_ch];

    // ===================================================================
    // RESPONSE PATH  --  per-channel FIFO + round-robin drain arbiter
    // ===================================================================
    reg [PAY_W-1:0] fifo [0:N_CH-1][0:RESP_QD-1];
    reg [PTR_W-1:0] head [0:N_CH-1];
    reg [PTR_W-1:0] tail [0:N_CH-1];
    reg [CNT_W-1:0] cnt  [0:N_CH-1];
    reg [CH_IDX_W-1:0] rr;   // round-robin pointer

    // round-robin grant (declared before use; driven combinationally below)
    reg                gnt_valid;
    reg [CH_IDX_W-1:0] gnt;
    integer ai; integer idx;

    wire [N_CH-1:0] fifo_ne;     // FIFO non-empty
    wire [N_CH-1:0] deq_fire;    // arbiter drains this channel this cycle
    generate
        for (c = 0; c < N_CH; c = c + 1) begin : g_occ
            // back-pressure the channel when its FIFO is full
            assign mem_resp_ready[c] = (cnt[c] != RESP_QD[CNT_W-1:0]);
            assign fifo_ne[c]        = (cnt[c] != {CNT_W{1'b0}});
            assign deq_fire[c]       =
                gnt_valid && (gnt == c[CH_IDX_W-1:0]) && resp_ready;
        end
    endgenerate

    // ---- combinational round-robin grant (reads REGISTERED occupancy) ----
    always @* begin
        gnt_valid = 1'b0;
        gnt       = {CH_IDX_W{1'b0}};
        for (ai = 0; ai < N_CH; ai = ai + 1) begin
            idx = ai + {{(32-CH_IDX_W){1'b0}}, rr};
            if (idx >= N_CH) idx = idx - N_CH;
            if (!gnt_valid && fifo_ne[idx[CH_IDX_W-1:0]]) begin
                gnt_valid = 1'b1;
                gnt       = idx[CH_IDX_W-1:0];
            end
        end
        // drive requester response port from the granted FIFO head
        resp_valid = gnt_valid;
        resp_data  = fifo[gnt][head[gnt]][DATA_W-1:0];
        resp_tag   = fifo[gnt][head[gnt]][DATA_W +: TAG_W];
    end

    // ---- sequential FIFO update + rr advance ----
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
                head[k] <= {PTR_W{1'b0}};
                tail[k] <= {PTR_W{1'b0}};
                cnt [k] <= {CNT_W{1'b0}};
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
                cnt[k] <= cnt[k]
                          + {{(CNT_W-1){1'b0}}, enq_fire[k]}
                          - {{(CNT_W-1){1'b0}}, deq_fire[k]};
            end
            // advance round-robin past the channel we just drained
            if (gnt_valid && resp_ready)
                rr <= (gnt == CH_LAST)
                          ? {CH_IDX_W{1'b0}} : gnt + 1'b1;
        end
    end

endmodule
/* verilator lint_on DECLFILENAME */
