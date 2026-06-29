`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// boot_loader.v  --  power-up resident-set DMA sequencer  (SYSTEM_SINGLE_PACKAGE)
//----------------------------------------------------------------------------
// ROLE
//   The GLM-5.2-FP8 chip keeps the entire 753 GB model in Flash, but the
//   ~28 GB HOT/RESIDENT working set (all-layer attention, shared expert, dense
//   FFN, router, embed/LM-head, norms) must live in fast DDR5 for per-token
//   reuse.  Before ANY inference, this engine COPIES the resident set from
//   Flash into DDR5 at power-up; its registered `done` is the single gate that
//   releases inference.  (Routed experts stay in Flash, demand-streamed later
//   by expert_cache_pf -- NOT this engine's job.)
//
//   This is PURE integer/control RTL: a raw word-for-word block move, no
//   arithmetic on the data, no rounding.  Both the Flash READ port and the
//   DDR5 WRITE port are simple addressable-memory + latency STUBS owned by the
//   TB (real ONFI/NVMe + DDR5 controllers are vendor IP).
//
// RESIDENT-SET DESCRIPTOR  (a list of up to SEG_MAX segments + a count)
//   The resident set is generally NON-contiguous (attention here, router there,
//   embed elsewhere), so it is described as SEG_MAX segments, each a triple
//        { flash_base[ADDR_W], ddr5_base[ADDR_W], len[LEN_W] }  (len in WORDS)
//   presented FLAT-PACKED on three buses (segment k in bits [k*W +: W]) plus a
//   live `seg_count`.  A 1-cycle `start` (power-on) pulse LATCHES the whole
//   descriptor, so the caller may change the inputs immediately after.  Each
//   active segment k is copied word-for-word:
//        for w in 0..len[k]-1:  DDR5[ddr5_base[k]+w] = Flash[flash_base[k]+w]
//   Segments are walked k = 0,1,...,seg_count-1 in order; a len==0 segment is
//   skipped (one cycle).  seg_count==0 (or all-zero lengths) => nothing to do,
//   `done` rises immediately.
//
// FLASH-READ / DDR5-WRITE PORTS and HOW THEY PIPELINE
//   Flash READ  (req -> data, latency FLASH_LAT, owned/modelled by the TB):
//       flash_req  + flash_addr  presented combinationally;
//       flash_ready (input) gates acceptance (req issued only when both high);
//       flash_rvalid + flash_rdata return the word IN ORDER FLASH_LAT later.
//   DDR5 WRITE  (addr+data+we, with back-pressure):
//       ddr_we + ddr_addr + ddr_wdata presented; the write retires only when
//       ddr_ready (input) is also high (DDR5 back-pressure / refresh stall).
//
//   A small skid FIFO (depth BURST) DECOUPLES the two so reads and writes
//   pipeline across FLASH_LAT and DDR5 back-pressure:
//       * READ side issues Flash reads as long as the FIFO has reserved room
//         (outstanding = issued-but-not-yet-written  <  BURST).  Up to BURST
//         reads are in flight, hiding FLASH_LAT.
//       * Returned words (flash_rvalid) push into the FIFO IN ORDER.
//       * WRITE side pops the FIFO head and writes it to DDR5 whenever a word
//         is available and ddr_ready is high; if DDR5 back-pressures, the FIFO
//         fills, `outstanding` hits BURST, and the read side naturally stalls.
//   Because Flash returns are in order and BOTH the read and the write address
//   generators walk the SAME segment geometry (skipping the same len==0 gaps),
//   the w-th FIFO word is exactly the w-th word to write -- no per-entry
//   address tag is needed; the write generator reconstructs the DDR5 address.
//
// PROGRESS / DONE SEMANTICS
//   words_done : registered count of WORDS retired into DDR5 (readable status).
//   busy       : high from `start` until the whole resident set is in DDR5.
//   done       : registered LEVEL (the gate).  Rises ONLY when EVERY word of
//                EVERY active segment has been written to DDR5, and STAYS high
//                until a new `start` (or reset).  Never a pulse -- inference
//                samples it as a steady release.
//
// CONVENTIONS: synchronous ACTIVE-HIGH reset (ALL state); NO latch (every
//   combinational output is fully defaulted); NO combinational loop (all
//   outputs derive from registered state only).
//============================================================================
module boot_loader #(
    parameter integer ADDR_W  = 32,   // Flash / DDR5 word-address width
    parameter integer DATA_W  = 64,   // copy word (block) width, bits
    parameter integer SEG_MAX = 4,    // max resident-set segments
    parameter integer BURST   = 8,    // skid-FIFO depth / max reads in flight
    parameter integer LEN_W   = 16,   // per-segment length field width (words)
    // ---- derived geometry ----
    localparam integer SEGW   = (SEG_MAX < 2) ? 1 : $clog2(SEG_MAX + 1),
    localparam integer DEPTHW = (BURST   < 2) ? 1 : $clog2(BURST),     // FIFO ptr
    localparam integer OCCW   = $clog2(BURST + 1),                     // occupancy
    localparam integer PROG_W = LEN_W + SEGW                           // words_done
) (
    input  wire                       clk,
    input  wire                       rst,         // sync, active-high

    // ---- power-on command + resident-set descriptor ----
    input  wire                       start,       // 1-cycle pulse: latch + begin
    input  wire [SEGW-1:0]            seg_count,   // # active segments (<= SEG_MAX)
    input  wire [SEG_MAX*ADDR_W-1:0]  seg_flash_base, // per-seg Flash base (packed)
    input  wire [SEG_MAX*ADDR_W-1:0]  seg_ddr_base,   // per-seg DDR5  base (packed)
    input  wire [SEG_MAX*LEN_W-1:0]   seg_len,        // per-seg length, words (packed)

    // ---- Flash READ port (req -> data, latency FLASH_LAT; TB-stubbed) ----
    output reg                        flash_req,   // read request strobe (comb)
    output reg  [ADDR_W-1:0]          flash_addr,  // read address       (comb)
    input  wire                       flash_ready, // Flash accepts a request
    input  wire                       flash_rvalid,// a returned word is valid
    input  wire [DATA_W-1:0]          flash_rdata, // returned word (in order)

    // ---- DDR5 WRITE port (addr+data+we, back-pressured; TB-stubbed) ----
    output reg                        ddr_we,      // write strobe       (comb)
    output reg  [ADDR_W-1:0]          ddr_addr,    // write address      (comb)
    output reg  [DATA_W-1:0]          ddr_wdata,   // write data         (comb)
    input  wire                       ddr_ready,   // DDR5 accepts the write

    // ---- status ----
    output reg                        busy,        // engine is copying
    output reg                        done,        // LEVEL: resident set in DDR5
    output reg  [PROG_W-1:0]          words_done   // # words retired (progress)
);
    // -----------------------------------------------------------------------
    // sized occupancy constant
    // -----------------------------------------------------------------------
    localparam integer    IDXW    = (SEG_MAX < 2) ? 1 : $clog2(SEG_MAX); // array index
    localparam [OCCW-1:0] DEPTHC  = OCCW'(BURST);    // FIFO depth, sized
    localparam [SEGW-1:0] SEGMAXC = SEGW'(SEG_MAX);  // SEG_MAX, sized

    // -----------------------------------------------------------------------
    // latched descriptor (so caller may change inputs after `start`)
    // -----------------------------------------------------------------------
    reg [ADDR_W-1:0] fbase_q [0:SEG_MAX-1];
    reg [ADDR_W-1:0] dbase_q [0:SEG_MAX-1];
    reg [LEN_W-1:0]  len_q   [0:SEG_MAX-1];
    reg [SEGW-1:0]   ncount_q;

    // -----------------------------------------------------------------------
    // segment walkers : READ-issue (r*) and WRITE-retire (w*) cursors
    // -----------------------------------------------------------------------
    reg [SEGW-1:0]   rseg;   reg [LEN_W-1:0] roff;   // next Flash read
    reg [SEGW-1:0]   wseg;   reg [LEN_W-1:0] woff;   // next DDR5  write

    // -----------------------------------------------------------------------
    // skid FIFO (depth BURST) + occupancy accounting
    //   occ  = issued - written   (slots reserved; gates the read side)
    //   fcnt = returned - written  (words physically in FIFO; gates write side)
    // -----------------------------------------------------------------------
    reg [DATA_W-1:0] fifo_mem [0:BURST-1];
    reg [DEPTHW-1:0] head, tail;
    reg [OCCW-1:0]   occ;
    reg [OCCW-1:0]   fcnt;

    // engine running?
    wire running = busy;

    // bounded segment indices (never index out of [0:SEG_MAX-1])
    wire [IDXW-1:0]  rsi = (rseg < SEGMAXC) ? rseg[IDXW-1:0] : {IDXW{1'b0}};
    wire [IDXW-1:0]  wsi = (wseg < SEGMAXC) ? wseg[IDXW-1:0] : {IDXW{1'b0}};

    // current-cursor segment fields (registered array reads)
    wire [LEN_W-1:0] rlen  = len_q[rsi];
    wire [LEN_W-1:0] wlen  = len_q[wsi];
    wire [ADDR_W-1:0] rbase = fbase_q[rsi];
    wire [ADDR_W-1:0] wbase = dbase_q[wsi];

    // segment "still has work" flags
    wire read_active  = running & (rseg < ncount_q) & (rlen != {LEN_W{1'b0}});
    wire write_active = running & (wseg < ncount_q) & (wlen != {LEN_W{1'b0}});

    // -----------------------------------------------------------------------
    // combinational port drives (fully defaulted -> no latch; pure functions of
    // registered state -> no comb loop)
    // -----------------------------------------------------------------------
    always @* begin
        // Flash read: issue while this segment has words AND the FIFO has room.
        flash_req  = read_active & (occ < DEPTHC);
        flash_addr = read_active
                   ? (rbase + {{(ADDR_W-LEN_W){1'b0}}, roff})
                   : {ADDR_W{1'b0}};

        // DDR5 write: present FIFO head while this segment has words AND a word
        // is available; it retires only when ddr_ready is also high.
        ddr_we    = write_active & (fcnt != {OCCW{1'b0}});
        ddr_addr  = write_active
                  ? (wbase + {{(ADDR_W-LEN_W){1'b0}}, woff})
                  : {ADDR_W{1'b0}};
        ddr_wdata = ddr_we ? fifo_mem[head] : {DATA_W{1'b0}};
    end

    // per-cycle events
    wire issue_fire = flash_req & flash_ready;          // a read accepted by Flash
    wire write_fire = ddr_we   & ddr_ready;             // a word retired to DDR5
    wire push_fire  = running  & flash_rvalid;          // a returned word arrives

    // pointer increments (wrap at BURST)
    wire [DEPTHW-1:0] tail_nxt = (tail == DEPTHW'(BURST-1)) ? {DEPTHW{1'b0}}
                                                           : tail + {{(DEPTHW-1){1'b0}},1'b1};
    wire [DEPTHW-1:0] head_nxt = (head == DEPTHW'(BURST-1)) ? {DEPTHW{1'b0}}
                                                           : head + {{(DEPTHW-1){1'b0}},1'b1};

    integer i;

    // -----------------------------------------------------------------------
    // single synchronous control process (active-high reset; no latch/comb loop)
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            busy       <= 1'b0;
            done       <= 1'b0;
            words_done <= {PROG_W{1'b0}};
            ncount_q   <= {SEGW{1'b0}};
            rseg <= {SEGW{1'b0}}; roff <= {LEN_W{1'b0}};
            wseg <= {SEGW{1'b0}}; woff <= {LEN_W{1'b0}};
            head <= {DEPTHW{1'b0}}; tail <= {DEPTHW{1'b0}};
            occ  <= {OCCW{1'b0}};  fcnt <= {OCCW{1'b0}};
            for (i = 0; i < SEG_MAX; i = i + 1) begin
                fbase_q[i] <= {ADDR_W{1'b0}};
                dbase_q[i] <= {ADDR_W{1'b0}};
                len_q[i]   <= {LEN_W{1'b0}};
            end
        end else if (start & ~busy) begin
            // ---- power-on: latch the resident-set descriptor & begin ----
            busy       <= 1'b1;
            done       <= 1'b0;
            words_done <= {PROG_W{1'b0}};
            ncount_q   <= seg_count;
            rseg <= {SEGW{1'b0}}; roff <= {LEN_W{1'b0}};
            wseg <= {SEGW{1'b0}}; woff <= {LEN_W{1'b0}};
            head <= {DEPTHW{1'b0}}; tail <= {DEPTHW{1'b0}};
            occ  <= {OCCW{1'b0}};  fcnt <= {OCCW{1'b0}};
            for (i = 0; i < SEG_MAX; i = i + 1) begin
                fbase_q[i] <= seg_flash_base[i*ADDR_W +: ADDR_W];
                dbase_q[i] <= seg_ddr_base [i*ADDR_W +: ADDR_W];
                len_q[i]   <= seg_len      [i*LEN_W  +: LEN_W ];
            end
        end else if (busy) begin
            // ============ FIFO push (returned Flash word, in order) ============
            if (push_fire) begin
                fifo_mem[tail] <= flash_rdata;
                tail           <= tail_nxt;
            end

            // ============ occupancy / fcnt update (issue, write, push) =========
            // occ  += issue_fire - write_fire ;  fcnt += push_fire - write_fire
            occ  <= occ  + (issue_fire ? {{(OCCW-1){1'b0}},1'b1} : {OCCW{1'b0}})
                         - (write_fire ? {{(OCCW-1){1'b0}},1'b1} : {OCCW{1'b0}});
            fcnt <= fcnt + (push_fire  ? {{(OCCW-1){1'b0}},1'b1} : {OCCW{1'b0}})
                         - (write_fire ? {{(OCCW-1){1'b0}},1'b1} : {OCCW{1'b0}});

            // ============ READ cursor advance / len==0 skip ====================
            if (read_active) begin
                if (issue_fire) begin
                    if (roff == rlen - {{(LEN_W-1){1'b0}},1'b1}) begin
                        rseg <= rseg + {{(SEGW-1){1'b0}},1'b1};
                        roff <= {LEN_W{1'b0}};
                    end else begin
                        roff <= roff + {{(LEN_W-1){1'b0}},1'b1};
                    end
                end
            end else if (running & (rseg < ncount_q)) begin
                // current read segment is len==0 : skip it (one cycle)
                rseg <= rseg + {{(SEGW-1){1'b0}},1'b1};
                roff <= {LEN_W{1'b0}};
            end

            // ============ WRITE cursor advance / len==0 skip ===================
            if (write_active) begin
                if (write_fire) begin
                    head       <= head_nxt;
                    words_done <= words_done + {{(PROG_W-1){1'b0}},1'b1};
                    if (woff == wlen - {{(LEN_W-1){1'b0}},1'b1}) begin
                        wseg <= wseg + {{(SEGW-1){1'b0}},1'b1};
                        woff <= {LEN_W{1'b0}};
                    end else begin
                        woff <= woff + {{(LEN_W-1){1'b0}},1'b1};
                    end
                end
            end else if (running & (wseg < ncount_q)) begin
                // current write segment is len==0 : skip it (one cycle)
                wseg <= wseg + {{(SEGW-1){1'b0}},1'b1};
                woff <= {LEN_W{1'b0}};
            end

            // ============ completion gate ======================================
            // Every active segment fully written -> raise the (level) done gate.
            if (wseg == ncount_q) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end

`ifdef FORMAL
    // (placeholder for future formal hooks)
`endif
endmodule
/* verilator lint_on DECLFILENAME */
