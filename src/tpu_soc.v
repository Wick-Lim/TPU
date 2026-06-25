`include "tpu_defs.vh"
`timescale 1ns/1ps
//============================================================================
// tpu_soc.v  --  two-clock autonomous SoC top for the TPU core (module: tpu_soc)
//----------------------------------------------------------------------------
// PURPOSE
//   A self-driving system-on-chip wrapper that makes the TPU core fetch and run
//   a program out of EXTERNAL memory, all by itself, across an asynchronous
//   clock-domain boundary.  A host configures a tiny descriptor (program base
//   address + word count) over an AXI4-Lite SLAVE, then writes a START bit.  In
//   response the SoC:
//       1. commands its AXI4-Lite MASTER DMA (the reused axi_master_dma) to READ
//          DMA_LEN 32-bit instruction words from external memory at DMA_SRC;
//       2. streams each fetched word, in order, through an asynchronous instr
//          FIFO (cdc_async_fifo) from the bus clock domain (ACLK) into the core
//          clock domain (CCLK);
//       3. a CCLK SEQUENCER pops one instruction at a time and drives the core's
//          instruction_in, advancing only when the core front end is free (it
//          respects dbg_pipe_stall exactly like the core's own testbench step()),
//          never dropping an instruction;
//       4. after the last fetched instruction is issued, the sequencer drains a
//          fixed run of NOPs (>= pipeline depth) so the final write-back lands,
//          captures result_out into a CCLK register and pulses result_valid;
//       5. the captured result crosses back CCLK->ACLK through a second async
//          FIFO; the host reads it from the RESULT register, with STATUS bits
//          reflecting dma_done / result_valid / err.
//
// CLOCKS / RESETS (TWO ASYNCHRONOUS DOMAINS)
//   * ACLK / ARESETn  : bus + AXI domain (host control, slave regs, master DMA,
//                       FIFO write side of instr FIFO / read side of result FIFO).
//                       ARESETn is AXI active-LOW; internal aclk reset arst = ~ARESETn.
//   * CCLK / CRESETn  : core domain (the TPU core, the sequencer, FIFO read side
//                       of instr FIFO / write side of result FIFO).
//                       CRESETn is active-LOW; internal cclk reset crst = ~CRESETn.
//   The two clocks are unrelated.  EVERYTHING that crosses between them does so
//   ONLY through (a) the two async FIFOs' gray-coded pointers, or (b) explicit
//   2-FF synchronizers carrying SINGLE-BIT level/toggle signals.  No raw
//   multi-bit binary value is ever sampled directly in the opposite domain.
//
// CDC STRUCTURE
//   START   ACLK -> CCLK : a TOGGLE bit (flips once per START kick) crossed via a
//                          2-FF synchronizer; the core domain edge-detects it to
//                          arm one program run.  (Single bit => metastability-safe.)
//   INSTR   ACLK -> CCLK : cdc_async_fifo #(DATA_W=32).  The DMA sink
//                          {wr_en,wr_idx,wr_data} pushes each fetched word on the
//                          ACLK write side; the sequencer pops on the CCLK side.
//   DMALEN  ACLK -> CCLK : the program length is delivered IMPLICITLY -- the core
//                          domain counts the words it pops; the host's START
//                          carries no length across the boundary directly.  The
//                          number of instructions to issue is bounded by what the
//                          DMA actually pushes into the FIFO, and the core run
//                          completes after a fixed NOP drain following the last
//                          popped word (tracked entirely within the CCLK domain).
//   RESULT  CCLK -> ACLK : cdc_async_fifo #(DATA_W=32).  result_valid pushes the
//                          captured result_out on the CCLK write side; the slave
//                          pops it on the ACLK side for the RESULT register.
//   DONE    CCLK -> ACLK : a TOGGLE bit (flips once per completed run) crossed via
//                          a 2-FF synchronizer; the ACLK side edge-detects it to
//                          set the sticky core-done / result-ready status.
//   DMAERR  (ACLK only)  : the master DMA's sticky err lives entirely in ACLK; no
//                          crossing needed.
//
// AXI4-LITE SLAVE REGISTER MAP  (byte offset; 32-bit words; ADDR_W byte addr)
//   0x00  DMA_SRC   RW  : external BYTE base address of the program in ext. memory.
//   0x04  DMA_LEN   RW  : number of 32-bit instruction words to fetch (>= 1).
//                         Clamped to [1 .. INSTR_FIFO_DEPTH] on use so a fetch can
//                         never overrun the instruction FIFO.
//   0x08  DMA_CTRL  W   : bit0 START -- write-1 kicks the DMA fetch + core run.
//                     R  : bit0 DMA_BUSY (master DMA running),
//                          bit1 DONE     (last run's core result is ready).
//   0x0C  RESULT    RO  : the core's result_out after the fetched program ran,
//                         popped from the result FIFO (crossed CCLK->ACLK).
//   0x10  STATUS    RO  : bit0 dma_done   (sticky: last DMA fetch finished),
//                         bit1 result_valid (a result word is available to read),
//                         bit2 err        (master DMA saw a non-OKAY response).
//   Any other offset reads 0 and write-responds OKAY (benign); unmapped accesses
//   are not error-flagged so a host probe cannot wedge the bus.
//
// SYNTHESIS
//   Synchronous resets on every flop (each reg assigned on every path of its
//   clocked block -> no inferred latch); the only cross-domain paths are the two
//   async FIFOs and the 2-FF single-bit synchronizers; no combinational loops.
//   The TPU core and the two reused modules (axi_master_dma, cdc_async_fifo) are
//   instantiated UNCHANGED.
//============================================================================
/* verilator lint_off DECLFILENAME */
module tpu_soc #(
    // External AXI byte-address width (master + slave decode share the convention).
    parameter integer ADDR_W      = 32,
    // Master DMA word-count width (also the FIFO index width on the sink).
    parameter integer DMA_LENW    = 8,
    // Slave register file byte-address width (5 regs => 3 bits is plenty; keep 8).
    parameter integer SADDR_W     = 8,
    // Instruction CDC FIFO address width => depth = 2**INSTR_AW words.
    parameter integer INSTR_AW    = 5,
    // Result CDC FIFO address width.
    parameter integer RESULT_AW   = 2,
    // NOP drain cycles after the last instruction issues (>= core pipeline depth).
    parameter integer DRAIN_NOPS  = 8,
    // Core instruction_in -> result_out latency, in CCLK cycles, for a
    // non-stalling scalar instruction.  This is the depth of the "real-instruction
    // valid" shadow shift register that times the RESULT capture: when a REAL
    // instruction's valid bit reaches the shadow's last stage, that instruction's
    // write-back is now on core_result and is captured.  See the capture logic
    // below; verified by simulation against the two scalar test programs.
    parameter integer RESULT_LAT  = 5
) (
    // ====================================================================
    // BUS / AXI domain.
    // ====================================================================
    input  wire                 ACLK,
    input  wire                 ARESETn,     // active-LOW

    // ---- AXI4-Lite SLAVE (host control) ----
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [SADDR_W-1:0]   S_AWADDR,
    input  wire [2:0]           S_AWPROT,    // accepted, no protection policy
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire                 S_AWVALID,
    output wire                 S_AWREADY,
    input  wire [31:0]          S_WDATA,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [3:0]           S_WSTRB,     // accepted, full-word writes only
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire                 S_WVALID,
    output wire                 S_WREADY,
    output wire [1:0]           S_BRESP,
    output wire                 S_BVALID,
    input  wire                 S_BREADY,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [SADDR_W-1:0]   S_ARADDR,
    input  wire [2:0]           S_ARPROT,    // accepted, no protection policy
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire                 S_ARVALID,
    output wire                 S_ARREADY,
    output wire [31:0]          S_RDATA,
    output wire [1:0]           S_RRESP,
    output wire                 S_RVALID,
    input  wire                 S_RREADY,

    // ---- AXI4-Lite MASTER (to external memory; driven by axi_master_dma) ----
    output wire [ADDR_W-1:0]    M_AWADDR,
    output wire [2:0]           M_AWPROT,
    output wire                 M_AWVALID,
    input  wire                 M_AWREADY,
    output wire [31:0]          M_WDATA,
    output wire [3:0]           M_WSTRB,
    output wire                 M_WVALID,
    input  wire                 M_WREADY,
    input  wire [1:0]           M_BRESP,
    input  wire                 M_BVALID,
    output wire                 M_BREADY,
    output wire [ADDR_W-1:0]    M_ARADDR,
    output wire [2:0]           M_ARPROT,
    output wire                 M_ARVALID,
    input  wire                 M_ARREADY,
    input  wire [31:0]          M_RDATA,
    input  wire [1:0]           M_RRESP,
    input  wire                 M_RVALID,
    output wire                 M_RREADY,

    // ====================================================================
    // CORE domain.
    // ====================================================================
    input  wire                 CCLK,
    input  wire                 CRESETn      // active-LOW
);

    // ======================================================================
    // Local parameters / derived constants.
    // ======================================================================
    localparam [1:0] AXI_OKAY = 2'b00;       // only response this slave returns
    localparam integer INSTR_DEPTH = (1 << INSTR_AW);

    // Active-HIGH synchronous resets derived from the active-LOW domain resets.
    wire arst = ~ARESETn;                    // ACLK-domain reset
    wire crst = ~CRESETn;                    // CCLK-domain reset

    // Slave register WORD offsets (byte addr = word << 2).
    localparam [2:0] REG_DMA_SRC  = 3'd0;    // 0x00
    localparam [2:0] REG_DMA_LEN  = 3'd1;    // 0x04
    localparam [2:0] REG_DMA_CTRL = 3'd2;    // 0x08
    localparam [2:0] REG_RESULT   = 3'd3;    // 0x0C
    localparam [2:0] REG_STATUS   = 3'd4;    // 0x10

    // ======================================================================
    // ============================  ACLK DOMAIN  ===========================
    // ======================================================================

    // ----------------------------------------------------------------------
    // Descriptor registers (host-programmed).
    // ----------------------------------------------------------------------
    reg [ADDR_W-1:0]   dma_src;              // program base byte address
    reg [DMA_LENW-1:0] dma_len;              // program length in words (>= 1)

    // ----------------------------------------------------------------------
    // AXI4-Lite SLAVE handshake state (registered, single outstanding) -- the
    // same registered-READY decomposition used by tpu_axi.v in this repo.
    // ----------------------------------------------------------------------
    reg s_awready, s_wready, s_bvalid;
    reg s_arready, s_rvalid;
    reg [31:0] s_rdata;

    reg [2:0] s_awaddr_word;                 // latched write word-offset
    reg       s_aw_seen, s_w_seen;           // per-channel handshake captured
    reg [31:0] s_wdata_lat;                  // latched WDATA if W arrives early

    assign S_AWREADY = s_awready;
    assign S_WREADY  = s_wready;
    assign S_BVALID  = s_bvalid;
    assign S_BRESP   = AXI_OKAY;
    assign S_ARREADY = s_arready;
    assign S_RVALID  = s_rvalid;
    assign S_RRESP   = AXI_OKAY;
    assign S_RDATA   = s_rdata;

    // Selected write word-offset: latched offset if AW came first, else live.
    wire [2:0] s_wr_word = s_aw_seen ? s_awaddr_word : S_AWADDR[4:2];
    wire [31:0] s_wr_data = s_w_seen ? s_wdata_lat : S_WDATA;
    // A write COMMITS the cycle both AW and W have handshaked (this cycle or via
    // an earlier-captured *_seen) and no B response is already pending.
    wire s_aw_now = (S_AWVALID && s_awready);
    wire s_w_now  = (S_WVALID  && s_wready);
    wire s_write_commit = !s_bvalid &&
                          (s_aw_now || s_aw_seen) &&
                          (s_w_now  || s_w_seen);

    // ----------------------------------------------------------------------
    // Master-DMA command + status (ACLK).
    // ----------------------------------------------------------------------
    wire dma_busy, dma_done_pulse, dma_err;
    // DMA sink stream (READ dir): each fetched instruction word, in order.
    wire                  dma_sink_we;
    wire [31:0]           dma_sink_data;
    // READ-mode unused master outputs: the sink index (FIFO order is implicit) and
    // the WRITE-direction source request strobe/index (no WRITE ever issued).
    /* verilator lint_off UNUSEDSIGNAL */
    wire [DMA_LENW-1:0]   dma_sink_idx;       // order is implicit in the FIFO
    wire                  dma_src_req;        // WRITE dir only -> never pulses
    wire [DMA_LENW-1:0]   dma_src_idx;        // WRITE dir only -> never advances
    /* verilator lint_on UNUSEDSIGNAL */

    // DMA length actually commanded: clamp host DMA_LEN to [1 .. INSTR_DEPTH] so a
    // fetch can never overrun the instruction FIFO, and never command len 0.
    wire [DMA_LENW-1:0] dma_len_clamped =
        (dma_len == {DMA_LENW{1'b0}})                ? {{(DMA_LENW-1){1'b0}}, 1'b1} :
        (dma_len > INSTR_DEPTH[DMA_LENW-1:0])        ? INSTR_DEPTH[DMA_LENW-1:0]    :
                                                       dma_len;

    // START kick: a write-1 to DMA_CTRL bit0.  Pulsed for one ACLK cycle.
    wire start_kick = s_write_commit && (s_wr_word == REG_DMA_CTRL) && s_wr_data[0];

    // The master DMA accepts `start` only when idle; guard the kick so a START
    // pressed while a fetch is in flight is ignored (cannot corrupt a run).
    wire dma_cmd_start = start_kick && !dma_busy;

    // Sticky DMA-done flag for STATUS.bit0 (set on DMA done, cleared on next kick).
    reg dma_done_sticky;
    always @(posedge ACLK) begin
        if (arst)                 dma_done_sticky <= 1'b0;
        else if (dma_cmd_start)   dma_done_sticky <= 1'b0;
        else if (dma_done_pulse)  dma_done_sticky <= 1'b1;
    end

    // ----------------------------------------------------------------------
    // START toggle: ACLK -> CCLK (single-bit, 2-FF synced in the core domain).
    //   Flip once per kick that actually launches a DMA run; the core domain
    //   edge-detects the toggle to arm exactly one program run.
    // ----------------------------------------------------------------------
    reg start_tgl_a;
    always @(posedge ACLK) begin
        if (arst)               start_tgl_a <= 1'b0;
        else if (dma_cmd_start) start_tgl_a <= ~start_tgl_a;
    end

    // ----------------------------------------------------------------------
    // DONE toggle: CCLK -> ACLK (single-bit, 2-FF synced here in the bus domain).
    //   `done_tgl_c` (core domain) flips once per completed run; sync + edge
    //   detect here sets the sticky core-done flag for STATUS / DMA_CTRL.DONE.
    // ----------------------------------------------------------------------
    wire done_tgl_c;                          // from CCLK domain (declared below)
    reg  done_tgl_a1, done_tgl_a2, done_tgl_a3;
    wire core_done_edge = (done_tgl_a3 ^ done_tgl_a2);  // toggle observed
    always @(posedge ACLK) begin
        if (arst) begin
            done_tgl_a1 <= 1'b0; done_tgl_a2 <= 1'b0; done_tgl_a3 <= 1'b0;
        end else begin
            done_tgl_a1 <= done_tgl_c;
            done_tgl_a2 <= done_tgl_a1;
            done_tgl_a3 <= done_tgl_a2;
        end
    end

    // Sticky core-done (a completed core run has produced a result).  Cleared by
    // a new START kick (a fresh run is beginning).
    reg core_done_sticky;
    always @(posedge ACLK) begin
        if (arst)               core_done_sticky <= 1'b0;
        else if (dma_cmd_start) core_done_sticky <= 1'b0;
        else if (core_done_edge) core_done_sticky <= 1'b1;
    end

    // ----------------------------------------------------------------------
    // RESULT FIFO read side (ACLK) with a HEAD PREFETCH (first-word-fall-through).
    //   The cdc_async_fifo presents its read-data REGISTERED: a word only appears
    //   on res_rd_data the cycle AFTER a pop (rd_en & ~empty) is accepted, and the
    //   reset value is 0.  The slave latches RDATA at the AR handshake (same cycle
    //   as the pop), so a naive "pop on read" hands the host the STALE pre-pop
    //   res_rd_data (== 0) -- the result word would arrive one cycle too late.
    //
    //   To make RESULT reads return the ACTUAL head word, we PREFETCH: whenever the
    //   FIFO is non-empty and our head holding register is invalid, we issue a pop
    //   to drain one word out of the FIFO; the cycle after, res_rd_data carries that
    //   word and we latch it into result_head (result_head_valid=1).  The host
    //   RESULT read returns result_head directly and CONSUMES it (invalidating the
    //   holding reg so the next word is prefetched).  The FIFO is thus always read
    //   one word ahead, and the registered-read latency is hidden from the host.
    // ----------------------------------------------------------------------
    wire        res_rd_empty;
    wire [31:0] res_rd_data;

    reg [31:0] result_head;        // prefetched FIFO head word
    reg        result_head_valid;  // result_head holds a not-yet-consumed word
    reg        res_prefetch_busy;  // a pop is in flight; res_rd_data lands next cyc

    wire s_read_commit = S_ARVALID && s_arready;   // AR handshake = one read
    // The host consumes the prefetched head on a committed RESULT read.
    wire result_head_consume =
        s_read_commit && (S_ARADDR[4:2] == REG_RESULT) && result_head_valid;
    // Issue a FIFO pop whenever the FIFO has a word, no pop is already in flight,
    // and the holding register is (or is about to become) free to accept it.
    wire res_pop = !res_rd_empty && !res_prefetch_busy &&
                   (!result_head_valid || result_head_consume);

    always @(posedge ACLK) begin
        if (arst) begin
            result_head       <= 32'b0;
            result_head_valid <= 1'b0;
            res_prefetch_busy <= 1'b0;
        end else begin
            // A pop accepted this cycle => its word lands on res_rd_data next cycle.
            res_prefetch_busy <= res_pop;
            // The word popped last cycle is now on res_rd_data: latch it as the head.
            if (res_prefetch_busy) begin
                result_head       <= res_rd_data;
                result_head_valid <= 1'b1;
            end else if (result_head_consume) begin
                // Consumed and nothing newly prefetched this cycle: head goes empty.
                result_head_valid <= 1'b0;
            end
        end
    end

    // ----------------------------------------------------------------------
    // SLAVE write datapath: descriptor regs + handshake FSM.
    // ----------------------------------------------------------------------
    always @(posedge ACLK) begin
        if (arst) begin
            dma_src       <= {ADDR_W{1'b0}};
            dma_len       <= {DMA_LENW{1'b0}};
            s_awready     <= 1'b0;
            s_wready      <= 1'b0;
            s_bvalid      <= 1'b0;
            s_awaddr_word <= 3'd0;
            s_aw_seen     <= 1'b0;
            s_w_seen      <= 1'b0;
            s_wdata_lat   <= 32'b0;
        end else begin
            // ---- AW channel: assert READY for one cycle when a write can start.
            if (!s_awready && S_AWVALID && !s_aw_seen && !s_bvalid)
                s_awready <= 1'b1;
            else
                s_awready <= 1'b0;

            // ---- W channel: assert READY for one cycle when a write can start.
            if (!s_wready && S_WVALID && !s_w_seen && !s_bvalid)
                s_wready <= 1'b1;
            else
                s_wready <= 1'b0;

            // ---- Capture per-channel handshakes that arrive before the other.
            if (s_aw_now) begin
                s_awaddr_word <= S_AWADDR[4:2];
                s_aw_seen     <= 1'b1;
            end
            if (s_w_now) begin
                s_wdata_lat <= S_WDATA;
                s_w_seen    <= 1'b1;
            end

            // ---- Commit: apply the register write and raise BVALID.
            if (s_write_commit) begin
                case (s_wr_word)
                    REG_DMA_SRC: dma_src <= s_wr_data[ADDR_W-1:0];
                    REG_DMA_LEN: dma_len <= s_wr_data[DMA_LENW-1:0];
                    default:     ; // DMA_CTRL handled via start_kick; others benign
                endcase
                s_bvalid  <= 1'b1;
                s_aw_seen <= 1'b0;
                s_w_seen  <= 1'b0;
            end

            // ---- Retire the write response.
            if (s_bvalid && S_BREADY)
                s_bvalid <= 1'b0;
        end
    end

    // ----------------------------------------------------------------------
    // SLAVE read datapath: AR handshake + RDATA mux.
    // ----------------------------------------------------------------------
    // STATUS word assembled combinationally from the ACLK status flops.
    wire [31:0] status_word = { 29'b0, dma_err, core_done_sticky, dma_done_sticky };
    wire [31:0] ctrl_rdata  = { 30'b0, core_done_sticky, dma_busy };

    always @(posedge ACLK) begin
        if (arst) begin
            s_arready <= 1'b0;
            s_rvalid  <= 1'b0;
            s_rdata   <= 32'b0;
        end else begin
            // AR: accept one read when no read response is in flight.
            if (!s_arready && S_ARVALID && !s_rvalid) begin
                s_arready <= 1'b1;
                // Latch the addressed register's value as RDATA (registered).
                case (S_ARADDR[4:2])
                    REG_DMA_SRC:  s_rdata <= { {(32-ADDR_W){1'b0}},   dma_src };
                    REG_DMA_LEN:  s_rdata <= { {(32-DMA_LENW){1'b0}}, dma_len };
                    REG_DMA_CTRL: s_rdata <= ctrl_rdata;
                    REG_RESULT:   s_rdata <= result_head;   // prefetched FIFO head
                    REG_STATUS:   s_rdata <= status_word;
                    default:      s_rdata <= 32'b0;
                endcase
                s_rvalid <= 1'b1;
            end else begin
                s_arready <= 1'b0;
                if (s_rvalid && S_RREADY)
                    s_rvalid <= 1'b0;
            end
        end
    end

    // ======================================================================
    // MASTER DMA instance (ACLK).  READ dir: ext memory -> instr FIFO sink.
    //   dir tied 0 (READ); WRITE-side source is unused (rd_data tied 0).
    // ======================================================================
    axi_master_dma #(
        .ADDR_W (ADDR_W),
        .LENW   (DMA_LENW)
    ) u_dma (
        .ACLK     (ACLK),
        .ARESETn  (ARESETn),
        // command
        .start    (dma_cmd_start),
        .ext_addr (dma_src),
        .len      (dma_len_clamped),
        .dir      (1'b0),                     // READ ext -> sink
        .busy     (dma_busy),
        .done     (dma_done_pulse),
        .err      (dma_err),
        // sink stream (READ): each fetched word
        .wr_en    (dma_sink_we),
        .wr_idx   (dma_sink_idx),
        .wr_data  (dma_sink_data),
        // source stream (WRITE) unused in READ mode
        .rd_req   (dma_src_req),
        .rd_idx   (dma_src_idx),
        .rd_data  (32'b0),
        // AXI master channels -> SoC ports
        .AWADDR   (M_AWADDR),
        .AWPROT   (M_AWPROT),
        .AWVALID  (M_AWVALID),
        .AWREADY  (M_AWREADY),
        .WDATA    (M_WDATA),
        .WSTRB    (M_WSTRB),
        .WVALID   (M_WVALID),
        .WREADY   (M_WREADY),
        .BRESP    (M_BRESP),
        .BVALID   (M_BVALID),
        .BREADY   (M_BREADY),
        .ARADDR   (M_ARADDR),
        .ARPROT   (M_ARPROT),
        .ARVALID  (M_ARVALID),
        .ARREADY  (M_ARREADY),
        .RDATA    (M_RDATA),
        .RRESP    (M_RRESP),
        .RVALID   (M_RVALID),
        .RREADY   (M_RREADY)
    );

    // ======================================================================
    // INSTRUCTION CDC FIFO (DATA_W=32).  Written ACLK from the DMA sink; read
    // CCLK by the sequencer.  The DMA only emits wr_en when ~full is irrelevant
    // here because DMA_LEN is clamped to INSTR_DEPTH, so the FIFO never overflows
    // within a single run (the sequencer drains it as the core consumes).
    // ======================================================================
    wire        instr_fifo_full;             // ACLK side (observed, not gating)
    wire        instr_fifo_empty;            // CCLK side
    wire        instr_rd_en;                 // CCLK side pop strobe
    wire [31:0] instr_rd_data;               // CCLK side popped word

    /* verilator lint_off UNUSEDSIGNAL */
    wire _u_instr_full = instr_fifo_full;    // visibility only
    /* verilator lint_on UNUSEDSIGNAL */

    cdc_async_fifo #(
        .DATA_W (32),
        .ADDR_W (INSTR_AW)
    ) u_instr_fifo (
        // write (ACLK)
        .wclk    (ACLK),
        .wrst_n  (ARESETn),
        .wr_en   (dma_sink_we),
        .wr_data (dma_sink_data),
        .full    (instr_fifo_full),
        // read (CCLK)
        .rclk    (CCLK),
        .rrst_n  (CRESETn),
        .rd_en   (instr_rd_en),
        .rd_data (instr_rd_data),
        .empty   (instr_fifo_empty)
    );

    // ======================================================================
    // ============================  CCLK DOMAIN  ===========================
    // ======================================================================

    // ----------------------------------------------------------------------
    // START toggle 2-FF synchronizer (ACLK->CCLK) + edge detect = arm one run.
    // ----------------------------------------------------------------------
    reg start_tgl_c1, start_tgl_c2, start_tgl_c3;
    wire start_arm = (start_tgl_c3 ^ start_tgl_c2);   // toggle observed in CCLK
    always @(posedge CCLK) begin
        if (crst) begin
            start_tgl_c1 <= 1'b0; start_tgl_c2 <= 1'b0; start_tgl_c3 <= 1'b0;
        end else begin
            start_tgl_c1 <= start_tgl_a;
            start_tgl_c2 <= start_tgl_c1;
            start_tgl_c3 <= start_tgl_c2;
        end
    end

    // ----------------------------------------------------------------------
    // CORE instance (CCLK).  The sequencer owns instruction_in.
    // ----------------------------------------------------------------------
    reg  [31:0] core_instr;                   // driven by the sequencer
    wire [31:0] core_result;
    /* verilator lint_off UNUSEDSIGNAL */
    wire        core_illegal;                 // surfaced only as run side-effect
    wire [`ST_W-1:0] core_status;             // debug view; not crossed
    /* verilator lint_on UNUSEDSIGNAL */
    wire        core_stall;                   // dbg_pipe_stall: front-end hold

    TPU u_core (
        .clk            (CCLK),
        .rst            (crst),               // sync active-HIGH internally
        .instruction_in (core_instr),
        .result_out     (core_result),
        .illegal_opcode (core_illegal),
        .dbg_status     (core_status),
        .dbg_pipe_stall (core_stall)
    );

    // ----------------------------------------------------------------------
    // CORE-DOMAIN SEQUENCER.
    //   States:
    //     C_IDLE  : present NOP; wait for start_arm (a fresh program run).
    //     C_RUN   : pop instructions from the instr FIFO and issue them one at a
    //               time, advancing ONLY when the core front end is free
    //               (!core_stall), exactly like the core's testbench step().
    //               When the FIFO runs dry, present NOP while waiting for more
    //               words (the DMA may still be filling it).  The run ends when
    //               all DMA_LEN words have been issued -- counted locally by
    //               watching the DMA's terminal condition via the FIFO + a word
    //               budget delivered through the FIFO itself (see below).
    //     C_DRAIN : issue DRAIN_NOPS NOP cycles (>= pipeline depth) so the final
    //               write-back lands, then capture result_out and pulse valid.
    //
    //   WORD BUDGET.  The number of instructions to issue equals the number of
    //   words the DMA pushes.  Rather than cross the multi-bit DMA_LEN, the
    //   sequencer keys off the FIFO: it issues every word the FIFO delivers, and
    //   decides the program is COMPLETE when (a) at least one word has been
    //   issued AND (b) the FIFO has been empty for a guard window long enough to
    //   guarantee the DMA has finished pushing (the DMA pushes faster than the
    //   core consumes only transiently; once the last word is popped and the
    //   FIFO stays empty past the guard, the program body is done).  The guard
    //   eliminates the race where the FIFO momentarily empties mid-fetch.
    // ----------------------------------------------------------------------
    localparam [1:0] C_IDLE=2'd0, C_RUN=2'd1, C_DRAIN=2'd2;
    reg [1:0] cstate;

    // FIFO "stayed empty" guard counter: how many consecutive CCLK cycles the
    // FIFO has been empty while we had nothing pending to issue.  Sized to safely
    // cover the async-FIFO empty-deassert latency plus DMA inter-word gaps.
    localparam integer EMPTY_GUARD = 64;
    localparam integer GUARD_W = 7;           // ceil(log2(EMPTY_GUARD))+1
    reg [GUARD_W-1:0] empty_cnt;

    reg [$clog2(DRAIN_NOPS+1)-1:0] drain_cnt; // counts NOP drain cycles
    reg issued_any;                           // at least one real instr issued

    // The FIFO read-data is REGISTERED: a word popped on cycle N appears on
    // instr_rd_data on cycle N+1.  We therefore issue with a one-cycle pipeline:
    //   - pop_pending: a pop was accepted last cycle, so instr_rd_data is the word
    //     to issue THIS cycle (if the core front end is free).
    // To respect the core hold, we DO NOT pop a new word while an already-popped
    // word is still waiting to be accepted by the core (core_stall high).
    reg pop_pending;                          // a popped word awaits issue
    reg [31:0] held_instr;                    // the popped word, held across stalls
    reg        held_valid;                    // held_instr carries a real instr

    // We may pop when: in RUN, FIFO non-empty, and we are not currently holding an
    // unissued word (held_valid low) and not mid-pop (pop_pending low).
    wire can_pop = (cstate == C_RUN) && !instr_fifo_empty &&
                   !pop_pending && !held_valid;
    assign instr_rd_en = can_pop;

    // The instruction we PRESENT to the core this cycle.  Priority: a held
    // (already-popped, not-yet-accepted) word, else the freshly arrived popped
    // word (pop_pending), else OP_NOP (all-zero word).  "Real vs NOP" is tracked
    // structurally by held_valid / pop_pending, so no separate flag is needed.
    reg [31:0] present_instr;
    always @(*) begin
        if (held_valid)
            present_instr = held_instr;       // re-present across a core stall
        else if (pop_pending)
            present_instr = instr_rd_data;    // freshly popped word
        else
            present_instr = 32'b0;            // OP_NOP
    end

    // The core accepts the presented instruction this cycle iff its front end is
    // free (!core_stall) -- mirrors the core testbench step(): hold until accepted.
    wire core_accepts = !core_stall;

    // ----------------------------------------------------------------------
    // REAL-INSTRUCTION VALID SHADOW (RESULT capture timing).
    //   The sequencer registers present_instr into core_instr, so the word the
    //   core actually sees on its instruction_in NEXT cycle is real iff a real
    //   word is being PRESENTED and the core ACCEPTS it this cycle (a held/popped
    //   word that the core front end is free to take).  `present_real_now` is that
    //   condition for the word about to load into core_instr.
    //
    //   We shift that "real" bit through a RESULT_LAT-deep register that mirrors the
    //   core's instruction_in -> result_out latency.  When the bit reaching the
    //   LAST stage is set, a REAL instruction's write-back is now on core_result,
    //   so we capture it into result_captured.  Because trailing NOPs shift in
    //   valid=0, the LAST real instruction's capture is never overwritten by the
    //   drain tail -- result_captured holds the program's true final write-back.
    //
    //   present_real_now corresponds to the value loading into core_instr THIS
    //   cycle (core_instr <= present_instr); real_shadow[0] is therefore aligned
    //   one cycle behind, i.e. with core_instr being driven to the core, so the
    //   total instruction_in -> result_out path is covered by RESULT_LAT stages.
    wire present_real_now = (held_valid || pop_pending) && core_accepts;

    reg [RESULT_LAT-1:0] real_shadow;          // valid bits in flight to result_out
    // The instruction at the core's write-back / result_out this cycle was REAL
    // iff its valid bit has reached the shadow's last stage.
    wire result_real_now = real_shadow[RESULT_LAT-1];

    // ----------------------------------------------------------------------
    // RESULT FIFO write side (CCLK).  Push the captured result on result_valid.
    // ----------------------------------------------------------------------
    reg        result_valid;                  // 1-cycle pulse when a run completes
    reg [31:0] result_captured;               // last REAL write-back, captured live
    wire       res_wr_full;                   // CCLK side full (not expected)
    /* verilator lint_off UNUSEDSIGNAL */
    wire _u_res_full = res_wr_full;
    /* verilator lint_on UNUSEDSIGNAL */

    // DONE toggle (CCLK): flip once per completed run.  Crossed to ACLK above.
    reg done_tgl_c_r;
    assign done_tgl_c = done_tgl_c_r;

    // Program-complete detection (within C_RUN): we have issued at least one real
    // instruction, nothing is pending/held to issue, and the FIFO has stayed
    // empty through the guard window (so the DMA has finished pushing).
    wire run_body_done = (cstate == C_RUN) && issued_any &&
                         !pop_pending && !held_valid && instr_fifo_empty &&
                         (empty_cnt >= EMPTY_GUARD[GUARD_W-1:0]);

    always @(posedge CCLK) begin
        if (crst) begin
            cstate          <= C_IDLE;
            core_instr      <= 32'b0;
            empty_cnt       <= {GUARD_W{1'b0}};
            drain_cnt       <= {$clog2(DRAIN_NOPS+1){1'b0}};
            issued_any      <= 1'b0;
            pop_pending     <= 1'b0;
            held_instr      <= 32'b0;
            held_valid      <= 1'b0;
            result_valid    <= 1'b0;
            result_captured <= 32'b0;
            done_tgl_c_r    <= 1'b0;
            real_shadow     <= {RESULT_LAT{1'b0}};
        end else begin
            // one-cycle pulse default
            result_valid <= 1'b0;

            // Drive the instruction the combinational mux selected onto the core.
            core_instr <= present_instr;

            // ---- REAL-instruction valid shadow ----
            // Shift the "real & accepted" bit in at stage 0 (aligned with the word
            // loaded into core_instr this cycle), advancing one stage per cycle.
            real_shadow <= {real_shadow[RESULT_LAT-2:0], present_real_now};

            // When a REAL instruction's valid bit reaches the last stage, its
            // write-back is now on core_result -- capture it.  The LAST real
            // instruction's capture survives because the trailing NOPs shift in 0.
            if (result_real_now)
                result_captured <= core_result;

            // ---- pop pipeline bookkeeping ----
            // A pop accepted this cycle => its data is valid NEXT cycle.
            if (can_pop)
                pop_pending <= 1'b1;
            else if (pop_pending && !held_valid)
                // freshly popped word becomes the held word next evaluation
                pop_pending <= 1'b0;

            // Capture a freshly popped word into the held register when it arrives
            // and is not immediately accepted; release the held word when accepted.
            if (held_valid) begin
                if (core_accepts) begin
                    held_valid <= 1'b0;        // accepted -> free to pop again
                    issued_any <= 1'b1;
                end
            end else if (pop_pending) begin
                // instr_rd_data is the freshly popped word THIS cycle.
                if (core_accepts) begin
                    // accepted immediately; no need to hold
                    issued_any  <= 1'b1;
                end else begin
                    // core busy: stash it to re-present until accepted
                    held_instr <= instr_rd_data;
                    held_valid <= 1'b1;
                end
            end

            // ---- empty guard counter ----
            if (cstate == C_RUN && instr_fifo_empty && !pop_pending && !held_valid) begin
                if (empty_cnt < EMPTY_GUARD[GUARD_W-1:0])
                    empty_cnt <= empty_cnt + {{(GUARD_W-1){1'b0}}, 1'b1};
            end else begin
                empty_cnt <= {GUARD_W{1'b0}};
            end

            // ---- state machine ----
            case (cstate)
                C_IDLE: begin
                    drain_cnt  <= {$clog2(DRAIN_NOPS+1){1'b0}};
                    issued_any <= 1'b0;
                    empty_cnt  <= {GUARD_W{1'b0}};
                    if (start_arm)
                        cstate <= C_RUN;
                end

                C_RUN: begin
                    if (run_body_done) begin
                        drain_cnt <= {$clog2(DRAIN_NOPS+1){1'b0}};
                        cstate    <= C_DRAIN;
                    end
                end

                C_DRAIN: begin
                    // Issue NOPs (nothing held/pending here -> present_instr=NOP).
                    if (drain_cnt >= DRAIN_NOPS[$clog2(DRAIN_NOPS+1)-1:0]) begin
                        // The drain tail guarantees the final write-back has long
                        // since landed and been captured LIVE by the real-valid
                        // shadow above (result_captured already holds the LAST real
                        // instruction's write-back; the trailing NOPs shifted in
                        // valid=0 and never overwrote it).  Just pulse valid, flip
                        // the DONE toggle, and return to IDLE -- pushing the true
                        // program result into the result FIFO.
                        result_valid    <= 1'b1;
                        done_tgl_c_r    <= ~done_tgl_c_r;
                        cstate          <= C_IDLE;
                    end else begin
                        // Advance the drain only on free front-end cycles so the
                        // pipeline genuinely retires DRAIN_NOPS instructions.
                        if (core_accepts)
                            drain_cnt <= drain_cnt +
                                {{($clog2(DRAIN_NOPS+1)-1){1'b0}}, 1'b1};
                    end
                end

                default: cstate <= C_IDLE;
            endcase
        end
    end

    // ======================================================================
    // RESULT CDC FIFO (DATA_W=32).  Written CCLK on result_valid; read ACLK by
    // the slave's RESULT register.
    // ======================================================================
    cdc_async_fifo #(
        .DATA_W (32),
        .ADDR_W (RESULT_AW)
    ) u_result_fifo (
        // write (CCLK)
        .wclk    (CCLK),
        .wrst_n  (CRESETn),
        .wr_en   (result_valid),
        .wr_data (result_captured),
        .full    (res_wr_full),
        // read (ACLK)
        .rclk    (ACLK),
        .rrst_n  (ARESETn),
        .rd_en   (res_pop),
        .rd_data (res_rd_data),
        .empty   (res_rd_empty)
    );

endmodule
/* verilator lint_on DECLFILENAME */
