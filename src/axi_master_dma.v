`timescale 1ns/1ps
//============================================================================
// axi_master_dma.v  --  AXI4-Lite MASTER DMA engine (module: axi_master_dma)
//----------------------------------------------------------------------------
// PURPOSE
//   Moves a contiguous run of LEN 32-bit words between an EXTERNAL AXI4-Lite
//   memory and a simple INTERNAL streaming port, so the accelerator can fetch
//   or store its own data autonomously instead of the host pushing every word.
//
//        dir == 0 (READ) : external memory  ->  internal sink
//                          for i in 0..len-1:
//                              AR ext_base+4*i ; R word -> {wr_en,wr_idx,wr_data}
//        dir == 1 (WRITE): internal source   ->  external memory
//                          for i in 0..len-1:
//                              {rd_req,rd_idx} ; rd_data -> AW/W ext_base+4*i ; B
//
//   This module is itself an AXI4-Lite *master* on the memory side.  The small
//   command interface (start/ext_addr/len/dir) is NOT AXI -- it is meant to be
//   driven later by the SoC's existing AXI4-Lite *slave* register block.
//
// CLOCK / RESET
//   Single clock domain: ACLK.  ARESETn is AXI active-LOW reset; this module's
//   internal reset is synchronous active-HIGH:  rst = ~ARESETn.  ARESETn is
//   assumed synchronous to ACLK (standard AXI assumption).  EVERY state element
//   resets synchronously and is assigned on every path of the clocked block, so
//   there is no inferred latch and no combinational loop.
//
// AXI4-LITE MASTER PROTOCOL
//   * 32-bit data bus, word-aligned byte addresses incrementing by 4.
//   * ONE outstanding transaction at a time (fully legal for AXI4-Lite):
//       - a WRITE beat drives AW and W, then waits for B before the next word;
//       - a READ  beat drives AR,       then waits for R before the next word.
//   * VALID/READY: the master raises *VALID and holds the payload stable until
//     the slave raises the matching *READY (the handshake).  *VALID is a
//     registered output that is only ever lowered the cycle AFTER a handshake,
//     so there is no combinational AWVALID<->AWREADY (or ARVALID<->ARREADY)
//     dependency -- a slave whose READY combinationally depends on VALID cannot
//     create a loop through this master.
//   * AWVALID and WVALID are asserted together for a write and each is cleared
//     independently when its own channel handshakes (the slave may accept AW and
//     W on different cycles).  The master then waits for BVALID.
//   * Response checking: BRESP / RRESP are sampled at the B / R handshake.  Any
//     response other than OKAY (2'b00) -- i.e. SLVERR (2'b10) or DECERR (2'b11)
//     -- sets the sticky `err` output, aborts the run, and pulses `done`.
//     EXOKAY (2'b01) cannot occur on AXI4-Lite and is also treated as an error.
//
// INTERNAL STREAMING PORT  (simple, registered, one word per beat)
//   READ  dir: as each external word's R beat completes, the engine drives a
//              1-cycle  {wr_en=1, wr_idx=i, wr_data=R.DATA}  into the sink.
//   WRITE dir: before issuing each external word, the engine drives a 1-cycle
//              {rd_req=1, rd_idx=i}; the source must return rd_data for that
//              index in the SAME cycle (combinational source) so the engine can
//              register it and drive it onto the W channel.  rd_idx walks
//              0..len-1 in order.
//
// HANDSHAKE  (command side)
//   `start` is a 1-cycle pulse, sampled ONLY when idle; it latches
//   {ext_addr,len,dir}.  `len` must be >= 1.  `busy` is high for the whole run.
//   `done` is a 1-cycle pulse the cycle the LAST word retires (or the cycle an
//   error aborts the run).  `err` is sticky from the faulting beat until the
//   next `start` (which clears it) -- so a host can read it after `done`.
//
// PARAMETERS
//   ADDR_W : external AXI byte-address width (default 32).
//   LENW   : width of the word-count `len` / internal index (default 8 -> up to
//            255 words per descriptor).  Kept small per the spec.
//============================================================================
module axi_master_dma #(
    parameter integer ADDR_W = 32,
    parameter integer LENW   = 8
) (
    // ---- AXI4-Lite global signals ----
    input  wire                 ACLK,
    input  wire                 ARESETn,      // active-LOW; internal rst = ~ARESETn

    // ---- Command / control interface (NOT AXI; driven by SoC slave later) ----
    input  wire                 start,        // 1-cycle pulse, sampled only when idle
    input  wire [ADDR_W-1:0]    ext_addr,     // external base BYTE address (word aligned)
    input  wire [LENW-1:0]      len,          // number of 32-bit words, >= 1
    input  wire                 dir,          // 0 = READ ext->sink, 1 = WRITE source->ext
    output reg                  busy,         // engine is running
    output reg                  done,         // 1-cycle pulse when the run retires/aborts
    output reg                  err,          // sticky: a non-OKAY response was seen

    // ---- Internal streaming SINK (READ dir): each returned word, in order ----
    output reg                  wr_en,        // 1-cycle strobe per returned word
    output reg  [LENW-1:0]      wr_idx,       // word index 0..len-1
    output reg  [31:0]          wr_data,      // returned external word

    // ---- Internal streaming SOURCE (WRITE dir): pull each word to send ----
    output reg                  rd_req,       // 1-cycle request for word rd_idx
    output reg  [LENW-1:0]      rd_idx,       // word index 0..len-1
    input  wire [31:0]          rd_data,      // source word for rd_idx (same cycle)

    // ---- AXI4-Lite MASTER: write address channel ----
    output reg  [ADDR_W-1:0]    AWADDR,
    output wire [2:0]           AWPROT,       // tied to 0 (non-priv, secure, data)
    output reg                  AWVALID,
    input  wire                 AWREADY,

    // ---- AXI4-Lite MASTER: write data channel ----
    output reg  [31:0]          WDATA,
    output wire [3:0]           WSTRB,        // tied all-ones (full 32-bit word)
    output reg                  WVALID,
    input  wire                 WREADY,

    // ---- AXI4-Lite MASTER: write response channel ----
    input  wire [1:0]           BRESP,
    input  wire                 BVALID,
    output reg                  BREADY,

    // ---- AXI4-Lite MASTER: read address channel ----
    output reg  [ADDR_W-1:0]    ARADDR,
    output wire [2:0]           ARPROT,       // tied to 0
    output reg                  ARVALID,
    input  wire                 ARREADY,

    // ---- AXI4-Lite MASTER: read data channel ----
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [31:0]          RDATA,        // [31:0] used; declared wide for clarity
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [1:0]           RRESP,
    input  wire                 RVALID,
    output reg                  RREADY
);

    // ======================================================================
    // Constants.
    // ======================================================================
    localparam [1:0] RESP_OKAY = 2'b00;       // only response treated as success

    // Active-HIGH synchronous reset derived from the active-LOW AXI reset.
    wire rst = ~ARESETn;

    // AXI sideband tie-offs: full-word, default-protection accesses.
    assign AWPROT = 3'b000;
    assign ARPROT = 3'b000;
    assign WSTRB  = 4'b1111;

    // ======================================================================
    // FSM states.  One outstanding AXI transaction at a time.
    //   IDLE  : waiting for start.
    //   RADDR : READ  -- drive ARVALID, wait for AR handshake.
    //   RDATA : READ  -- drive RREADY, wait for R handshake (capture word/resp).
    //   WADDR : WRITE -- request source word; drive AW+W, wait for both hs.
    //   WRESP : WRITE -- drive BREADY, wait for B handshake (capture resp).
    //   FIN   : pulse done (and busy->0), return to IDLE.
    // ======================================================================
    localparam [2:0] S_IDLE  = 3'd0,
                     S_RADDR = 3'd1,
                     S_RDATA = 3'd2,
                     S_WADDR = 3'd3,
                     S_WRESP = 3'd4,
                     S_FIN   = 3'd5;
    reg [2:0] state;

    // ======================================================================
    // Latched descriptor + iterator.
    //   base_addr : latched external base byte address.
    //   total     : latched word count (len).
    //   idx       : index of the word CURRENTLY in flight (0..total-1).
    //   cur_addr  : base_addr + 4*idx, the byte address of the current word.
    // The direction does NOT need a latched copy: `start` routes into the READ
    // (S_RADDR) or WRITE (S_WADDR) sub-FSM, and the active state thereafter
    // encodes the direction for the rest of the run.
    // idx is LENW+1 wide so the terminal compare idx==total (total up to 2^LENW)
    // never truncates; the index FIELDS exposed externally take the low LENW bits.
    // ======================================================================
    reg [ADDR_W-1:0] base_addr;
    reg [LENW:0]     total;
    reg [LENW:0]     idx;

    // Per-channel "this beat's AXI phase already handshaked" flags, so AW and W
    // can complete on different cycles before the master moves on to wait for B.
    reg aw_done;     // AW handshake captured for the current write word
    reg w_done;      // W  handshake captured for the current write word

    // Width helpers.
    localparam [LENW:0] IDX_ONE  = { {LENW{1'b0}}, 1'b1 };
    localparam [LENW:0] IDX_ZERO = { (LENW+1){1'b0} };

    // Current byte address = base + 4*idx (idx low bits; <<2 for word stride).
    wire [ADDR_W-1:0] cur_addr =
        base_addr + { {(ADDR_W-LENW-2){1'b0}}, idx[LENW-1:0], 2'b00 };

    // Channel handshake pulses (combinational; VALID is registered so no loop).
    wire ar_hs = ARVALID && ARREADY;
    wire r_hs  = RREADY  && RVALID;
    wire aw_hs = AWVALID && AWREADY;
    wire w_hs  = WVALID  && WREADY;
    wire b_hs  = BREADY  && BVALID;

    // A response code that is not OKAY is an error (covers SLVERR/DECERR/EXOKAY).
    wire r_err = r_hs && (RRESP != RESP_OKAY);
    wire b_err = b_hs && (BRESP != RESP_OKAY);

    // ======================================================================
    // Main sequential FSM.  Every reg is assigned on every path (defaults at
    // the top of the else-branch), so no latch is inferred.
    // ======================================================================
    always @(posedge ACLK) begin
        if (rst) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            err       <= 1'b0;
            wr_en     <= 1'b0;
            wr_idx    <= {LENW{1'b0}};
            wr_data   <= 32'b0;
            rd_req    <= 1'b0;
            rd_idx    <= {LENW{1'b0}};
            AWADDR    <= {ADDR_W{1'b0}};
            AWVALID   <= 1'b0;
            WDATA     <= 32'b0;
            WVALID    <= 1'b0;
            BREADY    <= 1'b0;
            ARADDR    <= {ADDR_W{1'b0}};
            ARVALID   <= 1'b0;
            RREADY    <= 1'b0;
            base_addr <= {ADDR_W{1'b0}};
            total     <= IDX_ZERO;
            idx       <= IDX_ZERO;
            aw_done   <= 1'b0;
            w_done    <= 1'b0;
        end else begin
            // ---- one-cycle-pulse defaults (assigned on every path) ----
            done   <= 1'b0;
            wr_en  <= 1'b0;
            rd_req <= 1'b0;

            case (state)
                // ----------------------------------------------------- IDLE
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        // Latch the descriptor and clear the sticky error.
                        base_addr <= ext_addr;
                        total     <= { 1'b0, len };
                        idx       <= IDX_ZERO;
                        err       <= 1'b0;
                        busy      <= 1'b1;
                        aw_done   <= 1'b0;
                        w_done    <= 1'b0;
                        if (dir) begin
                            // WRITE: request the first source word; AW+W issued
                            // in S_WADDR next cycle once rd_data is registered.
                            rd_req <= 1'b1;
                            rd_idx <= {LENW{1'b0}};
                            state  <= S_WADDR;
                        end else begin
                            // READ: drive the first AR.
                            ARADDR  <= ext_addr;
                            ARVALID <= 1'b1;
                            state   <= S_RADDR;
                        end
                    end
                end

                // ------------------------------------------ READ: AR phase
                S_RADDR: begin
                    if (ar_hs) begin
                        // AR accepted -> drop ARVALID, raise RREADY for the data.
                        ARVALID <= 1'b0;
                        RREADY  <= 1'b1;
                        state   <= S_RDATA;
                    end
                end

                // ------------------------------------------ READ: R phase
                S_RDATA: begin
                    if (r_hs) begin
                        RREADY <= 1'b0;          // single outstanding: drop READY
                        if (r_err) begin
                            // Non-OKAY response: flag, abort, retire.
                            err   <= 1'b1;
                            state <= S_FIN;
                        end else begin
                            // Push the returned word into the internal sink.
                            wr_en   <= 1'b1;
                            wr_idx  <= idx[LENW-1:0];
                            wr_data <= RDATA;
                            if ((idx + IDX_ONE) == total) begin
                                // Last word retired.
                                state <= S_FIN;
                            end else begin
                                // Advance to the next word: issue its AR.
                                idx     <= idx + IDX_ONE;
                                ARADDR  <= base_addr +
                                    { {(ADDR_W-LENW-2){1'b0}},
                                      (idx[LENW-1:0] + 1'b1), 2'b00 };
                                ARVALID <= 1'b1;
                                state   <= S_RADDR;
                            end
                        end
                    end
                end

                // ------------------------------------------ WRITE: AW+W phase
                S_WADDR: begin
                    // On entry from IDLE / previous word, rd_req was pulsed the
                    // PREVIOUS cycle, so rd_data is valid THIS cycle.  Register
                    // it and present AW+W on the first cycle of this state.
                    if (!AWVALID && !WVALID && !aw_done && !w_done) begin
                        // First cycle in this state for this word: launch AW+W.
                        AWADDR  <= cur_addr;
                        AWVALID <= 1'b1;
                        WDATA   <= rd_data;
                        WVALID  <= 1'b1;
                    end else begin
                        // Independently retire AW and W as each handshakes.
                        if (aw_hs) AWVALID <= 1'b0;
                        if (w_hs)  WVALID  <= 1'b0;
                        if (aw_hs) aw_done <= 1'b1;
                        if (w_hs)  w_done  <= 1'b1;

                        // Once BOTH address and data are accepted, await B.
                        if ((aw_hs || aw_done) && (w_hs || w_done)) begin
                            BREADY <= 1'b1;
                            state  <= S_WRESP;
                        end
                    end
                end

                // ------------------------------------------ WRITE: B phase
                S_WRESP: begin
                    if (b_hs) begin
                        BREADY  <= 1'b0;          // single outstanding
                        aw_done <= 1'b0;          // reset per-word phase flags
                        w_done  <= 1'b0;
                        if (b_err) begin
                            err   <= 1'b1;
                            state <= S_FIN;
                        end else if ((idx + IDX_ONE) == total) begin
                            state <= S_FIN;        // last word committed
                        end else begin
                            // Next word: request the source data, then S_WADDR.
                            idx    <= idx + IDX_ONE;
                            rd_req <= 1'b1;
                            rd_idx <= idx[LENW-1:0] + 1'b1;
                            state  <= S_WADDR;
                        end
                    end
                end

                // ----------------------------------------------------- FIN
                S_FIN: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                // ----------------------------------------------------- safe
                default: begin
                    state <= S_IDLE;
                    busy  <= 1'b0;
                end
            endcase
        end
    end

endmodule
