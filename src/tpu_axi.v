`include "tpu_defs.vh"
`timescale 1ns/1ps
//============================================================================
// tpu_axi.v  --  AXI4-Lite SLAVE wrapper for the TPU v2.0 core (module: tpu_axi)
//----------------------------------------------------------------------------
// Makes the verified scalar 5-stage TPU core (src/tpu_top.v, module `TPU`) a
// drop-in, SoC-integratable AXI4-Lite IP.  The core is WRAPPED -- never edited.
//
// CLOCK / RESET
//   * Single clock domain: ACLK drives both the AXI slave logic and the core.
//   * ARESETn is AXI active-LOW reset; the core's `rst` is synchronous active-
//     HIGH, so  core.rst = ~ARESETn.  ARESETn is assumed synchronous to ACLK
//     (standard AXI assumption); all wrapper state resets synchronously.
//
// CORE DRIVE MODEL  (mirrors the testbench step() one-instruction-at-a-time)
//   The wrapper OWNS the core's `instruction_in`.  The core executes whatever
//   word is presented EACH ACLK (one instr/cycle; tensor ops self-stall).  A
//   "single-step issue" model is used:
//     * INSTR register (RW): the 32-bit instruction word to issue.
//     * Writing the CTRL/STEP register (bit0=STEP) drives the INSTR word onto
//       instruction_in for EXACTLY ONE ACLK cycle -- the cycle the W-channel
//       write COMMITS (BVALID asserted).  On every other cycle instruction_in
//       is forced to NOP (`OP_NOP` = 0x00).  A host therefore runs any program
//       by writing INSTR then CTRL.STEP per instruction, in sequence, and reads
//       RESULT/STATUS back after the pipeline latency (5 stages + any stall).
//   The core's result_out and illegal_opcode are continuously mirrored into
//   read-only registers (RESULT, STATUS); STATUS.illegal is sticky and is
//   cleared via CTRL.CLR_ILL (bit1) on a CTRL write.
//
// REGISTER MAP  (32-bit registers; byte address = word offset * 4)
//   word  byte   name      access  description
//   0x00  0x00   CTRL/STEP  W : bit0 STEP   -> issue INSTR for one ACLK cycle
//                               bit1 CLR_ILL-> clear sticky STATUS.illegal
//                          R : bit0 STEP_DONE (1 once any STEP has been issued)
//                              bit1 LAST_ILL  (illegal_opcode seen on last step
//                                              issue cycle; debug view)
//   0x04  0x04   INSTR     RW : the 32-bit instruction word to issue (bit0=step
//                               of CTRL is separate; this holds the program word)
//   0x08  0x08   RESULT    RO : mirror of core result_out (committed WB word)
//   0x0C  0x0C   STATUS    RO : bit0      sticky illegal_opcode (set on any step
//                                          whose decode raised illegal_opcode;
//                                          cleared by CTRL.CLR_ILL / ARESETn)
//                               bit1      core illegal_opcode (live, this cycle)
//                               bit2      STEP_DONE (>=1 step issued since reset)
//                               bit3      BUSY (a STEP is still being held into a
//                                          stalled core front; poll BUSY==0 before
//                                          issuing the next STEP)
//                               [31:4]    reserved (read 0)
//   Unmapped word offsets: reads return 0; writes are dropped.  Both return
//   SLVERR (RRESP/BRESP = 2'b10); mapped accesses return OKAY (2'b00).
//
// AXI4-LITE PROTOCOL
//   * 32-bit data; 4-bit WSTRB byte strobes honoured on register writes.
//   * Five channels AW/W/B/AR/R with registered VALID/READY handshakes -- no
//     combinational AWREADY<->AWVALID (or ARREADY<->ARVALID) loops.
//   * Single outstanding transaction (legal for AXI4-Lite): a write commits
//     only when BOTH AW and W have handshaked; the read data is registered.
//
// SYNTHESIS
//   Synchronous reset on every state element; every reg assigned on every path
//   (no inferred latch); no combinational loops.  Passes verilator
//   --lint-only -Wall (top=tpu_axi) and yosys `check -assert` (top=tpu_axi).
//============================================================================
module tpu_axi #(
    parameter ADDR_W = 4   // byte-address width decoded inside the slave (>=4)
) (
    // ---- AXI4-Lite global signals ----
    input  wire                 ACLK,
    input  wire                 ARESETn,     // active-LOW (core rst = ~ARESETn)

    // ---- Write address channel ----
    // AWADDR[1:0] (byte-within-word) and AWPROT are intentionally accepted but
    // not consumed: this slave is word-addressed (32-bit regs) and applies no
    // access-protection policy -- standard for a simple AXI4-Lite register block.
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [ADDR_W-1:0]    AWADDR,
    input  wire [2:0]           AWPROT,      // unused (accepted, no protection)
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire                 AWVALID,
    output wire                 AWREADY,

    // ---- Write data channel ----
    input  wire [31:0]          WDATA,
    input  wire [3:0]           WSTRB,
    input  wire                 WVALID,
    output wire                 WREADY,

    // ---- Write response channel ----
    output wire [1:0]           BRESP,
    output wire                 BVALID,
    input  wire                 BREADY,

    // ---- Read address channel ----
    // ARADDR[1:0] / ARPROT accepted-but-unused for the same word-addressed,
    // no-protection-policy reason as the AW channel above.
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [ADDR_W-1:0]    ARADDR,
    input  wire [2:0]           ARPROT,      // unused (accepted)
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire                 ARVALID,
    output wire                 ARREADY,

    // ---- Read data channel ----
    output wire [31:0]          RDATA,
    output wire [1:0]           RRESP,
    output wire                 RVALID,
    input  wire                 RREADY
);

    // ======================================================================
    // Local parameters: register word offsets and AXI response codes.
    // ======================================================================
    localparam [1:0] OKAY   = 2'b00;   // AXI4-Lite OKAY response
    localparam [1:0] SLVERR = 2'b10;   // AXI4-Lite SLVERR (unmapped address)

    // Word offsets (byte addr = offset << 2).  ADDR_W-2 word-index bits.
    localparam [ADDR_W-3:0] REG_CTRL   = (ADDR_W-2)'('h0);  // 0x00
    localparam [ADDR_W-3:0] REG_INSTR  = (ADDR_W-2)'('h1);  // 0x04
    localparam [ADDR_W-3:0] REG_RESULT = (ADDR_W-2)'('h2);  // 0x08
    localparam [ADDR_W-3:0] REG_STATUS = (ADDR_W-2)'('h3);  // 0x0C

    // CTRL/STEP write-bit positions.
    localparam integer CTRL_STEP_BIT    = 0;   // bit0: issue INSTR for one cycle
    localparam integer CTRL_CLR_ILL_BIT = 1;   // bit1: clear sticky illegal

    // Active-HIGH synchronous reset for the wrapper + the core.
    wire rst = ~ARESETn;

    // ======================================================================
    // Architectural registers (program-visible state).
    // ======================================================================
    reg [31:0] instr_reg;     // INSTR  : instruction word to issue
    reg        sticky_illegal;// STATUS bit0 : sticky illegal_opcode
    reg        step_done;     // a STEP has been issued since reset
    reg        last_illegal;  // illegal_opcode observed on last step issue cycle

    // ======================================================================
    // AXI write channel handshake state (registered; single outstanding).
    //   awready/wready pulse for exactly one cycle once both AWVALID & WVALID
    //   are seen and no response is already in flight.  The commit happens the
    //   same cycle (write_commit), so STEP issues INSTR on exactly that cycle.
    // ======================================================================
    reg awready_r, wready_r, bvalid_r;
    reg [1:0] bresp_r;

    // Latched write address (captured when AW handshakes; may arrive before W).
    reg [ADDR_W-3:0] awaddr_word;
    reg              aw_seen;   // AW handshake captured, awaiting W
    reg              w_seen;    // W  handshake captured, awaiting AW
    reg [31:0]       wdata_lat; // latched WDATA when W handshakes early
    reg [3:0]        wstrb_lat; // latched WSTRB when W handshakes early

    // ======================================================================
    // AXI read channel handshake state (registered; single outstanding).
    // ======================================================================
    reg arready_r, rvalid_r;
    reg [1:0]  rresp_r;
    reg [31:0] rdata_r;

    // ----------------------------------------------------------------------
    // Output wiring.
    // ----------------------------------------------------------------------
    assign AWREADY = awready_r;
    assign WREADY  = wready_r;
    assign BVALID  = bvalid_r;
    assign BRESP   = bresp_r;
    assign ARREADY = arready_r;
    assign RVALID  = rvalid_r;
    assign RRESP   = rresp_r;
    assign RDATA   = rdata_r;

    // ======================================================================
    // Core instance.
    //   instruction_in = INSTR for exactly the one cycle a CTRL.STEP write
    //   commits; NOP on every other cycle.
    // ======================================================================
    wire [31:0] core_result;
    wire        core_illegal;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [`ST_W-1:0] core_dbg_status;  // debug view; not surfaced over AXI
    /* verilator lint_on UNUSEDSIGNAL */
    // The core latches instruction_in ONLY when its front end is not held; this
    // signal is that hold (load-use / tensor self-stall / multi-cycle mem stall).
    // We MUST keep presenting an issued instruction until the hold clears, else a
    // STEP that lands during a stall would be silently dropped.
    wire             core_dbg_stall;

    // step_issue is the single-cycle pulse that drives INSTR onto the core.
    // It is a write commit to the CTRL register with the STEP bit set.
    reg  step_issue;

    // step_active: a STEP was issued while the core front was held, so the core
    // has not yet latched it.  We hold the instruction on instruction_in until the
    // hold clears (the core then captures it), so the instruction is NEVER dropped.
    // Surfaced as STATUS.BUSY; a host must poll BUSY==0 before the next STEP.
    reg  step_active;

    // Present INSTR on the issue cycle AND while the core front is still held;
    // NOP on every other cycle (NOP is always legal, so the core idles safely).
    wire [31:0] core_instr = (step_issue || step_active) ? instr_reg
                                                         : {24'b0, `OP_NOP};

    TPU u_core (
        .clk            (ACLK),
        .rst            (rst),
        .instruction_in (core_instr),
        .result_out     (core_result),
        .illegal_opcode (core_illegal),
        .dbg_status     (core_dbg_status),
        .dbg_pipe_stall (core_dbg_stall)
    );

    // ======================================================================
    // step_active hold: keeps an issued instruction on instruction_in until the
    // core's front end is free to capture it, so a STEP landing during a pipeline
    // stall (load-use / tensor self-stall / multi-cycle mem op) is NEVER dropped.
    //   - set on a STEP that commits while the front is held (core_dbg_stall);
    //   - cleared the cycle the front is free (the core captures it then);
    //   - never asserts when the front is free at issue -> identical to a plain
    //     one-cycle issue, so the common (no-stall) path is unchanged.
    // A host polls STATUS.BUSY (== step_active) and issues the next STEP only when
    // it reads 0.
    // ======================================================================
    always @(posedge ACLK) begin
        if (rst)
            step_active <= 1'b0;
        else if (step_issue)
            step_active <= core_dbg_stall;
        else if (step_active && !core_dbg_stall)
            step_active <= 1'b0;
    end

    // ======================================================================
    // Write-channel sequencing.
    //   AW and W may arrive in any order.  We register AWREADY/WREADY pulses;
    //   a write COMMITS the cycle both have been seen and no B is pending.
    // ======================================================================
    // A handshake completes this cycle when VALID & READY are both high.
    wire aw_hs = AWVALID && awready_r;
    wire w_hs  = WVALID  && wready_r;

    // Both AW and W are available to commit this cycle (either just handshaked
    // now, or was captured on an earlier cycle).
    wire aw_avail = aw_hs || aw_seen;
    wire w_avail  = w_hs  || w_seen;
    wire write_commit = aw_avail && w_avail && !bvalid_r;

    // Effective write payload/address for the committing transaction (use the
    // freshly-handshaked value if present this cycle, else the latched copy).
    wire [ADDR_W-3:0] commit_word  = aw_hs ? AWADDR[ADDR_W-1:2] : awaddr_word;
    wire [31:0]       commit_wdata = w_hs  ? WDATA : wdata_lat;
    wire [3:0]        commit_wstrb = w_hs  ? WSTRB : wstrb_lat;

    // Per-byte write-strobe mask expanded to a 32-bit bit mask.
    wire [31:0] wmask = {{8{commit_wstrb[3]}}, {8{commit_wstrb[2]}},
                         {8{commit_wstrb[1]}}, {8{commit_wstrb[0]}}};

    // Address decode for the committing write.
    wire commit_is_ctrl  = (commit_word == REG_CTRL);
    wire commit_is_instr = (commit_word == REG_INSTR);
    wire commit_mapped_w = commit_is_ctrl || commit_is_instr ||
                           (commit_word == REG_RESULT) ||
                           (commit_word == REG_STATUS);

    // Strobe-masked INSTR update value (RMW so partial WSTRB is honoured).
    wire [31:0] instr_next = (commit_wdata & wmask) | (instr_reg & ~wmask);

    // CTRL.STEP / CTRL.CLR_ILL are taken from the (byte0) write data; STEP is
    // honoured only if byte lane 0 is actually strobed.
    wire ctrl_byte0    = commit_wstrb[0];
    wire ctrl_do_step  = commit_is_ctrl && ctrl_byte0 &&
                         commit_wdata[CTRL_STEP_BIT];
    wire ctrl_do_clr   = commit_is_ctrl && ctrl_byte0 &&
                         commit_wdata[CTRL_CLR_ILL_BIT];

    // ----------------------------------------------------------------------
    // step_issue: combinational single-cycle pulse (exactly the commit cycle).
    // ----------------------------------------------------------------------
    always @(*) begin
        step_issue = write_commit && ctrl_do_step;
    end

    // ----------------------------------------------------------------------
    // Write FSM registers.
    // ----------------------------------------------------------------------
    always @(posedge ACLK) begin
        if (rst) begin
            awready_r   <= 1'b0;
            wready_r    <= 1'b0;
            bvalid_r    <= 1'b0;
            bresp_r     <= OKAY;
            aw_seen     <= 1'b0;
            w_seen      <= 1'b0;
            awaddr_word <= {(ADDR_W-2){1'b0}};
            wdata_lat   <= 32'b0;
            wstrb_lat   <= 4'b0;
            instr_reg   <= 32'b0;
        end else begin
            // --- AWREADY: assert when we can accept a new address (none held
            //     and no response pending), drop after the handshake. ---
            if (awready_r) begin
                awready_r <= 1'b0;                      // single-cycle pulse
            end else if (!aw_seen && !bvalid_r && AWVALID) begin
                awready_r <= 1'b1;
            end

            // --- WREADY: same policy for the data beat. ---
            if (wready_r) begin
                wready_r <= 1'b0;
            end else if (!w_seen && !bvalid_r && WVALID) begin
                wready_r <= 1'b1;
            end

            // --- Capture early-arriving AW (W not yet here) ---
            if (aw_hs && !write_commit) begin
                aw_seen     <= 1'b1;
                awaddr_word <= AWADDR[ADDR_W-1:2];
            end

            // --- Capture early-arriving W (AW not yet here) ---
            if (w_hs && !write_commit) begin
                w_seen    <= 1'b1;
                wdata_lat <= WDATA;
                wstrb_lat <= WSTRB;
            end

            // --- Commit: apply register effects, raise BVALID. ---
            if (write_commit) begin
                aw_seen  <= 1'b0;
                w_seen   <= 1'b0;
                bvalid_r <= 1'b1;
                bresp_r  <= commit_mapped_w ? OKAY : SLVERR;
                if (commit_is_instr)
                    instr_reg <= instr_next;
            end

            // --- B handshake: clear BVALID once accepted. ---
            if (bvalid_r && BREADY) begin
                bvalid_r <= 1'b0;
            end
        end
    end

    // ======================================================================
    // Read-channel sequencing.
    //   Register ARREADY (single-cycle pulse) and the read data.  The read
    //   reflects the register file as of the AR handshake cycle.
    // ======================================================================
    wire ar_hs = ARVALID && arready_r;
    wire [ADDR_W-3:0] ar_word = ARADDR[ADDR_W-1:2];

    // Mapped-read decode + data mux (combinational on the AR handshake).
    reg [31:0] rdata_mux;
    reg        rmapped;
    always @(*) begin
        rdata_mux = 32'b0;
        rmapped   = 1'b1;
        case (ar_word)
            REG_CTRL:   rdata_mux = {30'b0, last_illegal, step_done};
            REG_INSTR:  rdata_mux = instr_reg;
            REG_RESULT: rdata_mux = core_result;
            REG_STATUS: rdata_mux = {28'b0, step_active, step_done, core_illegal,
                                     sticky_illegal};
            default:    begin rdata_mux = 32'b0; rmapped = 1'b0; end
        endcase
    end

    always @(posedge ACLK) begin
        if (rst) begin
            arready_r <= 1'b0;
            rvalid_r  <= 1'b0;
            rresp_r   <= OKAY;
            rdata_r   <= 32'b0;
        end else begin
            // --- ARREADY: accept a read address when none is in flight. ---
            if (arready_r) begin
                arready_r <= 1'b0;                      // single-cycle pulse
            end else if (!rvalid_r && ARVALID) begin
                arready_r <= 1'b1;
            end

            // --- Latch read data/response on the AR handshake. ---
            if (ar_hs) begin
                rvalid_r <= 1'b1;
                rdata_r  <= rdata_mux;
                rresp_r  <= rmapped ? OKAY : SLVERR;
            end else if (rvalid_r && RREADY) begin
                rvalid_r <= 1'b0;                       // R accepted
            end
        end
    end

    // ======================================================================
    // Status / control side-effect registers.
    //   step_done   : set once any STEP issues.
    //   sticky_illegal: latched when the core raises illegal_opcode; cleared by
    //                   CTRL.CLR_ILL (CLR wins over a concurrent set) or reset.
    //   last_illegal: whether the MOST RECENTLY stepped instruction was illegal.
    //
    //   TIMING.  illegal_opcode asserts the cycle AFTER step_issue: the core
    //   registers instruction_in into IF/ID before the decoder flags it.  So we
    //   must NOT gate on `step_issue && core_illegal` (same cycle) -- that always
    //   samples the previous NOP and never fires.  Because the wrapper drives
    //   instruction_in = NOP (legal) on every non-step cycle, core_illegal is
    //   high ONLY for the one stepped illegal instruction; sampling it every
    //   cycle therefore captures exactly that instruction, timing-robustly.
    // ======================================================================
    always @(posedge ACLK) begin
        if (rst) begin
            sticky_illegal <= 1'b0;
            step_done      <= 1'b0;
            last_illegal   <= 1'b0;
        end else begin
            // Track "last stepped instr illegal?": a new step clears the snapshot;
            // its illegal flag (if any) lands the next cycle and sets it, where it
            // persists until the next step.
            if (step_issue) begin
                step_done    <= 1'b1;
                last_illegal <= 1'b0;
            end else if (core_illegal) begin
                last_illegal <= 1'b1;
            end
            // Sticky-illegal: clear wins over set; core_illegal (only ever high
            // for a stepped illegal instruction) latches the sticky bit.
            if (write_commit && ctrl_do_clr)
                sticky_illegal <= 1'b0;
            else if (core_illegal)
                sticky_illegal <= 1'b1;
        end
    end

endmodule
