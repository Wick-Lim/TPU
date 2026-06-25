`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// test/tpu_axi_tb.v  --  AXI4-Lite BUS-FUNCTIONAL-MODEL (BFM) testbench for the
//                        tpu_axi slave wrapper (src/tpu_axi.v) around module TPU.
//----------------------------------------------------------------------------
// PURPOSE
//   Prove the unchanged TPU core is correctly CONTROLLED THROUGH THE AXI BUS:
//   every program-visible action (issue an instruction, read RESULT, read
//   STATUS, clear the sticky illegal flag) is driven over the five AXI4-Lite
//   channels (AW/W/B/AR/R) by reusable BFM tasks that honour the full
//   VALID/READY handshake -- there are NO protocol shortcuts and NO hierarchical
//   pokes of wrapper or core state.  The ONLY thing the TB reaches into the DUT
//   for is... nothing: every check is against a value READ BACK OVER AXI and
//   compared to an INDEPENDENT integer/real golden computed here in the TB.
//
// REGISTER MAP (from src/tpu_axi.v, byte address = word offset):
//   0x00 CTRL   W : bit0 STEP (issue INSTR for one ACLK cycle),
//                   bit1 CLR_ILL (clear sticky STATUS.illegal)
//        CTRL   R : bit0 STEP_DONE, bit1 LAST_ILL
//   0x04 INSTR  RW: 32-bit instruction word to issue (WSTRB-maskable)
//   0x08 RESULT RO: mirror of core result_out (committed WB word)
//   0x0C STATUS RO: bit0 sticky illegal, bit1 live illegal_opcode,
//                   bit2 STEP_DONE, [31:3] reserved 0
//
// STEP-ISSUE SEMANTICS (from src/tpu_axi.v): writing CTRL with bit0=1 drives the
//   INSTR word onto instruction_in for EXACTLY the one ACLK cycle the CTRL write
//   COMMITS; every other cycle instruction_in is NOP.  The core latches that one
//   word into its IF stage (multi-cycle tensor ops self-stall internally, so a
//   single STEP launches even a tensor op).  So a host runs a program by, per
//   instruction: axi_write(INSTR, word); axi_write(CTRL, STEP).  Because the core
//   is a 5-stage pipeline, RESULT/STATUS are read back AFTER a drain of NOP steps.
//
// GOLDENS are INDEPENDENT (computed in the integer domain here), never mirrored
//   from DUT internals.  Prints "ALL N TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module tpu_axi_tb;

    // ===================== AXI4-Lite signals (TB drives master side) =========
    // ADDR_W=8 so the four mapped word offsets (0x0/0x4/0x8/0xC) sit inside a
    // larger decode window, leaving higher offsets (e.g. 0x10) UNMAPPED so the
    // SLVERR path can be exercised.  (The wrapper decodes word = ADDR[ADDR_W-1:2].)
    localparam ADDR_W = 8;

    reg                 ACLK;
    reg                 ARESETn;

    // Write address channel
    reg  [ADDR_W-1:0]   AWADDR;
    reg  [2:0]          AWPROT;
    reg                 AWVALID;
    wire                AWREADY;
    // Write data channel
    reg  [31:0]         WDATA;
    reg  [3:0]          WSTRB;
    reg                 WVALID;
    wire                WREADY;
    // Write response channel
    wire [1:0]          BRESP;
    wire                BVALID;
    reg                 BREADY;
    // Read address channel
    reg  [ADDR_W-1:0]   ARADDR;
    reg  [2:0]          ARPROT;
    reg                 ARVALID;
    wire                ARREADY;
    // Read data channel
    wire [31:0]         RDATA;
    wire [1:0]          RRESP;
    wire                RVALID;
    reg                 RREADY;

    // ===================== DUT =====================
    tpu_axi #(.ADDR_W(ADDR_W)) dut (
        .ACLK(ACLK), .ARESETn(ARESETn),
        .AWADDR(AWADDR), .AWPROT(AWPROT), .AWVALID(AWVALID), .AWREADY(AWREADY),
        .WDATA(WDATA), .WSTRB(WSTRB), .WVALID(WVALID), .WREADY(WREADY),
        .BRESP(BRESP), .BVALID(BVALID), .BREADY(BREADY),
        .ARADDR(ARADDR), .ARPROT(ARPROT), .ARVALID(ARVALID), .ARREADY(ARREADY),
        .RDATA(RDATA), .RRESP(RRESP), .RVALID(RVALID), .RREADY(RREADY)
    );

    // 10 ns clock.
    initial ACLK = 1'b0;
    always #5 ACLK = ~ACLK;

    // ===================== register byte-addresses =====================
    localparam [ADDR_W-1:0] A_CTRL    = 8'h00;
    localparam [ADDR_W-1:0] A_INSTR   = 8'h04;
    localparam [ADDR_W-1:0] A_RESULT  = 8'h08;
    localparam [ADDR_W-1:0] A_STATUS  = 8'h0C;
    localparam [ADDR_W-1:0] A_UNMAPPED= 8'h10;   // word 4: past the 4-register map

    // CTRL write bits / STATUS read bits.
    localparam CTRL_STEP_BIT    = 0;
    localparam CTRL_CLR_ILL_BIT = 1;
    localparam ST_STICKY_ILL    = 0;   // STATUS bit0 : sticky illegal
    localparam ST_LIVE_ILL      = 1;   // STATUS bit1 : live illegal_opcode
    localparam ST_STEP_DONE     = 2;   // STATUS bit2 : step_done
    localparam ST_BUSY          = 3;   // STATUS bit3 : BUSY (held STEP in stall)

    // AXI response codes.
    localparam [1:0] R_OKAY   = 2'b00;
    localparam [1:0] R_SLVERR = 2'b10;

    // ===================== bookkeeping =====================
    integer pass, fail;

    // Coverage flags: prove the BFM actually exercised each channel handshake.
    integer cov_aw, cov_w, cov_b, cov_ar, cov_r;

    // ===================== check primitives =====================
    task chk_eq;
        input [511:0] name;
        input integer got, exp;
        begin
            if (got === exp) begin
                pass = pass + 1;
                $display("  PASS %0s : %0d", name, got);
            end else begin
                fail = fail + 1;
                $display("  FAIL %0s : got=%0d exp=%0d", name, got, exp);
                $fatal(1, "AXI BFM test mismatch");
            end
        end
    endtask

    // ===================== instruction builders (same as tpu_tb.v) ==========
    function [31:0] R;
        input [7:0] op; input [3:0] a, b, c; input [11:0] im;
        R = {op, a, b, c, im};
    endfunction
    function [31:0] Iimm;
        input [7:0] op; input [3:0] c; input [19:0] im;
        Iimm = {op, c, im};
    endfunction

    //========================================================================
    // AXI4-Lite BFM master tasks.  Full VALID/READY handshakes; single
    // outstanding; AW and W are issued concurrently (legal), then B is awaited.
    //========================================================================

    // -- axi_write(addr, data) : full WSTRB=all-bytes write, wait for BRESP. --
    task axi_write;
        input [ADDR_W-1:0] addr;
        input [31:0]       data;
        reg aw_done, w_done;
        begin
            @(negedge ACLK);
            AWADDR  = addr; AWPROT = 3'b000; AWVALID = 1'b1;
            WDATA   = data; WSTRB  = 4'hF;   WVALID  = 1'b1;
            BREADY  = 1'b1;
            aw_done = 1'b0; w_done = 1'b0;
            // Hold AWVALID/WVALID until each is accepted (READY seen on posedge).
            while (!(aw_done && w_done)) begin
                @(posedge ACLK);
                if (AWVALID && AWREADY) begin aw_done = 1'b1; cov_aw = cov_aw + 1; end
                if (WVALID  && WREADY ) begin w_done  = 1'b1; cov_w  = cov_w  + 1; end
                @(negedge ACLK);
                if (aw_done) AWVALID = 1'b0;
                if (w_done ) WVALID  = 1'b0;
            end
            // Await the write response (BVALID) with BREADY asserted.
            forever begin
                @(posedge ACLK);
                if (BVALID && BREADY) begin
                    cov_b = cov_b + 1;
                    if (BRESP !== R_OKAY) begin
                        fail = fail + 1;
                        $display("  FAIL axi_write BRESP : got=%b exp=00 @addr=%h",
                                 BRESP, addr);
                        $fatal(1, "AXI write response not OKAY");
                    end
                    @(negedge ACLK); BREADY = 1'b0;
                    disable axi_write;
                end
            end
        end
    endtask

    // -- axi_write_err(addr,data) : write expecting SLVERR (unmapped address). --
    task axi_write_err;
        input [ADDR_W-1:0] addr;
        input [31:0]       data;
        reg aw_done, w_done;
        begin
            @(negedge ACLK);
            AWADDR  = addr; AWPROT = 3'b000; AWVALID = 1'b1;
            WDATA   = data; WSTRB  = 4'hF;   WVALID  = 1'b1;
            BREADY  = 1'b1;
            aw_done = 1'b0; w_done = 1'b0;
            while (!(aw_done && w_done)) begin
                @(posedge ACLK);
                if (AWVALID && AWREADY) begin aw_done = 1'b1; cov_aw = cov_aw + 1; end
                if (WVALID  && WREADY ) begin w_done  = 1'b1; cov_w  = cov_w  + 1; end
                @(negedge ACLK);
                if (aw_done) AWVALID = 1'b0;
                if (w_done ) WVALID  = 1'b0;
            end
            forever begin
                @(posedge ACLK);
                if (BVALID && BREADY) begin
                    cov_b = cov_b + 1;
                    chk_eq("unmapped write SLVERR", (BRESP === R_SLVERR) ? 1 : 0, 1);
                    @(negedge ACLK); BREADY = 1'b0;
                    disable axi_write_err;
                end
            end
        end
    endtask

    // -- axi_read(addr, data) : AR handshake then capture R; check RRESP=OKAY. --
    task axi_read;
        input  [ADDR_W-1:0] addr;
        output [31:0]       data;
        reg ar_done;
        begin
            @(negedge ACLK);
            ARADDR = addr; ARPROT = 3'b000; ARVALID = 1'b1; RREADY = 1'b1;
            ar_done = 1'b0;
            // Drive ARVALID until accepted.
            while (!ar_done) begin
                @(posedge ACLK);
                if (ARVALID && ARREADY) begin ar_done = 1'b1; cov_ar = cov_ar + 1; end
                @(negedge ACLK);
                if (ar_done) ARVALID = 1'b0;
            end
            // Await read data.
            forever begin
                @(posedge ACLK);
                if (RVALID && RREADY) begin
                    cov_r = cov_r + 1;
                    data = RDATA;
                    if (RRESP !== R_OKAY) begin
                        fail = fail + 1;
                        $display("  FAIL axi_read RRESP : got=%b exp=00 @addr=%h",
                                 RRESP, addr);
                        $fatal(1, "AXI read response not OKAY");
                    end
                    @(negedge ACLK); RREADY = 1'b0;
                    disable axi_read;
                end
            end
        end
    endtask

    //========================================================================
    // Program-driving helpers built ON TOP of the BFM tasks.
    //========================================================================

    // Issue exactly one instruction word through the bus: write INSTR then pulse
    // CTRL.STEP.  The wrapper presents INSTR on instruction_in for the one ACLK
    // cycle the CTRL write commits (the core latches it into IF).
    task issue;
        input [31:0] word;
        begin
            axi_write(A_INSTR, word);
            axi_write(A_CTRL, 32'h1 << CTRL_STEP_BIT);   // STEP
        end
    endtask

    // Issue N NOP steps to drain the pipeline so all in-flight results retire to
    // the architectural register file.  Each axi_write pair already consumes
    // several ACLK cycles, so a handful of NOP issues comfortably exceeds the
    // 5-stage latency (plus any tensor self-stall).
    task drain;
        input integer n;
        integer k;
        begin
            for (k = 0; k < n; k = k + 1) issue(R(`OP_NOP, 0, 0, 0, 0));
        end
    endtask

    // Capture a scalar instruction's RESULT over the bus.  The wrapper mirrors
    // the core's result_out LIVE (it pulses for exactly the one cycle that
    // instruction's write-back retires), so the only reliable way to observe a
    // value is to issue the producing instruction LAST and read RESULT
    // IMMEDIATELY: the AR-channel handshake of the read naturally lands ~4 ACLK
    // cycles after the STEP commit -- exactly the 5-stage WB cycle -- and the
    // wrapper latches that word into RDATA at the AR handshake (then holds it
    // until R is consumed).  `producer` must therefore be a scalar value-writing
    // op (LOADI/ADD/SUB/ADDI/RDSTATUS/...), issued as the final program step.
    task issue_then_read_result;
        input  [31:0] producer;
        output [31:0] data;
        begin
            issue(producer);
            axi_read(A_RESULT, data);
        end
    endtask

    // Read the CORE's internal status word over the bus.  RDSTATUS rC=status is a
    // scalar op (writes_reg), so its WB lands at the deterministic 5-stage offset;
    // we drain first so any in-flight tensor/scalar status updates have settled,
    // issue RDSTATUS to capture the core status into a scratch reg, drain so that
    // register is committed, then SURFACE it via a scalar move issued LAST and
    // read RESULT immediately.  This is the authoritative way to observe the
    // core's sticky illegal/sat status (set by an illegal decode, cleared by
    // CLRSTATUS) entirely over AXI.
    task read_core_status;
        output [31:0] data;
        begin
            drain(4);
            issue(R(`OP_RDSTATUS, 0, 0, 9, 12'd0));   // r9 = core status word
            drain(4);
            issue_then_read_result(R(`OP_ADDI, 9, 0, 10, 12'd0), data);  // r10=r9
        end
    endtask

    // ===================== local readback regs =====================
    reg [31:0] rd;
    reg [31:0] rd_instr;
    reg [31:0] rd_status;
    reg [31:0] rd_ctrl;

    integer expv;

    //========================================================================
    // MAIN
    //========================================================================
    initial begin
        $dumpfile("tpu_axi_waveform.vcd");
        $dumpvars(0, tpu_axi_tb);

        pass = 0; fail = 0;
        cov_aw = 0; cov_w = 0; cov_b = 0; cov_ar = 0; cov_r = 0;

        // Idle all master-driven signals.
        AWADDR = 0; AWPROT = 0; AWVALID = 0;
        WDATA  = 0; WSTRB  = 0; WVALID  = 0;
        BREADY = 0;
        ARADDR = 0; ARPROT = 0; ARVALID = 0; RREADY = 0;

        // ---- Reset (active-low) ----
        ARESETn = 1'b0;
        repeat (5) @(posedge ACLK);
        @(negedge ACLK);
        ARESETn = 1'b1;
        @(posedge ACLK);

        //====================================================================
        // TEST 0 -- INSTR register is plain RW over the bus (WSTRB full + byte).
        //   Independent golden: what we wrote is what we read.
        //====================================================================
        $display("[AXI TEST 0] INSTR register RW + WSTRB byte-masking over the bus");
        axi_write(A_INSTR, 32'hDEAD_BEEF);
        axi_read (A_INSTR, rd_instr);
        chk_eq("INSTR readback full-word", rd_instr, 32'hDEAD_BEEF);

        // Byte-masked write: change only byte lane 1 (bits[15:8]) to 0x55.
        @(negedge ACLK);
        AWADDR=A_INSTR; AWPROT=0; AWVALID=1'b1;
        WDATA=32'h0000_5500; WSTRB=4'b0010; WVALID=1'b1; BREADY=1'b1;
        begin : bytewr
            reg awd, wd;
            awd=0; wd=0;
            while (!(awd && wd)) begin
                @(posedge ACLK);
                if (AWVALID && AWREADY) begin awd=1; cov_aw=cov_aw+1; end
                if (WVALID  && WREADY ) begin wd =1; cov_w =cov_w +1; end
                @(negedge ACLK);
                if (awd) AWVALID=0;
                if (wd ) WVALID =0;
            end
            forever begin
                @(posedge ACLK);
                if (BVALID && BREADY) begin
                    cov_b=cov_b+1; @(negedge ACLK); BREADY=0; disable bytewr;
                end
            end
        end
        axi_read(A_INSTR, rd_instr);
        // golden: 0xDEADBEEF with byte1 replaced by 0x55 -> 0xDEAD55EF
        chk_eq("INSTR WSTRB byte-merge", rd_instr, 32'hDEAD_55EF);

        //====================================================================
        // TEST 1 -- SCALAR PROGRAM over the bus.  LOADI r1=5; LOADI r2=7;
        //   ADD r3=r1+r2; ADDI r3=r3+10; read RESULT -> must equal 22.
        //   INDEPENDENT golden computed in plain integers below.
        //====================================================================
        $display("[AXI TEST 1] scalar program LOADI/ADD/ADDI through the bus");
        // independent golden
        expv = 5 + 7;          // ADD r3
        expv = expv + 10;      // ADDI r3 += 10  => 22

        issue(Iimm(`OP_LOADI, 1, 20'd5));        // r1 = 5
        issue(Iimm(`OP_LOADI, 2, 20'd7));        // r2 = 7
        issue(R(`OP_ADD,  1, 2, 3, 12'd0));      // r3 = r1 + r2 = 12
        // ADDI is the FINAL producer: issue it last, read RESULT immediately so
        // the read's AR handshake lands on its WB cycle.
        issue_then_read_result(R(`OP_ADDI, 3, 0, 3, 12'd10), rd);   // r3 = 12+10 = 22
        chk_eq("scalar RESULT (5+7+10)", $signed(rd), expv);

        //====================================================================
        // TEST 2 -- a SECOND scalar program proves RESULT tracks fresh work.
        //   LOADI r4=100; LOADI r5=37; SUB r6=r4-r5 -> 63.
        //====================================================================
        $display("[AXI TEST 2] second scalar program SUB through the bus");
        expv = 100 - 37;       // 63
        issue(Iimm(`OP_LOADI, 4, 20'd100));
        issue(Iimm(`OP_LOADI, 5, 20'd37));
        issue_then_read_result(R(`OP_SUB, 4, 5, 6, 12'd0), rd);  // r6 = 100-37 = 63
        chk_eq("scalar RESULT (100-37)", $signed(rd), expv);

        // STATUS step_done bit must now be set (we have stepped many times).
        axi_read(A_STATUS, rd_status);
        chk_eq("STATUS step_done set", (rd_status >> ST_STEP_DONE) & 1, 1);
        // No illegal yet -> sticky illegal clear.
        chk_eq("STATUS sticky-illegal clear (pre)", (rd_status >> ST_STICKY_ILL) & 1, 0);
        // BUSY must be 0 when idle: every prior STEP was issued into a non-stalled
        // front, so the wrapper never had to hold an instruction across a stall.
        chk_eq("STATUS BUSY clear when idle", (rd_status >> ST_BUSY) & 1, 0);
        // CTRL read: STEP_DONE bit0 set.
        axi_read(A_CTRL, rd_ctrl);
        chk_eq("CTRL STEP_DONE", (rd_ctrl >> CTRL_STEP_BIT) & 1, 1);

        //====================================================================
        // TEST 3 -- ILLEGAL OPCODE through the bus sets the core's sticky illegal
        //   status; the CLRSTATUS path clears it.  INDEPENDENT golden: opcode 0xAA
        //   is not in tpu_defs.vh, so the core MUST raise illegal_opcode and latch
        //   status[ST_ILLEGAL_BIT].  We observe the core's status word -- read
        //   entirely over AXI via RDSTATUS + a scalar surface read -- which is the
        //   authoritative, software-visible illegal flag (exactly what tpu_tb.v
        //   PART 7 checks, here driven through the bus instead of a direct port).
        //====================================================================
        $display("[AXI TEST 3] illegal opcode sets core status.illegal; CLRSTATUS clears");
        // Baseline: status illegal bit must be 0 before the illegal op.
        read_core_status(rd_status);
        chk_eq("core status illegal clear (pre)", (rd_status >> `ST_ILLEGAL_BIT) & 1, 0);

        issue(R(8'hAA, 1, 2, 3, 12'h000));       // illegal opcode (0xAA not in ISA)
        read_core_status(rd_status);
        chk_eq("core status illegal SET after illegal op",
               (rd_status >> `ST_ILLEGAL_BIT) & 1, 1);

        // The WRAPPER's own STATUS.sticky_illegal must ALSO latch now: the fixed
        // logic samples illegal_opcode the cycle it actually asserts (one cycle
        // after step_issue), not gated on the same-cycle step_issue (which never
        // fired).  This guards that regression.
        axi_read(A_STATUS, rd_status);
        chk_eq("wrapper STATUS sticky-illegal SET after illegal step",
               (rd_status >> ST_STICKY_ILL) & 1, 1);

        // Clear the core sticky bits via CLRSTATUS, re-read over the bus.
        issue(R(`OP_CLRSTATUS, 0, 0, 0, 12'd0));
        read_core_status(rd_status);
        chk_eq("core status illegal CLEARED after CLRSTATUS",
               (rd_status >> `ST_ILLEGAL_BIT) & 1, 0);

        // A legal instruction after clear must NOT re-set the sticky bit.
        issue(Iimm(`OP_LOADI, 7, 20'd9));
        read_core_status(rd_status);
        chk_eq("core status stays clear after legal op",
               (rd_status >> `ST_ILLEGAL_BIT) & 1, 0);

        // Also exercise the wrapper's CTRL.CLR_ILL write path and the STATUS/CTRL
        // register READS over the bus (AXI coverage + sticky-state regs).  The
        // wrapper's own sticky-illegal mirror is cleared here and STEP_DONE stays
        // set (we have issued many steps).
        axi_write(A_CTRL, 32'h1 << CTRL_CLR_ILL_BIT);   // wrapper CLR_ILL
        axi_read(A_STATUS, rd_status);
        chk_eq("wrapper STATUS sticky-illegal clear after CLR_ILL",
               (rd_status >> ST_STICKY_ILL) & 1, 0);
        chk_eq("wrapper STATUS step_done still set", (rd_status >> ST_STEP_DONE) & 1, 1);
        axi_read(A_CTRL, rd_ctrl);
        chk_eq("wrapper CTRL STEP_DONE still set", (rd_ctrl >> CTRL_STEP_BIT) & 1, 1);

        //====================================================================
        // TEST 4 -- illegal opcode is a SAFE NOP: a register written just before
        //   the illegal op survives, and a legal op after it commits to RESULT.
        //   LOADI r8=123; <illegal>; ADDI r8=r8+1 -> RESULT must be 124, proving
        //   the bus-driven illegal neither corrupted r8 nor wedged the pipeline.
        //====================================================================
        $display("[AXI TEST 4] illegal op is a safe NOP (state survives over the bus)");
        expv = 123 + 1;        // 124
        issue(Iimm(`OP_LOADI, 8, 20'd123));      // r8 = 123
        issue(R(8'hBB, 0, 0, 0, 12'h000));       // illegal: must not touch r8
        // ADDI is the final producer -> read RESULT immediately.
        issue_then_read_result(R(`OP_ADDI, 8, 0, 8, 12'd1), rd);   // r8 = 123+1 = 124
        chk_eq("RESULT after illegal-as-NOP (123+1)", $signed(rd), expv);
        // (sticky illegal will be set again from the 0xBB op) -- clear for tidiness
        axi_write(A_CTRL, 32'h1 << CTRL_CLR_ILL_BIT);

        //====================================================================
        // TEST 5 -- TENSOR op end-to-end over the bus: SOFTMAX.  Seed an 8-wide
        //   logit vector into DMEM (via LOADI/STORE), TLOAD it to TM, run SOFTMAX,
        //   then read the RETURNED STATUS WORD over RESULT and check its UNIT and
        //   ARGMAX fields against an INDEPENDENT integer golden (argmax = index of
        //   the largest logit; unit = UNIT_SOFTMAX from tpu_defs.vh).
        //   This proves a multi-cycle self-stalling tensor op is launched by a
        //   single bus STEP and retires its status to RESULT over AXI.
        //====================================================================
        $display("[AXI TEST 5] SOFTMAX tensor op end-to-end over the bus");
        begin : softmax_test
            // Logits chosen so the argmax is unambiguous (index 5 is the largest).
            // Stored as Q7.8-ish raw 16-bit values; softmax argmax only depends on
            // their ORDER, so the integer golden is simply argmax of this array.
            integer L [0:7];
            integer gi, gmax, gargmax;
            integer got_unit, got_argmax;
            integer line, lane, addr16;
            L[0]=1*256; L[1]=2*256; L[2]=0;     L[3]=-1*256;
            L[4]=1*256; L[5]=6*256; L[6]=1*256; L[7]=-2*256;   // max at idx 5
            gmax = L[0]; gargmax = 0;
            for (gi = 1; gi < 8; gi = gi + 1)
                if (L[gi] > gmax) begin gmax = L[gi]; gargmax = gi; end

            // Seed logits into DMEM[80..87] using LOADI(r14=val)+LOADI(r12=addr)+
            // STORE.  Each value is sign-extended into a 32-bit reg by LOADI.
            for (gi = 0; gi < 8; gi = gi + 1) begin
                addr16 = 80 + gi;
                issue(Iimm(`OP_LOADI, 14, {{4{L[gi][15]}}, L[gi][15:0]})); // r14=val
                issue(Iimm(`OP_LOADI, 12, {12'd0, addr16[7:0]}));         // r12=addr
                issue(R(`OP_STORE, 12, 14, 0, 12'd0));                    // DMEM=r14
            end
            drain(4);

            // TLOAD DMEM[80..83]->TM[16], DMEM[84..87]->TM[17].
            issue(Iimm(`OP_LOADI, 3, 20'd80));
            issue(R(`OP_TLOAD, 3, 0, 0, {7'd0, 5'd16}));
            drain(4);
            issue(Iimm(`OP_LOADI, 3, 20'd84));
            issue(R(`OP_TLOAD, 3, 0, 0, {7'd0, 5'd17}));
            drain(4);

            // Clear any prior sticky illegal (from TEST 4's 0xBB) so the
            // post-SOFTMAX core-status illegal check below is meaningful.
            issue(R(`OP_CLRSTATUS, 0, 0, 0, 12'd0));
            drain(2);

            // SOFTMAX x_base=line16, probs->TM[18..19], status word -> rC(=r7).
            issue(Iimm(`OP_LOADI, 5, 20'd16));
            issue(R(`OP_SOFTMAX, 5, 0, 7, 12'd18));   // rC=r7 gets STATUS word
            drain(10);                                 // let the tensor op retire to r7

            // The SOFTMAX retired its STATUS word into r7.  A multi-cycle tensor
            // op's WB pulse is NOT timing-alignable to a single bus read, so we
            // SURFACE the register through a scalar move (ADDI r9 = r7 + 0) issued
            // LAST -- that scalar WB lands at the deterministic 5-stage offset, so
            // reading RESULT immediately captures exactly the status word.  This
            // mirrors how a real host would read a tensor op's status reg.
            issue_then_read_result(R(`OP_ADDI, 7, 0, 9, 12'd0), rd);  // r9 = r7
            got_unit   = (rd >> `ST_UNIT_LO)   & ((1 << (`ST_UNIT_HI   - `ST_UNIT_LO   + 1)) - 1);
            got_argmax = (rd >> `ST_ARGMAX_LO) & ((1 << (`ST_ARGMAX_HI - `ST_ARGMAX_LO + 1)) - 1);
            chk_eq("SOFTMAX status UNIT field (over bus)", got_unit, `UNIT_SOFTMAX);
            chk_eq("SOFTMAX status ARGMAX field (over bus)", got_argmax, gargmax);

            // Core status after the SOFTMAX: UNIT field == UNIT_SOFTMAX and the
            // illegal bit clear (CLRSTATUS above + no illegal op since), all read
            // over the bus via RDSTATUS.
            read_core_status(rd_status);
            chk_eq("core status UNIT==SOFTMAX after op",
                   (rd_status >> `ST_UNIT_LO) &
                   ((1 << (`ST_UNIT_HI - `ST_UNIT_LO + 1)) - 1), `UNIT_SOFTMAX);
            chk_eq("core status illegal clear after SOFTMAX",
                   (rd_status >> `ST_ILLEGAL_BIT) & 1, 0);
        end

        //====================================================================
        // TEST 6 -- GEMM tensor op end-to-end over the bus.  Seed two 4x4 Q7.8
        //   matrices via LOADI/STORE/TLOAD, run GEMM, and check the returned
        //   STATUS WORD's UNIT field == UNIT_GEMM over RESULT.  (The full product
        //   tile is checked bit-exactly by the system TB; here we prove the bus
        //   launches the systolic op and its status retires over AXI.)
        //====================================================================
        $display("[AXI TEST 6] GEMM tensor op launched + status retired over the bus");
        begin : gemm_test
            integer r, c;
            integer A [0:3][0:3];
            integer B [0:3][0:3];
            integer got_unit;
            integer dbase, addr16, tline;
            // Simple operands (identity-ish): A = I*1.0, B arbitrary.  Status UNIT
            // is independent of the data, so any operands prove the launch.
            for (r = 0; r < 4; r = r + 1)
                for (c = 0; c < 4; c = c + 1) begin
                    A[r][c] = (r == c) ? 256 : 0;          // 1.0 on diagonal (Q7.8)
                    B[r][c] = ((r + c) % 4) * 64;          // arbitrary small values
                end
            // Seed A -> DMEM[0..15] -> TM[0..3].
            for (r = 0; r < 4; r = r + 1) begin
                for (c = 0; c < 4; c = c + 1) begin
                    addr16 = r*4 + c;
                    issue(Iimm(`OP_LOADI, 14, {{4{A[r][c][15]}}, A[r][c][15:0]}));
                    issue(Iimm(`OP_LOADI, 12, {12'd0, addr16[7:0]}));
                    issue(R(`OP_STORE, 12, 14, 0, 12'd0));
                end
                dbase = r*4;
                issue(Iimm(`OP_LOADI, 3, {12'd0, dbase[7:0]}));
                issue(R(`OP_TLOAD, 3, 0, 0, {7'd0, r[4:0]}));   // TM[r] = A row r
            end
            // Seed B -> DMEM[16..31] -> TM[4..7].
            for (r = 0; r < 4; r = r + 1) begin
                for (c = 0; c < 4; c = c + 1) begin
                    addr16 = 16 + r*4 + c;
                    issue(Iimm(`OP_LOADI, 14, {{4{B[r][c][15]}}, B[r][c][15:0]}));
                    issue(Iimm(`OP_LOADI, 12, {12'd0, addr16[7:0]}));
                    issue(R(`OP_STORE, 12, 14, 0, 12'd0));
                end
                dbase = 16 + r*4;
                tline = 4 + r;
                issue(Iimm(`OP_LOADI, 3, {12'd0, dbase[7:0]}));
                issue(R(`OP_TLOAD, 3, 0, 0, {7'd0, tline[4:0]}));  // TM[4+r] = B row
            end
            drain(4);
            // GEMM: A base line 0, B base line 4, C base imm12=8, status -> r7.
            issue(Iimm(`OP_LOADI, 5, 20'd0));
            issue(Iimm(`OP_LOADI, 6, 20'd4));
            issue(R(`OP_GEMM, 5, 6, 7, 12'd8));   // C->TM[8..11], status->r7
            drain(10);                             // let the systolic op retire to r7
            // Surface the GEMM status register via a scalar move (see TEST 5).
            issue_then_read_result(R(`OP_ADDI, 7, 0, 9, 12'd0), rd);  // r9 = r7
            got_unit = (rd >> `ST_UNIT_LO) & ((1 << (`ST_UNIT_HI - `ST_UNIT_LO + 1)) - 1);
            chk_eq("GEMM status UNIT field (over bus)", got_unit, `UNIT_GEMM);
        end

        //====================================================================
        // TEST 7 -- UNMAPPED address returns SLVERR on BOTH read and write, and
        //   an unmapped read returns 0 data.  Offset 0x10 is past the map.
        //====================================================================
        $display("[AXI TEST 7] unmapped address -> SLVERR (read + write)");
        // unmapped write
        axi_write_err(A_UNMAPPED, 32'h1234_5678);
        // unmapped read: expect SLVERR + zero data.  (axi_read asserts RRESP=OKAY,
        // so drive the read inline here to allow checking the error response.)
        begin : unmapped_rd
            reg ard;
            @(negedge ACLK);
            ARADDR = A_UNMAPPED; ARPROT=0; ARVALID=1'b1; RREADY=1'b1;
            ard=0;
            while (!ard) begin
                @(posedge ACLK);
                if (ARVALID && ARREADY) begin ard=1; cov_ar=cov_ar+1; end
                @(negedge ACLK);
                if (ard) ARVALID=0;
            end
            forever begin
                @(posedge ACLK);
                if (RVALID && RREADY) begin
                    cov_r=cov_r+1;
                    chk_eq("unmapped read SLVERR", (RRESP === R_SLVERR) ? 1 : 0, 1);
                    chk_eq("unmapped read data 0", RDATA, 0);
                    @(negedge ACLK); RREADY=0; disable unmapped_rd;
                end
            end
        end

        //====================================================================
        // COVERAGE -- the BFM must have actually driven every channel handshake.
        //====================================================================
        $display("[AXI COVERAGE] AW=%0d W=%0d B=%0d AR=%0d R=%0d",
                 cov_aw, cov_w, cov_b, cov_ar, cov_r);
        chk_eq("AW channel exercised", (cov_aw > 0) ? 1 : 0, 1);
        chk_eq("W  channel exercised", (cov_w  > 0) ? 1 : 0, 1);
        chk_eq("B  channel exercised", (cov_b  > 0) ? 1 : 0, 1);
        chk_eq("AR channel exercised", (cov_ar > 0) ? 1 : 0, 1);
        chk_eq("R  channel exercised", (cov_r  > 0) ? 1 : 0, 1);

        //====================================================================
        // SUMMARY
        //====================================================================
        if (fail == 0)
            $display("\nALL %0d TESTS PASSED", pass);
        else begin
            $display("\n%0d TESTS FAILED (of %0d)", fail, pass + fail);
            $fatal(1, "AXI BFM tests failed");
        end
        $finish;
    end

    // global timeout guard.
    initial begin
        #20000000;
        $display("FATAL: AXI TB timeout");
        $fatal(1, "timeout");
    end

endmodule
