`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// test/tpu_soc_tb.v  --  dual-clock, end-to-end autonomous-execution TB for the
//                        two-clock SoC top `tpu_soc` (src/tpu_soc.v)
//----------------------------------------------------------------------------
// WHAT THIS PROVES
//   The SoC, given ONLY a host descriptor over its AXI4-Lite SLAVE (program base
//   address + word count) and a START bit, AUTONOMOUSLY:
//       (1) commands its reused AXI4-Lite MASTER DMA to READ the program out of
//           an EXTERNAL memory that this TB models as an AXI4-Lite slave;
//       (2) streams the fetched instruction words across an ASYNCHRONOUS clock
//           boundary (ACLK -> CCLK) into the unchanged TPU core;
//       (3) its CCLK sequencer issues every instruction (honouring the core's
//           dbg_pipe_stall front-end hold), drains the pipeline, and signals
//           completion back to the host (CCLK -> ACLK) so STATUS.result_valid
//           rises and the host can read the RESULT register.
//   Nothing on the core side is poked by the TB: the host only ever touches the
//   AXI SLAVE port, and the program is delivered ONLY through the AXI MASTER
//   read path + the SoC's internal CDC FIFOs.
//
// TWO ASYNCHRONOUS CLOCKS
//   ACLK = 10.0 ns period, CCLK = 13.0 ns period.  The ratio 10:13 is irrational
//   in integer-cycle terms (LCM = 130 ns), so over a run the two clocks sample at
//   continually shifting phase relationships -- a genuine CDC stress.  The TB
//   asserts (a) the two periods are unequal and (b) the live phase between the
//   clocks actually drifts (it samples a non-trivial spread of CCLK-vs-ACLK edge
//   offsets), so the pass cannot be an artifact of accidentally-aligned clocks.
//
// INDEPENDENT GOLDEN (the crux)
//   Each program computes a known SCALAR via real instructions.  The golden value
//   for each program is computed HERE in the TB from first principles (plain
//   integer arithmetic over the program's operands) -- never read back from any
//   DUT internal.  Program 1: LOADI r1=5 ; ADDI r1 = r1 + 7  => 12.
//   Program 2: LOADI r2=100 ; LOADI r3=37 ; SUB r4 = r2 - r3 => 63.
//   The committed scalar (the architectural result the program produced) is read
//   back from the core register file AFTER the SoC reports the run complete, and
//   checked bit-exactly against that independent golden -- this is the load-
//   bearing proof that the fetched program actually RAN to a correct result.
//
//   The host-visible RESULT register is ALSO exercised end-to-end (host AR read
//   over the AXI slave, crossing the result CDC FIFO CCLK->ACLK) and is checked
//   against the SAME independent golden as the committed scalar: the SoC's
//   real-instruction-valid shadow captures the LAST real instruction's write-back
//   live (surviving the NOP drain tail), so RESULT now returns the program's true
//   result (12 for P1, 63 for P2).  This verifies the full host read path -- AR
//   handshake, FIFO pop, RDATA mux -- AND that the captured value is correct.
//
// REUSABILITY
//   After program 1 completes, the SoC is RE-ARMED with a DIFFERENT DMA_SRC and a
//   DIFFERENT DMA_LEN and run again with program 2, proving the SoC returns to a
//   clean idle and is reusable without a reset.
//
// COVERAGE / NEGATIVE GUARDS
//   * AXI MASTER coverage counters assert the DMA genuinely drove its READ
//     channels (AR handshakes and R handshakes both occurred, one pair per
//     fetched word) -- the program did NOT arrive by some side channel.
//   * The DMA error path stays clean (STATUS.err == 0) for every well-formed run.
//   * STATUS.dma_done and STATUS.result_valid both rise per run.
//
// Prints "ALL <N> TESTS PASSED"; $fatal on any mismatch.  A real VCD is dumped.
//============================================================================
module tpu_soc_tb;

    // ====================================================================
    // Parameters mirrored from the DUT defaults (NOT read from the DUT).
    // ====================================================================
    localparam integer ADDR_W   = 32;
    localparam integer SADDR_W  = 8;
    localparam integer DMA_LENW = 8;

    // Slave register byte offsets (must match the SoC register map).
    localparam [SADDR_W-1:0] REG_DMA_SRC  = 8'h00;
    localparam [SADDR_W-1:0] REG_DMA_LEN  = 8'h04;
    localparam [SADDR_W-1:0] REG_DMA_CTRL = 8'h08;
    localparam [SADDR_W-1:0] REG_RESULT   = 8'h0C;
    localparam [SADDR_W-1:0] REG_STATUS   = 8'h10;

    // STATUS bit positions.
    localparam integer ST_DMA_DONE = 0;
    localparam integer ST_RES_VLD  = 1;
    localparam integer ST_ERR      = 2;

    // The SoC presents an all-zero (OP_NOP) word during its DRAIN_NOPS tail, but a
    // "real-instruction valid" shadow shift register inside the CCLK sequencer
    // captures core_result LIVE the cycle each REAL instruction reaches write-back.
    // The trailing NOPs shift in valid=0 and never overwrite that capture, so the
    // host-visible RESULT now returns the LAST real instruction's actual write-back
    // value -- the program's true result.  RESULT is therefore checked against the
    // SAME independent golden (computed in the TB from the program operands) as the
    // committed register-file scalar, exercising the whole host read path end to end
    // (AR handshake -> result-FIFO pop CCLK->ACLK -> RDATA mux).

    // ====================================================================
    // Two asynchronous clocks + independent active-low resets.
    // ====================================================================
    real ACLK_HALF = 5.0;    // ACLK period 10.0 ns
    real CCLK_HALF = 6.5;    // CCLK period 13.0 ns

    reg ACLK = 1'b0;
    reg CCLK = 1'b0;
    reg ARESETn = 1'b0;
    reg CRESETn = 1'b0;

    always #(ACLK_HALF) ACLK = ~ACLK;
    always #(CCLK_HALF) CCLK = ~CCLK;

    // ====================================================================
    // SoC port nets.
    // ====================================================================
    // ---- AXI4-Lite SLAVE (TB drives as host) ----
    reg  [SADDR_W-1:0] S_AWADDR;
    reg  [2:0]         S_AWPROT;
    reg                S_AWVALID;
    wire               S_AWREADY;
    reg  [31:0]        S_WDATA;
    reg  [3:0]         S_WSTRB;
    reg                S_WVALID;
    wire               S_WREADY;
    wire [1:0]         S_BRESP;
    wire               S_BVALID;
    reg                S_BREADY;
    reg  [SADDR_W-1:0] S_ARADDR;
    reg  [2:0]         S_ARPROT;
    reg                S_ARVALID;
    wire               S_ARREADY;
    wire [31:0]        S_RDATA;
    wire [1:0]         S_RRESP;
    wire               S_RVALID;
    reg                S_RREADY;

    // ---- AXI4-Lite MASTER (TB models external memory as the slave) ----
    wire [ADDR_W-1:0]  M_AWADDR;
    wire [2:0]         M_AWPROT;
    wire               M_AWVALID;
    reg                M_AWREADY;
    wire [31:0]        M_WDATA;
    wire [3:0]         M_WSTRB;
    wire               M_WVALID;
    reg                M_WREADY;
    reg  [1:0]         M_BRESP;
    reg                M_BVALID;
    wire               M_BREADY;
    wire [ADDR_W-1:0]  M_ARADDR;
    wire [2:0]         M_ARPROT;
    wire               M_ARVALID;
    reg                M_ARREADY;
    reg  [31:0]        M_RDATA;
    reg  [1:0]         M_RRESP;
    reg                M_RVALID;
    wire               M_RREADY;

    // ====================================================================
    // DUT.
    // ====================================================================
    tpu_soc dut (
        .ACLK(ACLK), .ARESETn(ARESETn),
        .S_AWADDR(S_AWADDR), .S_AWPROT(S_AWPROT), .S_AWVALID(S_AWVALID), .S_AWREADY(S_AWREADY),
        .S_WDATA(S_WDATA), .S_WSTRB(S_WSTRB), .S_WVALID(S_WVALID), .S_WREADY(S_WREADY),
        .S_BRESP(S_BRESP), .S_BVALID(S_BVALID), .S_BREADY(S_BREADY),
        .S_ARADDR(S_ARADDR), .S_ARPROT(S_ARPROT), .S_ARVALID(S_ARVALID), .S_ARREADY(S_ARREADY),
        .S_RDATA(S_RDATA), .S_RRESP(S_RRESP), .S_RVALID(S_RVALID), .S_RREADY(S_RREADY),
        .M_AWADDR(M_AWADDR), .M_AWPROT(M_AWPROT), .M_AWVALID(M_AWVALID), .M_AWREADY(M_AWREADY),
        .M_WDATA(M_WDATA), .M_WSTRB(M_WSTRB), .M_WVALID(M_WVALID), .M_WREADY(M_WREADY),
        .M_BRESP(M_BRESP), .M_BVALID(M_BVALID), .M_BREADY(M_BREADY),
        .M_ARADDR(M_ARADDR), .M_ARPROT(M_ARPROT), .M_ARVALID(M_ARVALID), .M_ARREADY(M_ARREADY),
        .M_RDATA(M_RDATA), .M_RRESP(M_RRESP), .M_RVALID(M_RVALID), .M_RREADY(M_RREADY),
        .CCLK(CCLK), .CRESETn(CRESETn)
    );

    // ====================================================================
    // EXTERNAL MEMORY model: an AXI4-Lite SLAVE (READ path is what the DMA uses).
    //   256 words, word-addressed by ARADDR[9:2].  Single outstanding read with a
    //   registered-READY handshake.  The WRITE channels are accepted-and-OKAY'd
    //   (the SoC never issues a master WRITE in READ mode, but a well-behaved
    //   slave must not wedge if one ever appeared).
    // ====================================================================
    localparam integer XMEM_WORDS = 256;
    reg [31:0] xmem [0:XMEM_WORDS-1];

    // READ-channel coverage + handshake state.
    integer cov_ar_hs;        // AR handshakes observed (one per fetched word)
    integer cov_r_hs;         // R  handshakes observed (one per fetched word)
    integer cov_aw_hs;        // AW handshakes (expected 0 in READ mode)
    reg [ADDR_W-1:0] xmem_araddr_lat;
    reg              xmem_rpend;     // an accepted AR awaits its R data

    always @(posedge ACLK) begin
        if (!ARESETn) begin
            M_ARREADY      <= 1'b0;
            M_RVALID       <= 1'b0;
            M_RDATA        <= 32'b0;
            M_RRESP        <= 2'b00;
            xmem_rpend     <= 1'b0;
            xmem_araddr_lat<= {ADDR_W{1'b0}};
            // write side
            M_AWREADY      <= 1'b0;
            M_WREADY       <= 1'b0;
            M_BVALID       <= 1'b0;
            M_BRESP        <= 2'b00;
            // coverage
            cov_ar_hs      <= 0;
            cov_r_hs       <= 0;
            cov_aw_hs      <= 0;
        end else begin
            // ---- READ address: accept one AR when idle ----
            M_ARREADY <= 1'b0;
            if (M_ARVALID && !M_ARREADY && !xmem_rpend && !M_RVALID) begin
                M_ARREADY       <= 1'b1;
                xmem_araddr_lat <= M_ARADDR;
                xmem_rpend      <= 1'b1;
                cov_ar_hs       <= cov_ar_hs + 1;
            end
            // ---- READ data: drive R the cycle after AR, hold until accepted ----
            if (xmem_rpend && !M_RVALID) begin
                M_RDATA  <= xmem[xmem_araddr_lat[9:2]];
                M_RRESP  <= 2'b00;                 // OKAY
                M_RVALID <= 1'b1;
                xmem_rpend <= 1'b0;
            end else if (M_RVALID && M_RREADY) begin
                M_RVALID <= 1'b0;
                cov_r_hs <= cov_r_hs + 1;
            end

            // ---- WRITE side: benign accept-and-OKAY (never used in READ mode) ---
            M_AWREADY <= 1'b0;
            M_WREADY  <= 1'b0;
            if (M_AWVALID && !M_AWREADY) begin M_AWREADY <= 1'b1; cov_aw_hs <= cov_aw_hs + 1; end
            if (M_WVALID  && !M_WREADY ) M_WREADY  <= 1'b1;
            if (M_AWVALID && M_WVALID && !M_BVALID) begin M_BVALID <= 1'b1; M_BRESP <= 2'b00; end
            else if (M_BVALID && M_BREADY) M_BVALID <= 1'b0;
        end
    end

    // ====================================================================
    // Instruction encoders -- identical convention to test/tpu_tb.v / tpu_defs.vh.
    // ====================================================================
    function [31:0] Iimm;                 // I-format (LOADI): op|rC|imm20
        input [7:0] op; input [3:0] c; input [19:0] im;
        Iimm = {op, c, im};
    endfunction
    function [31:0] R;                    // R-format: op|rA|rB|rC|imm12
        input [7:0] op; input [3:0] a, b, c; input [11:0] im;
        R = {op, a, b, c, im};
    endfunction

    // ====================================================================
    // HOST AXI4-Lite SLAVE-side bus driver tasks (drive the SoC slave port).
    //   These are the ONLY way the TB touches the SoC -- the host model.
    // ====================================================================
    // Single 32-bit write: independent AW/W handshakes, then B retire.
    task host_write;
        input [SADDR_W-1:0] addr;
        input [31:0]        data;
        begin
            @(posedge ACLK);
            S_AWADDR  <= addr;  S_AWVALID <= 1'b1;
            S_WDATA   <= data;  S_WVALID  <= 1'b1;
            S_WSTRB   <= 4'hF;  S_BREADY  <= 1'b1;
            // wait for AW accept
            @(posedge ACLK);
            while (!S_AWREADY) @(posedge ACLK);
            S_AWVALID <= 1'b0;
            // wait for W accept (may be same cycle as AW or later)
            while (!S_WREADY) @(posedge ACLK);
            S_WVALID <= 1'b0;
            // wait for B response then retire
            while (!S_BVALID) @(posedge ACLK);
            @(posedge ACLK);
            S_BREADY <= 1'b0;
        end
    endtask

    // Single 32-bit read: AR handshake, then capture RDATA at R handshake.
    task host_read;
        input  [SADDR_W-1:0] addr;
        output [31:0]        data;
        begin
            @(posedge ACLK);
            S_ARADDR  <= addr;  S_ARVALID <= 1'b1;  S_RREADY <= 1'b1;
            @(posedge ACLK);
            while (!S_ARREADY) @(posedge ACLK);
            S_ARVALID <= 1'b0;
            while (!S_RVALID) @(posedge ACLK);
            data = S_RDATA;
            @(posedge ACLK);
            S_RREADY <= 1'b0;
        end
    endtask

    // Poll STATUS until result_valid (bit1) is set, with a bounded timeout.
    task host_wait_result;
        output [31:0] status;
        integer guard;
        reg [31:0] s;
        begin
            status = 32'b0;
            for (guard = 0; guard < 5000; guard = guard + 1) begin
                host_read(REG_STATUS, s);
                if (s[ST_RES_VLD]) begin
                    status = s;
                    guard  = 6000;   // break
                end
            end
            if (status[ST_RES_VLD] !== 1'b1) begin
                $display("  FAIL: timed out waiting for STATUS.result_valid");
                $fatal(1, "result_valid timeout");
            end
        end
    endtask

    // ====================================================================
    // Bookkeeping + checkers.
    // ====================================================================
    integer pass = 0, fail = 0;

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
                $fatal(1, "tpu_soc TB mismatch");
            end
        end
    endtask

    task chk_true;
        input [511:0] name;
        input         cond;
        begin
            if (cond === 1'b1) begin
                pass = pass + 1;
                $display("  PASS %0s", name);
            end else begin
                fail = fail + 1;
                $display("  FAIL %0s (condition false)", name);
                $fatal(1, "tpu_soc TB condition failed");
            end
        end
    endtask

    // ====================================================================
    // Asynchronous-clock witness.  Records the time-offset of each CCLK posedge
    // relative to the most recent ACLK posedge; an unrelated clock pair produces
    // a WIDE spread of offsets, an accidentally-related pair a narrow/locked one.
    // ====================================================================
    real last_aclk_edge;
    real phase_min, phase_max;
    integer phase_samples;
    real off;

    always @(posedge ACLK) last_aclk_edge = $realtime;
    always @(posedge CCLK) begin
        if (last_aclk_edge >= 0.0) begin
            off = $realtime - last_aclk_edge;     // 0 .. (ACLK period)
            if (phase_samples == 0) begin
                phase_min = off; phase_max = off;
            end else begin
                if (off < phase_min) phase_min = off;
                if (off > phase_max) phase_max = off;
            end
            phase_samples = phase_samples + 1;
        end
    end

    // ====================================================================
    // Run one program already resident in xmem and check its committed scalar.
    //   src_word : program base in xmem in WORDS (byte addr = src_word*4)
    //   len      : program length in words
    //   reg_idx  : the architectural register the program's result lands in
    //   golden   : INDEPENDENT expected value of that register (computed in TB)
    //   tag      : label
    // Records pre-run master coverage so per-run AR/R handshake deltas are checked.
    // ====================================================================
    task run_program_check;
        input [31:0]          src_word;
        input [DMA_LENW-1:0]  len;
        input integer         reg_idx;
        input integer         golden;
        input [127:0]         tag;
        integer ar0, r0;
        reg [31:0] st, res;
        begin
            ar0 = cov_ar_hs;
            r0  = cov_r_hs;

            // ---- host programs the descriptor + kicks START over the AXI slave --
            host_write(REG_DMA_SRC,  src_word << 2);     // byte base address
            host_write(REG_DMA_LEN,  {{(32-DMA_LENW){1'b0}}, len});
            host_write(REG_DMA_CTRL, 32'h0000_0001);     // bit0 START

            // ---- wait (host-visible) for the autonomous run to complete ---------
            host_wait_result(st);

            $display("  [%0s] STATUS=0x%08h (dma_done=%0b result_valid=%0b err=%0b)",
                     tag, st, st[ST_DMA_DONE], st[ST_RES_VLD], st[ST_ERR]);

            // ---- host-visible status assertions ---------------------------------
            chk_true({tag, " STATUS.dma_done set"},     st[ST_DMA_DONE]);
            chk_true({tag, " STATUS.result_valid set"}, st[ST_RES_VLD]);
            chk_eq  ({tag, " STATUS.err clear"}, st[ST_ERR], 0);

            // ---- host reads the RESULT register (full CCLK->ACLK read path) ------
            //      RESULT now returns the LAST real instruction's actual write-back
            //      (captured live by the sequencer's real-instruction shadow), so it
            //      must equal the program's INDEPENDENT golden -- not 0.
            host_read(REG_RESULT, res);
            chk_eq({tag, " RESULT reg == golden"}, res, golden);

            // ---- AXI MASTER coverage for THIS run (one AR+R pair per word) -------
            chk_eq({tag, " DMA AR handshakes == len"}, cov_ar_hs - ar0, len);
            chk_eq({tag, " DMA R  handshakes == len"}, cov_r_hs  - r0,  len);

            // ---- THE LOAD-BEARING CHECK: the committed scalar == INDEPENDENT -----
            //      golden.  This is the architectural result the autonomously-
            //      fetched program produced; read back from the core register file
            //      only AFTER the SoC reported the run complete.
            chk_eq({tag, " committed scalar == golden"},
                   dut.u_core.regfile.registers[reg_idx], golden);
        end
    endtask

    // ====================================================================
    // Independent goldens (plain integer arithmetic over program operands).
    // ====================================================================
    integer p1_golden;   // r1 = 5 + 7
    integer p2_golden;   // r4 = 100 - 37

    integer i;

    // ====================================================================
    // Main stimulus.
    // ====================================================================
    initial begin
        $dumpfile("tpu_soc_waveform.vcd");
        $dumpvars(0, tpu_soc_tb);

        // bus idle
        S_AWADDR=0; S_AWPROT=0; S_AWVALID=0;
        S_WDATA=0;  S_WSTRB=4'hF; S_WVALID=0;
        S_BREADY=0;
        S_ARADDR=0; S_ARPROT=0; S_ARVALID=0; S_RREADY=0;

        // async-clock witness init
        last_aclk_edge = -1.0;
        phase_min = 0.0; phase_max = 0.0; phase_samples = 0;

        // clear external memory
        for (i = 0; i < XMEM_WORDS; i = i + 1) xmem[i] = 32'b0;

        // ---- PROGRAM 1 @ word 0 : LOADI r1=5 ; ADDI r1 = r1 + 7  => 12 ----------
        xmem[0] = Iimm(`OP_LOADI, 4'd1, 20'd5);          // r1 = 5
        xmem[1] = R(`OP_ADDI, 4'd1, 4'd0, 4'd1, 12'd7);  // r1 = r1 + 7 = 12
        p1_golden = 5 + 7;                                // independent: 12

        // ---- PROGRAM 2 @ word 8 : LOADI r2=100 ; LOADI r3=37 ; SUB r4=r2-r3 => 63
        xmem[8]  = Iimm(`OP_LOADI, 4'd2, 20'd100);                 // r2 = 100
        xmem[9]  = Iimm(`OP_LOADI, 4'd3, 20'd37);                  // r3 = 37
        xmem[10] = R(`OP_SUB, 4'd2, 4'd3, 4'd4, 12'd0);            // r4 = r2 - r3 = 63
        p2_golden = 100 - 37;                                      // independent: 63

        // ---- independent, asynchronous resets (different lengths/clocks) --------
        ARESETn = 1'b0; CRESETn = 1'b0;
        repeat (6) @(posedge ACLK); ARESETn = 1'b1;
        repeat (6) @(posedge CCLK); CRESETn = 1'b1;
        repeat (4) @(posedge ACLK);

        $display("[tpu_soc TB] two async clocks: ACLK=%.1f ns, CCLK=%.1f ns",
                 2.0*ACLK_HALF, 2.0*CCLK_HALF);

        // ================= PROGRAM 1 =================
        $display("\n[PROGRAM 1] LOADI r1=5 ; ADDI r1=r1+7  (golden r1 = %0d)", p1_golden);
        run_program_check(32'd0, 8'd2, 1, p1_golden, "P1");

        // ================= PROGRAM 2 (RE-ARM, different SRC + LEN) =================
        $display("\n[PROGRAM 2] LOADI r2=100 ; LOADI r3=37 ; SUB r4=r2-r3  (golden r4 = %0d)", p2_golden);
        run_program_check(32'd8, 8'd3, 4, p2_golden, "P2");

        // ================= AXI MASTER drove its channels (whole run) =================
        $display("\n[COVERAGE] AXI master + clock asynchrony");
        chk_true("master AR driven (>0 hs)", (cov_ar_hs > 0));
        chk_true("master R  driven (>0 hs)", (cov_r_hs  > 0));
        chk_eq  ("master AR == R (read-mode)", cov_ar_hs, cov_r_hs);
        chk_eq  ("master AR == words (2+3)", cov_ar_hs, 5);
        // READ-mode DMA must never issue a master WRITE.
        chk_eq  ("master NO writes (READ mode)", cov_aw_hs, 0);

        // ================= the two clocks are genuinely unrelated =================
        // Different periods...
        chk_true("ACLK and CCLK periods differ", (ACLK_HALF != CCLK_HALF));
        // ...and the live CCLK-vs-ACLK phase actually DRIFTS across a wide span
        // (an aligned/locked pair would show a near-zero spread).  Spread is in ns;
        // for 10 ns ACLK with a 13 ns CCLK the offset sweeps most of the period.
        $display("  phase offset spread: min=%.2f ns max=%.2f ns over %0d samples",
                 phase_min, phase_max, phase_samples);
        chk_true("CCLK/ACLK phase drift > 3 ns",
                 ((phase_max - phase_min) > 3.0));

        // ================= summary =================
        if (fail == 0)
            $display("\nALL %0d TESTS PASSED", pass);
        else begin
            $display("\n%0d TESTS FAILED (of %0d)", fail, pass + fail);
            $fatal(1, "tpu_soc integration tests failed");
        end
        $finish;
    end

    // Global timeout guard.
    initial begin
        #5_000_000;
        $display("FATAL: tpu_soc TB timeout");
        $fatal(1, "timeout");
    end

endmodule
