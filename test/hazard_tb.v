`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// test/hazard_tb.v  --  TPU v2.0 focused hazard / pipeline testbench
//----------------------------------------------------------------------------
// Self-checking, program-driven tests of the pipeline's HAZARD machinery in the
// integrated core (src/tpu_top.v module TPU).  Every program-visible value is
// seeded by REAL INSTRUCTIONS (LOADI etc.); the only hierarchical accesses are
// read-back peeks for CHECKING and a couple of stall-signal observations to
// PROVE a stall actually fired (timing assertions).
//
// HAZARDS COVERED (SPEC §3, docs/ISA.md §5):
//   H1  back-to-back scalar RAW  -> EX/EX-MEM/MEM-WB FORWARDING (no manual NOPs)
//   H2  LOAD-use stall            -> a LOAD producer followed by a dependent op
//                                    inserts exactly one bubble (observed) and
//                                    still computes the correct value
//   H3  tensor BUSY stall         -> a GEMM immediately followed by a dependent
//                                    RDSTATUS reads the GEMM's status (the busy
//                                    stall serialized them; no manual spacing),
//                                    and a TLOAD/GEMM->TSTORE chain round-trips
//   H4  illegal opcode            -> illegal_opcode flag + sticky status bit +
//                                    safe NOP (no register write)
//   H5  r0 write ignored          -> a LOADI to r0 leaves r0 == 0
//   H6  LOADI seeding             -> the program seeds RF with no TB pokes
//
// Self-checking; prints "ALL N TESTS PASSED"; $fatal on any failure.
//============================================================================
module hazard_tb;

    reg         clk, rst;
    reg  [31:0] instr;
    wire [31:0] result_out;
    wire        illegal_opcode;
    wire [`ST_W-1:0] dbg_status;
    wire        dbg_pipe_stall;

    TPU dut (
        .clk(clk), .rst(rst),
        .instruction_in(instr),
        .result_out(result_out),
        .illegal_opcode(illegal_opcode),
        .dbg_status(dbg_status),
        .dbg_pipe_stall(dbg_pipe_stall)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    integer pass, fail;
    integer illegal_seen;
    // H7: capture the o_base the ATTENTION unit latches at dispatch (the value it
    // will write its context tiles to).  Sampled on the attention start pulse.
    reg [`TM_IDX_W-1:0] attn_obase;
    always @(posedge clk)
        if (dut.attn_start) attn_obase <= dut.opC_ex[`TM_IDX_W-1:0];
    integer stall_cycles;      // counts cycles dbg_pipe_stall was high during step

    function [31:0] R;
        input [7:0] op; input [3:0] a, b, c; input [11:0] im;
        R = {op, a, b, c, im};
    endfunction
    function [31:0] Iimm;
        input [7:0] op; input [3:0] c; input [19:0] im;
        Iimm = {op, c, im};
    endfunction

    // step: present one instruction, hold while the front is stalled, and COUNT
    // how many extra stall cycles it incurred (for hazard timing assertions).
    task step;
        input [31:0] ins;
        begin
            @(negedge clk); instr = ins;
            @(posedge clk);
            if (illegal_opcode) illegal_seen = 1;
            while (dbg_pipe_stall === 1'b1) begin
                stall_cycles = stall_cycles + 1;
                @(negedge clk);
                @(posedge clk);
            end
        end
    endtask

    task drain;
        input integer n;
        integer k;
        begin
            for (k = 0; k < n; k = k + 1) step(R(`OP_NOP, 0, 0, 0, 0));
            for (k = 0; k < n; k = k + 1) @(posedge clk);
        end
    endtask

    task store16;
        input [7:0] addr; input signed [15:0] val;
        begin
            step(Iimm(`OP_LOADI, 14, {{4{val[15]}}, val}));
            step(Iimm(`OP_LOADI, 12, {12'd0, addr}));
            step(R(`OP_STORE, 12, 14, 0, 0));
        end
    endtask

    task tload;
        input [7:0] base; input [`TM_IDX_W-1:0] line;
        begin
            step(Iimm(`OP_LOADI, 3, {12'd0, base}));
            step(R(`OP_TLOAD, 3, 0, 0, {7'd0, line}));
        end
    endtask

    task chk_eq;
        input [255:0] name; input integer got, exp;
        begin
            if (got === exp) begin
                pass = pass + 1;
                $display("  PASS %0s : %0d", name, got);
            end else begin
                fail = fail + 1;
                $display("  FAIL %0s : got=%0d exp=%0d", name, got, exp);
                $fatal(1, "hazard test mismatch");
            end
        end
    endtask
    task chk_ge;
        input [255:0] name; input integer got, lo;
        begin
            if (got >= lo) begin
                pass = pass + 1;
                $display("  PASS %0s : %0d (>=%0d)", name, got, lo);
            end else begin
                fail = fail + 1;
                $display("  FAIL %0s : got=%0d expected >= %0d", name, got, lo);
                $fatal(1, "hazard timing assertion failed");
            end
        end
    endtask

    function signed [15:0] tm_lane;
        input [`TM_IDX_W-1:0] line; input integer c;
        begin tm_lane = dut.tmem.lines[line][c*`LANE_W +: `ELEM_W]; end
    endfunction

    integer i, j;
    reg signed [15:0] Cgold;
    real racc;

    initial begin
        $dumpfile("hazard_waveform.vcd");
        $dumpvars(0, hazard_tb);

        pass = 0; fail = 0; illegal_seen = 0;

        instr = R(`OP_NOP, 0, 0, 0, 0);
        rst = 1'b1;
        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        //====================================================================
        // H6 + H1 : LOADI seeding and back-to-back scalar RAW forwarding.
        //   A long dependent chain with NO manual NOP spacing -- if forwarding
        //   were broken these would read stale RF values.
        //====================================================================
        $display("[H1/H6] LOADI seed + back-to-back RAW forwarding");
        step(Iimm(`OP_LOADI, 1, 20'd3));         // r1 = 3
        step(Iimm(`OP_LOADI, 2, 20'd4));         // r2 = 4
        step(R(`OP_ADD, 1, 2, 3, 0));            // r3 = r1+r2 = 7      (fwd both)
        step(R(`OP_ADD, 3, 1, 4, 0));            // r4 = r3+r1 = 10     (fwd r3 EX)
        step(R(`OP_SUB, 4, 3, 5, 0));            // r5 = r4-r3 = 3      (fwd r4 EX, r3 EXMEM)
        step(R(`OP_ADD, 5, 4, 6, 0));            // r6 = r5+r4 = 13
        step(R(`OP_XOR, 6, 5, 7, 0));            // r7 = 13^3 = 14
        drain(6);
        chk_eq("RAW r3", dut.regfile.registers[3], 7);
        chk_eq("RAW r4 (EX fwd)", dut.regfile.registers[4], 10);
        chk_eq("RAW r5 (EX+EXMEM fwd)", dut.regfile.registers[5], 3);
        chk_eq("RAW r6", dut.regfile.registers[6], 13);
        chk_eq("RAW r7", dut.regfile.registers[7], 14);

        //====================================================================
        // H2 : LOAD-use stall.  A LOAD producer immediately followed by a
        //   dependent ADD must insert >=1 bubble (observed via dbg_pipe_stall)
        //   AND still produce the correct value (the dependent op reads the
        //   load result via the post-stall forward, not a stale register).
        //====================================================================
        $display("[H2] LOAD-use stall");
        store16(50, 16'sd77);                    // DMEM[50] = 77
        step(Iimm(`OP_LOADI, 1, 20'd50));        // r1 = 50 (address)
        stall_cycles = 0;
        step(R(`OP_LOAD, 1, 0, 8, 0));           // r8 = DMEM[50] = 77 (LOAD producer)
        step(R(`OP_ADD, 8, 0, 9, 0));            // r9 = r8 + 0 = 77   (load-use!)
        drain(6);
        chk_ge("LOAD-use inserted a stall", stall_cycles, 1);
        chk_eq("LOAD-use result correct", dut.regfile.registers[9], 77);

        // Control: a LOAD followed by an INDEPENDENT op must NOT stall.
        $display("[H2b] no false stall on independent op after LOAD");
        store16(51, 16'sd55);
        step(Iimm(`OP_LOADI, 1, 20'd51));
        stall_cycles = 0;
        step(R(`OP_LOAD, 1, 0, 8, 0));           // r8 = DMEM[51]
        step(R(`OP_ADD, 2, 2, 10, 0));           // r10 = r2+r2 (independent of r8)
        drain(6);
        chk_eq("no stall on independent op", stall_cycles, 0);
        chk_eq("LOAD value still correct", dut.regfile.registers[8], 55);

        //====================================================================
        // H3 : TENSOR busy stall.  A GEMM immediately followed by a dependent
        //   RDSTATUS must read the GEMM's status word -- the busy stall has to
        //   serialize them with NO manual spacing, and the GEMM must have
        //   incurred a multi-cycle stall (observed).
        //====================================================================
        $display("[H3] tensor busy stall + dependent RDSTATUS");
        // Seed a tiny 4x4 GEMM (identity A * B) into TM[0..7].
        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1) begin
                store16(i[7:0]*4 + j[7:0], (i == j) ? 16'sd256 : 16'sd0);       // A=I
                store16(16 + i[7:0]*4 + j[7:0], ((i*4 + j + 1)) * 16'sd256);    // B
            end
        for (i = 0; i < 4; i = i + 1) tload(i[7:0]*4, i[`TM_IDX_W-1:0]);        // A->TM0..3
        for (i = 0; i < 4; i = i + 1) tload(16 + i[7:0]*4, (4 + i[`TM_IDX_W-1:0])); // B->TM4..7
        // golden C = B (A is identity).
        step(Iimm(`OP_LOADI, 5, 20'd0));
        step(Iimm(`OP_LOADI, 6, 20'd4));
        stall_cycles = 0;
        step(R(`OP_GEMM, 5, 6, 7, 12'd8));       // C -> TM[8..11], status -> r7
        // Immediately a DEPENDENT RDSTATUS (no spacing): reads the status the
        // GEMM wrote.  Because the busy stall held the pipeline, RDSTATUS sees
        // the post-GEMM status register.
        step(R(`OP_RDSTATUS, 0, 0, 11, 0));      // r11 = status
        drain(6);
        chk_ge("GEMM busy stall was multi-cycle", stall_cycles, 8);
        // status unit field must be GEMM (the busy stall serialized them).
        chk_eq("RDSTATUS sees GEMM unit",
               (dut.regfile.registers[11] >> `ST_UNIT_LO) &
               ((1 << (`ST_UNIT_HI - `ST_UNIT_LO + 1)) - 1), `UNIT_GEMM);
        // And the GEMM result actually landed in TM (= B since A=I).
        chk_eq("GEMM result in TM[8] lane0", tm_lane(5'd8, 0), 16'sd256);
        chk_eq("GEMM result in TM[11] lane3", tm_lane(5'd11, 3), 16'sd4096);

        //====================================================================
        // H3b : tensor op immediately followed by a dependent TLOAD/TSTORE
        //   round-trip (read the TM result the tensor op produced, the cycle
        //   after done, through the scalar TM port -- the busy stall + drain
        //   serialization must make TM[8..11] visible).
        //====================================================================
        $display("[H3b] tensor result -> dependent TSTORE round-trip");
        step(Iimm(`OP_LOADI, 4, 20'd100));
        step(R(`OP_TSTORE, 4, 0, 0, 12'd8));     // DMEM[100..103] = TM[8] (= B row0)
        drain(4);
        // B row 0 = {256,512,768,1024} (lane low16).
        chk_eq("TSTORE TM[8] w0", $signed(dut.u_dmem.sram[100][15:0]), 256);
        chk_eq("TSTORE TM[8] w3", $signed(dut.u_dmem.sram[103][15:0]), 1024);

        //====================================================================
        // H4 : illegal opcode -> flag + sticky status + safe NOP.
        //====================================================================
        $display("[H4] illegal opcode -> flag + safe NOP");
        step(Iimm(`OP_LOADI, 8, 20'd321));       // r8 = 321
        illegal_seen = 0;
        step(R(8'h7F, 8, 9, 8, 12'h111));        // illegal: must NOT write r8
        drain(4);
        chk_eq("illegal flagged", illegal_seen, 1);
        chk_eq("illegal safe-NOP (r8 kept)", dut.regfile.registers[8], 321);
        step(R(`OP_RDSTATUS, 0, 0, 9, 0));
        drain(4);
        chk_eq("illegal sticky bit set",
               (dut.regfile.registers[9] >> `ST_ILLEGAL_BIT) & 1, 1);

        //====================================================================
        // H5 : r0 hardwired zero -- write ignored, read returns 0.
        //====================================================================
        $display("[H5] r0 write ignored");
        step(Iimm(`OP_LOADI, 0, 20'd12345));     // attempt to write r0
        drain(4);
        chk_eq("r0 stays 0 after LOADI r0", dut.regfile.registers[0], 0);
        // r0 used as a source reads 0: r10 = r0 + 5-imm via ADDI.
        step(R(`OP_ADDI, 0, 0, 10, 12'd5));      // r10 = r0 + 5 = 5
        drain(4);
        chk_eq("r0 reads as 0 (ADDI r0+5)", dut.regfile.registers[10], 5);

        //====================================================================
        // H7 : ATTENTION o_base (rC source) load-use stall.  A LOAD that writes
        //   rC immediately followed by an ATTENTION that uses rC as its o_base
        //   MUST insert a bubble (rC is a SOURCE for ATTENTION), so ATTENTION
        //   latches the LOADed o_base, not the stale register value.  Without the
        //   stall ATTENTION would write its 4 context lines to the WRONG TM region.
        //   Observed via the attention unit's latched o_base_q at start.
        //====================================================================
        $display("[H7] ATTENTION o_base load-use stall");
        // Seed minimal Q/K/V tiles so ATTENTION runs (values irrelevant to the
        // o_base hazard; we check WHERE it would write, via o_base_q).
        for (i = 0; i < 4; i = i + 1) begin
            store16(i[7:0]*4,      (i == 0) ? 16'sd256 : 16'sd0);   // Q seed word
            store16(40 + i[7:0]*4, (i == 0) ? 16'sd256 : 16'sd0);   // K seed word
            store16(80 + i[7:0]*4, 16'sd256);                       // V seed word
        end
        for (i = 0; i < 4; i = i + 1) tload(i[7:0]*4,      i[`TM_IDX_W-1:0]);      // Q->TM0..3
        for (i = 0; i < 4; i = i + 1) tload(40 + i[7:0]*4, (4 + i[`TM_IDX_W-1:0])); // K->TM4..7
        for (i = 0; i < 4; i = i + 1) tload(80 + i[7:0]*4, (8 + i[`TM_IDX_W-1:0])); // V->TM8..11
        step(Iimm(`OP_LOADI, 5, 20'd0));         // q_base = 0
        step(Iimm(`OP_LOADI, 6, 20'd4));         // k_base = 4
        step(Iimm(`OP_LOADI, 7, 20'd15));        // STALE r7 = 15 (wrong o_base)
        store16(70, 16'sd20);                    // correct o_base 20 in DMEM[70]
        step(Iimm(`OP_LOADI, 9, 20'd70));        // address
        attn_obase = 5'd31;
        step(R(`OP_LOAD, 9, 0, 7, 0));           // r7 = DMEM[70] = 20  (LOAD producer of rC=r7)
        step(R(`OP_ATTENTION, 5, 6, 7, 12'd8));  // v_base=8, o_base = r7 (must be 20, not 15)
        drain(6);
        chk_eq("ATTENTION o_base load-use: latched correct base",
               attn_obase, 5'd20);

        //====================================================================
        // SUMMARY
        //====================================================================
        if (fail == 0)
            $display("\nALL %0d TESTS PASSED", pass);
        else begin
            $display("\n%0d TESTS FAILED (of %0d)", fail, pass + fail);
            $fatal(1, "hazard tests failed");
        end
        $finish;
    end

    initial begin
        #10000000;
        $display("FATAL: hazard TB timeout");
        $fatal(1, "timeout");
    end

endmodule
