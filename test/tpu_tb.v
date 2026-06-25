`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// test/tpu_tb.v  --  TPU v2.0 PROGRAM-DRIVEN system integration testbench
//----------------------------------------------------------------------------
// This is the end-to-end system test of the integrated v2.0 core (src/tpu_top.v
// module TPU).  It is PROGRAM-DRIVEN: every program-visible piece of state is
// seeded by REAL INSTRUCTIONS (LOADI / arithmetic to compose 32-bit words /
// STORE into DMEM / TLOAD into TM) issued on `instruction_in` -- there are NO
// hierarchical pokes of architectural state.  The only hierarchical accesses
// are READ-BACK peeks of TM / DMEM / RF for CHECKING, after the program has
// committed them (the DUT owns and produced every value being checked).
//
// COVERAGE: the program issues EVERY opcode in docs/ISA.md §3:
//   NOP, LOADI, LOAD, STORE, ADD, SUB, AND, OR, XOR, SHL, SHR, RELU, ADDI,
//   RDSTATUS, CLRSTATUS, DMA, GATHER, SCATTER, TLOAD, TSTORE, GEMM, CONV2D,
//   SOFTMAX, ATTENTION, FUSE_GEMM_RELU, FUSE_CONV_RELU, plus an illegal opcode.
// The headline tensor flow is, end-to-end:
//     GEMM  ->  FUSE_GEMM_RELU  ->  SOFTMAX  ->  ATTENTION
// each checked against an INDEPENDENT `real`-typed golden (real matmul / real
// exp-softmax / real attention), bit-exact for the exact ops (GEMM/FUSE/CONV)
// and within a STATED tolerance for the LUT-approx ops (softmax/attention).
//
// TOLERANCE (stated): softmax & attention use the unit's documented LUT-exp +
//   reciprocal approximation, so their probabilities / context elements are
//   checked to ATOL_Q016 (= 4 LSB of Q0.16) and ATOL_Q78 (= 3 LSB of Q7.8)
//   respectively.  GEMM/FUSE_GEMM_RELU/CONV are EXACT (round-half-up + saturate
//   identical to the boundary quantization of the real golden) and checked
//   bit-exactly.
//
// TM PACKING CONVENTIONS (confirmed against the leaf units):
//   * GEMM/SOFTMAX/ATTENTION lanes: one value in the LOW 16 bits of each 32-bit
//     lane; lane c at bits [32*c +: 16].  4 values per 128-bit line.
//   * CONV2D input/kernel/output: DENSE 16-bit packing; value v at bits
//     [16*v +: 16] (8 px or 4 out-px per line; high bits unused).
//
// A REAL VCD is produced ($dumpfile/$dumpvars).  Counts tests; prints
// "ALL N TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module tpu_tb;

    // ===================== DUT =====================
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

    // 10 ns clock.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ===================== bookkeeping =====================
    integer pass, fail;
    integer illegal_seen;

    // ===================== instruction builders =====================
    function [31:0] R;
        input [7:0] op; input [3:0] a, b, c; input [11:0] im;
        R = {op, a, b, c, im};
    endfunction
    function [31:0] Iimm;
        input [7:0] op; input [3:0] c; input [19:0] im;
        Iimm = {op, c, im};
    endfunction

    // Issue one instruction, honoring any pipeline stall by holding the same
    // instruction word until the front un-stalls.  Drives on the negedge so the
    // word is stable before the sampling posedge.
    task step;
        input [31:0] ins;
        begin
            @(negedge clk); instr = ins;
            @(posedge clk);
            if (illegal_opcode) illegal_seen = 1;
            while (dbg_pipe_stall === 1'b1) begin
                @(negedge clk);
                @(posedge clk);
            end
        end
    endtask

    // Drain the pipeline so all in-flight results retire to RF/TM/DMEM.
    task drain;
        input integer n;
        integer k;
        begin
            for (k = 0; k < n; k = k + 1) step(R(`OP_NOP, 0, 0, 0, 0));
            for (k = 0; k < n; k = k + 1) @(posedge clk);
        end
    endtask

    // ===================== program helpers (all via real instructions) =====================
    // Store a 16-bit (sign-extended) value into DMEM[addr] using LOADI+STORE.
    task store16;
        input [7:0]         addr;
        input signed [15:0] val;
        begin
            step(Iimm(`OP_LOADI, 14, {{4{val[15]}}, val}));   // r14 = sext(val)
            step(Iimm(`OP_LOADI, 12, {12'd0, addr}));         // r12 = addr
            step(R(`OP_STORE, 12, 14, 0, 0));                 // DMEM[r12] = r14
        end
    endtask

    // Store a full 32-bit word = (hi<<16)|lo16 into DMEM[addr], composed with
    // real ALU ops (LOADI/SHL/OR) -- the software path for packed CONV lanes.
    task store_word;
        input [7:0]  addr;
        input [15:0] lo, hi;
        begin
            step(Iimm(`OP_LOADI, 14, {4'd0, lo}));            // r14 = lo (zero-ext)
            step(Iimm(`OP_LOADI, 15, {4'd0, hi}));            // r15 = hi
            step(Iimm(`OP_LOADI, 13, 16));                    // r13 = 16
            step(R(`OP_SHL, 15, 13, 15, 0));                  // r15 = hi << 16
            step(R(`OP_OR,  14, 15, 14, 0));                  // r14 = (hi<<16)|lo
            step(Iimm(`OP_LOADI, 12, {12'd0, addr}));         // r12 = addr
            step(R(`OP_STORE, 12, 14, 0, 0));                 // DMEM[r12] = r14
        end
    endtask

    // TLOAD DMEM[base..base+3] into TM line `line` (4 words -> 1 line).
    task tload;
        input [7:0]           base;
        input [`TM_IDX_W-1:0] line;
        begin
            step(Iimm(`OP_LOADI, 3, {12'd0, base}));          // r3 = DMEM base
            step(R(`OP_TLOAD, 3, 0, 0, {7'd0, line}));        // TM[line]=DMEM[r3..]
        end
    endtask

    // ===================== TM readback (checking only) =====================
    // 32-bit-lane convention (GEMM/SOFTMAX/ATTENTION): lane c low 16 bits.
    function signed [15:0] tm_lane;
        input [`TM_IDX_W-1:0] line; input integer c;
        begin tm_lane = dut.tmem.lines[line][c*`LANE_W +: `ELEM_W]; end
    endfunction
    // 16-bit dense convention (CONV in/kernel/out): value v at bits[16*v].
    function signed [15:0] tm_dense;
        input [`TM_IDX_W-1:0] line; input integer v;
        begin tm_dense = dut.tmem.lines[line][v*`ELEM_W +: `ELEM_W]; end
    endfunction

    function integer absdiff;
        input integer a, b;
        begin absdiff = (a > b) ? (a - b) : (b - a); end
    endfunction

    // real <-> Q7.8 helpers (boundary quantization for exact-op goldens).
    function signed [15:0] rq78;
        input real x; real s;
        begin
            s = $rtoi($floor(x * 256.0 + 0.5));
            if (s >  32767.0) rq78 = 16'sh7FFF;
            else if (s < -32768.0) rq78 = 16'sh8000;
            else rq78 = $rtoi(s);
        end
    endfunction
    function real q78r;
        input signed [15:0] q;
        begin q78r = $itor(q) / 256.0; end
    endfunction

    // ===================== check primitives =====================
    task chk_eq;
        input [255:0] name;
        input integer got, exp;
        begin
            if (got === exp) begin
                pass = pass + 1;
                $display("  PASS %0s : %0d", name, got);
            end else begin
                fail = fail + 1;
                $display("  FAIL %0s : got=%0d exp=%0d", name, got, exp);
                $fatal(1, "system test mismatch");
            end
        end
    endtask
    task chk_tol;
        input [255:0] name;
        input integer got, exp, tol;
        begin
            if (absdiff(got, exp) <= tol) begin
                pass = pass + 1;
                $display("  PASS %0s : got=%0d exp=%0d (|d|<=%0d)", name, got, exp, tol);
            end else begin
                fail = fail + 1;
                $display("  FAIL %0s : got=%0d exp=%0d tol=%0d", name, got, exp, tol);
                $fatal(1, "system test tolerance exceeded");
            end
        end
    endtask

    // ===================== golden data =====================
    localparam integer ATOL_Q016 = 4;   // softmax probability tolerance (Q0.16 LSB)
    localparam integer ATOL_Q78  = 3;    // attention context tolerance (Q7.8 LSB)

    integer i, j, k, d;

    // GEMM operands (Q7.8 16-bit) and goldens.
    reg signed [15:0] Amat [0:3][0:3];
    reg signed [15:0] Bmat [0:3][0:3];
    reg signed [15:0] Cgold[0:3][0:3];    // raw GEMM golden
    reg signed [15:0] Crelu[0:3][0:3];    // FUSE_GEMM_RELU golden
    real racc;

    // SOFTMAX logits and golden probs.
    reg signed [15:0] Lvec [0:7];
    real    sm_e [0:7];
    real    sm_sum;
    integer sm_gp [0:7];
    integer sm_argmax, sm_gmax;

    // ATTENTION operands and golden.
    reg signed [15:0] Qm [0:3][0:3];
    reg signed [15:0] Km [0:3][0:3];
    reg signed [15:0] Vm [0:3][0:3];
    real Smat [0:3][0:3];
    real Wmat [0:3][0:3];
    real Omat [0:3][0:3];
    reg signed [15:0] Ogold [0:3][0:3];
    real rmax, rsum;

    // CONV operands and golden.
    reg signed [15:0] IN  [0:7][0:7];
    reg signed [15:0] KER [0:2][0:2];
    reg signed [15:0] CONVg [0:5][0:5];   // raw conv golden
    reg signed [15:0] CRELUg[0:5][0:5];   // FUSE_CONV_RELU golden
    // NEGATIVE-producing conv: a -1.0 impulse kernel over the (non-negative) IN
    // yields all-NEGATIVE conv outputs, so the FUSE RELU is LOAD-BEARING on ALL
    // 9 output lines (the impulse-kernel case above masks it -- RELU is a no-op
    // there).  NEGCRELUg is therefore all-zero; any un-RELU'd line shows as a
    // negative leak.  This exercises the v2.0 FUSE_CONV_RELU line-count fix
    // (9 lines, not 4) and the dense-16 RELU granularity fix.
    reg signed [15:0] NEGKER  [0:2][0:2]; // -1.0 center-impulse kernel
    reg signed [15:0] NEGCONVg [0:5][0:5]; // raw conv golden (negative kernel)
    reg signed [15:0] NEGCRELUg[0:5][0:5]; // FUSE_CONV_RELU golden (all zero)
    integer oy, ox, wr, wc, iy, ix;
    real cacc, ncacc;

    // ===================== compute all goldens =====================
    task compute_goldens;
        begin
            // ---- GEMM: A,B chosen to mix signs; produce some negatives ----
            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1) begin
                    Amat[i][j] = (i - j) * 16'sd96;                 // -1.125..1.125
                    Bmat[i][j] = (j - 1) * 16'sd256 + (i * 16'sd64);
                end
            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1) begin
                    racc = 0.0;
                    for (k = 0; k < 4; k = k + 1)
                        racc = racc + q78r(Amat[i][k]) * q78r(Bmat[k][j]);
                    Cgold[i][j] = rq78(racc);
                    Crelu[i][j] = (Cgold[i][j] < 0) ? 16'sd0 : Cgold[i][j];
                end

            // ---- SOFTMAX logits ----
            Lvec[0]=16'sd256; Lvec[1]=16'sd512; Lvec[2]=16'sd0;   Lvec[3]=-16'sd256;
            Lvec[4]=16'sd128; Lvec[5]=16'sd384; Lvec[6]=16'sd64;  Lvec[7]=-16'sd128;
            sm_gmax = Lvec[0]; sm_argmax = 0;
            for (k = 1; k < 8; k = k + 1)
                if (Lvec[k] > sm_gmax) begin sm_gmax = Lvec[k]; sm_argmax = k; end
            sm_sum = 0.0;
            for (k = 0; k < 8; k = k + 1) begin
                sm_e[k] = $exp((Lvec[k] - sm_gmax) / 256.0);
                sm_sum  = sm_sum + sm_e[k];
            end
            for (k = 0; k < 8; k = k + 1) begin
                sm_gp[k] = $rtoi((sm_e[k] / sm_sum) * 65536.0 + 0.5);
                if (sm_gp[k] > 65535) sm_gp[k] = 65535;
                if (sm_gp[k] < 0)     sm_gp[k] = 0;
            end

            // ---- ATTENTION Q,K,V ----
            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1) begin
                    Qm[i][j] = (((i + j) % 3))         * 16'sd128;
                    Km[i][j] = (((i*2 + j) % 4))       * 16'sd128;
                    Vm[i][j] = ((((i + 2*j) % 5) - 2)) * 16'sd128;
                end
            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1) begin
                    Smat[i][j] = 0.0;
                    for (d = 0; d < 4; d = d + 1)
                        Smat[i][j] = Smat[i][j] + q78r(Qm[i][d]) * q78r(Km[j][d]);
                    Smat[i][j] = Smat[i][j] / 2.0;                 // 1/sqrt(4)
                end
            for (i = 0; i < 4; i = i + 1) begin
                rmax = Smat[i][0];
                for (j = 1; j < 4; j = j + 1) if (Smat[i][j] > rmax) rmax = Smat[i][j];
                rsum = 0.0;
                for (j = 0; j < 4; j = j + 1) begin
                    Wmat[i][j] = $exp(Smat[i][j] - rmax); rsum = rsum + Wmat[i][j];
                end
                for (j = 0; j < 4; j = j + 1) Wmat[i][j] = Wmat[i][j] / rsum;
            end
            for (i = 0; i < 4; i = i + 1)
                for (d = 0; d < 4; d = d + 1) begin
                    Omat[i][d] = 0.0;
                    for (j = 0; j < 4; j = j + 1)
                        Omat[i][d] = Omat[i][d] + Wmat[i][j] * q78r(Vm[j][d]);
                    Ogold[i][d] = rq78(Omat[i][d]);
                end

            // ---- CONV: impulse (center) kernel -> valid center crop ----
            for (i = 0; i < 8; i = i + 1)
                for (j = 0; j < 8; j = j + 1)
                    IN[i][j] = ((i*8 + j) % 10) * 16'sd256;
            for (i = 0; i < 3; i = i + 1)
                for (j = 0; j < 3; j = j + 1) KER[i][j] = 16'sd0;
            KER[1][1] = 16'sd256;                                  // 1.0 impulse
            for (oy = 0; oy < 6; oy = oy + 1)
                for (ox = 0; ox < 6; ox = ox + 1) begin
                    cacc = 0.0;
                    for (wr = 0; wr < 3; wr = wr + 1)
                        for (wc = 0; wc < 3; wc = wc + 1) begin
                            iy = oy + wr; ix = ox + wc;            // stride1 pad0
                            cacc = cacc + q78r(IN[iy][ix]) * q78r(KER[wr][wc]);
                        end
                    CONVg[oy][ox]  = rq78(cacc);
                    CRELUg[oy][ox] = (CONVg[oy][ox] < 0) ? 16'sd0 : CONVg[oy][ox];
                end

            // ---- NEGATIVE conv golden: same IN, a -1.0 center-impulse kernel ----
            //   KER'[1][1] = -256 (= -1.0), all other taps 0.  out = -IN(center)
            //   (all <= 0 since IN >= 0), so RELU -> all 0.  Computed with the SAME
            //   nested-loop convolution as the positive case (independent of the
            //   DUT) using the negated kernel array NEGKER.
            for (i = 0; i < 3; i = i + 1)
                for (j = 0; j < 3; j = j + 1) NEGKER[i][j] = 16'sd0;
            NEGKER[1][1] = -16'sd256;                              // -1.0 impulse
            for (oy = 0; oy < 6; oy = oy + 1)
                for (ox = 0; ox < 6; ox = ox + 1) begin
                    ncacc = 0.0;
                    for (wr = 0; wr < 3; wr = wr + 1)
                        for (wc = 0; wc < 3; wc = wc + 1) begin
                            iy = oy + wr; ix = ox + wc;            // stride1 pad0
                            ncacc = ncacc + q78r(IN[iy][ix]) * q78r(NEGKER[wr][wc]);
                        end
                    NEGCONVg[oy][ox]  = rq78(ncacc);
                    NEGCRELUg[oy][ox] = (NEGCONVg[oy][ox] < 0) ? 16'sd0
                                                               : NEGCONVg[oy][ox];
                end
        end
    endtask

    // ===================== seeding tasks (via instructions) =====================
    // Seed a 4x4 Q7.8 matrix into DMEM at `dbase` (row = 4 words, low-16 of each
    // holds the value), then TLOAD the 4 rows to TM lines `tbase..tbase+3`.
    task seed_matrix_4x4;
        input [7:0]           dbase;
        input [`TM_IDX_W-1:0] tbase;
        input integer         which;     // 0=A 1=B 2=Q 3=K 4=V
        integer r, c;
        reg signed [15:0] v;
        begin
            for (r = 0; r < 4; r = r + 1) begin
                for (c = 0; c < 4; c = c + 1) begin
                    case (which)
                        0: v = Amat[r][c];
                        1: v = Bmat[r][c];
                        2: v = Qm[r][c];
                        3: v = Km[r][c];
                        default: v = Vm[r][c];
                    endcase
                    store16(dbase + r[7:0]*4 + c[7:0], v);
                end
                tload(dbase + r[7:0]*4, tbase + r[`TM_IDX_W-1:0]);
            end
        end
    endtask

    // ===================== main program =====================
    initial begin
        $dumpfile("tpu_waveform.vcd");
        $dumpvars(0, tpu_tb);

        pass = 0; fail = 0; illegal_seen = 0;
        compute_goldens();

        instr = R(`OP_NOP, 0, 0, 0, 0);
        rst = 1'b1;
        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        //====================================================================
        // PART 0 -- scalar ISA smoke.
        //====================================================================
        $display("[PART 0] scalar ISA");
        step(Iimm(`OP_LOADI, 1, 20'h00005));     // r1 = 5
        step(Iimm(`OP_LOADI, 2, 20'h00007));     // r2 = 7
        step(R(`OP_ADD, 1, 2, 3, 0));            // r3 = 12  (fwd r1,r2)
        step(R(`OP_SUB, 2, 1, 4, 0));            // r4 = 2
        step(R(`OP_AND, 1, 2, 5, 0));            // r5 = 5&7 = 5
        step(R(`OP_OR,  1, 2, 6, 0));            // r6 = 5|7 = 7
        step(R(`OP_XOR, 1, 2, 7, 0));            // r7 = 5^7 = 2
        step(Iimm(`OP_LOADI, 8, 20'h00001));
        step(R(`OP_SHL, 1, 8, 9, 0));            // r9 = 5<<1 = 10
        step(R(`OP_SHR, 1, 8, 10, 0));           // r10 = 5>>1 = 2
        step(Iimm(`OP_LOADI, 11, 20'hFFFFB));    // r11 = -5 (sext)
        step(R(`OP_RELU, 11, 0, 13, 0));         // r13 = RELU(-5) = 0
        step(R(`OP_ADDI, 1, 0, 14, 12'h003));    // r14 = r1 + 3 = 8
        drain(6);
        chk_eq("ADD r3",  dut.regfile.registers[3],  12);
        chk_eq("SUB r4",  dut.regfile.registers[4],  2);
        chk_eq("AND r5",  dut.regfile.registers[5],  5);
        chk_eq("OR  r6",  dut.regfile.registers[6],  7);
        chk_eq("XOR r7",  dut.regfile.registers[7],  2);
        chk_eq("SHL r9",  dut.regfile.registers[9],  10);
        chk_eq("SHR r10", dut.regfile.registers[10], 2);
        chk_eq("RELU r13",dut.regfile.registers[13], 0);
        chk_eq("ADDI r14",dut.regfile.registers[14], 8);

        store16(40, 16'sd1234);
        step(Iimm(`OP_LOADI, 1, 20'd40));
        step(R(`OP_LOAD, 1, 0, 5, 0));           // r5 = DMEM[40] = 1234
        drain(4);
        chk_eq("LOAD r5", dut.regfile.registers[5], 1234);

        //====================================================================
        // PART 1 -- GEMM (TM->TM, exact vs real matmul golden).
        //====================================================================
        $display("[PART 1] GEMM (4x4, exact)");
        seed_matrix_4x4(8'd0,  5'd0, 0);         // A -> TM[0..3]
        seed_matrix_4x4(8'd16, 5'd4, 1);         // B -> TM[4..7]
        step(Iimm(`OP_LOADI, 5, 20'd0));
        step(Iimm(`OP_LOADI, 6, 20'd4));
        step(R(`OP_GEMM, 5, 6, 7, 12'd8));       // C -> TM[8..11], status -> r7
        drain(6);
        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                chk_eq("GEMM C", tm_lane(8 + i[`TM_IDX_W-1:0], j), Cgold[i][j]);
        chk_eq("GEMM status unit",
               (dut.regfile.registers[7] >> `ST_UNIT_LO) &
               ((1 << (`ST_UNIT_HI - `ST_UNIT_LO + 1)) - 1), `UNIT_GEMM);

        //====================================================================
        // PART 2 -- FUSE_GEMM_RELU (GEMM then in-place RELU on the C tile).
        //====================================================================
        $display("[PART 2] FUSE_GEMM_RELU (exact)");
        step(Iimm(`OP_LOADI, 5, 20'd0));
        step(Iimm(`OP_LOADI, 6, 20'd4));
        step(R(`OP_FUSE_GEMM_RELU, 5, 6, 7, 12'd12)); // -> TM[12..15]
        drain(8);
        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                chk_eq("FUSE_GEMM_RELU C", tm_lane(12 + i[`TM_IDX_W-1:0], j),
                       Crelu[i][j]);

        // TSTORE TM[12] -> DMEM[60..63], LOAD them back, check row 0.
        step(Iimm(`OP_LOADI, 4, 20'd60));
        step(R(`OP_TSTORE, 4, 0, 0, 12'd12));    // DMEM[60..63] = TM[12]
        drain(4);
        for (j = 0; j < 4; j = j + 1) begin
            step(Iimm(`OP_LOADI, 1, 60 + j[19:0]));
            step(R(`OP_LOAD, 1, 0, 9, 0));       // r9 = DMEM[60+j]
            drain(4);
            chk_eq("TSTORE/LOAD C row0", $signed(dut.regfile.registers[9][15:0]),
                   Crelu[0][j]);
        end

        //====================================================================
        // PART 3 -- SOFTMAX (len-8, tolerance vs real exp-softmax golden).
        //====================================================================
        $display("[PART 3] SOFTMAX (len-8, tolerance)");
        for (i = 0; i < 8; i = i + 1) store16(80 + i[7:0], Lvec[i]);
        tload(8'd80, 5'd16);                     // TM[16] = L0..3
        tload(8'd84, 5'd17);                     // TM[17] = L4..7
        step(Iimm(`OP_LOADI, 5, 20'd16));        // x base line = 16
        step(R(`OP_SOFTMAX, 5, 0, 7, 12'd18));   // probs -> TM[18..19], status->r7
        drain(6);
        for (i = 0; i < 4; i = i + 1) begin
            chk_tol("SOFTMAX p", tm_lane(5'd18, i) & 16'hFFFF, sm_gp[i], ATOL_Q016);
            chk_tol("SOFTMAX p", tm_lane(5'd19, i) & 16'hFFFF, sm_gp[4+i], ATOL_Q016);
        end
        chk_eq("SOFTMAX status unit",
               (dut.regfile.registers[7] >> `ST_UNIT_LO) &
               ((1 << (`ST_UNIT_HI - `ST_UNIT_LO + 1)) - 1), `UNIT_SOFTMAX);
        chk_eq("SOFTMAX status argmax",
               (dut.regfile.registers[7] >> `ST_ARGMAX_LO) &
               ((1 << (`ST_ARGMAX_HI - `ST_ARGMAX_LO + 1)) - 1), sm_argmax);

        //====================================================================
        // PART 4 -- ATTENTION (seq4 d4, tolerance vs real attention golden).
        //====================================================================
        $display("[PART 4] ATTENTION (seq4 d4, tolerance)");
        seed_matrix_4x4(8'd96,  5'd20, 2);       // Q -> TM[20..23]
        seed_matrix_4x4(8'd112, 5'd24, 3);       // K -> TM[24..27]
        seed_matrix_4x4(8'd128, 5'd28, 4);       // V -> TM[28..31]
        step(Iimm(`OP_LOADI, 5, 20'd20));        // Q base
        step(Iimm(`OP_LOADI, 6, 20'd24));        // K base
        step(Iimm(`OP_LOADI, 7, 20'd8));         // O base line = 8 (rC-source)
        step(R(`OP_ATTENTION, 5, 6, 7, 12'd28)); // V base=28, O base=r7=8, st->r7
        drain(8);
        for (i = 0; i < 4; i = i + 1)
            for (d = 0; d < 4; d = d + 1)
                chk_tol("ATTENTION O", tm_lane(8 + i[`TM_IDX_W-1:0], d),
                        Ogold[i][d], ATOL_Q78);
        chk_eq("ATTENTION status unit",
               (dut.regfile.registers[7] >> `ST_UNIT_LO) &
               ((1 << (`ST_UNIT_HI - `ST_UNIT_LO + 1)) - 1), `UNIT_ATTENTION);

        //====================================================================
        // PART 5 -- CONV2D + FUSE_CONV_RELU (opcode coverage; exact goldens).
        //====================================================================
        $display("[PART 5] CONV2D + FUSE_CONV_RELU (exact)");
        for (i = 0; i < 8; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                store_word(150 + i[7:0]*4 + j[7:0],
                           IN[i][2*j][15:0], IN[i][2*j+1][15:0]);
        for (i = 0; i < 3; i = i + 1) begin
            store_word(190 + i[7:0]*4 + 0, KER[i][0][15:0], KER[i][1][15:0]);
            store_word(190 + i[7:0]*4 + 1, KER[i][2][15:0], 16'd0);
        end
        for (i = 0; i < 8; i = i + 1) tload(150 + i[7:0]*4, i[`TM_IDX_W-1:0]);
        for (i = 0; i < 3; i = i + 1) tload(190 + i[7:0]*4, (10 + i[`TM_IDX_W-1:0]));
        step(Iimm(`OP_LOADI, 5, 20'd0));         // in base line = 0
        step(Iimm(`OP_LOADI, 6, 20'd10));        // kernel base line = 10
        step(R(`OP_CONV2D, 5, 6, 7, (12'd16 << 5) | 12'd1));   // out -> TM[16..]
        drain(10);
        for (oy = 0; oy < 6; oy = oy + 1)
            for (ox = 0; ox < 6; ox = ox + 1) begin : conv_chk
                integer p, ln, lane;
                p = oy*6 + ox; ln = 16 + p/4; lane = p % 4;
                chk_eq("CONV2D out", tm_dense(ln[`TM_IDX_W-1:0], lane),
                       CONVg[oy][ox]);
            end
        step(Iimm(`OP_LOADI, 5, 20'd0));
        step(Iimm(`OP_LOADI, 6, 20'd10));
        step(R(`OP_FUSE_CONV_RELU, 5, 6, 7, (12'd26 << 5) | 12'd1));
        drain(10);
        for (oy = 0; oy < 6; oy = oy + 1)
            for (ox = 0; ox < 6; ox = ox + 1) begin : fconv_chk
                integer p, ln, lane;
                p = oy*6 + ox; ln = 26 + p/4; lane = p % 4;
                chk_eq("FUSE_CONV_RELU out", tm_dense(ln[`TM_IDX_W-1:0], lane),
                       CRELUg[oy][ox]);
            end

        //----------------------------------------------------------------
        // PART 5b -- FUSE_CONV_RELU with a NEGATIVE conv output (RELU is
        //   LOAD-BEARING on ALL 9 output lines).  Reuses the same IN (already in
        //   TM[0..7]) with a -1.0 center-impulse kernel; every conv pixel is
        //   negative so the FUSE RELU must zero ALL 36 pixels -- a missed line or
        //   a wrong RELU granularity leaks a negative value.  This is the
        //   regression guard for the FUSE_CONV_RELU 9-line + dense-16 fixes.
        //----------------------------------------------------------------
        $display("[PART 5b] FUSE_CONV_RELU negative output (RELU load-bearing)");
        // Re-seed IN into TM[0..7] (defensive: earlier parts may touch TM).
        for (i = 0; i < 8; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                store_word(150 + i[7:0]*4 + j[7:0],
                           IN[i][2*j][15:0], IN[i][2*j+1][15:0]);
        for (i = 0; i < 8; i = i + 1) tload(150 + i[7:0]*4, i[`TM_IDX_W-1:0]);
        // negative kernel into DMEM 190.. then TM[10..12]: center = -1.0
        store_word(190 + 0*4 + 0, 16'sd0, 16'sd0);
        store_word(190 + 0*4 + 1, 16'sd0, 16'd0);
        store_word(190 + 1*4 + 0, 16'sd0, -16'sd256);   // KER[1][0]=0, KER[1][1]=-256
        store_word(190 + 1*4 + 1, 16'sd0, 16'd0);
        store_word(190 + 2*4 + 0, 16'sd0, 16'sd0);
        store_word(190 + 2*4 + 1, 16'sd0, 16'd0);
        for (i = 0; i < 3; i = i + 1) tload(190 + i[7:0]*4, (10 + i[`TM_IDX_W-1:0]));
        step(Iimm(`OP_LOADI, 5, 20'd0));
        step(Iimm(`OP_LOADI, 6, 20'd10));
        // raw CONV first to confirm the outputs are genuinely negative
        step(R(`OP_CONV2D, 5, 6, 7, (12'd16 << 5) | 12'd1));   // out -> TM[16..]
        drain(10);
        for (oy = 0; oy < 6; oy = oy + 1)
            for (ox = 0; ox < 6; ox = ox + 1) begin : negconv_chk
                integer p, ln, lane;
                p = oy*6 + ox; ln = 16 + p/4; lane = p % 4;
                chk_eq("CONV2D neg out", tm_dense(ln[`TM_IDX_W-1:0], lane),
                       NEGCONVg[oy][ox]);
            end
        // now FUSE_CONV_RELU: every one of the 36 pixels (9 lines) must be 0.
        step(Iimm(`OP_LOADI, 5, 20'd0));
        step(Iimm(`OP_LOADI, 6, 20'd10));
        step(R(`OP_FUSE_CONV_RELU, 5, 6, 7, (12'd26 << 5) | 12'd1));
        drain(20);
        for (oy = 0; oy < 6; oy = oy + 1)
            for (ox = 0; ox < 6; ox = ox + 1) begin : negfconv_chk
                integer p, ln, lane;
                p = oy*6 + ox; ln = 26 + p/4; lane = p % 4;
                chk_eq("FUSE_CONV_RELU neg->0 all lines",
                       tm_dense(ln[`TM_IDX_W-1:0], lane), NEGCRELUg[oy][ox]);
            end

        //====================================================================
        // PART 6 -- DMA / GATHER / SCATTER.
        //====================================================================
        $display("[PART 6] DMA / GATHER / SCATTER");
        for (i = 0; i < 5; i = i + 1) store16(200 + i[7:0], (300 + i[15:0]));
        step(Iimm(`OP_LOADI, 3, 20'd200));       // DMA src
        step(Iimm(`OP_LOADI, 4, 20'd220));       // DMA dst
        step(R(`OP_DMA, 3, 4, 5, 12'd5));        // copy 5 words; r5 = #words
        drain(6);
        for (i = 0; i < 5; i = i + 1)
            chk_eq("DMA copy", $signed(dut.u_dmem.sram[220 + i][15:0]), 300 + i);
        chk_eq("DMA words", dut.regfile.registers[5], 5);

        store16(236, 16'sd4242);
        step(Iimm(`OP_LOADI, 1, 20'd230));       // base
        step(Iimm(`OP_LOADI, 2, 20'd3));         // index
        step(R(`OP_GATHER, 1, 2, 6, 12'd1));     // r6 = DMEM[230+(3<<1)] = DMEM[236]
        drain(6);
        chk_eq("GATHER r6", $signed(dut.regfile.registers[6][15:0]), 4242);

        step(Iimm(`OP_LOADI, 7, 20'd5555));      // value (rC-source)
        step(Iimm(`OP_LOADI, 1, 20'd240));       // base
        step(Iimm(`OP_LOADI, 2, 20'd2));         // index
        step(R(`OP_SCATTER, 1, 2, 7, 12'd1));    // DMEM[244] = r7 = 5555
        drain(6);
        chk_eq("SCATTER DMEM[244]", $signed(dut.u_dmem.sram[244][15:0]), 5555);

        //====================================================================
        // PART 7 -- STATUS / RDSTATUS / CLRSTATUS / illegal opcode.
        //====================================================================
        $display("[PART 7] status + illegal opcode");
        step(Iimm(`OP_LOADI, 8, 20'd123));       // r8 = 123 (must survive)
        step(R(8'hAA, 8, 9, 8, 12'h000));        // illegal: must NOT write r8
        drain(4);
        chk_eq("illegal opcode flagged", illegal_seen, 1);
        chk_eq("illegal = safe NOP (r8 kept)", dut.regfile.registers[8], 123);
        step(R(`OP_RDSTATUS, 0, 0, 9, 0));       // r9 = status
        drain(4);
        chk_eq("RDSTATUS illegal bit",
               (dut.regfile.registers[9] >> `ST_ILLEGAL_BIT) & 1, 1);
        step(R(`OP_CLRSTATUS, 0, 0, 0, 0));
        drain(2);
        step(R(`OP_RDSTATUS, 0, 0, 10, 0));      // r10 = status
        drain(4);
        chk_eq("CLRSTATUS cleared illegal",
               (dut.regfile.registers[10] >> `ST_ILLEGAL_BIT) & 1, 0);

        step(Iimm(`OP_LOADI, 0, 20'd999));       // r0 write must be ignored
        drain(4);
        chk_eq("r0 hardwired zero", dut.regfile.registers[0], 0);

        //====================================================================
        // SUMMARY
        //====================================================================
        if (fail == 0)
            $display("\nALL %0d TESTS PASSED", pass);
        else begin
            $display("\n%0d TESTS FAILED (of %0d)", fail, pass + fail);
            $fatal(1, "system integration tests failed");
        end
        $finish;
    end

    // global timeout guard.
    initial begin
        #20000000;
        $display("FATAL: system TB timeout");
        $fatal(1, "timeout");
    end

endmodule
