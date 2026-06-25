`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// vector_alu_tb.v  --  self-checking unit TB for src/vector_alu.v
//----------------------------------------------------------------------------
// DUT: vector_alu (combinational scalar ALU, SPEC.md §5.5).
//
// GOLDEN MODEL (independence):
//   vector_alu performs EXACT integer / bitwise / shift ops, so the reference
//   computes each result with the NATIVE Verilog operator for that op and the
//   comparison is BIT-EXACT (no tolerance).  The golden is structured as an
//   independent decode (a separate case in the TB task `expected`) so a typo in
//   the DUT's opcode->operation mapping is caught: e.g. if the DUT wired SHL to
//   ">>", the golden still uses "<<" and the vectors diverge.  For RELU the
//   golden uses a signed compare ($signed(in1) < 0) computed independently of
//   the DUT's bit-31 test.  This satisfies "compare against native Verilog
//   operators; negative/zero/overflow edges".
//
// TESTS:
//   * Directed per-op truth-table corner cases: zero, +/- one, max, min,
//     all-ones, alternating-bits, identity/impulse shifts, full-width shifts,
//     RELU on negative/zero/positive/min/max, ADD/SUB overflow (wrap) edges.
//   * Constrained-random: seeded $random, >=200 vectors per opcode across all
//     9 ALU opcodes plus illegal-opcode -> 0 default checks.
//   * $fatal on ANY mismatch; prints "ALL <N> TESTS PASSED".
//============================================================================
module vector_alu_tb;

    // ---- DUT ports ----
    reg  [`OPCODE_W-1:0] opcode;
    reg  [`XLEN-1:0]     in1;
    reg  [`XLEN-1:0]     in2;
    wire [`XLEN-1:0]     out;

    vector_alu dut (
        .opcode (opcode),
        .in1    (in1),
        .in2    (in2),
        .out    (out)
    );

    integer pass_count = 0;
    integer fail_count = 0;
    integer i;

    // -----------------------------------------------------------------
    // Independent golden: native-operator reference for each ALU opcode.
    // Returns the expected 32-bit result for (op, a, b).
    // -----------------------------------------------------------------
    function [`XLEN-1:0] expected;
        input [`OPCODE_W-1:0] op;
        input [`XLEN-1:0]     a;
        input [`XLEN-1:0]     b;
        reg signed [`XLEN-1:0] sa;
        begin
            sa = a;
            case (op)
                `OP_RELU: expected = (sa < 0) ? {`XLEN{1'b0}} : a; // signed test
                `OP_ADD:  expected = a + b;
                `OP_ADDI: expected = a + b;
                `OP_SUB:  expected = a - b;
                `OP_AND:  expected = a & b;
                `OP_OR:   expected = a | b;
                `OP_XOR:  expected = a ^ b;
                `OP_SHL:  expected = a << b[4:0];   // logical
                `OP_SHR:  expected = a >> b[4:0];   // logical
                default:  expected = {`XLEN{1'b0}}; // non-ALU op -> 0
            endcase
        end
    endfunction

    // -----------------------------------------------------------------
    // Drive one vector, compare DUT vs golden, count, $fatal on mismatch.
    // (combinational DUT: settle with #1 before sampling)
    // -----------------------------------------------------------------
    task check;
        input [`OPCODE_W-1:0] op;
        input [`XLEN-1:0]     a;
        input [`XLEN-1:0]     b;
        input [255:0]         label;     // ascii tag for diagnostics
        reg   [`XLEN-1:0]     exp;
        begin
            opcode = op;
            in1    = a;
            in2    = b;
            #1;
            exp = expected(op, a, b);
            if (out !== exp) begin
                fail_count = fail_count + 1;
                $display("FAIL [%0s] op=0x%02h in1=0x%08h in2=0x%08h : out=0x%08h exp=0x%08h",
                         label, op, a, b, out, exp);
                $fatal(1, "vector_alu mismatch");
            end else begin
                pass_count = pass_count + 1;
            end
        end
    endtask

    initial begin
        opcode = `OP_NOP;
        in1    = {`XLEN{1'b0}};
        in2    = {`XLEN{1'b0}};

        //==============================================================
        // DIRECTED: RELU corner cases
        //==============================================================
        check(`OP_RELU, 32'h0000_0000, 32'h0, "relu_zero");
        check(`OP_RELU, 32'h0000_0001, 32'h0, "relu_pos_one");
        check(`OP_RELU, 32'h7FFF_FFFF, 32'h0, "relu_pos_max");
        check(`OP_RELU, 32'hFFFF_FFFF, 32'h0, "relu_neg_one");      // -1 -> 0
        check(`OP_RELU, 32'h8000_0000, 32'h0, "relu_neg_min");      // INT_MIN -> 0
        check(`OP_RELU, 32'h8000_0001, 32'h0, "relu_neg_large");
        check(`OP_RELU, 32'h0000_8000, 32'h0, "relu_small_pos");

        //==============================================================
        // DIRECTED: ADD / ADDI corner cases (wrap, no saturation)
        //==============================================================
        check(`OP_ADD,  32'd0,         32'd0,         "add_zero");
        check(`OP_ADD,  32'd5,         32'd7,         "add_basic");
        check(`OP_ADD,  32'hFFFF_FFFF, 32'd1,         "add_wrap_to_zero");   // -1 + 1
        check(`OP_ADD,  32'h7FFF_FFFF, 32'd1,         "add_signed_overflow"); // wraps to MIN
        check(`OP_ADD,  32'hFFFF_FFFF, 32'hFFFF_FFFF, "add_neg_neg");
        check(`OP_ADD,  32'h8000_0000, 32'h8000_0000, "add_min_min");        // wraps to 0
        check(`OP_ADDI, 32'd100,       32'hFFFF_FFF6, "addi_neg_imm");       // +100 + (-10)
        check(`OP_ADDI, 32'h7FFF_FFFF, 32'd1,         "addi_overflow");

        //==============================================================
        // DIRECTED: SUB corner cases
        //==============================================================
        check(`OP_SUB,  32'd0,         32'd0,         "sub_zero");
        check(`OP_SUB,  32'd7,         32'd5,         "sub_basic");
        check(`OP_SUB,  32'd5,         32'd7,         "sub_neg_result");     // -> 0xFFFFFFFE
        check(`OP_SUB,  32'd0,         32'd1,         "sub_underflow");       // 0-1 = -1
        check(`OP_SUB,  32'h8000_0000, 32'd1,         "sub_min_minus_one");  // wraps to MAX
        check(`OP_SUB,  32'h7FFF_FFFF, 32'hFFFF_FFFF, "sub_max_minus_neg1"); // MAX-(-1) wraps

        //==============================================================
        // DIRECTED: AND / OR / XOR truth-table edges
        //==============================================================
        check(`OP_AND,  32'hFFFF_FFFF, 32'hFFFF_FFFF, "and_ones");
        check(`OP_AND,  32'hFFFF_FFFF, 32'h0000_0000, "and_ones_zero");
        check(`OP_AND,  32'hAAAA_AAAA, 32'h5555_5555, "and_alt");           // -> 0
        check(`OP_AND,  32'hF0F0_F0F0, 32'hFF00_FF00, "and_mask");
        check(`OP_OR,   32'h0000_0000, 32'h0000_0000, "or_zero");
        check(`OP_OR,   32'hAAAA_AAAA, 32'h5555_5555, "or_alt");            // -> all ones
        check(`OP_OR,   32'hF0F0_F0F0, 32'h0F0F_0F0F, "or_complement");
        check(`OP_XOR,  32'hFFFF_FFFF, 32'hFFFF_FFFF, "xor_self_ones");     // -> 0
        check(`OP_XOR,  32'hAAAA_AAAA, 32'h5555_5555, "xor_alt");           // -> all ones
        check(`OP_XOR,  32'h1234_5678, 32'h1234_5678, "xor_identical");     // -> 0
        check(`OP_XOR,  32'hDEAD_BEEF, 32'h0000_0000, "xor_with_zero");     // identity

        //==============================================================
        // DIRECTED: SHL / SHR (logical) corner cases
        //==============================================================
        check(`OP_SHL,  32'h0000_0001, 32'd0,  "shl_by_0_identity");
        check(`OP_SHL,  32'h0000_0001, 32'd1,  "shl_by_1");
        check(`OP_SHL,  32'h0000_0001, 32'd31, "shl_by_31_to_msb");
        check(`OP_SHL,  32'hFFFF_FFFF, 32'd4,  "shl_drop_high_bits");
        check(`OP_SHL,  32'h8000_0000, 32'd1,  "shl_msb_off_end");          // -> 0
        check(`OP_SHL,  32'h1234_5678, 32'd33, "shl_amt_low5_only");        // 33&31=1
        check(`OP_SHR,  32'h8000_0000, 32'd0,  "shr_by_0_identity");
        check(`OP_SHR,  32'h8000_0000, 32'd1,  "shr_logical_zero_fill");    // -> 0x40000000
        check(`OP_SHR,  32'h8000_0000, 32'd31, "shr_by_31");                // -> 1
        check(`OP_SHR,  32'hFFFF_FFFF, 32'd4,  "shr_logical_not_arith");    // -> 0x0FFFFFFF
        check(`OP_SHR,  32'h0000_0001, 32'd1,  "shr_off_end");              // -> 0
        check(`OP_SHR,  32'h1234_5678, 32'd32, "shr_amt_low5_only");        // 32&31=0 identity

        //==============================================================
        // DIRECTED: illegal / non-ALU opcodes -> default 0
        //==============================================================
        check(`OP_NOP,      32'hDEAD_BEEF, 32'hCAFE_BABE, "nop_default_zero");
        check(`OP_GEMM,     32'hDEAD_BEEF, 32'hCAFE_BABE, "gemm_not_alu_zero");
        check(`OP_SOFTMAX,  32'hDEAD_BEEF, 32'hCAFE_BABE, "softmax_not_alu_zero");
        check(8'hFF,        32'hDEAD_BEEF, 32'hCAFE_BABE, "unknown_op_zero");

        //==============================================================
        // CONSTRAINED-RANDOM: seeded, >=200 vectors per ALU opcode.
        // Each iteration exercises every ALU opcode with the same operands,
        // so 250 iterations -> 250 * 9 = 2250 random ALU checks (>>200).
        //==============================================================
        for (i = 0; i < 250; i = i + 1) begin
            in1 = $random;
            in2 = $random;
            check(`OP_RELU, in1, in2, "rand_relu");
            check(`OP_ADD,  in1, in2, "rand_add");
            check(`OP_ADDI, in1, in2, "rand_addi");
            check(`OP_SUB,  in1, in2, "rand_sub");
            check(`OP_AND,  in1, in2, "rand_and");
            check(`OP_OR,   in1, in2, "rand_or");
            check(`OP_XOR,  in1, in2, "rand_xor");
            check(`OP_SHL,  in1, in2, "rand_shl");
            check(`OP_SHR,  in1, in2, "rand_shr");
        end

        //==============================================================
        // Summary
        //==============================================================
        if (fail_count != 0) begin
            $display("VECTOR_ALU: %0d FAILURES out of %0d tests", fail_count,
                     pass_count + fail_count);
            $fatal(1, "vector_alu TB failed");
        end
        $display("ALL %0d TESTS PASSED", pass_count);
        $finish;
    end

endmodule
