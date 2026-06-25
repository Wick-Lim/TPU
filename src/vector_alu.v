`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// vector_alu.v  --  TPU v2.0 scalar ALU  (SPEC.md §5.5, docs/ISA.md §3)
//----------------------------------------------------------------------------
// PURPOSE
//   Pure-COMBINATIONAL 32-bit scalar ALU for the v2.0 scalar pipeline EX stage.
//   It performs the integer / bit / shift operations needed by real address,
//   loop-counter and control math driven by LOADI programs.  It REPLACES the
//   v1.5 "fake SOFTMAX" lane-normalization (true exp-softmax is now its own
//   real unit, src/softmax_unit.v); this module no longer has any softmax path.
//
// OPERATIONS (opcode constants from tpu_defs.vh, ISA.md §3)
//   OP_RELU (0x0B) : out = (signed in1 < 0) ? 0 : in1   -- rectified linear unit
//   OP_ADD  (0x04) : out = in1 + in2                    -- wrapping 32-bit add
//   OP_SUB  (0x05) : out = in1 - in2                    -- wrapping 32-bit sub
//   OP_AND  (0x06) : out = in1 & in2                    -- bitwise AND
//   OP_OR   (0x07) : out = in1 | in2                    -- bitwise OR
//   OP_XOR  (0x08) : out = in1 ^ in2                    -- bitwise XOR
//   OP_SHL  (0x09) : out = in1 << in2[4:0]              -- logical left shift
//   OP_SHR  (0x0A) : out = in1 >> in2[4:0]              -- logical right shift
//   OP_ADDI (0x0C) : out = in1 + in2                    -- add (in2 = sext imm12,
//                                                          supplied by datapath)
//   default        : out = 0                            -- not an ALU op
//
//   ADD and ADDI are arithmetically identical AT THIS BOUNDARY: the datapath
//   selects the second operand (register vs. sign-extended imm12) upstream and
//   presents it on in2, so both reduce to a 32-bit two's-complement sum here.
//
// FORMAT / SEMANTICS
//   All operands are bare 32-bit two's-complement words (XLEN = 32).  This unit
//   is NOT fixed-point/Q-format scaled: it is scalar/integer control math.  ADD
//   and SUB WRAP modulo 2^32 (no saturation -- saturation is a tensor-unit
//   policy, SPEC.md §1.3, and does not apply to scalar address arithmetic).
//   RELU treats in1 as a single signed 32-bit value (sign = bit 31).  SHR is
//   LOGICAL (zero-fill); there is no arithmetic-right ALU op in the v2.0 ISA.
//   Shift amount is in2[4:0] (0..31); upper bits of in2 are ignored for shifts.
//
// LATENCY / INTERFACE
//   Combinational (0 cycles, resolves within the EX cycle).  No state, no clock,
//   no reset -- nothing to reset.  Drop-in: ports are {opcode, in1, in2, out}.
//============================================================================
module vector_alu (
    input  wire [`OPCODE_W-1:0] opcode,   // 8-bit opcode (tpu_defs.vh)
    input  wire [`XLEN-1:0]     in1,      // operand A (rA)
    input  wire [`XLEN-1:0]     in2,      // operand B (rB) or sign-extended imm
    output reg  [`XLEN-1:0]     out       // result word
);

    // RELU sign test uses bit 31 of in1 directly (two's-complement sign bit);
    // no full signed copy is kept so every declared bit is consumed.
    wire in1_neg = in1[`XLEN-1];

    // Logical shifts use only the low 5 bits of in2 (0..31 positions); the
    // upper 27 bits are intentionally unused for shift ops.  Verilator would
    // flag those bits as UNUSED in a shift context, but they ARE used by the
    // add/sub/bitwise ops, so no lint_off is needed -- in2 is fully consumed
    // across the opcode set.  We slice explicitly here for shift clarity.
    wire [4:0] shamt = in2[4:0];

    always @(*) begin
        case (opcode)
            `OP_RELU: out = in1_neg ? {`XLEN{1'b0}} : in1;
            `OP_ADD:  out = in1 + in2;
            `OP_ADDI: out = in1 + in2;            // identical to ADD here
            `OP_SUB:  out = in1 - in2;
            `OP_AND:  out = in1 & in2;
            `OP_OR:   out = in1 | in2;
            `OP_XOR:  out = in1 ^ in2;
            `OP_SHL:  out = in1 << shamt;         // logical left  (zero-fill)
            `OP_SHR:  out = in1 >> shamt;         // logical right (zero-fill)
            default:  out = {`XLEN{1'b0}};        // not an ALU opcode
        endcase
    end

endmodule
