`include "tpu_defs.vh"
//============================================================================
// instruction_decoder.v  --  TPU v2.0 combinational instruction field extractor
//----------------------------------------------------------------------------
// ALGORITHM
//   Pure combinational (zero-latency) field extraction from the fixed 32-bit
//   instruction word, for BOTH instruction formats defined by the ISA
//   (docs/ISA.md §1, field positions in tpu_defs.vh §3).  The decoder extracts
//   every field UNCONDITIONALLY (format-agnostic slicing); each instruction
//   documents which fields it actually consumes.  No opcode-dependent muxing of
//   the raw fields is done here -- that is the control decoder's job -- so this
//   block has no state and cannot stall.
//
//   R-format (register operands), used by every op except LOADI:
//     [31:24] opcode | [23:20] rA | [19:16] rB | [15:12] rC | [11:0] imm12
//
//   I-format (load-immediate), used ONLY by LOADI (`OP_LOADI`):
//     [31:24] opcode | [23:20] rC | [19:0] imm20  (sign-extended to 32 bits)
//
//   Note the rC field position differs between formats:
//     * R-format rC = instr[15:12]
//     * I-format rC = instr[23:20]   (overlaps the R-format rA position)
//   Both are exposed separately (rC for R-format, rC_i for I-format) so the
//   caller selects by format without re-slicing.
//
//   Modifier sub-fields packed into imm12 (docs/ISA.md §3, §4):
//     * tile-base line index           imm12[4:0]   (GEMM C, SOFTMAX p, ATTN V,
//                                                     TLOAD/TSTORE line)
//     * CONV2D output tile base         imm12[9:5]
//     * CONV2D stride                   imm12[1:0]   (in {1,2})
//     * CONV2D pad                      imm12[3:2]   (in {0,1})
//     * GATHER/SCATTER stride exponent  imm12[1:0]   (effective stride = 1<<e)
//     * DMA copy length                 imm12        (1..4095 words)
//   These overlap intentionally (the same imm12 bits mean different things per
//   opcode); the decoder presents the slices, the consuming unit picks one.
//
// Q-FORMAT
//   This unit performs NO arithmetic on tensor data; it only sign-extends the
//   I-format 20-bit immediate to a 32-bit (XLEN) scalar value (two's complement,
//   plain integer -- NOT a Q-format quantity).  `imm12_sext` is likewise the
//   sign-extension of the 12-bit R-format immediate (used by ADDI/LOAD/STORE
//   displacement math).
//
// LATENCY
//   Combinational (0 cycles).  No clock, no reset, no state.
//
// INTERFACE
//   input  instruction[31:0] : the 32-bit instruction word
//   output opcode[7:0]       : instr[31:24]
//   output rA[3:0]           : R-format source A reg / tensor A tile-base src
//   output rB[3:0]           : R-format source B reg / tensor B tile-base src
//   output rC[3:0]           : R-format dest reg / status dest (instr[15:12])
//   output rC_i[3:0]         : I-format (LOADI) dest reg (instr[23:20])
//   output imm12[11:0]       : R-format 12-bit immediate (raw)
//   output imm12_sext[31:0]  : imm12 sign-extended to 32 bits
//   output imm20[19:0]       : I-format 20-bit immediate (raw)
//   output imm20_sext[31:0]  : imm20 sign-extended to 32 bits (LOADI value)
//   output tile_base[4:0]    : imm12[4:0]  generic tile-base line index
//   output conv_out_base[4:0]: imm12[9:5]  CONV2D output tile base
//   output conv_stride[1:0]  : imm12[1:0]  CONV2D stride field
//   output conv_pad[1:0]     : imm12[3:2]  CONV2D pad field
//   output gs_stride_exp[1:0]: imm12[1:0]  GATHER/SCATTER stride exponent
//   output dma_len[11:0]     : imm12       DMA copy length
//============================================================================
module instruction_decoder (
    input  wire [31:0]               instruction,

    // --- common opcode / register fields ---
    output wire [`OPCODE_W-1:0]      opcode,        // instr[31:24]
    output wire [`RF_IDX_W-1:0]      rA,            // instr[23:20]
    output wire [`RF_IDX_W-1:0]      rB,            // instr[19:16]
    output wire [`RF_IDX_W-1:0]      rC,            // instr[15:12]  (R-format)
    output wire [`RF_IDX_W-1:0]      rC_i,          // instr[23:20]  (I-format)

    // --- immediates (raw + sign-extended) ---
    output wire [`IMM12_W-1:0]       imm12,         // instr[11:0]
    output wire [`XLEN-1:0]          imm12_sext,    // sign-extend(imm12)->32b
    output wire [`IMM20_W-1:0]       imm20,         // instr[19:0]
    output wire [`XLEN-1:0]          imm20_sext,    // sign-extend(imm20)->32b

    // --- imm12 modifier sub-fields (overlapping, opcode selects meaning) ---
    output wire [`TM_IDX_W-1:0]      tile_base,     // imm12[4:0]
    output wire [`TM_IDX_W-1:0]      conv_out_base, // imm12[9:5]
    output wire [1:0]                conv_stride,   // imm12[1:0]
    output wire [1:0]                conv_pad,      // imm12[3:2]
    output wire [1:0]                gs_stride_exp, // imm12[1:0]
    output wire [`IMM12_W-1:0]       dma_len        // imm12  (copy length)
);

    //------------------------------------------------------------------------
    // Common opcode + register fields (positions from tpu_defs.vh §3).
    //------------------------------------------------------------------------
    assign opcode = instruction[`OPCODE_HI:`OPCODE_LO];   // [31:24]
    assign rA     = instruction[`RA_HI:`RA_LO];           // [23:20]
    assign rB     = instruction[`RB_HI:`RB_LO];           // [19:16]
    assign rC     = instruction[`RC_HI:`RC_LO];           // [15:12]  R-format

    // I-format LOADI destination shares the rA bit-positions [23:20].
    assign rC_i   = instruction[`RA_HI:`RA_LO];           // [23:20]  I-format

    //------------------------------------------------------------------------
    // Immediates -- raw slices and sign-extensions.
    //------------------------------------------------------------------------
    assign imm12 = instruction[`IMM12_HI:`IMM12_LO];      // [11:0]
    assign imm20 = instruction[`IMM20_HI:`IMM20_LO];      // [19:0]

    // Sign-extend imm12 (12 bits) to XLEN (32 bits): replicate bit[11].
    assign imm12_sext =
        { {(`XLEN-`IMM12_W){imm12[`IMM12_W-1]}}, imm12 };

    // Sign-extend imm20 (20 bits) to XLEN (32 bits): replicate bit[19].
    // This is the value LOADI writes to rC.
    assign imm20_sext =
        { {(`XLEN-`IMM20_W){imm20[`IMM20_W-1]}}, imm20 };

    //------------------------------------------------------------------------
    // imm12 modifier sub-fields (overlapping; the consuming unit picks one).
    //------------------------------------------------------------------------
    assign tile_base     = imm12[`TM_IDX_W-1:0];          // imm12[4:0]
    assign conv_out_base = imm12[9:5];                    // CONV2D out base
    assign conv_stride   = imm12[1:0];                    // CONV2D stride
    assign conv_pad      = imm12[3:2];                    // CONV2D pad
    assign gs_stride_exp = imm12[1:0];                    // GATHER/SCATTER exp
    assign dma_len       = imm12;                         // DMA length

endmodule
