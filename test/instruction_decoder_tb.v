`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// instruction_decoder_tb.v  --  self-checking TB for instruction_decoder.v
//----------------------------------------------------------------------------
// VERIFIES (docs/ISA.md §1, tpu_defs.vh §3):
//   * R-format field extraction: opcode/rA/rB/rC/imm12 for known encodings.
//   * I-format (LOADI) field extraction: opcode/rC_i/imm20 and the differing
//     rC bit-position between formats.
//   * Sign extension of imm12 (12->32) and imm20 (20->32), incl. the negative,
//     all-ones, max-positive, and zero boundaries.
//   * imm12 modifier sub-fields (tile_base, conv_out_base, conv_stride,
//     conv_pad, gs_stride_exp, dma_len).
//
// INDEPENDENT GOLDEN MODEL
//   The decoder is an EXACT integer bit-field extractor, so the golden is
//   compared BIT-EXACT (no tolerance).  Crucially the golden computes the
//   expected values a DIFFERENT WAY than the DUT:
//     * The DUT slices the instruction word with part-selects / sized
//       concatenations driven by named bit-position macros.
//     * The golden RECONSTRUCTS each field from the independently-chosen
//       generator inputs (op, ra, rb, rc, imm) that were PACKED into the word
//       by a separate packer, and computes sign-extension with signed-`real`
//       /signed-integer arithmetic (value = imm - 2^W when the sign bit is set)
//       rather than by bit replication.  A slicing bug in the DUT (wrong bit
//       range, swapped fields, wrong sign bit) is therefore caught because the
//       golden never touches the DUT's bit ranges.
//
// SELF-CONTAINED: no other src/ modules are instantiated (the decoder is pure
// combinational and has no external memory).  $fatal on any mismatch.
//============================================================================
module instruction_decoder_tb;

    //------------------------------------------------------------------------
    // DUT I/O
    //------------------------------------------------------------------------
    reg  [31:0]               instruction;

    wire [`OPCODE_W-1:0]      opcode;
    wire [`RF_IDX_W-1:0]      rA, rB, rC, rC_i;
    wire [`IMM12_W-1:0]       imm12;
    wire [`XLEN-1:0]          imm12_sext;
    wire [`IMM20_W-1:0]       imm20;
    wire [`XLEN-1:0]          imm20_sext;
    wire [`TM_IDX_W-1:0]      tile_base, conv_out_base;
    wire [1:0]                conv_stride, conv_pad, gs_stride_exp;
    wire [`IMM12_W-1:0]       dma_len;

    instruction_decoder dut (
        .instruction   (instruction),
        .opcode        (opcode),
        .rA            (rA),
        .rB            (rB),
        .rC            (rC),
        .rC_i          (rC_i),
        .imm12         (imm12),
        .imm12_sext    (imm12_sext),
        .imm20         (imm20),
        .imm20_sext    (imm20_sext),
        .tile_base     (tile_base),
        .conv_out_base (conv_out_base),
        .conv_stride   (conv_stride),
        .conv_pad      (conv_pad),
        .gs_stride_exp (gs_stride_exp),
        .dma_len       (dma_len)
    );

    //------------------------------------------------------------------------
    // Book-keeping
    //------------------------------------------------------------------------
    integer tests;
    integer fails;

    //------------------------------------------------------------------------
    // Independent packers (build the instruction word a known way).
    //   These intentionally use SHIFT/OR composition (a different syntactic
    //   construction than the DUT's named-macro part-selects) so the packer
    //   and the DUT do not share extraction code.
    //------------------------------------------------------------------------
    function [31:0] pack_r;
        input [7:0]  op;
        input [3:0]  ra;
        input [3:0]  rb;
        input [3:0]  rc;
        input [11:0] im12;
        begin
            pack_r = (op  << 24)
                   | ({28'd0, ra}   << 20)
                   | ({28'd0, rb}   << 16)
                   | ({28'd0, rc}   << 12)
                   | ({20'd0, im12});
        end
    endfunction

    function [31:0] pack_i;
        input [7:0]  op;
        input [3:0]  rc;        // I-format dest sits in [23:20]
        input [19:0] im20;
        begin
            pack_i = (op  << 24)
                   | ({28'd0, rc}  << 20)
                   | ({12'd0, im20});
        end
    endfunction

    //------------------------------------------------------------------------
    // Independent sign-extension golden (signed arithmetic, NOT bit replication)
    //   value = imm                 if sign bit == 0
    //   value = imm - 2^W           if sign bit == 1
    //   Returned as a 32-bit two's complement pattern.
    //------------------------------------------------------------------------
    function [31:0] sext_golden;
        input integer imm;     // unsigned magnitude of the W-bit field
        input integer width;   // field width W
        integer signed_val;
        integer half;          // 2^(W-1)
        integer full;          // 2^W
        begin
            half = (1 << (width-1));
            full = (1 << width);
            if (imm >= half) signed_val = imm - full;  // sign bit set
            else             signed_val = imm;
            sext_golden = signed_val[31:0];
        end
    endfunction

    //------------------------------------------------------------------------
    // Field check helpers (bit-exact; $fatal on mismatch).
    //------------------------------------------------------------------------
    task chk32;
        input [255:0] name;
        input [31:0]  got;
        input [31:0]  exp;
        begin
            tests = tests + 1;
            if (got !== exp) begin
                fails = fails + 1;
                $display("FAIL [%0s] instr=%h got=%h exp=%h",
                         name, instruction, got, exp);
                $fatal(1, "field mismatch");
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Full R-format check: independently reconstruct EVERY expected field from
    // the generator inputs and compare to the DUT outputs.
    //------------------------------------------------------------------------
    task check_r;
        input [7:0]  op;
        input [3:0]  ra;
        input [3:0]  rb;
        input [3:0]  rc;
        input [11:0] im12;
        reg   [4:0]  exp_tile, exp_convout;
        begin
            instruction = pack_r(op, ra, rb, rc, im12);
            #1;  // settle combinational logic

            chk32("opcode",      {24'd0, opcode},        {24'd0, op});
            chk32("rA",          {28'd0, rA},            {28'd0, ra});
            chk32("rB",          {28'd0, rB},            {28'd0, rb});
            chk32("rC",          {28'd0, rC},            {28'd0, rc});
            chk32("imm12",       {20'd0, imm12},         {20'd0, im12});
            chk32("imm12_sext",  imm12_sext,             sext_golden(im12, 12));

            // imm12 modifier sub-fields, reconstructed by independent masking.
            exp_tile    = im12 & 12'h01F;          // [4:0]
            exp_convout = (im12 >> 5) & 12'h01F;   // [9:5]
            chk32("tile_base",     {27'd0, tile_base},     {27'd0, exp_tile});
            chk32("conv_out_base", {27'd0, conv_out_base}, {27'd0, exp_convout});
            chk32("conv_stride",   {30'd0, conv_stride},   {30'd0, (im12 & 12'h003)});
            chk32("conv_pad",      {30'd0, conv_pad},      {30'd0, ((im12 >> 2) & 12'h003)});
            chk32("gs_stride_exp", {30'd0, gs_stride_exp}, {30'd0, (im12 & 12'h003)});
            chk32("dma_len",       {20'd0, dma_len},       {20'd0, im12});
        end
    endtask

    //------------------------------------------------------------------------
    // Full I-format (LOADI) check.
    //------------------------------------------------------------------------
    task check_i;
        input [7:0]  op;
        input [3:0]  rc;
        input [19:0] im20;
        begin
            instruction = pack_i(op, rc, im20);
            #1;

            chk32("i_opcode",     {24'd0, opcode},      {24'd0, op});
            chk32("i_rC_i",       {28'd0, rC_i},        {28'd0, rc});
            chk32("i_imm20",      {12'd0, imm20},       {12'd0, im20});
            chk32("i_imm20_sext", imm20_sext,           sext_golden(im20, 20));
        end
    endtask

    //------------------------------------------------------------------------
    // Random helpers
    //------------------------------------------------------------------------
    reg [31:0] r;
    integer    i;

    initial begin
        tests = 0;
        fails = 0;
        instruction = 32'd0;

        //--------------------------------------------------------------------
        // DIRECTED R-FORMAT corner cases
        //--------------------------------------------------------------------
        // All-zero instruction (NOP, every field zero).
        check_r(`OP_NOP,   4'h0, 4'h0, 4'h0, 12'h000);
        // All-ones fields (max opcode/regs/imm).
        check_r(8'hFF,     4'hF, 4'hF, 4'hF, 12'hFFF);
        // Distinct nibble pattern so a swapped field is caught.
        check_r(8'hA5,     4'h1, 4'h2, 4'h3, 12'hABC);
        check_r(8'h5A,     4'hC, 4'hD, 4'hE, 12'h123);

        // imm12 sign-extension boundaries.
        check_r(`OP_ADDI,  4'h1, 4'h0, 4'h2, 12'h000);  // 0      -> +0
        check_r(`OP_ADDI,  4'h1, 4'h0, 4'h2, 12'h7FF);  // +2047  (max pos)
        check_r(`OP_ADDI,  4'h1, 4'h0, 4'h2, 12'h800);  // -2048  (min neg)
        check_r(`OP_ADDI,  4'h1, 4'h0, 4'h2, 12'hFFF);  // -1     (all ones)
        check_r(`OP_ADDI,  4'h1, 4'h0, 4'h2, 12'h001);  // +1

        // Real opcodes exercising the modifier sub-fields.
        // CONV2D: out base in [9:5], stride [1:0], pad [3:2].
        //   imm12 = {2'b00, out_base=5'h1A, pad=2'b01, stride=2'b10}
        //         = 9'b? -> build: out_base<<5 | pad<<2 | stride
        check_r(`OP_CONV2D, 4'h2, 4'h3, 4'h4,
                (12'h1A << 5) | (2'b01 << 2) | 2'b10);
        // CONV2D max-ish: out_base=5'h1F, pad=1, stride=2.
        check_r(`OP_CONV2D, 4'h2, 4'h3, 4'h4,
                (12'h1F << 5) | (2'b01 << 2) | 2'b10);
        // GEMM: C tile base in imm12[4:0].
        check_r(`OP_GEMM,   4'h0, 4'h4, 4'h5, 12'h008);
        // SOFTMAX: p base in imm12[4:0].
        check_r(`OP_SOFTMAX, 4'h1, 4'h0, 4'h6, 12'h01F);
        // GATHER/SCATTER stride exponent in imm12[1:0].
        check_r(`OP_GATHER, 4'h1, 4'h2, 4'h7, 12'h003);
        check_r(`OP_SCATTER, 4'h1, 4'h2, 4'h0, 12'h002);
        // DMA length = full imm12.
        check_r(`OP_DMA,    4'h1, 4'h2, 4'h3, 12'hFFF);
        check_r(`OP_DMA,    4'h1, 4'h2, 4'h3, 12'h001);

        //--------------------------------------------------------------------
        // DIRECTED I-FORMAT (LOADI) corner cases
        //--------------------------------------------------------------------
        check_i(`OP_LOADI, 4'h0, 20'h00000);  // 0
        check_i(`OP_LOADI, 4'hF, 20'h7FFFF);  // +524287  (max pos)
        check_i(`OP_LOADI, 4'h5, 20'h80000);  // -524288  (min neg)
        check_i(`OP_LOADI, 4'hA, 20'hFFFFF);  // -1       (all ones)
        check_i(`OP_LOADI, 4'h3, 20'h00001);  // +1
        check_i(`OP_LOADI, 4'h7, 20'h12345);  // arbitrary positive
        check_i(`OP_LOADI, 4'hC, 20'hABCDE);  // arbitrary negative

        // Cross-check: rC field position DIFFERS between formats.
        //   For an I-format word, the R-format rC slice (instr[15:12]) is part
        //   of imm20, while rC_i (instr[23:20]) is the real LOADI dest.  Verify
        //   they are read from the correct, distinct positions.
        instruction = pack_i(`OP_LOADI, 4'h9, 20'hF00F0);
        #1;
        // rC_i must be 0x9 (the dest); R-format rC (instr[15:12]) must equal
        // bits [15:12] of imm20=0xF00F0 -> nibble at [15:12] = 0x0.
        chk32("fmt_rC_i",   {28'd0, rC_i}, 32'h00000009);
        chk32("fmt_rC_pos", {28'd0, rC},   {28'd0, (20'hF00F0 >> 12) & 20'hF});

        //--------------------------------------------------------------------
        // CONSTRAINED-RANDOM (seeded) -- >=200 vectors per format.
        //--------------------------------------------------------------------
        r = 32'h1234_5678;          // seed $random deterministically
        i = $random(r);             // prime the generator with the seed
        for (i = 0; i < 256; i = i + 1) begin
            check_r($random & 32'hFF,
                    $random & 32'hF,
                    $random & 32'hF,
                    $random & 32'hF,
                    $random & 32'hFFF);
        end
        for (i = 0; i < 256; i = i + 1) begin
            check_i($random & 32'hFF,
                    $random & 32'hF,
                    $random & 32'hFFFFF);
        end

        //--------------------------------------------------------------------
        // Report
        //--------------------------------------------------------------------
        if (fails != 0) begin
            $display("TESTS FAILED: %0d/%0d", fails, tests);
            $fatal(1, "instruction_decoder_tb FAILED");
        end
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end

endmodule
