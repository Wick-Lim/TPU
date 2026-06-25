`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// register_file.v  --  TPU v2.0 scalar register file (SPEC.md §3.1, §5.5)
//----------------------------------------------------------------------------
// ALGORITHM / FUNCTION
//   A 16 x 32-bit scalar register file (`RF_REGS` = 16, `XLEN` = 32) that backs
//   the v2.0 scalar pipeline's operand domain.  It provides:
//     * THREE combinational (asynchronous) read ports
//         - read_data1 / read_data2 : the rA / rB operands resolved in ID, so
//           the forwarding/timing model can observe operands in the same cycle
//           the instruction is decoded.
//         - read_data3              : a third operand port (used e.g. to read a
//           tensor op's IMM-selected source register).
//     * ONE synchronous write port (committed in WB).
//     * Synchronous reset that clears every architectural register to 0, so
//       every readable value is defined immediately after reset (no X, no
//       reliance on a non-synthesizable `initial`).
//
//   HARDWIRED-ZERO r0 (SPEC.md §2, §3.1):
//     * Reads of index 0 ALWAYS return 32'h0000_0000, regardless of any prior
//       write.  This is enforced at every read port by a combinational mux on
//       the address, so it holds even on the same cycle as (an ignored) write.
//     * Writes that target index 0 are IGNORED: the write enable is gated so
//       registers[0] is never updated by the write path.  registers[0] is held
//       at 0 by reset as well, so it is doubly safe; the read mux is the
//       architectural guarantee.
//
// Q-FORMAT
//   None -- this is a raw 32-bit scalar word store (XLEN).  The contents are
//   opcode-defined integers / addresses; the RF imposes no fixed-point format.
//
// LATENCY
//   Reads : 0 cycles (purely combinational).
//   Writes: 1 cycle  (registered; visible on the read ports the cycle AFTER
//           the posedge that commits the write).
//
// INTERFACE
//   clk, rst                          : clock, synchronous active-high reset.
//   read_addr1/2/3 [RF_IDX_W-1:0]     : three independent read indices.
//   write_addr     [RF_IDX_W-1:0]     : write index (ignored when == 0).
//   write_data     [XLEN-1:0]         : write payload.
//   write_enable                      : commit the write on the next posedge.
//   read_data1/2/3 [XLEN-1:0]         : combinational read results (0 for r0).
//
// SYNTHESIS
//   Synchronous reset on all state; single clocked block with a write under a
//   gated enable (no inferred latch); no combinational loops; no
//   non-synthesizable constructs.  File name == module name (DECLFILENAME).
//============================================================================
module register_file #(
    parameter integer REGS   = `RF_REGS,   // number of architectural registers
    parameter integer IDX_W  = `RF_IDX_W,  // register index width
    parameter integer WORD_W = `XLEN       // scalar word width
) (
    input  wire              clk,
    input  wire              rst,
    input  wire [IDX_W-1:0]  read_addr1,
    input  wire [IDX_W-1:0]  read_addr2,
    input  wire [IDX_W-1:0]  read_addr3,
    input  wire [IDX_W-1:0]  write_addr,
    input  wire [WORD_W-1:0] write_data,
    input  wire              write_enable,
    output wire [WORD_W-1:0] read_data1,
    output wire [WORD_W-1:0] read_data2,
    output wire [WORD_W-1:0] read_data3
);
    // Architectural state.  registers[0] is the hardwired-zero r0; it is never
    // written (write enable is gated) and reads of it are forced to 0 below.
    reg [WORD_W-1:0] registers [0:REGS-1];
    integer i;

    // ----- Combinational read ports with r0 hardwired to zero -----
    // The address-0 mux is the architectural guarantee: even though reset and
    // the write gate already keep registers[0] at 0, forcing the read result
    // here makes r0==0 independent of the storage cell's value.
    assign read_data1 = (read_addr1 == {IDX_W{1'b0}}) ? {WORD_W{1'b0}}
                                                      : registers[read_addr1];
    assign read_data2 = (read_addr2 == {IDX_W{1'b0}}) ? {WORD_W{1'b0}}
                                                      : registers[read_addr2];
    assign read_data3 = (read_addr3 == {IDX_W{1'b0}}) ? {WORD_W{1'b0}}
                                                      : registers[read_addr3];

    // ----- Synchronous reset + synchronous, r0-protected write -----
    // Write enable is gated by (write_addr != 0) so a write to r0 is silently
    // ignored, never disturbing the hardwired-zero register.
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < REGS; i = i + 1) begin
                registers[i] <= {WORD_W{1'b0}};
            end
        end else if (write_enable && (write_addr != {IDX_W{1'b0}})) begin
            registers[write_addr] <= write_data;
        end
    end
endmodule
