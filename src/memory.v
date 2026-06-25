`include "tpu_defs.vh"
`timescale 1ns/1ps
//============================================================================
// memory  --  DMEM scalar data memory (TPU v2.0)                  (SPEC §2, §5.5)
//----------------------------------------------------------------------------
// 256 x 32-bit word data SRAM model (widened from the v1.5 16-word DMEM).
//
//   * Depth : `DMEM_DEPTH = 256 words  (8-bit address, `DMEM_ADDR_W = 8)
//   * Width : `XLEN       = 32-bit words (plain integer / scalar domain)
//
// Semantics (SPEC.md §2: "all three storages: combinational read, synchronous
// write, synchronous reset to 0"):
//   * READ  is COMBINATIONAL on BOTH ports, so a word written on one clock edge
//     is observable in the same MEM cycle, and so the GATHER / DMA effective-
//     address math can read its word inside the single MEM stage.
//   * A SECOND combinational read port (raddr2/rdata2) lets the DMA engine read
//     its source word while the primary port services LOAD / STORE / GATHER /
//     SCATTER, all within the same MEM cycle.
//   * WRITE is SYNCHRONOUS (one write port, on the rising clock edge).
//   * RESET is SYNCHRONOUS and clears every word to 0 (no X after reset, fully
//     synthesizable -- no `initial`, no asynchronous reset).
//
// Q-format: NONE -- DMEM holds raw 32-bit scalar words (no fixed-point scaling
// is applied here; Q-format interpretation lives in the tensor units that
// consume tiles loaded from DMEM via TLOAD/TSTORE).
//
// Latency: combinational read (0-cycle); write/reset take effect on the NEXT
// rising edge (1-cycle visibility).
//
// Interface:
//   clk, rst                         clock and synchronous reset
//   addr        [`DMEM_ADDR_W-1:0]   primary port address (LOAD/STORE/GATHER)
//   data_in     [`XLEN-1:0]          primary write data
//   write_enable                     1 => synchronous write of data_in @ addr
//   data_out    [`XLEN-1:0]          primary combinational read data @ addr
//   raddr2      [`DMEM_ADDR_W-1:0]   secondary read port address (DMA source)
//   rdata2      [`XLEN-1:0]          secondary combinational read data @ raddr2
//============================================================================
module memory (
    input  wire                     clk,
    input  wire                     rst,
    input  wire [`DMEM_ADDR_W-1:0]  addr,         // primary port (LOAD/STORE/GATHER)
    input  wire [`XLEN-1:0]         data_in,      // primary write data
    input  wire                     write_enable,
    output wire [`XLEN-1:0]         data_out,     // primary read data  (combinational)
    input  wire [`DMEM_ADDR_W-1:0]  raddr2,       // secondary read port (DMA source)
    output wire [`XLEN-1:0]         rdata2        // secondary read data (combinational)
);
    // 256 x 32-bit storage array.
    reg [`XLEN-1:0] sram [0:`DMEM_DEPTH-1];

    // Reset-clear loop index.  `integer` (32-bit) so the loop bound DMEM_DEPTH
    // (256) is reached without an 8-bit counter wrapping.
    integer i;

    //------------------------------------------------------------------
    // Combinational dual-port read.
    //------------------------------------------------------------------
    assign data_out = sram[addr];
    assign rdata2   = sram[raddr2];

    //------------------------------------------------------------------
    // Synchronous reset (clear all) / synchronous single-port write.
    // Reset has priority so post-reset state is fully defined (all 0).
    //------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < `DMEM_DEPTH; i = i + 1)
                sram[i] <= {`XLEN{1'b0}};
        end else if (write_enable) begin
            sram[addr] <= data_in;
        end
    end
endmodule
