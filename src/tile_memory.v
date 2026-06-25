`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// tile_memory  --  TPU v2.0 tensor tile memory  (SPEC.md §1.1, §2, §5.5)
//----------------------------------------------------------------------------
// The tensor operand store: TM_LINES (32) x LINE_W (128) bits.  One 128-bit
// TM line = 4 x 32-bit lanes (one matrix row or four vector elements; see
// SPEC §1.1).  This module is the shared backing store that every tensor unit
// (GEMM / CONV2D / SOFTMAX / ATTENTION) reads operands from and writes results
// to.  It exposes raw ACCESS PORTS only -- it instantiates nothing and is in
// turn instantiated once and wired to the tensor FSMs.
//
// ALGORITHM / BEHAVIOR
//   * TWO independent COMBINATIONAL (asynchronous) read ports
//       raddr1 -> rdata1 , raddr2 -> rdata2
//     Combinational read lets a tensor FSM present an operand line index and
//     consume the 128-bit line within the same cycle (SPEC §2: "combinational
//     read so the tensor FSMs can read operands within a cycle"), and lets the
//     two read ports service two distinct operands (e.g. an A row and a B row)
//     in parallel with full independence.
//   * ONE SYNCHRONOUS write port  (we, waddr, wdata).  A line written on a
//     posedge is observable on the combinational read ports from the next
//     cycle onward.  Write has lower priority than reset.
//   * SYNCHRONOUS RESET clears every line to 0 (no X after reset; no
//     non-synthesizable `initial`).  Reset dominates write.
//
// Q-FORMAT
//   This memory is FORMAT-AGNOSTIC: it stores raw 128-bit lines.  The Q-format
//   of the packed lanes (Q7.8 element data, Q0.16 probabilities, etc.) is owned
//   by the tensor unit that produced the line, never interpreted here.
//
// LATENCY
//   Reads : combinational (0-cycle, same-cycle visible).
//   Writes: synchronous (visible on reads the cycle AFTER the posedge).
//
// INTERFACE (all addresses are TM line indices, TM_IDX_W = 5 bits wide)
//   clk, rst                         clock / synchronous active-high reset
//   raddr1  [TM_IDX_W-1:0]  (in)     read port 1 line index   (combinational)
//   rdata1  [LINE_W-1:0]    (out)    read port 1 line data
//   raddr2  [TM_IDX_W-1:0]  (in)     read port 2 line index   (combinational)
//   rdata2  [LINE_W-1:0]    (out)    read port 2 line data
//   we                      (in)     write enable
//   waddr   [TM_IDX_W-1:0]  (in)     write port line index
//   wdata   [LINE_W-1:0]    (in)     write port line data
//
// SYNTHESIZABILITY
//   Single clocked always block, every state element (the line array) assigned
//   on every path (reset or write-or-hold), no latches, no combinational loops,
//   no non-synthesizable constructs.  Passes verilator --lint-only -Wall.
//============================================================================
module tile_memory (
    input  wire                  clk,
    input  wire                  rst,

    // Read port 1 (combinational).
    input  wire [`TM_IDX_W-1:0]  raddr1,
    output wire [`LINE_W-1:0]    rdata1,

    // Read port 2 (combinational, fully independent of port 1).
    input  wire [`TM_IDX_W-1:0]  raddr2,
    output wire [`LINE_W-1:0]    rdata2,

    // Write port (synchronous).
    input  wire                  we,
    input  wire [`TM_IDX_W-1:0]  waddr,
    input  wire [`LINE_W-1:0]    wdata
);

    // 32 x 128-bit tile-line storage.
    reg [`LINE_W-1:0] lines [0:`TM_LINES-1];
    integer i;

    // --- Combinational (asynchronous) reads: same-cycle operand fetch. ---
    assign rdata1 = lines[raddr1];
    assign rdata2 = lines[raddr2];

    // --- Synchronous reset (dominates) + synchronous write. ---
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < `TM_LINES; i = i + 1)
                lines[i] <= {`LINE_W{1'b0}};
        end else if (we) begin
            lines[waddr] <= wdata;
        end
    end
endmodule
