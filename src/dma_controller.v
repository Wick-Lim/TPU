`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// dma_controller.v  --  TPU v2.0 multi-word DMA copy FSM        (SPEC §2, §5.5)
//----------------------------------------------------------------------------
// ALGORITHM
//   A real multi-word block copy through DMEM (NOT v1.5's single-word stub):
//
//        for (i = 0; i < len; i = i + 1)
//            DMEM[dst + i] = DMEM[src + i];
//
//   One 32-bit word moved per cycle.  The engine does NOT contain the data
//   memory; it DRIVES external DMEM access ports (combinational read,
//   synchronous write) so the surrounding datapath (or, in the unit TB, a
//   behavioral DMEM model) owns the storage and there is no module-internal
//   aliasing.  The source read of DMEM[src+i] issued in cycle T returns
//   `rd_data` combinationally that same cycle; the engine registers it and
//   commits DMEM[dst+i] in cycle T+1.  This 1-deep read->write pipeline both
//   honours the synchronous-write contract and lets the read address advance
//   one word every cycle.
//
// Q-FORMAT
//   None.  This is an EXACT integer (raw 32-bit word) block move: no fixed-
//   point arithmetic, no rounding, no saturation.  Words are copied bit-exact.
//
// HANDSHAKE / LATENCY                                              (SPEC §3.3)
//   `start` (a 1-cycle pulse, sampled only in IDLE) latches {src,dst,len} and
//   begins the copy.  `busy` is high while transferring; `done` is a 1-cycle
//   pulse on the cycle the FINAL word is written (or, for len=0, the ack).
//   For a transfer of L words the engine takes  L + 2  cycles from the start
//   pulse to the done pulse:
//        cycle 0  : start sampled, descriptor latched, READ src+0 issued
//        cycle 1  : word0 in pipe -> WRITE dst+0; READ src+1 issued (busy)
//        ...       one READ issued and one (previous) word WRITTEN per cycle
//        cycle L+1: WRITE dst+(L-1) (final), done=1, busy->0
//      (start pulse is cycle 0; done pulse is cycle L+1 => L+2 cycles total.)
//   Edge cases:
//        len = 0 : no-op.  No DMEM write ever asserts.  done pulses the cycle
//                  after start (a 2-cycle start->done ack), busy stays 0.
//        len = 1 : exactly one word copied; done at start+2  (L+2, L=1).
//   Overlapping src/dst regions are copied in ascending order (i = 0,1,...),
//   identical to the C loop above; the golden TB models the same ordering.
//
// INTERFACE
//   clk, rst                       : clock; synchronous active-high reset (ALL state).
//   start                          : begin a copy of the presented descriptor.
//   src_addr[DMEM_ADDR_W-1:0]      : base DMEM word index of the source run.
//   dst_addr[DMEM_ADDR_W-1:0]      : base DMEM word index of the dest   run.
//   len[DMEM_ADDR_W:0]             : number of words to copy (0..DMEM_DEPTH).
//   busy                           : engine is transferring (stall signal).
//   done                           : 1-cycle pulse when the copy has retired.
//   words[DMEM_ADDR_W:0]           : # words moved (== latched len) for rC
//                                    write-back (SPEC OP_DMA: rC = #words).
//   -- DMEM access ports the engine DRIVES (TB / datapath owns the memory) --
//   rd_addr[DMEM_ADDR_W-1:0]       : combinational DMEM read address (source).
//   rd_data[XLEN-1:0]              : DMEM read data for rd_addr (combinational).
//   wr_en                          : DMEM write strobe (one word/cycle).
//   wr_addr[DMEM_ADDR_W-1:0]       : DMEM write address (destination).
//   wr_data[XLEN-1:0]              : DMEM write data (the copied word).
//
// SYNTHESIS NOTES
//   All state has a synchronous reset; every reg is assigned on every path of
//   the clocked block (no inferred latch); no combinational loop; no real /
//   $display / $random / initial in this module.  `len`, `total`, `rd_idx` and
//   `wr_idx` are one bit wider than the address so the full count DMEM_DEPTH
//   (256) and the terminal compare against `total` fit without truncation.
//============================================================================
module dma_controller (
    input  wire                      clk,
    input  wire                      rst,

    // descriptor + handshake
    input  wire                      start,
    input  wire [`DMEM_ADDR_W-1:0]   src_addr,
    input  wire [`DMEM_ADDR_W-1:0]   dst_addr,
    input  wire [`DMEM_ADDR_W:0]     len,        // 0..DMEM_DEPTH
    output reg                        busy,
    output reg                        done,
    output reg  [`DMEM_ADDR_W:0]      words,      // # words moved (== len) -> rC

    // DMEM access ports this engine drives (memory lives OUTSIDE the module)
    output reg  [`DMEM_ADDR_W-1:0]   rd_addr,    // source read address (comb use)
    input  wire [`XLEN-1:0]          rd_data,    // source read data (combinational)
    output reg                        wr_en,      // destination write strobe
    output reg  [`DMEM_ADDR_W-1:0]   wr_addr,    // destination write address
    output reg  [`XLEN-1:0]          wr_data     // destination write data
);

    // --- one-hot-ish FSM state --------------------------------------------
    localparam [0:0] S_IDLE = 1'b0,  // waiting for start
                     S_RUN  = 1'b1;  // issuing reads / committing writes
    reg state;

    // --- latched descriptor + iterators -----------------------------------
    //   src_base/dst_base : run bases latched at start.
    //   total             : latched length (== words to move).
    //   rd_idx            : count of source reads ISSUED so far (next offset).
    //   pipe_vld          : a read word is sitting in the 1-deep pipeline.
    //   wr_off_q          : destination offset for the word now in the pipe.
    reg [`DMEM_ADDR_W-1:0] src_base;
    reg [`DMEM_ADDR_W-1:0] dst_base;
    reg [`DMEM_ADDR_W:0]   total;
    reg [`DMEM_ADDR_W:0]   rd_idx;
    reg                    pipe_vld;
    reg [`DMEM_ADDR_W-1:0] wr_off_q;

    // Constant-width "one" increment for the (DMEM_ADDR_W+1)-wide counters.
    localparam [`DMEM_ADDR_W:0] CNT_ONE  = { {(`DMEM_ADDR_W){1'b0}}, 1'b1 };
    localparam [`DMEM_ADDR_W:0] CNT_ZERO = {(`DMEM_ADDR_W+1){1'b0}};

    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            busy     <= 1'b0;
            done     <= 1'b0;
            words    <= CNT_ZERO;
            rd_addr  <= {`DMEM_ADDR_W{1'b0}};
            wr_en    <= 1'b0;
            wr_addr  <= {`DMEM_ADDR_W{1'b0}};
            wr_data  <= {`XLEN{1'b0}};
            src_base <= {`DMEM_ADDR_W{1'b0}};
            dst_base <= {`DMEM_ADDR_W{1'b0}};
            total    <= CNT_ZERO;
            rd_idx   <= CNT_ZERO;
            pipe_vld <= 1'b0;
            wr_off_q <= {`DMEM_ADDR_W{1'b0}};
        end else begin
            // Defaults each cycle (assigned on every path -> no latch).
            done  <= 1'b0;
            wr_en <= 1'b0;

            case (state)
                // -------------------------------------------------- IDLE ---
                S_IDLE: begin
                    busy     <= 1'b0;
                    pipe_vld <= 1'b0;
                    if (start) begin
                        src_base <= src_addr;
                        dst_base <= dst_addr;
                        total    <= len;
                        words    <= len;                 // visible result (# words)
                        if (len == CNT_ZERO) begin
                            // len == 0 : no-op.  Pulse done next cycle, no write.
                            busy  <= 1'b0;
                            done  <= 1'b1;
                            state <= S_IDLE;
                            rd_idx   <= CNT_ZERO;
                            pipe_vld <= 1'b0;
                        end else begin
                            // Issue the first source read this cycle (src + 0).
                            busy     <= 1'b1;
                            rd_addr  <= src_addr;
                            wr_off_q <= {`DMEM_ADDR_W{1'b0}};   // dst offset 0
                            rd_idx   <= CNT_ONE;                // 1 read issued
                            pipe_vld <= 1'b1;
                            state    <= S_RUN;
                        end
                    end
                end

                // --------------------------------------------------- RUN ---
                // Each cycle: commit the word read last cycle (pipe_vld), and
                // issue the next source read if any remain.
                S_RUN: begin
                    // Commit the pipelined word to its destination.
                    if (pipe_vld) begin
                        wr_en   <= 1'b1;
                        wr_addr <= dst_base + wr_off_q;
                        wr_data <= rd_data;
                    end

                    if (rd_idx < total) begin
                        // More words to read: issue next source read.
                        busy     <= 1'b1;
                        rd_addr  <= src_base + rd_idx[`DMEM_ADDR_W-1:0];
                        wr_off_q <= rd_idx[`DMEM_ADDR_W-1:0];
                        rd_idx   <= rd_idx + CNT_ONE;
                        pipe_vld <= 1'b1;
                        state    <= S_RUN;
                    end else begin
                        // All reads issued; the write committed THIS cycle (if
                        // pipe_vld) is the final word -> retire.
                        busy     <= 1'b0;
                        done     <= 1'b1;
                        pipe_vld <= 1'b0;
                        state    <= S_IDLE;
                    end
                end

                default: begin
                    state    <= S_IDLE;
                    busy     <= 1'b0;
                    pipe_vld <= 1'b0;
                end
            endcase
        end
    end

endmodule
