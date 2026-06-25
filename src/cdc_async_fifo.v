`timescale 1ns/1ps
//============================================================================
// cdc_async_fifo.v  --  textbook asynchronous (dual-clock) FIFO for CDC
//----------------------------------------------------------------------------
// ALGORITHM / FUNCTION
//   A classic Cummings-style asynchronous FIFO that safely moves DATA_W-bit
//   words from an independent WRITE clock domain (wclk/wrst_n) to an
//   independent READ clock domain (rclk/rrst_n).  Depth is 2**ADDR_W words
//   (ADDR_W = 4 => depth 16, DATA_W = 32 by default).
//
//   STRUCTURE (the standard CDC-safe async-FIFO, Cummings SNUG2002):
//     * A dual-port RAM: written synchronously on wclk, read synchronously on
//       rclk (registered read-data output -> well-defined timing, no async RAM
//       read glitches crossing into the read domain's flops).
//     * Binary read/write pointers are ONE BIT WIDER than the address (ADDR_W+1)
//       so that "full" and "empty" can be distinguished even when the low
//       ADDR_W bits of the two pointers are equal: the extra MSB records how
//       many times each pointer has wrapped.
//     * Each binary pointer is converted to GRAY code locally.  Only the GRAY
//       pointer crosses the clock boundary, through a 2-FF synchronizer, into
//       the opposite domain.  Gray code changes exactly ONE bit per increment,
//       so even if the synchronizer samples mid-transition the captured value
//       is either the old or the new pointer -- never a corrupt intermediate.
//       (A binary pointer could glitch on multiple bits at once and is NEVER
//       crossed directly -- that would be metastability-unsafe.)
//
//   FULL / EMPTY ARE REGISTERED (no combinational loop):
//     The pointer increments are gated by the *current* (registered) full/empty
//     flops, and full/empty are recomputed each clock from the *next* gray
//     pointer versus the synchronized opposite gray pointer.  Because the
//     gating signal (the flop output) is NOT in the same combinational cone as
//     the value that recomputes it, there is no feedback loop -- this is the
//     textbook decomposition that the naive "full feeds wbin_next feeds full"
//     version would violate.
//
//     EMPTY: next read-gray equals the synchronized write-gray exactly
//            (read pointer has caught up to write pointer).
//     FULL : next write-gray equals the synchronized read-gray with the top
//            TWO bits inverted -- the gray-coded statement that the write
//            pointer is exactly one full lap (DEPTH words) ahead of the read
//            pointer (low bits equal, both wrap-MSBs differ).
//
//   Writes are accepted only when (wr_en & ~full); reads only when
//   (rd_en & ~empty).  The pointers therefore never advance past one another,
//   so no word is ever lost, duplicated, or overwritten in flight.
//
// LATENCY / CDC
//   A written word becomes visible to the read side after the write gray
//   pointer propagates through the read domain's 2-FF synchronizer (=> a couple
//   of rclk edges of latency on `empty` deassertion); symmetric for `full`.
//   This conservative latency can only make the FIFO look MORE full / LESS empty
//   than reality, so it is always safe (never permits over/underflow).
//   The RAM read output is REGISTERED: when (rd_en & ~empty) is accepted at an
//   rclk edge the corresponding word appears on rd_data after that edge.
//
// INTERFACE
//   Write domain: wclk, wrst_n (sync, active-low), wr_en, wr_data[DATA_W-1:0],
//                 full.
//   Read  domain: rclk, rrst_n (sync, active-low), rd_en, rd_data[DATA_W-1:0],
//                 empty.
//   NOTE: resets are ACTIVE-LOW here (the conventional CDC-FIFO convention,
//   reflected in the *_n suffix) and are sampled SYNCHRONOUSLY to their own
//   clock -- every flop is assigned on every path, so there is no inferred latch
//   and no async-reset recovery hazard.
//
// SYNTHESIS
//   Two independent clocked domains; every reg assigned on every path; gray
//   pointers crossed only through 2-FF synchronizers; NO combinational loops
//   (full/empty are flops; the gray<->bin conversions are feed-forward); no
//   non-synthesizable constructs.  File name == module name.
//============================================================================
module cdc_async_fifo #(
    parameter integer DATA_W = 32,  // data word width
    parameter integer ADDR_W = 4    // address width => depth = 2**ADDR_W
) (
    // ---- write clock domain ----
    input  wire              wclk,
    input  wire              wrst_n,   // synchronous, active-low
    input  wire              wr_en,
    input  wire [DATA_W-1:0] wr_data,
    output reg               full,
    // ---- read clock domain ----
    input  wire              rclk,
    input  wire              rrst_n,   // synchronous, active-low
    input  wire              rd_en,
    output reg  [DATA_W-1:0] rd_data,
    output reg               empty
);
    localparam integer DEPTH = (1 << ADDR_W);

    // ---- dual-port storage ----
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    // ========================================================================
    // WRITE-DOMAIN pointers (binary + gray), ADDR_W+1 bits wide.
    //   The increment is gated by the REGISTERED `full` (a flop output), so it
    //   does NOT depend combinationally on the freshly-computed full -> no loop.
    // ========================================================================
    reg  [ADDR_W:0] wbin;          // write pointer, binary
    reg  [ADDR_W:0] wgray;         // write pointer, gray (this is what crosses)
    wire            do_write = wr_en & ~full;
    wire [ADDR_W:0] wbin_next  = wbin + {{ADDR_W{1'b0}}, do_write};
    wire [ADDR_W:0] wgray_next = (wbin_next >> 1) ^ wbin_next;

    // ========================================================================
    // READ-DOMAIN pointers (binary + gray), ADDR_W+1 bits wide.
    //   Gated by the REGISTERED `empty` flop output -> no combinational loop.
    // ========================================================================
    reg  [ADDR_W:0] rbin;          // read pointer, binary
    reg  [ADDR_W:0] rgray;         // read pointer, gray (this is what crosses)
    wire            do_read = rd_en & ~empty;
    wire [ADDR_W:0] rbin_next  = rbin + {{ADDR_W{1'b0}}, do_read};
    wire [ADDR_W:0] rgray_next = (rbin_next >> 1) ^ rbin_next;

    // ---- 2-FF synchronizers ----
    reg  [ADDR_W:0] rgray_wq1, rgray_wq2;   // read gray  -> write domain (full)
    reg  [ADDR_W:0] wgray_rq1, wgray_rq2;   // write gray -> read  domain (empty)

    // next-state full / empty (combinational, feed-forward only)
    wire full_next  = (wgray_next == {~rgray_wq2[ADDR_W:ADDR_W-1],
                                       rgray_wq2[ADDR_W-2:0]});
    wire empty_next = (rgray_next == wgray_rq2);

    //========================================================================
    // WRITE DOMAIN (wclk)
    //========================================================================
    // Synchronize the read gray pointer into the write domain (2 FF).
    always @(posedge wclk) begin
        if (!wrst_n) begin
            rgray_wq1 <= {(ADDR_W+1){1'b0}};
            rgray_wq2 <= {(ADDR_W+1){1'b0}};
        end else begin
            rgray_wq1 <= rgray;
            rgray_wq2 <= rgray_wq1;
        end
    end

    // Write pointer update + registered FULL.
    always @(posedge wclk) begin
        if (!wrst_n) begin
            wbin  <= {(ADDR_W+1){1'b0}};
            wgray <= {(ADDR_W+1){1'b0}};
            full  <= 1'b0;
        end else begin
            wbin  <= wbin_next;
            wgray <= wgray_next;
            full  <= full_next;
        end
    end

    // RAM write port (synchronous on wclk).  The low ADDR_W bits of the binary
    // write pointer address the memory.  Guarded by (~full) so a word is never
    // overwritten before it has been read.
    always @(posedge wclk) begin
        if (do_write) begin
            mem[wbin[ADDR_W-1:0]] <= wr_data;
        end
    end

    //========================================================================
    // READ DOMAIN (rclk)
    //========================================================================
    // Synchronize the write gray pointer into the read domain (2 FF).
    always @(posedge rclk) begin
        if (!rrst_n) begin
            wgray_rq1 <= {(ADDR_W+1){1'b0}};
            wgray_rq2 <= {(ADDR_W+1){1'b0}};
        end else begin
            wgray_rq1 <= wgray;
            wgray_rq2 <= wgray_rq1;
        end
    end

    // Read pointer update + registered EMPTY.
    always @(posedge rclk) begin
        if (!rrst_n) begin
            rbin  <= {(ADDR_W+1){1'b0}};
            rgray <= {(ADDR_W+1){1'b0}};
            empty <= 1'b1;             // FIFO is empty out of reset
        end else begin
            rbin  <= rbin_next;
            rgray <= rgray_next;
            empty <= empty_next;
        end
    end

    // RAM read port (synchronous on rclk, registered output).  Reads the word
    // at the current read pointer when a read is accepted; otherwise holds.
    // rd_data is assigned on every path of this block (no inferred latch).
    always @(posedge rclk) begin
        if (!rrst_n) begin
            rd_data <= {DATA_W{1'b0}};
        end else if (do_read) begin
            rd_data <= mem[rbin[ADDR_W-1:0]];
        end else begin
            rd_data <= rd_data;
        end
    end
endmodule
