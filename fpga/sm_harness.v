`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// sm_harness.v  --  SYNTHESIZABLE place-and-route harness for softmax_unit
//----------------------------------------------------------------------------
// PURPOSE
//   softmax_unit exposes RAW tile-memory (TM) access ports: a combinational
//   read port (tm_raddr -> tm_rdata) and a synchronous write port
//   (tm_we / tm_waddr / tm_wdata), each carrying a full 128-bit TM line.
//   Synthesizing the unit STANDALONE forces those 128-bit ports (plus the two
//   5-bit address ports etc.) to become top-level package pins, which overflows
//   any ECP5 package ("no BELs of type TRELLIS_IO").
//
//   This harness BURIES every wide port as an INTERNAL net.  It contains a small
//   synthesizable TM model (a reg [127:0] tm[0:TM_LINES-1] array) wired to the
//   unit exactly as test/softmax_unit_tb.v wires it (combinational read, sync
//   write), preloads tm deterministically from an LFSR on reset, pulses `start`
//   once, and XOR-accumulates everything the unit writes back (its probabilities
//   on the TM write port) into a 32-bit signature register `sig`.  The ONLY
//   top-level ports are clk, rst, start, busy, done, sig[31:0] -- a handful of
//   pins -- so place-and-route fits and measures the unit's REAL internal
//   critical path.
//
//   The signature feedback (sig depends on tm_wdata which depends on the unit's
//   whole datapath, and the unit's reads depend on the LFSR-seeded tm) keeps the
//   synthesizer from optimizing the unit away: every output the unit produces is
//   observable on `sig`, and every input it consumes is a non-constant net.
//
// SYNTHESIZABILITY
//   No $display / $finish / initial-with-delays / real / $random.  All state has
//   a synchronous reset; every reg is assigned on every path of its clocked
//   block.  Passes iverilog -g2012 -Wall and verilator --lint-only -Wall.
//============================================================================
module sm_harness (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,     // external 1-cycle request (gated internally)
    output reg         busy,
    output reg         done,
    output reg  [31:0] sig
);

    // ---- internal TM <-> unit access nets (BURIED; never top-level pins) ----
    wire [`TM_IDX_W-1:0] tm_raddr;
    reg  [`LINE_W-1:0]   tm_rdata;
    wire                 tm_we;
    wire [`TM_IDX_W-1:0] tm_waddr;
    wire [`LINE_W-1:0]   tm_wdata;

    // ---- unit status nets (buried; surfaced only through busy/done/sig) ----
    wire                 u_busy;
    wire                 u_done;
    wire                 u_sat;
    wire [2:0]           u_argmax;

    // ---- fixed operand/result base lines (mirrors the unit TB: x@0, p@8) ----
    localparam [`TM_IDX_W-1:0] X_BASE = 5'd0;
    localparam [`TM_IDX_W-1:0] P_BASE = 5'd8;

    // ---- internal tile memory model (32 x 128b), exactly like the TB ----
    reg [`LINE_W-1:0] tm [0:`TM_LINES-1];

    // Combinational read port (present the addressed line on tm_rdata).
    always @(*) tm_rdata = tm[tm_raddr];

    // ---- 32-bit Fibonacci LFSR used to deterministically seed the logits ----
    // Maximal-length 32-bit polynomial (CRC-32 taps 0x04C11DB7); the exact
    // polynomial is irrelevant -- we only need a deterministic, well-mixed,
    // non-constant sequence to fill the logit lanes so the unit sees real data.
    reg [31:0] lfsr;
    wire [31:0] lfsr_step =
        {lfsr[30:0], 1'b0} ^ ({32{lfsr[31]}} & 32'h04C1_1DB7);

    // ---- preload / start sequencing state ----
    // After reset we wait in H_IDLE for the external `start` request, fill the
    // two logit lines (X_BASE, X_BASE+1) from the LFSR, then emit a single
    // internal start pulse to the unit.  After the unit finishes we return to
    // H_IDLE so a later `start` runs another softmax (start is load-bearing).
    localparam [2:0] H_IDLE = 3'd0;   // wait for external start
    localparam [2:0] H_SEED = 3'd1;   // load logit lines from LFSR
    localparam [2:0] H_KICK = 3'd2;   // emit one internal start pulse
    localparam [2:0] H_RUN  = 3'd3;   // unit running; capture writes
    localparam [2:0] H_HOLD = 3'd4;   // finished; hold done/sig one cycle

    reg [2:0]  hstate;
    reg        u_start;
    reg [`TM_IDX_W-1:0] seed_line;    // which TM line we are seeding (0 or 1)

    integer li;

    // ---- the softmax DUT (default LEN = `SM_LEN = 8) ----
    softmax_unit dut (
        .clk      (clk),
        .rst      (rst),
        .start    (u_start),
        .x_base   (X_BASE),
        .p_base   (P_BASE),
        .busy     (u_busy),
        .done     (u_done),
        .sat      (u_sat),
        .argmax   (u_argmax),
        .tm_raddr (tm_raddr),
        .tm_rdata (tm_rdata),
        .tm_we    (tm_we),
        .tm_waddr (tm_waddr),
        .tm_wdata (tm_wdata)
    );

    // ----------------------------------------------------------------------
    // Single clocked control + memory-write + signature block.
    //   * synchronous write to tm[] from the unit's write port (like the TB);
    //   * XOR-accumulate the unit's write data, status and argmax into `sig`;
    //   * sequence: seed two logit lines from the LFSR, pulse u_start once.
    // ----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            for (li = 0; li < `TM_LINES; li = li + 1)
                tm[li] <= {`LINE_W{1'b0}};
            lfsr      <= 32'hACE1_2345;   // fixed non-zero seed
            hstate    <= H_IDLE;
            seed_line <= {`TM_IDX_W{1'b0}};
            u_start   <= 1'b0;
            busy      <= 1'b0;
            done      <= 1'b0;
            sig       <= 32'd0;
        end else begin
            // Advance the LFSR every cycle (free-running mixer).
            lfsr    <= lfsr_step;
            u_start <= 1'b0;            // default: start is a 1-cycle pulse
            done    <= 1'b0;

            // Synchronous TM write captured from the unit (TB-identical), AND
            // fold every written line into the signature so the unit's output
            // is observable (prevents dead-code elimination of the datapath).
            if (tm_we) begin
                tm[tm_waddr] <= tm_wdata;
                sig <= sig ^ tm_wdata[31:0]   ^ tm_wdata[63:32]
                           ^ tm_wdata[95:64]  ^ tm_wdata[127:96]
                           ^ {28'd0, u_sat, u_argmax};
            end

            case (hstate)
                // Wait for the external start request, then begin seeding.
                H_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy      <= 1'b1;
                        seed_line <= {`TM_IDX_W{1'b0}};
                        hstate    <= H_SEED;
                    end
                end

                // Seed the two logit lines (X_BASE, X_BASE+1) from the LFSR.
                H_SEED: begin
                    tm[X_BASE + seed_line] <=
                        { lfsr_step[31:0], lfsr[31:0],
                          {lfsr[15:0], lfsr[31:16]}, ~lfsr[31:0] };
                    if (seed_line == 5'd1) begin
                        hstate    <= H_KICK;
                        seed_line <= {`TM_IDX_W{1'b0}};
                    end else begin
                        seed_line <= seed_line + 5'd1;
                    end
                end

                // Emit exactly one internal start pulse to the unit.
                H_KICK: begin
                    u_start <= 1'b1;
                    busy    <= 1'b1;
                    hstate  <= H_RUN;
                end

                // Unit running: wait for its done, captured above into sig.
                H_RUN: begin
                    busy <= u_busy;
                    if (u_done) begin
                        done   <= 1'b1;
                        busy   <= 1'b0;
                        hstate <= H_HOLD;
                    end
                end

                // Finished: drop busy and return to idle (await next start).
                H_HOLD: begin
                    busy   <= 1'b0;
                    hstate <= H_IDLE;
                end

                default: hstate <= H_IDLE;
            endcase
        end
    end

endmodule
