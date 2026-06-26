`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// gemm_harness.v  --  SYNTHESIZABLE place-and-route harness for gemm_systolic
//----------------------------------------------------------------------------
// PURPOSE
//   gemm_systolic exposes RAW tile-memory (TM) access ports: TWO combinational
//   read ports (tm_raddr1->tm_rdata1 for A rows, tm_raddr2->tm_rdata2 for B
//   rows) and ONE synchronous write port (tm_we / tm_waddr / tm_wdata for C
//   rows), each carrying a full 128-bit TM line.  Synthesizing the unit
//   STANDALONE forces those wide ports to become top-level package pins, which
//   overflows any ECP5 package ("no BELs of type TRELLIS_IO").
//
//   This harness BURIES every wide port as an INTERNAL net.  It contains a small
//   synthesizable TM model (a reg [127:0] tm[0:TM_LINES-1] array) wired to the
//   unit exactly as test/gemm_systolic_tb.v wires it (two combinational read
//   ports, one synchronous write port), preloads the A and B tiles
//   deterministically from an LFSR on reset, pulses `start` once, and
//   XOR-accumulates everything the unit writes back (the C-matrix rows on the TM
//   write port, plus the sat flag) into a 32-bit signature register `sig`.  The
//   ONLY top-level ports are clk, rst, start, busy, done, sig[31:0] -- a handful
//   of pins -- so place-and-route fits and measures the unit's REAL internal
//   critical path (the N*N MAC mesh + the Q15.16->Q7.8 round/sat narrowing).
//
//   The signature feedback (sig <- tm_wdata, which depends on the whole MAC
//   datapath; the unit's reads depend on the LFSR-seeded A/B tiles) keeps the
//   synthesizer from optimizing the systolic array away: every output is
//   observable on `sig` and every operand it consumes is a non-constant net, so
//   the MULT18X18D DSPs and the accumulator mesh are all retained and placed.
//
// SYNTHESIZABILITY
//   No $display / $finish / initial-with-delays / real / $random.  All state has
//   a synchronous reset; every reg is assigned on every path of its clocked
//   block.  Passes iverilog -g2012 -Wall and verilator --lint-only -Wall.
//============================================================================
module gemm_harness (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,     // external 1-cycle request (gated internally)
    output reg         busy,
    output reg         done,
    output reg  [31:0] sig
);

    // ---- internal TM <-> unit access nets (BURIED; never top-level pins) ----
    wire [`TM_IDX_W-1:0] tm_raddr1, tm_raddr2;
    reg  [`LINE_W-1:0]   tm_rdata1, tm_rdata2;
    wire                 tm_we;
    wire [`TM_IDX_W-1:0] tm_waddr;
    wire [`LINE_W-1:0]   tm_wdata;

    // ---- unit status nets (buried; surfaced only through busy/done/sig) ----
    wire                 u_busy;
    wire                 u_done;
    wire                 u_sat;

    // ---- fixed operand/result base lines (A@0, B@8, C@16) ----
    localparam [`TM_IDX_W-1:0] A_BASE = 5'd0;
    localparam [`TM_IDX_W-1:0] B_BASE = 5'd8;
    localparam [`TM_IDX_W-1:0] C_BASE = 5'd16;

    // GEMM tile dimension (default N = `GEMM_N = 4 -> 4 A rows + 4 B rows).
    localparam integer N = `GEMM_N;

    // ---- internal tile memory model (32 x 128b), dual combinational read ----
    reg [`LINE_W-1:0] tm [0:`TM_LINES-1];
    always @(*) tm_rdata1 = tm[tm_raddr1];
    always @(*) tm_rdata2 = tm[tm_raddr2];

    // ---- 32-bit Fibonacci LFSR to deterministically seed the A/B operands ----
    // Maximal-length 32-bit polynomial (CRC-32 taps 0x04C11DB7); only a
    // deterministic, well-mixed, non-constant sequence is needed so the systolic
    // array sees real (non-trivial, non-constant) operands on every lane.
    reg [31:0] lfsr;
    wire [31:0] lfsr_step =
        {lfsr[30:0], 1'b0} ^ ({32{lfsr[31]}} & 32'h04C1_1DB7);

    // ---- preload / start sequencing state ----
    // After reset we wait in H_IDLE for the external `start` request, fill the N
    // A-rows (A_BASE..A_BASE+N-1) and the N B-rows (B_BASE..B_BASE+N-1) from the
    // LFSR, then emit one internal start pulse.  `seed_cnt` walks 0..2N-1 (A
    // rows first, then B rows).  After the unit finishes we return to H_IDLE so
    // a later `start` runs another GEMM (start is load-bearing).
    localparam [2:0] H_IDLE = 3'd0;   // wait for external start
    localparam [2:0] H_SEED = 3'd1;   // load A and B tile rows from LFSR
    localparam [2:0] H_KICK = 3'd2;   // emit one internal start pulse
    localparam [2:0] H_RUN  = 3'd3;   // unit running; capture C-row writes
    localparam [2:0] H_HOLD = 3'd4;   // finished; hold done/sig one cycle

    // Last seed index (2N-1), sized to the TM index width for clean compares.
    localparam [`TM_IDX_W-1:0] SEED_LAST = `TM_IDX_W'(2*N - 1);

    reg [2:0]  hstate;
    reg        u_start;
    reg [`TM_IDX_W-1:0] seed_cnt;     // 0..2N-1 across A then B rows

    integer li;

    // ---- the GEMM DUT (default N = `GEMM_N = 4) ----
    gemm_systolic dut (
        .clk       (clk),
        .rst       (rst),
        .start     (u_start),
        .a_base    (A_BASE),
        .b_base    (B_BASE),
        .c_base    (C_BASE),
        .busy      (u_busy),
        .done      (u_done),
        .sat       (u_sat),
        .tm_raddr1 (tm_raddr1),
        .tm_rdata1 (tm_rdata1),
        .tm_raddr2 (tm_raddr2),
        .tm_rdata2 (tm_rdata2),
        .tm_we     (tm_we),
        .tm_waddr  (tm_waddr),
        .tm_wdata  (tm_wdata)
    );

    // A line seeded from the LFSR: 4 lanes, each a Q7.8-ish value derived from a
    // different slice of the LFSR state so adjacent lanes differ.
    wire [`LINE_W-1:0] seed_line =
        { lfsr_step[31:0], lfsr[31:0],
          {lfsr[15:0], lfsr[31:16]}, ~lfsr[31:0] };

    // ----------------------------------------------------------------------
    // Single clocked control + memory-write + signature block.
    //   * synchronous write to tm[] from the unit's write port (like the TB);
    //   * XOR-accumulate the unit's written C rows and sat flag into `sig`;
    //   * sequence: seed N A-rows then N B-rows from the LFSR, pulse start once.
    // ----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            for (li = 0; li < `TM_LINES; li = li + 1)
                tm[li] <= {`LINE_W{1'b0}};
            lfsr     <= 32'h1357_BD9F;  // fixed non-zero seed
            hstate   <= H_IDLE;
            seed_cnt <= {`TM_IDX_W{1'b0}};
            u_start  <= 1'b0;
            busy     <= 1'b0;
            done     <= 1'b0;
            sig      <= 32'd0;
        end else begin
            lfsr    <= lfsr_step;       // free-running mixer
            u_start <= 1'b0;            // start is a 1-cycle pulse
            done    <= 1'b0;

            // Synchronous TM write captured from the unit (TB-identical), AND
            // fold every written C row + the sat flag into the signature so the
            // unit's output is observable (prevents dead-code elimination of the
            // MAC mesh / DSPs).
            if (tm_we) begin
                tm[tm_waddr] <= tm_wdata;
                sig <= sig ^ tm_wdata[31:0]   ^ tm_wdata[63:32]
                           ^ tm_wdata[95:64]  ^ tm_wdata[127:96]
                           ^ {31'd0, u_sat};
            end

            case (hstate)
                // Wait for the external start request, then begin seeding.
                H_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy     <= 1'b1;
                        seed_cnt <= {`TM_IDX_W{1'b0}};
                        hstate   <= H_SEED;
                    end
                end

                // Seed N A-rows (A_BASE..) then N B-rows (B_BASE..) from the LFSR.
                // seed_cnt 0..N-1 -> A rows ; N..2N-1 -> B rows.
                H_SEED: begin
                    if (seed_cnt < `TM_IDX_W'(N))
                        tm[A_BASE + seed_cnt] <= seed_line;
                    else
                        tm[B_BASE + (seed_cnt - `TM_IDX_W'(N))] <= seed_line;

                    if (seed_cnt == SEED_LAST) begin
                        hstate   <= H_KICK;
                        seed_cnt <= {`TM_IDX_W{1'b0}};
                    end else begin
                        seed_cnt <= seed_cnt + 5'd1;
                    end
                end

                // Emit exactly one internal start pulse to the unit.
                H_KICK: begin
                    u_start <= 1'b1;
                    busy    <= 1'b1;
                    hstate  <= H_RUN;
                end

                // Unit running: wait for its (combinational 1-cycle) done.
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
