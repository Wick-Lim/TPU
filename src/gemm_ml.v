`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// gemm_ml  --  output-stationary NxN GEMM with MULTI-LINE TM ROWS (ROADMAP §4)
//----------------------------------------------------------------------------
// PURPOSE (the limitation this unit resolves)
//   The architecture's TM line is FIXED at `LINE_W==128 bits = `LINE_LANES==4
//   lanes (tpu_defs.vh / tile_memory.v).  Every EXISTING tensor unit packs one
//   matrix/feature ROW into exactly ONE TM line, which structurally caps tile
//   sizes at N <= `LINE_LANES (e.g. gemm_systolic supports only N<=4).  This
//   self-contained unit PROVES the way past that cap: a matrix ROW spans
//   LINES_PER_ROW = ceil(N/`LINE_LANES) CONSECUTIVE TM lines, so tiles of N up
//   to (and beyond) `LINE_LANES work.  It does NOT touch the existing
//   single-line units -- those remain the verified production path; this unit
//   is the multi-line packing proof.
//
// PARAMETERIZATION (size N)
//   parameter integer N  -- the GEMM tile is NxN.  Supported & verified at
//   N=4 (LINES_PER_ROW==1, degenerate single-line case -- identical packing to
//   the legacy units) and N=8 (LINES_PER_ROW==2, the genuine multi-line case).
//   The unit is written for any N>=2: EVERY line index, lane index, counter
//   width and loop bound is DERIVED from N and `LINE_LANES -- there are NO size
//   literals (no hardcoded 4/8, no fixed 2-bit lane slices).
//
//   localparam LINES_PER_ROW = ceil(N / `LINE_LANES)
//            = (N + `LINE_LANES - 1) / `LINE_LANES        (1 @N<=4, 2 @N=5..8)
//
// TM LAYOUT  (multi-line rows -- the whole point of this unit)
//   Matrix M (one of A, B, C) with TM base line `base` occupies
//   N * LINES_PER_ROW consecutive lines.  ROW r of M occupies the
//   LINES_PER_ROW lines  [ base + r*LINES_PER_ROW .. base + r*LINES_PER_ROW
//   + LINES_PER_ROW-1 ].  Within that group, ELEMENT k (column k, 0..N-1) of
//   row r lives at:
//        line = base + r*LINES_PER_ROW + (k / `LINE_LANES)
//        lane =                          (k % `LINE_LANES)        (0..3)
//   and occupies the LOW `ELEM_W (16) bits of that 32-bit lane (the Q7.8
//   value).  Reads sign-extend those 16 bits to a signed Q7.8; writes re-pack
//   the sign-extended result.  When N is not a multiple of `LINE_LANES the
//   final line of a row is partially used (high lanes are don't-care on read
//   and written as 0 on write).
//   Tile bases on the interface:
//        A : a_base .. a_base + N*LINES_PER_ROW - 1
//        B : b_base .. b_base + N*LINES_PER_ROW - 1
//        C : c_base .. c_base + N*LINES_PER_ROW - 1
//
// ALGORITHM (output-stationary tiled / streaming MAC)
//   C = A . B for NxN matrices, C[i][j] = sum_{k=0..N-1} A[i][k]*B[k][j].
//   This is a CLEAR tiled MAC, not a skewed mesh (the deliverable is the
//   multi-line packing -- the MAC is the simple, obviously-correct kind):
//     1. LOAD A: walk A's lines in order, LINES_PER_ROW lines per row, and
//        assemble each full N-element row into the operand bank `afull`.
//     2. LOAD B: same for B into `bfull`.  A row load therefore takes
//        LINES_PER_ROW cycles, exactly as required.
//     3. MAC: with all operands banked, for each k=0..N-1 do one rank-1
//        update of ALL N*N output accumulators in parallel:
//            acc[i][j] += A[i][k] * B[k][j].
//        N such cycles fully evaluate every dot product (output-stationary:
//        each acc[i][j] is updated in place).
//     4. WRITEBACK C: for each output row r (0..N-1), round-half-up+saturate
//        its N accumulators to Q7.8 and PACK them back across LINES_PER_ROW
//        TM lines (one line driven per cycle), reproducing the multi-line row
//        layout above.
//
// Q-FORMATS  (single source of truth: tpu_defs.vh -- never re-defined here)
//   A,B,C elements : Q7.8 (16-bit signed in the low bits of a 32-bit lane).
//   MAC product    : Q7.8 * Q7.8 = Q14.16 (30-bit signed, fits 32 bits).
//   Accumulator    : Q15.16, 48-bit signed (`ACC_W) -- holds the sum of N
//                    Q14.16 products without overflow for N up to 8.
//   Output narrow  : round-half-up + saturate to Q7.8 via a LOCAL signed
//                    helper that is bit-identical to the canonical
//                    `TPU_RND_SAT_Q78 / `TPU_SAT_HIT macros (same idiom as
//                    gemm_systolic.v): the helper is named so the per-element
//                    `sat` reduction can index the unpacked acc[] array.
//                    No silent truncation anywhere.
//
// HANDSHAKE / LATENCY (measured, deterministic)
//   `start`  : 1-cycle pulse sampled in S_IDLE; latches a/b/c bases, begins.
//   `busy`   : registered, high from the cycle after start through the final
//              run cycle.
//   `done`   : COMBINATIONAL 1-cycle pulse on the final writeback cycle (the
//              cycle the last C line is driven onto the write port).
//   `sat`    : COMBINATIONAL OR-reduction over all N*N final accumulators,
//              valid together with `done`; 1 iff any C element saturated.
//   Phases (cycles after the start-sample cycle), ROW_LINES = N*LINES_PER_ROW:
//        load A          : ROW_LINES cycles  (LINES_PER_ROW per row)
//        load B          : ROW_LINES cycles  (LINES_PER_ROW per row)
//        MAC             : N cycles          (one rank-1 update per k)
//        writeback C     : ROW_LINES cycles  (drives one strobe per line)
//        drain           : 1 cycle           (last strobe on the bus; done)
//   MEASURED start->done LATENCY = 3 * N * LINES_PER_ROW + N + 1 cycles
//     = 3*ROW_LINES + N + 1.   e.g. N=4,LPR=1 -> 3*4 + 4 + 1 = 17 ;
//                                    N=8,LPR=2 -> 3*16 + 8 + 1 = 57.
//   Because the TM write is SYNCHRONOUS the write STROBE for a line driven in
//   S_WB appears on the bus the next cycle and COMMITS the cycle after that;
//   `done` is pulsed in the drain cycle, aligned with the LAST strobe's bus
//   presentation, so the last C line LANDS in TM exactly the cycle AFTER
//   `done`.  A consumer reads the full C tile from the cycle after `done`.
//
// INTERFACE  (raw TM ports, exactly like the other units -- TB owns the TM)
//   clk, rst                          clock / synchronous active-high reset
//   start                             1-cycle pulse: latch bases, begin GEMM
//   a_base,b_base,c_base [4:0]        TM line index of A/B/C tile (row 0, line 0)
//   busy                              high while a GEMM is in flight (registered)
//   done                              1-cycle pulse when last C line is driven
//   sat                               valid with done; 1 iff any C elem saturated
//   tm_raddr1 [4:0] / tm_rdata1 [127:0]   combinational TM read port 1 (A load)
//   tm_raddr2 [4:0] / tm_rdata2 [127:0]   combinational TM read port 2 (B load)
//   tm_we / tm_waddr [4:0] / tm_wdata [127:0]   synchronous TM write port (C)
//
// SYNTHESIZABILITY
//   All sequential state has a synchronous active-high reset; every reg is
//   assigned on every path of its clocked block (no inferred latch); the
//   combinational outputs (done, sat) are pure functions of registered state
//   with no feedback (no comb loop); no non-synthesizable constructs.  Passes
//   iverilog -g2012 -Wall, verilator --lint-only -Wall and yosys check -assert.
//============================================================================
module gemm_ml #(
    // GEMM tile dimension.  Verified at N=4 (single-line rows) and N=8
    // (two-line rows -- the multi-line packing proof).  Any N>=2 is structural.
    parameter integer N = 8
) (
    input  wire                  clk,
    input  wire                  rst,

    // Control handshake.
    input  wire                  start,
    input  wire [`TM_IDX_W-1:0]  a_base,
    input  wire [`TM_IDX_W-1:0]  b_base,
    input  wire [`TM_IDX_W-1:0]  c_base,
    output reg                   busy,
    output wire                  done,
    output wire                  sat,

    // Tile-memory access ports (the memory itself lives outside this unit).
    output reg  [`TM_IDX_W-1:0]  tm_raddr1,
    input  wire [`LINE_W-1:0]    tm_rdata1,
    output reg  [`TM_IDX_W-1:0]  tm_raddr2,
    input  wire [`LINE_W-1:0]    tm_rdata2,
    output reg                   tm_we,
    output reg  [`TM_IDX_W-1:0]  tm_waddr,
    output reg  [`LINE_W-1:0]    tm_wdata
);

    // ----------------------------------------------------------------------
    // Geometry -- ALL derived from N and `LINE_LANES (no size literals).
    //   LINES_PER_ROW = ceil(N / `LINE_LANES)  -- TM lines spanned by one row.
    //   ROW_LINES     = N * LINES_PER_ROW       -- lines spanned by a matrix.
    // Phase lengths (in S_RUN cycles):
    //   LOAD_CYC = ROW_LINES   (one TM read per line; LINES_PER_ROW per row)
    //   MAC_CYC  = N           (one rank-1 update per k)
    //   WB_CYC   = ROW_LINES   (one TM write per line)
    // ----------------------------------------------------------------------
    localparam integer LINES_PER_ROW =
        (N + `LINE_LANES - 1) / `LINE_LANES;        // ceil(N / LINE_LANES)
    localparam integer ROW_LINES = N * LINES_PER_ROW;

    localparam integer LOAD_A_END = ROW_LINES;                 // exclusive
    localparam integer LOAD_B_END = ROW_LINES + ROW_LINES;     // exclusive
    localparam integer MAC_END    = LOAD_B_END + N;            // exclusive
    localparam integer WB_END     = MAC_END + ROW_LINES;       // exclusive
    // tcnt walks 0..LAST_T; the drain cycle (last C strobe on the bus, `done`
    // asserted) is tcnt == WB_END, the final run cycle.
    localparam integer LAST_T     = WB_END;                    // drain cycle

    // Derived widths (no hardcoded slice widths).
    //   TCNT_W : run-cycle counter 0..LAST_T.
    //   ROW_W  : a matrix index 0..N-1 (row/col/k).  >=1 for N>=2.
    //   LPR_W  : a within-row line offset 0..LINES_PER_ROW-1.  >=1 always.
    //   LANE_W_IDX : a lane index 0..`LINE_LANES-1.
    localparam integer TCNT_W     = $clog2(LAST_T + 1);
    //   LOFF_W : a matrix line offset 0..ROW_LINES-1.  ROW_LINES <= `TM_LINES so
    //            LOFF_W <= `TM_IDX_W -- a line offset always zero-extends cleanly
    //            into a `TM_IDX_W TM address (no truncation, no unused bits).
    localparam integer LOFF_W     = $clog2(ROW_LINES);
    localparam integer ROW_W      = $clog2(N);
    localparam integer LPR_W      = (LINES_PER_ROW > 1) ? $clog2(LINES_PER_ROW)
                                                        : 1;
    localparam integer LANE_W_IDX = $clog2(`LINE_LANES);

    // FSM states.
    localparam [2:0] S_IDLE   = 3'd0;
    localparam [2:0] S_LOAD_A = 3'd1;   // read A: LINES_PER_ROW lines per row
    localparam [2:0] S_LOAD_B = 3'd2;   // read B: LINES_PER_ROW lines per row
    localparam [2:0] S_MAC    = 3'd3;   // N rank-1 updates of all acc[i][j]
    localparam [2:0] S_WB     = 3'd4;   // write C: LINES_PER_ROW lines per row
    localparam [2:0] S_DRAIN  = 3'd5;   // 1 cycle: last C strobe on bus, done

    reg [2:0]             state;
    reg [`TM_IDX_W-1:0]   a_base_r, b_base_r, c_base_r;
    reg [TCNT_W-1:0]      tcnt;        // run-cycle counter 0..LAST_T

    // Operand banks (Q7.8 sign-extended to 32 bits).
    //   afull[i][k] = A[i][k] ,  bfull[k][j] = B[k][j].
    reg signed [`XLEN-1:0] afull [0:N-1][0:N-1];
    reg signed [`XLEN-1:0] bfull [0:N-1][0:N-1];

    // N*N output accumulators, Q15.16, 48-bit signed.
    reg signed [`ACC_W-1:0] acc [0:N-1][0:N-1];

    integer i, j;

    // ----------------------------------------------------------------------
    // Local round-half-up + saturate to Q7.8  (SPEC §1.3 policy) -- a named,
    // bit-exact wrapper of the canonical tpu_defs.vh `TPU_RND_SAT_Q78 /
    // `TPU_SAT_HIT, identical to the helper in gemm_systolic.v.  It is retained
    // as a function (not the bare macro) so the per-element `sat` reduction can
    // index the unpacked acc[] array and evaluate the round-shift once.
    //   rounded = (acc + (1<<(FRAC-1))) >>> FRAC          [arithmetic shift]
    //   sat: rounded > 32767 -> 32767 ; < -32768 -> -32768 ; else value
    // ----------------------------------------------------------------------
    localparam signed [`ACC_W-1:0] RND_BIAS =
        `ACC_W'sd1 <<< (`Q78_FRAC-1);          // +128 = 1<<(FRAC-1), signed

    function signed [`ACC_W-1:0] round_shift;
        input signed [`ACC_W-1:0] acc_in;
        begin
            round_shift = (acc_in + RND_BIAS) >>> `Q78_FRAC;
        end
    endfunction

    function signed [`ELEM_W-1:0] rnd_sat_q78;
        input signed [`ACC_W-1:0] acc_in;
        reg   signed [`ACC_W-1:0] r;
        begin
            r = round_shift(acc_in);
            if (r > `ACC_W'sd32767)       rnd_sat_q78 = `Q78_MAX;
            else if (r < -`ACC_W'sd32768) rnd_sat_q78 = `Q78_MIN;
            else                          rnd_sat_q78 = r[`ELEM_W-1:0];
        end
    endfunction

    function sat_hit_q78;
        input signed [`ACC_W-1:0] acc_in;
        reg   signed [`ACC_W-1:0] r;
        begin
            r = round_shift(acc_in);
            sat_hit_q78 = (r > `ACC_W'sd32767) || (r < -`ACC_W'sd32768);
        end
    endfunction

    // ----------------------------------------------------------------------
    // Combinational helper: sign-extend one lane of a 128-bit TM line.
    // Only the low `ELEM_W (16) bits of the 32-bit lane carry the Q7.8 value.
    // ----------------------------------------------------------------------
    function signed [`XLEN-1:0] sext_lane;
        input [`LINE_W-1:0]    line;
        input [LANE_W_IDX-1:0] lane;     // 0..`LINE_LANES-1
        reg   [`ELEM_W-1:0]    raw;
        begin
            raw       = line[lane*`LANE_W +: `ELEM_W];
            sext_lane = `TPU_SEXT16(raw);
        end
    endfunction

    // ----------------------------------------------------------------------
    // Load-address decomposition.  During the load phases `lcnt` walks the
    // ROW_LINES lines of a matrix in order (0..ROW_LINES-1).  The line's
    //   row index    = lcnt / LINES_PER_ROW
    //   within-row   = lcnt % LINES_PER_ROW   (which line of that row)
    // are computed as TRUE integer div/mod (NOT bit truncation) so they are
    // correct for ANY N -- not just powers of two.  The within-row offset
    // selects which `LINE_LANES-element slice of the row this line carries.
    // ----------------------------------------------------------------------
    function [ROW_W-1:0] line_row;          // row index of load line `lc`
        input [TCNT_W-1:0] lc;
        begin
            line_row = ROW_W'(lc / LINES_PER_ROW[TCNT_W-1:0]);
        end
    endfunction

    function [LPR_W-1:0] line_off;          // within-row line offset of `lc`
        input [TCNT_W-1:0] lc;
        begin
            line_off = LPR_W'(lc % LINES_PER_ROW[TCNT_W-1:0]);
        end
    endfunction

    // A matrix line offset `lc` (already narrowed to LOFF_W bits by the caller,
    // value in [0,ROW_LINES-1]) as a `TM_IDX_W-wide TM address offset to add to
    // a tile base.  ROW_LINES <= `TM_LINES gives LOFF_W <= `TM_IDX_W, so this is
    // a pure zero-extension: every input bit is used, nothing is truncated, and
    // the lint stays clean (no out-of-range part-select, no WIDTHTRUNC/UNUSED).
    // Callers pass LOFF_W'(<TCNT_W expr>); the expression value is always in
    // [0,ROW_LINES-1] so that narrowing to LOFF_W bits is lossless.
    function [`TM_IDX_W-1:0] loff;
        input [LOFF_W-1:0] lc;
        begin
            loff = {{(`TM_IDX_W-LOFF_W){1'b0}}, lc};
        end
    endfunction

    // ----------------------------------------------------------------------
    // `done` : 1-cycle pulse asserted on the cycle the LAST C-line write STROBE
    //   is presented on the TM write port.  Because the write port is
    //   synchronous (tm_we/tm_waddr/tm_wdata are driven by NBA), the strobe for
    //   the line whose S_WB iteration ran at cycle t appears on the bus at t+1
    //   and the data COMMITS into TM at t+2.  Asserting `done` together with the
    //   final strobe presentation makes the last C line LAND in TM exactly the
    //   cycle AFTER `done` -- the same consume contract as gemm_systolic (read C
    //   one cycle after observing `done`).  `done` is therefore a registered
    //   pulse: it is raised by the S_WB tail (see the sequential core) and held
    //   for exactly one cycle.
    // `sat`  : combinational OR-reduction over all N*N (final, stable) accs,
    //   valid together with `done`.  Pure function of registered `acc`.
    // ----------------------------------------------------------------------
    reg done_r;
    assign done = done_r;

    reg     sat_comb;
    integer si, sj;
    always @(*) begin
        sat_comb = 1'b0;
        for (si = 0; si < N; si = si + 1)
            for (sj = 0; sj < N; sj = sj + 1)
                if (sat_hit_q78(acc[si][sj]))
                    sat_comb = 1'b1;
    end
    assign sat = sat_comb;

    // ----------------------------------------------------------------------
    // Working indices (registered) used inside the sequential core.
    //   load_row / load_off : row & within-row offset of the line being
    //                         banked THIS cycle (from the address driven LAST
    //                         cycle -- read data is combinational off raddr).
    //   wb_line             : 0..ROW_LINES-1 line counter of the writeback.
    // ----------------------------------------------------------------------
    reg [ROW_W-1:0] load_row;      // row of the line whose data is live now
    reg [LPR_W-1:0] load_off;      // within-row offset of that line
    reg [TCNT_W-1:0] mac_k;        // current k (0..N-1) during S_MAC
    reg [TCNT_W-1:0] wb_line;      // current writeback line (0..ROW_LINES-1)

    integer e;                     // element-within-line loop variable

    // ----------------------------------------------------------------------
    // Sequential core.
    // ----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            tcnt      <= {TCNT_W{1'b0}};
            a_base_r  <= {`TM_IDX_W{1'b0}};
            b_base_r  <= {`TM_IDX_W{1'b0}};
            c_base_r  <= {`TM_IDX_W{1'b0}};
            tm_raddr1 <= {`TM_IDX_W{1'b0}};
            tm_raddr2 <= {`TM_IDX_W{1'b0}};
            tm_we     <= 1'b0;
            tm_waddr  <= {`TM_IDX_W{1'b0}};
            tm_wdata  <= {`LINE_W{1'b0}};
            done_r    <= 1'b0;
            load_row  <= {ROW_W{1'b0}};
            load_off  <= {LPR_W{1'b0}};
            mac_k     <= {TCNT_W{1'b0}};
            wb_line   <= {TCNT_W{1'b0}};
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    afull[i][j] <= {`XLEN{1'b0}};
                    bfull[i][j] <= {`XLEN{1'b0}};
                    acc[i][j]   <= {`ACC_W{1'b0}};
                end
        end else begin
            // Defaults: single-cycle write strobe and single-cycle done pulse
            // (assigned every path -> no latch on tm_we / done_r).
            tm_we  <= 1'b0;
            done_r <= 1'b0;

            case (state)
                // ----------------------------------------------------------
                // S_IDLE : on `start`, latch bases, clear accumulators, and
                //   present A's first line (line 0 of row 0) on read port 1 so
                //   its data is live on the first S_LOAD_A cycle.
                // ----------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        a_base_r <= a_base;
                        b_base_r <= b_base;
                        c_base_r <= c_base;
                        busy     <= 1'b1;
                        tcnt     <= {TCNT_W{1'b0}};
                        for (i = 0; i < N; i = i + 1)
                            for (j = 0; j < N; j = j + 1)
                                acc[i][j] <= {`ACC_W{1'b0}};
                        // Present A line 0 now; it is live next cycle (S_LOAD_A
                        // tcnt==0).  load_row/load_off describe THAT line.
                        tm_raddr1 <= a_base;
                        load_row  <= {ROW_W{1'b0}};
                        load_off  <= {LPR_W{1'b0}};
                        state     <= S_LOAD_A;
                    end
                end

                // ----------------------------------------------------------
                // S_LOAD_A : bank the A line presented LAST cycle (its data is
                //   live on tm_rdata1 now), then address the NEXT A line.  A
                //   line carries `LINE_LANES elements: columns
                //   k = load_off*`LINE_LANES + e  for e=0..`LINE_LANES-1,
                //   skipping any k >= N on the final partial line.  Each ROW
                //   takes LINES_PER_ROW cycles, so the whole load is ROW_LINES
                //   cycles -- the required multi-line row load.
                // ----------------------------------------------------------
                S_LOAD_A: begin
                    for (e = 0; e < `LINE_LANES; e = e + 1) begin
                        // absolute column index k of lane e on this line
                        if ((load_off * `LINE_LANES + e) < N) begin
                            afull[load_row]
                                 [load_off * `LINE_LANES + e] <=
                                 sext_lane(tm_rdata1, e[LANE_W_IDX-1:0]);
                        end
                    end

                    // Next line index (within the matrix) = tcnt+1; while still
                    // inside A, decompose it and address port 1 for it.
                    if ((tcnt + 1'b1) < LOAD_A_END[TCNT_W-1:0]) begin
                        tm_raddr1 <= a_base_r + loff(LOFF_W'(tcnt + 1'b1));
                        load_row  <= line_row(tcnt + 1'b1);
                        load_off  <= line_off(tcnt + 1'b1);
                        tcnt      <= tcnt + 1'b1;
                    end else begin
                        // A fully loaded.  Present B's first line on port 2 so
                        // it is live on the first S_LOAD_B cycle; restart the
                        // line decomposition at B's row 0 / offset 0.
                        tm_raddr2 <= b_base_r;
                        load_row  <= {ROW_W{1'b0}};
                        load_off  <= {LPR_W{1'b0}};
                        tcnt      <= tcnt + 1'b1;
                        state     <= S_LOAD_B;
                    end
                end

                // ----------------------------------------------------------
                // S_LOAD_B : symmetric to S_LOAD_A on read port 2 into bfull.
                //   `lc` = tcnt - LOAD_A_END is the 0-based B line counter.
                // ----------------------------------------------------------
                S_LOAD_B: begin
                    for (e = 0; e < `LINE_LANES; e = e + 1) begin
                        if ((load_off * `LINE_LANES + e) < N) begin
                            bfull[load_row]
                                 [load_off * `LINE_LANES + e] <=
                                 sext_lane(tm_rdata2, e[LANE_W_IDX-1:0]);
                        end
                    end

                    if ((tcnt + 1'b1) < LOAD_B_END[TCNT_W-1:0]) begin
                        // next B line counter = (tcnt+1) - LOAD_A_END
                        tm_raddr2 <= b_base_r +
                            loff(LOFF_W'((tcnt + 1'b1)
                                         - LOAD_A_END[TCNT_W-1:0]));
                        load_row  <= line_row((tcnt + 1'b1)
                                              - LOAD_A_END[TCNT_W-1:0]);
                        load_off  <= line_off((tcnt + 1'b1)
                                              - LOAD_A_END[TCNT_W-1:0]);
                        tcnt      <= tcnt + 1'b1;
                    end else begin
                        // B fully loaded -> begin the MAC sweep at k=0.
                        mac_k <= {TCNT_W{1'b0}};
                        tcnt  <= tcnt + 1'b1;
                        state <= S_MAC;
                    end
                end

                // ----------------------------------------------------------
                // S_MAC : one rank-1 update per cycle.  For the current k,
                //   acc[i][j] += A[i][k] * B[k][j]  for all i,j.  N cycles
                //   (k=0..N-1) complete every dot product (output-stationary).
                // ----------------------------------------------------------
                S_MAC: begin
                    for (i = 0; i < N; i = i + 1)
                        for (j = 0; j < N; j = j + 1) begin
                            acc[i][j] <= acc[i][j]
                                + ($signed(afull[i][mac_k[ROW_W-1:0]])
                                 * $signed(bfull[mac_k[ROW_W-1:0]][j]));
                        end

                    if ((mac_k + 1'b1) < N[TCNT_W-1:0]) begin
                        mac_k <= mac_k + 1'b1;
                        tcnt  <= tcnt + 1'b1;
                    end else begin
                        // MAC done -> start writeback at line 0.
                        wb_line <= {TCNT_W{1'b0}};
                        tcnt    <= tcnt + 1'b1;
                        state   <= S_WB;
                    end
                end

                // ----------------------------------------------------------
                // S_WB : pack C back across multi-line rows, one TM line per
                //   cycle.  Line `wb_line` belongs to row
                //   r = wb_line / LINES_PER_ROW and carries the `LINE_LANES
                //   columns  k = (wb_line % LINES_PER_ROW)*`LINE_LANES + e.
                //   Each used lane gets the rounded/saturated C[r][k]; unused
                //   high lanes (final partial line) are written as 0.  The
                //   strobe (tm_we/tm_waddr/tm_wdata) is driven via NBA so it
                //   appears on the bus NEXT cycle; this state runs ROW_LINES
                //   cycles (wb_line 0..ROW_LINES-1) and then hands to S_DRAIN.
                // ----------------------------------------------------------
                S_WB: begin
                    tm_we    <= 1'b1;
                    tm_waddr <= c_base_r + loff(LOFF_W'(wb_line));
                    for (e = 0; e < `LINE_LANES; e = e + 1) begin
                        // column k of lane e on this writeback line
                        if ((line_off(wb_line) * `LINE_LANES + e) < N) begin
                            tm_wdata[e*`LANE_W +: `LANE_W] <=
                                `TPU_SEXT16(rnd_sat_q78(
                                    acc[line_row(wb_line)]
                                       [line_off(wb_line) * `LINE_LANES + e]));
                        end else begin
                            tm_wdata[e*`LANE_W +: `LANE_W] <= {`LANE_W{1'b0}};
                        end
                    end

                    if ((wb_line + 1'b1) < ROW_LINES[TCNT_W-1:0]) begin
                        wb_line <= wb_line + 1'b1;
                        tcnt    <= tcnt + 1'b1;
                    end else begin
                        // Last line's strobe is being driven this cycle (it
                        // reaches the bus NEXT cycle, in S_DRAIN).  Assert `done`
                        // (NBA) so it is high DURING S_DRAIN, aligned with that
                        // final strobe's bus presentation; the last C line then
                        // COMMITS into TM exactly one cycle after `done`.
                        done_r <= 1'b1;
                        tcnt   <= tcnt + 1'b1;
                        state  <= S_DRAIN;
                    end
                end

                // ----------------------------------------------------------
                // S_DRAIN : single cycle.  The LAST C line's write strobe is on
                //   the TM bus this cycle (driven by NBA in the final S_WB
                //   cycle); it COMMITS into TM next cycle.  `done` is high this
                //   cycle (asserted in the final S_WB cycle), so the last C line
                //   lands exactly one cycle after `done`.  Drop `busy`, clear
                //   tcnt and return to idle.  No new strobe (tm_we defaults low);
                //   done_r defaults low so the pulse is exactly one cycle.
                // ----------------------------------------------------------
                S_DRAIN: begin
                    busy  <= 1'b0;
                    tcnt  <= {TCNT_W{1'b0}};
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
