`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// gemm_systolic  --  output-stationary NxN systolic GEMM   (SPEC.md §5.1)
//----------------------------------------------------------------------------
// PARAMETERIZATION (size N)
//   This unit is parameterized over the GEMM tile dimension N (module
//   parameter N, DEFAULT `GEMM_N == 4).  With the default the module is
//   byte-identical in behavior to the original fixed-4x4 implementation (the
//   per-unit TB still passes with the same assertion count).  Every internal
//   index, counter width, loop bound, operand bank and accumulator array is
//   derived from N (via $clog2) -- no hardcoded 4's or fixed 2-bit slices
//   remain.
//
//   SUPPORTED RANGE:  2 <= N <= `LINE_LANES  (== 4).
//   * UPPER BOUND N<=4 is ARCHITECTURAL: a matrix ROW is packed as ONE TM line
//     of `LINE_LANES==4 lanes (baked into tpu_defs.vh / tile_memory.v /
//     tpu_top.v), so one line holds at most 4 elements.  A single-line-per-row
//     GEMM therefore cannot exceed N=4.  N>4 would require multi-line rows = a
//     TM port / packing re-architecture and is OUT OF SCOPE for this unit.
//   * LOWER BOUND N>=2 keeps every $clog2(N)-derived index width >= 1 bit
//     ($clog2(1)==0 is degenerate); N=1 is not a meaningful GEMM tile.
//
//   The size-specific mod-4 BIT-TRUNCATION tricks of the original
//   (arr_k = tcnt[1:0]-i-j, wr_row = tcnt[1:0]-FIRST_WR[1:0], tcnt[3:0]) have
//   been replaced by PARAMETER-CORRECT equivalents: a true subtraction guarded
//   to [0,N-1] for the arrival index, a true (tcnt-FIRST_WR) for the write row,
//   and a $clog2-sized counter -- correct for ANY N in range, not just powers
//   of two.
//----------------------------------------------------------------------------
// ALGORITHM (output-stationary skewed wavefront mesh)
//   Computes C = A . B for NxN matrices held in tile memory (TM).  The array
//   is an OUTPUT-STATIONARY NxN grid of N*N processing elements (PEs); PE[i][j]
//   owns the 48-bit Q15.16 accumulator for output element C[i][j].
//
//   Operands are SKEWED at the array edges exactly as in a wavefront array:
//     * A streams from the WEST, row i delayed by i, then shifts EAST one PE
//       per cycle:  A[i][k] reaches PE[i][j] at cycle  t = i + j + k.
//     * B streams from the NORTH, col j delayed by j, then shifts SOUTH one PE
//       per cycle:  B[k][j] reaches PE[i][j] at cycle  t = i + j + k.
//   Because A[i][k] and B[k][j] reach PE[i][j] on the SAME cycle t = i+j+k,
//   each PE multiply-accumulates exactly the N products of its dot product:
//        C[i][j] = sum_{k=0..N-1} A[i][k] * B[k][j].
//   The unique partial product consumed by PE[i][j] on cycle t is index
//   k = t - i - j, valid only while 0 <= k <= N-1.  We MAC precisely that
//   product each cycle, which is algebraically identical to a physical mesh
//   that shifts a/b operands between neighbouring PEs.
//
//   WAVEFRONT TIMING.  PE[i][j] performs its last (k=N-1) MAC at cycle
//   t = i + j + N-1.  The whole array therefore drains over t = 0 .. 3(N-1)
//   = 3N-3 cycles (10 cycles for N=4).  Output ROW i is fully accumulated once
//   its rightmost PE[i][N-1] finishes, at t = i + 2N-2 (rows finish at
//   t = 6,7,8,9 for N=4).  Results are written back ONE ROW PER CYCLE,
//   OVERLAPPED with the wavefront tail: row r is driven onto the TM write port
//   the cycle after it becomes final, i.e. at t = r + 2N-1 (t = 7,8,9,10).
//   This overlap is what yields the SPEC §3.3 latency 2N-1 + N (11 for N=4).
//
//   OPERAND DELIVERY.  TM has two combinational read ports.  During the run
//   the unit walks A row r on port 1 and B row r on port 2 (r = 0..N-1),
//   capturing one (A,B) row pair per cycle into the operand banks afull/bfull.
//   Row r is presented combinationally on cycle r and banked for cycle r+1.
//   A PE needing a row on the very cycle it is presented (the leading array
//   edge) reads it straight from the live TM read data via the operand
//   selectors a_sel/b_sel, so no PE ever sees a stale/not-yet-loaded operand.
//
// Q-FORMATS  (single source of truth: tpu_defs.vh)
//   A, B, C elements : Q7.8   (16-bit signed value sign-extended in a 32-bit
//                              TM lane; N lanes of a 4-lane TM line = one row).
//   MAC product      : Q7.8 * Q7.8 = Q14.16 (30-bit signed, fits 32 bits).
//   Accumulator      : Q15.16, 48-bit signed (N Q14.16 products w/o overflow).
//   Output narrowing : round-half-up + saturate to Q7.8 via the shared
//                      `TPU_RND_SAT_Q78 macro; saturation flagged via
//                      `TPU_SAT_HIT.
//
// MEMORY LAYOUT (TM, base line indices on the interface)
//   A : lines a_base..a_base+N-1 ; line a_base+i = row i = lanes
//       {A[i][N-1]..A[i][0]} (lane 0 = column 0).
//   B : lines b_base..b_base+N-1 ; line b_base+k = row k of B.
//   C : lines c_base..c_base+N-1 ; line c_base+i = row i of C.
//   Only the low ELEM_W (16) bits of each lane carry Q7.8 data; the unit
//   sign-extends on read and re-packs sign-extended results on write.
//
// LATENCY / HANDSHAKE (measured, deterministic)  -- SPEC §3.3 = 2N-1 + N
//   Cycle 0       : `start` sampled high in S_IDLE (bases latched).
//   Cycles 1..L   : S_RUN (L = 3N-2 + 1 = latency).  MAC wavefront on
//                   t = 0..3N-3 with overlapped row writes on t = 2N-1..3N-2;
//                   the LAST C row (row N-1) is driven on the final cycle.
//   `done` is a COMBINATIONAL 1-cycle pulse asserted on the final run cycle (the
//   cycle the last C row write is driven).  Measured start->done = 2N-1 + N
//   (11 for N=4).
//   `sat` is a COMBINATIONAL reduction over all N*N (final, stable)
//   accumulators, valid together with `done`; high iff ANY C element saturated.
//   `busy` is registered, high from cycle 1 through the final cycle.
//   Because the TM write is SYNCHRONOUS, the last C row LANDS in TM the cycle
//   AFTER `done`; a consumer reads C from the cycle after `done` onward.
//
// INTERFACE
//   clk, rst                         clock / synchronous active-high reset
//   start                            1-cycle pulse: latch bases, begin GEMM
//   a_base,b_base,c_base [4:0]       TM line indices of A, B, C tiles
//   busy                             high while a GEMM is in flight (registered)
//   done                             1-cycle pulse when last C row is driven
//   sat                              valid with done; 1 iff any C elem saturated
//   tm_raddr1,tm_raddr2  [4:0]       combinational TM read addrs (A row, B row)
//   tm_rdata1,tm_rdata2  [127:0]     combinational TM read data
//   tm_we / tm_waddr[4:0] / tm_wdata[127:0]   synchronous TM write port
//
// SYNTHESIZABILITY
//   All sequential state has a synchronous reset; every reg is assigned on every
//   path of its clocked block (no inferred latch); the combinational outputs
//   (done, sat, and the MAC/operand-select logic) are pure functions of
//   registered state with no feedback (no comb loop); no non-synthesizable
//   constructs.  Passes verilator --lint-only -Wall and iverilog -g2012 -Wall.
//============================================================================
module gemm_systolic #(
    // GEMM tile dimension.  DEFAULT keeps the historical 4x4 behavior exactly.
    // Architectural envelope: 2 <= N <= `LINE_LANES (one TM line per row).
    parameter integer N = `GEMM_N
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

    // Tile-memory access ports (memory itself lives outside this unit).
    output reg  [`TM_IDX_W-1:0]  tm_raddr1,
    input  wire [`LINE_W-1:0]    tm_rdata1,
    output reg  [`TM_IDX_W-1:0]  tm_raddr2,
    input  wire [`LINE_W-1:0]    tm_rdata2,
    output reg                   tm_we,
    output reg  [`TM_IDX_W-1:0]  tm_waddr,
    output reg  [`LINE_W-1:0]    tm_wdata
);

    // ----------------------------------------------------------------------
    // Geometry (local loop bounds only -- never redefines tpu_defs sizes).
    // All bounds and index/counter widths are DERIVED from the parameter N so
    // the unit is correct for any N in [2,`LINE_LANES]; defaults reproduce the
    // historical 4x4 sizing (LAST_MAC=9, FIRST_WR=7, LAST_T=10, 4-bit tcnt,
    // 2-bit indices).
    // ----------------------------------------------------------------------
    localparam integer LAST_MAC = 3*N - 3;   // last MAC cycle t = 3N-3  (9 @N=4)
    localparam integer FIRST_WR = 2*N - 1;   // first row write at t=2N-1 (7 @N=4)
    localparam integer LAST_T   = 3*N - 2;   // final run cycle t = 3N-2  (10 @N=4)

    // Derived widths (replace the original hardcoded 4/2-bit slices):
    //   TCNT_W : holds run-cycle counter 0..LAST_T (max value 3N-2).  Width
    //            $clog2(LAST_T+1) = $clog2(3N-1).  4 bits @N=4 (counts 0..10).
    //   IDX_W  : holds a matrix index 0..N-1 (row/col/arrival-k/write-row).
    //            $clog2(N).  2 bits @N=4.  >=1 for N>=2 (supported range).
    localparam integer TCNT_W = $clog2(LAST_T + 1);
    localparam integer IDX_W  = $clog2(N);

    // FSM states.
    localparam [0:0] S_IDLE = 1'b0;  // wait for start
    localparam [0:0] S_RUN  = 1'b1;  // skew-fed MAC wavefront + overlapped write

    reg                  state;
    reg [`TM_IDX_W-1:0]  a_base_r, b_base_r, c_base_r;
    reg [TCNT_W-1:0]     tcnt;     // run cycle counter 0..LAST_T (TCNT_W bits)

    // Operand register banks (Q7.8 sign-extended to 32 bits).
    //   afull[i][k] = A[i][k] ,  bfull[k][j] = B[k][j].
    reg signed [`XLEN-1:0] afull [0:N-1][0:N-1];
    reg signed [`XLEN-1:0] bfull [0:N-1][0:N-1];

    // N*N output accumulators, Q15.16, 48-bit signed.
    reg signed [`ACC_W-1:0] acc [0:N-1][0:N-1];

    integer i, j;

    // ----------------------------------------------------------------------
    // Combinational helper: sign-extend lane `lane` of a 128-bit TM line.
    // Only the low ELEM_W (16) bits of a lane carry Q7.8 data.
    // ----------------------------------------------------------------------
    function signed [`XLEN-1:0] sext_lane;
        input [`LINE_W-1:0] line;
        input [IDX_W-1:0]   lane;     // 0..N-1
        reg   [`ELEM_W-1:0] raw;
        begin
            raw       = line[lane*`LANE_W +: `ELEM_W];   // low 16 bits of lane
            sext_lane = `TPU_SEXT16(raw);
        end
    endfunction

    // Arrival index k = tcnt - i - j for PE[i][j].  PARAMETER-CORRECT form:
    // this is the TRUE non-negative difference (NOT a mod-N bit-truncation), so
    // it is exact for ANY N -- not just powers of two.  It is consulted only
    // when the wavefront guard at the call site holds (tcnt >= i+j and
    // tcnt-i-j < N), in which case the difference lies in [0,N-1] and fits the
    // IDX_W-bit result; the wide intermediate makes the subtraction true integer
    // arithmetic before the (lossless, in-range) narrow to IDX_W bits.
    function [IDX_W-1:0] arr_k;
        input [TCNT_W-1:0] ii;
        input [TCNT_W-1:0] jj;
        begin
            // tcnt-ii-jj evaluates at the full TCNT_W operand width (a TRUE
            // integer difference, NOT a mod-N truncation of pre-narrowed tcnt);
            // under the call-site guard the value is in [0,N-1] and is captured
            // EXACTLY by the IDX_W-bit return.  The IDX_W'() cast marks the
            // narrowing INTENTIONAL (lossless here) -- same idiom as the shared
            // `XLEN'/`ELEM_W' casts in tpu_defs.vh; keeps verilator -Wall clean.
            arr_k = IDX_W'(tcnt - ii - jj);
        end
    endfunction

    // Current write-back row index = tcnt - (2N-1), 0..N-1.  PARAMETER-CORRECT:
    // a true subtraction (consulted only when tcnt >= FIRST_WR, so the result is
    // in [0,N-1]) narrowed losslessly to IDX_W bits -- no mod-N bit-truncation.
    function [IDX_W-1:0] wr_row_f;
        begin
            wr_row_f = IDX_W'(tcnt - FIRST_WR[TCNT_W-1:0]);
        end
    endfunction

    // ----------------------------------------------------------------------
    // Local round-half-up + saturate to Q7.8  (SPEC §1.3 policy).
    //
    // NOTE on the shared macros: this local helper is a thin, bit-exact wrapper
    // of the canonical tpu_defs.vh `TPU_RND_SAT_Q78` / `TPU_ROUND_SHIFT /
    // `TPU_SAT_HIT.  The shared macros previously built their round-half-up bias
    // as an UNSIGNED concatenation, which forced (signed acc + unsigned bias)
    // UNSIGNED and turned the `>>>` into a LOGICAL shift, mis-narrowing NEGATIVE
    // accumulators (e.g. acc=-12800 -> +32767 instead of -50).  That header bug
    // is now FIXED (the bias is a signed ACC_W constant), so the macros are
    // signed-correct and this helper computes the IDENTICAL result; it is
    // retained as a named function only because the per-element `sat` reduction
    // below indexes the unpacked `acc[si][sj]` array and a function evaluates
    // the round-shift once.  Documented policy (matches `TPU_RND_SAT_Q78):
    //   rounded = (acc + (1 << (FRAC-1))) >>> FRAC      [arithmetic shift]
    //   sat: rounded > 32767 -> 32767 ; rounded < -32768 -> -32768 ; else value
    // Bias is a localparam (declared inside this module, per the rules).
    // ----------------------------------------------------------------------
    localparam signed [`ACC_W-1:0] RND_BIAS =
        `ACC_W'sd1 <<< (`Q78_FRAC-1);          // +128 = 1<<(FRAC-1), signed

    // Round-half-up + arithmetic shift; returns the (still wide) signed value.
    function signed [`ACC_W-1:0] round_shift;
        input signed [`ACC_W-1:0] acc_in;
        begin
            round_shift = (acc_in + RND_BIAS) >>> `Q78_FRAC;
        end
    endfunction

    // Saturate the rounded value to the signed Q7.8 range, return 16-bit Q7.8.
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

    // Did the canonical narrowing of `acc_in` saturate?  (1 bit)
    function sat_hit_q78;
        input signed [`ACC_W-1:0] acc_in;
        reg   signed [`ACC_W-1:0] r;
        begin
            r = round_shift(acc_in);
            sat_hit_q78 = (r > `ACC_W'sd32767) || (r < -`ACC_W'sd32768);
        end
    endfunction

    // ----------------------------------------------------------------------
    // Operand selectors.  At run cycle `tcnt` the read ports present A row
    // `tcnt` (port1) and B row `tcnt` (port2) while tcnt < N -- captured into
    // the banks for the NEXT cycle.  A PE needing row k == tcnt THIS cycle
    // reads it live from the TM read data; otherwise the needed row (k < tcnt)
    // is already banked.  (Needed row index never exceeds tcnt: an A row index
    // i is live only when tcnt==i, i.e. j==k==0; a B row index k is live only
    // when tcnt==k, i.e. i==j==0.)  Row indices are compared at TCNT_W width;
    // column/bank indices are IDX_W bits.
    // ----------------------------------------------------------------------
    function signed [`XLEN-1:0] a_sel;
        input [TCNT_W-1:0] rowi;   // A row index i  (compared with tcnt)
        input [IDX_W-1:0]  colk;   // A column index k
        begin
            if (rowi == tcnt) a_sel = sext_lane(tm_rdata1, colk);
            else              a_sel = afull[rowi[IDX_W-1:0]][colk];
        end
    endfunction

    function signed [`XLEN-1:0] b_sel;
        input [TCNT_W-1:0] rowk;   // B row index k  (compared with tcnt)
        input [IDX_W-1:0]  colj;   // B column index j
        begin
            if (rowk == tcnt) b_sel = sext_lane(tm_rdata2, colj);
            else              b_sel = bfull[rowk[IDX_W-1:0]][colj];
        end
    endfunction

    // ----------------------------------------------------------------------
    // `done` : combinational 1-cycle pulse on the final run cycle (last C row
    //          driven).  start->done latency = 3N-2 + 1 from start sample.
    // `sat`  : combinational reduction over ALL N*N (final, stable)
    //          accumulators, valid together with `done`.  High iff any C
    //          element saturated.
    // Both are pure functions of registered state -> glitch-free at sample.
    // ----------------------------------------------------------------------
    assign done = (state == S_RUN) && (tcnt == LAST_T[TCNT_W-1:0]);

    // `sat` : OR of the per-element saturation flags over all N*N (final,
    // stable) accumulators -- a pure combinational reduction of `acc`.  Uses an
    // `always @(*)` so the sensitivity correctly covers every word of the `acc`
    // array (a continuous assign calling a function that reads an unpacked
    // array is NOT reliably re-evaluated on array writes under iverilog).
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
    // Sequential core.
    // ----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            // --- synchronous reset: clear ALL state ---
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
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    afull[i][j] <= {`XLEN{1'b0}};
                    bfull[i][j] <= {`XLEN{1'b0}};
                    acc[i][j]   <= {`ACC_W{1'b0}};
                end
        end else begin
            // Default (assigned every cycle -> no latch; single-cycle write).
            tm_we <= 1'b0;

            case (state)
                // ----------------------------------------------------------
                // S_IDLE: on `start`, latch bases, clear accumulators, present
                //   A row 0 / B row 0 so they are available on the first MAC.
                // ----------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        a_base_r  <= a_base;
                        b_base_r  <= b_base;
                        c_base_r  <= c_base;
                        busy      <= 1'b1;
                        tcnt      <= {TCNT_W{1'b0}};
                        for (i = 0; i < N; i = i + 1)
                            for (j = 0; j < N; j = j + 1)
                                acc[i][j] <= {`ACC_W{1'b0}};
                        tm_raddr1 <= a_base;       // A row 0 presented next cyc
                        tm_raddr2 <= b_base;       // B row 0 presented next cyc
                        state     <= S_RUN;
                    end
                end

                // ----------------------------------------------------------
                // S_RUN: drive the skewed wavefront and overlapped writeback.
                //   (a) bank the presented A/B row (tcnt < N) and address next.
                //   (b) MAC every PE[i][j] whose arrival k = tcnt-i-j is in
                //       [0,N-1] (tcnt <= 3N-3 = LAST_MAC).
                //   (c) overlapped write: once row r = tcnt-(2N-1) is final
                //       (tcnt >= 2N-1), drive C row r to the TM write port.
                //   On tcnt == LAST_T (last row driven) return to idle; `done`
                //   is the combinational pulse asserted on this same cycle.
                // ----------------------------------------------------------
                S_RUN: begin
                    // (a) bank presented rows (valid only for tcnt < N).  The
                    //     next operand row to present is tcnt+1; this branch is
                    //     only reached while tcnt < N, so tcnt+1 <= N is a true
                    //     count and NO mod-N wrap is needed (none is performed).
                    //     At tcnt == N-1 the presented row tcnt+1 == N is a
                    //     don't-care (never banked: the guard blocks tcnt >= N).
                    if (tcnt < N[TCNT_W-1:0]) begin
                        for (j = 0; j < N; j = j + 1) begin
                            afull[tcnt[IDX_W-1:0]][j] <=
                                sext_lane(tm_rdata1, j[IDX_W-1:0]);
                            bfull[tcnt[IDX_W-1:0]][j] <=
                                sext_lane(tm_rdata2, j[IDX_W-1:0]);
                        end
                        tm_raddr1 <= a_base_r +
                            ({{(`TM_IDX_W-TCNT_W){1'b0}}, tcnt} +
                             {{(`TM_IDX_W-1){1'b0}}, 1'b1});
                        tm_raddr2 <= b_base_r +
                            ({{(`TM_IDX_W-TCNT_W){1'b0}}, tcnt} +
                             {{(`TM_IDX_W-1){1'b0}}, 1'b1});
                    end

                    // (b) MAC the skewed wavefront for cycle `tcnt` (t<=3N-3).
                    //     PE[i][j] MACs iff its arrival k = tcnt-i-j is in
                    //     [0,N-1]; the operand selectors fetch A[i][k], B[k][j]
                    //     from the live read ports or the banks as appropriate.
                    if (tcnt <= LAST_MAC[TCNT_W-1:0]) begin
                        for (i = 0; i < N; i = i + 1)
                            for (j = 0; j < N; j = j + 1) begin
                                if ((tcnt >= (i[TCNT_W-1:0] + j[TCNT_W-1:0])) &&
                                    ((tcnt - i[TCNT_W-1:0] - j[TCNT_W-1:0])
                                                       < N[TCNT_W-1:0])) begin
                                    acc[i][j] <= acc[i][j]
                                      + ($signed(a_sel(i[TCNT_W-1:0],
                                                  arr_k(i[TCNT_W-1:0],
                                                        j[TCNT_W-1:0])))
                                       * $signed(b_sel(
                                                  {{(TCNT_W-IDX_W){1'b0}},
                                                   arr_k(i[TCNT_W-1:0],
                                                         j[TCNT_W-1:0])},
                                                  j[IDX_W-1:0])));
                                end else begin
                                    acc[i][j] <= acc[i][j];   // hold (no latch)
                                end
                            end
                    end

                    // (c) overlapped writeback: row wr_row_f() = tcnt-(2N-1) is
                    //     final (its last MAC committed last cycle).  Round/sat
                    //     its N accumulators (lanes 0..N-1 of the TM line) and
                    //     drive the TM write port.
                    if (tcnt >= FIRST_WR[TCNT_W-1:0]) begin
                        tm_we    <= 1'b1;
                        tm_waddr <= c_base_r +
                                    {{(`TM_IDX_W-IDX_W){1'b0}}, wr_row_f()};
                        for (j = 0; j < N; j = j + 1) begin
                            tm_wdata[j*`LANE_W +: `LANE_W] <=
                                `TPU_SEXT16(rnd_sat_q78(acc[wr_row_f()][j]));
                        end
                    end

                    if (tcnt == LAST_T[TCNT_W-1:0]) begin
                        busy  <= 1'b0;
                        state <= S_IDLE;
                    end else begin
                        tcnt <= tcnt + {{(TCNT_W-1){1'b0}}, 1'b1};
                    end
                end
            endcase
        end
    end
endmodule
