`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// attention_param_tb.v  --  2nd-SIZE proof for the parameterized attention_unit
//----------------------------------------------------------------------------
// PURPOSE
//   attention_unit was parameterized over SEQ / D (defaults `SEQ_LEN,`D_MODEL =
//   4,4).  attention_unit_tb exercises the DEFAULT size exhaustively and must
//   stay byte-identical / same assertion count.  This SEPARATE testbench
//   instantiates the unit at a DIFFERENT, in-range size to prove the
//   parameterization is structurally correct: the score (QK^T) walk, the
//   softmax-row packing (LEN=SEQ, no padding), the *V context, every counter/index width
//   ($clog2), the TM row (un)packing and the latency closed form must all track
//   SEQ and D -- not a hardcoded 4.
//
//   2nd SIZE:  SEQ = 2, D = 2.
//     * Q/K/V/O each occupy SEQ=2 TM lines (vs 4 at the default); each row packs
//       D=2 elements into lanes 0..1 of ONE 128-bit line (vs 4 lanes) -- so a
//       fixed-4-lane / fixed-4-row implementation would mis-address.  The score
//       matrix is 2x2 (vs 4x4), and the softmax runs over EXACTLY SEQ=2 real
//       logits (LEN=SEQ, NO padding -- the committed softmax_unit is reused
//       unchanged at LEN=2), so the score row spans NSCR_X=ceil(2/4)=1 X-line.
//       The committed start->done latency for SEQ=2 is, from the LEN=SEQ closed
//       form, LAT = (3*SEQ+1) + SEQ*(SM_LAT+1+3) + 2 with SM_LAT=53+ceil(2/4)+2*2
//       = 58, i.e. 7 + 2*62 + 2 = 133 cycles, checked EXACTLY (it differs from the
//       default 279, so a fixed latency would fail).
//   This sits strictly inside the architectural envelope (D=2 <= LINE_LANES=4;
//   SEQ=2 <= SM_LEN=8).
//
// INDEPENDENT GOLDEN  (NOT mirrored from the DUT)
//   The golden computes the SAME attention a COMPLETELY DIFFERENT way and shares
//   NO arithmetic with the DUT's fixed-point/LUT path:
//     * every Q/K/V element -> Verilog `real` (val/256.0),
//     * scores S[i][j] = SUM_d Qr[i][d]*Kr[j][d] in `real`, scaled by the EXACT
//       1/sqrt(4)=0.5 (the unit's committed >>1, identical for any SEQ/D),
//     * row-wise softmax uses the TRUE math-library $exp() with real division,
//     * context O[i][d] = SUM_j Wr[i][j]*Vr[j][d] in `real`,
//     * quantization to Q7.8 (round-half-up + saturate) ONLY at the boundary.
//   Pure floating point vs fixed-point+LUT exp: the two cannot share an
//   arithmetic bug.  Same +/- ATOL=2 LSB per-element tolerance as the default TB.
//
// MEMORY MODEL
//   The TB models the tile memory: a 32x128 tm[] array driven by the DUT's
//   external TM access ports (combinational read, synchronous write).  The DUT's
//   one permitted submodule (softmax_unit) runs on the DUT's INTERNAL scratch.
//
// COVERAGE  (directed + constrained random, all generalized to SEQ=2,D=2)
//   D1 zeros                 -> uniform weights, O == 0
//   D2 identity Q=K=I, equal V rows -> O row == the (shared) V row (passthrough)
//   D3 dominant-key          -> the aligned key's V passes through
//   D5 negative Q/K/V        -> signed dot products, valid distribution
//   D8 convexity/saturation  -> all V = +max, uniform weights -> O == +max, sat 0
//   D9 base-offset independence (passthrough at non-default bases)
//   R  >=150 constrained-random Q/K/V within the accurate regime; per-element
//      tolerance, latency EXACT (133), sat==gsat every vector.
//
// GATE: prints "ALL <N> TESTS PASSED"; $fatal on ANY mismatch.
//============================================================================
module attention_param_tb;

    // ---- 2nd-size geometry (DIFFERENT from the default 4/4) ----
    localparam integer S    = 2;          // SEQ
    localparam integer D    = 2;          // D_MODEL
    // Committed start->done latency for SEQ=2, from the LEN=SEQ softmax closed
    // form (see the DUT header LATENCY note): SETUP=3*SEQ+1=7, SM_LAT=53+ceil(2/4)
    // +2*2=58, PER_ROW=SM_LAT+1+3=62, LAT=SETUP+SEQ*PER_ROW+2 = 7 + 2*62 + 2 = 133.
    // (Was 37 before the softmax reciprocal divide was PIPELINED into a multi-cycle
    // sequential divider: that added DIV_CYCLES=48 per softmax invocation, and
    // attention reuses the softmax once per output row, so LAT grew by SEQ*48 = 96.
    // The values are UNCHANGED -- only the latency grew -- and 133 != the default 279.)
    localparam integer LAT  = 133;        // committed start->done for SEQ=2
    localparam integer ATOL = 2;          // per-element Q7.8 tolerance (LSB)
    localparam integer NRAND = 180;       // constrained-random vectors

    // ---- clock / reset ----
    reg clk;
    reg rst;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ---- DUT control ----
    reg                  start;
    reg  [`TM_IDX_W-1:0] q_base, k_base, v_base, o_base;
    wire                 busy;
    wire                 done;
    wire                 sat;

    // ---- DUT <-> TM access ports ----
    wire [`TM_IDX_W-1:0] tm_raddr;
    reg  [`LINE_W-1:0]   tm_rdata;
    wire                 tm_we;
    wire [`TM_IDX_W-1:0] tm_waddr;
    wire [`LINE_W-1:0]   tm_wdata;

    // ---- TB-modelled tile memory ----
    reg [`LINE_W-1:0] tm [0:`TM_LINES-1];
    always @(*) tm_rdata = tm[tm_raddr];
    always @(posedge clk) begin
        if (tm_we)
            tm[tm_waddr] <= tm_wdata;
    end

    // ---- DUT instantiated at the 2nd size via parameter override ----
    attention_unit #(.SEQ(S), .D(D)) dut (
        .clk      (clk),
        .rst      (rst),
        .start    (start),
        .q_base   (q_base),
        .k_base   (k_base),
        .v_base   (v_base),
        .o_base   (o_base),
        .busy     (busy),
        .done     (done),
        .sat      (sat),
        .tm_raddr (tm_raddr),
        .tm_rdata (tm_rdata),
        .tm_we    (tm_we),
        .tm_waddr (tm_waddr),
        .tm_wdata (tm_wdata)
    );

    // ---- bookkeeping ----
    integer pass;
    integer fail;
    integer seed;
    integer t;
    integer i, j, d;
    integer cyc;

    // ======================================================================
    // Operand matrices, held as plain signed-16 Q7.8 integers (S x D).
    // ======================================================================
    reg signed [15:0] QM [0:S-1][0:D-1];
    reg signed [15:0] KM [0:S-1][0:D-1];
    reg signed [15:0] VM [0:S-1][0:D-1];

    // ----------------------------------------------------------------------
    // Write an S-row Q7.8 matrix into S TM lines.  Each of the D elements sits
    // in the LOW 16 bits of lane c (sign-extended), lanes D..3 left 0 (the unit
    // ignores them on read).  Generalized over D (NOT a fixed 4-lane pack).
    // ----------------------------------------------------------------------
    task write_mat;
        input integer which;                 // 0=Q 1=K 2=V
        input [`TM_IDX_W-1:0] base;
        integer r, c;
        reg signed [15:0] e;
        reg [`LINE_W-1:0] line;
        begin
            for (r = 0; r < S; r = r + 1) begin
                line = {`LINE_W{1'b0}};
                for (c = 0; c < D; c = c + 1) begin
                    e = (which == 0) ? QM[r][c] :
                        (which == 1) ? KM[r][c] : VM[r][c];
                    line[c*`LANE_W +: `LANE_W] = {{16{e[15]}}, e};
                end
                tm[base + r] = line;
            end
        end
    endtask

    // ======================================================================
    // INDEPENDENT real golden.  Pure floating-point attention, quantized only
    // at the boundary.  gO[i][d] = quantized Q7.8 context; gsat = 1 iff any real
    // context element falls outside the Q7.8 representable range.
    // ======================================================================
    real    Qr [0:S-1][0:D-1];
    real    Kr [0:S-1][0:D-1];
    real    Vr [0:S-1][0:D-1];
    real    Sr [0:S-1][0:S-1];   // scaled scores (the softmax logits, real)
    real    Wr [0:S-1][0:S-1];   // softmax weights (real)
    real    Or [0:S-1][0:D-1];   // context (real, before quantize)
    integer gO [0:S-1][0:D-1];   // quantized Q7.8 context
    reg     gsat;

    task golden;
        integer gi, gj, gd;
        real    acc, mx, esum, ev, oq;
        integer oi;
        begin
            for (gi = 0; gi < S; gi = gi + 1)
                for (gd = 0; gd < D; gd = gd + 1) begin
                    Qr[gi][gd] = QM[gi][gd] / 256.0;
                    Kr[gi][gd] = KM[gi][gd] / 256.0;
                    Vr[gi][gd] = VM[gi][gd] / 256.0;
                end
            // scaled scores S[i][j] = 0.5 * SUM_d Qr*Kr  (exact >>1, any SEQ/D).
            for (gi = 0; gi < S; gi = gi + 1)
                for (gj = 0; gj < S; gj = gj + 1) begin
                    acc = 0.0;
                    for (gd = 0; gd < D; gd = gd + 1)
                        acc = acc + Qr[gi][gd] * Kr[gj][gd];
                    Sr[gi][gj] = 0.5 * acc;
                end
            // row-wise softmax over the SEQ scaled scores (true $exp).
            for (gi = 0; gi < S; gi = gi + 1) begin
                mx = Sr[gi][0];
                for (gj = 1; gj < S; gj = gj + 1)
                    if (Sr[gi][gj] > mx) mx = Sr[gi][gj];
                esum = 0.0;
                for (gj = 0; gj < S; gj = gj + 1) begin
                    ev = $exp(Sr[gi][gj] - mx);
                    Wr[gi][gj] = ev;
                    esum = esum + ev;
                end
                for (gj = 0; gj < S; gj = gj + 1)
                    Wr[gi][gj] = Wr[gi][gj] / esum;
            end
            // context O[i][d] = SUM_j W[i][j]*V[j][d], real, then quantize Q7.8.
            gsat = 1'b0;
            for (gi = 0; gi < S; gi = gi + 1)
                for (gd = 0; gd < D; gd = gd + 1) begin
                    acc = 0.0;
                    for (gj = 0; gj < S; gj = gj + 1)
                        acc = acc + Wr[gi][gj] * Vr[gj][gd];
                    Or[gi][gd] = acc;
                    oq = acc * 256.0;
                    if (oq >= 0.0) oi = $rtoi(oq + 0.5);
                    else           oi = $rtoi(oq - 0.5);
                    if (oi > 32767)  begin oi = 32767;  gsat = 1'b1; end
                    if (oi < -32768) begin oi = -32768; gsat = 1'b1; end
                    gO[gi][gd] = oi;
                end
        end
    endtask

    // ----------------------------------------------------------------------
    // Read back the S x D DUT context O from the S TM lines at o_base.
    // ----------------------------------------------------------------------
    reg signed [15:0] DO [0:S-1][0:D-1];
    task read_ctx;
        input [`TM_IDX_W-1:0] base;
        integer r, c;
        begin
            for (r = 0; r < S; r = r + 1)
                for (c = 0; c < D; c = c + 1)
                    DO[r][c] = tm[base+r][c*`LANE_W +: `ELEM_W];
        end
    endtask

    function integer absdiff;
        input integer a;
        input integer b;
        begin
            absdiff = (a > b) ? (a - b) : (b - a);
        end
    endfunction

    // ----------------------------------------------------------------------
    // Run one attention op end-to-end and check against the golden.
    // ----------------------------------------------------------------------
    task run_check;
        input [255:0] tag;
        begin
            golden;

            write_mat(0, q_base);
            write_mat(1, k_base);
            write_mat(2, v_base);

            @(negedge clk);
            start = 1'b1;
            @(posedge clk);          // start edge: DUT samples start (cyc 1)
            #1;
            start = 1'b0;
            cyc = 1;
            if (busy !== 1'b1) begin
                $display("FAIL[%0s] busy not asserted after start (cyc=%0d)", tag, cyc);
                fail = fail + 1; $fatal(1, "attention busy timing");
            end
            while (done !== 1'b1) begin
                @(posedge clk);
                #1;
                cyc = cyc + 1;
                if (cyc > LAT + 8) begin
                    $display("FAIL[%0s] done never asserted (cyc=%0d)", tag, cyc);
                    fail = fail + 1; $fatal(1, "attention done timeout");
                end
            end
            if (cyc != LAT) begin
                $display("FAIL[%0s] done at cyc=%0d, expected %0d", tag, cyc, LAT);
                fail = fail + 1; $fatal(1, "attention latency mismatch");
            end else begin
                pass = pass + 1;
            end
            if (busy !== 1'b0) begin
                $display("FAIL[%0s] busy still high at done", tag);
                fail = fail + 1; $fatal(1, "attention busy-at-done");
            end else begin
                pass = pass + 1;
            end

            // last O row lands in TM on the NEXT posedge; advance then read.
            @(posedge clk);
            #1;
            read_ctx(o_base);

            for (i = 0; i < S; i = i + 1)
                for (j = 0; j < D; j = j + 1) begin
                    if (absdiff(DO[i][j], gO[i][j]) > ATOL) begin
                        $display("FAIL[%0s] O[%0d][%0d] dut=%0d gold=%0d (diff=%0d)",
                                 tag, i, j, DO[i][j], gO[i][j],
                                 absdiff(DO[i][j], gO[i][j]));
                        fail = fail + 1; $fatal(1, "attention context mismatch");
                    end else begin
                        pass = pass + 1;
                    end
                end

            // SAT CONSISTENCY: the context is a convex combination of in-range V,
            // so it provably never clamps; DUT sat must be 0 and match gsat.
            if (sat !== gsat) begin
                $display("FAIL[%0s] sat=%b but golden gsat=%b", tag, sat, gsat);
                fail = fail + 1; $fatal(1, "attention sat/gsat mismatch");
            end else begin
                pass = pass + 1;
            end
        end
    endtask

    // ----------------------------------------------------------------------
    // Operand-loading helpers.
    // ----------------------------------------------------------------------
    task set_all_zero;
        integer r, c;
        begin
            for (r = 0; r < S; r = r + 1)
                for (c = 0; c < D; c = c + 1) begin
                    QM[r][c] = 16'sd0; KM[r][c] = 16'sd0; VM[r][c] = 16'sd0;
                end
        end
    endtask

    task set_QK_identity;
        integer r, c;
        begin
            for (r = 0; r < S; r = r + 1)
                for (c = 0; c < D; c = c + 1) begin
                    QM[r][c] = (r == c) ? 16'sd256 : 16'sd0;
                    KM[r][c] = (r == c) ? 16'sd256 : 16'sd0;
                end
        end
    endtask

    function signed [15:0] rnd_elem;
        input integer lim;
        integer v;
        begin
            v = ($random(seed) % (2*lim + 1)) - lim;
            rnd_elem = v[15:0];
        end
    endfunction

    task randomize_ops;
        input integer lim;
        integer r, c;
        begin
            for (r = 0; r < S; r = r + 1)
                for (c = 0; c < D; c = c + 1) begin
                    QM[r][c] = rnd_elem(lim);
                    KM[r][c] = rnd_elem(lim);
                    VM[r][c] = rnd_elem(lim);
                end
        end
    endtask

    // ======================================================================
    integer r2, c2;
    initial begin
        pass   = 0;
        fail   = 0;
        seed   = 32'h0BADF00D;
        start  = 1'b0;
        q_base = 5'd0;
        k_base = 5'd2;
        v_base = 5'd4;
        o_base = 5'd6;

        for (i = 0; i < `TM_LINES; i = i + 1)
            tm[i] = {`LINE_W{1'b0}};

        // ---- synchronous reset ----
        rst = 1'b1;
        @(posedge clk);
        @(posedge clk);
        rst = 1'b0;
        @(posedge clk);
        if (busy !== 1'b0 || done !== 1'b0) begin
            $display("FAIL[reset] busy=%b done=%b after reset", busy, done);
            fail = fail + 1; $fatal(1, "attention reset state");
        end else begin
            pass = pass + 1;
        end

        // =========================== DIRECTED ===========================

        // D1 zeros -> uniform weights, O == 0.
        set_all_zero;
        run_check("D1-zeros");
        for (i = 0; i < S; i = i + 1)
            for (j = 0; j < D; j = j + 1)
                if (DO[i][j] !== 16'sd0) begin
                    $display("FAIL[D1-zero] O[%0d][%0d]=%0d !=0", i, j, DO[i][j]);
                    fail = fail + 1; $fatal(1, "attention zero-out");
                end else pass = pass + 1;

        // D2 identity Q=K=I with EQUAL V rows -> each O row == the shared V row.
        set_QK_identity;
        for (r2 = 0; r2 < S; r2 = r2 + 1)
            for (c2 = 0; c2 < D; c2 = c2 + 1)
                VM[r2][c2] = (c2 == 0) ? 16'sd96 : 16'sd200;
        run_check("D2-identity");
        for (i = 0; i < S; i = i + 1)
            if (DO[i][0] !== 16'sd96 || DO[i][1] !== 16'sd200) begin
                $display("FAIL[D2-passthru] O row %0d = %0d %0d",
                         i, DO[i][0], DO[i][1]);
                fail = fail + 1; $fatal(1, "attention passthrough");
            end else pass = pass + 1;

        // D3 dominant-key: row i query strongly aligns with key i (big diagonal),
        // distinct V rows -> O[i] ~ V[i] (the aligned value passes through).
        set_all_zero;
        for (r2 = 0; r2 < S; r2 = r2 + 1) begin
            QM[r2][r2] = 16'sd1024;   // 4.0 on the diagonal (huge alignment)
            KM[r2][r2] = 16'sd1024;
            for (c2 = 0; c2 < D; c2 = c2 + 1)
                VM[r2][c2] = (r2 + 1) * 16'sd20 + c2 * 16'sd10;
        end
        run_check("D3-dominant");
        for (i = 0; i < S; i = i + 1)
            for (j = 0; j < D; j = j + 1)
                if (absdiff(DO[i][j], VM[i][j]) > 8) begin
                    $display("FAIL[D3-pass] O[%0d][%0d]=%0d V=%0d", i, j,
                             DO[i][j], VM[i][j]);
                    fail = fail + 1; $fatal(1, "attention dominant passthrough");
                end else pass = pass + 1;

        // D5 negative Q/K/V -> signed dot products; still a valid distribution.
        for (r2 = 0; r2 < S; r2 = r2 + 1)
            for (c2 = 0; c2 < D; c2 = c2 + 1) begin
                QM[r2][c2] = (((r2 + c2) & 1) ? -1 : 1) * (16'sd64  + r2*16'sd32);
                KM[r2][c2] = (((r2 + c2) & 1) ? 1 : -1) * (16'sd48  + c2*16'sd24);
                VM[r2][c2] = (c2 - 1) * 16'sd96;
            end
        run_check("D5-negatives");

        // D8 CONVEXITY / saturation boundary: all V = +max with uniform weights.
        // Every O element must round to exactly +max WITHOUT clamping (sat 0).
        for (r2 = 0; r2 < S; r2 = r2 + 1)
            for (c2 = 0; c2 < D; c2 = c2 + 1) begin
                QM[r2][c2] = 16'sd100;
                KM[r2][c2] = 16'sd100;
                VM[r2][c2] = `Q78_MAX;
            end
        run_check("D8-convex-maxV");
        for (i = 0; i < S; i = i + 1)
            for (j = 0; j < D; j = j + 1)
                if (DO[i][j] !== `Q78_MAX) begin
                    $display("FAIL[D8-maxV] O[%0d][%0d]=%0d expected %0d",
                             i, j, DO[i][j], `Q78_MAX);
                    fail = fail + 1; $fatal(1, "attention convex max");
                end else pass = pass + 1;
        if (sat !== 1'b0) begin
            $display("FAIL[D8-nosat] sat=1 for convex max-V (should never clamp)");
            fail = fail + 1; $fatal(1, "attention spurious sat");
        end else pass = pass + 1;

        // ======================= BASE-OFFSET INDEPENDENCE =================
        q_base = 5'd16; k_base = 5'd18; v_base = 5'd0; o_base = 5'd24;
        set_QK_identity;
        for (r2 = 0; r2 < S; r2 = r2 + 1)
            for (c2 = 0; c2 < D; c2 = c2 + 1)
                VM[r2][c2] = (c2 == 0) ? -16'sd64 : 16'sd128;
        run_check("D9-baseoffset");
        for (i = 0; i < S; i = i + 1)
            if (DO[i][0] !== -16'sd64 || DO[i][1] !== 16'sd128) begin
                $display("FAIL[D9-base] O row %0d = %0d %0d",
                         i, DO[i][0], DO[i][1]);
                fail = fail + 1; $fatal(1, "attention base offset");
            end else pass = pass + 1;
        q_base = 5'd0; k_base = 5'd2; v_base = 5'd4; o_base = 5'd6;

        // ======================= CONSTRAINED RANDOM =======================
        for (t = 0; t < NRAND; t = t + 1) begin
            if (t < 60)        randomize_ops(64);    // small (~+/-0.25)
            else if (t < 120)  randomize_ops(256);   // mid   (~+/-1.0)
            else               randomize_ops(512);   // edge of regime (~+/-2.0)
            run_check("R-rand");
        end

        // ======================= SUMMARY =======================
        if (fail != 0) begin
            $display("ATTENTION_PARAM_TB: %0d FAILED, %0d passed", fail, pass);
            $fatal(1, "attention_param_tb had failures");
        end else begin
            $display("ALL %0d TESTS PASSED", pass);
        end
        $finish;
    end

endmodule
