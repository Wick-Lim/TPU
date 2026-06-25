`include "tpu_defs.vh"
//============================================================================
// fused_ops_unit_tb  --  self-checking unit TB for fused_ops_unit (SPEC §5.5)
//----------------------------------------------------------------------------
// DUT
//   fused_ops_unit: combinational tile-domain elementwise post-op.  Given an
//   opcode and one 128-bit input tile (4 packed Q7.8 lanes, each sign-extended
//   into a 32-bit lane), it applies a per-lane saturating RELU when the opcode
//   is FUSE_GEMM_RELU / FUSE_CONV_RELU, else passes the tile through unchanged.
//
// INDEPENDENT GOLDEN MODEL
//   The reference is computed a DIFFERENT way than the DUT: the TB unpacks the
//   tile into four 'real' floating-point values (interpreting each lane as a
//   signed Q7.8: real = signed_lane / 256.0), applies RELU in the REAL domain
//   (max(0.0, x)), then re-quantizes back to a signed Q7.8 lane at the boundary
//   ($rtoi(round(r * 256.0)), clamped to the Q7.8 range) and re-packs.  Pass-
//   through is modeled in real the same way (identity).  The golden therefore
//   never replicates the DUT's integer bit-twiddling; it exists to catch a DUT
//   arithmetic bug.  RELU on an already-in-range Q7.8 is EXACT, so the result
//   is compared BIT-EXACT (no tolerance) -- the only rounding in the golden is
//   the exact /256 then *256 round-trip of an integer, which is lossless.
//
// COVERAGE
//   T1  directed corner tiles: all-zero, all-positive, all-negative, mixed
//       sign, +0/-0, one-hot positive, one-hot negative, Q7.8 max/min lanes,
//       small +/-1 LSB values, full-range sign-extended extremes.
//   T2  pass-through: every NON-fused opcode (NOP, ALU ops, mem ops, the bare
//       tensor opcodes, an illegal opcode) leaves an arbitrary mixed-sign tile
//       BIT-IDENTICAL; and both FUSE opcodes RELU the same tile identically.
//   T3  sat output is ALWAYS 0 (RELU introduces no new saturation), checked on
//       every directed and random vector.
//   T4  constrained-random (seeded $random, >=200 vectors): random 128-bit
//       tiles x random opcode (biased so RELU and pass-through are both well
//       exercised), checked against the real-domain golden, bit-exact.
//
// GATES: prints "ALL <N> TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module fused_ops_unit_tb;

    // ---- DUT ports (combinational, no clock needed) ----
    reg  [`OPCODE_W-1:0] opcode;
    reg                  dense16;     // 0 = 32-bit lane (GEMM), 1 = dense-16 (CONV)
    reg  [`LINE_W-1:0]   tile_in;
    wire [`LINE_W-1:0]   tile_out;
    wire                 sat;

    // ---- DUT ----
    fused_ops_unit dut (
        .opcode   (opcode),
        .dense16  (dense16),
        .tile_in  (tile_in),
        .tile_out (tile_out),
        .sat      (sat)
    );

    integer pass;
    integer fail;
    integer seed;
    integer k;

    // ---------------------------------------------------------------------
    // INDEPENDENT golden, computed in the REAL domain.
    //   relu_active : does this opcode apply RELU?
    //   gold_line() : produce the expected 128-bit output line for (op, in).
    // The lanes are interpreted as signed Q7.8: value = signed(lane)/256.0.
    // ---------------------------------------------------------------------

    // Is this opcode a RELU-fusing opcode?  Modeled independently of the DUT's
    // boolean by an explicit two-way membership test.
    function relu_active;
        input [`OPCODE_W-1:0] op;
        begin
            if (op == `OP_FUSE_GEMM_RELU)      relu_active = 1'b1;
            else if (op == `OP_FUSE_CONV_RELU) relu_active = 1'b1;
            else                               relu_active = 1'b0;
        end
    endfunction

    // Signed integer value of one 32-bit lane.
    function signed [`LANE_W-1:0] lane_signed;
        input [`LANE_W-1:0] lane;
        begin
            lane_signed = $signed(lane);
        end
    endfunction

    // Golden RELU of one lane via REAL math:
    //   r = signed_lane / 256.0 ; y = max(0.0, r) ; quantize back to Q7.8.
    // Because the input lane is an integer multiple of 1/256, the /256 then
    // *256 round-trip is lossless; max(0,.) of a representable value is
    // representable, so this is exact (compared bit-exact, no tolerance).
    function [`LANE_W-1:0] gold_relu_lane;
        input [`LANE_W-1:0] lane;
        real                r;
        real                y;
        integer             q;
        begin
            r = lane_signed(lane) / 256.0;     // Q7.8 -> real
            y = (r < 0.0) ? 0.0 : r;           // RELU in the real domain
            // Re-quantize: round-half-away handled by adding/subtracting 0.5
            // before truncation; y >= 0 here so simple +0.5 rounding suffices.
            q = $rtoi(y * 256.0 + 0.5);        // real -> Q7.8 integer
            // Clamp to the Q7.8 signed range (defensive; RELU output of an
            // in-range tile is always within range so this never trims here).
            if (q > `Q78_MAX_VAL) q = `Q78_MAX_VAL;
            if (q < `Q78_MIN_VAL) q = `Q78_MIN_VAL;
            // Sign-extend the 16-bit Q7.8 result back into the 32-bit lane,
            // matching the DUT's lane format (SPEC §1.3).
            gold_relu_lane = `TPU_SEXT16(q[`ELEM_W-1:0]);
        end
    endfunction

    // Golden output LINE for a given opcode + input line.
    function [`LINE_W-1:0] gold_line;
        input [`OPCODE_W-1:0] op;
        input [`LINE_W-1:0]   in;
        reg   [`LINE_W-1:0]   out;
        begin
            if (relu_active(op)) begin
                out[0*`LANE_W +: `LANE_W] = gold_relu_lane(in[0*`LANE_W +: `LANE_W]);
                out[1*`LANE_W +: `LANE_W] = gold_relu_lane(in[1*`LANE_W +: `LANE_W]);
                out[2*`LANE_W +: `LANE_W] = gold_relu_lane(in[2*`LANE_W +: `LANE_W]);
                out[3*`LANE_W +: `LANE_W] = gold_relu_lane(in[3*`LANE_W +: `LANE_W]);
            end else begin
                out = in;                      // pass-through (identity)
            end
            gold_line = out;
        end
    endfunction

    // Pack a 128-bit line from four signed Q7.8 element values (sign-extended).
    function [`LINE_W-1:0] pack_q78;
        input signed [`ELEM_W-1:0] e3, e2, e1, e0;
        begin
            pack_q78 = { `TPU_SEXT16(e3), `TPU_SEXT16(e2),
                         `TPU_SEXT16(e1), `TPU_SEXT16(e0) };
        end
    endfunction

    // Drive the DUT with (op,in), settle, and compare tile_out + sat against
    // the independent real-domain golden.  Bit-exact (exact integer RELU).
    task check;
        input [`OPCODE_W-1:0] op;
        input [`LINE_W-1:0]   in;
        input [255:0]         tag;
        reg   [`LINE_W-1:0]   exp;
        begin
            opcode  = op;
            dense16 = 1'b0;                    // 32-bit-lane (GEMM) packing
            tile_in = in;
            #1;                                // settle combinational logic
            exp = gold_line(op, in);
            if (tile_out !== exp) begin
                $display("FAIL[%0s] op=%02x in=%032x exp=%032x got=%032x",
                         tag, op, in, exp, tile_out);
                fail = fail + 1;
                $fatal(1, "fused_ops_unit tile_out mismatch");
            end else begin
                pass = pass + 1;
            end
            // sat must always be 0 for this stage.
            if (sat !== 1'b0) begin
                $display("FAIL[%0s] op=%02x sat asserted (got %b, expected 0)",
                         tag, op, sat);
                fail = fail + 1;
                $fatal(1, "fused_ops_unit sat should never assert");
            end else begin
                pass = pass + 1;
            end
        end
    endtask

    // ---------------------------------------------------------------------
    // DENSE-16 (CONV) golden + check.  In dense16 mode the 128-bit line is 8
    // independent 16-bit Q7.8 elements [16*e +: 16]; RELU each independently.
    // The golden RELUs each 16-bit element in the REAL domain, independent of
    // the DUT's bit-twiddling, and is bit-exact for in-range Q7.8 data.
    // ---------------------------------------------------------------------
    function [`ELEM_W-1:0] gold_relu_elem;
        input [`ELEM_W-1:0] e;
        real    r; real y; integer q;
        begin
            r = $signed(e) / 256.0;
            y = (r < 0.0) ? 0.0 : r;
            q = $rtoi(y * 256.0 + 0.5);
            if (q > `Q78_MAX_VAL) q = `Q78_MAX_VAL;
            if (q < `Q78_MIN_VAL) q = `Q78_MIN_VAL;
            gold_relu_elem = q[`ELEM_W-1:0];
        end
    endfunction

    function [`LINE_W-1:0] gold_line_dense;
        input [`OPCODE_W-1:0] op;
        input [`LINE_W-1:0]   in;
        reg   [`LINE_W-1:0]   out;
        integer               e;
        begin
            if (relu_active(op)) begin
                for (e = 0; e < (`LINE_W/`ELEM_W); e = e + 1)
                    out[e*`ELEM_W +: `ELEM_W] =
                        gold_relu_elem(in[e*`ELEM_W +: `ELEM_W]);
            end else begin
                out = in;
            end
            gold_line_dense = out;
        end
    endfunction

    task check_dense;
        input [`OPCODE_W-1:0] op;
        input [`LINE_W-1:0]   in;
        input [255:0]         tag;
        reg   [`LINE_W-1:0]   exp;
        begin
            opcode  = op;
            dense16 = 1'b1;                    // dense-16 (CONV) packing
            tile_in = in;
            #1;
            exp = gold_line_dense(op, in);
            if (tile_out !== exp) begin
                $display("FAIL[%0s] op=%02x dense in=%032x exp=%032x got=%032x",
                         tag, op, in, exp, tile_out);
                fail = fail + 1;
                $fatal(1, "fused_ops_unit dense16 tile_out mismatch");
            end else pass = pass + 1;
            if (sat !== 1'b0) begin
                $display("FAIL[%0s] dense sat asserted", tag);
                fail = fail + 1;
                $fatal(1, "fused_ops_unit dense sat should never assert");
            end else pass = pass + 1;
        end
    endtask

    // Pack a 128-bit line from eight DENSE 16-bit Q7.8 elements (low-first).
    function [`LINE_W-1:0] pack_dense;
        input signed [`ELEM_W-1:0] e7,e6,e5,e4,e3,e2,e1,e0;
        begin
            pack_dense = {e7,e6,e5,e4,e3,e2,e1,e0};
        end
    endfunction

    initial begin
        pass = 0;
        fail = 0;
        seed = 32'h0F00D123;

        // ---------------------------------------------------------------
        // T1: directed corner-case tiles under FUSE_GEMM_RELU.
        // ---------------------------------------------------------------
        // all-zero
        check(`OP_FUSE_GEMM_RELU, pack_q78(16'sd0,16'sd0,16'sd0,16'sd0), "T1-allzero");
        // all-positive (mix of magnitudes)
        check(`OP_FUSE_GEMM_RELU, pack_q78(16'sd1,16'sd100,16'sd5000,16'sd32767), "T1-allpos");
        // all-negative -> all clamp to 0
        check(`OP_FUSE_GEMM_RELU, pack_q78(-16'sd1,-16'sd100,-16'sd5000,-16'sd32768), "T1-allneg");
        // mixed sign (alternating)
        check(`OP_FUSE_GEMM_RELU, pack_q78(16'sd256,-16'sd256,16'sd1,-16'sd1), "T1-mixed");
        // one-hot positive lane, rest negative
        check(`OP_FUSE_GEMM_RELU, pack_q78(-16'sd5,16'sd12345,-16'sd9,-16'sd3), "T1-onehotpos");
        // one-hot negative lane, rest positive
        check(`OP_FUSE_GEMM_RELU, pack_q78(16'sd5,-16'sd12345,16'sd9,16'sd3), "T1-onehotneg");
        // Q7.8 max/min in lanes
        check(`OP_FUSE_GEMM_RELU, pack_q78(16'sd32767,-16'sd32768,16'sd32767,-16'sd32768), "T1-extremes");
        // +/-1 LSB
        check(`OP_FUSE_GEMM_RELU, pack_q78(16'sd1,-16'sd1,16'sd1,-16'sd1), "T1-lsb");
        // largest negative only
        check(`OP_FUSE_GEMM_RELU, pack_q78(-16'sd32768,-16'sd32768,-16'sd32768,-16'sd32768), "T1-allmin");
        // same set under FUSE_CONV_RELU (must behave identically)
        check(`OP_FUSE_CONV_RELU, pack_q78(16'sd256,-16'sd256,16'sd1,-16'sd1), "T1-conv-mixed");
        check(`OP_FUSE_CONV_RELU, pack_q78(-16'sd1,-16'sd100,-16'sd5000,-16'sd32768), "T1-conv-allneg");

        // ---------------------------------------------------------------
        // T2: pass-through for every non-fused opcode (BIT-IDENTICAL),
        //     plus both FUSE opcodes RELU the same tile identically.
        // ---------------------------------------------------------------
        // an arbitrary mixed-sign tile to push through
        // (lane3=+1000, lane2=-2000, lane1=+30000, lane0=-30000)
        check(`OP_NOP,       pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-nop");
        check(`OP_LOADI,     pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-loadi");
        check(`OP_LOAD,      pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-load");
        check(`OP_STORE,     pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-store");
        check(`OP_ADD,       pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-add");
        check(`OP_SUB,       pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-sub");
        check(`OP_AND,       pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-and");
        check(`OP_OR,        pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-or");
        check(`OP_XOR,       pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-xor");
        check(`OP_SHL,       pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-shl");
        check(`OP_SHR,       pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-shr");
        check(`OP_RELU,      pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-scalarrelu");
        check(`OP_DMA,       pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-dma");
        check(`OP_GATHER,    pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-gather");
        check(`OP_TLOAD,     pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-tload");
        check(`OP_GEMM,      pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-gemm");
        check(`OP_CONV2D,    pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-conv2d");
        check(`OP_SOFTMAX,   pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-softmax");
        check(`OP_ATTENTION, pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-attention");
        check(8'hFE,         pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-illegal");
        // both FUSE opcodes on the same tile
        check(`OP_FUSE_GEMM_RELU, pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-fusegemm");
        check(`OP_FUSE_CONV_RELU, pack_q78(16'sd1000,-16'sd2000,16'sd30000,-16'sd30000), "T2-fuseconv");

        // ---------------------------------------------------------------
        // T4: constrained-random sweep, >=200 vectors.
        //   Random 128-bit tile (packed from four random signed Q7.8 lanes so
        //   every lane is a legal sign-extended Q7.8) x random opcode chosen so
        //   both RELU and pass-through are well exercised.  Checked bit-exact
        //   against the real-domain golden; sat asserted-0 checked too.
        // ---------------------------------------------------------------
        for (k = 0; k < 300; k = k + 1) begin : rnd_loop
            reg [`LINE_W-1:0]   rin;
            reg [`OPCODE_W-1:0] rop;
            reg [15:0]          e [0:3];
            integer             j;
            begin
                // Build four random legal Q7.8 lanes and sign-extend them.
                for (j = 0; j < 4; j = j + 1)
                    e[j] = $random(seed);              // 16-bit signed Q7.8 word
                rin = { `TPU_SEXT16(e[3]), `TPU_SEXT16(e[2]),
                        `TPU_SEXT16(e[1]), `TPU_SEXT16(e[0]) };
                // Choose an opcode: ~1/3 GEMM_RELU, ~1/3 CONV_RELU, ~1/3 a
                // random pass-through opcode, so all paths get heavy coverage.
                case (k % 3)
                    0:       rop = `OP_FUSE_GEMM_RELU;
                    1:       rop = `OP_FUSE_CONV_RELU;
                    default: rop = $random(seed);      // arbitrary -> pass-through path
                endcase
                check(rop, rin, "T4-rand");
            end
        end

        // ---------------------------------------------------------------
        // T5: DENSE-16 (CONV) RELU granularity.  CONV2D packs 2 px per 32-bit
        //   lane, so RELU MUST run per-16-bit element.  These cases FAIL a
        //   32-bit-lane RELU and PASS a dense-16 RELU.
        // ---------------------------------------------------------------
        // The exact audit regression case: dense pixels (p0=+50,p1=-30,p2=-40,
        //   p3=+60) -> (50,0,0,60).  A 32-bit-lane RELU would give (0,0,-40,60).
        check_dense(`OP_FUSE_CONV_RELU,
            pack_dense(16'sd0,16'sd0,16'sd0,16'sd0,
                       16'sd60,-16'sd40,-16'sd30,16'sd50), "T5-audit");
        // Positive-even / negative-odd in the SAME lane: the even pixel must
        //   SURVIVE (a 32-bit RELU would zero the whole lane on the odd sign).
        check_dense(`OP_FUSE_CONV_RELU,
            pack_dense(16'sd0,16'sd0,16'sd0,16'sd0,
                       16'sd0,16'sd0,-16'sd1,16'sd100), "T5-posEven-negOdd");
        // Negative-even / positive-odd: even must be zeroed, odd survive (a
        //   32-bit RELU would let the negative even pixel LEAK through).
        check_dense(`OP_FUSE_CONV_RELU,
            pack_dense(16'sd0,16'sd0,16'sd0,16'sd0,
                       16'sd0,16'sd0,16'sd77,-16'sd5), "T5-negEven-posOdd");
        // All-negative dense -> all zero; all-positive dense -> unchanged.
        check_dense(`OP_FUSE_CONV_RELU,
            pack_dense(-16'sd1,-16'sd2,-16'sd3,-16'sd4,
                       -16'sd5,-16'sd6,-16'sd7,-16'sd8), "T5-allneg");
        check_dense(`OP_FUSE_CONV_RELU,
            pack_dense(16'sd1,16'sd2,16'sd3,16'sd4,
                       16'sd5,16'sd6,16'sd7,16'sd8), "T5-allpos");
        // FUSE_GEMM_RELU under dense16 also RELUs per-element (opcode-agnostic
        //   granularity); pass-through opcode under dense16 is still identity.
        check_dense(`OP_NOP,
            pack_dense(16'sd9,-16'sd9,16'sd9,-16'sd9,
                       16'sd9,-16'sd9,16'sd9,-16'sd9), "T5-passthru");
        // Constrained-random dense-16 sweep (>=200 vectors).
        for (k = 0; k < 250; k = k + 1) begin : rnd_dense
            reg [`LINE_W-1:0]   rin;
            reg [`OPCODE_W-1:0] rop;
            reg [15:0]          de [0:7];
            integer             j;
            begin
                for (j = 0; j < 8; j = j + 1) de[j] = $random(seed);
                rin = {de[7],de[6],de[5],de[4],de[3],de[2],de[1],de[0]};
                rop = (k % 2 == 0) ? `OP_FUSE_CONV_RELU : `OP_FUSE_GEMM_RELU;
                check_dense(rop, rin, "T5-rand");
            end
        end

        // ---------------------------------------------------------------
        if (fail != 0) begin
            $display("FUSED_OPS_UNIT TB: %0d FAILURES", fail);
            $fatal(1, "fused_ops_unit_tb FAILED");
        end
        $display("ALL %0d TESTS PASSED", pass);
        $finish;
    end
endmodule
