`timescale 1ns/1ps
//============================================================================
// ecc_secded_tb.v  --  BINDING correctness TB for the SECDED codec (P2.1)
//----------------------------------------------------------------------------
// Verifies that src/ecc_secded.v actually implements an extended-Hamming
// SECDED (72,64) codec, against an INDEPENDENT, X-aware golden model that
// re-derives the construction from first principles (not the same source).
//
//   CLEAN          : random words encode->decode with no injected error.
//                    Golden encode must match DUT code_out bit-for-bit;
//                    decode must return the word, single_err=0, double_err=0.
//   SINGLE-CORRECT : for several random words flip EVERY codeword bit, one at
//                    a time, all CODE_W positions -> single_err=1, double_err=0,
//                    data_out == original (every single-bit error corrected;
//                    the overall-parity-bit flip is the data-OK single case).
//   DOUBLE-DETECT  : for several random words flip EVERY distinct pair of
//                    codeword bits (exhaustive C(CODE_W,2)) -> double_err=1,
//                    single_err=0, and NEVER silently miscorrected to a wrong
//                    single (the core SECDED guarantee).
//
// Every DUT response is also cross-checked, bit-for-bit, against the golden
// decode for that exact received word.  X-aware throughout (=== / !==).
//============================================================================
module ecc_secded_tb;
    // TB-only width lint relaxations: fixed-width string message ports and
    // 1-bit shift counts. The DUT (src/ecc_secded.v) is linted separately.
    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off WIDTHTRUNC */

    //------------------------------------------------------------------
    // Geometry (independently derived; must match the DUT's localparams)
    //------------------------------------------------------------------
    localparam integer DATA_W = 64;

    function integer calc_p;
        input integer dw;
        integer p;
        begin
            p = 0;
            while ((1 << p) < (dw + p + 1)) p = p + 1;
            calc_p = p;
        end
    endfunction

    localparam integer P      = calc_p(DATA_W);
    localparam integer HCW    = DATA_W + P;
    localparam integer CODE_W = DATA_W + P + 1;

    function gv_is_pow2;                  // 1 iff i is a power of two
        input integer i;
        begin
            gv_is_pow2 = (i > 0) && ((i & (i - 1)) == 0);
        end
    endfunction

    //------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------
    reg  [DATA_W-1:0] data_in;
    wire [CODE_W-1:0] code_out;
    reg  [CODE_W-1:0] code_in;
    wire [DATA_W-1:0] data_out;
    wire              single_err, double_err;

    ecc_secded #(.DATA_W(DATA_W)) dut (
        .data_in(data_in), .code_out(code_out),
        .code_in(code_in), .data_out(data_out),
        .single_err(single_err), .double_err(double_err)
    );

    //------------------------------------------------------------------
    // INDEPENDENT GOLDEN  --  encode
    //------------------------------------------------------------------
    function [CODE_W-1:0] golden_encode;
        input [DATA_W-1:0] d;
        reg [HCW:1] g;
        integer i, k, di;
        reg       par, ov;
        begin
            g = {HCW{1'b0}};
            // scatter data into the non-power-of-two positions (ascending)
            di = 0;
            for (i = 1; i <= HCW; i = i + 1)
                if (!gv_is_pow2(i)) begin
                    g[i] = d[di];
                    di   = di + 1;
                end
            // each Hamming parity bit = XOR of the DATA bits it covers
            for (k = 0; k < P; k = k + 1) begin
                par = 1'b0;
                for (i = 1; i <= HCW; i = i + 1)
                    if (!gv_is_pow2(i) && ((i & (1 << k)) != 0))
                        par = par ^ g[i];
                g[(1 << k)] = par;
            end
            // overall parity = XOR of the whole Hamming codeword
            ov = ^g;
            golden_encode = {CODE_W{1'b0}};
            for (i = 1; i <= HCW; i = i + 1)
                golden_encode[i-1] = g[i];
            golden_encode[CODE_W-1] = ov;
        end
    endfunction

    //------------------------------------------------------------------
    // INDEPENDENT GOLDEN  --  decode  (re-derives syndrome + the 4 cases)
    //------------------------------------------------------------------
    task golden_decode;
        input  [CODE_W-1:0] cin;
        output [DATA_W-1:0] g_data;
        output              g_single;
        output              g_double;
        reg [HCW:1] rx, cor;
        reg [P:0]   synd;       // wide enough to hold a position up to HCW
        reg         oc;
        integer i, k, di;
        begin
            for (i = 1; i <= HCW; i = i + 1)
                rx[i] = cin[i-1];

            synd = 0;
            for (k = 0; k < P; k = k + 1)
                for (i = 1; i <= HCW; i = i + 1)
                    if ((i & (1 << k)) != 0)
                        synd[k] = synd[k] ^ rx[i];

            oc = ^cin;

            cor      = rx;
            g_single = 1'b0;
            g_double = 1'b0;

            if (synd == 0) begin
                if (oc) g_single = 1'b1;          // case 4: parity-bit flip
            end else if (oc) begin                // case 2: single correctable
                g_single = 1'b1;
                if (synd <= HCW[P:0]) cor[synd] = ~rx[synd];
            end else begin                        // case 3: double detect
                g_double = 1'b1;
            end

            g_data = {DATA_W{1'b0}};
            di = 0;
            for (i = 1; i <= HCW; i = i + 1)
                if (!gv_is_pow2(i)) begin
                    g_data[di] = cor[i];
                    di = di + 1;
                end
        end
    endtask

    //------------------------------------------------------------------
    // Harness
    //------------------------------------------------------------------
    integer tests   = 0;
    integer fails   = 0;
    integer n_clean = 0, n_single = 0, n_double = 0;

    task check;
        input [320:0] name;
        input         cond;
        begin
            tests = tests + 1;
            if (cond !== 1'b1) begin
                fails = fails + 1;
                $display("FAIL [%0d]: %0s", fails, name);
            end
        end
    endtask

    // compare a full DUT decode response to the golden, X-aware
    task expect_decode;
        input [320:0] tag;
        input [DATA_W-1:0] gd;
        input              gs;
        input              gdo;
        begin
            check({tag, " data==golden"},   data_out   === gd);
            check({tag, " single==golden"}, single_err === gs);
            check({tag, " double==golden"}, double_err === gdo);
        end
    endtask

    localparam integer CLEAN_WORDS  = 256;
    localparam integer SINGLE_WORDS = 64;
    localparam integer DOUBLE_WORDS = 16;

    reg  [DATA_W-1:0] dword, gdata;
    reg               gsing, gdoub;
    reg  [CODE_W-1:0] enc, rx, gcode;
    integer w, a, b;

    initial begin
        if (CODE_W != 72)
            $display("NOTE: CODE_W=%0d (expected 72 for DATA_W=64)", CODE_W);

        //==============================================================
        // 1) CLEAN
        //==============================================================
        for (w = 0; w < CLEAN_WORDS; w = w + 1) begin
            dword   = {$random, $random};
            data_in = dword; #1;
            enc     = code_out;
            gcode   = golden_encode(dword);

            check("clean: code_out==golden_encode", enc === gcode);

            code_in = enc; #1;
            golden_decode(enc, gdata, gsing, gdoub);
            expect_decode("clean", gdata, gsing, gdoub);
            check("clean: data recovered",  data_out   === dword);
            check("clean: no single",       single_err === 1'b0);
            check("clean: no double",       double_err === 1'b0);
            n_clean = n_clean + 1;
        end

        //==============================================================
        // 2) SINGLE-CORRECT  -- exhaustive over ALL codeword positions
        //==============================================================
        for (w = 0; w < SINGLE_WORDS; w = w + 1) begin
            dword   = {$random, $random};
            data_in = dword; #1;
            enc     = code_out;

            for (a = 0; a < CODE_W; a = a + 1) begin
                rx      = enc ^ (1'b1 << a);
                code_in = rx; #1;
                golden_decode(rx, gdata, gsing, gdoub);
                expect_decode("single", gdata, gsing, gdoub);
                // semantic guarantees for ANY single flip:
                check("single: single_err asserted", single_err === 1'b1);
                check("single: no double",           double_err === 1'b0);
                check("single: data recovered",      data_out   === dword);
                n_single = n_single + 1;
            end
        end

        //==============================================================
        // 3) DOUBLE-DETECT  -- exhaustive over ALL distinct bit pairs
        //==============================================================
        for (w = 0; w < DOUBLE_WORDS; w = w + 1) begin
            dword   = {$random, $random};
            data_in = dword; #1;
            enc     = code_out;

            for (a = 0; a < CODE_W; a = a + 1)
                for (b = a + 1; b < CODE_W; b = b + 1) begin
                    rx      = enc ^ (1'b1 << a) ^ (1'b1 << b);
                    code_in = rx; #1;
                    golden_decode(rx, gdata, gsing, gdoub);
                    expect_decode("double", gdata, gsing, gdoub);
                    // core SECDED guarantee: a 2-bit error is FLAGGED, and
                    // never silently miscorrected to a (wrong) single.
                    check("double: double_err asserted", double_err === 1'b1);
                    check("double: single deasserted",   single_err === 1'b0);
                    n_double = n_double + 1;
                end
        end

        //==============================================================
        // verdict
        //==============================================================
        $display("clean words           : %0d", n_clean);
        $display("single-flip cases     : %0d (all %0d positions x %0d words)",
                 n_single, CODE_W, SINGLE_WORDS);
        $display("double-flip pair cases: %0d (all C(%0d,2) x %0d words)",
                 n_double, CODE_W, DOUBLE_WORDS);

        if (fails == 0)
            $display("ALL %0d TESTS PASSED", tests);
        else begin
            $display("%0d of %0d CHECKS FAILED", fails, tests);
            $fatal;
        end
        $finish;
    end
    /* verilator lint_on WIDTHTRUNC  */
    /* verilator lint_on WIDTHEXPAND */
endmodule
