`timescale 1ns/1ps
//============================================================================
// ecc_secded.v  --  parameterized Hamming SECDED ECC codec  (ROADMAP P2.1)
//----------------------------------------------------------------------------
// Single-Error-Correct, Double-Error-Detect codec for the DDR5/Flash memory
// path (expert_cache_pf / ddr5_xbar / flash_xbar data words).
//
// Construction (extended Hamming code):
//   - DATA_W payload bits.
//   - P Hamming parity bits, P = smallest p with 2^p >= DATA_W + p + 1.
//   - 1 extra OVERALL parity bit covering the whole codeword (this is what
//     upgrades SEC Hamming to SECDED).
//   - CODE_W = DATA_W + P + 1.
//
// Bit layout (1-indexed Hamming positions 1..HCW, HCW = DATA_W + P):
//   - Position i is a Hamming PARITY bit iff i is a power of two (1,2,4,8,...).
//   - All other positions carry DATA bits, in ascending order.
//   - Hamming parity bit at position 2^k = XOR of every DATA position whose
//     index has bit k set (it covers itself, but only data sits elsewhere).
//   - code_out[i-1] = ham[i]   for i in 1..HCW  (Hamming codeword, LSB-first).
//   - code_out[CODE_W-1]       = OVERALL parity = XOR of all HCW Hamming bits
//                                (so XOR of the entire CODE_W word == 0 clean).
//
// DECODE:
//   syndrome[k] = XOR of received bits at positions with bit k set  (k=0..P-1).
//   syndrome    = sum syndrome[k]<<k  -> 1-indexed position of a single error.
//   oc          = XOR of the entire received codeword (overall-parity check):
//                 oc==1 => an ODD number of bit errors, oc==0 => EVEN.
//   Four cases:
//     1) syndrome==0 & oc==0 : NO ERROR.
//     2) syndrome!=0 & oc==1 : SINGLE error at bit[syndrome] -> CORRECT it.
//     3) syndrome!=0 & oc==0 : DOUBLE error -> double_err, do NOT correct.
//     4) syndrome==0 & oc==1 : single error IS the overall-parity bit;
//                              data is fine (single_err, data untouched).
//
// Pure combinational; no latch, no comb loop.  Provides BOTH encode + decode.
//============================================================================
module ecc_secded #(
    parameter integer DATA_W = 64
)(
    data_in, code_out,           // ENCODE: data_in -> code_out
    code_in, data_out, single_err, double_err  // DECODE: code_in -> out+flags
);

    //------------------------------------------------------------------
    // Parameter derivation (functions + localparams precede port widths)
    //------------------------------------------------------------------
    function integer calc_p;
        input integer dw;
        integer p;
        begin
            p = 0;
            while ((1 << p) < (dw + p + 1)) p = p + 1;
            calc_p = p;
        end
    endfunction

    localparam integer P      = calc_p(DATA_W); // # Hamming parity bits
    localparam integer HCW    = DATA_W + P;     // Hamming codeword width
    localparam integer CODE_W = DATA_W + P + 1; // full SECDED width
    localparam [P-1:0] HCW_W  = HCW[P-1:0];      // HCW as a P-bit bound

    function is_pow2;                            // 1 iff i is a power of two
        input integer i;
        begin
            is_pow2 = (i > 0) && ((i & (i - 1)) == 0);
        end
    endfunction

    // ---- ports (widths depend on the localparams above) ----------------
    input  wire [DATA_W-1:0]   data_in;
    output reg  [CODE_W-1:0]   code_out;
    input  wire [CODE_W-1:0]   code_in;
    output reg  [DATA_W-1:0]   data_out;
    output reg                 single_err;
    output reg                 double_err;

    integer i, k, di;

    //------------------------------------------------------------------
    // ENCODE
    //------------------------------------------------------------------
    reg [HCW:1] enc_ham;   // 1-indexed Hamming codeword
    reg         enc_par;

    always @(data_in) begin
        enc_ham = {HCW{1'b0}};
        // place data into non-power-of-two positions (ascending)
        di = 0;
        for (i = 1; i <= HCW; i = i + 1) begin
            if (!is_pow2(i)) begin
                enc_ham[i] = data_in[di];
                di = di + 1;
            end
        end
        // Hamming parity bits at each power-of-two position
        for (k = 0; k < P; k = k + 1) begin
            enc_ham[(1 << k)] = 1'b0;
            for (i = 1; i <= HCW; i = i + 1) begin
                if ((i != (1 << k)) && ((i & (1 << k)) != 0))
                    enc_ham[(1 << k)] = enc_ham[(1 << k)] ^ enc_ham[i];
            end
        end
        // overall parity over the whole Hamming codeword
        enc_par = ^enc_ham;
        // assemble: [CODE_W-1]=overall, [HCW-1:0]=ham (pos i -> bit i-1)
        for (i = 1; i <= HCW; i = i + 1)
            code_out[i-1] = enc_ham[i];
        code_out[CODE_W-1] = enc_par;
    end

    //------------------------------------------------------------------
    // DECODE
    //------------------------------------------------------------------
    reg [HCW:1]   dec_ham;     // received Hamming bits, 1-indexed
    reg [P-1:0]   syndrome;
    reg           oc;          // overall-parity check (XOR of whole word)
    reg [HCW:1]   cor_ham;     // corrected Hamming bits

    always @(code_in) begin
        // unpack received codeword
        for (i = 1; i <= HCW; i = i + 1)
            dec_ham[i] = code_in[i-1];

        // recompute syndrome
        for (k = 0; k < P; k = k + 1) begin
            syndrome[k] = 1'b0;
            for (i = 1; i <= HCW; i = i + 1)
                if ((i & (1 << k)) != 0)
                    syndrome[k] = syndrome[k] ^ dec_ham[i];
        end

        // overall-parity check: XOR of the ENTIRE received codeword
        oc = ^code_in;

        // default: pass through, no flags
        cor_ham    = dec_ham;
        single_err = 1'b0;
        double_err = 1'b0;

        if (syndrome == 0) begin
            if (oc) begin
                // Case 4: error in the overall-parity bit only; data OK
                single_err = 1'b1;
            end
            // else Case 1: no error
        end else begin
            if (oc) begin
                // Case 2: single correctable error at position syndrome
                single_err = 1'b1;
                if (syndrome <= HCW_W)
                    cor_ham[syndrome] = ~dec_ham[syndrome];
            end else begin
                // Case 3: double error detected, uncorrectable
                double_err = 1'b1;
            end
        end

        // extract data from corrected Hamming codeword
        di = 0;
        data_out = {DATA_W{1'b0}};
        for (i = 1; i <= HCW; i = i + 1) begin
            if (!is_pow2(i)) begin
                data_out[di] = cor_ham[i];
                di = di + 1;
            end
        end
    end

endmodule
