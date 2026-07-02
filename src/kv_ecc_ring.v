`timescale 1ns/1ps
//============================================================================
// kv_ecc_ring.v  --  SECDED-protected wide-row KV ring buffer  (ROADMAP P2 / C6)
//----------------------------------------------------------------------------
// A tiny stand-in for the kv_cache_pager resident-window `ring` array whose row
// (ROW_BITS, e.g. 768 in the pager) is NOT natively a multiple of the 64-bit
// SECDED lane (see docs/P2_MEMORY_MAP.md "ECC lane-partition note").  This module
// exercises the lane-partition logic that C6 must add:
//
//   * Each wide ROW_BITS row is split into NLANES = ceil(ROW_BITS/64) lanes of
//     64 bits.  The FINAL lane is RAGGED when ROW_BITS is not a multiple of 64:
//     its high (64 - valid) bits are ZERO-PADDED before encode so a full 64-bit
//     `ecc_secded` codec still applies.  Only the valid low bits are returned on
//     read; the pad bits are re-created (constant 0) and never stored as data.
//   * Each lane is protected by an extended-Hamming SECDED codeword (CODE_W=72
//     for a 64-bit payload) stored ALONGSIDE the data lane -- the raw payload is
//     never stored, only the codeword.  Encode on write, decode+correct on read.
//   * Per-row flags aggregate the lanes: serr = OR of every lane's corrected
//     single-bit error, derr = OR of every lane's detected double-bit error.  A
//     DBU in ANY lane poisons the whole row (derr high) -- the conservative
//     choice flagged in the memory map.
//   * A back-door fault-injection port XORs an arbitrary mask into ONE stored
//     lane codeword (flip 1 stored bit -> correctable SBU; flip 2 -> DBU),
//     mirroring ecc_mem_wrap's bd_* port at ring granularity.
//
// RING SHAPE (mirrors kv_cache_pager access at tiny scale)
//   DEPTH rows (power-of-two).  Append writes row waddr; read-by-index reads
//   row raddr.  Synchronous 2-stage read like ecc_mem_wrap: re latches the
//   fetched lane codewords (cycle +1), the combinational decode is registered
//   (cycle +2), so rdata/serr/derr are valid two rising edges after re/raddr.
//
// The Hamming math is NOT reimplemented -- 2*NLANES ecc_secded instances (one
// encode + one decode per lane) do it.  Pure register array + combinational
// codecs; synchronous active-high reset; no latch, no comb loop.
//============================================================================
module kv_ecc_ring #(
    parameter integer ROW_BITS = 100,  // wide KV row width (may be ragged mod 64)
    parameter integer DEPTH    = 4     // ring rows (power-of-two)
)(
    clk, rst,
    // append / write-by-index port (SECDED-encoded)
    we, waddr, wdata,
    // read-by-index port (SECDED-decoded, 2-cycle)
    re, raddr, rdata, serr, derr,
    // back-door raw-codeword fault injection: XOR bd_xor into stored lane
    bd_we, bd_addr, bd_lane, bd_xor
);

    //------------------------------------------------------------------
    // Geometry.  LANE_W-bit SECDED lanes; CODE_W derived exactly as
    // ecc_secded derives it from a LANE_W payload.
    //------------------------------------------------------------------
    localparam integer LANE_W = 64;

    function integer calc_p;                 // Hamming parity bits for dw payload
        input integer dw;
        integer p;
        begin
            p = 0;
            while ((1 << p) < (dw + p + 1)) p = p + 1;
            calc_p = p;
        end
    endfunction

    function integer clog2;                  // index bits for n items (min 1)
        input integer n;
        integer v;
        begin
            clog2 = 0;
            v     = n - 1;
            while (v > 0) begin
                clog2 = clog2 + 1;
                v     = v >> 1;
            end
            if (clog2 == 0) clog2 = 1;
        end
    endfunction

    localparam integer P        = calc_p(LANE_W);            // = 7 for 64b
    localparam integer CODE_W   = LANE_W + P + 1;            // = 72 for 64b
    localparam integer NLANES   = (ROW_BITS + LANE_W - 1) / LANE_W;
    localparam integer ADDR_W   = clog2(DEPTH);
    localparam integer LANE_IW  = clog2(NLANES);
    localparam integer NWORDS   = DEPTH * NLANES;

    // ---- ports (widths depend on the localparams above) ----------------
    input  wire                  clk;
    input  wire                  rst;

    input  wire                  we;
    input  wire [ADDR_W-1:0]     waddr;
    input  wire [ROW_BITS-1:0]   wdata;

    input  wire                  re;
    input  wire [ADDR_W-1:0]     raddr;
    output reg  [ROW_BITS-1:0]   rdata;
    output reg                   serr;
    output reg                   derr;

    input  wire                  bd_we;
    input  wire [ADDR_W-1:0]     bd_addr;
    input  wire [LANE_IW-1:0]    bd_lane;
    input  wire [CODE_W-1:0]     bd_xor;

    //------------------------------------------------------------------
    // Per-lane payload slices (write side) + reconstruction (read side).
    // The final lane is zero-padded to LANE_W when ROW_BITS % 64 != 0.
    //------------------------------------------------------------------
    wire [LANE_W-1:0] wlane   [0:NLANES-1];   // padded write payload per lane
    wire [CODE_W-1:0] enc_code[0:NLANES-1];   // encoded codeword per lane
    wire [LANE_W-1:0] dec_data[0:NLANES-1];   // decoded/corrected payload
    wire [NLANES-1:0] dec_serr;               // per-lane corrected single error
    wire [NLANES-1:0] dec_derr;               // per-lane detected double error
    wire [ROW_BITS-1:0] rdata_c;              // reconstructed row (combinational)

    reg  [CODE_W-1:0] mem [0:NWORDS-1];       // stored codewords (never raw data)
    reg  [CODE_W-1:0] rd_code [0:NLANES-1];   // fetched codewords (read stage 1)
    reg               rd_valid_q;             // rd_code holds a genuine fetch

    genvar j;
    generate
        for (j = 0; j < NLANES; j = j + 1) begin : LANE
            // Valid payload bits of this lane: full LANE_W except a ragged tail.
            localparam integer LO  = j * LANE_W;
            localparam integer WID = (LO + LANE_W <= ROW_BITS)
                                     ? LANE_W : (ROW_BITS - LO);

            // Zero-pad the ragged final lane's high bits before encode.
            assign wlane[j] = { {(LANE_W-WID){1'b0}}, wdata[LO +: WID] };

            // ENCODE on write: payload -> codeword (decode side unused).
            wire [LANE_W-1:0] enc_d_unused;
            wire              enc_s_unused, enc_d2_unused;
            ecc_secded #(.DATA_W(LANE_W)) u_enc (
                .data_in    (wlane[j]),
                .code_out   (enc_code[j]),
                .code_in    ({CODE_W{1'b0}}),
                .data_out   (enc_d_unused),
                .single_err (enc_s_unused),
                .double_err (enc_d2_unused)
            );

            // DECODE on read: fetched codeword -> corrected payload + flags.
            wire [CODE_W-1:0] dec_c_unused;
            ecc_secded #(.DATA_W(LANE_W)) u_dec (
                .data_in    ({LANE_W{1'b0}}),
                .code_out   (dec_c_unused),
                .code_in    (rd_code[j]),
                .data_out   (dec_data[j]),
                .single_err (dec_serr[j]),
                .double_err (dec_derr[j])
            );

            // Reconstruct only the valid low bits of this lane (pad discarded).
            assign rdata_c[LO +: WID] = dec_data[j][WID-1:0];
        end
    endgenerate

    // Per-row aggregation: any lane's SBU/DBU raises the row flag.
    wire serr_c = |dec_serr;
    wire derr_c = |dec_derr;

    //------------------------------------------------------------------
    // Codeword array + read-fetch pipeline (single clocked block).
    //------------------------------------------------------------------
    integer k;
    always @(posedge clk) begin
        if (rst) begin
            for (k = 0; k < NWORDS; k = k + 1)
                mem[k] <= {CODE_W{1'b0}};
            for (k = 0; k < NLANES; k = k + 1)
                rd_code[k] <= {CODE_W{1'b0}};
            rd_valid_q <= 1'b0;
        end else begin
            // Encoded write: store all NLANES codewords of row waddr.
            if (we) begin
                for (k = 0; k < NLANES; k = k + 1)
                    mem[waddr*NLANES + k] <= enc_code[k];
            end
            // Back-door fault injection: XOR mask into one stored lane codeword.
            if (bd_we)
                mem[bd_addr*NLANES + bd_lane] <=
                    mem[bd_addr*NLANES + bd_lane] ^ bd_xor;
            // Synchronous read fetch (stage 1): grab all lane codewords.
            rd_valid_q <= re;
            if (re) begin
                for (k = 0; k < NLANES; k = k + 1)
                    rd_code[k] <= mem[raddr*NLANES + k];
            end
        end
    end

    //------------------------------------------------------------------
    // Register the decode results (stage 2) -> every output is a register.
    //------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            rdata <= {ROW_BITS{1'b0}};
            serr  <= 1'b0;
            derr  <= 1'b0;
        end else begin
            rdata <= rdata_c;
            serr  <= rd_valid_q & serr_c;
            derr  <= rd_valid_q & derr_c;
        end
    end

endmodule
