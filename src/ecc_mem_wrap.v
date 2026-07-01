`timescale 1ns/1ps
//============================================================================
// ecc_mem_wrap.v  --  SECDED-protected synchronous memory wrapper (ROADMAP P2.1)
//----------------------------------------------------------------------------
// A DATA_W x DEPTH synchronous RAM that transparently protects every stored
// word with an extended-Hamming SECDED code (see src/ecc_secded.v).  The
// wrapper NEVER stores raw payload: on a write it ENCODES wdata into a
// CODE_W-bit codeword (payload + Hamming parity + overall parity) and stores
// the codeword; on a read it DECODES the stored codeword back to the payload,
// transparently CORRECTING any single-bit flip and DETECTING any double-bit
// flip that occurred while the word sat in the array.
//
// This is the memory-path use of the codec: a bit that rots in DRAM/Flash is
// silently repaired (serr) rather than corrupting an activation/weight, and a
// 2-bit event is surfaced (derr) instead of being silently miscorrected.
//
// CODEC
//   The Hamming math is NOT reimplemented here -- two ecc_secded instances do
//   it: one drives the WRITE (encode) path, one drives the READ (decode) path.
//   CODE_W is derived from DATA_W exactly as ecc_secded derives it, so the
//   stored word width tracks the codec automatically.
//
// ALGORITHM / TIMING  (one-cycle synchronous read, every output registered)
//   WRITE (synchronous, dominated by reset):
//     we -> mem[waddr] <= encode(wdata)                (codeword stored)
//   READ (synchronous, 1-cycle latency):
//     re -> rd_code <= mem[raddr]                       (fetch codeword)
//        then decode(rd_code) is combinational, and its results are registered:
//        rdata <= corrected payload , serr <= single_err , derr <= double_err
//   So a word written at cycle N is, from cycle N+1 on, returned on rdata one
//   cycle after its raddr/re are presented, with serr/derr valid alongside.
//
//   To let a testbench (or a scrubber) INJECT a fault into a stored word, an
//   optional back-door write port overwrites a raw codeword in place:
//     bd_we -> mem[bd_addr] <= bd_code                  (raw, NOT re-encoded)
//   Left unconnected, bd_we defaults low -> pure ECC RAM (old behavior).
//
// SYNTHESIZABILITY
//   Synchronous ACTIVE-HIGH reset (dominates writes); every output registered;
//   single clocked always block for the array + read pipe; the codec instances
//   are pure combinational.  No latch, no combinational loop.
//============================================================================
module ecc_mem_wrap #(
    parameter integer DATA_W = 64,   // payload width per word
    parameter integer DEPTH  = 256   // number of words
)(
    clk, rst,
    // write port (synchronous, ECC-encoded)
    we, waddr, wdata,
    // read port (synchronous, 1-cycle, ECC-decoded)
    re, raddr, rdata, serr, derr,
    // optional back-door raw-codeword write (fault injection / scrub); off = old behavior
    bd_we, bd_addr, bd_code
);

    //------------------------------------------------------------------
    // Geometry -- CODE_W derived exactly as ecc_secded derives it.
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

    function integer clog2;                 // bits to index DEPTH words
        input integer n;
        integer v;
        begin
            clog2 = 0;
            v     = n - 1;
            while (v > 0) begin
                clog2 = clog2 + 1;
                v     = v >> 1;
            end
            if (clog2 == 0) clog2 = 1;       // at least 1 addr bit
        end
    endfunction

    localparam integer P      = calc_p(DATA_W);   // Hamming parity bits
    localparam integer CODE_W = DATA_W + P + 1;   // full SECDED codeword width
    localparam integer ADDR_W = clog2(DEPTH);     // address width

    // ---- ports (widths depend on the localparams above) ----------------
    input  wire                clk;
    input  wire                rst;

    input  wire                we;
    input  wire [ADDR_W-1:0]   waddr;
    input  wire [DATA_W-1:0]   wdata;

    input  wire                re;
    input  wire [ADDR_W-1:0]   raddr;
    output reg  [DATA_W-1:0]   rdata;
    output reg                 serr;
    output reg                 derr;

    input  wire                bd_we;
    input  wire [ADDR_W-1:0]   bd_addr;
    input  wire [CODE_W-1:0]   bd_code;

    //------------------------------------------------------------------
    // WRITE-path codec: encode payload -> codeword (pure combinational).
    //------------------------------------------------------------------
    wire [CODE_W-1:0] enc_code;
    // decode side of this instance is unused; feed a constant to keep it quiet.
    wire [DATA_W-1:0] enc_data_unused;
    wire              enc_serr_unused, enc_derr_unused;

    ecc_secded #(.DATA_W(DATA_W)) u_enc (
        .data_in    (wdata),
        .code_out   (enc_code),
        .code_in    ({CODE_W{1'b0}}),
        .data_out   (enc_data_unused),
        .single_err (enc_serr_unused),
        .double_err (enc_derr_unused)
    );

    //------------------------------------------------------------------
    // Backing store of CODEWORDS + the read-fetch pipeline register.
    //------------------------------------------------------------------
    reg [CODE_W-1:0] mem [0:DEPTH-1];
    reg [CODE_W-1:0] rd_code;          // codeword fetched on the read posedge
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < DEPTH; i = i + 1)
                mem[i] <= {CODE_W{1'b0}};
            rd_code <= {CODE_W{1'b0}};
        end else begin
            // Normal encoded write.
            if (we)
                mem[waddr] <= enc_code;
            // Back-door raw-codeword write (fault injection / scrubber).
            // Independent address: may coexist with a normal write elsewhere.
            if (bd_we)
                mem[bd_addr] <= bd_code;
            // Synchronous read fetch (1-cycle latency).
            if (re)
                rd_code <= mem[raddr];
        end
    end

    //------------------------------------------------------------------
    // READ-path codec: decode the fetched codeword (pure combinational).
    //------------------------------------------------------------------
    wire [DATA_W-1:0] dec_data;
    wire              dec_serr, dec_derr;
    wire [CODE_W-1:0] dec_code_unused;

    ecc_secded #(.DATA_W(DATA_W)) u_dec (
        .data_in    ({DATA_W{1'b0}}),
        .code_out   (dec_code_unused),
        .code_in    (rd_code),
        .data_out   (dec_data),
        .single_err (dec_serr),
        .double_err (dec_derr)
    );

    //------------------------------------------------------------------
    // Register the decode results -> every output is a register.
    //------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            rdata <= {DATA_W{1'b0}};
            serr  <= 1'b0;
            derr  <= 1'b0;
        end else begin
            rdata <= dec_data;
            serr  <= dec_serr;
            derr  <= dec_derr;
        end
    end

endmodule
