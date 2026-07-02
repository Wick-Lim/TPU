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
// SCRUB WRITE-BACK (P2.1 heal-on-read):
//   Correcting a single-bit flip on the READ output alone leaves the rotted
//   bit in the array, where it can accumulate a second flip and become an
//   uncorrectable double error.  This wrapper therefore SCRUBS: whenever a read
//   detects+corrects a single-bit error, the corrected codeword (re-encoded
//   from the corrected payload) is written back into the array one cycle later,
//   healing the bit in place.  The external read-latency contract is unchanged
//   (rdata/serr/derr still valid 2 rising edges after re); scrub is an extra
//   internal write that lands on the same cycle the corrected read is
//   registered.  A subsequent read of the same address therefore sees serr=0.
//
// STICKY STATUS (serr_sticky/derr_sticky + err_ack):
//   The per-read serr/derr pulses are momentary.  Two sticky flags LATCH that
//   at least one single- / double-error was ever observed since the last clear,
//   so slow software can poll them.  A synchronous err_ack clears both; a new
//   error observed in the same cycle as err_ack still latches (errors are never
//   lost).  rst clears them.
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
    // sticky error status + synchronous ack/clear
    serr_sticky, derr_sticky, err_ack,
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

    output reg                 serr_sticky;   // latched: a single-error was seen
    output reg                 derr_sticky;   // latched: a double-error was seen
    input  wire                err_ack;       // sync clear for both sticky flags

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
    reg [ADDR_W-1:0] rd_addr_q;        // address that produced rd_code (for scrub)
    reg              rd_valid_q;       // rd_code holds a genuine fetched word
    integer i;

    //------------------------------------------------------------------
    // READ-path codec: decode the fetched codeword (pure combinational).
    // Declared before the array block so the scrub write-back can consume
    // dec_serr/dec_derr in the same clocked block.
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
    // SCRUB-path codec: re-ENCODE the corrected payload -> clean codeword.
    // For a single-bit error dec_data is the corrected payload, so scrub_code
    // is the original clean codeword; writing it back heals the array bit.
    //------------------------------------------------------------------
    wire [CODE_W-1:0] scrub_code;
    wire [DATA_W-1:0] scr_data_unused;
    wire              scr_serr_unused, scr_derr_unused;

    ecc_secded #(.DATA_W(DATA_W)) u_scrub (
        .data_in    (dec_data),
        .code_out   (scrub_code),
        .code_in    ({CODE_W{1'b0}}),
        .data_out   (scr_data_unused),
        .single_err (scr_serr_unused),
        .double_err (scr_derr_unused)
    );

    // A scrub write-back is due this cycle iff the word currently in rd_code is
    // a genuine fetch that decoded to a CORRECTABLE single error.
    wire scrub_we = rd_valid_q & dec_serr & ~dec_derr;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < DEPTH; i = i + 1)
                mem[i] <= {CODE_W{1'b0}};
            rd_code    <= {CODE_W{1'b0}};
            rd_addr_q  <= {ADDR_W{1'b0}};
            rd_valid_q <= 1'b0;
        end else begin
            // Track the in-flight read so the scrub write can target its addr.
            rd_valid_q <= re;
            if (re)
                rd_addr_q <= raddr;
            // Scrub write-back (lowest priority: an explicit we/bd_we to the
            // same address this cycle overrides the heal).
            if (scrub_we)
                mem[rd_addr_q] <= scrub_code;
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
    // Register the decode results -> every output is a register.
    //------------------------------------------------------------------
    // A genuine read completes (decode result becomes valid) this cycle iff
    // rd_valid_q is set; only then do the sticky flags accumulate.
    wire read_serr = rd_valid_q & dec_serr;
    wire read_derr = rd_valid_q & dec_derr;

    always @(posedge clk) begin
        if (rst) begin
            rdata       <= {DATA_W{1'b0}};
            serr        <= 1'b0;
            derr        <= 1'b0;
            serr_sticky <= 1'b0;
            derr_sticky <= 1'b0;
        end else begin
            rdata <= dec_data;
            serr  <= dec_serr;
            derr  <= dec_derr;
            // Sticky latch: err_ack clears, but a new error the same cycle
            // still latches (set dominates clear -> errors are never lost).
            serr_sticky <= (serr_sticky & ~err_ack) | read_serr;
            derr_sticky <= (derr_sticky & ~err_ack) | read_derr;
        end
    end

endmodule
