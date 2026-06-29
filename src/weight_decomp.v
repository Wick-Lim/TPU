`timescale 1ns/1ps
//============================================================================
// weight_decomp.v  --  GLM-5.2 STREAMING FP8 WEIGHT DECOMPRESSOR
//                                          (IMPROVEMENT_PLAN.md P2.1, §P2)
//----------------------------------------------------------------------------
// PURPOSE
//   The expert weights of GLM-5.2-FP8 are E4M3 *bytes*.  Trained-weight FP8
//   values are near-zero-heavy (roughly normal/Laplacian), so the 256-symbol
//   byte stream has entropy WELL under 8 bits -- typically ~5-6.5 bits/symbol.
//   tok/s and J/token are Flash-byte-bound (see the governing model in
//   docs/IMPROVEMENT_PLAN.md), so storing experts LOSSLESSLY COMPRESSED in
//   Flash and decompressing them on-chip during the Flash->DDR5 fetch shrinks
//   Flash bytes/expert by that ratio -- a direct ~1.3-1.6x tok/s + energy win.
//
//   THIS module is the ON-CHIP half: a streaming lossless decompressor.
//   Compressed bytes in (Flash side, in_valid/in_ready) -> FP8 bytes out
//   (DDR5 side, out_valid/out_ready), handshake-driven, with end-of-block.
//
//   NOTE: this is PURE INTEGER / CONTROL RTL.  It operates on the FP8 *bytes*
//   as opaque 8-bit symbols -- it NEVER interprets their numeric value, so
//   there is NO floating point anywhere in here.  Losslessness is bit-exact:
//   the FP8 byte that comes out is the FP8 byte that went in to the offline
//   compressor, so the downstream FP8 datapath is bit-identical.
//
//----------------------------------------------------------------------------
// SCHEME  --  CANONICAL (static) HUFFMAN over the 257-symbol alphabet
//   The HW-cheap lossless coder that captures most of the entropy gain without
//   an arithmetic coder is a CANONICAL HUFFMAN code.  "Canonical" is the key
//   property: codes of a given length are a CONTIGUOUS numeric range assigned
//   to symbols in sorted order, so the decoder needs only two tiny tables --
//     count_table[len] : how many codewords have each length (1..MAXLEN), and
//     symbol_table[i]  : the symbols listed in canonical order --
//   and decodes with the classic per-bit DEFLATE recurrence (no big LUT, no
//   multiply, no divide, just shift/add/compare).  See the per-bit step below.
//
//   ALPHABET: 0..255 = the 256 FP8 E4M3 byte codes, plus one EOB symbol
//   (EOB_SYM, default 256) that the encoder appends so the stream is
//   self-delimiting -> the `eob` output marks end-of-block with NO side length
//   channel.  The matching OFFLINE ENCODER (length-limited package-merge
//   Huffman + canonical-code assignment) lives in tools/fp8_huff.py; the TB
//   compresses with it and checks bit-exact round-trip + prints the ratio.
//
//   TABLE SIZE (must fit on-chip -- it does, trivially):
//     count_table  : (MAXLEN+1) entries x COUNTW bits  = 16 x 10 = 160 bits
//     symbol_table : <=257 entries x SYMW bits         ~ 257 x 9 ~ 2313 bits
//   => well under 0.4 KB total.  The tables are LOADABLE (per-block or global)
//   via the tbl_we/tbl_sel/tbl_addr/tbl_wdata port, so one global table or a
//   per-block table can be installed without resynthesis.
//
//----------------------------------------------------------------------------
// PER-BIT CANONICAL DECODE  (one bit consumed per cycle, the DEFLATE loop)
//   State carried between bits of ONE codeword: ccode (bits so far), clen
//   (#bits so far, starts 1), cfirst (first numeric code of length clen),
//   cindex (count of all shorter codes).  Each accepted bit b:
//       code_L = (ccode<<1) | b ;  cnt = count_table[clen]
//       if (code_L - cfirst) < cnt :          // codeword of length clen
//            symbol = symbol_table[cindex + (code_L - cfirst)] ; emit; reset
//       else :                                 // need more bits
//            cfirst <= (cfirst+cnt)<<1 ; cindex <= cindex+cnt ; clen <= clen+1
//   This is exactly the canonical recurrence the encoder uses to lay out codes
//   (cfirst[1]=0, cfirst[l+1]=(cfirst[l]+count[l])<<1; cindex[l]=Sum count[<l]),
//   so decode is the bijective inverse of encode -- LOSSLESS by construction.
//
//----------------------------------------------------------------------------
// STREAMING HANDSHAKE
//   COMPRESSED IN (Flash side):  in_byte/in_valid/in_ready.  A compressed byte
//     is accepted when in_valid & in_ready; in_ready is high whenever the bit
//     buffer has room for another byte and the block is not finished.  Bits are
//     read MSB-first (bit 7 of the first byte is the first code bit), matching
//     the encoder's MSB-first packing.
//   FP8 OUT (DDR5 side):  out_byte/out_valid/out_ready.  out_valid pulses (and
//     holds until out_ready) for each decoded FP8 byte; decoding back-pressures
//     when the consumer is not ready (no new symbol is produced while one is
//     pending).  out_byte carries the raw FP8 E4M3 byte.
//   END OF BLOCK:  eob goes (and stays) high when the EOB symbol is decoded; no
//     data byte is emitted for it.  Assert `start` (1-cycle) to begin the next
//     block (clears bit state + eob; leaves the loaded tables intact).
//
//----------------------------------------------------------------------------
// STRUCTURE  --  combinational next-state + clocked register; sync active-high
//   reset, no latch, no comb loop.
//============================================================================
module weight_decomp #(
    parameter integer MAXLEN  = 15,   // max canonical Huffman code length
    parameter integer SYMW    = 9,    // symbol width (holds 0..256)
    parameter integer COUNTW  = 10,   // per-length count width (holds 0..257)
    parameter integer AW      = 9,    // table address width (load port)
    parameter integer BUFW    = 32,   // bit-buffer width (>= 16)
    parameter integer EOB_SYM = 256   // end-of-block symbol value
)(
    input  wire              clk,
    input  wire              rst,        // synchronous, active-high

    // ---- canonical Huffman table load (per-block or global) ----
    input  wire              tbl_we,     // write enable
    input  wire              tbl_sel,    // 1 = symbol_table, 0 = count_table
    input  wire [AW-1:0]     tbl_addr,   // count: 1..MAXLEN ; symbol: 0..ncodes-1
    input  wire [COUNTW-1:0] tbl_wdata,  // count value (or symbol in low SYMW bits)

    // ---- begin a new block (1-cycle pulse) ----
    input  wire              start,

    // ---- compressed input (Flash side) ----
    input  wire [7:0]        in_byte,
    input  wire              in_valid,
    output wire              in_ready,

    // ---- decompressed FP8 output (DDR5 side) ----
    output reg  [7:0]        out_byte,
    output reg               out_valid,
    input  wire              out_ready,
    output reg               eob         // end-of-block (level, held until start)
);
    localparam integer CW    = MAXLEN + 1;          // code/first accumulator width
    localparam integer CB    = $clog2(BUFW + 1);    // bit-count width
    localparam integer LW    = $clog2(MAXLEN + 2);  // code-length width
    localparam integer CTADW = $clog2(MAXLEN + 1);  // count_table address width
    localparam integer NSYMMAX = 1 << SYMW;         // symbol_table depth

    localparam [CB-1:0] ROOM = CB'(BUFW - 8);       // load when bitcnt <= ROOM

    // ---- tables (loaded via tbl_* ; NOT reset -- they are memories) ----
    reg [COUNTW-1:0] count_table  [0:MAXLEN];        // count_table[len]
    reg [SYMW-1:0]   symbol_table [0:NSYMMAX-1];     // canonical-order symbols

    // ---- bit buffer (MSB-first: next bit = bitbuf[BUFW-1]) ----
    reg [BUFW-1:0]   bitbuf;
    reg [CB-1:0]     bitcnt;

    // ---- per-codeword decode state ----
    reg [CW-1:0]     ccode;     // bits accumulated for current codeword
    reg [CW-1:0]     cfirst;    // first numeric code of length clen
    reg [SYMW-1:0]   cindex;    // number of codewords shorter than clen
    reg [LW-1:0]     clen;      // current codeword length (>=1)

    reg              done;      // block finished (EOB seen), gates input

    // ---- combinational next-state ----
    reg [BUFW-1:0]   nbitbuf;
    reg [CB-1:0]     nbitcnt;
    reg [CW-1:0]     nccode, ncfirst;
    reg [SYMW-1:0]   ncindex;
    reg [LW-1:0]     nclen;
    reg [7:0]        nout_byte;
    reg              nout_valid, neob, ndone;

    // ---- combinational temporaries ----
    reg              have_bit, can_proc, do_bit, ld;
    reg              cur_bit;
    reg [CW:0]       code_L, diff;
    reg [COUNTW-1:0] cnt;
    reg [SYMW-1:0]   sidx, symv;
    reg [BUFW-1:0]   buf_c;
    reg [CB-1:0]     cnt_c, shamt;

    // in_ready: room for a whole byte and block not finished
    assign in_ready = (bitcnt <= ROOM) && !done && !rst;

    // ----------------------------------------------------------------------
    // combinational: next-state of the decode datapath
    // ----------------------------------------------------------------------
    always @* begin
        // defaults: hold current state
        nccode    = ccode;
        ncfirst   = cfirst;
        ncindex   = cindex;
        nclen     = clen;
        nout_byte = out_byte;
        nout_valid= out_valid;
        neob      = eob;
        ndone     = done;

        // output back-pressure: clear when consumed
        if (out_valid && out_ready)
            nout_valid = 1'b0;

        have_bit = (bitcnt != {CB{1'b0}});
        can_proc = (!out_valid) || out_ready;   // free to produce a symbol
        do_bit   = have_bit && can_proc && !done;

        // default temporaries (avoid latches in comb block)
        cur_bit = 1'b0;
        code_L  = {(CW+1){1'b0}};
        diff    = {(CW+1){1'b0}};
        cnt     = {COUNTW{1'b0}};
        sidx    = {SYMW{1'b0}};
        symv    = {SYMW{1'b0}};

        buf_c   = bitbuf;
        cnt_c   = bitcnt;

        if (do_bit) begin
            cur_bit = bitbuf[BUFW-1];
            buf_c   = bitbuf << 1;
            cnt_c   = bitcnt - 1'b1;

            code_L  = ({1'b0, ccode} << 1) | {{CW{1'b0}}, cur_bit};
            cnt     = count_table[clen[CTADW-1:0]];
            diff    = code_L - {1'b0, cfirst};

            if (diff < {{(CW+1-COUNTW){1'b0}}, cnt}) begin
                // matched a codeword of length clen
                sidx = cindex + diff[SYMW-1:0];
                symv = symbol_table[sidx];
                // reset codeword state for the next symbol
                nccode  = {CW{1'b0}};
                ncfirst = {CW{1'b0}};
                ncindex = {SYMW{1'b0}};
                nclen   = {{(LW-1){1'b0}}, 1'b1};
                if (symv == EOB_SYM[SYMW-1:0]) begin
                    neob  = 1'b1;
                    ndone = 1'b1;
                end else begin
                    nout_byte  = symv[7:0];
                    nout_valid = 1'b1;
                end
            end else begin
                // need more bits: advance the canonical recurrence
                nccode  = code_L[CW-1:0];
                ncfirst = (cfirst + {{(CW-COUNTW){1'b0}}, cnt}) << 1;
                ncindex = cindex + cnt[SYMW-1:0];
                nclen   = clen + 1'b1;
            end
        end

        // input load: place new byte just below the valid bits
        ld    = in_valid && in_ready;
        shamt = ROOM - cnt_c;           // = BUFW-8-cnt_c, in range (cnt_c<=ROOM)
        if (ld) begin
            buf_c = buf_c | ( {{(BUFW-8){1'b0}}, in_byte} << shamt );
            cnt_c = cnt_c + 6'd8;
        end

        nbitbuf = buf_c;
        nbitcnt = cnt_c;
    end

    // ----------------------------------------------------------------------
    // clocked: registers + table load
    // ----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            bitbuf    <= {BUFW{1'b0}};
            bitcnt    <= {CB{1'b0}};
            ccode     <= {CW{1'b0}};
            cfirst    <= {CW{1'b0}};
            cindex    <= {SYMW{1'b0}};
            clen      <= {{(LW-1){1'b0}}, 1'b1};
            out_byte  <= 8'h00;
            out_valid <= 1'b0;
            eob       <= 1'b0;
            done      <= 1'b0;
        end else begin
            // table load (independent of the decode datapath)
            if (tbl_we) begin
                if (tbl_sel) symbol_table[tbl_addr]            <= tbl_wdata[SYMW-1:0];
                else         count_table[tbl_addr[CTADW-1:0]]  <= tbl_wdata;
            end

            if (start) begin
                // begin a new block: clear bit + codeword state and EOB
                bitbuf    <= {BUFW{1'b0}};
                bitcnt    <= {CB{1'b0}};
                ccode     <= {CW{1'b0}};
                cfirst    <= {CW{1'b0}};
                cindex    <= {SYMW{1'b0}};
                clen      <= {{(LW-1){1'b0}}, 1'b1};
                out_valid <= 1'b0;
                eob       <= 1'b0;
                done      <= 1'b0;
            end else begin
                bitbuf    <= nbitbuf;
                bitcnt    <= nbitcnt;
                ccode     <= nccode;
                cfirst    <= ncfirst;
                cindex    <= ncindex;
                clen      <= nclen;
                out_byte  <= nout_byte;
                out_valid <= nout_valid;
                eob       <= neob;
                done      <= ndone;
            end
        end
    end

`ifdef FORMAL
    always @(posedge clk) if (!rst) assert (bitcnt <= BUFW);
`endif

endmodule
