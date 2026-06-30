`timescale 1ns/1ps
//============================================================================
// weight_decomp2.v  --  CONTEXT-MODELED STREAMING FP8 WEIGHT DECOMPRESSOR
//                                 (IMPROVEMENT_PLAN.md ULTRA_PERF #3, beats P2.1)
//----------------------------------------------------------------------------
// PURPOSE
//   The committed `weight_decomp` is ORDER-0 canonical Huffman: ONE static
//   table for the whole stream -> 1.34x (5.97 bits/sym) on representative FP8
//   E4M3 weights.  The residual entropy is in the CONTEXT: trained FP8 weights,
//   scanned in memory order inside a [128,128] block, have strong MAGNITUDE
//   LOCALITY (neighbouring weights share an exponent range) plus sign/exponent
//   correlation.  An ORDER-1 model -- decode each byte with one of a few
//   Huffman tables selected by a cheap function of the PREVIOUS decoded byte --
//   captures that and reaches ~1.4-1.5x (a measured ~1.13x over order-0 on the
//   SAME correlated weight blocks; see test/weight_decomp2_tb.v), for a few
//   small extra on-chip tables.  Bigger Flash->DDR5 compression = tok/s + J.
//
//   PURE INTEGER / CONTROL RTL: it operates on the FP8 *bytes* as opaque 8-bit
//   symbols and NEVER interprets their numeric value -- no floating point.
//   Losslessness is bit-exact: the FP8 byte that comes out is the FP8 byte the
//   offline compressor (tools/fp8_ctxpack.py) started from.
//
//----------------------------------------------------------------------------
// CONTEXT SCHEME  --  NCTX (=4) classes, selected by ctxfn(prev_byte)
//   The context for the NEXT symbol is a cheap function of the LAST decoded FP8
//   byte: drop the sign bit (sign is ~uncorrelated), bucket the 7-bit magnitude
//   m = byte[6:0] with two threshold compares + a zero test:
//       m == 0           -> class 0   (exact zero)
//       m <  THRESH1     -> class 1   (tiny  magnitude, exp 0..2)
//       m <  THRESH2     -> class 2   (small/mid magnitude, exp 3..7)
//       else             -> class 3   (large magnitude)
//   => TWO unsigned comparators + one equality; no multiply/divide, no LUT.
//   The first symbol uses INIT_CTX (=0).  EACH class carries its own canonical
//   Huffman table; the active class selects which table the next codeword
//   decodes against.  Within one (multi-bit) codeword the class is FIXED (it
//   only updates when a data byte is emitted), so the per-bit recurrence below
//   is identical to weight_decomp -- just table-base-offset by the class.
//
//----------------------------------------------------------------------------
// CANONICAL DECODE  (reused per-bit DEFLATE recurrence; shift/add/compare only)
//   Per codeword: ccode (bits so far), clen (#bits, starts 1), cfirst (first
//   numeric code of length clen), cindex (count of all shorter codes).  Bit b:
//       code_L = (ccode<<1)|b ;  cnt = count_table[ctx][clen]
//       if (code_L - cfirst) < cnt :  symbol = symbol_table[ctx][cindex+(code_L-cfirst)]
//       else : cfirst<=(cfirst+cnt)<<1 ; cindex<=cindex+cnt ; clen<=clen+1
//   On emitting a DATA byte, ctx <= ctxfn(byte) for the next codeword.  EOB_SYM
//   (=256) ends the block (no data byte emitted); the encoder appends it under
//   the final context, so every consulted table that can terminate carries EOB.
//
//----------------------------------------------------------------------------
// TABLES  (loadable, small on-chip; per-context, power-of-two strides)
//   count_table  : NCTX x (1<<CTADW) entries x COUNTW   = 4x16x10 =   640 bits
//   symbol_table : NCTX x (1<<SYMW)  entries x SYMW     = 4x512x9 = 18432 bits
//   => ~2.4 KB total, trivially on-chip.  Loaded via tbl_we/tbl_sel/tbl_ctx/
//   tbl_addr/tbl_wdata (per-block or global); a context is addressed by tbl_ctx.
//   Index math is pure shift/OR:  count: (ctx<<CTADW)|clen ;  sym: (ctx<<SYMW)|i.
//
//----------------------------------------------------------------------------
// STREAMING HANDSHAKE  (identical contract to weight_decomp)
//   COMPRESSED IN : in_byte/in_valid/in_ready (Flash side, MSB-first packing).
//   FP8 OUT       : out_byte/out_valid/out_ready (DDR5 side), back-pressured.
//   END OF BLOCK  : eob latches high on EOB_SYM; pulse `start` to begin the
//                   next block (clears bit + codeword + context state, keeps the
//                   loaded tables).  Sync active-high reset, no latch, no comb loop.
//============================================================================
module weight_decomp2 #(
    parameter integer NCTX    = 4,    // # context classes (this ctxfn yields 4)
    parameter integer MAXLEN  = 15,   // max canonical Huffman code length
    parameter integer SYMW    = 9,    // symbol width (holds 0..256)
    parameter integer COUNTW  = 10,   // per-length count width (holds 0..257)
    parameter integer AW      = 9,    // table address width (per-context load)
    parameter integer BUFW    = 32,   // bit-buffer width (>= 16)
    parameter integer EOB_SYM = 256,  // end-of-block symbol value
    parameter integer THRESH1 = 'h18, // magnitude bucket edge 1 (class 0/1 vs ...)
    parameter integer THRESH2 = 'h40, // magnitude bucket edge 2
    parameter integer INIT_CTX= 0,    // context of the first symbol
    parameter integer CTXW    = (NCTX <= 1) ? 1 : $clog2(NCTX)  // ctx index width (derived)
)(
    input  wire              clk,
    input  wire              rst,        // synchronous, active-high

    // ---- canonical Huffman table load (per-context, per-block or global) ----
    input  wire              tbl_we,     // write enable
    input  wire              tbl_sel,    // 1 = symbol_table, 0 = count_table
    input  wire [CTXW-1:0]   tbl_ctx,    // which context class to load
    input  wire [AW-1:0]     tbl_addr,   // count: 0..MAXLEN ; symbol: 0..ncodes-1
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
    localparam integer CTADW = $clog2(MAXLEN + 1);  // per-context count addr width
    localparam integer CTDEP = NCTX << CTADW;       // count_table depth
    localparam integer SYDEP = NCTX << SYMW;        // symbol_table depth

    localparam [CB-1:0] ROOM = CB'(BUFW - 8);       // load when bitcnt <= ROOM

    // ---- per-context tables (loaded via tbl_* ; NOT reset -- they are memories) ----
    reg [COUNTW-1:0] count_table  [0:CTDEP-1];       // count_table[(ctx<<CTADW)|len]
    reg [SYMW-1:0]   symbol_table [0:SYDEP-1];       // symbol_table[(ctx<<SYMW)|i]

    // ---- bit buffer (MSB-first: next bit = bitbuf[BUFW-1]) ----
    reg [BUFW-1:0]   bitbuf;
    reg [CB-1:0]     bitcnt;

    // ---- per-codeword decode state ----
    reg [CW-1:0]     ccode;     // bits accumulated for current codeword
    reg [CW-1:0]     cfirst;    // first numeric code of length clen
    reg [SYMW-1:0]   cindex;    // number of codewords shorter than clen
    reg [LW-1:0]     clen;      // current codeword length (>=1)
    reg [CTXW-1:0]   ctx;       // active context class (fixed within a codeword)

    reg              done;      // block finished (EOB seen), gates input

    // ---- combinational next-state ----
    reg [BUFW-1:0]   nbitbuf;
    reg [CB-1:0]     nbitcnt;
    reg [CW-1:0]     nccode, ncfirst;
    reg [SYMW-1:0]   ncindex;
    reg [LW-1:0]     nclen;
    reg [CTXW-1:0]   nctx;
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
    reg [CTADW-1:0]  caddr;
    reg [SYMW+CTXW-1:0] saddr;

    // ---- context function: cheap magnitude bucket of a decoded FP8 byte ----
    function [CTXW-1:0] ctxfn(input [6:0] m);
        begin                                            // m = magnitude (sign dropped)
            if (m == 7'd0)               ctxfn = 0;
            else if (m < THRESH1[6:0])   ctxfn = 1;
            else if (m < THRESH2[6:0])   ctxfn = 2;
            else                         ctxfn = 3;
        end
    endfunction

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
        nctx      = ctx;
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
        caddr   = {CTADW{1'b0}};
        saddr   = {(SYMW+CTXW){1'b0}};

        buf_c   = bitbuf;
        cnt_c   = bitcnt;

        if (do_bit) begin
            cur_bit = bitbuf[BUFW-1];
            buf_c   = bitbuf << 1;
            cnt_c   = bitcnt - 1'b1;

            code_L  = ({1'b0, ccode} << 1) | {{CW{1'b0}}, cur_bit};
            caddr   = clen[CTADW-1:0];
            // per-context count: base (ctx<<CTADW) | clen  (pure shift/OR)
            cnt     = count_table[ {ctx, caddr} ];
            diff    = code_L - {1'b0, cfirst};

            if (diff < {{(CW+1-COUNTW){1'b0}}, cnt}) begin
                // matched a codeword of length clen in context `ctx`
                sidx  = cindex + diff[SYMW-1:0];
                saddr = {ctx, sidx};                 // (ctx<<SYMW) | sidx
                symv  = symbol_table[saddr];
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
                    nctx       = ctxfn(symv[6:0]);   // context for the NEXT symbol (drop sign)
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
            ctx       <= INIT_CTX[CTXW-1:0];
            out_byte  <= 8'h00;
            out_valid <= 1'b0;
            eob       <= 1'b0;
            done      <= 1'b0;
        end else begin
            // table load (independent of the decode datapath)
            if (tbl_we) begin
                if (tbl_sel) symbol_table[ {tbl_ctx, tbl_addr[SYMW-1:0]} ]   <= tbl_wdata[SYMW-1:0];
                else         count_table [ {tbl_ctx, tbl_addr[CTADW-1:0]} ]  <= tbl_wdata;
            end

            if (start) begin
                // begin a new block: clear bit + codeword + context state and EOB
                bitbuf    <= {BUFW{1'b0}};
                bitcnt    <= {CB{1'b0}};
                ccode     <= {CW{1'b0}};
                cfirst    <= {CW{1'b0}};
                cindex    <= {SYMW{1'b0}};
                clen      <= {{(LW-1){1'b0}}, 1'b1};
                ctx       <= INIT_CTX[CTXW-1:0];
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
                ctx       <= nctx;
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
