`include "tpu_defs.vh"
//============================================================================
// conv2d_unit  --  TPU v2.0 true 2-D convolution 8x8 * 3x3   (SPEC.md §5.2)
//----------------------------------------------------------------------------
// ALGORITHM
//   Valid/same-padding 2-D convolution (cross-correlation; no kernel flip, to
//   match the spec's "MAC the 9 window*kernel products" and the impulse->
//   identity directed test) of an 8x8 Q7.8 input feature map with a 3x3 Q7.8
//   kernel.  The unit walks the output grid in raster order one pixel per
//   cycle; for each output pixel it accumulates the nine window*kernel products
//   in a single 48-bit Q15.16 accumulator, then round-half-up + saturates the
//   accumulator back to a Q7.8 element.  The narrowing SEMANTICS are the
//   canonical tpu_defs.vh contract (round-half-up of acc/2^FRAC, then clamp to
//   the signed Q7.8 range, with a saturation-hit flag).  The round-shift uses a
//   local helper (`round_shift_q78`) that is now bit-exact to the shared
//   `TPU_ROUND_SHIFT macro; the macro's old UNSIGNED-bias bug (which turned the
//   `>>> into a LOGICAL shift and corrupted NEGATIVE accumulators, which conv
//   produces constantly) is FIXED in the header.  The clamp reuses the shared,
//   always-correct `TPU_SAT_Q78` macro.  See the round_shift_q78 comment.
//
// MICROARCHITECTURE (line buffers + 3x3 window)
//   The spec mandates "line buffers (2 rows x 8) + a 3x3 window register".
//   Because the whole map is tiny (8x8) and TM has only two combinational read
//   ports (we need three image rows AND three kernel rows per window), we
//   realize the line-buffer store as an explicit on-chip ROW BUFFER: at launch
//   we stream the 8 input rows and 3 kernel rows out of TM (one TM line per
//   cycle, using one read port) into registers (`imap` 8x8 Q7.8, `kreg` 3x3
//   Q7.8).  After load, the COMPUTE phase reads no TM at all: each cycle it
//   selects the current 3x3 window out of `imap` (zero outside the map, i.e.
//   zero edge padding when pad=1), MACs it against `kreg`, narrows, and packs
//   the Q7.8 result into the output line buffer.  This is functionally the
//   line-buffer + sliding-window datapath (one output pixel/cycle) with the
//   small map fully buffered on chip; it keeps the explicit 48-bit accumulator
//   and the visible saturation flag.
//
//   stride in {1,2} and pad in {0,1} are REAL input fields (no dead path):
//     out_dim = floor((CONV_H + 2*pad - CONV_K)/stride) + 1
//       pad=0,stride=1 -> 6x6  (36 px, the committed default)
//       pad=1,stride=1 -> 8x8  (64 px)
//       pad=0,stride=2 -> 3x3  ( 9 px)
//       pad=1,stride=2 -> 4x4  (16 px)
//   Source samples outside [0,7]x[0,7] (reachable only when pad=1) contribute 0
//   (zero edge padding).  stride 0/3 are clamped to 1, pad>=1 treated as 1 --
//   so there is no illegal/undefined dead path.
//
// Q-FORMATS
//   input feature map : Q7.8  (16-bit signed element, sign-extended in lane)
//   kernel            : Q7.8
//   product Q7.8*Q7.8 : Q14.16 (30-bit signed)
//   accumulator       : Q15.16, ACC_W = 48-bit signed (9 taps never overflow)
//   output pixel      : Q7.8, via round-half-up + saturate (round_shift_q78 +
//                       `TPU_SAT_Q78`)
//   sat               : sticky OR of the per-pixel saturation-hit across all px
//
// MEMORY PACKING  (the TB models TM; this unit only drives TM access ports)
//   INPUT  (in_base):  8 consecutive TM lines; line in_base+r = image row r.
//                      Within a line the 8 columns pack 16 bits each, low-first:
//                      bits [16*c +: 16] = pixel(row=r, col=c), Q7.8.
//                      (8 cols * 16b = 128b = exactly one TM line.)
//   KERNEL (k_base):   3 consecutive TM lines; line k_base+kr = kernel row kr.
//                      bits [16*kc +: 16] = kernel(kr,kc), Q7.8, in the low 48
//                      bits; the upper 80 bits of each kernel line are ignored.
//   OUTPUT (out_base): output pixels in raster order, 4 px per TM line, packed
//                      bits [16*(p%4) +: 16] = output pixel p (p = oy*OW+ox),
//                      Q7.8, low-lane first.  6x6 => 36 px => 9 lines.  A
//                      partial final line has its unused upper lanes written 0.
//                      Number of output lines = ceil(OH*OW / 4).
//
// INTERFACE / HANDSHAKE
//   start                 1-cycle launch pulse (ignored while busy)
//   in_base,k_base,out_base [TM_IDX_W-1:0]  base TM line indices
//   stride[1:0]           1 or 2 (0/3 clamp to 1)
//   pad[1:0]              0 or 1 (>=1 -> 1)
//   busy                  high from the cycle after accept until (but NOT
//                         including) the cycle done pulses; busy is low when
//                         done is high
//   done                  1-cycle pulse when all output lines are committed
//   sat                   sticky run saturation flag (valid at done; held to
//                         next start)
//   TM ports: one combinational read port used (rd_addr/rd_data) for the load
//             stream; one synchronous write port (tm_we/tm_waddr/tm_wdata) for
//             results.  This unit instantiates NO memory; the TB/top wires the
//             ports to the shared tile_memory.
//
// LATENCY  (deterministic; the TB asserts it)
//   start sampled in S_IDLE -> busy rises next cycle.  Then:
//     LOAD phase : H + K = 8 + 3 = 11 cycles  (8 image rows + 3 kernel rows,
//                  one TM line/cycle through the single read port)
//     COMPUTE    : OH*OW cycles (one output pixel/cycle); output TM lines are
//                  flushed inline as each group of 4 pixels (or the final
//                  partial group) completes.
//     DONE       : 1 cycle (done pulse).
//   Counting posedges from the edge that samples start to the edge that
//   raises done:  (H+K) load edges + OH*OW compute edges + 1 done edge.
//     default (6x6): 11 + 36 + 1 = 48 cycles.  All bounded constants; the TB
//     both checks this closed form and measures done dynamically.
//
// SYNTHESIZABILITY
//   One clocked always block holds ALL state with a synchronous reset on every
//   reg (assigned on every path -> no inferred latch); one combinational block
//   is the pure next-state/datapath function (every output assigned on every
//   path; no comb loop).  No real/$display/$random/initial in the module.
//   Lints clean under verilator -Wall.
//============================================================================
module conv2d_unit (
    input  wire                   clk,
    input  wire                   rst,

    // Control handshake.
    input  wire                   start,
    input  wire [`TM_IDX_W-1:0]   in_base,
    input  wire [`TM_IDX_W-1:0]   k_base,
    input  wire [`TM_IDX_W-1:0]   out_base,
    input  wire [1:0]             stride,
    input  wire [1:0]             pad,
    output reg                    busy,
    output reg                    done,
    output reg                    sat,

    // Tile-memory access ports.
    output reg  [`TM_IDX_W-1:0]   rd_addr,   // combinational read (load stream)
    input  wire [`LINE_W-1:0]     rd_data,
    output reg                    tm_we,     // synchronous write (results)
    output reg  [`TM_IDX_W-1:0]   tm_waddr,
    output reg  [`LINE_W-1:0]     tm_wdata
);

    //------------------------------------------------------------------------
    // Local geometry constants (NOT in tpu_defs.vh; declared here per RULES).
    //------------------------------------------------------------------------
    localparam integer H    = `CONV_H;     // 8  input height
    localparam integer W    = `CONV_W;     // 8  input width
    localparam integer K    = `CONV_K;     // 3  kernel size
    localparam integer ELEM = `ELEM_W;     // 16 bits per Q7.8 element

    // Counter widths (small, bounded by H=8 and 64 pixels).
    localparam integer CW = 4;  // 0..8   (coords, row/col, load index)
    localparam integer PW = 7;  // 0..64  (linear pixel index / pixel count)

    //------------------------------------------------------------------------
    // FSM states.
    //------------------------------------------------------------------------
    localparam [1:0] S_IDLE = 2'd0;
    localparam [1:0] S_LOAD = 2'd1;
    localparam [1:0] S_COMP = 2'd2;
    localparam [1:0] S_DONE = 2'd3;

    //------------------------------------------------------------------------
    // State registers (+ their combinational "_n" next values).
    //------------------------------------------------------------------------
    reg [1:0]            state, state_n;

    reg [`TM_IDX_W-1:0] in_b,  in_b_n;
    reg [`TM_IDX_W-1:0] k_b,   k_b_n;
    reg [`TM_IDX_W-1:0] out_b, out_b_n;
    reg [1:0]           strd,  strd_n;
    reg [1:0]           pd,    pd_n;
    reg [CW-1:0]        od,    od_n;        // output dim (OH == OW, square map)
    reg [PW-1:0]        npix,  npix_n;      // total output pixels = od*od

    // Load index: 0..H+K-1 (rows of image then rows of kernel).
    reg [CW-1:0]        lidx,  lidx_n;

    // Output-pixel walk.
    reg [CW-1:0]        oy,    oy_n;
    reg [CW-1:0]        ox,    ox_n;
    reg [PW-1:0]        pidx,  pidx_n;      // linear output pixel index
    reg [1:0]           lane,  lane_n;      // which of 4 lanes in current line
    reg [`LINE_W-1:0]   obuf,  obuf_n;      // output line buffer (4 px)
    reg [`TM_IDX_W-1:0] oline, oline_n;     // current output TM line index

    reg                 sat_n;
    reg                 busy_n;

    // On-chip buffers (the "line buffers"/window store + kernel store).
    reg signed [ELEM-1:0] imap [0:H*W-1];   // 8x8 input map, Q7.8
    reg signed [ELEM-1:0] imap_n [0:H*W-1];
    reg signed [ELEM-1:0] kreg [0:K*K-1];   // 3x3 kernel, Q7.8
    reg signed [ELEM-1:0] kreg_n [0:K*K-1];

    integer bi;

    //------------------------------------------------------------------------
    // Canonical accumulator -> Q7.8 narrowing (round-half-up + saturate).
    //----------------------------------------------------------------------------
    // CONTRACT: identical SEMANTICS to tpu_defs.vh's TPU_RND_SAT_Q78 /
    // TPU_SAT_HIT (round-half-up of acc/2^FRAC, then clamp to the signed Q7.8
    // range, with a saturation-hit flag).  This local round-shift is now
    // bit-exact to the canonical `TPU_ROUND_SHIFT.  HISTORICAL NOTE: the shared
    // macro used to add an UNSIGNED rounding-bias concatenation, so (signed acc
    // + unsigned bias) evaluated unsigned and the `>>>` became a LOGICAL shift,
    // wrong (large positive) for NEGATIVE accumulators.  Conv accumulators are
    // routinely negative (signed inputs/kernels), so this unit kept a signed
    // local bias.  The header bias is now a signed ACC_W constant (fixed), so
    // the macro and this helper agree; the helper is retained only to keep the
    // per-pixel datapath wording unchanged.  The clamp reuses the shared (and
    // always-correct) TPU_SAT_Q78 macro verbatim.
    function signed [`ACC_W-1:0] round_shift_q78;
        input signed [`ACC_W-1:0] a;
        begin
            // round-half-up (ties toward +inf) then arithmetic >> FRAC, all signed
            round_shift_q78 = (a + `ACC_W'sd128) >>> `Q78_FRAC;
        end
    endfunction

    //------------------------------------------------------------------------
    // Combinational datapath scratch.
    //------------------------------------------------------------------------
    reg signed [`ACC_W-1:0] acc;            // 48-bit Q15.16 MAC accumulator
    reg signed [`ACC_W-1:0] rnd;            // round-half-up+shifted (pre-clamp)
    reg signed [ELEM-1:0]   px_out;         // saturated Q7.8 output pixel
    reg                     px_sat;         // saturation hit for this pixel
    reg signed [`PROD_W-1:0] prod;          // single Q14.16 product

    integer wr, wc;                         // window row/col (0..K-1)
    integer iy, ix;                         // source coord (signed via int)
    integer base_iy, base_ix;               // window-base source coord (signed)
    integer pp, ss, eff, dd;                // output-dim computation scratch
    integer kr;                             // kernel row index during load
    reg [PW-1:0] npix32;                    // dd*dd narrowed to the pixel count
    reg signed [ELEM-1:0] samp;             // selected window sample
    reg                   last_in_line;     // this px fills lane 3 OR is final px
    reg                   last_px;          // this px is the final output px

    //========================================================================
    // Combinational: next-state, TM addressing, window MAC, output packing.
    //========================================================================
    always @(*) begin
        // ---- hold-by-default (every state output assigned on every path) ---
        state_n  = state;
        in_b_n   = in_b;
        k_b_n    = k_b;
        out_b_n  = out_b;
        strd_n   = strd;
        pd_n     = pd;
        od_n     = od;
        npix_n   = npix;
        lidx_n   = lidx;
        oy_n     = oy;
        ox_n     = ox;
        pidx_n   = pidx;
        lane_n   = lane;
        obuf_n   = obuf;
        oline_n  = oline;
        sat_n    = sat;
        busy_n   = busy;
        for (bi = 0; bi < H*W; bi = bi + 1) imap_n[bi] = imap[bi];
        for (bi = 0; bi < K*K; bi = bi + 1) kreg_n[bi] = kreg[bi];

        // TM ports idle by default.
        rd_addr  = in_b;
        tm_we    = 1'b0;
        tm_waddr = oline;
        tm_wdata = obuf;
        done     = 1'b0;

        // datapath scratch defaults (assigned on every path -> no latch).
        acc          = {`ACC_W{1'b0}};
        rnd          = {`ACC_W{1'b0}};
        prod         = {`PROD_W{1'b0}};
        px_out       = {ELEM{1'b0}};
        px_sat       = 1'b0;
        samp         = {ELEM{1'b0}};
        iy           = 0;
        ix           = 0;
        base_iy      = 0;
        base_ix      = 0;
        pp           = 0;
        ss           = 0;
        eff          = 0;
        dd           = 0;
        kr           = 0;
        npix32       = {PW{1'b0}};
        last_in_line = 1'b0;
        last_px      = 1'b0;

        case (state)
            //----------------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    in_b_n  = in_base;
                    k_b_n   = k_base;
                    out_b_n = out_base;
                    strd_n  = (stride == 2'd2) ? 2'd2 : 2'd1; // {0,1,3} -> 1
                    pd_n    = (pad   != 2'd0)  ? 2'd1 : 2'd0; // >=1     -> 1
                    sat_n   = 1'b0;
                    busy_n  = 1'b1;

                    // Output dimension floor((H+2*pad-K)/stride)+1, sanitized.
                    pp  = (pad != 2'd0)  ? 1 : 0;
                    ss  = (stride == 2'd2) ? 2 : 1;
                    eff = H + 2*pp - K;              // 5 or 7
                    dd  = (eff / ss) + 1;            // {6,3} or {8,4}
                    npix32 = PW'(dd * dd);          // {36,9} or {64,16}
                    od_n   = dd[CW-1:0];
                    npix_n = npix32;

                    lidx_n  = {CW{1'b0}};
                    oy_n    = {CW{1'b0}};
                    ox_n    = {CW{1'b0}};
                    pidx_n  = {PW{1'b0}};
                    lane_n  = 2'd0;
                    obuf_n  = {`LINE_W{1'b0}};
                    oline_n = out_base;
                    state_n = S_LOAD;
                end
            end

            //----------------------------------------------------------------
            // S_LOAD: stream H image rows then K kernel rows from TM, one line
            // per cycle, through the single combinational read port.
            //   lidx 0 .. H-1   : image row lidx          (8 columns)
            //   lidx H .. H+K-1 : kernel row (lidx-H)      (3 columns)
            //----------------------------------------------------------------
            S_LOAD: begin
                if (lidx < H[CW-1:0]) begin
                    rd_addr = in_b + {{(`TM_IDX_W-CW){1'b0}}, lidx};
                    // latch 8 columns of this image row
                    imap_n[ lidx*W + 0 ] = $signed(rd_data[ 0*ELEM +: ELEM]);
                    imap_n[ lidx*W + 1 ] = $signed(rd_data[ 1*ELEM +: ELEM]);
                    imap_n[ lidx*W + 2 ] = $signed(rd_data[ 2*ELEM +: ELEM]);
                    imap_n[ lidx*W + 3 ] = $signed(rd_data[ 3*ELEM +: ELEM]);
                    imap_n[ lidx*W + 4 ] = $signed(rd_data[ 4*ELEM +: ELEM]);
                    imap_n[ lidx*W + 5 ] = $signed(rd_data[ 5*ELEM +: ELEM]);
                    imap_n[ lidx*W + 6 ] = $signed(rd_data[ 6*ELEM +: ELEM]);
                    imap_n[ lidx*W + 7 ] = $signed(rd_data[ 7*ELEM +: ELEM]);
                end else begin
                    // kernel row (lidx - H), 3 taps from low 48 bits
                    kr      = 32'(lidx) - H;
                    rd_addr = k_b + {{(`TM_IDX_W-CW){1'b0}}, kr[CW-1:0]};
                    kreg_n[ kr*K + 0 ] = $signed(rd_data[0*ELEM +: ELEM]);
                    kreg_n[ kr*K + 1 ] = $signed(rd_data[1*ELEM +: ELEM]);
                    kreg_n[ kr*K + 2 ] = $signed(rd_data[2*ELEM +: ELEM]);
                end

                if (lidx == CW'(H + K - 1)) begin // 10 == final load index
                    lidx_n  = {CW{1'b0}};
                    state_n = S_COMP;
                end else begin
                    lidx_n  = lidx + {{(CW-1){1'b0}}, 1'b1};
                end
            end

            //----------------------------------------------------------------
            // S_COMP: one output pixel per cycle.  Build the 3x3 window from
            // imap (zero outside the map), MAC against kreg in a 48-bit
            // accumulator, round+saturate to Q7.8, pack into the output line
            // buffer, and flush the line to TM when 4 lanes are filled or this
            // is the final pixel.
            //----------------------------------------------------------------
            S_COMP: begin
                // ---- 9-tap MAC over the current window ----
                // Window-base source coords (signed; can be negative for pad=1).
                // Everything is promoted to 32-bit signed `integer` here, so the
                // subtraction of `pd` is a signed (not wrapping unsigned) op.
                base_iy = (32'(oy) * 32'(strd)) - 32'(pd);
                base_ix = (32'(ox) * 32'(strd)) - 32'(pd);
                acc = {`ACC_W{1'b0}};
                for (wr = 0; wr < K; wr = wr + 1) begin
                    for (wc = 0; wc < K; wc = wc + 1) begin
                        // signed source coordinate with stride and padding
                        iy = base_iy + wr;
                        ix = base_ix + wc;
                        if ((iy >= 0) && (iy < H) && (ix >= 0) && (ix < W))
                            samp = imap[ iy*W + ix ];
                        else
                            samp = {ELEM{1'b0}};        // zero edge padding
                        prod = $signed(samp) * $signed(kreg[wr*K + wc]);
                        // sign-extend the 32-bit Q14.16 product into the 48-bit acc
                        acc = acc + {{(`ACC_W-`PROD_W){prod[`PROD_W-1]}}, prod};
                    end
                end

                // ---- narrow Q15.16 -> Q7.8 (round-half-up + saturate) ----
                // round-half-up + arithmetic shift (signed; see round_shift_q78)
                rnd    = round_shift_q78(acc);
                // clamp to Q7.8 range via the shared (correct) saturate macro
                px_out = `TPU_SAT_Q78(rnd);
                // saturation hit iff the clamp actually fired
                px_sat = ($signed(rnd) > $signed(`ACC_W'sd32767)) ||
                         ($signed(rnd) < $signed(-`ACC_W'sd32768));
                sat_n  = sat | px_sat;

                // ---- pack into the output line buffer at the current lane ----
                obuf_n = obuf;
                obuf_n[ lane*ELEM +: ELEM ] = px_out;

                // is this the last pixel overall?
                last_px = (pidx == (npix - {{(PW-1){1'b0}}, 1'b1}));
                // flush the line when lane==3 (line full) or on the last pixel
                last_in_line = (lane == 2'd3) || last_px;

                if (last_in_line) begin
                    tm_we    = 1'b1;
                    tm_waddr = oline;
                    tm_wdata = obuf_n;          // includes the just-packed lane
                    oline_n  = oline + {{(`TM_IDX_W-1){1'b0}}, 1'b1};
                    obuf_n   = {`LINE_W{1'b0}}; // start next line cleared
                    lane_n   = 2'd0;
                end else begin
                    lane_n   = lane + 2'd1;
                end

                // ---- advance the raster scan ----
                if (last_px) begin
                    state_n = S_DONE;
                    busy_n  = 1'b0;     // busy low on the cycle done pulses
                end else if (ox == (od - {{(CW-1){1'b0}}, 1'b1})) begin
                    ox_n = {CW{1'b0}};
                    oy_n = oy + {{(CW-1){1'b0}}, 1'b1};
                    pidx_n = pidx + {{(PW-1){1'b0}}, 1'b1};
                end else begin
                    ox_n   = ox + {{(CW-1){1'b0}}, 1'b1};
                    pidx_n = pidx + {{(PW-1){1'b0}}, 1'b1};
                end
            end

            //----------------------------------------------------------------
            S_DONE: begin
                done    = 1'b1;
                busy_n  = 1'b0;
                state_n = S_IDLE;
            end

            default: state_n = S_IDLE;
        endcase
    end

    //========================================================================
    // Sequential state update (synchronous reset on ALL state).
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            in_b  <= {`TM_IDX_W{1'b0}};
            k_b   <= {`TM_IDX_W{1'b0}};
            out_b <= {`TM_IDX_W{1'b0}};
            strd  <= 2'd1;
            pd    <= 2'd0;
            od    <= {CW{1'b0}};
            npix  <= {PW{1'b0}};
            lidx  <= {CW{1'b0}};
            oy    <= {CW{1'b0}};
            ox    <= {CW{1'b0}};
            pidx  <= {PW{1'b0}};
            lane  <= 2'd0;
            obuf  <= {`LINE_W{1'b0}};
            oline <= {`TM_IDX_W{1'b0}};
            sat   <= 1'b0;
            busy  <= 1'b0;
            for (bi = 0; bi < H*W; bi = bi + 1) imap[bi] <= {ELEM{1'b0}};
            for (bi = 0; bi < K*K; bi = bi + 1) kreg[bi] <= {ELEM{1'b0}};
        end else begin
            state <= state_n;
            in_b  <= in_b_n;
            k_b   <= k_b_n;
            out_b <= out_b_n;
            strd  <= strd_n;
            pd    <= pd_n;
            od    <= od_n;
            npix  <= npix_n;
            lidx  <= lidx_n;
            oy    <= oy_n;
            ox    <= ox_n;
            pidx  <= pidx_n;
            lane  <= lane_n;
            obuf  <= obuf_n;
            oline <= oline_n;
            sat   <= sat_n;
            busy  <= busy_n;
            for (bi = 0; bi < H*W; bi = bi + 1) imap[bi] <= imap_n[bi];
            for (bi = 0; bi < K*K; bi = bi + 1) kreg[bi] <= kreg_n[bi];
        end
    end

endmodule
