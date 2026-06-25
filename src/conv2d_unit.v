`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// conv2d_unit  --  TPU v2.0 true 2-D convolution  IMG_H x IMG_W * K x K
//                  (SPEC.md §5.2)
//----------------------------------------------------------------------------
// ALGORITHM
//   Valid/same-padding 2-D convolution (cross-correlation; no kernel flip, to
//   match the spec's "MAC the window*kernel products" and the impulse->
//   identity directed test) of an IMG_H x IMG_W Q7.8 input feature map with a
//   K x K Q7.8 kernel.  The unit walks the output grid in raster order one
//   pixel per cycle; for each output pixel it accumulates the K*K window*kernel
//   products in a single 48-bit Q15.16 accumulator, then round-half-up +
//   saturates the accumulator back to a Q7.8 element.  The narrowing SEMANTICS
//   are the canonical tpu_defs.vh contract (round-half-up of acc/2^FRAC, then
//   clamp to the signed Q7.8 range, with a saturation-hit flag).  The
//   round-shift uses a local helper (`round_shift_q78`) that is now bit-exact
//   to the shared `TPU_ROUND_SHIFT macro; the macro's old UNSIGNED-bias bug
//   (which turned the `>>> into a LOGICAL shift and corrupted NEGATIVE
//   accumulators, which conv produces constantly) is FIXED in the header.  The
//   clamp reuses the shared, always-correct `TPU_SAT_Q78` macro.  See the
//   round_shift_q78 comment.
//
// PARAMETERIZATION  (NEW in v2.0; default == tpu_defs.vh, byte-identical)
//   parameter IMG_H = `CONV_H, IMG_W = `CONV_W, K = `CONV_K  (defaults 8,8,3).
//   Output dims are localparams OH = IMG_H-K+1, OW = IMG_W-K+1 (the VALID-pad,
//   stride-1 dims; default 6,6); the run-time stride/pad fields further shrink
//   the effective output dims ODH/ODW computed at launch.  All line-buffer
//   depths, loop bounds, counter widths ($clog2) and packing indices are
//   derived from these parameters -- no size-literals remain.
//
//   SUPPORTED RANGE (architectural envelope):
//     * IMG_W <= LINE_W/ELEM_W = 4*32/16 = 8 : one image row must pack into ONE
//       128-bit TM line (ELEM=16 bits/col), so at most 8 columns per line.
//     * K     <= LINE_W/ELEM_W = 8           : one kernel row must likewise pack
//       into one TM line.
//     * K     <= IMG_H  and  K <= IMG_W      : valid output dims must be >= 1.
//     * The OUTPUT line packing is FIXED at LINE_LANES = 4 px per TM line
//       (LINE_W/LANE_W); this matches the architecture's fixed 4-lane TM line
//       and is parameterized against `LINE_LANES, not a literal 4.
//   The default (8,8,3) sits at the IMG_W upper bound (8 cols == full line) and
//   is exercised exhaustively by conv2d_unit_tb; a 2nd in-range size
//   (6,6,3 => 4x4) is proven by a second instance in that TB.
//
// MICROARCHITECTURE (line buffers + K x K window)
//   The spec mandates "line buffers + a window register".  Because the whole
//   map is tiny and TM has only two combinational read ports (we need K image
//   rows AND K kernel rows per window), we realize the line-buffer store as an
//   explicit on-chip ROW BUFFER: at launch we stream the IMG_H input rows and K
//   kernel rows out of TM (one TM line per cycle, using one read port) into
//   registers (`imap` IMG_H*IMG_W Q7.8, `kreg` K*K Q7.8).  After load, the
//   COMPUTE phase reads no TM at all: each cycle it selects the current K x K
//   window out of `imap` (zero outside the map, i.e. zero edge padding when
//   pad=1), MACs it against `kreg`, narrows, and packs the Q7.8 result into the
//   output line buffer.  This is functionally the line-buffer + sliding-window
//   datapath (one output pixel/cycle) with the small map fully buffered on
//   chip; it keeps the explicit 48-bit accumulator and the visible saturation
//   flag.
//
//   stride in {1,2} and pad in {0,1} are REAL input fields (no dead path):
//     out_dim = floor((dim + 2*pad - K)/stride) + 1     (per axis)
//       pad=0,stride=1 -> OH x OW   (default 6x6, the committed default)
//       pad=1,stride=1 -> IMG_H x IMG_W
//       pad=0,stride=2 / pad=1,stride=2 -> proportionally smaller
//   Source samples outside [0,IMG_H-1]x[0,IMG_W-1] (reachable only when pad=1)
//   contribute 0 (zero edge padding).  stride 0/3 are clamped to 1, pad>=1
//   treated as 1 -- so there is no illegal/undefined dead path.
//
// Q-FORMATS
//   input feature map : Q7.8  (16-bit signed element, sign-extended in lane)
//   kernel            : Q7.8
//   product Q7.8*Q7.8 : Q14.16 (30-bit signed)
//   accumulator       : Q15.16, ACC_W = 48-bit signed (K*K taps never overflow
//                       for the supported envelope)
//   output pixel      : Q7.8, via round-half-up + saturate (round_shift_q78 +
//                       `TPU_SAT_Q78`)
//   sat               : sticky OR of the per-pixel saturation-hit across all px
//
// MEMORY PACKING  (the TB models TM; this unit only drives TM access ports)
//   INPUT  (in_base):  IMG_H consecutive TM lines; line in_base+r = image row r.
//                      Within a line the IMG_W columns pack 16 bits each,
//                      low-first:  bits [16*c +: 16] = pixel(row=r, col=c),
//                      Q7.8.  (IMG_W cols * 16b <= 128b = one TM line.)
//   KERNEL (k_base):   K consecutive TM lines; line k_base+kr = kernel row kr.
//                      bits [16*kc +: 16] = kernel(kr,kc), Q7.8, in the low
//                      K*16 bits; the upper bits of each kernel line are ignored.
//   OUTPUT (out_base): output pixels in raster order, LINE_LANES (=4) px per TM
//                      line, packed bits [16*(p%LANES) +: 16] = output pixel p
//                      (p = oy*ODW+ox), Q7.8, low-lane first.  A partial final
//                      line has its unused upper lanes written 0.
//                      Number of output lines = ceil(ODH*ODW / LINE_LANES).
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
//     LOAD phase : IMG_H + K cycles  (IMG_H image rows + K kernel rows, one TM
//                  line/cycle through the single read port)
//     COMPUTE    : ODH*ODW cycles (one output pixel/cycle); output TM lines are
//                  flushed inline as each group of LINE_LANES pixels (or the
//                  final partial group) completes.
//     DONE       : 1 cycle (done pulse).
//   Counting posedges from the edge that samples start to the edge that
//   raises done:  (IMG_H+K) load edges + ODH*ODW compute edges + 1 done edge.
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
module conv2d_unit #(
    // Conv geometry parameters; defaults are the tpu_defs.vh committed sizes so
    // an unparameterized instance is byte-identical to the v1.x fixed unit.
    parameter integer IMG_H = `CONV_H,     // 8  input height
    parameter integer IMG_W = `CONV_W,     // 8  input width
    parameter integer K     = `CONV_K      // 3  kernel size (K x K)
) (
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
    // Derived from the IMG_H/IMG_W/K parameters -- no size-literals.
    //------------------------------------------------------------------------
    localparam integer H    = IMG_H;       // input height  (parameterized)
    localparam integer W    = IMG_W;       // input width   (parameterized)
    localparam integer ELEM = `ELEM_W;     // 16 bits per Q7.8 element
    localparam integer LANES = `LINE_LANES; // 4  output px packed per TM line

    // VALID/stride-1 output dims (pad=0,stride=1).  These are the SMALLEST
    // output dims.  Used below to size the coordinate/pixel counters from the
    // geometry rather than from any size-literal.
    localparam integer OH   = IMG_H - K + 1; // default 6
    localparam integer OW   = IMG_W - K + 1; // default 6

    // Maximum run-time output dims/pixels.  The run-time output dim is
    //   od = (IMG + 2*pad - K)/stride + 1 = OH + 2*pad   (stride=1),
    // maximized at pad=PAD_MAX:  od_max = OH + 2*PAD_MAX.
    // NOTE: the previous OH+(K-1) == IMG basis was exact ONLY for K=3 (where
    // pad=1 grows the output back to exactly IMG).  For K<3 (e.g. K=2) the pad=1
    // output is OH+2 > IMG, which would OVERFLOW the PW-bit pixel counter and
    // truncate the run.  Adding 2*PAD_MAX (independent of K) makes every K in the
    // supported range safe.  Supported zero-pad is pad in {0,1}; pad=1 is the
    // documented max-grow case.  (For the default K=3 these stay = IMG.)
    localparam integer PAD_MAX  = 1;
    localparam integer ODH_MAX  = OH + 2*PAD_MAX;
    localparam integer ODW_MAX  = OW + 2*PAD_MAX;
    localparam integer NPIX_MAX = ODH_MAX * ODW_MAX;

    // Counter widths derived from the geometry via the built-in $clog2 (the
    // same idiom gemm_systolic uses); no size-literals.
    //   CW  : holds 0..ODH_MAX/ODW_MAX coords AND the 0..H+K-1 load index.
    //   PW  : holds 0..NPIX_MAX output pixels (linear pixel index / count).
    //   LW  : output-lane index 0..LANES-1 (FIXED 4-lane TM packing).
    localparam integer CW = ($clog2(H + K) > $clog2(ODW_MAX + 1))
                            ? $clog2(H + K) : $clog2(ODW_MAX + 1);
    localparam integer PW = $clog2(NPIX_MAX + 1);
    localparam integer LW = $clog2(LANES);

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
    reg [CW-1:0]        odh,   odh_n;       // effective output height (run-time)
    reg [CW-1:0]        odw,   odw_n;       // effective output width  (run-time)
    reg [PW-1:0]        npix,  npix_n;      // total output pixels = odh*odw

    // Load index: 0..H+K-1 (rows of image then rows of kernel).
    reg [CW-1:0]        lidx,  lidx_n;

    // Output-pixel walk.
    reg [CW-1:0]        oy,    oy_n;
    reg [CW-1:0]        ox,    ox_n;
    reg [PW-1:0]        pidx,  pidx_n;      // linear output pixel index
    reg [LW-1:0]        lane,  lane_n;      // which of LANES lanes in current line
    reg [`LINE_W-1:0]   obuf,  obuf_n;      // output line buffer (LANES px)
    reg [`TM_IDX_W-1:0] oline, oline_n;     // current output TM line index

    reg                 sat_n;
    reg                 busy_n;

    // On-chip buffers (the "line buffers"/window store + kernel store).
    reg signed [ELEM-1:0] imap [0:H*W-1];   // IMG_H x IMG_W input map, Q7.8
    reg signed [ELEM-1:0] imap_n [0:H*W-1];
    reg signed [ELEM-1:0] kreg [0:K*K-1];   // K x K kernel, Q7.8
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
    integer pp, ss, effh, effw, ddh, ddw;   // output-dim computation scratch
    integer kr;                             // kernel row index during load
    integer lc;                             // load-column index (image/kernel)
    reg [PW-1:0] npix32;                    // ddh*ddw narrowed to the pixel count
    reg signed [ELEM-1:0] samp;             // selected window sample
    reg                   last_in_line;     // this px fills lane LANES-1 OR final
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
        odh_n    = odh;
        odw_n    = odw;
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
        effh         = 0;
        effw         = 0;
        ddh          = 0;
        ddw          = 0;
        kr           = 0;
        lc           = 0;
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

                    // Output dims floor((dim+2*pad-K)/stride)+1, per axis, sanitized.
                    pp   = (pad != 2'd0)  ? 1 : 0;
                    ss   = (stride == 2'd2) ? 2 : 1;
                    effh = H + 2*pp - K;            // height span
                    effw = W + 2*pp - K;            // width  span
                    ddh  = (effh / ss) + 1;         // effective output height
                    ddw  = (effw / ss) + 1;         // effective output width
                    npix32 = PW'(ddh * ddw);
                    odh_n  = ddh[CW-1:0];
                    odw_n  = ddw[CW-1:0];
                    npix_n = npix32;

                    lidx_n  = {CW{1'b0}};
                    oy_n    = {CW{1'b0}};
                    ox_n    = {CW{1'b0}};
                    pidx_n  = {PW{1'b0}};
                    lane_n  = {LW{1'b0}};
                    obuf_n  = {`LINE_W{1'b0}};
                    oline_n = out_base;
                    state_n = S_LOAD;
                end
            end

            //----------------------------------------------------------------
            // S_LOAD: stream H image rows then K kernel rows from TM, one line
            // per cycle, through the single combinational read port.
            //   lidx 0 .. H-1   : image row lidx          (W columns)
            //   lidx H .. H+K-1 : kernel row (lidx-H)      (K columns)
            //----------------------------------------------------------------
            S_LOAD: begin
                if (lidx < H[CW-1:0]) begin
                    rd_addr = in_b + {{(`TM_IDX_W-CW){1'b0}}, lidx};
                    // latch W columns of this image row (one TM line)
                    for (lc = 0; lc < W; lc = lc + 1)
                        imap_n[ lidx*W + lc ] =
                            $signed(rd_data[ lc*ELEM +: ELEM ]);
                end else begin
                    // kernel row (lidx - H), K taps from the low K*ELEM bits
                    kr      = 32'(lidx) - H;
                    rd_addr = k_b + {{(`TM_IDX_W-CW){1'b0}}, kr[CW-1:0]};
                    for (lc = 0; lc < K; lc = lc + 1)
                        kreg_n[ kr*K + lc ] =
                            $signed(rd_data[ lc*ELEM +: ELEM ]);
                end

                if (lidx == CW'(H + K - 1)) begin // final load index
                    lidx_n  = {CW{1'b0}};
                    state_n = S_COMP;
                end else begin
                    lidx_n  = lidx + {{(CW-1){1'b0}}, 1'b1};
                end
            end

            //----------------------------------------------------------------
            // S_COMP: one output pixel per cycle.  Build the K x K window from
            // imap (zero outside the map), MAC against kreg in a 48-bit
            // accumulator, round+saturate to Q7.8, pack into the output line
            // buffer, and flush the line to TM when LANES lanes are filled or
            // this is the final pixel.
            //----------------------------------------------------------------
            S_COMP: begin
                // ---- K*K-tap MAC over the current window ----
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
                // flush the line when the lane group is full or on the last pixel
                last_in_line = (lane == LW'(LANES-1)) || last_px;

                if (last_in_line) begin
                    tm_we    = 1'b1;
                    tm_waddr = oline;
                    tm_wdata = obuf_n;          // includes the just-packed lane
                    oline_n  = oline + {{(`TM_IDX_W-1){1'b0}}, 1'b1};
                    obuf_n   = {`LINE_W{1'b0}}; // start next line cleared
                    lane_n   = {LW{1'b0}};
                end else begin
                    lane_n   = lane + {{(LW-1){1'b0}}, 1'b1};
                end

                // ---- advance the raster scan ----
                if (last_px) begin
                    state_n = S_DONE;
                    busy_n  = 1'b0;     // busy low on the cycle done pulses
                end else if (ox == (odw - {{(CW-1){1'b0}}, 1'b1})) begin
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
            odh   <= {CW{1'b0}};
            odw   <= {CW{1'b0}};
            npix  <= {PW{1'b0}};
            lidx  <= {CW{1'b0}};
            oy    <= {CW{1'b0}};
            ox    <= {CW{1'b0}};
            pidx  <= {PW{1'b0}};
            lane  <= {LW{1'b0}};
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
            odh   <= odh_n;
            odw   <= odw_n;
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
