`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// kv_cache_pager.v  --  GLM-5.2 MLA LATENT-KV ring cache with Flash overflow
//                                                   (ACCEL_GLM52 §4.1/§4.2, paging)
//----------------------------------------------------------------------------
// FUNCTION  (the KV-cache that feeds mla_attn(_fp8)'s kc_* read protocol)
//   MLA stores ONE small LATENT KV row per token per layer = [c_kv | k_rope]
//   (in mla_attn_fp8 the row is kc_ckv[KV_LORA*16] concatenated with
//    kc_krope[ROPE*16] -- ROW_BITS bits total; the real GLM-5.2 row is 576
//    elements: c_kv(512)+k_rope(64)).  At 1M context the full cache is ~94 GB,
//   far larger than fast memory.  This pager owns the bounded RESIDENT WINDOW
//   that lives in fast on-chip memory as a RING BUFFER; tokens older than the
//   window have spilled to Flash (COLD) and are demand-fetched on a gather.
//
//   The DSA indexer (src/dsa_indexer.v + topk_select) selects up to index_topk
//   logical row indices; attention gathers ONLY those rows.  This unit answers
//   each gather: a RESIDENT row comes straight from the ring (fast, 1 cycle);
//   a COLD row is pulled from Flash (FLASH_LAT latency) and returned.  Rows are
//   OPAQUE bit vectors here -- the pager only ADDRESSES / MOVES them, never does
//   any floating-point math on their contents.
//
//----------------------------------------------------------------------------
// APPEND  (write the newest token's latent at the ring head)
//   append_valid pulses append_row in.  The row is written to ring slot
//   (append_count mod RESIDENT) for logical position = append_count, then
//   append_count advances (the ring head wraps at RESIDENT).  The last RESIDENT
//   appended positions are RESIDENT; older ones are COLD (conceptually evicted
//   to Flash).  When append_count exceeds RESIDENT the oldest resident row's
//   slot is overwritten by the newest append -- that is the overflow EVICTION,
//   and `overflowed` goes high (the evicted positions now only live in Flash).
//
// GATHER  (the attention read path -- mla kc_* protocol)
//   gather_valid pulses a DSA-selected logical row index gather_idx in.
//     * RESIDENT (resident_lo <= gather_idx < append_count):
//         row_out = ring[gather_idx mod RESIDENT], row_valid pulses NEXT cycle.
//         busy stays low (fast path, fully pipelineable 1 row/cycle).
//     * COLD (anything else, i.e. an evicted/older position):
//         flash_req + flash_idx are issued and HELD; busy rises; when the TB/DMA
//         returns flash_done (after FLASH_LAT) with flash_row, that row is driven
//         on row_out with a row_valid pulse and busy drops.  A cold fetch is NOT
//         re-staged into the ring (that would corrupt the contiguous resident
//         window mapping slot = pos mod RESIDENT); it is simply returned.
//
//----------------------------------------------------------------------------
// LATENCY
//   APPEND  : 1 cycle (synchronous write; append_count updates same edge).
//   GATHER  resident : row_valid 1 cycle after the accepted gather (registered
//                      ring read); busy never rises -> back-to-back gathers ok.
//   GATHER  cold     : flash_req asserted the cycle after accept and HELD; busy
//                      high until flash_done is seen, then row_valid pulses and
//                      busy drops.  Only one gather is in flight while busy.
//
// RESIDENCY (combinational, from append_count):
//   resident_lo = (append_count > RESIDENT) ? append_count-RESIDENT : 0
//   resident    = (gather_idx < append_count) && (gather_idx >= resident_lo)
//
//----------------------------------------------------------------------------
// STYLE: synchronous, ACTIVE-HIGH reset; NO latch (every reg holds or updates
//   in a clocked block, row_valid defaults low each cycle); NO comb loop.
//   Deterministic.  Rows are opaque [ROW_BITS] vectors -- pure addressing/move.
//   NOTE: RESIDENT must be a POWER OF TWO (the ring uses bit-slice modulo
//   slot = pos[RPTRW-1:0]); POSW must be >= RPTRW.
//============================================================================
module kv_cache_pager #(
    // one [c_kv | k_rope] latent row width.  Default = mla_attn_fp8's row:
    // (KV_LORA=32 + ROPE=16) * 16 bits = 768.
    parameter integer ROW_BITS  = 768,
    parameter integer RESIDENT  = 32,    // ring capacity in rows (POWER OF TWO)
    parameter integer S_MAX     = 1024,  // logical context capacity (positions)
    parameter integer POSW      = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer FLASH_LAT = 8,     // cold-row fetch latency (doc; TB models it)
    // ---- derived (do NOT override) ----
    parameter integer RPTRW     = (RESIDENT <= 1) ? 1 : $clog2(RESIDENT)
)(
    input  wire                 clk,
    input  wire                 rst,          // synchronous, ACTIVE-HIGH

    // ---- APPEND : write newest token latent at the ring head ----
    input  wire                 append_valid,
    input  wire [ROW_BITS-1:0]  append_row,

    // ---- GATHER : attention read path (satisfies mla kc_* protocol) ----
    input  wire                 gather_valid,
    input  wire [POSW-1:0]      gather_idx,    // DSA-selected logical row index
    output reg                  row_valid,     // 1-cycle pulse: row_out is valid
    output reg  [ROW_BITS-1:0]  row_out,
    output reg                  busy,          // high while a cold Flash fetch drains

    // ---- FLASH overflow fetch (TB/DMA models backing store + latency) ----
    output reg                  flash_req,     // held high until flash_done
    output reg  [POSW-1:0]      flash_idx,     // cold logical row index to fetch
    input  wire                 flash_done,    // 1-cycle: flash_row is ready
    input  wire [ROW_BITS-1:0]  flash_row,     // fetched cold row

    // ---- observability ----
    output wire [POSW-1:0]      append_count,  // total rows appended (= next logical pos)
    output wire [POSW-1:0]      resident_lo,   // lowest RESIDENT logical position
    output wire                 overflowed     // older positions have spilled to Flash
);

    // touch FLASH_LAT (a documentation parameter; the TB/DMA models the latency)
    // so lint never flags it as dead -- has no effect on hardware behaviour.
    localparam integer FLASH_LAT_DOC = FLASH_LAT;

    //------------------------------------------------------------------------
    // RING storage + append counter.
    //------------------------------------------------------------------------
    reg [ROW_BITS-1:0] ring [0:RESIDENT-1];
    reg [POSW-1:0]     count;        // number of rows appended so far
    integer            i;

    assign append_count = count;

    // resident window low bound (combinational from the counter).
    wire over = (count > RESIDENT[POSW-1:0]);
    assign overflowed  = over;
    assign resident_lo = over ? (count - RESIDENT[POSW-1:0]) : {POSW{1'b0}};

    // residency decode for the incoming gather index.
    wire g_resident = (gather_idx < count) && (gather_idx >= resident_lo);

    //------------------------------------------------------------------------
    // GATHER FSM : IDLE serves resident rows in 1 cycle and launches cold
    // fetches; FLASH holds flash_req until flash_done returns the cold row.
    //------------------------------------------------------------------------
    localparam G_IDLE = 1'b0, G_FLASH = 1'b1;
    reg g_state;

    always @(posedge clk) begin
        if (rst) begin
            count     <= {POSW{1'b0}};
            row_valid <= 1'b0;
            row_out   <= {ROW_BITS{1'b0}};
            busy      <= 1'b0;
            flash_req <= 1'b0;
            flash_idx <= {POSW{1'b0}};
            g_state   <= G_IDLE;
            for (i = 0; i < RESIDENT; i = i + 1)
                ring[i] <= {ROW_BITS{1'b0}};
        end else begin
            // row_valid is a 1-cycle pulse -> default low every cycle.
            row_valid <= 1'b0;

            //----------------------------------------------------------------
            // APPEND port (independent of the gather path).  Writes position
            // `count` at slot count mod RESIDENT, then advances the head.  A
            // concurrent resident gather of the oldest slot reads the OLD
            // (pre-eviction) value (nonblocking), the correct boundary value.
            //----------------------------------------------------------------
            if (append_valid) begin
                ring[count[RPTRW-1:0]] <= append_row;
                count                  <= count + 1'b1;
            end

            //----------------------------------------------------------------
            // GATHER FSM.
            //----------------------------------------------------------------
            case (g_state)
                G_IDLE: begin
                    if (gather_valid && !busy) begin
                        if (g_resident) begin
                            // fast path: registered ring read, 1-cycle latency.
                            row_out   <= ring[gather_idx[RPTRW-1:0]];
                            row_valid <= 1'b1;
                        end else begin
                            // cold path: issue + hold a Flash fetch, go busy.
                            flash_req <= 1'b1;
                            flash_idx <= gather_idx;
                            busy      <= 1'b1;
                            g_state   <= G_FLASH;
                        end
                    end
                end
                G_FLASH: begin
                    if (flash_done) begin
                        flash_req <= 1'b0;
                        row_out   <= flash_row;
                        row_valid <= 1'b1;
                        busy      <= 1'b0;
                        g_state   <= G_IDLE;
                    end
                end
                default: g_state <= G_IDLE;
            endcase
        end
    end

    // keep the documentation localparam observably alive (no hardware effect).
    /* verilator lint_off UNUSEDPARAM */
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused_doc = (FLASH_LAT_DOC == 0);
    /* verilator lint_on UNUSEDSIGNAL */
    /* verilator lint_on UNUSEDPARAM */

endmodule
