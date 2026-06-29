`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// weight_loader.v  --  WEIGHT-SIDE DMA / sequencer for glm_matmul_fp8 (ACCEL_GLM52)
//----------------------------------------------------------------------------
// FUNCTION
//   glm_matmul_fp8 is a [128,128]-block-scaled FP8 E4M3 GEMM that PULLS its
//   weights: at `start` it latches one bf16 BLOCK scale per (output column pj,
//   K-block bj), then it walks K consuming one FP8 weight ROW w_row[k][*] each
//   beat that `in_valid` is asserted.  This loader is the master that DRIVES
//   that pull from a simple read-memory (the TB models real DDR5/Flash):
//
//     1. Given a tile DESCRIPTOR {base, k_len, nblk}, read the tile's bf16 block
//        scales into the packed w_scale bus, then present them with the `start`
//        pulse (clearing glm_matmul_fp8's banks + latching k_len/scales).
//     2. Stream the FP8 weight rows w_row[k] one per beat, asserting in_valid,
//        so the GEMM consumes exactly k_len beats k=0..k_len-1.
//
//   The activation side (a_col / a_shift) is driven separately; this loader owns
//   ONLY the weight side.  It is the beat master: it drives in_valid, so the
//   activation provider keeps pace with the weight stream.
//
// WEIGHT-MEMORY (DESCRIPTOR) LAYOUT  -- word-addressed:
//   A tile occupies a contiguous region from `base`:
//     * SCALE region : base + (bj*PE_N + pj),  for bj=0..nblk-1, pj=0..PE_N-1
//                      one bf16 block scale per word (low 16 bits), stored in the
//                      EXACT (bj*PE_N+pj) order glm_matmul_fp8 packs w_scale.
//                      ( nblk*PE_N words )
//     * CODE  region : base + nblk*PE_N + k,   for k=0..k_len-1
//                      one weight ROW per word: word[8*pj +: 8] = W[k][pj] (E4M3),
//                      the EXACT packing glm_matmul_fp8 expects on w_row.
//                      ( k_len words )
//   Unused scale banks (nblk < NB) are ZERO-filled in w_scale so the GEMM's
//   dequant fold over all NB banks multiplies a cleared (=0) accumulator by 0
//   (never X).
//
// READ-MEMORY INTERFACE (TB models DDR5/Flash):
//   mem_en + mem_addr presented (combinationally) on the bus cycle t  ->
//   mem_data valid on cycle t+1  (a standard registered single-port RAM read;
//   latency = 1).  The loader presents the address combinationally so the RAM's
//   own output register supplies exactly the one-cycle latency.
//
// START / STREAM TIMING this loader drives (matches glm_matmul_fp8):
//   cyc S    : mm_start=1, mm_w_scale=<assembled>, mm_k_len=k_len; present code k=0.
//              (start latches scales/k_len, clears banks, raises GEMM streaming
//               on the NEXT cycle.)
//   cyc S+1  : mem_data=w_row[0]; GEMM streaming now high -> in_valid=1,
//              w_row=w_row[0]; present code k=1.
//   ...      : in_valid + w_row[k] each beat; present code k+1 one cycle ahead.
//   cyc S+L  : w_row[k_len-1] (last beat); GEMM hits last_issue, drops streaming.
//
// CONVENTIONS: synchronous ACTIVE-HIGH reset; NO latch (the combinational
//   request block fully defaults its outputs); NO combinational loop.
//============================================================================
module weight_loader #(
    parameter integer PE_N   = 4,        // output columns (== glm_matmul_fp8 PE_N)
    parameter integer KMAX   = 256,      // max K (== glm_matmul_fp8 KMAX)
    parameter integer BLK    = 128,      // K-block size [128,128] (== glm_matmul_fp8 BLK)
    parameter integer ADDR_W = 24,       // weight-memory address width
    // memory data width: wide enough for a packed weight row AND a bf16 scale.
    parameter integer DATA_W = (8*PE_N >= 16) ? 8*PE_N : 16,
    // ---- derived geometry (mirror glm_matmul_fp8) ----
    localparam integer NB  = (KMAX + BLK - 1) / BLK,   // #K-blocks (== w_scale banks)
    localparam integer KW  = $clog2(KMAX + 1),         // k_len width
    localparam integer BKW = $clog2(NB + 1),           // K-block-count width
    localparam integer NSW = $clog2(NB*PE_N + 1)       // scale-word counter width
) (
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    // ---- tile descriptor command ----
    input  wire                       load,       // 1-cycle pulse: start a tile load
    input  wire [ADDR_W-1:0]          desc_base,  // tile base address in weight mem
    input  wire [KW-1:0]              desc_klen,  // K length (#weight rows)
    input  wire [BKW-1:0]             desc_nblk,  // #K-blocks for this tile (<= NB)

    // ---- read-memory interface (TB models DDR5/Flash; mem_data valid t+1) ----
    output reg                        mem_en,     // combinational request strobe
    output reg  [ADDR_W-1:0]          mem_addr,   // combinational read address
    input  wire [DATA_W-1:0]          mem_data,

    // ---- glm_matmul_fp8 WEIGHT-side drive ----
    output wire                       mm_start,
    output wire [KW-1:0]              mm_k_len,
    output wire [8*PE_N-1:0]          mm_w_row,
    output wire [16*PE_N*NB-1:0]      mm_w_scale,
    output wire                       mm_in_valid,

    // ---- status ----
    output reg                        busy,
    output reg                        done        // 1-cycle pulse when tile streamed
);
    // -----------------------------------------------------------------------
    // FSM states
    // -----------------------------------------------------------------------
    localparam [2:0] S_IDLE   = 3'd0,
                     S_SCALE  = 3'd1,   // read bf16 block scales into w_scale_q
                     S_START  = 3'd2,   // pulse start (+ latch scales) ; present code k=0
                     S_STREAM = 3'd3,   // stream weight rows w_row[k] + in_valid
                     S_DONE   = 3'd4;   // signal completion

    reg [2:0]                state;

    // latched descriptor
    reg [ADDR_W-1:0]         base_q;
    reg [KW-1:0]             klen_q;
    reg [BKW-1:0]            nblk_q;

    // assembled weight scale bus (zero-filled for unused banks)
    reg [16*PE_N*NB-1:0]     w_scale_q;

    // scale-phase counters / latency-1 capture pipeline
    reg [NSW-1:0]            sc_iss;     // #scale reads issued
    reg [NSW-1:0]            sc_cap;     // #scale words captured
    reg                      rd_v;       // a scale read was requested LAST cycle
    reg [NSW-1:0]            rd_slot;    // its (bj*PE_N+pj) slot

    // code-phase counters / in-flight read
    reg [KW-1:0]             cd_iss;     // #code reads requested (next k to fetch)
    reg [KW-1:0]             beat_cnt;   // #weight-row beats driven to the GEMM
    reg                      code_pending; // mem_data this cycle is a valid w_row

    // total scale words for this tile, and the code-region base address.
    wire [NSW-1:0]           ns        = NSW'(nblk_q) * NSW'(PE_N);
    wire [ADDR_W-1:0]        ns_ext    = {{(ADDR_W-NSW){1'b0}}, ns};
    wire [ADDR_W-1:0]        code_base = base_q + ns_ext;

    // -----------------------------------------------------------------------
    // Combinational weight-side outputs.
    //   start is a clean 1-cycle pulse (S_START lasts one cycle); the GEMM
    //   latches w_scale/k_len there and raises streaming next cycle.
    //   in_valid asserts for exactly the k_len streamed beats; w_row tracks the
    //   registered mem_data on those beats, 0 otherwise (no X on the bus).
    // -----------------------------------------------------------------------
    assign mm_start    = (state == S_START);
    assign mm_k_len    = klen_q;
    assign mm_w_scale  = w_scale_q;
    assign mm_in_valid = (state == S_STREAM) & code_pending & (beat_cnt < klen_q);
    assign mm_w_row    = mm_in_valid ? mem_data[8*PE_N-1:0] : {(8*PE_N){1'b0}};

    // -----------------------------------------------------------------------
    // Combinational read-request: present the next address on the bus so the
    // registered RAM returns its data one cycle later.  Fully defaulted -> no
    // latch.  Address advances off the registered counters (no comb loop).
    // -----------------------------------------------------------------------
    always @* begin
        mem_en   = 1'b0;
        mem_addr = {ADDR_W{1'b0}};
        case (state)
            S_SCALE:  if (sc_iss < ns) begin
                          mem_en   = 1'b1;
                          mem_addr = base_q + {{(ADDR_W-NSW){1'b0}}, sc_iss};
                      end
            S_START:  begin
                          mem_en   = 1'b1;             // prefetch code word k=0
                          mem_addr = code_base;
                      end
            S_STREAM: if (cd_iss < klen_q) begin
                          mem_en   = 1'b1;             // prefetch next code word
                          mem_addr = code_base + {{(ADDR_W-KW){1'b0}}, cd_iss};
                      end
            default:  begin
                          mem_en   = 1'b0;
                          mem_addr = {ADDR_W{1'b0}};
                      end
        endcase
    end

    // -----------------------------------------------------------------------
    // Control FSM (single synchronous, active-high reset, no latch/comb loop).
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            base_q       <= {ADDR_W{1'b0}};
            klen_q       <= {KW{1'b0}};
            nblk_q       <= {BKW{1'b0}};
            w_scale_q    <= {(16*PE_N*NB){1'b0}};
            sc_iss       <= {NSW{1'b0}};
            sc_cap       <= {NSW{1'b0}};
            rd_v         <= 1'b0;
            rd_slot      <= {NSW{1'b0}};
            cd_iss       <= {KW{1'b0}};
            beat_cnt     <= {KW{1'b0}};
            code_pending <= 1'b0;
        end else begin
            // ---- per-cycle defaults (1-cycle pulses / capture marker) ----
            done <= 1'b0;
            rd_v <= 1'b0;

            // ---- latency-1 scale CAPTURE: a read requested last cycle returns now ----
            if (rd_v) begin
                w_scale_q[16*rd_slot +: 16] <= mem_data[15:0];
                sc_cap                      <= sc_cap + 1'b1;
            end

            case (state)
                // ---- idle: wait for a descriptor ----
                S_IDLE: begin
                    if (load) begin
                        base_q       <= desc_base;
                        klen_q       <= desc_klen;
                        nblk_q       <= desc_nblk;
                        w_scale_q    <= {(16*PE_N*NB){1'b0}}; // zero-fill unused banks
                        sc_iss       <= {NSW{1'b0}};
                        sc_cap       <= {NSW{1'b0}};
                        cd_iss       <= {KW{1'b0}};
                        beat_cnt     <= {KW{1'b0}};
                        code_pending <= 1'b0;
                        busy         <= 1'b1;
                        state        <= S_SCALE;
                    end
                end

                // ---- read all bf16 block scales into w_scale_q (latency-1) ----
                S_SCALE: begin
                    if (sc_iss < ns) begin           // a request is on the bus now
                        rd_v    <= 1'b1;             // its data returns next cycle
                        rd_slot <= sc_iss;
                        sc_iss  <= sc_iss + 1'b1;
                    end
                    // all scales captured (also covers ns==0: no-scale tile)
                    if (sc_cap == ns)
                        state <= S_START;
                end

                // ---- start pulse (latch scales/k_len, clear banks); prefetch k=0 ----
                S_START: begin
                    code_pending <= 1'b1;            // mem_data next cycle = w_row[0]
                    cd_iss       <= {{(KW-1){1'b0}}, 1'b1};
                    state        <= S_STREAM;
                end

                // ---- stream weight rows: drive in_valid+w_row, prefetch next k ----
                S_STREAM: begin
                    if (mm_in_valid)
                        beat_cnt <= beat_cnt + 1'b1;

                    if (cd_iss < klen_q) begin
                        code_pending <= 1'b1;        // another row in flight
                        cd_iss       <= cd_iss + 1'b1;
                    end else begin
                        code_pending <= 1'b0;        // no more rows in flight
                    end

                    // last beat just driven -> finish next cycle.
                    if (mm_in_valid && (beat_cnt == klen_q - 1'b1))
                        state <= S_DONE;
                end

                // ---- completion ----
                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
