`timescale 1ns/1ps
//============================================================================
// weight_loader_tb.v  --  BINDING check for weight_loader.
//
//   The same glm_matmul_fp8 GEMM, given the SAME weight tile two ways, must
//   produce the BIT-EXACT same C tile:
//     * u_ref : weights fed DIRECTLY (the existing direct-feed pattern:
//               TB drives start/k_len/w_row/w_scale/in_valid, plus a_col/
//               a_shift) -- this is the trusted reference path.
//     * u_dut : the SAME weights laid into a memory image in the storage
//               layout weight_loader expects; weight_loader reads the image
//               and DRIVES u_dut's weight pull (mm_start/mm_k_len/mm_w_row/
//               mm_w_scale/mm_in_valid).  The TB drives the SAME activation
//               (a_col/a_shift), tracking the loader's beat stream.
//
//   The GEMM is deterministic, so for identical (w_row[k], w_scale, a_col[k],
//   a_shift, k_len) the two c_out busses must be EQUAL bit-for-bit.  The only
//   thing under test is whether the loader reconstructs the weight pull from
//   the memory image with the exact packing/order/timing the GEMM consumes.
//
//   Reference and DUT are run SEQUENTIALLY on the shared clk so there is no
//   cross-path timing coupling; each tile's c_ref is captured, then c_dut, and
//   compared element-by-element.  X-AWARE: any X bit is a hard failure.
//   Covers: k_len < KMAX, single + multiple K-blocks (w_scale indexing), and
//   several random tiles.  Emits "ALL <N> TESTS PASSED" + $fatal on mismatch.
//============================================================================
module weight_loader_tb;
    localparam integer PE_M   = 4;
    localparam integer PE_N   = 4;
    localparam integer KMAX   = 256;
    localparam integer BLK    = 128;
    localparam integer ADDR_W = 16;
    localparam integer DATA_W = 8*PE_N;          // 32: holds a packed row OR a bf16 scale
    localparam integer NB     = (KMAX+BLK-1)/BLK;
    localparam integer KW     = $clog2(KMAX+1);
    localparam integer BKW    = $clog2(NB+1);

    integer errors = 0;
    integer checks = 0;

    // ---------------- shared clock / reset ----------------
    reg clk = 1'b0;
    reg rst;
    always #5 clk = ~clk;

    // ---------------- tile storage (declared before use) ----------------
    reg [15:0] A     [0:PE_M-1][0:KMAX-1];   // bf16 activations
    reg [ 7:0] W     [0:KMAX-1][0:PE_N-1];   // E4M3 weights
    reg [ 7:0] ash_v [0:PE_M-1];             // per-row activation pow2 shift (signed)
    reg [15:0] wsc_v [0:PE_N-1][0:NB-1];     // bf16 weight BLOCK scale per (col, K-block)

    // ====================================================================
    // REFERENCE PATH : glm_matmul_fp8 fed the weights DIRECTLY by the TB.
    // ====================================================================
    reg                       start_r = 1'b0;
    reg  [KW-1:0]             klen_r  = 0;
    reg                       inv_r   = 1'b0;
    reg  [16*PE_M-1:0]        acol_r  = 0;
    reg  [ 8*PE_N-1:0]        wrow_r  = 0;
    reg  [ 8*PE_M-1:0]        ash_r   = 0;
    reg  [16*PE_N*NB-1:0]     wsc_r   = 0;
    wire                      ref_busy;
    wire                      ref_ov;
    wire [16*PE_M*PE_N-1:0]   ref_cout;

    glm_matmul_fp8 #(.PE_M(PE_M), .PE_N(PE_N), .KMAX(KMAX), .BLK(BLK)) u_ref (
        .clk(clk), .rst(rst), .start(start_r), .k_len(klen_r),
        .in_valid(inv_r), .a_col(acol_r), .w_row(wrow_r),
        .a_shift(ash_r), .w_scale(wsc_r),
        .busy(ref_busy), .out_valid(ref_ov), .c_out(ref_cout)
    );

    // ====================================================================
    // DUT PATH : weight_loader reads a memory image and DRIVES glm_matmul_fp8.
    // ====================================================================
    reg                       load = 1'b0;
    reg  [ADDR_W-1:0]         desc_base;
    reg  [KW-1:0]             desc_klen;
    reg  [BKW-1:0]            desc_nblk;

    wire                      mem_en;
    wire [ADDR_W-1:0]         mem_addr;
    reg  [DATA_W-1:0]         mem_data;

    wire                      mm_start;
    wire [KW-1:0]             mm_k_len;
    wire [8*PE_N-1:0]         mm_w_row;
    wire [16*PE_N*NB-1:0]     mm_w_scale;
    wire                      mm_in_valid;
    wire                      ld_busy;
    wire                      ld_done;

    weight_loader #(
        .PE_N(PE_N), .KMAX(KMAX), .BLK(BLK), .ADDR_W(ADDR_W), .DATA_W(DATA_W)
    ) u_ld (
        .clk(clk), .rst(rst),
        .load(load), .desc_base(desc_base), .desc_klen(desc_klen), .desc_nblk(desc_nblk),
        .mem_en(mem_en), .mem_addr(mem_addr), .mem_data(mem_data),
        .mm_start(mm_start), .mm_k_len(mm_k_len), .mm_w_row(mm_w_row),
        .mm_w_scale(mm_w_scale), .mm_in_valid(mm_in_valid),
        .busy(ld_busy), .done(ld_done)
    );

    // activation side for the DUT: the TB owns it (the loader is the beat
    // master).  a_col tracks the loader's beat stream; a_shift is latched at
    // mm_start and held constant.
    reg  [ 8*PE_M-1:0]        ash_d = 0;
    reg  [KW:0]              dut_beat;
    integer                  dut_K;
    reg  [16*PE_M-1:0]        acol_d;
    wire                      dut_busy;
    wire                      dut_ov;
    wire [16*PE_M*PE_N-1:0]   dut_cout;

    glm_matmul_fp8 #(.PE_M(PE_M), .PE_N(PE_N), .KMAX(KMAX), .BLK(BLK)) u_dut (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_k_len),
        .in_valid(mm_in_valid), .a_col(acol_d), .w_row(mm_w_row),
        .a_shift(ash_d), .w_scale(mm_w_scale),
        .busy(dut_busy), .out_valid(dut_ov), .c_out(dut_cout)
    );

    // beat counter: reset at mm_start, ++ on each streamed weight beat, so on
    // the cycle mm_in_valid is high for beat k, dut_beat == k.
    always @(posedge clk) begin
        if (rst)            dut_beat <= {(KW+1){1'b0}};
        else if (mm_start)  dut_beat <= {(KW+1){1'b0}};
        else if (mm_in_valid) dut_beat <= dut_beat + 1'b1;
    end

    // combinational activation feed -> A[*][dut_beat] on the in_valid cycle.
    integer ai;
    always @* begin
        acol_d = {(16*PE_M){1'b0}};
        for (ai = 0; ai < PE_M; ai = ai + 1)
            acol_d[16*ai +: 16] = (dut_beat < dut_K) ? A[ai][dut_beat] : 16'h0000;
    end

    // ---------------- latency-1 read memory (DDR5/Flash stub) ----------------
    localparam integer MEM_WORDS = 1024;
    reg [DATA_W-1:0] mem [0:MEM_WORDS-1];
    always @(posedge clk) begin
        if (mem_en) mem_data <= mem[mem_addr];
    end

    // ====================================================================
    // Random generators (independent of the DUT functions).
    // ====================================================================
    // moderate normal bf16 (exp in [lo_e,hi_e], hi_e<255 so never inf/nan).
    function [15:0] rnd_bf16(input integer lo_e, input integer hi_e);
        reg [7:0] e; reg [6:0] m; reg s;
        begin
            s = $random & 1;
            e = lo_e + ({$random} % (hi_e - lo_e + 1));
            m = $random;
            rnd_bf16 = {s, e, m};
        end
    endfunction

    // non-NaN E4M3 code.
    function [7:0] rnd_e4m3;
        reg [7:0] x;
        begin
            x = $random;
            if (x[6:0] == 7'b1111111) x[0] = 1'b0;   // dodge the NaN pattern
            rnd_e4m3 = x;
        end
    endfunction

    // fill a fresh random tile: K activation/weight rows, nblk block scales.
    task gen_tile(input integer K, input integer nblk);
        integer pi, pj, k, b;
        begin
            for (pi = 0; pi < PE_M; pi = pi + 1)
                ash_v[pi] = $random % 5;            // small +/- pow2 shift
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (b = 0; b < NB; b = b + 1)
                    wsc_v[pj][b] = (b < nblk) ? rnd_bf16(122, 132) : 16'h0000;
            for (k = 0; k < K; k = k + 1) begin
                for (pi = 0; pi < PE_M; pi = pi + 1) A[pi][k] = rnd_bf16(118, 132);
                for (pj = 0; pj < PE_N; pj = pj + 1) W[k][pj] = rnd_e4m3();
            end
        end
    endtask

    // ====================================================================
    // REFERENCE drive: weights packed DIRECTLY, capture c_ref.
    // ====================================================================
    reg [16*PE_M*PE_N-1:0] c_ref;
    task run_ref(input integer K, input integer nblk);
        integer pi, pj, k, b;
        begin
            @(negedge clk);
            for (pi = 0; pi < PE_M; pi = pi + 1) ash_r[8*pi +: 8] = ash_v[pi];
            wsc_r = {(16*PE_N*NB){1'b0}};                 // zero-fill (match loader)
            for (b = 0; b < nblk; b = b + 1)
                for (pj = 0; pj < PE_N; pj = pj + 1)
                    wsc_r[16*(b*PE_N + pj) +: 16] = wsc_v[pj][b];
            klen_r  = K[KW-1:0];
            start_r = 1'b1;
            inv_r   = 1'b0;
            @(negedge clk);
            start_r = 1'b0;
            for (k = 0; k < K; k = k + 1) begin
                inv_r = 1'b1;
                for (pi = 0; pi < PE_M; pi = pi + 1) acol_r[16*pi +: 16] = A[pi][k];
                for (pj = 0; pj < PE_N; pj = pj + 1) wrow_r[8*pj  +:  8] = W[k][pj];
                @(negedge clk);
            end
            inv_r = 1'b0; acol_r = 0; wrow_r = 0;
            do @(negedge clk); while (ref_ov !== 1'b1);
            c_ref = ref_cout;
        end
    endtask

    // ====================================================================
    // DUT drive: lay the SAME weights into the image, let the loader pull.
    // ====================================================================
    reg [16*PE_M*PE_N-1:0] c_dut;

    // build the memory image in the storage layout the loader expects.
    task build_mem(input [ADDR_W-1:0] base, input integer K, input integer nblk);
        integer i, k, pj, b;
        reg [DATA_W-1:0] word;
        begin
            for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = {DATA_W{1'b0}};
            // SCALE region: base + (b*PE_N + pj), low 16 bits = bf16 block scale.
            for (b = 0; b < nblk; b = b + 1)
                for (pj = 0; pj < PE_N; pj = pj + 1)
                    mem[base + (b*PE_N + pj)] = {{(DATA_W-16){1'b0}}, wsc_v[pj][b]};
            // CODE region: base + nblk*PE_N + k, word[8*pj+:8] = W[k][pj].
            for (k = 0; k < K; k = k + 1) begin
                word = {DATA_W{1'b0}};
                for (pj = 0; pj < PE_N; pj = pj + 1) word[8*pj +: 8] = W[k][pj];
                mem[base + nblk*PE_N + k] = word;
            end
        end
    endtask

    task run_dut(input [ADDR_W-1:0] base, input integer K, input integer nblk);
        integer pi;
        begin
            build_mem(base, K, nblk);
            for (pi = 0; pi < PE_M; pi = pi + 1) ash_d[8*pi +: 8] = ash_v[pi];
            dut_K     = K;
            desc_base = base;
            desc_klen = K[KW-1:0];
            desc_nblk = nblk[BKW-1:0];
            @(negedge clk);
            load = 1'b1;
            @(negedge clk);
            load = 1'b0;
            do @(negedge clk); while (dut_ov !== 1'b1);
            c_dut = dut_cout;
        end
    endtask

    // ====================================================================
    // bit-exact compare (X-aware), one check per output element.
    // ====================================================================
    task compare(input [127:0] tag, input integer K, input integer nblk);
        integer e;
        reg [15:0] r, d;
        begin
            for (e = 0; e < PE_M*PE_N; e = e + 1) begin
                checks = checks + 1;
                r = c_ref[16*e +: 16];
                d = c_dut[16*e +: 16];
                if (^r === 1'bx) begin
                    errors = errors + 1;
                    $display("  FAIL [%0s] elem %0d: REF has X (%b)", tag, e, r);
                end else if (^d === 1'bx) begin
                    errors = errors + 1;
                    $display("  FAIL [%0s] elem %0d: DUT(loader) has X (%b)", tag, e, d);
                end else if (d !== r) begin
                    errors = errors + 1;
                    $display("  FAIL [%0s] elem %0d (K=%0d nblk=%0d): loader=%h direct=%h",
                             tag, e, K, nblk, d, r);
                end
            end
        end
    endtask

    // one tile end-to-end: generate, run reference, run DUT-via-loader, compare.
    task do_tile(input [ADDR_W-1:0] base, input integer K, input integer nblk,
                 input [127:0] tag);
        begin
            gen_tile(K, nblk);
            run_ref(K, nblk);
            run_dut(base, K, nblk);
            compare(tag, K, nblk);
            repeat (2) @(negedge clk);
        end
    endtask

    // ====================================================================
    // Stimulus
    // ====================================================================
    integer ti, Kr, nbr;
    initial begin
        rst = 1'b1;
        repeat (5) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ---- directed: single K-block, k_len << KMAX ----
        do_tile(16'd64,  4,   1, "K4_1BLK");
        do_tile(16'd128, 17,  1, "K17_1BLK");
        do_tile(16'd256, 128, 1, "K128_1BLK");      // full single block

        // ---- directed: TWO K-blocks (exercises w_scale (bj*PE_N+pj) indexing) ----
        do_tile(16'd400, 200, 2, "K200_2BLK");      // full + partial block
        do_tile(16'd700, 256, 2, "K256_2BLK");      // both blocks full

        // ---- randomized tiles, varied K / nblk = ceil(K/BLK) ----
        for (ti = 0; ti < 10; ti = ti + 1) begin
            Kr  = 1 + ({$random} % KMAX);            // 1..256
            nbr = (Kr + BLK - 1) / BLK;
            // image is re-zeroed per tile -> a fixed base is safe; vary it a
            // little to also exercise a non-trivial descriptor base.
            do_tile(16'd64 + (ti % 4)*16'd128, Kr, nbr, "RND");
        end

        if (errors == 0)
            $display("ALL %0d TESTS PASSED", checks);
        else begin
            $display("%0d/%0d CHECKS FAILED", errors, checks);
            $fatal(1, "weight_loader binding check FAILED");
        end
        $finish;
    end

    // safety timeout
    initial begin
        #200000;
        $display("TIMEOUT");
        $fatal(1, "weight_loader binding check TIMEOUT");
    end
endmodule
