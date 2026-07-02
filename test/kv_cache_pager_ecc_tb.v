`timescale 1ns/1ps
//============================================================================
// kv_cache_pager_ecc_tb.v  --  ECC=1 fault-injection proof for kv_cache_pager
//----------------------------------------------------------------------------
// Drives the SECDED-protected (ECC=1) variant of the latent-KV ring pager and
// proves the three C6 obligations against a stored-codeword fault model:
//
//   (1) CLEAN     : appended rows gather back EXACT, ecc_serr=ecc_derr=0.
//   (2) SBU       : XOR one bit into a STORED lane codeword (back-door poke of
//                   dut.ring[slot], the on-die codeword store) -> the gather
//                   still returns the EXACT row (SECDED corrected it) and the
//                   sticky ecc_serr rises; ecc_derr stays low.
//   (3) DBU       : XOR two bits into one stored lane codeword -> the gather
//                   sets the sticky ecc_derr (detected, uncorrectable).
//
// Two DUTs cover the lane geometry the memory map flags:
//   * DUT A ROW_BITS=100  -> NLANES=2: a FULL 64-bit lane0 + a RAGGED 36-bit
//     lane1 (28 pad bits) -> exercises the zero-pad ragged-lane path.
//   * DUT B ROW_BITS=768  -> NLANES=12: the REAL GLM-5.2 latent row, 12 clean
//     64-bit lanes.
//
// Fault injection uses a hierarchical procedural write into the DUT's on-die
// `ring` codeword array (dut.ring[slot]) -- the ring-granularity "back door"
// mirrored on kv_ecc_ring's bd_* port, needing NO new DUT ports.
//
//   Prints "ALL <N> TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module kv_cache_pager_ecc_tb;

    localparam integer LANE_W = 64;
    localparam integer CODE_W = 72;               // (72,64) SECDED codeword

    integer tests, errors;

    //------------------------------------------------------------------ clock
    reg clk = 1'b0;
    always #5 clk = ~clk;

    //==================================================================
    // DUT A : ragged ROW_BITS=100 (full lane0 + ragged lane1)
    //==================================================================
    localparam integer A_ROW    = 100;
    localparam integer A_RES    = 4;
    localparam integer A_SMAX   = 16;
    localparam integer A_POSW   = 4;              // clog2(16)
    localparam integer A_NLANES = (A_ROW + LANE_W - 1) / LANE_W;   // = 2
    localparam integer A_RINGW  = A_NLANES * CODE_W;               // = 144

    reg                 a_rst;
    reg                 a_append_valid;
    reg  [A_ROW-1:0]    a_append_row;
    reg                 a_gather_valid;
    reg  [A_POSW-1:0]   a_gather_idx;
    wire                a_row_valid;
    wire [A_ROW-1:0]    a_row_out;
    wire                a_busy;
    wire                a_flash_req;
    wire [A_POSW-1:0]   a_flash_idx;
    wire                a_serr, a_derr;
    wire [A_POSW-1:0]   a_append_count, a_resident_lo;
    wire                a_overflowed;

    kv_cache_pager #(
        .ROW_BITS(A_ROW), .RESIDENT(A_RES), .S_MAX(A_SMAX),
        .POSW(A_POSW), .FLASH_LAT(4), .ECC(1)
    ) duta (
        .clk(clk), .rst(a_rst),
        .append_valid(a_append_valid), .append_row(a_append_row),
        .gather_valid(a_gather_valid), .gather_idx(a_gather_idx),
        .row_valid(a_row_valid), .row_out(a_row_out), .busy(a_busy),
        .flash_req(a_flash_req), .flash_idx(a_flash_idx),
        .flash_done(1'b0), .flash_row({A_ROW{1'b0}}),
        .append_count(a_append_count), .resident_lo(a_resident_lo),
        .overflowed(a_overflowed),
        .ecc_serr(a_serr), .ecc_derr(a_derr)
    );

    function [A_ROW-1:0] a_genrow(input [A_POSW-1:0] p);
        a_genrow = { p, {3{ 32'hA5A50000 | {24'b0, p} }} };   // 4 + 96 = 100
    endfunction

    task a_append(input [A_POSW-1:0] p);
        begin
            @(negedge clk);
            a_append_valid = 1'b1; a_append_row = a_genrow(p);
            @(negedge clk);
            a_append_valid = 1'b0;
        end
    endtask

    // gather resident row p; check data (unless skip_data) + serr/derr flags.
    task a_gather_chk(input [A_POSW-1:0] p, input skip_data,
                      input exp_serr, input exp_derr);
        integer wd; reg [A_ROW-1:0] got;
        begin
            @(negedge clk);
            a_gather_valid = 1'b1; a_gather_idx = p;
            @(negedge clk);
            a_gather_valid = 1'b0;
            wd = 0;
            while (!a_row_valid) begin
                @(negedge clk); wd = wd + 1;
                if (wd > 20) begin $display("FAIL A: gather p=%0d timeout", p);
                                   $fatal(1, "timeout"); end
            end
            got = a_row_out;
            tests = tests + 1;
            if (!skip_data && got !== a_genrow(p)) begin
                errors = errors + 1;
                $display("FAIL A data: p=%0d got=%h exp=%h", p, got, a_genrow(p));
            end
            tests = tests + 1;
            if (a_serr !== exp_serr) begin errors = errors + 1;
                $display("FAIL A serr: p=%0d got=%b exp=%b", p, a_serr, exp_serr); end
            tests = tests + 1;
            if (a_derr !== exp_derr) begin errors = errors + 1;
                $display("FAIL A derr: p=%0d got=%b exp=%b", p, a_derr, exp_derr); end
        end
    endtask

    // XOR a mask into ONE stored lane codeword of slot (back-door poke).
    task a_inject(input [A_POSW-1:0] slot, input integer lane, input [CODE_W-1:0] mask);
        reg [A_RINGW-1:0] w;
        begin
            @(negedge clk);
            w = duta.ring[slot];
            w[lane*CODE_W +: CODE_W] = w[lane*CODE_W +: CODE_W] ^ mask;
            duta.ring[slot] = w;
        end
    endtask

    //==================================================================
    // DUT B : clean ROW_BITS=768 (the real GLM-5.2 latent row, 12 lanes)
    //==================================================================
    localparam integer B_ROW    = 768;
    localparam integer B_RES    = 4;
    localparam integer B_SMAX   = 16;
    localparam integer B_POSW   = 4;
    localparam integer B_NLANES = (B_ROW + LANE_W - 1) / LANE_W;   // = 12
    localparam integer B_RINGW  = B_NLANES * CODE_W;               // = 864

    reg                 b_rst;
    reg                 b_append_valid;
    reg  [B_ROW-1:0]    b_append_row;
    reg                 b_gather_valid;
    reg  [B_POSW-1:0]   b_gather_idx;
    wire                b_row_valid;
    wire [B_ROW-1:0]    b_row_out;
    wire                b_busy;
    wire                b_flash_req;
    wire [B_POSW-1:0]   b_flash_idx;
    wire                b_serr, b_derr;
    wire [B_POSW-1:0]   b_append_count, b_resident_lo;
    wire                b_overflowed;

    kv_cache_pager #(
        .ROW_BITS(B_ROW), .RESIDENT(B_RES), .S_MAX(B_SMAX),
        .POSW(B_POSW), .FLASH_LAT(4), .ECC(1)
    ) dutb (
        .clk(clk), .rst(b_rst),
        .append_valid(b_append_valid), .append_row(b_append_row),
        .gather_valid(b_gather_valid), .gather_idx(b_gather_idx),
        .row_valid(b_row_valid), .row_out(b_row_out), .busy(b_busy),
        .flash_req(b_flash_req), .flash_idx(b_flash_idx),
        .flash_done(1'b0), .flash_row({B_ROW{1'b0}}),
        .append_count(b_append_count), .resident_lo(b_resident_lo),
        .overflowed(b_overflowed),
        .ecc_serr(b_serr), .ecc_derr(b_derr)
    );

    function [B_ROW-1:0] b_genrow(input [B_POSW-1:0] p);
        b_genrow = {24{ 32'hBEEF0000 | {24'b0, p} }};   // 24 * 32 = 768
    endfunction

    task b_append(input [B_POSW-1:0] p);
        begin
            @(negedge clk);
            b_append_valid = 1'b1; b_append_row = b_genrow(p);
            @(negedge clk);
            b_append_valid = 1'b0;
        end
    endtask

    task b_gather_chk(input [B_POSW-1:0] p, input skip_data,
                      input exp_serr, input exp_derr);
        integer wd; reg [B_ROW-1:0] got;
        begin
            @(negedge clk);
            b_gather_valid = 1'b1; b_gather_idx = p;
            @(negedge clk);
            b_gather_valid = 1'b0;
            wd = 0;
            while (!b_row_valid) begin
                @(negedge clk); wd = wd + 1;
                if (wd > 20) begin $display("FAIL B: gather p=%0d timeout", p);
                                   $fatal(1, "timeout"); end
            end
            got = b_row_out;
            tests = tests + 1;
            if (!skip_data && got !== b_genrow(p)) begin
                errors = errors + 1;
                $display("FAIL B data: p=%0d got=%h exp=%h", p, got, b_genrow(p));
            end
            tests = tests + 1;
            if (b_serr !== exp_serr) begin errors = errors + 1;
                $display("FAIL B serr: p=%0d got=%b exp=%b", p, b_serr, exp_serr); end
            tests = tests + 1;
            if (b_derr !== exp_derr) begin errors = errors + 1;
                $display("FAIL B derr: p=%0d got=%b exp=%b", p, b_derr, exp_derr); end
        end
    endtask

    task b_inject(input [B_POSW-1:0] slot, input integer lane, input [CODE_W-1:0] mask);
        reg [B_RINGW-1:0] w;
        begin
            @(negedge clk);
            w = dutb.ring[slot];
            w[lane*CODE_W +: CODE_W] = w[lane*CODE_W +: CODE_W] ^ mask;
            dutb.ring[slot] = w;
        end
    endtask

    //------------------------------------------------------------- stimulus
    integer p;
    initial begin
        tests = 0; errors = 0;
        a_append_valid = 0; a_append_row = 0; a_gather_valid = 0; a_gather_idx = 0;
        b_append_valid = 0; b_append_row = 0; b_gather_valid = 0; b_gather_idx = 0;

        a_rst = 1'b1; b_rst = 1'b1;
        repeat (3) @(negedge clk);
        a_rst = 1'b0; b_rst = 1'b0;
        @(negedge clk);

        //================================================================
        // DUT A (ragged 100).  Append 4 rows -> all resident.
        //================================================================
        for (p = 0; p < 4; p = p + 1) a_append(p[A_POSW-1:0]);

        // (1) CLEAN: every row exact, no flags.
        for (p = 0; p < 4; p = p + 1)
            a_gather_chk(p[A_POSW-1:0], 1'b0, 1'b0, 1'b0);

        // (2a) SBU in the FULL lane0 of slot 0: flip codeword bit 3.
        a_inject(4'd0, 0, 72'h0000_0000_0000_0000_08);   // 1 bit set (bit 3)
        a_gather_chk(4'd0, 1'b0, 1'b1, 1'b0);            // corrected exact, serr=1

        // (2b) SBU in the RAGGED lane1 of slot 2: flip codeword bit 5.
        a_inject(4'd2, 1, 72'h0000_0000_0000_0000_20);   // 1 bit set (bit 5)
        a_gather_chk(4'd2, 1'b0, 1'b1, 1'b0);            // corrected exact, serr stays 1

        // a clean row still reads exact (serr sticky high, derr still 0).
        a_gather_chk(4'd1, 1'b0, 1'b1, 1'b0);

        // (3) DBU in lane0 of slot 3: flip TWO codeword bits (0 and 1).
        a_inject(4'd3, 0, 72'h0000_0000_0000_0000_03);   // 2 bits set (bits 0,1)
        a_gather_chk(4'd3, 1'b1, 1'b1, 1'b1);            // derr=1 (data uncorrectable)

        //================================================================
        // DUT B (clean 768, real latent row).  Append 4 rows -> resident.
        //================================================================
        for (p = 0; p < 4; p = p + 1) b_append(p[B_POSW-1:0]);

        // (1) CLEAN: exact, no flags.
        for (p = 0; p < 4; p = p + 1)
            b_gather_chk(p[B_POSW-1:0], 1'b0, 1'b0, 1'b0);

        // (2) SBU in lane 11 (top clean lane) of slot 1: flip codeword bit 7.
        b_inject(4'd1, 11, 72'h0000_0000_0000_0000_80);  // 1 bit set (bit 7)
        b_gather_chk(4'd1, 1'b0, 1'b1, 1'b0);            // corrected exact, serr=1

        // (3) DBU in lane 0 of slot 2: flip TWO codeword bits (2 and 4).
        b_inject(4'd2, 0, 72'h0000_0000_0000_0000_14);   // 2 bits set (bits 2,4)
        b_gather_chk(4'd2, 1'b1, 1'b1, 1'b1);            // derr=1

        //------------------------------------------------------------ tally
        if (errors != 0) begin
            $display("FAILED: %0d errors out of %0d checks", errors, tests);
            $fatal(1, "kv_cache_pager_ecc_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end

    // safety net.
    initial begin
        #500000;
        $fatal(1, "TIMEOUT: kv_cache_pager_ecc_tb did not finish");
    end

endmodule
