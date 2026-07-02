`timescale 1ns/1ps
//============================================================================
// kv_ecc_ring_tb.v  --  self-checking TB for the SECDED KV ring  (C6)
//----------------------------------------------------------------------------
// ROW_BITS = 100 is deliberately NOT a multiple of 64: it partitions into
//   lane 0 = 64 valid bits (full clean lane)
//   lane 1 = 36 valid bits, zero-padded to 64 (RAGGED final lane)
// so every test that crosses a lane boundary also exercises the ragged tail.
//
// Proves, with $fatal on any mismatch / X:
//   T1  write 3 rows, read each back CLEAN (data exact, serr=0, derr=0)
//   T2  single-bit fault in the FULL lane 0        -> corrected data, serr=1
//   T3  single-bit fault in the RAGGED lane 1      -> corrected data, serr=1
//   T4  double-bit fault in the FULL lane 0        -> derr=1
//   T5  double-bit fault in the RAGGED lane 1      -> derr=1
// Prints "ALL N TESTS PASSED".
//============================================================================
module kv_ecc_ring_tb;

    localparam integer ROW_BITS = 100;   // ragged: 64 + 36
    localparam integer DEPTH    = 4;
    localparam integer CODE_W   = 72;    // 64b payload SECDED codeword

    reg                    clk = 1'b0;
    reg                    rst;

    reg                    we;
    reg  [1:0]             waddr;        // clog2(DEPTH)=2
    reg  [ROW_BITS-1:0]    wdata;

    reg                    re;
    reg  [1:0]             raddr;
    wire [ROW_BITS-1:0]    rdata;
    wire                   serr;
    wire                   derr;

    reg                    bd_we;
    reg  [1:0]             bd_addr;
    reg  [0:0]             bd_lane;      // clog2(NLANES=2)=1
    reg  [CODE_W-1:0]      bd_xor;

    integer                npass = 0;

    kv_ecc_ring #(.ROW_BITS(ROW_BITS), .DEPTH(DEPTH)) dut (
        .clk(clk), .rst(rst),
        .we(we), .waddr(waddr), .wdata(wdata),
        .re(re), .raddr(raddr), .rdata(rdata), .serr(serr), .derr(derr),
        .bd_we(bd_we), .bd_addr(bd_addr), .bd_lane(bd_lane), .bd_xor(bd_xor)
    );

    always #5 clk = ~clk;

    // Single-bit and double-bit fault masks on a stored lane codeword.
    // bit2 = a DATA-carrying position (Hamming pos 3) so correction is
    // observable; bits{2,3} = two distinct positions -> guaranteed DBU.
    localparam [CODE_W-1:0] MASK_SBU = 72'h4;
    localparam [CODE_W-1:0] MASK_DBU = 72'hC;

    task do_write(input [1:0] a, input [ROW_BITS-1:0] d);
        begin
            @(negedge clk); we = 1'b1; waddr = a; wdata = d;
            @(negedge clk); we = 1'b0;
        end
    endtask

    task do_inject(input [1:0] a, input [0:0] lane, input [CODE_W-1:0] mask);
        begin
            @(negedge clk); bd_we = 1'b1; bd_addr = a; bd_lane = lane; bd_xor = mask;
            @(negedge clk); bd_we = 1'b0;
        end
    endtask

    // Read row a (2-cycle latency); returned outputs valid when task returns.
    task do_read(input [1:0] a);
        begin
            @(negedge clk); re = 1'b1; raddr = a;   // stage-1 fetch on next posedge
            @(negedge clk); re = 1'b0;              // stage-2 decode on next posedge
            @(negedge clk);                         // outputs now registered/valid
        end
    endtask

    task check_clean(input [ROW_BITS-1:0] exp);
        begin
            if ((^{rdata, serr, derr}) === 1'bx) begin
                $display("FAIL: X on clean read"); $fatal;
            end
            if (rdata !== exp) begin
                $display("FAIL: clean data exp=%h got=%h", exp, rdata); $fatal;
            end
            if (serr !== 1'b0 || derr !== 1'b0) begin
                $display("FAIL: clean flags serr=%b derr=%b", serr, derr); $fatal;
            end
            npass = npass + 1;
        end
    endtask

    task check_corrected(input [ROW_BITS-1:0] exp);
        begin
            if ((^{rdata, serr, derr}) === 1'bx) begin
                $display("FAIL: X on corrected read"); $fatal;
            end
            if (rdata !== exp) begin
                $display("FAIL: SBU not corrected exp=%h got=%h", exp, rdata); $fatal;
            end
            if (serr !== 1'b1) begin
                $display("FAIL: SBU serr not set (serr=%b)", serr); $fatal;
            end
            if (derr !== 1'b0) begin
                $display("FAIL: SBU spurious derr (derr=%b)", derr); $fatal;
            end
            npass = npass + 1;
        end
    endtask

    task check_double;
        begin
            if ((^{serr, derr}) === 1'bx) begin
                $display("FAIL: X on double-error flags"); $fatal;
            end
            if (derr !== 1'b1) begin
                $display("FAIL: DBU not detected (derr=%b)", derr); $fatal;
            end
            npass = npass + 1;
        end
    endtask

    reg [ROW_BITS-1:0] pA, pB, pC, pD;

    initial begin
        // distinct 100-bit patterns; low 64b -> lane0, high 36b -> lane1
        pA = {36'hA5A5A5A5A, 64'h0123456789ABCDEF};
        pB = {36'h5A5A5A5A5, 64'hFEDCBA9876543210};
        pC = {36'hF0F0F0F0F, 64'hCAFEF00DDEADBEEF};
        pD = {36'h123456789, 64'h8000000000000001};

        we = 0; waddr = 0; wdata = 0;
        re = 0; raddr = 0;
        bd_we = 0; bd_addr = 0; bd_lane = 0; bd_xor = 0;

        rst = 1'b1;
        repeat (3) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ---- T1: write 3 rows, read each back CLEAN ----
        do_write(2'd0, pA);
        do_write(2'd1, pB);
        do_write(2'd2, pC);
        do_read(2'd0); check_clean(pA);
        do_read(2'd1); check_clean(pB);
        do_read(2'd2); check_clean(pC);

        // ---- T2: single-bit fault in FULL lane 0 of row 1 -> corrected ----
        do_inject(2'd1, 1'b0, MASK_SBU);
        do_read(2'd1); check_corrected(pB);

        // ---- T3: single-bit fault in RAGGED lane 1 of row 2 -> corrected ----
        do_inject(2'd2, 1'b1, MASK_SBU);
        do_read(2'd2); check_corrected(pC);

        // sanity: heal-free re-read still corrects (fault persists, no scrub)
        do_read(2'd1); check_corrected(pB);

        // ---- T4: double-bit fault in FULL lane 0 of a fresh row -> derr ----
        do_write(2'd3, pD);
        do_inject(2'd3, 1'b0, MASK_DBU);
        do_read(2'd3); check_double;

        // ---- T5: double-bit fault in RAGGED lane 1 of row 0 -> derr ----
        do_inject(2'd0, 1'b1, MASK_DBU);
        do_read(2'd0); check_double;

        $display("ALL %0d TESTS PASSED", npass);
        $finish;
    end

    // safety net: never hang
    initial begin
        #100000;
        $display("FAIL: timeout"); $fatal;
    end

endmodule
