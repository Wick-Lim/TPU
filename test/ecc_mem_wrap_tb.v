`timescale 1ns/1ps
//============================================================================
// ecc_mem_wrap_tb.v  --  correctness TB for the SECDED memory wrapper (P2.1)
//----------------------------------------------------------------------------
// Drives src/ecc_mem_wrap.v (a DATA_W x DEPTH SECDED-protected synchronous
// RAM) and verifies the three memory-path guarantees:
//
//   1) CLEAN         : write words, read them back -> payload identical to what
//                      was written, serr=0 and derr=0 (no phantom flags).
//   2) SINGLE-CORRECT: after a clean write, corrupt ONE bit of the stored
//                      codeword in place (back-door raw write), then read ->
//                      payload is CORRECTED back to the original AND serr=1,
//                      derr=0.  Exhaustive over every codeword-bit position.
//   3) DOUBLE-DETECT : corrupt TWO distinct bits of a stored codeword, read ->
//                      derr=1 (the double error is flagged, not miscorrected).
//                      Exhaustive over every distinct pair of positions.
//
// The corrupted codeword is produced by an INDEPENDENT ecc_secded encode
// instance in the TB (not by reading the DUT's internal array), then flipped
// and written through the wrapper's back-door raw-write port -- so the DUT's
// read/decode path is exercised exactly as it would be on real bit-rot.
//
// READ TIMING (matches the DUT's 2-stage synchronous read pipeline):
//   edge A: present raddr/re -> DUT latches mem[raddr] into rd_code
//   edge B: DUT registers decode(rd_code) into rdata/serr/derr
//   so rdata/serr/derr are valid 2 rising edges after re is sampled.
//============================================================================
module ecc_mem_wrap_tb;
    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off WIDTHTRUNC */

    //------------------------------------------------------------------
    // Geometry (independently derived; must match the DUT's localparams)
    //------------------------------------------------------------------
    localparam integer DATA_W = 64;
    localparam integer DEPTH  = 32;

    function integer calc_p;
        input integer dw;
        integer p;
        begin
            p = 0;
            while ((1 << p) < (dw + p + 1)) p = p + 1;
            calc_p = p;
        end
    endfunction

    function integer clog2;
        input integer n;
        integer v;
        begin
            clog2 = 0;
            v     = n - 1;
            while (v > 0) begin
                clog2 = clog2 + 1;
                v     = v >> 1;
            end
            if (clog2 == 0) clog2 = 1;
        end
    endfunction

    localparam integer P      = calc_p(DATA_W);
    localparam integer CODE_W = DATA_W + P + 1;
    localparam integer ADDR_W = clog2(DEPTH);

    //------------------------------------------------------------------
    // Clock / reset
    //------------------------------------------------------------------
    reg clk = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    reg rst;

    //------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------
    reg  [ADDR_W-1:0] waddr;
    reg               we;
    reg  [DATA_W-1:0] wdata;

    reg  [ADDR_W-1:0] raddr;
    reg               re;
    wire [DATA_W-1:0] rdata;
    wire              serr, derr;

    wire              serr_sticky, derr_sticky;
    reg               err_ack;

    reg               bd_we;
    reg  [ADDR_W-1:0] bd_addr;
    reg  [CODE_W-1:0] bd_code;

    ecc_mem_wrap #(.DATA_W(DATA_W), .DEPTH(DEPTH)) dut (
        .clk(clk), .rst(rst),
        .we(we), .waddr(waddr), .wdata(wdata),
        .re(re), .raddr(raddr), .rdata(rdata), .serr(serr), .derr(derr),
        .serr_sticky(serr_sticky), .derr_sticky(derr_sticky), .err_ack(err_ack),
        .bd_we(bd_we), .bd_addr(bd_addr), .bd_code(bd_code)
    );

    //------------------------------------------------------------------
    // Independent encode helper: payload -> codeword (for fault injection).
    // Uses a fresh ecc_secded instance (encode side only).
    //------------------------------------------------------------------
    reg  [DATA_W-1:0] enc_data;
    wire [CODE_W-1:0] enc_code;
    wire [DATA_W-1:0] enc_du;
    wire              enc_su, enc_dbu;

    ecc_secded #(.DATA_W(DATA_W)) u_encref (
        .data_in(enc_data), .code_out(enc_code),
        .code_in({CODE_W{1'b0}}), .data_out(enc_du),
        .single_err(enc_su), .double_err(enc_dbu)
    );

    //------------------------------------------------------------------
    // Harness
    //------------------------------------------------------------------
    integer tests = 0;
    integer fails = 0;
    integer n_clean = 0, n_single = 0, n_double = 0, n_scrub = 0;

    task check;
        input [400:0] name;
        input         cond;
        begin
            tests = tests + 1;
            if (cond !== 1'b1) begin
                fails = fails + 1;
                $display("FAIL [%0d]: %0s", fails, name);
            end
        end
    endtask

    // --- Bus-idle helper (all controls low). ---
    task bus_idle;
        begin
            we = 1'b0; re = 1'b0; bd_we = 1'b0; err_ack = 1'b0;
        end
    endtask

    // --- Pulse err_ack for one clock to clear the sticky flags. ---
    task ack_sticky;
        begin
            @(negedge clk);
            bus_idle;
            err_ack = 1'b1;
            @(posedge clk); #1;
            err_ack = 1'b0;
        end
    endtask

    // --- Encoded write of `d` to `addr` (1 posedge). ---
    task ecc_write;
        input [ADDR_W-1:0] addr;
        input [DATA_W-1:0] d;
        begin
            @(negedge clk);
            bus_idle;
            we = 1'b1; waddr = addr; wdata = d;
            @(posedge clk); #1;
            we = 1'b0;
        end
    endtask

    // --- Raw back-door codeword write of `c` to `addr` (fault inject). ---
    task raw_write;
        input [ADDR_W-1:0] addr;
        input [CODE_W-1:0] c;
        begin
            @(negedge clk);
            bus_idle;
            bd_we = 1'b1; bd_addr = addr; bd_code = c;
            @(posedge clk); #1;
            bd_we = 1'b0;
        end
    endtask

    // --- Synchronous read of `addr`; result valid after the 2-stage pipe. ---
    task ecc_read;
        input [ADDR_W-1:0] addr;
        begin
            @(negedge clk);
            bus_idle;
            re = 1'b1; raddr = addr;
            @(posedge clk); #1;   // edge A: mem[addr] -> rd_code
            re = 1'b0;
            @(posedge clk); #1;   // edge B: decode(rd_code) -> rdata/serr/derr
        end
    endtask

    //------------------------------------------------------------------
    // Test bodies
    //------------------------------------------------------------------
    localparam integer CLEAN_WORDS  = DEPTH;   // fill the whole array
    localparam integer SINGLE_WORDS = 8;
    localparam integer DOUBLE_WORDS = 4;

    reg  [DATA_W-1:0] dword;
    reg  [CODE_W-1:0] cw, bad;
    integer w, a, b, addr;

    initial begin
        if (CODE_W != 72)
            $display("NOTE: CODE_W=%0d (expected 72 for DATA_W=64)", CODE_W);

        // reset (synchronous, active high)
        bus_idle;
        we = 1'b0; re = 1'b0; bd_we = 1'b0; err_ack = 1'b0;
        waddr = 0; wdata = 0; raddr = 0; bd_addr = 0; bd_code = 0;
        rst = 1'b1;
        repeat (3) @(posedge clk);
        #1 rst = 1'b0;
        @(negedge clk);

        //==============================================================
        // 1) CLEAN write/read -- payload preserved, no flags.
        //==============================================================
        for (w = 0; w < CLEAN_WORDS; w = w + 1) begin
            dword = {$random, $random};
            addr  = w % DEPTH;
            ecc_write(addr[ADDR_W-1:0], dword);
            ecc_read (addr[ADDR_W-1:0]);
            check("clean: data preserved", rdata === dword);
            check("clean: no single",      serr  === 1'b0);
            check("clean: no double",      derr  === 1'b0);
            n_clean = n_clean + 1;
        end

        //==============================================================
        // 2) SINGLE-bit flip -> corrected + serr, exhaustive positions.
        //==============================================================
        for (w = 0; w < SINGLE_WORDS; w = w + 1) begin
            dword    = {$random, $random};
            addr     = w % DEPTH;
            // reference codeword for this payload (independent encode)
            enc_data = dword; #1;
            cw       = enc_code;

            for (a = 0; a < CODE_W; a = a + 1) begin
                bad = cw ^ (1'b1 << a);          // flip exactly one bit
                raw_write(addr[ADDR_W-1:0], bad); // corrupt the stored word
                ecc_read (addr[ADDR_W-1:0]);
                // single-bit error is CORRECTED back to the original payload
                check("single: data corrected", rdata === dword);
                check("single: serr=1",         serr  === 1'b1);
                check("single: derr=0",         derr  === 1'b0);
                n_single = n_single + 1;
            end
        end

        //==============================================================
        // 3) DOUBLE-bit flip -> derr, exhaustive distinct pairs.
        //==============================================================
        for (w = 0; w < DOUBLE_WORDS; w = w + 1) begin
            dword    = {$random, $random};
            addr     = w % DEPTH;
            enc_data = dword; #1;
            cw       = enc_code;

            for (a = 0; a < CODE_W; a = a + 1)
                for (b = a + 1; b < CODE_W; b = b + 1) begin
                    bad = cw ^ (1'b1 << a) ^ (1'b1 << b);  // flip two bits
                    raw_write(addr[ADDR_W-1:0], bad);
                    ecc_read (addr[ADDR_W-1:0]);
                    // double error is DETECTED (never silently miscorrected)
                    check("double: derr=1", derr === 1'b1);
                    check("double: serr=0", serr === 1'b0);
                    n_double = n_double + 1;
                end
        end

        //==============================================================
        // 4) SCRUB WRITE-BACK + STICKY flags.
        //    A single-bit fault is injected into a stored word; the first read
        //    corrects it (serr=1) AND heals the array; a second read of the
        //    SAME address then sees serr=0.  Sticky flags latch across reads
        //    and are cleared by err_ack.  A double fault sets derr (+sticky)
        //    and is NOT healed.
        //==============================================================
        // Start from a clean sticky state (prior sections tripped the flags).
        ack_sticky;
        check("scrub: sticky serr cleared by ack", serr_sticky === 1'b0);
        check("scrub: sticky derr cleared by ack", derr_sticky === 1'b0);

        // ---- single-bit fault -> corrected, serr, then scrubbed away ----
        dword    = {$random, $random};
        addr     = 7;
        enc_data = dword; #1;
        cw       = enc_code;

        ecc_write(addr[ADDR_W-1:0], dword);      // clean codeword in the array
        bad = cw ^ (1'b1 << 3);                  // rot a single data bit
        raw_write(addr[ADDR_W-1:0], bad);        // inject the fault in place

        ecc_read(addr[ADDR_W-1:0]);              // read #1: correct + heal
        check("scrub: read1 data corrected", rdata       === dword);
        check("scrub: read1 serr=1",         serr        === 1'b1);
        check("scrub: read1 derr=0",         derr        === 1'b0);
        check("scrub: read1 sticky serr=1",  serr_sticky === 1'b1);
        check("scrub: read1 sticky derr=0",  derr_sticky === 1'b0);

        ecc_read(addr[ADDR_W-1:0]);              // read #2: bit was healed
        check("scrub: read2 data still ok",  rdata === dword);
        check("scrub: read2 serr=0 (healed)",serr  === 1'b0);
        check("scrub: read2 derr=0",         derr  === 1'b0);
        // sticky serr still latched from read #1 until acked
        check("scrub: sticky serr still latched", serr_sticky === 1'b1);
        n_scrub = n_scrub + 1;

        ack_sticky;                              // clear the latch
        check("scrub: sticky serr cleared", serr_sticky === 1'b0);

        // ---- double-bit fault -> derr, sticky derr, NOT healed ----
        dword    = {$random, $random};
        addr     = 9;
        enc_data = dword; #1;
        cw       = enc_code;

        ecc_write(addr[ADDR_W-1:0], dword);
        bad = cw ^ (1'b1 << 3) ^ (1'b1 << 10);   // two distinct rotted bits
        raw_write(addr[ADDR_W-1:0], bad);

        ecc_read(addr[ADDR_W-1:0]);              // read #1: detect double
        check("scrub: dbl read1 derr=1",        derr        === 1'b1);
        check("scrub: dbl read1 serr=0",        serr        === 1'b0);
        check("scrub: dbl read1 sticky derr=1", derr_sticky === 1'b1);

        ecc_read(addr[ADDR_W-1:0]);              // read #2: still bad (no heal)
        check("scrub: dbl read2 derr=1 (not healed)", derr === 1'b1);
        check("scrub: dbl sticky derr still latched",  derr_sticky === 1'b1);
        n_scrub = n_scrub + 1;

        ack_sticky;
        check("scrub: sticky derr cleared", derr_sticky === 1'b0);

        //==============================================================
        // verdict
        //==============================================================
        $display("clean words        : %0d", n_clean);
        $display("single-flip cases  : %0d (all %0d positions x %0d words)",
                 n_single, CODE_W, SINGLE_WORDS);
        $display("double-flip cases  : %0d (all C(%0d,2) x %0d words)",
                 n_double, CODE_W, DOUBLE_WORDS);
        $display("scrub/sticky cases : %0d", n_scrub);

        if (fails == 0)
            $display("ALL %0d TESTS PASSED", tests);
        else begin
            $display("%0d of %0d CHECKS FAILED", fails, tests);
            $fatal;
        end
        $finish;
    end

    /* verilator lint_on WIDTHTRUNC  */
    /* verilator lint_on WIDTHEXPAND */
endmodule
