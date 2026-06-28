`timescale 1ns/1ps
//============================================================================
// kv_cache_pager_tb.v  --  smoke TB for the MLA latent-KV ring + Flash pager
//----------------------------------------------------------------------------
// INDEPENDENT X-aware golden model:
//   * sw_row[p]  : the opaque row appended at logical position p (the truth).
//   * sw_count   : positions appended so far.
//   * resident window = last RESIDENT positions [sw_count-RESIDENT, sw_count-1].
//   A Flash backing model returns sw_row[idx] FLASH_LAT cycles after a held
//   flash_req (exactly the cold-row DMA the pager expects).
//
//   Checks: resident gathers come from the ring same fast latency; cold
//   (evicted) gathers come back via Flash; values always match what was
//   appended; ring WRAPAROUND and overflow EVICTION (overflowed / resident_lo)
//   behave; a mixed resident+cold gather stream is exercised.
//
//   Prints "ALL <N> TESTS PASSED" with zero failures; $fatal on any mismatch.
//============================================================================
module kv_cache_pager_tb;

    localparam integer ROW_BITS  = 64;
    localparam integer RESIDENT  = 8;     // power of two
    localparam integer S_MAX     = 64;
    localparam integer POSW      = 6;     // clog2(64)
    localparam integer FLASH_LAT = 5;

    reg                 clk, rst;
    reg                 append_valid;
    reg  [ROW_BITS-1:0] append_row;
    reg                 gather_valid;
    reg  [POSW-1:0]     gather_idx;
    wire                row_valid;
    wire [ROW_BITS-1:0] row_out;
    wire                busy;
    wire                flash_req;
    wire [POSW-1:0]     flash_idx;
    reg                 flash_done;
    reg  [ROW_BITS-1:0] flash_row;
    wire [POSW-1:0]     append_count;
    wire [POSW-1:0]     resident_lo;
    wire                overflowed;

    kv_cache_pager #(
        .ROW_BITS(ROW_BITS), .RESIDENT(RESIDENT),
        .S_MAX(S_MAX), .POSW(POSW), .FLASH_LAT(FLASH_LAT)
    ) dut (
        .clk(clk), .rst(rst),
        .append_valid(append_valid), .append_row(append_row),
        .gather_valid(gather_valid), .gather_idx(gather_idx),
        .row_valid(row_valid), .row_out(row_out), .busy(busy),
        .flash_req(flash_req), .flash_idx(flash_idx),
        .flash_done(flash_done), .flash_row(flash_row),
        .append_count(append_count), .resident_lo(resident_lo),
        .overflowed(overflowed)
    );

    //------------------------------------------------------------------ clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    //------------------------------------------------ independent golden model
    reg [ROW_BITS-1:0] sw_row [0:S_MAX-1];
    integer            sw_count;
    integer            tests, errors;
    integer            k;
    integer            ridx, rlo;

    // unique opaque pattern per logical position (exercises the full row width).
    function [ROW_BITS-1:0] genrow(input [POSW-1:0] p);
        genrow = { (32'hC0DE0000 | {26'b0, p}), (32'hBEEF0000 | {26'b0, p}) };
    endfunction

    //------------------------------------------------------- Flash DMA model
    // On a HELD flash_req, wait FLASH_LAT cycles then pulse flash_done for one
    // cycle with sw_row[flash_idx] (the DUT holds flash_idx until done).
    integer flash_cnt;
    reg     flash_active;
    always @(posedge clk) begin
        if (rst) begin
            flash_done   <= 1'b0;
            flash_row    <= {ROW_BITS{1'b0}};
            flash_active <= 1'b0;
            flash_cnt    <= 0;
        end else begin
            flash_done <= 1'b0;
            if (flash_req && !flash_active && !flash_done) begin
                flash_active <= 1'b1;
                flash_cnt    <= FLASH_LAT;
            end else if (flash_active) begin
                if (flash_cnt <= 1) begin
                    flash_done   <= 1'b1;
                    flash_row    <= sw_row[flash_idx];
                    flash_active <= 1'b0;
                end else begin
                    flash_cnt <= flash_cnt - 1;
                end
            end
        end
    end

    //----------------------------------------------------------------- tasks
    task do_append(input [POSW-1:0] p);
        begin
            @(negedge clk);
            append_valid = 1'b1;
            append_row   = genrow(p);
            @(negedge clk);
            append_valid = 1'b0;
            sw_row[p]    = genrow(p);
            sw_count     = sw_count + 1;
        end
    endtask

    // issue one gather; wait for row_valid; check against the golden row.
    task do_gather(input [POSW-1:0] idx, input expect_cold);
        reg [ROW_BITS-1:0] got;
        integer wd;
        begin
            @(negedge clk);
            gather_valid = 1'b1;
            gather_idx   = idx;
            @(negedge clk);
            gather_valid = 1'b0;
            wd = 0;
            while (!row_valid) begin
                @(negedge clk);
                wd = wd + 1;
                if (wd > FLASH_LAT + 30) begin
                    $display("FAIL: gather idx=%0d timed out", idx);
                    $fatal(1, "gather timeout");
                end
            end
            got   = row_out;
            tests = tests + 1;
            if (got !== sw_row[idx]) begin
                errors = errors + 1;
                $display("FAIL: gather idx=%0d got=%h exp=%h (cold=%0d wd=%0d)",
                         idx, got, sw_row[idx], expect_cold, wd);
            end
            // latency sanity: resident must be fast (no Flash); cold must use it.
            if (!expect_cold && wd > 1) begin
                errors = errors + 1;
                $display("FAIL: resident idx=%0d took %0d cycles (expected fast)",
                         idx, wd);
            end
            if (expect_cold && wd < 2) begin
                errors = errors + 1;
                $display("FAIL: cold idx=%0d returned too fast (%0d) -- no Flash?",
                         idx, wd);
            end
        end
    endtask

    task check_obs(input [POSW-1:0] exp_cnt, input [POSW-1:0] exp_lo,
                   input exp_over);
        begin
            tests = tests + 1;
            if (append_count !== exp_cnt || resident_lo !== exp_lo ||
                overflowed !== exp_over) begin
                errors = errors + 1;
                $display("FAIL obs: cnt=%0d/%0d lo=%0d/%0d over=%0b/%0b",
                         append_count, exp_cnt, resident_lo, exp_lo,
                         overflowed, exp_over);
            end
        end
    endtask

    //------------------------------------------------------------- stimulus
    integer p;
    initial begin
        append_valid = 0; append_row = 0;
        gather_valid = 0; gather_idx = 0;
        sw_count = 0; tests = 0; errors = 0;

        rst = 1'b1;
        repeat (3) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // reset observability.
        check_obs(0, 0, 1'b0);

        //==================================================================
        // PHASE 1: fewer than RESIDENT appends -> everything resident, no Flash.
        //==================================================================
        for (p = 0; p < 5; p = p + 1) do_append(p[POSW-1:0]);
        check_obs(5, 0, 1'b0);                 // 5 < 8 -> no eviction, lo=0
        for (p = 0; p < 5; p = p + 1)
            do_gather(p[POSW-1:0], 1'b0);      // all resident (fast)

        //==================================================================
        // PHASE 2: fill exactly to RESIDENT -> ring full, still no eviction.
        //==================================================================
        for (p = 5; p < 8; p = p + 1) do_append(p[POSW-1:0]);
        check_obs(8, 0, 1'b0);                 // exactly full, lo=0, no overflow
        do_gather(6'd0, 1'b0);                 // oldest resident, fast
        do_gather(6'd7, 1'b0);                 // newest resident, fast

        //==================================================================
        // PHASE 3: overflow past RESIDENT -> wraparound + eviction to Flash.
        //   append positions 8..19 (12 more).  Ring keeps last 8 -> 12..19.
        //   Positions 0..11 are now COLD (only in Flash).
        //==================================================================
        for (p = 8; p < 20; p = p + 1) do_append(p[POSW-1:0]);
        check_obs(20, 12, 1'b1);               // lo=20-8=12, overflowed

        // resident window [12..19]: fast ring reads (wraparound slots).
        do_gather(6'd12, 1'b0);                // oldest resident
        do_gather(6'd19, 1'b0);                // newest resident
        do_gather(6'd15, 1'b0);                // middle resident

        // evicted/cold positions [0..11]: served via Flash.
        do_gather(6'd0,  1'b1);                // oldest cold
        do_gather(6'd11, 1'b1);                // newest cold (just evicted)
        do_gather(6'd3,  1'b1);
        do_gather(6'd7,  1'b1);

        //==================================================================
        // PHASE 4: mixed resident + cold stream (DSA-style scattered indices).
        //==================================================================
        do_gather(6'd18, 1'b0);   // resident
        do_gather(6'd2,  1'b1);   // cold
        do_gather(6'd13, 1'b0);   // resident
        do_gather(6'd9,  1'b1);   // cold
        do_gather(6'd17, 1'b0);   // resident
        do_gather(6'd5,  1'b1);   // cold

        //==================================================================
        // PHASE 5: append a few more, re-check the window slides correctly.
        //   append 20..23 -> count=24, resident [16..23], lo=16.
        //==================================================================
        for (p = 20; p < 24; p = p + 1) do_append(p[POSW-1:0]);
        check_obs(24, 16, 1'b1);
        do_gather(6'd23, 1'b0);   // newest resident
        do_gather(6'd16, 1'b0);   // oldest resident now
        do_gather(6'd15, 1'b1);   // just fell out of the window -> cold
        do_gather(6'd12, 1'b1);   // older cold

        //==================================================================
        // PHASE 6: randomized append/gather stream.
        //   Each step either appends the next logical position (while there is
        //   room < S_MAX) or gathers a random already-appended index; the cold
        //   class is predicted by the SAME golden window the pager must obey:
        //     lo = (count > RESIDENT) ? count-RESIDENT : 0; cold iff idx < lo.
        //   Also re-gathers the SAME index back-to-back to exercise repeats.
        //==================================================================
        for (k = 0; k < 60; k = k + 1) begin
            // bias toward appending early so a deep cold region builds up.
            if (sw_count < S_MAX-2 &&
                ($random % 3 != 0 || sw_count < RESIDENT+2)) begin
                do_append(sw_count[POSW-1:0]);
            end else begin
                // pick a random valid logical index in [0, sw_count-1].
                ridx = $unsigned($random) % sw_count;
                rlo  = (sw_count > RESIDENT) ? (sw_count - RESIDENT) : 0;
                do_gather(ridx[POSW-1:0], (ridx < rlo) ? 1'b1 : 1'b0);
                // gather the very same index again (repeat / cache-stability).
                do_gather(ridx[POSW-1:0], (ridx < rlo) ? 1'b1 : 1'b0);
            end
        end

        //------------------------------------------------------------ tally
        if (errors != 0) begin
            $display("FAILED: %0d errors out of %0d checks", errors, tests);
            $fatal(1, "kv_cache_pager_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end

    // safety net: never hang.
    initial begin
        #500000;
        $fatal(1, "TIMEOUT: kv_cache_pager_tb did not finish");
    end

endmodule
