`include "tpu_defs.vh"
`timescale 1ns/1ps
//============================================================================
// scatter_gather_tb  --  self-checking unit TB for scatter_gather.v
//----------------------------------------------------------------------------
// Verifies the v2.0 indexed DMEM access engine (SPEC §2, §5.5):
//   * effective address  eff = base + (index << stride_exp)
//   * GATHER  (is_scatter=0): data_out = DMEM[eff]   (combinational load)
//   * SCATTER (is_scatter=1): DMEM[eff] = data_in     (synchronous store)
//   * stride_exp in {0,1,2,3}; index = 0 and index = max corner cases.
//
// SELF-CONTAINED MEMORY MODEL:
//   The DUT exposes DMEM ACCESS PORTS and does NOT instantiate any memory, so
//   the TB models DMEM itself (a flat 256x32 reg array `dmem`) and drives the
//   DUT's dmem_rdata from it / commits the DUT's dmem_we+dmem_wdata into it on
//   the clock edge.  No src/ memory module is instantiated.  The combinational
//   read path mirrors the v2.0 DMEM contract (combinational read, sync write).
//
// INDEPENDENT GOLDEN MODEL (must NOT replicate the DUT's arithmetic):
//   1. Effective address: the DUT computes  base + (index << stride_exp).  The
//      golden instead computes it by MULTIPLICATION in a 64-bit space and wraps
//      at XLEN:  exp_addr = (base + index * (64'd1 << stride_exp)) mod 2^XLEN.
//      Using *, a wide accumulator, and an explicit modulo is a DIFFERENT
//      mechanism than the DUT's truncating left-shift, so an off-by-shift or a
//      width-wrap bug in the DUT is caught.
//   2. Memory contents: a SEPARATE golden shadow array `gold[]` receives the
//      same architectural scatter writes via straight-line procedural stores,
//      independent of the TB's own `dmem[]` that feeds the DUT -- so a gather
//      is checked against gold[exp_addr], and the round-trip cross-checks both.
//
// All comparisons are BIT-EXACT (the unit is pure integer address math + word
// movement -- no Q-format, no LUT, tolerance = 0 LSB).  $fatal on any mismatch.
// Prints "ALL <N> TESTS PASSED" with zero failures at the end.
//============================================================================
module scatter_gather_tb;

    // ---- sizes mirrored (read-only) from tpu_defs ----
    localparam integer XLEN   = `XLEN;          // 32
    localparam integer AW     = `DMEM_ADDR_W;   // 8
    localparam integer DEPTH  = `DMEM_DEPTH;    // 256
    localparam integer SEW    = 2;              // stride-exp width (imm12[1:0])
    localparam integer NRAND  = 400;            // constrained-random vectors

    // ---- clock / DMEM-commit handshake ----
    reg clk;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ---- DUT ports ----
    reg              start;
    reg              is_scatter;
    reg  [XLEN-1:0]  base_addr;
    reg  [XLEN-1:0]  index;
    reg  [SEW-1:0]   stride_exp;
    reg  [XLEN-1:0]  data_in;
    wire [XLEN-1:0]  dmem_rdata;     // driven combinationally from TB dmem[]
    wire [XLEN-1:0]  addr_out;
    wire [AW-1:0]    dmem_addr;
    wire [XLEN-1:0]  dmem_wdata;
    wire             dmem_we;
    wire [XLEN-1:0]  data_out;

    // ---- DUT ----
    scatter_gather #(.STRIDE_EXP_W(SEW)) dut (
        .start      (start),
        .is_scatter (is_scatter),
        .base_addr  (base_addr),
        .index      (index),
        .stride_exp (stride_exp),
        .data_in    (data_in),
        .dmem_rdata (dmem_rdata),
        .addr_out   (addr_out),
        .dmem_addr  (dmem_addr),
        .dmem_wdata (dmem_wdata),
        .dmem_we    (dmem_we),
        .data_out   (data_out)
    );

    // ---- TB-modelled DMEM (feeds the DUT; combinational read, sync write) ----
    reg [XLEN-1:0] dmem [0:DEPTH-1];
    assign dmem_rdata = dmem[dmem_addr];     // combinational read port

    // Commit a started SCATTER on the rising edge (synchronous write), exactly
    // as the real v2.0 DMEM would.  Independent of the golden shadow below.
    always @(posedge clk) begin
        if (dmem_we)
            dmem[dmem_addr] <= dmem_wdata;
    end

    // ---- Independent golden shadow (updated by a DIFFERENT path) ----
    reg [XLEN-1:0] gold [0:DEPTH-1];

    // ---- bookkeeping ----
    integer tests;
    integer errors;
    integer i;
    integer s;
    integer seed;

    // ---- golden effective-address: multiply + 64-bit accumulate + wrap ----
    // Deliberately computed by MULTIPLY (not the DUT's left-shift) so the two
    // share no arithmetic.  Returns the wrapped XLEN word address.
    function [XLEN-1:0] gold_addr(input [XLEN-1:0] b,
                                  input [XLEN-1:0] idx,
                                  input [SEW-1:0]  se);
        reg [63:0] mult;
        reg [63:0] sum;
        begin
            mult     = {32'd0, idx} * (64'd1 << se);   // index * stride (no shift)
            sum      = {32'd0, b} + mult;              // base + that
            gold_addr = sum[XLEN-1:0];                 // wrap at XLEN
        end
    endfunction

    // ---- golden DMEM physical word address (low AW bits, DMEM wraps) ----
    function [AW-1:0] gold_pa(input [XLEN-1:0] b,
                             input [XLEN-1:0] idx,
                             input [SEW-1:0]  se);
        reg [XLEN-1:0] ea;
        begin
            ea      = gold_addr(b, idx, se);
            gold_pa = ea[AW-1:0];
        end
    endfunction

    //------------------------------------------------------------------
    // Check the combinational effective-address outputs for a settled stimulus.
    //------------------------------------------------------------------
    task check_addr;
        reg [XLEN-1:0] eA;
        reg [AW-1:0]   pA;
        begin
            #1;                                   // allow combinational settle
            eA = gold_addr(base_addr, index, stride_exp);
            pA = gold_pa  (base_addr, index, stride_exp);

            tests = tests + 1;
            if (addr_out !== eA) begin
                errors = errors + 1;
                $display("FAIL[addr] base=%h idx=%h se=%0d got=%h exp=%h",
                         base_addr, index, stride_exp, addr_out, eA);
                $fatal(1, "effective-address mismatch");
            end
            tests = tests + 1;
            if (dmem_addr !== pA) begin
                errors = errors + 1;
                $display("FAIL[paddr] base=%h idx=%h se=%0d got=%h exp=%h",
                         base_addr, index, stride_exp, dmem_addr, pA);
                $fatal(1, "physical-address mismatch");
            end
        end
    endtask

    //------------------------------------------------------------------
    // GATHER (load): drive a started gather and check data_out == gold[eff].
    // No memory write occurs (dmem_we must stay low).
    //------------------------------------------------------------------
    task do_gather(input [XLEN-1:0] b, input [XLEN-1:0] idx, input [SEW-1:0] se);
        reg [AW-1:0] pA;
        begin
            @(negedge clk);
            start      = 1'b1;
            is_scatter = 1'b0;
            base_addr  = b;
            index      = idx;
            stride_exp = se;
            data_in    = 32'h0;
            check_addr;                           // also validates the address

            pA = gold_pa(b, idx, se);
            tests = tests + 1;
            if (dmem_we !== 1'b0) begin
                errors = errors + 1;
                $display("FAIL[gather-we] base=%h idx=%h se=%0d we=%b (must be 0)",
                         b, idx, se, dmem_we);
                $fatal(1, "gather asserted write-enable");
            end
            tests = tests + 1;
            if (data_out !== gold[pA]) begin
                errors = errors + 1;
                $display("FAIL[gather] base=%h idx=%h se=%0d pa=%0d got=%h exp=%h",
                         b, idx, se, pA, data_out, gold[pA]);
                $fatal(1, "gather load data mismatch");
            end
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    //------------------------------------------------------------------
    // SCATTER (store): drive a started scatter, confirm we/addr/wdata, let the
    // edge commit, then update the golden shadow by an INDEPENDENT plain store.
    //------------------------------------------------------------------
    task do_scatter(input [XLEN-1:0] b, input [XLEN-1:0] idx,
                    input [SEW-1:0] se, input [XLEN-1:0] d);
        reg [AW-1:0] pA;
        begin
            @(negedge clk);
            start      = 1'b1;
            is_scatter = 1'b1;
            base_addr  = b;
            index      = idx;
            stride_exp = se;
            data_in    = d;
            check_addr;

            pA = gold_pa(b, idx, se);
            tests = tests + 1;
            if (dmem_we !== 1'b1) begin
                errors = errors + 1;
                $display("FAIL[scatter-we] base=%h idx=%h se=%0d we=%b (must be 1)",
                         b, idx, se, dmem_we);
                $fatal(1, "scatter did not assert write-enable");
            end
            tests = tests + 1;
            if (dmem_wdata !== d) begin
                errors = errors + 1;
                $display("FAIL[scatter-wd] got=%h exp=%h", dmem_wdata, d);
                $fatal(1, "scatter write-data mismatch");
            end
            @(posedge clk);                       // DUT+TB dmem commit here
            #1;
            gold[pA] = d;                         // golden: independent store
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    //------------------------------------------------------------------
    // Round-trip: scatter a value, then gather it back -> must equal the value.
    //------------------------------------------------------------------
    task round_trip(input [XLEN-1:0] b, input [XLEN-1:0] idx,
                    input [SEW-1:0] se, input [XLEN-1:0] d);
        begin
            do_scatter(b, idx, se, d);
            do_gather (b, idx, se);
            tests = tests + 1;
            if (data_out !== d) begin
                errors = errors + 1;
                $display("FAIL[round-trip] base=%h idx=%h se=%0d wrote=%h read=%h",
                         b, idx, se, d, data_out);
                $fatal(1, "scatter->gather round-trip mismatch");
            end
        end
    endtask

    //------------------------------------------------------------------
    // start=0 must NOT enable a write and must drive data_out to 0.
    //------------------------------------------------------------------
    task check_idle(input [XLEN-1:0] b, input [XLEN-1:0] idx, input [SEW-1:0] se);
        begin
            @(negedge clk);
            start      = 1'b0;
            is_scatter = 1'b1;            // even with scatter selected...
            base_addr  = b;
            index      = idx;
            stride_exp = se;
            data_in    = 32'hDEAD_BEEF;
            #1;
            tests = tests + 1;
            if (dmem_we !== 1'b0) begin
                errors = errors + 1;
                $display("FAIL[idle-we] start=0 but we=%b", dmem_we);
                $fatal(1, "idle cycle asserted write-enable");
            end
            // gather output must be flushed to 0 when not a started gather
            is_scatter = 1'b0;
            #1;
            tests = tests + 1;
            if (data_out !== 32'h0) begin
                errors = errors + 1;
                $display("FAIL[idle-dout] start=0 data_out=%h (must be 0)", data_out);
                $fatal(1, "idle cycle leaked load data");
            end
        end
    endtask

    //------------------------------------------------------------------
    // Stimulus
    //------------------------------------------------------------------
    reg [XLEN-1:0] rb, ridx, rd;
    reg [SEW-1:0]  rse;

    initial begin
        tests      = 0;
        errors     = 0;
        start      = 1'b0;
        is_scatter = 1'b0;
        base_addr  = 32'h0;
        index      = 32'h0;
        stride_exp = 2'd0;
        data_in    = 32'h0;
        seed       = 32'h5CA77E61;

        for (i = 0; i < DEPTH; i = i + 1) begin
            dmem[i] = {XLEN{1'b0}};
            gold[i] = {XLEN{1'b0}};
        end

        // ============================================================
        // 1. DIRECTED: effective-address math for every stride_exp with
        //    index 0 and index "max", plus a mid index.
        // ============================================================
        // index = 0  -> eff = base, for all stride_exp.
        for (s = 0; s < 4; s = s + 1) begin
            base_addr  = 32'd17;
            index      = 32'd0;
            stride_exp = s[SEW-1:0];
            check_addr;
        end
        // index = max 32-bit -> exercises XLEN wrap of the shift+add.
        for (s = 0; s < 4; s = s + 1) begin
            base_addr  = 32'd5;
            index      = 32'hFFFF_FFFF;        // index "max"
            stride_exp = s[SEW-1:0];
            check_addr;
        end
        // base = max, index small.
        for (s = 0; s < 4; s = s + 1) begin
            base_addr  = 32'hFFFF_FFFF;
            index      = 32'd3;
            stride_exp = s[SEW-1:0];
            check_addr;
        end
        // a handful of small ramps so the *(1<<se) scaling is exercised exactly.
        for (i = 0; i < 8; i = i + 1) begin
            for (s = 0; s < 4; s = s + 1) begin
                base_addr  = 32'd4;
                index      = i[XLEN-1:0];
                stride_exp = s[SEW-1:0];
                check_addr;
            end
        end

        // ============================================================
        // 2. DIRECTED round-trips: scatter then gather, varying stride_exp,
        //    at low addresses and at the index-0 corner.
        // ============================================================
        round_trip(32'd0,  32'd0, 2'd0, 32'h0000_0000);   // zero data
        round_trip(32'd0,  32'd0, 2'd2, 32'hFFFF_FFFF);   // all-ones data
        round_trip(32'd0,  32'd0, 2'd2, 32'h8000_0001);   // sign-ish corner
        round_trip(32'd10, 32'd5, 2'd0, 32'h0000_002A);   // stride 1
        round_trip(32'd10, 32'd5, 2'd1, 32'hABCD_0000);   // stride 2
        round_trip(32'd2,  32'd7, 2'd2, 32'h1357_9BDF);   // stride 4
        round_trip(32'd1,  32'd3, 2'd3, 32'h2468_ACE0);   // stride 8

        // index = "max" that still lands in DMEM after wrap (exercise wrap into
        // a low physical address while writing a distinctive payload).
        round_trip(32'd0, 32'hFFFF_FFFF, 2'd0, 32'hCAFE_BABE); // eff wraps to 255
        round_trip(32'd1, 32'hFFFF_FFFF, 2'd0, 32'h0BADF00D);  // eff wraps to 0

        // ============================================================
        // 3. DIRECTED: gather sees a value parked directly in golden+dmem
        //    (independent of any scatter) -> pure load path.
        // ============================================================
        gold[33] = 32'h1111_2222;  dmem[33] = 32'h1111_2222;
        do_gather(32'd33, 32'd0, 2'd0);                   // base hits addr 33
        gold[40] = 32'h3333_4444;  dmem[40] = 32'h3333_4444;
        do_gather(32'd32, 32'd2, 2'd2);                   // 32 + (2<<2)=40

        // ============================================================
        // 4. DIRECTED: idle (start=0) must not write and must flush data_out.
        // ============================================================
        check_idle(32'd7, 32'd9, 2'd1);
        check_idle(32'd0, 32'd0, 2'd0);

        // ============================================================
        // 5. DIRECTED: scatter must NOT disturb a neighbour word.
        // ============================================================
        do_scatter(32'd50, 32'd0, 2'd0, 32'hA1A1_A1A1);   // write addr 50
        do_scatter(32'd51, 32'd0, 2'd0, 32'hB2B2_B2B2);   // write addr 51
        do_gather (32'd50, 32'd0, 2'd0);                  // re-read 50 unchanged
        tests = tests + 1;
        if (data_out !== 32'hA1A1_A1A1) begin
            errors = errors + 1;
            $display("FAIL[neighbour] addr50 disturbed: got=%h", data_out);
            $fatal(1, "scatter disturbed neighbour word");
        end

        // ============================================================
        // 6. CONSTRAINED-RANDOM round-trips (seeded, >=200 vectors).
        //    Random base/index/stride/data; physical address derived from the
        //    golden multiply so the DUT's shift is independently checked, and
        //    every round-trip re-reads exactly what was written.
        // ============================================================
        for (i = 0; i < NRAND; i = i + 1) begin
            rb   = $random(seed);
            ridx = $random(seed);
            rd   = $random(seed);
            rse  = $random(seed);
            round_trip(rb, ridx, rse, rd);
        end

        // also a block of random PURE gathers against pre-seeded memory to keep
        // the load path exercised when no immediately-preceding scatter exists.
        for (i = 0; i < DEPTH; i = i + 1) begin
            rd      = $random(seed);
            dmem[i] = rd;
            gold[i] = rd;
        end
        for (i = 0; i < NRAND; i = i + 1) begin
            rb   = $random(seed);
            ridx = $random(seed);
            rse  = $random(seed);
            do_gather(rb, ridx, rse);
        end

        // ---- final tally ----
        if (errors != 0) begin
            $display("FAILED: %0d errors out of %0d checks", errors, tests);
            $fatal(1, "scatter_gather_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED", tests);
        $finish;
    end

    // safety net: never hang.
    initial begin
        #5_000_000;
        $fatal(1, "TIMEOUT: scatter_gather_tb did not finish");
    end

endmodule
