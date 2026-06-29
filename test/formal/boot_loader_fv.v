`timescale 1ns/1ps
//============================================================================
// boot_loader_fv.v -- formal harness (BMC) for src/boot_loader.v  (READ-ONLY DUT)
//----------------------------------------------------------------------------
// Properties proven (bounded model checking):
//   A_STABLE : `done` is a STABLE LEVEL.  Once high it cannot fall except on a
//              reset or on a fresh `start` (which is only honoured while ~busy).
//              Expressed: (done & ~rst & ~(start & ~busy)) |=> done.
//   B_TOTAL  : `done` only rises AFTER all words are copied: whenever done is
//              high, words_done == (sum of active-segment lengths latched at the
//              last start).  i.e. done can NEVER assert early.
//
// All DUT inputs are FREE formal signals.  We add only:
//   * reset asserted at t=0 (assume rst when cyc==0),
//   * legal descriptor: seg_count <= SEG_MAX (the module's documented contract),
//   * a shadow `exp_total` latched on the SAME (start & ~busy & ~rst) edge the
//     DUT latches its descriptor, computed from the SAME seg_len/seg_count.
// Inputs flash_ready/flash_rvalid/flash_rdata/ddr_ready are left fully free
// (no liveness assumed -- these are pure SAFETY properties; if the environment
// never completes, done never rises and both asserts hold vacuously).
//
// Params kept SMALL so BMC is tractable (tiny FIFO, addr, data, lengths).
//============================================================================
module boot_loader_fv (
    input wire clk,

    // ---- free formal stimulus (DUT inputs) ----
    input wire                         rst,
    input wire                         start,
    input wire [SEGW-1:0]              seg_count,
    input wire [SEG_MAX*ADDR_W-1:0]    seg_flash_base,
    input wire [SEG_MAX*ADDR_W-1:0]    seg_ddr_base,
    input wire [SEG_MAX*LEN_W-1:0]     seg_len,
    input wire                         flash_ready,
    input wire                         flash_rvalid,
    input wire [DATA_W-1:0]            flash_rdata,
    input wire                         ddr_ready
);
    // ---- SMALL params (tractable BMC) ----
    localparam integer ADDR_W  = 4;
    localparam integer DATA_W  = 4;
    localparam integer SEG_MAX = 2;
    localparam integer BURST   = 2;
    localparam integer LEN_W   = 2;
    // ---- derived geometry (mirror the DUT) ----
    localparam integer SEGW   = (SEG_MAX < 2) ? 1 : $clog2(SEG_MAX + 1);
    localparam integer PROG_W = LEN_W + SEGW;
    localparam [SEGW-1:0] SEGMAXC = SEG_MAX[SEGW-1:0];

    // ---- DUT outputs ----
    wire                  flash_req;
    wire [ADDR_W-1:0]     flash_addr;
    wire                  ddr_we;
    wire [ADDR_W-1:0]     ddr_addr;
    wire [DATA_W-1:0]     ddr_wdata;
    wire                  busy;
    wire                  done;
    wire [PROG_W-1:0]     words_done;

    // ---- DUT (committed, READ-ONLY) ----
    boot_loader #(
        .ADDR_W (ADDR_W),
        .DATA_W (DATA_W),
        .SEG_MAX(SEG_MAX),
        .BURST  (BURST),
        .LEN_W  (LEN_W)
    ) dut (
        .clk(clk), .rst(rst),
        .start(start), .seg_count(seg_count),
        .seg_flash_base(seg_flash_base),
        .seg_ddr_base(seg_ddr_base),
        .seg_len(seg_len),
        .flash_req(flash_req), .flash_addr(flash_addr),
        .flash_ready(flash_ready), .flash_rvalid(flash_rvalid),
        .flash_rdata(flash_rdata),
        .ddr_we(ddr_we), .ddr_addr(ddr_addr), .ddr_wdata(ddr_wdata),
        .ddr_ready(ddr_ready),
        .busy(busy), .done(done), .words_done(words_done)
    );

    // -----------------------------------------------------------------------
    // t=0 reset + legal-descriptor assumptions
    // -----------------------------------------------------------------------
    reg [3:0] cyc = 4'd0;
    always @(posedge clk) begin
        // reset asserted at t=0; legal descriptor every cycle
        if (cyc == 4'd0) assume (rst);
        assume (seg_count <= SEGMAXC);
        cyc <= (cyc == 4'hf) ? cyc : (cyc + 4'd1);
    end

    // -----------------------------------------------------------------------
    // shadow expected total = sum over active segments of latched length
    // -----------------------------------------------------------------------
    reg  [PROG_W-1:0] total_in;
    integer k;
    always @* begin
        total_in = {PROG_W{1'b0}};
        for (k = 0; k < SEG_MAX; k = k + 1)
            if (k < seg_count)
                total_in = total_in + seg_len[k*LEN_W +: LEN_W];
    end

    reg [PROG_W-1:0] exp_total = {PROG_W{1'b0}};
    always @(posedge clk) begin
        if (rst)                  exp_total <= {PROG_W{1'b0}};
        else if (start & ~busy)   exp_total <= total_in;   // mirrors DUT latch
    end

    // -----------------------------------------------------------------------
    // SAFETY PROPERTIES (immediate asserts; A uses a registered antecedent for
    // the next-cycle |=> obligation)
    // -----------------------------------------------------------------------
    reg pv      = 1'b0;   // "a previous cycle exists"
    reg p_hold  = 1'b0;   // antecedent of A, registered from previous cycle
    always @(posedge clk) begin
        // A: done is a STABLE LEVEL -- if last cycle done held & no reset & no
        //    honoured start, then done must still be high now.
        if (pv && p_hold)
            A_STABLE : assert (done);

        // B: done never asserts EARLY -- whenever done is high, every word of
        //    every active segment is already retired (words_done == total).
        //    Gated by `pv` to skip the single pre-reset (t=0) garbage state:
        //    reset is SYNCHRONOUS, so the forced rst@t=0 only cleans state at
        //    t=1; from t>=1 every state is reset-rooted and legitimate.
        if (pv && done)
            B_TOTAL  : assert (words_done == exp_total);

        p_hold <= (done & ~rst & ~(start & ~busy));
        pv     <= 1'b1;
    end

endmodule
