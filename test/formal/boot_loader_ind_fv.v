`timescale 1ns/1ps
//============================================================================
// boot_loader_ind_fv.v -- k-INDUCTION harness for src/boot_loader.v (READ-ONLY DUT)
//----------------------------------------------------------------------------
// GOAL: prove the "done-gate is a stable, quiescent, well-counted LEVEL"
// safety properties UNBOUNDED (for ALL reachable states), via
//   yosys-smtbmc -i  (temporal k-induction: base case + induction step).
//
// TOOLING CONSTRAINT (load-bearing): yosys 0.66 in the write_smt2 flow gives
// NO internal observability of the DUT -- a Verilog hierarchical reference such
// as `dut.wseg` is silently turned into a fresh implicit free wire (verified:
// an `assert(dut.cnt==cnt)` probe FAILS at step 1), and `bind` is dropped.
// Therefore every strengthening invariant here is expressed ONLY over the DUT's
// PRIMARY I/O plus harness-owned shadow registers driven by that same I/O.
//
// PROVEN UNBOUNDED (all asserts below are observably k-inductive):
//   P_STABLE : `done` is a STABLE LEVEL -- once high it stays high until a reset
//              or an honoured `start`.  (done & ~rst & ~(start&~busy)) |=> done.
//   P_EXCL   : busy and done are mutually exclusive: ~(busy & done).  The gate is
//              a clean level -- the engine is never "still copying" and "released".
//   P_DGATE_W: done => ~ddr_we     (no DDR5 write strobe once released)
//   P_DGATE_R: done => ~flash_req  (no Flash read request once released)
//              -> the engine is fully QUIESCENT behind a raised done.
//   P_SHEQ   : words_done == (count of observable write retirements ddr_we&ddr_ready
//              since the last reset/honoured-start).  The progress counter is
//              EXACTLY the number of words pushed into DDR5 -- it never miscounts.
//
// NOT provable unbounded in THIS build (kept BOUNDED in boot_loader_fv.v):
//   B_TOTAL  : done => words_done == sum-of-active-segment-lengths.  The induction
//              step admits a spurious state (done=1 with words_done != total) that
//              only the INTERNAL write-cursor/length-sum progress invariant
//              words_done == SUM_{k<wseg} len_q[k] + woff  (plus wseg<=ncount<=SEG_MAX,
//              occ==issued-written, fcnt==returned-written, occ<=BURST) would
//              exclude -- and those reference DUT-internal registers this flow
//              cannot name.  See the RETURN note / docs.  Stays BMC-bounded.
//
// Params identical to boot_loader_fv.v so both harnesses cover the same instance.
//============================================================================
module boot_loader_ind_fv (
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
    // ---- SMALL params (mirror boot_loader_fv.v) ----
    localparam integer ADDR_W  = 4;
    localparam integer DATA_W  = 4;
    localparam integer SEG_MAX = 2;
    localparam integer BURST   = 2;
    localparam integer LEN_W   = 2;
    localparam integer SEGW    = (SEG_MAX < 2) ? 1 : $clog2(SEG_MAX + 1);
    localparam integer PROG_W  = LEN_W + SEGW;
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
        .ADDR_W (ADDR_W), .DATA_W (DATA_W), .SEG_MAX(SEG_MAX),
        .BURST  (BURST),  .LEN_W  (LEN_W)
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
    // ENVIRONMENT ASSUMPTION (legal descriptor; the DUT's documented contract).
    // NOTE: induction does NOT force reset -- the step starts from an ARBITRARY
    // legal state, so there is intentionally no "assume rst at t=0" here.  The
    // BASE case of k-induction still needs reachability rooted at reset; smtbmc
    // gets that from the reset logic + this legal-input assume.  We additionally
    // assume reset at t=0 only to keep the base case well-formed.
    // -----------------------------------------------------------------------
    reg first = 1'b1;
    always @(posedge clk) begin
        if (first) assume (rst);          // base-case reset root (t=0 only)
        assume (seg_count <= SEGMAXC);    // legal segment count, every cycle
        first <= 1'b0;
    end

    // observable per-cycle write retirement (== DUT's internal write_fire)
    wire write_fire = ddr_we & ddr_ready;
    // observable "honoured start" / reset of the progress counter
    wire clear_prog = rst | (start & ~busy);

    // -----------------------------------------------------------------------
    // STRENGTHENING SHADOW: count write retirements from OBSERVABLE I/O exactly
    // as the DUT counts words_done.  Identical update + identical clear => the
    // equality P_SHEQ is 1-inductive (both registers move by the same observable
    // delta each cycle), and it pins words_done to a concrete observable meaning.
    // -----------------------------------------------------------------------
    reg [PROG_W-1:0] shadow_words = {PROG_W{1'b0}};
    always @(posedge clk) begin
        if (clear_prog)        shadow_words <= {PROG_W{1'b0}};
        else if (write_fire)   shadow_words <= shadow_words
                                            + {{(PROG_W-1){1'b0}}, 1'b1};
    end

    // -----------------------------------------------------------------------
    // pv : "this is not the standalone pre-reset (t=0) state".  pv has init 0 and
    // an UNCONDITIONAL `pv<=1`, so its next-state is constant 1.  In k-induction
    // the successor state checked by the step therefore ALWAYS has pv=1 -> the
    // gating is SOUND (it never lets the step skip a real successor); only the
    // single free t=0 garbage state (synchronous reset cleans it at t=1) is
    // excluded -- exactly the state that is not a reachable operating state.
    // p_hold : registered antecedent for the next-cycle (|=>) stability check.
    // -----------------------------------------------------------------------
    reg pv     = 1'b0;
    reg p_hold = 1'b0;
    always @(posedge clk) begin
        pv     <= 1'b1;
        p_hold <= (done & ~rst & ~(start & ~busy));
    end

    // -----------------------------------------------------------------------
    // SAFETY PROPERTIES (all observably k-inductive)
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        // done is a stable level (antecedent p_hold is 0 at/after t=0 reset)
        if (pv && p_hold) P_STABLE : assert (done);

        if (pv) begin
            // done is a clean, exclusive, quiescent level
            P_EXCL    : assert (~(busy & done));
            P_DGATE_W : assert (~(done & ddr_we));
            P_DGATE_R : assert (~(done & flash_req));
            // progress counter == observable words retired into DDR5
            P_SHEQ    : assert (words_done == shadow_words);
        end
    end
endmodule
