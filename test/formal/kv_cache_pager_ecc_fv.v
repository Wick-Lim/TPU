`timescale 1ns/1ps
//============================================================================
// kv_cache_pager_ecc_fv.v -- BMC formal harness for src/kv_cache_pager.v with
//                            ECC=1 (lane-partitioned SECDED ring)   READ-ONLY DUT
//----------------------------------------------------------------------------
// The committed kv_cache_pager_fv.v proves the DEFAULT (ECC=0) datapath: the
// ring stores RAW rows, and a resident gather returns row_out == the appended
// row.  This harness proves the SAME correctness survives the ECC=1 lane-SECDED
// wrapping, whose ring instead stores CODEWORDS: on append each ROW_BITS row is
// partitioned into NLANES = ceil(ROW_BITS/64) 64-bit lanes, each ENCODED by an
// ecc_secded #(64) into a CODE_W(=72)-bit codeword and packed into the ring;
// on a resident gather each lane codeword is DECODED+CORRECTED back to the row.
//
// CRUX vs the ECC=0 harness: the ring no longer holds the row -- it holds the
// encode() of the row.  So the shadow compare here asserts the DUT's DECODED
// gather output row_out equals the appended row, i.e. it forces the solver to
// discharge the ENCODE-THEN-DECODE == IDENTITY lemma of the (72,64) SECDED
// codec (a purely combinational, XOR-only linear identity that z3 bit-blasts).
// We do NOT model fault injection here (that is exhaustively covered by
// test/kv_cache_pager_ecc_tb.v); we prove the NO-FAULT datapath, which also
// means the sticky ecc_serr / ecc_derr flags must NEVER rise (no false alarm).
//
// PROPERTIES:
//   (A) IN-BOUNDS / WINDOW  (identical to the ECC=0 harness; outputs only)
//        - resident_lo <= append_count                 (window never inverts)
//        - (append_count - resident_lo) <= RESIDENT     (window <= ring cap)
//        - on a resident gather the ring read slot is < RESIDENT
//        - while a cold fetch is held, flash_idx < append_count
//   (B) RESIDENT GATHER CORRECT THROUGH ECC   <-- the ECC crux
//        - an accepted resident gather drives row_valid next cycle and
//          row_out == the appended row  (== decode(encode(appended row)))
//   (C) NO FALSE ECC ALARM
//        - the sticky ecc_serr / ecc_derr stay 0 forever on the no-fault path
//   (D) COLD GATHER CORRECT (bonus; cold path bypasses the codec)
//        - a cold fetch returns row_out == the row appended at that position.
//
// INSTANCE = the committed ECC=0 harness's sizes (ROW_BITS=4, RESIDENT=4,
// S_MAX=8, FLASH_LAT=2) but with ECC=1 -- i.e. literally "the committed pager
// proof, now with the SECDED ring."  ROW_BITS=4 -> NLANES=1: a single SECDED
// lane whose 4 payload bits are ZERO-PADDED to 64 before encode and whose low 4
// bits are extracted after decode, so the RAGGED-lane encode/decode + pad-drop
// path IS exercised (the non-trivial lane shape).  BMC PROVES this single-lane
// ECC ring datapath at K=16 (and K=12) with z3.
//
// WHY SINGLE-LANE (NLANES=1) and not the multi-lane real row: with ROW_BITS>64
// each ring element becomes NLANES*72 bits of CODEWORD stored in a flop array,
// and the resident-gather read `decode(ring[slot])` forces z3 to reason about a
// symbolic-address mux over multi-hundred-bit codewords fed into the codec --
// this array reasoning bit-blasts badly and z3 does NOT return on the crux step
// (empirically: a 2-lane ROW_BITS=68 build with even a 2-slot ring hangs on the
// first resident-gather step; the standalone 64-bit decode(encode)==identity is
// 0.14 s, so the cost is the memory indirection, not the codec).  Multi-lane is
// N INDEPENDENT structural copies of the SAME proven single-lane identity (fixed
// bit-slice pack/unpack, zero cross-lane logic), and the full ROW_BITS=100 /768
// multi-lane geometry + fault correction is exhaustively covered by
// test/kv_cache_pager_ecc_tb.v.  See the report / docs/FORMAL.md.
//============================================================================
module kv_cache_pager_ecc_fv #(
    parameter integer ROW_BITS  = 4,     // -> NLANES=1 (single ragged SECDED lane)
    parameter integer RESIDENT  = 4,     // power of two
    parameter integer S_MAX     = 8,
    parameter integer FLASH_LAT = 2,
    parameter integer POSW      = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer RPTRW     = (RESIDENT <= 1) ? 1 : $clog2(RESIDENT)
)(
    input  wire                 clk,
    // free formal stimulus (constrained below)
    input  wire                 rst,
    input  wire                 append_valid,
    input  wire [ROW_BITS-1:0]  append_row,
    input  wire                 gather_valid,
    input  wire [POSW-1:0]      gather_idx,
    input  wire                 flash_done
);
    //------------------------------------------------------------------------
    // DUT outputs
    //------------------------------------------------------------------------
    wire                 row_valid;
    wire [ROW_BITS-1:0]  row_out;
    wire                 busy;
    wire                 flash_req;
    wire [POSW-1:0]      flash_idx;
    wire [POSW-1:0]      append_count;
    wire [POSW-1:0]      resident_lo;
    wire                 overflowed;
    wire                 ecc_serr;      // sticky: a resident gather CORRECTED an SBU
    wire                 ecc_derr;      // sticky: a resident gather DETECTED a DBU

    // Flash backing store modelled by the SAME shadow content the appends saw,
    // so a correct cold fetch returns exactly the appended row.
    wire [ROW_BITS-1:0]  flash_row;

    //------------------------------------------------------------------------
    // DUT with ECC=1 : the ring stores lane-SECDED codewords.
    //   The parameter override .ECC(1) makes chparam unnecessary; the build
    //   still needs src/ecc_secded.v in the read_verilog list.
    //------------------------------------------------------------------------
    kv_cache_pager #(
        .ROW_BITS (ROW_BITS),
        .RESIDENT (RESIDENT),
        .S_MAX    (S_MAX),
        .FLASH_LAT(FLASH_LAT),
        .ECC      (1)
    ) dut (
        .clk          (clk),
        .rst          (rst),
        .append_valid (append_valid),
        .append_row   (append_row),
        .gather_valid (gather_valid),
        .gather_idx   (gather_idx),
        .row_valid    (row_valid),
        .row_out      (row_out),
        .busy         (busy),
        .flash_req    (flash_req),
        .flash_idx    (flash_idx),
        .flash_done   (flash_done),
        .flash_row    (flash_row),
        .append_count (append_count),
        .resident_lo  (resident_lo),
        .overflowed   (overflowed),
        .ecc_serr     (ecc_serr),
        .ecc_derr     (ecc_derr)
    );

    //------------------------------------------------------------------------
    // Free-running step counter -> forces reset at t==0 only.
    //------------------------------------------------------------------------
    reg [15:0] t = 16'd0;
    always @(posedge clk) t <= t + 16'd1;

    //------------------------------------------------------------------------
    // PROTOCOL ASSUMPTIONS (legal stimulus) -- identical to the ECC=0 harness.
    //------------------------------------------------------------------------
    always @* begin
        // reset asserted exactly on the first cycle, deasserted afterwards.
        assume (rst == (t == 16'd0));

        // legal gather: only request a position that has actually been
        // appended (gather_idx < append_count).  Also forbids gather when
        // append_count == 0.
        if (gather_valid)
            assume (gather_idx < append_count);

        // never append past the (small) shadow capacity in the bounded window.
        if (append_valid)
            assume (append_count < (S_MAX[POSW-1:0] - 1'b1));
    end

    //------------------------------------------------------------------------
    // REFERENCE SHADOW : row appended at each logical position (the RAW row,
    // NOT a codeword).  The DUT independently stores encode(append_row); the
    // assert forces decode(that) == this shadow, i.e. the ECC identity.
    //------------------------------------------------------------------------
    reg [ROW_BITS-1:0] shadow [0:S_MAX-1];
    always @(posedge clk) begin
        if (!rst && append_valid)
            shadow[append_count] <= append_row;
    end

    // Flash backing returns the row stored at the requested cold position.
    assign flash_row = shadow[flash_idx];

    //------------------------------------------------------------------------
    // residency / acceptance decode (mirrors the DUT, computed from outputs).
    //------------------------------------------------------------------------
    wire g_resident = (gather_idx < append_count) && (gather_idx >= resident_lo);
    wire acc        = gather_valid && !busy && !rst;
    wire acc_res    = acc &&  g_resident;
    wire acc_cold   = acc && !g_resident;
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused_acc_cold = acc_cold;       // documentation handle
    wire _unused_of       = overflowed;
    wire _unused_fr       = flash_req || (|append_row);
    /* verilator lint_on UNUSEDSIGNAL */

    //------------------------------------------------------------------------
    // (B) RESIDENT GATHER CORRECT (through ECC) -- registered expectation,
    //     checked next cycle against the DUT's DECODED row_out.
    //------------------------------------------------------------------------
    reg                res_v;
    reg [ROW_BITS-1:0] res_row;
    always @(posedge clk) begin
        if (rst) begin
            res_v   <= 1'b0;
            res_row <= {ROW_BITS{1'b0}};
        end else begin
            res_v   <= acc_res;
            res_row <= shadow[gather_idx];
        end
    end

    //------------------------------------------------------------------------
    // (D) COLD GATHER CORRECT -- busy(=G_FLASH) & flash_done => deliver next.
    //------------------------------------------------------------------------
    reg                cold_v;
    reg [ROW_BITS-1:0] cold_row;
    always @(posedge clk) begin
        if (rst) begin
            cold_v   <= 1'b0;
            cold_row <= {ROW_BITS{1'b0}};
        end else begin
            cold_v   <= (busy && flash_done);
            cold_row <= flash_row;            // == shadow[flash_idx]
        end
    end

    //------------------------------------------------------------------------
    // ASSERTIONS
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst) begin
            // (A) in-bounds / window invariants
            a_lo  : assert (resident_lo <= append_count);
            a_win : assert ((append_count - resident_lo) <= RESIDENT[POSW-1:0]);
            if (acc_res)
                a_slot : assert (gather_idx[RPTRW-1:0] < RESIDENT[RPTRW:0]);
            if (flash_req)
                a_fidx : assert (flash_idx < append_count);

            // (B) resident gather correctness THROUGH the SECDED codec:
            //     row_out == decode(encode(appended row)) == appended row.
            if (res_v) begin
                a_res_valid : assert (row_valid);
                a_res_row   : assert (row_out == res_row);
            end

            // (C) NO FALSE ECC ALARM on the no-fault datapath: the sticky
            //     serr/derr flags must never rise (encode-decode leaves the
            //     syndrome clean, so no lane ever flags a correction/detect).
            a_no_serr : assert (!ecc_serr);
            a_no_derr : assert (!ecc_derr);

            // (D) cold gather correctness (cold path bypasses the codec)
            if (cold_v) begin
                a_cold_valid : assert (row_valid);
                a_cold_row   : assert (row_out == cold_row);
            end
        end
    end

    //------------------------------------------------------------------------
    // NON-VACUITY / REACHABILITY PROBES : each is a DELIBERATELY FALSE claim.
    // BMC returning a counterexample for it proves the corresponding state IS
    // reachable under our assumptions (so the real asserts above are not
    // vacuously true).  Selected one at a time with -D<NAME>; compiled OUT of
    // the default PASS run.  (The Makefile's assert-count>0 guard already bars
    // the trivial 0-assert vacuity; these confirm the crux path is exercised.)
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst) begin
`ifdef PROBE_RESV
            // expect FAIL: a resident gather DOES complete (res_v reachable),
            // so a_res_row (the encode-decode identity check) really fires.
            p_resv : assert (!res_v);
`endif
`ifdef PROBE_COLD
            // expect FAIL: a cold fetch DOES complete (cold_v reachable).
            p_cold : assert (!cold_v);
`endif
`ifdef PROBE_ROWNZ
            // expect FAIL: a decoded resident row_out CAN be non-zero, i.e.
            // real payload flows through encode->store->decode intact.
            if (res_v) p_rownz : assert (row_out == {ROW_BITS{1'b0}});
`endif
        end
    end

endmodule
