`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// mbist_ctrl.v  --  March C- memory BIST controller for a single-port SRAM
//                                                                (DFT / P2.3)
//----------------------------------------------------------------------------
// ROLE
//   Design-For-Test engine that exercises an on-die single-port SRAM (a weight
//   / KV-cache / scratch RAM) with the industry-standard March C- algorithm and
//   flags any cell that fails to store the value written to it (stuck-at, plus
//   many transition and coupling faults).  It DRIVES the RAM's address / write-
//   enable / write-data ports and OBSERVES its read-data port; the RAM itself
//   lives OUTSIDE this module (real macro in silicon; a behavioral model in the
//   unit TB), exactly like boot_loader.v / dma_controller.v drive external
//   memory ports rather than owning storage.
//
// MARCH C-  (10N accesses; N = DEPTH cells)
//   A "March element" is an address sweep (UP = 0..DEPTH-1, or DOWN =
//   DEPTH-1..0) applying a fixed sequence of read/write OPERATIONS to EVERY
//   cell before moving to the next address.  March C- is six elements:
//
//        M0  up   : w0                 (init every cell to 0)
//        M1  up   : r0, w1             (expect 0, then write 1)
//        M2  up   : r1, w0             (expect 1, then write 0)
//        M3  down : r0, w1             (expect 0, then write 1)
//        M4  down : r1, w0             (expect 1, then write 0)
//        M5  down : r0                 (expect 0)
//
//   "w0"/"w1" write the all-zeros / all-ones WIDTH-bit word; "r0"/"r1" read and
//   compare against all-zeros / all-ones.  A read that does not return the
//   expected word is a FAIL at that address.
//
// SINGLE-PORT TIMING  (synchronous write, combinational read RAM)
//   The controller drives ONE access per cycle.  The RAM has a COMBINATIONAL
//   read port (rdata = RAM[addr]) and a synchronous write -- the project's
//   memory.v contract (assign data_out = sram[addr]; write on the clock edge).
//   Because the controller's `addr` is a REGISTERED output, an address driven
//   at edge T is stable throughout cycle T+1, so RAM[addr] (hence rdata) is
//   valid during cycle T+1.  A read OPERATION therefore spans, from the point
//   the controller DRIVES the address:
//        cycle T   : drive addr = A, we = 0            (issue read of cell A)
//        cycle T+1 : rdata == RAM[A]  -> COMPARE       (and issue the next op)
//   A write OPERATION is single-cycle: drive addr = A, we = 1, wdata = pattern;
//   it retires on the clock edge (the NEXT march element reads it back).  The
//   controller keeps, alongside the op it is DRIVING this cycle, a one-deep
//   "pending read" record (valid, expected pattern, address) captured whenever
//   it drove a read; on the FOLLOWING cycle it compares rdata against that
//   record.  This 1-deep read pipeline lets a new access issue every cycle.
//
// OUTPUTS  (ALL registered -- the RAM-facing strobes too, so the RAM sees clean
//   glitch-free accesses)
//   busy       : high from `start` until the march completes (or a fail stops).
//   done       : registered LEVEL; rises when the march finishes (pass OR fail)
//                and stays high until the next `start`/reset.  NOT a pulse.
//   fail       : registered LEVEL; set on the FIRST mismatch and latched until
//                the next `start`/reset (0 == RAM good).
//   fail_addr  : address of that first failing read (meaningful when fail == 1).
//   addr,we,wdata : single-port SRAM access driven this cycle.
//   rdata         : RAM read data (combinational; valid the cycle after addr).
//
//   On the FIRST mismatch the engine latches fail + fail_addr and STOPS (raises
//   done): a March run aborts at the first defect (results past a fault are not
//   meaningful).  A fault-free RAM runs all 10N operations and finishes fail==0.
//
// CONVENTIONS  (match memory.v / dma_controller.v / boot_loader.v)
//   Synchronous ACTIVE-HIGH reset clears ALL state; every output is registered
//   and assigned on every path (NO inferred latch); NO combinational loop (all
//   outputs derive from registered state only); fully parameterized.
//============================================================================
module mbist_ctrl #(
    parameter integer DEPTH = 16,   // # SRAM cells (addresses)
    parameter integer WIDTH = 8,    // bits per SRAM word
    // ---- derived geometry ----
    localparam integer AW = (DEPTH < 2) ? 1 : $clog2(DEPTH)  // address width
) (
    input  wire              clk,
    input  wire              rst,        // sync, active-high (ALL state)

    // ---- command / status ----
    input  wire              start,      // 1-cycle pulse: begin a march run
    output reg               busy,       // engine is running the march
    output reg               done,       // LEVEL: march finished (pass or fail)
    output reg               fail,       // LEVEL: a cell failed (latched)
    output reg  [AW-1:0]     fail_addr,  // address of the first failing read

    // ---- single-port SRAM access ports the engine DRIVES ----
    output reg  [AW-1:0]     addr,       // access address
    output reg               we,         // write strobe (1 == write wdata)
    output reg  [WIDTH-1:0]  wdata,      // write data (all-0 or all-1 pattern)
    input  wire [WIDTH-1:0]  rdata       // RAM read data (combinational; 1-cyc after addr)
);
    // -----------------------------------------------------------------------
    // The march is flattened into a linear PROGRAM of operations.  Each op is
    // { direction, is_read, pattern }.  There are 10 ops (M0:1 + M1:2 + M2:2 +
    // M3:2 + M4:2 + M5:1); each is applied to all DEPTH addresses in its
    // element's direction.  The engine steps (opi, cnt) through the program.
    //   Program (opi : element : operation):
    //     0 : M0 up   : w0
    //     1 : M1 up   : r0    2 : M1 up   : w1
    //     3 : M2 up   : r1    4 : M2 up   : w0
    //     5 : M3 down : r0    6 : M3 down : w1
    //     7 : M4 down : r1    8 : M4 down : w0
    //     9 : M5 down : r0
    // -----------------------------------------------------------------------
    localparam integer NOPS = 10;                   // ops in the March C- program
    localparam integer OPW  = (NOPS < 2) ? 1 : $clog2(NOPS);

    // Constant program table, encoded as three packed vectors so the whole
    // table is a set of parameters (no `initial`, no inferred state) and any op
    // field is a simple bit-select of `opi`.  Bit i (LSB = op0) describes op i:
    //   DIR_UP[i] : op i sweeps addresses UP (else DOWN)
    //   IS_RD [i] : op i is a read+compare (else a write)
    //   PAT1  [i] : op i uses the all-ones pattern (else all-zeros)
    // Derivation vs. the program list above (opi in 0..9):
    //   UP  ops   : 0,1,2,3,4        -> bits 0..4 = 1     -> 10'b00_0001_1111
    //   READ ops  : 1,3,5,7,9        -> bits 1,3,5,7,9 =1 -> 10'b10_1010_1010
    //   PAT1 ops  : 2,3,6,7 (r1/w1)  -> bits 2,3,6,7   =1 -> 10'b00_1100_1100
    localparam [NOPS-1:0] DIR_UP = 10'b00_0001_1111;
    localparam [NOPS-1:0] IS_RD  = 10'b10_1010_1010;
    localparam [NOPS-1:0] PAT1   = 10'b00_1100_1100;

    // all-ones / all-zeros WIDTH-bit words
    localparam [WIDTH-1:0] WORD0 = {WIDTH{1'b0}};
    localparam [WIDTH-1:0] WORD1 = {WIDTH{1'b1}};

    // sized constants for the address counter (AW+1 wide so the sweep terminal
    // compare against DEPTH never truncates, even at DEPTH == 2^AW).
    localparam integer  CW      = AW + 1;
    localparam [CW-1:0] C_ONE   = { {(CW-1){1'b0}}, 1'b1 };
    localparam [CW-1:0] C_ZERO  = { CW{1'b0} };
    localparam [CW-1:0] DEPTHC  = CW'(DEPTH);
    localparam [CW-1:0] DEPTHM1 = DEPTHC - C_ONE;   // DEPTH-1

    // -----------------------------------------------------------------------
    // run state
    //   opi : current op index (0..NOPS); opi == NOPS means the program is done.
    //   cnt : # addresses ALREADY issued in the current op (0..DEPTH-1 sweep).
    //   run : the march is actively issuing accesses.
    // -----------------------------------------------------------------------
    reg [OPW:0]  opi;      // one extra bit so opi == NOPS (done) is representable
    reg [CW-1:0] cnt;
    reg          run;

    // one-deep pending-read record (a read issued LAST cycle, compared NOW)
    reg          rd_pend;   // a read comparison is due this cycle
    reg          rd_pat1;   // expected pattern of that pending read (1 = all-ones)
    reg [AW-1:0] rd_addr;   // address of that pending read (for fail_addr)

    // -----------------------------------------------------------------------
    // combinational view of the op the engine is ABOUT to drive this cycle
    // (pure functions of registered state -> no latch, no comb loop)
    // -----------------------------------------------------------------------
    wire         active   = run & (opi < OPW'(NOPS));         // program not done
    wire [OPW-1:0] oidx   = opi[OPW-1:0];
    wire         cur_up   = active ? DIR_UP[oidx] : 1'b0;
    wire         cur_rd   = active ? IS_RD [oidx] : 1'b0;
    wire         cur_pat1 = active ? PAT1  [oidx] : 1'b0;
    // physical address for the cnt-th access of this op:
    //   UP   : cnt            (0,1,...,DEPTH-1)
    //   DOWN : DEPTH-1 - cnt  (DEPTH-1,...,0)
    wire [CW-1:0] addr_c  = cur_up ? cnt : (DEPTHM1 - cnt);
    wire          has_addr = active & (cnt < DEPTHC);         // an access to issue

    // the pending read (if any) MISMATCHES the expected pattern this cycle
    wire          mismatch = rd_pend &
                             (rdata !== (rd_pat1 ? WORD1 : WORD0));

    // -----------------------------------------------------------------------
    // single synchronous control process (active-high reset; no latch/comb loop)
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            busy      <= 1'b0;
            done      <= 1'b0;
            fail      <= 1'b0;
            fail_addr <= {AW{1'b0}};
            addr      <= {AW{1'b0}};
            we        <= 1'b0;
            wdata     <= WORD0;
            opi       <= {(OPW+1){1'b0}};
            cnt       <= C_ZERO;
            run       <= 1'b0;
            rd_pend   <= 1'b0;
            rd_pat1   <= 1'b0;
            rd_addr   <= {AW{1'b0}};
        end else if (start & ~busy) begin
            // ---- begin a fresh march run ----
            busy      <= 1'b1;
            done      <= 1'b0;
            fail      <= 1'b0;
            fail_addr <= {AW{1'b0}};
            addr      <= {AW{1'b0}};
            we        <= 1'b0;
            wdata     <= WORD0;
            opi       <= {(OPW+1){1'b0}};   // start at op 0
            cnt       <= C_ZERO;
            run       <= 1'b1;
            rd_pend   <= 1'b0;              // no comparison due yet
            rd_pat1   <= 1'b0;
            rd_addr   <= {AW{1'b0}};
        end else if (busy) begin
            // Default RAM strobe low; overwritten below if this cycle issues a
            // write.  (Assigned on every path -> no inferred latch.)
            we <= 1'b0;

            if (mismatch) begin
                //============================================================
                // FIRST failure : latch fail + fail_addr and ABORT the run.
                //============================================================
                fail      <= 1'b1;
                fail_addr <= rd_addr;
                run       <= 1'b0;
                busy      <= 1'b0;
                done      <= 1'b1;
                rd_pend   <= 1'b0;
                // we already defaulted low (no access issued on the abort cycle)
            end else if (has_addr) begin
                //============================================================
                // ISSUE the next op.  Read: we=0, record pending compare.
                //                     Write: we=1, drive the pattern.
                //============================================================
                addr    <= addr_c[AW-1:0];
                we      <= ~cur_rd;                 // write when NOT a read
                wdata   <= cur_pat1 ? WORD1 : WORD0;
                // record a pending read comparison for next cycle
                rd_pend <= cur_rd;
                rd_pat1 <= cur_pat1;
                rd_addr <= addr_c[AW-1:0];

                // advance the address sweep / op program
                if (cnt == DEPTHM1) begin
                    cnt <= C_ZERO;
                    opi <= opi + {{OPW{1'b0}}, 1'b1};   // next op
                end else begin
                    cnt <= cnt + C_ONE;
                end
            end else begin
                //============================================================
                // No access left to issue (program exhausted).  The final read
                // (if there was a pending one) was compared THIS cycle without
                // mismatch; clear it and finish -- the march passed.
                //============================================================
                rd_pend <= 1'b0;
                run     <= 1'b0;
                busy    <= 1'b0;
                done    <= 1'b1;
            end
        end
        // when idle (!busy): busy/done/fail/fail_addr hold their level; the
        // RAM strobe stays deasserted (we was set 0 on the cycle we went idle).
    end

endmodule
/* verilator lint_on DECLFILENAME */
