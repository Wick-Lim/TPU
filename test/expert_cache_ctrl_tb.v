`timescale 1ns/1ps
//============================================================================
// expert_cache_ctrl_tb.v -- self-checking TB for expert_cache_ctrl
//                           (GLM-5.2-FP8 MoE expert-weight HBM cache controller)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN (no DUT logic shared)
//   A behavioral fully-associative LRU cache mirrors the DUT policy:
//     - resident set : ref_valid[slot] + ref_tag[slot]
//     - EXACT LRU     : ref_ord[] is a MRU..LRU recency STACK of slot indices
//                       (move-to-front on every touch).  This is the textbook
//                       form of the DUT's rank permutation, derived here
//                       INDEPENDENTLY from the request stream.
//   For each request the golden predicts hit/miss, the answered SLOT, and the
//   running hit/miss counters, then checks them BIT-EXACT against the DUT.
//   Victim policy: lowest-index INVALID slot first, else the LRU (ref_ord tail).
//   X-AWARE: any X on resp_valid/hit/resp_slot/busy/counts at a check -> FAIL.
//
// PART A -- CORRECTNESS (directed + random), pass/fail gated:
//   * all-cold-miss fill of every empty slot
//   * all-hit repeat of a resident id (single-id hammer)
//   * classic LRU eviction order (SLOTS=4: access A B C D, re-access A,
//       insert E -> B must be evicted, NOT A) + B then misses again
//   * thrash: more distinct ids than slots, repeatedly
//   * constrained-random stream over a small id space
//   Two harness width shapes (4/8 and 8/64) plus 2/64 and 16/64 in the sweep
//   exercise the SLOT_W/ID_W derivation (incl. the degenerate SLOTS=2 width).
//
// PART B -- HIT-RATE MEASUREMENT (system value, measurement not a gate):
//   A longer SKEWED (Zipf-like: min-of-k) trace with temporal locality is run
//   through the DUT; per-request hit/miss is STILL checked against the golden.
//   The measured DUT hit_count/miss_count is reported, and SLOTS is swept
//   (2,4,8,16 at N_EXPERT=64 on the IDENTICAL trace) to show
//   "bigger HBM cache -> higher hit rate".
//============================================================================
module ecc_harness #(
    parameter integer SLOTS     = 4,
    parameter integer N_EXPERT  = 8,
    parameter integer FLASH_LAT = 5,
    parameter integer NRAND     = 200,
    parameter integer NZIPF     = 1500
)(
    input  wire    clk,
    input  wire    go,
    output reg     done_all,
    output integer pass,
    output integer fail,
    output integer z_hits,    // Part B: measured DUT hits on the zipf trace
    output integer z_miss     // Part B: measured DUT misses on the zipf trace
);
    localparam integer ID_W   = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT);
    localparam integer SLOT_W = (SLOTS    <= 1) ? 1 : $clog2(SLOTS);

    reg                rst;
    reg                req_valid;
    reg  [ID_W-1:0]    req_expert_id;
    wire               resp_valid;
    wire               hit;
    wire [SLOT_W-1:0]  resp_slot;
    wire               busy;
    wire               flash_req;
    wire [ID_W-1:0]    flash_expert_id;
    reg                flash_done;
    wire [31:0]        hit_count;
    wire [31:0]        miss_count;

    expert_cache_ctrl #(.SLOTS(SLOTS), .N_EXPERT(N_EXPERT), .FLASH_LAT(FLASH_LAT)) dut (
        .clk(clk), .rst(rst),
        .req_valid(req_valid), .req_expert_id(req_expert_id),
        .resp_valid(resp_valid), .hit(hit), .resp_slot(resp_slot), .busy(busy),
        .flash_req(flash_req), .flash_expert_id(flash_expert_id), .flash_done(flash_done),
        .hit_count(hit_count), .miss_count(miss_count)
    );

    // ---- Flash backing-store model: FLASH_LAT-cycle latency, 1-cycle done ----
    reg     servicing;
    integer fcnt;
    always @(posedge clk) begin
        if (rst) begin
            flash_done <= 1'b0;
            servicing  <= 1'b0;
            fcnt       <= 0;
        end else begin
            flash_done <= 1'b0;
            if (!servicing && flash_req) begin
                servicing <= 1'b1;
                fcnt      <= FLASH_LAT - 1;
            end else if (servicing) begin
                if (fcnt == 0) begin
                    flash_done <= 1'b1;
                    servicing  <= 1'b0;
                end else begin
                    fcnt <= fcnt - 1;
                end
            end
        end
    end

    // ---- behavioral reference cache (the INDEPENDENT golden) ----
    reg               ref_valid [0:SLOTS-1];
    integer           ref_tag   [0:SLOTS-1];   // resident id (or -1)
    integer           ref_ord   [0:SLOTS-1];   // MRU..LRU stack of slot indices
    integer           ref_hits, ref_miss;

    integer g_hit, g_slot;

    task ref_reset;
        integer a;
        begin
            for (a = 0; a < SLOTS; a = a + 1) begin
                ref_valid[a] = 1'b0;
                ref_tag[a]   = -1;
                ref_ord[a]   = a;            // initial recency order = 0..SLOTS-1
            end
            ref_hits = 0;
            ref_miss = 0;
        end
    endtask

    // move slot `s` to MRU (front) of ref_ord
    task ref_touch(input integer s);
        integer a, ppos;
        reg     fnd;
        begin
            fnd = 1'b0; ppos = 0;
            for (a = 0; a < SLOTS; a = a + 1)
                if (!fnd && ref_ord[a] == s) begin ppos = a; fnd = 1'b1; end
            for (a = ppos; a > 0; a = a - 1)
                ref_ord[a] = ref_ord[a-1];
            ref_ord[0] = s;
        end
    endtask

    // compute golden hit/slot for a request id, and UPDATE the ref model
    task ref_request(input integer id);
        integer a, fslot, fv;
        begin
            fslot = -1;
            for (a = 0; a < SLOTS; a = a + 1)
                if (ref_valid[a] && ref_tag[a] == id) fslot = a;

            if (fslot != -1) begin
                g_hit  = 1;
                g_slot = fslot;
                ref_hits = ref_hits + 1;
                ref_touch(fslot);
            end else begin
                g_hit = 0;
                // victim: lowest-index invalid, else LRU (tail of ref_ord)
                fv = -1;
                for (a = SLOTS-1; a >= 0; a = a - 1)
                    if (!ref_valid[a]) fv = a;
                if (fv == -1) fv = ref_ord[SLOTS-1];
                g_slot = fv;
                ref_valid[fv] = 1'b1;
                ref_tag[fv]   = id;
                ref_miss = ref_miss + 1;
                ref_touch(fv);
            end
        end
    endtask

    // drive ONE request, wait for the response, check BIT-EXACT vs the golden
    task do_req(input integer id);
        begin
            // golden first (pure function of the request stream)
            ref_request(id);

            // present the request for one cycle (DUT is idle here)
            @(posedge clk);
            req_valid     <= 1'b1;
            req_expert_id <= id[ID_W-1:0];
            @(posedge clk);
            req_valid     <= 1'b0;

            // wait for the response pulse
            while (resp_valid !== 1'b1) @(posedge clk);

            // ---- checks at the response beat ----
            if ((^resp_valid === 1'bx) || (^hit === 1'bx) ||
                (^resp_slot === 1'bx) || (^busy === 1'bx) ||
                (^hit_count === 1'bx) || (^miss_count === 1'bx)) begin
                $display("FAIL X-bit at resp id=%0d", id);
                fail = fail + 1; $fatal(1, "X-bit");
            end
            if (hit !== g_hit[0]) begin
                $display("FAIL hit id=%0d got=%b exp=%0d", id, hit, g_hit);
                fail = fail + 1; $fatal(1, "hit mismatch");
            end
            if (resp_slot !== g_slot[SLOT_W-1:0]) begin
                $display("FAIL slot id=%0d got=%0d exp=%0d", id, resp_slot, g_slot);
                fail = fail + 1; $fatal(1, "slot mismatch");
            end
            if (busy !== 1'b0) begin
                $display("FAIL busy not low at resp id=%0d", id);
                fail = fail + 1; $fatal(1, "busy");
            end
            if (hit_count !== ref_hits[31:0] || miss_count !== ref_miss[31:0]) begin
                $display("FAIL counts id=%0d hc=%0d(exp %0d) mc=%0d(exp %0d)",
                         id, hit_count, ref_hits, miss_count, ref_miss);
                fail = fail + 1; $fatal(1, "counts");
            end
            pass = pass + 1;
            @(posedge clk);   // settle before next request
        end
    endtask

    // re-assert synchronous reset and clear the golden (isolate a measurement)
    task do_reset;
        begin
            req_valid <= 1'b0;
            rst = 1'b1;
            ref_reset;
            @(posedge clk); @(posedge clk); @(posedge clk);
            rst = 1'b0;
            @(posedge clk);
            if (hit_count !== 32'd0 || miss_count !== 32'd0 || busy !== 1'b0) begin
                $display("FAIL post-reset state");
                fail = fail + 1; $fatal(1, "reset");
            end
            pass = pass + 1;
        end
    endtask

    // PART A -- directed coverage
    task run_directed;
        integer a, A, B, C, D, E;
        begin
            // (1) cold misses fill every empty slot, in slot order 0..SLOTS-1
            for (a = 0; a < SLOTS; a = a + 1) do_req(a + 1);      // ids 1..SLOTS

            // (2) immediate re-request of a resident id -> HIT, same slot
            do_req(1);
            do_req(SLOTS);

            // (3) single-id HAMMER: same id many times -> first is resident, rest hits
            for (a = 0; a < 6; a = a + 1) do_req(2);

            // (4) CLASSIC LRU eviction order (only well-defined for SLOTS==4):
            //     access A B C D, re-access A, insert E -> B is the LRU victim.
            if (SLOTS == 4 && N_EXPERT >= 6) begin
                do_reset;
                A = 1; B = 2; C = 3; D = 4; E = 5;
                do_req(A); do_req(B); do_req(C); do_req(D);  // fill: LRU stack D C B A
                do_req(A);                                   // touch A -> LRU is now B
                do_req(E);                                   // MISS -> must evict B's slot
                do_req(B);                                   // B evicted -> MISS again
                do_req(A);                                   // A still resident -> HIT
                do_reset;                                    // restore for the rest
                // refill so the cache is non-empty going into thrash/random
                for (a = 0; a < SLOTS; a = a + 1) do_req(a + 1);
            end

            // (5) THRASH: cycle through more distinct ids than slots, repeatedly,
            //     so every access evicts the LRU (a pathological 0%-locality stream)
            for (a = 0; a < (SLOTS + 3) * 2; a = a + 1)
                do_req((a % (SLOTS + 3)) + 1);
        end
    endtask

    integer seed;
    task run_random;
        integer a, id;
        begin
            seed = 32'hC0FFEE ^ (SLOTS << 8) ^ N_EXPERT;
            for (a = 0; a < NRAND; a = a + 1) begin
                id = $unsigned($random(seed)) % N_EXPERT;
                do_req(id);
            end
        end
    endtask

    // ---- PART B : Zipf-like skewed trace generator (pure integer) ----------
    // Popularity skew via MIN-OF-K uniform draws (concentrates mass on small
    // ids = "popular" experts); temporal locality by repeating the previous id.
    // The generator depends ONLY on (zseed, N_EXPERT), NOT on SLOTS, so the
    // trace is IDENTICAL across the SLOTS sweep -> a fair cache-size comparison.
    integer zseed;
    function integer zipf_next(input integer prev);
        integer r, m, t, c;
        begin
            r = $unsigned($random(zseed)) % 100;
            if (prev >= 0 && r < 30) begin
                zipf_next = prev;                 // 30% temporal-locality repeat
            end else begin
                m = N_EXPERT;
                for (c = 0; c < 4; c = c + 1) begin
                    t = $unsigned($random(zseed)) % N_EXPERT;
                    if (t < m) m = t;             // min-of-4 -> Zipf-like skew
                end
                zipf_next = m;
            end
        end
    endfunction

    task run_zipf;
        integer a, id, prev;
        begin
            // isolate the measurement: fresh DUT + golden
            do_reset;
            zseed = 32'h5EED1234 ^ (N_EXPERT << 3);   // SLOTS-independent on purpose
            prev  = -1;
            for (a = 0; a < NZIPF; a = a + 1) begin
                id = zipf_next(prev);
                do_req(id);                           // STILL checked vs golden
                prev = id;
            end
            z_hits = hit_count;                       // measured from the DUT
            z_miss = miss_count;
        end
    endtask

    initial begin
        pass = 0; fail = 0; done_all = 1'b0;
        z_hits = 0; z_miss = 0;
        req_valid = 1'b0; req_expert_id = {ID_W{1'b0}};
        rst = 1'b1;
        @(posedge go);

        do_reset;                 // initial reset + post-reset sanity
        run_directed;             // PART A directed
        run_random;               // PART A random
        run_zipf;                 // PART B measurement (+ per-req PART A check)

        done_all = 1'b1;
    end
endmodule

//----------------------------------------------------------------------------
module expert_cache_ctrl_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg     go = 1'b0;

    localparam integer NZ = 1500;

    // width shapes for PART A: 4/8 and 8/64 (8/64 also serves as a sweep point)
    wire    dA, d2, d4, d8, d16;
    integer pA, fA, zhA, zmA;
    integer p2, f2, zh2, zm2;
    integer p4, f4, zh4, zm4;
    integer p8, f8, zh8, zm8;
    integer p16, f16, zh16, zm16;

    // PART A width-derivation shape: SLOTS=4, N_EXPERT=8
    ecc_harness #(.SLOTS(4), .N_EXPERT(8), .FLASH_LAT(5), .NRAND(200), .NZIPF(NZ)) hA (
        .clk(clk), .go(go), .done_all(dA), .pass(pA), .fail(fA), .z_hits(zhA), .z_miss(zmA));

    // SLOTS sweep (all N_EXPERT=64, IDENTICAL zipf trace): 2, 4, 8, 16
    ecc_harness #(.SLOTS(2),  .N_EXPERT(64), .FLASH_LAT(4),  .NRAND(150), .NZIPF(NZ)) h2 (
        .clk(clk), .go(go), .done_all(d2), .pass(p2), .fail(f2), .z_hits(zh2), .z_miss(zm2));
    ecc_harness #(.SLOTS(4),  .N_EXPERT(64), .FLASH_LAT(4),  .NRAND(150), .NZIPF(NZ)) h4 (
        .clk(clk), .go(go), .done_all(d4), .pass(p4), .fail(f4), .z_hits(zh4), .z_miss(zm4));
    ecc_harness #(.SLOTS(8),  .N_EXPERT(64), .FLASH_LAT(20), .NRAND(300), .NZIPF(NZ)) h8 (
        .clk(clk), .go(go), .done_all(d8), .pass(p8), .fail(f8), .z_hits(zh8), .z_miss(zm8));
    ecc_harness #(.SLOTS(16), .N_EXPERT(64), .FLASH_LAT(4),  .NRAND(150), .NZIPF(NZ)) h16 (
        .clk(clk), .go(go), .done_all(d16), .pass(p16), .fail(f16), .z_hits(zh16), .z_miss(zm16));

    integer total, totfail;

    // print one sweep row: hit-rate as integer percent with 2 decimals
    task show_row(input integer slots, input integer h, input integer m);
        integer tot, pctx100;
        begin
            tot     = h + m;
            pctx100 = (tot == 0) ? 0 : (h * 10000) / tot;
            $display("  SLOTS=%2d   hits=%0d  miss=%0d   hit_rate=%0d.%02d%%",
                     slots, h, m, pctx100 / 100, pctx100 % 100);
        end
    endtask

    initial begin
        @(posedge clk);
        go = 1'b1;
        wait (dA && d2 && d4 && d8 && d16);
        @(posedge clk);

        total   = pA + p2 + p4 + p8 + p16;
        totfail = fA + f2 + f4 + f8 + f16;
        if (totfail != 0) begin
            $display("FAIL: fA=%0d f2=%0d f4=%0d f8=%0d f16=%0d",
                     fA, f2, f4, f8, f16);
            $fatal(1, "FAILURES");
        end

        // ---- PART B report : hit-rate vs cache size ----
        $display("");
        $display("PART B  hit-rate vs HBM cache size (N_EXPERT=64, Zipf-like trace, %0d reqs/size):", NZ);
        show_row(2,  zh2,  zm2);
        show_row(4,  zh4,  zm4);
        show_row(8,  zh8,  zm8);
        show_row(16, zh16, zm16);
        $display("  (monotone increasing hit-rate => bigger HBM cache amortizes more Flash misses)");
        $display("");

        $display("widthA(4/8): %0d passed   sweep 2/4/8/16(N=64): %0d/%0d/%0d/%0d passed",
                 pA, p2, p4, p8, p16);
        $display("ALL %0d TESTS PASSED", total);
        $finish;
    end
endmodule
