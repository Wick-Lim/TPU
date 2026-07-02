# ============================================================================
# glm_fp8_system_cdc.sdc  --  CDC timing constraints for the 2-clock product top
# ----------------------------------------------------------------------------
# Task C8 (the CDC SIGN-OFF half).  glm_fp8_system_cdc runs the compute box
# entirely on core_clk and presents a host_clk (USB-C device) interface.  EVERY
# signal that crosses between the two asynchronous domains does so through a
# recognized synchronizer:
#   * u_req_fifo  (cdc_async_fifo) : host_clk -> core_clk, {prompt_tok,start_pos,s_len}
#   * u_tok_fifo  (cdc_async_fifo) : core_clk -> host_clk, next_tok
#   * busy_s1/busy_s2              : 2-FF sync of sys_busy (core -> host)
#   * done_tgl_c -> done_tgl_h1/h2/h3 : toggle + 3-FF sync of sys_done (core -> host)
#   * u_host_rst_sync/u_core_rst_sync (reset_sync) : per-domain async-assert reset
#
# These constraints tell STA the two clocks are asynchronous and cut/bound the
# crossing paths so the tool does not try (and fail) to meet a single-cycle
# setup/hold across domains.  Periods below are REPRESENTATIVE placeholders --
# retarget them to the real device/compute clock plan at P&R.
# ============================================================================

# ---- clocks (representative periods; retarget at sign-off) ------------------
create_clock -name host_clk -period 10.0 [get_ports host_clk]   ;# 100 MHz host/USB device
create_clock -name core_clk -period  2.0 [get_ports core_clk]   ;# 500 MHz compute die

# ---- the two domains are ASYNCHRONOUS ---------------------------------------
set_clock_groups -asynchronous \
    -group [get_clocks host_clk] \
    -group [get_clocks core_clk]

# ---- REQUEST async FIFO (host -> core): gray pointers cross domains ----------
# The gray-coded write pointer is synchronized into the read (core) domain and
# vice-versa inside cdc_async_fifo; cut those pointer-crossing paths.
set_false_path -through [get_pins -hierarchical *u_req_fifo*wq2_rptr*]
set_false_path -through [get_pins -hierarchical *u_req_fifo*rq2_wptr*]
# The FIFO payload RAM is written by host_clk and read by core_clk after the
# pointer handshake guarantees stability; bound it datapath-only.
set_max_delay -datapath_only 8.0 -from [get_clocks host_clk] -through [get_pins -hierarchical *u_req_fifo*mem*] -to [get_clocks core_clk]

# ---- TOKEN async FIFO (core -> host) ----------------------------------------
set_false_path -through [get_pins -hierarchical *u_tok_fifo*wq2_rptr*]
set_false_path -through [get_pins -hierarchical *u_tok_fifo*rq2_wptr*]
set_max_delay -datapath_only 8.0 -from [get_clocks core_clk] -through [get_pins -hierarchical *u_tok_fifo*mem*] -to [get_clocks host_clk]

# ---- busy: 2-FF level synchronizer (core sys_busy -> host busy_s1/busy_s2) ---
# First-stage capture flop busy_s1 is the metastability point; false-path its D.
set_false_path -to [get_pins -hierarchical *busy_s1*/D]

# ---- done: toggle + 3-FF synchronizer (core done_tgl_c -> host done_tgl_h1..3)
set_false_path -to [get_pins -hierarchical *done_tgl_h1*/D]

# ---- per-domain reset synchronizers: async ASSERT path is by design async ----
# reset_sync asserts asynchronously (async clear of the chain) and deasserts
# synchronously; cut the async-assert path into the first chain flop.
set_false_path -to [get_pins -hierarchical *u_host_rst_sync*sync_chain*/CLR*]
set_false_path -to [get_pins -hierarchical *u_core_rst_sync*sync_chain*/CLR*]
