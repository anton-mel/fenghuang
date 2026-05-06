// =============================================================================
// tb_fenghuang_tab.v
// Testbench for FengHuang Tensor Addressable Bridge (TAB)
//
// Tests three key communication patterns from the paper (§3.3.2):
//
//   Test 1 — P2P Send/Recv (Fig 3.7)
//     xPU 0 writes a tensor slice to shared memory; xPU 1 reads it back.
//     TAB issues write-completion notification; xPU 1 waits for sync.
//
//   Test 2 — AllReduce / ReduceScatter (Fig 3.5)
//     All 4 xPUs write-accumulate their local partial sums to the same
//     address.  After all banks signal completion, TAB broadcasts a sync
//     notification.  xPU 0 reads back the accumulated result.
//
//   Test 3 — AllGather (Fig 3.6)
//     Each xPU writes its local data to a distinct region of shared memory.
//     TAB notifies all xPUs upon completion.  Each xPU reads back its
//     peer's region.
// =============================================================================
`timescale 1ns/1ps

module tb_fenghuang_tab;

    // =========================================================================
    // Parameters — kept small for simulation speed
    // =========================================================================
    localparam N_XPU     = 4;
    localparam N_BANKS   = 4;
    localparam ADDR_W    = 16;
    localparam DATA_W    = 64;    // 8-byte words for readability
    localparam MEM_DEPTH = 256;
    localparam TXN_ID_W  = 8;
    localparam CNT_W     = 16;

    // =========================================================================
    // DUT interface signals
    // =========================================================================
    reg                          clk;
    reg                          rst_n;

    reg  [N_XPU-1:0]             xpu_req_valid;
    wire [N_XPU-1:0]             xpu_req_ready;
    reg  [N_XPU*2-1:0]           xpu_req_op;
    reg  [N_XPU*ADDR_W-1:0]      xpu_req_addr;
    reg  [N_XPU*DATA_W-1:0]      xpu_req_wdata;
    reg  [N_XPU*TXN_ID_W-1:0]    xpu_req_txn_id;
    reg  [N_XPU*N_XPU-1:0]       xpu_req_notify_mask;

    wire [N_XPU-1:0]             xpu_rsp_valid;
    reg  [N_XPU-1:0]             xpu_rsp_ready;
    wire [N_XPU*2-1:0]           xpu_rsp_type;
    wire [N_XPU*DATA_W-1:0]      xpu_rsp_rdata;
    wire [N_XPU*TXN_ID_W-1:0]    xpu_rsp_txn_id;

    wire [N_XPU-1:0]             xpu_sync_notify_valid;
    reg  [N_XPU-1:0]             xpu_sync_notify_ack;

    // =========================================================================
    // DUT instantiation
    // =========================================================================
    fenghuang_tab #(
        .N_XPU     (N_XPU),
        .N_BANKS   (N_BANKS),
        .ADDR_W    (ADDR_W),
        .DATA_W    (DATA_W),
        .MEM_DEPTH (MEM_DEPTH),
        .TXN_ID_W  (TXN_ID_W),
        .CNT_W     (CNT_W)
    ) dut (
        .clk                   (clk),
        .rst_n                 (rst_n),

        .xpu_req_valid         (xpu_req_valid),
        .xpu_req_ready         (xpu_req_ready),
        .xpu_req_op            (xpu_req_op),
        .xpu_req_addr          (xpu_req_addr),
        .xpu_req_wdata         (xpu_req_wdata),
        .xpu_req_txn_id        (xpu_req_txn_id),
        .xpu_req_notify_mask   (xpu_req_notify_mask),

        .xpu_rsp_valid         (xpu_rsp_valid),
        .xpu_rsp_ready         (xpu_rsp_ready),
        .xpu_rsp_type          (xpu_rsp_type),
        .xpu_rsp_rdata         (xpu_rsp_rdata),
        .xpu_rsp_txn_id        (xpu_rsp_txn_id),

        .xpu_sync_notify_valid (xpu_sync_notify_valid),
        .xpu_sync_notify_ack   (xpu_sync_notify_ack)
    );

    // =========================================================================
    // Clock generation — 1 GHz (1 ns period)
    // =========================================================================
    initial clk = 0;
    always #0.5 clk = ~clk;

    // =========================================================================
    // Helper tasks
    // =========================================================================
    localparam OP_READ    = 2'b00;
    localparam OP_WRITE   = 2'b01;
    localparam OP_WR_ACC  = 2'b10;
    localparam OP_WC_SYNC = 2'b11;

    // Issue a single-cycle request from xpu_id and wait for acceptance
    task automatic issue_req;
        input integer         xpu_id;
        input [1:0]           op;
        input [ADDR_W-1:0]    addr;
        input [DATA_W-1:0]    wdata;
        input [TXN_ID_W-1:0]  tid;
        input [N_XPU-1:0]     notify_mask; // only used for WC_SYNC
        begin
            @(posedge clk);
            #0.1; // small skew after clock edge
            xpu_req_valid[xpu_id]                          = 1'b1;
            xpu_req_op   [xpu_id*2       +: 2]             = op;
            xpu_req_addr [xpu_id*ADDR_W  +: ADDR_W]        = addr;
            xpu_req_wdata[xpu_id*DATA_W  +: DATA_W]        = wdata;
            xpu_req_txn_id[xpu_id*TXN_ID_W +: TXN_ID_W]   = tid;
            xpu_req_notify_mask[xpu_id*N_XPU +: N_XPU]     = notify_mask;

            // Wait until the TAB accepts the request — poll at clock edges
            // to avoid triggering on combinatorial glitches during the delta
            // cycles when the parallel fork threads are assigning their own
            // valid bits (each assignment re-evaluates the crossbar, briefly
            // producing transient ready pulses for other xPUs).
            do @(posedge clk); while (!xpu_req_ready[xpu_id]);
            #0.1; // hold valid past posedge so bank samples cleanly
            xpu_req_valid[xpu_id] = 1'b0;
        end
    endtask

    // Wait for a read response on xpu_id; return data and tid
    task automatic wait_rsp_read;
        input  integer        xpu_id;
        output [DATA_W-1:0]   rdata;
        output [TXN_ID_W-1:0] ret_tid;
        begin
            xpu_rsp_ready[xpu_id] = 1'b1;
            wait(xpu_rsp_valid[xpu_id] &&
                 xpu_rsp_type[xpu_id*2 +: 2] == 2'b00);
            rdata   = xpu_rsp_rdata  [xpu_id*DATA_W   +: DATA_W];
            ret_tid = xpu_rsp_txn_id [xpu_id*TXN_ID_W +: TXN_ID_W];
            @(posedge clk);
            xpu_rsp_ready[xpu_id] = 1'b0;
        end
    endtask

    // Wait for write-completion notification on xpu_id (type==01)
    task automatic wait_wr_compl;
        input integer xpu_id;
        begin
            xpu_rsp_ready[xpu_id] = 1'b1;
            wait(xpu_rsp_valid[xpu_id] &&
                 xpu_rsp_type[xpu_id*2 +: 2] == 2'b01);
            @(posedge clk);
            xpu_rsp_ready[xpu_id] = 1'b0;
        end
    endtask

    // Wait for sync notification on xpu_id, then ack
    task automatic wait_sync_notify;
        input integer xpu_id;
        begin
            wait(xpu_sync_notify_valid[xpu_id]);
            @(posedge clk);
            xpu_sync_notify_ack[xpu_id] = 1'b1;
            @(posedge clk);
            xpu_sync_notify_ack[xpu_id] = 1'b0;
        end
    endtask

    // =========================================================================
    // Result checking
    // =========================================================================
    integer pass_cnt;
    integer fail_cnt;

    task automatic check_eq;
        input [DATA_W-1:0]  got;
        input [DATA_W-1:0]  exp;
        input [127:0]       label; // use short string
        begin
            if (got === exp) begin
                $display("[PASS] %0s  got=0x%0h", label, got);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %0s  got=0x%0h  exp=0x%0h", label, got, exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    // =========================================================================
    // Test variables
    // =========================================================================
    reg [DATA_W-1:0]   rd_data;
    reg [TXN_ID_W-1:0] rd_tid;
    integer            i;

    // =========================================================================
    // Main stimulus
    // =========================================================================
    initial begin
        // ---- Initialise signals ----
        rst_n                = 1'b0;
        xpu_req_valid        = {N_XPU{1'b0}};
        xpu_req_op           = {(N_XPU*2){1'b0}};
        xpu_req_addr         = {(N_XPU*ADDR_W){1'b0}};
        xpu_req_wdata        = {(N_XPU*DATA_W){1'b0}};
        xpu_req_txn_id       = {(N_XPU*TXN_ID_W){1'b0}};
        xpu_req_notify_mask  = {(N_XPU*N_XPU){1'b0}};
        xpu_rsp_ready        = {N_XPU{1'b1}};
        xpu_sync_notify_ack  = {N_XPU{1'b0}};
        pass_cnt             = 0;
        fail_cnt             = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // =====================================================================
        // TEST 1 — P2P Send / Recv  (Fig 3.7)
        // xPU 0 writes 0xDEADBEEF_CAFEBABE to address 0x0004
        // xPU 1 reads it back after receiving write-completion notification
        // =====================================================================
        $display("\n=== Test 1: P2P Send/Recv ===");

        // xPU 0 writes to shared address 0x0004
        // Bank selected by addr[1:0] = 2'b00 → bank 0
        // Bank-local address = 0x0004 >> 2 = 0x0001
        fork
            begin
                issue_req(0, OP_WRITE, 16'h0004, 64'hDEADBEEF_CAFEBABE, 8'h01, 4'b0000);
            end
            begin
                // Issue WC_SYNC immediately after write, notify xPU 1
                @(posedge clk); @(posedge clk);
                issue_req(0, OP_WC_SYNC, 16'h0000, 64'h0, 8'h02, 4'b0010);
            end
        join

        // xPU 1 waits for sync notification, then reads
        wait_sync_notify(1);
        $display("  xPU 1 received sync notification");

        issue_req(1, OP_READ, 16'h0004, 64'h0, 8'h10, 4'b0000);
        wait_rsp_read(1, rd_data, rd_tid);
        check_eq(rd_data, 64'hDEADBEEF_CAFEBABE, "P2P_READ");

        // =====================================================================
        // TEST 2 — AllReduce via Write-Accumulate (Fig 3.5)
        // All 4 xPUs accumulate their local partial sums into address 0x0008.
        // Expected result: 0x10 + 0x20 + 0x30 + 0x40 = 0xA0
        // =====================================================================
        $display("\n=== Test 2: AllReduce (Write-Accumulate) ===");

        // First, clear the target address by writing 0
        issue_req(0, OP_WRITE, 16'h0008, 64'h00, 8'h20, 4'b0000);
        @(posedge clk); @(posedge clk); // allow write to complete

        // All 4 xPUs write-accumulate in parallel
        fork
            issue_req(0, OP_WR_ACC, 16'h0008, 64'h10, 8'h21, 4'b0000);
            issue_req(1, OP_WR_ACC, 16'h0008, 64'h20, 8'h22, 4'b0000);
            issue_req(2, OP_WR_ACC, 16'h0008, 64'h30, 8'h23, 4'b0000);
            issue_req(3, OP_WR_ACC, 16'h0008, 64'h40, 8'h24, 4'b0000);
        join
        // Issue WC_SYNC from xPU 0, notify all 4 xPUs
        issue_req(0, OP_WC_SYNC, 16'h0000, 64'h0, 8'h25, 4'b1111);

        // All xPUs wait for sync
        fork
            wait_sync_notify(0);
            wait_sync_notify(1);
            wait_sync_notify(2);
            wait_sync_notify(3);
        join
        $display("  All 4 xPUs received AllReduce sync notification");

        // xPU 0 reads back the accumulated result
        issue_req(0, OP_READ, 16'h0008, 64'h0, 8'h26, 4'b0000);
        wait_rsp_read(0, rd_data, rd_tid);
        check_eq(rd_data, 64'hA0, "AllReduce_result");

        // =====================================================================
        // TEST 3 — AllGather (Fig 3.6)
        // Each xPU writes its local tensor to a distinct address region.
        // Bank striping: xPU i → address (i*4), which maps to bank i.
        // All xPUs then read back their peers' data.
        // =====================================================================
        $display("\n=== Test 3: AllGather ===");

        // Each xPU writes to its own slot: addr = i * 4  (bank = addr[1:0] = i mod 4)
        fork
            issue_req(0, OP_WRITE, 16'h0000, 64'hAAAA_0000_0000_AAAA, 8'h30, 4'b0000);
            issue_req(1, OP_WRITE, 16'h0001, 64'hBBBB_1111_1111_BBBB, 8'h31, 4'b0000);
            issue_req(2, OP_WRITE, 16'h0002, 64'hCCCC_2222_2222_CCCC, 8'h32, 4'b0000);
            issue_req(3, OP_WRITE, 16'h0003, 64'hDDDD_3333_3333_DDDD, 8'h33, 4'b0000);
        join

        // xPU 0 issues WC_SYNC, notify all
        issue_req(0, OP_WC_SYNC, 16'h0000, 64'h0, 8'h34, 4'b1111);

        fork
            wait_sync_notify(0);
            wait_sync_notify(1);
            wait_sync_notify(2);
            wait_sync_notify(3);
        join
        $display("  AllGather writes complete, all xPUs notified");

        // xPU 1 reads xPU 0's slot
        issue_req(1, OP_READ, 16'h0000, 64'h0, 8'h40, 4'b0000);
        wait_rsp_read(1, rd_data, rd_tid);
        check_eq(rd_data, 64'hAAAA_0000_0000_AAAA, "AllGather_xPU0_data_via_xPU1");

        // xPU 0 reads xPU 3's slot
        issue_req(0, OP_READ, 16'h0003, 64'h0, 8'h41, 4'b0000);
        wait_rsp_read(0, rd_data, rd_tid);
        check_eq(rd_data, 64'hDDDD_3333_3333_DDDD, "AllGather_xPU3_data_via_xPU0");

        // =====================================================================
        // Summary
        // =====================================================================
        $display("\n=== Summary: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("ALL TESTS PASSED");
        else
            $display("FAILURES DETECTED");

        repeat (10) @(posedge clk);
        $finish;
    end

    // =========================================================================
    // Timeout watchdog
    // =========================================================================
    initial begin
        #100000;
        $display("[ERROR] Simulation timeout!");
        $finish;
    end


    // =========================================================================
    // VCD dump — limit to top-level signals only (avoids iverilog hang on
    // large mem arrays inside generate blocks)
    // =========================================================================
    initial begin
        $dumpfile("fenghuang_tab.vcd");
        $dumpvars(1, tb_fenghuang_tab);  // depth 1: top-level only
    end

endmodule
