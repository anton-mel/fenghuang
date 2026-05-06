// =============================================================================
// fenghuang_tab.v
// FengHuang Tensor Addressable Bridge (TAB) — Top Level
//
// Reference: "FengHuang: Next-Generation Memory Orchestration for AI
//             Inferencing", Microsoft Research, November 2025 (arXiv 2511.10753)
//
// Architecture (Fig 1.3 / Fig 3.2 / Fig 3.4):
//
//   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
//   │ xPU 0 │   │ xPU 1 │   │ xPU 2 │   │ xPU 3 │   ← N_XPU accelerators
//   └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘
//       │   PHXLINK (SerDes 224G/448G)        │
//   ────┴───────────┴───────────┴─────────────┴────
//                    FengHuang TAB
//   ──┬──────────────┬─────────────────┬───────────┬─
//     │              │                 │           │
//   ┌─┴──┐        ┌──┴─┐           ┌──┴─┐      ┌──┴─┐
//   │LP6 │        │LP6 │           │LP6 │      │LP6 │  ← N_BANKS LPDDR6 banks
//   └────┘        └────┘           └────┘      └────┘
//
// Supported operations (per xPU request):
//   OP_READ    (2'b00) — load tensor slice from remote memory
//   OP_WRITE   (2'b01) — store tensor slice (post-write / fire-and-notify)
//   OP_WR_ACC  (2'b10) — write-accumulate: mem[addr] += data  (in-memory reduce)
//   OP_WC_SYNC (2'b11) — request write-completion notification
//                         (fields: notify_mask selects which xPUs receive the sync)
//
// Communication primitives built on top (Fig 3.5 – 3.7):
//   AllReduce / ReduceScatter — all xPUs issue OP_WR_ACC; TAB reduces in-place
//   AllGather / AllToAll      — all xPUs issue OP_WRITE to distinct addresses
//   P2P Send/Recv             — single xPU issues OP_WRITE to a shared slot
//   In all cases a trailing OP_WC_SYNC ensures synchronisation.
//
// Latency reference (Table 3.1):
//   Read     220 ns  |  Write   90 ns  |  WrAcc  90 ns  |  Sync   40 ns
//
// Parameters
// ----------
//   N_XPU      : number of attached xPUs (default 4)
//   N_BANKS    : number of remote memory banks (default 4 = N_XPU, for full
//                bisection bandwidth; can be scaled independently)
//   ADDR_W     : global word-address width (default 32)
//   DATA_W     : data-path width in bits   (default 512 = 64 B, one cache-line)
//   MEM_DEPTH  : simulated depth per bank  (default 4096 entries)
//   TXN_ID_W   : transaction tag width     (default 8 — 256 outstanding per xPU)
// =============================================================================
`timescale 1ns/1ps

module fenghuang_tab #(
    parameter N_XPU      = 4,
    parameter N_BANKS    = 4,
    parameter ADDR_W     = 32,
    parameter DATA_W     = 512,
    parameter MEM_DEPTH  = 4096,
    parameter TXN_ID_W   = 8,
    parameter CNT_W      = 16
)(
    input  wire                           clk,
    input  wire                           rst_n,

    // =========================================================================
    // xPU Request Interfaces  (one bundle per xPU)
    // =========================================================================
    // Each xPU presents a simple valid/ready command interface.
    // Packed arrays: [xpu_idx * FIELD_W +: FIELD_W]

    input  wire [N_XPU-1:0]              xpu_req_valid,
    output wire [N_XPU-1:0]             xpu_req_ready,

    // op: 00=READ 01=WRITE 10=WR_ACC 11=WC_SYNC
    input  wire [N_XPU*2-1:0]           xpu_req_op,

    // Global shared-memory word address.
    // Lower $clog2(N_BANKS) bits select the bank (striping); upper bits are
    // the bank-local address.
    input  wire [N_XPU*ADDR_W-1:0]      xpu_req_addr,

    // Write / accumulate data
    input  wire [N_XPU*DATA_W-1:0]      xpu_req_wdata,

    // Transaction ID — echoed back in the response
    input  wire [N_XPU*TXN_ID_W-1:0]    xpu_req_txn_id,

    // WC_SYNC only: one-hot mask of xPUs to notify when pending writes drain
    input  wire [N_XPU*N_XPU-1:0]       xpu_req_notify_mask,

    // =========================================================================
    // xPU Response Interfaces
    // =========================================================================
    output wire [N_XPU-1:0]             xpu_rsp_valid,
    input  wire [N_XPU-1:0]             xpu_rsp_ready,

    // type: 00=read-data  01=write-complete  10=sync-notify
    output wire [N_XPU*2-1:0]           xpu_rsp_type,

    // Read data payload (valid when type==00)
    output wire [N_XPU*DATA_W-1:0]      xpu_rsp_rdata,

    // Transaction ID echo
    output wire [N_XPU*TXN_ID_W-1:0]    xpu_rsp_txn_id,

    // Sync notification (valid when type==10; one bit per xPU, one cycle)
    output wire [N_XPU-1:0]             xpu_sync_notify_valid,
    input  wire [N_XPU-1:0]             xpu_sync_notify_ack
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    localparam SRC_ID_W   = (N_XPU > 1) ? $clog2(N_XPU) : 1;
    localparam BANK_SEL_W = $clog2(N_BANKS);
    localparam BANK_ADDR_W = ADDR_W - BANK_SEL_W;

    localparam OP_READ   = 2'b00;
    localparam OP_WRITE  = 2'b01;
    localparam OP_WR_ACC = 2'b10;
    localparam OP_WC_SYNC = 2'b11;

    // =========================================================================
    // OP_WC_SYNC interception
    // =========================================================================
    // WC_SYNC requests are not forwarded to the crossbar; they are captured
    // here and routed directly to the completion tracker.

    wire [N_XPU-1:0]   is_sync_req;       // per-xPU: this request is WC_SYNC
    wire [N_XPU-1:0]   pass_req_valid;    // forwarded to crossbar (non-sync)
    wire [N_XPU-1:0]   sync_req_ready_cb; // crossbar req_ready for non-sync

    // Sync request arbitration: one sync accepted per cycle (tracker has depth-1 buffer)
    wire               tracker_sync_ready;
    reg  [N_XPU-1:0]   sync_req_grant;    // one-hot: which xPU wins sync slot
    wire [N_XPU-1:0]   sync_notify_mask_out; // resolved mask for tracker

    genvar gi;
    generate
        for (gi = 0; gi < N_XPU; gi = gi + 1) begin : sync_decode
            assign is_sync_req[gi] = xpu_req_valid[gi] &&
                                     (xpu_req_op[gi*2 +: 2] == OP_WC_SYNC);
        end
    endgenerate

    // First requesting xPU wins sync arbitration this cycle
    integer x;
    reg    sync_fired;
    always @(*) begin
        sync_req_grant = {N_XPU{1'b0}};
        sync_fired     = 1'b0;
        for (x = 0; x < N_XPU; x = x + 1) begin
            if (!sync_fired && is_sync_req[x] && tracker_sync_ready) begin
                sync_req_grant[x] = 1'b1;
                sync_fired        = 1'b1;
            end
        end
    end

    // Resolve notify_mask (OR of all granted sync requestors' masks)
    reg [N_XPU-1:0] resolved_mask;
    always @(*) begin
        resolved_mask = {N_XPU{1'b0}};
        for (x = 0; x < N_XPU; x = x + 1) begin
            if (sync_req_grant[x])
                resolved_mask = resolved_mask | xpu_req_notify_mask[x*N_XPU +: N_XPU];
        end
    end
    assign sync_notify_mask_out = resolved_mask;

    // Non-sync requests pass through to the crossbar
    generate
        for (gi = 0; gi < N_XPU; gi = gi + 1) begin : fwd_req
            assign pass_req_valid[gi] = xpu_req_valid[gi] && !is_sync_req[gi];
        end
    endgenerate

    // xPU req_ready: accept sync immediately if tracker ready, else crossbar ready
    generate
        for (gi = 0; gi < N_XPU; gi = gi + 1) begin : req_ready_mux
            assign xpu_req_ready[gi] = is_sync_req[gi] ? sync_req_grant[gi]
                                                        : sync_req_ready_cb[gi];
        end
    endgenerate

    // =========================================================================
    // Bank wires (crossbar ↔ banks)
    // =========================================================================
    wire [N_BANKS-1:0]                   bank_req_valid;
    wire [N_BANKS-1:0]                   bank_req_ready;
    wire [N_BANKS*2-1:0]                 bank_req_op;
    wire [N_BANKS*BANK_ADDR_W-1:0]       bank_req_addr;
    wire [N_BANKS*DATA_W-1:0]            bank_req_wdata;
    wire [N_BANKS*SRC_ID_W-1:0]          bank_req_src_id;
    wire [N_BANKS*TXN_ID_W-1:0]          bank_req_txn_id;

    wire [N_BANKS-1:0]                   bank_rsp_valid;
    wire [N_BANKS-1:0]                   bank_rsp_ready;
    wire [N_BANKS*DATA_W-1:0]            bank_rsp_rdata;
    wire [N_BANKS*SRC_ID_W-1:0]          bank_rsp_src_id;
    wire [N_BANKS*TXN_ID_W-1:0]          bank_rsp_txn_id;

    wire [N_BANKS-1:0]                   bank_compl_valid;
    wire [N_BANKS*SRC_ID_W-1:0]          bank_compl_src_id;
    wire [N_BANKS*TXN_ID_W-1:0]          bank_compl_txn_id;

    // =========================================================================
    // Crossbar instantiation
    // =========================================================================
    tab_crossbar #(
        .N_XPU     (N_XPU),
        .N_BANKS   (N_BANKS),
        .ADDR_W    (ADDR_W),
        .DATA_W    (DATA_W),
        .SRC_ID_W  (SRC_ID_W),
        .TXN_ID_W  (TXN_ID_W)
    ) u_crossbar (
        .clk               (clk),
        .rst_n             (rst_n),

        .xpu_req_valid     (pass_req_valid),
        .xpu_req_ready     (sync_req_ready_cb),
        .xpu_req_op        (xpu_req_op),
        .xpu_req_addr      (xpu_req_addr),
        .xpu_req_wdata     (xpu_req_wdata),
        .xpu_req_txn_id    (xpu_req_txn_id),

        .xpu_rsp_valid     (xpu_rsp_valid),
        .xpu_rsp_ready     (xpu_rsp_ready),
        .xpu_rsp_type      (xpu_rsp_type),
        .xpu_rsp_rdata     (xpu_rsp_rdata),
        .xpu_rsp_txn_id    (xpu_rsp_txn_id),

        .bank_req_valid    (bank_req_valid),
        .bank_req_ready    (bank_req_ready),
        .bank_req_op       (bank_req_op),
        .bank_req_addr     (bank_req_addr),
        .bank_req_wdata    (bank_req_wdata),
        .bank_req_src_id   (bank_req_src_id),
        .bank_req_txn_id   (bank_req_txn_id),

        .bank_rsp_valid    (bank_rsp_valid),
        .bank_rsp_ready    (bank_rsp_ready),
        .bank_rsp_rdata    (bank_rsp_rdata),
        .bank_rsp_src_id   (bank_rsp_src_id),
        .bank_rsp_txn_id   (bank_rsp_txn_id),

        .bank_compl_valid  (bank_compl_valid),
        .bank_compl_src_id (bank_compl_src_id),
        .bank_compl_txn_id (bank_compl_txn_id)
    );

    // =========================================================================
    // Memory bank instantiation (one per bank)
    // =========================================================================
    generate
        for (gi = 0; gi < N_BANKS; gi = gi + 1) begin : gen_banks
            tab_mem_bank #(
                .ADDR_W    (BANK_ADDR_W),
                .DATA_W    (DATA_W),
                .MEM_DEPTH (MEM_DEPTH),
                .SRC_ID_W  (SRC_ID_W),
                .TXN_ID_W  (TXN_ID_W)
            ) u_bank (
                .clk           (clk),
                .rst_n         (rst_n),

                .req_valid     (bank_req_valid  [gi]),
                .req_ready     (bank_req_ready  [gi]),
                .req_op        (bank_req_op     [gi*2       +: 2]),
                .req_addr      (bank_req_addr   [gi*BANK_ADDR_W +: BANK_ADDR_W]),
                .req_wdata     (bank_req_wdata  [gi*DATA_W  +: DATA_W]),
                .req_src_id    (bank_req_src_id [gi*SRC_ID_W +: SRC_ID_W]),
                .req_txn_id    (bank_req_txn_id [gi*TXN_ID_W +: TXN_ID_W]),

                .rsp_valid     (bank_rsp_valid  [gi]),
                .rsp_ready     (bank_rsp_ready  [gi]),
                .rsp_rdata     (bank_rsp_rdata  [gi*DATA_W  +: DATA_W]),
                .rsp_src_id    (bank_rsp_src_id [gi*SRC_ID_W +: SRC_ID_W]),
                .rsp_txn_id    (bank_rsp_txn_id [gi*TXN_ID_W +: TXN_ID_W]),

                .compl_valid   (bank_compl_valid  [gi]),
                .compl_src_id  (bank_compl_src_id [gi*SRC_ID_W +: SRC_ID_W]),
                .compl_txn_id  (bank_compl_txn_id [gi*TXN_ID_W +: TXN_ID_W])
            );
        end
    endgenerate

    // =========================================================================
    // Write-issued count for completion tracker
    // =========================================================================
    // Count how many write / write-accumulate requests are accepted by the
    // crossbar this cycle.  Multiple xPUs can be granted simultaneously when
    // they target distinct banks (e.g., AllGather), so a count is needed
    // rather than a single-bit valid.
    reg [CNT_W-1:0]  wr_issued_count;
    reg [N_XPU-1:0]  wr_issued_mask;

    always @(*) begin
        wr_issued_count = {CNT_W{1'b0}};
        wr_issued_mask  = {N_XPU{1'b0}};
        for (x = 0; x < N_XPU; x = x + 1) begin
            if (pass_req_valid[x] && sync_req_ready_cb[x] &&
                (xpu_req_op[x*2 +: 2] == OP_WRITE ||
                 xpu_req_op[x*2 +: 2] == OP_WR_ACC)) begin
                wr_issued_count   = wr_issued_count + 1;
                wr_issued_mask[x] = 1'b1;
            end
        end
    end

    // =========================================================================
    // Completion tracker instantiation
    // =========================================================================
    tab_compl_tracker #(
        .N_XPU    (N_XPU),
        .N_BANKS  (N_BANKS),
        .SRC_ID_W (SRC_ID_W),
        .TXN_ID_W (TXN_ID_W),
        .CNT_W    (CNT_W)
    ) u_compl_tracker (
        .clk                (clk),
        .rst_n              (rst_n),

        .wr_issued_count    (wr_issued_count),
        .wr_issued_mask     (wr_issued_mask),

        .bank_compl_valid   (bank_compl_valid),

        .sync_req_valid     (|sync_req_grant),
        .sync_req_ready     (tracker_sync_ready),
        .sync_notify_mask   (sync_notify_mask_out),

        .sync_notify_valid  (xpu_sync_notify_valid),
        .sync_notify_ack    (xpu_sync_notify_ack)
    );

endmodule
