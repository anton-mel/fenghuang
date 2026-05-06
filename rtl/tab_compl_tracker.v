// =============================================================================
// tab_compl_tracker.v
// FengHuang Tensor Addressable Bridge — Write-Completion Notification Tracker
//
// The completion tracker implements the synchronisation primitive described in
// Section 3.3.1 of the FengHuang paper:
//
//   "Write Completion Notification: Ensures synchronization among xPUs by
//    signalling when data writes have completed."
//
// Usage flow (AllReduce example with N_XPU=4, see Fig 3.5):
//   1.  Each xPU issues N_BANKS Write-Accumulate requests (one per bank).
//       Each request increments pending_count.
//   2.  As each bank completes its atomic add, it pulses compl_valid.
//       Each pulse decrements pending_count.
//   3.  Any xPU may issue an OP_WC_SYNC request (via the TAB top-level)
//       carrying a notify_mask that specifies which xPUs to wake up.
//   4.  Once pending_count reaches 0, the tracker broadcasts a one-cycle
//       sync_valid pulse to all xPUs in the stored notify_mask.
//       Latency: 40 ns (Table 3.1, "Atomic operation completion notification").
//
// The tracker maintains a single global pending counter, which is sufficient
// for the AllReduce / ReduceScatter / AllGather patterns described in the paper
// where all write-accumulates belong to a single collective operation at a time.
// For overlapping collectives, extend with multiple counter slots indexed by
// a group/barrier ID.
// =============================================================================
`timescale 1ns/1ps

module tab_compl_tracker #(
    parameter N_XPU     = 4,    // number of xPUs
    parameter N_BANKS   = 4,    // number of memory banks
    parameter SRC_ID_W  = 2,    // log2(N_XPU)
    parameter TXN_ID_W  = 8,    // transaction ID width
    parameter CNT_W     = 16    // pending-count width (max 2^16 outstanding writes)
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // -------------------------------------------------------------------------
    // Write-increment interface — driven each cycle with the count of
    // Write / Write-Acc requests dispatched to banks that cycle.
    // Multiple writes can be issued in one cycle (e.g., AllGather to distinct
    // banks), so a saturating count is used rather than a single-bit valid.
    // -------------------------------------------------------------------------
    input  wire [CNT_W-1:0]       wr_issued_count,  // # writes issued this cycle
    input  wire [N_XPU-1:0]       wr_issued_mask,   // OR of participating xPUs

    // -------------------------------------------------------------------------
    // Write-completion inputs from all banks
    // -------------------------------------------------------------------------
    input  wire [N_BANKS-1:0]     bank_compl_valid,  // per-bank completion pulses

    // -------------------------------------------------------------------------
    // Sync-request interface — OP_WC_SYNC command from any xPU
    // -------------------------------------------------------------------------
    input  wire                   sync_req_valid,
    output wire                   sync_req_ready,
    input  wire [N_XPU-1:0]       sync_notify_mask,  // which xPUs to wake

    // -------------------------------------------------------------------------
    // Sync-notification output — broadcast to xPUs when pending hits 0
    // -------------------------------------------------------------------------
    output reg  [N_XPU-1:0]       sync_notify_valid, // one-hot per xPU, one cycle
    input  wire [N_XPU-1:0]       sync_notify_ack    // xPU acknowledges notification
);

    // =========================================================================
    // Pending write counter
    // =========================================================================
    reg [CNT_W-1:0]   pending_count;

    // Number of completions arriving this cycle (sum across all banks)
    integer b;
    reg [CNT_W-1:0] completions_this_cycle;
    always @(*) begin
        completions_this_cycle = {CNT_W{1'b0}};
        for (b = 0; b < N_BANKS; b = b + 1)
            completions_this_cycle = completions_this_cycle + {{(CNT_W-1){1'b0}}, bank_compl_valid[b]};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_count <= {CNT_W{1'b0}};
        end else begin
            // Both issue and completion can happen in the same cycle (net delta).
            // wr_issued_count counts how many writes were dispatched this cycle;
            // completions_this_cycle counts how many banks sent a compl pulse.
            pending_count <= pending_count + wr_issued_count - completions_this_cycle;
        end
    end

    // =========================================================================
    // Sync request FIFO (depth-1 — one pending sync request at a time)
    // =========================================================================
    reg                sync_pending;       // a sync request is queued
    reg [N_XPU-1:0]    pending_mask;       // notify_mask for the queued sync

    assign sync_req_ready = !sync_pending; // accept new sync only when idle

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sync_pending <= 1'b0;
            pending_mask <= {N_XPU{1'b0}};
        end else begin
            if (sync_req_valid && sync_req_ready) begin
                sync_pending <= 1'b1;
                pending_mask <= sync_notify_mask;
            end else if (sync_pending && (pending_count == {CNT_W{1'b0}})) begin
                // All pending writes have drained — clear the request
                // (notification is issued below in the same cycle)
                sync_pending <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Broadcast notification when pending drains to zero
    // =========================================================================
    // One-cycle pulse to each xPU in the notify mask.
    // Paper §3.3.2: "The TAB sends a write-completion notification for
    //                synchronization" after all write-accumulates finish.

    reg [N_XPU-1:0] notify_latch;     // hold notification until all xPUs ack
    reg             notify_active;

    integer x;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            notify_latch  <= {N_XPU{1'b0}};
            notify_active <= 1'b0;
            sync_notify_valid <= {N_XPU{1'b0}};
        end else begin
            if (!notify_active) begin
                // Fire notification when sync is pending and all writes done
                if (sync_pending && (pending_count == {CNT_W{1'b0}})) begin
                    notify_latch      <= pending_mask;
                    notify_active     <= 1'b1;
                    sync_notify_valid <= pending_mask;
                end else begin
                    sync_notify_valid <= {N_XPU{1'b0}};
                end
            end else begin
                // Clear per-xPU valid bit as each xPU acknowledges
                for (x = 0; x < N_XPU; x = x + 1) begin
                    if (sync_notify_ack[x])
                        sync_notify_valid[x] <= 1'b0;
                end
                // All xPUs acked → de-assert
                if ((sync_notify_valid & ~sync_notify_ack) == {N_XPU{1'b0}})
                    notify_active <= 1'b0;
            end
        end
    end

endmodule
