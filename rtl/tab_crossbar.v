// =============================================================================
// tab_crossbar.v
// FengHuang Tensor Addressable Bridge — N_XPU × N_BANKS Crossbar Switch
//
// Routes xPU requests to memory banks based on address striping:
//   bank_index  = req_addr[BANK_SEL_W-1:0]       (lower address bits)
//   bank_offset = req_addr[ADDR_W-1:BANK_SEL_W]  (upper address bits)
//
// Arbitration: per-bank round-robin among all requesting xPUs.
// Only one request per bank per clock (bank is single-ported).
// Backpressure: xPU req_ready deasserted until its turn.
//
// Implementation note: all 2D arrays are flattened to 1D packed vectors to
// ensure iverilog correctly infers sensitivity lists in always @(*) blocks.
// =============================================================================
`timescale 1ns/1ps

module tab_crossbar #(
    parameter N_XPU      = 4,
    parameter N_BANKS    = 4,
    parameter ADDR_W     = 32,
    parameter DATA_W     = 512,
    parameter SRC_ID_W   = 2,
    parameter TXN_ID_W   = 8
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // xPU → TAB request
    input  wire [N_XPU-1:0]              xpu_req_valid,
    output reg  [N_XPU-1:0]             xpu_req_ready,
    input  wire [N_XPU*2-1:0]           xpu_req_op,
    input  wire [N_XPU*ADDR_W-1:0]      xpu_req_addr,
    input  wire [N_XPU*DATA_W-1:0]      xpu_req_wdata,
    input  wire [N_XPU*TXN_ID_W-1:0]    xpu_req_txn_id,

    // TAB → xPU response
    output reg  [N_XPU-1:0]             xpu_rsp_valid,
    input  wire [N_XPU-1:0]             xpu_rsp_ready,
    output reg  [N_XPU*2-1:0]           xpu_rsp_type,
    output reg  [N_XPU*DATA_W-1:0]      xpu_rsp_rdata,
    output reg  [N_XPU*TXN_ID_W-1:0]    xpu_rsp_txn_id,

    // TAB → bank request
    output reg  [N_BANKS-1:0]            bank_req_valid,
    input  wire [N_BANKS-1:0]            bank_req_ready,
    output reg  [N_BANKS*2-1:0]          bank_req_op,
    output reg  [N_BANKS*(ADDR_W-$clog2(N_BANKS))-1:0]  bank_req_addr,
    output reg  [N_BANKS*DATA_W-1:0]     bank_req_wdata,
    output reg  [N_BANKS*SRC_ID_W-1:0]   bank_req_src_id,
    output reg  [N_BANKS*TXN_ID_W-1:0]   bank_req_txn_id,

    // Bank → TAB read response
    input  wire [N_BANKS-1:0]            bank_rsp_valid,
    output reg  [N_BANKS-1:0]            bank_rsp_ready,
    input  wire [N_BANKS*DATA_W-1:0]     bank_rsp_rdata,
    input  wire [N_BANKS*SRC_ID_W-1:0]   bank_rsp_src_id,
    input  wire [N_BANKS*TXN_ID_W-1:0]   bank_rsp_txn_id,

    // Bank → TAB write-completion
    input  wire [N_BANKS-1:0]            bank_compl_valid,
    input  wire [N_BANKS*SRC_ID_W-1:0]   bank_compl_src_id,
    input  wire [N_BANKS*TXN_ID_W-1:0]   bank_compl_txn_id
);

    localparam BANK_SEL_W  = $clog2(N_BANKS);
    localparam BANK_ADDR_W = ADDR_W - BANK_SEL_W;

    // =========================================================================
    // Per-xPU unpacked signals
    // =========================================================================
    wire [1:0]           req_op_x    [0:N_XPU-1];
    wire [ADDR_W-1:0]    req_addr_x  [0:N_XPU-1];
    wire [DATA_W-1:0]    req_wdata_x [0:N_XPU-1];
    wire [TXN_ID_W-1:0]  req_tid_x   [0:N_XPU-1];
    wire [BANK_SEL_W-1:0] req_bsel_x [0:N_XPU-1];
    wire [BANK_ADDR_W-1:0] req_baddr_x[0:N_XPU-1];

    genvar gi;
    generate
        for (gi = 0; gi < N_XPU; gi = gi + 1) begin : unpack_xpu
            assign req_op_x   [gi] = xpu_req_op   [gi*2      +: 2];
            assign req_addr_x [gi] = xpu_req_addr  [gi*ADDR_W +: ADDR_W];
            assign req_wdata_x[gi] = xpu_req_wdata [gi*DATA_W +: DATA_W];
            assign req_tid_x  [gi] = xpu_req_txn_id[gi*TXN_ID_W +: TXN_ID_W];
            assign req_bsel_x [gi] = xpu_req_addr  [gi*ADDR_W +: BANK_SEL_W];
            assign req_baddr_x[gi] = xpu_req_addr  [gi*ADDR_W+BANK_SEL_W +: BANK_ADDR_W];
        end
    endgenerate

    // =========================================================================
    // Round-robin priority registers, one per bank.
    // Flattened to 1D: rr_prio_flat[bx*N_XPU +: N_XPU] = priority for bank bx.
    // Using flat vectors ensures always @(posedge clk) correctly captures all
    // bits and always @(*) sensitivity lists cover all reads.
    // =========================================================================
    reg [N_XPU*N_BANKS-1:0] rr_prio_flat;

    // =========================================================================
    // Per-bank combinatorial arbiter
    // Flat grant vector: bank_grant_flat[bx*N_XPU +: N_XPU] = grants for bank bx
    // =========================================================================
    reg [N_XPU*N_BANKS-1:0] bank_grant_flat;  // combinatorial
    reg [N_XPU-1:0]         xpu_ready_c;       // combinatorial result

    integer bx, xx, xidx;
    integer prio_start;

    // Helper: extract per-bank priority slice (wire for sensitivity list clarity)
    wire [N_XPU-1:0] rr_prio_b [0:N_BANKS-1];
    genvar gb;
    generate
        for (gb = 0; gb < N_BANKS; gb = gb + 1) begin : unpack_prio
            assign rr_prio_b[gb] = rr_prio_flat[gb*N_XPU +: N_XPU];
        end
    endgenerate

    always @(*) begin
        // Default: nothing granted
        xpu_ready_c    = {N_XPU{1'b0}};
        bank_grant_flat = {(N_XPU*N_BANKS){1'b0}};

        for (bx = 0; bx < N_BANKS; bx = bx + 1) begin
            bank_req_valid [bx]                          = 1'b0;
            bank_req_op    [bx*2      +: 2]              = 2'b00;
            bank_req_addr  [bx*BANK_ADDR_W +: BANK_ADDR_W] = {BANK_ADDR_W{1'b0}};
            bank_req_wdata [bx*DATA_W  +: DATA_W]        = {DATA_W{1'b0}};
            bank_req_src_id[bx*SRC_ID_W +: SRC_ID_W]    = {SRC_ID_W{1'b0}};
            bank_req_txn_id[bx*TXN_ID_W +: TXN_ID_W]    = {TXN_ID_W{1'b0}};
        end

        for (bx = 0; bx < N_BANKS; bx = bx + 1) begin
            if (bank_req_ready[bx]) begin
                prio_start = find_lsb(rr_prio_b[bx]);
                for (xx = 0; xx < N_XPU; xx = xx + 1) begin
                    xidx = (xx + prio_start) % N_XPU;

                    if (!bank_grant_flat[bx*N_XPU + xidx] &&  // bank not yet granted
                        !xpu_ready_c[xidx]              &&  // xPU not granted elsewhere
                        xpu_req_valid[xidx]             &&
                        (req_bsel_x[xidx] == bx[BANK_SEL_W-1:0]))
                    begin
                        // Check that no earlier winner was chosen for this bank
                        // (bank_grant_flat for this bank is still 0 for all bits
                        //  because we only set one bit per bank per cycle)
                        if (!(|bank_grant_flat[bx*N_XPU +: N_XPU])) begin
                            bank_grant_flat[bx*N_XPU + xidx]            = 1'b1;
                            xpu_ready_c[xidx]                            = 1'b1;
                            bank_req_valid [bx]                          = 1'b1;
                            bank_req_op    [bx*2      +: 2]              = req_op_x[xidx];
                            bank_req_addr  [bx*BANK_ADDR_W +: BANK_ADDR_W] = req_baddr_x[xidx];
                            bank_req_wdata [bx*DATA_W  +: DATA_W]        = req_wdata_x[xidx];
                            bank_req_src_id[bx*SRC_ID_W +: SRC_ID_W]    = xidx[SRC_ID_W-1:0];
                            bank_req_txn_id[bx*TXN_ID_W +: TXN_ID_W]    = req_tid_x[xidx];
                        end
                    end
                end
            end
        end
    end

    // xPU ready driven combinatorially
    always @(*) begin
        xpu_req_ready = xpu_ready_c;
    end

    // Round-robin priority advance (registered)
    integer bi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (bi = 0; bi < N_BANKS; bi = bi + 1)
                rr_prio_flat[bi*N_XPU +: N_XPU] <= {{(N_XPU-1){1'b0}}, 1'b1};
        end else begin
            for (bi = 0; bi < N_BANKS; bi = bi + 1) begin
                if (|bank_grant_flat[bi*N_XPU +: N_XPU]) begin
                    // Rotate priority left by one (next xPU gets preference)
                    rr_prio_flat[bi*N_XPU +: N_XPU] <=
                        {rr_prio_flat[bi*N_XPU +: N_XPU-1],
                         rr_prio_flat[bi*N_XPU + N_XPU-1]};
                end
            end
        end
    end

    // =========================================================================
    // Helper function: return binary index of LSB set bit
    // =========================================================================
    function integer find_lsb;
        input [N_XPU-1:0] v;
        integer fi;
        begin
            find_lsb = 0;
            for (fi = N_XPU-1; fi >= 0; fi = fi - 1)
                if (v[fi]) find_lsb = fi;
        end
    endfunction

    // =========================================================================
    // Return path: bank read responses → xPU (demux by src_id)
    // =========================================================================
    integer rx, rb;

    always @(*) begin
        bank_rsp_ready = {N_BANKS{1'b0}};

        for (rx = 0; rx < N_XPU; rx = rx + 1) begin
            xpu_rsp_valid [rx]                     = 1'b0;
            xpu_rsp_type  [rx*2      +: 2]         = 2'b00;
            xpu_rsp_rdata [rx*DATA_W   +: DATA_W]  = {DATA_W{1'b0}};
            xpu_rsp_txn_id[rx*TXN_ID_W +: TXN_ID_W] = {TXN_ID_W{1'b0}};
        end

        // Read responses (higher priority)
        for (rb = 0; rb < N_BANKS; rb = rb + 1) begin
            if (bank_rsp_valid[rb]) begin
                rx = bank_rsp_src_id[rb*SRC_ID_W +: SRC_ID_W];
                if (!xpu_rsp_valid[rx]) begin // first bank wins for this xPU
                    xpu_rsp_valid [rx]                    = 1'b1;
                    xpu_rsp_type  [rx*2      +: 2]        = 2'b00;
                    xpu_rsp_rdata [rx*DATA_W   +: DATA_W] = bank_rsp_rdata [rb*DATA_W   +: DATA_W];
                    xpu_rsp_txn_id[rx*TXN_ID_W +: TXN_ID_W] = bank_rsp_txn_id[rb*TXN_ID_W +: TXN_ID_W];
                    bank_rsp_ready[rb]                    = xpu_rsp_ready[rx];
                end
            end
        end

        // Write completions (lower priority; fill xPUs not already served)
        for (rb = 0; rb < N_BANKS; rb = rb + 1) begin
            if (bank_compl_valid[rb]) begin
                rx = bank_compl_src_id[rb*SRC_ID_W +: SRC_ID_W];
                if (!xpu_rsp_valid[rx]) begin
                    xpu_rsp_valid [rx]                    = 1'b1;
                    xpu_rsp_type  [rx*2      +: 2]        = 2'b01;
                    xpu_rsp_txn_id[rx*TXN_ID_W +: TXN_ID_W] = bank_compl_txn_id[rb*TXN_ID_W +: TXN_ID_W];
                end
            end
        end
    end

endmodule
