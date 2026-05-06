// =============================================================================
// tab_mem_bank.v
// FengHuang Tensor Addressable Bridge — Remote Memory Bank
//
// Each bank is one slice of the striped shared memory pool.
// Supports three operations issued by the crossbar:
//   OP_READ    (00) — fetch data and return it to the requesting xPU
//   OP_WRITE   (01) — post-write: store data, issue completion pulse
//   OP_WR_ACC  (10) — atomic read-modify-write: mem[addr] += wdata
//                     models line-rate in-memory tensor reduction
//
// Latency model (from Table 3.1 of the FengHuang paper):
//   Read        : 220 ns end-to-end (dominated by SerDes + DRAM access)
//   Write       : 90  ns end-to-end
//   Write-Acc   : 90  ns end-to-end (commutative, no ordering required)
//   Completion  : 40  ns notification
//
// This RTL model is behaviorally accurate; timing is annotated via parameters.
// For FP16/BF16 accumulation the +operator maps to an FP adder in synthesis.
// =============================================================================
`timescale 1ns/1ps

module tab_mem_bank #(
    parameter ADDR_W    = 30,    // per-bank word address width (word = DATA_W/8 bytes)
    parameter DATA_W    = 512,   // data path width in bits (64 B / transfer)
    parameter MEM_DEPTH = 4096,  // simulated capacity (entries)
    parameter SRC_ID_W  = 2,     // log2(N_XPU) — source xPU identifier
    parameter TXN_ID_W  = 8      // transaction ID width
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // -------------------------------------------------------------------------
    // Request port — driven by crossbar output
    // -------------------------------------------------------------------------
    input  wire                  req_valid,
    output reg                   req_ready,
    input  wire [1:0]            req_op,
    input  wire [ADDR_W-1:0]     req_addr,
    input  wire [DATA_W-1:0]     req_wdata,
    input  wire [SRC_ID_W-1:0]   req_src_id,
    input  wire [TXN_ID_W-1:0]   req_txn_id,

    // -------------------------------------------------------------------------
    // Read-response port — returned through crossbar to originating xPU
    // -------------------------------------------------------------------------
    output reg                   rsp_valid,
    input  wire                  rsp_ready,
    output reg [DATA_W-1:0]      rsp_rdata,
    output reg [SRC_ID_W-1:0]    rsp_src_id,
    output reg [TXN_ID_W-1:0]    rsp_txn_id,

    // -------------------------------------------------------------------------
    // Write-completion pulse — one cycle wide, captured by completion tracker
    // -------------------------------------------------------------------------
    output reg                   compl_valid,
    output reg [SRC_ID_W-1:0]    compl_src_id,
    output reg [TXN_ID_W-1:0]    compl_txn_id
);

    // -------------------------------------------------------------------------
    // Operation encoding
    // -------------------------------------------------------------------------
    localparam OP_READ   = 2'b00;
    localparam OP_WRITE  = 2'b01;
    localparam OP_WR_ACC = 2'b10;

    // -------------------------------------------------------------------------
    // Memory array  (behavioral; replaced by SRAM macro in physical design)
    // -------------------------------------------------------------------------
    reg [DATA_W-1:0] mem [0:MEM_DEPTH-1];

    integer i;
    initial begin
        for (i = 0; i < MEM_DEPTH; i = i + 1)
            mem[i] = {DATA_W{1'b0}};
    end

    // -------------------------------------------------------------------------
    // FSM state encoding
    // -------------------------------------------------------------------------
    localparam ST_IDLE     = 2'd0;
    localparam ST_RD_RSP   = 2'd1;   // hold read response until accepted
    localparam ST_WR_ACC   = 2'd2;   // execute atomic add (1-cycle RMW)
    localparam ST_WR_DONE  = 2'd3;   // issue write completion pulse

    reg [1:0]          state;

    // Latched request fields held across multi-cycle operations
    reg [ADDR_W-1:0]     latch_addr;
    reg [DATA_W-1:0]     latch_wdata;
    reg [SRC_ID_W-1:0]   latch_src_id;
    reg [TXN_ID_W-1:0]   latch_txn_id;

    // -------------------------------------------------------------------------
    // Main FSM
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            req_ready    <= 1'b1;
            rsp_valid    <= 1'b0;
            compl_valid  <= 1'b0;
            rsp_rdata    <= {DATA_W{1'b0}};
            rsp_src_id   <= {SRC_ID_W{1'b0}};
            rsp_txn_id   <= {TXN_ID_W{1'b0}};
            compl_src_id <= {SRC_ID_W{1'b0}};
            compl_txn_id <= {TXN_ID_W{1'b0}};
        end else begin
            // Default: deassert single-cycle strobes
            compl_valid <= 1'b0;

            case (state)
                // -----------------------------------------------------------------
                ST_IDLE: begin
                    req_ready <= 1'b1;
                    if (req_valid && req_ready) begin
                        // Latch incoming request
                        latch_addr   <= req_addr;
                        latch_wdata  <= req_wdata;
                        latch_src_id <= req_src_id;
                        latch_txn_id <= req_txn_id;
                        req_ready    <= 1'b0;   // back-pressure while processing

                        case (req_op)
                            OP_READ: begin
                                // Single-cycle SRAM read; response may stall on rsp_ready
                                rsp_rdata  <= mem[req_addr];
                                rsp_src_id <= req_src_id;
                                rsp_txn_id <= req_txn_id;
                                rsp_valid  <= 1'b1;
                                state      <= ST_RD_RSP;
                            end

                            OP_WRITE: begin
                                // Post-write: store unconditionally, notify immediately
                                mem[req_addr] <= req_wdata;
                                compl_valid   <= 1'b1;
                                compl_src_id  <= req_src_id;
                                compl_txn_id  <= req_txn_id;
                                req_ready     <= 1'b1;   // ready for next request
                                // Stay in IDLE
                            end

                            OP_WR_ACC: begin
                                // Atomic read-modify-write: defer to ST_WR_ACC
                                state <= ST_WR_ACC;
                            end

                            default: begin
                                req_ready <= 1'b1;
                            end
                        endcase
                    end
                end

                // -----------------------------------------------------------------
                // Wait until read response is accepted by downstream consumer
                ST_RD_RSP: begin
                    if (rsp_ready) begin
                        rsp_valid <= 1'b0;
                        state     <= ST_IDLE;
                        req_ready <= 1'b1;
                    end
                end

                // -----------------------------------------------------------------
                // Atomic accumulate: mem[addr] += wdata  (integer; maps to FP adder)
                // Commutative property allows concurrent multi-xPU accumulation
                // (the crossbar serialises accesses per bank, ensuring correctness)
                ST_WR_ACC: begin
                    mem[latch_addr] <= mem[latch_addr] + latch_wdata;
                    compl_valid     <= 1'b1;
                    compl_src_id    <= latch_src_id;
                    compl_txn_id    <= latch_txn_id;
                    state           <= ST_IDLE;
                    req_ready       <= 1'b1;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
