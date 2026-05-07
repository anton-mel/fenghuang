# FengHuang Tensor Addresable Bridge

Implementation of the Tensor Addressable Bridge from  
_"FengHuang: Next-Generation Memory Orchestration for AI Inferencing"_, Microsoft Research, arXiv 2511.10753v1.

For our final project for CPSC 4261 (Spring 2026, Prof. Richard Yang), we implemented FengHuang, a hardware-software co-design proposal from
Microsoft Research Asia for next-generation AI inference infrastructure. FengHuang's core primitive is the Tensor Addressable Bridge (TAB), a
shared remote-memory fabric that decouples model-weight storage from per-GPU HBM, streaming weights and KV-cache on demand to a pool of LPDDR6
banks over 224G/448G SerDes links. We implemented the TAB as four synthesizable Verilog modules (top-level integrator, round-robin crossbar,
banked memory, and global completion tracker), built a roofline-based Python simulator whose bandwidth-sensitivity curve reproduces the paper's
Figure 4.2 to within 1% at the 4.8 TB/s design point, and profiled Qwen3-235B-A22B on Yale's 8-GPU H200 cluster. In the accompanying report we
discuss the challenges surfaced during implementation and empirical evaluation and propose a redesign that addresses them.

#### Team Members

Anton Melnychuk, am3785, anton.melnychuk@yale.edu

<img width="844" height="170" alt="image" src="https://github.com/user-attachments/assets/fcbdcbbe-9cf3-406b-9694-7aeb8e87dcc9" />

</br>

<img width="1234" height="316" alt="image" src="https://github.com/user-attachments/assets/71299b60-eb6c-4485-88ef-4ff4e3b63b1f" />


---

### RTL
- `tab_mem_bank.v` - memory bank FSM (OP_READ / OP_WRITE / OP_WR_ACC)
- `tab_crossbar.v` - N×M crossbar with round-robin arbitration
- `tab_compl_tracker.v` - pending write counter + WC_SYNC notification
- `fenghuang_tab.v` - all submodules
- `tb/tb_fenghuang_tab.v` - testbench (P2P, AllReduce, AllGather)
- All 4 tests passing under iverilog

Run this to see output:
```bash
./sim/run.sh
```

### Nsight Simulation
- Profiled Qwen3-235B-A22B on Yale cluster (H100 80GB × 8) with Nsight Systems
- Extracted GPU utilization and compute/overhead breakdown from trace
- Built Python simulator: TAB bandwidth prefetch model (simulator_v2.py)
  - Model: paper scenario (20 GB local HBM, 38.75 GB remote per GPU)
  - Sweep: 4.0 => 6.4 TB/s remote bandwidth
- Reproduced Figure 4.2 from paper: see `sim/results/comparison_final.png`
  - Our sim @ 4.8 TB/s: −18.2% TPOT vs Baseline8 (paper: −19.2%)

### Write-Up
- Section 1: RTL implementation
- Section 2: Specification gaps found
- Section 3: Nsight simulation methodology and results
- Section 4: Comparison to paper's claims
- Section 5: Proposed Redesign

---

## Hardware Setups

### 1. Paper Baseline (Figure 3.8)
What the paper compares FengHuang against:
- GPUs: 8 × H200 (141 GB HBM each)
- Total memory:** 1152 GB HBM across all GPUs
- HBM bandwidth:** 38.4 TB/s aggregate
- Inter-GPU:** NVLink 4.0, 900 GB/s bidirectional per GPU
- Parallelism:** TP=8 FP8
- TPOT (Qwen3-235B, batch=8):** ~780 µs

### 2. Paper FengHuang Simulation Setup (Figure 3.8)
The hypothetical FengHuang system the paper models:
- GPUs: N × H200 with only 20 GB local HBM per GPU
- Remote memory: 144 GB × N LPDDR6 shared across all GPUs via TAB
- TAB bandwidth: 4.8 TB/s per GPU, full duplex
- Parallelism: TP=4 FP8 (FH4 config)
- TPOT (Qwen3-235B, batch=8, 4.8 TB/s):** ~630 µs

### 3. Our Cluster Setup
- GPUs: 8 × H200 SXM5 80 GB HBM each
- Total memory: 640 GB HBM across all GPUs
- Inter-GPU: NVLink
- Software: SGLang inference server
- Parallelism: TP=4 + DP=2, BF16 weights

---

## Simulation

```bash
iverilog -g2005-sv -ofenghuang_sim \
  rtl/tab_mem_bank.v rtl/tab_crossbar.v rtl/tab_compl_tracker.v \
  rtl/fenghuang_tab.v tb/tb_fenghuang_tab.v
vvp fenghuang_sim
```

Expected output:
```
[PASS] P2P_READ
[PASS] AllReduce_result  got=0xa0
[PASS] U0_data_via_xPU1
[PASS] U3_data_via_xPU0
ALL TESTS PASSED
```

---

## FPGA Synthesys

| Operation | Latency |
|---|---|
| Read | 220 ns |
| Write | 90 ns |
| Write-Accumulate | 90 ns |
| WC-Sync notification | 40 ns |
