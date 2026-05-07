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
cd rtl && ./run.sh
```

### Nsight Simulation
- Profiled Qwen3-235B-A22B on Yale cluster (H100 80GB × 8) with Nsight Systems
- Extracted GPU utilization and compute/overhead breakdown from trace
- Built Python simulator: TAB bandwidth prefetch model (simulator_v2.py)
  - Model: paper scenario (20 GB local HBM, 38.75 GB remote per GPU)
  - Sweep: 4.0 => 6.4 TB/s remote bandwidth
- Reproduced Figure 4.2 from paper: see `sim/results/comparison_final.png`
  - Our sim @ 4.8 TB/s: −18.2% TPOT vs Baseline8 (paper: −19.2%)

Run command (requires nsys `.sqlite` traces from the Yale cluster, not checked in — pre-computed outputs in `sim/results/*.json`):
```bash
python sim/simulator_v2.py \
  --baseline-trace trace_baseline8.sqlite \
  --fh4-trace      trace_fh4_qa.sqlite \
  --bw-sweep 4.0 4.8 6.4 \
  --output sim/results/results_final.json
```

### Write-Up
1. Introduction
2. Problem Statement
3. Prefetch Simulation
4. Design and RTL Implementation
5. Architectural Gaps and Failure Modes
6. TAB Hybrid Redesign
7. Conclusion

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
