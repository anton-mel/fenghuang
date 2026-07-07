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

<img width="1440" height="804" alt="image" src="https://github.com/user-attachments/assets/8b5f722a-1849-4a40-8bf2-d2e0d9b10de4" />
