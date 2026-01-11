# 🚀 20-Day GPU, CUDA, and Triton Challenge

This repository tracks my journey through an intensive 20-day challenge focused on GPU programming, ranging from basic vector operations to advanced implementations like **Flash Attention 2 (FA2)** and **Rotary Positional Encodings (RoPE)**.

## 📁 Repository Structure

I am following a modular structure to keep kernels, PyTorch bindings, and testing scripts organized.

```text
.
├── common/                # Shared utilities (error checking, timers)
│   └── helper.cuh         # CUDA error checking macros
├── day01-addition/        # Simple vector addition
├── day06-pytorch-ext/     # First PyTorch + CUDA JIT integration
├── day14-flash-forward/   # FA2 Forward Pass implementation
├── day20-final-tasks/     # FA2 Backward & RoPE
├── scripts/               # Global build or benchmarking scripts
└── README.md              # Progress tracking and notes

```

---

## 📅 Challenge Roadmap & Progress

| Day | Task / Topic | Status | Key Files |
| --- | --- | --- | --- |
| **D1** | **Vector Addition**: Basics of memory allocation & 1D indexing. | 🟢 | `addition.cu` |
| **D2** | **Device Functions**: Using `__device__` for per-thread math. | ⚪ |  |
| **D3** | **Matrix Addition**: 2D indexing & row/column mapping. | ⚪ |  |
| **D4** | **Layer Norm**: Shared memory & mean/variance reduction. | ⚪ |  |
| **D5** | **Vector Reductions**: Parallel sum & optimization tricks. | ⚪ |  |
| **D6** | **PyTorch Integration**: Building CUDA extensions with `cpp_extension`. | ⚪ |  |
| **D7** | **Tiled MatMul**: Shared memory tiling to improve throughput. | ⚪ |  |
| **D8** | **Self-Attention**: Naive self-attention with online softmax. | ⚪ |  |
| **D9** | **Minimal Flash Attention**: Tiled implementation for memory efficiency. | ⚪ |  |
| **D10** | **Linker & Build Systems**: Managing complex CUDA/C++ builds. | ⚪ |  |
| **D11** | **Backwards Benchmarking**: Comparing kernel grads vs. PyTorch Autograd. | ⚪ |  |
| **D12** | **Advanced Softmax**: Shared memory & Warp-level primitives. | ⚪ |  |
| **D13** | **RMSNorm**: Optimization with `float4` and warp-reduce. | ⚪ |  |
| **D14** | **2D Convolution**: Shared memory tiling for image processing. | ⚪ |  |
| **D15** | ⭐ **Mandatory: FA2 Forward**: Implement Forward Pass for FA2. | ⚪ |  |
| **D16** | **Attention Gradients**: Extending attention with backward logic. | ⚪ |  |
| **D17** | **cuBLAS**: Integrating vendor-optimized BLAS libraries. | ⚪ |  |
| **D18** | **Atomics & PTX**: Inline PTX for warp-based reduction. | ⚪ |  |
| **D19** | **Fused Kernels**: Combining cuBLAS with custom attention logic. | ⚪ |  |
| **D20** | ⭐ **Mandatory: FA2 Backward**: Complete Gradient computation. | ⚪ |  |

> **Optional Bonus:** Fused Chunked Cross-Entropy Loss + Backwards.

---

## 🛠️ Tech Stack & Requirements

* **Language:** C++ / CUDA / Python
* **Libraries:** PyTorch (LibTorch), Ninja, cuBLAS
* **Hardware:** NVIDIA GPU (Compute Capability 7.0+ recommended)
* **Compiler:** `nvcc` (CUDA Toolkit)

### Environment Setup

1. **CUDA Toolkit:** Ensure `nvcc --version` is accessible.
2. **Python Deps:** `pip install torch ninja numpy`
3. **Error Checking:** Use the macros in `common/helper.cuh` for all kernels.

---

## 📝 Learning Notes

* **Memory Hierarchy:** Understanding the latency difference between Global Memory (HBM) and Shared Memory (SRAM) is the key to FA2.
* **Kernel Fusion:** Reducing memory trips by combining operations (e.g., Softmax + MatMul) provides the biggest speedups.

---

**Would you like me to generate the `common/helper.cuh` file next so you have the error-checking macros ready for Day 1?**
