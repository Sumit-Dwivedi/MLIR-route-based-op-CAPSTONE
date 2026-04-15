# Adaptive MLIR MatMul Optimization Framework

A shape-aware, hardware-targeted optimization pipeline for `linalg.matmul` operations. This framework transforms high-level tensor contractions into highly optimized, vectorized native code using the MLIR (Multi-Level Intermediate Representation) ecosystem.

## 🚀 Overview

The **Adaptive MLIR MatMul Framework** is designed to solve the "one size fits all" problem in matrix multiplication. Instead of applying a single tiling strategy to every operation, it uses an **Adaptive Router** to inspect matrix shapes (static or dynamic) and hardware features (AVX2, AVX-512, GPU) to dispatch the most efficient optimization strategy.

### Key Features

*   **Adaptive Routing:** Automatically selects between `Square`, `Skinny`, `Small`, or `GPU` strategies.
*   **Dynamic Shape Support:** Emits runtime dispatch (`scf.if`) for matrices whose dimensions are unknown at compile time.
*   **3-Level Cache Blocking:** Implements L2, L1, and Register-level tiling to maximize cache locality and SIMD throughput.
*   **INT8 Quantization Support:** Automatically detects INT8/I32 patterns and applies 4x vector scaling for integer SIMD instructions.
*   **GPU Offloading:** Identifies massive compute workloads and prepares them for GPU execution via `scf.forall` mapping.
*   **End-to-End JIT Execution:** Includes a custom OrcJIT runner with automated benchmarking and correctness verification.

---

## 🔥 Why This Project? (The "Adaptive Edge")

Unlike general-purpose libraries (TensorFlow/XLA, PyTorch Inductor) that use fixed heuristics or vendor black-boxes (cuBLAS, MKL), this framework functions as a **precision optimization tool**:

1.  **Shape-Aware Routing:** Instead of a "one-size-fits-all" tiling, it selects specialized strategies for **Skinny**, **Square**, or **Small** matrices to maximize cache utilization.
2.  **AOT Polyhedral Fallback:** For **Dynamic Shapes**, it generates compile-time multi-versioned code with runtime dispatch (`scf.if`), providing optimized vectorized paths without the overhead of runtime recompilation.
3.  **Datatype-Aware Scaling:** Automatically scales vectorization factors (e.g., **4x scaling for INT8**) to saturate hardware-specific instructions (AVX-512 VNNI).
4.  **Standalone & Transparent:** Provides production-grade optimizations (3-level blocking, fusion, SIMD) in a lightweight standalone tool, making it ideal for specialized performance tuning and research.

---

## 👥 Target Audience

*   **Compiler Engineers:** Exploring MLIR-based codegen and transformation pipelines.
*   **MLIR Infrastructure Developers:** Looking for reference implementations of tile-and-fuse, vectorization, and dialect lowering.
*   **High-Performance Computing (HPC) Researchers:** Benchmarking automated vs. manual optimization strategies on modern x86 and GPU hardware.

---

## 💡 How to Use (Sample Cases)

The framework is invoked through `adaptive-opt`, a custom tool that applies the optimization pipeline.

### Case 1: Square Matrix (Standard Blocking)
Optimizes a 1024x1024 matrix using 3-level cache-aware blocking.
```bash
./build/adaptive-opt examples/square_matmul.mlir \
  --kernel-fusion --adaptive-router --square-blocking \
  --simd-vectorization --lower-to-llvm --jit-run
```

### Case 2: Skinny Matrix (Tall/Wide Tiling)
Optimizes matrices with extreme aspect ratios (e.g., 4096x32) by tiling only the dominant dimensions.
```bash
./build/adaptive-opt examples/skinny_matmul.mlir \
  --kernel-fusion --adaptive-router --skinny-tiling \
  --simd-vectorization --lower-to-llvm --jit-run
```

### Case 3: Dynamic Shapes (Runtime Dispatch)
For matrices like `tensor<?x?xf32>`, the router generates code that decides the strategy at runtime.
```bash
ADAPTIVE_M=512 ADAPTIVE_N=512 ADAPTIVE_K=512 \
  ./build/adaptive-opt tests/router_dynamic_dispatch.mlir \
  --adaptive-router --lower-to-llvm --jit-run
```

### Case 4: GPU Offload
Massive matrices (>10^8 FLOPs) are automatically prepared for GPU execution.
```bash
./build/adaptive-opt tests/gpu_offload_test.mlir \
  --adaptive-router --gpu-offload --lower-to-llvm
```

---

## 🛠️ Step-by-Step Guide to Run

### 1. Prerequisites
Ensure you have the following installed (Ubuntu 22.04+ recommended):
```bash
sudo apt update && sudo apt install -y \
  build-essential cmake ninja-build git python3 lld clang ccache
```

### 2. Build LLVM/MLIR from Source
This framework requires a custom build of the LLVM Project.
```bash
git clone https://github.com/llvm/llvm-project.git
cmake -G Ninja -S llvm-project/llvm -B mlir-build \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_INSTALL_UTILS=ON
cmake --build mlir-build -j$(nproc)
```

### 3. Build the Adaptive Framework
1.  Clone this repository.
2.  Edit `mlir-adaptive-matmul/CMakeLists.txt` to set `MLIR_DIR` and `LLVM_DIR` to your `mlir-build` paths.
3.  Build:
```bash
cd mlir-adaptive-matmul
mkdir build && cd build
cmake .. -G Ninja
ninja
```

### 4. Run a Benchmark
Execute a standard 512x512 matrix multiplication with performance metrics:
```bash
ADAPTIVE_M=512 ADAPTIVE_N=512 ADAPTIVE_K=512 ADAPTIVE_RUNS=10 \
  ./build/adaptive-opt ../examples/square_matmul.mlir \
  --kernel-fusion --adaptive-router --square-blocking \
  --simd-vectorization --lower-to-llvm --jit-run
```

---

## 📂 Project Structure

*   `mlir-adaptive-matmul/src/router/`: Strategy dispatch and dynamic dispatch logic.
*   `mlir-adaptive-matmul/src/generic_passes/`: 3-level blocking, SIMD vectorization, and GPU offload.
*   `mlir-adaptive-matmul/src/lowering/`: 17-step lowering pipeline to LLVM dialect.
*   `mlir-adaptive-matmul/src/jit/`: OrcJIT runner, aligned memory allocation, and stack hardening.
*   `docs/`: Deep-dives into architecture, workflow, and systems requirements.
