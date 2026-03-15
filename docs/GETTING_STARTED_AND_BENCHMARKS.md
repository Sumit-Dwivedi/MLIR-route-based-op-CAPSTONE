# Getting Started & Benchmarking Guide

**Framework:** Adaptive MLIR MatMul Optimization Framework
**Target Environment:** Ubuntu 22.04+ on WSL2 (also works on native Linux)
**Architecture:** x86_64 (AVX2 or AVX-512 required for vectorized paths)

---

## 1. Prerequisites

### 1a. System Packages

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake ninja-build git python3 python3-pip python3-venv \
  lld clang ccache zlib1g-dev libncurses-dev libxml2-dev
```

| Package | Purpose |
|---------|---------|
| `cmake` (≥ 3.10) | Build system for both LLVM and the project |
| `ninja-build` | Parallel build driver — 3-5× faster than Make for LLVM |
| `lld` | LLVM linker — required for linking LLVM/MLIR (GNU ld runs out of memory) |
| `clang` | Recommended compiler for building LLVM (GCC works but is slower) |
| `ccache` | Compiler cache — essential for iterative LLVM rebuilds |
| `python3` + `pip` | Required for `lit` (LLVM Integrated Tester) and `FileCheck` |

### 1b. Hardware Verification

Verify your CPU supports AVX2 (minimum) or AVX-512 (optimal):

```bash
grep -o 'avx2\|avx512f' /proc/cpuinfo | sort -u
```

Expected output for AVX2: `avx2`
Expected output for AVX-512: `avx2` and `avx512f`

If neither flag appears, the framework will still compile but the JIT-compiled code will use SSE2 scalar fallback (significantly slower).

### 1c. WSL2-Specific Configuration

WSL2's default memory limit may be insufficient for building LLVM. Create or edit `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=16GB
swap=8GB
processors=8
```

Restart WSL after editing: `wsl --shutdown` from PowerShell.

---

## 2. Building LLVM/MLIR from Source

The framework links against MLIR libraries that are not distributed in binary packages. You must build LLVM/MLIR from source.

### 2a. Clone LLVM

```bash
cd ~
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout main  # or a specific release tag (e.g., llvmorg-19.1.0)
```

### 2b. Configure the Build

```bash
cmake -G Ninja -S llvm -B ../mlir-build \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_USE_LINKER=lld \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
```

**Critical flags:**
- `LLVM_ENABLE_PROJECTS="mlir"` — builds MLIR alongside LLVM
- `LLVM_TARGETS_TO_BUILD="X86"` — reduces build time by skipping ARM/RISC-V/etc.
- `LLVM_USE_LINKER=lld` — prevents OOM during linking (GNU ld requires 32GB+ for LLVM)
- `LLVM_INSTALL_UTILS=ON` — installs `FileCheck` and `not` utilities needed for lit tests

### 2c. Build

```bash
cmake --build ../mlir-build -- -j$(nproc)
```

**Expected build time:** 30-90 minutes depending on CPU and available RAM. With `ccache`, incremental rebuilds take < 5 minutes.

**Expected disk usage:** ~20GB for the build directory.

### 2d. Verify the Build

```bash
~/mlir-build/bin/mlir-opt --version
~/mlir-build/bin/FileCheck --version
ls ~/mlir-build/lib/libmlir_c_runner_utils.so
```

All three commands must succeed. The shared library `libmlir_c_runner_utils.so` is loaded at runtime by the JIT runner.

---

## 3. Building the Adaptive MatMul Framework

### 3a. Clone the Repository

```bash
cd ~
git clone <repository-url> mlir-capstone
cd mlir-capstone/mlir-adaptive-matmul
```

### 3b. Update Build Paths

Edit `CMakeLists.txt` lines 8-9 to point to your MLIR/LLVM build:

```cmake
set(MLIR_DIR "/home/<your-username>/mlir-build/lib/cmake/mlir")
set(LLVM_DIR "/home/<your-username>/mlir-build/lib/cmake/llvm")
```

### 3c. Build

```bash
mkdir -p build && cd build
cmake .. -G Ninja
ninja
```

**Expected output:** The `adaptive-opt` binary at `build/adaptive-opt`.

### 3d. Verify the Build

```bash
./adaptive-opt --help 2>&1 | grep -E "adaptive-router|square-blocking|jit-run"
```

Expected: Three lines showing the custom pass descriptions.

---

## 4. Running the Pipeline

### 4a. Basic Invocation

The full pipeline is invoked as a sequence of pass flags on `adaptive-opt`:

```bash
./build/adaptive-opt input.mlir \
  --kernel-fusion \
  --adaptive-router \
  --square-blocking \
  --simd-vectorization \
  --lower-to-llvm \
  --jit-run
```

**Pass ordering is critical.** The flags must appear in this exact sequence:

1. `--kernel-fusion` — fuse elementwise chains before tiling
2. `--adaptive-router` — classify shapes and assign strategy attributes
3. `--square-blocking` (or `--skinny-tiling`, `--gpu-offload`) — strategy-specific tiling
4. `--simd-vectorization` — linalg → vector
5. `--lower-to-llvm` — vector → LLVM dialect
6. `--jit-run` — JIT compile and execute

### 4b. Generating Input MLIR

A minimal tensor-semantic matmul input:

```mlir
// file: matmul_256.mlir
func.func @matmul(%A: tensor<256x256xf32>, %B: tensor<256x256xf32>,
                   %C: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %result = linalg.matmul ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>)
                          outs(%C : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}
```

### 4c. Debugging IR Transformations

Use MLIR's built-in IR printing to inspect the output of each pass:

```bash
./build/adaptive-opt input.mlir \
  --kernel-fusion \
  --adaptive-router \
  --print-ir-after=adaptive-router \
  2>&1 | head -100
```

Or print after every pass:

```bash
./build/adaptive-opt input.mlir \
  --kernel-fusion --adaptive-router --square-blocking \
  --print-ir-after-all 2>&1 | less
```

---

## 5. Running the Test Suite

### 5a. Python Environment for `lit`

```bash
cd ~/mlir-capstone
python3 -m venv compiler-env
source compiler-env/bin/activate
pip install lit
```

### 5b. `lit` Configuration

The test suite lives in `mlir-adaptive-matmul/tests/`. The `lit.cfg.py` configures:
- The `adaptive-opt` binary path
- The `FileCheck` binary path (from the MLIR build)
- File extensions to scan (`.mlir`)

Verify `FileCheck` is accessible:

```bash
~/mlir-build/bin/FileCheck --version
```

### 5c. Running the Tests

```bash
source ~/mlir-capstone/compiler-env/bin/activate

lit mlir-adaptive-matmul/tests/ \
  --path ~/mlir-build/bin \
  --path ~/mlir-capstone/mlir-adaptive-matmul/build \
  -v
```

**Expected output:**

```
PASS: router_dynamic_dispatch.mlir
PASS: quantized_matmul_test.mlir
PASS: gpu_offload_test.mlir
```

### 5d. Test Descriptions

| Test | What It Validates |
|------|-------------------|
| `router_dynamic_dispatch.mlir` | `scf.if` emission for dynamic shapes, `tensor.dim` → `arith.cmpi sgt` → branch, `optimization_strategy` attributes on both branches |
| `quantized_matmul_test.mlir` | INT8 datatype detection (`i8` inputs + `i32` accumulator → `adaptive.datatype = "int8"`), 4× vector scaling on K dimension |
| `gpu_offload_test.mlir` | `scf.forall` with `#gpu.block<y>`, `#gpu.block<x>` mapping, 64×64 tile grid computation, threshold gating (4096³ → GPU, 64³ → square) |

---

## 6. Benchmarking

### 6a. Environment Variables

The JIT runner is parameterized entirely via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ADAPTIVE_M` | 256 | M dimension of the matmul |
| `ADAPTIVE_N` | 256 | N dimension (must match MLIR input or use dynamic shapes) |
| `ADAPTIVE_K` | 256 | K dimension |
| `ADAPTIVE_RUNS` | 10 | Number of timed benchmark iterations (after warmup) |
| `ADAPTIVE_JIT_OPT` | 2 | LLVM optimization level (0-3) for JIT-compiled code |
| `ADAPTIVE_GPU_TARGET` | *(empty)* | GPU architecture string (e.g., `sm_80`) — enables GPU routing |
| `MLIR_RUNNER_UTILS` | `/home/sumit/mlir-build/lib/libmlir_c_runner_utils.so` | Path to MLIR runtime utilities shared library |

### 6b. Running a Single Benchmark

```bash
cd ~/mlir-capstone/mlir-adaptive-matmul

ADAPTIVE_M=512 ADAPTIVE_N=512 ADAPTIVE_K=512 ADAPTIVE_RUNS=20 \
  ./build/adaptive-opt examples/square_matmul.mlir \
  --kernel-fusion --adaptive-router --square-blocking \
  --simd-vectorization --lower-to-llvm --jit-run
```

**Expected output:**

```
[JITRunner] Entry function: matmul
[JITRunner] Detected 3 memref arg(s): 2D, 2D, 2D
[JITRunner] Matrix: 512x512 * 512x512 (20 runs)
[JITRunner] Results (20 runs):
  Median:  72.817 ms
  Mean:    73.452 ms
  Min:     71.203 ms
  Max:     78.901 ms
  GFLOPS:  3.69
  Correctness: PASS (5/5 verified, all = 512.0)
```

### 6c. Running the Full Benchmark Sweep

The automated benchmark harness iterates over standard matrix sizes:

```bash
cd ~/mlir-capstone/mlir-adaptive-matmul
chmod +x tests/run_benchmark_sweep.sh
./tests/run_benchmark_sweep.sh
```

This executes the pipeline for 32×32, 64×64, 256×256, 512×512, and 1024×1024, collecting correctness and performance data for each.

### 6d. Reference Performance Numbers

Measured on x86_64 / AVX2 (256-bit SIMD) / WSL2 Linux 6.6.87, LLVM O2 optimization:

| Shape | Strategy | Median (ms) | GFLOPS | Correctness |
|-------|----------|------------|--------|-------------|
| 32×32 | small | 0.022 | 3.00 | PASS (5/5) |
| 64×64 | square | 0.122 | 4.31 | PASS (5/5) |
| 256×256 | square | 8.156 | 4.18 | PASS (5/5) |
| 512×512 | square | 72.817 | 3.69 | PASS (5/5) |
| 1024×1024 | square | 668.252 | 3.21 | PASS (5/5) |

**Theoretical peak (single-core AVX2 FMA at 3.0 GHz):** 2 × 8 × 3.0 = 48 GFLOPS. The framework achieves ~4 GFLOPS (8.3% of peak), which is expected for a research-grade compiler without register-level scheduling, software pipelining, or micro-kernel assembly.

### 6e. Tuning the JIT Optimization Level

The LLVM optimization level significantly affects both compilation time and execution speed:

```bash
# No optimization (fastest compilation, slowest execution)
ADAPTIVE_JIT_OPT=0 ADAPTIVE_M=256 ADAPTIVE_N=256 ADAPTIVE_K=256 \
  ./build/adaptive-opt examples/square_matmul.mlir \
  --kernel-fusion --adaptive-router --square-blocking \
  --simd-vectorization --lower-to-llvm --jit-run

# Maximum optimization (slowest compilation, fastest execution)
ADAPTIVE_JIT_OPT=3 ADAPTIVE_M=256 ADAPTIVE_N=256 ADAPTIVE_K=256 \
  ./build/adaptive-opt examples/square_matmul.mlir \
  --kernel-fusion --adaptive-router --square-blocking \
  --simd-vectorization --lower-to-llvm --jit-run
```

O2 (default) provides the best tradeoff for benchmarking. O3 enables auto-vectorization of scalar residuals but increases JIT compilation time by 2-3×.

---

## 7. Low-Level Systems Requirements for JIT Execution

### 7a. 64-Byte Aligned Memory Allocation

**Requirement:** All input and output buffers passed to JIT-compiled code must be 64-byte aligned.

**Root cause:** The vectorized matmul micro-kernel lowers to AVX2 `vmovaps` (aligned packed single-precision move) and AVX-512 `vmovaps` / `vmovdqa64` instructions. These instructions require the memory operand address to be aligned to the vector width:

| ISA | Instruction | Required Alignment |
|-----|------------|-------------------|
| AVX2 | `vmovaps ymm, [mem]` | 32 bytes |
| AVX-512 | `vmovaps zmm, [mem]` | 64 bytes |
| SSE | `movaps xmm, [mem]` | 16 bytes |

A misaligned access triggers a `#GP` exception (x86 General Protection fault), delivered as `SIGSEGV` to the process.

**Implementation:** The JIT runner uses `posix_memalign`:

```c++
constexpr size_t kAlignment = 64;
void *ptr = nullptr;
if (posix_memalign(&ptr, kAlignment, count * sizeof(float)) != 0)
  return nullptr;
```

**Why not `std::vector<float>`:** The C++ standard guarantees `std::vector` allocates with `alignof(float)` = 4 bytes. Some implementations over-align to 16 bytes (SSE), but none guarantee 32 or 64 bytes. Using `std::vector` with AVX2/AVX-512 vectorized code causes intermittent `SIGSEGV` — intermittent because `malloc` may happen to return an aligned address by chance.

**Why not `aligned_alloc` (C11):** `aligned_alloc` requires `size` to be a multiple of `alignment`. For non-round buffer sizes (e.g., 127 × 4 = 508 bytes, not a multiple of 64), `aligned_alloc` returns `nullptr` or invokes undefined behavior. `posix_memalign` has no such restriction.

**Cleanup:** Buffers allocated with `posix_memalign` must be freed with `free()` (not `delete` or `delete[]`).

### 7b. Stack Limit Elevation via `setrlimit`

**Requirement:** The main thread's stack soft limit must be raised before executing JIT-compiled code for matrices ≥ 320×320.

**Root cause analysis:**

Three-level tiling + vectorization produces deeply nested LLVM IR. For a 512×512 matrix with 64/64/8 blocking:

```
Loop nest depth: 6 (M_l2 × N_l2 × K_l1 × M_reg × N_reg × K_reg)
Vector spills per micro-kernel: ~32 × vector<8xf32> = 1KB
Spill slots across loop nest: ~32 × 64 iterations = ~2MB
Plus: alloca for memref descriptors, loop variables, intermediate results
Estimated total stack: 12-20MB for 512×512
```

The default Linux soft stack limit is 8MB (`ulimit -s` = 8192). When the JIT-compiled code's stack usage exceeds this, the kernel delivers `SIGSEGV` (stack overflow).

**Implementation:**

```c++
#include <sys/resource.h>

struct rlimit rl;
if (getrlimit(RLIMIT_STACK, &rl) == 0) {
  rl.rlim_cur = rl.rlim_max;  // raise soft limit to hard limit
  setrlimit(RLIMIT_STACK, &rl);
}
```

**How it works:** The Linux kernel auto-grows the main thread's stack on demand. When the process accesses a stack page beyond the current allocation, the kernel intercepts the page fault and extends the stack mapping — but only up to `RLIMIT_STACK` (soft limit). By raising the soft limit to the hard limit (typically 64MB or `RLIM_INFINITY`), the kernel is permitted to auto-grow the stack as needed.

**Why this does NOT work for pthread stacks:** Stacks allocated by `pthread_create` use `mmap` with a fixed size and guard page. The guard page causes immediate `SIGSEGV` on overflow — there is no auto-growth mechanism. Additionally, deeply nested JIT code may access stack memory non-sequentially (e.g., jumping from the top of a large alloca to the bottom), skipping the guard page entirely (stack clash vulnerability). The main thread's stack does not have this limitation because the kernel handles its growth at the page fault level, regardless of access pattern.

**Verification:** To confirm the stack limit is sufficient:

```bash
# Check current limits
ulimit -s      # soft limit (KB)
ulimit -Hs     # hard limit (KB)

# Temporarily set unlimited (for debugging)
ulimit -s unlimited
```

**Fallback for restricted environments:** If `setrlimit` fails (e.g., hard limit is also 8MB), reduce the register tile sizes in `SquareBlockingPass.cpp`:

```c++
// In computeBlockingSizes():
bs.regM = 4;  bs.regN = 4;  bs.regK = 4;  // Smaller micro-kernel = less stack
```

This reduces vectorization quality but prevents stack overflow.

---

## 8. Troubleshooting

### 8a. Build Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `undefined reference to mlir::linalg::tileLinalgOp` | MLIR build missing Linalg transforms | Rebuild LLVM with `-DLLVM_ENABLE_PROJECTS="mlir"` |
| `error: use of undeclared identifier 'gpu'` | GPU dialect not registered | Verify `MLIRGPUDialect` is in `target_link_libraries` in CMakeLists.txt |
| Linker OOM (killed) | GNU ld cannot link LLVM | Use `lld`: add `-DLLVM_USE_LINKER=lld` to LLVM cmake |
| `-fno-rtti` mismatch | Mixing RTTI modes | Ensure the project's `CMakeLists.txt` has `set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")` |

### 8b. Runtime Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `SIGSEGV` in `invokePacked` at 320+ matrix size | Stack overflow | Add `setrlimit` call or run with `ulimit -s unlimited` |
| `SIGSEGV` on vector load/store | Misaligned buffer | Use `posix_memalign` with 64-byte alignment |
| `Failed to create ExecutionEngine` | Missing LLVM target | Add `llvm::InitializeNativeTarget()` before engine creation |
| `memrefCopy: symbol not found` | Missing runtime library | Set `MLIR_RUNNER_UTILS` env var to path of `libmlir_c_runner_utils.so` |
| `Correctness: FAIL` with NaN values | Uninitialized output buffer | Ensure output buffer is zero-filled before each invocation |
| Pipeline timeout at 512+ | Missing K-dimension tiling | Verify `SquareBlockingPass` includes Level 1 K-tiling (check for `square_k_tiled` attribute in IR) |

### 8c. Debugging JIT-Compiled Code

Dump the LLVM IR before JIT compilation:

```bash
./build/adaptive-opt input.mlir \
  --kernel-fusion --adaptive-router --square-blocking \
  --simd-vectorization --lower-to-llvm \
  --print-ir-after=lower-to-llvm 2>&1 | head -200
```

Dump the generated x86 assembly (requires `ADAPTIVE_JIT_OPT=0` to prevent LLVM from inlining everything):

```bash
ADAPTIVE_JIT_OPT=0 ./build/adaptive-opt input.mlir \
  --kernel-fusion --adaptive-router --square-blocking \
  --simd-vectorization --lower-to-llvm --jit-run \
  2>&1 | grep -A 20 "vmovaps\|vfmadd"
```

---

## 9. Project Structure Reference

```
mlir-capstone/
├── docs/                          # Documentation (this file)
├── mlir-adaptive-matmul/
│   ├── CMakeLists.txt             # Build configuration
│   ├── include/AdaptiveMatmul/    # Header files
│   │   ├── CostModel.h
│   │   ├── HardwareInfo.h
│   │   ├── MatrixAnalyzer.h
│   │   ├── Passes.h               # All 13 pass class declarations
│   │   └── Router.h
│   ├── src/
│   │   ├── main.cpp               # adaptive-opt entry point
│   │   ├── analysis/              # Shape analysis
│   │   ├── cost_model/            # Heuristic scoring
│   │   ├── hardware/              # CPU feature detection (cpuid)
│   │   ├── router/                # Strategy dispatch + dynamic scf.if
│   │   ├── generic_passes/        # Square/SIMD/GPU/Fusion passes
│   │   ├── skinny_passes/         # Skinny-matrix tiling
│   │   ├── small_passes/          # Small-matrix placeholder
│   │   ├── lowering/              # 17-step lowering to LLVM dialect
│   │   └── jit/                   # OrcJIT execution + benchmarking
│   ├── examples/                  # Reference MLIR inputs
│   └── tests/                     # FileCheck lit tests + benchmarks
├── tests/python/                  # Legacy Python orchestration (Phase 1)
└── scripts/                       # Legacy shell scripts
```
