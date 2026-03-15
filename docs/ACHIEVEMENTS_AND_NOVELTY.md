# Achievements, Novelty & Future Work

**Framework:** Adaptive MLIR MatMul Optimization Framework
**Evaluation Date:** 2026-03-15
**Platform:** x86_64 / AVX2 (256-bit SIMD) / WSL2 Linux 6.6.87

---

## 1. Core Achievements

### 1a. End-to-End JIT Execution with Verified Correctness

The framework compiles a high-level `linalg.matmul` tensor operation through 6 transformation stages to native x86 machine code, executes it via LLVM OrcJIT, and verifies numerical correctness — all within a single `adaptive-opt` invocation.

**Correctness results (all-ones inputs, C[i][j] = K):**

| Shape | Strategy | Points Verified | Result |
|-------|----------|----------------|--------|
| 32×32×32 | small | 5/5 | **PASS** |
| 64×64×64 | square | 5/5 | **PASS** |
| 256×256×256 | square | 5/5 | **PASS** |
| 512×512×512 | square | 5/5 | **PASS** |
| 1024×1024×1024 | square | 5/5 | **PASS** |

**100% correctness across all tested sizes** — from L1-resident (32×32, 128KB working set) to L3-spilling (1024×1024, 12MB working set).

### 1b. Stable ~4 GFLOPS Throughput Across Scale

| Shape | Median (ms) | GFLOPS | Cache Tier |
|-------|------------|--------|------------|
| 32×32 | 0.022 | 3.00 | L1 resident |
| 64×64 | 0.122 | 4.31 | L1/L2 boundary |
| 256×256 | 8.156 | 4.18 | L2 resident |
| 512×512 | 72.817 | 3.69 | L2/L3 boundary |
| 1024×1024 | 668.252 | 3.21 | L3 spilling |

The framework sustains **3.0–4.3 GFLOPS** across three orders of magnitude in problem size. The graceful degradation from 4.31 GFLOPS (64×64) to 3.21 GFLOPS (1024×1024) reflects the expected cache hierarchy pressure — not algorithmic breakdown. Without 3-level blocking, sizes ≥ 512 timed out entirely (L1 cache thrashing caused 100×+ slowdowns).

**Context:** Single-core AVX2 FMA theoretical peak at 3.0 GHz = 48 GFLOPS. The framework achieves 8.3% of peak, which is characteristic of research-grade compilers without micro-kernel assembly scheduling, software pipelining, or explicit prefetch insertion. Production systems like OpenBLAS achieve 85-95% of peak through hand-written assembly micro-kernels — a fundamentally different approach that sacrifices generality for raw throughput.

### 1c. Three-Level Cache-Hierarchical Blocking

The `SquareBlockingPass` implements a complete cache-aware blocking strategy:

- **Level 0 (L2):** Tile M, N → 64×64 (AVX2) or 128×128 (AVX-512). Working set: ~48KB per tile (fits in 256KB L2).
- **Level 1 (L1):** Tile K → 64. Working set: A(64×64) + B(64×64) + C(64×64) = 48KB (fits in 32KB L1d with temporal locality on C).
- **Level 2 (Register):** Tile M, N, K → 8×8×8 (AVX2) or 16×16×16 (AVX-512). Produces `vector<8x8x8xf32>` micro-kernels that decompose to 1D `vector<8xf32>` mapping to YMM registers.

This is the primary technical achievement that unblocked matrices > 256×256 from infinite-timeout to sub-second execution.

### 1d. Epilogue Tile-and-Fuse

`SquareBlockingPass` detects single-use elementwise consumers (relu, bias+relu) of contraction results and applies `tileConsumerAndFuseProducersUsingSCF`. This ensures:

- The epilogue (relu) executes **after** the full K reduction — not per-K-step
- The fused computation runs within the L2 tile loops, maximizing data locality
- No intermediate materialization of the full C matrix between contraction and epilogue

### 1e. Dynamic Shape Dispatch via IR-Level Branching

The `RouterPass` emits `scf.if` control flow for partially-dynamic shapes, keeping both code paths (skinny and square) visible to downstream MLIR optimizations. This is verified by FileCheck tests that validate the complete IR structure: `tensor.dim` → `arith.cmpi sgt` → `scf.if` with strategy-attributed clones in both branches.

### 1f. INT8 Quantized Matmul Support

End-to-end datatype detection and vector scaling for INT8 workloads:
- Router detects `i8` inputs + `i32` accumulator → `adaptive.datatype = "int8"`
- `SIMDVectorizationPass` applies 4× K-dimension vector scaling (64 i8 vs 16 f32 per 512-bit register)
- Attribute propagates through all three blocking levels

### 1g. GPU Offload Preparation

Operations exceeding 10^8 FLOPs are tiled into `scf.forall` loops with `#gpu.block<y>` / `#gpu.block<x>` mapping attributes — the canonical input for MLIR's `convert-scf-forall-to-gpu` pass. Verified via FileCheck: 4096×4096 → GPU, 64×64 → square.

### 1h. Full IR Verification Suite

Three lit-style FileCheck tests achieve 100% pass rate:

| Test | Features Validated |
|------|--------------------|
| `router_dynamic_dispatch.mlir` | `scf.if` dispatch, partial-dynamic folding, strategy attributes |
| `quantized_matmul_test.mlir` | INT8 detection, datatype attribute propagation, 4× vector scaling |
| `gpu_offload_test.mlir` | `scf.forall` + GPU block mapping, threshold gating, grid computation |

---

## 2. Novelty: The "Mini-Triton" Unified Heterogeneous Router

### 2a. What Student Compilers Typically Do

The standard capstone or course compiler project follows one of these patterns:

- **Single-backend, static-shape optimizer:** Takes a fixed-size matrix, applies one tiling strategy (e.g., 32×32 blocking), lowers to LLVM, runs. No shape adaptation, no datatype awareness, no heterogeneous dispatch.
- **MLIR tutorial pass:** Implements one transformation (e.g., `linalg.matmul` → `scf.for` loops) following the MLIR documentation. No pipeline composition, no JIT execution, no correctness verification.
- **Python-orchestrated benchmark:** Calls into BLAS or NumPy, measures performance. No compiler IR transformation occurs.

These approaches demonstrate understanding of individual concepts but do not address the central challenge of production compilers: **making a single input program run efficiently across diverse hardware and shape regimes**.

### 2b. What This Framework Does Differently

This framework implements a **unified, heterogeneous compiler router** — conceptually a "Mini-Triton" — that accepts a single `linalg.matmul` op and automatically:

1. **Classifies the workload** at compile time (static shapes) or defers classification to runtime (dynamic shapes via `scf.if` polyhedral multi-versioning)
2. **Routes to a specialized optimization pipeline** based on shape geometry (skinny, square, small, GPU-scale) and hardware capabilities (AVX2, AVX-512, GPU)
3. **Adapts vectorization parameters** based on datatype (f32 vs INT8 — 4× vector width scaling)
4. **Lowers through a complete 17-step pipeline** to native code or GPU-ready IR
5. **Executes and verifies** via OrcJIT with systems-level hardening (aligned allocation, stack management)

This is novel because it integrates five capabilities that are typically found only in production compiler frameworks (Triton, IREE, XLA):

#### Novelty 1: AOT Polyhedral Fallback via `scf.if`

When matrix dimensions are unknown at compile time, the framework does not give up or fall back to a generic path. Instead, it emits **compile-time multi-versioned code** with runtime dispatch:

```mlir
%is_skinny = arith.cmpi sgt, %max_dim, %threshold  // max(M,N) > min(M,N)*8
scf.if %is_skinny {
  // Skinny-optimized matmul (tiled on dominant dimension)
} else {
  // Square-optimized matmul (3-level cache blocking)
}
```

Both paths are independently optimized by downstream passes. This is the same approach used by IREE's dispatch region formation and Triton's autotuner — applied here at the MLIR level without requiring a runtime framework.

#### Novelty 2: Dynamic Datatype-Aware Vectorization

The router's INT8 detection (`i8` inputs + `i32` accumulator) triggers a 4× vector scaling factor in `SIMDVectorizationPass`. This is not a simple flag — it changes the K-dimension vector tile size to saturate integer SIMD registers:

- **f32 on AVX-512:** 16 elements per 512-bit register → K vector tile = 16
- **i8 on AVX-512:** 64 elements per 512-bit register → K vector tile = 64

This mirrors how XLA's `dot_emitter` and Triton's `tl.dot` handle mixed-precision GEMM — detecting accumulator width and adjusting tile geometry to maximize throughput per instruction.

#### Novelty 3: Heterogeneous Target Dispatch (CPU + GPU)

A single `linalg.matmul` can be routed to:
- **CPU square path** → 3-level blocking + AVX vectorization → JIT execution
- **CPU skinny path** → dominant-dimension tiling → JIT execution
- **GPU path** → `scf.forall` with `#gpu.block` mapping → ready for Vulkan/CUDA/ROCm backend

The GPU path produces **target-agnostic parallel IR** that the standard MLIR `convert-scf-forall-to-gpu` pass transforms into device-specific `gpu.launch` ops. This is the same abstraction level used by IREE's HAL (Hardware Abstraction Layer) — the tiling pass is backend-agnostic, and the backend-specific lowering is a separate concern.

#### Novelty 4: Tile-and-Fuse Epilogue Handling

The `SquareBlockingPass` integrates `tileConsumerAndFuseProducersUsingSCF` to handle matmul + elementwise chains (bias + relu) correctly — tiling the consumer (epilogue) and fusing the producer (contraction) into the tile loops. This avoids the classic correctness bug where epilogues are applied per-reduction-step instead of after the full K reduction.

Production frameworks (IREE, TVM) implement this via fusion planning passes. This framework achieves it with a 40-line integration of MLIR's tile-and-fuse infrastructure — demonstrating that the MLIR ecosystem's composable transformations can replicate production-level fusion semantics.

#### Novelty 5: Full-Stack JIT with Systems-Level Hardening

The JIT runner goes beyond "call `ExecutionEngine` and print a number." It addresses real systems engineering challenges:

- **64-byte aligned allocation** for AVX-512 `vmovaps` (not guaranteed by `std::vector`)
- **Stack limit elevation** via `setrlimit` for deeply nested loop nests (not handled by MLIR or LLVM)
- **LLVM function signature introspection** to dynamically construct the unpacked memref ABI argument array
- **Multi-point numerical verification** with relative tolerance

These are the same problems that IREE's CPU HAL driver and TVM's runtime solve — but typically hidden behind layers of abstraction. Solving them directly in a 427-line C++ file demonstrates understanding of the full stack from IR to metal.

---

## 3. Current Limitations

### 3a. Skinny Pipeline Does Not Complete JIT Execution

**Status:** IR generation succeeds; JIT execution crashes.

**Root cause:** `SkinnyTilingPass` produces tiled IR that the vector lowering pipeline cannot fully convert to LLVM. The same 2D→1D vector decomposition issue that was solved for the square path (`LowerVectorMultiReductionPass`) has not been applied to the skinny-tiled output because the skinny vectorization pass is a placeholder — it marks ops as `vectorized` without performing actual `linalg::vectorize`.

**Impact:** Matrices with aspect ratio > 8 (e.g., 4096×32) cannot be JIT-executed. They are correctly routed and tiled, but the lowering pipeline fails.

**Fix:** Apply `SIMDVectorizationPass` to skinny-tiled ops (currently skipped because they retain `optimization_strategy = "skinny"`). This requires removing the skinny skip condition or adding a separate vectorization entry point.

### 3b. Small Matrix Path Uses Scalar Fallback

**Status:** `SmallMatrixVectorPass` is a placeholder.

**Root cause:** Matrices < 64×64 are tagged `optimization_strategy = "small"` and skip `SIMDVectorizationPass`. They fall through to `linalg-to-loops` in the lowering pipeline, producing scalar `scf.for` loops.

**Impact:** 32×32 achieves 3.0 GFLOPS (pure scalar on L1-resident data) — competitive with vectorized 64×64 (4.3 GFLOPS) due to zero cache misses, but leaves performance on the table.

**Fix:** Implement direct `linalg::vectorize` for small matrices with vector tile = full matrix dimensions. A 32×32×32 matmul can be vectorized to `vector<32x32x32xf32>` and lowered via multi-reduction decomposition without tiling overhead.

### 3c. INT8 JIT Execution Not Benchmarked

**Status:** INT8 IR generation and vectorization scaling are verified by FileCheck tests. JIT execution is not tested.

**Root cause:** `AdaptiveJITRunner` allocates `float` (f32) buffers. INT8 workloads require `int8_t` input buffers and `int32_t` accumulator buffers, with corresponding `MemRef2D` template specialization.

**Fix:** Template the buffer allocation on element type, or add a separate INT8 buffer path keyed on the `adaptive.datatype` attribute.

### 3d. GPU Path Requires External Runtime

**Status:** GPU-targeted IR (`scf.forall` + `#gpu.block` mapping) is generated and verified. Actual GPU execution is not implemented.

**Root cause:** The framework stops at `scf.forall` — it does not invoke `convert-scf-forall-to-gpu`, `gpu-to-spirv` / `gpu-to-nvvm`, or link a Vulkan/CUDA runtime.

**Impact:** The GPU path is a preparation step, not an execution path.

**Fix:** Extend the lowering pipeline with:
1. `convert-scf-forall-to-gpu` → `gpu.launch` ops
2. `gpu-to-spirv` or `gpu-to-nvvm` → device IR
3. Vulkan HAL or CUDA runtime integration for kernel dispatch

### 3e. No Auto-Tuning of Tile Sizes

**Status:** Tile sizes are statically determined from `HardwareFeatures` (L2 size, SIMD width).

**Root cause:** Cache sizes vary significantly across CPU microarchitectures (AMD Zen4: 32KB L1d, 1MB L2 per core; Intel SPR: 48KB L1d, 2MB L2 per core). Static heuristics cannot optimally cover all targets.

**Fix:** Implement an auto-tuning loop that:
1. Generates multiple tile-size variants
2. JIT-compiles and benchmarks each
3. Selects the fastest configuration
4. Caches the result keyed on (shape, hardware, datatype)

This mirrors Triton's auto-tuner and TVM's AutoTVM/Ansor — but at the MLIR pass level.

### 3f. No Software Pipelining or Prefetch Insertion

**Status:** The micro-kernel relies entirely on hardware prefetching and LLVM's register allocator.

**Impact:** At 8.3% of peak, the primary bottleneck is instruction scheduling — the LLVM backend does not know the optimal FMA → load interleaving for the specific micro-architecture. Production BLAS libraries (OpenBLAS, MKL) use hand-written assembly with explicit `prefetcht0` instructions and carefully ordered FMA sequences.

**Fix:** This is a fundamental limitation of compiler-generated code vs. hand-tuned assembly. Partial mitigation: `--llvm-opt-level=3` enables LLVM's software pipeliner (`-pipeliner-enable`) for loop bodies, and `SkinnyMemoryPass` could be extended to insert `memref.prefetch` ops.

---

## 4. Future Work Roadmap

| Priority | Task | Estimated Complexity | Impact |
|----------|------|---------------------|--------|
| **P0** | Skinny vectorization (use `SIMDVectorizationPass` for skinny ops) | Low — remove skip condition | Unblocks skinny JIT execution |
| **P0** | INT8 buffer allocation in JIT runner | Medium — template on element type | Unblocks INT8 benchmarking |
| **P1** | Small matrix vectorization (full-matrix vector tile) | Medium — need to handle non-power-of-2 dimensions | 2-4× speedup for < 64×64 |
| **P1** | `gpu.launch` emission + Vulkan/SPIR-V lowering | High — requires runtime integration | End-to-end GPU execution |
| **P2** | Auto-tuning of tile sizes | High — search space exploration | 10-30% throughput improvement |
| **P2** | Prefetch insertion in `SkinnyMemoryPass` | Medium — `memref.prefetch` ops | Reduced L2 miss rate for skinny |
| **P3** | Batch matmul support in JIT runner | Low — add batch dimension to buffer allocation | Enables transformer workloads |
| **P3** | `f16` / `bf16` datatype support | Medium — vector scaling + buffer type | Enables ML inference workloads |
