# Adaptive MLIR MatMul Framework — Codebase Manifest

**Generated:** 2026-03-15
**Branch:** `first-version-exp`
**Total Source Files:** 23 (5 headers + 18 implementations)
**Total Lines of Code:** 2,116
**Build System:** CMake 3.10+ / C++17 / `-fno-rtti`
**MLIR/LLVM Dependency:** Custom trunk build at `/home/sumit/mlir-build/`

---

## Commit Archaeology

| Commit | Description | Files Touched |
|--------|------------|---------------|
| `d539404` | Initial commit: research-grade scaffolding + Python orchestration | 38 files (all A) |
| `6898d36` | Phase 2: Python→C++/LLVM compiler-style orchestration rewrite | 13 files (2 A, 11 M) |
| `811b24b` | Phase 3: Epilogue fusion fix, 3-level blocking, JIT hardening | 8 files (all M) |
| *(uncommitted)* | Phase 4: K-dimension tiling, aligned buffers, stack limit, tests | 8+ files |

---

## File Categorization

### Bucket 1: Pre-built / Untouched (Infrastructure Scaffolding)

These files were created in the initial commit (`d539404`) and have never been modified. They provide foundational abstractions or serve as placeholder stubs that the active pipeline bypasses entirely.

| File | Lines | Criticality | Purpose |
|------|-------|-------------|---------|
| `include/AdaptiveMatmul/CostModel.h` | 17 | Tier 3 | Interface declaration for `CostModel::computeScore()` — a weighted heuristic used only in the router's diagnostic output, not in any control-flow decision. |
| `include/AdaptiveMatmul/MatrixAnalyzer.h` | 32 | Tier 3 | Defines the `MatrixShape` enum (`Skinny`, `Square`, `Small`, `Generic`) and the `MatrixInfo` struct (`M`, `N`, `K`, `aspect_ratio`). Pure data types consumed by the router. |
| `include/AdaptiveMatmul/Router.h` | 36 | Tier 3 | Header for `RouterPass` with `getDependentDialects` registering `scf::SCFDialect`, `arith::ArithDialect`, `tensor::TensorDialect`, `memref::MemRefDialect`. The real logic is in RouterPass.cpp. |
| `src/cost_model/CostModel.cpp` | 20 | Tier 3 | Implements the cost heuristic: `alpha * aspect_ratio + beta * sqrt(M*N*K) + gamma * hw_factor`. Provides estimated throughput numbers printed during routing — informational only, does not affect strategy selection. |
| `src/generic_passes/WideTilingPass.cpp` | 1 | Tier 3 | Stub — single-line `runOnOperation` that prints "Applied Wide Tiling". No IR transformation occurs. The "wide" strategy is never emitted by the current router. |
| `src/skinny_passes/SkinnyMemoryPass.cpp` | 11 | Tier 3 | Stub — prints "Applying Skinny Memory Optimization..." with no IR transformation. Placeholder for future prefetch/packing optimizations. |
| `examples/skinny_matmul.mlir` | ~20 | Tier 3 | Reference MLIR input: 4096×32 memref-based matmul for testing the skinny pipeline. |
| `examples/small_matmul.mlir` | ~20 | Tier 3 | Reference MLIR input: small matrix matmul for the `SmallMatrixVectorPass` path. |
| `examples/square_matmul.mlir` | ~20 | Tier 3 | Reference MLIR input: 1024×1024 memref-based matmul for the square blocking pipeline. |

---

### Bucket 2: Pre-built but Modified (Injected Logic)

These files existed in the initial scaffolding commit (`d539404`) but were substantially rewritten in `6898d36` and/or `811b24b` to implement the actual compiler optimizations. The original versions were either stubs or skeleton implementations.

| File | Lines | Commits | Criticality |
|------|-------|---------|-------------|
| `include/AdaptiveMatmul/HardwareInfo.h` | 36 | d539404 → 6898d36 | Tier 2 |
| `include/AdaptiveMatmul/Passes.h` | 128 | d539404 → 6898d36 | Tier 2 |
| `CMakeLists.txt` | 143 | d539404 → 6898d36 → 811b24b | Tier 2 |
| `src/main.cpp` | 40 | d539404 → 6898d36 → 811b24b | Tier 2 |
| `src/analysis/MatrixAnalyzer.cpp` | 49 | d539404 → 811b24b | Tier 2 |
| `src/hardware/HardwareInfo.cpp` | 136 | d539404 → 6898d36 | Tier 2 |
| `src/router/RouterPass.cpp` | 240 | d539404 → 6898d36 | **Tier 1** |
| `src/generic_passes/GenericTilingPass.cpp` | 42 | d539404 → 6898d36 | Tier 3 |
| `src/generic_passes/KernelFusionPass.cpp` | 70 | d539404 → 6898d36 → 811b24b | **Tier 1** |
| `src/generic_passes/SquareBlockingPass.cpp` | 296 | d539404 → 6898d36 → 811b24b | **Tier 1** |
| `src/generic_passes/SIMDVectorizationPass.cpp` | 120 | d539404 → 6898d36 | **Tier 1** |
| `src/generic_passes/GPUOffloadPass.cpp` | 105 | d539404 | **Tier 1** |
| `src/skinny_passes/SkinnyTilingPass.cpp` | 124 | d539404 → 6898d36 | Tier 2 |
| `src/skinny_passes/SkinnyVectorizationPass.cpp` | 50 | d539404 → 811b24b | Tier 3 |
| `src/small_passes/SmallMatrixVectorPass.cpp` | 48 | d539404 → 811b24b | Tier 3 |

#### Deep-Dive: Modified Files

**`include/AdaptiveMatmul/Passes.h`** (Tier 2)
Central pass registry header declaring all 13 pass classes as `mlir::PassWrapper<T, OperationPass<ModuleOp>>` with `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID`. Modified in `6898d36` to add `GPUOffloadPass`, `SIMDVectorizationPass`, `KernelFusionPass`, `LowerToLLVMPass`, and `JITRunnerPass` declarations. Each pass declares `getDependentDialects` to register the MLIR dialects it introduces — critically, `GPUOffloadPass` registers `gpu::GPUDialect` and `scf::SCFDialect`, while `SIMDVectorizationPass` registers `arith::ArithDialect`, `memref::MemRefDialect`, `vector::VectorDialect`, and `ub::UBDialect`. Without these registrations, the MLIR context would crash on unregistered dialect operations during IR construction.

**`include/AdaptiveMatmul/HardwareInfo.h`** (Tier 2)
Defines the `HardwareFeatures` struct with fields for AVX2/AVX-512 capability, SIMD register width in bits, L1/L3 cache sizes in KB, core count, and GPU architecture string. Modified in `6898d36` to add `detectFromModule(mlir::ModuleOp)` — a module-level detection path that reads MLIR `DataLayout` attributes and custom `adaptive.target` / `adaptive.simd_width` attributes to override host-detected defaults. This enables cross-compilation scenarios where the target machine differs from the build host.

**`CMakeLists.txt`** (Tier 2)
Build configuration linking against 40+ MLIR/LLVM libraries. Modified across all three commits to add new source files and link dependencies: `MLIRGPUDialect` (GPU block mapping), `MLIRVectorTransforms` (multi-reduction lowering), `MLIRExecutionEngine` + `LLVMOrcJIT` (JIT compilation), `MLIRSCFTransforms` (tile-and-fuse), and all dialect-to-LLVM conversion libraries. The `-fno-rtti` flag is critical — MLIR is built without RTTI, and mixing RTTI modes causes `dynamic_cast` failures in the pass infrastructure.

**`src/main.cpp`** (Tier 2)
The `adaptive-opt` tool entry point, modeled after `mlir-opt`. Registers all 13 custom passes in pipeline order: `KernelFusionPass` → `RouterPass` → strategy passes (`SkinnyTiling`, `SkinnyVectorization`, `SkinnyMemory`, `GenericTiling`, `SmallMatrixVector`, `WideTiling`, `SquareBlocking`) → `SIMDVectorizationPass` → `GPUOffloadPass` → `LowerToLLVMPass` → `JITRunnerPass`. Uses `mlir::MlirOptMain` to inherit the full `mlir-opt` CLI including `--pass-pipeline`, `--print-ir-after-all`, and FileCheck integration. Modified to add pass registrations as new passes were implemented.

**`src/analysis/MatrixAnalyzer.cpp`** (Tier 2)
Extracts M, N, K dimensions from `linalg::MatmulOp` and `linalg::BatchMatmulOp` via `getStaticLoopRanges()`. Computes the aspect ratio as `max(M,N) / min(M,N)` and classifies into `Skinny` (ratio > 8), `Square` (ratio < 1.5 and M > 128), `Small` (both < 128), or `Generic`. For dynamic shapes (where `M == ShapedType::kDynamic`), defaults aspect ratio to 1.0 — the router's dynamic dispatch handles the actual runtime classification. Modified in `811b24b` for batch matmul dimension mapping correctness.

**`src/hardware/HardwareInfo.cpp`** (Tier 2)
Dual-path hardware detection: `detect()` uses x86 `cpuid` intrinsics (`__get_cpuid_count`) to probe AVX2 (leaf 7, EBX bit 5), AVX-512F (leaf 7, EBX bit 16), and L3 cache parameters (leaf 4, sub-leaf 3 — ways × partitions × line_size × sets). GPU detection reads the `ADAPTIVE_GPU_TARGET` environment variable following IREE/XLA conventions. `detectFromModule()` overlays IR-level attributes: `adaptive.target` for GPU/ISA override, `dlti.dl_spec` for pointer width detection (32-bit → smaller cache assumption), and `adaptive.simd_width` for explicit SIMD width override. Modified in `6898d36` to add the module-level detection path.

**`src/router/RouterPass.cpp`** (Tier 1)
The central dispatch brain of the framework. For static shapes, `chooseStaticStrategy()` routes based on aspect ratio (>16 → skinny/wide), size (<64 → small), compute volume (>10^8 FLOPs with GPU → gpu), ISA (AVX-512 → simd), or default (square). For dynamic shapes, the pass emits `scf.if` control flow at the IR level: it builds `tensor.dim` → `arith.index_cast` → `arith.maxsi`/`arith.minsi` → `arith.muli` → `arith.cmpi sgt` to compute `max(M,N) > min(M,N) * 8` at runtime, then branches into cloned matmul ops tagged with `optimization_strategy = "skinny"` or `"square"`. The pass also invokes `detectDataType()` which inspects operand element types — `i8` inputs with `i32` accumulator produce `adaptive.datatype = "int8"`, otherwise `"f32"`. Both attributes propagate through the tiling pipeline via attribute forwarding. Modified in `6898d36` to add dynamic dispatch, INT8 detection, and `IRMapping`-based op cloning inside `scf.if` regions.

**`src/generic_passes/KernelFusionPass.cpp`** (Tier 1)
Pre-tiling elementwise chain fusion using MLIR's `populateElementwiseOpsFusionPatterns`. This pass fuses chains like `bias_add → relu` into single `linalg.generic` ops but deliberately does NOT fuse elementwise ops into contractions — doing so would incorrectly apply the epilogue per-reduction-step instead of after the full K reduction. The separation is architecturally critical: this pass runs BEFORE tiling, and contraction-epilogue fusion is handled by `tileConsumerAndFuseProducersUsingSCF` in `SquareBlockingPass`. Uses `GreedyRewriteConfig` with top-down traversal and max 10 iterations to reach a fixpoint. Modified across `6898d36` and `811b24b` to refine the fusion control function and add diagnostic reporting of remaining contractions vs. elementwise epilogues.

**`src/generic_passes/SquareBlockingPass.cpp`** (Tier 1)
The most complex and architecturally significant pass in the framework, implementing three-level cache-aware blocking for near-square matrices. **Level 0 (L2 blocking):** Tiles M and N dimensions into blocks (128×128 for AVX-512, 64×64 for AVX2) that fit in L2 cache, with an epilogue-aware path that uses `tileConsumerAndFuseProducersUsingSCF` when a single-use elementwise consumer (relu, bias+relu) is detected via `findEpilogueConsumer()`. Tile-and-fuse tiles the epilogue on M,N and fuses the contraction producer into the tile loops, ensuring the epilogue executes AFTER the full K reduction — not per K step. **Level 1 (L1 K-tiling):** Tiles the K (reduction) dimension with `{0, 0, tK}` where `tK = min(64, K)`, creating `scf.for(M,N L2) → scf.for(K L1)` nesting that keeps A/B slices in L1. Skips if K ≤ 64 or dynamic. **Level 2 (Register tiling):** Tiles M, N, and K to micro-kernel dimensions (8×8×8 for AVX2, 16×16×16 for AVX-512) to produce compact vector tiles that lower cleanly through `LowerVectorMultiReductionPass`. Gate attributes (`square_l2_blocked`, `square_k_tiled`, `square_reg_tiled`) prevent re-processing across levels. `BlockingSizes` is computed from `HardwareFeatures` with cache-size-aware downscaling for smaller L2 caches. Modified in every commit: `d539404` (skeleton), `6898d36` (L2 + register tiling with tile-and-fuse), `811b24b` (K-dimension tiling, attribute propagation fix for `adaptive.datatype`).

**`src/generic_passes/SIMDVectorizationPass.cpp`** (Tier 1)
Hardware-aware vectorization that converts tiled `linalg.matmul` ops into the `vector` dialect using `mlir::linalg::vectorize()`. The pass computes vector lengths from the SIMD register width and element type bitwidth (`computeVectorLen`: 512-bit AVX-512 / 32-bit f32 = 16 elements). For INT8 workloads (detected via `adaptive.datatype` attribute), it applies a 4× vector scaling factor on the K dimension to saturate integer SIMD registers (64 i8 elements per 512-bit register vs. 16 f32). Uses walk-based `IRRewriter` instead of the greedy pattern driver to match the codepath that MLIR's transform dialect uses successfully. Skips ops with `optimization_strategy = "small"` (scalar fallback) or `"gpu"` (device-specific vectorization). The `vecSizes` vector uses the full static loop ranges as vector tile sizes, and `scalableDims` is set to all-false (fixed-width vectors only). Modified in `6898d36` to add INT8 scaling, hardware-aware vector length computation, and strategy-aware skip logic.

**`src/generic_passes/GPUOffloadPass.cpp`** (Tier 1)
Maps operations tagged with `optimization_strategy = "gpu"` to GPU execution by tiling M and N dimensions into 64×64 blocks wrapped in `scf.forall` loops with `gpu::GPUBlockMappingAttr` (`DimY` for M, `DimX` for N). Uses `mlir::scf::tileUsingSCF` with `LoopType::ForallOp` to emit parallel loop structure. K is left un-tiled (full reduction per GPU block). The `scf.forall` with GPU block mapping is the canonical input for MLIR's `convert-scf-forall-to-gpu` pass that creates `gpu.launch` ops. The router gates this path behind `M * N * K > 10^8` to avoid GPU kernel launch overhead dominating small problems. Though created in `d539404`, the implementation was already functional — it represents IREE-style GPU dispatch preparation without requiring actual GPU hardware at compile time.

**`src/generic_passes/GenericTilingPass.cpp`** (Tier 3)
Simple 32×32×32 tiling for ops with `optimization_strategy = "square"`. This was the original square-matrix tiling implementation, now superseded by `SquareBlockingPass`'s three-level blocking. Still registered in the pass pipeline but effectively a no-op for the active square path since `SquareBlockingPass` processes ops before this pass sees them. Modified in `6898d36` to use `IRRewriter` instead of the deprecated pattern-based tiling API.

**`src/skinny_passes/SkinnyTilingPass.cpp`** (Tier 2)
Two-level tiling for skinny (tall-and-thin or short-and-wide) matrices. **Level 0 (Workgroup):** Tiles the dominant parallel dimension — M for tall matrices (`tileM = 128/256`, `tileN = 0`), N for wide matrices (`tileN = 128/256`, `tileM = 0`). AVX-512 systems get 256-element workgroup tiles. **Level 1 (UKernel/L1):** Tiles K to 32 for L1 residency. Dynamic shapes are guarded with a `skinny_dynamic_fallback` attribute and left untiled for runtime dispatch. Uses `linalg::tileLinalgOp` with `LinalgTilingOptions` and propagates the `optimization_strategy` attribute to tiled ops. Modified in `6898d36` to add hardware-aware tile size selection and dynamic shape handling.

**`src/skinny_passes/SkinnyVectorizationPass.cpp`** (Tier 3)
Placeholder — pattern-based rewriter that matches `linalg::MatmulOp` with `optimization_strategy = "skinny"` and marks them with a `vectorized` attribute without performing actual vectorization. The `SIMDVectorizationPass` handles real vectorization for all strategies. Modified in `811b24b` to add the early-exit guard checking for skinny ops.

**`src/small_passes/SmallMatrixVectorPass.cpp`** (Tier 3)
Placeholder — pattern-based rewriter identical in structure to `SkinnyVectorizationPass` but matching `optimization_strategy = "small"`. Marks ops as `vectorized` without transformation. Small matrices (< 64×64) fall through to `linalg-to-loops` in the lowering pipeline, resulting in scalar execution. Modified in `811b24b` for the early-exit guard.

---

### Bucket 3: Completely New (Engineered from Scratch)

These files did not exist in the initial commit and were created entirely during Phase 2 (`6898d36`) or Phase 4 (uncommitted) to enable end-to-end JIT execution, full dialect lowering, and IR verification.

| File | Lines | Commit | Criticality |
|------|-------|--------|-------------|
| `src/lowering/LowerToLLVMPass.cpp` | 108 | 6898d36 | **Tier 1** |
| `src/jit/AdaptiveJITRunner.cpp` | 427 | 6898d36 → 811b24b | **Tier 1** |
| `tests/router_dynamic_dispatch.mlir` | ~60 | uncommitted | Tier 2 |
| `tests/quantized_matmul_test.mlir` | ~60 | uncommitted | Tier 2 |
| `tests/gpu_offload_test.mlir` | ~45 | uncommitted | Tier 2 |
| `tests/lit.cfg.py` | ~15 | uncommitted | Tier 3 |
| `tests/run_benchmark_sweep.sh` | ~50 | uncommitted | Tier 2 |
| `tests/BENCHMARK_REPORT.md` | ~110 | uncommitted | Tier 2 |

#### Deep-Dive: New Files

**`src/lowering/LowerToLLVMPass.cpp`** (Tier 1)
The full 17-step lowering pipeline from vectorized MLIR to LLVM dialect, encapsulated in a single `OperationPass<ModuleOp>` that internally constructs a `mlir::PassManager`. **Phase 1 (Bufferization):** `OneShotBufferize` with `bufferizeFunctionBoundaries = true` converts tensor semantics to memref semantics — this is the point where the compiler transitions from value-based SSA to memory-based computation. **Phase 2 (Vector dimension lowering):** `LowerVectorMultiReductionPass` with `InnerParallel` strategy decomposes 2D/3D `vector.multi_reduction` ops (produced by vectorizing 3-level-tiled matmul micro-kernels like `vector<8x8x8xf32>`) into 1D reductions that LLVM can lower to native SIMD instructions. Without this phase, `VectorToLLVM` fails because LLVM IR has no concept of multi-dimensional vectors. **Phase 3 (High-level lowering):** `VectorToSCF` → `LinalgToLoops` → `LowerAffine` → `SCFToControlFlow` progressively eliminates all high-level MLIR abstractions. **Phase 4 (Dialect→LLVM):** `VectorToLLVM` → `MathToLLVM` → `ExpandStridedMetadata` → `FinalizeMemRefToLLVM` → `FuncToLLVM` → `ArithToLLVM` → `ControlFlowToLLVM` → `IndexToLLVM` → `UBToLLVM` → `ReconcileUnrealizedCasts`. The ordering is critical — `ExpandStridedMetadata` must precede `FinalizeMemRefToLLVM` because it introduces affine operations that need a second `LowerAffine` pass. `ReconcileUnrealizedCasts` is the final cleanup that removes any remaining `builtin.unrealized_conversion_cast` ops left by partial dialect conversions.

**`src/jit/AdaptiveJITRunner.cpp`** (Tier 1)
The largest and most systems-level file in the framework (427 lines), implementing end-to-end JIT compilation and benchmarking via `mlir::ExecutionEngine` backed by LLVM OrcJIT. **Signature parsing:** `parseMemRefRanks()` introspects the `LLVM::LLVMFuncOp` parameter types to determine memref dimensionalities — each `memref<MxNxf32>` unpacks to 7 LLVM scalar arguments (2 pointers + offset + 2 sizes + 2 strides), and each `memref<Nxf32>` unpacks to 5. **Buffer management:** Uses `posix_memalign` with 64-byte alignment to satisfy AVX-512 `vmovaps` / AVX2 `vmovaps` alignment requirements — `std::vector` only guarantees 16-byte alignment which causes `SIGSEGV` on aligned vector load instructions. **POD descriptors:** `MemRef2D` and `MemRef1D` are plain-old-data structs with `packArgs()` methods that build the `void**` argument array expected by `invokePacked`. These MUST have no constructors/destructors because the JIT engine writes raw bytes into the result descriptor. **Stack hardening:** `setrlimit(RLIMIT_STACK, rl.rlim_max)` raises the main thread's soft stack limit to the hard limit before JIT execution. Deeply-tiled vectorized code (3-level blocking × vectorization) generates LLVM IR with deeply nested loop nests that consume significant stack space — the kernel auto-grows the main thread stack up to `RLIMIT_STACK` but not beyond. **Benchmarking:** Warmup runs (3) + timed runs (configurable via `ADAPTIVE_RUNS` env var) with `std::chrono::high_resolution_clock`. Reports median/mean/min/max wall-clock time and GFLOPS (2×M×N×K / median_seconds / 10^9). **Correctness verification:** With all-ones input buffers, every `C[r][c]` should equal K. Checks 5 strategic points (4 corners + center) with relative tolerance of 10^-3. **Configurable optimization:** `ADAPTIVE_JIT_OPT` env var controls LLVM optimization level (default O2) via `makeOptimizingTransformer` and `CodeGenOptLevel`. Loads `libmlir_c_runner_utils.so` for `memrefCopy` support needed when tile-and-fuse bufferization generates `memref.copy` ops.

**`tests/router_dynamic_dispatch.mlir`** (Tier 2)
FileCheck-based lit test verifying the router's dynamic dispatch codegen. Feeds a `linalg.matmul` with one dynamic dimension (`tensor<?x64xf32>`) and checks for: `tensor.dim` extraction, `arith.cmpi sgt` aspect-ratio comparison, `scf.if` branching, and `optimization_strategy = "skinny"` / `"square"` attributes on cloned ops inside each branch. Validates partial-dynamic folding where one dimension is static.

**`tests/quantized_matmul_test.mlir`** (Tier 2)
FileCheck test for INT8 quantized matmul routing. Defines a `linalg.matmul` with `i8` inputs and `i32` accumulator, verifies that the router produces `adaptive.datatype = "int8"` and that `SIMDVectorizationPass` applies the 4× vector scaling factor on the K dimension. Also tests the `f32` path as a control.

**`tests/gpu_offload_test.mlir`** (Tier 2)
FileCheck test for GPU offload preparation. Feeds a 4096×4096 matmul (exceeding the 10^8 FLOPs GPU threshold) and verifies: `scf.forall` emission, `#gpu.block<y>` / `#gpu.block<x>` mapping attributes, 64×64 tiling grid calculation, and that a 64×64 matmul below the threshold is routed to "square" instead of "gpu".

**`tests/run_benchmark_sweep.sh`** (Tier 2)
Automated benchmark harness that iterates over matrix sizes (32, 64, 256, 512, 1024), sets `ADAPTIVE_M/N/K` environment variables, invokes `adaptive-opt` with the full pass pipeline (`--kernel-fusion --adaptive-router --square-blocking --simd-vectorization --lower-to-llvm --jit-run`), and collects timing/correctness results.

**`tests/BENCHMARK_REPORT.md`** (Tier 2)
Engineering validation report documenting IR verification results (3/3 FileCheck tests pass), correctness matrix (5-point verification per shape), performance table (median/mean/min/max/GFLOPS), and root-cause analysis of known limitations (K-dimension tiling gap, skinny pipeline vector lowering, small-matrix scalar fallback, INT8 buffer allocation).

---

## Criticality Tiers

### Tier 1 — Core Architecture (7 files)

These files ARE the compiler. Removing or breaking any one of them collapses the entire optimization pipeline. They implement the four pillars of the framework: shape-aware routing, cache-hierarchical blocking, hardware-targeted vectorization, and end-to-end execution.

| File | Role | Key MLIR/LLVM APIs |
|------|------|--------------------|
| `RouterPass.cpp` | Shape classification + dynamic `scf.if` dispatch | `IRRewriter`, `IRMapping`, `scf::IfOp::create`, `arith::CmpIOp`, `tensor::DimOp` |
| `SquareBlockingPass.cpp` | 3-level cache blocking (L2→L1→register) + epilogue fusion | `linalg::tileLinalgOp`, `scf::tileConsumerAndFuseProducersUsingSCF`, `TilingInterface` |
| `SIMDVectorizationPass.cpp` | Linalg→Vector lowering with INT8 scaling | `linalg::vectorize`, `linalg::hasVectorizationImpl`, `IRRewriter` |
| `KernelFusionPass.cpp` | Pre-tiling elementwise chain fusion | `linalg::populateElementwiseOpsFusionPatterns`, `applyPatternsGreedily` |
| `GPUOffloadPass.cpp` | GPU block mapping via `scf.forall` | `scf::tileUsingSCF`, `gpu::GPUBlockMappingAttr`, `scf::SCFTilingOptions` |
| `LowerToLLVMPass.cpp` | 17-step lowering pipeline to LLVM dialect | `PassManager`, `OneShotBufferize`, `LowerVectorMultiReduction`, 12 conversion passes |
| `AdaptiveJITRunner.cpp` | OrcJIT execution + aligned buffers + benchmarking | `ExecutionEngine::create`, `invokePacked`, `posix_memalign`, `setrlimit` |

### Tier 2 — Supporting Infrastructure (9 files)

Essential for the framework to function but replaceable or reconfigurable without redesigning the core pipeline.

| File | Role |
|------|------|
| `Passes.h` | Pass class declarations + dialect registration |
| `HardwareInfo.h` / `HardwareInfo.cpp` | CPU feature detection (cpuid) + MLIR module attribute overlay |
| `CMakeLists.txt` | Build graph — 40+ library dependencies |
| `main.cpp` | CLI entry point + pass registration order |
| `MatrixAnalyzer.cpp` | Shape extraction + aspect-ratio classification |
| `SkinnyTilingPass.cpp` | Two-level tiling for tall/thin matrices |
| `tests/*.mlir` + `BENCHMARK_REPORT.md` | Verification and validation artifacts |

### Tier 3 — Stubs and Placeholders (7 files)

Non-functional or bypassed. Present for architectural completeness or future development.

| File | Status |
|------|--------|
| `CostModel.h` / `CostModel.cpp` | Informational scoring — no control-flow impact |
| `MatrixAnalyzer.h` / `Router.h` | Pure type/interface declarations |
| `WideTilingPass.cpp` | Single-line stub |
| `SkinnyMemoryPass.cpp` | Print-only stub |
| `SkinnyVectorizationPass.cpp` | Placeholder — marks ops without transforming |
| `SmallMatrixVectorPass.cpp` | Placeholder — marks ops without transforming |
| `GenericTilingPass.cpp` | Superseded by SquareBlockingPass |

---

## Pipeline Flow (File-to-File Data Dependency)

```
Input MLIR (linalg.matmul on tensor types)
    │
    ▼
KernelFusionPass.cpp ─── fuse bias+relu chains (GreedyPatternRewrite)
    │                     does NOT fuse into contractions
    ▼
RouterPass.cpp ────────── classify shape → set optimization_strategy attr
    │                     dynamic shapes → emit scf.if with runtime check
    │                     detect i8/i32 → set adaptive.datatype = "int8"
    ▼
┌─── Strategy Dispatch (based on optimization_strategy attribute) ───┐
│                                                                     │
│  "square" → SquareBlockingPass.cpp                                 │
│              Level 0: tile M,N (L2: 64×64 or 128×128)             │
│              + tile-and-fuse epilogue if present                   │
│              Level 1: tile K (L1: 64)                              │
│              Level 2: tile M,N,K (register: 8×8×8 or 16×16×16)    │
│                                                                     │
│  "skinny" → SkinnyTilingPass.cpp                                   │
│              Level 0: tile dominant dim (workgroup: 128/256)       │
│              Level 1: tile K (ukernel: 32)                         │
│                                                                     │
│  "gpu"    → GPUOffloadPass.cpp                                     │
│              tile M,N → 64×64 scf.forall + gpu.block mapping      │
│                                                                     │
│  "small"  → SmallMatrixVectorPass.cpp (placeholder → scalar)      │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
SIMDVectorizationPass.cpp ── linalg → vector dialect (skip small/gpu)
    │                         INT8: 4× K-dimension vector scaling
    ▼
LowerToLLVMPass.cpp ──────── OneShotBufferize → LowerVectorMultiReduction
    │                         → VectorToSCF → LinalgToLoops → LowerAffine
    │                         → SCFToControlFlow → VectorToLLVM → ...
    │                         → ReconcileUnrealizedCasts
    ▼
AdaptiveJITRunner.cpp ─────── ExecutionEngine (OrcJIT) → native execution
                               64-byte aligned buffers, stack hardening
                               warmup + benchmark + 5-point correctness
```

---

## External Dependencies (Non-Project Files)

| Dependency | Path | Purpose |
|-----------|------|---------|
| MLIR/LLVM build | `/home/sumit/mlir-build/` | Provides all `MLIR*` and `LLVM*` libraries |
| C runner utils | `/home/sumit/mlir-build/lib/libmlir_c_runner_utils.so` | Runtime `memrefCopy` for tile-and-fuse bufferization |
| Python test suite | `/home/sumit/mlir-capstone/tests/python/` | Legacy orchestration scripts from Phase 1 (pre-C++ rewrite) |
| Shell scripts | `/home/sumit/mlir-capstone/scripts/` | Legacy benchmark/research runner scripts |
