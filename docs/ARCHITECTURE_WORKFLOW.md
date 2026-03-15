# Architecture & Workflow Deep-Dive

**Framework:** Adaptive MLIR MatMul Optimization Framework
**Target Audience:** Compiler Engineers, SREs, MLIR Infrastructure Developers
**Pipeline Version:** Phase 4 (3-level blocking + JIT hardening)

---

## 1. Operation Lifecycle: `linalg.matmul` Ingestion to Native Execution

A `linalg.matmul` operation enters the framework as a high-level tensor-semantic contraction in the Linalg dialect. It undergoes six transformation stages before executing as native x86 machine code via LLVM OrcJIT. Each stage operates on `mlir::ModuleOp` and communicates downstream via IR attributes (`optimization_strategy`, `adaptive.datatype`, gate attributes).

```
linalg.matmul {ins = %A, %B : tensor<MxKxf32>, tensor<KxNxf32>}
                {outs = %C : tensor<MxNxf32>} -> tensor<MxNxf32>
```

### Stage 1: Kernel Fusion (`KernelFusionPass`)

The matmul's output may feed into an elementwise consumer chain (e.g., `linalg.generic` implementing bias addition followed by ReLU). Before any tiling occurs, `KernelFusionPass` fuses adjacent elementwise operations into single `linalg.generic` ops.

**What it does NOT do:** It does not fuse elementwise ops into the contraction itself. This is a deliberate architectural decision.

**Why:** If `bias_add(relu(matmul(A, B)))` were fused into a single op, the epilogue (bias+relu) would execute per-reduction-step — i.e., inside the K loop — producing incorrect results. The matmul accumulates `C[i][j] += A[i][k] * B[k][j]` across all K; the epilogue must run only after the full K reduction completes. Separating elementwise-chain fusion (this pass) from contraction-epilogue fusion (tile-and-fuse in `SquareBlockingPass`) enforces this invariant.

**Mechanism:** Invokes `mlir::linalg::populateElementwiseOpsFusionPatterns` with a permissive `ControlFusionFn` (always returns `true`) inside a `GreedyRewriteConfig` with top-down traversal and max 10 iterations. The greedy driver reaches a fixpoint where no further elementwise-to-elementwise fusions are possible.

**Post-condition:** The IR contains unfused contractions (`linalg.matmul`) with optional single-use elementwise consumers (`linalg.generic`). The producer-consumer relationship is preserved for tile-and-fuse in Stage 3.

---

### Stage 2: Adaptive Router (`RouterPass`)

The router inspects each `linalg.matmul` / `linalg.batch_matmul` and assigns an `optimization_strategy` string attribute that gates all downstream passes.

#### 2a. Static Shape Routing

When M, N, and K are compile-time constants, `chooseStaticStrategy()` applies a decision tree:

| Condition | Strategy | Rationale |
|-----------|----------|-----------|
| `aspect_ratio > 16` | `"skinny"` or `"wide"` | Extreme aspect ratios benefit from tiling only the dominant dimension; the short dimension fits in registers. |
| `M < 64 && N < 64 && K < 64` | `"small"` | Kernel launch overhead (even JIT) dominates; scalar fallback is competitive. |
| `has_gpu && M*N*K > 10^8` | `"gpu"` | GPU kernel launch latency (~5μs) is amortized only above ~10^8 FLOPs. Below this threshold, PCIe transfer + launch overhead exceeds compute time. |
| `has_avx512` | `"simd"` | AVX-512 enables 16-wide f32 or 64-wide i8 vectorization without multi-level blocking. |
| default | `"square"` | Near-square matrices benefit from cache-hierarchical blocking. |

The strategy attribute is set directly on the operation: `op->setAttr("optimization_strategy", StringAttr::get(ctx, "square"))`.

#### 2b. Dynamic Shape Dispatch (`scf.if` Polyhedral Fallback)

When M or N is `ShapedType::kDynamic` (unknown at compile time), the router emits runtime control flow that defers the strategy decision to execution time.

**IR construction sequence:**

1. **Dimension extraction:** `tensor.dim %A, 0` → runtime M value (or `arith.constant` if static).
2. **Type casting:** `arith.index_cast` to `i64` for integer arithmetic (index type has platform-dependent width).
3. **Aspect ratio computation:**
   ```
   %max = arith.maxsi %M_i64, %N_i64
   %min = arith.minsi %M_i64, %N_i64
   %threshold = arith.muli %min, %c8_i64
   %is_skinny = arith.cmpi sgt, %max, %threshold
   ```
4. **Branch emission:** `scf.if %is_skinny` with two regions, each containing a cloned matmul via `IRMapping`:
   - **then-region:** Clone with `optimization_strategy = "skinny"`
   - **else-region:** Clone with `optimization_strategy = "square"`
5. **Result threading:** `scf.yield` returns the cloned op's results; `rewriter.replaceOp` substitutes the original.

**Why `scf.if` instead of a runtime function call:** The `scf.if` approach keeps both code paths visible to downstream MLIR passes. Each branch is independently optimized by the strategy-specific tiling passes. After lowering, LLVM's branch predictor handles the runtime dispatch with near-zero overhead for consistent shapes.

#### 2c. INT8 Datatype Detection

`detectDataType()` inspects the element types of the matmul's DPS (destination-passing style) operands:

- **Inputs A, B:** `getElemType(op.getDpsInputOperand(0)->get())` → check `isInteger(8)`
- **Accumulator C:** `getElemType(op.getDpsInits()[0])` → check `isInteger(32)`

If inputs are `i8`/`ui8` and accumulator is `i32`, the operation is tagged with `adaptive.datatype = "int8"`. This attribute propagates through all tiling levels and triggers 4× vector scaling in `SIMDVectorizationPass`.

**Why detect at the router level:** Datatype affects both tiling decisions (INT8 micro-kernels can be 4× wider) and vectorization parameters. Detecting once and propagating via attribute avoids redundant type inspection in every downstream pass.

---

### Stage 3: Strategy-Specific Tiling Passes

Each strategy attribute gates a dedicated tiling pass. Only ops with the matching attribute are transformed; all others are skipped via early-exit guards.

#### 3a. Square Blocking (`SquareBlockingPass`) — 3-Level Cache Hierarchy

This is the most architecturally significant pass, implementing a three-level blocking strategy that maps the computation onto the CPU cache hierarchy.

**Why 3-level blocking:** A naive `linalg.matmul` on a 512×512 matrix performs 512³ = 134M multiply-accumulate operations. Without blocking, the inner loop streams through 512×512×4 = 1MB of B-matrix data per row of A — far exceeding the 32KB L1 data cache. Each cache miss costs ~10 cycles (L2) or ~40 cycles (L3), dominating the 4-cycle FMA latency. Three-level blocking ensures that at each loop nest level, the working set fits in the corresponding cache tier.

##### Level 0: L2 Cache Blocking (M, N dimensions)

**Tile sizes:** 128×128 (AVX-512) or 64×64 (AVX2), scaled down for L2 caches < 4MB.

**Working set calculation (AVX2, 64×64):**
- A slice: 64 × K × 4 bytes (streamed)
- B slice: K × 64 × 4 bytes (streamed)
- C tile: 64 × 64 × 4 bytes = 16KB (resident)
- Total per-tile: 16KB + streaming ≈ fits in 256KB L2

**Epilogue fusion path:** When `findEpilogueConsumer()` detects a single-use `linalg.generic` consumer that is elementwise (e.g., ReLU), the pass uses `scf::tileConsumerAndFuseProducersUsingSCF`:

1. Tile the **epilogue** (consumer) on M, N dimensions.
2. Fuse the **contraction** (producer) into the tile loops.
3. The fused contraction performs the full K reduction within each M×N tile, then the epilogue executes on the completed tile.

This ensures correctness: `relu(C[i][j])` runs after `C[i][j] = sum_k(A[i][k] * B[k][j])`, not per-K-step.

**Fallback:** If tile-and-fuse fails (e.g., non-fusible consumer), falls through to `linalg::tileLinalgOp` with `{tM, tN, 0}` — tiling only M, N and leaving K for Level 1.

**Gate attribute:** Sets `square_l2_blocked` on tiled ops.

##### Level 1: L1 Cache Blocking (K dimension)

**Tile size:** `tK = min(64, K)`. Skipped if K ≤ 64 (already fits) or K is dynamic.

**Tile specification:** `{0, 0, tK}` — zeros for M, N mean "don't tile these dimensions" (already tiled at L2).

**Working set calculation (64×64 L2 tile, K=64 L1 tile):**
- A slice: 64 × 64 × 4 = 16KB
- B slice: 64 × 64 × 4 = 16KB
- C tile: 64 × 64 × 4 = 16KB (accumulate in-place)
- Total: 48KB — fits in 2-way 32KB L1d with temporal locality on C

**Loop nest produced:** `scf.for(M_l2) → scf.for(N_l2) → scf.for(K_l1) → matmul(M_l2, N_l2, K_l1)`

**Gate attribute:** Sets `square_k_tiled`. Propagates `optimization_strategy` and `adaptive.datatype`.

##### Level 2: Register Tiling (M, N, K micro-kernel)

**Tile sizes:** 8×8×8 (AVX2) or 16×16×16 (AVX-512).

**Tile specification:** `{tM, tN, tK}` — all three dimensions, producing the innermost micro-kernel.

**Why tile K at register level:** Without register-level K tiling, the vectorizer creates `vector<8x8x64xf32>` (for a 64-deep K tile) — a 16KB vector that cannot be register-allocated and spills to stack. With `regK=8`, the vectorizer creates `vector<8x8x8xf32>` = 2KB, which `LowerVectorMultiReductionPass` decomposes into 1D `vector<8xf32>` operations mapping directly to 256-bit YMM registers.

**Gate attribute:** Requires `square_l2_blocked` + `square_k_tiled`. Sets `square_reg_tiled`.

**Final loop nest:**
```
scf.for M_l2 (step=64):
  scf.for N_l2 (step=64):
    scf.for K_l1 (step=64):
      scf.for M_reg (step=8):
        scf.for N_reg (step=8):
          scf.for K_reg (step=8):
            matmul_8x8x8  ← vectorized to vector<8x8x8xf32>
```

#### 3b. Skinny Tiling (`SkinnyTilingPass`)

Two-level tiling for matrices with aspect ratio > 8:

- **Level 0 (Workgroup):** Tile the dominant parallel dimension. For tall matrices (M >> N): `{128, 0, 32}` — tile M, leave N untiled, tile K for L1. For wide matrices (N >> M): `{0, 128, 32}`.
- **Level 1 (UKernel/L1):** K-dimension tiling to 32 elements for L1 residency.

AVX-512 systems double the workgroup tile to 256 elements to amortize loop overhead across wider SIMD execution.

Dynamic shapes are tagged with `skinny_dynamic_fallback` and left untiled — the runtime `scf.if` dispatch handles them.

#### 3c. GPU Offload (`GPUOffloadPass`)

Prepares massive matmuls (> 10^8 FLOPs) for GPU execution by tiling M and N into 64×64 blocks wrapped in `scf.forall` with GPU block mapping attributes:

```mlir
scf.forall (%bidy, %bidx) in (M/64, N/64)
    mapping = [#gpu.block<y>, #gpu.block<x>] {
  // 64x64 matmul tile with full K reduction
}
```

**Why `scf.forall` instead of `gpu.launch`:** `scf.forall` is a structured parallel loop that the standard `convert-scf-forall-to-gpu` pass transforms into `gpu.launch` ops. This keeps the tiling pass target-agnostic — the same IR can lower to CUDA, ROCm, or Vulkan/SPIR-V depending on which GPU backend conversion is applied downstream.

**Why 64×64 blocks:** Maps to CUDA's 64-thread threadblocks (8×8 threads, each computing one element) or Vulkan's workgroup size limits. The K dimension is left un-tiled (full reduction per block) to avoid cross-block synchronization.

---

### Stage 4: SIMD Vectorization (`SIMDVectorizationPass`)

Converts tiled `linalg` operations into the `vector` dialect using `mlir::linalg::vectorize()`.

**Skip conditions:**
- `optimization_strategy = "small"` → scalar fallback (vectorization overhead > benefit at < 64×64)
- `optimization_strategy = "gpu"` → GPU-specific vectorization handled by device backend
- `!linalg::hasVectorizationImpl(op)` → unsupported op variant
- Any dynamic dimension → runtime dispatch handles these

**Vector size computation:**
- `vecSizes` is set to the full static loop ranges of the (already register-tiled) matmul
- For an 8×8×8 micro-kernel: `vecSizes = {8, 8, 8}`, producing `vector<8x8x8xf32>`
- `scalableDims` is all-false (fixed-width vectors only — no SVE/RVV)

**INT8 vector scaling:**
When `adaptive.datatype = "int8"`, the K-dimension vector tile is scaled by 4×:
```
vecSizes[2] = min(vecSizes[2] * 4, K)
```

**Rationale:** A 256-bit AVX2 register holds 8 × f32 = 8 elements, but 32 × i8 = 32 elements. A 512-bit AVX-512 register holds 16 × f32 but 64 × i8. The 4× scaling factor (32/8 or 64/16) saturates integer SIMD registers by packing 4× more elements per vector operation, directly exploiting the `VPMADDUBSW` / `VPDPBUSD` instruction throughput.

**API used:** `mlir::linalg::vectorize(rewriter, op, vecSizes, scalableDims)` returns a `VectorizationResult` with replacement values. The walk-based `IRRewriter` approach (not the greedy pattern driver) matches the codepath that MLIR's transform dialect uses successfully — the greedy driver has known issues with vectorization pattern ordering.

---

### Stage 5: Lowering to LLVM Dialect (`LowerToLLVMPass`)

A 17-step pipeline encapsulated in a single pass that constructs an internal `mlir::PassManager`. The ordering is not arbitrary — each step has dependencies on its predecessors.

#### Phase 1: Bufferization (tensor → memref)

`OneShotBufferize` with `bufferizeFunctionBoundaries = true` converts tensor-semantic IR to memref-based IR. Every `tensor<MxNxf32>` becomes `memref<MxNxf32>` with explicit memory allocation. Function signatures change from tensor arguments to memref arguments.

**Why one-shot:** One-shot bufferization performs a single analysis pass to determine buffer reuse opportunities, avoiding the phase-ordering issues of dialect-specific bufferization passes.

#### Phase 2: Vector Dimension Lowering

`LowerVectorMultiReductionPass` with `InnerParallel` strategy is the critical bridge between MLIR's multi-dimensional vector abstraction and LLVM's 1D-only vector type.

**Problem:** The vectorizer produces `vector.multi_reduction` ops on `vector<8x8x8xf32>` — an 8×8×8 3D vector. LLVM IR has no concept of multi-dimensional vectors. Without this pass, `VectorToLLVM` crashes with "cannot convert multi-dimensional vector to LLVM type."

**Solution:** `InnerParallel` decomposes `vector.multi_reduction<add, vector<8x8x8xf32>>` into nested loops of 1D `vector.reduction<add, vector<8xf32>>` operations. Each `vector<8xf32>` maps directly to a 256-bit YMM register (AVX2) or the lower 256 bits of a ZMM register (AVX-512).

#### Phase 3: High-Level Dialect Lowering

1. `VectorToSCF` — `vector.transfer_read/write` → `scf.for` loops with scalar loads/stores
2. `LinalgToLoops` — residual `linalg` ops (unfused, unvectorized) → `scf.for` scalar loops
3. `LowerAffine` — `affine.apply`, `affine.for` → `arith` index arithmetic
4. `SCFToControlFlow` — `scf.for`, `scf.if`, `scf.while` → `cf.br`, `cf.cond_br`

#### Phase 4: Dialect → LLVM Conversions

The ordering respects dialect dependencies:

1. `VectorToLLVM` — `vector.*` → `llvm.intr.fma`, `llvm.shufflevector`, etc.
2. `MathToLLVM` — `math.sqrt`, `math.exp` → `llvm.intr.sqrt`, `llvm.intr.exp`
3. `ExpandStridedMetadata` — complex `memref.subview` → primitive pointer arithmetic (introduces affine ops)
4. `LowerAffine` (second pass) — clean up affine ops from step 3
5. `FinalizeMemRefToLLVM` — `memref.alloc`, `memref.load/store` → `llvm.alloca`, `llvm.load/store`
6. `FuncToLLVM` — `func.func`, `func.call`, `func.return` → `llvm.func`, `llvm.call`, `llvm.return`
7. `ArithToLLVM` — `arith.addf`, `arith.muli`, etc. → `llvm.fadd`, `llvm.mul`
8. `ControlFlowToLLVM` — `cf.br`, `cf.cond_br` → `llvm.br`, `llvm.cond_br`
9. `IndexToLLVM` — `index.casts`, `index.sizeof` → `llvm.sext`, `llvm.constant`
10. `UBToLLVM` — `ub.poison` → `llvm.mlir.poison`
11. `ReconcileUnrealizedCasts` — removes residual `builtin.unrealized_conversion_cast` ops

**Post-condition:** The module contains only `llvm.*` operations — a valid input for `mlir::ExecutionEngine`.

---

### Stage 6: JIT Execution (`AdaptiveJITRunner`)

#### 6a. Engine Creation

`mlir::ExecutionEngine::create(module, opts)` translates LLVM dialect to LLVM IR, runs the LLVM optimization pipeline (`makeOptimizingTransformer` at O2 by default), and JIT-compiles to native x86 via OrcJIT.

**Shared library loading:** `libmlir_c_runner_utils.so` provides `memrefCopy` — required when tile-and-fuse bufferization generates `memref.copy` ops for partial tile results.

#### 6b. Signature Introspection

`parseMemRefRanks()` walks the LLVM function's parameter types to reconstruct the memref ABI:

```
memref<MxNxf32> → (ptr, ptr, i64, i64, i64, i64, i64)
                    base  data  off  sz0  sz1  str0 str1
```

Two consecutive `LLVMPointerType` args start a new memref; subsequent `i64` args encode offset, sizes, and strides. `rank = (num_i64_args - 1) / 2` (1 offset + rank sizes + rank strides).

#### 6c. Buffer Allocation

`posix_memalign(&ptr, 64, count * sizeof(float))` allocates 64-byte-aligned buffers.

**Why 64-byte alignment:** AVX-512 `vmovaps` (aligned packed single-precision move) requires 64-byte alignment. AVX2 `vmovaps` requires 32-byte alignment. Misaligned access causes `#GP` (General Protection fault) → `SIGSEGV`. `std::vector<float>` allocators guarantee only `alignof(float)` = 4 bytes, or at best 16 bytes (SSE). The 64-byte alignment satisfies all current x86 SIMD ISAs.

#### 6d. Stack Hardening

```c++
struct rlimit rl;
getrlimit(RLIMIT_STACK, &rl);
rl.rlim_cur = rl.rlim_max;
setrlimit(RLIMIT_STACK, &rl);
```

**Why:** Three-level tiling + vectorization produces deeply nested loop nests in LLVM IR. At O2, LLVM may not fully collapse these loops, resulting in stack frames with large alloca regions for vector spills. The default soft stack limit (typically 8MB) is insufficient for 512×512+ matrices. The kernel auto-grows the main thread's stack up to `RLIMIT_STACK`; raising the soft limit to the hard limit (typically 64MB+) prevents `SIGSEGV` from stack overflow.

**Why not `pthread_create` with large stack:** mmap'd pthread stacks have hard boundaries. The JIT code's stack access pattern can skip guard pages (stack clash), causing immediate `SIGSEGV`. The main thread's stack uses the kernel's auto-growth mechanism which handles non-sequential stack access.

#### 6e. Execution & Verification

- **Warmup:** 3 runs to populate instruction/data caches and trigger JIT code caching.
- **Benchmark:** N runs (default 10) with `std::chrono::high_resolution_clock` per-invocation timing.
- **Correctness:** With all-ones inputs (A=1, B=1), every `C[r][c] = K`. Five strategic points (4 corners + center) are checked with relative tolerance 10^-3 to catch both gross errors and floating-point accumulation drift.

---

## Design Decision Summary

| Decision | Alternative Considered | Why This Approach |
|----------|----------------------|-------------------|
| Attribute-based pass gating | Pass pipeline branching | Attributes survive tiling; pipeline branching requires separate PassManager instances per strategy |
| `scf.if` for dynamic dispatch | Runtime function pointer table | `scf.if` keeps both paths visible to MLIR optimizations; function pointers are opaque |
| Tile-and-fuse for epilogues | Post-tiling fusion | Post-tiling fusion requires complex inter-loop analysis; tile-and-fuse handles it structurally |
| Walk-based vectorization | Greedy pattern driver | Greedy driver has known ordering issues with vectorization patterns; walk matches transform dialect codepath |
| `posix_memalign` over `std::vector` | `aligned_alloc` (C11) | `posix_memalign` is available on all POSIX systems; `aligned_alloc` requires size to be a multiple of alignment |
| `setrlimit` over `pthread` stack | mmap'd stack with guard pages | Main thread stack auto-grows; mmap'd stacks have hard boundaries vulnerable to stack clash |
| One-shot bufferization | Dialect-specific bufferization | Avoids phase-ordering issues; single analysis pass determines buffer reuse |
| `LowerVectorMultiReduction` before `VectorToLLVM` | Direct LLVM lowering | LLVM IR has no multi-dimensional vectors; this pass is mandatory, not optional |
