# MLIR Glossary for the Adaptive MatMul Framework

**Scope:** Terms used in this project, defined from first principles with references to our specific implementation.
**Audience:** Beginners seeking foundational understanding; intermediate MLIR engineers seeking project-specific context.

---

## A

### Affine Dialect (`affine`)

An MLIR dialect for polyhedral-model-style loop and memory access representations. `affine.for` loops have trip counts that are affine functions of outer loop induction variables and symbolic constants. `affine.apply` computes affine expressions (linear combinations of values plus a constant).

**In this project:** `affine` operations appear as intermediate artifacts during lowering. `ExpandStridedMetadata` (which decomposes complex `memref.subview` ops into pointer arithmetic) introduces `affine.apply` ops that must be lowered to `arith` via `LowerAffine`. This is why `LowerToLLVMPass` calls `LowerAffine` **twice** — once for affine ops from the original IR, and once for affine ops introduced by `ExpandStridedMetadata`.

### Ahead-of-Time (AOT) Compilation

Compilation that occurs before program execution, as opposed to Just-in-Time (JIT) compilation which occurs during execution. AOT produces a static binary or object file; JIT produces machine code in memory at runtime.

**In this project:** The framework uses a hybrid approach. The MLIR-level transformations (routing, tiling, vectorization, lowering) are performed AOT — they produce LLVM dialect IR that is a complete, optimized program. The final step (LLVM IR → native x86) is JIT-compiled via `ExecutionEngine` / OrcJIT. The `scf.if` dynamic dispatch is an AOT construct — the branch is compiled into the binary, but the branch condition is evaluated at runtime.

### Aligned Memory Access

A memory read or write where the address is a multiple of the data's natural alignment. x86 SIMD instructions have strict alignment requirements:

| Instruction | Vector Width | Required Alignment |
|------------|-------------|-------------------|
| `movaps` (SSE) | 128-bit | 16 bytes |
| `vmovaps` (AVX) | 256-bit | 32 bytes |
| `vmovaps` (AVX-512) | 512-bit | 64 bytes |

Misaligned access on these instructions triggers a `#GP` (General Protection) exception → `SIGSEGV`.

**In this project:** `AdaptiveJITRunner` uses `posix_memalign(&ptr, 64, size)` to guarantee 64-byte alignment for all buffers passed to JIT-compiled code.

### AVX2 (Advanced Vector Extensions 2)

Intel ISA extension (Haswell, 2013+) providing 256-bit SIMD registers (YMM0–YMM15). Key instructions for matmul:

- `vfmadd231ps ymm, ymm, ymm` — fused multiply-add on 8 × f32
- `vmovaps ymm, [mem]` — aligned load of 8 × f32

**In this project:** AVX2 is the baseline SIMD target. `HardwareInfo::detect()` probes AVX2 via `cpuid` leaf 7, EBX bit 5. When detected, the framework uses 256-bit vector widths, 64×64 L2 tiles, and 8×8×8 register tiles.

### AVX-512

Intel ISA extension (Skylake-X, 2017+) providing 512-bit SIMD registers (ZMM0–ZMM31). Doubles the vector width and register count vs. AVX2.

**In this project:** When `HardwareInfo` detects AVX-512 (`cpuid` leaf 7, EBX bit 16), tile sizes double: 128×128 L2 tiles, 16×16×16 register tiles. The `SIMDVectorizationPass` computes vector length as `simd_width / element_bitwidth` = 512/32 = 16 for f32.

---

## B

### Bufferization

The transformation from **tensor semantics** (value-based, SSA, no side effects) to **memref semantics** (memory-based, pointer aliasing, side effects). Tensors are immutable values; memrefs are mutable memory regions.

```mlir
// Before bufferization (tensor):
%result = linalg.matmul ins(%A, %B : tensor<MxKxf32>, tensor<KxNxf32>)
                        outs(%C : tensor<MxNxf32>) -> tensor<MxNxf32>

// After bufferization (memref):
linalg.matmul ins(%A, %B : memref<MxKxf32>, memref<KxNxf32>)
              outs(%C : memref<MxNxf32>)
```

**In this project:** `LowerToLLVMPass` uses `OneShotBufferize` with `bufferizeFunctionBoundaries = true`. This is the point where the compiler transitions from a purely functional representation (where every operation produces a new tensor) to an imperative representation (where operations mutate memory in-place). One-shot bufferization performs a single analysis pass to determine which tensors can share buffers, avoiding unnecessary memory allocation.

### Blocking (Cache Blocking / Loop Tiling)

Restructuring loop nests so that the working set of each inner iteration fits in a specific cache level. Also called "loop tiling" or "strip-mining."

**In this project:** `SquareBlockingPass` implements three-level blocking:
- L2 blocking: M, N → 64 or 128 (working set fits in 256KB–1MB L2)
- L1 blocking: K → 64 (A + B slices fit in 32KB L1d)
- Register blocking: M, N, K → 8 (vectors fit in YMM/ZMM registers)

---

## C

### Canonicalization

An MLIR transformation that applies algebraic simplifications and constant folding to operations. Each dialect defines its own canonicalization patterns (e.g., `arith.addi %x, 0` → `%x`).

**In this project:** `createCanonicalizerPass()` runs after bufferization and after vector dimension lowering to clean up redundant operations.

### Contraction

A linear algebra operation that reduces one or more dimensions via multiply-accumulate. `linalg.matmul` is a 2D contraction: `C[i][j] += A[i][k] * B[k][j]` reduces over K.

**In this project:** The router identifies contractions via `mlir::linalg::isaContractionOpInterface`. Only contractions (not elementwise ops) are routed to strategy-specific tiling passes. The `KernelFusionPass` counts remaining contractions vs. elementwise ops after fusion.

### Common Subexpression Elimination (CSE)

Identifies and removes duplicate computations. If two operations have identical operands and produce the same result, one is deleted and its uses are redirected to the other.

**In this project:** `createCSEPass()` runs after canonicalization to eliminate redundant address computations introduced by tiling and bufferization.

---

## D

### Datatype Detection

Runtime or compile-time identification of the numerical type of a matrix operation's operands.

**In this project:** `RouterPass::detectDataType()` inspects the DPS operand element types of each `linalg.matmul`:
- `i8` inputs + `i32` accumulator → `"int8"` (quantized inference workload)
- Everything else → `"f32"`

The result is stored as `adaptive.datatype` attribute and consumed by `SIMDVectorizationPass` to apply 4× vector scaling for INT8.

### Dependent Dialects

MLIR passes must declare which dialects they may introduce into the IR via `getDependentDialects`. The pass manager uses this to pre-load dialect definitions before the pass runs.

**In this project:** `GPUOffloadPass` declares `gpu::GPUDialect` and `scf::SCFDialect`. `SIMDVectorizationPass` declares `arith`, `memref`, `vector`, and `ub` dialects. Without these declarations, the MLIR context would crash when the pass creates operations from an unregistered dialect.

### Destination-Passing Style (DPS)

A calling convention where the output buffer is passed as an input argument (the "destination"), rather than being allocated by the callee. Linalg operations use DPS: `ins(...)` are read-only inputs, `outs(...)` are read-write destinations.

```mlir
linalg.matmul ins(%A, %B) outs(%C) -> tensor<MxNxf32>
// %C is both the initial accumulator AND the destination for results
```

**In this project:** The JIT runner respects DPS by pre-allocating the output buffer (zeroed) and passing it as the third memref argument.

---

## E

### Elementwise Operation

A computation where each output element depends only on the corresponding input elements (no reduction, no cross-element dependency). Examples: ReLU (`max(0, x)`), bias addition (`x + b`), scaling (`x * alpha`).

**In this project:** `KernelFusionPass` fuses chains of elementwise operations (e.g., `bias_add → relu`) into single `linalg.generic` ops using `populateElementwiseOpsFusionPatterns`. The fused elementwise op becomes the epilogue that is later tile-and-fused with the contraction.

### Epilogue

An elementwise operation applied to the output of a contraction. In `C = relu(matmul(A, B) + bias)`, the epilogue is `relu(... + bias)`.

**In this project:** Epilogue handling is architecturally critical. The epilogue must execute **after** the full K reduction, not per-K-step. `KernelFusionPass` fuses elementwise chains; `SquareBlockingPass` then tiles the epilogue on M, N and fuses the contraction producer via `tileConsumerAndFuseProducersUsingSCF`.

### Epilogue Fusion

See **Tile-and-Fuse**.

### ExecutionEngine (`mlir::ExecutionEngine`)

MLIR's interface to LLVM's OrcJIT. Takes an MLIR module in LLVM dialect, translates it to LLVM IR, runs the LLVM optimization pipeline, JIT-compiles to native code, and provides `invokePacked` for calling the compiled function.

**In this project:** `AdaptiveJITRunner` creates an `ExecutionEngine` with configurable optimization level (`ADAPTIVE_JIT_OPT`), loads `libmlir_c_runner_utils.so` for runtime `memrefCopy` support, and invokes the entry function via `invokePacked` with manually constructed argument arrays.

---

## F

### FileCheck

An LLVM utility that verifies text output against patterns. Used in `lit` tests to validate IR transformations.

```
// CHECK: scf.if
// CHECK-DAG: optimization_strategy = "square"
// CHECK-DAG: adaptive.datatype = "f32"
```

**In this project:** Three FileCheck tests validate router dispatch (`scf.if` emission), INT8 detection (datatype attributes), and GPU offload (`scf.forall` + block mapping).

### Fused Multiply-Add (FMA)

A single instruction that computes `a * b + c` with a single rounding step (more accurate and faster than separate multiply and add). On AVX2: `vfmadd231ps` operates on 8 × f32 in a single cycle.

**In this project:** The vectorized micro-kernel's inner loop body lowers to FMA instructions. Peak throughput: 2 FMA units × 8 f32/FMA × frequency = theoretical GFLOPS ceiling.

---

## G

### Gate Attribute

An attribute set on an MLIR operation to indicate that a specific transformation has been applied. Downstream passes check for gate attributes to avoid re-processing.

**In this project:** `SquareBlockingPass` uses three gate attributes:
- `square_l2_blocked` — L2 tiling complete
- `square_k_tiled` — L1 K-tiling complete
- `square_reg_tiled` — register tiling complete

Each level requires the previous level's gate and sets its own. This prevents infinite reprocessing when the pass walks the module multiple times.

### GPU Block Mapping

MLIR's `#gpu.block<x>` / `#gpu.block<y>` attributes that map parallel loop iterations to GPU thread block indices (`blockIdx.x`, `blockIdx.y` in CUDA).

**In this project:** `GPUOffloadPass` attaches `GPUBlockMappingAttr` to `scf.forall` loops: M → `block<y>`, N → `block<x>`. The standard `convert-scf-forall-to-gpu` pass transforms these into `gpu.launch` ops with the correct grid dimensions.

### Greedy Pattern Rewrite Driver

MLIR's `applyPatternsGreedily` function that applies a set of rewrite patterns repeatedly until no more patterns match (fixpoint) or a maximum iteration count is reached.

**In this project:** `KernelFusionPass` uses the greedy driver with `GreedyRewriteConfig{.maxIterations = 10, .useTopDownTraversal = true}` to fuse elementwise chains. Top-down traversal is preferred because producer-consumer relationships flow downward in the IR.

---

## I

### INT8 Quantization

Representing tensor values as 8-bit integers instead of 32-bit floats. Used in ML inference to reduce memory bandwidth and leverage integer SIMD instructions. The accumulator remains 32-bit integer (`i32`) to avoid overflow during the reduction.

**In this project:** The router detects `i8 × i8 → i32` matmuls and tags them with `adaptive.datatype = "int8"`. The vectorization pass scales the K-dimension vector tile by 4× because each SIMD register holds 4× more `i8` elements than `f32` elements.

### `invokePacked`

`ExecutionEngine::invokePacked(funcName, args)` calls a JIT-compiled function using the "packed" calling convention where `args` is a `void**` array. Each element points to one scalar argument of the LLVM function.

**In this project:** Each `memref<MxNxf32>` unpacks to 7 arguments: 2 pointers (base, data), 1 offset, 2 sizes, 2 strides. The JIT runner builds the `void**` array via `MemRef2D::packArgs()`.

### IRMapping

An MLIR utility (`mlir::IRMapping`) that maps values from one region to another during cloning. When `builder.clone(*op, mapping)` is called, `mapping` redirects operand references from the original op's operands to their cloned counterparts.

**In this project:** `RouterPass` uses `IRMapping` to clone matmul ops into `scf.if` then/else regions. Each clone operates on the original operands (the `scf.if` is inserted at the same point), but the cloned ops receive different strategy attributes.

### IRRewriter

A subclass of `OpBuilder` that tracks operation replacements and erasures. Unlike the greedy pattern driver, `IRRewriter` gives the pass author full control over the rewriting sequence.

**In this project:** All tiling passes (Square, Skinny, GPU) use `IRRewriter` instead of the greedy pattern driver. Walk-based rewriting with `IRRewriter` avoids the pattern ordering and invalidation issues that the greedy driver is known to have with tiling transformations.

---

## J

### JIT (Just-in-Time) Compilation

Translating IR to native machine code at runtime, immediately before execution. The compiled code lives in memory and is never written to disk.

**In this project:** `mlir::ExecutionEngine` wraps LLVM OrcJIT. The LLVM dialect module is translated to LLVM IR, optimized (O0–O3), and compiled to native x86. The JIT compilation step adds 100-500ms of overhead but enables runtime parameterization (matrix dimensions, optimization level) without re-running the MLIR pipeline.

---

## L

### Linalg Dialect (`linalg`)

MLIR's dialect for structured linear algebra operations. Provides named operations (`linalg.matmul`, `linalg.batch_matmul`, `linalg.generic`) with explicit loop semantics (iteration domains, indexing maps, reduction dimensions).

**Key properties exploited in this project:**
- `getStaticLoopRanges()` — extracts M, N, K dimensions for shape classification
- `isaContractionOpInterface` — identifies matmul-like operations
- `hasVectorizationImpl` — checks if the op can be vectorized
- `tileLinalgOp` — tiles a linalg op with specified tile sizes, producing `scf.for` loops

### Lowering

Transforming operations from a higher-level dialect to a lower-level dialect. Lowering is irreversible — information is lost (e.g., tensor semantics → memref, structured loops → branch-based control flow).

**In this project:** The lowering chain is: `linalg` → `vector` → `scf` → `cf` → `llvm`. Each step removes abstractions: tiling removes the matmul structure, vectorization removes the loop body, SCF-to-CF removes structured control flow, and dialect-to-LLVM removes all high-level types.

### `LowerVectorMultiReductionPass`

An MLIR pass that decomposes multi-dimensional `vector.multi_reduction` operations into nested 1D reductions. Required because LLVM IR only supports 1D vector types.

**In this project:** This pass is the critical bridge between MLIR's `vector<8x8x8xf32>` (produced by vectorizing a 3-level-tiled matmul) and LLVM's `<8 x float>` (the only vector type LLVM can codegen). Without it, `VectorToLLVM` fails with "cannot convert multi-dimensional vector to LLVM type." Placed after bufferization, before `VectorToSCF`.

---

## M

### MemRef (Memory Reference)

MLIR's type for a reference to a contiguous block of memory with known rank, element type, and layout. Analogous to a pointer + metadata (sizes, strides, offset).

```mlir
memref<256x256xf32>  // 2D, 256×256, row-major (default), f32 elements
```

**In this project:** After bufferization, all tensor types become memrefs. The JIT runner constructs `MemRef2D` POD structs that match the LLVM-lowered memref layout: `{basePtr, dataPtr, offset, sizes[2], strides[2]}`.

### MLIR (Multi-Level Intermediate Representation)

A compiler infrastructure for defining and transforming domain-specific IRs. Unlike LLVM IR (single-level, close to machine code), MLIR supports multiple abstraction levels (dialects) in the same module, with progressive lowering between them.

**In this project:** MLIR is the entire compilation substrate. The input is `linalg` dialect, transformations operate across `scf`, `vector`, `arith`, `tensor`, `memref`, `gpu`, and `cf` dialects, and the output is `llvm` dialect.

### MLIR Dialect

A namespace of operations, types, and attributes within MLIR. Each dialect represents a specific abstraction level or domain. Dialects are composable — a single module can contain operations from multiple dialects.

**Dialects used in this project:**

| Dialect | Abstraction Level | Operations Used |
|---------|-------------------|-----------------|
| `linalg` | Structured linear algebra | `matmul`, `generic`, `batch_matmul` |
| `tensor` | Immutable value tensors | `dim`, `extract_slice`, `insert_slice` |
| `memref` | Mutable memory references | `alloc`, `load`, `store`, `subview` |
| `scf` | Structured control flow | `for`, `if`, `forall`, `yield` |
| `arith` | Arithmetic | `addf`, `muli`, `cmpi`, `constant` |
| `vector` | Fixed-width SIMD vectors | `transfer_read`, `transfer_write`, `multi_reduction` |
| `gpu` | GPU abstractions | `GPUBlockMappingAttr` |
| `cf` | Unstructured control flow | `br`, `cond_br` |
| `llvm` | LLVM IR | `func`, `call`, `load`, `store`, `fadd` |

### ModuleOp (`mlir::ModuleOp`)

The top-level container operation in MLIR. All functions and global state live inside a `ModuleOp`. Passes operate on `ModuleOp` (module-level) or on `FuncOp` (function-level, via `addNestedPass`).

**In this project:** All passes are `OperationPass<ModuleOp>`. The JIT runner walks the module to find the entry `LLVMFuncOp` for signature introspection.

### Multi-Versioning

See **Polyhedral Fallback**.

---

## O

### One-Shot Bufferization

An MLIR bufferization strategy that performs a single global analysis to determine buffer allocation and reuse, then converts all tensor operations to memref operations in one pass. Avoids the phase-ordering issues of per-dialect bufferization.

**In this project:** `LowerToLLVMPass` uses `OneShotBufferize` with `bufferizeFunctionBoundaries = true` as the first step of lowering. The `bufferizeFunctionBoundaries` flag is required because the matmul function signature uses tensor types that must be converted to memref types.

### OrcJIT

LLVM's Just-in-Time compilation framework. Supports lazy compilation, concurrent compilation, and symbol resolution. Named after the character "Orc" in a video game (not an acronym).

**In this project:** `mlir::ExecutionEngine` wraps OrcJIT. The engine translates LLVM dialect → LLVM IR → native x86 machine code, all in memory. `CodeGenOptLevel::Aggressive` enables LLVM's full backend optimization pipeline (instruction selection, register allocation, instruction scheduling).

---

## P

### Pass (`mlir::Pass`)

A unit of IR transformation in MLIR. Passes are registered with the pass manager and executed in order. Each pass declares which operation type it operates on and which dialects it may introduce.

**In this project:** 13 custom passes are registered in `main.cpp`. Each is a `PassWrapper<T, OperationPass<ModuleOp>>` with `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` for RTTI-free type identification (required by `-fno-rtti`).

### PassManager (`mlir::PassManager`)

Orchestrates the execution of passes on an MLIR module. Handles pass dependencies, verification between passes, and nested pass execution.

**In this project:** `LowerToLLVMPass` constructs an **internal** `PassManager` to run the 17-step lowering pipeline. This is unusual — most passes perform a single transformation. The internal PassManager pattern is used because the 17 steps must execute in strict order and a failure at any step should abort the entire lowering.

### Polyhedral Fallback / Multi-Versioning

Generating multiple optimized code paths for the same operation, with a runtime condition selecting the appropriate path. Named after the polyhedral model of loop optimization, where loop bounds define a polyhedron and different optimizations apply to different regions.

**In this project:** `RouterPass` emits `scf.if` multi-versioned code for dynamic shapes. The runtime condition `max(M,N) > min(M,N) * 8` selects between skinny-optimized and square-optimized code paths. Both paths are fully compiled; the branch is resolved at execution time with near-zero overhead (branch predictor converges after 1-2 iterations for consistent shapes).

### `posix_memalign`

A POSIX function that allocates memory aligned to a specified boundary: `posix_memalign(&ptr, alignment, size)`. Unlike `aligned_alloc` (C11), it does not require `size` to be a multiple of `alignment`.

**In this project:** All JIT runner buffers are allocated with `posix_memalign(&ptr, 64, count * sizeof(float))`. The 64-byte alignment satisfies AVX-512 requirements.

---

## R

### Reduction Dimension

A loop dimension that is contracted (summed over) in a contraction operation. In `C[i][j] += A[i][k] * B[k][j]`, **K is the reduction dimension** — it does not appear in the output indices.

**In this project:** The K dimension receives special treatment: it is tiled at L1 level (Level 1 in `SquareBlockingPass`) to control the working set of A and B slices, and at register level (Level 2) to control vector spill size.

### `RLIMIT_STACK`

A POSIX resource limit controlling the maximum stack size for a process. Has two values: soft limit (current effective limit, adjustable by the process) and hard limit (maximum the soft limit can be raised to, set by the administrator).

**In this project:** `setrlimit(RLIMIT_STACK, {.rlim_cur = rl.rlim_max})` raises the soft limit to the hard limit before JIT execution. The kernel auto-grows the main thread's stack up to this limit via page fault handling.

---

## S

### SCF Dialect (`scf` — Structured Control Flow)

An MLIR dialect for structured loop and conditional constructs. Operations: `scf.for` (counted loop), `scf.while` (while loop), `scf.if` (conditional), `scf.forall` (parallel loop), `scf.yield` (return values from regions).

**In this project:**
- `scf.for` — produced by tiling (L2, L1, register loops)
- `scf.if` — produced by `RouterPass` for dynamic shape dispatch
- `scf.forall` — produced by `GPUOffloadPass` for GPU block mapping
- `scf.yield` — threads tensor values through tiled loop nests

### SPIR-V

An intermediate language for GPU compute and graphics shaders, defined by the Khronos Group. Target for Vulkan GPU compute kernels.

**In this project:** The GPU offload path produces `scf.forall` with GPU block mapping that is compatible with MLIR's `gpu-to-spirv` conversion. The full Vulkan/SPIR-V lowering and runtime execution is future work — the framework stops at the `scf.forall` abstraction level.

### Static Loop Ranges

The compile-time-known trip counts of a linalg operation's iteration domain. Extracted via `linalgOp.getStaticLoopRanges()`.

**In this project:** The router, tiling passes, and vectorization pass all use static loop ranges to determine M, N, K dimensions. Dimensions with value `ShapedType::kDynamic` (-1) are handled by the dynamic dispatch path.

---

## T

### Tensor Dialect (`tensor`)

MLIR's dialect for immutable, value-semantic tensor operations. `tensor.dim` extracts runtime dimension sizes; `tensor.extract_slice` produces sub-tensors without copying.

**In this project:** `RouterPass` uses `tensor.dim` to extract runtime M, N values for dynamic shapes. Tiling produces `tensor.extract_slice` / `tensor.insert_slice` pairs that are later eliminated by bufferization.

### Tile-and-Fuse

An MLIR transformation that tiles a consumer operation and fuses its producer into the tile loops. The producer executes inside the consumer's tile iteration, maximizing data locality.

```
// Before tile-and-fuse:
%matmul = linalg.matmul ...     // produces MxN result
%relu = linalg.generic (%matmul) // consumes entire MxN result

// After tile-and-fuse (tiling relu on M,N, fusing matmul):
scf.for M_tile:
  scf.for N_tile:
    %tiled_matmul = linalg.matmul [M_tile, N_tile, K_full]
    %tiled_relu = linalg.generic (%tiled_matmul)
```

**In this project:** `SquareBlockingPass` uses `tileConsumerAndFuseProducersUsingSCF` to tile the epilogue (relu/bias) on M, N and fuse the contraction producer into the tile loops. This ensures the epilogue runs after the full K reduction within each tile.

### Tiling

Partitioning a loop's iteration space into smaller blocks (tiles). Each tile processes a subset of the data, improving cache utilization.

```
// Before tiling: matmul(256, 256, 256)
// After tiling with {64, 64, 0}: 4×4 = 16 tiles of matmul(64, 64, 256)
```

A tile size of 0 means "don't tile this dimension."

**In this project:** `linalg::tileLinalgOp(rewriter, op, opts)` performs the tiling transformation. The tiled op is wrapped in `scf.for` loops and the original op is replaced with the loop results via `rewriter.replaceOp`.

### TilingInterface (`mlir::TilingInterface`)

An MLIR interface that operations must implement to be tiled. Provides methods for computing tile offsets, sizes, and generating the tiled computation.

**In this project:** `linalg.matmul` and `linalg.generic` both implement `TilingInterface`. `GPUOffloadPass` casts to `TilingInterface` before calling `tileUsingSCF` (the `scf.forall` tiling API).

---

## U

### Unpacked MemRef ABI

The calling convention where each memref argument is expanded into its constituent scalar values (pointers, offset, sizes, strides). This is the default ABI for MLIR's `ExecutionEngine`.

```
memref<256x256xf32> → (ptr, ptr, i64, i64, i64, i64, i64)
                       base data  off  M    N   str0  str1
```

**In this project:** `parseMemRefRanks()` reconstructs the memref ranks from the LLVM function signature by counting the number of `i64` arguments between consecutive pointer pairs.

---

## V

### Vector Dialect (`vector`)

MLIR's dialect for fixed-width SIMD vector operations. Supports multi-dimensional vectors (e.g., `vector<8x8xf32>`), which are a higher-level abstraction than LLVM's 1D vectors.

Key operations:
- `vector.transfer_read` — load from memref into vector (with optional padding)
- `vector.transfer_write` — store vector to memref
- `vector.multi_reduction` — reduce one or more dimensions of a multi-dimensional vector
- `vector.contract` — generalized contraction on vectors

**In this project:** `SIMDVectorizationPass` produces `vector` operations via `linalg::vectorize`. `LowerVectorMultiReductionPass` decomposes multi-dimensional vectors to 1D. `VectorToLLVM` converts 1D vector operations to LLVM intrinsics.

### Vectorization

Transforming scalar loop bodies into vector (SIMD) operations that process multiple data elements per instruction.

**In this project:** `mlir::linalg::vectorize(rewriter, op, vecSizes, scalableDims)` converts a `linalg.matmul` micro-kernel into `vector.contract` and `vector.transfer_read/write` operations. The `vecSizes` parameter specifies the vector tile dimensions (matching the register tile sizes from `SquareBlockingPass`).

### VNNI (Vector Neural Network Instructions)

Intel AVX-512 extension for accelerating INT8/INT16 dot products. `VPDPBUSD` computes `i8 × i8 → i32` dot product on 64 elements per cycle.

**In this project:** When `adaptive.datatype = "int8"`, the 4× vector scaling factor in `SIMDVectorizationPass` is designed to saturate VNNI's 64-element-per-cycle throughput on AVX-512 systems. On AVX2 systems without VNNI, the scaling still improves throughput by packing more `i8` elements per `VPMADDUBSW` instruction.

---

## W

### Walk (`mlir::Operation::walk`)

An MLIR method that traverses all operations nested within an operation (recursively). Used to collect transformation targets before modifying the IR.

**In this project:** Every pass uses `module.walk([&](OperationType op) { ... })` to collect target operations into a `SmallVector`, then iterates the vector to apply transformations. This two-phase approach (collect then modify) is required because modifying the IR during a walk invalidates the walk iterator.

---

## Y

### Yield (`scf.yield`)

An operation that returns values from an `scf.for`, `scf.if`, or `scf.forall` region. In MLIR's SSA-based tensor semantics, `scf.yield` is how updated tensor values "escape" from loop bodies.

**In this project:** `RouterPass` uses `scf.yield` to return the results of cloned matmul ops from `scf.if` then/else regions. Tiling produces `scf.yield` ops that thread the updated output tensor through `scf.for` iterations.
