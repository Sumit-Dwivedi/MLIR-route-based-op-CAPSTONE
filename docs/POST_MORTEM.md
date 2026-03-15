# Engineering Post-Mortem

**Framework:** Adaptive MLIR MatMul Optimization Framework
**Scope:** All engineering hurdles encountered during Phases 1–4
**Intended Audience:** Compiler engineers debugging similar MLIR/LLVM JIT pipelines

---

## Incident 1: L1 Cache Thrashing (Matrices > 256×256 Timeout)

### Symptoms

- 32×32, 64×64, 256×256: correct results in < 1 second
- 512×512: JIT execution never completes (timeout after 60+ seconds)
- 1024×1024: same behavior
- No crash, no error — the process simply hangs in the JIT-compiled inner loop

### Diagnosis

**Step 1:** Verified the IR was correct — `--print-ir-after=lower-to-llvm` showed valid LLVM dialect with the expected loop structure.

**Step 2:** Profiled the loop nest. The `SquareBlockingPass` (Phase 2 implementation) tiled only M and N at the L2 level with tile sizes `{tM, tN, tK}` where `tK = min(64, K)`. However, the tile specification was `{tM, tN, tK}` — **tiling all three dimensions at L2 level in a single pass**, not in a hierarchical sequence. After L2 tiling, the register tiling pass further tiled M and N but left K un-tiled at the register level.

The result: the innermost micro-kernel had dimensions `MxNxK_full` where K_full could be 512. For a 512×512 matrix with 64×64 L2 tiles:

```
Inner kernel: 8×8×512 (M_reg × N_reg × K_full)
B-slice per kernel: 512 × 8 × 4 bytes = 16KB
A-slice per kernel: 8 × 512 × 4 bytes = 16KB
C-tile: 8 × 8 × 4 bytes = 256 bytes
Total working set: ~32KB per micro-kernel invocation
```

The 32KB L1 data cache (2-way associative on most x86) cannot hold both A and B slices simultaneously. Every iteration of the K loop evicts B-slice cache lines loaded in the previous iteration — **classic L1 cache thrashing**. The effective memory bandwidth drops from ~100 GB/s (L1 hit) to ~30 GB/s (L2 hit), causing a 3× slowdown per micro-kernel. With 64×64 = 4096 micro-kernels per matrix, the aggregate slowdown makes 512×512 appear to "hang."

**Step 3:** Confirmed by running 256×256 (K=256, inner kernel 8×8×256): A+B working set = 16KB, which fits in L1 with room for C. This explains why 256×256 worked but 512×512 did not — the boundary is where the K-slice exceeds L1 capacity.

### Resolution

**Implemented 3-level blocking in `SquareBlockingPass`:**

1. **Level 0 (L2):** Tile M, N with `{tM, tN, 0}` — zeros for K means "don't tile K at this level"
2. **Level 1 (L1, NEW):** Tile K with `{0, 0, tK}` — zeros for M, N means "already tiled"
3. **Level 2 (Register):** Tile M, N, K with `{regM, regN, regK}` — 8×8×8 micro-kernels

**Gate attributes** (`square_l2_blocked`, `square_k_tiled`, `square_reg_tiled`) prevent re-processing across levels. Each level walks the module, finds ops with the prerequisite gates but missing its own gate, applies tiling, and sets its gate.

**Post-fix working set (512×512, 64/64/8 blocking):**
```
Inner kernel: 8×8×8
A-slice: 8 × 8 × 4 = 256 bytes
B-slice: 8 × 8 × 4 = 256 bytes
C-tile: 8 × 8 × 4 = 256 bytes
Total: 768 bytes — trivially fits in L1
```

**Result:** 512×512 dropped from timeout to 72.8ms. 1024×1024 completed in 668ms. 256×256 improved from 20.2ms to 8.2ms (2.5× faster due to better L1 utilization).

### Key Lesson

Tile size selection is meaningless without tile **nesting**. A single `{64, 64, 64}` tile produces the same inner kernel dimensions as no tiling at all for the K dimension — because `tileLinalgOp` with non-zero K creates a K loop that iterates once if K equals the tile size, but the inner kernel still has the full original K when K > tile size. The fix was to tile K **separately** in a second pass, creating a proper loop nest: `for_M(64) → for_N(64) → for_K(64) → kernel(8,8,8)`.

---

## Incident 2: AVX Memory Alignment Segfaults

### Symptoms

- JIT execution crashes with `SIGSEGV` during `invokePacked`
- Crash occurs inside the JIT-compiled function (not in MLIR/LLVM infrastructure)
- Intermittent: sometimes works for the same matrix size, sometimes crashes
- Consistently reproducible at 320×320+ with vectorization enabled

### Diagnosis

**Step 1:** Disabled vectorization (`SIMDVectorizationPass` skip all) — JIT execution succeeded at all sizes. This narrowed the issue to vectorized code paths.

**Step 2:** Examined the generated LLVM IR. The vectorized micro-kernel lowers to AVX2 instructions:

```llvm
%vec = load <8 x float>, ptr %addr, align 32
```

The `align 32` annotation tells LLVM the pointer is 32-byte aligned. LLVM emits `vmovaps` (aligned packed single-precision) which **faults on misaligned addresses**.

**Step 3:** Checked buffer allocation. The original code used `std::vector<float>`:

```c++
std::vector<float> bufA(M * K, 1.0f);
// bufA.data() alignment = alignof(float) = 4 bytes
// Some allocators over-align to 16 bytes, but never 32 or 64
```

**Step 4:** Verified with `printf("addr=%p\n", bufA.data())` — addresses were 16-byte aligned (typical glibc `malloc` behavior), not 32-byte aligned. The `vmovaps` instruction requires 32 bytes (AVX2) or 64 bytes (AVX-512).

**Why intermittent:** `malloc` returns addresses from a free-list. Depending on prior allocations, an address may happen to be 32-byte aligned by chance — causing the same binary to succeed on one run and crash on the next.

### Resolution

Replaced `std::vector<float>` with `posix_memalign`:

```c++
constexpr size_t kAlignment = 64;  // Satisfies both AVX2 (32) and AVX-512 (64)

auto alignedAlloc = [](size_t count, size_t align) -> float * {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, align, count * sizeof(float)) != 0)
    return nullptr;
  return static_cast<float *>(ptr);
};
```

Used 64-byte alignment (not 32) to future-proof for AVX-512 targets where `vmovaps zmm, [mem]` requires 64-byte alignment.

Added `free(rawBuffers[i])` cleanup — `posix_memalign` allocations must be freed with `free()`, not `delete[]`.

### Key Lesson

MLIR's bufferization and LLVM's code generator assume aligned memory. There is no runtime check — a misaligned `vmovaps` is an immediate `#GP` fault. Any JIT runner that feeds buffers to vectorized MLIR code **must** use aligned allocation. This is not documented in MLIR's `ExecutionEngine` API.

---

## Incident 3: OS Stack Overflow on Deeply Nested Loop Nests

### Symptoms

- JIT execution crashes with `SIGSEGV` at matrix sizes ≥ 320×320
- Crash occurs even with aligned buffers (Incident 2 fix applied)
- Works with `ulimit -s unlimited` but not with default 8MB stack
- Exact boundary: 319×319 PASS, 320×320 CRASH (first multiple of 64 beyond 256 that produces a deeper loop nest)

### Diagnosis

**Step 1:** Added a signal handler for `SIGSEGV` to print the faulting address. The address was near the bottom of the stack region (confirmed via `/proc/self/maps`).

**Step 2:** Tested with `ulimit -s 65536` (64MB) — 320×320 worked. Reduced to `ulimit -s 16384` (16MB) — still worked. This confirmed stack overflow, not a code bug.

**Step 3:** Analyzed the LLVM IR stack usage. Three-level tiling (64/64/8) + vectorization produces:

```
Loop depth: 6 (M_l2 × N_l2 × K_l1 × M_reg × N_reg × K_reg)
Per-level alloca: ~16 vector<8xf32> spill slots = 512 bytes
Nested frames: 6 levels × 512 bytes = ~3KB per innermost iteration
Total across iterations: variable, but LLVM may not fully merge allocas
```

At O2 optimization, LLVM merges some allocas but not all — the total stack usage scales with the number of live vector registers across the loop nest.

**Step 4: First fix attempt — `pthread_create` with 64MB stack:**

```c++
pthread_attr_setstacksize(&attr, 64 * 1024 * 1024);
pthread_create(&thread, &attr, jitWork, nullptr);
```

Result: 320×320 PASS, 512×512 CRASH. The pthread stack is allocated via `mmap` with a fixed size and a guard page at the bottom. The JIT code's stack access pattern skipped the guard page (accessing memory below it without touching it first), causing `SIGSEGV` on a non-stack page. This is a **stack clash** — the stack pointer jumps past the guard page into unmapped memory.

Increasing to 256MB pthread stack did not help — the access pattern, not the size, was the issue.

**Step 5: Second fix attempt — `setrlimit` on the main thread:**

```c++
struct rlimit rl;
getrlimit(RLIMIT_STACK, &rl);
rl.rlim_cur = rl.rlim_max;
setrlimit(RLIMIT_STACK, &rl);
```

Result: All sizes PASS. The main thread's stack uses the kernel's auto-growth mechanism — when a page fault occurs on the stack, the kernel extends the stack mapping regardless of whether the access was sequential. This handles the JIT code's non-sequential stack access pattern.

### Resolution

Applied `setrlimit(RLIMIT_STACK)` in `AdaptiveJITRunner.cpp` before JIT invocation. The call raises the soft limit to the hard limit (typically 64MB or `RLIM_INFINITY`). Removed the pthread approach entirely — direct main-thread execution is simpler and more robust.

### Key Lesson

`pthread_create` stacks and main-thread stacks have fundamentally different growth semantics:

- **Main thread:** Kernel handles page faults by extending the stack mapping. Grows on demand up to `RLIMIT_STACK`. Handles non-sequential access (stack clash safe).
- **pthread:** `mmap`'d with fixed size + guard page. No auto-growth. Non-sequential access past the guard page = immediate `SIGSEGV` on unmapped memory.

For JIT-compiled code with unpredictable stack access patterns (compiler-generated allocas, spill slots), **always use the main thread with an elevated stack limit**.

---

## Incident 4: FileCheck Test Brittleness

### Symptoms

- `lit` tests fail with `CHECK-SAME: expected string not found` even though the IR appears correct when printed manually
- Failures are sensitive to MLIR's IR printing format (whitespace, line breaks, attribute ordering)

### Diagnosis

**Step 1:** MLIR's IR printer may break a single operation across multiple lines. For example:

```mlir
%result = scf.if %cond -> (tensor<64x64xf32>) {
  // ...
}
```

vs.

```mlir
%result = scf.if %cond
    -> (tensor<64x64xf32>) {
  // ...
}
```

`CHECK-SAME` requires the matched text to be on the **same line** as the previous `CHECK`. If MLIR inserts a line break, `CHECK-SAME` fails.

**Step 2:** Attribute ordering in MLIR is not guaranteed. `{optimization_strategy = "square", adaptive.datatype = "f32"}` may print as `{adaptive.datatype = "f32", optimization_strategy = "square"}` depending on the internal attribute storage order.

### Resolution

1. **Replaced `CHECK-SAME` with `CHECK` where line breaks are possible.** `CHECK` matches anywhere on subsequent lines; `CHECK-SAME` requires the same line.

2. **Used `CHECK-DAG` for attribute checking** where ordering is non-deterministic:
   ```
   // CHECK-DAG: optimization_strategy = "square"
   // CHECK-DAG: adaptive.datatype = "f32"
   ```

3. **Anchored checks to unique IR patterns** rather than full operation strings:
   ```
   // CHECK: arith.cmpi sgt
   // CHECK: scf.if
   ```
   Instead of:
   ```
   // CHECK: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : i64
   ```

### Key Lesson

FileCheck tests should be **resilient to IR formatting changes**. Use:
- `CHECK` (not `CHECK-SAME`) unless same-line co-occurrence is the actual invariant
- `CHECK-DAG` for unordered attribute sets
- `%{{.*}}` for SSA value names (they are not stable across MLIR versions)
- Minimal matching: check the opcode and critical operands, not the full operation string

---

## Incident 5: 2D/3D Vector Lowering Failure

### Symptoms

- `LowerToLLVMPass` fails with: `cannot convert multi-dimensional vector to LLVM type`
- Failure occurs at the `VectorToLLVM` conversion step
- Only happens when vectorization produces vectors with rank > 1 (e.g., `vector<8x8xf32>` or `vector<8x8x8xf32>`)
- Scalar (un-vectorized) paths lower successfully

### Diagnosis

**Step 1:** Examined the IR after vectorization. `SIMDVectorizationPass` produces:

```mlir
%result = vector.multi_reduction <add>, %vec, %acc
    [2] : vector<8x8x8xf32> to vector<8x8xf32>
```

This is a 3D vector reduced along dimension 2 to a 2D vector.

**Step 2:** LLVM IR supports only 1D vectors (`<8 x float>`, `<16 x float>`). There is no LLVM type for `<8 x 8 x float>`. The `VectorToLLVM` conversion pass cannot translate `vector<8x8xf32>` — it requires all vectors to be 1D before conversion.

**Step 3:** Checked MLIR's own test suite. The canonical lowering pipeline (`test-lower-to-llvm`) includes `--test-vector-multi-reduction-lowering-patterns` before `--convert-vector-to-llvm`. This pass decomposes multi-dimensional reductions into nested 1D reductions.

### Resolution

Added `LowerVectorMultiReductionPass` to the lowering pipeline **before** `VectorToSCF` and `VectorToLLVM`:

```c++
pm.addNestedPass<mlir::func::FuncOp>(
    mlir::vector::createLowerVectorMultiReductionPass(
        mlir::vector::VectorMultiReductionLowering::InnerParallel));
```

The `InnerParallel` strategy decomposes:

```
vector.multi_reduction<add, vector<8x8x8xf32>> [2]
```

into:

```
// 8 iterations of:
vector.reduction<add, vector<8xf32>>  // 1D reduction, maps to LLVM
```

The resulting `vector<8xf32>` operations map directly to 256-bit YMM registers (AVX2) or ZMM registers (AVX-512).

**Placement is critical:** This pass must run:
- **After** bufferization (operates on memref, not tensor)
- **Before** `VectorToSCF` (which also cannot handle multi-dimensional vectors)
- **Before** `VectorToLLVM` (the pass that actually fails without it)

A canonicalize + CSE pass follows to clean up redundant operations introduced by the decomposition.

### Key Lesson

MLIR's vector dialect is richer than LLVM's vector type system. Any pipeline that vectorizes to rank > 1 (which `linalg::vectorize` does for matmul) **must** include `LowerVectorMultiReductionPass` before `VectorToLLVM`. This is not enforced by the MLIR infrastructure — it is a pipeline composition requirement that must be discovered through failure or by reading MLIR's own test pipelines.

---

## Summary of Engineering Debt Resolved

| Incident | Time to Diagnose | Time to Fix | Root Cause Category |
|----------|-----------------|-------------|-------------------|
| L1 Cache Thrashing | ~3 hours | ~2 hours | Algorithm design (missing tiling level) |
| AVX Alignment | ~1 hour | ~30 minutes | Systems programming (memory allocator) |
| Stack Overflow | ~4 hours | ~1 hour | OS internals (stack growth semantics) |
| FileCheck Brittleness | ~1 hour | ~30 minutes | Test infrastructure (format sensitivity) |
| 2D Vector Lowering | ~2 hours | ~15 minutes | Pipeline composition (missing pass) |

**Total debugging time:** ~12 hours across 5 incidents. The stack overflow incident consumed the most time due to the misleading pthread detour — the initial hypothesis (insufficient stack size) was correct, but the fix mechanism (pthread vs setrlimit) required understanding kernel-level stack management semantics.
