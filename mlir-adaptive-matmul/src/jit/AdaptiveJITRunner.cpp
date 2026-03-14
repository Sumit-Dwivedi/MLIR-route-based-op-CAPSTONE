//===- AdaptiveJITRunner.cpp - JIT execution + benchmarking ---------------===//
//
// Takes an MLIR module already lowered to LLVM dialect, JIT-compiles it via
// mlir::ExecutionEngine, and benchmarks execution with std::chrono timers.
//
// Usage from adaptive-opt:
//   adaptive-opt input.mlir --adaptive-router --square-blocking \
//       --simd-vectorization --lower-to-llvm --jit-run
//
// The pass allocates input/output buffers, invokes the JIT-compiled function,
// and reports wall-clock execution time. This replaces the Python-based
// simulated timing with real hardware measurement.
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace adaptive_matmul {

/// Allocate a flat float buffer and fill with a value.
static std::vector<float> allocBuffer(int64_t size, float fillVal) {
  std::vector<float> buf(size, fillVal);
  return buf;
}

/// Build a StridedMemRefType<float, 2> descriptor pointing into `data`.
static StridedMemRefType<float, 2> makeMemRef2D(float *data, int64_t M,
                                                  int64_t N) {
  StridedMemRefType<float, 2> ref;
  ref.basePtr = data;
  ref.data = data;
  ref.offset = 0;
  ref.sizes[0] = M;
  ref.sizes[1] = N;
  ref.strides[0] = N;
  ref.strides[1] = 1;
  return ref;
}

void JITRunnerPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  // ---- Initialize LLVM native target ----
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Register LLVM IR translation.
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  module->getContext()->appendDialectRegistry(registry);

  // ---- Create ExecutionEngine with O2 optimization ----
  mlir::ExecutionEngineOptions opts;
  opts.transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  opts.enableObjectDump = false;

  auto maybeEngine = mlir::ExecutionEngine::create(module, opts);
  if (!maybeEngine) {
    llvm::errs() << "[JITRunner] Failed to create ExecutionEngine: "
                 << llvm::toString(maybeEngine.takeError()) << "\n";
    signalPassFailure();
    return;
  }
  auto &engine = *maybeEngine;

  // ---- Detect the entry function ----
  // Look for the first non-declaration function in the module.
  mlir::LLVM::LLVMFuncOp entryFunc;
  module.walk([&](mlir::LLVM::LLVMFuncOp func) {
    if (!func.isExternal() && !entryFunc)
      entryFunc = func;
  });

  if (!entryFunc) {
    llvm::errs() << "[JITRunner] No entry function found in module.\n";
    signalPassFailure();
    return;
  }

  llvm::StringRef funcName = entryFunc.getName();
  llvm::outs() << "[JITRunner] Entry function: " << funcName << "\n";

  // ---- Determine matrix dimensions from function signature ----
  // Convention: the function takes 3 memref<MxNxf32> arguments (A, B, C)
  // and returns a memref descriptor. We parse M, N from the first argument.
  //
  // For the packed interface, ExecutionEngine wraps to:
  //   void _mlir_<funcName>(void **args)
  // where args[0..2] = pointers to input memref descriptors,
  //       args[3]    = pointer to output memref descriptor.

  // Default dimensions — overridden by matrix_m/matrix_n env vars.
  int64_t M = 256, N = 256, K = 256;
  if (const char *envM = std::getenv("ADAPTIVE_M"))
    M = std::atol(envM);
  if (const char *envN = std::getenv("ADAPTIVE_N"))
    N = std::atol(envN);
  if (const char *envK = std::getenv("ADAPTIVE_K"))
    K = std::atol(envK);

  int warmupRuns = 3;
  int benchRuns = 10;
  if (const char *envR = std::getenv("ADAPTIVE_RUNS"))
    benchRuns = std::atoi(envR);

  llvm::outs() << "[JITRunner] Matrix: " << M << "x" << K << " * " << K << "x"
               << N << " (" << benchRuns << " runs)\n";

  // ---- Allocate buffers ----
  auto bufA = allocBuffer(M * K, 1.0f);
  auto bufB = allocBuffer(K * N, 1.0f);
  auto bufC = allocBuffer(M * N, 0.0f);

  auto refA = makeMemRef2D(bufA.data(), M, K);
  auto refB = makeMemRef2D(bufB.data(), K, N);
  auto refC = makeMemRef2D(bufC.data(), M, N);

  // Output descriptor — same layout as C.
  auto bufOut = allocBuffer(M * N, 0.0f);
  auto refOut = makeMemRef2D(bufOut.data(), M, N);

  // ---- Warmup runs ----
  for (int i = 0; i < warmupRuns; ++i) {
    void *args[] = {&refA, &refB, &refC, &refOut};
    auto err = engine->invokePacked(funcName, llvm::MutableArrayRef(args));
    if (err) {
      llvm::errs() << "[JITRunner] Warmup invocation failed: "
                   << llvm::toString(std::move(err)) << "\n";
      signalPassFailure();
      return;
    }
  }

  // ---- Benchmark runs ----
  std::vector<double> timings;
  timings.reserve(benchRuns);

  for (int i = 0; i < benchRuns; ++i) {
    // Reset output buffer.
    std::memset(bufOut.data(), 0, bufOut.size() * sizeof(float));
    std::memset(bufC.data(), 0, bufC.size() * sizeof(float));
    refOut = makeMemRef2D(bufOut.data(), M, N);
    refC = makeMemRef2D(bufC.data(), M, N);

    void *args[] = {&refA, &refB, &refC, &refOut};

    auto start = std::chrono::high_resolution_clock::now();
    auto err = engine->invokePacked(funcName, llvm::MutableArrayRef(args));
    auto end = std::chrono::high_resolution_clock::now();

    if (err) {
      llvm::errs() << "[JITRunner] Invocation " << i
                   << " failed: " << llvm::toString(std::move(err)) << "\n";
      signalPassFailure();
      return;
    }

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    timings.push_back(ms);
  }

  // ---- Report results ----
  std::sort(timings.begin(), timings.end());
  double median = timings[timings.size() / 2];
  double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
  double mean = sum / timings.size();
  double minT = timings.front();
  double maxT = timings.back();

  // GFLOPS: 2*M*N*K operations for matmul.
  double gflops = (2.0 * M * N * K) / (median * 1e-3) / 1e9;

  llvm::outs() << "[JITRunner] Results (" << benchRuns << " runs):\n"
               << "  Median:  " << llvm::format("%.3f", median) << " ms\n"
               << "  Mean:    " << llvm::format("%.3f", mean) << " ms\n"
               << "  Min:     " << llvm::format("%.3f", minT) << " ms\n"
               << "  Max:     " << llvm::format("%.3f", maxT) << " ms\n"
               << "  GFLOPS:  " << llvm::format("%.2f", gflops) << "\n";

  // ---- Sanity check: C[0][0] should be K * 1.0 * 1.0 = K ----
  float expected = static_cast<float>(K);
  float actual = bufOut[0];
  if (std::abs(actual - expected) > 1e-3f * expected) {
    llvm::outs() << "  WARNING: C[0][0] = " << actual << ", expected "
                 << expected << "\n";
  } else {
    llvm::outs() << "  Correctness: PASS (C[0][0] = " << actual << ")\n";
  }
}

} // namespace adaptive_matmul
