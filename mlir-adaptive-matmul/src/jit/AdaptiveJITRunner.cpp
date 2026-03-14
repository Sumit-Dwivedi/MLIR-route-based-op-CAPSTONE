//===- AdaptiveJITRunner.cpp - JIT execution + benchmarking ---------------===//
//
// Takes an MLIR module already lowered to LLVM dialect, JIT-compiles it via
// mlir::ExecutionEngine, and benchmarks execution with std::chrono timers.
//
// Usage from adaptive-opt:
//   adaptive-opt input.mlir --lower-to-llvm --jit-run
//
// The pass allocates input/output buffers, invokes the JIT-compiled function,
// and reports wall-clock execution time.
//
// The LLVM function uses the unpacked memref ABI where each memref<MxNxf32>
// argument becomes 7 scalar values: (basePtr, dataPtr, offset, size0, size1,
// stride0, stride1). The return value is a memref descriptor struct.
// invokePacked expects void** where each element points to one scalar arg.
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace adaptive_matmul {

/// POD memref descriptors matching MLIR's flat ABI layout exactly.
/// These MUST be plain structs with no constructors/destructors so the JIT
/// can safely write raw bytes into them.
struct MemRef2D {
  float *basePtr;
  float *dataPtr;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];

  void packArgs(llvm::SmallVectorImpl<void *> &args) {
    args.push_back(&basePtr);
    args.push_back(&dataPtr);
    args.push_back(&offset);
    args.push_back(&sizes[0]);
    args.push_back(&sizes[1]);
    args.push_back(&strides[0]);
    args.push_back(&strides[1]);
  }
};

struct MemRef1D {
  float *basePtr;
  float *dataPtr;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];

  void packArgs(llvm::SmallVectorImpl<void *> &args) {
    args.push_back(&basePtr);
    args.push_back(&dataPtr);
    args.push_back(&offset);
    args.push_back(&sizes[0]);
    args.push_back(&strides[0]);
  }
};

static MemRef2D makeDesc2D(float *data, int64_t M, int64_t N) {
  MemRef2D d{};
  d.basePtr = data;
  d.dataPtr = data;
  d.offset = 0;
  d.sizes[0] = M;
  d.sizes[1] = N;
  d.strides[0] = N;
  d.strides[1] = 1;
  return d;
}

static MemRef1D makeDesc1D(float *data, int64_t N) {
  MemRef1D d{};
  d.basePtr = data;
  d.dataPtr = data;
  d.offset = 0;
  d.sizes[0] = N;
  d.strides[0] = 1;
  return d;
}

/// Parse the LLVM function signature to determine argument memref ranks.
/// Each memref<MxNxf32> becomes (ptr, ptr, i64, i64, i64, i64, i64) = 7 args.
/// Each memref<Nxf32> becomes (ptr, ptr, i64, i64, i64) = 5 args.
/// Returns a list of ranks (2 for 2D, 1 for 1D).
static llvm::SmallVector<unsigned>
parseMemRefRanks(mlir::LLVM::LLVMFuncOp func) {
  llvm::SmallVector<unsigned> ranks;
  auto funcTy = func.getFunctionType();
  unsigned numArgs = funcTy.getNumParams();

  unsigned i = 0;
  while (i < numArgs) {
    // Each memref starts with 2 ptr args.
    auto paramTy = funcTy.getParamType(i);
    if (!mlir::isa<mlir::LLVM::LLVMPointerType>(paramTy))
      break;

    // Count: 2 ptrs + offset(i64) + sizes + strides.
    // 2D memref = 7 args (ptr, ptr, i64, i64, i64, i64, i64)
    // 1D memref = 5 args (ptr, ptr, i64, i64, i64)
    // Detect by counting remaining i64 args after the 2 ptrs.
    unsigned j = i + 2;
    while (j < numArgs &&
           !mlir::isa<mlir::LLVM::LLVMPointerType>(funcTy.getParamType(j)))
      ++j;

    unsigned numI64 = j - i - 2; // offset + sizes + strides
    // rank = (numI64 - 1) / 2  (1 offset + rank sizes + rank strides)
    unsigned rank = (numI64 - 1) / 2;
    ranks.push_back(rank);
    i = j;
  }
  return ranks;
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

  // Register MLIR runtime utilities (provides memrefCopy, etc.)
  // needed when tile-and-fuse bufferization generates memref.copy ops.
  llvm::SmallVector<llvm::StringRef> sharedLibs;
  if (const char *libPath = std::getenv("MLIR_RUNNER_UTILS"))
    sharedLibs.push_back(libPath);
  else
    sharedLibs.push_back(
        "/home/sumit/mlir-build/lib/libmlir_c_runner_utils.so");
  opts.sharedLibPaths = sharedLibs;

  auto maybeEngine = mlir::ExecutionEngine::create(module, opts);
  if (!maybeEngine) {
    llvm::errs() << "[JITRunner] Failed to create ExecutionEngine: "
                 << llvm::toString(maybeEngine.takeError()) << "\n";
    signalPassFailure();
    return;
  }
  auto &engine = *maybeEngine;

  // ---- Detect the entry function ----
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

  // ---- Parse function signature to determine memref ranks ----
  auto ranks = parseMemRefRanks(entryFunc);
  unsigned numMemRefs = ranks.size();

  llvm::outs() << "[JITRunner] Detected " << numMemRefs << " memref arg(s): ";
  for (unsigned i = 0; i < numMemRefs; ++i)
    llvm::outs() << (i ? ", " : "") << ranks[i] << "D";
  llvm::outs() << "\n";

  // ---- Determine matrix dimensions ----
  // Convention: first 2 args are A(MxK), B(KxN); last 2D arg is C(MxN).
  // Any 1D args between B and C are extra inputs (e.g., bias vector).
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

  // ---- Allocate buffers for each argument ----
  // Convention: arg0 = A(MxK), arg1 = B(KxN), last 2D = C(MxN) output,
  // middle 1D args = vectors of size N (e.g., bias).
  llvm::SmallVector<std::vector<float>> buffers(numMemRefs);
  llvm::SmallVector<MemRef2D> descs2D(numMemRefs);
  llvm::SmallVector<MemRef1D> descs1D(numMemRefs);

  for (unsigned i = 0; i < numMemRefs; ++i) {
    if (ranks[i] == 2) {
      if (i == 0) {
        buffers[i].assign(M * K, 1.0f);
        descs2D[i] = makeDesc2D(buffers[i].data(), M, K);
      } else if (i == 1) {
        buffers[i].assign(K * N, 1.0f);
        descs2D[i] = makeDesc2D(buffers[i].data(), K, N);
      } else {
        buffers[i].assign(M * N, 0.0f);
        descs2D[i] = makeDesc2D(buffers[i].data(), M, N);
      }
    } else if (ranks[i] == 1) {
      buffers[i].assign(N, 0.0f);
      descs1D[i] = makeDesc1D(buffers[i].data(), N);
    }
  }

  // Result descriptor — POD struct so JIT can write raw bytes safely.
  MemRef2D descResult{};

  auto buildArgs = [&]() -> llvm::SmallVector<void *> {
    llvm::SmallVector<void *> args;
    for (unsigned i = 0; i < numMemRefs; ++i) {
      if (ranks[i] == 2)
        descs2D[i].packArgs(args);
      else
        descs1D[i].packArgs(args);
    }
    args.push_back(&descResult);
    return args;
  };

  auto resetOutputs = [&]() {
    for (unsigned i = 0; i < numMemRefs; ++i) {
      if (ranks[i] == 2 && i >= 2) {
        std::memset(buffers[i].data(), 0, buffers[i].size() * sizeof(float));
        descs2D[i] = makeDesc2D(buffers[i].data(), M, N);
      }
    }
    std::memset(&descResult, 0, sizeof(descResult));
  };

  // ---- Warmup runs ----
  for (int i = 0; i < warmupRuns; ++i) {
    resetOutputs();
    auto args = buildArgs();
    auto err =
        engine->invokePacked(funcName, llvm::MutableArrayRef(args));
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
    resetOutputs();
    auto args = buildArgs();

    auto start = std::chrono::high_resolution_clock::now();
    auto err =
        engine->invokePacked(funcName, llvm::MutableArrayRef(args));
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
  float *resultData = descResult.dataPtr;
  if (resultData) {
    float expected = static_cast<float>(K);
    float actual = resultData[0];
    if (std::abs(actual - expected) > 1e-3f * expected) {
      llvm::outs() << "  WARNING: C[0][0] = " << actual << ", expected "
                   << expected << "\n";
    } else {
      llvm::outs() << "  Correctness: PASS (C[0][0] = " << actual << ")\n";
    }
  }
}

} // namespace adaptive_matmul
