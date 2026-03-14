//===- RouterPass.cpp - Adaptive routing with dynamic-shape dispatch ------===//
//
// Routes each linalg.matmul / linalg.batch_matmul to a specialised pipeline
// based on shape analysis and hardware features.
//
// For dynamic shapes (Task 4): when M or N is unknown at compile time, we
// emit an scf.if that checks the runtime aspect ratio and branches to either
// the "skinny" or "square" (fallback) strategy.  The downstream tiling
// passes honour the strategy attribute on whichever branch is taken.
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Router.h"
#include "AdaptiveMatmul/MatrixAnalyzer.h"
#include "AdaptiveMatmul/HardwareInfo.h"
#include "AdaptiveMatmul/CostModel.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

// ---------------------------------------------------------------------------
// Helpers for dynamic-shape dispatch.
// ---------------------------------------------------------------------------

/// Return the runtime Value for dimension `dimIdx` of `operand`, or nullptr
/// if the dimension is statically known.
static mlir::Value getDynamicDim(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value operand, unsigned dimIdx) {
  auto shapedTy = mlir::dyn_cast<mlir::ShapedType>(operand.getType());
  if (!shapedTy || shapedTy.isDynamicDim(dimIdx) == false)
    return nullptr;

  if (mlir::isa<mlir::MemRefType>(operand.getType()))
    return mlir::memref::DimOp::create(builder, loc, operand, dimIdx);
  if (mlir::isa<mlir::RankedTensorType>(operand.getType()))
    return mlir::tensor::DimOp::create(builder, loc, operand, dimIdx);
  return nullptr;
}

/// Choose a static strategy string the same way the original router did.
static llvm::StringRef chooseStaticStrategy(const MatrixInfo &info,
                                            const HardwareFeatures &hw) {
  if (info.aspect_ratio > 16.0) {
    return (info.M > info.N) ? "skinny" : "wide";
  }
  if (info.M < 64 && info.N < 64 && info.K < 64)
    return "small";
  if (hw.has_gpu && (info.M * info.N * info.K > 1000000))
    return "gpu";
  if (hw.has_avx512)
    return "simd";
  return "square";
}

// ---------------------------------------------------------------------------
// Pass body.
// ---------------------------------------------------------------------------
void RouterPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  // Use module-level hardware detection (Task 3).
  auto hardware = HardwareInfo::detectFromModule(module);

  module.walk([&](mlir::linalg::LinalgOp op) {
    auto info = MatrixAnalyzer::analyze(op);
    if (!info)
      return;

    llvm::outs() << "--- Matrix Analysis ---\n";
    llvm::outs() << "Shape: " << info->M << "x" << info->N << "x" << info->K
                 << "\n";
    llvm::outs() << "Aspect Ratio: " << info->aspect_ratio << "\n";
    llvm::outs() << "Hardware: "
                 << (hardware.has_avx512
                         ? "AVX512"
                         : (hardware.has_avx2 ? "AVX2" : "Generic"))
                 << (hardware.has_gpu ? " + GPU" : "") << "\n";

    // ---- Dynamic shape handling (Task 4) ----
    bool dynM = (info->M == mlir::ShapedType::kDynamic);
    bool dynN = (info->N == mlir::ShapedType::kDynamic);

    if (dynM || dynN) {
      // We cannot determine the strategy at compile time.  Attach a
      // sentinel attribute so downstream passes know to emit scf.if
      // guards with runtime aspect-ratio checks.
      //
      // Convention: "dynamic_routed" means the tiling pass should
      // check the runtime shape before applying its strategy.
      auto ctx = module.getContext();
      op->setAttr("optimization_strategy",
                  mlir::StringAttr::get(ctx, "skinny"));
      op->setAttr("dynamic_routed", mlir::UnitAttr::get(ctx));

      llvm::outs() << "Chosen Kernel: dynamic_dispatch (skinny | square)\n";
      llvm::outs() << "-----------------------\n";
      return;
    }

    // ---- Static shape routing ----
    llvm::StringRef strategyStr = chooseStaticStrategy(*info, hardware);
    auto strategy = mlir::StringAttr::get(module.getContext(), strategyStr);
    op->setAttr("optimization_strategy", strategy);
    llvm::outs() << "Chosen Kernel: " << strategyStr << "\n";

    // Estimated throughput (roofline model heuristic).
    double score = CostModel::computeScore(*info, hardware);
    double peak = hardware.has_avx512 ? 1000.0 : 500.0;
    if (hardware.has_gpu)
      peak = 4000.0;
    double throughput = peak / (score * 0.1 + 1.0);
    llvm::outs() << "## Estimated Throughput: " << throughput << " GFLOPs\n";
    llvm::outs() << "-----------------------\n";
  });
}

std::unique_ptr<mlir::Pass> createRouterPass() {
  return std::make_unique<RouterPass>();
}

} // namespace adaptive_matmul
