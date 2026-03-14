#include "AdaptiveMatmul/Router.h"
#include "AdaptiveMatmul/MatrixAnalyzer.h"
#include "AdaptiveMatmul/HardwareInfo.h"
#include "AdaptiveMatmul/CostModel.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

void RouterPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto hardware = HardwareInfo::detect();

  module.walk([&](mlir::linalg::LinalgOp op) {
    auto info = MatrixAnalyzer::analyze(op);
    if (info) {
      double score = CostModel::computeScore(*info, hardware);
      
      llvm::outs() << "--- Matrix Analysis ---\n";
      llvm::outs() << "Shape: " << info->M << "x" << info->N << "x" << info->K << "\n";
      llvm::outs() << "Aspect Ratio: " << info->aspect_ratio << "\n";
      llvm::outs() << "Hardware: " << (hardware.has_avx512 ? "AVX512" : (hardware.has_avx2 ? "AVX2" : "Generic")) 
                   << (hardware.has_gpu ? " + GPU" : "") << "\n";

      mlir::StringAttr strategy;
      
      // Advanced Routing Logic (Research-grade)
      if (info->aspect_ratio > 16.0) {
          if (info->M > info->N) {
              strategy = mlir::StringAttr::get(module.getContext(), "skinny");
          } else {
              strategy = mlir::StringAttr::get(module.getContext(), "wide");
          }
      } else if (info->M < 64 && info->N < 64 && info->K < 64) {
          strategy = mlir::StringAttr::get(module.getContext(), "small");
      } else if (hardware.has_gpu && (info->M * info->N * info->K > 1000000)) {
          strategy = mlir::StringAttr::get(module.getContext(), "gpu");
      } else if (hardware.has_avx512) {
          strategy = mlir::StringAttr::get(module.getContext(), "simd");
      } else {
          strategy = mlir::StringAttr::get(module.getContext(), "square");
      }

      op->setAttr("optimization_strategy", strategy);
      llvm::outs() << "Chosen Kernel: " << strategy.getValue() << "\n";
      
      // Estimated throughput (mocked based on Roofline Model concept)
      double peak = hardware.has_avx512 ? 1000.0 : 500.0;
      double throughput = peak / (score * 0.1 + 1.0); 
      llvm::outs() << "## Estimated Throughput: " << throughput << " GFLOPs\n";
      llvm::outs() << "-----------------------\n";
    }
  });
}

std::unique_ptr<mlir::Pass> createRouterPass() {
  return std::make_unique<RouterPass>();
}

} // namespace adaptive_matmul
