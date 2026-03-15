//===- LowerToLLVMPass.cpp - Full pipeline: vectorized IR → LLVM dialect --===//
//
// Mirrors the canonical MLIR lowering pipeline (test-lower-to-llvm) with
// one-shot bufferization prepended for tensor → memref conversion.
//
// Pipeline:
//   1. One-Shot Bufferize          (tensor → memref)
//   2. Vector → SCF                (transfer ops → loops)
//   3. Linalg → Loops              (residual linalg → scf)
//   4. Affine → Standard           (affine maps → arith)
//   5. SCF → ControlFlow           (scf → cf)
//   6. Canonicalize + CSE
//   7. Vector → LLVM
//   8. Math → LLVM
//   9. ExpandStridedMetadata       (complex memref ops → primitives)
//  10. Affine → Standard           (affine from expansion)
//  11. MemRef → LLVM (finalize)
//  12. Func → LLVM
//  13. Arith → LLVM
//  14. ControlFlow → LLVM
//  15. Index → LLVM
//  16. UB → LLVM
//  17. Reconcile Unrealized Casts
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

void LowerToLLVMPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::PassManager pm(module->getName());

  // ---- Phase 1: Bufferization (tensor → memref) ----
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // ---- Phase 2: Vector dimension lowering (2D/3D → 1D) ----
  // Lower multi_reduction and contract before VectorToSCF/VectorToLLVM,
  // because LLVM IR does not support multi-dimensional vectors.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::vector::createLowerVectorMultiReductionPass(
          mlir::vector::VectorMultiReductionLowering::InnerParallel));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // ---- Phase 3: High-level dialect lowering ----
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToSCFPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // ---- Phase 4: Dialect → LLVM conversions ----
  pm.addPass(mlir::createConvertVectorToLLVMPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createUBToLLVMConversionPass());

  // ---- Phase 4: Cleanup ----
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "[LowerToLLVM] Pipeline failed.\n";
    signalPassFailure();
    return;
  }

  llvm::outs() << "[LowerToLLVM] Full lowering to LLVM dialect complete.\n";
}

} // namespace adaptive_matmul
