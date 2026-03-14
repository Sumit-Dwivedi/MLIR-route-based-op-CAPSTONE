#ifndef ADAPTIVE_MATMUL_PASSES_H
#define ADAPTIVE_MATMUL_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace adaptive_matmul {

class SkinnyTilingPass : public mlir::PassWrapper<SkinnyTilingPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SkinnyTilingPass)
  llvm::StringRef getArgument() const final { return "skinny-tiling"; }
  llvm::StringRef getDescription() const final { return "Tiling optimization for skinny matrices."; }
  void runOnOperation() override;
};

class SkinnyVectorizationPass : public mlir::PassWrapper<SkinnyVectorizationPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SkinnyVectorizationPass)
  llvm::StringRef getArgument() const final { return "skinny-vectorization"; }
  llvm::StringRef getDescription() const final { return "Vectorization optimization for skinny matrices."; }
  void runOnOperation() override;
};

class SkinnyMemoryPass : public mlir::PassWrapper<SkinnyMemoryPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SkinnyMemoryPass)
  llvm::StringRef getArgument() const final { return "skinny-memory"; }
  llvm::StringRef getDescription() const final { return "Memory optimization for skinny matrices."; }
  void runOnOperation() override;
};

class GenericTilingPass : public mlir::PassWrapper<GenericTilingPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenericTilingPass)
  llvm::StringRef getArgument() const final { return "generic-tiling"; }
  llvm::StringRef getDescription() const final { return "Generic tiling for square matrices."; }
  void runOnOperation() override;
};

class SmallMatrixVectorPass : public mlir::PassWrapper<SmallMatrixVectorPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SmallMatrixVectorPass)
  llvm::StringRef getArgument() const final { return "small-matrix-vector"; }
  llvm::StringRef getDescription() const final { return "Vectorization for small matrices."; }
  void runOnOperation() override;
};

class WideTilingPass : public mlir::PassWrapper<WideTilingPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WideTilingPass)
  llvm::StringRef getArgument() const final { return "wide-tiling"; }
  llvm::StringRef getDescription() const final { return "Tiling optimization for wide matrices."; }
  void runOnOperation() override;
};

class SquareBlockingPass : public mlir::PassWrapper<SquareBlockingPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SquareBlockingPass)
  llvm::StringRef getArgument() const final { return "square-blocking"; }
  llvm::StringRef getDescription() const final { return "Blocking optimization for square matrices."; }
  void runOnOperation() override;
};

class GPUOffloadPass : public mlir::PassWrapper<GPUOffloadPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUOffloadPass)
  llvm::StringRef getArgument() const final { return "gpu-offload"; }
  llvm::StringRef getDescription() const final { return "Offload matmul to GPU."; }
  void runOnOperation() override;
};

class SIMDVectorizationPass : public mlir::PassWrapper<SIMDVectorizationPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SIMDVectorizationPass)
  llvm::StringRef getArgument() const final { return "simd-vectorization"; }
  llvm::StringRef getDescription() const final { return "SIMD specific vectorization."; }
  void runOnOperation() override;
};

class KernelFusionPass : public mlir::PassWrapper<KernelFusionPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KernelFusionPass)
  llvm::StringRef getArgument() const final { return "kernel-fusion"; }
  llvm::StringRef getDescription() const final { return "Fuses matmul with bias_add/relu."; }
  void runOnOperation() override;
};

} // namespace adaptive_matmul

#endif // ADAPTIVE_MATMUL_PASSES_H
