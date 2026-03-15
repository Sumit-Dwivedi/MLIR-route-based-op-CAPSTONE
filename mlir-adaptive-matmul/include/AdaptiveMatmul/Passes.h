#ifndef ADAPTIVE_MATMUL_PASSES_H
#define ADAPTIVE_MATMUL_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect,
                    mlir::scf::SCFDialect>();
  }
  void runOnOperation() override;
};

class SIMDVectorizationPass : public mlir::PassWrapper<SIMDVectorizationPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SIMDVectorizationPass)
  llvm::StringRef getArgument() const final { return "simd-vectorization"; }
  llvm::StringRef getDescription() const final { return "SIMD specific vectorization."; }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect,
                    mlir::vector::VectorDialect,
                    mlir::ub::UBDialect>();
  }
  void runOnOperation() override;
};

class KernelFusionPass : public mlir::PassWrapper<KernelFusionPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KernelFusionPass)
  llvm::StringRef getArgument() const final { return "kernel-fusion"; }
  llvm::StringRef getDescription() const final { return "Fuses matmul with bias_add/relu."; }
  void runOnOperation() override;
};

class LowerToLLVMPass : public mlir::PassWrapper<LowerToLLVMPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLLVMPass)
  llvm::StringRef getArgument() const final { return "lower-to-llvm"; }
  llvm::StringRef getDescription() const final { return "Lower vectorized IR through the full pipeline to LLVM dialect."; }
  void runOnOperation() override;
};

class JITRunnerPass : public mlir::PassWrapper<JITRunnerPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(JITRunnerPass)
  llvm::StringRef getArgument() const final { return "jit-run"; }
  llvm::StringRef getDescription() const final { return "JIT-compile and benchmark the LLVM dialect module."; }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};

} // namespace adaptive_matmul

#endif // ADAPTIVE_MATMUL_PASSES_H
