#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "AdaptiveMatmul/Router.h"
#include "AdaptiveMatmul/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // KernelFusion runs FIRST: fuses elementwise chains (bias+relu → one op).
  // Contractions (matmul) remain as named ops — epilogue fusion happens
  // via tile-and-fuse in the tiling passes.
  mlir::PassRegistration<adaptive_matmul::KernelFusionPass>();
  mlir::PassRegistration<adaptive_matmul::RouterPass>();
  mlir::PassRegistration<adaptive_matmul::SkinnyTilingPass>();
  mlir::PassRegistration<adaptive_matmul::SkinnyVectorizationPass>();
  mlir::PassRegistration<adaptive_matmul::SkinnyMemoryPass>();
  mlir::PassRegistration<adaptive_matmul::GenericTilingPass>();
  mlir::PassRegistration<adaptive_matmul::SmallMatrixVectorPass>();
  mlir::PassRegistration<adaptive_matmul::WideTilingPass>();
  mlir::PassRegistration<adaptive_matmul::SquareBlockingPass>();
  mlir::PassRegistration<adaptive_matmul::SIMDVectorizationPass>();
  mlir::PassRegistration<adaptive_matmul::GPUOffloadPass>();
  mlir::PassRegistration<adaptive_matmul::LowerToLLVMPass>();
  mlir::PassRegistration<adaptive_matmul::JITRunnerPass>();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Adaptive Matmul Optimizer\n", registry));
}
