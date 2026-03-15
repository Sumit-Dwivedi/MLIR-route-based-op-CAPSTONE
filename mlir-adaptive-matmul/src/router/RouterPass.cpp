//===- RouterPass.cpp - Adaptive routing with dynamic-shape dispatch ------===//
//
// Routes each linalg.matmul / linalg.batch_matmul to a specialised pipeline
// based on shape analysis and hardware features.
//
// For dynamic shapes: when M or N is unknown at compile time, we emit an
// scf.if that checks the runtime aspect ratio (max(M,N) > min(M,N) * 8)
// and branches to either the "skinny" or "square" strategy.  Downstream
// tiling passes honour the strategy attribute on whichever branch executes.
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
#include "mlir/IR/IRMapping.h"
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
  if (!shapedTy || !shapedTy.isDynamicDim(dimIdx))
    return nullptr;

  if (mlir::isa<mlir::MemRefType>(operand.getType()))
    return mlir::memref::DimOp::create(builder, loc, operand, dimIdx);
  if (mlir::isa<mlir::RankedTensorType>(operand.getType()))
    return mlir::tensor::DimOp::create(builder, loc, operand, dimIdx);
  return nullptr;
}

/// Return the runtime Value for a dimension: dynamic via DimOp, or static
/// via arith.constant.
static mlir::Value getDimValue(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value operand, unsigned dimIdx,
                               int64_t staticVal) {
  if (staticVal == mlir::ShapedType::kDynamic)
    return getDynamicDim(builder, loc, operand, dimIdx);
  return mlir::arith::ConstantIndexOp::create(builder, loc, staticVal);
}

/// Choose a static strategy string based on shape and hardware.
static llvm::StringRef chooseStaticStrategy(const MatrixInfo &info,
                                            const HardwareFeatures &hw) {
  if (info.aspect_ratio > 16.0) {
    return (info.M > info.N) ? "skinny" : "wide";
  }
  if (info.M < 64 && info.N < 64 && info.K < 64)
    return "small";
  if (hw.has_gpu && (info.M * info.N * info.K > 100000000))
    return "gpu";
  if (hw.has_avx512)
    return "simd";
  return "square";
}

/// Detect the data type category from operand element types.
/// Returns "int8" if inputs are i8/ui8 and accumulator is i32, else "f32".
static llvm::StringRef detectDataType(mlir::linalg::LinalgOp op) {
  if (op.getNumDpsInputs() < 2 || op.getNumDpsInits() < 1)
    return "f32";

  auto getElemType = [](mlir::Value v) -> mlir::Type {
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(v.getType()))
      return shaped.getElementType();
    return {};
  };

  mlir::Type inA = getElemType(op.getDpsInputOperand(0)->get());
  mlir::Type inB = getElemType(op.getDpsInputOperand(1)->get());
  mlir::Type out = getElemType(op.getDpsInits()[0]);

  if (!inA || !inB || !out)
    return "f32";

  bool inputsI8 = (inA.isInteger(8) && inB.isInteger(8));
  bool accumI32 = out.isInteger(32);

  if (inputsI8 && accumI32)
    return "int8";
  return "f32";
}

/// Determine the operand dim indices for M and N given a matmul variant.
/// Returns {operand_for_M, dimIdx_M, operand_for_N, dimIdx_N}.
static std::tuple<unsigned, unsigned, unsigned, unsigned>
getMatmulDimMapping(mlir::linalg::LinalgOp op) {
  if (llvm::isa<mlir::linalg::BatchMatmulOp>(op.getOperation()))
    return {0, 1, 1, 2}; // A[batch,M,K], B[batch,K,N]
  return {0, 0, 1, 1};   // A[M,K], B[K,N]
}

// ---------------------------------------------------------------------------
// Pass body.
// ---------------------------------------------------------------------------
void RouterPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto hardware = HardwareInfo::detectFromModule(module);
  auto *ctx = module.getContext();

  // Collect ops first — cannot modify IR during walk.
  struct WorkItem {
    mlir::linalg::LinalgOp op;
    MatrixInfo info;
    bool isDynamic;
  };
  llvm::SmallVector<WorkItem> work;

  module.walk([&](mlir::linalg::LinalgOp op) {
    auto info = MatrixAnalyzer::analyze(op);
    if (!info)
      return;
    bool dynM = (info->M == mlir::ShapedType::kDynamic);
    bool dynN = (info->N == mlir::ShapedType::kDynamic);
    work.push_back({op, *info, dynM || dynN});
  });

  mlir::IRRewriter rewriter(ctx);

  for (auto &item : work) {
    auto op = item.op;
    auto &info = item.info;

    // ---- Data type detection ----
    llvm::StringRef dtStr = detectDataType(op);
    auto dtAttr = mlir::StringAttr::get(ctx, dtStr);

    llvm::outs() << "--- Matrix Analysis ---\n";
    llvm::outs() << "Shape: " << info.M << "x" << info.N << "x" << info.K
                 << "\n";
    llvm::outs() << "Aspect Ratio: " << info.aspect_ratio << "\n";
    llvm::outs() << "Hardware: "
                 << (hardware.has_avx512
                         ? "AVX512"
                         : (hardware.has_avx2 ? "AVX2" : "Generic"))
                 << (hardware.has_gpu ? " + GPU" : "") << "\n";

    if (item.isDynamic) {
      // ---- Dynamic dispatch: emit scf.if with runtime aspect-ratio check ----
      mlir::Location loc = op.getLoc();
      rewriter.setInsertionPoint(op);

      // Determine which operand dims map to M and N.
      auto [opIdxM, dimIdxM, opIdxN, dimIdxN] = getMatmulDimMapping(op);
      mlir::Value A = op.getDpsInputOperand(opIdxM)->get();
      mlir::Value B = op.getDpsInputOperand(opIdxN)->get();

      // Get runtime M and N values (DimOp for dynamic, constant for static).
      mlir::Value dimM = getDimValue(rewriter, loc, A, dimIdxM, info.M);
      mlir::Value dimN = getDimValue(rewriter, loc, B, dimIdxN, info.N);

      // Cast to i64 for integer arithmetic (maxsi/minsi/muli/cmpi).
      auto i64Ty = rewriter.getI64Type();
      mlir::Value mI64 =
          mlir::arith::IndexCastOp::create(rewriter, loc, i64Ty, dimM);
      mlir::Value nI64 =
          mlir::arith::IndexCastOp::create(rewriter, loc, i64Ty, dimN);

      // is_skinny = max(M,N) > min(M,N) * 8
      mlir::Value maxDim =
          mlir::arith::MaxSIOp::create(rewriter, loc, mI64, nI64);
      mlir::Value minDim =
          mlir::arith::MinSIOp::create(rewriter, loc, mI64, nI64);
      mlir::Value c8 = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(8));
      mlir::Value threshold =
          mlir::arith::MulIOp::create(rewriter, loc, minDim, c8);
      mlir::Value isSkinny = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::sgt, maxDim, threshold);

      // Build scf.if: then → skinny, else → square.
      // Result types are inferred from the yield operands.
      auto ifOp = mlir::scf::IfOp::create(
          rewriter, loc, isSkinny,
          /*thenBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location l) {
            mlir::IRMapping mapping;
            auto *cloned = b.clone(*op, mapping);
            cloned->setAttr("optimization_strategy",
                            mlir::StringAttr::get(ctx, "skinny"));
            cloned->setAttr("adaptive.datatype", dtAttr);
            mlir::scf::YieldOp::create(b, l, cloned->getResults());
          },
          /*elseBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location l) {
            mlir::IRMapping mapping;
            auto *cloned = b.clone(*op, mapping);
            cloned->setAttr("optimization_strategy",
                            mlir::StringAttr::get(ctx, "square"));
            cloned->setAttr("adaptive.datatype", dtAttr);
            mlir::scf::YieldOp::create(b, l, cloned->getResults());
          });

      rewriter.replaceOp(op, ifOp.getResults());

      llvm::outs() << "Chosen Kernel: dynamic_dispatch (skinny | square)\n";
      llvm::outs() << "-----------------------\n";
      continue;
    }

    // ---- Static shape routing ----
    llvm::StringRef strategyStr = chooseStaticStrategy(info, hardware);
    auto strategy = mlir::StringAttr::get(ctx, strategyStr);
    op->setAttr("optimization_strategy", strategy);
    op->setAttr("adaptive.datatype", dtAttr);
    llvm::outs() << "Chosen Kernel: " << strategyStr << "\n";
    llvm::outs() << "Data Type: " << dtStr << "\n";

    double score = CostModel::computeScore(info, hardware);
    double peak = hardware.has_avx512 ? 1000.0 : 500.0;
    if (hardware.has_gpu)
      peak = 4000.0;
    double throughput = peak / (score * 0.1 + 1.0);
    llvm::outs() << "## Estimated Throughput: " << throughput << " GFLOPs\n";
    llvm::outs() << "-----------------------\n";
  }
}

std::unique_ptr<mlir::Pass> createRouterPass() {
  return std::make_unique<RouterPass>();
}

} // namespace adaptive_matmul
