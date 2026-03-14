#include "AdaptiveMatmul/Passes.h"
#include "llvm/Support/raw_ostream.h"
namespace adaptive_matmul { void SquareBlockingPass::runOnOperation() { llvm::outs() << "Applied Square Blocking\n"; } }