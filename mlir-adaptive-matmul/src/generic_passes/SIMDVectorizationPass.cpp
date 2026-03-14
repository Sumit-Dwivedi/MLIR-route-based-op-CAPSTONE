#include "AdaptiveMatmul/Passes.h"
#include "llvm/Support/raw_ostream.h"
namespace adaptive_matmul { void SIMDVectorizationPass::runOnOperation() { llvm::outs() << "Applied SIMD Vectorization\n"; } }