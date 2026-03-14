#include "AdaptiveMatmul/Passes.h"
#include "llvm/Support/raw_ostream.h"

namespace adaptive_matmul {

void SkinnyMemoryPass::runOnOperation() {
  llvm::outs() << "Applying Skinny Memory Optimization...\n";
}

} // namespace adaptive_matmul
