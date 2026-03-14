#include "AdaptiveMatmul/Passes.h"
#include "llvm/Support/raw_ostream.h"
namespace adaptive_matmul { void WideTilingPass::runOnOperation() { llvm::outs() << "Applied Wide Tiling\n"; } }