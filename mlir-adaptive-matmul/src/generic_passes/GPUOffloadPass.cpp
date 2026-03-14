#include "AdaptiveMatmul/Passes.h"
#include "llvm/Support/raw_ostream.h"
namespace adaptive_matmul { void GPUOffloadPass::runOnOperation() { llvm::outs() << "Applied GPU Offload\n"; } }