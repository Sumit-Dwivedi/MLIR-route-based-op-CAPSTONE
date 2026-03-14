//===- HardwareInfo.cpp - Hardware feature detection ----------------------===//
//
// Two detection paths:
//
//   1. detect()            — probes the host CPU via cpuid intrinsics (x86)
//                            and environment variables (GPU_TARGET, etc.).
//
//   2. detectFromModule()  — reads MLIR DataLayout attributes and target
//                            attributes attached to the ModuleOp (e.g. by
//                            IREE's HAL target pipeline or XLA's auto-tuner).
//                            Falls back to detect() for anything the IR
//                            doesn't specify.
//
//===----------------------------------------------------------------------===//

#include "AdaptiveMatmul/HardwareInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ADAPTIVE_HAS_X86
#ifdef __GNUC__
#include <cpuid.h>
#endif
#endif

namespace adaptive_matmul {

// ---------------------------------------------------------------------------
// Host-level detection.
// ---------------------------------------------------------------------------
HardwareFeatures HardwareInfo::detect() {
  HardwareFeatures f{};
  f.has_avx2    = false;
  f.has_avx512  = false;
  f.has_gpu     = false;
  f.sim_width   = 128;  // SSE baseline
  f.cache_size_kb = 32768;
  f.l1_cache_kb   = 32;
  f.num_cores     = 1;
  f.gpu_arch      = "";

#ifdef ADAPTIVE_HAS_X86
  unsigned int eax, ebx, ecx, edx;

  // Leaf 7, sub-leaf 0: structured extended feature flags.
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    // AVX2: EBX bit 5
    f.has_avx2 = (ebx >> 5) & 1;
    // AVX-512F: EBX bit 16
    f.has_avx512 = (ebx >> 16) & 1;
  }

  if (f.has_avx512) {
    f.sim_width = 512;
  } else if (f.has_avx2) {
    f.sim_width = 256;
  }

  // Leaf 4: deterministic cache parameters (EAX[31:26]+1 = cores sharing).
  // We just grab the L3 size from sub-leaf 3 if available.
  if (__get_cpuid_count(4, 3, &eax, &ebx, &ecx, &edx)) {
    unsigned ways       = ((ebx >> 22) & 0x3FF) + 1;
    unsigned partitions = ((ebx >> 12) & 0x3FF) + 1;
    unsigned line_size  = (ebx & 0xFFF) + 1;
    unsigned sets       = ecx + 1;
    unsigned long long cache_bytes =
        (unsigned long long)ways * partitions * line_size * sets;
    f.cache_size_kb = static_cast<int>(cache_bytes / 1024);
  }
#endif

  // GPU detection via environment (IREE / XLA convention).
  const char *gpuTarget = std::getenv("ADAPTIVE_GPU_TARGET");
  if (gpuTarget && std::strlen(gpuTarget) > 0) {
    f.has_gpu  = true;
    f.gpu_arch = gpuTarget;
  }

  return f;
}

// ---------------------------------------------------------------------------
// MLIR module-level detection.
// ---------------------------------------------------------------------------
HardwareFeatures HardwareInfo::detectFromModule(mlir::ModuleOp module) {
  // Start with host defaults.
  HardwareFeatures f = detect();

  if (!module)
    return f;

  // --- Check for a target attribute (IREE convention) ---
  // IREE attaches  #hal.device.target<"cpu", ...>  or  #hal.device.target<"cuda", ...>
  // as a module-level attribute.  We look for a simple string marker.
  if (auto targetAttr =
          module->getAttrOfType<mlir::StringAttr>("adaptive.target")) {
    llvm::StringRef target = targetAttr.getValue();
    if (target.contains("gpu") || target.contains("cuda") ||
        target.contains("rocm") || target.contains("vulkan")) {
      f.has_gpu = true;
      if (f.gpu_arch.empty())
        f.gpu_arch = target.str();
    }
    if (target.contains("avx512"))
      f.has_avx512 = true;
    if (target.contains("avx2"))
      f.has_avx2 = true;
  }

  // --- Check DataLayout for pointer/index widths ---
  // A 32-bit data layout hints at an embedded target with smaller caches.
  if (auto dlAttr =
          module->getAttrOfType<mlir::StringAttr>("dlti.dl_spec")) {
    llvm::StringRef dl = dlAttr.getValue();
    if (dl.contains("p:32"))
      f.cache_size_kb = std::min(f.cache_size_kb, 4096);
  }

  // --- SIMD width override attribute ---
  if (auto simdAttr =
          module->getAttrOfType<mlir::IntegerAttr>("adaptive.simd_width")) {
    f.sim_width = static_cast<int>(simdAttr.getInt());
    f.has_avx512 = (f.sim_width >= 512);
    f.has_avx2   = (f.sim_width >= 256);
  }

  return f;
}

} // namespace adaptive_matmul
