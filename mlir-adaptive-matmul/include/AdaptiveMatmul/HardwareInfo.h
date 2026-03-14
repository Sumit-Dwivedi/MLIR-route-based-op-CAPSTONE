#ifndef ADAPTIVE_MATMUL_HARDWARE_INFO_H
#define ADAPTIVE_MATMUL_HARDWARE_INFO_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace adaptive_matmul {

struct HardwareFeatures {
  bool has_avx2;
  bool has_avx512;
  bool has_gpu;
  int sim_width;      // SIMD register width in bits
  int cache_size_kb;  // L3 cache size in KB

  // Extended fields for IREE/XLA dispatch decisions.
  int l1_cache_kb;
  int num_cores;
  std::string gpu_arch; // e.g. "sm_80", "gfx90a", "" if no GPU
};

class HardwareInfo {
public:
  /// Host-level detection (cpuid probing, /proc, env vars).
  static HardwareFeatures detect();

  /// MLIR-level detection: inspects DataLayout and target attributes on
  /// the module to override or refine the host-detected defaults.
  /// This is the preferred entry point when a ModuleOp is available.
  static HardwareFeatures detectFromModule(mlir::ModuleOp module);
};

} // namespace adaptive_matmul

#endif // ADAPTIVE_MATMUL_HARDWARE_INFO_H
