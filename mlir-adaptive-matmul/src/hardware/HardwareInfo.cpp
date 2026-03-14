#include "AdaptiveMatmul/HardwareInfo.h"

namespace adaptive_matmul {

HardwareFeatures HardwareInfo::detect() {
  HardwareFeatures features;
  // Basic mock detection
  features.has_avx2 = true;
  features.has_avx512 = false;
  features.has_gpu = false;
  features.sim_width = 256; // 256-bit
  features.cache_size_kb = 32768; // 32MB L3
  return features;
}

} // namespace adaptive_matmul
