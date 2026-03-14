#ifndef ADAPTIVE_MATMUL_HARDWARE_INFO_H
#define ADAPTIVE_MATMUL_HARDWARE_INFO_H

#include <string>

namespace adaptive_matmul {

struct HardwareFeatures {
  bool has_avx2;
  bool has_avx512;
  bool has_gpu;
  int sim_width;
  int cache_size_kb;
};

class HardwareInfo {
public:
  static HardwareFeatures detect();
};

} // namespace adaptive_matmul

#endif // ADAPTIVE_MATMUL_HARDWARE_INFO_H
