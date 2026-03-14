#ifndef ADAPTIVE_MATMUL_COST_MODEL_H
#define ADAPTIVE_MATMUL_COST_MODEL_H

#include "AdaptiveMatmul/MatrixAnalyzer.h"
#include "AdaptiveMatmul/HardwareInfo.h"

namespace adaptive_matmul {

class CostModel {
public:
  static double computeScore(const MatrixInfo& info, const HardwareFeatures& hardware);
};

} // namespace adaptive_matmul

#endif // ADAPTIVE_MATMUL_COST_MODEL_H
