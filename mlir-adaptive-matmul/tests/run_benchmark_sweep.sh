#!/bin/bash
# Benchmark sweep for the Adaptive Matmul Framework.
# Generates MLIR, runs the full pipeline, collects correctness + performance.
set -euo pipefail

OPT="$(dirname "$0")/../build/adaptive-opt"
PIPELINE="--kernel-fusion --adaptive-router --square-blocking --simd-vectorization --lower-to-llvm --jit-run"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

gen_f32() {
  local M=$1 N=$2 K=$3 FILE=$4
  cat > "$FILE" <<EOF
func.func @matmul(%A: tensor<${M}x${K}xf32>, %B: tensor<${K}x${N}xf32>,
                   %C: tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<${M}x${K}xf32>, tensor<${K}x${N}xf32>)
                      outs(%C : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  return %0 : tensor<${M}x${N}xf32>
}
EOF
}

run_bench() {
  local LABEL="$1" M=$2 N=$3 K=$4 RUNS=$5
  local FILE="$TMPDIR/${LABEL}.mlir"
  gen_f32 "$M" "$N" "$K" "$FILE"
  echo "=== BENCH: $LABEL (${M}x${N}x${K}, ${RUNS} runs) ==="
  ADAPTIVE_M=$M ADAPTIVE_N=$N ADAPTIVE_K=$K ADAPTIVE_RUNS=$RUNS \
    "$OPT" "$FILE" $PIPELINE 2>&1 || echo "  STATUS: ERROR"
  echo "=== END $LABEL ==="
  echo ""
}

echo "########## BENCHMARK SWEEP ##########"
echo ""
run_bench "small_32"     32   32   32   20
run_bench "skinny_4096"  4096 32   32   10
run_bench "square_256"   256  256  256  20
run_bench "square_1024"  1024 1024 1024 5
echo "########## SWEEP COMPLETE ##########"
