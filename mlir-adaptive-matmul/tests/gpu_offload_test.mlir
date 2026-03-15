// RUN: ADAPTIVE_GPU_TARGET=vulkan adaptive-opt %s --adaptive-router --gpu-offload 2>&1 | FileCheck %s

// Test: Massive matmul (4096x4096x4096, 6.9e10 ops > 1e8 threshold)
// should be routed to "gpu" and offloaded into scf.forall with GPU block mapping.

func.func @matmul_massive(%A: tensor<4096x4096xf32>, %B: tensor<4096x4096xf32>,
                           %C: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<4096x4096xf32>, tensor<4096x4096xf32>)
                      outs(%C : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %0 : tensor<4096x4096xf32>
}

// CHECK-LABEL: func.func @matmul_massive
// CHECK: scf.forall
// CHECK:   linalg.matmul
// CHECK-SAME: optimization_strategy = "gpu"
// CHECK: scf.forall.in_parallel
// CHECK: mapping = [#gpu.block<y>, #gpu.block<x>]

// -----

// Test: Small matmul should NOT be routed to GPU even with GPU available.

func.func @matmul_small(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>,
                         %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>)
                      outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @matmul_small
// CHECK: linalg.matmul
// CHECK-SAME: optimization_strategy = "square"
// CHECK-NOT: scf.forall
