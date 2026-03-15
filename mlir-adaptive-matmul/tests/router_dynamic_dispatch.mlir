// RUN: adaptive-opt %s --adaptive-router 2>&1 | FileCheck %s

// Test: dynamic-shape matmul should produce scf.if with runtime aspect-ratio
// dispatch.  The then-branch gets "skinny", the else-branch gets "square".

func.func @matmul_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                           %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                      outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK: dynamic_dispatch
// CHECK: scf.if
// CHECK:   linalg.matmul
// CHECK-SAME: optimization_strategy = "skinny"
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   linalg.matmul
// CHECK-SAME: optimization_strategy = "square"
// CHECK:   scf.yield

// -----

// Test: static-shape matmul should NOT produce scf.if.

func.func @matmul_static(%A: tensor<256x256xf32>, %B: tensor<256x256xf32>,
                          %C: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>)
                      outs(%C : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %0 : tensor<256x256xf32>
}

// CHECK-NOT: scf.if
// CHECK: linalg.matmul
// CHECK-SAME: optimization_strategy

// -----

// Test: partially dynamic (only M dynamic) should still produce scf.if.

func.func @matmul_partial_dynamic(%A: tensor<?x128xf32>, %B: tensor<128x64xf32>,
                                   %C: tensor<?x64xf32>) -> tensor<?x64xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x128xf32>, tensor<128x64xf32>)
                      outs(%C : tensor<?x64xf32>) -> tensor<?x64xf32>
  return %0 : tensor<?x64xf32>
}

// CHECK: scf.if
// CHECK:   linalg.matmul
// CHECK-SAME: optimization_strategy = "skinny"
// CHECK:   linalg.matmul
// CHECK-SAME: optimization_strategy = "square"
