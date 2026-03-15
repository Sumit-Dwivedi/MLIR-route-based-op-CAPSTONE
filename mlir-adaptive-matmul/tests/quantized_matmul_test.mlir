// RUN: adaptive-opt %s --adaptive-router 2>&1 | FileCheck %s

// Test 1: INT8 matmul (i8 inputs, i32 accumulator) should be tagged "int8".

func.func @matmul_int8(%A: tensor<128x64xi8>, %B: tensor<64x128xi8>,
                        %C: tensor<128x128xi32>) -> tensor<128x128xi32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x64xi8>, tensor<64x128xi8>)
                      outs(%C : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %0 : tensor<128x128xi32>
}

// CHECK-LABEL: func.func @matmul_int8
// CHECK: linalg.matmul
// CHECK-SAME: adaptive.datatype = "int8"
// CHECK-SAME: optimization_strategy = "square"

// -----

// Test 2: Standard f32 matmul should be tagged "f32".

func.func @matmul_f32(%A: tensor<128x64xf32>, %B: tensor<64x128xf32>,
                       %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x64xf32>, tensor<64x128xf32>)
                      outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @matmul_f32
// CHECK: linalg.matmul
// CHECK-SAME: adaptive.datatype = "f32"

// -----

// Test 3: Dynamic INT8 matmul should produce scf.if with "int8" on both branches.

func.func @matmul_int8_dynamic(%A: tensor<?x?xi8>, %B: tensor<?x?xi8>,
                                %C: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xi8>, tensor<?x?xi8>)
                      outs(%C : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// CHECK-LABEL: func.func @matmul_int8_dynamic
// CHECK: scf.if
// CHECK:   linalg.matmul
// CHECK-SAME: adaptive.datatype = "int8"
// CHECK-SAME: optimization_strategy = "skinny"
// CHECK: } else {
// CHECK:   linalg.matmul
// CHECK-SAME: adaptive.datatype = "int8"
// CHECK-SAME: optimization_strategy = "square"
