func.func @skinny_matmul(%arg0: memref<4096x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<4096x32xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<4096x32xf32>, memref<32x32xf32>)
               outs(%arg2 : memref<4096x32xf32>)
  return
}
