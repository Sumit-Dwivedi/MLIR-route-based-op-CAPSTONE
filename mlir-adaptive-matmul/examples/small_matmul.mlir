func.func @small_matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
               outs(%arg2 : memref<64x64xf32>)
  return
}
