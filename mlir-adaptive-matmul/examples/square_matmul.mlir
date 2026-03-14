func.func @square_matmul(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024x1024xf32>)
               outs(%arg2 : memref<1024x1024xf32>)
  return
}
