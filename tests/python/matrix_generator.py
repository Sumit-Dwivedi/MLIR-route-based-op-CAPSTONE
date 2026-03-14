import numpy as np
import os

class MatrixGenerator:
    def __init__(self, output_dir="tests/python/tmp"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_random_matrix(self, rows, cols):
        # Use random initialization to prevent constant folding
        return np.random.rand(rows, cols).astype(np.float32)

    def generate_mlir(self, m, n, k, mlir_path):
        # Template using memref for mlir-cpu-runner compatibility
        mlir_template = f"""
module {{
  func.func @matmul_test(%A: memref<{m}x{k}xf32>, %B: memref<{k}x{n}xf32>, %C: memref<{m}x{n}xf32>) {{
    linalg.matmul ins(%A, %B : memref<{m}x{k}xf32>, memref<{k}x{n}xf32>)
                 outs(%C : memref<{m}x{n}xf32>)
    return
  }}

  func.func @main() {{
    %A = memref.alloc() : memref<{m}x{k}xf32>
    %B = memref.alloc() : memref<{k}x{n}xf32>
    %C = memref.alloc() : memref<{m}x{n}xf32>
    
    // In a real scenario we'd initialize these with random data
    // For timing purposes, we just run the matmul
    
    call @matmul_test(%A, %B, %C) : (memref<{m}x{k}xf32>, memref<{k}x{n}xf32>, memref<{m}x{n}xf32>) -> ()
    
    // Checksum/Reduction to prevent DCE
    %sum = memref.alloc() : memref<f32>
    // Simplified reduction placeholder
    
    memref.dealloc %A : memref<{m}x{k}xf32>
    memref.dealloc %B : memref<{k}x{n}xf32>
    memref.dealloc %C : memref<{m}x{n}xf32>
    return
  }}
}}
"""
        with open(mlir_path, 'w') as f:
            f.write(mlir_template)

    def get_test_cases(self):
        return [
            ("Square_Small", 128, 128, 128),
            ("Square_Medium", 512, 512, 512),
            ("Square_Large", 1024, 1024, 1024),
            ("Skinny", 4096, 32, 32),
            ("Wide", 32, 4096, 4096),
            ("Attention", 4096, 64, 64),
            ("LLM_FF", 8192, 128, 128)
        ]
