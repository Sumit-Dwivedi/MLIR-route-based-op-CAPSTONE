class GraphSimulator:
    @staticmethod
    def generate_fused_graph_mlir(m, n, k, mlir_path):
        mlir_template = f"""
module {{
  func.func @fused_graph(%A: memref<{m}x{k}xf32>, %B: memref<{k}x{n}xf32>, %Bias: memref<{m}x{n}xf32>, %Out: memref<{m}x{n}xf32>) {{
    %tmp = memref.alloc() : memref<{m}x{n}xf32>
    
    // Operation 1: Matmul
    linalg.matmul ins(%A, %B : memref<{m}x{k}xf32>, memref<{k}x{n}xf32>)
                 outs(%tmp : memref<{m}x{n}xf32>)
                 
    // Operation 2: Bias Add (Simplified as another linalg op)
    linalg.add ins(%tmp, %Bias : memref<{m}x{n}xf32>, memref<{m}x{n}xf32>)
               outs(%Out : memref<{m}x{n}xf32>)
               
    memref.dealloc %tmp : memref<{m}x{n}xf32>
    return
  }}
  
  func.func @main() {{
    %A = memref.alloc() : memref<{m}x{k}xf32>
    %B = memref.alloc() : memref<{k}x{n}xf32>
    %Bias = memref.alloc() : memref<{m}x{n}xf32>
    %Out = memref.alloc() : memref<{m}x{n}xf32>
    
    call @fused_graph(%A, %B, %Bias, %Out) : (memref<{m}x{k}xf32>, memref<{k}x{n}xf32>, memref<{m}x{n}xf32>, memref<{m}x{n}xf32>) -> ()
    
    memref.dealloc %A : memref<{m}x{k}xf32>
    memref.dealloc %B : memref<{k}x{n}xf32>
    memref.dealloc %Bias : memref<{m}x{n}xf32>
    memref.dealloc %Out : memref<{m}x{n}xf32>
    return
  }}
}}
"""
        with open(mlir_path, 'w') as f:
            f.write(mlir_template)

if __name__ == "__main__":
    GraphSimulator.generate_fused_graph_mlir(512, 512, 512, "tests/python/tmp/graph_test.mlir")
    print("Graph MLIR generated")
