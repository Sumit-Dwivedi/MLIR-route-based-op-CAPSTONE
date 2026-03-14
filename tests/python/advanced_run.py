import torch
import time
import json
import os
import numpy as np
from matrix_generator import MatrixGenerator
from mlir_runner import MLIRRunner
from benchmark_runner import BenchmarkRunner
from hardware_detector import HardwareDetector
from cost_model import RooflineModel

class AdvancedBenchmark:
    def __init__(self):
        self.hp = HardwareDetector.get_hardware_profile()
        self.rm = RooflineModel(self.hp)
        self.results = []

    def run_torch_benchmark(self, m, n, k, device="cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            return None
            
        A = torch.randn(m, k).to(device)
        B = torch.randn(k, n).to(device)
        
        # Warmup
        for _ in range(5):
            torch.matmul(A, B)
        
        start = time.time()
        for _ in range(20):
            torch.matmul(A, B)
        end = time.time()
        
        return (end - start) / 20.0

    def run_all(self):
        gen = MatrixGenerator(output_dir="tests/python/tmp")
        runner = MLIRRunner(opt_path="mlir-adaptive-matmul/build/adaptive-opt")
        
        # Research-grade shapes
        test_cases = [
            ("Attention", 4096, 64, 64),
            ("Attention_Wide", 4096, 4096, 64),
            ("LLM_Inference", 16384, 32, 8192),
            ("Tokenization", 8192, 128, 128),
            ("Square_Large", 2048, 2048, 2048)
        ]
        
        for name, m, n, k in test_cases:
            print(f"\nBenchmarking {name} ({m}x{k} * {k}x{n})")
            
            torch_cpu = self.run_torch_benchmark(m, n, k, "cpu")
            torch_gpu = self.run_torch_benchmark(m, n, k, "cuda")
            
            mlir_path = os.path.join("tests/python/tmp", f"adv_{m}_{n}_{k}.mlir")
            gen.generate_mlir(m, n, k, "", "", "", mlir_path)
            mlir_res = runner.run_pipeline(mlir_path)
            
            # Record
            res = {
                "name": name,
                "shape": (m, n, k),
                "torch_cpu": torch_cpu,
                "torch_gpu": torch_gpu,
                "mlir_opt": mlir_res["execution_time"],
                "strategy": mlir_res["strategy"],
                "stdout": mlir_res["stdout"]
            }
            self.results.append(res)
            print(f"Strategy: {mlir_res['strategy']}")
            print(f"MLIR Time: {mlir_res['execution_time']:.6f}s")
            print(f"Torch CPU: {torch_cpu:.6f}s")

        with open("results/advanced_benchmarks.json", 'w') as f:
            json.dump(self.results, f, indent=4)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    bench = AdvancedBenchmark()
    bench.run_all()
