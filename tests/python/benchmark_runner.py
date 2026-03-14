import json
import time
import os

class BenchmarkRunner:
    def __init__(self, results_path="results/benchmark_results.json"):
        self.results_path = results_path
        self.results = []

    def record_result(self, mtype, shape, strategy, mlir_time, numpy_time, success):
        result = {
            "matrix_type": mtype,
            "shape": f"{shape[0]}x{shape[1]} * {shape[1]}x{shape[2]}",
            "m": shape[0], "n": shape[2], "k": shape[1],
            "chosen_strategy": strategy,
            "mlir_execution_time": mlir_time,
            "numpy_execution_time": numpy_time,
            "speedup": numpy_time / mlir_time if mlir_time > 0 else 0,
            "success": success,
            "timestamp": time.time()
        }
        self.results.append(result)

    def save_results(self):
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f, indent=4)

if __name__ == "__main__":
    runner = BenchmarkRunner()
    print("Benchmark runner initialized")
