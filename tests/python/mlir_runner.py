import subprocess
import time
import os
import re

class MLIRRunner:
    def __init__(self, opt_path="mlir-adaptive-matmul/build/adaptive-opt", 
                 runner_path="/home/sumit/mlir-build/bin/mlir-runner"):
        self.opt_path = opt_path
        self.runner_path = runner_path
        self.shared_libs = [
            "/home/sumit/mlir-build/lib/libmlir_c_runner_utils.so",
            "/home/sumit/mlir-build/lib/libmlir_runner_utils.so"
        ]

    def run_full_pipeline(self, mlir_path, iterations=50, verbose=False):
        tmp_prefix = mlir_path.replace(".mlir", "")
        opt_mlir = f"{tmp_prefix}_opt.mlir"
        
        # 1. MLIR Optimization Phase
        start_opt = time.time()
        opt_cmd = [
            self.opt_path,
            "--adaptive-router",
            "--skinny-tiling", "--wide-tiling", "--square-blocking",
            "--skinny-vectorization", "--small-matrix-vector",
            "--simd-vectorization", "--gpu-offload",
            mlir_path, "-o", opt_mlir
        ]
        res_opt = subprocess.run(opt_cmd, capture_output=True, text=True)
        opt_time = time.time() - start_opt
        
        if res_opt.returncode != 0:
            return {"success": False, "error": res_opt.stderr}
            
        strategy = self._extract_strategy(res_opt.stdout)
        
        # 2. Lowering to LLVM Dialect for Execution
        lower_cmd = [
            "/home/sumit/mlir-build/bin/mlir-opt",
            opt_mlir,
            "--convert-linalg-to-loops",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-vector-to-llvm",
            "--convert-func-to-llvm",
            "--finalize-memref-to-llvm",
            "--reconcile-unrealized-casts"
        ]
        res_lower = subprocess.run(lower_cmd, capture_output=True, text=True)
        if res_lower.returncode != 0:
            return {"success": False, "error": res_lower.stderr}
            
        lower_mlir = f"{tmp_prefix}_lower.mlir"
        with open(lower_mlir, 'w') as f:
            f.write(res_lower.stdout)

        # 3. Execution Benchmark using mlir-cpu-runner
        exec_times = []
        # In a real research environment, we'd wrap the main with a loop or call it multiple times
        # For this framework, we measure total time and divide.
        
        # Measure peak GFLOPS of CPU (Estimate for i5-10300H: ~300 GFLOPS AVX2)
        peak_gflops = 300.0 
        
        total_exec_start = time.time()
        for _ in range(iterations):
            run_cmd = [
                self.runner_path,
                "-e", "main",
                "-entry-point-result=void",
            ]
            for lib in self.shared_libs:
                run_cmd += ["--shared-libs", lib]
            run_cmd += [lower_mlir]
            
            subprocess.run(run_cmd, capture_output=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        total_exec_time = time.time() - total_exec_start
        avg_exec_time = total_exec_time / iterations

        return {
            "success": True,
            "strategy": strategy,
            "compile_time": opt_time,
            "execution_time": avg_exec_time,
            "peak_gflops": peak_gflops,
            "stdout": res_opt.stdout
        }

    def _extract_strategy(self, output):
        match = re.search(r"Chosen Kernel: (\w+)", output)
        if match:
            return match.group(1)
        return "Unknown"
