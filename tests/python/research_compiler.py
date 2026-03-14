import os
import argparse
import time
import numpy as np
import pandas as pd
from matrix_generator import MatrixGenerator
from mlir_runner import MLIRRunner
from benchmark_runner import BenchmarkRunner
from hardware_detector import HardwareDetector
from cost_model import HybridCostModel
from learned_cost_model import LearnedCostModel
from autotuner import DynamicAutotuner
from workload_classifier import WorkloadClassifier
from visualizer import Visualizer

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    rich_available = True
except ImportError:
    rich_available = False

def main():
    parser = argparse.ArgumentParser(description="Adaptive MLIR Research Compiler")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    args = parser.parse_args()

    os.makedirs("results/plots", exist_ok=True)
    console = Console() if rich_available else None

    # 1. Hardware Detection
    hw = HardwareDetector.get_hardware_profile()
    peak_gflops = 300.0 # Placeholder for AVX2 peak
    if rich_available:
        console.print(Panel(f"[bold green]Adaptive MLIR Research Compiler[/bold green]\nHardware: {hw['cpu_brand']}"))

    # 2. Initialize Research Modules
    gen = MatrixGenerator()
    runner = MLIRRunner()
    learned_model = LearnedCostModel()
    autotuner = DynamicAutotuner()
    classifier = WorkloadClassifier()

    test_cases = gen.get_test_cases()
    
    results_data = []

    for name, m, n, k in test_cases:
        workload_type = classifier.classify(m, n, k)
        if rich_available:
            console.print(f"\n[bold blue]>>> Processing: {name} ({workload_type})[/bold blue]")

        # 3. Kernel Selection (Hybrid: Analytical + Learned)
        candidates = ["skinny", "square", "wide", "small", "simd"]
        best_kernel = learned_model.predict_best_kernel(m, n, k, candidates, hw)
        
        # 4. Autotuning parameters for selected kernel
        cfg = autotuner.tune_configuration(m, n, k, best_kernel)

        # 5. MLIR Execution (Proper pipeline)
        mlir_path = os.path.join("tests/python/tmp", f"research_{m}_{n}_{k}.mlir")
        gen.generate_mlir(m, n, k, mlir_path)
        
        res = runner.run_full_pipeline(mlir_path, iterations=args.iterations)
        
        if res["success"]:
            # 6. Performance Diagnostics
            flops = 2.0 * m * n * k
            measured_gflops = (flops / res["execution_time"]) / 1e9 if res["execution_time"] > 0 else 0
            efficiency = (measured_gflops / peak_gflops) * 100
            
            # 7. Feedback Loop (Learned Cost Model update)
            learned_model.record_performance(m, n, k, res["strategy"], hw["cpu_brand"], res["execution_time"])
            
            # Record results
            results_data.append({
                "name": name,
                "workload": workload_type,
                "strategy": res["strategy"],
                "gflops": measured_gflops,
                "efficiency": efficiency,
                "exec_time": res["execution_time"]
            })

            if rich_available:
                table = Table(show_header=False, box=None)
                table.add_row("Measured Performance", f"[bold cyan]{measured_gflops:.2f} GFLOPs[/bold cyan]")
                table.add_row("Hardware Efficiency", f"{efficiency:.2f}%")
                table.add_row("Execution Time (avg)", f"{res['execution_time']:.6f} s")
                table.add_row("Selected Strategy", res["strategy"])
                console.print(table)
        else:
            err_msg = res.get('error', 'Unknown Error')
            if rich_available:
                console.print(f"[bold red]Failed to run {name}:[/bold red] {err_msg}")
            else:
                print(f"Failed to run {name}: {err_msg}")

    # 8. Final Report
    df = pd.DataFrame(results_data)
    df.to_json("results/research_results.json", indent=4)
    
    if rich_available:
        console.print(Panel("[bold green]Research Framework Run Complete! Results in results/[/bold green]"))

if __name__ == "__main__":
    main()
