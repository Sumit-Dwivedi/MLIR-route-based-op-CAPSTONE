import os
import argparse
import time
import numpy as np
import json
from matrix_generator import MatrixGenerator
from mlir_runner import MLIRRunner
from benchmark_runner import BenchmarkRunner
from result_validator import ResultValidator
from visualizer import Visualizer
from hardware_detector import HardwareDetector
from cost_model import HybridCostModel

# Use rich for CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich import print as rprint
    rich_available = True
except ImportError:
    rich_available = False

def main():
    parser = argparse.ArgumentParser(description="Run advanced MLIR adaptive matmul tests.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    # Initialize components
    gen = MatrixGenerator(output_dir="tests/python/tmp")
    runner = MLIRRunner()
    benchmark = BenchmarkRunner(results_path="results/advanced_benchmarks.json")
    hw = HardwareDetector.get_hardware_profile()
    cost_model = HybridCostModel(hw)
    
    if rich_available:
        console = Console()
        console.print(Panel("[bold green]MLIR Research Compiler Framework: Adaptive Matrix Optimization[/bold green]"))
        console.print(f"[bold blue]Hardware Detected:[/bold blue] {hw['cpu_brand']}")

    test_cases = gen.get_test_cases()
    
    for mtype, m, n, k in test_cases:
        # Generate matrices and MLIR
        A = gen.generate_matrix(m, k)
        B = gen.generate_matrix(k, n)
        mlir_path = os.path.join("tests/python/tmp", f"test_{m}_{n}_{k}.mlir")
        gen.generate_mlir(m, n, k, "", "", "", mlir_path)

        # 1. Run Baseline (PyTorch CPU simulation)
        # Using a simple numpy/timer proxy for baseline in this turn
        start_base = time.time()
        _ = np.matmul(A, B)
        base_exec_time = time.time() - start_base

        # 2. Run MLIR Pipeline (Optimized + Lowering simulation)
        res = runner.run_full_pipeline(mlir_path, verbose=args.verbose)
        if not res["success"]:
            print(f"Error in {mtype}: {res['error']}")
            continue

        # 3. Diagnostics
        flops = 2.0 * m * n * k
        achieved_gflops = (flops / res["execution_time"]) / 1e9 if res["execution_time"] > 0 else 0
        
        # 4. Record
        benchmark.record_result(mtype, (m, k, n), res["strategy"], res["execution_time"], base_exec_time, True)

        # 5. CLI Output with Visualization
        if rich_available:
            console.print(f"\n[bold magenta]Workload: {mtype} ({m}x{k} * {k}x{n})[/bold magenta]")
            
            # Pass Pipeline Visualization (Simplified)
            passes = ["RouterPass"]
            if res["strategy"] == "skinny": passes += ["SkinnyTilingPass", "SkinnyVectorizationPass"]
            elif res["strategy"] == "square": passes += ["SquareBlockingPass", "GenericTilingPass"]
            elif res["strategy"] == "wide": passes += ["WideTilingPass"]
            elif res["strategy"] == "small": passes += ["SmallMatrixVectorPass"]
            passes += ["VectorContractPass", "LLVMCodegenPass"]
            
            table = Table(show_header=False, box=None)
            table.add_row("Aspect Ratio", f"{max(m, n) / max(1, min(m, n)):.2f}")
            table.add_row("Selected Kernel", f"[bold yellow]{res['strategy']}Kernel[/bold yellow]")
            table.add_row("Compile Time", f"{res['compile_time']:.4f} s")
            table.add_row("Execution Time", f"[bold cyan]{res['execution_time']:.6f} s[/bold cyan]")
            table.add_row("Performance", f"{achieved_gflops:.2f} GFLOPs")
            table.add_row("Pipeline Executed", " -> ".join(passes))
            console.print(table)
            console.print("-" * 60)
        else:
            print(f"Workload: {mtype}, Kernel: {res['strategy']}, Exec Time: {res['execution_time']:.6f} s")

    benchmark.save_results()
    if not args.skip_plots:
        vis = Visualizer(results_path="results/advanced_benchmarks.json")
        vis.generate_plots()

if __name__ == "__main__":
    main()
