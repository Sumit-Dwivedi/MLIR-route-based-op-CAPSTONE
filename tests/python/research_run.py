import os
import json
import time
import argparse
from hardware_detector import HardwareDetector
from cost_model import RooflineModel
from autotuner import Autotuner
from advanced_run import AdvancedBenchmark
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    rich_available = True
except ImportError:
    rich_available = False

def main():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("tests/python/tmp", exist_ok=True)
    
    if rich_available:
        console = Console()
        console.print(Panel("[bold green]MLIR Heterogeneous Kernel Selection Research Framework[/bold green]"))

    # 1. Hardware Detection
    hp = HardwareDetector.get_hardware_profile()
    if rich_available:
        console.print(f"[blue]Detected Hardware:[/blue] {hp['cpu_brand']}")
        console.print(f"[blue]SIMD:[/blue] AVX2={hp['simd_support']['avx2']}, AVX512={hp['simd_support']['avx512']}")
        console.print(f"[blue]GPU:[/blue] {'Available' if hp['gpu_present'] else 'Not Found'}")

    # 2. Autotuning
    tuner = Autotuner()
    tuner.tune()

    # 3. Advanced Benchmarking
    bench = AdvancedBenchmark()
    bench.run_all()

    # 4. Results Analysis & Visualization
    df = pd.DataFrame(bench.results)
    df['speedup_vs_pytorch'] = df['torch_cpu'] / df['mlir_opt']
    
    # Speedup vs PyTorch Plot
    plt.figure(figsize=(10, 6))
    plt.bar(df['name'], df['speedup_vs_pytorch'])
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.ylabel("Speedup (PyTorch CPU Time / MLIR Opt Time)")
    plt.title("MLIR Research Pipeline Speedup vs PyTorch CPU")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/plots/speedup_vs_pytorch.png")
    
    # Kernel Selection Summary
    if rich_available:
        console.print("\n[bold cyan]Kernel Selection Research Summary[/bold cyan]")
        table = Table(title="Research-grade Matrix Benchmarks")
        table.add_column("Case", style="magenta")
        table.add_column("Shape", style="green")
        table.add_column("Strategy", style="yellow")
        table.add_column("Speedup (vs Torch)", style="cyan")
        
        for _, row in df.iterrows():
            table.add_row(row['name'], str(row['shape']), row['strategy'], f"{row['speedup_vs_pytorch']:.2f}x")
        console.print(table)

    print("\nResearch results saved to results/advanced_benchmarks.json")
    print("Plots generated in results/plots/")

if __name__ == "__main__":
    main()
