import matplotlib.pyplot as plt
import json
import os
import pandas as pd

class Visualizer:
    def __init__(self, results_path="results/benchmark_results.json", plots_dir="results/plots"):
        self.results_path = results_path
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def load_data(self):
        if os.path.exists(self.results_path):
            with open(self.results_path, 'r') as f:
                return pd.DataFrame(json.load(f))
        return pd.DataFrame()

    def generate_plots(self):
        df = self.load_data()
        if df.empty:
            print("No data available for plotting")
            return

        # 1. Runtime Comparison
        plt.figure(figsize=(10, 6))
        for strategy in df['chosen_strategy'].unique():
            subset = df[df['chosen_strategy'] == strategy]
            plt.scatter(subset['m'] * subset['n'] * subset['k'], subset['mlir_execution_time'], label=f"MLIR - {strategy}")
        plt.scatter(df['m'] * df['n'] * df['k'], df['numpy_execution_time'], label="Numpy Baseline", marker='x')
        plt.xlabel("Matrix Operation Volume (M * N * K)")
        plt.ylabel("Execution Time (s)")
        plt.title("Runtime Comparison: MLIR Optimization vs Numpy")
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(self.plots_dir, "runtime_comparison.png"))
        plt.close()

        # 2. Pipeline Distribution
        plt.figure(figsize=(10, 6))
        strategy_counts = df['chosen_strategy'].value_counts()
        strategy_counts.plot(kind='bar')
        plt.xlabel("Optimization Strategy")
        plt.ylabel("Count")
        plt.title("Optimization Pipeline Selection Distribution")
        plt.savefig(os.path.join(self.plots_dir, "pipeline_distribution.png"))
        plt.close()

        # 3. Speedup Chart
        plt.figure(figsize=(10, 6))
        df['speedup'] = df['numpy_execution_time'] / df['mlir_execution_time']
        plt.bar(df['shape'], df['speedup'])
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Matrix Shape")
        plt.ylabel("Speedup (Numpy Time / MLIR Opt Time)")
        plt.title("MLIR Optimization Speedup over Baseline")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "speedup_chart.png"))
        plt.close()

if __name__ == "__main__":
    vis = Visualizer()
    vis.generate_plots()
