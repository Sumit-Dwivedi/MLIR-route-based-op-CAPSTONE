import json
import os
import numpy as np

class LearnedCostModel:
    def __init__(self, data_path="results/kernel_training_data.json"):
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                return json.load(f)
        return []

    def record_performance(self, m, n, k, kernel_type, hardware, runtime):
        entry = {
            "m": m, "n": n, "k": k,
            "kernel": kernel_type,
            "hardware": hardware,
            "runtime": runtime
        }
        self.data.append(entry)
        with open(self.data_path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def predict_best_kernel(self, m, n, k, candidates, hardware):
        # Research Placeholder: simple KNN-like search or fallback to analytical
        # In a real system, we'd use a small regression model here.
        best_kernel = candidates[0]
        min_expected_time = float('inf')
        
        # Look for similar shapes in data
        for entry in self.data:
            if abs(entry["m"] - m) < 10 and abs(entry["n"] - n) < 10:
                if entry["runtime"] < min_expected_time:
                    min_expected_time = entry["runtime"]
                    best_kernel = entry["kernel"]
                    
        return best_kernel
