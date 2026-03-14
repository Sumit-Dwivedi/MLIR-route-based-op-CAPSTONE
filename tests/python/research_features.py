import json
import os
import hashlib

class WorkloadProfilingCache:
    def __init__(self, cache_path="results/workload_cache.json"):
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        return {}

    def get_fingerprint(self, m, n, k):
        # Create a unique key for the matrix shape
        return hashlib.md5(f"{m}_{n}_{k}".encode()).hexdigest()

    def lookup(self, m, n, k):
        fp = self.get_fingerprint(m, n, k)
        return self.cache.get(fp)

    def update(self, m, n, k, strategy, execution_time):
        fp = self.get_fingerprint(m, n, k)
        self.cache[fp] = {
            "strategy": strategy,
            "execution_time": execution_time,
            "m": m, "n": n, "k": k
        }
        self._save_cache()

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=4)

if __name__ == "__main__":
    cache = WorkloadProfilingCache()
    print("Workload Profiling Cache initialized")
