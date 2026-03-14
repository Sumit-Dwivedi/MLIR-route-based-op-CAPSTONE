import math

class HybridCostModel:
    def __init__(self, hardware_profile):
        self.profile = hardware_profile
        self.peak_gflops = 1000.0 if hardware_profile["simd_support"]["avx512"] else 500.0
        self.bandwidth_gbs = 50.0 # GB/s
        self.cache_size_mb = 32.0

    def compute_kernel_score(self, m, n, k, kernel_type):
        flops = 2.0 * m * n * k
        mem_access = (m * k + k * n + m * n) * 4 # bytes
        oi = (flops / mem_access) if mem_access > 0 else 0
        
        # Roofline logic
        predicted_throughput = min(self.peak_gflops, oi * self.bandwidth_gbs)
        
        # 1. Compute Bound Score (0-1, higher is more compute bound)
        compute_bound_score = predicted_throughput / self.peak_gflops
        
        # 2. Memory Bound Score (0-1, higher is more memory bound)
        memory_bound_score = 1.0 - compute_bound_score
        
        # 3. Vector Efficiency
        vector_efficiency = 0.8 # default
        if kernel_type == "simd" and self.profile["simd_support"]["avx512"]:
            vector_efficiency = 0.95
        elif kernel_type == "small":
            vector_efficiency = 0.4 # higher overhead for small matrices
            
        # 4. Cache Reuse Score
        # Square blocking better for cache reuse
        cache_reuse_score = 0.5
        if kernel_type == "square" and (m*n*k*4 < self.cache_size_mb * 1024 * 1024):
            cache_reuse_score = 0.9
        elif kernel_type == "skinny":
            cache_reuse_score = 0.7 # row streaming
            
        # Weights
        w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
        
        # Score calculation: we want to maximize throughput and minimize latency
        # Higher score = Better performance predicted
        score = (w1 * compute_bound_score + 
                 w2 * (1.0 - memory_bound_score) + 
                 w3 * vector_efficiency + 
                 w4 * cache_reuse_score)
                 
        return {
            "score": score,
            "predicted_gflops": predicted_throughput * score,
            "oi": oi,
            "is_compute_bound": oi > (self.peak_gflops / self.bandwidth_gbs)
        }

if __name__ == "__main__":
    from hardware_detector import HardwareDetector
    hp = HardwareDetector.get_hardware_profile()
    model = HybridCostModel(hp)
    print(model.compute_kernel_score(4096, 64, 64, "skinny"))
