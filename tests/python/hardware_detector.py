import cpuinfo
import GPUtil
import platform
import subprocess

class HardwareDetector:
    @staticmethod
    def get_hardware_profile():
        cpu_info = cpuinfo.get_cpu_info()
        gpus = GPUtil.getGPUs()
        
        profile = {
            "cpu_vendor": cpu_info.get("vendor_id_raw", "Unknown"),
            "cpu_brand": cpu_info.get("brand_raw", "Unknown"),
            "cpu_cores": cpu_info.get("count", 0),
            "simd_support": {
                "avx2": "avx2" in cpu_info.get("flags", []),
                "avx512": any("avx512" in f for f in cpu_info.get("flags", [])),
            },
            "gpu_present": len(gpus) > 0,
            "gpus": [],
            "os": platform.system(),
            "arch": platform.machine()
        }
        
        for gpu in gpus:
            profile["gpus"].append({
                "name": gpu.name,
                "vendor": "NVIDIA", # GPUtil is for NVIDIA
                "memory_total": gpu.memoryTotal,
                "driver": gpu.driver
            })
            
        return profile

if __name__ == "__main__":
    profile = HardwareDetector.get_hardware_profile()
    import json
    print(json.dumps(profile, indent=2))
