class WorkloadClassifier:
    @staticmethod
    def classify(m, n, k):
        aspect_ratio = max(m, n) / max(1, min(m, n))
        
        if aspect_ratio > 32 and (m > 4096 or n > 4096):
            return "LLM_Inference_Large"
        elif aspect_ratio > 16:
            return "Skinny_Streaming"
        elif m == n == k and m >= 1024:
            return "HPC_Square_GEMM"
        elif m < 64 and n < 64 and k < 64:
            return "Tiny_Tensor_Op"
        elif k in [64, 128, 256, 512] and m > 1024:
            return "Transformer_Attention_Pattern"
        else:
            return "Generic_Workload"

if __name__ == "__main__":
    classifier = WorkloadClassifier()
    print(classifier.classify(4096, 64, 64))
