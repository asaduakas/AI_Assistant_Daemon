from llama_cpp import Llama
import time

#----------CONFIG----------
model_path = "./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
# model_path = "./models/mistral-7b-instruct-v0.1.Q8_0.gguf"

n_threads = 8

prompt = "Write a short poem about AI programming"

results = []

for layers in [8, 16, 24, 32]:
    n_layer = layers
    llm = Llama(model_path=model_path, n_threads=n_threads, n_gpu_layers=layers)
    start = time.time()
    result = llm(prompt, max_tokens=128)
    end = time.time()

    results.append({
        "gpu_layers": layers,
        "inference_time": end - start,
        "output": result["choices"][0]["text"] # type: ignore
    })

print("\n=== Summary ===\n")
for r in results:
    print(f"GPU Layers: {r['gpu_layers']}")
    print(f"Inference time: {r['inference_time']:.2f}s")
    print("Output:")
    print(r["output"])
    print("-" * 40)

