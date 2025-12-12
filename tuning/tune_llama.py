import time
import subprocess
from llama_cpp import Llama
import psutil
import os

MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q8_0.gguf"
TEST_PROMPT = "Write a short poem about artificial intelligence."

def get_free_vram():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        return int(output.decode("utf-8").strip())
    except:
        return None

def run_test(n_gpu_layers, n_threads):
    print(f"\n=== Testing GPU Layers: {n_gpu_layers}, Threads: {n_threads} ===")

    start = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_ctx=512
    )

    output = llm(TEST_PROMPT, max_tokens=64)
    end = time.time()

    inference_time = end - start
    print(f"Inference Time: {inference_time:.2f}s")
    return inference_time

def main():
    free_vram = get_free_vram()
    print(f"Free VRAM: {free_vram} MB")

    cpu_cores = psutil.cpu_count(logical=False)
    print(f"Physical CPU Cores: {cpu_cores}")

    layer_candidates = [8, 16, 24, 32]
    thread_candidates = [cpu_cores, cpu_cores // 2] # type: ignore

    results = []

    for layers in layer_candidates:
        for threads in thread_candidates:
            try:
                t = run_test(layers, threads)
                results.append((t, layers, threads))
            except RuntimeError as e:
                print(f"Failed @ GPU Layers {layers}: {e}")

    results.sort(key=lambda x: x[0])

    print("\n=== Best Settings ===")
    best = results[0]
    print(f"Best Inference Time: {best[0]:.2f}s")
    print(f"Best GPU Layers: {best[1]}")
    print(f"Best Threads: {best[2]}")

if __name__ == "__main__":
    main()
