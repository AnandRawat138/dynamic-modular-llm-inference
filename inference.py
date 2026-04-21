import torch
import time
from loader import load_base_model, ModuleLoader
from router import EmbeddingRouter
from utils.config import MAX_NEW_TOKENS
from utils.memory import get_ram_usage, get_gpu_usage


class ModularInference:
    def __init__(self):
        self.base_model, self.tokenizer = load_base_model()
        self.loader = ModuleLoader(self.base_model)
        self.router = EmbeddingRouter()

    def generate(self, prompt):
        start_time = time.time()

        ram_before = get_ram_usage()
        gpu_before = get_gpu_usage()

        modules = self.router.route(prompt, top_k=1)
        print(f"\n[Router] Selected modules: {modules}")

        for m in modules:
            self.loader.load_module(m)

        self.loader.activate_adapters(modules)

        model = self.loader.get_model()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

        ram_after = get_ram_usage()
        gpu_after = get_gpu_usage()
        latency = time.time() - start_time

        print(f"\n[Metrics]")
        print(f"Latency: {latency:.2f}s")
        print(f"RAM Δ: {ram_after - ram_before:.2f} GB")
        print(f"GPU Δ: {gpu_after - gpu_before:.2f} GB")

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)