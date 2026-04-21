import time
import pandas as pd
from inference import ModularInference
from loader import load_base_model, ModuleLoader
from utils.memory import get_ram_usage

prompts = [
    "What is overfitting in machine learning?",
    "Summarize climate change impacts",
    "Write Python code for quicksort",
    "Explain neural networks",
    "Summarize artificial intelligence"
]


def run_modular():
    system = ModularInference()
    results = []

    for p in prompts:
        start = time.time()
        ram_before = get_ram_usage()

        system.generate(p)

        ram_after = get_ram_usage()
        latency = time.time() - start

        results.append({
            "type": "modular",
            "prompt": p,
            "latency": latency,
            "ram": ram_after - ram_before
        })

    return results


def run_baseline():
    model, tokenizer = load_base_model()
    loader = ModuleLoader(model)

    # Load ALL adapters
    for m in ["qa_adapter", "summarization_adapter", "code_adapter"]:
        loader.load_module(m)

    loader.activate_adapters(["qa_adapter"])  # arbitrary

    results = []

    for p in prompts:
        start = time.time()
        ram_before = get_ram_usage()

        inputs = tokenizer(p, return_tensors="pt")
        model = loader.get_model()

        outputs = model.generate(**inputs, max_new_tokens=128)

        ram_after = get_ram_usage()
        latency = time.time() - start

        results.append({
            "type": "baseline",
            "prompt": p,
            "latency": latency,
            "ram": ram_after - ram_before
        })

    return results


if __name__ == "__main__":
    modular = run_modular()
    baseline = run_baseline()

    df = pd.DataFrame(modular + baseline)
    df.to_csv("results.csv", index=False)

    print(df)