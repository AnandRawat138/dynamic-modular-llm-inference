# Dynamic Modular LLM Inference

A research prototype for **memory-efficient Large Language Model (LLM) inference** using **dynamic modular composition** and **selective loading of model components**.

This repository accompanies the research paper:

> **Dynamic Modular Composition for Memory-Efficient Large Language Model Inference**

---

## 📌 Overview

Conventional LLM deployment requires loading the full model into memory, which creates a major bottleneck for systems with limited RAM/VRAM.

This project proposes a modular inference architecture where:

```text
Input Prompt
   ↓
Router Model
   ↓
Relevant Module Selection
   ↓
Dynamic Loading into Memory
   ↓
Base Model Composition
   ↓
Response Generation
Only task-relevant components are loaded at runtime, reducing peak memory footprint while maintaining practical inference performance.

🚀 Key Features
✅ Prompt-aware routing mechanism
✅ Dynamic module loading / unloading
✅ Memory-efficient inference pipeline
✅ Baseline vs modular benchmarking
✅ Latency and RAM usage comparison
✅ Experimental result visualization
✅ Reproducible research prototype
🧠 Core Idea

Instead of loading one monolithic LLM, the system divides capabilities into lightweight functional modules (adapters / domain units).

Examples:

Finance module
Medical module
Technical / Coding module
General reasoning module

A lightweight router analyzes the prompt and loads only relevant modules.

📂 Repository Structure
dynamic-modular-llm-inference/
│── main.py                 # Main execution entry point
│── router.py              # Prompt routing logic
│── loader.py              # Dynamic module loading
│── inference.py           # Response generation pipeline
│── experiment.py          # Benchmark experiments
│── result_ploter.py       # Plot generation
│── results.csv            # Benchmark data
│── results.png            # Latency comparison graph
│── results_memory_graph.png
│── README.md
│── LICENSE
│── .gitignore
│
├── modules/
│   └── create_adapter.py  # Example adapter/module creator
│
└── utils/
    ├── config.py
    └── memory.py
⚙️ Installation

Clone repository:

git clone https://github.com/AnandRawat138/dynamic-modular-llm-inference.git
cd dynamic-modular-llm-inference

Install dependencies:

pip install torch transformers psutil pandas matplotlib
▶️ Run Prototype
python main.py
🧪 Run Experiments
python experiment.py

This evaluates:

Modular vs baseline latency
Memory usage
Throughput
Routing overhead
📊 Example Results

Prototype evaluation demonstrates:

Up to ~49% reduction in peak memory usage
Competitive latency under warm-cache settings
Minimal quality degradation
Improved scalability with modular expansion
📈 Included Figures
Latency Comparison
results.png
Memory Usage Comparison
results_memory_graph.png
🔬 Research Contribution

This repository demonstrates that memory sparsity can be a practical alternative to compute sparsity approaches such as Mixture-of-Experts.

Instead of activating fewer parameters while keeping all weights resident, this system loads only required components into memory.

📄 Paper Citation
@article{rawat2026dynamic,
  title={Dynamic Modular Composition for Memory-Efficient Large Language Model Inference},
  author={Rawat, Anand and others},
  journal={Under Review},
  year={2026}
}
👥 Authors
Anand Rawat
Raghvendra Singh
Sanjeev Kumar
Nand Kishore Sharma
Vipin Kumar Jaiswal
📜 License

MIT License

📬 Contact

Anand Rawat
GitHub: https://github.com/AnandRawat138

⭐ Future Work
Real HuggingFace adapter loading
LoRA / PEFT integration
GPU memory benchmarks
Hybrid routing models
Edge deployment optimization
🤝 Contributions Welcome

Suggestions, forks, and research collaboration discussions are welcome.
