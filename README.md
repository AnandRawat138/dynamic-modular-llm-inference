<div align="center">

# 🚀 Dynamic Modular LLM Inference

### Memory-Efficient Large Language Model Inference using Dynamic Modular Composition

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Research](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Architecture](https://img.shields.io/badge/Focus-LLM%20Systems-purple.svg)]()

</div>

---

## 📌 Overview

Large Language Models (LLMs) typically require loading the **entire model into memory**, resulting in high RAM / VRAM usage and limiting deployment on resource-constrained devices.

This repository presents a **dynamic modular inference architecture** that loads only task-relevant model components during runtime.

Instead of monolithic loading:

```text
Load Full LLM → High Memory Usage
We use:

Prompt → Router → Select Modules → Load On Demand → Generate Output
```
This reduces active memory footprint while maintaining practical inference performance.

🧠 Core Idea

The proposed system introduces memory sparsity:

Traditional systems keep all parameters loaded.
Our architecture loads only the required modules.

Examples of functional modules:
```text
Module	Purpose
💰 Finance	Market / stock / economics prompts
🩺 Medical	Health / biomedical prompts
💻 Technical	Coding / engineering prompts
🌐 General	Common reasoning prompts
```
A lightweight router model determines which modules are required.
```text
⚙️ System Pipeline
┌──────────────┐
│ Input Prompt │
└──────┬───────┘
       ↓
┌──────────────┐
│ Router Model │
└──────┬───────┘
       ↓
┌─────────────────────┐
│ Relevant Modules    │
│ Selected Dynamically│
└──────┬──────────────┘
       ↓
┌──────────────┐
│ Load Modules │
└──────┬───────┘
       ↓
┌──────────────┐
│ Base Model   │
│ Composition  │
└──────┬───────┘
       ↓
┌──────────────┐
│ Final Output │
└──────────────┘
```
✨ Key Features
✅ Prompt-aware routing

✅ Selective module loading

✅ Lower peak memory footprint

✅ Baseline vs modular benchmarking

✅ Latency / RAM / throughput evaluation

✅ Reproducible research prototype

✅ Expandable modular architecture

📂 Repository Structure
```text
dynamic-modular-llm-inference/
│── main.py
│── router.py
│── loader.py
│── inference.py
│── experiment.py
│── result_ploter.py
│── results.csv
│── results.png
│── results_memory_graph.png
│── README.md
│── LICENSE
│── .gitignore
│
├── modules/
│   └── create_adapter.py
│
└── utils/
    ├── config.py
    └── memory.py
```
⚙️ Installation
1️⃣ Clone Repository
```text
git clone https://github.com/AnandRawat138/dynamic-modular-llm-inference.git
cd dynamic-modular-llm-inference
```
2️⃣ Install Dependencies
```text
pip install torch transformers psutil pandas matplotlib
```
▶️ Run Prototype
```text
python main.py
```
🧪 Run Benchmark Experiments
```text
python experiment.py
```
Evaluates:

```text
Latency comparison
Memory usage
Throughput
Modular loading overhead
📊 Example Results
Metric	Baseline	Proposed System
Peak Memory	100%	51%
Warm Cache Latency	1.0x	1.1x
Accuracy	100%	98.8%
Scalability	Moderate	High
📈 Included Graphs
Latency Comparison
```
results.png
Memory Usage Comparison
results_memory_graph.png

🔬 Research Contribution

This project explores an alternative to compute-sparsity systems such as Mixture-of-Experts.
```text
Traditional MoE:
Fewer active parameters
All experts often remain in memory
Proposed Approach:
Only selected modules loaded
Lower active memory footprint
```

This makes the design attractive for:
```text
Edge devices
Resource-limited servers
Multi-tenant inference systems
Memory-aware AI deployment
```
📄 Associated Research Paper

Dynamic Modular Composition for Memory-Efficient Large Language Model Inference

@article{rawat2026dynamic,
  title={Dynamic Modular Composition for Memory-Efficient Large Language Model Inference},
  author={Rawat, Anand and others},
  journal={Under Review},
  year={2026}
}

👨‍💻 Authors
```text
Anand Rawat
Raghvendra Singh
Sanjeev Kumar
Nand Kishore Sharma
Vipin Kumar Jaiswal
```
📜 License

MIT License

🔭 Future Roadmap
```text
🔹 Real HuggingFace adapter loading
🔹 LoRA / PEFT integration
🔹 GPU memory benchmarks
🔹 Hybrid routing strategies
🔹 Edge deployment optimization
🔹 Multi-node serving support
```
🤝 Contributions
```text
Contributions, suggestions, forks, and collaborations are welcome.
```
