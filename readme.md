<p align="center">
 <!-- <img src="docs/logo.png" style="height: 80px;"> -->
 <h1 align="center">Improving LLM Reasoning in Materials Science</h1>
</p>
<p align="center">
 <a><img alt="Status" src="https://img.shields.io/badge/status-research_prototype-6a5acd"></a>
 <a><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue"></a>
 <a><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
</p>

## About


This repository is a part of Master's thesis in AI: "Enhancing the Reasoning of LLMs in Materials Discovery via Causal Contracts"

## 📋 Overview

This repository contains a modular, DAG‑based framework for experimenting with **pipelines that test and improve the reasoning capabilities of large language models (LLMs)**. The system offers two end‑to‑end pipelines:

* **Document Pipeline** — downloads papers, extracts pages, performs multi‑stage analysis, and synthesizes structured artifacts (semantic summaries and contracts) before generating QA datasets.
* **Answer & Evaluation Pipeline** — answers the generated questions under configurable context settings and evaluates performance with transparent, reproducible artifacts.

The pipelines are built from interoperable **agents** orchestrated by a lightweight **DAG runner**, making it easy to plug in new components, models, or evaluation strategies. Code quality, reproducibility, and documentation are priorities.


> 📖 Paper: *coming soon*

## 🏁 Highlights

* **DAG‑orchestrated research pipelines** with explicit data contracts between tasks.
* **Agentic design**: discrete agents for download, extraction, analysis, contract writing, QA generation, answering, and evaluation.
* **Reproducible runs**: each experiment writes all inputs/outputs to a timestamped folder under `runs/`.
* **Configurable model runtime** via CLI flags (`--model`, `--temperature`, `--retries`).
* **Context controls** for ablations (e.g., raw text vs. additional context flags).

## 📋 Table of Contents

* [🚀 Get Started](#-get-started)

  * [System Requirements](#system-requirements)
  * [Installation](#installation)
  * [Environment Setup](#environment-setup)
  * [Quick Start](#quick-start)
* [🔧 Framework Components](#-framework-components)

  * [🧬 Document Pipeline](#-document-pipeline)
  * [🧪 Answer & Evaluation Pipeline](#-answer--evaluation-pipeline)
  * [⚙️ Core Runtime](#️-core-runtime)
  * [🗂️ Utilities & Middleware](#️-utilities--middleware)
* [📦 Project Structure](#-project-structure)
* [📈 Reproducibility & Experiment Artifacts](#-reproducibility--experiment-artifacts)
* [📝 Configuration Reference](#-configuration-reference)
* [🙌 Acknowledgements](#-acknowledgements)
* [📚 Citation](#-citation)

## 🚀 Get Started

#### System Requirements

* Python **3.10+**
* macOS/Linux/WSL2 (Windows native should also work)
* Recommended: virtual environment (Conda/venv)

#### Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd <your-repo>
```

2. **Create and activate a Conda environment**

```bash
conda create -n llm-reasoning python=3.10 -y
conda activate llm-reasoning
```

3. **Install dependencies**

* Using pip (from `requirements.txt`):

```bash
pip install -r requirements.txt
```

* Or, if you maintain an `environment.yml`:

```bash
conda env update -n llm-reasoning -f environment.yml
```

> Tip: If you're on Apple Silicon or using CUDA, prefer installing any platform‑specific packages via conda first, then fall back to pip.

#### Environment Setup

Minimal configuration is required for local runs. By default, artifacts are written to `./runs`. Ensure the directories under `data/` and `test_data/` exist and contain the input files referenced below.

#### Quick Start

**Document Pipeline** — given a list of paper URLs:

```bash
python app.py document  --working-dir runs \
  --papers ./test_data/papers_1.lst \
  --run-id doc_pipeline_test  \
  --model openai/o4-mini --temperature 1.0 --retries 3
```

**Answer & Evaluation Pipeline** — given a QA dataset and contracts:

```bash
python app.py answer --working-dir runs \
  --contracts ./data/contracts2.json \
  --dataset   ./data/full_dataset2.json \
  --flags RAW_TEXT \
  --model openai/o4-mini --temperature 1.0 --retries 3
```

```bash
python app.py design  --working-dir runs \
   --contracts ./data/contracts2.json \
   --contract_id 21-0 \
   --run-id design_pipeline-20250915-065507 \
   --model openai/o4-mini --temperature 1.0 --retries 3

```

> Both commands create a timestamped experiment folder inside `runs/` that contains all intermediate and final artifacts.

## 🔧 Framework Components

### General notes on code architecture

The LLM client part is implemented using LiteLLM library, and this makes our tool model-agnostic. When using two providers,
appropriate API keys must be provided. 

### 🧬 Document Pipeline

**Goal:** transform raw papers into structured, analysis‑ready artifacts and generate a high‑quality QA dataset for reasoning evaluation.

**Stages (agents):**

1. **DownloadAgent** — fetches documents from URLs listed in `--papers`.
2. **ExtractionAgent** — extracts pages and creates normalized document IDs.
3. **DocumentAnalyzerAgent** — runs first‑pass analysis on the processed documents.
4. **SemanticAnalyzerAgent** — performs deeper semantic analysis to derive structured representations.
5. **ContractWriterAgent** — synthesizes *contracts* (concise, schema‑constrained summaries) to support controlled QA.
6. **QADatasetGeneratorAgent** — produces a QA dataset aligned with the contracts and selected context flags.

**Outputs:** processed IDs, semantic documents, contracts, and a generated QA dataset — all stored under the current run directory.


### 🧪 Answer & Evaluation Pipeline

**Goal:** answer questions under specified context settings and evaluate performance.

**Stages (agents):**

1. **QAAnswerAgent** — answers questions from a QA dataset using the selected model configuration and context flags.
2. **QAEvaluationAgent** — computes evaluation metrics and produces analysis artifacts.

**Outputs:** model answers and evaluation reports (JSON/CSV/plots as configured) under the current run directory.

### ⚙️ Core Runtime

* **DAG / DAGRunner** — schedules and executes agents given explicit data dependencies; ensures artifact lineage, reproducibility, and fault isolation.
* **Composability** — use the DAG builders in `app4.py` as examples to assemble new experimental pipelines.

### 🗂️ Utilities & Middleware

* **`utils.prompt_manager.PromptManager`** — centralizes prompts used by analysis/answering agents.
* **`utils.common.ModelConfig`** — encapsulates model runtime knobs (`name`, `temperature`, `retries`).
* **`middleware.ImageStorage`** — manages image assets produced along the pipeline.

## 📦 Project Structure


```
<repo-root>/
├─ agents/                  # Agent implementations (download, extract, analyze, QA, eval, ...)
├─ core/                    # DAG & runtime abstractions
├─ data/                    # Generated artifacts (contracts, datasets, etc)
├─ docs/                    # Technical documentation
├─ docucache/              
├─ middleware/              # Shared services (e.g., image storage)
├─ notebooks/               # Analytics for thesis            
├─ runs/                    # Experiment outputs (created at runtime)
├─ test_data/               # Test lists / small fixtures (e.g., paper URLs)
├─ utils/                   # Prompt manager, settings, model config, logging helpers
├─ requirements.txt         # Python dependencies
└─ app.py                  # CLI entrypoint: build & run DAG pipelines
```

## The file structure of DocuCache

```shell
└── docucache/
    ├── metadata.db            <-- The SQLite database file
    ├── 1/                     <-- First paper's folder (ID from DB)
    │   ├── assets/
    │   └── tmp/
    │       └── 1706.03762.pdf
    ├── 2/
    │   ├── assets/
    │   └── tmp/
    │       └── 2203.02155.pdf
    └── 3/
        ├── assets/
        └── tmp/
            └── 2307.09288.pdf
```



## 📈 Reproducibility & Experiment Artifacts

Every run produces a folder `runs/<pipeline-name>-YYYYMMDD-HHMMSS/` containing:

* **Inputs:** exact copies or references to all inputs used by each agent.
* **Intermediate Artifacts:** JSON manifests (e.g., processed document IDs, semantic docs, contracts, model answers).
* **Final Reports:** evaluation summaries and any generated plots/tables.
* **Logs:** structured logs with timestamps for traceability.

You can pass a custom `--run-id` to name the experiment folder deterministically.

## 📝 Configuration Reference

### Common CLI Flags

* `--working-dir` (default: `runs`) — base directory for artifacts.
* `--model` (default: `openai/o4-mini`) — model name/ID in a standard format provider/model_name.
* `--temperature` (default: `1.0`) — sampling temperature.
* `--retries` (default: `3`) — retry attempts for model calls.

### Document Pipeline

* `--papers` — path to a text file containing one URL per line (default: `./test_data/papers_1.lst`).
* `--run-id` — optional experiment/run identifier.

### Answer & Evaluation Pipeline

* `--dataset` — path to a QA dataset JSON (default: `./data/test_dataset.json`).
* `--contracts` — path to a contracts JSON (default: `./data/test_contracts.json`).
* `--flags` — context flags (e.g., `RAW_TEXT`, `CC`, or `CC+RAW_TEXT`).
* `--run-id` — optional experiment/run identifier.

### Programmatic Use

Import and extend the DAG builders from `app.py`:

```python
from app import get_document_pipeline_dag, get_answer_pipeline_dag

# Build a custom DAG and run it with your own runner/configuration
```

## 📚 Citation

If you use this repository in academic work, please cite it as:

```bibtex
@misc{kaliutau2025,
  author = {Kaliutau, A.},
  title = {{AI in Material Science}},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/stable-reasoning/ai-material-science}},
}
```

---

### 💡 Tips

* Keep the structure of your input files stable; the agents rely on consistent schemas.
* For ablations, duplicate a prior run with only one change (e.g., `--flags`, `--temperature`).
* Use `--run-id` to align artifacts across multiple pipelines when running a batched study.

