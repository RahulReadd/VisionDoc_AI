# VLM PDF Extractor

End-to-end pipeline that converts PDF documents into structured JSON using open-source Vision-Language Models (VLMs), running entirely on a free Google Colab T4 GPU (15 GB VRAM).

Supports **receipt parsing**, **signature detection**, **form field extraction**, and **key-value extraction**.

---

## Architecture

```
PDF file
   │
   ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  pdf_to_image.py │────▶│   extract.py     │────▶│   evaluate.py   │
│                  │     │                  │     │                  │
│ • Page diagnosis │     │ • Prompt engine  │     │ • Exact Match    │
│ • DPI selection  │     │ • VLM inference  │     │ • Token F1       │
│ • Skew detection │     │ • JSON parsing   │     │ • Menu Item F1   │
│ • Deskew         │     │                  │     │                  │
| • Resize         │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
   PIL Images                Structured JSON          Scores / CSV

                  ┌──────────────────────┐
                  │   src/models/         │
                  │ • VLMAdapter (ABC)    │
                  │ • ModelRegistry       │
                  │ • Qwen3-VL adapter    │
                  │ • Qwen2.5-VL adapter    │
                  │ • InternVL adapter    │
                  │ • Florence-2 adapter  │
                  │ • Pixtral adapter     │
                  │ • Llama Vision adapter│
                  └──────────────────────┘
```

**Orchestration:** `run_pipeline.py` ties all three modules together via CLI.  
**Demo:** `app/colab_demo.ipynb` provides a Gradio UI for interactive use in Colab.

---

## Model Selection

We benchmarked 5 open-source VLM families on the **CORD v2** receipt dataset using a Google Colab T4 GPU (15 GB VRAM). All models achieved **100% JSON parse success rate**.

### Benchmark Results

| Model | Model ID | Load (s) | VRAM Alloc (MB) | VRAM Reserve (MB) | Avg Inference (s) | JSON % |
|-------|----------|----------|-----------------|--------------------|--------------------|--------|
| **Qwen3-VL-2B** | Qwen/Qwen3-VL-2B-Instruct | 122.51 | 4,058 | 4,078 | **8.41** | 100% |
| Qwen3-VL-4B | Qwen/Qwen3-VL-4B-Instruct | 528.24 | 8,465 | 8,466 | 10.43 | 100% |
| Qwen2.5-VL-3B | Qwen/Qwen2.5-VL-3B-Instruct | 541.91 | 7,171 | 7,246 | 8.98 | 100% |
| InternVL 3.5-8B (4-bit) | OpenGVLab/InternVL3_5-8B-HF | 1,327.14 | 6,307 | 6,916 | 14.62 | 100% |
| Pixtral-12B (4-bit) | mistral-community/pixtral-12b | 2,846.68 | 8,693 | 8,704 | 22.82 | 100% |

> **Florence-2-large** and **Llama-3.2-11B-Vision** did not complete the benchmark due to compatibility issues on Colab T4

### Why Qwen3-VL-2B?

| Criteria | Qwen3-VL-2B | Runner-up (Qwen2.5-VL-3B) |
|----------|-------------|---------------------------|
| **Load time** | 122s (fastest) | 542s |
| **VRAM** | 4,058 MB (lowest — leaves 11 GB headroom) | 7,171 MB |
| **Inference speed** | 8.41s/image | 8.98s/image |
| **JSON success** | 100% | 100% |

**Qwen3-VL-2B** is the clear winner: it loads **4x faster**, uses **half the VRAM**, and matches or beats all other models on inference speed — all while maintaining 100% structured output reliability. The low VRAM footprint is critical because it leaves ample headroom for high-resolution document images on the T4's 15 GB budget.

### Models That Did Not Run

- **Florence-2-large**: Uses a different architecture (`AutoModelForCausalLM` with task-specific tokens like `<OCR_WITH_REGION>`) that expects Florence-specific prompt formats rather than free-form text prompts. Requires adapter-level prompt remapping to work with our generic benchmark harness.
- **Llama-3.2-11B-Vision**: Requires accepting Meta's community license on HuggingFace and `huggingface-cli login`. Additionally, at 11B parameters even with 4-bit quantization (~8 GB), it can hit CUDA memory limits when combined with large image tensors on T4.

The Adapter + Registry pattern makes swapping to any model a one-line change if these issues are resolved later.

Full benchmarking analysis is in `notebooks/01_model_selection.ipynb`.

---

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (or Google Colab with T4 runtime)

### Local / Colab Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/VLM_PDF_Extractor.git
cd VLM_PDF_Extractor

# Install dependencies
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
```

**Colab quick-start:** Open `app/colab_demo.ipynb` in Colab — it installs everything automatically.

### Model Download

Models are downloaded automatically on first use from HuggingFace Hub. For gated models like Llama-3.2:

```bash
huggingface-cli login --token YOUR_TOKEN
```

---

## Usage

### CLI — Single PDF

```bash
python -m app.run_pipeline --input invoice.pdf --task receipt --model qwen3-vl-2b
```

### CLI — CORD Dataset Evaluation

```bash
python -m app.run_pipeline --dataset cord --n 50 --task receipt --output results/json_outputs
```

### Gradio Demo (Colab)

Open `app/colab_demo.ipynb` in Google Colab, run all cells, and use the Gradio interface to upload PDFs or images.

### Available Tasks

| Task | Description |
|------|-------------|
| `receipt` | Extract menu items, subtotal, tax, total |
| `signature` | Detect handwritten signatures |
| `form_fields` | List form fields with filled/empty status |
| `key_value` | Extract all key-value pairs |

### Available Models

```python
from src.models import ModelRegistry
print(ModelRegistry.list_models())
# ['qwen3-vl-2b', 'qwen3-vl-4b', 'qwen3-vl-4b-4bit',
#  'internvl35-8b-4bit', 'florence2-large',
#  'pixtral-12b-4bit', 'llama32-11b-4bit']
```

---

## Project Structure

```
VLM_PDF_Extractor/
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/                          # Deliverable application modules
│   ├── __init__.py
│   ├── pdf_to_image.py           # PDF → VLM-ready images (deskew, DPI selection)
│   ├── extract.py                # VLM inference + prompt engineering
│   ├── evaluate.py               # Scoring against ground truth
│   ├── run_pipeline.py           # End-to-end CLI pipeline
│   └── colab_demo.ipynb          # Gradio UI for interactive demo
│
├── src/                          # Model adapter framework
│   ├── __init__.py
│   ├── benchmark.py              # Benchmarking utilities
│   └── models/
│       ├── __init__.py
│       ├── base.py               # VLMAdapter ABC + ModelConfig
│       ├── registry.py           # Model presets + registry
│       ├── qwen3_vl.py           # Qwen3-VL adapter
│       ├── internvl.py           # InternVL adapter
│       ├── florence2.py          # Florence-2 adapter
│       ├── pixtral.py            # Pixtral adapter
│       └── llama_vision.py       # Llama Vision adapter
│
├── results/                      # Pipeline outputs
│   ├── sample_inputs/            # Sample PDFs/images
│   ├── json_outputs/             # Extracted JSON (10+ documents)
│   ├── qualitative/              # Side-by-side examples
│   └── eval_summary.csv          # Metrics summary
│
│
└── Planning/                     # Research & planning documents
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **JSON Valid Rate** | % of outputs that parse as valid JSON |
| **Field Exact Match** | Exact match on scalar fields (total, subtotal, tax) |
| **Field Token F1** | Token-level F1 on scalar fields |
| **Menu Item F1** | Set F1 on extracted menu item names vs. ground truth |

---

## Key Design Decisions

1. **Adapter Pattern** — Each VLM family has its own adapter class implementing a common `VLMAdapter` interface. Swapping models requires changing a single string.

2. **Smart PDF Preprocessing** — Pages are diagnosed for size, type, and embedded DPI. Render DPI is auto-selected to maximize text clarity within the VLM's pixel budget. Global skew is detected and corrected.

3. **Prompt Engineering** — Task-specific prompts are structured to elicit valid JSON from the VLM, with fallback parsing for markdown-fenced or noisy outputs.

4. **Colab-First** — Everything is designed to run on a free T4 GPU. The notebooks document the full research and development process.
