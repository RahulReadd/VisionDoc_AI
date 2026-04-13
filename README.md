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

We benchmarked 6 open-source VLMs on **20 CORD v2** receipt images using a Google Colab T4 GPU (15 GB VRAM), evaluating both operational and accuracy metrics. A weighted scoring system (Accuracy 40%, Speed 25%, VRAM 20%, JSON reliability 15%) produced the final ranking.

### Benchmark Results

| Model | Load (s) | VRAM (MB) | Avg Inf (s) | JSON % | Field EM | Field F1 | Menu F1 | Weighted Score |
|-------|----------|-----------|-------------|--------|----------|----------|---------|----------------|
| **Qwen3-VL-2B** ★ | 77.2 | 4,058 | 13.0 | 100% | 0.492 | 0.525 | 0.768 | **0.855** |
| Qwen2.5-VL-3B | 161.1 | 7,171 | 14.31 | 100% | 0.658 | 0.658 | 0.859 | 0.790 |
| Qwen3-VL-4B | 194.4 | 8,480 | 12.83 | 100% | 0.542 | 0.575 | 0.834 | 0.778 |
| InternVL 3.5-8B (4-bit) | 558.6 | 6,312 | 18.69 | 100% | 0.492 | 0.525 | 0.838 | 0.723 |
| Pixtral-12B (4-bit) | 916.3 | 8,705 | 23.79 | 100% | 0.208 | 0.208 | 0.735 | 0.567 |
| Llama-3.2-11B (4-bit) | 330.6 | 7,311 | 19.2 | 70% | 0.0 | 0.0 | 0.421 | 0.467 |

> **Florence-2-large** did not complete the benchmark — its task-token architecture is incompatible with free-form extraction prompts.

> **Note:** Qwen3-VL-2B was tested on raw CORD images (up to 2304×4096). All other models used preprocessed images (≤401K pixels) to prevent CUDA crashes. Qwen3-VL-2B's scores are therefore measured under harder conditions.

### Why Qwen3-VL-2B?

| Criteria | Qwen3-VL-2B | Runner-up (Qwen2.5-VL-3B) |
|----------|-------------|---------------------------|
| **Weighted Score** | **0.855** (1st) | 0.790 (2nd) |
| **Load time** | 77s (fastest) | 161s |
| **VRAM** | 4,058 MB (lowest — leaves 11 GB headroom) | 7,171 MB |
| **JSON success** | 100% | 100% |
| **Menu F1** | 0.768 | 0.859 (highest) |

**Qwen3-VL-2B** wins overall: it loads **2x faster**, uses **half the VRAM**, and maintains 100% JSON reliability. While Qwen2.5-VL-3B has higher raw accuracy (Menu F1 0.859), the VRAM and speed advantages of the 2B model are decisive on a T4's 15 GB budget — especially since it achieved these scores on unprocessed images, meaning production accuracy (with preprocessing) would be higher.

### Models That Did Not Complete / Underperformed

- **Florence-2-large**: Task-token architecture (`<OCR>`, `<OCR_WITH_REGION>`) is incompatible with free-form text prompts. Requires adapter-level prompt remapping.
- **Llama-3.2-11B-Vision (4-bit)**: Ran but achieved only 70% JSON parse rate and near-zero accuracy (Field EM = 0.0). Its strength is reasoning, not structured document extraction.

The Adapter + Registry pattern makes swapping to any model a one-line change.

Full benchmarking analysis is in `app/model_selection.ipynb`.

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
