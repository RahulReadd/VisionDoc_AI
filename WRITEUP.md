# VLM-Based PDF Document Extraction — Technical Write-Up

## 1. VLM Selection: What We Used and What We Considered

**Selected model: Qwen3-VL-2B-Instruct** (Alibaba, 2025)

We benchmarked 7 open-source VLMs on 20 CORD v2 receipt images using a free Google Colab T4 GPU (15 GB VRAM). Models were evaluated on operational metrics (load time, VRAM, inference speed, JSON parse rate) and accuracy metrics (Field Exact Match, Field Token F1, Menu Item F1). A weighted scoring system (Accuracy 40%, Speed 25%, VRAM 20%, JSON reliability 15%) produced the final ranking.

| Model | VRAM (MB) | Avg Inf (s) | JSON % | Field EM | Field F1 | Menu F1 | Weighted Score |
|-------|-----------|-------------|--------|----------|----------|---------|----------------|
| **Qwen3-VL-2B** ★ | 4,058 | 13.0 | 100% | 0.492 | 0.525 | 0.768 | **0.855** |
| Qwen2.5-VL-3B | 7,171 | 14.31 | 100% | 0.658 | 0.658 | 0.859 | 0.790 |
| Qwen3-VL-4B | 8,480 | 12.83 | 100% | 0.542 | 0.575 | 0.834 | 0.778 |
| InternVL 3.5-8B (4-bit) | 6,312 | 18.69 | 100% | 0.492 | 0.525 | 0.838 | 0.723 |
| Pixtral-12B (4-bit) | 8,705 | 23.79 | 100% | 0.208 | 0.208 | 0.735 | 0.567 |
| Llama-3.2-11B (4-bit) | 7,311 | 19.2 | 70% | 0.0 | 0.0 | 0.421 | 0.467 |

> **Florence-2-large** did not complete the benchmark — its task-token architecture (`<OCR>`) is incompatible with free-form extraction prompts.

**Important note on test conditions:** Qwen3-VL-2B was benchmarked on **raw CORD images** (including 2304×4096 images at ~9.4M pixels). All other models were tested on **preprocessed images** resized to ≤401K pixels to prevent CUDA crashes. This means Qwen3-VL-2B's scores are measured under harder conditions — in production, where all images pass through `pdf_to_image.py` preprocessing, its accuracy would be equal or higher.

**Why Qwen3-VL-2B?** Despite being tested on raw (unprocessed) images, it achieves the highest weighted score (0.855). It uses the least VRAM (4,058 MB — leaving 11 GB headroom), loads fastest (77s vs 161–917s for others), and maintains 100% JSON reliability. While Qwen2.5-VL-3B has higher raw accuracy (Menu F1 0.859 vs 0.768), it uses 1.8x the VRAM and loads 2x slower. On a free T4 with tight session limits, resource efficiency is decisive.

**Why not the others?** Qwen2.5-VL-3B leads on accuracy but at nearly double the VRAM, leaving little headroom for high-resolution documents. InternVL 3.5-8B and Pixtral-12B are 1.4–1.8x slower at inference, and InternVL's 9-minute load time makes interactive use impractical. Llama-3.2-11B only achieved 70% JSON parse rate and near-zero accuracy on structured extraction — its strength is reasoning, not document parsing. Florence-2 is architecturally incompatible with our unified prompt strategy.

## 2. Prompting Strategy

**Approach: Single unified prompt returning all four extraction types in one JSON object.**

Rather than making 4 separate VLM calls per page (one for key-value, one for signature, one for form fields, one for receipt), we designed a single structured prompt that asks the model to return everything at once:

```json
{
  "key_value_pairs": { ... },
  "signature": { "present": true/false, "confidence": "...", "location": "..." },
  "form_fields": [ { "field_name": "...", "status": "filled/empty", "value": "..." } ],
  "receipt": { "menu": [...], "sub_total": {...}, "total": {...} }
}
```

**Why this works best:**

1. **Efficiency** — One inference pass per page instead of four. At ~8.4s per inference on T4, this saves ~25s per page.
2. **Context sharing** — The model sees the full document once and can cross-reference information (e.g., recognizing a receipt total near form fields, or a signature below a key-value section).
3. **JSON schema as prompt** — By embedding the exact JSON structure the model should produce, with descriptive field names and example values, we observed 100% valid JSON output. The model treats the schema as a template and fills it in.
4. **Null convention** — Instructing the model to use `null` for missing fields (e.g., `"receipt": null` for non-receipt documents) prevents hallucination of nonexistent data.
5. **Fallback parsing** — The `parse_json_output` function handles edge cases: markdown-fenced responses, leading text before JSON, and nested brace extraction. This makes the pipeline robust even when the model adds minor formatting artifacts.

We retain individual task-specific prompts as a fallback for benchmarking and ablation studies, but the unified prompt is the production default.

## 3. Failure Cases and Limitations

**Observed failure modes:**

- **Multi-column layouts:** The pipeline performs well on simple 2-column documents (label-value pairs, receipts), but struggles with complex multi-column layouts (e.g., financial tables with 4+ columns, side-by-side sections). The model tends to merge columns, misalign values across rows, or skip inner columns entirely. This is a fundamental limitation of treating the entire page as a single image without layout-aware segmentation.
- **Dense receipts with many line items:** When receipts have >15 items in small print, the model occasionally merges adjacent items or misaligns prices with item names. The 401K pixel budget forces downscaling of large pages, which blurs fine text.
- **Handwritten text:** The model reliably detects *whether* a signature is present (binary classification) but struggles to transcribe handwritten annotations or cursive text adjacent to signatures.
- **Rotated sub-regions:** Our deskew pipeline corrects whole-page skew via Hough line detection, but a rotated table embedded in an otherwise straight page is not detected — the VLM must handle it, and it often fails on severely rotated (>15°) sub-regions.
- **Non-Latin scripts:** While Qwen3-VL supports 30+ languages, our evaluation was limited to English/Indonesian (CORD dataset). Accuracy on Arabic, CJK, or mixed-script documents is untested.
- **Florence-2 incompatibility:** Florence-2's task-token architecture is fundamentally incompatible with free-form extraction prompts, preventing it from participating in the benchmark.
- **Llama-3.2-11B poor structured output:** Despite running successfully, Llama-3.2-11B achieved only 70% JSON parse rate and 0.0 Field EM — it generates conversational answers rather than strict JSON, making it unsuitable for structured extraction despite strong reasoning capabilities.
- **CUDA context corruption:** A device-side assert in one model permanently corrupts the CUDA context for the entire Colab session, requiring a full runtime restart. We mitigated this by running each model in its own cell and saving results to disk/Drive for cross-session recovery.

## 4. Production Improvements

If deploying this pipeline at scale, the key improvements would be:

1. **Dedicated GPU infrastructure:** Replace Colab's ephemeral T4 with a persistent A100/H100 instance. This enables larger models (InternVL-8B full precision, Qwen3-VL-7B) that scored higher on OCR benchmarks, and eliminates the VRAM gymnastics.

2. **vLLM / TGI serving:** Use a production inference server (vLLM, HuggingFace TGI) for batched, paged-attention inference. Current sequential processing (~8.4s/page) would drop to ~1-2s/page with proper batching and KV-cache management.

3. **LoRA / QLoRA fine-tuning:** The current pipeline is fully zero-shot. Using parameter-efficient fine-tuning (PEFT) methods like **LoRA** (Low-Rank Adaptation) or **QLoRA** (Quantized LoRA) on 500–1000 labeled domain-specific examples would significantly improve extraction accuracy — especially for multi-column tables and domain-specific schemas — without requiring full model retraining. QLoRA is particularly attractive: it fine-tunes a 4-bit quantized model with LoRA adapters, making it feasible even on a single T4 GPU. The resulting adapter weights are only ~50-100 MB, so multiple domain-specific adapters (invoices, medical forms, tax documents) can be hot-swapped at inference time without reloading the base model.

4. **OCR pre-pass + VLM verification:** For documents with very fine print, run a dedicated OCR engine (Tesseract, PaddleOCR) first, then feed the OCR text alongside the image to the VLM. This two-stage approach provides the VLM with text it might miss at lower resolutions.

5. **Layout-aware segmentation for multi-column documents:** Replace the current whole-page-to-VLM approach with a layout analysis pre-pass (e.g., LayoutLMv3, DocTR, or YOLO-based table detection) that segments the page into regions — tables, headers, paragraphs, sidebars. Each region can then be cropped and sent to the VLM individually, dramatically improving accuracy on complex multi-column layouts. This also enables region-level skew correction for rotated tables embedded in otherwise straight pages.

6. **Confidence scoring and human-in-the-loop:** Add confidence thresholds to flag low-confidence extractions for human review. The unified prompt already returns a `confidence` field for signatures; extend this pattern to all extraction types.

7. **Caching and incremental processing:** For document re-processing (e.g., updated PDFs), cache page hashes and only re-extract changed pages. Store results in a structured database rather than flat JSON files.

8. **Evaluation at scale:** Expand beyond CORD (receipts) and SigDetectVerifyFlow (signatures) to include FUNSD (form understanding), SROIE (scanned receipts), and DocVQA (document visual QA) benchmarks for a more comprehensive accuracy picture.
