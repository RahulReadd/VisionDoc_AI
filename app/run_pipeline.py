"""Module 4: End-to-End VLM PDF Extraction Pipeline.

Orchestrates: pdf_to_image -> extract -> evaluate.

By default extracts ALL aspects (key-value, signature, form fields, receipt)
in a single unified pass. Use --task to run a single-task prompt instead.

Usage:
    python -m app.run_pipeline --input invoice.pdf
    python -m app.run_pipeline --dataset cord --n 50 --task receipt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from app.pdf_to_image import convert_pdf, load_image
from app.extract import DocumentExtractor
from app.evaluate import (
    evaluate_single, parse_cord_ground_truth,
    evaluate_signature_single, evaluate_signature_batch,
)


def run_on_file(extractor: DocumentExtractor, file_path: str, task: str | None, output_dir: str) -> list[dict]:
    """Process a single PDF or image file."""
    path = Path(file_path)
    results = []

    if path.suffix.lower() == ".pdf":
        print(f"Converting PDF: {path.name}")
        images, diagnostics = convert_pdf(path)
    else:
        images = [load_image(path)]

    label = task or "full"
    for i, img in enumerate(images):
        print(f"  Page {i+1}/{len(images)}: extracting ({label})...", end=" ")
        pred = extractor.extract(img, task=task)

        result = {
            "source": path.name,
            "page": i,
            "task": label,
            "prediction": pred["parsed_json"],
            "json_valid": pred["json_valid"],
            "raw_output": pred["raw_output"],
        }
        results.append(result)

        out_path = os.path.join(output_dir, f"{path.stem}_p{i}_{label}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        status = "OK" if pred["json_valid"] else "FAIL"
        print(f"[{status}]")

    return results


def run_on_cord(extractor: DocumentExtractor, n: int, task: str | None, output_dir: str) -> list[dict]:
    """Process N images from the CORD dataset with evaluation.

    Note: evaluation metrics (Field EM, F1, Menu F1) are computed when using
    task='receipt' or task=None (unified prompt). The unified prompt wraps
    receipt fields inside a 'receipt' key, which evaluate_single handles.
    """
    from datasets import load_dataset

    cord = load_dataset("naver-clova-ix/cord-v2", split="test")
    n = min(n, len(cord))
    results = []

    for i in range(n):
        img = cord[i]["image"].convert("RGB")
        gt_raw = cord[i]["ground_truth"]
        gt_parsed = parse_cord_ground_truth(gt_raw)

        pred = extractor.extract(img, task=task)

        pred_json = pred["parsed_json"]
        # If unified prompt, receipt fields are nested under "receipt" key
        eval_json = pred_json
        if pred_json and "receipt" in pred_json and isinstance(pred_json["receipt"], dict):
            eval_json = pred_json["receipt"]

        scores = evaluate_single(eval_json, gt_parsed)

        result = {
            "image_idx": i,
            "source": "cord",
            "prediction": pred["parsed_json"],
            "json_valid": pred["json_valid"],
            "scores": scores,
        }
        results.append(result)

        out_path = os.path.join(output_dir, f"cord_{i:03d}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  Processed {i+1}/{n}")

    valid = sum(1 for r in results if r["json_valid"])
    avg_f1 = sum(r["scores"].get("field_f1", 0) for r in results) / len(results)
    avg_menu = sum(r["scores"].get("menu_f1", 0) for r in results) / len(results)

    print(f"\n  JSON valid: {valid}/{n} ({valid/n:.1%})")
    print(f"  Avg Field F1: {avg_f1:.3f}")
    print(f"  Avg Menu F1:  {avg_menu:.3f}")

    return results


def run_on_signatures(extractor: DocumentExtractor, n: int, task: str | None, output_dir: str) -> dict:
    """Evaluate signature detection using a balanced set:
      - Positive samples: SigDetectVerifyFlow (documents WITH signatures)
      - Negative samples: CORD receipts (documents WITHOUT signatures)

    Uses n//2 images from each source for a balanced binary evaluation.
    """
    from datasets import load_dataset
    import numpy as np

    half = n // 2

    # Positive: documents with signatures
    print(f"  Loading {half} positive samples (SigDetectVerifyFlow)...")
    sig_ds = load_dataset("Mels22/SigDetectVerifyFlow", split="test")
    pos_indices = np.linspace(0, len(sig_ds) - 1, half, dtype=int).tolist()
    pos_images = [sig_ds[i]["document"].convert("RGB") for i in pos_indices]
    pos_labels = [True] * half

    # Negative: receipts without signatures
    print(f"  Loading {half} negative samples (CORD receipts)...")
    cord = load_dataset("naver-clova-ix/cord-v2", split="test")
    neg_indices = np.linspace(0, len(cord) - 1, half, dtype=int).tolist()
    neg_images = [cord[i]["image"].convert("RGB") for i in neg_indices]
    neg_labels = [False] * half

    all_images = pos_images + neg_images
    all_labels = pos_labels + neg_labels

    print(f"  Running signature detection on {len(all_images)} images...")

    predictions = []
    for i, img in enumerate(all_images):
        pred = extractor.extract(img, task=task)
        predictions.append(pred["parsed_json"])

        out_path = os.path.join(output_dir, f"sig_{i:03d}.json")
        with open(out_path, "w") as f:
            json.dump({
                "image_idx": i,
                "gt_signature_present": all_labels[i],
                "source": "sigdetect" if i < half else "cord",
                "prediction": pred["parsed_json"],
                "json_valid": pred["json_valid"],
            }, f, indent=2, ensure_ascii=False)

        if (i + 1) % 10 == 0 or i == len(all_images) - 1:
            print(f"    Processed {i+1}/{len(all_images)}")

    results = evaluate_signature_batch(predictions, all_labels)

    print(f"\n  Signature Detection Results ({results['n']} images):")
    print(f"    Accuracy:  {results['accuracy']:.3f}")
    print(f"    Precision: {results['precision']:.3f}")
    print(f"    Recall:    {results['recall']:.3f}")
    print(f"    F1:        {results['f1']:.3f}")
    print(f"    (TP={results['tp']}, FP={results['fp']}, FN={results['fn']}, TN={results['tn']})")

    # Save summary
    summary_path = os.path.join(output_dir, "signature_eval_summary.json")
    summary = {k: v for k, v in results.items() if k != "per_sample"}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="VLM PDF Extraction Pipeline")
    parser.add_argument("--input", type=str, help="Path to a PDF or image file")
    parser.add_argument("--dataset", type=str, choices=["cord", "signature"],
                        help="Run on a dataset: 'cord' for receipts, 'signature' for detection eval")
    parser.add_argument("--n", type=int, default=10, help="Number of dataset images to process")
    parser.add_argument("--task", type=str, default=None,
                        choices=["receipt", "signature", "form_fields", "key_value"],
                        help="Single-task mode. Omit for unified full extraction.")
    parser.add_argument("--model", type=str, default="qwen3-vl-2b", help="Model name from registry")
    parser.add_argument("--output", type=str, default="results/json_outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Initializing model: {args.model}")
    extractor = DocumentExtractor(model_name=args.model)

    if args.input:
        run_on_file(extractor, args.input, args.task, args.output)
    elif args.dataset == "cord":
        run_on_cord(extractor, args.n, args.task, args.output)
    elif args.dataset == "signature":
        run_on_signatures(extractor, args.n, args.task, args.output)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m app.run_pipeline --input invoice.pdf")
        print("  python -m app.run_pipeline --dataset cord --n 50 --task receipt")
        print("  python -m app.run_pipeline --dataset signature --n 20")


if __name__ == "__main__":
    main()
