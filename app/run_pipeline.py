"""Module 4: End-to-End VLM PDF Extraction Pipeline.

Orchestrates: pdf_to_image -> extract -> evaluate.

Usage:
    python -m app.run_pipeline --input invoice.pdf --task receipt
    python -m app.run_pipeline --dataset cord --n 50 --task receipt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from app.pdf_to_image import convert_pdf, load_image
from app.extract import DocumentExtractor
from app.evaluate import evaluate_single, parse_cord_ground_truth


def run_on_file(extractor: DocumentExtractor, file_path: str, task: str, output_dir: str) -> list[dict]:
    """Process a single PDF or image file."""
    path = Path(file_path)
    results = []

    if path.suffix.lower() == ".pdf":
        print(f"Converting PDF: {path.name}")
        images, diagnostics = convert_pdf(path)
    else:
        images = [load_image(path)]

    for i, img in enumerate(images):
        print(f"  Page {i+1}/{len(images)}: extracting ({task})...", end=" ")
        pred = extractor.extract(img, task=task)

        result = {
            "source": path.name,
            "page": i,
            "task": task,
            "prediction": pred["parsed_json"],
            "json_valid": pred["json_valid"],
            "raw_output": pred["raw_output"],
        }
        results.append(result)

        out_path = os.path.join(output_dir, f"{path.stem}_p{i}_{task}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        status = "OK" if pred["json_valid"] else "FAIL"
        print(f"[{status}]")

    return results


def run_on_cord(extractor: DocumentExtractor, n: int, task: str, output_dir: str) -> list[dict]:
    """Process N images from the CORD dataset with evaluation."""
    from datasets import load_dataset

    cord = load_dataset("naver-clova-ix/cord-v2", split="test")
    n = min(n, len(cord))
    results = []

    for i in range(n):
        img = cord[i]["image"].convert("RGB")
        gt_raw = cord[i]["ground_truth"]
        gt_parsed = parse_cord_ground_truth(gt_raw)

        pred = extractor.extract(img, task=task)
        scores = evaluate_single(pred["parsed_json"], gt_parsed)

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


def main():
    parser = argparse.ArgumentParser(description="VLM PDF Extraction Pipeline")
    parser.add_argument("--input", type=str, help="Path to a PDF or image file")
    parser.add_argument("--dataset", type=str, choices=["cord"], help="Run on a dataset")
    parser.add_argument("--n", type=int, default=10, help="Number of dataset images to process")
    parser.add_argument("--task", type=str, default="receipt",
                        choices=["receipt", "signature", "form_fields", "key_value"])
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
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m app.run_pipeline --input invoice.pdf --task receipt")
        print("  python -m app.run_pipeline --dataset cord --n 50 --task receipt")


if __name__ == "__main__":
    main()
