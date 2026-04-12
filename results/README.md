# Results

This directory is populated by running the pipeline.

## Structure

```
results/
  sample_inputs/    — Sample PDFs/images used for testing
  json_outputs/     — JSON extraction outputs (at least 10 documents)
  qualitative/      — Side-by-side comparison images (input + extracted JSON)
  eval_summary.csv  — Aggregate evaluation metrics
```

## How to generate

```bash
# Run on a single PDF
python -m app.run_pipeline --input sample.pdf --task receipt --output results/json_outputs

# Run on CORD dataset (10 images with evaluation)
python -m app.run_pipeline --dataset cord --n 10 --task receipt --output results/json_outputs
```
