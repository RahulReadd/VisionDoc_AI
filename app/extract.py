"""Module 2: VLM-Based Information Extraction.

Loads a Vision-Language Model and extracts structured JSON from document images.
A single unified prompt extracts all 4 required aspects from every page:
  1. Key-value pairs
  2. Signature detection
  3. Form field status (filled/empty)
  4. Receipt/financial fields (total, tax, items)

Usage:
    from app.extract import DocumentExtractor
    extractor = DocumentExtractor(model_name="qwen3-vl-2b")
    result = extractor.extract(image)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.models import get_model


# ── Unified extraction prompt ────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a document extraction system. Analyze this document image and return a JSON object with exactly these 4 keys:

1. "key_value_pairs": A flat dictionary of every labeled piece of information. Map each label to its value. For example, if the document says "Date: April 12", output "Date": "April 12". Include names, dates, numbers, addresses, totals, IDs — every piece of data that has a label.

2. "signature": Detect whether a handwritten signature (not printed text, not a logo) is physically present.
   Return: {"present": true/false, "confidence": "high"/"medium"/"low", "location": "bottom right"} or {"present": false, "confidence": "high", "location": null}

3. "form_fields": List input fields that a human is expected to fill in (text boxes, checkboxes, blanks, lines to write on). Table cells and printed data are NOT form fields.
   Return: [{"field_name": "Name", "status": "filled", "value": "John"}] or [] if there are no form fields.

4. "receipt": If this is a receipt or invoice with line items, extract them. If NOT a receipt/invoice, set this to null.
   Return: {"menu": [{"nm": "Coffee", "cnt": "2", "price": "8.00"}], "sub_total": {"subtotal_price": "8.00", "tax_price": "0.40", "discount_price": null}, "total": {"total_price": "8.40", "cashprice": null, "changeprice": null}}

STRICT RULES:
- Return ONLY the JSON object. No text before or after. No markdown fences.
- Use null for missing values. Do NOT copy these instructions into the output.
- key_value_pairs must be a flat dict like {"Date": "April 12", "Total": "$100"}, NOT nested.
- form_fields must be [] (empty list) if there are no fillable form fields.
- receipt must be null if the document is not a receipt or invoice.
"""

# Keep individual prompts available for backward compatibility and benchmarking
PROMPTS = {
    "receipt": """Extract all information from this receipt/invoice image as JSON.

Return exactly this structure (use null for missing fields):
{"menu": [{"nm": "item name", "cnt": "quantity", "price": "unit price"}], "sub_total": {"subtotal_price": "...", "tax_price": "...", "discount_price": null}, "total": {"total_price": "...", "cashprice": null, "changeprice": null}}

Include every line item. Return ONLY the JSON, no other text.""",

    "signature": """Look at this document image carefully. Is there a handwritten signature physically present? A signature is a handwritten cursive/scribble mark, NOT printed text or a typed name.

Return ONLY this JSON:
{"signature_present": true or false, "confidence": "high" or "medium" or "low", "location": "where on the page, or null if not present"}""",

    "form_fields": """Identify all fillable form fields in this document. A form field is an input area meant for a human to write or type into (text boxes, checkboxes, blank lines, dropdown areas). Printed labels and table data are NOT form fields.

Return ONLY this JSON:
{"fields": [{"field_name": "Name", "status": "filled", "value": "John Doe"}]}

If there are no fillable form fields, return: {"fields": []}""",

    "key_value": """Extract every labeled piece of information from this document as a flat JSON dictionary. Map each label/heading to its value.

Example: if the document says "Date: April 12" and "Total: $100", return {"Date": "April 12", "Total": "$100"}.

Include all names, dates, IDs, amounts, addresses, and any other labeled data. Return ONLY the JSON object.""",
}


def parse_json_output(text: str) -> tuple[dict | None, bool]:
    """Attempt to extract JSON from model output. Handles markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1]), True
        except json.JSONDecodeError:
            pass
    return None, False


class DocumentExtractor:
    """High-level extraction interface."""

    def __init__(self, model_name: str = "qwen3-vl-2b", max_new_tokens: int = 2048):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.adapter = get_model(model_name)
        print(f"DocumentExtractor ready: {model_name}")

    def extract(self, image: Image.Image, task: str | None = None) -> dict:
        """Extract structured data from a document image.

        Args:
            image: PIL Image of the document page.
            task: Optional — if None, uses the unified prompt that extracts
                  all 4 aspects at once. If set to a specific task name
                  ('receipt', 'signature', 'form_fields', 'key_value'),
                  uses the single-task prompt instead (useful for benchmarking).

        Returns:
            Dict with 'raw_output', 'parsed_json', 'json_valid' keys.
        """
        if task is not None:
            if task not in PROMPTS:
                raise ValueError(f"Unknown task '{task}'. Available: {list(PROMPTS.keys())}")
            prompt = PROMPTS[task]
        else:
            prompt = EXTRACTION_PROMPT

        raw_output = self.adapter.run_inference(image, prompt, max_new_tokens=self.max_new_tokens)
        parsed, valid = parse_json_output(raw_output)

        return {
            "raw_output": raw_output,
            "parsed_json": parsed,
            "json_valid": valid,
            "task": task or "full",
            "model": self.model_name,
        }

    def extract_batch(self, images: list[Image.Image], task: str | None = None) -> list[dict]:
        """Run extraction on multiple images."""
        return [self.extract(img, task) for img in images]


if __name__ == "__main__":
    from app.pdf_to_image import load_image

    if len(sys.argv) < 2:
        print("Usage: python -m app.extract <image_path> [task]")
        print("  task: omit for full extraction, or one of: receipt, signature, form_fields, key_value")
        sys.exit(1)

    img_path = sys.argv[1]
    task = sys.argv[2] if len(sys.argv) > 2 else None

    extractor = DocumentExtractor()
    img = load_image(img_path)
    result = extractor.extract(img, task=task)

    print(f"\nJSON valid: {result['json_valid']}")
    print(json.dumps(result["parsed_json"], indent=2, ensure_ascii=False))
