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

EXTRACTION_PROMPT = """Analyze this document image thoroughly and extract ALL of the following into a single JSON object.

{
  "key_value_pairs": {
    "description": "Every key-value pair visible in the document",
    "example": {"vendor_name": "...", "date": "...", "invoice_number": "..."}
  },
  "signature": {
    "present": true or false,
    "confidence": "high" or "medium" or "low",
    "location": "description of where on the page, or null"
  },
  "form_fields": [
    {"field_name": "...", "status": "filled" or "empty", "value": "content or null"}
  ],
  "receipt": {
    "menu": [
      {"nm": "item name", "cnt": "quantity", "price": "unit price"}
    ],
    "sub_total": {
      "subtotal_price": "...",
      "tax_price": "...",
      "discount_price": "..."
    },
    "total": {
      "total_price": "...",
      "cashprice": "...",
      "changeprice": "..."
    }
  }
}

Rules:
- Return ONLY valid JSON. No extra text, explanation, or markdown.
- Use null for any field you cannot find in the image.
- For key_value_pairs: include EVERY piece of text information visible as key-value.
- For signature: always report whether a handwritten signature is present or not.
- For form_fields: list ALL visible form fields with their filled/empty status.
- For receipt: include ALL line items. If the document is not a receipt, set receipt to null.
"""

# Keep individual prompts available for backward compatibility and benchmarking
PROMPTS = {
    "receipt": """Analyze this receipt image. Extract all information and return a JSON object with these fields:

{
  "menu": [
    {"nm": "item name", "cnt": "quantity", "price": "unit price"}
  ],
  "sub_total": {
    "subtotal_price": "...",
    "tax_price": "...",
    "discount_price": "..."
  },
  "total": {
    "total_price": "...",
    "cashprice": "...",
    "changeprice": "..."
  }
}

Rules:
- Return ONLY valid JSON, no extra text or explanation.
- Use null for any field you cannot find in the image.
- Include ALL menu items visible on the receipt.
""",

    "signature": """Is there a handwritten signature present on this document? Answer with a JSON object:
{"signature_present": true/false, "confidence": "high/medium/low", "location": "description or null"}

Return ONLY the JSON object.""",

    "form_fields": """Analyze this document image. List ALL form fields visible and indicate whether each is filled or empty.

Return a JSON object:
{
  "fields": [
    {"field_name": "...", "status": "filled" or "empty", "value": "content or null"}
  ]
}

Return ONLY the JSON object, no extra text.""",

    "key_value": """Extract ALL key-value pairs from this document as a JSON object.

Format: {"key1": "value1", "key2": "value2", ...}

Rules:
- Include every piece of text information visible.
- Use descriptive key names.
- Return ONLY the JSON object.""",
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
