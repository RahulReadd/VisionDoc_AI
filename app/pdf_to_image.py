"""Module 1: PDF to Image Conversion & Preprocessing.

Converts PDF pages to PIL Images optimized for VLM input.
Handles: multi-page, skewed scans, varied page sizes, scanned & vector PDFs.

Usage:
    from app.pdf_to_image import convert_pdf, load_image
    images, diagnostics = convert_pdf("invoice.pdf")

Dependencies: PyMuPDF (fitz), Pillow, OpenCV, NumPy
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


# VLM vision encoder pixel budget (Qwen3-VL / InternVL default)
DEFAULT_MAX_PIXELS = 512 * 28 * 28  # ~401K


@dataclass
class PageDiagnostic:
    """Diagnostic information for a single PDF page."""
    page_num: int
    width_in: float
    height_in: float
    page_type: str          # "A4/Letter", "receipt", "small-form", "other"
    rotation: int           # PDF metadata rotation flag
    embedded_images: int
    embedded_dpi: int | None
    skew_angle: float       # degrees, positive = clockwise
    optimal_dpi: int
    final_size: tuple[int, int] = (0, 0)


def _classify_page_size(w_in: float, h_in: float) -> str:
    long_side = max(w_in, h_in)
    short_side = min(w_in, h_in)
    if 7.5 < short_side < 9 and 10 < long_side < 12.5:
        return "A4/Letter"
    if short_side < 4.5 and (long_side / short_side if short_side > 0 else 999) > 2:
        return "receipt"
    if short_side < 5:
        return "small-form"
    return "other"


def _choose_dpi(w_in: float, h_in: float,
                embedded_dpi: int | None,
                max_pixels: int = DEFAULT_MAX_PIXELS) -> int:
    """Pick DPI so rendered image is close to max_pixels."""
    if w_in <= 0 or h_in <= 0:
        return 200
    ideal_dpi = int(math.sqrt(max_pixels / (w_in * h_in)))
    dpi = max(100, min(ideal_dpi, 300))
    if embedded_dpi and embedded_dpi < dpi:
        dpi = embedded_dpi
    return dpi


def _detect_skew(img: Image.Image) -> float:
    """Detect skew angle using Hough line detection on text edges."""
    gray = np.array(img.convert("L"))
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=gray.shape[1] // 4, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 30:
            angles.append(angle)
    return round(float(np.median(angles)), 2) if angles else 0.0


def _deskew(img: Image.Image, angle: float, threshold: float = 0.5) -> Image.Image:
    """Rotate image to correct skew. Only applies if |angle| > threshold."""
    if abs(angle) < threshold:
        return img
    return img.rotate(-angle, resample=Image.BICUBIC,
                      expand=True, fillcolor=(255, 255, 255))


def _render_page(page, dpi: int) -> Image.Image:
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _diagnose_page(page, page_num: int, max_pixels: int) -> PageDiagnostic:
    """Build diagnostic for a single page."""
    rect = page.rect
    w_in = rect.width / 72
    h_in = rect.height / 72

    embedded_dpi = None
    images = page.get_images(full=True)
    if images:
        try:
            xref = images[0][0]
            base = page.parent.extract_image(xref)
            if base.get("height", 0) > 0 and h_in > 0:
                embedded_dpi = round(base["height"] / h_in)
        except Exception:
            pass

    optimal_dpi = _choose_dpi(w_in, h_in, embedded_dpi, max_pixels)

    return PageDiagnostic(
        page_num=page_num,
        width_in=round(w_in, 2),
        height_in=round(h_in, 2),
        page_type=_classify_page_size(w_in, h_in),
        rotation=page.rotation,
        embedded_images=len(images),
        embedded_dpi=embedded_dpi,
        skew_angle=0.0,
        optimal_dpi=optimal_dpi,
    )


# ── Public API ────────────────────────────────────────────────────────────

def convert_pdf(
    pdf_path: str | Path,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    deskew: bool = True,
    deskew_threshold: float = 0.5,
) -> tuple[list[Image.Image], list[PageDiagnostic]]:
    """Convert a PDF into VLM-ready images.

    For each page:
      1. Diagnose size, type, embedded DPI
      2. Pick optimal render DPI based on max_pixels budget
      3. Render to PIL Image
      4. Detect and correct skew (if enabled)

    Returns:
        (images, diagnostics) — list of PIL Images + per-page diagnostics.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    images: list[Image.Image] = []
    diagnostics: list[PageDiagnostic] = []

    for i in range(len(doc)):
        page = doc[i]
        diag = _diagnose_page(page, page_num=i + 1, max_pixels=max_pixels)
        img = _render_page(page, dpi=diag.optimal_dpi)

        if deskew:
            diag.skew_angle = _detect_skew(img)
            img = _deskew(img, diag.skew_angle, threshold=deskew_threshold)

        diag.final_size = img.size
        images.append(img)
        diagnostics.append(diag)

    doc.close()
    return images, diagnostics


def load_image(image_path: str | Path) -> Image.Image:
    """Load a single image file (PNG, JPG, etc.) as PIL RGB Image."""
    return Image.open(str(image_path)).convert("RGB")


def print_diagnostics(diagnostics: list[PageDiagnostic]) -> None:
    """Pretty-print diagnostics for all pages."""
    for d in diagnostics:
        px = d.final_size[0] * d.final_size[1]
        print(f"  Page {d.page_num}: {d.width_in}\"x{d.height_in}\" ({d.page_type}) "
              f"| DPI={d.optimal_dpi} | skew={d.skew_angle} deg "
              f"| output={d.final_size[0]}x{d.final_size[1]} ({px/1e6:.1f}M px)"
              + (f" | native DPI={d.embedded_dpi}" if d.embedded_dpi else ""))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m app.pdf_to_image <path.pdf>")
        sys.exit(1)

    imgs, diags = convert_pdf(sys.argv[1])
    print_diagnostics(diags)
    for i, img in enumerate(imgs):
        out = f"page_{i+1}.png"
        img.save(out)
        print(f"  Saved {out}")
