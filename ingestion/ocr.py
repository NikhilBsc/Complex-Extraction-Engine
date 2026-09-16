"""
ingestion/ocr.py
----------------
Thin wrapper around pytesseract + PyMuPDF that renders a PDF page to a
high-resolution image and returns the OCR'd text.

Responsibilities:
  - Render a single fitz.Page to a PIL Image at configurable DPI.
  - Run Tesseract OCR and return the resulting string.
  - Raise a clear exception if Tesseract is not installed / misconfigured.

Nothing in this module knows about quality evaluation or selection strategy.
"""

from __future__ import annotations

from typing import Optional

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError(
        "PyMuPDF (fitz) is required for OCR rendering. "
        "Install it with:  pip install pymupdf"
    ) from exc

try:
    import pytesseract
    from PIL import Image
    import os
    _TESS_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_TESS_WIN):
        pytesseract.pytesseract.tesseract_cmd = _TESS_WIN
except ImportError as exc:
    raise ImportError(
        "pytesseract and Pillow are required for OCR. "
        "Install them with:  pip install pytesseract pillow"
    ) from exc

# Default render resolution – 300 DPI gives a good balance of quality / speed
DEFAULT_DPI: int = 300


def page_to_image(page: fitz.Page, dpi: int = DEFAULT_DPI) -> Image.Image:
    """
    Render a PyMuPDF page to a PIL Image.

    Parameters
    ----------
    page : fitz.Page
        The page object from an open fitz.Document.
    dpi : int
        Dots-per-inch for rendering. Higher values improve OCR accuracy
        at the cost of speed and memory. Default is 300.

    Returns
    -------
    PIL.Image.Image
        RGB image of the rendered page.
    """
    zoom = dpi / 72  # 72 pt is the default PDF unit
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    # Convert raw pixmap bytes to PIL Image
    img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return img


def ocr_page(
    page: fitz.Page,
    dpi: int = DEFAULT_DPI,
    lang: str = "eng",
    tesseract_config: str = "--oem 3 --psm 6",
) -> str:
    """
    Run Tesseract OCR on a single PDF page and return the extracted text.

    Parameters
    ----------
    page : fitz.Page
        PyMuPDF page to OCR.
    dpi : int
        Render resolution. Default 300.
    lang : str
        Tesseract language code. Default "eng" (English).
    tesseract_config : str
        Tesseract configuration string.
        --oem 3  → use LSTM + legacy engine (best accuracy)
        --psm 6  → assume a single uniform block of text

    Returns
    -------
    str
        OCR-extracted text (may be empty if the page image is blank).
    """
    img = page_to_image(page, dpi=dpi)
    text: str = pytesseract.image_to_string(img, lang=lang, config=tesseract_config)
    return text
