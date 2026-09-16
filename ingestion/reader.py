"""
ingestion/reader.py
-------------------
Hybrid PDF reader – the core of the ingestion layer.

For every page in the document it:
  1. Extracts native text via PyMuPDF.
  2. Evaluates whether that text is usable (via ingestion.quality).
  3. Runs OCR via Tesseract (via ingestion.ocr) regardless of native quality,
     so that both representations are always available for downstream layers.
  4. Selects the best representation (native | ocr) and records the source.
  5. Returns a list of PageResult dataclass instances – one per page.

Design decisions
----------------
- We always run OCR, not just when native text fails. This means downstream
  components can always compare both if they need to.
- The selection logic lives here (not in quality.py) so the policy is in one
  place and easy to change.
- We use a dataclass (not a plain dict) to get type hints and a clean repr.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

try:
    import pymupdf as fitz  # PyMuPDF (pymupdf >= 1.24 uses `pymupdf` namespace)
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required. Install with:  pip install pymupdf"
    ) from exc

from ingestion import ocr as ocr_module
from ingestion import quality as quality_module

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public data contract
# --------------------------------------------------------------------------- #

SourceType = Literal["native", "ocr", "empty"]


@dataclass
class PageResult:
    """Standardised per-page output of the ingestion layer."""

    page_number: int           # 1-indexed
    native_text: str           # Raw text from the PDF text layer (may be "")
    ocr_text: str              # Text produced by Tesseract OCR (may be "")
    selected_text: str         # The text chosen for downstream processing
    source_type: SourceType    # Which source was chosen: "native" | "ocr" | "empty"

    # Quality metadata – useful for debugging / auditing
    quality_report: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = (self.selected_text or "")[:60].replace("\n", " ")
        return (
            f"<PageResult page={self.page_number} "
            f"source={self.source_type!r} "
            f"preview={preview!r}>"
        )


# --------------------------------------------------------------------------- #
# Main reader
# --------------------------------------------------------------------------- #


def read_pdf(
    pdf_path: str | Path,
    *,
    ocr_dpi: int = 300,
    ocr_lang: str = "eng",
    tesseract_config: str = "--oem 3 --psm 6",
    run_ocr: bool = True,
) -> List[PageResult]:
    """
    Open a PDF and return a list of PageResult objects, one per page.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    ocr_dpi : int
        DPI to use when rendering pages for OCR. Default 300.
    ocr_lang : str
        Tesseract language. Default "eng".
    tesseract_config : str
        Tesseract configuration flags.
    run_ocr : bool
        Set to False to skip OCR entirely (useful in tests where
        Tesseract may not be installed). Default True.

    Returns
    -------
    List[PageResult]
        One PageResult per page, in page order.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Opening PDF: %s", pdf_path)

    results: List[PageResult] = []

    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
        logger.info("Total pages: %d", total_pages)

        for page_index in range(total_pages):
            page_number = page_index + 1
            page = doc[page_index]

            # ----------------------------------------------------------------
            # Step 1: Native text extraction
            # ----------------------------------------------------------------
            native_text: str = page.get_text("text") or ""

            # ----------------------------------------------------------------
            # Step 2: Evaluate native text quality
            # ----------------------------------------------------------------
            quality = quality_module.evaluate(native_text)
            logger.debug(
                "Page %d quality: usable=%s, reason=%s",
                page_number,
                quality["usable"],
                quality["reason"],
            )

            # ----------------------------------------------------------------
            # Step 3: OCR (always run so downstream has both options)
            # ----------------------------------------------------------------
            ocr_text: str = ""
            if run_ocr:
                try:
                    ocr_text = ocr_module.ocr_page(
                        page,
                        dpi=ocr_dpi,
                        lang=ocr_lang,
                        tesseract_config=tesseract_config,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "OCR failed on page %d: %s", page_number, exc
                    )
                    ocr_text = ""

            # ----------------------------------------------------------------
            # Step 4: Select best representation
            # ----------------------------------------------------------------
            source_type: SourceType
            selected_text: str

            if quality["usable"]:
                # Native text is clean enough – prefer it (deterministic, fast)
                source_type = "native"
                selected_text = native_text
            elif ocr_text.strip():
                # Native text is unusable but OCR produced something
                source_type = "ocr"
                selected_text = ocr_text
            else:
                # Both sources are empty / failed
                source_type = "empty"
                selected_text = ""

            logger.info(
                "Page %d → source=%s chars=%d",
                page_number,
                source_type,
                len(selected_text),
            )

            results.append(
                PageResult(
                    page_number=page_number,
                    native_text=native_text,
                    ocr_text=ocr_text,
                    selected_text=selected_text,
                    source_type=source_type,
                    quality_report=quality,
                )
            )

    return results
