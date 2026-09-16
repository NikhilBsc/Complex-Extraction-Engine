"""
preprocessing/pipeline.py
--------------------------
Preprocessing pipeline: takes ingestion output and returns preprocessed pages.

Orchestrates:
  1. cleaner.clean()     — fix encoding, OCR noise, hyphens, whitespace
  2. normalizer.normalize_page() — structure detection, header/footer removal

Input:  List[ingestion.PageResult]
Output: List[PreprocessedPage]

PreprocessedPage is the contract between preprocessing and all downstream
layers (masking, segmentation, GPT).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from ingestion.reader import PageResult, SourceType
from preprocessing.cleaner import clean
from preprocessing.normalizer import StructuredPage, normalize_page

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public data contract
# --------------------------------------------------------------------------- #

@dataclass
class PreprocessedPage:
    """
    The output of the preprocessing layer for a single page.

    Downstream layers (masking, segmentation, GPT) should consume this object.
    """
    page_number: int
    source_type: SourceType          # inherited from ingestion ("native"|"ocr"|"empty")
    original_text: str               # raw selected text from ingestion (unchanged)
    cleaned_text: str                # after cleaner.clean()
    structured: StructuredPage       # after normalizer.normalize_page()

    @property
    def text(self) -> str:
        """Convenience: the best available text for downstream use."""
        return self.structured.plain_text or self.cleaned_text

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"<PreprocessedPage page={self.page_number} "
            f"source={self.source_type!r} "
            f"has_kv={self.structured.has_key_value} "
            f"has_table={self.structured.has_table} "
            f"preview={preview!r}>"
        )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def preprocess(pages: List[PageResult]) -> List[PreprocessedPage]:
    """
    Run the full preprocessing pipeline on a list of ingested pages.

    Parameters
    ----------
    pages : List[PageResult]
        Output from ingestion.read_pdf().

    Returns
    -------
    List[PreprocessedPage]
        One PreprocessedPage per input page, in page order.
        Pages with empty selected_text are included but flagged via
        source_type == "empty".
    """
    results: List[PreprocessedPage] = []

    for page in pages:
        pn = page.page_number

        if not page.selected_text.strip():
            logger.warning("Page %d has no usable text — skipping preprocessing.", pn)
            # Still emit the page so downstream knows it exists
            results.append(PreprocessedPage(
                page_number=pn,
                source_type=page.source_type,
                original_text="",
                cleaned_text="",
                structured=StructuredPage(
                    page_number=pn,
                    raw_cleaned_text="",
                    regions=[],
                    plain_text="",
                ),
            ))
            continue

        # Step 1 — Clean
        cleaned = clean(page.selected_text)
        logger.debug("Page %d: cleaned %d→%d chars", pn, len(page.selected_text), len(cleaned))

        # Step 2 — Structural normalization
        structured = normalize_page(pn, cleaned)
        logger.info(
            "Page %d: regions=%d has_kv=%s has_table=%s",
            pn,
            len(structured.regions),
            structured.has_key_value,
            structured.has_table,
        )

        results.append(PreprocessedPage(
            page_number=pn,
            source_type=page.source_type,
            original_text=page.selected_text,
            cleaned_text=cleaned,
            structured=structured,
        ))

    logger.info(
        "Preprocessing complete: %d pages, %d with key-value, %d with tables.",
        len(results),
        sum(1 for p in results if p.structured.has_key_value),
        sum(1 for p in results if p.structured.has_table),
    )
    return results
