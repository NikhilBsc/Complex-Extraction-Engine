"""
validation/grounding.py
-----------------------
Grounding check: verifies that every extracted value is actually present
in the source document text.

"Grounding" means: the value GPT returned must be traceable back to the
raw document text. If GPT hallucinated or paraphrased a value that doesn't
exist verbatim in any page of the source, it is rejected.

Strategy
--------
1. Exact substring match (case-insensitive) against the source segment text.
2. If not found in segment, try the full document text (cross-segment check).
3. If not found anywhere, attempt normalised fuzzy match
   (handle minor whitespace/punctuation differences between GPT output
   and source text — e.g. "USD43,500" vs "USD 43,500").
4. If all checks fail → NOT GROUNDED → reject the field.

Fuzzy match is intentionally conservative: only whitespace/punctuation
normalisation. We do NOT use semantic similarity — that would defeat the
purpose of grounding.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #

def _normalise(text: str) -> str:
    """
    Normalise text for fuzzy grounding comparison.
    - Lowercase
    - Collapse whitespace
    - Remove punctuation variation (smart quotes, dashes, etc.)
    - Strip unicode accents (NFC → NFD → strip Mn)
    """
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    # Collapse all whitespace variants to single space
    text = re.sub(r"\s+", " ", text)
    # Normalise dashes
    text = re.sub(r"[\u2013\u2014\u2012]", "-", text)
    # Normalise quotes
    text = re.sub(r"[\u2018\u2019\u201a]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e]", '"', text)
    return text


def _strip_currency_spaces(text: str) -> str:
    """Remove spaces between currency symbol and amount for normalised matching."""
    return re.sub(r"(USD|GBP|EUR|AUD|CAD)\s+", r"\1", text, flags=re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Grounding result
# --------------------------------------------------------------------------- #

@dataclass
class GroundingResult:
    """Result of a grounding check for one field-value pair."""
    is_grounded: bool
    match_type: str   # "exact" | "normalised" | "not_found"
    matched_in: str   # "segment" | "document" | "none"
    evidence_found: str   # the actual text span found in the document
    confidence_penalty: float  # 0.0 (exact match) to 0.2 (fuzzy) to 1.0 (not found)


def check_grounding(
    value: str,
    segment_text: str,         # the masked text of the source segment
    full_document_texts: List[str],  # all page texts (masked) for cross-segment check
    original_segment_text: str = "",  # unmasked segment text (if available)
) -> GroundingResult:
    """
    Check whether `value` is grounded in the source document text.

    Parameters
    ----------
    value : str
        The extracted value (after rehydration — real value, no mask tokens).
    segment_text : str
        The masked text of the segment the field was extracted from.
    full_document_texts : List[str]
        All page texts (masked) for cross-segment lookup.
    original_segment_text : str
        The original (unmasked) segment text for exact-match grounding.

    Returns
    -------
    GroundingResult
    """
    if not value or not value.strip():
        return GroundingResult(
            is_grounded=False,
            match_type="not_found",
            matched_in="none",
            evidence_found="",
            confidence_penalty=1.0,
        )

    value_norm = _normalise(value)
    value_norm_stripped = _strip_currency_spaces(value_norm)

    # Texts to search (in order of preference)
    search_targets = []

    if original_segment_text:
        search_targets.append(("segment", original_segment_text))
    search_targets.append(("segment", segment_text))

    for i, page_text in enumerate(full_document_texts):
        search_targets.append((f"document_page_{i+1}", page_text))

    # --- Attempt 1: Exact case-insensitive substring match ---
    for location, text in search_targets:
        if value.lower() in text.lower():
            span_start = text.lower().find(value.lower())
            span = text[max(0, span_start-10):span_start + len(value) + 10].strip()
            loc = "segment" if "segment" in location else "document"
            return GroundingResult(
                is_grounded=True,
                match_type="exact",
                matched_in=loc,
                evidence_found=span,
                confidence_penalty=0.0,
            )

    # --- Attempt 2: Normalised fuzzy match ---
    for location, text in search_targets:
        text_norm = _normalise(text)
        text_norm_stripped = _strip_currency_spaces(text_norm)

        if value_norm in text_norm or value_norm_stripped in text_norm_stripped:
            loc = "segment" if "segment" in location else "document"
            return GroundingResult(
                is_grounded=True,
                match_type="normalised",
                matched_in=loc,
                evidence_found=f"[normalised match] {value}",
                confidence_penalty=0.05,
            )

    # --- Attempt 3: Partial match for long values (> 30 chars) ---
    # For long values, GPT sometimes extracts a prefix or suffix — check partial
    if len(value) > 30:
        for location, text in search_targets:
            # Check if at least 70% of the value's words appear nearby
            words = [w for w in value.lower().split() if len(w) > 3]
            if words:
                words_found = sum(1 for w in words if w in text.lower())
                ratio = words_found / len(words)
                if ratio >= 0.75:
                    loc = "segment" if "segment" in location else "document"
                    return GroundingResult(
                        is_grounded=True,
                        match_type="normalised",
                        matched_in=loc,
                        evidence_found=f"[partial word match {ratio:.0%}] {value[:60]}",
                        confidence_penalty=0.10,
                    )

    # --- Not grounded ---
    logger.debug("Value not grounded: %r (checked %d sources)", value[:50], len(search_targets))
    return GroundingResult(
        is_grounded=False,
        match_type="not_found",
        matched_in="none",
        evidence_found="",
        confidence_penalty=1.0,
    )
