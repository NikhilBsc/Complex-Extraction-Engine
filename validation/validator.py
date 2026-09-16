"""
validation/validator.py
-----------------------
Validation orchestrator — Phase 7 core.

Applies all validation checks to extracted fields and produces a final
validation verdict for each field.

Validation pipeline per field
------------------------------
1. Rehydrate mask tokens → real values (MaskingContext)
2. Field quality checks (field_checks.py)
   - Boilerplate field name → REJECT
   - Null/empty value → REJECT
   - Generic field name → FLAG
3. Grounding check (grounding.py)
   - Value found in source text → grounding OK
   - Value not found → REJECT
4. Final verdict assignment
   - All checks passed + grounded → ACCEPTED
   - Quality check failed → REJECTED
   - Grounding failed → REJECTED
   - Flagged but grounded → FLAGGED (valid but review recommended)
   - Confidence below threshold → FLAGGED

Output
------
- Each ExtractedField gets its validation_status updated in place.
- A ValidationReport is returned with counts and rejected/flagged lists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gpt.models import ExtractedField, ExtractionResult
from masking.context import MaskingContext
from masking.rehydrator import rehydrate_value
from validation.grounding import check_grounding
from validation.field_checks import check_field_quality

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Fields below this confidence threshold are flagged for review (not rejected)
FLAG_CONFIDENCE_THRESHOLD: float = 0.80

# Grounding failure → always reject (no exceptions in POC)
REJECT_UNGROUNDED: bool = True


# --------------------------------------------------------------------------- #
# Document context
# --------------------------------------------------------------------------- #

@dataclass
class DocumentContext:
    """
    Provides the source text for grounding validation.

    Built from the masked pages so that grounding works against the
    same text that GPT saw. After rehydration, the values are real —
    so we also need original page texts for cross-checking real values.
    """
    # Masked page texts: segment_id → masked_text
    segment_texts: Dict[str, str]

    # All masked page texts (for cross-segment grounding)
    all_masked_texts: List[str]

    # All original (unmasked) page texts (for exact grounding of rehydrated values)
    all_original_texts: List[str]

    def get_segment_text(self, segment_id: str) -> str:
        return self.segment_texts.get(segment_id, "")


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #

@dataclass
class ValidationReport:
    """Summary of validation results for a full document."""
    total_fields: int = 0
    accepted: int = 0
    rejected: int = 0
    flagged: int = 0
    rejected_fields: List[ExtractedField] = field(default_factory=list)
    flagged_fields: List[ExtractedField] = field(default_factory=list)
    accepted_fields: List[ExtractedField] = field(default_factory=list)

    @property
    def precision_estimate(self) -> float:
        """Fraction of returned fields that were accepted."""
        if self.accepted + self.rejected == 0:
            return 0.0
        return self.accepted / (self.accepted + self.rejected)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of all fields that were accepted."""
        if self.total_fields == 0:
            return 0.0
        return self.accepted / self.total_fields

    def __repr__(self) -> str:
        return (
            f"<ValidationReport total={self.total_fields} "
            f"accepted={self.accepted} "
            f"rejected={self.rejected} "
            f"flagged={self.flagged} "
            f"precision~{self.precision_estimate:.1%}>"
        )


# --------------------------------------------------------------------------- #
# Core validator
# --------------------------------------------------------------------------- #

def validate_field(
    extracted: ExtractedField,
    context: DocumentContext,
    mask_context: MaskingContext,
) -> None:
    """
    Validate a single ExtractedField in place.

    Updates extracted.validation_status and extracted.rejection_reason.

    Parameters
    ----------
    extracted : ExtractedField
        The field to validate. Modified in place.
    context : DocumentContext
        Source texts for grounding.
    mask_context : MaskingContext
        For rehydrating mask tokens in the value before grounding.
    """
    # --- Step 1: Rehydrate the value ---
    real_value = rehydrate_value(extracted.value, mask_context)
    extracted.value = real_value
    extracted.is_rehydrated = True

    # Also rehydrate evidence
    extracted.evidence = rehydrate_value(extracted.evidence, mask_context)

    # --- Step 2: Field quality checks ---
    quality = check_field_quality(extracted.field_name, real_value)

    if not quality.is_valid:
        extracted.validation_status = "rejected"
        extracted.rejection_reason = quality.rejection_reason
        logger.debug(
            "REJECTED (quality): %r = %r — %s",
            extracted.field_name, real_value[:40], quality.rejection_reason,
        )
        return

    # --- Step 3: Grounding check ---
    segment_text = context.get_segment_text(extracted.segment_id)

    grounding = check_grounding(
        value=real_value,
        segment_text=segment_text,
        full_document_texts=context.all_original_texts,
        original_segment_text=segment_text,
    )

    if not grounding.is_grounded and REJECT_UNGROUNDED:
        extracted.validation_status = "rejected"
        extracted.rejection_reason = f"not grounded in source text (value: {real_value[:60]!r})"
        logger.debug(
            "REJECTED (grounding): %r = %r",
            extracted.field_name, real_value[:40],
        )
        return

    # Apply grounding confidence penalty
    adjusted_confidence = extracted.confidence - grounding.confidence_penalty
    extracted.confidence = max(0.0, adjusted_confidence)

    # --- Step 4: Confidence threshold flag ---
    should_flag = quality.should_flag
    flag_reason = quality.flag_reason or ""

    if extracted.confidence < FLAG_CONFIDENCE_THRESHOLD:
        should_flag = True
        flag_reason = (
            flag_reason + " | " if flag_reason else ""
        ) + f"low confidence ({extracted.confidence:.2f})"

    # --- Final verdict ---
    if should_flag:
        extracted.validation_status = "flagged"
        extracted.rejection_reason = flag_reason  # reuse field for flag reason
        logger.debug(
            "FLAGGED: %r = %r — %s",
            extracted.field_name, real_value[:40], flag_reason,
        )
    else:
        extracted.validation_status = "accepted"
        logger.debug(
            "ACCEPTED: %r = %r (conf=%.2f, grounding=%s)",
            extracted.field_name, real_value[:40],
            extracted.confidence, grounding.match_type,
        )


def validate_document(
    results: List[ExtractionResult],
    context: DocumentContext,
    mask_context: MaskingContext,
) -> tuple[List[ExtractedField], ValidationReport]:
    """
    Validate all extracted fields from a full document.

    Parameters
    ----------
    results : List[ExtractionResult]
        Raw GPT extraction results from Phase 6.
    context : DocumentContext
        Source texts for grounding.
    mask_context : MaskingContext
        Masking context for token rehydration.

    Returns
    -------
    (all_fields, report)
        all_fields : All ExtractedField objects with validation_status set.
        report     : ValidationReport with counts and categorised lists.
    """
    report = ValidationReport()
    all_fields: List[ExtractedField] = []

    for result in results:
        for ef in result.fields:
            report.total_fields += 1
            validate_field(ef, context, mask_context)
            all_fields.append(ef)

            if ef.validation_status == "accepted":
                report.accepted += 1
                report.accepted_fields.append(ef)
            elif ef.validation_status == "rejected":
                report.rejected += 1
                report.rejected_fields.append(ef)
            elif ef.validation_status == "flagged":
                report.flagged += 1
                report.flagged_fields.append(ef)

    logger.info(
        "Validation complete: %d total → %d accepted, %d flagged, %d rejected "
        "(precision estimate: %.1f%%)",
        report.total_fields,
        report.accepted,
        report.flagged,
        report.rejected,
        report.precision_estimate * 100,
    )

    return all_fields, report


# --------------------------------------------------------------------------- #
# Context builder helper
# --------------------------------------------------------------------------- #

def build_document_context(
    masked_pages,          # List[MaskedPage]
    preprocessed_pages,    # List[PreprocessedPage]
    segments,              # List[Segment]
) -> DocumentContext:
    """
    Build a DocumentContext from pipeline outputs.

    Parameters
    ----------
    masked_pages : List[MaskedPage]
    preprocessed_pages : List[PreprocessedPage]
    segments : List[Segment]

    Returns
    -------
    DocumentContext
    """
    # Map segment_id → masked text (from masked pages, matched by page)
    # Since segments may span multiple pages, use the first page's masked text
    page_masked: Dict[int, str] = {mp.page_number: mp.masked_text for mp in masked_pages}
    page_original: Dict[int, str] = {pp.page_number: pp.original_text for pp in preprocessed_pages}

    segment_texts: Dict[str, str] = {}
    for seg in segments:
        # Concatenate masked text from all pages in this segment
        seg_text = "\n".join(
            page_masked.get(pn, "") for pn in seg.page_numbers
        )
        segment_texts[seg.segment_id] = seg_text

    all_masked = [page_masked.get(i + 1, "") for i in range(len(masked_pages))]
    all_original = [page_original.get(i + 1, "") for i in range(len(preprocessed_pages))]

    return DocumentContext(
        segment_texts=segment_texts,
        all_masked_texts=all_masked,
        all_original_texts=all_original,
    )
