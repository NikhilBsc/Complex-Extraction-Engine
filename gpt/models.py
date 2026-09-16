"""
gpt/models.py
-------------
Data models for the GPT extraction layer.

ExtractedField is the output contract of Phase 6.
All downstream phases (validation, consolidation, output) consume this type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtractedField:
    """
    A single field-value pair discovered and extracted by GPT.

    Attributes
    ----------
    field_name : str
        The business field label as GPT identified it.
        Examples: "Letter Date", "Proposed Fee", "Payment Term (Duration)"
    value : str
        The extracted value string (may still contain mask tokens before rehydration).
    confidence : float
        GPT's self-reported confidence [0.0 – 1.0].
        Used in validation to decide whether to accept, flag, or reject.
    evidence : str
        The exact text span from the source segment that supports this extraction.
        Used in grounding validation.
    segment_id : str
        Which segment this field was extracted from.
    page_numbers : List[int]
        Pages the source segment spans.
    source_type : str
        "primary" (main extraction) or "fallback" (retry/secondary call).
    raw_gpt_field : str
        The exact field name as returned by GPT (before normalisation).
    is_rehydrated : bool
        True once mask tokens have been replaced with original values.
    validation_status : str
        Set by the validation phase: "accepted" | "rejected" | "flagged" | "pending"
    rejection_reason : Optional[str]
        Populated by validation if the field is rejected.
    """

    field_name: str
    value: str
    confidence: float
    evidence: str
    segment_id: str
    page_numbers: List[int]
    source_type: str = "primary"
    raw_gpt_field: str = ""
    is_rehydrated: bool = False
    validation_status: str = "pending"
    rejection_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.raw_gpt_field:
            self.raw_gpt_field = self.field_name
        # Clamp confidence
        self.confidence = max(0.0, min(1.0, self.confidence))

    def __repr__(self) -> str:
        val_preview = str(self.value)[:40]
        return (
            f"<ExtractedField '{self.field_name}' = '{val_preview}' "
            f"conf={self.confidence:.2f} status={self.validation_status!r}>"
        )

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.85

    @property
    def is_accepted(self) -> bool:
        return self.validation_status == "accepted"


@dataclass
class ExtractionResult:
    """
    Full extraction result for one document segment.
    """
    segment_id: str
    fields: List[ExtractedField]
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    source_type: str = "primary"   # "primary" or "fallback"
    error: Optional[str] = None

    @property
    def field_count(self) -> int:
        return len(self.fields)

    def __repr__(self) -> str:
        return (
            f"<ExtractionResult segment={self.segment_id!r} "
            f"fields={self.field_count} "
            f"tokens={self.total_tokens} "
            f"model={self.model_used!r}>"
        )
