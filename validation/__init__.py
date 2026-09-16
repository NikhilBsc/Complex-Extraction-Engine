"""
validation/__init__.py
----------------------
Public surface of the validation package.
"""

from validation.validator import (
    validate_document,
    validate_field,
    build_document_context,
    DocumentContext,
    ValidationReport,
)
from validation.grounding import check_grounding, GroundingResult
from validation.field_checks import check_field_quality, FieldCheckResult

__all__ = [
    "validate_document",
    "validate_field",
    "build_document_context",
    "DocumentContext",
    "ValidationReport",
    "check_grounding",
    "GroundingResult",
    "check_field_quality",
    "FieldCheckResult",
]
