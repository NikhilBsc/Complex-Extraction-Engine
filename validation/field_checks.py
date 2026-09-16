"""
validation/field_checks.py
--------------------------
Field name and value quality checks — independent of grounding.

These checks catch issues that grounding alone cannot:
  1. Boilerplate / noise field names (page numbers, copyright lines, etc.)
  2. Suspiciously short or empty values
  3. Values that are obviously wrong type for a well-known field label
  4. Placeholder / null-equivalent values that slipped through parsing
  5. Field names that are too long or clearly not a business field

Design principle: be conservative. Only reject fields with strong evidence
of being invalid. When in doubt, FLAG (not reject) — let the human decide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- #
# Boilerplate / noise patterns
# --------------------------------------------------------------------------- #

# Field names matching these patterns are almost certainly boilerplate
_BOILERPLATE_FIELD_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^page\s*\d+",
        r"^\d+\s*of\s*\d+$",
        r"^copyright",
        r"^©",
        r"^all rights reserved",
        r"^confidential",
        r"^draft",
        r"^version\s*\d",
        r"^document id$",
        r"^ref(erence)?\s*:?\s*$",
        r"^\s*$",
        r"^table of contents",
        r"^appendix\s*[a-z]?$",
        r"^exhibit\s*[a-z0-9]?$",
        r"^schedule\s*[a-z0-9]?$",
        r"^signature",
        r"^date signed$",
        r"^printed name$",
        r"^title$",
    ]
]

# Values that are obviously null/placeholder (should have been filtered by extractor)
_NULL_VALUES = frozenset({
    "n/a", "na", "none", "null", "not applicable", "not specified",
    "unknown", "tbd", "tba", "to be determined", "to be advised",
    "-", "--", "---", "...", ".", "",
})

# Field names that are too generic alone to be meaningful
_OVERLY_GENERIC = frozenset({
    "name", "date", "value", "amount", "number", "id", "type",
    "code", "note", "notes", "comment", "comments", "description",
    "field", "item", "row", "column", "text", "data",
})

# Min/max field name length
MIN_FIELD_NAME_LEN = 2
MAX_FIELD_NAME_LEN = 120

# Min value length (single character values are almost always garbage)
MIN_VALUE_LEN = 1

# Max value length (very long values may be entire paragraphs — flag, don't reject)
MAX_VALUE_LEN_FLAG = 500


# --------------------------------------------------------------------------- #
# Check result
# --------------------------------------------------------------------------- #

@dataclass
class FieldCheckResult:
    """Result of field quality checks."""
    is_valid: bool
    should_flag: bool         # True if valid but worth reviewing
    rejection_reason: Optional[str]
    flag_reason: Optional[str]


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #

def _check_field_name(field_name: str) -> Optional[str]:
    """Return rejection reason if field name is invalid, else None."""
    if not field_name or not field_name.strip():
        return "empty field name"

    name = field_name.strip()

    if len(name) < MIN_FIELD_NAME_LEN:
        return f"field name too short ({len(name)} chars)"

    if len(name) > MAX_FIELD_NAME_LEN:
        return f"field name too long ({len(name)} chars)"

    for pattern in _BOILERPLATE_FIELD_PATTERNS:
        if pattern.match(name):
            return f"boilerplate field name: {name!r}"

    return None


def _check_value(value: str, field_name: str) -> tuple[Optional[str], Optional[str]]:
    """
    Return (rejection_reason, flag_reason).
    Both can be None if the value is perfectly valid.
    """
    if not value or not value.strip():
        return "empty value", None

    val = value.strip().lower()

    if val in _NULL_VALUES:
        return f"null-equivalent value: {value!r}", None

    if len(value.strip()) < MIN_VALUE_LEN:
        return f"value too short: {value!r}", None

    flag_reason = None
    if len(value) > MAX_VALUE_LEN_FLAG:
        flag_reason = f"value is very long ({len(value)} chars) — may be a paragraph, not a field value"

    return None, flag_reason


def _check_field_value_association(field_name: str, value: str) -> Optional[str]:
    """
    Check obvious field-value type mismatches.
    Only fires for well-known field name patterns where the value is clearly wrong type.
    Conservative — only catches obvious mismatches.
    """
    name_lower = field_name.lower()
    val_lower = value.lower().strip()

    # Date fields should look like dates
    date_keywords = ("date", "effective", "expiry", "expiration", "commencement", "inception")
    if any(kw in name_lower for kw in date_keywords):
        # If value contains no digits and no month name → suspicious
        months = ("jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec")
        has_digits = bool(re.search(r"\d", val_lower))
        has_month = any(m in val_lower for m in months)
        if not has_digits and not has_month and len(val_lower) > 5:
            return None  # Flag but don't reject — field name might be misleading

    return None


# --------------------------------------------------------------------------- #
# Public composite check
# --------------------------------------------------------------------------- #

def check_field_quality(field_name: str, value: str) -> FieldCheckResult:
    """
    Run all field quality checks and return a composite result.

    Parameters
    ----------
    field_name : str
        The extracted field name.
    value : str
        The extracted value (after rehydration).

    Returns
    -------
    FieldCheckResult
    """
    # Field name checks
    name_rejection = _check_field_name(field_name)
    if name_rejection:
        return FieldCheckResult(
            is_valid=False,
            should_flag=False,
            rejection_reason=name_rejection,
            flag_reason=None,
        )

    # Value checks
    val_rejection, val_flag = _check_value(value, field_name)
    if val_rejection:
        return FieldCheckResult(
            is_valid=False,
            should_flag=False,
            rejection_reason=val_rejection,
            flag_reason=None,
        )

    # Association check
    assoc_rejection = _check_field_value_association(field_name, value)
    if assoc_rejection:
        return FieldCheckResult(
            is_valid=False,
            should_flag=False,
            rejection_reason=assoc_rejection,
            flag_reason=None,
        )

    # Generic name flag (valid but low-specificity)
    flag_reason = val_flag
    if field_name.strip().lower() in _OVERLY_GENERIC and not flag_reason:
        flag_reason = f"overly generic field name: {field_name!r} — consider reviewing"

    return FieldCheckResult(
        is_valid=True,
        should_flag=bool(flag_reason),
        rejection_reason=None,
        flag_reason=flag_reason,
    )
