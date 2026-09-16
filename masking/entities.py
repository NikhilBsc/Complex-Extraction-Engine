"""
masking/entities.py
-------------------
Central configuration for which entity types to MASK vs. KEEP.
"""

from __future__ import annotations

# Entity types that WILL be masked by Presidio before text is sent to AI
MASK_ENTITY_TYPES: frozenset[str] = frozenset({
    # Contact, Names & Corporate PII
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "PERSON",        # Contact Person, Signatory, Individual Names
    "ORGANIZATION",  # Client Name, Firm Name, Service Provider Name
    "LOCATION",      # Client Address, Office Address, Street Locations

    # Financial & Government IDs
    "US_SSN",
    "US_ITIN",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",

    # Credit / financial instruments
    "CREDIT_CARD",
    "IBAN_CODE",

    # Medical / government IDs
    "MEDICAL_LICENSE",
    "UK_NHS",
    "AU_TFN",
    "IN_PAN",       # India Permanent Account Number
    "IN_AADHAAR",   # India Aadhaar
})

# Minimum confidence score (0.0 to 1.0)
MIN_SCORE: float = 0.65

# Token format
TOKEN_PREFIX = "["
TOKEN_SUFFIX = "]"


def make_token(entity_type: str, counter: int) -> str:
    return f"{TOKEN_PREFIX}{entity_type}_{counter}{TOKEN_SUFFIX}"


def should_mask(entity_type: str, score: float) -> bool:
    return entity_type in MASK_ENTITY_TYPES and score >= MIN_SCORE
