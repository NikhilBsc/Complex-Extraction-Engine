"""
masking/__init__.py
-------------------
Public surface of the masking package.
"""

from masking.context import MaskingContext, MaskEntry, MaskOccurrence
from masking.engine import mask_document, mask_page, MaskedPage
from masking.rehydrator import rehydrate_fields, rehydrate_value

__all__ = [
    # Context
    "MaskingContext",
    "MaskEntry",
    "MaskOccurrence",
    # Engine
    "mask_document",
    "mask_page",
    "MaskedPage",
    # Rehydration
    "rehydrate_fields",
    "rehydrate_value",
]
