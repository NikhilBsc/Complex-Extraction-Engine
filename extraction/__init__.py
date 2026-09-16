"""
extraction/__init__.py
----------------------
Public surface of the extraction package.
"""

from extraction.segment import Segment, estimate_tokens
from extraction.segmenter import DocumentSegmenter, segment_document, MAX_TOKENS_PER_SEGMENT

__all__ = [
    "Segment",
    "estimate_tokens",
    "DocumentSegmenter",
    "segment_document",
    "MAX_TOKENS_PER_SEGMENT",
]
