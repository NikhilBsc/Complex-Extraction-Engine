"""
extraction/segment.py
---------------------
Segment data model — the output contract of the segmentation phase.

A Segment is a self-contained, GPT-ready chunk of document content with
enough metadata for GPT to understand the context and for downstream
validation to trace results back to source pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Segment:
    """
    A logical, GPT-ready chunk of document content.

    One document produces N segments. Each segment is sent to GPT
    as an independent extraction unit, then consolidated downstream.
    """

    segment_id: str              # "seg_001", "seg_002", ...
    text: str                    # masked text content (what GPT sees)
    page_numbers: List[int]      # pages this segment spans (1-indexed)
    region_types: List[str]      # RegionType values present in this segment
    estimated_tokens: int        # rough token estimate (chars / 4)
    section_header: str          # nearest section heading (for GPT context)
    is_key_value_block: bool     # primarily KV content → extraction priority
    is_table: bool               # contains table structure
    is_mixed: bool               # mix of KV + paragraph content

    # Internal: original (unmasked) text — set by engine if needed for validation
    original_text: str = field(default="", repr=False)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"<Segment id={self.segment_id!r} "
            f"pages={self.page_numbers} "
            f"tokens~{self.estimated_tokens} "
            f"kv={self.is_key_value_block} "
            f"preview={preview!r}>"
        )

    @property
    def char_count(self) -> int:
        return len(self.text)


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation: 1 token ≈ 4 characters.
    Good enough for chunking decisions. Not a substitute for a real tokenizer.
    """
    return max(1, len(text) // 4)
