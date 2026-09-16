"""
extraction/segmenter.py
-----------------------
Logical document segmentation — Phase 5.

Converts the flat list of masked pages into a structured list of Segments
that are:
  1. Semantically coherent  — KV blocks stay together, tables stay together,
     paragraphs are grouped under their section heading.
  2. Token-budget-aware     — no single segment exceeds MAX_TOKENS_PER_SEGMENT
     (prevents GPT context overflow on large documents).
  3. Context-preserving     — each segment carries a section_header so GPT
     understands where in the document the content comes from.
  4. Ordered and numbered   — segment_id is sequential, preserving document order.

Input:
  - List[MaskedPage]        — from masking.engine
  - List[PreprocessedPage]  — from preprocessing.pipeline (for structural metadata)

Output:
  - List[Segment]           — ready to be sent to GPT one at a time

Algorithm
---------
For each page (in order):
  1. Pair MaskedPage.masked_text with PreprocessedPage.structured.regions.
  2. Re-apply region boundaries to the masked text (masking preserves structure).
  3. Walk regions:
       HEADING   → update current_section_header; flush current buffer if non-empty
       KEY_VALUE → append to KV buffer
       TABLE     → flush KV buffer; emit table segment; continue
       PARAGRAPH → append to paragraph buffer
  4. At each flush point or when token budget exceeded: emit a Segment.
  5. At end of document: flush any remaining buffers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from extraction.segment import Segment, estimate_tokens
from preprocessing.normalizer import RegionType

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Maximum tokens per segment sent to GPT.
# GPT-4o: 128k context. We keep segments small for precision + parallelism.
MAX_TOKENS_PER_SEGMENT: int = 3000

# Minimum tokens before we emit a segment (avoid trivially small segments)
MIN_TOKENS_PER_SEGMENT: int = 50


# --------------------------------------------------------------------------- #
# Internal buffer
# --------------------------------------------------------------------------- #

@dataclass
class _Buffer:
    """Accumulates text regions before emitting a Segment."""
    lines: List[str] = field(default_factory=list)
    region_types: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    section_header: str = ""
    has_kv: bool = False
    has_table: bool = False

    def add(self, text: str, region_type: str, page_number: int) -> None:
        self.lines.append(text)
        if region_type not in self.region_types:
            self.region_types.append(region_type)
        if page_number not in self.page_numbers:
            self.page_numbers.append(page_number)
        if region_type == RegionType.KEY_VALUE:
            self.has_kv = True
        if region_type == RegionType.TABLE:
            self.has_table = True

    def text(self) -> str:
        return "\n\n".join(self.lines)

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.text())

    def is_empty(self) -> bool:
        return not self.lines

    def clear(self) -> None:
        self.lines.clear()
        self.region_types.clear()
        self.page_numbers.clear()
        self.has_kv = False
        self.has_table = False
        # Note: section_header is NOT cleared — it persists across segments


# --------------------------------------------------------------------------- #
# Region extraction from masked text
# --------------------------------------------------------------------------- #

def _extract_masked_regions(
    masked_text: str,
    regions,   # List[TextRegion] from PreprocessedPage
) -> List[Tuple[str, str]]:
    """
    Map structural regions from preprocessing onto the masked text.

    Strategy: use the region text as a search key against the masked text.
    Masking only replaces value spans; the surrounding label/key text is
    unchanged, so substring matching works reliably.

    Falls back to the entire masked text as a single PARAGRAPH if matching fails.

    Returns: List of (region_text, region_type_value) tuples.
    """
    result = []

    for region in regions:
        region_type = (region.region_type.value
                       if hasattr(region.region_type, 'value')
                       else str(region.region_type))

        # For HEADING: the text is the heading itself — check if it's in masked text
        if region_type == "heading":
            if region.text.strip() in masked_text:
                result.append((region.text.strip(), region_type))
            continue

        # For other regions: try to find the first line of the region in the masked text
        # and extract a block of similar size
        first_line = region.text.strip().splitlines()[0].strip() if region.text.strip() else ""
        if not first_line:
            continue

        # Key-value regions: use the full masked text lines that look like KV pairs
        if region_type == "key_value":
            # Extract lines matching KV pattern from masked text
            from preprocessing.normalizer import _KV_PATTERN
            kv_lines = [
                line.strip() for line in masked_text.splitlines()
                if _KV_PATTERN.match(line.strip())
            ]
            if kv_lines:
                # De-duplicate and preserve order
                seen = set()
                unique_kv = []
                for line in kv_lines:
                    if line not in seen:
                        seen.add(line)
                        unique_kv.append(line)
                # Only add once (the segmenter will add it once per region)
                kv_text = "\n".join(unique_kv)
                if kv_text and kv_text not in [r[0] for r in result]:
                    result.append((kv_text, region_type))
            continue

        # Paragraphs and tables: use the original region text if it appears in masked text
        # (masking doesn't change structure, just replaces specific value tokens)
        first_50 = first_line[:50]
        if first_50 in masked_text:
            result.append((region.text.strip(), region_type))
        else:
            # Fallback: add it anyway — don't drop content
            result.append((region.text.strip(), region_type))

    return result


# --------------------------------------------------------------------------- #
# Main segmenter
# --------------------------------------------------------------------------- #

class DocumentSegmenter:
    """
    Converts masked pages + preprocessing structure into a list of Segments.

    Usage:
        segmenter = DocumentSegmenter()
        segments = segmenter.segment(masked_pages, preprocessed_pages)
    """

    def __init__(
        self,
        max_tokens: int = MAX_TOKENS_PER_SEGMENT,
        min_tokens: int = MIN_TOKENS_PER_SEGMENT,
    ) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self._counter: int = 0
        self._buffer: _Buffer = _Buffer()

    def _next_id(self) -> str:
        self._counter += 1
        return f"seg_{self._counter:03d}"

    def _flush(self) -> Optional[Segment]:
        """Emit current buffer as a Segment, then clear the buffer."""
        if self._buffer.is_empty():
            return None

        text = self._buffer.text()
        tokens = estimate_tokens(text)

        if tokens < self.min_tokens:
            logger.debug("Buffer too small (%d tokens) — not emitting.", tokens)
            return None

        seg = Segment(
            segment_id=self._next_id(),
            text=text,
            page_numbers=list(self._buffer.page_numbers),
            region_types=list(self._buffer.region_types),
            estimated_tokens=tokens,
            section_header=self._buffer.section_header,
            is_key_value_block=self._buffer.has_kv and not self._buffer.has_table,
            is_table=self._buffer.has_table,
            is_mixed=self._buffer.has_kv and bool(
                [r for r in self._buffer.region_types
                 if r not in (RegionType.KEY_VALUE, RegionType.HEADING)]
            ),
        )
        self._buffer.clear()
        return seg

    def _force_flush_if_over_budget(self, segments: List[Segment]) -> None:
        """If buffer is over token budget, flush it now."""
        if self._buffer.estimated_tokens() >= self.max_tokens:
            seg = self._flush()
            if seg:
                segments.append(seg)
                logger.debug("Budget flush: emitted %s (%d tokens)", seg.segment_id, seg.estimated_tokens)

    def segment(
        self,
        masked_pages,    # List[MaskedPage]
        preprocessed_pages,  # List[PreprocessedPage]
    ) -> List[Segment]:
        """
        Segment an entire document.

        Parameters
        ----------
        masked_pages : List[MaskedPage]
        preprocessed_pages : List[PreprocessedPage]

        Returns
        -------
        List[Segment]
        """
        # Reset state
        self._counter = 0
        self._buffer = _Buffer()
        segments: List[Segment] = []

        # Build a page_number → PreprocessedPage lookup
        pp_map = {pp.page_number: pp for pp in preprocessed_pages}

        for masked_page in masked_pages:
            pn = masked_page.page_number
            masked_text = masked_page.masked_text

            if not masked_text.strip():
                logger.debug("Page %d: empty — skipping.", pn)
                continue

            # Get structural regions for this page
            pp = pp_map.get(pn)
            if pp and pp.structured.regions:
                region_pairs = _extract_masked_regions(masked_text, pp.structured.regions)
            else:
                # No structural info — treat entire page as one paragraph
                region_pairs = [(masked_text, RegionType.PARAGRAPH.value)]

            for region_text, region_type in region_pairs:
                if not region_text.strip():
                    continue

                # --- HEADING: flush current buffer, update section context ---
                if region_type == RegionType.HEADING.value:
                    seg = self._flush()
                    if seg:
                        segments.append(seg)
                    self._buffer.section_header = region_text
                    # Don't add headings to buffer as standalone segments;
                    # they'll prefix the next content region
                    continue

                # --- TABLE: flush current buffer, emit table as its own segment ---
                if region_type == RegionType.TABLE.value:
                    seg = self._flush()
                    if seg:
                        segments.append(seg)
                    # Table is its own segment
                    table_seg = Segment(
                        segment_id=self._next_id(),
                        text=region_text,
                        page_numbers=[pn],
                        region_types=[RegionType.TABLE.value],
                        estimated_tokens=estimate_tokens(region_text),
                        section_header=self._buffer.section_header,
                        is_key_value_block=False,
                        is_table=True,
                        is_mixed=False,
                    )
                    segments.append(table_seg)
                    logger.debug("Emitted table segment %s (page %d)", table_seg.segment_id, pn)
                    continue

                # --- KV / PARAGRAPH: accumulate in buffer ---
                # If this single region already exceeds the budget, split it line-by-line
                if estimate_tokens(region_text) >= self.max_tokens:
                    lines = region_text.splitlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        self._buffer.add(line, region_type, pn)
                        self._force_flush_if_over_budget(segments)
                else:
                    self._buffer.add(region_text, region_type, pn)
                    self._force_flush_if_over_budget(segments)

            # End of page — check if we should flush at page boundaries
            # (for large KV blocks that span multiple pages, keep them together
            #  unless they'd exceed the budget)
            if self._buffer.estimated_tokens() >= self.max_tokens:
                seg = self._flush()
                if seg:
                    segments.append(seg)

        # Final flush — emit whatever's left in the buffer
        seg = self._flush()
        if seg:
            segments.append(seg)

        logger.info(
            "Segmentation complete: %d pages → %d segments "
            "(KV: %d, Table: %d, Mixed: %d)",
            len(masked_pages),
            len(segments),
            sum(1 for s in segments if s.is_key_value_block),
            sum(1 for s in segments if s.is_table),
            sum(1 for s in segments if s.is_mixed),
        )

        return segments


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #

def segment_document(
    masked_pages,
    preprocessed_pages,
    max_tokens: int = MAX_TOKENS_PER_SEGMENT,
) -> List[Segment]:
    """
    Convenience wrapper around DocumentSegmenter.

    Parameters
    ----------
    masked_pages : List[MaskedPage]
    preprocessed_pages : List[PreprocessedPage]
    max_tokens : int
        Maximum tokens per segment. Default 3000.

    Returns
    -------
    List[Segment]
    """
    segmenter = DocumentSegmenter(max_tokens=max_tokens)
    return segmenter.segment(masked_pages, preprocessed_pages)
