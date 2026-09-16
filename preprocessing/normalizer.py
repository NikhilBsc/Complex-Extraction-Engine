"""
preprocessing/normalizer.py
---------------------------
Structural normalization for business document text.

After basic cleaning (cleaner.py), this module understands the *shape* of
the document and normalises it so that GPT receives well-structured input.

Responsibilities:
  - Detect and remove repetitive page headers / footers
  - Detect key-value patterns and preserve their alignment
  - Detect table-like structures and normalise whitespace within them
  - Detect section headings and mark them consistently
  - Produce a per-page StructuredText object with typed regions

Design rule: Every transformation must be reversible / non-destructive to
business content. When uncertain, leave the text as-is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Region types
# --------------------------------------------------------------------------- #

class RegionType(str, Enum):
    HEADING   = "heading"
    KEY_VALUE = "key_value"
    TABLE     = "table"
    PARAGRAPH = "paragraph"
    FOOTER    = "footer"
    HEADER    = "header"
    UNKNOWN   = "unknown"


@dataclass
class TextRegion:
    """A typed block of text within a page."""
    region_type: RegionType
    text: str
    line_start: int   # 0-indexed line number within the page
    line_end: int


@dataclass
class StructuredPage:
    """Normalised, structured representation of one document page."""
    page_number: int
    raw_cleaned_text: str          # text after cleaner.py
    regions: List[TextRegion]      # typed regions detected on this page
    plain_text: str                # flat text with headers/footers stripped
    has_table: bool = False
    has_key_value: bool = False


# --------------------------------------------------------------------------- #
# Detection patterns
# --------------------------------------------------------------------------- #

# Key-value: "Label: Value" or "Label .... Value" or "Label\t Value"
_KV_PATTERN = re.compile(
    r"^(?P<key>[A-Z][A-Za-z0-9 /\-\(\)]{1,60})"  # key part
    r"\s*[:]\s*"                                    # separator
    r"(?P<value>.+)$",                             # value part
    re.MULTILINE,
)

# Table-like: lines that contain 2+ tab-separated or multiple-space-separated columns
_TABLE_LINE_PATTERN = re.compile(r"^.+(\t{1,}|\s{3,}).+$", re.MULTILINE)

# Section heading: short all-caps line OR title-case line ending without punctuation
_HEADING_PATTERN = re.compile(
    r"^(?:[A-Z][A-Z\s\-\/]{3,60}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,8})$",
    re.MULTILINE,
)

# Page number lines: "Page X of Y", "- X -", standalone digits
_PAGE_NUM_PATTERN = re.compile(
    r"^\s*(?:Page\s+\d+\s+of\s+\d+|-\s*\d+\s*-|\d{1,3})\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Likely footer/header: very short line at top or bottom of page with ref codes
_REF_CODE_PATTERN = re.compile(r"^\s*[A-Z0-9]{3,20}[-\/][A-Z0-9]{2,15}\s*$", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Header / footer removal
# --------------------------------------------------------------------------- #

def _strip_page_artifacts(lines: list[str]) -> tuple[list[str], list[str]]:
    """
    Heuristically remove page-number lines and short repetitive header/footer
    lines from the top and bottom of a page.

    Returns (clean_lines, removed_lines).
    """
    removed: list[str] = []
    # Strip from top (up to 3 lines)
    start = 0
    for i in range(min(3, len(lines))):
        line = lines[i].strip()
        if (
            not line
            or _PAGE_NUM_PATTERN.match(line)
            or _REF_CODE_PATTERN.match(line)
            or len(line) < 6  # very short lines at top are usually artefacts
        ):
            removed.append(lines[i])
            start = i + 1
        else:
            break

    # Strip from bottom (up to 3 lines)
    end = len(lines)
    for i in range(len(lines) - 1, max(len(lines) - 4, start - 1), -1):
        line = lines[i].strip()
        if (
            not line
            or _PAGE_NUM_PATTERN.match(line)
            or _REF_CODE_PATTERN.match(line)
            or len(line) < 6
        ):
            removed.append(lines[i])
            end = i
        else:
            break

    return lines[start:end], removed


# --------------------------------------------------------------------------- #
# Region detection
# --------------------------------------------------------------------------- #

def _detect_regions(lines: list[str]) -> list[TextRegion]:
    """
    Scan page lines and classify each group into typed TextRegions.
    Uses a simple rule-based state machine.
    """
    regions: list[TextRegion] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines (they delimit regions, not regions themselves)
        if not stripped:
            i += 1
            continue

        # --- Heading detection ---
        if _HEADING_PATTERN.match(stripped) and len(stripped) < 80:
            regions.append(TextRegion(
                region_type=RegionType.HEADING,
                text=stripped,
                line_start=i,
                line_end=i,
            ))
            i += 1
            continue

        # --- Key-value block detection ---
        # Collect consecutive KV lines into one region
        kv_lines = []
        j = i
        while j < len(lines) and _KV_PATTERN.match(lines[j].strip()):
            kv_lines.append(lines[j].strip())
            j += 1

        if len(kv_lines) >= 1:
            regions.append(TextRegion(
                region_type=RegionType.KEY_VALUE,
                text="\n".join(kv_lines),
                line_start=i,
                line_end=j - 1,
            ))
            i = j
            continue

        # --- Table detection ---
        if _TABLE_LINE_PATTERN.match(stripped):
            table_lines = []
            j = i
            while j < len(lines) and (
                _TABLE_LINE_PATTERN.match(lines[j].strip()) or not lines[j].strip()
            ):
                table_lines.append(lines[j])
                j += 1
            regions.append(TextRegion(
                region_type=RegionType.TABLE,
                text="\n".join(table_lines),
                line_start=i,
                line_end=j - 1,
            ))
            i = j
            continue

        # --- Default: paragraph ---
        para_lines = []
        j = i
        while j < len(lines) and lines[j].strip():
            para_lines.append(lines[j].strip())
            j += 1
        regions.append(TextRegion(
            region_type=RegionType.PARAGRAPH,
            text=" ".join(para_lines),
            line_start=i,
            line_end=j - 1,
        ))
        i = j

    return regions


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def normalize_page(page_number: int, cleaned_text: str) -> StructuredPage:
    """
    Normalize a single cleaned page into a StructuredPage.

    Parameters
    ----------
    page_number : int
        1-indexed page number.
    cleaned_text : str
        Text that has already been processed by cleaner.clean().

    Returns
    -------
    StructuredPage
        Typed, structured representation ready for masking and segmentation.
    """
    lines = cleaned_text.splitlines()
    clean_lines, _removed = _strip_page_artifacts(lines)
    regions = _detect_regions(clean_lines)

    plain_text = "\n".join(
        r.text for r in regions
        if r.region_type not in (RegionType.HEADER, RegionType.FOOTER)
    )

    has_table = any(r.region_type == RegionType.TABLE for r in regions)
    has_kv = any(r.region_type == RegionType.KEY_VALUE for r in regions)

    return StructuredPage(
        page_number=page_number,
        raw_cleaned_text=cleaned_text,
        regions=regions,
        plain_text=plain_text,
        has_table=has_table,
        has_key_value=has_kv,
    )
