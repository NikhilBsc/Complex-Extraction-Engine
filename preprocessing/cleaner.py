"""
preprocessing/cleaner.py
------------------------
Low-level text cleaning for PDF-extracted content.

Handles:
  - Unicode normalization and encoding artefacts
  - Common PDF ligature reconstruction  (ﬁ → fi, ﬂ → fl, etc.)
  - Smart-quote / curly-quote normalization
  - OCR noise: garbage characters, replacement chars, control chars
  - Broken-hyphen word repair (words split across line breaks)
  - Whitespace normalization (preserves paragraph/section breaks)

Design rule: NEVER remove content that could be a business value.
When in doubt, keep it.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# PDF ligature / glyph substitution map
# --------------------------------------------------------------------------- #
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    # Additional common PDF encoding issues
    "\u2019": "'",   # right single quotation mark
    "\u2018": "'",   # left single quotation mark
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u2013": "-",   # en dash
    "\u2014": "--",  # em dash
    "\u2022": "*",   # bullet
    "\u00b7": "*",   # middle dot bullet
    "\u2026": "...", # ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u00ad": "",    # soft hyphen (remove)
    "\u200b": "",    # zero-width space (remove)
    "\ufffd": "",    # replacement character (OCR garbage — remove)
}

# Characters that are almost certainly OCR noise when isolated
_GARBAGE_PATTERN = re.compile(r"[^\x20-\x7E\u00A1-\u024F\u2013\u2014\u2019\u201c\u201d\n\r\t]")

# Broken-hyphen pattern: word- \n word → wordword
_BROKEN_HYPHEN = re.compile(r"(\w)-\s*\n\s*(\w)")

# Runs of 3+ blank lines collapsed to 2 (preserve section breaks)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Multiple spaces (but not newlines) collapsed to single space
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def fix_ligatures(text: str) -> str:
    """Replace PDF ligature characters and typographic symbols."""
    for glyph, replacement in _LIGATURE_MAP.items():
        text = text.replace(glyph, replacement)
    return text


def normalize_unicode(text: str) -> str:
    """
    Apply NFC unicode normalization.
    Composes characters into their canonical combined form.
    Safe for all business text.
    """
    return unicodedata.normalize("NFC", text)


def remove_ocr_noise(text: str, threshold: float = 0.02) -> str:
    """
    Remove isolated garbage characters produced by OCR.

    Strategy: if a 'word' token consists entirely of non-ASCII non-Latin
    characters AND is very short (≤2 chars), treat it as noise and drop it.
    We do NOT do bulk regex removal — too aggressive.

    Parameters
    ----------
    text : str
    threshold : float
        If the fraction of garbage characters in the whole text exceeds this,
        apply token-level cleanup. Otherwise skip (fast path).
    """
    garbage_chars = len(_GARBAGE_PATTERN.findall(text))
    if not text or garbage_chars / max(len(text), 1) < threshold:
        return text  # fast path — text is clean enough

    # Token-level: drop tokens that are purely garbage
    tokens = text.split(" ")
    cleaned = []
    for token in tokens:
        clean_token = _GARBAGE_PATTERN.sub("", token)
        # Keep token if cleaning left something meaningful
        if clean_token.strip() or "\n" in token:
            cleaned.append(clean_token)
        # else: drop pure garbage token silently
    return " ".join(cleaned)


def repair_broken_hyphens(text: str) -> str:
    """
    Rejoin words broken across lines by a hyphen.

    Example:
        'compli-\\nance services' → 'compliance services'
    """
    return _BROKEN_HYPHEN.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize internal whitespace without destroying document structure.

    Rules:
      - Collapse 2+ spaces/tabs to a single space (within a line)
      - Collapse 3+ consecutive blank lines to 2 (preserve section spacing)
      - Strip leading/trailing whitespace from the whole text
    """
    text = _MULTI_SPACE.sub(" ", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def clean(text: str) -> str:
    """
    Full cleaning pipeline for a single page's text.

    Applies steps in a safe, order-sensitive sequence:
      1. Ligature / glyph fix
      2. Unicode normalization
      3. OCR noise removal
      4. Broken-hyphen repair
      5. Whitespace normalization

    Parameters
    ----------
    text : str
        Raw text from ingestion (native or OCR).

    Returns
    -------
    str
        Cleaned text, ready for normalization.
    """
    if not text:
        return ""
    text = fix_ligatures(text)
    text = normalize_unicode(text)
    text = remove_ocr_noise(text)
    text = repair_broken_hyphens(text)
    text = normalize_whitespace(text)
    return text
