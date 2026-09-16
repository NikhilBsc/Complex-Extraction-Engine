"""
ingestion/quality.py
--------------------
Heuristics that decide whether native-extracted text from a PDF page
is usable or whether we should fall back to OCR.

A page's native text is considered USABLE when:
  1. It contains a minimum number of meaningful characters.
  2. A sufficient fraction of those characters are printable ASCII / common
     Unicode (i.e. not garbage / replacement chars).
  3. The ratio of alphanumeric characters to total characters is above a
     threshold (guards against pages made entirely of symbols/artefacts).

These thresholds are intentionally conservative for a POC and can be
tuned once real documents are available.
"""

import unicodedata

# --------------------------------------------------------------------------- #
# Tuneable thresholds
# --------------------------------------------------------------------------- #
MIN_CHAR_COUNT: int = 20        # Pages with fewer chars are treated as empty
MIN_PRINTABLE_RATIO: float = 0.70  # At least 70 % of chars must be printable
MIN_ALPHA_RATIO: float = 0.40      # At least 40 % of chars must be alphanumeric


def _is_printable(char: str) -> bool:
    """Return True for characters that are visually meaningful."""
    cat = unicodedata.category(char)
    # Accept letters, numbers, punctuation, symbols, and spaces.
    # Reject control characters (Cc), surrogates (Cs), etc.
    return cat not in ("Cc", "Cs", "Co", "Cn")


def evaluate(text: str) -> dict:
    """
    Evaluate the quality of native-extracted text from one PDF page.

    Parameters
    ----------
    text : str
        Raw text string extracted by the native text layer.

    Returns
    -------
    dict with keys:
        usable          – bool   whether native text should be used
        char_count      – int    total character count
        printable_ratio – float  fraction of printable characters
        alpha_ratio     – float  fraction of alphanumeric characters
        reason          – str    human-readable explanation of the decision
    """
    if not text or not text.strip():
        return {
            "usable": False,
            "char_count": 0,
            "printable_ratio": 0.0,
            "alpha_ratio": 0.0,
            "reason": "empty or whitespace-only text",
        }

    chars = list(text)
    total = len(chars)
    printable = sum(1 for c in chars if _is_printable(c))
    alpha = sum(1 for c in chars if c.isalnum())

    printable_ratio = printable / total
    alpha_ratio = alpha / total

    if total < MIN_CHAR_COUNT:
        reason = f"too few characters ({total} < {MIN_CHAR_COUNT})"
        usable = False
    elif printable_ratio < MIN_PRINTABLE_RATIO:
        reason = (
            f"low printable ratio ({printable_ratio:.2f} < {MIN_PRINTABLE_RATIO})"
        )
        usable = False
    elif alpha_ratio < MIN_ALPHA_RATIO:
        reason = (
            f"low alphanumeric ratio ({alpha_ratio:.2f} < {MIN_ALPHA_RATIO})"
        )
        usable = False
    else:
        reason = "native text passes all quality checks"
        usable = True

    return {
        "usable": usable,
        "char_count": total,
        "printable_ratio": round(printable_ratio, 4),
        "alpha_ratio": round(alpha_ratio, 4),
        "reason": reason,
    }
