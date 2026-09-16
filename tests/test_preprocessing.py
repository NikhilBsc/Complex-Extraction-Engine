"""
tests/test_preprocessing.py
----------------------------
Smoke-test for the preprocessing layer.

Usage (from project root):
    python -m tests.test_preprocessing
    python -m tests.test_preprocessing path/to/your.pdf

Tests:
  1. cleaner.clean() handles ligatures, hyphens, whitespace correctly.
  2. normalizer.normalize_page() detects KV regions from real business text.
  3. Full pipeline: ingestion → preprocessing on a real or synthetic PDF.
"""

from __future__ import annotations

import sys
import io
import logging
from pathlib import Path

# UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.cleaner import clean
from preprocessing.normalizer import normalize_page, RegionType
from preprocessing.pipeline import preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {message}")
    print(f"  [PASS] {message}")


# --------------------------------------------------------------------------- #
# 1. Unit test: cleaner
# --------------------------------------------------------------------------- #

def test_cleaner() -> None:
    print("\n" + "=" * 60)
    print("1. Cleaner unit tests")
    print("=" * 60)

    # Ligatures
    result = clean("The \ufb01rm provides \ufb02exible services.")
    _assert("fi" in result and "fl" in result, "Ligatures fi/fl resolved")

    # Smart quotes
    result = clean("\u201cHello\u201d and \u2018world\u2019")
    _assert('"Hello"' in result, "Smart double quotes normalized")

    # Broken hyphen
    result = clean("infor-" + "\n" + "mation services")
    _assert("information" in result, "Broken hyphen repaired")

    # Whitespace
    result = clean("Hello    World\n\n\n\nNext section")
    _assert("Hello World" in result, "Multiple spaces collapsed")
    _assert(result.count("\n") <= 2, "Excess blank lines collapsed")

    # OCR noise: replacement chars removed
    result = clean("Invoice\ufffdNumber: INV-001")
    _assert("\ufffd" not in result, "Replacement character removed")

    # Empty input
    result = clean("")
    _assert(result == "", "Empty input returns empty string")


# --------------------------------------------------------------------------- #
# 2. Unit test: normalizer
# --------------------------------------------------------------------------- #

def test_normalizer() -> None:
    print("\n" + "=" * 60)
    print("2. Normalizer unit tests")
    print("=" * 60)

    # Key-value detection
    kv_text = (
        "Letter Date: January 22, 2026\n"
        "Client Name: Foundry Associates REIT LLC\n"
        "Client Address: 1270 Avenue of the Americas, New York\n"
        "Service Type: U.S. tax compliance services\n"
        "Proposed Fee: USD 43,500\n"
        "Payment Term (Duration): 10 working days\n"
    )
    page = normalize_page(1, kv_text)
    _assert(page.has_key_value, "Key-value region detected")
    kv_regions = [r for r in page.regions if r.region_type == RegionType.KEY_VALUE]
    _assert(len(kv_regions) >= 1, "At least one KV region found")
    _assert("Letter Date" in page.plain_text, "Letter Date preserved in plain_text")
    _assert("USD 43,500" in page.plain_text, "Fee value preserved")

    # Section heading detection
    heading_text = "PAYMENT TERMS AND CONDITIONS\n\nThis section describes the terms.\n"
    page2 = normalize_page(2, heading_text)
    heading_regions = [r for r in page2.regions if r.region_type == RegionType.HEADING]
    _assert(len(heading_regions) >= 1, "Heading region detected")

    # Empty page handling
    page3 = normalize_page(3, "")
    _assert(page3.plain_text == "", "Empty page returns empty plain_text")
    _assert(len(page3.regions) == 0, "Empty page has no regions")


# --------------------------------------------------------------------------- #
# 3. Integration test: full pipeline on PDF
# --------------------------------------------------------------------------- #

def test_pipeline_on_pdf(pdf_path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"3. Full pipeline: ingestion → preprocessing on {pdf_path.name}")
    print("=" * 60)

    from ingestion import read_pdf

    pages = read_pdf(pdf_path, run_ocr=False)
    preprocessed = preprocess(pages)

    _assert(len(preprocessed) == len(pages), "One PreprocessedPage per ingested page")

    total_kv = sum(1 for p in preprocessed if p.structured.has_key_value)
    total_table = sum(1 for p in preprocessed if p.structured.has_table)

    print(f"\n  Pages processed : {len(preprocessed)}")
    print(f"  Pages with KV   : {total_kv}")
    print(f"  Pages with table: {total_table}")

    print("\n  --- Per-page summary ---")
    for p in preprocessed:
        regions_summary = ", ".join(
            f"{r.region_type.value}({len(r.text)}ch)"
            for r in p.structured.regions[:5]
        )
        preview = p.text[:80].replace("\n", " ")
        print(f"  Page {p.page_number:>2} | src={p.source_type:<6} | regions: {regions_summary}")
        print(f"           | preview: {preview!r}")

    for p in preprocessed:
        _assert(p.page_number >= 1, f"Page {p.page_number}: valid page number")
        _assert(p.original_text is not None, f"Page {p.page_number}: original_text not None")
        _assert(p.cleaned_text is not None, f"Page {p.page_number}: cleaned_text not None")
        _assert(p.structured is not None, f"Page {p.page_number}: structured not None")


def test_pipeline_synthetic() -> None:
    """Run preprocessing on a synthetic in-memory page (no PDF needed)."""
    print("\n" + "=" * 60)
    print("3. Synthetic pipeline test (no PDF required)")
    print("=" * 60)

    from ingestion.reader import PageResult

    fake_page = PageResult(
        page_number=1,
        native_text=(
            "ENGAGEMENT LETTER\n\n"
            "Letter Date: January 22, 2026\n"
            "Client Name: Foundry Associates REIT LLC\n"
            "Client Address: 1270 Avenue of the Americas, STE 1907, New York\n"
            "Service Type: U.S. tax compliance services\n"
            "Proposed Fee (Tax Compliance Services): USD 43,500 (cap)\n"
            "Payment Term (Duration): 10 working days\n\n"
            "This engagement letter sets out the terms under which KPMG will "
            "provide tax compliance services to the client.\n\n"
            "Page 1 of 5\n"
        ),
        ocr_text="",
        selected_text=(
            "ENGAGEMENT LETTER\n\n"
            "Letter Date: January 22, 2026\n"
            "Client Name: Foundry Associates REIT LLC\n"
            "Client Address: 1270 Avenue of the Americas, STE 1907, New York\n"
            "Service Type: U.S. tax compliance services\n"
            "Proposed Fee (Tax Compliance Services): USD 43,500 (cap)\n"
            "Payment Term (Duration): 10 working days\n\n"
            "This engagement letter sets out the terms under which KPMG will "
            "provide tax compliance services to the client.\n\n"
            "Page 1 of 5\n"
        ),
        source_type="native",
        quality_report={"usable": True},
    )

    preprocessed = preprocess([fake_page])

    _assert(len(preprocessed) == 1, "One PreprocessedPage returned")
    p = preprocessed[0]
    _assert(p.page_number == 1, "Page number correct")
    _assert(p.structured.has_key_value, "KV detected in synthetic business content")
    _assert("Foundry Associates REIT LLC" in p.text, "Client name preserved")
    _assert("USD 43,500" in p.text, "Fee value preserved")
    _assert("Page 1 of 5" not in p.text, "Page number footer stripped")

    print(f"\n  Regions detected: {len(p.structured.regions)}")
    for r in p.structured.regions:
        print(f"    [{r.region_type.value:12s}] {r.text[:80]!r}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_cleaner()
    test_normalizer()

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"Error: file not found → {pdf_path}", file=sys.stderr)
            sys.exit(1)
        test_pipeline_on_pdf(pdf_path)
    else:
        test_pipeline_synthetic()

    print("\n" + "=" * 60)
    print("All preprocessing tests passed.")
    print("=" * 60)
