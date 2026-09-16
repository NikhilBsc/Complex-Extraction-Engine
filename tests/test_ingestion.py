"""
tests/test_ingestion.py
-----------------------
Smoke-test / example script for the ingestion layer.

Usage (from the project root):
    python -m tests.test_ingestion path/to/your.pdf

What it verifies
----------------
1. read_pdf() accepts a valid PDF and returns a list of PageResult objects.
2. Every PageResult has the required fields: page_number, native_text,
   ocr_text, selected_text, source_type, quality_report.
3. page_number values are sequential, starting from 1.
4. source_type is always one of the three expected values.
5. selected_text is never None.
6. A summary table is printed to stdout so you can visually inspect the output.

If no PDF path is supplied on the command line, the script creates a tiny
synthetic in-memory PDF with known text, runs the ingestion layer against it
(OCR disabled so no Tesseract dependency is needed), and asserts the result.
"""

from __future__ import annotations

import sys
import io
import logging
from pathlib import Path

# Force UTF-8 output on Windows consoles to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --------------------------------------------------------------------------- #
# Allow running from the project root without installing the package
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion import read_pdf, PageResult  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("test_ingestion")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

VALID_SOURCE_TYPES = {"native", "ocr", "empty"}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {message}")
    print(f"  ✓  {message}")


def _print_summary(pages: list[PageResult]) -> None:
    """Print a compact table of results to stdout."""
    col = {
        "page": 6,
        "source": 8,
        "native_chars": 13,
        "ocr_chars": 10,
        "selected_chars": 15,
        "preview": 50,
    }
    header = (
        f"{'Page':>{col['page']}} | "
        f"{'Source':<{col['source']}} | "
        f"{'Native Chars':>{col['native_chars']}} | "
        f"{'OCR Chars':>{col['ocr_chars']}} | "
        f"{'Selected Chars':>{col['selected_chars']}} | "
        f"Preview"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for p in pages:
        preview = (p.selected_text or "")[:col["preview"]].replace("\n", " ")
        print(
            f"{p.page_number:>{col['page']}} | "
            f"{p.source_type:<{col['source']}} | "
            f"{len(p.native_text):>{col['native_chars']}} | "
            f"{len(p.ocr_text):>{col['ocr_chars']}} | "
            f"{len(p.selected_text):>{col['selected_chars']}} | "
            f"{preview!r}"
        )
    print(sep + "\n")


# --------------------------------------------------------------------------- #
# Synthetic PDF test (no Tesseract required)
# --------------------------------------------------------------------------- #

def _create_synthetic_pdf() -> Path:
    """
    Create a minimal, in-memory PDF with one page of known text and save it
    to a temp file. Returns the path.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF is required. pip install pymupdf")

    tmp_path = ROOT / "tests" / "_synthetic_test.pdf"
    tmp_path.parent.mkdir(exist_ok=True)

    doc = fitz.open()
    page = doc.new_page()
    # Insert text so native extraction finds it
    page.insert_text(
        (72, 100),
        "Invoice Number: INV-2024-0042\n"
        "Date: 2024-03-15\n"
        "Vendor: Acme Corporation\n"
        "Total Amount: $12,500.00\n"
        "Payment Terms: Net 30\n",
        fontsize=12,
    )
    doc.save(str(tmp_path))
    doc.close()
    return tmp_path


def run_synthetic_test() -> None:
    """Run assertions against a synthetic PDF (no real Tesseract needed)."""
    print("=" * 60)
    print("Running synthetic PDF test (OCR disabled)")
    print("=" * 60)

    pdf_path = _create_synthetic_pdf()
    logger.info("Synthetic PDF written to: %s", pdf_path)

    pages = read_pdf(pdf_path, run_ocr=False)

    # Basic structural assertions
    _assert(isinstance(pages, list), "read_pdf returns a list")
    _assert(len(pages) == 1, "synthetic PDF has exactly 1 page")

    p = pages[0]
    _assert(isinstance(p, PageResult), "items are PageResult instances")
    _assert(p.page_number == 1, "page_number starts at 1")
    _assert(p.source_type in VALID_SOURCE_TYPES, f"source_type is valid ({p.source_type})")
    _assert(p.selected_text is not None, "selected_text is not None")
    _assert(p.native_text is not None, "native_text is not None")
    _assert(p.ocr_text is not None, "ocr_text is not None (empty str when OCR disabled)")
    _assert("usable" in p.quality_report, "quality_report contains 'usable' key")
    _assert(p.source_type == "native", "synthetic text-layer PDF selects 'native' source")
    _assert("Invoice Number" in p.selected_text, "known text is present in selected_text")

    _print_summary(pages)
    print("All synthetic assertions passed.\n")


# --------------------------------------------------------------------------- #
# Real PDF test
# --------------------------------------------------------------------------- #

def run_real_pdf_test(pdf_path: Path) -> None:
    """Run the ingestion layer on a real PDF and print the summary table."""
    print("=" * 60)
    print(f"Running ingestion on: {pdf_path.name}")
    print("=" * 60)

    pages = read_pdf(pdf_path, run_ocr=True)

    _assert(isinstance(pages, list), "read_pdf returns a list")
    _assert(len(pages) > 0, "at least one page returned")

    for i, p in enumerate(pages, start=1):
        _assert(isinstance(p, PageResult), f"page {i} is a PageResult")
        _assert(p.page_number == i, f"page_number is sequential ({i})")
        _assert(p.source_type in VALID_SOURCE_TYPES, f"page {i} source_type is valid")
        _assert(p.selected_text is not None, f"page {i} selected_text is not None")

    _print_summary(pages)
    print(f"All assertions passed for {len(pages)}-page document.\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Real PDF supplied via command line
        supplied_path = Path(sys.argv[1])
        if not supplied_path.exists():
            print(f"Error: file not found → {supplied_path}", file=sys.stderr)
            sys.exit(1)
        run_real_pdf_test(supplied_path)
    else:
        # No PDF supplied – use synthetic in-memory document
        run_synthetic_test()
