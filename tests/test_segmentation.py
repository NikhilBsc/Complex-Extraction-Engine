"""
tests/test_segmentation.py
--------------------------
Tests for the document segmentation layer (Phase 5).

Usage:
    python -m tests.test_segmentation
    python -m tests.test_segmentation path/to/document.pdf

Test groups:
  1. Segment dataclass and estimate_tokens
  2. Segmenter with synthetic structured pages
  3. Token budget enforcement (large documents)
  4. Full pipeline: ingestion → preprocessing → masking → segmentation (real PDF)
"""

from __future__ import annotations

import sys
import io
import logging
import tempfile
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {message}")
    print(f"  [PASS] {message}")


# --------------------------------------------------------------------------- #
# 1. Segment dataclass tests
# --------------------------------------------------------------------------- #

def test_segment_model() -> None:
    print("\n" + "=" * 60)
    print("1. Segment dataclass tests")
    print("=" * 60)

    from extraction.segment import Segment, estimate_tokens

    seg = Segment(
        segment_id="seg_001",
        text="Letter Date: January 22, 2026\nClient Name: Acme Corp",
        page_numbers=[1],
        region_types=["key_value"],
        estimated_tokens=15,
        section_header="ENGAGEMENT LETTER",
        is_key_value_block=True,
        is_table=False,
        is_mixed=False,
    )

    _assert(seg.segment_id == "seg_001", "segment_id set correctly")
    _assert(seg.char_count == len(seg.text), "char_count property works")
    _assert(seg.is_key_value_block, "is_key_value_block flag set")
    _assert(not seg.is_table, "is_table flag unset")

    # Token estimation
    tokens_short = estimate_tokens("Hello world")      # 11 chars → ~2 tokens
    tokens_long = estimate_tokens("A" * 400)           # 400 chars → 100 tokens
    _assert(tokens_short >= 1, "estimate_tokens returns at least 1")
    _assert(tokens_long == 100, "estimate_tokens: 400 chars = 100 tokens")


# --------------------------------------------------------------------------- #
# 2. Segmenter with synthetic pages
# --------------------------------------------------------------------------- #

def _make_synthetic_pipeline():
    """
    Build a minimal ingestion + preprocessing + masking result from a
    synthetic KPMG-style engagement letter — no real PDF needed.
    """
    from ingestion.reader import PageResult
    from preprocessing.pipeline import preprocess
    from masking.engine import mask_document

    page_text = (
        "ENGAGEMENT LETTER\n\n"
        "Letter Date: January 22, 2026\n"
        "Client Name: Foundry Associates REIT LLC\n"
        "Client Address: 1270 Avenue of the Americas, STE 1907, New York\n"
        "Auditor / Firm Name: KPMG Samjong Accounting Corp.\n"
        "KPMG Contact: Sang Bum Oh, +82-2-2112-0721\n"
        "Engagement Lead: Sang Bum Oh\n"
        "Service Type: U.S. tax compliance services\n"
        "Tax Year Covered: 2025\n"
        "Pricing Method: Fixed Fee\n"
        "Proposed Fee (Tax Compliance Services): USD 43,500 (cap)\n"
        "Fee Exclusions: Exclusive of VAT and reasonable out-of-pocket expenses\n"
        "Payment Term (Duration): 10 working days\n"
        "Payment Term Start Description: Runs from the invoice date\n\n"
        "SCOPE OF SERVICES\n\n"
        "This engagement covers U.S. federal and state tax compliance for the "
        "fiscal year 2025. Services include preparation of Form 1120-REIT, "
        "Form 1042, and related schedules as applicable.\n\n"
        "BILLING AND PAYMENT\n\n"
        "Billing / Invoicing Schedule: 50% billed in April 2026, remaining 50% by August 2026\n"
        "Additional Fee Trigger: USD 2,300 per Form 8288-B filed if extra distributions require it\n"
        "Out-of-Scope Pricing: Time and Materials billed at 60% of standard hourly rates\n"
    )

    fake_pages = [
        PageResult(
            page_number=1,
            native_text=page_text,
            ocr_text="",
            selected_text=page_text,
            source_type="native",
            quality_report={"usable": True},
        )
    ]

    preprocessed = preprocess(fake_pages)

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        masked_pages, context = mask_document(
            preprocessed,
            doc_id="synthetic_kpmg",
            save_mask_file=False,
        )

    return masked_pages, preprocessed, context


def test_segmenter_synthetic() -> None:
    print("\n" + "=" * 60)
    print("2. Segmenter with synthetic KPMG-style engagement letter")
    print("=" * 60)

    from extraction.segmenter import segment_document

    masked_pages, preprocessed, _ = _make_synthetic_pipeline()
    segments = segment_document(masked_pages, preprocessed)

    _assert(len(segments) >= 1, "At least one segment produced")

    # All segments have required fields
    for seg in segments:
        _assert(seg.segment_id.startswith("seg_"), f"{seg.segment_id}: ID format correct")
        _assert(len(seg.page_numbers) >= 1, f"{seg.segment_id}: has page numbers")
        _assert(len(seg.text) > 0, f"{seg.segment_id}: non-empty text")
        _assert(seg.estimated_tokens > 0, f"{seg.segment_id}: estimated_tokens > 0")

    # At least one KV segment (engagement letter is KV-heavy)
    kv_segs = [s for s in segments if s.is_key_value_block]
    _assert(len(kv_segs) >= 1, "At least one KV segment detected")

    # Key business values survive into segments
    all_text = "\n".join(s.text for s in segments)
    _assert("Foundry Associates REIT LLC" in all_text, "Client name in segments")
    _assert("USD 43,500" in all_text, "Fee amount in segments")
    _assert("January 22, 2026" in all_text, "Date in segments")
    _assert("10 working days" in all_text, "Payment term in segments")

    # Print segment summary
    print(f"\n  Total segments: {len(segments)}")
    for seg in segments:
        kv_flag = "[KV]" if seg.is_key_value_block else "[TB]" if seg.is_table else "[PR]"
        header = f"'{seg.section_header[:30]}'" if seg.section_header else "no header"
        print(f"  {seg.segment_id} {kv_flag} pages={seg.page_numbers} "
              f"~{seg.estimated_tokens}tok section={header}")
        print(f"    preview: {seg.text[:80].replace(chr(10), ' ')!r}")


# --------------------------------------------------------------------------- #
# 3. Token budget enforcement
# --------------------------------------------------------------------------- #

def test_token_budget() -> None:
    print("\n" + "=" * 60)
    print("3. Token budget enforcement")
    print("=" * 60)

    from extraction.segmenter import DocumentSegmenter
    from masking.engine import MaskedPage

    # Create a page with very large text (exceeds any reasonable budget)
    large_text = "\n".join(
        f"Field {i}: Value for field number {i} with some additional descriptive text"
        for i in range(1, 201)  # 200 KV pairs → ~200 * 70 chars ≈ 14000 chars ≈ 3500 tokens
    )

    from ingestion.reader import PageResult
    from preprocessing.pipeline import preprocess
    from masking.engine import mask_document

    fake_page = PageResult(
        page_number=1,
        native_text=large_text,
        ocr_text="",
        selected_text=large_text,
        source_type="native",
        quality_report={"usable": True},
    )
    preprocessed = preprocess([fake_page])
    masked_pages, _ = mask_document(preprocessed, doc_id="budget_test", save_mask_file=False)

    # Segment with a small budget to force splitting
    segmenter = DocumentSegmenter(max_tokens=500, min_tokens=10)
    segments = segmenter.segment(masked_pages, preprocessed)

    _assert(len(segments) > 1, f"Large doc split into multiple segments (got {len(segments)})")

    for seg in segments:
        _assert(
            seg.estimated_tokens <= 600,  # some slack for last partial segment
            f"{seg.segment_id}: within token budget (~{seg.estimated_tokens} tokens)"
        )

    print(f"\n  Large doc → {len(segments)} segments (budget=500 tokens each)")


# --------------------------------------------------------------------------- #
# 4. Full pipeline on real PDF
# --------------------------------------------------------------------------- #

def test_full_pipeline_pdf(pdf_path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"4. Full pipeline: ingestion→preprocessing→masking→segmentation")
    print(f"   Document: {pdf_path.name}")
    print("=" * 60)

    from ingestion import read_pdf
    from preprocessing import preprocess
    from masking.engine import mask_document
    from extraction.segmenter import segment_document

    pages = read_pdf(pdf_path, run_ocr=False)
    preprocessed = preprocess(pages)
    masked_pages, context = mask_document(
        preprocessed, doc_id=pdf_path.stem, save_mask_file=False
    )
    segments = segment_document(masked_pages, preprocessed)

    _assert(len(segments) >= 1, "At least one segment produced")

    total_tokens = sum(s.estimated_tokens for s in segments)
    kv_count = sum(1 for s in segments if s.is_key_value_block)
    table_count = sum(1 for s in segments if s.is_table)

    print(f"\n  Pages          : {len(pages)}")
    print(f"  Segments       : {len(segments)}")
    print(f"  Total ~tokens  : {total_tokens}")
    print(f"  KV segments    : {kv_count}")
    print(f"  Table segments : {table_count}")
    print(f"  Entities masked: {context.total_masked()}")

    print("\n  --- Segment breakdown ---")
    for seg in segments:
        flag = "[KV]" if seg.is_key_value_block else "[TB]" if seg.is_table else "[PR]"
        print(f"  {seg.segment_id} {flag} pages={seg.page_numbers} ~{seg.estimated_tokens}tok")
        print(f"    {seg.text[:100].replace(chr(10), ' ')!r}")

    for seg in segments:
        _assert(seg.estimated_tokens <= MAX_TOKENS_PER_SEGMENT + 200,
                f"{seg.segment_id}: within token budget")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from extraction.segmenter import MAX_TOKENS_PER_SEGMENT

    test_segment_model()
    test_segmenter_synthetic()
    test_token_budget()

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"Error: not found → {pdf_path}", file=sys.stderr)
            sys.exit(1)
        test_full_pipeline_pdf(pdf_path)

    print("\n" + "=" * 60)
    print("All segmentation tests passed.")
    print("=" * 60)
