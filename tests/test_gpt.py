"""
tests/test_gpt.py
-----------------
Tests for the GPT extraction layer (Phase 6).

Usage:
    python -m tests.test_gpt                        # mock tests (no API key needed)
    python -m tests.test_gpt --live                 # live tests (requires OPENAI_API_KEY in .env)
    python -m tests.test_gpt --live path/to/doc.pdf # live test on real PDF

Test groups:
  1. Data models (ExtractedField, ExtractionResult)
  2. Prompt builder (correct prompt selected per segment type)
  3. Field parsing (raw GPT JSON → ExtractedField)
  4. Fallback trigger logic
  5. Mock extraction (no API key — validates structure without calling OpenAI)
  6. [LIVE] Single segment extraction on synthetic KPMG engagement text
  7. [LIVE] Full document extraction on real PDF
"""

from __future__ import annotations

import sys
import io
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

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
# 1. Data model tests
# --------------------------------------------------------------------------- #

def test_models() -> None:
    print("\n" + "=" * 60)
    print("1. Data model tests")
    print("=" * 60)

    from gpt.models import ExtractedField, ExtractionResult

    f = ExtractedField(
        field_name="Letter Date",
        value="January 22, 2026",
        confidence=0.98,
        evidence="Letter Date: January 22, 2026",
        segment_id="seg_001",
        page_numbers=[1],
    )
    _assert(f.field_name == "Letter Date", "field_name set")
    _assert(f.value == "January 22, 2026", "value set")
    _assert(f.is_high_confidence, "confidence >= 0.85 → is_high_confidence")
    _assert(f.validation_status == "pending", "default validation_status is pending")
    _assert(f.raw_gpt_field == "Letter Date", "raw_gpt_field defaults to field_name")

    # Confidence clamping
    f_over = ExtractedField("X", "Y", 1.5, "", "seg_001", [1])
    _assert(f_over.confidence == 1.0, "confidence clamped to 1.0")
    f_under = ExtractedField("X", "Y", -0.5, "", "seg_001", [1])
    _assert(f_under.confidence == 0.0, "confidence clamped to 0.0")

    result = ExtractionResult(
        segment_id="seg_001",
        fields=[f],
        model_used="gpt-4o",
        prompt_tokens=500,
        completion_tokens=200,
        total_tokens=700,
    )
    _assert(result.field_count == 1, "field_count correct")
    _assert(result.total_tokens == 700, "total_tokens correct")


# --------------------------------------------------------------------------- #
# 2. Prompt builder tests
# --------------------------------------------------------------------------- #

def test_prompts() -> None:
    print("\n" + "=" * 60)
    print("2. Prompt builder tests")
    print("=" * 60)

    from gpt.prompts import get_user_prompt, SYSTEM_PROMPT
    from extraction.segment import Segment

    def make_seg(is_kv=False, is_table=False, is_mixed=False):
        return Segment(
            segment_id="seg_001",
            text="Letter Date: January 22, 2026\nClient Name: Acme Corp",
            page_numbers=[1],
            region_types=["key_value" if is_kv else "paragraph"],
            estimated_tokens=20,
            section_header="ENGAGEMENT LETTER",
            is_key_value_block=is_kv,
            is_table=is_table,
            is_mixed=is_mixed,
        )

    # KV prompt
    kv_prompt = get_user_prompt(make_seg(is_kv=True))
    _assert("key-value" in kv_prompt.lower() or "Field: Value" in kv_prompt, "KV prompt selected for KV segment")
    _assert("January 22, 2026" in kv_prompt or "Acme Corp" in kv_prompt, "Segment text in KV prompt")

    # Table prompt
    table_prompt = get_user_prompt(make_seg(is_table=True))
    _assert("table" in table_prompt.lower(), "Table prompt selected for table segment")

    # Paragraph prompt
    para_prompt = get_user_prompt(make_seg())
    _assert("narrative" in para_prompt.lower() or "paragraph" in para_prompt.lower(), "Paragraph prompt for default segment")

    # System prompt essentials
    _assert("JSON" in SYSTEM_PROMPT, "System prompt mentions JSON")
    _assert("evidence" in SYSTEM_PROMPT.lower(), "System prompt requires evidence")
    _assert("hallucin" in SYSTEM_PROMPT.lower(), "System prompt forbids hallucination")
    _assert("confidence" in SYSTEM_PROMPT.lower(), "System prompt mentions confidence")


# --------------------------------------------------------------------------- #
# 3. Field parsing tests
# --------------------------------------------------------------------------- #

def test_field_parsing() -> None:
    print("\n" + "=" * 60)
    print("3. Field parsing tests")
    print("=" * 60)

    from gpt.extractor import _parse_gpt_fields

    raw = [
        {"field": "Letter Date", "value": "January 22, 2026", "confidence": 0.98,
         "evidence": "Letter Date: January 22, 2026"},
        {"field": "Client Name", "value": "Foundry Associates REIT LLC", "confidence": 0.97,
         "evidence": "Client Name: Foundry Associates REIT LLC"},
        {"field": "", "value": "empty field name should be skipped", "confidence": 0.9, "evidence": ""},
        {"field": "Bad Value", "value": "N/A", "confidence": 0.8, "evidence": ""},
        {"field": "Masked Field", "value": "[US_SSN_1]", "confidence": 0.95,
         "evidence": "SSN: [US_SSN_1]"},
    ]

    fields = _parse_gpt_fields(raw, segment_id="seg_001", page_numbers=[1])

    _assert(len(fields) == 3, f"3 valid fields parsed (got {len(fields)})")
    _assert(fields[0].field_name == "Letter Date", "First field name correct")
    _assert(fields[0].confidence == 0.98, "Confidence parsed correctly")
    _assert(fields[2].value == "[US_SSN_1]", "Mask token preserved as value")

    # Missing confidence → default
    raw_no_conf = [{"field": "X", "value": "Y", "evidence": "X: Y"}]
    fields_nc = _parse_gpt_fields(raw_no_conf, "seg_001", [1])
    _assert(fields_nc[0].confidence == 0.80, "Missing confidence → default 0.80")

    # Empty list
    fields_empty = _parse_gpt_fields([], "seg_001", [1])
    _assert(fields_empty == [], "Empty raw list → empty fields")


# --------------------------------------------------------------------------- #
# 4. Fallback trigger tests
# --------------------------------------------------------------------------- #

def test_fallback_trigger() -> None:
    print("\n" + "=" * 60)
    print("4. Fallback trigger logic tests")
    print("=" * 60)

    from gpt.extractor import _should_trigger_fallback
    from gpt.models import ExtractedField
    from extraction.segment import Segment

    def make_kv_seg(text: str) -> Segment:
        return Segment(
            segment_id="seg_001",
            text=text,
            page_numbers=[1],
            region_types=["key_value"],
            estimated_tokens=50,
            section_header="",
            is_key_value_block=True,
            is_table=False,
            is_mixed=False,
        )

    def make_field(conf: float) -> ExtractedField:
        return ExtractedField("F", "V", conf, "F: V", "seg_001", [1])

    # KV segment with 0 fields → fallback
    kv_seg = make_kv_seg("A: 1\nB: 2\nC: 3\nD: 4\nE: 5")
    _assert(_should_trigger_fallback(kv_seg, []), "KV segment with 0 fields → fallback triggered")

    # High confidence → no fallback
    high_conf_fields = [make_field(0.95), make_field(0.92), make_field(0.98)]
    _assert(not _should_trigger_fallback(kv_seg, high_conf_fields), "High confidence → no fallback")

    # Low average confidence → fallback
    low_conf_fields = [make_field(0.60), make_field(0.55)]
    _assert(_should_trigger_fallback(kv_seg, low_conf_fields), "Low confidence → fallback triggered")

    # KV sparsity: 5 KV lines but only 1 field
    sparse_fields = [make_field(0.90)]
    _assert(_should_trigger_fallback(kv_seg, sparse_fields), "KV sparsity → fallback triggered")


# --------------------------------------------------------------------------- #
# 5. Mock extraction test (no API key needed)
# --------------------------------------------------------------------------- #

def test_mock_extraction() -> None:
    print("\n" + "=" * 60)
    print("5. Mock extraction test (no API key required)")
    print("=" * 60)

    from gpt.extractor import extract_segment
    from extraction.segment import Segment

    seg = Segment(
        segment_id="seg_001",
        text=(
            "Letter Date: January 22, 2026\n"
            "Client Name: Foundry Associates REIT LLC\n"
            "Proposed Fee: USD 43,500\n"
            "Payment Term: 10 working days\n"
        ),
        page_numbers=[1],
        region_types=["key_value"],
        estimated_tokens=40,
        section_header="ENGAGEMENT LETTER",
        is_key_value_block=True,
        is_table=False,
        is_mixed=False,
    )

    # Mock the GPT client to return a fixed response
    mock_response = {
        "fields": [
            {"field": "Letter Date", "value": "January 22, 2026",
             "confidence": 0.98, "evidence": "Letter Date: January 22, 2026"},
            {"field": "Client Name", "value": "Foundry Associates REIT LLC",
             "confidence": 0.97, "evidence": "Client Name: Foundry Associates REIT LLC"},
            {"field": "Proposed Fee", "value": "USD 43,500",
             "confidence": 0.99, "evidence": "Proposed Fee: USD 43,500"},
            {"field": "Payment Term", "value": "10 working days",
             "confidence": 0.96, "evidence": "Payment Term: 10 working days"},
        ]
    }
    mock_usage = {"prompt_tokens": 300, "completion_tokens": 150, "total_tokens": 450}

    with patch("gpt.extractor.call_gpt_with_fallback",
               return_value=(mock_response, mock_usage, "gpt-4o")):
        result = extract_segment(seg)

    _assert(result is not None, "extract_segment returns a result")
    _assert(result.segment_id == "seg_001", "segment_id preserved")
    _assert(result.field_count == 4, f"4 fields extracted (got {result.field_count})")
    _assert(result.model_used == "gpt-4o", "model_used set")
    _assert(result.total_tokens == 450, "token usage tracked")
    _assert(result.fields[0].field_name == "Letter Date", "First field name correct")
    _assert(result.fields[2].value == "USD 43,500", "Fee value extracted correctly")

    print(f"\n  Extracted {result.field_count} fields from mock GPT response:")
    for f in result.fields:
        print(f"  {f.field_name:<45} | {f.value}")


# --------------------------------------------------------------------------- #
# 6. Live extraction test
# --------------------------------------------------------------------------- #

def test_live_extraction() -> None:
    print("\n" + "=" * 60)
    print("6. LIVE extraction test (requires OPENAI_API_KEY)")
    print("=" * 60)

    from gpt.extractor import extract_segment
    from extraction.segment import Segment

    seg = Segment(
        segment_id="seg_live_001",
        text=(
            "Letter Date: January 22, 2026\n"
            "Client Name: Foundry Associates REIT LLC\n"
            "Client Address: 1270 Avenue of the Americas, STE 1907, New York, 10020, United States\n"
            "Auditor / Firm Name: KPMG Samjong Accounting Corp.\n"
            "KPMG Contact: Sang Bum Oh, +82-2-2112-0721\n"
            "Service Type: U.S. tax compliance services\n"
            "Tax Year Covered: 2025\n"
            "Proposed Fee (Tax Compliance Services): USD 43,500 (cap)\n"
            "Fee Exclusions: Exclusive of VAT and reasonable out-of-pocket expenses\n"
            "Payment Term (Duration): 10 working days\n"
            "Billing / Invoicing Schedule: 50% billed in April 2026, remaining 50% by August 2026\n"
        ),
        page_numbers=[1],
        region_types=["key_value"],
        estimated_tokens=120,
        section_header="ENGAGEMENT LETTER",
        is_key_value_block=True,
        is_table=False,
        is_mixed=False,
    )

    result = extract_segment(seg)

    _assert(result is not None, "Live extraction returned a result")
    _assert(result.field_count >= 5, f"At least 5 fields extracted (got {result.field_count})")
    _assert(result.error is None, "No error in result")

    field_names = {f.field_name for f in result.fields}
    print(f"\n  Model       : {result.model_used}")
    print(f"  Fields      : {result.field_count}")
    print(f"  Tokens used : {result.total_tokens}")
    print(f"\n  Extracted fields:")
    for f in result.fields:
        flag = "[high]" if f.is_high_confidence else "[low] "
        print(f"  {flag} {f.field_name:<50} | {f.value}")

    # Key fields must be extracted
    all_text = " ".join(f.field_name.lower() for f in result.fields)
    _assert("date" in all_text or "letter" in all_text, "Date-related field found")
    _assert(any("fee" in f.field_name.lower() or "43,500" in f.value for f in result.fields),
            "Fee field extracted")


# --------------------------------------------------------------------------- #
# 7. Live full document test
# --------------------------------------------------------------------------- #

def test_live_document(pdf_path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"7. LIVE full document extraction: {pdf_path.name}")
    print("=" * 60)

    from ingestion import read_pdf
    from preprocessing import preprocess
    from masking.engine import mask_document
    from extraction.segmenter import segment_document
    from gpt.extractor import extract_document

    pages = read_pdf(pdf_path, run_ocr=False)
    preprocessed = preprocess(pages)
    masked_pages, context = mask_document(preprocessed, doc_id=pdf_path.stem, save_mask_file=False)
    segments = segment_document(masked_pages, preprocessed)

    print(f"\n  Segments to process: {len(segments)}")
    results, summary = extract_document(segments)

    print(f"\n  Segments processed : {summary.segments_processed}")
    print(f"  Segments skipped   : {summary.segments_skipped}")
    print(f"  Total fields       : {summary.total_fields_extracted}")
    print(f"  Total tokens       : {summary.total_tokens}")
    print(f"  Estimated cost     : ${summary.estimated_cost_usd():.4f}")
    print(f"  Models used        : {summary.models_used}")

    _assert(summary.total_fields_extracted >= 10, "At least 10 fields extracted from real document")
    _assert(len(summary.errors) == 0 or len(summary.errors) < 3, "Fewer than 3 errors")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    live_mode = "--live" in sys.argv
    pdf_path = None

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        pdf_path = Path(args[0])

    # Always run these (no API key needed)
    test_models()
    test_prompts()
    test_field_parsing()
    test_fallback_trigger()
    test_mock_extraction()

    if live_mode:
        test_live_extraction()
        if pdf_path and pdf_path.exists():
            test_live_document(pdf_path)
        elif pdf_path:
            print(f"Warning: PDF not found: {pdf_path}", file=sys.stderr)
    else:
        print("\n  [INFO] Skipping live tests. Run with --live to test against OpenAI API.")

    print("\n" + "=" * 60)
    print("All GPT tests passed.")
    print("=" * 60)
