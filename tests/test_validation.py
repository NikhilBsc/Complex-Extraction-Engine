"""
tests/test_validation.py
------------------------
Tests for the validation layer (Phase 7).

Usage:
    python -m tests.test_validation

Test groups:
  1. Grounding checks (exact, normalised, partial, not found)
  2. Field quality checks (boilerplate, null values, generic names)
  3. Full validator with synthetic fields
  4. Rehydration integration (mask tokens → real values before grounding)
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
# Sample document text (KPMG-style engagement letter)
# --------------------------------------------------------------------------- #

SAMPLE_DOC = (
    "ENGAGEMENT LETTER\n\n"
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
    "Billing Schedule: 50% billed in April 2026, remaining 50% by August 2026\n"
)


# --------------------------------------------------------------------------- #
# 1. Grounding tests
# --------------------------------------------------------------------------- #

def test_grounding() -> None:
    print("\n" + "=" * 60)
    print("1. Grounding check tests")
    print("=" * 60)

    from validation.grounding import check_grounding

    # Exact match
    r = check_grounding("January 22, 2026", SAMPLE_DOC, [SAMPLE_DOC])
    _assert(r.is_grounded, "Exact value grounded")
    _assert(r.match_type == "exact", "Match type is exact")
    _assert(r.confidence_penalty == 0.0, "No confidence penalty for exact match")

    # Case-insensitive exact match
    r2 = check_grounding("january 22, 2026", SAMPLE_DOC, [SAMPLE_DOC])
    _assert(r2.is_grounded, "Case-insensitive match grounded")

    # Normalised match (extra spaces)
    r3 = check_grounding("USD  43,500  (cap)", SAMPLE_DOC, [SAMPLE_DOC])
    _assert(r3.is_grounded, "Normalised match (extra spaces) grounded")

    # Grounded in full document (cross-page)
    other_page = "Additional clause: Payment Term (Duration): 10 working days"
    r4 = check_grounding("10 working days", "", [SAMPLE_DOC, other_page])
    _assert(r4.is_grounded, "Cross-page grounding works")

    # Not grounded — hallucinated value
    r5 = check_grounding("USD 999,999 (hallucinated)", SAMPLE_DOC, [SAMPLE_DOC])
    _assert(not r5.is_grounded, "Hallucinated value not grounded")
    _assert(r5.match_type == "not_found", "Match type is not_found")
    _assert(r5.confidence_penalty == 1.0, "Max penalty for ungrounded value")

    # Empty value → not grounded
    r6 = check_grounding("", SAMPLE_DOC, [SAMPLE_DOC])
    _assert(not r6.is_grounded, "Empty value not grounded")

    # Long partial match
    long_value = "Exclusive of VAT and reasonable out-of-pocket expenses"
    r7 = check_grounding(long_value, SAMPLE_DOC, [SAMPLE_DOC])
    _assert(r7.is_grounded, "Long value grounded via exact match")


# --------------------------------------------------------------------------- #
# 2. Field quality checks
# --------------------------------------------------------------------------- #

def test_field_checks() -> None:
    print("\n" + "=" * 60)
    print("2. Field quality check tests")
    print("=" * 60)

    from validation.field_checks import check_field_quality

    # Valid fields
    r = check_field_quality("Letter Date", "January 22, 2026")
    _assert(r.is_valid, "Valid KV field accepted")
    _assert(not r.should_flag, "Valid field not flagged")

    r2 = check_field_quality("Proposed Fee (Tax Compliance Services)", "USD 43,500 (cap)")
    _assert(r2.is_valid, "Valid fee field accepted")

    # Boilerplate field name
    r3 = check_field_quality("Page 1 of 5", "some value")
    _assert(not r3.is_valid, "Boilerplate 'Page 1 of 5' rejected")

    r4 = check_field_quality("Copyright", "2024 KPMG")
    _assert(not r4.is_valid, "Copyright field rejected")

    # Null value
    r5 = check_field_quality("Client Name", "N/A")
    _assert(not r5.is_valid, "N/A value rejected")

    r6 = check_field_quality("Status", "TBD")
    _assert(not r6.is_valid, "TBD value rejected")

    # Empty field name
    r7 = check_field_quality("", "some value")
    _assert(not r7.is_valid, "Empty field name rejected")

    # Generic field name → flagged but valid
    r8 = check_field_quality("name", "Foundry Associates REIT LLC")
    _assert(r8.is_valid, "Generic 'name' field is valid")
    _assert(r8.should_flag, "Generic 'name' field is flagged for review")

    # Very long value → flagged
    r9 = check_field_quality("Scope of Services", "A" * 600)
    _assert(r9.is_valid, "Long value field is valid")
    _assert(r9.should_flag, "Long value field flagged for review")


# --------------------------------------------------------------------------- #
# 3. Full validator with synthetic extraction results
# --------------------------------------------------------------------------- #

def test_full_validator() -> None:
    print("\n" + "=" * 60)
    print("3. Full validator with synthetic extraction results")
    print("=" * 60)

    from gpt.models import ExtractedField, ExtractionResult
    from masking.context import MaskingContext
    from validation.validator import validate_document, DocumentContext
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        mask_ctx = MaskingContext(doc_id="val_test", masks_dir=Path(tmpdir))

        # Simulate extraction results
        fields = [
            ExtractedField("Letter Date", "January 22, 2026", 0.98,
                           "Letter Date: January 22, 2026", "seg_001", [1]),
            ExtractedField("Client Name", "Foundry Associates REIT LLC", 0.97,
                           "Client Name: Foundry Associates REIT LLC", "seg_001", [1]),
            ExtractedField("Proposed Fee", "USD 43,500 (cap)", 0.99,
                           "Proposed Fee: USD 43,500 (cap)", "seg_001", [1]),
            # Hallucinated field — NOT in document
            ExtractedField("Tax Rate", "35% flat rate (hallucinated)", 0.85,
                           "Tax Rate: 35%", "seg_001", [1]),
            # Boilerplate field
            ExtractedField("Page 1 of 5", "Header", 0.70,
                           "Page 1 of 5", "seg_001", [1]),
            # Low confidence
            ExtractedField("Payment Term (Duration)", "10 working days", 0.65,
                           "Payment Term (Duration): 10 working days", "seg_001", [1]),
        ]

        results = [
            ExtractionResult(
                segment_id="seg_001",
                fields=fields,
                model_used="gpt-4o",
                prompt_tokens=300,
                completion_tokens=150,
                total_tokens=450,
            )
        ]

        doc_context = DocumentContext(
            segment_texts={"seg_001": SAMPLE_DOC},
            all_masked_texts=[SAMPLE_DOC],
            all_original_texts=[SAMPLE_DOC],
        )

        all_fields, report = validate_document(results, doc_context, mask_ctx)

    print(f"\n  Total fields : {report.total_fields}")
    print(f"  Accepted     : {report.accepted}")
    print(f"  Flagged      : {report.flagged}")
    print(f"  Rejected     : {report.rejected}")
    print(f"  Precision est: {report.precision_estimate:.1%}")

    _assert(report.total_fields == 6, "6 total fields")
    _assert(report.accepted >= 2, "At least 2 accepted (date, client, fee)")
    _assert(report.rejected >= 2, "At least 2 rejected (hallucinated, boilerplate)")

    # Specific field statuses
    by_field = {f.field_name: f for f in all_fields}
    _assert(by_field["Letter Date"].validation_status == "accepted",
            "Letter Date accepted")
    _assert(by_field["Client Name"].validation_status == "accepted",
            "Client Name accepted")
    _assert(by_field["Tax Rate (hallucinated)"].validation_status == "rejected"
            if "Tax Rate (hallucinated)" in by_field
            else by_field["Tax Rate"].validation_status in ("rejected", "flagged"),
            "Hallucinated Tax Rate rejected or flagged")
    _assert(by_field["Page 1 of 5"].validation_status == "rejected",
            "Boilerplate field rejected")
    _assert(by_field["Payment Term (Duration)"].validation_status in ("accepted", "flagged"),
            "Low-confidence field accepted or flagged")

    print("\n  Field status summary:")
    for f in all_fields:
        icon = "[ACC]" if f.validation_status == "accepted" else \
               "[FLG]" if f.validation_status == "flagged" else "[REJ]"
        print(f"  {icon} {f.field_name:<50} = {str(f.value)[:40]}")


# --------------------------------------------------------------------------- #
# 4. Rehydration integration in validator
# --------------------------------------------------------------------------- #

def test_rehydration_in_validator() -> None:
    print("\n" + "=" * 60)
    print("4. Rehydration integration in validator")
    print("=" * 60)

    from gpt.models import ExtractedField, ExtractionResult
    from masking.context import MaskingContext
    from validation.validator import validate_document, DocumentContext
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        mask_ctx = MaskingContext(doc_id="rehydrate_val_test", masks_dir=Path(tmpdir))

        # Register a masked entity
        mask_ctx.get_or_create_token("123-45-6789", "US_SSN", 0.95, 1, 50, 61)

        # Document with the real value
        doc_with_ssn = "Employee SSN: 123-45-6789\nClient Name: Acme Corp\n"

        # Extracted field still has the mask token (as GPT returned it)
        fields = [
            ExtractedField("Employee SSN", "[US_SSN_1]", 0.95,
                           "Employee SSN: [US_SSN_1]", "seg_001", [1]),
            ExtractedField("Client Name", "Acme Corp", 0.97,
                           "Client Name: Acme Corp", "seg_001", [1]),
        ]
        results = [ExtractionResult("seg_001", fields, "gpt-4o", 100, 50, 150)]

        doc_context = DocumentContext(
            segment_texts={"seg_001": doc_with_ssn},
            all_masked_texts=[doc_with_ssn],
            all_original_texts=[doc_with_ssn],
        )

        all_fields, report = validate_document(results, doc_context, mask_ctx)

    by_field = {f.field_name: f for f in all_fields}

    _assert(by_field["Employee SSN"].value == "123-45-6789",
            "Mask token rehydrated to real SSN value")
    _assert(by_field["Employee SSN"].is_rehydrated, "is_rehydrated flag set")
    _assert(by_field["Client Name"].value == "Acme Corp",
            "Non-masked value unchanged")
    _assert(report.total_fields == 2, "Both fields processed")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_grounding()
    test_field_checks()
    test_full_validator()
    test_rehydration_in_validator()

    print("\n" + "=" * 60)
    print("All validation tests passed.")
    print("=" * 60)
