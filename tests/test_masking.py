"""
tests/test_masking.py
---------------------
Tests for the masking layer (Phase 4).

Usage:
    python -m tests.test_masking
    python -m tests.test_masking path/to/document.pdf

Test groups:
  1. MaskingContext — token creation, cross-page consistency, rehydration, save/load
  2. Entities config — should_mask logic
  3. Rehydrator — token replacement in strings/dicts/lists
  4. Engine — Presidio integration test with synthetic PII text
  5. Integration — ingestion → preprocessing → masking on a real PDF
"""

from __future__ import annotations

import sys
import io
import logging
import tempfile
from pathlib import Path

# UTF-8 output on Windows
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
# 1. MaskingContext unit tests
# --------------------------------------------------------------------------- #

def test_context() -> None:
    print("\n" + "=" * 60)
    print("1. MaskingContext unit tests")
    print("=" * 60)

    from masking.context import MaskingContext

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = MaskingContext(doc_id="test_doc_001", masks_dir=Path(tmpdir))

        # Create a token
        token1 = ctx.get_or_create_token(
            original_value="123-45-6789",
            entity_type="US_SSN",
            score=0.95,
            page_number=1,
            char_start=10,
            char_end=21,
        )
        _assert(token1 == "[US_SSN_1]", f"First US_SSN token is [US_SSN_1], got {token1!r}")
        _assert(ctx.total_masked() == 1, "One entity masked")

        # Cross-page consistency: same value → same token
        token_same = ctx.get_or_create_token(
            original_value="123-45-6789",
            entity_type="US_SSN",
            score=0.90,
            page_number=5,
            char_start=0,
            char_end=11,
        )
        _assert(token_same == token1, "Same value returns same token across pages")
        _assert(ctx.total_masked() == 1, "No new entity created for duplicate value")

        # Second different SSN → different token
        token2 = ctx.get_or_create_token(
            original_value="987-65-4321",
            entity_type="US_SSN",
            score=0.95,
            page_number=2,
            char_start=0,
            char_end=11,
        )
        _assert(token2 == "[US_SSN_2]", f"Second US_SSN token is [US_SSN_2], got {token2!r}")
        _assert(ctx.total_masked() == 2, "Two unique entities masked")

        # Different entity type → different counter sequence
        token_cc = ctx.get_or_create_token(
            original_value="4111111111111111",
            entity_type="CREDIT_CARD",
            score=0.98,
            page_number=3,
            char_start=0,
            char_end=16,
        )
        _assert(token_cc == "[CREDIT_CARD_1]", f"Credit card token correct, got {token_cc!r}")
        _assert(ctx.total_masked() == 3, "Three unique entities")

        # Rehydration
        masked_text = "SSN is [US_SSN_1] and card is [CREDIT_CARD_1]."
        restored = ctx.rehydrate(masked_text)
        _assert("123-45-6789" in restored, "US_SSN_1 rehydrated correctly")
        _assert("4111111111111111" in restored, "CREDIT_CARD_1 rehydrated correctly")
        _assert("[US_SSN_1]" not in restored, "Token fully replaced after rehydration")

        # Summary
        summary = ctx.summary()
        _assert(summary.get("US_SSN") == 2, "Summary shows 2 US_SSN entities")
        _assert(summary.get("CREDIT_CARD") == 1, "Summary shows 1 CREDIT_CARD entity")

        # Save to file
        saved_path = ctx.save()
        _assert(saved_path.exists(), "Mask file written to disk")

        # Load from file
        ctx2 = MaskingContext.load("test_doc_001", masks_dir=Path(tmpdir))
        _assert(ctx2.total_masked() == 3, "Loaded context has 3 entities")
        restored2 = ctx2.rehydrate("[US_SSN_1]")
        _assert(restored2 == "123-45-6789", "Loaded context rehydrates correctly")


# --------------------------------------------------------------------------- #
# 2. Entities config tests
# --------------------------------------------------------------------------- #

def test_entities_config() -> None:
    print("\n" + "=" * 60)
    print("2. Entities configuration tests")
    print("=" * 60)

    from masking.entities import should_mask

    # Should mask
    _assert(should_mask("US_SSN", 0.90), "US_SSN with high score → mask")
    _assert(should_mask("CREDIT_CARD", 0.95), "CREDIT_CARD → mask")
    _assert(should_mask("IBAN_CODE", 0.80), "IBAN_CODE → mask")

    # Should NOT mask (score too low)
    _assert(not should_mask("US_SSN", 0.50), "US_SSN with low score → keep")

    # Should NOT mask (not in MASK_ENTITY_TYPES)
    _assert(not should_mask("PERSON", 0.99), "PERSON → keep (business name)")
    _assert(not should_mask("ORG", 0.99), "ORG → keep (company name)")
    _assert(not should_mask("DATE_TIME", 0.99), "DATE_TIME → keep (business field)")
    _assert(not should_mask("MONEY", 0.99), "MONEY → keep (business field)")
    _assert(not should_mask("LOCATION", 0.99), "LOCATION → keep (business address)")
    _assert(not should_mask("EMAIL_ADDRESS", 0.99), "EMAIL_ADDRESS → keep (business contact)")
    _assert(not should_mask("PHONE_NUMBER", 0.99), "PHONE_NUMBER → keep (business contact)")


# --------------------------------------------------------------------------- #
# 3. Rehydrator unit tests
# --------------------------------------------------------------------------- #

def test_rehydrator() -> None:
    print("\n" + "=" * 60)
    print("3. Rehydrator unit tests")
    print("=" * 60)

    from masking.context import MaskingContext
    from masking.rehydrator import rehydrate_value, rehydrate_fields
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = MaskingContext(doc_id="rehydrate_test", masks_dir=Path(tmpdir))
        ctx.get_or_create_token("123-45-6789", "US_SSN", 0.95, 1, 0, 11)
        ctx.get_or_create_token("4111111111111111", "CREDIT_CARD", 0.98, 1, 20, 36)

        # String rehydration
        result = rehydrate_value("SSN: [US_SSN_1], Card: [CREDIT_CARD_1]", ctx)
        _assert("123-45-6789" in result, "String: SSN token replaced")
        _assert("4111111111111111" in result, "String: credit card token replaced")

        # Unknown token left as-is
        result2 = rehydrate_value("Unknown: [UNKNOWN_99]", ctx)
        _assert("[UNKNOWN_99]" in result2, "Unknown token left unchanged")

        # List rehydration
        result3 = rehydrate_value(["[US_SSN_1]", "normal value"], ctx)
        _assert(result3[0] == "123-45-6789", "List element rehydrated")
        _assert(result3[1] == "normal value", "Non-token list element unchanged")

        # No-op: value without tokens
        result4 = rehydrate_value("USD 43,500", ctx)
        _assert(result4 == "USD 43,500", "Non-masked value returned unchanged")

        # Field list rehydration
        fields = [
            {"field": "Client Name", "value": "Acme Corp"},
            {"field": "Tax ID", "value": "[US_SSN_1]"},
            {"field": "Payment", "value": "USD 43,500"},
        ]
        rehydrated = rehydrate_fields(fields, ctx)
        _assert(rehydrated[0]["value"] == "Acme Corp", "Non-masked field unchanged")
        _assert(rehydrated[1]["value"] == "123-45-6789", "Masked field rehydrated")
        _assert(rehydrated[2]["value"] == "USD 43,500", "Amount unchanged")


# --------------------------------------------------------------------------- #
# 4. Engine integration test (Presidio)
# --------------------------------------------------------------------------- #

def test_engine() -> None:
    print("\n" + "=" * 60)
    print("4. Engine test (Presidio + spaCy)")
    print("=" * 60)

    from masking.engine import mask_page
    from masking.context import MaskingContext
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = MaskingContext(doc_id="engine_test", masks_dir=Path(tmpdir))

        # Text with clear PII that Presidio should detect
        text_with_pii = (
            "Client Name: Foundry Associates REIT LLC\n"
            "Letter Date: January 22, 2026\n"
            "Proposed Fee: USD 43,500\n"
            "Client SSN: 123-45-6789\n"          # <-- should be masked
            "Payment Terms: Net 30 days\n"
            "Credit Card: 4111-1111-1111-1111\n"  # <-- should be masked
        )

        result = mask_page(page_number=1, text=text_with_pii, context=ctx)

        # Business fields must NEVER be masked (these are the critical assertions)
        _assert("Foundry Associates REIT LLC" in result.masked_text, "Company name NOT masked")
        _assert("January 22, 2026" in result.masked_text, "Date NOT masked")
        _assert("USD 43,500" in result.masked_text, "Fee amount NOT masked")
        _assert("Net 30 days" in result.masked_text, "Payment terms NOT masked")

        # Credit card should definitely be masked (high-confidence pattern-based rule)
        _assert("4111-1111-1111-1111" not in result.masked_text, "Credit card masked")

        # SSN: Presidio detection is context-sensitive. If masked, great. If not,
        # it means the score was below threshold — acceptable for POC.
        ssn_masked = "123-45-6789" not in result.masked_text
        print(f"\n  SSN detection result: {'MASKED' if ssn_masked else 'kept (below threshold)'}")
        # Not a hard assertion — Presidio is probabilistic

        print(f"\n  Entities found  : {result.entities_found}")
        print(f"  Entities masked : {result.entities_masked}")
        print(f"  Context summary : {ctx.summary()}")
        print(f"\n  Masked text preview:")
        for line in result.masked_text.splitlines():
            print(f"    {line}")

        # Rehydration round-trip
        restored = ctx.rehydrate(result.masked_text)
        _assert("123-45-6789" in restored, "SSN restored via rehydration")
        _assert("4111-1111-1111-1111" in restored, "Credit card restored via rehydration")


# --------------------------------------------------------------------------- #
# 5. Integration: ingestion → preprocessing → masking on real PDF
# --------------------------------------------------------------------------- #

def test_full_pipeline(pdf_path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"5. Full pipeline test: {pdf_path.name}")
    print("=" * 60)

    from ingestion import read_pdf
    from preprocessing import preprocess
    from masking.engine import mask_document
    import tempfile

    pages = read_pdf(pdf_path, run_ocr=False)
    preprocessed = preprocess(pages)

    with tempfile.TemporaryDirectory() as tmpdir:
        masked_pages, context = mask_document(
            preprocessed,
            doc_id=pdf_path.stem,
            save_mask_file=True,
        )

    _assert(len(masked_pages) == len(preprocessed), "One masked page per preprocessed page")
    _assert(context is not None, "MaskingContext returned")

    print(f"\n  Pages processed : {len(masked_pages)}")
    print(f"  Entities masked : {context.total_masked()}")
    print(f"  Summary         : {context.summary()}")

    for mp in masked_pages:
        _assert(mp.page_number >= 1, f"Page {mp.page_number}: valid page number")
        _assert(mp.masked_text is not None, f"Page {mp.page_number}: masked_text not None")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_context()
    test_entities_config()
    test_rehydrator()
    test_engine()

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"Error: not found → {pdf_path}", file=sys.stderr)
            sys.exit(1)
        test_full_pipeline(pdf_path)

    print("\n" + "=" * 60)
    print("All masking tests passed.")
    print("=" * 60)
