"""
gpt/prompts.py
--------------
Prompt templates for GPT dynamic field discovery and extraction.

Design principles
-----------------
1. PRECISION over RECALL — it is better to return fewer, correct fields than
   more, guessed ones. This directly targets the ≥95% accuracy requirement.
2. EVIDENCE-ANCHORED — every extracted value must be directly supported by
   an exact text span in the source. GPT is instructed to include this evidence.
3. STRUCTURED OUTPUT — always request JSON so we can parse reliably.
4. CONTEXT-AWARE — prompts vary by segment type (KV, paragraph, table)
   to maximise extraction quality for each content shape.
5. NO HALLUCINATION — explicit instructions to return nothing if unsure.
"""

from __future__ import annotations

from enum import Enum


class SegmentType(str, Enum):
    KEY_VALUE = "key_value"
    PARAGRAPH = "paragraph"
    TABLE     = "table"
    MIXED     = "mixed"
    UNKNOWN   = "unknown"


# --------------------------------------------------------------------------- #
# System prompt (shared across all segment types)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are an expert business document analyst specializing in high-density structured field extraction.

Your goal is COMPREHENSIVE EXTRACTION: extract ALL business field-value pairs from the document text. A typical business document contains 20+ distinct business fields. Do not skip details.

WHAT TO EXTRACT (be thorough):
1. Key Parties & Names: Client Name, Firm/Auditor Name, Signatories, Representatives, Addresses, Contact Details, Email, Phone.
2. Financials & Fees: Proposed Fee, Retainer, VAT/Tax Terms, Payment Terms, Hourly Rates, Expense Exclusions, Currency, Billing Frequency.
3. Dates & Deadlines: Letter/Contract Date, Effective Date, Termination Date, Billing Dates, Reporting Deadlines, Tax Year Covered.
4. Scope & Legal: Service Type/Scope, Deliverables, Governing Law, Jurisdiction, Confidentiality Terms, Liability Limits, Engagement References.
5. Identifiers: Document Ref No, Tax ID / EIN, VAT Registration, Invoice/Project Codes.

CRITICAL RULES:
1. Extract ALL field-value pairs stated in the text.
2. Values must be exact verbatim text fragments from the source.
3. Every field MUST include an 'evidence' string with the exact text span.
4. Field names should be clean, human-readable Title Case (e.g. "Tax Year Covered", "Proposed Fee").
5. Mask tokens like [US_SSN_1] or [CREDIT_CARD_1] represent sensitive data — include them as-is in values.

OUTPUT FORMAT — respond with ONLY valid JSON:
{
  "fields": [
    {
      "field": "<human-readable field name>",
      "value": "<exact verbatim value from text>",
      "confidence": 0.95,
      "evidence": "<exact text fragment from document>"
    }
  ]
}
"""

# --------------------------------------------------------------------------- #
# User prompt builders (vary by segment type)
# --------------------------------------------------------------------------- #

def build_kv_prompt(
    segment_text: str,
    section_header: str,
    page_numbers: list[int],
) -> str:
    """
    Prompt for KEY-VALUE structured segments.

    These segments are already in "Field: Value" format. GPT's job here is
    to cleanly parse and validate each pair, assess confidence, and return
    evidence. This is the highest-accuracy extraction scenario.
    """
    context = f"Section: {section_header}" if section_header else "Document section"
    pages = ", ".join(str(p) for p in page_numbers)

    return f"""Extract all business field-value pairs from the following document section.

Context: {context} | Pages: {pages}
Segment type: Structured key-value content

The text below is already in "Field: Value" format. Extract each pair precisely.
Pay special attention to:
- Multi-line values (a value that continues on the next line)
- Compound fields with multiple components in the value
- Numerical values with units (USD amounts, percentages, durations)

DOCUMENT TEXT:
---
{segment_text}
---

Return the JSON extraction. Include ALL field-value pairs found."""


def build_paragraph_prompt(
    segment_text: str,
    section_header: str,
    page_numbers: list[int],
) -> str:
    """
    Prompt for PARAGRAPH segments (narrative / prose content).

    GPT must identify business fields embedded in natural language sentences.
    This is harder — instruct GPT to be conservative and only extract
    clearly stated facts, not implied ones.
    """
    context = f"Section: {section_header}" if section_header else "Document section"
    pages = ", ".join(str(p) for p in page_numbers)

    return f"""Extract meaningful business facts from the following document paragraph(s).

Context: {context} | Pages: {pages}
Segment type: Narrative / prose content

Read the text carefully and identify explicitly stated business information such as:
- Named parties, roles, and responsibilities
- Specific dates, deadlines, and durations
- Monetary amounts, fees, rates, and pricing
- Obligations, rights, and conditions
- Document references and identifiers
- Geographic locations relevant to the business context
- Scope of services or products described

IMPORTANT: Only extract facts that are EXPLICITLY stated. Do not interpret or infer.
If a sentence says "payment is due within 30 days", extract field="Payment Due Period", value="30 days".
Do NOT rephrase. Keep values verbatim.

DOCUMENT TEXT:
---
{segment_text}
---

Return the JSON extraction. Be conservative — precision is more important than completeness."""


def build_table_prompt(
    segment_text: str,
    section_header: str,
    page_numbers: list[int],
) -> str:
    """
    Prompt for TABLE segments.

    Tables may contain column headers + rows of data. GPT should extract
    meaningful column-header:cell pairs.
    """
    context = f"Section: {section_header}" if section_header else "Document section"
    pages = ", ".join(str(p) for p in page_numbers)

    return f"""Extract business data from the following document table.

Context: {context} | Pages: {pages}
Segment type: Table / structured grid content

The text below represents a table. Column headers define the field names.
Each row represents one record. Extract each meaningful column-header:cell-value pair.
If the table has row labels, use them as field name context (e.g. "Service Type - Rate").

DOCUMENT TEXT:
---
{segment_text}
---

Return the JSON extraction with one entry per meaningful cell value."""


def build_mixed_prompt(
    segment_text: str,
    section_header: str,
    page_numbers: list[int],
) -> str:
    """
    Prompt for MIXED segments (KV + paragraph combined).
    """
    context = f"Section: {section_header}" if section_header else "Document section"
    pages = ", ".join(str(p) for p in page_numbers)

    return f"""Extract all business field-value pairs from the following mixed document section.

Context: {context} | Pages: {pages}
Segment type: Mixed content (structured fields and narrative text)

The text below contains both structured "Field: Value" lines and narrative paragraphs.
Extract ALL meaningful business information from both parts:
1. For structured lines: extract each "Field: Value" pair directly.
2. For narrative text: extract explicitly stated business facts only.

DOCUMENT TEXT:
---
{segment_text}
---

Return the JSON extraction. Combine results from both content types."""


# --------------------------------------------------------------------------- #
# Fallback / verification prompt
# --------------------------------------------------------------------------- #

FALLBACK_SYSTEM_PROMPT = """You are a careful document reviewer verifying extracted business data.

You will be given:
1. A piece of document text
2. A list of fields that were previously extracted from it

Your job is to:
- Verify each extracted field-value pair is genuinely supported by the text
- Correct any values that are wrong or incomplete
- Add any important business fields that were missed
- Remove any fields that cannot be supported by the text

Apply the same strict rules:
- Evidence must exist verbatim in the source text
- No hallucination or inference
- Return the corrected/complete list in the same JSON format

OUTPUT FORMAT:
{
  "fields": [
    {
      "field": "<field name>",
      "value": "<verbatim value>",
      "confidence": <float>,
      "evidence": "<exact source text fragment>"
    }
  ]
}
"""


def build_fallback_prompt(
    segment_text: str,
    previous_fields: list[dict],
    section_header: str,
    page_numbers: list[int],
) -> str:
    """
    Prompt for fallback/verification calls on segments where primary extraction
    returned low-confidence results or suspiciously few fields.
    """
    import json
    context = f"Section: {section_header}" if section_header else "Document section"
    pages = ", ".join(str(p) for p in page_numbers)
    prev_json = json.dumps({"fields": previous_fields}, indent=2)

    return f"""Review and improve the following extraction from a business document.

Context: {context} | Pages: {pages}

ORIGINAL DOCUMENT TEXT:
---
{segment_text}
---

PREVIOUSLY EXTRACTED FIELDS:
{prev_json}

Please:
1. Verify each extracted field is correct and supported by the text
2. Fix any wrong or incomplete values
3. Add any important business fields that were missed
4. Remove fields not supported by the text

Return the corrected complete JSON extraction."""


# --------------------------------------------------------------------------- #
# Prompt selector
# --------------------------------------------------------------------------- #

def get_user_prompt(
    segment,   # Segment
) -> str:
    """
    Select and build the appropriate user prompt based on segment type.
    """
    from extraction.segment import Segment

    if segment.is_table:
        return build_table_prompt(segment.text, segment.section_header, segment.page_numbers)
    elif segment.is_key_value_block:
        return build_kv_prompt(segment.text, segment.section_header, segment.page_numbers)
    elif segment.is_mixed:
        return build_mixed_prompt(segment.text, segment.section_header, segment.page_numbers)
    else:
        return build_paragraph_prompt(segment.text, segment.section_header, segment.page_numbers)
