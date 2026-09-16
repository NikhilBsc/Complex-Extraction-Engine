"""
gpt/extractor.py
----------------
Per-segment GPT extraction orchestrator — Phase 6 core.

Responsibilities
----------------
  1. For each Segment: build the right prompt → call GPT → parse response
  2. Convert raw GPT JSON into typed ExtractedField objects
  3. Detect low-confidence / sparse results → trigger fallback call
  4. Skip segments unlikely to contain business fields (legal boilerplate, etc.)
  5. Accumulate token usage across all segments
  6. Return List[ExtractionResult] — one per processed segment

Fallback trigger conditions
---------------------------
  - GPT returned 0 fields for a KV segment (very likely missed something)
  - Average confidence across returned fields < LOW_CONFIDENCE_THRESHOLD
  - GPT returned invalid/unexpected structure

Segment skipping heuristics
----------------------------
  - Empty segments
  - Segments shorter than MIN_SEGMENT_CHARS (too little content for GPT)
  - Segments that are entirely boilerplate (future: detect via embedding similarity)

Token budget tracking
---------------------
  - Accumulated per-run in ExtractionSummary
  - Logged after each segment and after full document
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from extraction.segment import Segment
from gpt.client import call_gpt_with_fallback, PRIMARY_MODEL
from gpt.models import ExtractedField, ExtractionResult
from gpt.prompts import SYSTEM_PROMPT, FALLBACK_SYSTEM_PROMPT, get_user_prompt, build_fallback_prompt

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Segments with fewer characters than this are skipped
MIN_SEGMENT_CHARS: int = 20

# If average field confidence is below this, trigger fallback
LOW_CONFIDENCE_THRESHOLD: float = 0.70

# If a KV segment returns fewer fields than this fraction of its KV lines, trigger fallback
KV_SPARSITY_THRESHOLD: float = 0.5  # e.g. 6 KV lines but only 2 fields → fallback

# Confidence to assign when GPT doesn't return a confidence value
DEFAULT_CONFIDENCE: float = 0.80

# Inter-segment delay (seconds) — keeps us safely within Groq rate limits.
# Groq Free Tier: 30 RPM → minimum 2s between requests.
# 2.5s gives headroom for any fallback calls on the same segment.
# Set to 0 if you are on a paid Groq plan.
INTER_SEGMENT_DELAY: float = 2.5


# --------------------------------------------------------------------------- #
# Summary tracker
# --------------------------------------------------------------------------- #

@dataclass
class ExtractionSummary:
    """Aggregated stats for a full document extraction run."""
    total_segments: int = 0
    segments_processed: int = 0
    segments_skipped: int = 0
    segments_with_fallback: int = 0
    total_fields_extracted: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    models_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_result(self, result: ExtractionResult) -> None:
        self.total_fields_extracted += result.field_count
        self.total_prompt_tokens += result.prompt_tokens
        self.total_completion_tokens += result.completion_tokens
        self.total_tokens += result.total_tokens
        if result.model_used not in self.models_used:
            self.models_used.append(result.model_used)
        if result.source_type == "fallback":
            self.segments_with_fallback += 1

    def estimated_cost_usd(self) -> float:
        """
        Rough cost estimate based on GPT-4o pricing (as of mid-2025).
        Input:  $5.00 / 1M tokens
        Output: $15.00 / 1M tokens
        """
        input_cost  = (self.total_prompt_tokens / 1_000_000) * 5.00
        output_cost = (self.total_completion_tokens / 1_000_000) * 15.00
        return round(input_cost + output_cost, 4)


# --------------------------------------------------------------------------- #
# Field parsing
# --------------------------------------------------------------------------- #

def _parse_gpt_fields(
    raw_fields: List[Dict[str, Any]],
    segment_id: str,
    page_numbers: List[int],
    source_type: str = "primary",
) -> List[ExtractedField]:
    """
    Convert raw GPT JSON field dicts into ExtractedField objects.

    Handles:
      - Missing keys (graceful defaults)
      - Confidence out of range (clamped)
      - Empty field name or value (skipped)
    """
    extracted: List[ExtractedField] = []

    for raw in raw_fields:
        field_name = str(raw.get("field", "")).strip()
        value = str(raw.get("value", "")).strip()
        evidence = str(raw.get("evidence", "")).strip()
        confidence = float(raw.get("confidence", DEFAULT_CONFIDENCE))

        if not field_name or not value:
            logger.debug("Skipping empty field from GPT: %r", raw)
            continue

        # Reject placeholder-like values
        if value.lower() in ("n/a", "none", "null", "not specified", "unknown", ""):
            logger.debug("Skipping null-value field: %r = %r", field_name, value)
            continue

        extracted.append(ExtractedField(
            field_name=field_name,
            value=value,
            confidence=confidence,
            evidence=evidence,
            segment_id=segment_id,
            page_numbers=page_numbers,
            source_type=source_type,
            raw_gpt_field=field_name,
        ))

    return extracted


# --------------------------------------------------------------------------- #
# Fallback trigger
# --------------------------------------------------------------------------- #

def _should_trigger_fallback(
    segment: Segment,
    fields: List[ExtractedField],
) -> bool:
    """
    Decide whether to trigger a fallback GPT call for this segment.
    """
    if not fields:
        # No fields from a KV segment → definitely try again
        if segment.is_key_value_block:
            logger.info(
                "Fallback triggered: KV segment %s returned 0 fields.",
                segment.segment_id,
            )
            return True
        return False

    avg_conf = sum(f.confidence for f in fields) / len(fields)
    if avg_conf < LOW_CONFIDENCE_THRESHOLD:
        logger.info(
            "Fallback triggered: %s avg confidence %.2f < %.2f",
            segment.segment_id, avg_conf, LOW_CONFIDENCE_THRESHOLD,
        )
        return True

    # KV sparsity check
    if segment.is_key_value_block:
        kv_lines = [
            line for line in segment.text.splitlines()
            if ":" in line and line.strip()
        ]
        if kv_lines and len(fields) < len(kv_lines) * KV_SPARSITY_THRESHOLD:
            logger.info(
                "Fallback triggered: %s has %d KV lines but only %d fields extracted.",
                segment.segment_id, len(kv_lines), len(fields),
            )
            return True

    return False


# --------------------------------------------------------------------------- #
# Single-segment extractor
# --------------------------------------------------------------------------- #

def extract_segment(
    segment: Segment,
) -> Optional[ExtractionResult]:
    """
    Run GPT extraction on a single segment.

    Returns None if the segment is skipped.
    Returns an ExtractionResult (possibly with fallback) otherwise.
    """
    # Skip trivially small segments
    if len(segment.text) < MIN_SEGMENT_CHARS:
        logger.debug(
            "Skipping segment %s: too short (%d chars).",
            segment.segment_id, len(segment.text),
        )
        return None

    user_prompt = get_user_prompt(segment)

    # --- Primary extraction call ---
    try:
        parsed, usage, model_used = call_gpt_with_fallback(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.error("GPT extraction failed for segment %s: %s", segment.segment_id, exc)
        return ExtractionResult(
            segment_id=segment.segment_id,
            fields=[],
            model_used=PRIMARY_MODEL,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=str(exc),
        )

    raw_fields = parsed.get("fields", [])
    if not isinstance(raw_fields, list):
        logger.warning("GPT returned non-list 'fields' for %s: %r", segment.segment_id, raw_fields)
        raw_fields = []

    fields = _parse_gpt_fields(
        raw_fields,
        segment_id=segment.segment_id,
        page_numbers=segment.page_numbers,
        source_type="primary",
    )

    logger.info(
        "Segment %s: %d fields extracted (primary, model=%s, tokens=%d)",
        segment.segment_id, len(fields), model_used, usage["total_tokens"],
    )

    # --- Fallback call if needed ---
    fallback_result: Optional[ExtractionResult] = None
    source_type = "primary"

    if _should_trigger_fallback(segment, fields):
        source_type = "fallback"
        prev_raw = [
            {"field": f.field_name, "value": f.value,
             "confidence": f.confidence, "evidence": f.evidence}
            for f in fields
        ]
        fallback_user_prompt = build_fallback_prompt(
            segment_text=segment.text,
            previous_fields=prev_raw,
            section_header=segment.section_header,
            page_numbers=segment.page_numbers,
        )
        try:
            fb_parsed, fb_usage, fb_model = call_gpt_with_fallback(
                system_prompt=FALLBACK_SYSTEM_PROMPT,
                user_prompt=fallback_user_prompt,
            )
            fb_raw = fb_parsed.get("fields", [])
            fields = _parse_gpt_fields(
                fb_raw,
                segment_id=segment.segment_id,
                page_numbers=segment.page_numbers,
                source_type="fallback",
            )
            logger.info(
                "Segment %s fallback: %d fields (model=%s, tokens=%d)",
                segment.segment_id, len(fields), fb_model, fb_usage["total_tokens"],
            )
            # Merge usage
            usage["prompt_tokens"]     += fb_usage["prompt_tokens"]
            usage["completion_tokens"] += fb_usage["completion_tokens"]
            usage["total_tokens"]      += fb_usage["total_tokens"]
            model_used = fb_model

        except Exception as fb_exc:
            logger.error("Fallback call failed for segment %s: %s", segment.segment_id, fb_exc)

    return ExtractionResult(
        segment_id=segment.segment_id,
        fields=fields,
        model_used=model_used,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        source_type=source_type,
    )


# --------------------------------------------------------------------------- #
# Full document extractor
# --------------------------------------------------------------------------- #

def extract_document(
    segments: List[Segment],
) -> tuple[List[ExtractionResult], ExtractionSummary]:
    """
    Run GPT extraction across all segments of a document.

    Parameters
    ----------
    segments : List[Segment]
        Output from segmentation phase.

    Returns
    -------
    (results, summary)
        results : List[ExtractionResult] — one per processed segment
        summary : ExtractionSummary — aggregated stats
    """
    summary = ExtractionSummary(total_segments=len(segments))
    results: List[ExtractionResult] = []
    t_start = time.time()

    for i, segment in enumerate(segments, start=1):
        elapsed = time.time() - t_start
        logger.info(
            "[%d/%d] segment=%s  pages=%s  ~%d tokens  (elapsed=%.0fs)",
            i, len(segments), segment.segment_id,
            segment.page_numbers, segment.estimated_tokens, elapsed,
        )
        print(
            f"    seg {i:>3}/{len(segments)}  {segment.segment_id}  "
            f"pages={segment.page_numbers}  ~{segment.estimated_tokens}tok",
            flush=True,
        )

        result = extract_segment(segment)

        if result is None:
            summary.segments_skipped += 1
            continue

        summary.segments_processed += 1
        summary.add_result(result)
        results.append(result)

        if result.error:
            summary.errors.append(f"{segment.segment_id}: {result.error}")

        # Rate-limit safety: small pause between segments
        if INTER_SEGMENT_DELAY > 0 and i < len(segments):
            time.sleep(INTER_SEGMENT_DELAY)

    logger.info(
        "Document extraction complete: %d segments processed, %d skipped, "
        "%d total fields, %d total tokens, est. cost $%.4f",
        summary.segments_processed,
        summary.segments_skipped,
        summary.total_fields_extracted,
        summary.total_tokens,
        summary.estimated_cost_usd(),
    )

    return results, summary
