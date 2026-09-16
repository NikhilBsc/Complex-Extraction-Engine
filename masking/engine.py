"""
masking/engine.py
-----------------
Presidio-based selective PII detection and masking engine.

Responsibilities
----------------
  - Initialize Presidio AnalyzerEngine with a spaCy NLP backend
  - For each page: detect PII entities → filter to MASK_ENTITY_TYPES → replace
  - Update MaskingContext with each new masked entity
  - Return masked text (token-substituted) + updated context

Key design decisions
--------------------
  1. Presidio runs ENTIRELY locally — no external API calls.
  2. spaCy model is loaded once and reused across all pages.
  3. Entity types NOT in MASK_ENTITY_TYPES are left untouched even if detected.
  4. Replacement is done right-to-left (by char offset) to avoid position drift.
  5. The engine is stateless — all state lives in MaskingContext.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from masking.context import MaskingContext
from masking.entities import MASK_ENTITY_TYPES, MIN_SCORE, should_mask

logger = logging.getLogger(__name__)

# spaCy model to use — en_core_web_lg gives best NER accuracy.
# Falls back to en_core_web_sm if large model not available.
_SPACY_MODELS = ["en_core_web_lg", "en_core_web_sm", "en_core_web_md"]


# --------------------------------------------------------------------------- #
# Lazy initialization of Presidio (expensive — load once)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _get_analyzer():
    """
    Initialize and cache the Presidio AnalyzerEngine.

    Tries spaCy models in order of preference. Raises RuntimeError if none
    are installed.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:
        raise ImportError(
            "presidio-analyzer is required. pip install presidio-analyzer"
        ) from exc

    last_error: Optional[Exception] = None
    for model_name in _SPACY_MODELS:
        try:
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
            logger.info("Presidio initialized with spaCy model: %s", model_name)
            return analyzer
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load spaCy model %r: %s", model_name, exc)
            last_error = exc
            continue

    raise RuntimeError(
        "No spaCy model found. Install one with:\n"
        "  python -m spacy download en_core_web_sm\n"
        "  python -m spacy download en_core_web_lg  (recommended)"
    ) from last_error


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class MaskedPage:
    """Result of masking a single page."""
    page_number: int
    original_text: str     # text before masking
    masked_text: str       # text with PII replaced by tokens
    entities_found: int    # total PII entities detected (before filtering)
    entities_masked: int   # entities actually masked


# --------------------------------------------------------------------------- #
# Core masking function
# --------------------------------------------------------------------------- #

def mask_page(
    page_number: int,
    text: str,
    context: MaskingContext,
    language: str = "en",
) -> MaskedPage:
    """
    Detect and mask PII in a single page's text.

    Parameters
    ----------
    page_number : int
        1-indexed page number (for occurrence tracking).
    text : str
        Cleaned, normalised text from preprocessing.
    context : MaskingContext
        The document-level masking context. Updated in place.
    language : str
        Language code for Presidio. Default "en".

    Returns
    -------
    MaskedPage
        Contains the masked text and stats. Context is updated as a side-effect.
    """
    if not text.strip():
        return MaskedPage(
            page_number=page_number,
            original_text=text,
            masked_text=text,
            entities_found=0,
            entities_masked=0,
        )

    analyzer = _get_analyzer()

    # Run Presidio — detect all entity types it knows about
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=list(MASK_ENTITY_TYPES),   # only detect what we care about
        score_threshold=MIN_SCORE,
    )

    entities_found = len(results)
    entities_masked = 0

    # Sort detections RIGHT-TO-LEFT so char positions stay valid during replacement
    results_sorted = sorted(results, key=lambda r: r.start, reverse=True)

    masked_text = text
    for result in results_sorted:
        if not should_mask(result.entity_type, result.score):
            continue

        original_value = text[result.start:result.end]
        token = context.get_or_create_token(
            original_value=original_value,
            entity_type=result.entity_type,
            score=result.score,
            page_number=page_number,
            char_start=result.start,
            char_end=result.end,
        )
        masked_text = masked_text[:result.start] + token + masked_text[result.end:]
        entities_masked += 1

    if entities_masked > 0:
        logger.info("Page %d: masked %d/%d entities", page_number, entities_masked, entities_found)
    else:
        logger.debug("Page %d: no entities masked (%d detected, none in mask list)", page_number, entities_found)

    return MaskedPage(
        page_number=page_number,
        original_text=text,
        masked_text=masked_text,
        entities_found=entities_found,
        entities_masked=entities_masked,
    )


def mask_document(
    pages: list,   # List[PreprocessedPage]
    doc_id: str,
    save_mask_file: bool = True,
) -> tuple[list[MaskedPage], MaskingContext]:
    """
    Mask an entire document (all preprocessed pages).

    Parameters
    ----------
    pages : List[PreprocessedPage]
        Output from preprocessing.preprocess().
    doc_id : str
        Unique document identifier (used for mask file naming).
    save_mask_file : bool
        Whether to persist the mask mapping to disk. Default True.

    Returns
    -------
    (List[MaskedPage], MaskingContext)
        Masked pages + the context object (needed for rehydration).
    """
    from preprocessing.pipeline import PreprocessedPage

    context = MaskingContext(doc_id=doc_id)
    masked_pages: list[MaskedPage] = []

    for page in pages:
        text = page.text  # best available text from preprocessing
        masked = mask_page(
            page_number=page.page_number,
            text=text,
            context=context,
        )
        masked_pages.append(masked)

    logger.info(
        "Document masking complete: %d pages, %d unique entities masked.",
        len(masked_pages),
        context.total_masked(),
    )

    if save_mask_file and context.has_masked_entities():
        context.save()
    elif save_mask_file:
        logger.info("No sensitive entities found — mask file not written.")

    return masked_pages, context
