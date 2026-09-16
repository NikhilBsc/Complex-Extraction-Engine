"""
masking/rehydrator.py
---------------------
Post-GPT rehydration: replace mask tokens in GPT output with original values.

After GPT processes masked text and returns a structured JSON of field-value
pairs, the values may contain mask tokens (e.g. "[US_SSN_1]").

This module restores those tokens to their original values using the
MaskingContext that was created during the masking phase.

It also handles edge cases:
  - Token appears inside a longer value string
  - Token appears multiple times in a value
  - Value contains no tokens (returned as-is)
  - Unknown token (logged as warning, returned as-is)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Union

from masking.context import MaskingContext

logger = logging.getLogger(__name__)

# Pattern to find any mask token in text: [ENTITY_TYPE_N]
_TOKEN_PATTERN = re.compile(r"\[([A-Z_]+)_(\d+)\]")


def rehydrate_value(value: Any, context: MaskingContext) -> Any:
    """
    Restore mask tokens in a single field value.

    Handles:
      - str: token replacement
      - list: element-wise replacement
      - dict: value-wise replacement
      - anything else: returned unchanged

    Parameters
    ----------
    value : Any
        The value as returned by GPT (may or may not contain tokens).
    context : MaskingContext
        The masking context holding token → original_value mappings.

    Returns
    -------
    Any
        Value with all tokens replaced by their original values.
    """
    if isinstance(value, str):
        return _rehydrate_string(value, context)
    elif isinstance(value, list):
        return [rehydrate_value(item, context) for item in value]
    elif isinstance(value, dict):
        return {k: rehydrate_value(v, context) for k, v in value.items()}
    else:
        return value


def _rehydrate_string(text: str, context: MaskingContext) -> str:
    """Replace all mask tokens found in a string with their original values."""
    def replace_token(match: re.Match) -> str:
        token = match.group(0)  # full match e.g. [US_SSN_1]
        entry = context.get_entry(token)
        if entry is None:
            logger.warning("Unknown mask token in GPT output: %r", token)
            return token  # leave as-is; do not guess
        return entry.original_value

    return _TOKEN_PATTERN.sub(replace_token, text)


def rehydrate_fields(
    fields: List[Dict[str, Any]],
    context: MaskingContext,
) -> List[Dict[str, Any]]:
    """
    Rehydrate a list of field-value dicts returned by GPT.

    Expected input format (from GPT extraction):
        [
          {"field": "Client Name", "value": "Acme Corp", ...},
          {"field": "SSN", "value": "[US_SSN_1]", ...},
          ...
        ]

    Returns the same list with all token values restored.

    Parameters
    ----------
    fields : List[Dict[str, Any]]
        GPT extraction output.
    context : MaskingContext
        Masking context from the same document session.

    Returns
    -------
    List[Dict[str, Any]]
        Fields with original values restored.
    """
    rehydrated = []
    for item in fields:
        new_item = dict(item)
        # Rehydrate the 'value' key — everything else stays as-is
        if "value" in new_item:
            new_item["value"] = rehydrate_value(new_item["value"], context)
        # Also rehydrate 'evidence' / 'source_text' snippets if present
        for key in ("evidence", "source_text", "raw_value"):
            if key in new_item:
                new_item[key] = rehydrate_value(new_item[key], context)
        rehydrated.append(new_item)

    tokens_found = sum(
        1 for item in fields
        if "value" in item and _TOKEN_PATTERN.search(str(item["value"]))
    )
    if tokens_found:
        logger.info("Rehydrated %d field values containing mask tokens.", tokens_found)

    return rehydrated
