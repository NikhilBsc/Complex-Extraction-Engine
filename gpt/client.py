"""
gpt/client.py
-------------
AI API client wrapper — currently configured for Groq.

Groq provides an OpenAI-compatible API (same Python SDK, same JSON format)
with FREE access to top-tier open-source models at high speed.

To switch provider later, only this file needs to change.
All extraction logic (extractor.py, prompts.py) remains untouched.

Provider: Groq  →  https://console.groq.com
Models:
  Primary  : llama-3.3-70b-versatile  (best accuracy, 128K context)
  Fallback : llama3-70b-8192          (stable, fast, 8K context)

Groq Free Tier limits (as of 2025):
  - 30 requests/minute
  - 14,400 tokens/minute
  - 500,000 tokens/day

Note: Set INTER_SEGMENT_DELAY = 2.5 in extractor.py to stay safe on free tier.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Groq model configuration
# --------------------------------------------------------------------------- #

PRIMARY_MODEL  = "openai/gpt-oss-120b"   # OpenAI's 120B parameter GPT model on Groq
FALLBACK_MODEL = "groq/compound"         # Groq compound fallback

GROQ_BASE_URL  = "https://api.groq.com/openai/v1"

# Retry config
MAX_RETRIES     = 3
INITIAL_BACKOFF = 2.0    # Groq rate limit window is 1 min, back off generously
BACKOFF_FACTOR  = 2.0

# Response config
MAX_RESPONSE_TOKENS = 4096
TEMPERATURE         = 0     # Deterministic — essential for extraction accuracy


# --------------------------------------------------------------------------- #
# Client factory
# --------------------------------------------------------------------------- #

def _get_client():
    """
    Build an OpenAI-compatible client pointed at Groq's endpoint.
    Reads GROQ_API_KEY from environment / .env file.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package required. Run: pip install openai") from exc

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found in environment.\n"
            "  1. Get a free key at: https://console.groq.com\n"
            "  2. Add to your .env file:\n"
            "       GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx"
        )

    return OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
    )


# --------------------------------------------------------------------------- #
# Core API call with retry
# --------------------------------------------------------------------------- #

def call_gpt(
    system_prompt: str,
    user_prompt: str,
    model: str = PRIMARY_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_RESPONSE_TOKENS,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Call the Groq API (OpenAI-compatible) with retry on rate limits.

    Returns
    -------
    (parsed_json, usage_dict)
    """
    client = _get_client()
    last_error: Optional[Exception] = None
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("Groq call attempt %d/%d model=%s", attempt, MAX_RETRIES, model)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},   # Groq supports JSON mode
            )

            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
            }

            # Parse JSON
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from the response
                import re
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                    except json.JSONDecodeError as e:
                        if attempt < MAX_RETRIES:
                            logger.warning("JSON parse failed attempt %d, retrying...", attempt)
                            time.sleep(backoff)
                            backoff *= BACKOFF_FACTOR
                            continue
                        raise ValueError(f"Groq returned invalid JSON after {MAX_RETRIES} attempts") from e
                else:
                    if attempt < MAX_RETRIES:
                        time.sleep(backoff)
                        backoff *= BACKOFF_FACTOR
                        continue
                    raise ValueError(f"No JSON found in Groq response: {content[:200]}")

            logger.info(
                "Groq call OK: model=%s tokens=%d (prompt=%d, completion=%d)",
                model, usage["total_tokens"],
                usage["prompt_tokens"], usage["completion_tokens"],
            )
            return parsed, usage

        except Exception as exc:
            err_str = str(exc).lower()

            # Rate limit → wait and retry
            if any(code in err_str for code in ["rate_limit", "429", "too many", "503", "timeout"]):
                wait = backoff + (attempt * 1.0)
                logger.warning(
                    "Rate limit hit (attempt %d/%d) — waiting %.0fs...",
                    attempt, MAX_RETRIES, wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                    backoff *= BACKOFF_FACTOR
                    last_error = exc
                    continue

            # Auth error → fail immediately
            elif any(code in err_str for code in ["401", "403", "invalid_api_key", "api key"]):
                raise RuntimeError(
                    f"Groq auth error: {exc}\n"
                    f"Check your GROQ_API_KEY in .env"
                ) from exc

            # Other errors → retry
            else:
                last_error = exc
                if attempt < MAX_RETRIES:
                    logger.warning("Unexpected error attempt %d: %s — retrying...", attempt, exc)
                    time.sleep(backoff)
                    backoff *= BACKOFF_FACTOR
                    continue

    raise RuntimeError(
        f"Groq API call failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def call_gpt_with_fallback(
    system_prompt: str,
    user_prompt: str,
) -> Tuple[Dict[str, Any], Dict[str, int], str]:
    """
    Try primary model first, then fallback models if it fails.
    """
    candidate_models = [
        "openai/gpt-oss-120b",    # Primary 120B model
        "qwen/qwen3.8-27b",       # Qwen 27B strong JSON extraction
        "groq/compound",          # Groq compound model
        "openai/gpt-oss-20b",     # 20B fast fallback
    ]

    last_exc = None
    for model_name in candidate_models:
        try:
            parsed, usage = call_gpt(system_prompt, user_prompt, model=model_name)
            return parsed, usage, model_name
        except Exception as exc:
            logger.warning("Model %s failed: %s — trying next fallback...", model_name, exc)
            last_exc = exc

    raise RuntimeError(f"All Groq models failed. Last error: {last_exc}")

