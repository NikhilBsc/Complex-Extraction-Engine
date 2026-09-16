"""
masking/context.py
------------------
MaskingContext — the authoritative in-session store for all masking operations.

Responsibilities
----------------
  1. Maintain an in-memory mapping:  mask_token  → MaskEntry
  2. Maintain a reverse lookup:      original_value → mask_token
     (ensures the SAME value always gets the SAME token across pages)
  3. Maintain per-entity-type counters so tokens are numbered sequentially
  4. Persist the full mapping to a protected JSON file (audit / authorizer access)
  5. Provide rehydration: given masked text, restore original values

Access model
------------
  - In-pipeline access: via the MaskingContext object held in memory.
    Only the code that created the context can use it.
  - Authorizer access: via the JSON file at masks/<doc_id>_masks.json.
    No other module should read this file directly.

The JSON file is written to the masks/ directory (created automatically).
For the POC it is not encrypted, but the path and content are documented as
requiring restricted access in a production system.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from masking.entities import make_token

logger = logging.getLogger(__name__)

# Default directory for mask files (relative to project root)
_DEFAULT_MASKS_DIR = Path(__file__).resolve().parents[1] / "masks"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class MaskOccurrence:
    """One location where the masked value appeared in the document."""
    page_number: int
    char_start: int
    char_end: int


@dataclass
class MaskEntry:
    """Full record for a single masked entity."""
    mask_token: str           # e.g. "[US_SSN_1]"
    original_value: str       # the real sensitive value
    entity_type: str          # Presidio entity type
    score: float              # detection confidence (0.0 – 1.0)
    first_seen_page: int      # page where it was first encountered
    occurrences: List[MaskOccurrence] = field(default_factory=list)

    def add_occurrence(self, page_number: int, char_start: int, char_end: int) -> None:
        self.occurrences.append(MaskOccurrence(page_number, char_start, char_end))


# --------------------------------------------------------------------------- #
# MaskingContext
# --------------------------------------------------------------------------- #

class MaskingContext:
    """
    Per-document masking session.

    Create one instance per document; do not share across documents.

    Parameters
    ----------
    doc_id : str
        Unique identifier for the document being processed.
        Used as the filename stem for the masks JSON file.
    masks_dir : Path | None
        Directory to write the JSON file. Defaults to <project_root>/masks/.
    """

    def __init__(self, doc_id: str, masks_dir: Optional[Path] = None) -> None:
        self.doc_id = doc_id
        self.masks_dir = Path(masks_dir) if masks_dir else _DEFAULT_MASKS_DIR
        self.created_at: str = datetime.now(timezone.utc).isoformat()

        # Primary store: token → entry
        self._token_map: Dict[str, MaskEntry] = {}

        # Reverse lookup: original_value → token (for cross-page consistency)
        self._value_map: Dict[str, str] = {}

        # Per-entity-type counter
        self._counters: Dict[str, int] = {}

        logger.info("MaskingContext created for doc_id=%r", doc_id)

    # ----------------------------------------------------------------------- #
    # Token management
    # ----------------------------------------------------------------------- #

    def get_or_create_token(
        self,
        original_value: str,
        entity_type: str,
        score: float,
        page_number: int,
        char_start: int,
        char_end: int,
    ) -> str:
        """
        Return the mask token for `original_value`.

        If this value has been seen before (on any page), returns the existing
        token — guaranteeing cross-page consistency.

        If it's new, creates a new sequential token, stores the entry, and
        returns the token.
        """
        # Normalise value for consistent lookup
        norm_value = original_value.strip()

        if norm_value in self._value_map:
            # Already masked — reuse token, just log the occurrence
            token = self._value_map[norm_value]
            entry = self._token_map[token]
            entry.add_occurrence(page_number, char_start, char_end)
            logger.debug(
                "Reused token %s for value %r on page %d",
                token, norm_value, page_number
            )
            return token

        # New entity — mint a fresh token
        self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
        token = make_token(entity_type, self._counters[entity_type])

        entry = MaskEntry(
            mask_token=token,
            original_value=norm_value,
            entity_type=entity_type,
            score=round(score, 4),
            first_seen_page=page_number,
        )
        entry.add_occurrence(page_number, char_start, char_end)

        self._token_map[token] = entry
        self._value_map[norm_value] = token

        logger.info(
            "New mask: %s → %r (type=%s, score=%.2f, page=%d)",
            token, norm_value, entity_type, score, page_number,
        )
        return token

    # ----------------------------------------------------------------------- #
    # Rehydration
    # ----------------------------------------------------------------------- #

    def rehydrate(self, text: str) -> str:
        """
        Replace all mask tokens in `text` with their original values.

        Parameters
        ----------
        text : str
            Text containing mask tokens such as [US_SSN_1].

        Returns
        -------
        str
            Text with original values restored.
        """
        for token, entry in self._token_map.items():
            text = text.replace(token, entry.original_value)
        return text

    # ----------------------------------------------------------------------- #
    # Accessors
    # ----------------------------------------------------------------------- #

    def get_entry(self, token: str) -> Optional[MaskEntry]:
        return self._token_map.get(token)

    def all_entries(self) -> List[MaskEntry]:
        return list(self._token_map.values())

    def total_masked(self) -> int:
        return len(self._token_map)

    def has_masked_entities(self) -> bool:
        return bool(self._token_map)

    def summary(self) -> Dict[str, int]:
        """Return count of masked entities per entity type."""
        counts: Dict[str, int] = {}
        for entry in self._token_map.values():
            counts[entry.entity_type] = counts.get(entry.entity_type, 0) + 1
        return counts

    # ----------------------------------------------------------------------- #
    # Persistence (authorizer file)
    # ----------------------------------------------------------------------- #

    def save(self) -> Path:
        """
        Write the full mask mapping to a JSON file in the masks/ directory.

        Returns the path of the written file.

        NOTE: This file contains original (sensitive) values. In production
        it must be access-controlled. For the POC it is written to disk
        for audit and disaster-recovery purposes.
        """
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.masks_dir / f"{self.doc_id}_masks.json"

        payload = {
            "doc_id": self.doc_id,
            "created_at": self.created_at,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "authorized_by": "system",
            "total_masked": self.total_masked(),
            "summary": self.summary(),
            "entries": [
                {
                    **{k: v for k, v in asdict(entry).items()},
                }
                for entry in self._token_map.values()
            ],
        }

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        # Attempt to restrict file permissions (best-effort on Windows)
        try:
            os.chmod(out_path, 0o600)
        except OSError:
            pass  # Windows may not support this; log silently

        logger.info("Mask file saved: %s (%d entries)", out_path, self.total_masked())
        return out_path

    @classmethod
    def load(cls, doc_id: str, masks_dir: Optional[Path] = None) -> "MaskingContext":
        """
        Load a previously saved MaskingContext from its JSON file.

        Only for authorizer / debugging use — not for normal pipeline flow.
        """
        masks_dir = Path(masks_dir) if masks_dir else _DEFAULT_MASKS_DIR
        in_path = masks_dir / f"{doc_id}_masks.json"

        if not in_path.exists():
            raise FileNotFoundError(f"Mask file not found: {in_path}")

        with open(in_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        ctx = cls(doc_id=doc_id, masks_dir=masks_dir)
        ctx.created_at = payload.get("created_at", ctx.created_at)

        for raw in payload.get("entries", []):
            entry = MaskEntry(
                mask_token=raw["mask_token"],
                original_value=raw["original_value"],
                entity_type=raw["entity_type"],
                score=raw["score"],
                first_seen_page=raw["first_seen_page"],
                occurrences=[
                    MaskOccurrence(**occ) for occ in raw.get("occurrences", [])
                ],
            )
            ctx._token_map[entry.mask_token] = entry
            ctx._value_map[entry.original_value] = entry.mask_token

            etype = entry.entity_type
            num = int(entry.mask_token.split("_")[-1].rstrip("]"))
            ctx._counters[etype] = max(ctx._counters.get(etype, 0), num)

        logger.info("MaskingContext loaded from %s (%d entries)", in_path, ctx.total_masked())
        return ctx

    def __repr__(self) -> str:
        return (
            f"<MaskingContext doc_id={self.doc_id!r} "
            f"total_masked={self.total_masked()} "
            f"summary={self.summary()}>"
        )
