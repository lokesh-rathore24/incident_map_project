"""Classify complaint descriptions into incident categories using OpenAI LLM."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Persistent classification cache  (description text -> category)
# ---------------------------------------------------------------------------
_CACHE_PATH = Path("classification_cache.json")


def _load_classification_cache() -> Dict[str, str]:
    """Load the classification cache from disk."""
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt classification cache – starting fresh.")
    return {}


def _save_classification_cache(cache: Dict[str, str]) -> None:
    """Persist the classification cache to disk."""
    _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_desc(text: str) -> str:
    """Normalise a description for cache-key purposes."""
    return " ".join(text.lower().split())

# The 20 valid categories – must stay in sync with EMOJI_MAPPING in app.py.
VALID_CATEGORIES: List[str] = [
    "Cyber Crime (other than financial fraud)",
    "Cyber Financial Fraud",
    "Other IPC/BNS Crimes",
    "Miscellaneous",
    "Crime Against SC/ST",
    "Crime against Children",
    "Matrimonial Dispute",
    "Illegal Immigration",
    "Job Related Fraud",
    "Property/Land Dispute",
    "Other Economic Offence",
    "Noise Pollution",
    "Runaway Couples",
    "Security Threat",
    "Minor Accident",
    "Crime Against Women",
    "Corruption/Demand of Bribe",
    "Lost Property",
    "Hurt",
    "Intimidation",
]

_CATEGORY_SET = {c.lower(): c for c in VALID_CATEGORIES}

FALLBACK_CATEGORY = "Miscellaneous"


def _normalize_category(raw: str) -> str:
    """Return the canonical category name, or FALLBACK_CATEGORY if unrecognised."""
    stripped = raw.strip().strip('"').strip("'")
    match = _CATEGORY_SET.get(stripped.lower())
    if match:
        return match
    # Fuzzy: try partial match (handles minor typos from the LLM)
    for key, canonical in _CATEGORY_SET.items():
        if key in stripped.lower() or stripped.lower() in key:
            return canonical
    return FALLBACK_CATEGORY


def _build_prompt(descriptions: List[str]) -> str:
    """Build the classification prompt for a batch of descriptions."""
    category_list = "\n".join(f"- {c}" for c in VALID_CATEGORIES)
    numbered = "\n".join(f"{i + 1}. \"{d}\"" for i, d in enumerate(descriptions))

    return f"""You are a crime/incident classifier for an Indian police complaint system.

Given the following complaint descriptions, classify each one into EXACTLY one of these categories:
{category_list}
Everything related to marpeet, ladai , jagda . fighting should be classified as "Hurt".

Rules:
- Return ONLY a valid JSON array of strings, one category per description, in the same order.
- Each string MUST be exactly one of the categories listed above (case-sensitive).
- If a description is empty, unclear, or does not fit any category, use "Miscellaneous".
- Do NOT include any explanation, markdown formatting, or extra text — just the JSON array.

Descriptions:
{numbered}"""


# Retry config for rate-limit (429) errors
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECS = 20.0


def classify_batch(
    descriptions: List[str],
    api_key: str,
    existing_classes: Optional[List[str]] = None,
    cache: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Classify a batch of descriptions using OpenAI.

    Checks *cache* first and only sends uncached descriptions to the API.
    New results are written back into *cache* (caller should persist it).
    On final failure the *existing_classes* are preserved.
    """
    if not descriptions:
        return []

    if cache is None:
        cache = {}

    fallback = []
    if existing_classes:
        for val in existing_classes:
            val_str = str(val).strip()
            # If original value is empty/whitespace or literal "None"/"nan"/"null", fall back to Miscellaneous
            if not val_str or val_str.lower() in ("none", "nan", "null", "<none>"):
                fallback.append(FALLBACK_CATEGORY)
            else:
                norm = _normalize_category(val_str)
                # If normalized result is not a valid category, use FALLBACK_CATEGORY
                if norm in VALID_CATEGORIES:
                    fallback.append(norm)
                else:
                    fallback.append(FALLBACK_CATEGORY)
    else:
        fallback = [FALLBACK_CATEGORY] * len(descriptions)

    results: List[Optional[str]] = [None] * len(descriptions)

    # --- Check cache first ---
    uncached_indices: List[int] = []
    uncached_descs: List[str] = []
    for i, desc in enumerate(descriptions):
        key = _norm_desc(desc)
        if not key:
            results[i] = fallback[i] if i < len(fallback) else FALLBACK_CATEGORY
            continue
        cached = cache.get(key)
        if cached:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_descs.append(desc)

    if not uncached_descs:
        msg = f"ℹ️ All {len(descriptions)} valid descriptions found in cache (or empty). Skipping OpenAI API call."
        logger.info(msg)
        print(msg, flush=True)
        return [r if r is not None else FALLBACK_CATEGORY for r in results]

    # --- Call API for uncached descriptions ---
    msg = f"🚀 Calling OpenAI API for {len(uncached_descs)} uncached descriptions..."
    logger.info(msg)
    print(msg, flush=True)
    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(uncached_descs)

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a crime/incident classifier. Respond only with a JSON array."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            raw_text = response.choices[0].message.content.strip()

            # Strip markdown code fences if present (```json ... ```)
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed: list = json.loads(raw_text)

            if not isinstance(parsed, list):
                raise ValueError("Response is not a JSON array")

            api_results = [_normalize_category(str(item)) for item in parsed]
            # Pad if needed
            while len(api_results) < len(uncached_descs):
                api_results.append(FALLBACK_CATEGORY)

            for j, idx in enumerate(uncached_indices):
                cat = api_results[j] if j < len(api_results) else FALLBACK_CATEGORY
                results[idx] = cat
                cache[_norm_desc(uncached_descs[j])] = cat

            return [r if r is not None else FALLBACK_CATEGORY for r in results]

        except Exception as exc:
            last_exc = exc
            is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                wait = _INITIAL_BACKOFF_SECS * (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d). Retrying in %.0fs...",
                    attempt + 1, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
            else:
                break

    logger.warning("OpenAI classification batch failed after %d attempts: %s", _MAX_RETRIES, last_exc)
    # Fill uncached slots with fallback
    for idx in uncached_indices:
        if results[idx] is None:
            results[idx] = fallback[idx] if idx < len(fallback) else FALLBACK_CATEGORY
    return [r if r is not None else FALLBACK_CATEGORY for r in results]


def classify_dataframe(
    df: pd.DataFrame,
    api_key: str,
    batch_size: int = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Classify all rows in *df* using OpenAI and update ``Class of Incident``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Complaint Description`` column.
    api_key : str
        OpenAI API key.
    batch_size : int
        Number of rows per LLM call (default 50).
    progress_callback : callable, optional
        Called as ``progress_callback(batches_done, total_batches)`` after each
        batch so the UI can update a progress bar.

    Returns
    -------
    pd.DataFrame
        The input dataframe with ``Class of Incident`` updated in-place.
    """
    out = df.copy()

    if "Complaint Description" not in out.columns:
        return out

    cache = _load_classification_cache()

    descriptions = out["Complaint Description"].fillna("").astype(str).tolist()
    existing_classes = (
        out["Class of Incident"].fillna("").astype(str).tolist()
        if "Class of Incident" in out.columns
        else [""] * len(out)
    )

    total_batches = (len(descriptions) + batch_size - 1) // batch_size
    all_categories: List[str] = []

    print(f"🤖 Starting AI classification of {len(descriptions)} rows in {total_batches} batches...", flush=True)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_descs = descriptions[start:end]
        batch_existing = existing_classes[start:end]
        categories = classify_batch(batch_descs, api_key, existing_classes=batch_existing, cache=cache)
        all_categories.extend(categories)

        if progress_callback:
            progress_callback(batch_idx + 1, total_batches)

    # Persist cache after all batches
    _save_classification_cache(cache)

    # For rows where the original description was empty, keep existing class (if valid)
    for i, desc in enumerate(descriptions):
        if not desc.strip():
            orig = str(existing_classes[i]).strip()
            if not orig or orig.lower() in ("none", "nan", "null", "<none>"):
                norm = FALLBACK_CATEGORY
            else:
                norm = _normalize_category(orig)
            all_categories[i] = norm

    out["Class of Incident"] = all_categories
    print(f"✅ Finished AI classification of {len(descriptions)} rows.", flush=True)
    return out
