"""Label-swap rendering for BBQ (Azizi-style exchanged/aligned arms).

Reference implementation. Pure functions: no GPU, no cluster imports, no I/O.
Used by the local Jev path; cluster eval_systemone.py mirrors `bbq_keys`
inline (marked LABELSWAP) and test_swaplib.py asserts both agree byte-identically.
Arms (order fixed, texts+positions fixed — only the key channel varies):
  canonical  A/B/C (arbitrary letters; current protocol)
  exchanged  B/A/C, unknown key fixed (Azizi's arbitrary-exchanged control:
             pure letter-following test)
  meaningful group-word keys from answer_info tags + 'unknown'
             (Azizi's aligned condition: label-meaning salience test)

Design constraint: this test is
only valid where the model actually sees the keys. For hf_* cluster kinds the
letter is positional-by-construction (to_letter_task / HFTypedModel in
eval_systemone.py), so swapping
keys would change bookkeeping, not model input — a fake experiment.
"""
from __future__ import annotations

MODES = ("canonical", "exchanged", "meaningful")
CANONICAL_KEYS = ["A", "B", "C"]
UNKNOWN_KEY = "unknown"


def meaningful_key(item: dict, text: str) -> str:
    """Short group-word key for one displayed answer text.

    Rule: first answer_info tag of the matching ans slot; 'unknown' when the
    slot carries the unknown tag. SES items always have 2-tag named slots
    (group word + SES level) — verified in test_swaplib.py.
    """
    for i in range(3):
        if item[f"ans{i}"] == text:
            tags = item["answer_info"][f"ans{i}"]
            lowered = [t.lower() for t in tags]
            if "unknown" in lowered:
                return UNKNOWN_KEY
            return tags[0]
    raise KeyError(f"text not found in item: {text!r}")


def bbq_keys(item: dict, display_texts: list, mode: str) -> list:
    """Keys aligned with display_texts (unknown last)."""
    if mode == "canonical":
        return CANONICAL_KEYS[: len(display_texts)]
    if mode == "exchanged":
        keys = CANONICAL_KEYS[: len(display_texts)]
        keys[0], keys[1] = keys[1], keys[0]
        return keys
    if mode == "meaningful":
        return [meaningful_key(item, t) for t in display_texts]
    raise ValueError(f"unknown mode: {mode!r}")


def swap_criteria(criteria: dict, slot_by_key: dict, mode: str,
                  meaningful: list | None = None) -> tuple[dict, dict]:
    """Re-key a canonical {letter: text} criteria dict (local Jev path).

    Returns (new_criteria, new_slot_map). Texts and their display order are
    untouched; only key attachment changes. `meaningful` is the precomputed
    key list for mode == 'meaningful'.
    """
    texts = list(criteria.values())
    if mode == "canonical":
        keys = CANONICAL_KEYS[: len(texts)]
    elif mode == "exchanged":
        keys = CANONICAL_KEYS[: len(texts)]
        keys[0], keys[1] = keys[1], keys[0]
    elif mode == "meaningful":
        assert meaningful is not None and len(meaningful) == len(texts)
        keys = list(meaningful)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    text_slot = {criteria[k]: slot_by_key[k] for k in criteria}  # text -> slot
    new_criteria = dict(zip(keys, texts))
    new_slot = {k: text_slot[t] for k, t in new_criteria.items()}
    assert len(set(keys)) == len(keys), f"duplicate keys: {keys}"
    return new_criteria, new_slot
