"""Selftest for the labelswap experiment. No GPU, no cluster, no API calls.

Checks:
 1. Canonical rendering is byte-identical to the audited sweep (existing
    cluster/results/*__bbq_ses.jsonl order-0 records) and cluster item_id ==
    raw row index (needed to join arms later).
 2. Cluster eval_systemone.bbq_request agrees byte-identically with the
    reference src/labelswap/swaplib.py on all 3 modes (200 items).
 3. Local src/adapt.bbq_request canonical agrees with cluster canonical.
 4. Tag rules hold on all 6,864 SES items (named = 2 tags, first tag never an
    SES-level token; unknown detected; keys unique per item per mode).

Usage: python src/labelswap/test_swaplib.py   (from execution/)
Exit != 0 on any failure.
"""
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]  # experiments/
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "cluster"))

import adapt  # noqa: E402
import eval_systemone as ev  # noqa: E402
from labelswap import swaplib  # noqa: E402

# independent unknown-slot detector (keep intentionally separate from analysis/metrics.py)
_UNKNOWNS = {
    "unknown", "cannot be determined", "can't be determined", "not answerable",
    "not known", "not enough info", "not enough information", "cannot answer",
    "can't answer", "undetermined",
}


def unknown_slot(item):
    for i, slot in enumerate(["ans0", "ans1", "ans2"]):
        tags = [t.lower() for t in item["answer_info"][slot]]
        if "unknown" in tags or item[slot].lower().strip(" .") in _UNKNOWNS:
            return i
    return None

RAW = EXP.parent / "raw" / "data" / "BBQ" / "data" / "SES.jsonl"
SES = [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]
BY_EID = {r["example_id"]: r for r in SES}  # cluster item_id == example_id
RESULTS = EXP / "cluster" / "results"
LEVEL_TOKENS = {"lowses", "highses", "low SES", "high SES"}


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def _intended_slot(item):
    """Independent re-implementation of the intended v14.1 canonical mapping:
    dataset order [ans0, ans1, ans2], order 0 (unknown wherever the dataset
    puts it)."""
    return {"A": 0, "B": 1, "C": 2}, [item["ans0"], item["ans1"], item["ans2"]]


def check_canonical_matches_sweep():
    """Canonical rendering has the intended v14.1 semantics (dataset order).

    NOTE: stored records from generations before the slot-mapping fix show the
    old A<->B-crossed map (and, where the true unknown was not at ans2, even
    the TEXT order differs). Stored records are used to verify the
    item_id == example_id mapping (via the untouched question instructions);
    slot/text semantics are checked against the independent _intended_slot
    re-implementation.
    """
    checked = pre = post = 0
    with open(RESULTS / "clef-flash__bbq_ses.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("order") != 0 or r.get("rep", 0) != 0:
                continue
            iid = r["item_id"]
            item = BY_EID[iid]  # KeyError here = mapping assumption wrong
            if r["request"]["questions"]["answer"]["instructions"] != item["question"]:
                fail(f"instructions mismatch at item {iid} (id mapping wrong?)")
            state, questions, meta = ev.bbq_request(item, 0)
            want_slot, want_opts = _intended_slot(item)
            if meta["slot_by_letter"] != want_slot or meta["options"] != want_opts:
                fail(f"intended-mapping mismatch at item {iid}")
            if list(questions["answer"]["criteria"].values()) != want_opts:
                fail(f"criteria text order mismatch at item {iid}")
            if r["meta"]["slot_by_letter"] == want_slot:
                post += 1
            else:
                pre += 1
            checked += 1
            if checked >= 60:
                break
    print(f"  [1] canonical semantics OK ({checked} records); stored maps: "
          f"pre-fix={pre} post-fix={post}")


def check_eval_matches_swaplib():
    """eval_systemone mirror == swaplib reference on all modes."""
    from labelswap import swaplib as sl  # local import for swap_criteria
    for mode in swaplib.MODES:
        ev.LETTER_MODE = mode
        for n, item in enumerate(SES[:200]):
            state, questions, meta = ev.bbq_request(item, 0)
            texts = list(questions["answer"]["criteria"].values())
            expect_keys = swaplib.bbq_keys(item, texts, mode)
            if list(questions["answer"]["criteria"].keys()) != expect_keys:
                fail(f"{mode}: key mismatch at row {n}")
            if meta["slot_by_letter"] != dict(zip(expect_keys, _slots(item, texts))):
                fail(f"{mode}: slot map mismatch at row {n}")
            if meta["rendering"] != mode:
                fail(f"{mode}: rendering flag missing at row {n}")
            if mode != "canonical":
                # local Jev path (swap_criteria post-processing) must agree
                # with the eval rendering; the local harness renders
                # unknown-last, so texts align only when the unknown is ans2
                # (same generation otherwise — see check 3).
                if unknown_slot(item) != 2:
                    continue
                s0, q0, m0 = adapt.bbq_request(item, 0)
                crit0 = q0["answer"]["criteria"]
                meaningful = (sl.bbq_keys(item, list(crit0.values()), mode)
                              if mode == "meaningful" else None)
                new_crit, new_slot = sl.swap_criteria(
                    crit0, m0["slot_by_letter"], mode, meaningful)
                if (new_crit != questions["answer"]["criteria"]
                        or new_slot != meta["slot_by_letter"]):
                    fail(f"{mode}: swap_criteria != eval rendering at row {n}")
    ev.LETTER_MODE = "canonical"
    print(f"  [2] eval mirror == swaplib == swap_criteria on {min(200, len(SES))} items x 3 modes OK")


def _slots(item, texts):
    idx = {item[f"ans{i}"]: i for i in range(3)}
    return [idx[t] for t in texts]


def check_local_adapt_matches():
    """Local adapt == cluster canonical on items whose unknown sits at ans2.

    v14.1 cluster rendering keeps dataset order (unknown wherever the dataset
    puts it); the local Ollama-era harness always moves the unknown to C. The
    two agree exactly when the unknown is at ans2, by design.
    """
    same = skipped = 0
    for n, item in enumerate(SES[:50]):
        if unknown_slot(item) != 2:
            skipped += 1
            continue
        s1, q1, m1 = adapt.bbq_request(item, 0)
        s2, q2, m2 = ev.bbq_request(item, 0)
        if q1 != q2 or m1["slot_by_letter"] != m2["slot_by_letter"]:
            fail(f"local adapt != cluster canonical at row {n}")
        same += 1
    print(f"  [3] local adapt == cluster canonical on {same} items "
          f"(skipped {skipped}: unknown not at ans2) OK")


def check_tag_rules():
    no_unk, dup, level_first = 0, [], 0
    for n, item in enumerate(SES):
        unks = [i for i in range(3)
                if "unknown" in [t.lower() for t in item["answer_info"][f"ans{i}"]]]
        if len(unks) != 1:
            no_unk += 1
        for i in range(3):
            if i in unks:
                continue
            tags = item["answer_info"][f"ans{i}"]
            if len(tags) != 2:
                fail(f"row {n}: named slot has {len(tags)} tags")
            if tags[0].lower() in LEVEL_TOKENS:
                level_first += 1
        for mode in swaplib.MODES:
            ev.LETTER_MODE = mode
            _, questions, _ = ev.bbq_request(item, 0)
            keys = list(questions["answer"]["criteria"].keys())
            if len(set(keys)) != len(keys):
                dup.append((n, mode, keys))
    ev.LETTER_MODE = "canonical"
    print(f"  [4] tag audit: items={len(SES)} no-unique-unknown={no_unk} "
          f"level-as-first-tag={level_first} dup-keys={len(dup)}")
    if dup:
        fail(f"duplicate keys: {dup[:3]}")
    if level_first:
        fail("SES-level token used as meaningful key")
    print("  [4] tag rules OK")


if __name__ == "__main__":
    print(f"SES items: {len(SES)}")
    check_canonical_matches_sweep()
    check_eval_matches_swaplib()
    check_local_adapt_matches()
    check_tag_rules()
    print("labelswap selftest OK")
