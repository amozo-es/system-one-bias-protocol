#!/usr/bin/env python3
"""Exclusive analysis for the labelswap experiment.

Reads ONLY one directory of labelswap arm logs: {model}__bbq_ses_ls_{arm}.jsonl
with meta.rendering set per arm. The cluster arms are produced by
cluster/eval_systemone.py --letter-order; the Jev arm by
src/labelswap/run_jev_ls.py. Compares arms at TEXT level — texts and
positions are fixed across arms, only key attachment varies, so
same-text == text-following.

Metrics per model (arms joined on item_id; identical sets asserted):
  same_text K-vs-X : text-following rate (1 - letter-following)
  same_text K-vs-M , TVD of choice-text distributions per pair
Verdict bands (descriptive, not a test):
  >= .98 Von-like (definitions decide), .90-.98 mixed, < .90 Laya-like.

Writes labelswap_results.csv into the output dir and prints a table.
No bias scores. No contact with sweep_tables.csv or the main results/.

Usage: python analysis/analyze_labelswap.py [--logs DIR] [--out-dir DIR]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "execution" / "cluster" / "labelswap" / "results"
ARMS = ("canonical", "exchanged", "meaningful")


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def load_arm(model, arm):
    path = RESULTS / f"{model}__bbq_ses_ls_{arm}.jsonl"
    if not path.exists():
        fail(f"missing file: {path.name}")
    picks, errors = {}, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("rendering", r.get("meta", {}).get("rendering")) != arm:
            fail(f"{path.name}: record rendering != {arm}")
        ans = r.get("answers")
        try:
            key = ans["answer"]["choice"]
            text = r["request"]["questions"]["answer"]["criteria"][key]
        except (TypeError, KeyError, AttributeError):
            errors += 1
            picks[r["item_id"]] = None
            continue
        picks[r["item_id"]] = text
    return picks, errors


def tvd(a, b):
    sup = set(a) | set(b)
    ca, cb = Counter(a), Counter(b)
    na, nb = len(a), len(b)
    return 0.5 * sum(abs(ca[t] / na - cb[t] / nb) for t in sup)


def verdict(same):
    if same >= 0.98:
        return "Von-like"
    if same >= 0.90:
        return "mixed"
    return "Laya-like"


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(RESULTS),
                    help="dir with {model}__bbq_ses_ls_{arm}.jsonl")
    ap.add_argument("--out-dir", default=str(HERE.parent / "results"), help="CSV output dir")
    args = ap.parse_args()
    RESULTS = Path(args.logs)
    if not RESULTS.exists():
        fail(f"no such directory: {RESULTS}")
    models = sorted({p.name.split("__bbq_ses_ls_")[0]
                     for p in RESULTS.glob("*__bbq_ses_ls_canonical.jsonl")})
    if not models:
        fail("no canonical arm files found")
    rows = []
    print(f"{'model':<12}{'pair':<6}{'n':>7}{'same_text':>11}{'tvd':>8}  verdict")
    for model in models:
        arms = {}
        for arm in ARMS:
            picks, errors = load_arm(model, arm)
            arms[arm] = picks
            rows.append({"model": model, "pair": f"err_{arm}", "n": len(picks),
                         "same": None, "tvd": None, "errors": errors})
        ids = [set(a) for a in arms.values()]
        if not (ids[0] == ids[1] == ids[2]):
            fail(f"{model}: arm item sets differ "
                 f"({sorted(len(i) for i in ids)})")
        n = len(ids[0])
        for pair in (("canonical", "exchanged"), ("canonical", "meaningful")):
            a, b = (arms[pair[0]], arms[pair[1]])
            ok = [(x, y) for x, y in
                  ((a[i], b[i]) for i in a) if x is not None and y is not None]
            same = (sum(1 for x, y in ok if x == y) / len(ok)) if ok else float("nan")
            d = tvd([x for x, _ in ok], [y for _, y in ok]) if ok else float("nan")
            tag = verdict(same) if pair[0] == "canonical" and pair[1] == "exchanged" else ""
            rows.append({"model": model, "pair": f"{pair[0][0]}vs{pair[1][0]}",
                         "n": n, "same": round(same, 4), "tvd": round(d, 4),
                         "errors": None, "verdict": tag})
            print(f"{model:<12}{pair[0][0]}vs{pair[1][0]:<5}{n:>7}"
                  f"{same:>11.4f}{d:>8.4f}  {tag}")
    out = Path(args.out_dir) / "labelswap_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("model,pair,n,same_text,tvd,errors,verdict\n")
        for r in rows:
            f.write(f"{r['model']},{r['pair']},{r['n']},{r.get('same')},"
                    f"{r.get('tvd')},{r.get('errors')},{r.get('verdict', '')}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
