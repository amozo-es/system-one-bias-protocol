"""Paired exact tests for cross-model flip-rate contrasts .

For each contrast, align the two models' BBQ items (both orders scored,
non-error), compute the per-item slot-flip indicator, and test H0: equal
flip rates on the shared items. The sign-flip permutation distribution over
discordant pairs is Binomial(n_discordant, 1/2); the exact two-sided test
(McNemar) is computed in log space. Holm correction across the whole family.

Usage: uv run python src/analyze_paired_tests.py [--out results/paired_tests.csv]
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets as ds
from analyze_sweep import BBQ_CATS, load_records, rebuild_slots

import analyze_sweep as sweep

HERE = Path(__file__).resolve().parent

CONTRASTS = [
    ("tev1-4b", "qwen3.5-4b"),
    ("tev1-0.8b", "qwen3.5-0.8b"),
    ("nimble-9b", "qwen3.5-9b"),
    ("qwen3.5-9b-nothink", "qwen3.5-9b"),
]


def flip_map(recs):
    """item_id -> True/False (both orders present, slot-level flip)."""
    by_item = {}
    for r in recs:
        if r.get("error") or not isinstance(r.get("answers"), dict):
            continue
        ans = r["answers"].get("answer") or {}
        letter = ans.get("choice")
        sbl = (r.get("meta") or {}).get("slot_by_letter") or {}
        slot = sbl.get(letter)
        if slot is None:
            continue
        by_item.setdefault(r["item_id"], {})[r["order"]] = slot
    return {i: v[0] != v[1] for i, v in by_item.items() if 0 in v and 1 in v}


def binom_two_sided(k, n):
    """Two-sided exact binomial p for k successes in n trials, p=1/2."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            for i in range(k + 1)]
    mx = max(logs)
    logsum = mx + math.log(sum(math.exp(v - mx) for v in logs))
    return min(1.0, 2.0 * math.exp(logsum - n * math.log(2)))


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, prev = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (m - rank) * pvals[i]))
        adj[i] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "results" / "paired_tests.csv"))
    ap.add_argument("--logs", default=str(sweep.CLUSTER),
                    help="decision-log dir (<model>__<dataset>.jsonl)")
    args = ap.parse_args()
    sweep.CLUSTER = Path(args.logs)
    assert abs(binom_two_sided(0, 10) - 2 / 1024) < 1e-12
    assert binom_two_sided(0, 1) == 1.0 and abs(binom_two_sided(0, 2) - 0.5) < 1e-12

    cache = {}

    def maps(model, name, cat):
        key = (model, name)
        if key not in cache:
            recs = load_records("cluster", model, name)
            if recs is None:
                cache[key] = {}
                return cache[key]
            items = {r["example_id"]: r for r in ds.load_bbq(cat)}
            rebuild_slots(recs, items)
            cache[key] = flip_map(recs)
        return cache[key]

    rows = []
    for a, b in CONTRASTS:
        for name, cat in BBQ_CATS.items():
            fa, fb = maps(a, name, cat), maps(b, name, cat)
            shared = sorted(set(fa) & set(fb))
            if not shared:
                continue
            va = [fa[i] for i in shared]
            vb = [fb[i] for i in shared]
            n10 = sum(1 for x, y in zip(va, vb) if x and not y)
            n01 = sum(1 for x, y in zip(va, vb) if y and not x)
            n = len(shared)
            ra, rb = sum(va) / n, sum(vb) / n
            rows.append(dict(contrast=f"{a} vs {b}", model_a=a, model_b=b, dataset=name,
                             n_paired=n, n_discordant=n10 + n01, rate_a=ra, rate_b=rb,
                             diff=ra - rb, p_exact=binom_two_sided(min(n10, n01), n10 + n01)))
    adj = holm([r["p_exact"] for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = p

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("contrast,dataset,n_paired,n_discordant,rate_a,rate_b,diff,p_exact,p_holm\n")
        for r in rows:
            f.write(f"{r['contrast']},{r['dataset']},{r['n_paired']},{r['n_discordant']},"
                    f"{r['rate_a']:.4f},{r['rate_b']:.4f},{r['diff']:+.4f},"
                    f"{r['p_exact']:.3g},{r['p_holm']:.3g}\n")
    print(f"wrote {out} ({len(rows)} tests)")
    if not rows:
        print("no paired items found (missing logs?)")
        return
    worst = max(r["p_holm"] for r in rows)
    sig = sum(1 for r in rows if r["p_holm"] < 0.05)
    for r in rows:
        tag = "sig" if r["p_holm"] < 0.05 else "n.s."
        print(f"  {r['contrast']:26s} {r['dataset']:26s} n={r['n_paired']:6d} "
              f"disc={r['n_discordant']:5d} {r['rate_a']:.3f} vs {r['rate_b']:.3f} "
              f"diff {r['diff']:+.3f} p_holm {r['p_holm']:.2g} {tag}")
    print(f"family: {sig}/{len(rows)} significant after Holm; max p_holm {worst:.3g}")


if __name__ == "__main__":
    main()
