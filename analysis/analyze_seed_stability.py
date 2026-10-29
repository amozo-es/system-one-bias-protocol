"""Stability under reseeding: results/ (seed 42) vs results_seedN/.

Joins records by (item_id, order) after canonicalizing by content
(rebuild_slots). Per model x cell x directory pair:
  - choice: % of items whose canonical slot changes; mean |dp| per slot
  - noul:   % of items crossing 0.5; mean |dp|
  - flip rate (order_flips) per directory and delta in percentage points

With --tests, for the seed pair A vs B:
  - exact McNemar on the per-item flip indicator (paired)
  - s_amb/s_dis delta with paired item-level bootstrap (with replacement)
  - Holm within each family (flips; scores)

Usage:
  python analyze_seed_stability.py --seeds 7,13
  python analyze_seed_stability.py --seeds 7,13 --tests
  python analyze_seed_stability.py --selftest   # base vs base => all zeros
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets as ds
from metrics import order_flips
from analyze_sweep import BBQ_CATS, MODELS, norm_record, rebuild_slots

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "execution" / "cluster" / "results"

REPEAT_MODELS = ["laya", "von", "tev1-4b", "tev1-0.8b", "qwen3.5-0.8b", "qwen3.5-4b",
                 "qwen3.5-9b-nothink", "nimble-9b", "clef", "clef-flash"]

CELLS = list(BBQ_CATS) + ["crows", "silverspoon"]
SLUG = {d: s for d, _s, s in MODELS}
_ITEMS = {}


def items_for(cell):
    if cell not in _ITEMS:
        _ITEMS[cell] = {r["example_id"]: r for r in ds.load_bbq(BBQ_CATS[cell])}
    return _ITEMS[cell]


def load_dir(base, slug, cell):
    f = Path(base) / f"{slug}__{cell}.jsonl"
    if not f.exists():
        return None
    recs = [norm_record(json.loads(l)) for l in f.read_text().splitlines() if l.strip()]
    if cell in BBQ_CATS:
        rebuild_slots(recs, items_for(cell))
    return recs


def extract_values(recs, cell):
    """(item_id, order) -> ('choice', slot, {slot: p}) | ('noul', p, {})."""
    vals = {}
    for r in recs:
        if r.get("error"):
            continue
        key = (int(r["item_id"]), int(r["order"]))
        if cell in BBQ_CATS:
            a = (r.get("answers") or {}).get("answer") or {}
            sbl = (r.get("meta") or {}).get("slot_by_letter") or {}
            letter = a.get("choice")
            if letter not in sbl:
                continue
            pslot = {sbl[l]: p for l, p in (a.get("probabilities") or {}).items() if l in sbl}
            vals[key] = ("choice", sbl[letter], pslot)
        elif cell == "crows":
            v = ((r.get("answers") or {}).get("stereotype") or {}).get("noul")
            if v is not None:
                vals[key] = ("noul", v, {})
        else:
            v = ((r.get("answers") or {}).get("justified") or {}).get("noul")
            if v is not None:
                vals[key] = ("noul", v, {})
    return vals


def compare(vals_a, vals_b):
    keys = sorted(set(vals_a) & set(vals_b))
    if not keys:
        return None
    kind = vals_a[keys[0]][0]
    if kind == "choice":
        changed = sum(1 for k in keys if vals_a[k][1] != vals_b[k][1])
        dps = [abs(p - vals_b[k][2][s])
               for k in keys for s, p in vals_a[k][2].items() if s in vals_b[k][2]]
        dp = sum(dps) / len(dps) if dps else 0.0
    else:
        changed = sum(1 for k in keys if (vals_a[k][1] >= 0.5) != (vals_b[k][1] >= 0.5))
        dp = sum(abs(vals_a[k][1] - vals_b[k][1]) for k in keys) / len(keys)
    return len(keys), changed / len(keys), dp


def compare_dirs(base_a, base_b, seed_a, seed_b, csv, selftest=False):
    for model in REPEAT_MODELS:
        slug = SLUG[model]
        for cell in CELLS:
            ra = load_dir(base_a, slug, cell)
            rb = load_dir(base_b, slug, cell)
            if ra is None or rb is None:
                continue
            cmp = compare(extract_values(ra, cell), extract_values(rb, cell))
            if cmp is None:
                continue
            n, changed_pct, dp = cmp
            flip_a = flip_b = None
            if cell in BBQ_CATS or cell == "crows":
                try:
                    fa, fb = order_flips(ra), order_flips(rb)
                    if fa["n"] and fb["n"]:
                        flip_a, flip_b = fa["rate"], fb["rate"]
                except KeyError:
                    pass
            row = [model, cell, f"{seed_a}->{seed_b}", n, f"{changed_pct:.5f}", f"{dp:.5f}"]
            row += ["" if flip_a is None else f"{flip_a:.5f}",
                    "" if flip_b is None else f"{flip_b:.5f}",
                    "" if flip_a is None else f"{(flip_b - flip_a) * 100:+.3f}"]
            csv.append(",".join(str(x) for x in row))
            print(f"  {model:18s} {cell:24s} n={n:<6d} changed={changed_pct*100:6.3f}% "
                  f"|dp|={dp:.5f}" + ("" if flip_a is None else
                                      f" flip {flip_a:.3f}->{flip_b:.3f} ({(flip_b-flip_a)*100:+.3f}pp)"))
            if selftest:
                assert changed_pct == 0.0 and dp == 0.0, f"selftest failed on {model} {cell}"


def flip_indicators(recs):
    by_item = {}
    for rec in recs:
        if rec["error"] or rec["answers"] is None:
            continue
        ans = rec["answers"]["answer"]
        slot = rec["meta"]["slot_by_letter"].get(ans.get("choice"))
        if slot is None:
            continue
        by_item.setdefault(rec["item_id"], {})[rec["order"]] = slot
    return {i: int(v[0] != v[1]) for i, v in by_item.items() if 0 in v and 1 in v}


def item_counts(recs, items):
    import numpy as np
    from metrics import extract_bbq_rows

    idx, mats = {}, []
    for item_id, cond, pol, wt, wnt, wc in extract_bbq_rows(recs, items):
        if item_id not in idx:
            idx[item_id] = len(mats)
            mats.append(np.zeros(12))
        v = mats[idx[item_id]]
        base = 0 if cond == "ambig" else 6
        p = 0 if pol == "neg" else 1
        v[base + p] += wt
        v[base + 2 + p] += wnt
        v[base + 4] += wc
        v[base + 5] += 1
    return idx, (np.array(mats) if mats else np.zeros((0, 12)))


def bbq_scores(mat):
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore"):
        t, nt = mat[..., 0] + mat[..., 1], mat[..., 2] + mat[..., 3]
        amb = np.where(t + nt > 0, 2 * t / (t + nt) - 1, 0.0)
        acc = np.where(mat[..., 5] > 0, mat[..., 4] / mat[..., 5], 0.0)
        t, nt = mat[..., 6] + mat[..., 7], mat[..., 8] + mat[..., 9]
        dis = np.where(t + nt > 0, 2 * t / (t + nt) - 1, 0.0)
    return 100 * amb * (1 - acc), 100 * dis


def significance(base_a, base_b, seed_a, seed_b, out, iters=500):
    """McNemar on flips + paired s_amb/s_dis bootstrap, Holm per family."""
    import numpy as np
    from analyze_paired_tests import binom_two_sided, holm

    flips, scores = [], []
    rng = np.random.default_rng(42)
    for model in REPEAT_MODELS:
        slug = SLUG[model]
        for cell in BBQ_CATS:
            ra, rb = load_dir(base_a, slug, cell), load_dir(base_b, slug, cell)
            if ra is None or rb is None:
                continue
            items = items_for(cell)

            fa, fb = flip_indicators(ra), flip_indicators(rb)
            common = sorted(set(fa) & set(fb))
            if common:
                a = sum(fa[i] for i in common)
                b = sum(fb[i] for i in common)
                c = sum(1 for i in common if fa[i] and not fb[i])
                d = sum(1 for i in common if fb[i] and not fa[i])
                flips.append(dict(model=model, cell=cell, n=len(common),
                                  rate_a=100 * a / len(common), rate_b=100 * b / len(common),
                                  delta=100 * (b - a) / len(common),
                                  p=binom_two_sided(min(c, d), c + d)))

            ia, ma = item_counts(ra, items)
            ib, mb = item_counts(rb, items)
            common = sorted(set(ia) & set(ib))
            if not common:
                continue
            A = np.stack([ma[ia[i]] for i in common])
            B = np.stack([mb[ib[i]] for i in common])
            pa, pb = bbq_scores(A.sum(0)), bbq_scores(B.sum(0))
            n = len(common)
            boots = None
            if not np.array_equal(A, B):
                idx = rng.integers(0, n, size=(iters, n))
                cnt = np.zeros((iters, n))
                np.add.at(cnt, (np.repeat(np.arange(iters), n), idx.ravel()), 1)
                boots = (bbq_scores(cnt @ A), bbq_scores(cnt @ B))
            for j, name in ((0, "s_amb"), (1, "s_dis")):
                if boots is None:
                    ci_a = (float(pa[j]), float(pa[j]))
                    ci_b = (float(pb[j]), float(pb[j]))
                    lo = hi = 0.0
                    pv = 1.0
                else:
                    delta = boots[0][j] - boots[1][j]
                    ci_a = tuple(float(x) for x in np.percentile(boots[0][j], [2.5, 97.5]))
                    ci_b = tuple(float(x) for x in np.percentile(boots[1][j], [2.5, 97.5]))
                    lo, hi = (float(x) for x in np.percentile(delta, [2.5, 97.5]))
                    pv = min(1.0, 2 * min(float((delta <= 0).mean()), float((delta >= 0).mean())))
                scores.append(dict(model=model, cell=cell, test=name, n=n,
                                   value_a=float(pa[j]), value_b=float(pb[j]),
                                   delta=float(pa[j] - pb[j]), ci_lo=lo, ci_hi=hi, p=pv,
                                   ci_a_lo=ci_a[0], ci_a_hi=ci_a[1],
                                   ci_b_lo=ci_b[0], ci_b_hi=ci_b[1],
                                   sig_a=ci_a[0] > 0 or ci_a[1] < 0,
                                   sig_b=ci_b[0] > 0 or ci_b[1] < 0))

    for r, adj in zip(flips, holm([r["p"] for r in flips])):
        r["p_holm"] = adj
    for r, adj in zip(scores, holm([r["p"] for r in scores])):
        r["p_holm"] = adj

    print(f"\n== Tests seed {seed_a} vs seed {seed_b} ==")
    nf = len(flips)
    print(f"flips: {sum(r['p'] < .05 for r in flips)}/{nf} uncorrected, "
          f"{sum(r['p_holm'] < .05 for r in flips)}/{nf} after Holm")
    for r in sorted(flips, key=lambda x: x["p"])[:3]:
        print(f"  {r['model']:18s} {r['cell']:24s} {r['rate_a']:6.3f}->{r['rate_b']:6.3f}pp "
              f"p={r['p']:.3f} p_holm={r['p_holm']:.3f}")
    ns = len(scores)
    print(f"scores: {sum(r['p'] < .05 for r in scores)}/{ns} uncorrected, "
          f"{sum(r['p_holm'] < .05 for r in scores)}/{ns} after Holm")
    for r in sorted(scores, key=lambda x: -abs(x["delta"]))[:5]:
        print(f"  {r['model']:18s} {r['cell']:24s} {r['test']:6s} "
              f"{r['value_a']:+7.3f}->{r['value_b']:+7.3f} d={r['delta']:+.3f} "
              f"CI[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] p_holm={r['p_holm']:.3f}")
    flips_sig = [r for r in scores if r["sig_a"] != r["sig_b"]]
    print(f"uncorrected significance verdicts that change: {len(flips_sig)}")
    for r in flips_sig:
        print(f"  {r['model']} {r['cell']} {r['test']}: "
              f"{r['value_a']:+.2f} (sig={r['sig_a']}) -> {r['value_b']:+.2f} (sig={r['sig_b']})")

    header = ("family,model,cell,test,n,value_a,value_b,delta,ci_lo,ci_hi,p,p_holm,"
              "ci_a_lo,ci_a_hi,ci_b_lo,ci_b_hi,sig_a,sig_b")
    lines = [header]
    for r in flips:
        lines.append(",".join(["flip", r["model"], r["cell"], "flip", str(r["n"]),
                               f"{r['rate_a']:.4f}", f"{r['rate_b']:.4f}", f"{r['delta']:+.4f}",
                               "", "", f"{r['p']:.6f}", f"{r['p_holm']:.6f}",
                               "", "", "", "", "", ""]))
    for r in scores:
        lines.append(",".join(["score", r["model"], r["cell"], r["test"], str(r["n"]),
                               f"{r['value_a']:.4f}", f"{r['value_b']:.4f}", f"{r['delta']:+.4f}",
                               f"{r['ci_lo']:+.4f}", f"{r['ci_hi']:+.4f}",
                               f"{r['p']:.6f}", f"{r['p_holm']:.6f}",
                               f"{r['ci_a_lo']:+.4f}", f"{r['ci_a_hi']:+.4f}",
                               f"{r['ci_b_lo']:+.4f}", f"{r['ci_b_hi']:+.4f}",
                               str(r["sig_a"]), str(r["sig_b"])]))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"CSV -> {out} ({len(lines) - 1} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,13", help="new seeds, comma-separated")
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--out", default=str(HERE.parent / "results" / "seed_stability.csv"))
    ap.add_argument("--tests", action="store_true", help="McNemar + score bootstrap (seed A vs B)")
    ap.add_argument("--tests-only", action="store_true", help="tests only (skip stability recomputation)")
    ap.add_argument("--pair", default=None,
                    help="dirA,dirB,labelA,labelB — compare two arbitrary directories "
                         "(e.g. batch arms) instead of results vs results_seedN")
    ap.add_argument("--tests-out", default=str(HERE.parent / "results" / "seed_significance.csv"))
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    csv = ["model,cell,comparison,n_matched,changed_pct,mean_abs_dp,flip_base,flip_seed,flip_delta_pp"]
    if args.selftest:
        print("== SELFTEST: base vs base ==")
        compare_dirs(args.base, args.base, "base", "base", csv, selftest=True)
        print("selftest OK (0 changes, 0 dp)")
        return

    if args.pair:
        da, db, la, lb = [x.strip() for x in args.pair.split(",")]
        print(f"== {la} vs {lb} ==")
        compare_dirs(da, db, la, lb, csv)
        if args.tests:
            significance(da, db, la, lb, args.tests_out, iters=args.iters)
        if not args.tests_only:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(csv) + "\n")
            print(f"\nCSV -> {out} ({len(csv) - 1} rows)")
        return

    if not args.tests_only:
        for seed in [s.strip() for s in args.seeds.split(",") if s.strip()]:
            seed_dir = Path(args.base).parent / f"results_seed{seed}"
            print(f"== seed {seed}: {seed_dir} ==")
            if not seed_dir.exists():
                print(f"  (missing {seed_dir})")
                continue
            compare_dirs(args.base, seed_dir, "42", seed, csv)
    seed_dirs = [(s.strip(), Path(args.base).parent / f"results_seed{s.strip()}")
                 for s in args.seeds.split(",") if s.strip()]
    if len(seed_dirs) >= 2 and all(d.exists() for _s, d in seed_dirs):
        (s1, d1), (s2, d2) = seed_dirs[0], seed_dirs[1]
        if not args.tests_only:
            print(f"== seed {s1} vs seed {s2} ==")
            compare_dirs(d1, d2, s1, s2, csv)
        if args.tests:
            significance(d1, d2, s1, s2, args.tests_out, iters=args.iters)
    elif args.tests:
        print("--tests requires two existing seed directories")
    if not args.tests_only:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(csv) + "\n")
        print(f"\nCSV -> {out} ({len(csv) - 1} rows)")


if __name__ == "__main__":
    main()
