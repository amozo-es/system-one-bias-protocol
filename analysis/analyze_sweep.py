"""Pan-category analysis: 13 configurations x 13 dataset families.

Sources: cluster/results (local + hosted-schema models, cluster schema) and
results/jev (Jev via OpenRouter, corpus schema). Reuses metrics.py
(official BBQ scoring with metadata, flips, bootstraps).

Normalization applied to cluster records:
  - choice without probabilities -> one-hot {letter: 1.0} (argmax, semantically identical)
  - answers._unparsed -> rec["error"]="parse" (excluded from metrics, counted)

Usage: python analyze_sweep.py [--iters 500] [--models von,clef] [--out results/sweep_tables.csv]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets as ds
from metrics import (bbq_bootstrap_ci, bootstrap_ci, crows_by_order, order_flips,
                     score_bbq, score_crows, score_silverspoon)

HERE = Path(__file__).resolve().parent
CLUSTER = HERE.parent / "execution" / "cluster" / "results"
JEV = HERE.parent / "results" / "jev"

# (display, source, file-slug) — source: cluster | jev
MODELS = [
    ("von", "cluster", "von"), ("laya", "cluster", "laya"),
    ("clef", "cluster", "clef"), ("clef-flash", "cluster", "clef-flash"),
    ("tev1-4b", "cluster", "tev1-4b"), ("tev1-0.8b", "cluster", "tev1-0.8b"),
    ("qwen3.5-0.8b", "cluster", "qwen3.5-0.8b"),
    ("nimble-9b", "cluster", "nimble-9b"), ("qwen3.5-4b", "cluster", "qwen3.5-4b"),
    ("qwen3.5-9b", "cluster", "qwen3.5-9b"),
    ("qwen3.5-9b-nothink", "cluster", "qwen3.5-9b-nothink"), ("gpt-oss-20b", "cluster", "gpt-oss-20b"),
    ("jev", "jev", "jev-or"),
]
TYPE = {"von": "encoder", "laya": "encoder", "clef": "hosted-SO", "clef-flash": "hosted-SO",
        "tev1-4b": "causal-SFT", "tev1-0.8b": "causal-SFT", "nimble-9b": "causal-LoRA",
        "qwen3.5-4b": "base", "qwen3.5-9b": "base", "qwen3.5-9b-nothink": "base-nothink", "gpt-oss-20b": "base", "jev": "hosted-SO",
        "qwen3.5-0.8b": "base"}

BBQ_CATS = {"bbq_age": "Age", "bbq_disability_status": "Disability_status",
            "bbq_gender_identity": "Gender_identity", "bbq_nationality": "Nationality",
            "bbq_physical_appearance": "Physical_appearance", "bbq_race_ethnicity": "Race_ethnicity",
            "bbq_race_x_gender": "Race_x_gender", "bbq_racexses": "Race_x_SES",
            "bbq_religion": "Religion", "bbq_ses": "SES", "bbq_sexual_orientation": "Sexual_orientation"}
CROWS_TYPES = ["age", "disability", "gender", "nationality", "physical-appearance",
               "race-color", "religion", "sexual-orientation", "socioeconomic"]


def norm_record(rec):
    """Cluster/corpus schema -> the schema metrics.py expects."""
    rec = dict(rec)
    try:
        rec["item_id"] = int(rec["item_id"])
        rec["order"] = int(rec["order"])
    except (KeyError, TypeError, ValueError):
        rec["error"] = rec.get("error") or "bad-id"
        return rec
    ans = rec.get("answers")
    if not isinstance(ans, dict):
        rec["error"] = rec.get("error") or "no-answers"
        return rec
    if any(isinstance(v, dict) and v.get("_unparsed") for v in ans.values()):
        rec["error"] = "parse"
        return rec
    for v in ans.values():
        if (isinstance(v, dict) and v.get("type") == "choice"
                and not v.get("probabilities") and v.get("choice")):
            v["probabilities"] = {v["choice"]: 1.0}  # one-hot = argmax
    return rec


def load_records(source, slug, name):
    base = CLUSTER if source == "cluster" else JEV
    if source == "jev" and name == "crows":
        # Jev: crows split by bias_type (jev-or__crows.jsonl = SES)
        parts = sorted(base.glob(f"{slug}__crows*.jsonl"))
        recs = []
        for f in parts:
            recs += [norm_record(json.loads(l)) for l in f.read_text().splitlines() if l.strip()]
        return recs or None
    f = base / f"{slug}__{name}.jsonl"
    if not f.exists():
        return None
    recs = [norm_record(json.loads(l)) for l in f.read_text().splitlines() if l.strip()]
    return recs


def rebuild_slots(recs, items):
    """v2: slot_by_letter is rebuilt by content. The v12/v13 generation logs
    mapped letters assuming unknown=ans2 (false in ~2/3 of BBQ items) with
    A<->B crossed."""
    for r in recs:
        crit = ((r.get("request") or {}).get("questions") or {}).get("answer", {}).get("criteria")
        it = items.get(r["item_id"])
        if not (crit and it):
            continue
        m2 = {}
        for L, t in crit.items():
            for i in range(3):
                if it[f"ans{i}"] == t:
                    m2[L] = i
                    break
        if m2:
            r.setdefault("meta", {})["slot_by_letter"] = m2


def parse_rate(recs):
    if not recs:
        return 0.0
    return sum(1 for r in recs if r.get("error")) / len(recs)


def run_bbq(model, source, slug, iters, csv):
    out_rows = []
    for name, cat in BBQ_CATS.items():
        recs = load_records(source, slug, name)
        if recs is None:
            print(f"  {model} {name}: NO FILE")
            continue
        items = {r["example_id"]: r for r in ds.load_bbq(cat)}
        rebuild_slots(recs, items)
        pr = parse_rate(recs)
        ci = bbq_bootstrap_ci(recs, items, prob=False, iters=iters)
        sc = score_bbq(recs, items, cat)
        fl = order_flips(recs)
        n_pairs = len({(r["item_id"], r["order"]) for r in recs})
        complete = "partial" if n_pairs < len(items) * 2 else "complete"
        for key, label in (("s_ambig", "s_amb"), ("s_disambig", "s_dis")):
            lo, hi, pt = ci[key]
            csv.append(f"{model},{TYPE[model]},{name},{label},{pt:.2f},{lo:.2f},{hi:.2f},"
                       f"{len(recs)},{pr:.3f},{complete}")
            out_rows.append((model, name, label, pt, lo, hi))
        csv.append(f"{model},{TYPE[model]},{name},flip_rate,{fl['rate']:.4f},,,{len(recs)},{pr:.3f},{complete}")
        print(f"  {model} {name:24s} s_amb {ci['s_ambig'][2]:+6.2f} [{ci['s_ambig'][0]:+6.2f},{ci['s_ambig'][1]:+6.2f}] "
              f"s_dis {ci['s_disambig'][2]:+6.2f} flip {fl['rate']:.3f} parse {pr:.3f} {complete}")
    return out_rows


def run_crows(model, source, slug, iters, csv):
    recs = load_records(source, slug, "crows")
    if recs is None:
        print(f"  {model} crows: NO FILE")
        return
    by = crows_by_order(recs)
    p0, p1 = by["order0"]["p_stereo"], by["order1"]["p_stereo"]
    swing = abs(p1 - p0)
    vals = []
    for r in recs:
        if r.get("error") or not isinstance(r.get("answers"), dict):
            continue
        p1_ = r["answers"]["stereotype"]["noul"]
        vals.append(p1_ if r["meta"]["s1_is_stereo"] else 1.0 - p1_)
    if not vals:
        print(f"  {model} crows: no valid records (parse {parse_rate(recs):.2f})")
        return
    lo, hi = bootstrap_ci(vals, iters=iters)
    print(f"  {model} crows: p_stereo {sum(vals)/max(len(vals),1):.3f} [{lo:.3f},{hi:.3f}] "
          f"| o0 {p0:.3f} o1 {p1:.3f} swing {swing:.3f} n {len(vals)} parse {parse_rate(recs):.3f}")
    csv.append(f"{model},{TYPE[model]},crows,p_stereo,{sum(vals)/max(len(vals),1):.4f},{lo:.4f},{hi:.4f},"
               f"{len(vals)},{parse_rate(recs):.3f},complete")
    csv.append(f"{model},{TYPE[model]},crows,order0,{p0:.4f},,,{len(vals)},,")
    csv.append(f"{model},{TYPE[model]},crows,order1,{p1:.4f},,,{len(vals)},,")
    csv.append(f"{model},{TYPE[model]},crows,swing,{swing:.4f},,,{len(vals)},,")
    # per bias type
    for t in CROWS_TYPES:
        sub = [r for r in recs if r.get("meta", {}).get("bias_type") == t and not r.get("error")]
        if not sub:
            continue
        s = score_crows(sub)
        csv.append(f"{model},{TYPE[model]},crows_{t},p_stereo,{s['p_stereo']:.4f},,,{s['n']},,")


def run_silverspoon(model, source, slug, iters, csv):
    recs = load_records(source, slug, "silverspoon")
    if recs is None:
        print(f"  {model} silverspoon: NO FILE")
        return
    items = {i: r for i, r in enumerate(ds.load_silverspoon())}
    sc = score_silverspoon(recs, items)
    if sc.get("n", 0) == 0:
        print(f"  {model} silverspoon: no valid records")
        return
    per_item = {}
    for r in recs:
        if r.get("error") or not isinstance(r.get("answers"), dict):
            continue
        try:
            per_item[int(r["item_id"])] = r["answers"]["justified"]["noul"]
        except (KeyError, ValueError):
            continue
    if not per_item:
        print(f"  {model} silverspoon: no valid records (parse {parse_rate(recs):.2f})")
        return
    # delta bootstrap: paired item resampling (p, Up_bin, Low_bin)
    trip = [(per_item[i], items[i]["Up_bin"], items[i]["Low_bin"])
            for i in per_item if i in items]

    def _delta(idxs):
        s = [trip[i] for i in idxs]
        au = sum(1 for p, u, _ in s if (p >= .5) == (u == "1")) / len(s)
        al = sum(1 for p, _, l in s if (p >= .5) == (l == "1")) / len(s)
        return au - al

    import random
    rng = random.Random(42)
    n_t = len(trip)
    stats = sorted(_delta([rng.randrange(n_t) for _ in range(n_t)]) for _ in range(iters))
    dlo, dhi = stats[int(0.025 * iters)], stats[int(0.975 * iters)]
    lo, hi = bootstrap_ci(list(per_item.values()), iters=iters)
    print(f"  {model} silverspoon: mean_p {sc['mean_p']:.4f} [{lo:.4f},{hi:.4f}] "
          f"align {sc['align_delta']:+.4f} [{dlo:+.4f},{dhi:+.4f}] n {sc['n']} parse {parse_rate(recs):.3f}")
    csv.append(f"{model},{TYPE[model]},silverspoon,mean_p,{sc['mean_p']:.4f},{lo:.4f},{hi:.4f},"
               f"{sc['n']},{parse_rate(recs):.3f},complete")
    csv.append(f"{model},{TYPE[model]},silverspoon,align_delta,{sc['align_delta']:.4f},{dlo:.4f},{dhi:.4f},"
               f"{sc['n']},{parse_rate(recs):.3f},complete")


def main():
    global CLUSTER, JEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--models", default="", help="coma de display names")
    ap.add_argument("--logs", default=str(CLUSTER),
                    help="decision-log dir (<model>__<dataset>.jsonl)")
    ap.add_argument("--jev", default=str(JEV), help="Jev decision-log dir")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "sweep_tables.csv"))
    args = ap.parse_args()
    CLUSTER, JEV = Path(args.logs), Path(args.jev)
    only = {s.strip() for s in args.models.split(",") if s.strip()}
    csv = ["model,type,dataset,metric,point,lo,hi,n,parse_rate,complete"]
    for display, source, slug in MODELS:
        if only and display not in only:
            continue
        print(f"== {display} ({source}) ==")
        run_bbq(display, source, slug, args.iters, csv)
        run_crows(display, source, slug, args.iters, csv)
        run_silverspoon(display, source, slug, args.iters, csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(csv) + "\n")
    print(f"\nCSV -> {out} ({len(csv) - 1} rows)")


if __name__ == "__main__":
    main()
