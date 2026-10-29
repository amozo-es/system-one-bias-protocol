"""Confidence-stratified probability-mass BBQ score.

Question: is the alignment hidden in the distribution (s_mass > s_argmax, Jev)
noise from near-uniform distributions, or does it persist among confident
answers? Recomputes s_amb with mass (prob=True) over all items and over the
conf >= 0.7 subset (same threshold as analyze_confidence.py).

Reads the JSONL logs on disk; runs no model.

Output: results/probmass_conf.csv
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import datasets as ds
from metrics import score_bbq

_ap = argparse.ArgumentParser()
_ap.add_argument("--logs", default=str(HERE.parent / "execution" / "cluster" / "results"),
                 help="decision-log dir (<model>__<dataset>.jsonl)")
_ap.add_argument("--jev", default=str(HERE.parent / "results" / "jev"),
                 help="Jev decision-log dir")
_ap.add_argument("--out-dir", default=str(HERE.parent / "results"), help="CSV output dir")
_args = _ap.parse_args()

CLUSTER = Path(_args.logs)
JEV = Path(_args.jev)
OUT = Path(_args.out_dir)
BBQ_CATS = {"bbq_age": "Age", "bbq_disability_status": "Disability_status",
            "bbq_gender_identity": "Gender_identity", "bbq_nationality": "Nationality",
            "bbq_physical_appearance": "Physical_appearance", "bbq_race_ethnicity": "Race_ethnicity",
            "bbq_race_x_gender": "Race_x_gender", "bbq_racexses": "Race_x_SES",
            "bbq_religion": "Religion", "bbq_ses": "SES", "bbq_sexual_orientation": "Sexual_orientation"}
TYPED = [("von", CLUSTER, "von"), ("laya", CLUSTER, "laya"),
         ("nimble-9b", CLUSTER, "nimble-9b"), ("clef", CLUSTER, "clef"),
         ("clef-flash", CLUSTER, "clef-flash"), ("jev", JEV, "jev-or")]
HI = 0.7


def load_records(base, slug, name):
    f = base / f"{slug}__{name}.jsonl"
    if not f.exists():
        return None
    out = []
    for line in open(f):
        if not line.strip():
            continue
        rec = json.loads(line)
        ans = rec.get("answers")
        if not isinstance(ans, dict) or any(
                isinstance(v, dict) and v.get("_unparsed") for v in ans.values()):
            rec["error"] = "parse"
            continue
        rec["error"] = None
        out.append(rec)
    return out


def normalize_slots(recs, items):
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


def conf_of(rec):
    c = (rec.get("answers") or {}).get("answer", {}).get("confidence")
    return c if isinstance(c, (int, float)) else None


print(f"{'model':<12} {'cat':<22} {'mass_all':>9} {'mass_hi':>9} {'arg_hi':>8} "
      f"{'gap_all':>8} {'gap_hi':>8} {'n_hi':>6}")
rows = ["model,dataset,s_amb_mass_all,s_amb_mass_hi,s_amb_argmax_hi,gap_all,gap_hi,n_hi"]
summary = {}
for model, base, slug in TYPED:
    gaps_all, gaps_hi = [], []
    for name, cat in BBQ_CATS.items():
        recs = load_records(base, slug, name)
        if recs is None:
            continue
        items = {r["example_id"]: r for r in ds.load_bbq(cat)}
        normalize_slots(recs, items)
        hi = [r for r in recs if (c := conf_of(r)) is not None and c >= HI]
        if not hi:
            continue
        m_all = score_bbq(recs, items, cat, prob=True)
        m_hi = score_bbq(hi, items, cat, prob=True)
        a_hi = score_bbq(hi, items, cat, prob=False)
        ga = m_all["s_amb"] - score_bbq(recs, items, cat, prob=False)["s_amb"]
        gh = m_hi["s_amb"] - a_hi["s_amb"]
        gaps_all.append(ga)
        gaps_hi.append(gh)
        rows.append(f"{model},{name},{m_all['s_amb']:.2f},{m_hi['s_amb']:.2f},{a_hi['s_amb']:.2f},"
                    f"{ga:.2f},{gh:.2f},{m_hi['n_ambig'] + m_hi['n_disambig']}")
        print(f"{model:<12} {name:<22} {m_all['s_amb']:>+9.2f} {m_hi['s_amb']:>+9.2f} "
              f"{a_hi['s_amb']:>+8.2f} {ga:>+8.2f} {gh:>+8.2f} {m_hi['n_ambig'] + m_hi['n_disambig']:>6}")
    if gaps_hi:
        summary[model] = (sum(g > 0 for g in gaps_hi), len(gaps_hi),
                          sum(gaps_all) / len(gaps_all), sum(gaps_hi) / len(gaps_hi))
print("\n== summary (gap_hi = mass_hi - argmax_hi) ==")
for m, (npos, n, mall, mhi) in summary.items():
    print(f"{m:<12} gap>0 en {npos}/{n} categorias | gap medio all {mall:+.2f} -> hi {mhi:+.2f}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "probmass_conf.csv").write_text("\n".join(rows) + "\n")
print(f"\nCSV -> {OUT / 'probmass_conf.csv'}")
