"""Probability-mass vs argmax (BBQ) + UNKNOWN rate in ambiguous contexts.

Only typed models (they expose per-option probabilities): von, laya, nimble-9b,
clef, clef-flash, jev. Letter models are excluded: their mass is one-hot by
construction (identical to argmax).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import datasets as ds
from metrics import score_bbq, bbq_bootstrap_ci

_ap = argparse.ArgumentParser()
_ap.add_argument("--logs", default=str(HERE.parent / "execution" / "cluster" / "results"),
                 help="decision-log dir (<model>__<dataset>.jsonl)")
_ap.add_argument("--jev", default=str(HERE.parent / "results" / "jev"),
                 help="Jev decision-log dir")
_ap.add_argument("--out-dir", default=str(HERE.parent / "results"), help="CSV output dir")
_args = _ap.parse_args()

CLUSTER = Path(_args.logs)
JEV = Path(_args.jev)
OUTDIR = Path(_args.out_dir)
BBQ_CATS = {"bbq_age": "Age", "bbq_disability_status": "Disability_status",
            "bbq_gender_identity": "Gender_identity", "bbq_nationality": "Nationality",
            "bbq_physical_appearance": "Physical_appearance", "bbq_race_ethnicity": "Race_ethnicity",
            "bbq_race_x_gender": "Race_x_gender", "bbq_racexses": "Race_x_SES",
            "bbq_religion": "Religion", "bbq_ses": "SES", "bbq_sexual_orientation": "Sexual_orientation"}
TYPED = [("von", CLUSTER, "von"), ("laya", CLUSTER, "laya"),
         ("nimble-9b", CLUSTER, "nimble-9b"), ("clef", CLUSTER, "clef"),
         ("clef-flash", CLUSTER, "clef-flash"), ("jev", JEV, "jev-or")]


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
    """v2: rebuilds slot_by_letter by content (v12/v13 logs assumed
    unknown=ans2 with A<->B crossed)."""
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


print(f"{'model':<12} {'cat':<22} {'s_amb_arg':>9} {'s_amb_mass':>10} {'gap':>6} {'unk_arg':>8} {'unk_mass':>8}")
gaps = {}
prow = ["model,dataset,s_amb_argmax,s_amb_mass,gap,unk_argmax,unk_mass"]
for model, base, slug in TYPED:
    mg = []
    for name, cat in BBQ_CATS.items():
        recs = load_records(base, slug, name)
        if recs is None:
            continue
        items = {r["example_id"]: r for r in ds.load_bbq(cat)}
        normalize_slots(recs, items)
        a = score_bbq(recs, items, cat, prob=False)
        m = score_bbq(recs, items, cat, prob=True)
        gap = m["s_amb"] - a["s_amb"]
        mg.append(gap)
        prow.append(f"{model},{name},{a['s_amb']:.2f},{m['s_amb']:.2f},{gap:.2f},"
                    f"{a['accuracy_ambig']:.4f},{m['accuracy_ambig']:.4f}")
        print(f"{model:<12} {name:<22} {a["s_amb"]:>+9.2f} {m["s_amb"]:>+10.2f} {gap:>+6.2f} "
              f"{a['accuracy_ambig']:>8.3f} {m['accuracy_ambig']:>8.3f}")
    if mg:
        gaps[model] = (sum(abs(g) for g in mg) / len(mg), max(mg, key=abs))
print("\n== summary ==")
for m, (mean_abs, mx) in gaps.items():
    print(f"{m:<12} |gap| medio {mean_abs:.2f}, max {mx:+.2f}")
out_path = OUTDIR / "probmass_gaps.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(prow) + "\n")
print(f"CSV -> {out_path} ({len(prow)-1} rows)")
