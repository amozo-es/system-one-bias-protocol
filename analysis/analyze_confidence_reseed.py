"""Confidence and repetitions: do low-confidence items change more across seeds 7/13?

Item-level join between cluster/results_seed7 and results_seed13 (same execution,
same stack, different shuffle seed). Only models with a confidence field and a
repetition arm (Jev is hosted and has no arm). Each pair's confidence is the mean
of the two runs; threshold 0.7 (the same as the confident-subset scores).

Output: results/confidence_reseed.csv
  model,dataset,lo_n,lo_changed,hi_n,hi_changed   (<0.7 and >=0.7)
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS_DEFAULT = HERE.parent / "execution" / "cluster" / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--logs", default=str(LOGS_DEFAULT),
                help="main-seed decision-log dir; seed arms live at <parent>/results_seed7|13")
ap.add_argument("--out", default=str(HERE.parent / "results" / "confidence_reseed.csv"))
args = ap.parse_args()

LOGS = Path(args.logs)
S7 = LOGS.parent / "results_seed7"
S13 = LOGS.parent / "results_seed13"
OUT = Path(args.out)

MODELS = ["von", "laya", "tev1-4b", "tev1-0.8b", "nimble-9b", "clef", "clef-flash"]
DATASETS = ["bbq_age", "bbq_disability_status", "bbq_gender_identity", "bbq_nationality",
            "bbq_physical_appearance", "bbq_race_ethnicity", "bbq_race_x_gender", "bbq_racexses",
            "bbq_religion", "bbq_ses", "bbq_sexual_orientation", "crows", "silverspoon"]
THR = 0.7


def load(path, model, ds):
    f = path / f"{model}__{ds}.jsonl"
    out = {}
    if not f.exists() or "smoke" in f.name:
        return out
    for line in open(f):
        r = json.loads(line)
        qid = "answer" if ds.startswith("bbq") else ("stereotype" if ds == "crows" else "justified")
        a = (r.get("answers") or {}).get(qid)
        if not isinstance(a, dict):
            continue
        conf = a.get("confidence")
        if not isinstance(conf, (int, float)):
            continue
        key = (int(r["item_id"]), int(r["order"]))
        if ds.startswith("bbq"):
            val = a.get("choice")
        else:
            v = a.get("noul")
            val = None if not isinstance(v, (int, float)) else (v > 0.5)
        if val is None:
            continue
        out[key] = (val, conf)
    return out


rows = ["model,dataset,lo_n,lo_changed,hi_n,hi_changed"]
tot = {}
for model in MODELS:
    for ds in DATASETS:
        a = load(S7, model, ds)
        b = load(S13, model, ds)
        lo_n = lo_c = hi_n = hi_c = 0
        for key in set(a) & set(b):
            v7, c7 = a[key]
            v13, c13 = b[key]
            changed = v7 != v13
            conf = (c7 + c13) / 2
            if conf >= THR:
                hi_n += 1
                hi_c += changed
            else:
                lo_n += 1
                lo_c += changed
        if lo_n + hi_n:
            rows.append(f"{model},{ds},{lo_n},{lo_c},{hi_n},{hi_c}")
            t = tot.setdefault(model, [0, 0, 0, 0])
            t[0] += lo_n; t[1] += lo_c; t[2] += hi_n; t[3] += hi_c

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(rows) + "\n")
print("pooled per model (<0.7 vs >=0.7):")
for m, (ln, lc, hn, hc) in tot.items():
    print(f"  {m:<12} lo {100*lc/ln if ln else 0:6.2f}% (n={ln:<7})  hi {100*hc/hn if hn else 0:6.2f}% (n={hn:<7})")
