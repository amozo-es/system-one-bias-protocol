"""Paper audit: generates the remaining CSVs used to verify the text.

- bbq_abstention.csv: accuracy_ambig (UNKNOWN rate) per model x category
- flip_destinations.csv: % flips target<->nontarget vs via UNKNOWN + the 4B SES case
- silverspoon_rates.csv: justified_rate/wrong_rate argmax + mean_p
- cost_totals.csv: total latency/tokens + ratios
- volume_counts.csv: decisions per model behind the 1,638,000 total
- dataset_composition.csv: items per dataset + SilverSpoon labels + excluded NAs
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import datasets as ds
from metrics import load_target_locs, score_bbq, score_silverspoon, unknown_slot

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
MODELS = [("von", CLUSTER, "von"), ("laya", CLUSTER, "laya"), ("clef", CLUSTER, "clef"),
          ("clef-flash", CLUSTER, "clef-flash"), ("tev1-4b", CLUSTER, "tev1-4b"),
          ("tev1-0.8b", CLUSTER, "tev1-0.8b"), ("qwen3.5-0.8b", CLUSTER, "qwen3.5-0.8b"),
          ("nimble-9b", CLUSTER, "nimble-9b"), ("qwen3.5-4b", CLUSTER, "qwen3.5-4b"),
          ("qwen3.5-9b", CLUSTER, "qwen3.5-9b"),
          ("qwen3.5-9b-nothink", CLUSTER, "qwen3.5-9b-nothink"),
          ("gpt-oss-20b", CLUSTER, "gpt-oss-20b"), ("jev", JEV, "jev-or")]


def norm(rec):
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
        if isinstance(v, dict) and v.get("type") == "choice" and not v.get("probabilities") and v.get("choice"):
            v["probabilities"] = {v["choice"]: 1.0}
    return rec


def load(source_base, slug, name):
    if slug == "jev-or" and name == "crows":
        recs = []
        for f in sorted(source_base.glob(f"{slug}__crows*.jsonl")):
            recs += [norm(json.loads(l)) for l in f.read_text().splitlines() if l.strip()]
        return recs or None
    f = source_base / f"{slug}__{name}.jsonl"
    if not f.exists():
        return None
    return [norm(json.loads(l)) for l in f.read_text().splitlines() if l.strip()]


def fix_slots(recs, items):
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


# --- 1. abstention + volume (13 cells: 11 BBQ + crows + spoon = 126,000) ---
abst = ["model,dataset,accuracy_ambig,n_ambig,accuracy_disambig,n_disambig"]
vol = ["model,file_records,valid_records"]
for model, base, slug in MODELS:
    tot_files = tot_valid = 0
    for name, cat in BBQ_CATS.items():
        recs = load(base, slug, name)
        if recs is None:
            continue
        items = {r["example_id"]: r for r in ds.load_bbq(cat)}
        fix_slots(recs, items)
        sc = score_bbq(recs, items, cat)
        abst.append(f"{model},{name},{sc['accuracy_ambig']:.4f},{sc['n_ambig']},"
                    f"{sc['accuracy_disambig']:.4f},{sc['n_disambig']}")
        tot_files += len(recs)
        tot_valid += sum(1 for r in recs if not r.get("error"))
    for extra in ("crows", "silverspoon"):
        recs = load(base, slug, extra)
        if recs is None:
            continue
        tot_files += len(recs)
        tot_valid += sum(1 for r in recs if not r.get("error"))
    vol.append(f"{model},{tot_files},{tot_valid}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "bbq_abstention.csv").write_text("\n".join(abst) + "\n")
(OUT / "volume_counts.csv").write_text("\n".join(vol) + "\n")
print(f"abstention: {len(abst)-1} rows | volume: {len(vol)-1} models")

# --- 2. flip destinations (+ per-order UNKNOWN counts, anchors the paper sentence) ---
tlocs = load_target_locs()
flip = ["model,dataset,n_both,n_flips,flip_rate,pct_target_nontarget,pct_with_unknown,"
        "unk_o0,unk_o1,unk_changed,unk_o0_ambig,unk_changed_ambig"]
for model, base, slug in MODELS:
    for name, cat in BBQ_CATS.items():
        recs = load(base, slug, name)
        if recs is None:
            continue
        items = {r["example_id"]: r for r in ds.load_bbq(cat)}
        fix_slots(recs, items)
        by_item = {}
        for r in recs:
            if r.get("error") or not r.get("answers"):
                continue
            letter = r["answers"]["answer"].get("choice")
            slot = r.get("meta", {}).get("slot_by_letter", {}).get(letter)
            if slot is None:
                continue
            by_item.setdefault(r["item_id"], {})[r["order"]] = (slot, r["item_id"])
        both = {k: v for k, v in by_item.items() if 0 in v and 1 in v}
        flips = {k: v for k, v in both.items() if v[0][0] != v[1][0]}
        direct = with_unk = 0
        unk_o0 = unk_o1 = unk_chg = unk_o0a = unk_chga = 0
        for item_id, v in both.items():
            it = items.get(item_id)
            unk = unknown_slot(it) if it else None
            s0, s1 = v[0][0], v[1][0]
            u0, u1 = (s0 == unk), (s1 == unk)
            unk_o0 += u0
            unk_o1 += u1
            unk_chg += (u0 != u1)
            if it and it.get("context_condition") == "ambig":
                unk_o0a += u0
                unk_chga += (u0 != u1)
        for item_id, v in flips.items():
            it = items.get(item_id)
            unk = unknown_slot(it) if it else None
            s0, s1 = v[0][0], v[1][0]
            if unk is not None and (s0 == unk or s1 == unk):
                with_unk += 1
            else:
                direct += 1
        nf, nb = len(flips), len(both)
        flip.append(f"{model},{name},{nb},{nf},{nf/nb if nb else 0:.4f},"
                    f"{direct/nf if nf else 0:.4f},{with_unk/nf if nf else 0:.4f},"
                    f"{unk_o0},{unk_o1},{unk_chg},{unk_o0a},{unk_chga}")
(OUT / "flip_destinations.csv").write_text("\n".join(flip) + "\n")
print(f"flip destinations: {len(flip)-1} rows")

# --- 3. silverspoon rates (argmax + mean_p + dataset labels) ---
sp = ["model,n,justified_rate,wrong_rate,mean_p,acc_up,acc_low,align_delta"]
for model, base, slug in MODELS:
    recs = load(base, slug, "silverspoon")
    if recs is None:
        continue
    items = {i: r for i, r in enumerate(ds.load_silverspoon())}
    sc = score_silverspoon(recs, items)
    if not sc.get("n"):
        continue
    ps = []
    for r in recs:
        if r.get("error") or not isinstance(r.get("answers"), dict):
            continue
        try:
            ps.append(r["answers"]["justified"]["noul"])
        except KeyError:
            continue
    jr = sum(1 for p in ps if p >= 0.5) / len(ps) if ps else 0
    sp.append(f"{model},{sc['n']},{jr:.4f},{1-jr:.4f},{sc['mean_p']:.4f},"
              f"{sc['acc_up']:.4f},{sc['acc_low']:.4f},{sc['align_delta']:.4f}")
# dataset labels: always-wrong baselines + disagreement
items = ds.load_silverspoon()
n = len(items)
up_wrong = sum(1 for r in items if r["Up_bin"] == "0") / n  # P(justified)<0.5 = wrong
low_wrong = sum(1 for r in items if r["Low_bin"] == "0") / n
disag = sum(1 for r in items if r["Up_bin"] != r["Low_bin"]) / n
sp.append(f"DATASET-always-wrong,{n},0.0000,1.0000,,{up_wrong:.4f},{low_wrong:.4f},")
# nota: acc always-wrong = frac label==0
sp2 = [f"# always-wrong acc_up={up_wrong:.4f} acc_low={low_wrong:.4f} disagreement={disag:.4f} n={n}"]
(OUT / "silverspoon_rates.csv").write_text("\n".join(sp + sp2) + "\n")
print(f"silverspoon: {len(sp)-1} rows | always-wrong up={up_wrong:.3f} low={low_wrong:.3f} disag={disag:.3f}")

# --- 4. total costs ---
tot_ms = tot_ctok = tot_ptok = tot_dec = 0
per = []
for mod in ["von", "laya", "tev1-0.8b", "tev1-4b", "nimble-9b", "qwen3.5-4b",
            "qwen3.5-0.8b", "qwen3.5-9b", "qwen3.5-9b-nothink", "gpt-oss-20b", "clef", "clef-flash"]:
    lat, ctok, ptok = [], [], []
    for f in (CLUSTER).glob(f"{mod}__*.jsonl"):
        if "smoke" in f.name:
            continue
        for line in open(f):
            r = json.loads(line)
            t, u = r.get("timing") or {}, r.get("usage") or {}
            if isinstance(t.get("latency_ms"), (int, float)):
                lat.append(t["latency_ms"])
            if isinstance(u.get("completion_tokens"), int):
                ctok.append(u["completion_tokens"])
            if isinstance(u.get("prompt_tokens"), int):
                ptok.append(u["prompt_tokens"])
    per.append((mod, len(lat), sum(lat), sum(ctok), sum(ptok)))
    tot_ms += sum(lat)
    tot_ctok += sum(ctok)
    tot_ptok += sum(ptok)
    tot_dec += len(lat)
share_9b = next(s for m, c, s, _, _ in per if m == "qwen3.5-9b")
share_ctok_9b = next(ct for m, _, _, ct, _ in per if m == "qwen3.5-9b")
lines = ["model,n_decisions,lat_sum_s,ctok_sum,ptok_sum,tok_sum",
         *[f"{m},{c},{s/1000:.1f},{ct},{pt},{ct+pt}" for m, c, s, ct, pt in per],
         f"TOTAL,{tot_dec},{tot_ms/1000:.1f},{tot_ctok},{tot_ptok},{tot_ctok+tot_ptok}",
         f"# gpu_hours_wall={tot_ms/3600000:.1f} total_tokens={(tot_ctok+tot_ptok)/1e9:.3f}B "
         f"share_9b_compute={share_9b/tot_ms:.3f} share_9b_ctok={share_ctok_9b/max(tot_ctok,1):.3f}"]
(OUT / "cost_totals.csv").write_text("\n".join(lines) + "\n")
print(f"costs: {tot_dec} decisions, {tot_ms/3600000:.1f} gpu-h, {(tot_ctok+tot_ptok)/1e9:.3f}B tokens")

# --- 5. dataset composition ---
cats = []
for name, cat in BBQ_CATS.items():
    try:
        rows = ds.load_bbq(cat)
    except FileNotFoundError:
        continue  # category not present in raw/
    amb = sum(1 for r in rows if r["context_condition"] == "ambig")
    cats.append(f"{name},{len(rows)},{amb},{len(rows)-amb}")
na_total = sum(1 for _ in open(HERE.parent / 'raw' / 'data' / 'BBQ' / 'analysis_scripts' / 'additional_metadata.csv')) - 1
comp = ["dataset,items,ambig,disambig", *cats,
        f"# crowspairs_total=1508 socioeconomic=172 silverspoon=3000"]
(OUT / "dataset_composition.csv").write_text("\n".join(comp) + "\n")
print("dataset composition OK")
