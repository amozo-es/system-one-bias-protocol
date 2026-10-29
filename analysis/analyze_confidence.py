"""Confidence diagnostics: does observed bias modulate confidence?

Reads the JSONL logs already on disk (cluster + jev); runs no model.

Provenance of the `confidence` field (see cluster/eval_systemone.py):
  - natives (von, laya, jev, clef): vendor scalar.
  - local-letter (tev1-4b, tev1-0.8b, nimble-9b): P(chosen letter token),
    softmax over candidates. Only on BBQ/choice; absent on noul.
  - qwen + gpt-oss bases: not emitted (free text + choice).
  - jev/von/tev1/nimble on noul: no field.

Outputs (results/):
  - confidence_stats.csv: coverage + per-condition means
  - confidence_subsets.csv: official s_amb/s_dis recomputed on
    conf>=HI and conf<LO, with N (sanity: _all column == sweep_tables)
  - confidence_calib.csv: accuracy per confidence bin (BBQ disambiguated)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import datasets as ds
from metrics import bbq_bootstrap_ci, bootstrap_ci, load_target_locs, score_bbq, score_crows, unknown_slot

CLUSTER = HERE.parent / "execution" / "cluster" / "results"
JEV = HERE.parent / "results" / "jev"
OUT = HERE.parent / "results"

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

HI = 0.7
LO = 0.5


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


def pick_slot(rec, items):
    """Slot argmax (same logic as metrics.score_bbq). Returns (best, probs_dict)."""
    item = items[rec["item_id"]]
    slot_by_letter = rec["meta"]["slot_by_letter"]
    ans = rec["answers"]["answer"]
    probs = ans.get("probabilities") or {}
    if probs:
        mass = {slot_by_letter.get(l): p for l, p in probs.items() if l in slot_by_letter}
    else:
        chosen = ans.get("choice")
        letter = next((l for l, s in slot_by_letter.items()
                       if chosen is not None and items[rec["item_id"]][f"ans{s}"] == chosen), None)
        mass = {slot_by_letter[letter]: 1.0} if letter else {}
    mass = {s: p for s, p in mass.items() if s is not None}
    if not mass:
        return None, {}
    return max(mass, key=mass.get), mass


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def q(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, int(p * len(s)))
    return s[i]


def main():
    global CLUSTER, JEV, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(CLUSTER),
                    help="decision-log dir (<model>__<dataset>.jsonl)")
    ap.add_argument("--jev", default=str(JEV), help="Jev decision-log dir")
    ap.add_argument("--out-dir", default=str(OUT), help="CSV output dir")
    args = ap.parse_args()
    CLUSTER, JEV, OUT = Path(args.logs), Path(args.jev), Path(args.out_dir)
    MODELS[:] = [(n, (JEV if n == "jev" else CLUSTER), s) for n, _, s in MODELS]

    stats = ["model,dataset,prim,n_total,n_conf,conf_cover,conf_mean,conf_p10,conf_p50,conf_p90,"
             "conf_aligned,conf_other,n_aligned,n_other,conf_flip,conf_stable,n_flip_items,"
             "conf_unk,conf_answered"]
    subsets = ["model,dataset,s_amb_all,s_amb_hi,s_amb_lo,s_dis_all,s_dis_hi,s_dis_lo,n_all,n_hi,n_lo,"
               "s_amb_hi_lo,s_amb_hi_hi,s_amb_lo_lo,s_amb_lo_hi"]
    calib = ["model,dataset,bin_lo,bin_hi,acc,n"]
    noul_rows = ["model,dataset,stat_all,stat_hi,stat_lo,ci_hi_lo,ci_hi_hi,ci_lo_lo,ci_lo_hi,n_hi,n_lo"]
    coverage_notes = []
    tlocs = load_target_locs()

    for model, base, slug in MODELS:
        for name in list(BBQ_CATS) + ["crows", "silverspoon"]:
            recs = load(base, slug, name)
            if recs is None:
                continue
            valid = [r for r in recs if not r.get("error")]
            qid = "answer" if name.startswith("bbq") else ("stereotype" if name == "crows" else "justified")
            confs = [(r, (r["answers"].get(qid) or {}).get("confidence")) for r in valid]
            have = [(r, c) for r, c in confs if isinstance(c, (int, float))]
            if not have:
                coverage_notes.append(f"{model}/{name}: sin confidence (n={len(valid)})")
                continue
            cs = [c for _, c in have]
            prim = "choice" if name.startswith("bbq") else "noul"
            row = {"model": model, "dataset": name, "prim": prim, "n_total": len(valid),
                   "n_conf": len(have), "conf_cover": len(have) / len(valid),
                   "conf_mean": mean(cs), "conf_p10": q(cs, .1), "conf_p50": q(cs, .5), "conf_p90": q(cs, .9)}
            if name.startswith("bbq"):
                items = {r["example_id"]: r for r in ds.load_bbq(BBQ_CATS[name])}
                fix_slots(recs, items)
                al, ot, unk, ansd = [], [], [], []
                by_item = {}
                for r, c in have:
                    if r["item_id"] not in items:
                        continue
                    best, _ = pick_slot(r, items)
                    if best is None:
                        continue
                    item = items[r["item_id"]]
                    key = (item["category"], str(item["question_index"]), r["item_id"])
                    tloc = tlocs.get(key)
                    u = unknown_slot(item)
                    (al if best == tloc else ot).append(c)
                    (unk if best == u else ansd).append(c)
                    by_item.setdefault(r["item_id"], []).append((r["order"], best, c))
                flips, stable = [], []
                for iid, os_ in by_item.items():
                    slots = {b for _, b, _ in os_}
                    if len(os_) < 2:
                        continue
                    (flips if len(slots) > 1 else stable).append(mean([c for _, _, c in os_]))
                row.update(conf_aligned=mean(al), conf_other=mean(ot), n_aligned=len(al), n_other=len(ot),
                           conf_flip=mean(flips), conf_stable=mean(stable), n_flip_items=len(flips),
                           conf_unk=mean(unk), conf_answered=mean(ansd))
                sc_all = score_bbq(recs, items, BBQ_CATS[name])
                rec_hi = [r for r in recs if isinstance(((r.get("answers") or {}).get(qid) or {}).get("confidence"), (int, float))
                          and ((r["answers"][qid]["confidence"]) >= HI)]
                rec_lo = [r for r in recs if isinstance(((r.get("answers") or {}).get(qid) or {}).get("confidence"), (int, float))
                          and ((r["answers"][qid]["confidence"]) < LO)]
                sc_hi = score_bbq(rec_hi, items, BBQ_CATS[name])
                sc_lo = score_bbq(rec_lo, items, BBQ_CATS[name])
                # bootstrap CI (500 iters, as the sweep) over the eleven categories
                ci_hi, ci_lo = ",,", ","
                for tag, rr, slot in (("hi", rec_hi, 0), ("lo", rec_lo, 1)):
                    txt = ","
                    if len(rr) > 1:
                        try:
                            ci = bbq_bootstrap_ci(rr, items, iters=500)
                            txt = f"{ci['s_ambig'][0]:.2f},{ci['s_ambig'][1]:.2f}"
                        except (KeyError, IndexError, ValueError):
                            txt = ","
                    if slot == 0:
                        ci_hi = txt + ","
                    else:
                        ci_lo = txt
                subsets.append(f"{model},{name},{sc_all['s_amb']:.2f},{sc_hi['s_amb']:.2f},{sc_lo['s_amb']:.2f},"
                               f"{sc_all['s_dis']:.2f},{sc_hi['s_dis']:.2f},{sc_lo['s_dis']:.2f},"
                               f"{sc_all['n_ambig'] + sc_all['n_disambig']},"
                               f"{sc_hi['n_ambig'] + sc_hi['n_disambig']},{sc_lo['n_ambig'] + sc_lo['n_disambig']},"
                               f"{ci_hi}{ci_lo}")
                # calibration on disambiguated: hit = argmax == label
                for blo, bhi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
                    ok = tot = 0
                    for r, c in have:
                        if r["item_id"] not in items:
                            continue
                        item = items[r["item_id"]]
                        if item["context_condition"] != "disambig":
                            continue
                        if not (blo <= c < bhi):
                            continue
                        best, _ = pick_slot(r, items)
                        if best is None:
                            continue
                        tot += 1
                        ok += 1 if best == item["label"] else 0
                    calib.append(f"{model},{name},{blo},{bhi},{(ok / tot if tot else float('nan')):.4f},{tot}")
            else:
                # noul with confidence: split by chosen side + cross-order agreement
                side_t, side_f, agree, disag = [], [], [], []
                by_item = {}
                for r, c in have:
                    v = (r["answers"].get(qid) or {}).get("noul")
                    if not isinstance(v, (int, float)):
                        continue
                    (side_t if v > 0.5 else side_f).append(c)
                    by_item.setdefault(r["item_id"], []).append((r["order"], v, c))
                for iid, os_ in by_item.items():
                    if len(os_) < 2:
                        continue
                    sides = {(v > 0.5) for _, v, _ in os_}
                    (agree if len(sides) == 1 else disag).append(mean([c for _, _, c in os_]))
                row.update(conf_aligned=mean(side_t), conf_other=mean(side_f),
                           n_aligned=len(side_t), n_other=len(side_f),
                           conf_flip=mean(disag), conf_stable=mean(agree), n_flip_items=len(disag),
                           conf_unk=float("nan"), conf_answered=float("nan"))
                # confidence subsets on noul: CrowS p_stereo / Spoon delta
                rec_hi = [r for r, c in have if c >= HI]
                rec_lo = [r for r, c in have if c < LO]
                if name == "crows":
                    def prefs(rr):
                        return [(r["answers"]["stereotype"]["noul"] if r["meta"]["s1_is_stereo"]
                                 else 1.0 - r["answers"]["stereotype"]["noul"]) for r in rr]
                    stat_all, ph, pl = mean(prefs([r for r, _ in have])), prefs(rec_hi), prefs(rec_lo)
                    stat_hi, stat_lo = mean(ph), mean(pl)
                else:  # silverspoon
                    items_by_index = {i: r for i, r in enumerate(ds.load_silverspoon())}

                    def triples(rr):
                        out = {}
                        for r in rr:
                            pv = r["answers"]["justified"]["noul"]
                            i = int(r["item_id"])
                            if i in items_by_index:
                                it = items_by_index[i]
                                out[i] = (pv, it["Up_bin"], it["Low_bin"])
                        return list(out.values())

                    def delta(pr):
                        n = len(pr)
                        if not n:
                            return float("nan")
                        au = sum(1 for pv, up, _ in pr if (pv >= .5) == (up == "1")) / n
                        al = sum(1 for pv, _, lo in pr if (pv >= .5) == (lo == "1")) / n
                        return au - al

                    stat_all = delta(triples([r for r, _ in have]))
                    th, tl = triples(rec_hi), triples(rec_lo)
                    ph, pl = th, tl
                    stat_hi, stat_lo = delta(th), delta(tl)
                try:
                    stat_fn = delta if name == "silverspoon" else None
                    chi = (bootstrap_ci(ph, stat=stat_fn, iters=500) if len(ph) > 1
                           else (float("nan"), float("nan")))
                    clo = (bootstrap_ci(pl, stat=stat_fn, iters=500) if len(pl) > 1
                           else (float("nan"), float("nan")))
                except (KeyError, IndexError, ValueError):
                    chi = clo = (float("nan"), float("nan"))
                noul_rows.append(f"{model},{name},{stat_all:.4f},{stat_hi:.4f},{stat_lo:.4f},"
                                 f"{chi[0]:.4f},{chi[1]:.4f},{clo[0]:.4f},{clo[1]:.4f},{len(rec_hi)},{len(rec_lo)}")
            stats.append(",".join(str(row[k]) for k in
                                  ["model", "dataset", "prim", "n_total", "n_conf", "conf_cover", "conf_mean",
                                   "conf_p10", "conf_p50", "conf_p90", "conf_aligned", "conf_other",
                                   "n_aligned", "n_other", "conf_flip", "conf_stable", "n_flip_items",
                                   "conf_unk", "conf_answered"]))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "confidence_stats.csv").write_text("\n".join(stats) + "\n")
    (OUT / "confidence_subsets.csv").write_text("\n".join(subsets) + "\n")
    (OUT / "confidence_calib.csv").write_text("\n".join(calib) + "\n")
    (OUT / "confidence_noul.csv").write_text("\n".join(noul_rows) + "\n")
    print(f"stats={len(stats) - 1} subsets={len(subsets) - 1} calib={len(calib) - 1}")
    print("SIN CONFIDENCE:")
    for n in coverage_notes:
        print(" ", n)


if __name__ == "__main__":
    main()