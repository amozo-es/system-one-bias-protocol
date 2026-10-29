"""Qualitative read of Nimble-9B stereotype alignment on BBQ.

From the cells with significant s_amb (sweep_tables.csv), extracts the ambiguous
items where Nimble-9B picked the stereotypical option (target slot, not UNKNOWN)
under order 0, with context/question/options/probabilities, and notes what it
picked under order 1. Post-hoc: runs no new inference.

Outputs:
  results/nimble_stereotype_items.csv   (samples, up to --per-cell per cell)
  results/nimble_stereotype_digest.md   (readable per-cell digest)

Usage: python analyze_nimble.py [--per-cell 25]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets as ds
from metrics import load_target_locs, unknown_slot
from analyze_sweep import BBQ_CATS, norm_record, rebuild_slots

HERE = Path(__file__).resolve().parent
CLUSTER = HERE.parent / "execution" / "cluster" / "results"
SWEEP = HERE.parent / "results" / "sweep_tables.csv"
OUT = HERE.parent / "results"

MODEL = "nimble-9b"


def significant_amb_cells():
    cells = []
    with open(SWEEP, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != MODEL or row["metric"] != "s_amb" or not row["lo"]:
                continue
            lo, hi = float(row["lo"]), float(row["hi"])
            if (lo > 0 and hi > 0) or (lo < 0 and hi < 0):
                cells.append((row["dataset"], float(row["point"]), lo, hi))
    return cells


def collect(dataset, per_cell):
    cat = BBQ_CATS[dataset]
    items = {r["example_id"]: r for r in ds.load_bbq(cat)}
    tlocs = load_target_locs()
    recs = [norm_record(json.loads(l))
            for l in (CLUSTER / f"{MODEL}__{dataset}.jsonl").read_text().splitlines() if l.strip()]
    rebuild_slots(recs, items)
    by_key = {(r["item_id"], r["order"]): r for r in recs if not r.get("error")}

    def chosen_slot(r):
        a = (r.get("answers") or {}).get("answer") or {}
        sbl = (r.get("meta") or {}).get("slot_by_letter") or {}
        return sbl.get(a.get("choice"))

    n_amb, hits = 0, []
    for item_id, it in sorted(items.items()):
        r0 = by_key.get((item_id, 0))
        if r0 is None or it["context_condition"] != "ambig":
            continue
        tloc = tlocs.get((it["category"], it["question_index"], item_id))
        if tloc is None:
            continue
        unk = unknown_slot(it)
        s0 = chosen_slot(r0)
        if s0 is None:
            continue
        n_amb += 1
        if s0 == tloc and s0 != unk:
            a = r0["answers"]["answer"]
            crit = r0["request"]["questions"]["answer"]["criteria"]
            r1 = by_key.get((item_id, 1))
            hits.append({
                "category": it["category"],
                "item_id": item_id,
                "context": it["context"],
                "question": it["question"],
                "option_target": it[f"ans{tloc}"],
                "chosen": a.get("choice"),
                "chosen_text": crit.get(a.get("choice"), ""),
                "probs": json.dumps({k: round(v, 4) for k, v in (a.get("probabilities") or {}).items()}),
                "conf": round(a.get("confidence") or 0.0, 4),
                "order1_slot": chosen_slot(r1) if r1 else "",
                "order1_target": tloc,
            })
    return items, n_amb, hits[:per_cell], len(hits)


def main():
    global CLUSTER, SWEEP, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=25)
    ap.add_argument("--logs", default=str(CLUSTER),
                    help="decision-log dir (<model>__<dataset>.jsonl)")
    ap.add_argument("--out-dir", default=str(OUT), help="CSV output dir")
    ap.add_argument("--sweep", default="", help="sweep_tables.csv (default: <out-dir>/sweep_tables.csv)")
    args = ap.parse_args()
    CLUSTER, OUT = Path(args.logs), Path(args.out_dir)
    SWEEP = Path(args.sweep) if args.sweep else OUT / "sweep_tables.csv"
    if not SWEEP.exists():
        sys.exit(f"sweep_tables.csv not found at {SWEEP}: run analyze_sweep.py first "
                 f"(or pass --sweep)")

    cells = significant_amb_cells()
    print(f"significant s_amb cells for {MODEL}: {len(cells)}")
    rows, digest = [], ["# Nimble-9B: ambiguous items with stereotypical choice",
                        "",
                        "Post-hoc extract from the cluster JSONL logs (order 0). Only cells with",
                        "significant s_amb. `order1_slot` = slot picked for the same item with options",
                        "swapped (target = order1_target).", ""]
    total_hits = total_amb = 0
    for dataset, point, lo, hi in cells:
        items, n_amb, sample, n_hits = collect(dataset, args.per_cell)
        total_hits += n_hits
        total_amb += n_amb
        digest += [f"## {dataset} (s_amb {point:+.2f} [{lo:+.2f}, {hi:+.2f}])",
                   f"ambiguous items: {n_amb} | stereotypical picks: {n_hits} "
                   f"({100 * n_hits / n_amb:.1f}%)", ""]
        for h in sample:
            rows.append(h)
            digest += [f"- [{h['item_id']}] {h['context']}",
                       f"  - Q: {h['question']} | elegida: {h['chosen_text']} ({h['chosen']}) "
                       f"conf {h['conf']} probs {h['probs']} | target: {h['option_target']} "
                       f"| orden1 eligio slot {h['order1_slot']} (target {h['order1_target']})"]
        digest.append("")
        print(f"  {dataset:26s} ambiguous={n_amb:6d} stereotypical={n_hits:5d} ({100 * n_hits / n_amb:5.1f}%)")

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "nimble_stereotype_items.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        csv_path.write_text("no stereotypical picks in the significant cells\n")
        print("no stereotypical picks found (did analyze_sweep.py run first?)")
    (OUT / "nimble_stereotype_digest.md").write_text("\n".join(digest) + "\n")
    pct = f" ({100 * total_hits / total_amb:.1f}%)" if total_amb else ""
    print(f"\ntotal: {total_hits}/{total_amb}{pct}")
    if rows:
        print(f"CSV -> {csv_path} ({len(rows)} rows)")
        print(f"digest -> {OUT / 'nimble_stereotype_digest.md'}")


if __name__ == "__main__":
    main()
