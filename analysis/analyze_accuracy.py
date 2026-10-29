"""Disambiguated accuracy (acc_disambig) and ambiguous UNKNOWN rate (acc_ambig) per config.

Mean over the 11 BBQ categories from the decision logs (cluster/results), counting
the correct slot per item (extract_bbq_rows -> w_correct). Chance is ~33% with 3 options.

Usage:
  python analyze_accuracy.py                       # -> results/acc_disambig.csv
  python analyze_accuracy.py --out another_path.csv
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_seed_stability as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(S.HERE.parent / "results" / "acc_disambig.csv"))
    ap.add_argument("--logs", default=str(S.BASE),
                    help="decision-log dir (<model>__<dataset>.jsonl)")
    args = ap.parse_args()
    S.BASE = Path(args.logs)

    confs = list(S.SLUG) + (["jev"] if "jev" not in S.SLUG else [])
    rows = ["model,n_cat,acc_disambig,acc_ambig"]
    for m in confs:
        slug = S.SLUG.get(m, m)
        accs, aacc = [], []
        for cell in S.BBQ_CATS:
            recs = S.load_dir(S.BASE, slug, cell)
            if recs is None:
                continue
            _idx, mat = S.item_counts(recs, S.items_for(cell))
            if not len(mat):
                continue
            accs.append(100 * mat[:, 10].sum() / mat[:, 11].sum())
            aacc.append(100 * mat[:, 4].sum() / mat[:, 5].sum())
        if accs:
            rows.append(f"{m},{len(accs)},{statistics.mean(accs):.2f},{statistics.mean(aacc):.2f}")
            print(f"  {m:20s} acc_dis={statistics.mean(accs):5.1f}%  acc_amb={statistics.mean(aacc):5.1f}%"
                  f"  (n_cat={len(accs)})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    print(f"CSV -> {out} ({len(rows) - 1} rows)")


if __name__ == "__main__":
    main()
