#!/usr/bin/env python3
"""Jev arm of the labelswap experiment (local API run, no GPU).

Same design as the cluster cells: BBQ SES, order 0, 3 letter-order arms.
Results land in cluster/labelswap/results/jev-or__bbq_ses_ls_{mode}.jsonl
with the same record schema (meta.rendering set per arm).

Requires JEV_API_KEY or OPENROUTER_API_KEY in the environment.

  python run_jev_ls.py --dry-run        # render + validate first items, no API calls
  python run_jev_ls.py --mode exchanged # single arm
  python run_jev_ls.py                  # all 3 arms sequentially
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))

import adapt  # noqa: E402
import runner  # noqa: E402
from labelswap import swaplib  # noqa: E402

RAW = HERE.parents[2] / "raw" / "data" / "BBQ" / "data" / "SES.jsonl"
OUTDIR = HERE.parents[1] / "cluster" / "labelswap" / "results"


def render(item, mode):
    """(state, questions, meta) under the requested letter-order arm."""
    state, questions, meta = adapt.bbq_request(item, 0)
    if mode == "canonical":
        meta = dict(meta, rendering="canonical")
        return state, questions, meta
    criteria = questions["answer"]["criteria"]
    texts = list(criteria.values())
    meaningful = (swaplib.bbq_keys(item, texts, "meaningful")
                  if mode == "meaningful" else None)
    new_criteria, new_slot = swaplib.swap_criteria(
        criteria, meta["slot_by_letter"], mode, meaningful)
    questions = {"answer": {"type": questions["answer"]["type"],
                            "instructions": questions["answer"]["instructions"],
                            "criteria": new_criteria}}
    meta = dict(meta, slot_by_letter=new_slot, rendering=mode)
    return state, questions, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all"] + list(swaplib.MODES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    rows = [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]
    modes = list(swaplib.MODES) if args.mode == "all" else [args.mode]
    for mode in modes:
        if args.dry_run:
            for r in rows[:3]:
                state, questions, meta = render(r, mode)
                keys = list(questions["answer"]["criteria"].keys())
                assert len(set(keys)) == len(keys) == 3, keys
                assert set(meta["slot_by_letter"]) == set(keys)
                print(f"[{mode}] {r['example_id']}: {keys} "
                      f"-> {meta['slot_by_letter']}")
            print(f"[{mode}] dry-run OK (no API calls)")
            continue
        items = [(r["example_id"], lambda o, r=r, m=mode: render(r, m))
                 for r in (rows[: args.limit] if args.limit else rows)]
        out = OUTDIR / f"jev-or__bbq_ses_ls_{mode}.jsonl"
        runner.run("jev-or", f"bbq_ses_ls_{mode}", items, orders=(0,),
                   out=out, workers=args.workers)


if __name__ == "__main__":
    main()
