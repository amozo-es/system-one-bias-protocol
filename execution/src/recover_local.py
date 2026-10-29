"""Local offline recovery of _unparsed answers (no DATASETS_DIR needed).

Replicates the cluster recovery pass, self-contained: derives the
options from the `request` stored in each record (criteria for choice, fixed
Yes./No. for noul) instead of rebuilding them from the datasets. Same rules,
same output format (_recovered: yesno|option_text).

Usage:
  python src/recover_local.py --models qwen3.5-4b --dry-run
  python src/recover_local.py --models qwen3.5-4b,qwen3.5-9b-nothink,gpt-oss-20b
"""
import argparse
import datetime
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "cluster" / "results"
YESNO = re.compile(r"^\s*(yes|no)\.?\s*$", re.I)

# noul: opciones fijas (eval_systemone.to_letter_task): A=true(Yes.), B=false(No.)
NOUL_LETTERS = {"A": "true", "B": "false"}


def final_segment(t):
    if not t:
        return ""
    i = t.rfind("assistantfinal")
    if i >= 0:
        t = t[i + len("assistantfinal"):]
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1]
    return t.strip()


def recover_record(r):
    """-> True if the record is recovered (modified in place)."""
    a = r.get("answers") or {}
    if "_unparsed" not in a or "_raw" not in a:
        return False
    seg = final_segment(a["_raw"])
    if not seg:
        return False
    questions = ((r.get("request") or {}).get("questions")) or {}
    if len(questions) != 1:
        return False
    (qid, q), = questions.items()
    letter, how = None, None
    if q.get("type") == "noul":
        m = YESNO.match(seg)
        if m:
            letter = "A" if m.group(1).lower() == "yes" else "B"
            how = "yesno"
    else:
        crit = q.get("criteria") or {}
        low = seg.lower()
        hits = [k for k, v in crit.items() if str(v).strip().lower() == low]
        if len(hits) == 1:
            letter, how = hits[0], "option_text"
    if letter is None:
        return False
    if q.get("type") == "noul":
        key = NOUL_LETTERS.get(letter)
        if key is None:
            return False
        ans = {"type": "noul", "noul": 1.0 if key == "true" else 0.0,
               "_raw": a["_raw"], "_recovered": how}
    else:
        if letter not in (q.get("criteria") or {}):
            return False
        ans = {"type": "choice", "choice": letter,
               "_raw": a["_raw"], "_recovered": how}
    r["error"] = False
    r["answers"] = {qid: ans}
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="",
                    help="coma de modelos; por defecto todos los letter")
    ap.add_argument("--dry-run", action="store_true",
                    help="solo cuenta recuperables, no escribe")
    args = ap.parse_args()
    only = {s.strip() for s in args.models.split(",") if s.strip()}
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = RESULTS.parent / f"results_backup_{stamp}"
    total_rec = total_left = 0
    for path in sorted(RESULTS.glob("*__*.jsonl")):
        if "smoke" in path.name:
            continue
        mod = path.name.split("__", 1)[0]
        if only and mod not in only:
            continue
        lines = path.read_text().splitlines()
        rec = left = 0
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("error"):
                continue
            if recover_record(r):
                rec += 1
                lines[i] = json.dumps(r, ensure_ascii=False)
            else:
                left += 1
        total_rec += rec
        total_left += left
        print(f"{path.name:<48} recoverable: {rec:>6}  residual: {left:>6}")
        if rec and not args.dry_run:
            bak.mkdir(exist_ok=True)
            dst = bak / path.name
            if not dst.exists():
                shutil.copy2(path, dst)
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(lines) + "\n")
            tmp.replace(path)
    print(f"TOTAL recovered: {total_rec}, residual: {total_left}")
    if not args.dry_run and total_rec:
        print(f"backup at {bak} (only modified files without a previous backup)")


if __name__ == "__main__":
    main()
