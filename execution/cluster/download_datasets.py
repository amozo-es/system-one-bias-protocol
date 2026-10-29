#!/usr/bin/env python3
"""Download datasets for the System One socioeconomic bias audit to $DATASETS_DIR.

Sources:
  BBQ          github.com/nyu-mll/BBQ            (11 category jsonl + additional_metadata.csv)
  CrowS-Pairs  github.com/nyu-mll/crows-pairs    (data/crows_pairs_anonymized.csv)
  SilverSpoon  distributed as a zip (no canonical public URL); pass --silverspoon-zip PATH
               pointing at the SilverSpoon.zip from the project's raw/data bundle.

Usage:
  python download_datasets.py                     # BBQ + CrowS only
  python download_datasets.py --silverspoon-zip ./SilverSpoon.zip
"""
import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
def _env_path(var, default):
    """Env var treated as unset when empty (exported-empty shells are common)."""
    import os
    v = os.environ.get(var, "").strip()
    return Path(v) if v else Path(default)



BBQ_ZIP = "https://github.com/nyu-mll/BBQ/archive/refs/heads/master.zip"
CROWS_CSV = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

EXPECTED = {  # category -> (items, jsonl name)
    "Age": 3680, "Disability_status": 1556, "Gender_identity": 5672,
    "Nationality": 3080, "Physical_appearance": 1576, "Race_ethnicity": 6880,
    "Race_x_SES": 11160, "Race_x_gender": 15960, "Religion": 1200,
    "SES": 6864, "Sexual_orientation": 864,
}
CROWS_TOTAL = 1508
CROWS_TYPES = {"race-color": 516, "gender": 262, "socioeconomic": 172, "nationality": 159,
               "religion": 105, "age": 87, "sexual-orientation": 84,
               "physical-appearance": 63, "disability": 60}


def fetch(url):
    print(f"  GET {url}")
    return urllib.request.urlopen(url, timeout=120).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default=None, help="default: $DATASETS_DIR")
    ap.add_argument("--silverspoon-zip", default="", help="path to SilverSpoon.zip (optional)")
    args = ap.parse_args()
    out = Path(args.datasets_dir) if args.datasets_dir else _env_path("DATASETS_DIR", "datasets")
    out.mkdir(parents=True, exist_ok=True)
    ok = True

    # ---- BBQ ----
    bbq_dir = out / "BBQ" / "data"
    if all((bbq_dir / f"{c}.jsonl").exists() for c in EXPECTED):
        print(f"[bbq] already present at {bbq_dir}")
    else:
        print("[bbq] downloading github.com/nyu-mll/BBQ ...")
        zf = zipfile.ZipFile(io.BytesIO(fetch(BBQ_ZIP)))
        first = next(n for n in zf.namelist() if n.endswith("/data/SES.jsonl"))
        top = first.split("data/SES.jsonl")[0]  # zip top-level repo dir
        bbq_dir.mkdir(parents=True, exist_ok=True)
        for name in zf.namelist():
            if (name.startswith(top) and name.endswith(".jsonl")) or name.endswith("additional_metadata.csv"):
                target = bbq_dir.parent / name[len(top):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
    counts = {}
    for cat, want in EXPECTED.items():
        f = bbq_dir / f"{cat}.jsonl"
        n = sum(1 for _ in f.open()) if f.exists() else -1
        counts[cat] = n
        if n != want:
            print(f"  ✗ {cat}: {n} != {want}"); ok = False
    meta = next((p for p in [bbq_dir.parent / "analysis_scripts" / "additional_metadata.csv",
                             bbq_dir.parent / "additional_metadata.csv"] if p.exists()), None)
    if not meta:
        print("  ✗ missing additional_metadata.csv"); ok = False
    print(f"[bbq] {'OK' if ok else 'ERROR'}: {counts}")

    # ---- CrowS-Pairs ----
    crows = out / "crows-pairs" / "crows_pairs_anonymized.csv"
    if not crows.exists():
        crows.parent.mkdir(parents=True, exist_ok=True)
        crows.write_bytes(fetch(CROWS_CSV))
    import csv
    rows = list(csv.DictReader(crows.open(encoding="utf-8")))
    from collections import Counter
    types = Counter(r["bias_type"] for r in rows)
    if len(rows) != CROWS_TOTAL or dict(types) != CROWS_TYPES:
        print(f"  ✗ crows: {len(rows)} rows, {dict(types)}"); ok = False
    else:
        print(f"[crows] OK: {len(rows)} pairs, {len(types)} types")

    # ---- SilverSpoon ----
    ss = out / "silverspoon" / "SilverSpoon.zip"
    if args.silverspoon_zip:
        import shutil
        src = Path(args.silverspoon_zip)
        if src.exists() and src.resolve() != ss.resolve():
            ss.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, ss)  # copy, never move (sources may be read-only)
        elif not src.exists():
            print(f"  ! SilverSpoon.zip not found at {src}: copy it from the project bundle "
                  "(raw/data/silverspoon/SilverSpoon.zip) and retry; meanwhile, "
                  "bbq:* and crows* cells can start now.")
    if ss.exists():
        with zipfile.ZipFile(ss) as z:
            names = z.namelist()
        need = ["silverspoonhumanbinary.csv", "silverspoonllmsbinary.csv"]
        lowered = [n.lower() for n in names]
        if all(any(n.endswith(x) for n in lowered) for x in need):
            print(f"[silverspoon] OK: {ss}")
        else:
            print(f"  ✗ silverspoon: missing {need} in the zip"); ok = False
    else:
        print("[silverspoon] SKIPPED: pass --silverspoon-zip (no canonical public URL)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
