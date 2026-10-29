"""Dataset loaders. raw/ is read-only; loaders never write there."""
import csv
import io
import json
import zipfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root (raw/ lives there)
# move the benchmark sources with AUDIT_RAW (default: <repo>/raw; contains data/)
RAW = (Path(os.environ["AUDIT_RAW"]) if os.environ.get("AUDIT_RAW")
       else ROOT / "raw") / "data"


def load_bbq(category):
    """category: SES | Race_x_SES. Returns list of raw BBQ dicts."""
    path = RAW / "BBQ" / "data" / f"{category}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_crows(bias_type="socioeconomic"):
    path = RAW / "crows-pairs" / "data" / "crows_pairs_anonymized.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if bias_type:
        rows = [r for r in rows if r["bias_type"] == bias_type]
    return rows


def load_silverspoon(which="human"):
    """which: 'human' (silverSpoonHumanBinary.csv) or 'llms'."""
    fname = f"silverSpoon{which.capitalize()}Binary.csv"
    with zipfile.ZipFile(RAW / "silverspoon" / "SilverSpoon.zip") as z:
        with z.open(fname) as f:
            return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))


# SocioEval: templates not public at download time. If the authors' dataset
# (Enkrypt AI) becomes available, add loader here; otherwise reconstruct from
# the paper's appendix (kumar-2026-socioeval-template-based.pdf).
def load_socioeval():
    raise NotImplementedError(
        "SocioEval templates pending: check paper appendix or author release"
    )


if __name__ == "__main__":
    bbq = load_bbq("SES")
    rx = load_bbq("Race_x_SES")
    cr = load_crows()
    ss = load_silverspoon()
    print(f"BBQ SES={len(bbq)} Race_x_SES={len(rx)} CrowS-SES={len(cr)} SilverSpoon={len(ss)}")
    print("CrowS bias_types:", sorted({r['bias_type'] for r in load_crows('') or []}) or "n/a")
