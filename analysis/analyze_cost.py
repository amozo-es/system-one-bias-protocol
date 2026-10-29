"""Cost per decision: latency and output tokens per model (cluster harness).

Reads timing.latency_ms and usage from each cluster/results record. Jev is
excluded: its API logs carry no per-decision usage/timing.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--logs", default=str(Path(__file__).resolve().parents[1] / "execution" / "cluster" / "results"),
                 help="decision-log dir (<model>__<dataset>.jsonl)")
_ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results"),
                 help="CSV output dir")
_args = _ap.parse_args()
RES = Path(_args.logs)
MODELS = ["von", "laya", "tev1-0.8b", "tev1-4b", "nimble-9b",
          "qwen3.5-4b", "qwen3.5-0.8b", "qwen3.5-9b", "qwen3.5-9b-nothink", "gpt-oss-20b", "clef", "clef-flash"]

FAMILY = {
    "von": "typed encoder", "laya": "typed encoder",
    "clef": "typed schema head", "clef-flash": "typed schema head",
    "nimble-9b": "letter fine-tune", "tev1-4b": "letter fine-tune", "tev1-0.8b": "letter fine-tune",
    "qwen3.5-4b": "letter base", "qwen3.5-0.8b": "letter base", "qwen3.5-9b": "letter base + thinking",
    "qwen3.5-9b-nothink": "letter base (no thinking)",
    "gpt-oss-20b": "letter base + thinking",
}

stats = {}
for mod in MODELS:
    lat, ctok, ptok = [], [], []
    for f in RES.glob(f"{mod}__*.jsonl"):
        if "smoke" in f.name:
            continue
        for line in open(f):
            r = json.loads(line)
            t = r.get("timing") or {}
            u = r.get("usage") or {}
            if isinstance(t.get("latency_ms"), (int, float)):
                lat.append(t["latency_ms"])
            if isinstance(u.get("completion_tokens"), int):
                ctok.append(u["completion_tokens"])
            if isinstance(u.get("prompt_tokens"), int):
                ptok.append(u["prompt_tokens"])
    if lat:
        stats[mod] = {
            "n": len(lat),
            "lat_med": statistics.median(lat),
            "lat_p90": sorted(lat)[int(0.9 * len(lat))],
            "ctok_med": statistics.median(ctok) if ctok else None,
            "ctok_p90": sorted(ctok)[int(0.9 * len(ctok))] if ctok else None,
            "ctok_mean": statistics.mean(ctok) if ctok else None,
            "ptok_med": statistics.median(ptok) if ptok else None,
        }

OUT = Path(_args.out_dir) / "cost_stats.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("model,family,n,lat_med_ms,lat_p90_ms,out_tok_med,out_tok_p90,in_tok_med\n")
print(f"{'model':<13} {'family':<22} {'n':>7} {'lat_med':>8} {'lat_p90':>8} "
      f"{'out_med':>8} {'out_p90':>8} {'out_mean':>9} {'in_med':>7}")
for m in MODELS:
    s = stats.get(m)
    if not s:
        continue
    print(f"{m:<13} {FAMILY[m]:<22} {s['n']:>7} {s['lat_med']:>7.0f} {s['lat_p90']:>8.0f} "
          f"{(s["ctok_med"] or 0):>8.0f} {(s["ctok_p90"] or 0):>8.0f} {(s["ctok_mean"] or 0):>9.0f} {(s["ptok_med"] or 0):>7.0f}")
    with OUT.open("a") as fh:
        fh.write(f"{m},{FAMILY[m]},{s['n']},{s['lat_med']:.1f},{s['lat_p90']:.1f},"
                 f"{s['ctok_med'] or ''},{s['ctok_p90'] or ''},{s['ptok_med'] or ''}\n")
