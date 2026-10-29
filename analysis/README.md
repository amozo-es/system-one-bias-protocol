# Analysis — from raw decision logs to the released CSVs

Each script reads per-decision JSONL logs and writes CSV aggregates. This
folder ships **code only**: the aggregate CSVs computed for the paper are kept
with the decision logs, outside this release. Paper tables and figures are
typeset in the paper repository.

## Inputs (not in this release; available on request)

Two input families. Scripts read both relative to this folder's parent
(the repository root):

1. Per-decision JSONL logs:

```
../execution/cluster/results/<model>__<dataset>.jsonl   # local + hosted-schema models
../results/jev/<model>__<dataset>.jsonl                 # Jev via OpenRouter
../execution/cluster/results_seed{7,13}/                # repetition arms (seed scripts)
../execution/cluster/labelswap/results/                 # labelswap arms (analyze_labelswap.py)
```

2. The source benchmarks, under `../raw/data/` (needed by `metrics.py` and
   `datasets.py` for official BBQ metadata and item texts):

```
../raw/data/BBQ/data/<Category>.jsonl                # all 11 official categories
../raw/data/BBQ/analysis_scripts/additional_metadata.csv
../raw/data/crows-pairs/crows_pairs_anonymized.csv
../raw/data/silverspoon/SilverSpoon.zip
```

BBQ and CrowS-Pairs download with `execution/cluster/download_datasets.py`
(point `--datasets-dir` at `../raw/data`); SilverSpoon comes from its authors.

## Dependencies

Python 3.12+, standard library only, except `numpy`, used by
`analyze_accuracy.py` and the `--tests` mode of `analyze_seed_stability.py`
(`pip install numpy`).

## Scripts

| Script | Writes | Feeds in the paper |
|---|---|---|
| `analyze_sweep.py` | `sweep_tables.csv` | Main metrics per model × dataset: official BBQ scores with bootstrap CIs, slot-level flip rates, CrowS-Pairs means, SilverSpoon deltas (§4.1–4.6; Tables 7–10; Figures 1–4) |
| `analyze_paired_tests.py` | `paired_tests.csv` | Fine-tune vs. base exact permutation tests with Holm correction (§3.3) |
| `analyze_confidence.py` | `confidence_stats.csv`, `confidence_subsets.csv`, `confidence_calib.csv`, `confidence_noul.csv` | Confidence table (Table 4) and §4.7 |
| `analyze_confidence_reseed.py` | `confidence_reseed.csv` | Confidence vs. change join between shuffle seeds (Table 4) |
| `analyze_seed_stability.py` | `seed_stability*.csv`, `seed_significance*.csv` | Repetition (seed 7 vs. 13) and batch-16 arms (Tables 11–12) |
| `analyze_cost.py` | `cost_stats.csv` | Median latency and token totals (Table 3) |
| `analyze_probmass.py` | `probmass_gaps.csv` | Probability-mass vs. argmax gap behind the abstention analysis (§4.1) |
| `analyze_probmass_conf.py` | `probmass_conf.csv` | Mass-vs-argmax gap among confident answers, ≥0.7 (§4.7) |
| `analyze_accuracy.py` | `acc_disambig.csv` | Disambiguated/ambiguous accuracy used as the parse-validity check (§4.2) |
| `analyze_paper_gaps.py` | `bbq_abstention.csv`, `flip_destinations.csv`, `silverspoon_rates.csv`, `cost_totals.csv`, `volume_counts.csv`, `dataset_composition.csv` | Abstention rates per model, flip destination bands, SilverSpoon justified/wrong rates, decision totals, dataset composition (§3.1, §4.1, §4.2, §4.6) |
| `analyze_nimble.py` | `nimble_stereotype_items.csv` | Post-hoc qualitative sample of stereotypical ambiguous choices (exploratory; no paper table) |
| `analyze_labelswap.py` | `labelswap_results.csv` | Key-label control: text-following rate and TVD per arm pair (§4.4; Table 6) |
| `metrics.py`, `datasets.py` | — | Shared scoring and loading code |

CSV output file names and column layouts match the paper's analysis runs.
When re-running against the audit logs, the suffix variants (`v14_1`,
`batch16`, `08b`) are produced by the flag combinations documented per script
(`--help`). Most scripts take `--help`; `analyze_confidence.py` and
`analyze_paper_gaps.py` run directly.

Bootstrap intervals use 500 item resamples; paired tests use exact
permutation with Holm correction. Both match Section 3.3 of the paper.

## Input/output paths

Every script accepts:

- `--logs DIR` — decision-log directory (default `../cluster/results`)
- `--out-dir DIR` — CSV output directory (default `../results`); scripts that
  write a single CSV take `--out FILE` instead
- `--jev DIR` — Jev decision-log directory, where used (default `../results/jev`)

The benchmark-source directory is configurable with the `AUDIT_RAW`
environment variable (default `../raw`, the directory containing `data/`). `analyze_seed_stability.py` keeps its
own flags (`--base`, `--pair`, `--out`, `--tests-out`); `analyze_sweep.py`,
`analyze_paired_tests.py`, and `analyze_accuracy.py` use `--out FILE`.

## Running (worked example)

```bash
cd analysis
python analyze_sweep.py --models clef-flash --iters 500   # one model first
head -5 ../results/sweep_tables.csv
```

Expected output (values will differ on your logs):

```
== clef-flash (cluster) ==
  clef-flash bbq_ses  s_amb  +12.34 [ +9.11,+15.20] s_dis  -8.90 flip 0.412 parse 0.000 complete
...
CSV -> ../results/sweep_tables.csv (N rows)
```

```
model,type,dataset,metric,point,lo,hi,n,parse_rate,complete
clef-flash,hosted-SO,bbq_ses,s_amb,12.34,9.11,15.20,2964,0.000,complete
...
```

Then run the rest in any order (`python analyze_confidence.py`,
`python analyze_paper_gaps.py`, …) — each writes its CSV into the output
directory (`--out-dir`, default `../results`). To check a reproduction,
compare the resulting CSVs with the numbers reported in the paper.
