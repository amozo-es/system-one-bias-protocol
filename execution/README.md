# Execution — audit harness

Code that turns a benchmark item into a typed request, sends it to a model, and logs the answer.

## Layout

```
src/                local harness (Ollama / hosted HTTP APIs; reads raw/data/)
  adapt.py            benchmark -> typed request (prompts live here)
  datasets.py         loaders for the source benchmarks (raw/data/ layout)
  models.py           model clients (Jev, Laya, Ollama, Together, OpenRouter)
  runner.py           local runner: model + dataset cell -> JSONL (Jev route)
  recover_local.py    offline repair of _unparsed answers
  labelswap/          key-label control, run side
    swaplib.py          reference rendering (canonical/exchanged/meaningful keys)
    run_jev_ls.py       Jev arms via the hosted API
    test_swaplib.py     selftest: rendering agrees everywhere (no GPU, no API)
cluster/            canonical sweep evaluator (reads $DATASETS_DIR)
  eval_systemone.py   single-job evaluator, one model x one dataset -> JSONL
  download_datasets.py  fetch BBQ + CrowS-Pairs (SilverSpoon needs its zip)
  download_models.py    fetch the model repos into the HF cache
pyproject.toml, uv.lock, .env.example
```

Two entry points, two data layouts: `cluster/eval_systemone.py` is the
canonical evaluator that produced the audit (data under `$DATASETS_DIR`,
layout of `download_datasets.py`). `src/` covers the hosted-API models that
have no local kind — Jev, audited through its hosted route — and the
labelswap Jev arms. Label-swap arms for the native kinds run through
`eval_systemone.py --letter-order {canonical,exchanged,meaningful}`.

## Setup

```bash
uv sync                          # base: requests + huggingface_hub
uv sync --extra gpu              # + torch/transformers/peft for the causal-LM kinds
uv sync --extra analysis         # + numpy if you also run ../analysis
# plain pip instead of uv:
pip install requests huggingface_hub        # base
pip install torch transformers accelerate peft sentencepiece protobuf pillow  # gpu
```

Encoder scorers need their vendor SDKs, on demand: `uv sync --extra laya`
(and `--extra von` for the `von serve` CLI).

Credentials go in the environment: copy `.env.example` to `.env` and fill in
`OPENROUTER_API_KEY` (Jev) and/or `TOGETHER_API_KEY` (Together baselines).
`src/models.py` loads `.env` automatically; `cluster/eval_systemone.py` reads
the same variables from the shell.

## 1. Download the datasets

BBQ and CrowS-Pairs download automatically from their official repositories
(no token needed). SilverSpoon ships as a zip without a canonical public URL:
get `SilverSpoon.zip` from its authors and pass it with `--silverspoon-zip`.

```bash
export DATASETS_DIR="$PWD/datasets"
python cluster/download_datasets.py                          # BBQ + CrowS-Pairs
python cluster/download_datasets.py --silverspoon-zip ./SilverSpoon.zip
```

Each run verifies the official counts (11 BBQ categories, 1,508 CrowS-Pairs
pairs by bias type, both SilverSpoon csvs inside the zip) and exits non-zero
on any mismatch. Successful output looks like:

```
[bbq] OK: {'Age': 3680, 'Disability_status': 1556, ..., 'SES': 6864, ...}
[crows] OK: 1508 pares, 9 tipos
[silverspoon] OK: datasets/silverspoon/SilverSpoon.zip
```

Layout created under `$DATASETS_DIR`:

```
datasets/
  BBQ/data/<Category>.jsonl          # 11 official categories
  BBQ/analysis_scripts/additional_metadata.csv
  crows-pairs/crows_pairs_anonymized.csv
  silverspoon/SilverSpoon.zip
```

The analysis scripts read the same benchmark sources from `../raw/data`:
point the downloader there instead — `python cluster/download_datasets.py
--datasets-dir ../raw/data` (plus the SilverSpoon zip) — or move the
`datasets/` folder after downloading. Models are optional for the datasets
step; datasets are required for every run.

## 2. Download the models

```bash
export HF_HOME="$PWD/models/hf"          # or your existing HF cache
python cluster/download_models.py --list             # the 12 repos, ~130 GB total
python cluster/download_models.py --only qwen3.5-0.8b
```

After downloading, runs work offline with `HF_HUB_OFFLINE=1`. The Qwen and
gpt-oss bases are public on the Hub; the fine-tunes (Tev1, Nimble) and the
typed encoders (Clef, Laya, Von) may require access requests to their
publishers.

## 3. Run one cell (worked example)

```bash
# 1. no GPU needed: loads every dataset unit and validates request rendering
python cluster/eval_systemone.py --selftest

# 2. one model x one dataset, both option orders, 200 items (needs a CUDA GPU)
python cluster/eval_systemone.py \
  --kind hf_letter \
  --model-path Qwen/Qwen3.5-0.8B \
  --dataset bbq:SES \
  --out results/qwen3.5-0.8b__bbq_ses.jsonl \
  --batch-size 64 --limit 200
```

Expected output:

```
[hf_letter] v14.1 | Qwen/Qwen3.5-0.8B | bbq:SES | pending 400/400
  400/400 (xx.x/s, err 0)
DONE bbq_ses: +400 (err 0) -> results/qwen3.5-0.8b__bbq_ses.jsonl
```

The paper's own sweep used the fine-tunes under the same kinds:
`--kind hf_letter --model-path togethercomputer/Tev1-4B-experimental` (Tev1),
`--kind hf_lora` (Nimble-9B base + adapter via `--adapter`), `--kind
hf_reasoning` (Qwen3.5-9B, gpt-oss-20B), `--kind clef` / `laya` / `von`.

Seeds, orders, and repetition arms follow the paper: shuffle seed 42 for the
main sweep, seeds 7 and 13 for ten configurations (`--seed`), and a fixed
batch of 16 for the batch arm (`--batch-size 16`). The runner is idempotent:
reruns skip completed items (and only those; failed batches are logged with
`_batch_error`).

Datasets: `bbq:<CAT>` (11 categories), `crows[:<bias_type>]`, `silverspoon`.
`python cluster/eval_systemone.py --help` lists every flag.

## Logs

Each row is one decision: request, chosen option, full probability
distribution where the interface exposes one, raw completion head and tail
for generative models, and parsing metadata. `../analysis/` turns these JSONL
logs into the paper's CSVs.
