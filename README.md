# Release package — Auditing Demographic Bias in System One Decision Models

This folder holds the code behind the paper. Two parts, both code-only:

| Folder | Contents | Where it goes |
|---|---|---|
| `execution/` | Harness, model clients, cluster evaluator, dataset/model downloaders | GitHub (code repo) |
| `analysis/` | Scripts that regenerate every aggregate CSV from the decision logs | GitHub (code repo) |

Not included: the raw per-decision logs (~2.3 GB), available on request; the
aggregate CSVs, kept with the logs (see the audit-data note in the project
wiki); and the adapted dataset, already on Hugging Face
(https://huggingface.co/datasets/amozo/system-one-bias-audit, private until
the paper is out).

## 1. Code to GitHub

Use one repository for `execution/` and `analysis/` (both are code).
Repository name: `system-one-bias-protocol`.

```bash
cd release
git init
git add README.md LICENSE execution analysis
git commit -m "System One bias audit: protocol harness, analysis, and reproduction scripts"
# create the repo on GitHub and push
gh repo create <user>/system-one-bias-protocol --public --source . --remote origin --push
git tag v1.0.0
git push origin v1.0.0
```

Before pushing:

- [x] `LICENSE` file (MIT) is present at the root.
- Do not add `.env`, API keys, or personal paths. `execution/.env.example` documents every key the harness reads.
- Cluster launch scripts (`.sbatch`, `run_*.sh`) are not part of the release; the portable entry point is `execution/cluster/eval_systemone.py`.
- `execution/README.md` explains setup, dataset/model downloads, and a single-cell run; `analysis/README.md` explains CSV reproduction.

## 2. Dataset on Hugging Face

Already uploaded: https://huggingface.co/datasets/amozo/system-one-bias-audit
(13 JSONL, 123,000 rows; **private until the paper is out** — switch to public
at launch). Its card declares 14 configs, so each benchmark loads by name:

```python
from datasets import load_dataset
ses = load_dataset("amozo/system-one-bias-audit", "bbq_ses")   # any of the 13 configs
all_rows = load_dataset("amozo/system-one-bias-audit", "all")
```

The adapted items are the exact requests the models saw: the harness renders
them from the official benchmarks with the same builders (`adapt.py`,
`eval_systemone.py`), so the dataset can be regenerated from `raw/` at any
time. Licenses stated in the card: BBQ CC BY 4.0, CrowS-Pairs CC BY-SA 4.0,
SilverSpoon CC BY 4.0. The dataset repository is CC BY-SA 4.0 as a whole
because CrowS-Pairs is ShareAlike; the code repository keeps its own license
(MIT, see `LICENSE`).

## 3. DOI (recommended)

Connect the GitHub repository to Zenodo, cut release `v1.0.0`, and Zenodo mints a DOI. Section 3.1 already links the dataset; add the code repository URL and the DOI to that sentence before submission. A camera-ready footnote or a sentence with the DOI is the usual form.

## 4. Pre-flight checklist

- [x] Dataset uploaded to https://huggingface.co/datasets/amozo/system-one-bias-audit (CC BY-SA 4.0; **private — switch to public at launch**)
- [x] `LICENSE` present in the code repo
- [x] Dataset card licenses checked (BBQ CC BY 4.0, CrowS-Pairs CC BY-SA 4.0, SilverSpoon CC BY 4.0; dataset repo CC BY-SA 4.0), attribution kept
- [ ] HF dataset switched to public at launch
- [ ] `load_dataset("amozo/system-one-bias-audit", "bbq_ses")` works anonymously (configs from the card)
- [ ] No `.env`, keys, or personal paths in the code repo
- [ ] `execution/README.md` quickstart runs on a clean checkout
- [ ] `analysis/README.md` reproduction path regenerates the paper's aggregate CSVs from the decision logs
- [ ] Paper links the code repository and the DOI (dataset URL already in §3.1)
