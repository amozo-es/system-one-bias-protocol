# Swap, Replicate, Read the Mass

### A Protocol for Auditing Demographic Bias in System One Decision Models

Alejandro Mozo Quesada · Sergio Gálvez Rojas · Francisco López Valverde — Universidad de Málaga

![arXiv](https://img.shields.io/badge/arXiv-PENDING-b31b1b.svg)
[![DOI](https://img.shields.io/badge/Zenodo%20DOI-10.5281%2Fzenodo.23763681-blue.svg)](https://doi.org/10.5281/zenodo.23763681)
[![Hugging Face dataset](https://img.shields.io/badge/dataset-amozo%2Fsystem--one--bias--audit-yellow.svg)](https://huggingface.co/datasets/amozo/system-one-bias-audit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Typed decision models answer structured questions with calibrated probabilities
instead of generating text, and are entering production decision paths. This
repository releases the audit protocol we built to test them for demographic
bias, together with the harness that ran it across thirteen configurations
(including the commercial reference Jev) and the analysis that produced every
number in the paper — 1,638,000 decisions over BBQ, CrowS-Pairs, and
SilverSpoon.

## The protocol

1. **Swap.** Every order-sensitive item runs twice with answer options
   swapped (native typed models also swap the option labels); boolean items
   run with the question reversed.
2. **Replicate.** Configurations repeat under new shuffle seeds and at a fixed
   batch size, bounding execution noise.
3. **Read the mass.** Scores read the full probability distribution, not only
   the winning answer, so abstention cannot hide alignment.

Adapt, do not invent: the protocol adapts three established benchmarks —
BBQ (all eleven categories), CrowS-Pairs (nine bias types), and SilverSpoon —
re-rendered onto the typed interface. The adapted items are on
[Hugging Face](https://huggingface.co/datasets/amozo/system-one-bias-audit)
(CC BY-SA 4.0, public).

## Repository layout

```
execution/   run the models -> per-decision JSONL logs
  cluster/     canonical evaluator (one model x one dataset), dataset and
               model downloaders
  src/         hosted-API route (Jev), request builders, label-swap arms
analysis/    JSONL logs -> the paper's aggregate CSVs (scoring, stability,
             confidence, costs, label-swap control)
```

## Quickstart

```bash
# base install (requests + huggingface_hub); GPU kinds need --extra gpu
cd execution && uv sync --extra gpu

# 1. datasets: BBQ + CrowS-Pairs download automatically; SilverSpoon zip from its authors
export DATASETS_DIR="$PWD/datasets"
python cluster/download_datasets.py

# 2. one model x one dataset, both option orders
python cluster/eval_systemone.py --selftest          # no GPU needed
python cluster/eval_systemone.py --kind hf_letter \
    --model-path Qwen/Qwen3.5-0.8B --dataset bbq:SES \
    --out results/qwen3.5-0.8b__bbq_ses.jsonl --batch-size 64

# 3. logs -> CSVs
cd ../analysis && python analyze_sweep.py
```

Hosted models authenticate through the environment (`OPENROUTER_API_KEY`,
`TOGETHER_API_KEY`); see `execution/.env.example`. Full details:
`execution/README.md` and `analysis/README.md`. The per-decision logs behind
the paper (~2.3 GB) are available on request.

## Citation

```bibtex
@inproceedings{mozo2026swap,
  title     = {Swap, Replicate, Read the Mass: A Protocol for Auditing
               Demographic Bias in System One Decision Models},
  author    = {Mozo Quesada, Alejandro and G{\'a}lvez Rojas, Sergio and
               L{\'o}pez Valverde, Francisco},
  year      = {2026},
  note      = {arXiv preprint coming soon}
}
```

## License

Code is released under the [MIT License](LICENSE). The benchmarks keep their
own licenses (BBQ CC BY 4.0, CrowS-Pairs CC BY-SA 4.0, SilverSpoon CC BY 4.0);
the adapted dataset on Hugging Face is CC BY-SA 4.0 as a whole.
