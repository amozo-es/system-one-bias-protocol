"""Runner: executes (model x dataset x item x rep) with incremental JSONL saving.

Idempotent: reruns skip (model, dataset, item_id, rep, order) keys already present.
Every line stores the full request and raw response for auditability.
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from models import Retry, get_model

RUNS = Path(__file__).resolve().parents[1] / "runs"


# BBQ category -> metadata/jsonl category name; CrowS bias_type registry.
BBQ_CATS = {
    "bbq_ses": "SES", "bbq_racexses": "Race_x_SES", "bbq_age": "Age",
    "bbq_disability_status": "Disability_status", "bbq_gender_identity": "Gender_identity",
    "bbq_nationality": "Nationality", "bbq_physical_appearance": "Physical_appearance",
    "bbq_race_ethnicity": "Race_ethnicity", "bbq_race_x_gender": "Race_x_gender",
    "bbq_religion": "Religion", "bbq_sexual_orientation": "Sexual_orientation",
}
CROWS_TYPES = {
    "crows": "socioeconomic", "crows_race_color": "race-color", "crows_gender": "gender",
    "crows_nationality": "nationality", "crows_religion": "religion", "crows_age": "age",
    "crows_sexual_orientation": "sexual-orientation",
    "crows_physical_appearance": "physical-appearance", "crows_disability": "disability",
}


def _key(model, dataset, item_id, rep, order):
    return f"{model}|{dataset}|{item_id}|{rep}|{order}"


def load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add(_key(r["model"], r["dataset"], r["item_id"], r["rep"], r["order"]))
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate truncated tail lines
    return done


def run(model_name, dataset_name, items, k_reps=1, orders=(0, 1), seed=42, out=None, workers=1):
    """items: list of (item_id, request_fn) where request_fn(order) -> (state, questions, meta)."""
    RUNS.mkdir(exist_ok=True)
    out = out or RUNS / f"{model_name}__{dataset_name}.jsonl"
    done = load_done(out)
    model = get_model(model_name)
    retry = Retry()
    rng = random.Random(seed)
    pending = [
        (item_id, order, rep)
        for item_id, _ in items
        for order in orders
        for rep in range(k_reps)
    ]
    pending = [t for t in pending if _key(model_name, dataset_name, t[0], t[2], t[1]) not in done]
    rng.shuffle(pending)  # decouple run order from item order (drift protection)
    req_fns = dict(items)
    n = 0
    t0 = time.time()
    lock = threading.Lock()

    def one(job):
        item_id, order, rep = job
        state, questions, meta = req_fns[item_id](order)
        try:
            answers = retry.call(model.ask, state, questions)
            error = None
        except Exception as e:  # noqa: BLE001 - log and continue
            answers, error = None, f"{type(e).__name__}: {e}"
        rec = {
            "model": model_name,
            "dataset": dataset_name,
            "item_id": item_id,
            "rep": rep,
            "order": order,
            "meta": meta,
            "request": {"state": state, "questions": questions},
            "answers": answers,
            "error": error,
            "ts": time.time(),
        }
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(one, pending)):
            n = i + 1
            if n % 100 == 0:
                rate = n / (time.time() - t0)
                print(f"{model_name}/{dataset_name}: {n}/{len(pending)} new, {rate:.1f}/s", flush=True)
    print(f"DONE {model_name}/{dataset_name}: +{n} records -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dataset", choices=sorted(list(BBQ_CATS) + list(CROWS_TYPES) + ["silverspoon"]))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke)")
    ap.add_argument("--orders", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1, help="parallel HTTP calls")
    args = ap.parse_args()

    import adapt
    import datasets as ds

    items = []
    if args.dataset in BBQ_CATS:
        rows = ds.load_bbq(BBQ_CATS[args.dataset])
        if args.limit:
            rows = rows[: args.limit]
        items = [(r["example_id"], lambda o, r=r: adapt.bbq_request(r, o)) for r in rows]
    elif args.dataset in CROWS_TYPES:
        rows = ds.load_crows(CROWS_TYPES[args.dataset])
        if args.limit:
            rows = rows[: args.limit]
        items = [(i, lambda o, r=r: adapt.crows_request(r, o)) for i, r in enumerate(rows)]
    elif args.dataset == "silverspoon":
        rows = ds.load_silverspoon()
        if args.limit:
            rows = rows[: args.limit]
        items = [(i, lambda o, r=r: adapt.silverspoon_request(r)) for i, r in enumerate(rows)]

    run(args.model, args.dataset, items, k_reps=args.reps, orders=tuple(args.orders),
        seed=args.seed, workers=args.workers)
