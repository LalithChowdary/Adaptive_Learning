"""Train the v2 learner-memory model ladder and save an auditable run."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import sklearn
import torch

from memory.data import DATA_ROOT, load_partition
from memory.evaluation import groups, paired_intervals, probability_report
from memory.tabular import TABULAR, fit_tabular, predict_tabular
from memory.training import fit_recurrent, make_batches, predict_recurrent, training_device


RECURRENT = ("time_input_gru", "time_decay_gru")
MODELS = (*TABULAR, *RECURRENT)
TIE_TOLERANCE = 0.001
MODEL_DIR = Path(__file__).resolve().parent


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def export_predictions(path, partition, predictions):
    fields = ("identifier", "learner", "skill", "purpose", "delay_days", "task_type")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow((*fields, "label", *predictions))
        for index, query in enumerate(partition.queries):
            metadata = [getattr(query, name) for name in fields]
            values = (float(p[index]) for p in predictions.values())
            writer.writerow((*metadata, int(partition.labels[index]), *values))


def write_report(output, run):
    lines = [
        "# Learner-memory model comparison", "",
        f"Run type: {run['mode']}. Recurrent device: `{run['recurrent_device']}`.", "",
        f"Validation choice: **`{run['selected_model']}`**. Models within {TIE_TOLERANCE} "
        "of the lowest validation delayed log loss are ranked by simplicity.", "",
        "| Model | Training device | Validation delayed log loss |",
        "|---|---|---:|",
    ]
    for name, result in run["models"].items():
        lines.append(f"| {name} | {result['device']} | {result['validation_delayed_log_loss']:.5f} |")

    for split, comparison in run["evaluation"].items():
        lines.extend([
            "", f"## {split}", "",
            "Each cell gives log loss / Brier score / 10-bin calibration error.", "",
            "| Model | Ordinary | Follow-up | 1 day | 7 days | 30 days |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, model_report in comparison["metrics"].items():
            values = []
            for group in ("ordinary", "followup", "day_1", "day_7", "day_30"):
                result = model_report.get(group)
                values.append(
                    " / ".join(f"{result[key]:.5f}" for key in ("log_loss", "brier", "ece_10"))
                    if result else "—"
                )
            lines.append(f"| {name} | {' | '.join(values)} |")
    lines.extend([
        "", "The JSON report includes query counts, calibration bins, learner-level paired "
        "confidence intervals, run settings, and the recurrent learning curves.",
        "Performance is measured on the synthetic v2 benchmark.", "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def experiment(args, device, run):
    started = time.monotonic()
    cohort = "pilot" if args.mode == "smoke" else "main"
    limit = 12 if args.mode == "smoke" else None
    train = load_partition(args.dataset, cohort, "train", learner_limit=limit)
    validation = load_partition(args.dataset, cohort, "validation", learner_limit=limit)
    train_learners = {query.learner for query in train.queries}
    validation_learners = {query.learner for query in validation.queries}
    if train_learners & validation_learners:
        raise ValueError("Training and validation learners overlap")
    if args.mode == "full" and (len(train_learners), len(validation_learners), len(train.labels)) != (1600, 200, 185510):
        raise ValueError("Unexpected main v2 fitting or validation coverage")
    run["training"] = {"learners": len(train_learners), "queries": len(train.labels)}
    print(f"{cohort}: {len(train_learners)} training learners, {len(train.labels):,} eligible queries", flush=True)

    masks = groups(validation.queries)
    models = {}
    for name in TABULAR:
        print(f"Fitting {name} on CPU", flush=True)
        model, result = fit_tabular(name, train, validation, masks, seed=args.seed)
        models[name], run["models"][name] = model, result
        folder = args.output / name
        folder.mkdir()
        joblib.dump(model, folder / "model.joblib")
        write_json(folder / "fit.json", result)

    training_batches = make_batches(train, args.batch_size, device)
    validation_batches = make_batches(validation, args.batch_size, device)
    for name in RECURRENT:
        mode = name.removesuffix("_gru")
        print(f"Fitting {name} on {device}", flush=True)
        model, result = fit_recurrent(
            mode, training_batches, validation, validation_batches, masks, device,
            epochs=args.epochs, seed=args.seed,
        )
        models[name], run["models"][name] = model, result
        folder = args.output / name
        folder.mkdir()
        torch.save({
            "mode": mode, "hidden_size": model.hidden_size,
            "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }, folder / "model.pt")
        write_json(folder / "fit.json", result)
    del training_batches, train

    scores = {name: info["validation_delayed_log_loss"] for name, info in run["models"].items()}
    run["selected_model"] = next(name for name in MODELS if scores[name] <= min(scores.values()) + TIE_TOLERANCE)
    best_tabular = min(TABULAR, key=scores.__getitem__)
    contrasts = (
        ("history_logistic", "task_logistic"),
        ("history_boosting", "history_logistic"),
        ("time_decay_gru", "time_input_gru"),
        ("time_decay_gru", best_tabular),
    )
    partitions = [(cohort, "validation")]
    if args.mode == "full":
        partitions.extend((name, "test") for name in ("main", "power_law", "long_gap"))
    seen = train_learners.copy()
    for name, split in partitions:
        key = f"{name}_{split}"
        print(f"Evaluating {key}", flush=True)
        if split == "validation":
            partition, batches = validation, validation_batches
        else:
            partition = load_partition(args.dataset, name, split)
            batches = make_batches(partition, args.batch_size, device)
        learners = {query.learner for query in partition.queries}
        if seen & learners:
            raise ValueError(f"Overlapping learners in {key}")
        seen.update(learners)
        group_masks = groups(partition.queries)
        predictions = {
            model_name: predict_tabular(model_name, model, partition) if model_name in TABULAR
            else predict_recurrent(model, batches, len(partition.labels))
            for model_name, model in models.items()
        }
        run["evaluation"][key] = {
            "metrics": {
                model_name: probability_report(partition, p, group_masks)
                for model_name, p in predictions.items()
            },
            "paired_intervals": paired_intervals(partition, predictions, contrasts, group_masks, seed=args.seed),
        }
        export_predictions(args.output / f"{key}_predictions.csv.gz", partition, predictions)
        del batches
    run["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if device.type == "mps":
        torch.mps.synchronize()
        run["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    write_report(args.output, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New run directory")
    parser.add_argument("--dataset", type=Path, default=DATA_ROOT)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 1 if args.mode == "smoke" else 8
    if args.epochs < 1 or args.batch_size < 1 or not 0 <= args.seed < 2**32:
        parser.error("epochs and batch size must be positive and the seed must be a uint32")
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    device = training_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    cohorts = ("pilot",) if args.mode == "smoke" else ("main", "power_law", "long_gap")
    run = {
        "status": "running", "mode": args.mode, "recurrent_device": str(device),
        "mps_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "selection_tolerance": TIE_TOLERANCE, "dataset": str(args.dataset),
        "dataset_manifests": {
            name: hashlib.sha256((args.dataset / name / "manifest.json").read_bytes()).hexdigest()
            for name in cohorts
        },
        "implementation_sha256": {
            str(path.relative_to(MODEL_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), *sorted((MODEL_DIR / "memory").glob("*.py")))
        },
        "environment": {
            "python": sys.version.split()[0], "architecture": platform.machine(),
            "macos": platform.mac_ver()[0], "torch": torch.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        },
        "models": {}, "evaluation": {},
    }
    write_json(args.output / "run.json", run)
    try:
        experiment(args, device, run)
    except BaseException as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        write_json(args.output / "run.json", run)
        raise
    run["status"] = "completed"
    write_json(args.output / "run.json", run)
    print(f"Selected {run['selected_model']}. Report: {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
