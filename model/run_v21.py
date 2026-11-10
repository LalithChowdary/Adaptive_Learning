"""Fit and compare v2.1 memory models with scheduled, multi-seed GRU training."""
import argparse
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

from memory.data import DATA_ROOT_V21, digest, load_partition
from memory.evaluation import groups, paired_intervals, probability_report
from memory.tabular import TABULAR, fit_tabular, predict_tabular
from memory.training import fit_recurrent, make_batches, predict_recurrent, training_device
from run import MODEL_DIR, MODELS, RECURRENT, TIE_TOLERANCE, export_predictions, write_json


VERSION = "2.1.0"
SEEDS = (20261110, 20261111, 20261112)
HORIZON_GROUPS = ("day_1", "day_7", "day_30")
METRICS = ("log_loss", "brier", "ece_10", "accuracy_05")


def summarize(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def fit_models(args, run, train, validation, device):
    masks = groups(validation.queries)
    models = {}
    for name in TABULAR:
        print(f"Fitting {name} on CPU", flush=True)
        model, fit = fit_tabular(name, train, validation, masks, seed=args.seeds[0], balanced=True)
        folder = args.output / name
        folder.mkdir()
        joblib.dump(model, folder / "model.joblib")
        write_json(folder / "fit.json", fit)
        models[name], run["models"][name] = [model], [fit]

    training_batches = make_batches(train, args.batch_size, device)
    validation_batches = make_batches(validation, args.batch_size, device)
    for name in RECURRENT:
        mode = name.removesuffix("_gru")
        models[name], run["models"][name] = [], []
        for seed in args.seeds:
            print(f"Fitting {name} on {device}, seed {seed}", flush=True)
            model, fit = fit_recurrent(
                mode, training_batches, validation, validation_batches, masks, device,
                epochs=args.epochs, seed=seed, scheduled=True,
            )
            folder = args.output / name / f"seed_{seed}"
            folder.mkdir(parents=True)
            torch.save({
                "dataset_version": VERSION, "mode": mode, "hidden_size": model.hidden_size,
                "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }, folder / "model.pt")
            write_json(folder / "fit.json", fit)
            models[name].append(model)
            run["models"][name].append(fit)
            write_json(args.output / "run.json", run)
    return models, validation_batches


def select_model(run):
    key = "validation_selection_log_loss"
    scores = {name: summarize([fit[key] for fit in fits]) for name, fits in run["models"].items()}
    best = min(score["mean"] for score in scores.values())
    selected = next(name for name in MODELS if scores[name]["mean"] <= best + TIE_TOLERANCE)
    if selected in TABULAR:
        checkpoint = f"{selected}/model.joblib"
    else:
        ranked = sorted(run["models"][selected], key=lambda fit: fit[key])
        checkpoint = f"{selected}/seed_{ranked[len(ranked) // 2]['seed']}/model.pt"
    run["validation_selection"] = scores
    run["selected_model"] = selected
    run["selected_checkpoint"] = checkpoint
    return selected


def evaluation_summary(reports):
    combined = {
        group: {
            "queries": reports[0][group]["queries"],
            **{metric: summarize([report[group][metric] for report in reports]) for metric in METRICS},
        }
        for group in reports[0]
    }
    combined["delayed_equal_horizon_log_loss"] = summarize([
        np.mean([report[horizon]["log_loss"] for horizon in HORIZON_GROUPS]) for report in reports
    ])
    return combined


def evaluate(args, run, models, partition, batches, cohort, split):
    from memory.oracle import oracle_predictions

    key = f"{cohort}_{split}"
    print(f"Evaluating {key}", flush=True)
    masks = groups(partition.queries)
    predictions = {}
    for name, fits in models.items():
        predictions[name] = [
            predict_tabular(name, model, partition) if name in TABULAR
            else predict_recurrent(model, batches, len(partition.labels))
            for model in fits
        ]
    # These references enter only reporting, after all fits and selection are fixed.
    predictions["constant"] = [np.full(len(partition.labels), run["training"]["base_rate"])]
    predictions["oracle"] = [oracle_predictions(args.dataset, cohort, split, partition)]

    metrics, summary, exported = {}, {}, {}
    for name, scores in predictions.items():
        names = [f"{name}_seed_{seed}" for seed in args.seeds] if name in RECURRENT else [name]
        metrics[name] = {
            identifier: probability_report(partition, probabilities, masks)
            for identifier, probabilities in zip(names, scores, strict=True)
        }
        summary[name] = evaluation_summary(list(metrics[name].values()))
        exported.update(zip(names, scores, strict=True))

    best_tabular = min(TABULAR, key=lambda name: run["validation_selection"][name]["mean"])
    contrasts = [
        ("history_logistic", "task_logistic"), ("history_boosting", "history_logistic"),
        ("time_decay_gru", "time_input_gru"), ("time_input_gru", best_tabular),
        ("time_decay_gru", best_tabular),
    ]
    paired = paired_intervals(partition, predictions, contrasts, masks, seed=args.seeds[0])
    signal_fraction = {}
    for group in ("ordinary", "followup", *HORIZON_GROUPS, "delayed_equal_horizon_log_loss"):
        losses = {
            name: value[group]["mean"] if group == "delayed_equal_horizon_log_loss"
            else value[group]["log_loss"]["mean"]
            for name, value in summary.items()
        }
        denominator = losses["constant"] - losses["oracle"]
        signal_fraction[group] = {
            name: (losses["constant"] - losses[name]) / denominator if denominator > 0 else None
            for name in MODELS
        }
    run["evaluation"][key] = {
        "by_seed": metrics, "summary": summary, "paired_intervals": paired,
        "constant_to_oracle_log_loss_fraction": signal_fraction,
    }
    export_predictions(args.output / f"{key}_predictions.csv.gz", partition, exported)


def show(score):
    mean, sd = score["mean"], score["sd"]
    return f"{mean:.5f}" if sd is None else f"{mean:.5f} ± {sd:.5f}"


def report(args, run):
    lines = [
        "# v2.1 learner-memory comparison", "",
        f"Selected **`{run['selected_model']}`** on main validation for the full run. "
        f"Representative checkpoint `{run['selected_checkpoint']}`.",
        "" if args.mode == "full" else "This is the pilot smoke result.",
        f"Recurrent device `{run['recurrent_device']}`. "
        f"Training used {run['training']['practice']:,} practice and "
        f"{run['training']['assessment']:,} delayed labels.", "",
        "Checkpoint selection weights practice log loss and equal-horizon delayed log loss equally. "
        "The score below aggregates recurrent fits across seeds. ± denotes sample standard deviation.", "",
        "| Model | Validation selection log loss |", "|---|---:|",
        *[f"| {name} | {show(score)} |" for name, score in run["validation_selection"].items()],
    ]
    for partition, evaluation in run["evaluation"].items():
        lines.extend([
            "", f"## {partition}", "",
            "Lower log loss is better. The constant and simulator oracle are evaluation references.",
            "", "| Model | Ordinary | Follow-up | 1 day | 7 days | 30 days | Delayed mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, result in evaluation["summary"].items():
            cells = [show(result[key]["log_loss"]) for key in ("ordinary", "followup", *HORIZON_GROUPS)]
            cells.append(show(result["delayed_equal_horizon_log_loss"]))
            lines.append(f"| {name} | {' | '.join(cells)} |")
    lines.extend([
        "", "The JSON contains per-seed probability metrics, calibration bins, paired learner intervals, "
        "and the fraction of the log-loss improvement from the constant to the simulator oracle.",
        "Each paired comparison averages each query's loss over seeds before resampling learners.",
        "The simulator oracle uses private pre-attempt strength and difficulty. It is a reference for this synthetic benchmark.", "",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def experiment(args, device, run):
    cohort, limit = ("pilot", 12) if args.mode == "smoke" else ("main", None)
    train = load_partition(args.dataset, cohort, "train", learner_limit=limit, version=VERSION)
    validation = load_partition(args.dataset, cohort, "validation", learner_limit=limit, version=VERSION)
    seen = {q.learner for q in train.queries}
    validation_learners = {q.learner for q in validation.queries}
    if seen & validation_learners:
        raise ValueError("Training and validation learners overlap")
    training_groups = groups(train.queries)
    run["training"] = {
        "learners": len(seen), "queries": len(train.labels),
        "base_rate": float(train.labels.mean(dtype=np.float64)),
        **{key: int(training_groups[key].sum()) for key in ("practice", "assessment", *HORIZON_GROUPS)},
    }
    if args.mode == "full":
        release = json.loads((args.dataset.parent / "reports" / "RELEASE_SUMMARY.json").read_text())
        if (len(seen), len(validation_learners), len(train.labels)) != (1600, 200, release["main_training_targets"]):
            raise ValueError("Unexpected v2.1 fitting or validation coverage")
        if any(not training_groups[key].any() for key in HORIZON_GROUPS):
            raise ValueError("The training cohort is missing a delayed label horizon")
    print(f"{cohort} fitting coverage {run['training']}", flush=True)
    models, batches = fit_models(args, run, train, validation, device)
    select_model(run)
    write_json(args.output / "run.json", run)

    seen.update(validation_learners)
    evaluate(args, run, models, validation, batches, cohort, "validation")
    if args.mode == "full":
        for name in ("main", "power_law", "long_gap"):
            partition = load_partition(args.dataset, name, "test", version=VERSION)
            learners = {q.learner for q in partition.queries}
            if seen & learners or len(learners) != 200:
                raise ValueError(f"Unexpected evaluation learner coverage in {name}")
            seen.update(learners)
            evaluate(
                args, run, models, partition, make_batches(partition, args.batch_size, device), name, "test",
            )
            write_json(args.output / "run.json", run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=DATA_ROOT_V21)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 1 if args.mode == "smoke" else 40
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    args.seeds = SEEDS[:1] if args.mode == "smoke" else SEEDS
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    device = training_device(args.device)
    cohorts = ("pilot",) if args.mode == "smoke" else ("main", "power_law", "long_gap")
    run = {
        "status": "running", "mode": args.mode, "dataset_version": VERSION,
        "recurrent_device": str(device), "mps_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "seeds": args.seeds, "epochs": args.epochs, "batch_size": args.batch_size,
        "selection_tolerance": TIE_TOLERANCE, "dataset": str(args.dataset),
        "dataset_manifests": {cohort: digest(args.dataset / cohort / "manifest.json") for cohort in cohorts},
        "implementation_sha256": {
            str(path.relative_to(MODEL_DIR)): digest(path)
            for path in (Path(__file__).resolve(), MODEL_DIR / "run.py", *sorted((MODEL_DIR / "memory").glob("*.py")))
        },
        "environment": {
            "python": sys.version.split()[0], "architecture": platform.machine(),
            "macos": platform.mac_ver()[0], "torch": torch.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        },
        "models": {}, "evaluation": {},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "run.json", run)
    started = time.monotonic()
    try:
        experiment(args, device, run)
        report(args, run)
    except BaseException as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        write_json(args.output / "run.json", run)
        raise
    run["status"] = "completed"
    run["elapsed_seconds"] = round(time.monotonic() - started, 3)
    write_json(args.output / "run.json", run)
    print(f"Selected {run['selected_model']}. Report at {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
