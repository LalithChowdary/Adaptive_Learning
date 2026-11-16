"""Fit and compare v2.2 memory models, combining predictive accuracy and behavioral sanity checks."""
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

from memory.behavioral import evaluate_behavioral_suite
from memory.data import DATA_ROOT_V22, digest, load_partition
from memory.evaluation import groups, paired_intervals, probability_report
from memory.tabular import TABULAR, fit_tabular, predict_tabular
from memory.training import fit_recurrent, make_batches, predict_recurrent, training_device
from run import MODEL_DIR, TIE_TOLERANCE, export_predictions, write_json


VERSION = "2.2.0"
SEEDS = (20261120, 20261121, 20261122)
RECURRENT_V22 = ("time_input_gru", "time_decay_gru", "kt_decay_gru", "monotonic_gru")
MODELS_V22 = (*TABULAR, *RECURRENT_V22)
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
    for name in RECURRENT_V22:
        mode = name.removesuffix("_gru")
        models[name], run["models"][name] = [], []
        for seed in args.seeds:
            print(f"Fitting {name} on {device}, seed {seed}", flush=True)
            model, fit = fit_recurrent(
                mode, training_batches, validation, validation_batches, masks, device,
                epochs=args.epochs, seed=seed, scheduled=True,
            )
            behavioral = evaluate_behavioral_suite(model)
            fit["behavioral"] = behavioral
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
    behavioral_summary = {}
    for name in RECURRENT_V22:
        fits = run["models"][name]
        passed_counts = [fit["behavioral"]["checks_passed"] for fit in fits]
        all_passed = all(fit["behavioral"]["all_passed"] for fit in fits)
        behavioral_summary[name] = {
            "all_seeds_passed": all_passed,
            "checks_passed_mean": float(np.mean(passed_counts)),
            "checks_total": fits[0]["behavioral"]["checks_total"],
            "representative_replay": fits[len(fits) // 2]["behavioral"],
        }
    run["behavioral_summary"] = behavioral_summary

    # Require passing behavioral sanity checks and pick the lowest validation selection log loss
    eligible = [name for name in RECURRENT_V22 if behavioral_summary[name]["all_seeds_passed"]]
    candidates = eligible if eligible else list(MODELS_V22)
    selected = min(candidates, key=lambda name: scores[name]["mean"])
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
    predictions["constant"] = [np.full(len(partition.labels), run["training"]["base_rate"])]
    predictions["oracle"] = [oracle_predictions(args.dataset, cohort, split, partition)]

    metrics, summary, exported = {}, {}, {}
    for name, scores in predictions.items():
        names = [f"{name}_seed_{seed}" for seed in args.seeds] if name in RECURRENT_V22 else [name]
        metrics[name] = {
            identifier: probability_report(partition, probabilities, masks)
            for identifier, probabilities in zip(names, scores, strict=True)
        }
        summary[name] = evaluation_summary(list(metrics[name].values()))
        exported.update(zip(names, scores, strict=True))

    best_tabular = min(TABULAR, key=lambda name: run["validation_selection"][name]["mean"])
    contrasts = [
        ("history_logistic", "task_logistic"),
        ("history_boosting", "history_logistic"),
        ("monotonic_gru", "kt_decay_gru"),
        ("monotonic_gru", "time_decay_gru"),
        ("monotonic_gru", best_tabular),
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
            for name in MODELS_V22
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
        "# v2.2 learner-memory comparison (accuracy + behavioral sanity suite)", "",
        f"Selected **`{run['selected_model']}`** (`{run['selected_checkpoint']}`).",
        f"Recurrent device `{run['recurrent_device']}`. "
        f"Training used {run['training']['practice']:,} practice and "
        f"{run['training']['assessment']:,} delayed labels.", "",
        "## Behavioral sanity check suite (6 controlled checks)", "",
        "| Sequence model | Checks passed | Fast 0% wrong (6.4s) delta | After 3 hints + 2nd 0% wrong | Non-increasing forgetting |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, b in run["behavioral_summary"].items():
        rep = b["representative_replay"]
        u = rep["user_conditionals_replay"]
        delta_first = (u["after_first_wrong_6.4s"] - u["before_attempt"]) * 100
        after_second = u["after_second_wrong_2.27s"] * 100
        before = u["before_attempt"] * 100
        forgetting_ok = "Pass" if rep["checks"]["non_increasing_forgetting_after_pass_and_fail"] else "Fail"
        lines.append(
            f"| {name} | {b['checks_passed_mean']:.1f} / {b['checks_total']} | "
            f"{before:.1f}% → {u['after_first_wrong_6.4s']*100:.1f}% ({delta_first:+.1f}%) | "
            f"{after_second:.1f}% | {forgetting_ok} |"
        )
    lines.extend([
        "", "## Validation selection log loss", "",
        "| Model | Validation selection log loss | Behavioral gate |", "|---|---:|---:|",
    ])
    for name, score in run["validation_selection"].items():
        gate = (
            "Pass (6/6)" if name in run["behavioral_summary"] and run["behavioral_summary"][name]["all_seeds_passed"]
            else ("Fail" if name in run["behavioral_summary"] else "Tabular baseline")
        )
        lines.append(f"| {name} | {show(score)} | {gate} |")

    for partition, evaluation in run["evaluation"].items():
        lines.extend([
            "", f"## {partition}", "",
            "| Model | Ordinary | Follow-up | 1 day | 7 days | 30 days | Delayed mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, result in evaluation["summary"].items():
            cells = [show(result[key]["log_loss"]) for key in ("ordinary", "followup", *HORIZON_GROUPS)]
            cells.append(show(result["delayed_equal_horizon_log_loss"]))
            lines.append(f"| {name} | {' | '.join(cells)} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            raise ValueError("Unexpected v2.2 fitting or validation coverage")
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
    parser.add_argument("--dataset", type=Path, default=DATA_ROOT_V22)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 1 if args.mode == "smoke" else 35
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
