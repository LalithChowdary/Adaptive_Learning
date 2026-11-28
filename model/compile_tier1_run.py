"""Compile read-only retrospective Tier 1 metrics and assemble unified tier1_run.json."""
import argparse
import copy
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from memory.evaluation import groups, paired_intervals, probability_report
from memory.tabular import TABULAR
from run import write_json
from run_v22 import (
    HORIZON_GROUPS,
    MODELS_V22,
    RECURRENT_V22,
    evaluation_summary,
    summarize,
)


DEFAULT_RETRO_DIR = Path("3/model/runs/v2.2-mps-transformer-comparison")
DEFAULT_PREREG_RUN = Path("3/eval_runs/v2.2-preregistered/tier1/run.json")
DEFAULT_OUTPUT = Path("3/eval_runs/v2.2-preregistered/tier1_run.json")


def _load_csv_partition(csv_path):
    with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    queries = [
        SimpleNamespace(
            learner=row["learner"],
            purpose=row["purpose"],
            delay_days=int(row["delay_days"]) if row["delay_days"] != "" else None,
            task_type=row["task_type"],
        )
        for row in rows
    ]
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    partition = SimpleNamespace(queries=queries, labels=labels)
    return partition, rows


def compile_retrospective(retro_dir, *, bootstrap_draws=2000):
    retro_dir = Path(retro_dir)
    orig = json.loads((retro_dir / "run.json").read_text(encoding="utf-8"))
    seeds = list(orig["seeds"])
    base_seed = int(seeds[0])

    models = copy.deepcopy(orig["models"])
    for name in TABULAR:
        if models[name] and "seed" not in models[name][0]:
            models[name][0]["seed"] = base_seed

    validation_selection = {
        name: summarize([fit["validation_selection_log_loss"] for fit in fits])
        for name, fits in models.items()
    }
    best_tabular = min(TABULAR, key=lambda name: validation_selection[name]["mean"])
    contrasts = [
        ("history_logistic", "task_logistic"),
        ("history_boosting", "history_logistic"),
        ("monotonic_gru", "kt_decay_gru"),
        ("monotonic_gru", "time_decay_gru"),
        ("monotonic_gru", "time_aware_transformer"),
        ("monotonic_transformer", "time_aware_transformer"),
        ("monotonic_transformer", "monotonic_gru"),
        ("monotonic_gru", best_tabular),
    ]

    selected_model = orig["selected_model"]
    selected_checkpoint = orig["selected_checkpoint"]
    selected_seed = Path(selected_checkpoint).parent.name.removeprefix("seed_")
    selected_identifier = f"{selected_model}_seed_{selected_seed}"

    evaluation = {}
    for split, orig_split in orig["evaluation"].items():
        csv_path = retro_dir / f"{split}_predictions.csv.gz"
        partition, rows = _load_csv_partition(csv_path)
        masks = groups(partition.queries)

        predictions = {}
        by_seed = {}
        summary = {}
        for name in (*MODELS_V22, "constant", "oracle"):
            identifiers = (
                [f"{name}_seed_{seed}" for seed in seeds]
                if name in RECURRENT_V22
                else [name]
            )
            score_arrays = [
                np.array([float(row[identifier]) for row in rows], dtype=np.float64)
                for identifier in identifiers
            ]
            predictions[name] = score_arrays
            by_seed[name] = {
                identifier: probability_report(partition, scores, masks)
                for identifier, scores in zip(identifiers, score_arrays, strict=True)
            }
            summary[name] = evaluation_summary(
                list(by_seed[name].values()),
                partition=partition,
                scores=score_arrays,
                masks=masks,
            )

        # Verify parity against historical run.json metrics
        for model_name, orig_summary in orig_split["summary"].items():
            for group_name, orig_group in orig_summary.items():
                if group_name == "delayed_equal_horizon_log_loss":
                    diff = abs(
                        summary[model_name][group_name]["mean"] - orig_group["mean"]
                    )
                    if diff >= 1e-7:
                        raise ValueError(
                            f"Parity mismatch in {split}.{model_name}.{group_name}: {diff}"
                        )
                    continue
                for metric in ("log_loss", "brier", "ece_10", "accuracy_05"):
                    diff = abs(
                        summary[model_name][group_name][metric]["mean"]
                        - orig_group[metric]["mean"]
                    )
                    if diff >= 1e-7:
                        raise ValueError(
                            f"Parity mismatch in {split}.{model_name}.{group_name}.{metric}: {diff}"
                        )

        paired_b2000 = paired_intervals(
            partition, predictions, contrasts, masks, seed=base_seed, draws=bootstrap_draws,
        )

        # Verify B=1000 reproduction from CSVs against historical run.json before archiving
        paired_b1000_check = paired_intervals(
            partition, predictions, contrasts, masks, seed=base_seed, draws=1000,
        )
        for group_name, orig_contrasts in orig_split["paired_intervals"].items():
            for contrast_name, orig_vals in orig_contrasts.items():
                for stat in ("delta_log_loss", "lower_95", "upper_95"):
                    diff = abs(
                        paired_b1000_check[group_name][contrast_name][stat]
                        - orig_vals[stat]
                    )
                    if diff >= 1e-7:
                        raise ValueError(
                            f"B=1000 CI parity mismatch in {split}.{group_name}.{contrast_name}.{stat}: {diff}"
                        )

        paired_b1000 = {}
        for group_name, orig_contrasts in orig_split["paired_intervals"].items():
            paired_b1000[group_name] = {
                contrast_name: {
                    **orig_vals,
                    "bootstrap_draws": 1000,
                    "bootstrap_seed": base_seed,
                }
                for contrast_name, orig_vals in orig_contrasts.items()
            }

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

        evaluation[split] = {
            "by_seed": by_seed,
            "summary": summary,
            "selected_checkpoint_calibration": {
                "checkpoint": selected_checkpoint,
                "ordinary": by_seed[selected_model][selected_identifier]["ordinary"]["calibration"],
                "assessment": by_seed[selected_model][selected_identifier]["assessment"]["calibration"],
            },
            "paired_intervals": paired_b2000,
            "paired_intervals_b1000": paired_b1000,
            "constant_to_oracle_log_loss_fraction": signal_fraction,
        }

    compiled = {
        "status": orig["status"],
        "mode": orig["mode"],
        "run_classification": "retrospective",
        "source_run_dir": str(retro_dir),
        "dataset_version": orig["dataset_version"],
        "reused_from": orig["reused_from"],
        "recurrent_device": orig["recurrent_device"],
        "mps_cpu_fallback_enabled": orig["mps_cpu_fallback_enabled"],
        "seeds": seeds,
        "epochs": orig["epochs"],
        "batch_size": orig["batch_size"],
        "bootstrap_draws": int(bootstrap_draws),
        "selection_tolerance": orig["selection_tolerance"],
        "dataset": orig["dataset"],
        "dataset_manifests": copy.deepcopy(orig["dataset_manifests"]),
        "implementation_sha256": copy.deepcopy(orig["implementation_sha256"]),
        "environment": copy.deepcopy(orig["environment"]),
        "training": copy.deepcopy(orig["training"]),
        "behavioral_summary": copy.deepcopy(orig["behavioral_summary"]),
        "validation_selection": validation_selection,
        "selected_model": selected_model,
        "selected_checkpoint": selected_checkpoint,
        "elapsed_seconds": orig["elapsed_seconds"],
        "models": models,
        "evaluation": evaluation,
    }
    return compiled


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrospective-dir", type=Path, default=DEFAULT_RETRO_DIR)
    parser.add_argument("--preregistered-run", type=Path, default=DEFAULT_PREREG_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--retrospective-only", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    retro = compile_retrospective(args.retrospective_dir, bootstrap_draws=args.bootstrap_draws)
    if args.retrospective_only:
        print(
            f"Compiled retrospective Tier 1 cleanly from {args.retrospective_dir} "
            f"(splits: {list(retro['evaluation'].keys())}, selected: {retro['selected_model']})."
        )
        return
    prereg = json.loads(args.preregistered_run.read_text(encoding="utf-8"))
    if prereg.get("status") != "completed":
        raise ValueError(f"Pre-registered run not completed: {prereg.get('status')}")
    if prereg.get("run_classification") != "pre_registered":
        raise ValueError(f"Unexpected run_classification: {prereg.get('run_classification')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    unified = {
        "retrospective": retro,
        "pre_registered": prereg,
    }
    write_json(args.output, unified)
    print(f"Wrote unified Tier 1 log to {args.output}")


if __name__ == "__main__":
    main()
