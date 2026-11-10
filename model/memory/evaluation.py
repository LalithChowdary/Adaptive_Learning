"""Probability metrics, subgroup reporting, and paired learner bootstrap."""
import numpy as np


HORIZONS = (1, 7, 30)


def loss_per_query(labels, probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != labels.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected one finite probability in [0, 1] per query")
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.where(labels == 1, np.log(p), np.log1p(-p))


def groups(queries):
    purpose = np.array([query.purpose for query in queries])
    delay = np.array([query.delay_days or 0 for query in queries])
    task = np.array([query.task_type for query in queries])
    return {
        "all": np.ones(len(queries), dtype=bool),
        "practice": delay == 0,
        "ordinary": (delay == 0) & (purpose == "ordinary"),
        "followup": (delay == 0) & (purpose == "followup"),
        "assessment": delay != 0,
        **{f"day_{days}": delay == days for days in HORIZONS},
        **{f"format/{name}": task == name for name in sorted(set(task))},
    }


def delayed_log_loss(labels, probabilities, masks):
    losses = loss_per_query(labels, probabilities)
    horizons = [masks[f"day_{days}"] for days in HORIZONS]
    if any(not mask.any() for mask in horizons):
        raise ValueError("Validation requires labeled examples at all assessment horizons")
    return float(np.mean([losses[mask].mean() for mask in horizons]))


def validation_losses(labels, probabilities, masks, *, balanced=False):
    practice = loss_per_query(labels, probabilities)[masks["practice"]]
    if not len(practice):
        raise ValueError("Validation requires labeled practice queries")
    delayed = delayed_log_loss(labels, probabilities, masks)
    practice_mean = float(practice.mean())
    return {
        "validation_practice_log_loss": practice_mean,
        "validation_delayed_log_loss": delayed,
        "validation_selection_log_loss": (practice_mean + delayed) / 2 if balanced else delayed,
    }


def probability_report(partition, probabilities, masks):
    labels = partition.labels
    losses = loss_per_query(labels, probabilities)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    learners = np.asarray([query.learner for query in partition.queries])
    report = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        scores, outcome = probabilities[mask], labels[mask]
        bins = np.minimum((scores * 10).astype(int), 9)
        calibration = []
        ece = 0.0
        for index in range(10):
            in_bin = bins == index
            count = int(in_bin.sum())
            if not count:
                continue
            confidence = float(scores[in_bin].mean())
            observed = float(outcome[in_bin].mean())
            ece += count / len(scores) * abs(confidence - observed)
            calibration.append({
                "lower": index / 10, "upper": (index + 1) / 10,
                "queries": count, "mean_prediction": confidence, "observed_rate": observed,
            })
        report[name] = {
            "queries": int(mask.sum()), "learners": int(len(np.unique(learners[mask]))),
            "log_loss": float(losses[mask].mean()),
            "brier": float(np.mean((scores - outcome) ** 2)),
            "accuracy_05": float(np.mean((scores >= 0.5) == outcome)),
            "ece_10": ece, "calibration": calibration,
        }
    return report


def paired_intervals(partition, predictions, contrasts, masks, *, seed, draws=1000):
    """Bootstrap whole learners with both models evaluated on the same queries."""
    learners = np.asarray([query.learner for query in partition.queries])
    losses = {
        model: np.mean([loss_per_query(partition.labels, seed) for seed in np.atleast_2d(p)], axis=0)
        for model, p in predictions.items()
    }
    result = {}
    rng = np.random.default_rng(seed)
    for group in ("ordinary", "followup", "day_1", "day_7", "day_30"):
        mask = masks[group]
        if not mask.any():
            continue
        _, indices = np.unique(learners[mask], return_inverse=True)
        counts = np.bincount(indices)
        samples = rng.integers(len(counts), size=(draws, len(counts)))
        denominators = counts[samples].sum(axis=1)
        result[group] = {}
        for candidate, reference in contrasts:
            diff = (losses[candidate] - losses[reference])[mask]
            sums = np.bincount(indices, weights=diff)
            simulated = sums[samples].sum(axis=1) / denominators
            lower, upper = np.quantile(simulated, [0.025, 0.975])
            result[group][f"{candidate} minus {reference}"] = {
                "delta_log_loss": float(diff.mean()),
                "lower_95": float(lower), "upper_95": float(upper),
            }
    return result
