"""Fit tabular comparators with selection on the explicit learner split."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .evaluation import validation_losses


TABULAR = ("task_logistic", "history_logistic", "history_boosting")


def fit_tabular(name, train, validation, masks, *, seed, balanced=False):
    history = name != "task_logistic"
    train_x, validation_x = train.tabular(history=history), validation.tabular(history=history)
    if name == "history_boosting":
        candidates = [
            ({"max_leaf_nodes": leaves, "max_iter": iterations}, HistGradientBoostingClassifier(
                max_leaf_nodes=leaves, max_iter=iterations, learning_rate=0.08,
                l2_regularization=1.0, early_stopping=False, random_state=seed,
            ))
            for leaves, iterations in ((15, 120), (31, 160))
        ]
    else:
        candidates = [
            ({"C": strength}, make_pipeline(StandardScaler(), LogisticRegression(
                C=strength, max_iter=1000, random_state=seed,
            )))
            for strength in (0.1, 1.0)
        ]
    best_model, best_loss = None, float("inf")
    trials = []
    with threadpool_limits(limits=4):
        for parameters, candidate in candidates:
            candidate.fit(train_x, train.labels)
            losses = validation_losses(
                validation.labels, candidate.predict_proba(validation_x)[:, 1], masks, balanced=balanced,
            )
            score = losses["validation_selection_log_loss"]
            trials.append({"parameters": parameters, **losses})
            if score < best_loss:
                best_model, best_loss, best_metrics = candidate, score, losses
    return best_model, {"device": "cpu", **best_metrics, "trials": trials}


def predict_tabular(name, model, partition):
    with threadpool_limits(limits=4):
        return model.predict_proba(partition.tabular(history=name != "task_logistic"))[:, 1]
