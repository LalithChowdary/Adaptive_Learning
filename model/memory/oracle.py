"""Evaluation-only reference with access to the v2.1 simulator's hidden inputs."""
import json

import numpy as np

from .data import digest, read_rows


def full_success_probability(strength, demand, task):
    """Integrate full success over the simulator's shared Gaussian attempt noise."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(32)
    return _probability(strength, demand, task, nodes, weights / weights.sum())


def _probability(strength, demand, task, nodes, normalized_weights):
    fmt = task["task_type"]
    z = 5 * (strength - demand) + 0.65 + 0.65 * nodes
    criteria = np.arange(len(task["criteria"]))[:, None]
    q = 0.97 / (1 + np.exp(-np.clip(z - 0.12 * criteria, -40, 40)))
    if fmt == "single_choice":
        success = q[0] + (1 - q[0]) / len(task["options"])
    elif fmt == "multiple_select":
        success = (0.5 + 0.5 * q).prod(axis=0)
    else:
        guess = {"output_prediction": 0.02, "fill_in_the_blank": 0.04, "code_completion": 0.01}[fmt]
        success = (guess + (1 - guess) * q).prod(axis=0)
    return float(success @ normalized_weights)


def oracle_predictions(root, cohort, split, partition):
    """Join exact task and response identifiers after model fitting and selection."""
    if split not in {"validation", "test"}:
        raise ValueError("Private oracle traces are for evaluation only")
    folder = root / cohort
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["version"] != "2.1.0":
        raise ValueError("The oracle requires v2.1 pre-attempt strength")
    catalog_path = root / "task_catalog.json"
    if digest(catalog_path) != manifest["catalog_files"]["task_catalog.json"]:
        raise ValueError("Oracle task catalog checksum mismatch")
    catalog = {task["task_id"]: task for task in json.loads(catalog_path.read_text())}

    wanted = {query.identifier: query for query in partition.queries}
    task_by_response = {}
    for kind in ("observations", "assessments"):
        relative = f"{kind}/{split}.jsonl.gz"
        if digest(folder / relative) != manifest["files"][relative]:
            raise ValueError(f"Oracle source checksum mismatch for {relative}")
        pending = {}
        for event in read_rows(folder / relative):
            interaction = event["interaction_id"]
            if event["event_type"] == "task" and event["event_id"] in wanted:
                query = wanted[event["event_id"]]
                if (event["learner_id"], event["skill_id"], event["purpose"]) != (
                    query.learner, query.skill, query.purpose,
                ):
                    raise ValueError(f"Oracle query routing mismatch for {query.identifier}")
                pending[interaction] = event
            elif event["event_type"] == "response" and event["attempt_number"] == 1 and interaction in pending:
                task = pending.pop(interaction)
                if (event["assistance"] != "independent" or event["hint_count"] != 0
                        or event["phase"] != task["phase"] or event["task_id"] != task["task_id"]):
                    raise ValueError("Oracle matched a response that was not an eligible first attempt")
                task_by_response[event["event_id"]] = (task["event_id"], task["task_id"])
        if pending:
            raise ValueError("Oracle could not link all queries to first attempts")

    relative = f"administration/{split}.jsonl.gz"
    if digest(folder / relative) != manifest["files"][relative]:
        raise ValueError("Oracle private trace checksum mismatch")
    nodes, weights = np.polynomial.hermite_e.hermegauss(32)
    normalized = weights / weights.sum()
    predictions = {}
    for row in read_rows(folder / relative):
        histories = [row["trace"], *(branch["trace"] for branch in row["assessments"])]
        for history in histories:
            for trace in history:
                matched = task_by_response.pop(trace["response_id"], None)
                if matched is None:
                    continue
                query_id, task_id = matched
                predictions[query_id] = _probability(
                    trace["strength_at_attempt"], trace["hidden_demand"], catalog[task_id], nodes, normalized,
                )
    if task_by_response or predictions.keys() != wanted.keys():
        raise ValueError("Oracle reference coverage differs from model evaluation queries")
    return np.asarray([predictions[query.identifier] for query in partition.queries])
