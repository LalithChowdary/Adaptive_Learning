"""Build tensors once per learner-skill history and align all model queries."""
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

from .features import (
    DAY, SUMMARY_SIZE, TASK_SIZE, TOKEN_SIZE, EvidenceStream, HistorySummary, matrix,
    task_features,
)

V2_ROOT = Path(__file__).resolve().parents[2] / "data_Set" / "v2"
DATA_ROOT = V2_ROOT / "dataset"
DATA_ROOT_V21 = V2_ROOT.with_name("v2.1") / "dataset"

# The audited v2 loader is a standalone module with sibling imports.
sys.path.insert(0, str(V2_ROOT))
try:
    from loader import checked_records
    from storage import digest, read_rows
finally:
    sys.path.pop(0)


@dataclass(frozen=True)
class Query:
    identifier: str
    learner: str
    skill: str
    purpose: str
    delay_days: int | None
    task_type: str


@dataclass(frozen=True)
class Sequence:
    tokens: np.ndarray
    gaps: np.ndarray
    positions: np.ndarray
    tasks: np.ndarray
    query_gaps: np.ndarray
    summaries: np.ndarray
    labels: np.ndarray
    queries: tuple[Query, ...]


def encode_record(record, *, training=False, include_training_assessments=False):
    stream = EvidenceStream()
    history = HistorySummary()
    events = record["events"]
    tokens, gaps = [], []
    positions, tasks, query_gaps, summaries, labels, queries = [], [], [], [], [], []
    anchor = None
    cursor = 0

    def consume_until(end):
        nonlocal cursor, anchor
        while cursor < end:
            evidence = stream.push(events[cursor])
            cursor += 1
            if evidence is None:
                continue
            gaps.append(0.0 if anchor is None else (evidence.time - anchor) / DAY)
            anchor = evidence.time
            tokens.append(evidence.vector())
            history.update(evidence)

    for query in record["queries"]:
        if training and query["phase"] != "practice" and not include_training_assessments:
            raise ValueError("Assessment query found in a training record")
        if query["phase"] == "assessment":
            if (query["history_end"] != len(events) or query["purpose"] != "delayed"
                    or query.get("delay_days") not in (1, 7, 30)):
                raise ValueError("Assessment query does not use a frozen practice prefix")
        elif query["phase"] != "practice":
            raise ValueError("Unexpected prediction phase")
        target = query["target"]
        if target is None:
            continue
        label = target["full_success"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Eligible targets must be binary integers")

        end = query["history_end"]
        if type(end) is not int or not cursor <= end <= len(events):
            raise ValueError("Invalid or out-of-order query cursor")
        consume_until(end)
        inputs = query["input"]
        now = stream.query_time(inputs["since_last_event_seconds"])
        gap = None if anchor is None else now - anchor
        task = inputs["target_task"]

        positions.append(len(tokens))
        tasks.append(task_features(task))
        query_gaps.append(0.0 if gap is None else gap / DAY)
        summaries.append(history.vector(now, inputs["since_last_response_seconds"], gap))
        labels.append(label)
        queries.append(Query(
            query["query_id"], record["learner_id"], record["skill_id"],
            query["purpose"], query.get("delay_days"), task["task_type"],
        ))

    if not queries:
        return None
    # Tokens after the last query have no loss or prediction attached.
    return Sequence(
        matrix(tokens, TOKEN_SIZE), np.asarray(gaps, dtype=np.float32),
        np.asarray(positions, dtype=np.int64), matrix(tasks, TASK_SIZE),
        np.asarray(query_gaps, dtype=np.float32), matrix(summaries, SUMMARY_SIZE),
        np.asarray(labels, dtype=np.float32), tuple(queries),
    )


@dataclass
class Partition:
    sequences: list[Sequence]

    def __post_init__(self):
        self.labels = np.concatenate([sequence.labels for sequence in self.sequences])
        self.queries = tuple(query for sequence in self.sequences for query in sequence.queries)
        if len({query.identifier for query in self.queries}) != len(self.queries):
            raise ValueError("Duplicate query ID in partition")

    def tabular(self, *, history):
        tasks = np.concatenate([sequence.tasks for sequence in self.sequences])
        if not history:
            return tasks
        summaries = np.concatenate([sequence.summaries for sequence in self.sequences])
        return np.concatenate((tasks, summaries), axis=1)


def checked_v21_records(cohort, split):
    """Accept audited v2.1 exports from the requested learner partition only."""
    manifest_path = cohort / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((cohort / "audit.json").read_text())
    if (manifest["version"] != "2.1.0" or audit["status"] != "passed"
            or audit["audited_manifest_sha256"] != digest(manifest_path)):
        raise ValueError("An audited v2.1 cohort is required")
    relative = f"model_data/{split}.jsonl.gz"
    for path in ("config.json", relative):
        if path not in manifest["files"] or digest(cohort / path) != manifest["files"][path]:
            raise ValueError(f"Missing or modified v2.1 input {path}")
    config = json.loads((cohort / "config.json").read_text())
    if config["evaluation_only"] and split != "test":
        raise ValueError("Stress cohorts may only be evaluated")
    allowed = {learner for learner, partition, _ in config["assignments"] if partition == split}
    for record in read_rows(cohort / relative):
        if record["split"] != split or record["learner_id"] not in allowed:
            raise ValueError("Learner partition mismatch")
        yield record


def load_partition(root, cohort, split, *, learner_limit=None, version="2.0.0"):
    learners = set()
    sequences = []
    record_keys = set()
    if version not in {"2.0.0", "2.1.0"}:
        raise ValueError(f"Unsupported dataset version {version}")
    load_records = checked_v21_records if version == "2.1.0" else checked_records
    for record in load_records(root / cohort, split):
        learner = record["learner_id"]
        if learner_limit is not None and learner not in learners and len(learners) >= learner_limit:
            break
        learners.add(learner)
        key = (learner, record["skill_id"])
        if key in record_keys:
            raise ValueError(f"Duplicate learner-skill record {key}")
        record_keys.add(key)
        sequence = encode_record(
            record, training=split == "train", include_training_assessments=version == "2.1.0",
        )
        if sequence is not None:
            sequences.append(sequence)
    if not sequences:
        raise ValueError(f"No eligible queries for {cohort}/{split}")
    return Partition(sequences)
