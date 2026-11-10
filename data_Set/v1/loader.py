"""Safe default loading: train split, practice labels, strict historical prefixes."""
import json
from pathlib import Path
from replay import training_examples, materialize_query
from storage import read_rows, digest


def checked_records(cohort, split):
    cohort = Path(cohort)
    manifest = json.loads((cohort / "manifest.json").read_text())
    config = json.loads((cohort / "config.json").read_text())
    relative = f"model_data/{split}.jsonl.gz"
    if relative not in manifest["files"] or digest(cohort / relative) != manifest["files"][relative]:
        raise ValueError("missing or modified model data")
    allowed = {a[0] for a in config["assignments"] if a[1] == split}
    for record in read_rows(cohort / relative):
        if record["split"] != split or record["learner_id"] not in allowed:
            raise ValueError("split mismatch")
        yield record


def load_training_examples(cohort):
    """Yield (input, binary label); never loads raw assessments or hidden files."""
    yield from training_examples(checked_records(cohort, "train"))


def load_evaluation_examples(cohort, split="validation", phase="assessment"):
    if split not in ("validation", "test") or phase not in ("practice", "assessment"):
        raise ValueError("evaluation requires explicit held-out partition and valid phase")
    for record in checked_records(cohort, split):
        for query in record["queries"]:
            if query["phase"] == phase and query["target"] is not None:
                yield materialize_query(record, query), query["target"]["full_success"]
