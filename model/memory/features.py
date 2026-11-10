"""Convert the audited v2 projection to causal model inputs."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

DAY = 86_400
FORMATS = ("single_choice", "multiple_select", "output_prediction", "fill_in_the_blank", "code_completion")
DIFFICULTIES = ("easy", "medium", "hard")
EVIDENCE_TYPES = ("attempt", "hint", "solution_reveal", "instruction")
HINT_TYPES = ("conceptual", "targeted", "scaffold")
ASSISTANCE_TYPES = ("independent", "assisted", "unknown")
GRADE_TYPES = ("resolved", "unresolved")
PAIR_FIELDS = ("task", "attempt_number", "assistance", "hint_count", "answer_revealed")


def indicators(value, vocabulary):
    if value is not None and value not in vocabulary:
        raise ValueError(f"Unexpected category {value!r} for {vocabulary}")
    return [float(value == category) for category in vocabulary]


def optional(value, *, scale=1, log=False):
    if value is None:
        return [0.0, 0.0]
    value = value / scale
    return [math.log1p(value) if log else value, 1.0]


def task_features(task):
    if task is None:
        return [0.0] * TASK_SIZE
    return [
        *indicators(task["task_type"], FORMATS),
        *indicators(task["difficulty"], DIFFICULTIES),
        *optional(task["reference_seconds"], scale=60, log=True),
        *(math.log1p(task[key]) for key in ("snippet_count", "option_count", "criterion_count")),
        *(value for key in ("credit_weights", "penalty_weights")
          for value in (math.log1p(len(task[key])), sum(task[key]), max(task[key], default=0.0))),
    ]


TASK_SIZE = len(task_features({
    "task_type": None, "difficulty": None, "reference_seconds": None,
    "snippet_count": 0, "option_count": 0, "criterion_count": 0,
    "credit_weights": [], "penalty_weights": [],
}))


@dataclass(frozen=True)
class Evidence:
    kind: str
    fields: dict
    time: float

    def vector(self):
        event = self.fields
        return [
            *indicators(self.kind, EVIDENCE_TYPES),
            *task_features(event.get("task")),
            *indicators(event.get("hint_type"), HINT_TYPES),
            *indicators(event.get("assistance"), ASSISTANCE_TYPES),
            *indicators(event.get("grading_status"), GRADE_TYPES),
            *optional(event.get("attempt_number"), log=True),
            *optional(event.get("hint_count"), log=True),
            *optional(event.get("active_response_seconds"), scale=60, log=True),
            *optional(event.get("relative_response_time"), log=True),
            *optional(event.get("score")),
            *optional(event.get("full_success")),
            *optional(event.get("answer_revealed")),
            *optional(event.get("new_session")),
        ]


TOKEN_SIZE = len(Evidence("instruction", {}, 0).vector())


class EvidenceStream:
    """Read one projected learner-skill stream in availability order."""

    def __init__(self):
        self.time = 0.0
        self.events_seen = 0
        self.pending = None

    def push(self, event):
        gap = event["gap_seconds"]
        if self.events_seen == 0:
            if gap is not None:
                raise ValueError("First projected event must have no previous-event gap")
        else:
            self.time += checked_gap(gap)
        self.events_seen += 1

        kind = event["event_type"]
        if self.pending is not None and kind != "grade":
            raise ValueError("Projected response must be followed by its matching grade")
        if kind == "response":
            self.pending = event
            return None
        if kind == "grade":
            response = self.pending
            if response is None or any(response[key] != event[key] for key in PAIR_FIELDS):
                raise ValueError("Cannot pair projected grade with the pending response")
            self.pending = None
            fields = dict(event)
            for key in ("active_response_seconds", "relative_response_time"):
                fields[key] = response[key]
            if response.get("new_session"):
                fields["new_session"] = True
            return Evidence("attempt", fields, self.time)
        if kind in EVIDENCE_TYPES[1:]:
            return Evidence(kind, event, self.time)
        return None

    def query_time(self, gap_since_event):
        if self.events_seen == 0:
            if gap_since_event is not None:
                raise ValueError("Empty history cannot have a previous-event gap")
            return 0.0
        return self.time + checked_gap(gap_since_event)


def checked_gap(value):
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid elapsed time {value!r}")
    return value


class Mean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, value):
        if value is not None:
            self.total += value
            self.count += 1

    def value(self):
        return self.total / self.count if self.count else None


class HistorySummary:
    """Shared fixed features for both history-aware tabular models."""

    def __init__(self):
        self.attempts = 0
        self.retries = 0
        self.unknown = 0
        self.unresolved = 0
        self.hints = dict.fromkeys(HINT_TYPES, 0)
        self.reveals = 0
        self.instructions = 0
        self.unaided_score = Mean()
        self.unaided_success = Mean()
        self.assisted_score = Mean()
        self.assisted_success = Mean()
        self.response_time = Mean()
        self.relative_time = Mean()
        self.recent_unaided = deque(maxlen=3)
        self.last_unaided_grade = None

    def update(self, evidence):
        fields = evidence.fields
        if evidence.kind == "hint":
            self.hints[fields["hint_type"]] += 1
        elif evidence.kind == "solution_reveal":
            self.reveals += 1
        elif evidence.kind == "instruction":
            self.instructions += 1
        elif evidence.kind == "attempt":
            self.attempts += 1
            self.retries += fields["attempt_number"] > 1
            self.unknown += fields["assistance"] == "unknown"
            self.unresolved += fields["grading_status"] == "unresolved"
            self.response_time.add(fields["active_response_seconds"])
            self.relative_time.add(fields["relative_response_time"])
            if fields["assistance"] == "independent" and fields["attempt_number"] == 1:
                self.unaided_score.add(fields["score"])
                self.unaided_success.add(fields["full_success"])
                if fields["full_success"] is not None:
                    self.last_unaided_grade = evidence.time
                    self.recent_unaided.append(float(fields["full_success"]))
            elif fields["assistance"] == "assisted":
                self.assisted_score.add(fields["score"])
                self.assisted_success.add(fields["full_success"])

    def vector(self, now, gap_since_response, gap_since_evidence):
        recent = self.recent_unaided
        last_unaided_gap = None if self.last_unaided_grade is None else now - self.last_unaided_grade
        counts = [
            self.attempts, self.retries, self.unknown, self.unresolved,
            *self.hints.values(), self.reveals, self.instructions,
            self.unaided_success.count, self.assisted_success.count, self.response_time.count,
        ]
        return [
            *(math.log1p(count) for count in counts),
            *optional(self.unaided_score.value()),
            *optional(self.unaided_success.value()),
            *optional(self.assisted_score.value()),
            *optional(self.assisted_success.value()),
            *optional(self.response_time.value(), scale=60, log=True),
            *optional(self.relative_time.value(), log=True),
            *optional(sum(recent) / len(recent) if recent else None),
            *optional(recent[-1] if recent else None),
            *optional(gap_since_response, scale=DAY, log=True),
            *optional(gap_since_evidence, scale=DAY, log=True),
            *optional(last_unaided_gap, scale=DAY, log=True),
        ]


SUMMARY_SIZE = len(HistorySummary().vector(0.0, None, None))


def matrix(rows, width):
    return np.asarray(rows, dtype=np.float32).reshape(-1, width)
