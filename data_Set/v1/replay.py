"""Causal feature assembly. Only materialize_query() supplies model inputs."""
from __future__ import annotations
from copy import deepcopy
from catalog import SKILLS, ACTION_NAMES
from simulator import VERSION, TASK_INDEX

COMMON = {"schema_version", "event_id", "learner_id", "skill_id", "course_id", "session_id",
          "branch", "interaction_id", "event_type", "occurred_at", "available_at"}
FIELDS = {
    "instruction": {"teaching_action"},
    "task": {"task_id", "difficulty", "task_type", "reference_seconds"},
    "help": {"task_id", "hint_count", "hint_type", "scaffold", "answer_revealed", "teaching_action"},
    "response": {"task_id", "attempt_number", "active_response_seconds", "hint_count", "hint_type",
                 "scaffold", "answer_revealed", "assistance"},
    "grade": {"task_id", "response_id", "grading_status", "criteria", "score", "full_success",
              "rubric_id", "grader", "grader_confidence"},
}
FEATURE_FIELDS = {"event_type", "gap_seconds", "new_session", "difficulty", "task_type", "reference_seconds",
                  "teaching_action", "attempt_number", "active_response_seconds", "relative_response_time",
                  "hint_count", "hint_type", "scaffold", "answer_revealed", "assistance",
                  "score", "full_success", "grading_status"}


def validate_events(events):
    seen, interactions, responses = set(), {}, {}
    last = float("-inf")
    learner = events[0]["learner_id"] if events else None
    for e in events:
        kind = e["event_type"]
        if kind not in FIELDS or set(e) != COMMON | FIELDS[kind]:
            raise ValueError("unexpected or missing event fields")
        if e["schema_version"] != VERSION or e["skill_id"] not in SKILLS or e["learner_id"] != learner:
            raise ValueError("invalid version, skill or learner")
        if e["event_id"] in seen:
            raise ValueError("duplicate event")
        seen.add(e["event_id"])
        if not last <= e["available_at"] or e["occurred_at"] > e["available_at"]:
            raise ValueError("noncausal event times")
        last = e["available_at"]
        if kind == "instruction":
            if e["teaching_action"] not in ACTION_NAMES[:-1]:
                raise ValueError("invalid standalone action")
            continue
        task = TASK_INDEX.get(e["task_id"])
        if task is None or task["skill_id"] != e["skill_id"]:
            raise ValueError("task mapping mismatch")
        key = (e["branch"], e["interaction_id"])
        if kind == "task":
            if key in interactions:
                raise ValueError("duplicate presentation")
            if e["difficulty"] != task["difficulty"] or e["task_type"] != task["task_type"]:
                raise ValueError("task metadata mismatch")
            if e["reference_seconds"] not in (None, task["reference_seconds"]):
                raise ValueError("reference mismatch")
            interactions[key] = dict(task_id=e["task_id"], attempts=0, help=None)
            continue
        if key not in interactions or interactions[key]["task_id"] != e["task_id"]:
            raise ValueError("event without matching presentation")
        state = interactions[key]
        if kind == "help":
            if e["hint_count"] != 1 or e["hint_type"] not in ("conceptual", "scaffold", "reveal"):
                raise ValueError("invalid help")
            if e["scaffold"] != (e["hint_type"] == "scaffold") or e["answer_revealed"] != (e["hint_type"] == "reveal"):
                raise ValueError("inconsistent help")
            state["help"] = e["hint_type"]
        elif kind == "response":
            state["attempts"] += 1
            if e["attempt_number"] != state["attempts"]:
                raise ValueError("attempt ordering mismatch")
            duration = e["active_response_seconds"]
            if duration is not None and not 0 < duration <= 600:
                raise ValueError("invalid duration")
            if e["assistance"] not in ("independent", "assisted", "unknown"):
                raise ValueError("invalid assistance")
            if e["assistance"] == "unknown":
                if any(e[k] is not None for k in ("hint_count", "hint_type", "scaffold", "answer_revealed")):
                    raise ValueError("unknown help must remain explicitly unknown")
            else:
                help_kind = state["help"]
                expected = dict(hint_count=int(help_kind is not None), hint_type=help_kind,
                                scaffold=help_kind == "scaffold", answer_revealed=help_kind == "reveal",
                                assistance="assisted" if help_kind else "independent")
                if any(e[k] != v for k, v in expected.items()):
                    raise ValueError("assistance contradicts history")
            responses[e["event_id"]] = {"event": e, "graded": False}
        elif kind == "grade":
            r = responses.get(e["response_id"])
            if not r or r["graded"] or r["event"]["interaction_id"] != e["interaction_id"]:
                raise ValueError("invalid grade linkage or duplicate grade")
            if r["event"]["skill_id"] != e["skill_id"] or r["event"]["task_id"] != e["task_id"]:
                raise ValueError("grade attribution mismatch")
            r["graded"] = True
            if e["grader"] != "simulated_rubric_v1" or e["grader_confidence"] is not None or e["rubric_id"] != "two_required_outputs_v1":
                raise ValueError("false grading provenance")
            if e["grading_status"] == "unresolved":
                if any(e[k] is not None for k in ("score", "full_success", "criteria")):
                    raise ValueError("unresolved grade carries a label")
            elif e["grading_status"] == "resolved":
                criteria = e["criteria"]
                if not isinstance(criteria, list) or len(criteria) != 2 or any(type(c) is not bool for c in criteria):
                    raise ValueError("invalid criteria")
                if e["score"] != sum(criteria) / 2 or e["full_success"] is not all(criteria):
                    raise ValueError("rubric mismatch")
            else:
                raise ValueError("invalid grading status")
    if any(not r["graded"] for r in responses.values()):
        raise ValueError("missing grade event")
    return True


def prepare(events):
    """One sequence per skill; task queries point to strict prefixes of events."""
    records, tasks, responses, queries = {}, {}, {}, {}
    last_event, last_response, last_session = {}, {}, {}
    for e in events:
        skill = e["skill_id"]
        rec = records.setdefault(skill, dict(learner_id=e["learner_id"], skill_id=skill, events=[], queries=[]))
        kind, interaction = e["event_type"], e["interaction_id"]
        task = tasks.get(interaction)
        if kind == "task":
            task = tasks[interaction] = e
            q = dict(query_id=e["event_id"], phase="practice" if e["branch"] == "practice" else "assessment",
                     history_end=len(rec["events"]),
                     input=dict(target_task={k: e[k] for k in ("difficulty", "task_type", "reference_seconds")},
                                since_last_event_seconds=None if skill not in last_event else e["available_at"] - last_event[skill],
                                since_last_response_seconds=None if skill not in last_response else e["available_at"] - last_response[skill]),
                     target=None)
            queries[interaction] = q
            rec["queries"].append(q)
        if kind == "response":
            responses[e["event_id"]] = e
            last_response[skill] = e["occurred_at"]
        response = responses.get(e.get("response_id")) if kind == "grade" else (e if kind == "response" else None)
        features = {k: None for k in FEATURE_FIELDS}
        features.update(event_type=kind, gap_seconds=None if skill not in last_event else e["available_at"] - last_event[skill],
                        new_session=None if skill not in last_session else e["session_id"] != last_session[skill])
        if task:
            features.update({k: task[k] for k in ("difficulty", "task_type", "reference_seconds")})
        for key in ("teaching_action", "hint_count", "hint_type", "scaffold", "answer_revealed",
                    "score", "full_success", "grading_status"):
            if key in e:
                features[key] = e[key]
        if response:
            features.update({k: response[k] for k in ("attempt_number", "hint_count", "hint_type", "scaffold", "answer_revealed", "assistance")})
            if kind == "response":
                duration = response["active_response_seconds"]
                features["active_response_seconds"] = duration
                features["relative_response_time"] = duration / task["reference_seconds"] if duration is not None and task["reference_seconds"] else None
        if kind == "grade" and response["attempt_number"] == 1:
            if e["grading_status"] == "resolved" and response["assistance"] == "independent":
                queries[interaction]["target"] = {"full_success": int(e["full_success"]), "score": e["score"]}
        rec["events"].append(features)
        last_event[skill], last_session[skill] = e["available_at"], e["session_id"]
    return records


def materialize_query(record, query):
    """Network-safe input: no IDs, labels, entire future sequence, or hidden parameters."""
    end = query["history_end"]
    if type(end) is not int or not 0 <= end <= len(record["events"]):
        raise ValueError("invalid history prefix")
    history = record["events"][:end]
    if any(set(e) != FEATURE_FIELDS for e in history):
        raise ValueError("feature schema mismatch")
    if set(query["input"]) != {"target_task", "since_last_event_seconds", "since_last_response_seconds"}:
        raise ValueError("query schema mismatch")
    if set(query["input"]["target_task"]) != {"difficulty", "task_type", "reference_seconds"}:
        raise ValueError("task feature schema mismatch")
    return deepcopy({"history": history, **query["input"]})


def training_examples(records):
    for record in records:
        for q in record["queries"]:
            if q["phase"] == "practice" and q["target"] is not None:
                yield materialize_query(record, q), q["target"]["full_success"]
