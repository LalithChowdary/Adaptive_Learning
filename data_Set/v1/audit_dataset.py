"""Reload all events, verify causal exports, counts, splits and integrity."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from itertools import groupby, zip_longest
import json
from pathlib import Path
from catalog import SKILLS
from simulator import DAY, TASK_INDEX, VERSION
from replay import validate_events, FEATURE_FIELDS
from build_dataset import export_records
from storage import read_rows, digest, save_json


def learner_groups(path):
    for learner, events in groupby(read_rows(path), key=lambda e: e["learner_id"]):
        yield learner, list(events)


def audit(root, check_sources=True):
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("version mismatch")
    for relative, checksum in manifest["files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"checksum mismatch: {relative}")
    if check_sources:
        for name, checksum in manifest["source_files"].items():
            if digest(Path(__file__).parent / name) != checksum:
                raise ValueError(f"generation source changed: {name}; reproduce with original source")
    assignments = {a[0]: (a[1], a[2]) for a in config["assignments"]}
    if len(assignments) != config["learners"]:
        raise ValueError("duplicate assignment")
    stats = defaultdict(Counter)
    gaps, durations, span_days = [], [], []
    observed_learners = set()
    primary_by_skill, delayed_by_day = defaultdict(Counter), defaultdict(Counter)
    for split in sorted({v[0] for v in assignments.values()}):
        raw = learner_groups(root / "observations" / f"{split}.jsonl.gz")
        assessment = learner_groups(root / "assessments" / f"{split}.jsonl.gz")
        models = learner_groups(root / "model_data" / f"{split}.jsonl.gz")
        hidden = iter(read_rows(root / "administration" / f"{split}.jsonl.gz"))
        for raw_group, assess_group, model_group, private in zip_longest(raw, assessment, models, hidden):
            if any(x is None for x in (raw_group, assess_group, model_group, private)):
                raise ValueError("missing or extra learner record")
            identity, practice = raw_group
            if identity in observed_learners or identity not in assignments or assignments[identity][0] != split:
                raise ValueError("learner split leakage or duplication")
            observed_learners.add(identity)
            if not identity == assess_group[0] == model_group[0] == private["learner_id"]:
                raise ValueError("learner routing mismatch")
            validate_events(practice)
            expected_tasks = config["config"]["tasks_per_skill"]
            tasks = [e for e in practice if e["event_type"] == "task"]
            if Counter(e["skill_id"] for e in tasks) != Counter({s: expected_tasks for s in SKILLS}):
                raise ValueError("skill task budget mismatch")
            if any(e["branch"] != "practice" for e in practice):
                raise ValueError("assessment in practice")
            if any(TASK_INDEX[e["task_id"]]["pool"] != "practice" for e in tasks):
                raise ValueError("assessment task in practice pool")
            if len({e["task_id"] for e in tasks}) != len(tasks):
                raise ValueError("repeated practice task ID")
            schedule = assignments[identity][1]
            skill_order = [e["skill_id"] for e in tasks]
            if schedule == "mixed":
                if any(set(skill_order[i:i+6]) != set(SKILLS) for i in range(0, len(tasks), 6)):
                    raise ValueError("mixed schedule mismatch")
            elif any(len(set(skill_order[i:i+expected_tasks])) != 1 for i in range(0, len(tasks), expected_tasks)):
                raise ValueError("blocked schedule mismatch")
            stats[split]["learners"] += 1
            stats[split][schedule] += 1
            probes = []
            seen_probes = set()
            assessment_tasks = set()
            for branch, branch_iter in groupby(assess_group[1], key=lambda e: e["branch"]):
                events = list(branch_iter)
                validate_events(events)
                if [e["event_type"] for e in events] != ["task", "response", "grade"]:
                    raise ValueError("assessment has assistance or retries")
                present, response, grade = events
                skill = present["skill_id"]
                delay = round((present["occurred_at"] - practice[-1]["available_at"]) / DAY)
                if abs(present["occurred_at"] - practice[-1]["available_at"] - delay * DAY) > 1e-5:
                    raise ValueError("assessment phase delay mismatch")
                if (delay, skill) in seen_probes or delay not in config["config"]["assessment_days"]:
                    raise ValueError("duplicate/unexpected assessment")
                seen_probes.add((delay, skill))
                if TASK_INDEX[present["task_id"]]["pool"] != "assessment" or response["assistance"] == "assisted":
                    raise ValueError("invalid assessment pool or assistance")
                if present["task_id"] in assessment_tasks:
                    raise ValueError("repeated assessment task ID")
                assessment_tasks.add(present["task_id"])
                probes.append(dict(skill_id=skill, delay_days=delay, events=events))
                if grade["grading_status"] == "resolved" and response["assistance"] == "independent":
                    delayed_by_day[f"{split}/{delay}d"]["eligible"] += 1
                    delayed_by_day[f"{split}/{delay}d"]["successes"] += int(grade["full_success"])
            if len(seen_probes) != len(SKILLS) * len(config["config"]["assessment_days"]):
                raise ValueError("missing delayed assessments")
            expected = export_records(practice, probes, split != "train")
            actual = model_group[1]
            if len(actual) != len(SKILLS):
                raise ValueError("missing skill sequence")
            for a, b in zip(actual, expected):
                b["split"] = split
                if a != b:
                    raise ValueError("model export differs from causal replay")
                for feature in a["events"]:
                    if set(feature) != FEATURE_FIELDS:
                        raise ValueError("hidden or unexpected model field")
                    if feature["gap_seconds"] is not None:
                        gaps.append(feature["gap_seconds"] / DAY)
                for q in a["queries"]:
                    if split == "train" and q["phase"] != "practice":
                        raise ValueError("delayed labels in fitting export")
                    if q["target"] is not None:
                        stats[split][f"eligible_{q['phase']}"] += 1
                        stats[split][f"successes_{q['phase']}"] += q["target"]["full_success"]
                        if q["phase"] == "practice":
                            primary_by_skill[f"{split}/{a['skill_id']}"]["eligible"] += 1
                            primary_by_skill[f"{split}/{a['skill_id']}"]["successes"] += q["target"]["full_success"]
            for e in practice + assess_group[1]:
                stats[split]["events"] += 1
                stats[split][f"event_{e['event_type']}"] += 1
                if e["event_type"] == "response":
                    stats[split][e["assistance"]] += 1
                    stats[split]["retries"] += int(e["attempt_number"] > 1)
                    if e["active_response_seconds"] is None:
                        stats[split]["missing_active_time"] += 1
                    else:
                        durations.append(e["active_response_seconds"])
                if e["event_type"] == "task" and e["reference_seconds"] is None:
                    stats[split]["missing_reference_time"] += 1
                if e["event_type"] == "grade":
                    stats[split][e["grading_status"]] += 1
                    stats[split]["partial_grades"] += int(e["score"] == .5)
            span_days.append((practice[-1]["available_at"] - practice[0]["available_at"]) / DAY)
    if observed_learners != set(assignments):
        raise ValueError("missing learners")
    for split, counter in stats.items():
        if counter["blocked"] != counter["mixed"]:
            raise ValueError("schedule imbalance")
    def distribution(values):
        values = sorted(values)
        return {"min": values[0], "median": values[len(values)//2], "p95": values[int(.95*(len(values)-1))], "max": values[-1]} if values else {}
    checks = ["file checksums", "generation source fingerprints", "unique disjoint learner assignments",
              "balanced declared schedules", "six-skill budgets", "strict event schema", "event and grade causality",
              "assistance consistency", "rubric derivation and unresolved labels", "simulated grading provenance",
              "independent delayed assessment branches", "assessment delays and coverage", "separate task pools",
              "all prepared records match causal replay", "no delayed fitting labels", "model feature allowlist",
              "no missing/extra observation, model, assessment or administration learner rows"]
    return dict(status="passed", cohort=config["name"], learner_count=len(observed_learners),
                check_categories_passed=len(checks), checks=checks, splits=dict(stats),
                primary_by_skill=dict(primary_by_skill), delayed_by_day=dict(delayed_by_day),
                skill_event_gaps_days=distribution(gaps), active_response_seconds=distribution(durations),
                practice_span_days=distribution(span_days),
                limitations="Internal consistency of synthetic data, not human fidelity or model effectiveness.")


def write_report(root, report):
    save_json(root / "audit.json", report)
    lines = [f"# {report['cohort']} dataset audit", "", f"Status: **{report['status']}**.", "",
             f"{report['learner_count']:,} learners; {report['check_categories_passed']} validation categories passed.", "",
             "| Split | Learners | Events | Responses (including retries) | Eligible practice targets | Eligible assessment targets |",
             "|---|---:|---:|---:|---:|---:|"]
    for split, s in report["splits"].items():
        lines.append(f"| {split} | {s['learners']} | {s['events']} | {s['event_response']} | {s['eligible_practice']} | {s['eligible_assessment']} |")
    lines += ["", "Training learners' assessment outcomes are retained in raw assessment files for audit only; they are absent from model fitting exports.", "",
              "## Delayed outcomes (descriptive)", "", "| Partition/delay | Eligible | Full successes |", "|---|---:|---:|"]
    for key, s in report["delayed_by_day"].items():
        lines.append(f"| {key} | {s['eligible']} | {s['successes']} |")
    lines += ["", "## Checks", ""] + [f"- Passed: {x}." for x in report["checks"]]
    lines += ["", "## Interpretation", "", report["limitations"], "",
              "Independent branches share a learner's practice history; they are correlated, not extra independent learners. No training or PPO effectiveness result is reported."]
    (root / "AUDIT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path)
    args = parser.parse_args()
    result = audit(args.cohort)
    write_report(args.cohort, result)
    print(json.dumps({k: result[k] for k in ("status", "cohort", "learner_count", "check_categories_passed")}, indent=2))


if __name__ == "__main__":
    main()
