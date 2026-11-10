"""Reload all events, verify causal exports, counts, splits and integrity."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from itertools import groupby, zip_longest
import json
from pathlib import Path
from catalog import SKILLS, FORMATS, grade, make_catalog
from simulator import DAY, PRACTICE_BLOCK_SIZE, TASK_INDEX, VERSION
from replay import validate_events, FEATURE_FIELDS
from build_dataset import export_records
from storage import read_rows, digest, save_json


def learner_groups(path):
    for learner, events in groupby(read_rows(path), key=lambda e: e["learner_id"]):
        yield learner, list(events)


def evidence_offsets(events):
    """Time since the most recent hybrid evidence at every raw cursor."""
    offsets = [None]
    offset = None
    for event in events:
        if event["event_type"] in {"grade", "hint", "solution_reveal", "instruction"}:
            offset = 0.0
        elif offset is not None:
            offset += event["gap_seconds"]
        offsets.append(offset)
    return offsets


def audit(root, check_sources=True):
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("version mismatch")
    for relative, checksum in manifest["files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"checksum mismatch: {relative}")
    for name, checksum in manifest["catalog_files"].items():
        if digest(root.parent / name) != checksum:
            raise ValueError("shared catalog checksum mismatch")
    if check_sources:
        for name, checksum in manifest["source_files"].items():
            if digest(Path(__file__).parent / name) != checksum:
                raise ValueError(f"generation source changed: {name}; reproduce with original source")
    assignments = {a[0]: (a[1], a[2]) for a in config["assignments"]}
    if len(assignments) != config["learners"]:
        raise ValueError("duplicate assignment")
    stats = defaultdict(Counter)
    if config["evaluation_only"] and any(split != "test" for split, _ in assignments.values()):
        raise ValueError("stress cohorts must be test-only")
    query_gaps, cold_queries = defaultdict(list), Counter()
    coverage, hint_counts, attempt_counts, exclusions = defaultdict(Counter), Counter(), Counter(), Counter()
    format_durations = defaultdict(list)
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
            all_tasks = [e for e in practice if e["event_type"] == "task"]
            tasks = [e for e in all_tasks if e["purpose"] == "ordinary"]
            followups = [e for e in all_tasks if e["purpose"] == "followup"]
            helped = {e["interaction_id"] for e in practice if e["event_type"] == "hint"}
            if Counter(e["parent_interaction_id"] for e in followups) != Counter({i: 1 for i in helped}):
                raise ValueError("missing or duplicate fresh followup")
            if any(e["event_type"] == "instruction" for e in practice):
                raise ValueError("routine pre-question instruction reintroduced")
            for e in all_tasks:
                if e["max_hints"] != config["config"]["max_hints"] or e["max_retries"] != config["config"]["max_retries"]:
                    raise ValueError("configured caps mismatch")
            if Counter(e["skill_id"] for e in tasks) != Counter({s: expected_tasks for s in SKILLS}):
                raise ValueError("skill task budget mismatch")
            if any(e["branch"] != "practice" for e in practice):
                raise ValueError("assessment in practice")
            if any(TASK_INDEX[e["task_id"]]["pool"] != "practice" for e in all_tasks):
                raise ValueError("assessment task in practice pool")
            if len({e["task_id"] for e in all_tasks}) != len(all_tasks):
                raise ValueError("repeated practice task ID")
            schedule = assignments[identity][1]
            skill_order = [e["skill_id"] for e in tasks]
            if schedule == "mixed":
                if any(set(skill_order[i:i+6]) != set(SKILLS) for i in range(0, len(tasks), 6)):
                    raise ValueError("mixed schedule mismatch")
            else:
                size = PRACTICE_BLOCK_SIZE if config["config"]["spaced_practice"] else expected_tasks
                cursor = 0
                for start in range(0, expected_tasks, size):
                    width = min(size, expected_tasks - start)
                    blocks = [skill_order[i:i + width] for i in range(cursor, cursor + width * len(SKILLS), width)]
                    if any(len(set(block)) != 1 for block in blocks) or {block[0] for block in blocks} != set(SKILLS):
                        raise ValueError("blocked revisit schedule mismatch")
                    cursor += width * len(SKILLS)
            stats[split]["learners"] += 1
            stats[split][schedule] += 1
            probes = []
            seen_probes = set()
            assessment_tasks = set()
            for branch, branch_iter in groupby(assess_group[1], key=lambda e: e["branch"]):
                events = list(branch_iter)
                validate_events(events)
                present = events[0]
                response = next(e for e in events if e["event_type"] == "response")
                first_grade = next(e for e in events if e["event_type"] == "grade")
                ending = next(e for e in events if e["event_type"] == "assessment_end")
                if ending["status"] == "incomplete" and response["phase"] != "review":
                    raise ValueError("interrupted assessment counted as scored")
                stats[split]["assessment_" + ending["status"]] += 1
                skill = present["skill_id"]
                delay = round((present["occurred_at"] - practice[-1]["available_at"]) / DAY)
                if abs(present["occurred_at"] - practice[-1]["available_at"] - delay * DAY) > 1e-5:
                    raise ValueError("assessment phase delay mismatch")
                if (delay, skill) in seen_probes or delay not in config["config"]["assessment_days"]:
                    raise ValueError("duplicate/unexpected assessment")
                seen_probes.add((delay, skill))
                if TASK_INDEX[present["task_id"]]["pool"] != "assessment" or (response["phase"] == "assessment" and response["assistance"] == "assisted"):
                    raise ValueError("invalid assessment pool or assistance")
                if present["task_id"] in assessment_tasks:
                    raise ValueError("repeated assessment task ID")
                assessment_tasks.add(present["task_id"])
                probes.append(dict(skill_id=skill, delay_days=delay, events=events))
                if first_grade["grading_status"] == "resolved" and response["assistance"] == "independent" and response["phase"] == "assessment":
                    delayed_by_day[f"{split}/{delay}d"]["eligible"] += 1
                    delayed_by_day[f"{split}/{delay}d"]["successes"] += int(first_grade["full_success"])
            if len(seen_probes) != len(SKILLS) * len(config["config"]["assessment_days"]):
                raise ValueError("missing delayed assessments")
            expected = export_records(practice, probes, True)
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
                offsets = evidence_offsets(a["events"])
                for q in a["queries"]:
                    cell = coverage[f"{split}/{a['skill_id']}/{q['input']['target_task']['task_type']}/{q['phase']}/{q.get('delay_days', 0)}d"]
                    cell["queries"] += 1
                    if q["target"] is None:
                        exclusions[q["target_exclusion"]] += 1
                    else:
                        cell["eligible"] += 1
                        cell["successes"] += q["target"]["full_success"]
                    if q["phase"] == "assessment" and q["history_end"] != len(a["events"]):
                        raise ValueError("delayed query must use only the frozen practice history")
                    if q["target"] is not None:
                        group = f"{split}/{q['phase']}"
                        offset = offsets[q["history_end"]]
                        if offset is None:
                            cold_queries[group] += 1
                        else:
                            gap_days = (offset + q["input"]["since_last_event_seconds"]) / DAY
                            query_gaps[group].append(gap_days)
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
                    hint_counts[str(e["hint_count"])] += 1
                    attempt_counts[str(e["attempt_number"])] += 1
                    if e["active_response_seconds"] is None:
                        stats[split]["missing_active_time"] += 1
                    else:
                        durations.append(e["active_response_seconds"])
                        format_durations[TASK_INDEX[e["task_id"]]["task_type"]].append(e["active_response_seconds"])
                if e["event_type"] == "task" and e["reference_seconds"] is None:
                    stats[split]["missing_reference_time"] += 1
                if e["event_type"] == "grade":
                    stats[split][e["grading_status"]] += 1
                    stats[split]["partial_grades"] += int(e["score"] is not None and 0 < e["score"] < 1)
            traces = private["trace"] + [t for branch in private["assessments"] for t in branch["trace"]]
            responses_by_id = {e["event_id"]: e for e in practice + assess_group[1] if e["event_type"] == "response"}
            grades_by_response = {e["response_id"]: e for e in practice + assess_group[1] if e["event_type"] == "grade"}
            if {t["response_id"] for t in traces} != set(responses_by_id) or len(traces) != len(responses_by_id):
                raise ValueError("hidden trace attribution mismatch")
            for trace in traces:
                if not (0 <= trace["strength_before"] <= trace["strength_after"] <= 1
                        and trace["strength_before"] <= trace["strength_at_attempt"] <= 1):
                    raise ValueError("unbounded retained learning")
                r = responses_by_id[trace["response_id"]]
                g = grades_by_response[trace["response_id"]]
                if r["branch"] != "practice" and trace["strength_before"] != trace["strength_after"]:
                    raise ValueError("assessment branch learns")
                if g["grading_status"] == "resolved" and grade(TASK_INDEX[r["task_id"]], trace["evidence_truth"])["score"] != g["score"]:
                    raise ValueError("hidden/observed score mismatch")
            span_days.append((practice[-1]["available_at"] - practice[0]["available_at"]) / DAY)
    if observed_learners != set(assignments):
        raise ValueError("missing learners")
    for split, counter in stats.items():
        if counter["blocked"] != counter["mixed"]:
            raise ValueError("schedule imbalance")
        if config["config"]["spaced_practice"] and split == "train":
            practice_gaps = query_gaps[f"{split}/practice"]
            if not any(gap > 30 for gap in practice_gaps):
                raise ValueError("training practice has no labels beyond 30 days")
            if any(not delayed_by_day[f"{split}/{delay}d"]["eligible"]
                   for delay in config["config"]["assessment_days"]):
                raise ValueError("training lacks eligible labels at a delayed horizon")
    def distribution(values):
        values = sorted(values)
        return {"min": values[0], "median": values[len(values)//2], "p95": values[int(.95*(len(values)-1))],
                "p99": values[int(.99*(len(values)-1))], "max": values[-1]} if values else {}
    missing_cells = []
    for split in stats:
        for skill in SKILLS:
            for fmt in FORMATS:
                if config["config"]["tasks_per_skill"] >= 5 and not coverage[f"{split}/{skill}/{fmt}/practice/0d"]["queries"]:
                    raise ValueError("missing practice skill/format coverage")
                for delay in config["config"]["assessment_days"]:
                    if not coverage[f"{split}/{skill}/{fmt}/assessment/{delay}d"]["queries"]:
                        missing_cells.append(f"{split}/{skill}/{fmt}/{delay}d")
    checks = ["question-first default with no routine instruction", "fresh followup for every assisted practice task",
              "configurable caps and per-hint/per-retry chronology", "assessment interruption and review semantics",
              "five-format skill coverage", "shared task and lesson catalog checksums", "hidden trace attribution and bounded response-only gains","file checksums", "generation source fingerprints", "unique disjoint learner assignments",
              "balanced declared schedules", "six-skill budgets", "strict event schema", "event and grade causality",
              "assistance consistency", "rubric derivation and unresolved labels", "simulated grading provenance",
              "independent delayed assessment branches", "assessment delays and coverage", "separate task pools",
              "all prepared records match causal replay", "delayed fitting labels use frozen practice prefixes",
              "spaced training gap and horizon coverage", "stress cohorts are test-only", "model feature allowlist",
              "no missing/extra observation, model, assessment or administration learner rows"]
    return dict(audited_manifest_sha256=digest(root / "manifest.json"), status="passed", cohort=config["name"], learner_count=len(observed_learners),
                check_categories_passed=len(checks), checks=checks, splits=dict(stats),
                primary_by_skill=dict(primary_by_skill), delayed_by_day=dict(delayed_by_day),
                coverage=dict(coverage), missing_assessment_coverage_cells=missing_cells,
                response_hint_counts=dict(hint_counts), response_attempt_numbers=dict(attempt_counts),
                target_exclusions=dict(exclusions), response_time_by_format={k: distribution(v) for k,v in format_durations.items()},
                labeled_query_gaps_days={k: dict(
                    **distribution(v), queries=len(v), cold_start=cold_queries[k],
                    over_5_days=sum(gap > 5 for gap in v),
                    over_30_days=sum(gap > 30 for gap in v),
                ) for k, v in query_gaps.items()},
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
    lines += ["", "Delayed training labels use only the train learners' frozen practice histories.", "",
              "## Labeled gaps from the last model evidence", "",
              "| Partition/phase | Non-cold queries | Cold starts | Median days | 99th percentile | Max days | Over 5 days | Over 30 days |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, s in report["labeled_query_gaps_days"].items():
        lines.append(f"| {group} | {s['queries']} | {s['cold_start']} | {s['median']:.3f} | {s['p99']:.2f} | {s['max']:.2f} | {s['over_5_days']} | {s['over_30_days']} |")
    lines += ["",
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
