"""Build independently seeded pilot/main/stress cohorts without overwriting data."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import platform
from catalog import SKILLS, make_catalog, LESSONS
from simulator import Config, Student, VERSION, rng_for
from replay import prepare, validate_events
from storage import writer, write_row, save_json, digest


def export_records(practice, probes, include_assessments):
    base = prepare(practice)
    if include_assessments:
        for probe in probes:
            combined = prepare(practice + probe["events"])
            skill = probe["skill_id"]
            q = combined[skill]["queries"][-1]
            q["delay_days"] = probe["delay_days"]
            # Prefix is exactly the pre-probe practice sequence; probe events not appended.
            if q["history_end"] != len(base[skill]["events"]):
                raise ValueError("assessment prefix contaminated")
            base[skill]["queries"].append(q)
    return list(base.values())


def assignment(n, seed, cohort, evaluation_only=False):
    if n < 20 or n % 20:
        raise ValueError("cohort size must be a positive multiple of 20")
    order = list(range(n))
    rng_for(seed, cohort, "split").shuffle(order)
    splits = {"test": order} if evaluation_only else {
        "train": order[:n * 8 // 10], "validation": order[n * 8 // 10:n * 9 // 10], "test": order[n * 9 // 10:]}
    return [(f"v21_{cohort}_{i:05d}", split, "mixed" if j % 2 else "blocked")
            for split, indices in splits.items() for j, i in enumerate(indices)]


def build_cohort(root, name, n, config, evaluation_only=False):
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    import json
    for filename, value in (("task_catalog.json", make_catalog()), ("lesson_catalog.json", LESSONS)):
        path = root / filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("existing shared catalog differs")
        if not path.exists(): save_json(path, value)
    destination.mkdir(parents=True)
    assignments = assignment(n, config.seed, name, evaluation_only)
    save_json(destination / "config.json", dict(name=name, learners=n, config=asdict(config),
                                              evaluation_only=evaluation_only, assignments=assignments))
    outputs = []
    with ExitStack() as stack:
        handles = {}
        for split in sorted({a[1] for a in assignments}):
            for role in ("observations", "assessments", "model_data", "administration"):
                path = destination / role / f"{split}.jsonl.gz"
                handles[role, split] = stack.enter_context(writer(path))
                outputs.append(path)
        for index, (identity, split, schedule) in enumerate(assignments):
            student = Student(identity, config).practice(schedule)
            validate_events(student.events)
            for event in student.events:
                write_row(handles["observations", split], event)
            probes, hidden_probes = [], []
            for delay in config.assessment_days:
                for skill in SKILLS:
                    clone = student.assessment(delay, skill)
                    validate_events(clone.events)
                    probe = dict(learner_id=identity, skill_id=skill, delay_days=delay, events=clone.events)
                    probes.append(probe)
                    hidden_probes.append(dict(delay_days=delay, skill_id=skill, trace=clone.trace))
                    for event in clone.events:
                        write_row(handles["assessments", split], event)
            for record in export_records(student.events, probes, True):
                record["split"] = split
                write_row(handles["model_data", split], record)
            write_row(handles["administration", split], dict(**student.private_record(), assessments=hidden_probes))
            if index == 0:
                save_json(destination / "example_learner.json", dict(
                    learner_id=identity, split=split, schedule=schedule,
                    first_events=student.events[:20], first_skill_record=export_records(student.events, [], False)[0]))
            if (index + 1) % 100 == 0:
                print(f"{name}: {index + 1}/{n} learners", flush=True)
    outputs += [destination / "config.json", destination / "example_learner.json"]
    save_json(destination / "manifest.json", dict(version=VERSION, python=platform.python_version(),
              files={str(p.relative_to(destination)): digest(p) for p in outputs},
              catalog_files={name: digest(root / name) for name in ("task_catalog.json", "lesson_catalog.json")},
              source_files={str(p.relative_to(Path(__file__).parent)): digest(p) for p in Path(__file__).parent.rglob("*.py")}))
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "dataset")
    parser.add_argument("--cohort", choices=("pilot", "main", "power_law", "long_gap", "all"), default="pilot")
    parser.add_argument("--max-hints", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output / "task_catalog.json"
    import json
    if catalog_path.exists() and json.loads(catalog_path.read_text()) != make_catalog():
        raise ValueError("existing catalog differs")
    save_json(catalog_path, make_catalog())
    specs = {
        "pilot": (200, Config(spaced_practice=True, max_hints=args.max_hints, max_retries=args.max_retries, seed=2026111001), False),
        "main": (2000, Config(spaced_practice=True, max_hints=args.max_hints, max_retries=args.max_retries, seed=2026111002), False),
        "power_law": (200, Config(spaced_practice=True, max_hints=args.max_hints, max_retries=args.max_retries, seed=2026111003, mechanism="power_law"), True),
        "long_gap": (200, Config(spaced_practice=True, max_hints=args.max_hints, max_retries=args.max_retries, seed=2026111004, long_gaps=True), True),
    }
    for name in specs if args.cohort == "all" else [args.cohort]:
        n, config, evaluation = specs[name]
        if name != "pilot":
            pilot_report = args.output / "pilot/audit.json"
            if not pilot_report.exists() or json.loads(pilot_report.read_text())["status"] != "passed":
                raise ValueError("generate and audit a pilot before larger cohorts")
        built = build_cohort(args.output, name, n, config, evaluation)
        from audit_dataset import audit, write_report
        report = audit(built)
        write_report(built, report)
        print(f"{name}: audit passed ({report['check_categories_passed']} categories)", flush=True)


if __name__ == "__main__":
    main()
