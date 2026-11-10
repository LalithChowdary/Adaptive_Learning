"""Three authored histories: forced outcomes test handling, not learner realism."""
from pathlib import Path
from simulator import Config, Student, DAY
from storage import save_json
from replay import validate_events, prepare


def fixtures():
    config = Config(unresolved_probability=0, missing_time_probability=0,
                    missing_reference_probability=0, missing_assistance_probability=0)
    out = {}
    independent = Student("fixture_independent", config)
    independent.task("range_boundaries", forced_first=[True, True])
    independent.advance(DAY)
    independent.task("range_boundaries", forced_first=[True, True])
    independent.events[-3]["course_id"] = "python_followup_v1"
    independent.events[-2]["course_id"] = "python_followup_v1"
    independent.events[-1]["course_id"] = "python_followup_v1"
    out["independent_and_cross_course"] = independent.events
    assisted = Student("fixture_assisted", config)
    assisted.task("loop_tracing", forced_first=[False, False], forced_retry=[True, True])
    # The next independently attempted task fails; disable automatic retry for clarity.
    assisted.opportunity += 1
    from simulator import POOLS
    task = POOLS["loop_tracing", "practice"][0]
    assisted.emit("task", "loop_tracing", "later_task", task_id=task["task_id"], difficulty=task["difficulty"],
                  task_type=task["task_type"], reference_seconds=task["reference_seconds"])
    assisted.attempt(task, "later_task", 1, forced_criteria=[False, False])
    out["assisted_success_then_independent_failure"] = assisted.events
    returning = Student("fixture_gap", config)
    returning.task("function_returns", forced_first=[True, True])
    returning.advance(DAY)
    returning.task("list_indexing", forced_first=[True, True])
    returning.advance(29 * DAY)
    returning.task("function_returns", forced_first=[True, True])
    out["return_after_gap"] = returning.events
    return out


def main():
    root = Path(__file__).parent / "fixtures"
    for name, events in fixtures().items():
        validate_events(events)
        save_json(root / f"{name}.json", dict(kind="authored_controlled_history", events=events,
                                            prepared=list(prepare(events).values())))
    print("Wrote and validated 3 authored histories.")


if __name__ == "__main__":
    main()
