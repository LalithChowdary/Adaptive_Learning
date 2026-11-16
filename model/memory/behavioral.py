"""Behavioral sanity check suite for learner-memory models under controlled scenarios."""
import math
import torch

from .features import DAY, FORMATS, EvidenceStream, task_features


def _make_task(fmt="output_prediction", difficulty="medium", ref_sec=75.0, snippets=2, options=0, criteria=2):
    weights = [round(1.0 / criteria, 6)] * criteria
    return {
        "task_type": fmt,
        "difficulty": difficulty,
        "reference_seconds": ref_sec,
        "snippet_count": snippets,
        "option_count": options,
        "criterion_count": criteria,
        "credit_weights": weights,
        "penalty_weights": [0.5, 0.5] if fmt == "multiple_select" else [],
    }


def _event(kind, gap_seconds=0.0, task=None, **fields):
    row = {
        "event_type": kind, "gap_seconds": gap_seconds, "new_session": False,
        "task": task, "attempt_number": None, "active_response_seconds": None,
        "relative_response_time": None, "hint_count": None, "hint_type": None,
        "answer_revealed": None, "assistance": None, "score": None,
        "full_success": None, "grading_status": None,
    }
    row.update(fields)
    return row


def _attempt_sequence(task, score, full_success, active_sec, hints=0, attempt=1, is_first=True):
    rel = (active_sec / task["reference_seconds"]) if (active_sec is not None and task["reference_seconds"]) else None
    assistance = "assisted" if hints > 0 else "independent"
    events = [_event("task", gap_seconds=None if is_first else 10.0, task=task, new_session=None if is_first else False)]
    for h in range(hints):
        events.append(_event("help_request", gap_seconds=2.0, task=task))
        events.append(_event(
            "hint", gap_seconds=6.0, task=task, hint_count=h + 1,
            hint_type=("conceptual", "targeted", "scaffold")[min(h, 2)],
            assistance="assisted",
        ))
    events.append(_event(
        "response", gap_seconds=active_sec if active_sec is not None else 30.0,
        task=task, attempt_number=attempt, active_response_seconds=active_sec,
        relative_response_time=rel, hint_count=hints, answer_revealed=False,
        assistance=assistance,
    ))
    events.append(_event(
        "grade", gap_seconds=0.5, task=task, attempt_number=attempt,
        hint_count=hints, answer_revealed=False, assistance=assistance,
        score=score, full_success=full_success, grading_status="resolved",
    ))
    return events


@torch.no_grad()
def predict_from_events(model, events, target_task, query_gap_days=0.0):
    device = next(model.parameters()).device
    model.eval()
    stream = EvidenceStream()
    state = model.initial_state(1)
    anchor = None
    for event in events:
        evidence = stream.push(event)
        if evidence is not None:
            gap_days = 0.0 if anchor is None else (evidence.time - anchor) / DAY
            token_t = torch.tensor([evidence.vector()], dtype=torch.float32, device=device)
            gap_t = torch.tensor([gap_days], dtype=torch.float32, device=device)
            state = model.observe(state, token_t, gap_t)
            anchor = evidence.time
    task_t = torch.tensor([task_features(target_task)], dtype=torch.float32, device=device)
    qgap_t = torch.tensor([float(query_gap_days)], dtype=torch.float32, device=device)
    logit = model.predict_logit(state, task_t, qgap_t)
    return float(torch.sigmoid(logit).item())


@torch.no_grad()
def evaluate_behavioral_suite(model):
    """Run all 6 pedagogical sanity checks and return detailed pass/fail diagnostics."""
    ref_task = _make_task("output_prediction", "medium", ref_sec=75.0, snippets=2, criteria=2)
    hard_task = _make_task("output_prediction", "hard", ref_sec=84.96, snippets=3, criteria=3)
    cold_p = predict_from_events(model, [], ref_task, 0.0)

    # 1. Wrong answer (0%) across speeds must decrease prediction
    wrong_cases = {}
    for label, sec in (("fast_5s", 5.0), ("user_6.4s", 6.426), ("medium_60s", 60.0), ("slow_180s", 180.0), ("missing_time", None)):
        evs = _attempt_sequence(hard_task, score=0.0, full_success=False, active_sec=sec, hints=0)
        p = predict_from_events(model, evs, ref_task, 0.0)
        wrong_cases[label] = {"probability": p, "delta": p - cold_p, "passed": p < cold_p - 1e-5}

    # 2. Correct answer (100%) across speeds must increase prediction
    correct_cases = {}
    for label, sec in (("fast_5s", 5.0), ("medium_60s", 60.0), ("slow_180s", 180.0), ("missing_time", None)):
        evs = _attempt_sequence(hard_task, score=1.0, full_success=True, active_sec=sec, hints=0)
        p = predict_from_events(model, evs, ref_task, 0.0)
        correct_cases[label] = {"probability": p, "delta": p - cold_p, "passed": p > cold_p + 1e-5}

    # 3. Ordering: Unaided 100% > Assisted (3 hints) 100% > Unaided 0%
    p_unaided_pass = correct_cases["medium_60s"]["probability"]
    p_assisted_pass = predict_from_events(
        model, _attempt_sequence(hard_task, score=1.0, full_success=True, active_sec=60.0, hints=3), ref_task, 0.0
    )
    p_unaided_fail = wrong_cases["medium_60s"]["probability"]
    ordering_passed = p_unaided_pass > p_assisted_pass > p_unaided_fail

    # 4. Difficulty ordering: Easy > Medium > Hard across formats and states
    difficulty_cases = {}
    for fmt in FORMATS:
        opts = 4 if fmt in ("single_choice", "multiple_select") else 0
        snips = 1 if fmt != "Multiple" else 1
        crits = 1 if fmt == "single_choice" else 3
        p_easy = predict_from_events(model, [], _make_task(fmt, "easy", 45.0, snips, opts, crits), 0.0)
        p_med = predict_from_events(model, [], _make_task(fmt, "medium", 45.0, snips, opts, crits), 0.0)
        p_hard = predict_from_events(model, [], _make_task(fmt, "hard", 45.0, snips, opts, crits), 0.0)
        difficulty_cases[fmt] = {
            "easy": p_easy, "medium": p_med, "hard": p_hard,
            "passed": p_easy > p_med > p_hard,
        }

    # 5. Non-increasing and realistic forgetting over time (both after success and after failure, plus 2-pass retention floor)
    horizons = (0.0, 1.0, 7.0, 14.0, 30.0, 60.0)
    pass_events = _attempt_sequence(hard_task, score=1.0, full_success=True, active_sec=45.0, hints=0)
    two_pass_events = pass_events + _attempt_sequence(hard_task, score=1.0, full_success=True, active_sec=45.0, hints=0, is_first=False)
    fail_events = _attempt_sequence(hard_task, score=0.0, full_success=False, active_sec=120.0, hints=0)
    curve_after_pass = [predict_from_events(model, pass_events, ref_task, d) for d in horizons]
    curve_after_two_pass = [predict_from_events(model, two_pass_events, ref_task, d) for d in horizons]
    curve_after_fail = [predict_from_events(model, fail_events, ref_task, d) for d in horizons]
    mono_after_pass = all(curve_after_pass[i + 1] <= curve_after_pass[i] + 1e-6 for i in range(len(horizons) - 1))
    mono_after_two_pass = all(curve_after_two_pass[i + 1] <= curve_after_two_pass[i] + 1e-6 for i in range(len(horizons) - 1))
    mono_after_fail = all(curve_after_fail[i + 1] <= curve_after_fail[i] + 1e-6 for i in range(len(horizons) - 1))
    realistic_long_horizon = (curve_after_two_pass[4] > cold_p) and (curve_after_two_pass[5] > cold_p) and (curve_after_two_pass[4] >= 0.60 * curve_after_two_pass[0])

    # 6. User's exact dashboard Conditionals trace (1st attempt 0% in 6.426s -> 3 hints -> retry 0% in 2.27s)
    trace_events = [
        _event("task", gap_seconds=None, task=hard_task, new_session=None),
        _event("response", gap_seconds=6.426, task=hard_task, attempt_number=1, active_response_seconds=6.426, relative_response_time=6.426 / 84.96, hint_count=0, answer_revealed=False, assistance="independent"),
        _event("grade", gap_seconds=0.0, task=hard_task, attempt_number=1, hint_count=0, answer_revealed=False, assistance="independent", score=0.0, full_success=False, grading_status="resolved"),
        _event("help_request", gap_seconds=2.0, task=hard_task),
        _event("hint", gap_seconds=1.0, task=hard_task, hint_count=1, hint_type="conceptual", assistance="assisted"),
        _event("help_request", gap_seconds=2.0, task=hard_task),
        _event("hint", gap_seconds=1.0, task=hard_task, hint_count=2, hint_type="targeted", assistance="assisted"),
        _event("help_request", gap_seconds=2.0, task=hard_task),
        _event("hint", gap_seconds=1.0, task=hard_task, hint_count=3, hint_type="scaffold", assistance="assisted"),
        _event("retry", gap_seconds=1.0, task=hard_task, attempt_number=2),
        _event("response", gap_seconds=2.27, task=hard_task, attempt_number=2, active_response_seconds=2.27, relative_response_time=2.27 / 84.96, hint_count=3, answer_revealed=False, assistance="assisted"),
        _event("grade", gap_seconds=0.0, task=hard_task, attempt_number=2, hint_count=3, answer_revealed=False, assistance="assisted", score=0.0, full_success=False, grading_status="resolved"),
    ]
    p_after_first_fail = predict_from_events(model, trace_events[:3], ref_task, 0.0)
    p_after_three_hints = predict_from_events(model, trace_events[:9], ref_task, 0.0)
    p_after_second_fail = predict_from_events(model, trace_events, ref_task, 0.0)
    user_trace_passed = (p_after_first_fail < cold_p - 1e-5) and (p_after_three_hints <= cold_p + 1e-5) and (p_after_second_fail < p_after_first_fail - 1e-5)

    checks = {
        "zero_score_decreases_all_speeds": all(c["passed"] for c in wrong_cases.values()),
        "full_success_increases_all_speeds": all(c["passed"] for c in correct_cases.values()),
        "assistance_and_outcome_ordering": ordering_passed,
        "difficulty_monotonicity_all_formats": all(c["passed"] for c in difficulty_cases.values()),
        "non_increasing_forgetting_after_pass_and_fail": mono_after_pass and mono_after_two_pass and mono_after_fail and realistic_long_horizon,
        "user_conditionals_trace_behaves_sensibly": user_trace_passed,
    }
    return {
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "all_passed": all(checks.values()),
        "checks": checks,
        "cold_start_medium_output": cold_p,
        "wrong_by_speed": wrong_cases,
        "correct_by_speed": correct_cases,
        "ordering": {
            "unaided_pass": p_unaided_pass,
            "assisted_pass": p_assisted_pass,
            "unaided_fail": p_unaided_fail,
            "passed": ordering_passed,
        },
        "difficulty_by_format": difficulty_cases,
        "forgetting_curves": {
            "horizons_days": list(horizons),
            "after_pass": curve_after_pass,
            "after_two_pass": curve_after_two_pass,
            "after_fail": curve_after_fail,
            "monotonic_after_pass": mono_after_pass,
            "monotonic_after_two_pass": mono_after_two_pass,
            "monotonic_after_fail": mono_after_fail,
            "realistic_long_horizon": realistic_long_horizon,
        },
        "user_conditionals_replay": {
            "before_attempt": cold_p,
            "after_first_wrong_6.4s": p_after_first_fail,
            "after_three_hints": p_after_three_hints,
            "after_second_wrong_2.27s": p_after_second_fail,
            "passed": user_trace_passed,
        },
    }
