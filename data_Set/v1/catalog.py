"""Authored Python task bank. Text/answers never enter the memory features."""
SKILLS = {
    "range_boundaries": "Exclusive upper bound of range",
    "loop_tracing": "Trace repeated updates in a loop",
    "accumulator_initialization": "Account for an accumulator's starting value",
    "conditional_evaluation": "Choose the branch of a conditional",
    "function_returns": "Evaluate a function's returned value",
    "list_indexing": "Resolve positive and negative list indices",
}
PREREQUISITES = {"loop_tracing": "range_boundaries",
                 "accumulator_initialization": "loop_tracing"}
DIFFICULTIES = ("easy", "medium", "hard")
ACTION_NAMES = ("explain", "worked_example", "prerequisite_review",
                "self_explanation", "partial_scaffold")


def snippet(skill, n, level):
    """Return trusted, authored snippet and independently computed stdout."""
    start = n % 5 + 1
    count = 2 + level * 2 + n % 3
    if skill == "range_boundaries":
        step = 1 + level
        stop = start + count * step - (n % 2)
        values = list(range(start, stop, step))
        return f"print(list(range({start}, {stop}, {step})))", str(values)
    if skill == "loop_tracing":
        increment = level + 1
        code = f"x = {start}\nfor _ in range({count}):\n    x += {increment}\nprint(x)"
        return code, str(start + count * increment)
    if skill == "accumulator_initialization":
        initial = start * (level + 1)
        values = list(range(1, count + 1))
        code = f"total = {initial}\nfor value in {values}:\n    total += value\nprint(total)"
        return code, str(initial + sum(values))
    if skill == "conditional_evaluation":
        x = n % 11
        threshold = 3 + level
        condition = f"x > {threshold}" if level < 2 else f"x > {threshold} and x % 2 == 0"
        truth = x > threshold and (level < 2 or x % 2 == 0)
        return f"x = {x}\nif {condition}:\n    print('A')\nelse:\n    print('B')", "A" if truth else "B"
    if skill == "function_returns":
        factor = 2 + level
        return f"def f(x):\n    return x * {factor} + {start}\nprint(f({count}))", str(count * factor + start)
    values = [start + i * (level + 1) for i in range(5 + level)]
    index = n % len(values) if level == 0 else -(1 + n % len(values))
    return f"values = {values}\nprint(values[{index}])", str(values[index])


def make_catalog():
    tasks = []
    for skill in SKILLS:
        for level, difficulty in enumerate(DIFFICULTIES):
            for i in range(24):
                code_a, answer_a = snippet(skill, i, level)
                code_b, answer_b = snippet(skill, i + 7, level)
                tasks.append({
                    "task_id": f"{skill}.{difficulty}.{i:02d}", "skill_id": skill,
                    "version": "1", "pool": "practice" if i < 16 else "assessment",
                    "task_type": "paired_output_prediction", "difficulty": difficulty,
                    "difficulty_source": "authored_unvalidated", "reference_seconds": (30, 60, 90)[level],
                    "reference_source": "authored_unvalidated", "rubric_id": "two_required_outputs_v1",
                    "prompt": "State the printed output of each independent snippet.",
                    "snippets": [code_a, code_b], "reference_outputs": [answer_a, answer_b],
                    "criteria": [{"id": "output_a", "weight": 0.5, "required": True},
                                 {"id": "output_b", "weight": 0.5, "required": True}],
                })
    return tasks
