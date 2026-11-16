"""Versioned authored tasks; text and answer keys are outside memory features."""
import re

SKILLS = {
    "range_boundaries": "Exclusive upper bound of range",
    "loop_tracing": "Trace repeated updates in a loop",
    "accumulator_initialization": "Account for an accumulator's starting value",
    "conditional_evaluation": "Choose the branch of a conditional",
    "function_returns": "Evaluate a function's returned value",
    "list_indexing": "Resolve positive and negative list indices",
}
FORMATS = ("single_choice", "multiple_select", "output_prediction", "fill_in_the_blank", "code_completion")
DIFFICULTIES = ("easy", "medium", "hard")
FORMAT_RULES = {
    "single_choice": dict(base_seconds=25, help_multiplier=.75, practice_dose=.30, guess=.25),
    "multiple_select": dict(base_seconds=45, help_multiplier=.9, practice_dose=.40, guess=.5),
    "output_prediction": dict(base_seconds=35, help_multiplier=1., practice_dose=.45, guess=.02),
    "fill_in_the_blank": dict(base_seconds=40, help_multiplier=1.1, practice_dose=.45, guess=.04),
    "code_completion": dict(base_seconds=75, help_multiplier=1.25, practice_dose=.55, guess=.01),
}


LESSONS = {
    "python_intro_v2/range_boundaries": "range(start, stop, step) begins at start and excludes stop. With a positive step, continue while the value is below stop. For example, list(range(1, 5, 2)) is [1, 3].",
    "python_intro_v2/loop_tracing": "Record the initial variable value, then apply the loop body once per iteration. Starting x at 2 and adding 3 twice gives 2, then 5, then 8.",
    "python_intro_v2/accumulator_initialization": "An accumulator starts with its initialized value. Add each contribution to that value. Starting a total at 5 and adding 1 and 2 produces 8.",
    "python_intro_v2/conditional_evaluation": "Evaluate the condition first. Execute the if branch when it is true and the else branch when it is false. A strict greater-than comparison is false at equality.",
    "python_intro_v2/function_returns": "Replace each function parameter with its supplied argument and evaluate the return expression. A function returning x * 2 + 1 returns 7 when x is 3.",
    "python_intro_v2/list_indexing": "Python lists use index 0 for the first element. Negative indices count from the end: -1 is the last element and -2 is the element before it.",
}


def case(skill, n, level):
    """Trusted reference examples and independently calculated outputs."""
    a, b, step = n + 2, 2 + n % 5 + level, 1 + level
    if skill == "range_boundaries":
        stop = a + b * step
        return f"print(list(range({a}, {stop}, {step})))", str(list(range(a, stop, step))), f"A Python range starts at {a}, stops before {stop}, and steps by {step}. What list does it produce?"
    if skill == "loop_tracing":
        return f"x = {a}\nfor _ in range({b}):\n    x += {step}\nprint(x)", str(a+b*step), f"A variable starts at {a} and is increased by {step} on each of {b} loop iterations. What is its final value?"
    if skill == "accumulator_initialization":
        return f"total = {a}\nfor value in range(1, {b+1}):\n    total += value\nprint(total)", str(a+b*(b+1)//2), f"An accumulator starts at {a}. A loop adds every integer from 1 through {b}, inclusive. What is the final total?"
    if skill == "conditional_evaluation":
        threshold = a + (1 if n % 2 else -1)
        return f"x = {a}\nif x > {threshold}:\n    print({b})\nelse:\n    print({b+step})", str(b if a > threshold else b+step), f"If {a} is greater than {threshold}, a program returns {b}; otherwise it returns {b+step}. What does it return?"
    if skill == "function_returns":
        return f"def f(x):\n    return x * {step+1} + {a}\nprint(f({b}))", str(b*(step+1)+a), f"A function returns its input multiplied by {step+1}, then increased by {a}. What does it return for input {b}?"
    values = [a+i*step for i in range(5+level)]
    index = n % len(values) if level == 0 else -(1+n % len(values))
    return f"values = {values}\nprint(values[{index}])", str(values[index]), f"A Python list contains {values}. What value is at index {index}?"


def wrong_answer(answer, offset):
    if answer.startswith("["):
        import ast
        return str([x + offset for x in ast.literal_eval(answer)])
    return str(int(answer) + offset)


def completion(skill, n, level):
    a, step = n + 2, level + 1
    if skill == "range_boundaries":
        body, desc = f"return list(range({a}, x, {step}))", f"Return a list starting at {a}, stepping by {step}, and stopping before x."
        inputs = [a, a+1, a+4*step, a-1]
        outputs = [list(range(a,x,step)) for x in inputs]
    elif skill == "loop_tracing":
        body, desc = f"value = {a}\nfor _ in range(x):\n    value += {step}\nreturn value", f"Start at {a}, perform x increments of {step}, and return the final value. x is nonnegative."
        inputs = [0, 1, 3, 5]; outputs = [a+x*step for x in inputs]
    elif skill == "accumulator_initialization":
        body, desc = f"total = {a}\nfor value in x:\n    total += value\nreturn total", f"Start a total at {a}, add each number in list x, and return it."
        inputs = [[], [2], [1,2,3], [-2,4]]; outputs = [a+sum(x) for x in inputs]
    elif skill == "conditional_evaluation":
        body, desc = f"if x > {a}:\n    return {step}\nreturn {-step}", f"Return {step} if x is greater than {a}; otherwise return {-step}."
        inputs = [a+1, a, a-1, a+5]; outputs = [step if x>a else -step for x in inputs]
    elif skill == "function_returns":
        body, desc = f"return x * {step+1} + {a}", f"Return x multiplied by {step+1}, plus {a}."
        inputs = [0, 1, -1, 3]; outputs = [x*(step+1)+a for x in inputs]
    else:
        body, desc = f"return x[{-step}]", f"Return the item at index {-step} of list x. x always contains at least {step} items."
        inputs = [[a+i for i in range(5)], [9,7,5,3], [1,1,2,2,3], [-1,0,1,2]]
        outputs = [x[-step] for x in inputs]
    solution = "def solve(x):\n" + "\n".join("    "+line for line in body.splitlines())
    return desc, solution, [dict(input=x, expected=y) for x,y in zip(inputs, outputs)]


def make_catalog():
    tasks = []
    for skill in SKILLS:
        for level, difficulty in enumerate(DIFFICULTIES):
            for fmt in FORMATS:
                for i in range(24):
                    # Disjoint numeric variants in assessment; families intentionally shared.
                    n = i + 30*FORMATS.index(fmt) + 150*level
                    code, answer, description = case(skill, n, level)
                    task = dict(task_id=f"{skill}.{difficulty}.{fmt}.{i:02d}", skill_id=skill,
                                version="2", pool="practice" if i < 16 else "assessment",
                                task_type=fmt, difficulty=difficulty, difficulty_source="authored_unvalidated",
                                reference_source="authored_unvalidated", prompt=description,
                                snippets=[], options=[], answer_key=None, reference_solution=None, tests=[])
                    if fmt == "single_choice":
                        if i % 2:
                            task.update(prompt="Choose the printed output.", snippets=[code])
                        opts = [answer] + [wrong_answer(answer,j) for j in (-1,1,2)]
                        rotation = i % 4; opts = opts[rotation:] + opts[:rotation]
                        task["options"] = [dict(id=f"o{j}", text=v) for j,v in enumerate(opts)]
                        task["answer_key"] = f"o{opts.index(answer)}"
                        weights = [1.]
                    elif fmt == "multiple_select":
                        count = 4 + i % 3
                        truth = [True, False, True, False, True, False][:count]
                        # Each option names a separate concrete example; exact set is required.
                        for j, correct in enumerate(truth):
                            c, ans, desc = case(skill, n*10+j, level)
                            text = desc + " Proposed result: " + (ans if correct else wrong_answer(ans,1))
                            task["options"].append(dict(id=f"o{j}", text=text))
                        task["prompt"] = "Select every example whose proposed result is correct. Incorrect selections lose credit."
                        task["answer_key"] = [f"o{j}" for j,t in enumerate(truth) if t]
                        positives = [j for j,t in enumerate(truth) if t]; negatives = [j for j,t in enumerate(truth) if not t]
                        denom = sum(range(1,len(positives)+1))
                        weights = [(positives.index(j)+1)/denom if t else 1/len(negatives) for j,t in enumerate(truth)]
                    elif fmt == "output_prediction":
                        pairs = [case(skill, n*10+j, level) for j in range(1+i%3)]
                        task.update(prompt="Give the printed output of each independent snippet.", snippets=[c for c,a,d in pairs], answer_key=[a for c,a,d in pairs])
                        weights = list(range(1,len(pairs)+1))
                    elif fmt == "fill_in_the_blank":
                        matches = list(re.finditer(r"(?<![A-Za-z_])\d+", code))[:1+i%3]
                        parts, keys, prev = [], [], 0
                        for j,m in enumerate(matches):
                            parts += [code[prev:m.start()], f"__BLANK_{j+1}__"]
                            keys.append(m.group()); prev=m.end()
                        parts.append(code[prev:])
                        task.update(prompt="Restore the missing integer literals so the code implements this specification: "+description,
                                    snippets=["".join(parts)], answer_key=keys, reference_solution=code)
                        weights = list(range(1,len(keys)+1))
                    else:
                        desc, solution, tests = completion(skill,n,level)
                        task.update(prompt="Complete solve(x). "+desc, snippets=["def solve(x):\n    # complete the body\n    pass"],
                                    reference_solution=solution, tests=tests[:2+i%3], answer_key=None)
                        weights = list(range(1,len(task["tests"])+1))
                    total = sum(weights)
                    task["criteria"] = [dict(id=f"c{j}", weight=(w if fmt=="multiple_select" else w/total), required=True,
                                             kind=("correct_selection" if f"o{j}" in task["answer_key"] else "incorrect_selection_penalty") if fmt=="multiple_select" else ("behavior_test" if fmt=="code_completion" else "exact_match"))
                                        for j,w in enumerate(weights)]
                    task["rubric_id"] = task["task_id"]+".rubric_v2"
                    size = len(weights)
                    task["reference_seconds"] = round(FORMAT_RULES[fmt]["base_seconds"]*(1+.4*level)*(1+.18*(size-1)), 2)
                    task["hints"] = [dict(level=j+1, kind=("conceptual","targeted","scaffold","scaffold")[min(j,3)],
                                         text=hint) for j,hint in enumerate([
                        "Identify the starting values and the exact operation requested.",
                        {"range_boundaries":"The stop value is excluded. Check the step.","loop_tracing":"Track the variable once per iteration.","accumulator_initialization":"Include the initial total before adding values.","conditional_evaluation":"Evaluate the condition before choosing a branch.","function_returns":"Substitute the argument into the return expression.","list_indexing":"Index zero is first; index minus one is last."}[skill],
                        "Make a small trace table with the starting state and each change; for selections check each option separately.",
                        "Check the boundary case first, then trace the remaining case one operation at a time."])]
                    tasks.append(task)
    return tasks


def grade(task, evidence):
    """Pure simulated adapter. Evidence is option ID(s) or per-criterion pass flags."""
    fmt = task["task_type"]
    if fmt == "single_choice":
        if evidence not in [o["id"] for o in task["options"]]: raise ValueError("invalid option")
        flags = [evidence == task["answer_key"]]; score = float(flags[0]); full = flags[0]
    elif fmt == "multiple_select":
        if not isinstance(evidence,list) or len(evidence)!=len(set(evidence)) or not set(evidence)<={o["id"] for o in task["options"]}: raise ValueError("invalid selection")
        selected, correct = set(evidence), set(task["answer_key"])
        flags = [(o["id"] in selected)==(o["id"] in correct) for o in task["options"]]
        score = max(0., min(1., sum(c["weight"]*(1 if o["id"] in correct else -1) for o,c in zip(task["options"],task["criteria"]) if o["id"] in selected)))
        full = selected == correct
    else:
        if not isinstance(evidence,list) or len(evidence)!=len(task["criteria"]) or any(type(x) is not bool for x in evidence): raise ValueError("invalid criterion results")
        flags = evidence[:]; score = sum(c["weight"] for c,passed in zip(task["criteria"],flags) if passed); full = all(flags)
    return dict(criteria=flags, score=round(score,10), full_success=full)
