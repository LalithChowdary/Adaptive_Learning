"""Read-only v2.2 catalogue with reviewed presentation hints and feature mapping."""
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data_Set' / 'v2.2'
spec = importlib.util.spec_from_file_location('dashboard_catalogue_source', DATA / 'catalog.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
SKILLS = {
    'range_boundaries': ('Range boundaries', 'Where sequences begin and end', '01'),
    'loop_tracing': ('Loop tracing', 'Follow a variable through each iteration', '02'),
    'accumulator_initialization': ('Accumulators', 'Build a total, one step at a time', '03'),
    'conditional_evaluation': ('Conditionals', 'Find the branch that runs', '04'),
    'function_returns': ('Function returns', 'Follow inputs to their returned values', '05'),
    'list_indexing': ('List indexing', 'Find an item by its position', '06'),
}
TASKS = {t['task_id']: t for t in json.loads((DATA / 'dataset/task_catalog.json').read_text())}
FORMATS = source.FORMATS
HINTS = {
 'range_boundaries': ['A range includes its start but excludes its stop.', 'Write the start, stop and step separately. With a positive step, keep only values below the stop.', 'Build a row of values: start, start + step, start + 2 × step. Stop before reaching or crossing the stop value.'],
 'loop_tracing': ['The loop body runs once for each item in the range.', 'Keep the starting value separate from the change made on each iteration.', 'Make a table with iteration 0 holding the starting value. For each following row, apply the update exactly once.'],
 'accumulator_initialization': ['The final total includes the initial value as well as the loop contributions.', 'Write down the initial total, then list the values added by the loop.', 'Use two columns: the next value to add and the running total. The first running total is the initialization, not necessarily zero.'],
 'conditional_evaluation': ['Only the branch whose condition is satisfied runs.', 'Substitute the given value into the comparison. A strict greater-than test is false at equality.', 'Write “condition: true or false” first. Then read only the corresponding branch; ignore the other branch.'],
 'function_returns': ['A function call substitutes its argument for the parameter.', 'Evaluate multiplication before addition in the return expression.', 'Rewrite the return expression with the input in place of x. Calculate the multiplication, then add the remaining value.'],
 'list_indexing': ['Positive indices start at zero. Negative indices count from the end.', 'For a positive index, label items 0, 1, 2… . For a negative index, label from the right −1, −2, −3… .', 'Write the indices under the list elements, using the direction of the requested index. Match the index to the item directly above it.'],
}

LESSONS = {
    'range_boundaries': {
        'concept': {
            'title': 'How Python range(start, stop, step) boundaries work',
            'summary': 'A Python range generates arithmetic values starting at start and stopping strictly before stop.',
            'rules': [
                'Inclusive start: the first generated number is always start (if startHasNotCrossedStop).',
                'Exclusive stop: stop is never included in the sequence, even if start + k * step == stop.',
                'Step size: each next term adds step (for positive step, keep values < stop; for negative step, keep values > stop).',
            ],
            'code': 'list(range(2, 9, 3))  # -> [2, 5, 8] (next would be 11 >= 9, so it stops)',
        },
        'worked_example': {
            'title': 'Worked Example: Tracing range(3, 11, 2)',
            'problem': 'Find every value produced by list(range(3, 11, 2)) and its length.',
            'code': 'values = list(range(3, 11, 2))\nprint(values[0], values[-1], len(values))',
            'steps': [
                'Identify start = 3, stop = 11 (excluded), step = +2.',
                'Term 0: start = 3 (3 < 11, include).',
                'Term 1: 3 + 2 = 5 (5 < 11, include).',
                'Term 2: 5 + 2 = 7 (7 < 11, include).',
                'Term 3: 7 + 2 = 9 (9 < 11, include). Next is 9 + 2 = 11, which reaches stop (11), so stop immediately.',
                'Result: [3, 5, 7, 9] -> first is 3, last is 9, length is 4.',
            ],
        },
        'self_explanation_prompt': 'Self-Explanation Check: Before submitting, explain to yourself why the stop boundary is excluded and verify whether your last generated value stays strictly inside the stop boundary.',
    },
    'loop_tracing': {
        'concept': {
            'title': 'Tracing state updates across loop iterations',
            'summary': 'A for-loop executes its body once per sequence item in order, updating state variables step by step.',
            'rules': [
                'Write down the initial variable value before iteration 1 begins.',
                'On each iteration, bind the loop variable to the next range element and evaluate the right-hand side using the current state.',
                'After the last element finishes, the variable holds the final updated state.',
            ],
            'code': 'x = 4\nfor step in range(1, 4):  # step = 1, 2, 3\n    x = x + 2 * step      # 4 -> 6 -> 10 -> 16',
        },
        'worked_example': {
            'title': 'Worked Example: Iteration table for a loop update',
            'problem': 'Trace the final value of value after the loop finishes.',
            'code': 'value = 5\nfor k in range(2, 7, 2):\n    value = value + k - 1\nprint(value)',
            'steps': [
                'Sequence from range(2, 7, 2) is [2, 4, 6]. Initial state: value = 5.',
                'Iteration 1 (k = 2): value = 5 + 2 - 1 = 6.',
                'Iteration 2 (k = 4): value = 6 + 4 - 1 = 9.',
                'Iteration 3 (k = 6): value = 9 + 6 - 1 = 14.',
                'Loop terminates; printed output is 14.',
            ],
        },
        'self_explanation_prompt': 'Self-Explanation Check: Trace each iteration on mental scratchpad from the initial state—verify both the number of iterations and the exact update rule on each pass.',
    },
    'accumulator_initialization': {
        'concept': {
            'title': 'Accumulator initialization and running totals',
            'summary': 'An accumulator combines a starting baseline with every contribution added inside the loop.',
            'rules': [
                'Never assume an accumulator starts at 0—always check its initialization line first.',
                'Total after loop = (Initial Value) + sum of each iteration contribution.',
                'When counting or summing filtered items, only add when the condition holds.',
            ],
            'code': 'total = 10\nfor n in [2, 5, 3]:\n    total += n  # 10 + 2 + 5 + 3 = 20',
        },
        'worked_example': {
            'title': 'Worked Example: Non-zero accumulator baseline',
            'problem': 'Compute total after accumulating transformed values.',
            'code': 'total = 3\nfor x in range(1, 4):\n    total = total + (2 * x)\nprint(total)',
            'steps': [
                'Baseline before loop: total = 3. The loop runs for x = 1, 2, 3.',
                'x = 1: add 2 * 1 = 2 -> running total = 3 + 2 = 5.',
                'x = 2: add 2 * 2 = 4 -> running total = 5 + 4 = 9.',
                'x = 3: add 2 * 3 = 6 -> running total = 9 + 6 = 15.',
                'Final printed total is 15 (3 initial + 12 from loop).',
            ],
        },
        'self_explanation_prompt': 'Self-Explanation Check: Explicitly separate the initial accumulator value from the loop contributions before answering unaided.',
    },
    'conditional_evaluation': {
        'concept': {
            'title': 'Mutual exclusion in if / elif / else chains',
            'summary': 'Python evaluates branch conditions top-to-bottom and executes only the first branch that evaluates to True.',
            'rules': [
                'Check boundary comparisons carefully: > and < are False at equality, whereas >= and <= are True.',
                'In an if / elif / else chain, as soon as one branch is True, all remaining elif/else branches are skipped.',
                'Two separate if statements (without elif) are independent and both must be checked.',
            ],
            'code': 'x = 10\nif x > 10:\n    out = "high"   # False (10 > 10 is False)\nelif x >= 10:\n    out = "edge"   # True -> runs this branch and skips else',
        },
        'worked_example': {
            'title': 'Worked Example: Boundary test in an if/elif/else chain',
            'problem': 'Determine which branch executes when score = 15.',
            'code': 'score = 15\nif score > 15:\n    tag = 1\nelif score >= 12:\n    tag = 2\nelse:\n    tag = 3\nprint(tag)',
            'steps': [
                'Substitute score = 15 into the first condition: 15 > 15 is False (strict inequality).',
                'Move to elif condition: 15 >= 12 is True.',
                'Execute tag = 2 and skip the else branch completely.',
                'Printed output is 2.',
            ],
        },
        'self_explanation_prompt': 'Self-Explanation Check: Verify whether the boundary comparison is strict (> or <) or inclusive (>= or <=), and confirm why the other branches do not overwrite the result.',
    },
    'function_returns': {
        'concept': {
            'title': 'Function parameters, operator precedence, and return values',
            'summary': 'Calling a function binds the argument to the parameter and immediately exits with the evaluated return expression.',
            'rules': [
                'Substitute the passed argument value for every occurrence of the parameter.',
                'Follow Python operator precedence: parentheses first, then multiplication/division/modulo, then addition/subtraction.',
                'A return statement exits the function immediately and hands that value back to the caller.',
            ],
            'code': 'def f(x):\n    return 3 * x + 4\n\nprint(f(5))  # 3 * 5 + 4 = 15 + 4 = 19',
        },
        'worked_example': {
            'title': 'Worked Example: Tracing a two-step function return',
            'problem': 'Evaluate compute(4) by tracing local variables to the return expression.',
            'code': 'def compute(x):\n    base = 2 * x - 1\n    return base + 5\n\nprint(compute(4))',
            'steps': [
                'Bind parameter x = 4.',
                'Evaluate base = 2 * 4 - 1 = 8 - 1 = 7.',
                'Evaluate return expression: base + 5 = 7 + 5 = 12.',
                'compute(4) returns 12.',
            ],
        },
        'self_explanation_prompt': 'Self-Explanation Check: Substitute the input argument step-by-step and verify operator precedence before submitting your unaided answer.',
    },
    'list_indexing': {
        'concept': {
            'title': 'Zero-based positive and negative list indexing',
            'summary': 'Python lists index elements from 0 on the left and from -1 on the right.',
            'rules': [
                'Positive indices: 0 is the first element, 1 is the second, and len(items) - 1 is the last.',
                'Negative indices: -1 is the last element, -2 is second-to-last, and -len(items) is the first.',
                'Nested or computed index items[i]: evaluate the index expression i first, then look up that position.',
            ],
            'code': 'items = [14, 28, 42, 56]\n# pos:    0   1   2   3\n# neg:   -4  -3  -2  -1\nprint(items[1], items[-2])  # -> 28 42',
        },
        'worked_example': {
            'title': 'Worked Example: Positive and negative index lookup',
            'problem': 'Find items[2] and items[-3] for a 5-element list.',
            'code': 'items = [10, 25, 40, 55, 70]\nprint(items[2], items[-3])',
            'steps': [
                'Label positions from the left (0-based): 0->10, 1->25, 2->40, 3->55, 4->70. So items[2] = 40.',
                'Label positions from the right: -1->70, -2->55, -3->40, -4->25, -5->10. So items[-3] = 40.',
                'Both expressions refer to the middle element 40.',
            ],
        },
        'self_explanation_prompt': 'Self-Explanation Check: Count carefully from 0 (left) or -1 (right) and confirm which element sits at the target index before answering unaided.',
    },
}

def features(task):
    return dict(difficulty=task['difficulty'], task_type=task['task_type'], reference_seconds=task['reference_seconds'],
        snippet_count=len(task['snippets']), option_count=len(task['options']), criterion_count=len(task['criteria']),
        credit_weights=sorted(c['weight'] for c in task['criteria'] if c['kind'] != 'incorrect_selection_penalty'),
        penalty_weights=sorted(c['weight'] for c in task['criteria'] if c['kind'] == 'incorrect_selection_penalty'))

def public_task(task, order):
    result = {k: copy.deepcopy(task[k]) for k in ('task_id','skill_id','task_type','difficulty','prompt','snippets','reference_seconds')}
    options = {o['id']: o for o in task['options']}
    result['options'] = [options[k] for k in order]
    result['criterion_count'] = len(task['criteria'])
    result['answer_count'] = len(task['answer_key']) if task['task_type'] in ('output_prediction','fill_in_the_blank') else 1
    return result

def hint(task, number):
    text = HINTS[task['skill_id']][number-1]
    if task['task_type'] == 'multiple_select': text += ' Check each example independently; more than one may be correct.'
    if task['task_type'] == 'code_completion' and number == 3: text += ' Put the result in a return statement inside solve(x).'
    return dict(level=number, kind=('conceptual','targeted','scaffold')[number-1], text=text, version='dashboard-hints-1')

def solution(task):
    if task['task_type'] == 'single_choice':
        return next(o['text'] for o in task['options'] if o['id'] == task['answer_key'])
    if task['task_type'] == 'multiple_select':
        return '\n\n'.join(o['text'] for o in task['options'] if o['id'] in task['answer_key'])
    return task['reference_solution'] or '\n'.join(f'{i+1}: {a}' for i,a in enumerate(task['answer_key']))
