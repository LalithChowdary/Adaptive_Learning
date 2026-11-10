"""Read-only v2.1 catalogue with reviewed presentation hints and feature mapping."""
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data_Set' / 'v2.1'
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
