# Loop Tracing (`for` and `while` State Progression)

## Core Concepts and Mental Model
Tracing a loop means constructing a **step-by-step state table** tracking every mutable variable before iteration 1, at the end of every pass through the loop body, and immediately after loop termination.

1. **Iteration Variable Binding**: At the start of each `for` iteration, the loop variable is rebound to the next value from the iterable (`range(...)`, `list`, etc.).
2. **Sequential Statement Execution Inside Body**: Top-to-bottom order inside the loop body determines whether an update uses the **old** or **updated** value of another variable (`x += 1` before `y += x` differs from `y += x` before `x += 1`).
3. **`while` Loop Condition Check**: Checked before every pass; modifying state inside the body affects the next condition check, not the remainder of the current pass unless `break` or `continue` executes.
4. **Post-Loop State**: Variables modified inside the loop retain their final values after the loop exits; variables inside a zero-iteration loop retain their pre-loop values.

## Common Misconceptions and Pitfalls
- **Skipping the last or first state row**: Jumping directly from `i = 0` to `stop` causes off-by-one arithmetic mistakes. Always evaluate `stop` exclusivity before entering an iteration.
- **Swapped update order**: Swapping two interdependent updates inside a loop changes every subsequent iteration exponentially or linearly.
- **Variable shadowing / re-assignment**: Manually assigning to a `for` loop target (`i = i + 2`) inside the body only persists until the end of that single iteration; the next pass overwrites `i` from the iterator.
- **Misplaced `print()` indentation**: A `print()` indented inside the loop prints on every pass; unindented at the loop's baseline prints only once after completion.

## Difficulty Progression and Worked Examples
- **Easy**: Single variable increment inside `range(3)` with `print(count)` after loop. Trace `i=0 -> count=1`, `i=1 -> count=2`, `i=2 -> count=3`.
- **Medium**: Two coupled state variables updated inside strided loops or conditional branches (`if i % 2 == 0: x += i else: y += 1`).
- **Hard**: Nested loops (`outer` x `inner` state tables), dependent inner bounds (`range(i, 5)`), or loops combining early state mutations with arithmetic transforms.

## Prerequisite Connections
- **Depends on**: `range_boundaries` (you must know the exact sequence visited by `for` loops before you can trace body state accurately).
- **Feeds into**: `accumulator_initialization` and `conditional_evaluation`.
