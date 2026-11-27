# Range Boundaries (`range(start, stop, step)`)

## Core Concepts and Mental Model
Python's built-in `range()` function generates integer sequences lazily using half-open interval semantics: `[start, stop)`.

1. **One-argument form `range(stop)`**: Starts at `0`, increments by `1`, and stops strictly **before** `stop`.
   - `range(5)` yields `0, 1, 2, 3, 4` (5 elements total; `5` is never produced).
   - If `stop <= 0`, `range(stop)` produces an empty sequence.
2. **Two-argument form `range(start, stop)`**: Starts inclusively at `start` and stops strictly **before** `stop` (`start, start+1, ..., stop-1`).
   - Count of items when `start <= stop`: `stop - start`.
3. **Three-argument form `range(start, stop, step)`**: Moves by `step` from `start` while strictly within bounds:
   - If `step > 0`: stops before `x >= stop` (yields `x < stop`).
   - If `step < 0`: counts **down**, stopping before `x <= stop` (yields `x > stop`).
   - `step == 0` raises `ValueError`.

## Common Misconceptions and Pitfalls
- **Off-by-one inclusive stop error**: Learners often assume `range(1, 5)` includes `5`. In Python, the `stop` boundary is always **exclusive**. To include `5`, write `range(1, 6)`.
- **Negative step stopping direction**: In `range(10, 2, -2)`, `2` is excluded, yielding `10, 8, 6, 4`. Conversely, `range(2, 10, -1)` is empty because moving by `-1` from `2` can never head toward `10`.
- **Empty range traps**: `range(5, 5)` and `range(5, 2)` (with default positive step `+1`) produce zero iterations without throwing an error.
- **Last emitted element vs `stop`**: For `range(1, 10, 3)`, elements are `1, 4, 7`. Adding `3` gives `10`, which equals `stop`, so iteration stops cleanly at `7`.

## Difficulty Progression and Worked Examples
- **Easy**: `list(range(4))` -> `[0, 1, 2, 3]`; `list(range(2, 6))` -> `[2, 3, 4, 5]`. Focus on inclusive `start` vs exclusive `stop`.
- **Medium**: Strided sequences like `range(3, 14, 3)` (`3, 6, 9, 12`) and countdowns like `range(8, 2, -2)` (`8, 6, 4`).
- **Hard**: Nested boundaries (`for i in range(n): for j in range(i + 1, n):`), empty intervals masked by negative steps, and off-by-one parity adjustments when `stop - start` is or is not divisible by `step`.

## Prerequisite Connections
- **Root Skill**: `range_boundaries` has no prerequisites and directly underpins `loop_tracing`, `accumulator_initialization`, and `list_indexing`. Mastery requires predicting both the first yielded integer and the exact final integer before termination.
