# Accumulator Initialization and Reductions

## Core Concepts and Mental Model
The **accumulator pattern** maintains a running aggregate across a sequence by combining three mandatory parts:

1. **Pre-Loop Initialization**: Set the accumulator variable **once before the loop** to the mathematical identity element (or explicit starting baseline) of the operation:
   - Addition / sum / count -> `total = 0`
   - Multiplication / product -> `product = 1` (initializing to `0` collapses every product to `0`!)
   - String or list concatenation -> `out = ""` or `items = []`
   - Running maximum/minimum -> initialize to the **first element** (or explicit seed baseline), avoiding arbitrary constants like `0` when all inputs may be negative.
2. **Loop Traversal**: Visit each item cleanly using bounded iteration (`range(...)` or element iteration).
3. **In-Body Combination Update**: Combine the current value cleanly (`total = total + x` or `total += x`), never overwriting the accumulator inside the loop.

## Common Misconceptions and Pitfalls
- **Re-initializing inside the loop**: Putting `total = 0` inside the `for` body wipes out every prior step so the final answer only reflects the final element.
- **Wrong identity for products vs sums**: Using `0` to initialize a multiplicative accumulator produces `0` regardless of loop inputs; using `1` to initialize an additive accumulator adds a phantom `+1` offset unless explicitly requested as a baseline.
- **Stale pre-existing baseline**: Starting from a non-zero seed (`total = 10`) is valid when accumulating onto a baseline, so tracing must include the initial seed before iteration 0!
- **Uninitialized `NameError`**: Updating `total += val` without defining `total` beforehand raises `UnboundLocalError` / `NameError`.

## Difficulty Progression and Worked Examples
- **Easy**: Identify correct pre-loop initialization (`total = 0` for sums, `prod = 1` for products) and trace simple sums over `range(1, 5)`.
- **Medium**: Baseline-seeded accumulators (`score = 50` plus loop updates) and filtered accumulators (`if x % 2 == 0: total += x`).
- **Hard**: Alternating sign sums (`+= val` vs `-= val`), multi-accumulator reductions (tracking both `sum` and `count` or weighted products), and loop-invariant verification.

## Prerequisite Connections
- **Depends on**: `range_boundaries` and `loop_tracing`.
- **Feeds into**: `function_returns` (returning accumulated results cleanly from helper functions).
