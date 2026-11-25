# Function Returns (`def`, parameter binding, `return` vs `print`, scope)

## Core Concepts and Mental Model
A Python function encapsulates a reusable computation that accepts inputs via parameters and hands a value back to the caller via `return`:

1. **Parameter Binding at Call Time**: Arguments in the call (`f(3, 5)`) bind positionally to the function's local parameters (`def f(a, b):`) before the body runs.
2. **Immediate Exit on `return`**: Executing `return expr` evaluates `expr` (respecting operator precedence), hands that value back to the call site, and **immediately exits** the function — any lines below that `return` in the active path never run!
3. **`return` vs `print`**:
   - `print(x)` displays text to standard output and evaluates to `None`.
   - `return x` produces no console output on its own, instead substituting `x` where the function call occurred (`res = f(4)`).
   - A function ending without hitting any `return` implicitly returns `None`.
4. **Local Scope & Composition**: Variables created inside `def` remain local to that call frame; nested calls `g(f(x))` evaluate innermost `f(x)` first, passing its returned result as the argument to `g(...)`.

## Common Misconceptions and Pitfalls
- **Dead code after `return`**: Assuming statements after an unconditional `return` still execute (or inside a loop that returns on the very first iteration).
- **Confusing `print` output with returned value**: Writing `print(ans)` instead of `return ans` yields `None` when the caller captures `y = fn(x)` or uses `fn(x) + 1` (`TypeError`).
- **Missing fallback `return`**: An `if` branch with `return` without an `else` (or post-loop) `return` unexpectedly yields `None` when the condition is `False`.
- **Early loop return**: Placing `return total` indented **inside** a `for` loop exits after only pass `0`, rather than returning the full accumulated sum after the loop!

## Difficulty Progression and Worked Examples
- **Easy**: Bind arguments, evaluate an arithmetic return expression (`return 2 * x + 3`), and inspect `print(f(4))`.
- **Medium**: Multi-line helper with local intermediate variables and conditional early `return` branches (`if x < 0: return 0`).
- **Hard**: Loop-accumulating functions, nested/composed calls (`outer(inner(x))`), and subtle early-return vs post-loop indentation bugs.

## Prerequisite Connections
- **Depends on**: `conditional_evaluation` and `accumulator_initialization` (the capstone of this Python curriculum DAG, combining conditional branching and loop accumulation inside callable functions).
