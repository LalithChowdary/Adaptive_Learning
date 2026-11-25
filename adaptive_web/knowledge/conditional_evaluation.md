# Conditional Evaluation (`if` / `elif` / `else` and Boolean Logic)

## Core Concepts and Mental Model
Python executes conditional statements by checking conditions sequentially and strictly separating **mutually exclusive chains** from **independent checks**:

1. **`if / elif / else` Chain (Mutually Exclusive)**: Conditions are tested top-to-bottom. The **first** condition evaluating to `True` executes its indented block, and **all remaining `elif`/`else` branches are skipped**, even if later conditions would also evaluate to `True`.
2. **Sequential `if / if` Statements (Independent)**: Multiple separate `if` statements without `elif` are each evaluated independently in sequence. More than one block can run!
3. **Boundary Inequalities**: Strictly distinguish `<` vs `<=` and `>` vs `>=`. At an exact threshold (`x == 10`), `x < 10` is `False` while `x <= 10` is `True`.
4. **Operator Precedence & Short-Circuiting**: Python evaluates `not` first, then `and`, then `or` (parentheses override all). `A and B` stops if `A` is falsy; `A or B` stops if `A` is truthy.

## Common Misconceptions and Pitfalls
- **Ordered threshold shadowing**: Writing `if score >= 60: grade = "D" elif score >= 90: grade = "A"` makes `"A"` unreachable because `95 >= 60` matches first! Always order strictest/highest bounds first or use explicit bounded intervals.
- **Confusing `elif` with stacked `if`**: Modifying a variable in the first `if` block CAN trigger a subsequent standalone `if` block on the exact same pass, whereas an `elif` branch would be skipped.
- **Off-by-one inequality comparisons**: Using `>` instead of `>=` misses exact equality boundaries; testing edge inputs (`threshold - 1`, `threshold`, `threshold + 1`) exposes boundary bugs immediately.
- **De Morgan's Law errors**: `not (a and b)` equals `(not a) or (not b)`, NOT `(not a) and (not b)`.

## Difficulty Progression and Worked Examples
- **Easy**: Single boundary comparison (`x >= 10`) or basic 2-branch `if/else` decision.
- **Medium**: Three-tier `if/elif/else` ladders inside a loop where inputs hit exact boundary values and test short-circuit ordering (`x != 0 and total // x > 2`).
- **Hard**: Nested conditionals combined with compound `and`/`or`/`not` expressions and sequential state-mutating `if` blocks inside loops.

## Prerequisite Connections
- **Depends on**: `loop_tracing` (tracking state mutations under conditional branches inside iteration).
- **Feeds into**: `function_returns` (early `return` inside conditional branches vs post-condition returns).
