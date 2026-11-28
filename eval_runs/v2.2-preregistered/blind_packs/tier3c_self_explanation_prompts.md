# Tier 3c Blinded Self-Explanation Authoring & Reference Pack (`N = 120`)

- **Prompt Pool:** All `18` production prompts from `se_prompt_bank` (`3` per skill × `6` skills; `7 + 7 + 6 = 20` response slots per skill; `rubric = None`).
- **Strict Blindness Rule:** Contributors authoring `learner_explanation` and reviewers assigning human reference scores MUST remain completely blind to automated grading rules and automated scores.
- **Quality-Band Authoring Targets (per skill):** `3` `empty_nonsense_or_code_only`, `5` `incorrect_or_misconception`, `6` `partial_coverage`, `6` `complete_causal`.
- **Human Reference Scale:** Continuous score on `[0.0, 1.0]` plus ordered 3-category label `{0: Incorrect/Absent, 1: Partial, 2: Full Conceptual Mastery}`.

## Production Prompt Bank (`18` Prompts)

### `range_boundaries`
- `se_range_boundaries_0`: Explain in your own words how start and stop boundaries work in Python range(start, stop).
- `se_range_boundaries_1`: Explain what happens when step is negative in range(start, stop, step) and how you find the last element.
- `se_range_boundaries_2`: Explain why range(n) produces n items and why stop is never included.

### `loop_tracing`
- `se_loop_tracing_0`: Explain in your own words how a loop iterates over a sequence.
- `se_loop_tracing_1`: Explain how to track a variable that is updated multiple times inside one loop iteration.
- `se_loop_tracing_2`: Explain what value a loop variable holds immediately after a for-loop finishes.

### `accumulator_initialization`
- `se_accumulator_initialization_0`: Explain in your own words how to initialize and update an accumulator variable.
- `se_accumulator_initialization_1`: Explain why initializing a product accumulator to 0 breaks multiplication loops.
- `se_accumulator_initialization_2`: Explain what goes wrong if an accumulator is initialized inside the loop body instead of before it.

### `conditional_evaluation`
- `se_conditional_evaluation_0`: Explain in your own words how an if/elif/else condition is evaluated.
- `se_conditional_evaluation_1`: Explain the difference between an if/elif/else chain and two separate if statements.
- `se_conditional_evaluation_2`: Explain why strict (<, >) vs inclusive (<=, >=) operators matter at exact boundary values.

### `list_indexing`
- `se_list_indexing_0`: Explain in your own words how zero-based list indexing works in Python.
- `se_list_indexing_1`: Explain how negative indices (-1, -2) map to elements in a list of length n.
- `se_list_indexing_2`: Explain why accessing lst[len(lst)] raises an IndexError.

### `function_returns`
- `se_function_returns_0`: Explain in your own words how a return statement exits a function and produces a value.
- `se_function_returns_1`: Explain the difference between print(x) and return x inside a Python function.
- `se_function_returns_2`: Explain how arguments bind to parameters and how operator precedence applies in return expressions.

## Response Authoring & Reference Slots (`120` items)

| Blind ID | Item ID | Prompt ID | Skill | Target Quality Band | Human Score `[0,1]` | Cat `(0/1/2)` |
|---|---|---|---|---|---|---|
| `T3C-001` | `t3c_range_boundaries_p0_s01` | `se_range_boundaries_0` | range_boundaries | incorrect_or_misconception | | |
| `T3C-002` | `t3c_range_boundaries_p0_s02` | `se_range_boundaries_0` | range_boundaries | partial_coverage | | |
| `T3C-003` | `t3c_range_boundaries_p0_s03` | `se_range_boundaries_0` | range_boundaries | partial_coverage | | |
| `T3C-004` | `t3c_range_boundaries_p0_s04` | `se_range_boundaries_0` | range_boundaries | partial_coverage | | |
| `T3C-005` | `t3c_range_boundaries_p0_s05` | `se_range_boundaries_0` | range_boundaries | partial_coverage | | |
| `T3C-006` | `t3c_range_boundaries_p0_s06` | `se_range_boundaries_0` | range_boundaries | complete_causal | | |
| `T3C-007` | `t3c_range_boundaries_p0_s07` | `se_range_boundaries_0` | range_boundaries | complete_causal | | |
| `T3C-008` | `t3c_range_boundaries_p1_s01` | `se_range_boundaries_1` | range_boundaries | complete_causal | | |
| `T3C-009` | `t3c_range_boundaries_p1_s02` | `se_range_boundaries_1` | range_boundaries | incorrect_or_misconception | | |
| `T3C-010` | `t3c_range_boundaries_p1_s03` | `se_range_boundaries_1` | range_boundaries | incorrect_or_misconception | | |
| `T3C-011` | `t3c_range_boundaries_p1_s04` | `se_range_boundaries_1` | range_boundaries | incorrect_or_misconception | | |
| `T3C-012` | `t3c_range_boundaries_p1_s05` | `se_range_boundaries_1` | range_boundaries | incorrect_or_misconception | | |
| `T3C-013` | `t3c_range_boundaries_p1_s06` | `se_range_boundaries_1` | range_boundaries | complete_causal | | |
| `T3C-014` | `t3c_range_boundaries_p1_s07` | `se_range_boundaries_1` | range_boundaries | empty_nonsense_or_code_only | | |
| `T3C-015` | `t3c_range_boundaries_p2_s01` | `se_range_boundaries_2` | range_boundaries | complete_causal | | |
| `T3C-016` | `t3c_range_boundaries_p2_s02` | `se_range_boundaries_2` | range_boundaries | partial_coverage | | |
| `T3C-017` | `t3c_range_boundaries_p2_s03` | `se_range_boundaries_2` | range_boundaries | empty_nonsense_or_code_only | | |
| `T3C-018` | `t3c_range_boundaries_p2_s04` | `se_range_boundaries_2` | range_boundaries | complete_causal | | |
| `T3C-019` | `t3c_range_boundaries_p2_s05` | `se_range_boundaries_2` | range_boundaries | partial_coverage | | |
| `T3C-020` | `t3c_range_boundaries_p2_s06` | `se_range_boundaries_2` | range_boundaries | empty_nonsense_or_code_only | | |
| `T3C-021` | `t3c_loop_tracing_p0_s01` | `se_loop_tracing_0` | loop_tracing | complete_causal | | |
| `T3C-022` | `t3c_loop_tracing_p0_s02` | `se_loop_tracing_0` | loop_tracing | partial_coverage | | |
| `T3C-023` | `t3c_loop_tracing_p0_s03` | `se_loop_tracing_0` | loop_tracing | complete_causal | | |
| `T3C-024` | `t3c_loop_tracing_p0_s04` | `se_loop_tracing_0` | loop_tracing | incorrect_or_misconception | | |
| `T3C-025` | `t3c_loop_tracing_p0_s05` | `se_loop_tracing_0` | loop_tracing | incorrect_or_misconception | | |
| `T3C-026` | `t3c_loop_tracing_p0_s06` | `se_loop_tracing_0` | loop_tracing | empty_nonsense_or_code_only | | |
| `T3C-027` | `t3c_loop_tracing_p0_s07` | `se_loop_tracing_0` | loop_tracing | partial_coverage | | |
| `T3C-028` | `t3c_loop_tracing_p1_s01` | `se_loop_tracing_1` | loop_tracing | empty_nonsense_or_code_only | | |
| `T3C-029` | `t3c_loop_tracing_p1_s02` | `se_loop_tracing_1` | loop_tracing | partial_coverage | | |
| `T3C-030` | `t3c_loop_tracing_p1_s03` | `se_loop_tracing_1` | loop_tracing | complete_causal | | |
| `T3C-031` | `t3c_loop_tracing_p1_s04` | `se_loop_tracing_1` | loop_tracing | incorrect_or_misconception | | |
| `T3C-032` | `t3c_loop_tracing_p1_s05` | `se_loop_tracing_1` | loop_tracing | partial_coverage | | |
| `T3C-033` | `t3c_loop_tracing_p1_s06` | `se_loop_tracing_1` | loop_tracing | incorrect_or_misconception | | |
| `T3C-034` | `t3c_loop_tracing_p1_s07` | `se_loop_tracing_1` | loop_tracing | incorrect_or_misconception | | |
| `T3C-035` | `t3c_loop_tracing_p2_s01` | `se_loop_tracing_2` | loop_tracing | complete_causal | | |
| `T3C-036` | `t3c_loop_tracing_p2_s02` | `se_loop_tracing_2` | loop_tracing | partial_coverage | | |
| `T3C-037` | `t3c_loop_tracing_p2_s03` | `se_loop_tracing_2` | loop_tracing | empty_nonsense_or_code_only | | |
| `T3C-038` | `t3c_loop_tracing_p2_s04` | `se_loop_tracing_2` | loop_tracing | complete_causal | | |
| `T3C-039` | `t3c_loop_tracing_p2_s05` | `se_loop_tracing_2` | loop_tracing | partial_coverage | | |
| `T3C-040` | `t3c_loop_tracing_p2_s06` | `se_loop_tracing_2` | loop_tracing | complete_causal | | |
| `T3C-041` | `t3c_accumulator_initialization_p0_s01` | `se_accumulator_initialization_0` | accumulator_initialization | incorrect_or_misconception | | |
| `T3C-042` | `t3c_accumulator_initialization_p0_s02` | `se_accumulator_initialization_0` | accumulator_initialization | partial_coverage | | |
| `T3C-043` | `t3c_accumulator_initialization_p0_s03` | `se_accumulator_initialization_0` | accumulator_initialization | complete_causal | | |
| `T3C-044` | `t3c_accumulator_initialization_p0_s04` | `se_accumulator_initialization_0` | accumulator_initialization | incorrect_or_misconception | | |
| `T3C-045` | `t3c_accumulator_initialization_p0_s05` | `se_accumulator_initialization_0` | accumulator_initialization | incorrect_or_misconception | | |
| `T3C-046` | `t3c_accumulator_initialization_p0_s06` | `se_accumulator_initialization_0` | accumulator_initialization | partial_coverage | | |
| `T3C-047` | `t3c_accumulator_initialization_p0_s07` | `se_accumulator_initialization_0` | accumulator_initialization | partial_coverage | | |
| `T3C-048` | `t3c_accumulator_initialization_p1_s01` | `se_accumulator_initialization_1` | accumulator_initialization | empty_nonsense_or_code_only | | |
| `T3C-049` | `t3c_accumulator_initialization_p1_s02` | `se_accumulator_initialization_1` | accumulator_initialization | complete_causal | | |
| `T3C-050` | `t3c_accumulator_initialization_p1_s03` | `se_accumulator_initialization_1` | accumulator_initialization | empty_nonsense_or_code_only | | |
| `T3C-051` | `t3c_accumulator_initialization_p1_s04` | `se_accumulator_initialization_1` | accumulator_initialization | complete_causal | | |
| `T3C-052` | `t3c_accumulator_initialization_p1_s05` | `se_accumulator_initialization_1` | accumulator_initialization | partial_coverage | | |
| `T3C-053` | `t3c_accumulator_initialization_p1_s06` | `se_accumulator_initialization_1` | accumulator_initialization | partial_coverage | | |
| `T3C-054` | `t3c_accumulator_initialization_p1_s07` | `se_accumulator_initialization_1` | accumulator_initialization | incorrect_or_misconception | | |
| `T3C-055` | `t3c_accumulator_initialization_p2_s01` | `se_accumulator_initialization_2` | accumulator_initialization | empty_nonsense_or_code_only | | |
| `T3C-056` | `t3c_accumulator_initialization_p2_s02` | `se_accumulator_initialization_2` | accumulator_initialization | complete_causal | | |
| `T3C-057` | `t3c_accumulator_initialization_p2_s03` | `se_accumulator_initialization_2` | accumulator_initialization | incorrect_or_misconception | | |
| `T3C-058` | `t3c_accumulator_initialization_p2_s04` | `se_accumulator_initialization_2` | accumulator_initialization | partial_coverage | | |
| `T3C-059` | `t3c_accumulator_initialization_p2_s05` | `se_accumulator_initialization_2` | accumulator_initialization | complete_causal | | |
| `T3C-060` | `t3c_accumulator_initialization_p2_s06` | `se_accumulator_initialization_2` | accumulator_initialization | complete_causal | | |
| `T3C-061` | `t3c_conditional_evaluation_p0_s01` | `se_conditional_evaluation_0` | conditional_evaluation | empty_nonsense_or_code_only | | |
| `T3C-062` | `t3c_conditional_evaluation_p0_s02` | `se_conditional_evaluation_0` | conditional_evaluation | partial_coverage | | |
| `T3C-063` | `t3c_conditional_evaluation_p0_s03` | `se_conditional_evaluation_0` | conditional_evaluation | partial_coverage | | |
| `T3C-064` | `t3c_conditional_evaluation_p0_s04` | `se_conditional_evaluation_0` | conditional_evaluation | incorrect_or_misconception | | |
| `T3C-065` | `t3c_conditional_evaluation_p0_s05` | `se_conditional_evaluation_0` | conditional_evaluation | partial_coverage | | |
| `T3C-066` | `t3c_conditional_evaluation_p0_s06` | `se_conditional_evaluation_0` | conditional_evaluation | empty_nonsense_or_code_only | | |
| `T3C-067` | `t3c_conditional_evaluation_p0_s07` | `se_conditional_evaluation_0` | conditional_evaluation | incorrect_or_misconception | | |
| `T3C-068` | `t3c_conditional_evaluation_p1_s01` | `se_conditional_evaluation_1` | conditional_evaluation | complete_causal | | |
| `T3C-069` | `t3c_conditional_evaluation_p1_s02` | `se_conditional_evaluation_1` | conditional_evaluation | incorrect_or_misconception | | |
| `T3C-070` | `t3c_conditional_evaluation_p1_s03` | `se_conditional_evaluation_1` | conditional_evaluation | partial_coverage | | |
| `T3C-071` | `t3c_conditional_evaluation_p1_s04` | `se_conditional_evaluation_1` | conditional_evaluation | partial_coverage | | |
| `T3C-072` | `t3c_conditional_evaluation_p1_s05` | `se_conditional_evaluation_1` | conditional_evaluation | complete_causal | | |
| `T3C-073` | `t3c_conditional_evaluation_p1_s06` | `se_conditional_evaluation_1` | conditional_evaluation | partial_coverage | | |
| `T3C-074` | `t3c_conditional_evaluation_p1_s07` | `se_conditional_evaluation_1` | conditional_evaluation | complete_causal | | |
| `T3C-075` | `t3c_conditional_evaluation_p2_s01` | `se_conditional_evaluation_2` | conditional_evaluation | complete_causal | | |
| `T3C-076` | `t3c_conditional_evaluation_p2_s02` | `se_conditional_evaluation_2` | conditional_evaluation | empty_nonsense_or_code_only | | |
| `T3C-077` | `t3c_conditional_evaluation_p2_s03` | `se_conditional_evaluation_2` | conditional_evaluation | incorrect_or_misconception | | |
| `T3C-078` | `t3c_conditional_evaluation_p2_s04` | `se_conditional_evaluation_2` | conditional_evaluation | complete_causal | | |
| `T3C-079` | `t3c_conditional_evaluation_p2_s05` | `se_conditional_evaluation_2` | conditional_evaluation | complete_causal | | |
| `T3C-080` | `t3c_conditional_evaluation_p2_s06` | `se_conditional_evaluation_2` | conditional_evaluation | incorrect_or_misconception | | |
| `T3C-081` | `t3c_list_indexing_p0_s01` | `se_list_indexing_0` | list_indexing | partial_coverage | | |
| `T3C-082` | `t3c_list_indexing_p0_s02` | `se_list_indexing_0` | list_indexing | incorrect_or_misconception | | |
| `T3C-083` | `t3c_list_indexing_p0_s03` | `se_list_indexing_0` | list_indexing | incorrect_or_misconception | | |
| `T3C-084` | `t3c_list_indexing_p0_s04` | `se_list_indexing_0` | list_indexing | partial_coverage | | |
| `T3C-085` | `t3c_list_indexing_p0_s05` | `se_list_indexing_0` | list_indexing | empty_nonsense_or_code_only | | |
| `T3C-086` | `t3c_list_indexing_p0_s06` | `se_list_indexing_0` | list_indexing | partial_coverage | | |
| `T3C-087` | `t3c_list_indexing_p0_s07` | `se_list_indexing_0` | list_indexing | partial_coverage | | |
| `T3C-088` | `t3c_list_indexing_p1_s01` | `se_list_indexing_1` | list_indexing | partial_coverage | | |
| `T3C-089` | `t3c_list_indexing_p1_s02` | `se_list_indexing_1` | list_indexing | partial_coverage | | |
| `T3C-090` | `t3c_list_indexing_p1_s03` | `se_list_indexing_1` | list_indexing | complete_causal | | |
| `T3C-091` | `t3c_list_indexing_p1_s04` | `se_list_indexing_1` | list_indexing | complete_causal | | |
| `T3C-092` | `t3c_list_indexing_p1_s05` | `se_list_indexing_1` | list_indexing | empty_nonsense_or_code_only | | |
| `T3C-093` | `t3c_list_indexing_p1_s06` | `se_list_indexing_1` | list_indexing | incorrect_or_misconception | | |
| `T3C-094` | `t3c_list_indexing_p1_s07` | `se_list_indexing_1` | list_indexing | complete_causal | | |
| `T3C-095` | `t3c_list_indexing_p2_s01` | `se_list_indexing_2` | list_indexing | complete_causal | | |
| `T3C-096` | `t3c_list_indexing_p2_s02` | `se_list_indexing_2` | list_indexing | complete_causal | | |
| `T3C-097` | `t3c_list_indexing_p2_s03` | `se_list_indexing_2` | list_indexing | incorrect_or_misconception | | |
| `T3C-098` | `t3c_list_indexing_p2_s04` | `se_list_indexing_2` | list_indexing | complete_causal | | |
| `T3C-099` | `t3c_list_indexing_p2_s05` | `se_list_indexing_2` | list_indexing | empty_nonsense_or_code_only | | |
| `T3C-100` | `t3c_list_indexing_p2_s06` | `se_list_indexing_2` | list_indexing | incorrect_or_misconception | | |
| `T3C-101` | `t3c_function_returns_p0_s01` | `se_function_returns_0` | function_returns | incorrect_or_misconception | | |
| `T3C-102` | `t3c_function_returns_p0_s02` | `se_function_returns_0` | function_returns | incorrect_or_misconception | | |
| `T3C-103` | `t3c_function_returns_p0_s03` | `se_function_returns_0` | function_returns | empty_nonsense_or_code_only | | |
| `T3C-104` | `t3c_function_returns_p0_s04` | `se_function_returns_0` | function_returns | empty_nonsense_or_code_only | | |
| `T3C-105` | `t3c_function_returns_p0_s05` | `se_function_returns_0` | function_returns | incorrect_or_misconception | | |
| `T3C-106` | `t3c_function_returns_p0_s06` | `se_function_returns_0` | function_returns | complete_causal | | |
| `T3C-107` | `t3c_function_returns_p0_s07` | `se_function_returns_0` | function_returns | partial_coverage | | |
| `T3C-108` | `t3c_function_returns_p1_s01` | `se_function_returns_1` | function_returns | partial_coverage | | |
| `T3C-109` | `t3c_function_returns_p1_s02` | `se_function_returns_1` | function_returns | incorrect_or_misconception | | |
| `T3C-110` | `t3c_function_returns_p1_s03` | `se_function_returns_1` | function_returns | complete_causal | | |
| `T3C-111` | `t3c_function_returns_p1_s04` | `se_function_returns_1` | function_returns | partial_coverage | | |
| `T3C-112` | `t3c_function_returns_p1_s05` | `se_function_returns_1` | function_returns | partial_coverage | | |
| `T3C-113` | `t3c_function_returns_p1_s06` | `se_function_returns_1` | function_returns | partial_coverage | | |
| `T3C-114` | `t3c_function_returns_p1_s07` | `se_function_returns_1` | function_returns | complete_causal | | |
| `T3C-115` | `t3c_function_returns_p2_s01` | `se_function_returns_2` | function_returns | partial_coverage | | |
| `T3C-116` | `t3c_function_returns_p2_s02` | `se_function_returns_2` | function_returns | complete_causal | | |
| `T3C-117` | `t3c_function_returns_p2_s03` | `se_function_returns_2` | function_returns | incorrect_or_misconception | | |
| `T3C-118` | `t3c_function_returns_p2_s04` | `se_function_returns_2` | function_returns | complete_causal | | |
| `T3C-119` | `t3c_function_returns_p2_s05` | `se_function_returns_2` | function_returns | complete_causal | | |
| `T3C-120` | `t3c_function_returns_p2_s06` | `se_function_returns_2` | function_returns | empty_nonsense_or_code_only | | |
