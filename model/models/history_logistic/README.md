# History-aware logistic regression

This baseline adds causal same-skill summaries to the question features. The summaries cover past scores and full successes, assisted versus unaided evidence, hints, retries, response durations, and elapsed gaps. A presence bit distinguishes an unavailable measure from a real zero.

We use it to measure how much learner history adds under a simple linear model. The learner's past responses enter through fixed summaries and logistic regression estimates their relationship with the next outcome. It fits on CPU and saves the estimator and the validation trials under `runs/<run>/history_logistic/`.

The history summary is shared with the boosted-tree comparator. See [feature construction](../../memory/features.py) and [tabular fitting](../../memory/tabular.py).

In v2.1, the same feature definitions apply to spaced practice and delayed fitting queries. [Current fitting protocol](../../METHODS_V21.md)
