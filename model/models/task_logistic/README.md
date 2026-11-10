# Task-only logistic regression

This baseline estimates full first-unaided success from the question's format, difficulty, reference time, and rubric structure. It establishes how much the task itself tells us and gives a reference for learners with little history.

It uses regularized logistic regression with training-fitted scaling. The v2.1 runner compares two regularization strengths on the shared practice-and-delayed validation score. Training uses scikit-learn on CPU. The fit and chosen setting are written under `runs/<run>/task_logistic/`.

The task-only and history-aware logistic implementations share [the fitting code](../../memory/tabular.py) and [task encoding](../../memory/features.py). Their comparison measures the added value of the history summaries.
