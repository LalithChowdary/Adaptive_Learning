# GRU with elapsed-time inputs

This comparator processes the ordered hybrid stream of graded attempts, delivered hints, reveals, and instruction. It receives `log1p(elapsed_days)` alongside each evidence token and again alongside the target task features at prediction time.

We use this model to compare elapsed time as a feature against a learned decay gate. Both GRUs share the tokenization, query boundaries, state width, head width, training split, and checkpoint selection rule.

The default runner fits on Apple GPU through PyTorch MPS. The v2.1 runner saves one checkpoint and learning curve per seed under `runs/<run>/time_input_gru/seed_<seed>/`. See [recurrent implementation](../../memory/recurrent.py), [batching and training](../../memory/training.py), and the [v2.1 protocol](../../METHODS_V21.md).
