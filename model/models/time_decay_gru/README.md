# Time-decayed GRU

This is the main sequence candidate. Each learner and skill has a compact state with an evidence timestamp. Before the next evidence update, the model moves the state toward its shared learned cold-start prior. The rate is positive and can differ by state dimension.

```text
rate = softplus(rate_layer(saved_state))
ready_state = prior + exp(-rate * elapsed_days) * (saved_state - prior)
```

The GRU cell uses the ready state when processing a token. For a prediction, a small head uses a temporary ready state and the next task's features. The original state and timestamp stay intact during prediction, so repeat queries do not advance the clock. A delivered hint can be the last encoded event, in which case the next gap starts at that hint.

The rate weights start at zero with rate biases corresponding to initial half-lives spaced between 0.5 and 120 days. The initialization provides a range of time scales. Training learns the rates from the next-answer prediction loss. The architecture follows the idea of trainable hidden-state decay from GRU-D. [Che et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5904216/)

Training uses MPS by default. The v2.1 runner saves one checkpoint and learning curve per seed under `runs/<run>/time_decay_gru/seed_<seed>/`. See [recurrent implementation](../../memory/recurrent.py), [batching and training](../../memory/training.py), and the [v2.1 protocol](../../METHODS_V21.md).
