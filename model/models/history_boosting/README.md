# History-aware histogram gradient boosting

This model uses scikit-learn's `HistGradientBoostingClassifier` with the same causal summary columns as the history-aware logistic regression. The trees can combine signals such as repeated difficulty, recent hints, and longer gaps through nonlinear splits. We use it as a stronger benchmark for the recurrent models.

The fit compares two tree sizes on the learner-level validation split. The dataset's missing numeric features are represented by values and presence bits in both tabular models. Training uses CPU and writes the chosen estimator and both validation trials under `runs/<run>/history_boosting/`. [Current fitting protocol](../../METHODS_V21.md)

See [tabular fitting](../../memory/tabular.py). [Estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
