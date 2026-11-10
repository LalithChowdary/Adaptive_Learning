# v2.1 learner-memory retraining

The v2.0 run fit on practice labels with gaps no longer than 3.06 days. It selected checkpoints using delayed assessments that extended well beyond that range. The first comparison remains a record of that extrapolation experiment. This run gives all five models supervised long-gap examples and a longer, scheduled training budget for the GRUs. [v2.1 dataset and audits](../data_Set/v2.1/README.md)

## Fitting and model selection

1. Load the audited v2.1 main training and validation projections. Use the same feature encoding and 32-unit recurrent architecture described in [the original methods](METHODS.md). The loader admits the new training probe labels only for v2.1. It checks that every probe reads the terminal practice prefix and that practice and delayed targets have valid phases.
2. Fit the three existing tabular models on the same 225,224 training targets as the GRUs. There are 198,651 ordinary or follow-up practice targets and 26,573 delayed probe targets. All fitting examples receive equal weight.
3. Train each recurrent architecture at seeds 20261110, 20261111, and 20261112. Batch 64 learner–skill sequences at a time. A full epoch has 150 optimizer steps. Keep AdamW, weight decay 0.001, clipping at 5, and a peak learning rate of 0.003. The OneCycle schedule steps after each optimizer step and spends its first 15% increasing the rate. Allow 40 epochs. After the first 30 epochs, stop if validation has failed to improve for eight epochs. Each run restores its best validation checkpoint. [PyTorch OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
4. Select the checkpoint using half practice log loss and half the unweighted mean of 1-, 7-, and 30-day validation log loss. Apply the same rule when choosing the tabular settings. Compare the recurrent architectures by mean validation score across seeds. If scores are within 0.001 of the minimum, use the simpler-model order recorded in the runner. For a winning recurrent architecture, save the median validation seed as the representative checkpoint.
5. After fitting and selection, score the main test split and both stress cohorts with the fixed checkpoints. Report log loss, Brier score, ten-bin calibration, and accuracy at a 0.5 cutoff. Report recurrent mean ± sample standard deviation over seeds. A paired learner bootstrap averages each query's loss over seeds before sampling learners. Its interval describes variation across learners for this set of seeds.

All recurrent fits use native MPS by default. The runner checks a GPU backward pass and refuses MPS training when CPU fallback is enabled. Scikit-learn baselines use CPU.

## Reporting references

**Constant.** Every held-out query receives the same probability, the success rate across eligible fitting labels. It provides a reference with no question or learner information.

**Simulator oracle.** After fitting and selection, an evaluation-only reader joins each scored task event to its first unaided response by explicit interaction and response identifiers. It then reads the private strength at the start of that attempt and the task's hidden demand. It computes the probability of all required criteria passing conditional on the shared Gaussian draw, then integrates over the draw using 32-point Gauss–Hermite quadrature. The oracle uses the simulator's distinct guessing rules for single choice, multiple select, and open response formats. This yields a privileged reference for the synthetic task. The reader accepts validation and test only. Private variables never enter the fitting loader or the checkpoint selection calculation.

The JSON report expresses the gain over the constant as a fraction of the constant-to-oracle log-loss gap. Its denominator and numerator both come from the same evaluation group. Finite-sample scores can put a model outside the 0–1 range. This fraction describes simulated signal, and the experiment does not estimate real learner performance.

## Reproduce the run

From `3/model` using the existing environment and an unused output directory:

```sh
unset PYTORCH_ENABLE_MPS_FALLBACK
MEMORY_TEST_DEVICE=mps .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_v21.py --mode smoke --device mps --output runs/v21-pilot-check
.venv/bin/python run_v21.py --device mps --output runs/v21-reproduction
```

The smoke command uses 12 training learners, 12 validation learners, and one epoch for one seed per GRU. The full run keeps separate checkpoints and learning curves for every seed. The JSON report identifies the selected checkpoint, stores complete metrics, and fingerprints the source and cohort manifests. The CSV exports store the probabilities for every seed and reference on matched evaluation queries. Dataset v2.0 still has its historical entry point `run.py`.

The two releases generate different learner histories and question outcomes. Model comparisons for this revision use the same v2.1 evaluation queries. The current experiment leaves hidden width at 32 and postpones the width comparison until after assessing the data and training changes.
