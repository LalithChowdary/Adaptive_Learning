# Learner-memory models

This experiment compares seven ways to predict full success on a new question's first unaided attempt, evaluated on both predictive accuracy and a 6-check behavioral sanity suite. The active protocol uses the audited [v2.2 dataset](../data_Set/v2.2/README.md) and the [agreed architecture](../../PROJECT_DECISIONS.md#learner-memory-model-specification).

| Location | Contents |
|---|---|
| [run_v22.py](run_v22.py) | v2.2 7-model training, behavioral sanity gate, multi-seed selection, and held-out evaluation |
| [run_v21.py](run_v21.py) | Historical v2.1 training, multi-seed model selection, and held-out evaluation |
| [run.py](run.py) | Historical v2.0 training entry point and shared run utilities |
| [memory/](memory/) | Shared causal encoding, estimators (`time_input`, `time_decay`, `kt_decay`, `monotonic`), behavioral suite, batched training, and metrics |
| [models/](models/) | A design note for each baseline model |
| [METHODS_V22_AND_MONOTONIC_GRU.md](METHODS_V22_AND_MONOTONIC_GRU.md) | Complete v2.2 dataset & Monotonic Grey-Box GRU mathematical specification (Duolingo HLR + Power-Law + Consolidated Floor) |
| [METHODS_V21.md](METHODS_V21.md) | Historical v2.1 fitting, split, selection, and evaluation protocol |
| [METHODS.md](METHODS.md) | Shared feature definitions and original v2.0 protocol |
| [tests/](tests/) | Causality, timing, replay, behavioral sanity, and gradient regression tests |
| [runs/](runs/) | Separate run directories for checkpoints, metrics, and predictions |

The verified **v2.2** run compared seven models (including `kt_decay_gru` and the bounded `monotonic_gru` with Duolingo HLR + Power-Law + Consolidated Permastore Floor) and selected the **Monotonic Grey-Box GRU (`monotonic_gru`)**, which passed all **6/6 behavioral sanity checks** across all seeds and achieved the lowest combined validation log loss (`0.55175 ± 0.00025`, beating `time_input_gru` at `0.55177 ± 0.00022`, `time_decay_gru` at `0.55177 ± 0.00022` which failed the forgetting check at `4.7/6`, `kt_decay_gru` at `0.55471 ± 0.00077`, and all tabular baselines). The selected checkpoint is [seed 20261122](runs/v2.2-mps-main/monotonic_gru/seed_20261122/model.pt). [v2.2 Technical Specification](METHODS_V22_AND_MONOTONIC_GRU.md) · [v2.2 Comparison report](runs/v2.2-mps-main/REPORT.md) · [v2.2 Full metrics](runs/v2.2-mps-main/run.json)

The preserved v2.1 run selected the **time-decayed GRU** on the combined validation score ([seed 20261111](runs/v2.1-mps-main/time_decay_gru/seed_20261111/model.pt)). [v2.1 Comparison report](runs/v2.1-mps-main/REPORT.md) · [v2.1 Full metrics](runs/v2.1-mps-main/run.json) · [v2.1 Verification](runs/v2.1-mps-main/VERIFICATION.md)

The historical v2.0 MPS run includes its [comparison report](runs/v2-mps-main/REPORT.md) and [verification record](runs/v2-mps-main/VERIFICATION.md). Its delayed validation results measure extrapolation beyond the fitted time gaps. See the v2.1 protocol for the revised comparison.

## Run on the MacBook GPU

From the project root, create the model environment if needed. Use a native arm64 Python 3.12 interpreter.

```sh
cd 3/model
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

From the model folder, run the following in macOS Terminal. Supply a fresh output name for each run.

```sh
unset PYTORCH_ENABLE_MPS_FALLBACK
MEMORY_TEST_DEVICE=mps .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_v21.py --device mps --mode smoke --output runs/v21-gpu-check
.venv/bin/python run_v21.py --device mps --output runs/v21-new-main-run
```

MPS is the default for both GRUs. The runner verifies a GPU backward pass before training, requires finite GRU gradients on the requested device, and records the device and learning curves. The three scikit-learn fits and data preparation run on CPU. In a process that lacks Metal access, run the same command from a Terminal session with GPU access. For an intentional CPU check, select `--device cpu`. [PyTorch MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html)

## Read the output

- `REPORT.md` gives the validation choice and the probability metrics for every reported group.
- `run.json` gives environment versions, source and dataset manifest hashes, model settings, calibration bins, paired intervals, and status. Use `status = completed` for a completed run.
- Each model's subfolder contains `fit.json` and a checkpoint. The GRU checkpoints store CPU tensors for portability. The reported gradient device describes where the model trained.
- The compressed prediction CSVs align all models to the same query IDs and preserve the original precision of the probabilities. IDs support reporting and do not enter model features.

The runner requires a new output directory. The pilot smoke run uses 12 pilot training and 12 pilot validation learners for a short check. The full run trains on the main training split and evaluates main validation, main test, power-law stress, and long-gap stress.
