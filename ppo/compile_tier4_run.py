"""Compile read-only retrospective Tier 4 metrics and assemble unified tier4_run.json."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ppo.agent import (  # noqa: E402
    FixedCurriculumPolicy,
    MaskedActorCritic,
    RandomValidPolicy,
    ThresholdHeuristicPolicy,
)
from ppo.env import CHECKPOINT, FrozenMemoryTracker, TOTAL_BUDGET  # noqa: E402
from ppo.train_ppo import (  # noqa: E402
    TEST_SEED_START,
    _evaluate_cohort_policies,
)

DEFAULT_RETRO_DIR = Path("3/ppo/runs/v2.2-ppo-main")
DEFAULT_PREREG_RUN = Path("3/eval_runs/v2.2-preregistered/tier4/run.json")
DEFAULT_OUTPUT = Path("3/eval_runs/v2.2-preregistered/tier4_run.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compile_retrospective(retro_dir: Path, *, bootstrap_draws: int = 2000):
    retro_dir = Path(retro_dir)
    orig = json.loads((retro_dir / "run.json").read_text(encoding="utf-8"))
    seeds = list(orig["seeds"])
    n_test = int(orig["test_learners"])
    test_learners = list(range(TEST_SEED_START, TEST_SEED_START + n_test))

    tracker = FrozenMemoryTracker()
    policies = {
        "random_valid": [RandomValidPolicy()],
        "fixed_curriculum": [FixedCurriculumPolicy()],
        "threshold_heuristic": [ThresholdHeuristicPolicy()],
        "myopic_rl_bandit": [],
        "delayed_retention_ppo": [],
    }
    for rl_name in ("myopic_rl_bandit", "delayed_retention_ppo"):
        for seed in seeds:
            ckpt = retro_dir / rl_name / f"seed_{seed}.pt"
            payload = torch.load(ckpt, map_location="cpu", weights_only=True)
            model = MaskedActorCritic()
            model.load_state_dict(payload["weights"])
            model.eval()
            policies[rl_name].append(model)

    cohorts = (
        ("main_test", "exponential", False),
        ("power_law_test", "power_law", False),
        ("long_gap_test", "exponential", True),
    )
    evaluation = {}
    for cohort_name, mech, lg in cohorts:
        res = _evaluate_cohort_policies(
            tracker,
            policies,
            test_learners,
            mech=mech,
            lg=lg,
            rng_seed=seeds[0],
            total_budget=TOTAL_BUDGET,
            bootstrap_draws=bootstrap_draws,
            include_ablation3=False,
        )
        # Verify parity against historical v2.2-ppo-main/run.json
        # Note: visited_topics set iteration under PYTHONHASHSEED can break rare p_30d ties
        # differently on random_valid (up to ~2.6e-5 on latent_mean_floor).
        orig_cohort = orig["evaluation"][cohort_name]
        orig_summary = orig_cohort["summary"]
        for pname, psum in orig_summary.items():
            for metric_key in ("day_1", "day_7", "day_30", "delayed_mean", "latent_mean_floor"):
                diff = abs(res["summary"][pname][metric_key]["mean"] - psum[metric_key]["mean"])
                if diff > 1e-4:
                    raise ValueError(
                        f"Retrospective PPO parity mismatch in {cohort_name}.{pname}.{metric_key}: {diff}"
                    )
            # Preserve exact historical summary numbers verbatim while adding min/max from by_seed
            seed_rows = orig_cohort["by_seed"][pname]
            for hist_key, hist_val in psum.items():
                val_copy = copy.deepcopy(hist_val)
                if isinstance(val_copy, dict) and "mean" in val_copy and hist_key in seed_rows[0]:
                    vals = [float(r[hist_key]) for r in seed_rows]
                    val_copy["min"] = float(np.min(vals))
                    val_copy["max"] = float(np.max(vals))
                res["summary"][pname][hist_key] = val_copy
        res["behavioral_by_seed"] = copy.deepcopy(orig_cohort["behavioral_by_seed"])
        evaluation[cohort_name] = res

    return {
        "run_classification": "retrospective",
        "frozen_memory_checkpoint": str(CHECKPOINT),
        "source_run": str(retro_dir / "run.json"),
        "seeds": seeds,
        "updates": orig["updates"],
        "episodes_per_update": orig["episodes_per_update"],
        "budget": float(TOTAL_BUDGET),
        "test_learners": n_test,
        "test_learner_seed_range": [TEST_SEED_START, TEST_SEED_START + n_test - 1],
        "bootstrap_draws": bootstrap_draws,
        "historical_artifact_notes": (
            "per_skill_day_30, min_skill_day_30, min/max, day_30 paired CIs, and budget_curve were not logged "
            "in historical 3/ppo/runs/v2.2-ppo-main/run.json; min/max were compiled from by_seed and "
            "per_skill_day_30 + B=2000 paired CIs were computed read-only from saved checkpoints without "
            "modifying historical files. budget_curve was not logged in the historical artifact."
        ),
        "training": orig["training"],
        "evaluation": evaluation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retro-dir", type=Path, default=DEFAULT_RETRO_DIR)
    parser.add_argument("--prereg-run", type=Path, default=DEFAULT_PREREG_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()

    retro = compile_retrospective(args.retro_dir, bootstrap_draws=args.bootstrap_draws)
    prereg_raw = json.loads(args.prereg_run.read_text(encoding="utf-8"))
    prereg = copy.deepcopy(prereg_raw)
    prereg["run_classification"] = "pre_registered"
    prereg["frozen_memory_checkpoint"] = prereg.get("frozen_memory_checkpoint", str(CHECKPOINT))
    prereg["budget"] = float(prereg.get("primary_budget", TOTAL_BUDGET))
    prereg["test_learner_seed_range"] = [
        int(prereg["test_seed_start"]),
        int(prereg["test_seed_start"]) + int(prereg["test_learners"]) - 1,
    ]

    unified = {
        "status": "completed",
        "frozen_memory_checkpoint": str(CHECKPOINT),
        "frozen_memory_checkpoint_sha256": _sha256(CHECKPOINT),
        "retrospective": retro,
        "pre_registered": prereg,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(unified, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote unified Tier 4 report JSON -> {args.output}")


if __name__ == "__main__":
    main()
