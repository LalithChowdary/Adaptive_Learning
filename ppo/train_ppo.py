"""Train and benchmark 5 teaching policies across 3 seeds and 3 held-out learner cohorts."""
import argparse
import json
from pathlib import Path
import sys
import time

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
    compute_gae,
    ppo_update,
)
from ppo.env import (  # noqa: E402
    ABLATED_OBS_DIM,
    ACTIONS,
    AdaptiveTutorEnv,
    CHECKPOINT,
    FrozenMemoryTracker,
    OBS_DIM,
    SKILLS_ORDER,
    TOTAL_BUDGET,
)

SEEDS = (20261120, 20261121, 20261122)
TRAIN_SEED_START = 10_000
TRAIN_LEARNER_POOL = 10_000
VAL_SEED_START = 20_000
TEST_SEED_START = 30_000
TEST_SEED_START_PREREG = 40_000


def _summarize(values):
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _bootstrap_diff(cand_scores, ref_scores, *, seed, draws=2000):
    diff = np.asarray(cand_scores, dtype=np.float64) - np.asarray(ref_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(diff), size=(draws, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "delta": float(diff.mean()),
        "lower_95": float(lo),
        "upper_95": float(hi),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
    }


def evaluate_policy(env, policy, learner_seeds, *, mechanism="exponential", long_gaps=False, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
    episodes = []
    for lseed in learner_seeds:
        if hasattr(policy, "reset"):
            policy.reset()
        obs, mask = env.reset(seed=int(lseed), mechanism=mechanism, long_gaps=long_gaps)
        while True:
            if isinstance(policy, MaskedActorCritic):
                action, _, _ = policy.act(obs, mask, deterministic=True)
            else:
                action = policy.select_action(obs, mask, rng)
            action_counts[action] += 1
            obs, _, done, mask, info = env.step(action)
            if done:
                episodes.append(info["delayed"])
                break

    total_actions = max(1, int(action_counts.sum()))
    metrics = {}
    for key in (
        "day_1",
        "day_7",
        "day_30",
        "delayed_mean",
        "day_1_expected",
        "day_7_expected",
        "day_30_expected",
        "delayed_expected_mean",
        "latent_mean_floor",
        "latent_mean_strength",
    ):
        metrics[key] = float(np.mean([ep[key] for ep in episodes]))
    metrics["action_shares"] = {
        ACTIONS[i]: float(action_counts[i] / total_actions) for i in range(len(ACTIONS))
    }
    per_skill_d30 = {
        skill: float(np.mean([ep["per_skill_day_30"][skill] for ep in episodes]))
        for skill in SKILLS_ORDER
    }
    lowest_skill = min(SKILLS_ORDER, key=lambda s: per_skill_d30[s])
    metrics["per_skill_day_30"] = per_skill_d30
    metrics["min_skill_day_30"] = float(per_skill_d30[lowest_skill])
    metrics["lowest_skill_name"] = lowest_skill
    metrics["per_learner_delayed_expected"] = [float(ep["delayed_expected_mean"]) for ep in episodes]
    metrics["per_learner_delayed_observed"] = [float(ep["delayed_mean"]) for ep in episodes]
    metrics["per_learner_day_30_observed"] = [float(ep["day_30"]) for ep in episodes]
    return metrics


def train_rl_policy(
    tracker,
    *,
    mode,
    seed,
    updates,
    episodes_per_update,
    val_learners,
    lr=2.5e-3,
):
    assert mode in ("delayed_retention_ppo", "myopic_rl_bandit", "ppo_no_longterm_state")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    ablate = mode == "ppo_no_longterm_state"
    obs_dim = ABLATED_OBS_DIM if ablate else OBS_DIM
    env = AdaptiveTutorEnv(tracker=tracker, no_longterm_state=ablate)
    model = MaskedActorCritic(obs_dim=obs_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    myopic = mode == "myopic_rl_bandit"
    gamma = 0.0 if myopic else 0.99
    gae_lambda = 0.0 if myopic else 0.95

    best_val = -1.0
    best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_any_val = -1.0
    best_any_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    curve = []
    started = time.monotonic()

    for upd in range(1, updates + 1):
        frac = (upd - 1) / max(1, updates - 1)
        cur_lr = lr * (1.0 - 0.85 * frac)
        cur_ent = 0.020 - 0.017 * frac
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        obs_buf, mask_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], [], []
        ep_delayed = []
        for _ in range(episodes_per_update):
            lseed = int(TRAIN_SEED_START + rng.integers(TRAIN_LEARNER_POOL))
            obs, mask = env.reset(seed=lseed, mechanism="exponential", long_gaps=False)
            while True:
                prev_pmed = float(obs[1])
                action, lp, val = model.act(obs, mask, deterministic=False)
                next_obs, shaped_rew, done, next_mask, info = env.step(action)
                if myopic:
                    if info["immediate_score"] is not None:
                        step_rew = float(info["immediate_score"])
                    else:
                        step_rew = 0.35 * max(0.0, float(next_obs[1]) - prev_pmed)
                else:
                    step_rew = shaped_rew

                obs_buf.append(obs)
                mask_buf.append(mask)
                act_buf.append(action)
                lp_buf.append(lp)
                rew_buf.append(step_rew)
                val_buf.append(val)
                done_buf.append(done)
                obs, mask = next_obs, next_mask
                if done:
                    ep_delayed.append(info["delayed"]["delayed_expected_mean"])
                    break

        advantages, returns = compute_gae(
            np.asarray(rew_buf, dtype=np.float32),
            np.asarray(val_buf, dtype=np.float32),
            np.asarray(done_buf, dtype=bool),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        stats = ppo_update(
            model,
            optimizer,
            np.asarray(obs_buf),
            np.asarray(mask_buf),
            np.asarray(act_buf),
            np.asarray(lp_buf, dtype=np.float32),
            returns,
            advantages,
            ent_coef=cur_ent,
            rng=rng,
        )
        if upd == 1 or upd % max(1, updates // 8) == 0 or upd == updates:
            val_metrics = evaluate_policy(env, model, val_learners, rng_seed=seed)
            val_score = val_metrics["delayed_expected_mean"]
            sh = val_metrics["action_shares"]
            if val_score > best_any_val:
                best_any_val = val_score
                best_any_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            healthy = True
            if not myopic and updates >= 20:
                healthy = sh["request_self_explanation"] <= 0.32 and sh["advance_next_topic"] >= 0.05
            if healthy and val_score > best_val:
                best_val = val_score
                best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            curve.append({
                "update": upd,
                "lr": cur_lr,
                "ent_coef": cur_ent,
                "train_expected_delayed": float(np.mean(ep_delayed)),
                "val_expected_delayed": val_score,
                "val_floor": val_metrics["latent_mean_floor"],
                "healthy_actions": healthy,
                **stats,
            })
            print(
                f"  [{mode} seed={seed}] upd {upd:3d}/{updates}: "
                f"val_delayed={val_score*100:.2f}% val_floor={val_metrics['latent_mean_floor']:.3f} "
                f"se={sh['request_self_explanation']*100:.1f}% adv={sh['advance_next_topic']*100:.1f}% "
                f"ent={stats['entropy']:.3f}",
                flush=True,
            )

    chosen_weights = best_weights if best_val >= 0.0 else best_any_weights
    model.load_state_dict(chosen_weights)
    model.eval()
    return model, {
        "mode": mode,
        "seed": seed,
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "best_val_expected_delayed": best_val if best_val >= 0.0 else best_any_val,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "learning_curve": curve,
    }


def evaluate_ppo_behavioral_suite(rep, baseline_summary):
    sh = rep["action_shares"]
    # Action 6 (review_prerequisite) also administers a prerequisite question turn via _emit_question_turn.
    q_share = (
        sh["give_easier_question"]
        + sh["give_same_difficulty_question"]
        + sh["give_harder_question"]
        + sh["review_prerequisite"]
    )
    checks = {
        "question_turns_at_least_40pct": q_share >= 0.40,
        "no_action_spam_over_32pct": (
            sh["request_self_explanation"] <= 0.32
            and sh["review_prerequisite"] <= 0.32
            and (sh["explain_concept"] + sh["show_worked_example"]) <= 0.32
        ),
        "curriculum_topic_advancement": 0.045 <= sh["advance_next_topic"] <= 0.25,
        "consolidated_floor_beats_heuristic": (
            rep["latent_mean_floor"] > baseline_summary["threshold_heuristic"]["latent_mean_floor"]["mean"]
        ),
        "delayed_retention_beats_all_baselines": rep["delayed_expected_mean"] > max(
            baseline_summary[b]["delayed_expected_mean"]["mean"]
            for b in ("random_valid", "fixed_curriculum", "threshold_heuristic", "myopic_rl_bandit")
        ),
    }
    return {
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "all_passed": all(checks.values()),
        "question_turns_share": float(q_share),
        "checks": checks,
    }


def _aggregate_seed_reports(reports):
    keys = (
        "day_1",
        "day_7",
        "day_30",
        "delayed_mean",
        "day_1_expected",
        "day_7_expected",
        "day_30_expected",
        "delayed_expected_mean",
        "latent_mean_floor",
        "latent_mean_strength",
    )
    summary = {k: _summarize([r[k] for r in reports]) for k in keys}
    summary["action_shares"] = {
        act: float(np.mean([r["action_shares"][act] for r in reports])) for act in ACTIONS
    }
    per_skill_d30 = {
        skill: _summarize([r["per_skill_day_30"][skill] for r in reports])
        for skill in SKILLS_ORDER
    }
    lowest_skill = min(SKILLS_ORDER, key=lambda s: per_skill_d30[s]["mean"])
    per_seed_mins = [r["min_skill_day_30"] for r in reports]
    summary["per_skill_day_30"] = per_skill_d30
    summary["min_skill_day_30"] = {
        "skill": lowest_skill,
        "rate": float(per_skill_d30[lowest_skill]["mean"]),
        **_summarize(per_seed_mins),
        "per_seed_lowest_skills": [r["lowest_skill_name"] for r in reports],
    }
    summary["lowest_skill_name"] = lowest_skill
    return summary


def write_report(output_dir, run):
    budget = run.get("primary_budget", TOTAL_BUDGET)
    lines = [
        "# PPO Adaptive Teaching Policy Benchmark (`v2.2` Frozen `monotonic_gru`)",
        "",
        f"Mode: `{run['mode']}` · Seeds: `{run['seeds']}` · Test learners per cohort: `{run['test_learners']}` · Budget: `{budget}`.",
        "",
    ]
    for cohort in run["evaluation"]:
        lines.extend([
            f"## {cohort}",
            "",
            "| Policy | 1-Day Pass | 7-Day Pass | 30-Day Pass | Delayed Mean (Obs) | Delayed Expected | Consolidated Floor $b_t$ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, res in run["evaluation"][cohort]["summary"].items():
            def fmt(entry, pct=True):
                m, sd = entry["mean"], entry["sd"]
                if pct:
                    return f"{m*100:.2f}%" if sd is None else f"{m*100:.2f}% ± {sd*100:.2f}%"
                return f"{m:.4f}" if sd is None else f"{m:.4f} ± {sd:.4f}"

            lines.append(
                f"| {name} | {fmt(res['day_1'])} | {fmt(res['day_7'])} | {fmt(res['day_30'])} | "
                f"{fmt(res['delayed_mean'])} | {fmt(res['delayed_expected_mean'])} | {fmt(res['latent_mean_floor'], pct=False)} |"
            )
        lines.extend(["", "### Action Distribution (`delayed_retention_ppo` vs. Baselines)", ""])
        lines.append("| Policy | " + " | ".join(ACTIONS) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(ACTIONS)) + "|")
        for name, res in run["evaluation"][cohort]["summary"].items():
            shares = [f"{res['action_shares'][a]*100:.1f}%" for a in ACTIONS]
            lines.append(f"| {name} | " + " | ".join(shares) + " |")
        lines.extend([
            "",
            "### 5-Check PPO Behavioral Sanity Suite (`delayed_retention_ppo`)",
            "",
            "| Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%-25%`) | Floor > Heuristic | Beats All Baselines |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for seed, b in zip(run["seeds"], run["evaluation"][cohort]["behavioral_by_seed"]):
            c = b["checks"]
            yn = lambda ok: "Pass" if ok else "Fail"
            lines.append(
                f"| `{seed}` | {b['checks_passed']} / {b['checks_total']} | "
                f"{yn(c['question_turns_at_least_40pct'])} ({b['question_turns_share']*100:.1f}%) | "
                f"{yn(c['no_action_spam_over_32pct'])} | {yn(c['curriculum_topic_advancement'])} | "
                f"{yn(c['consolidated_floor_beats_heuristic'])} | {yn(c['delayed_retention_beats_all_baselines'])} |"
            )
        lines.append("")

    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_cohort_policies(
    tracker,
    policies,
    test_learners,
    *,
    mech,
    lg,
    rng_seed,
    total_budget,
    bootstrap_draws,
    include_ablation3=False,
):
    eval_env = AdaptiveTutorEnv(tracker=tracker, total_budget=total_budget, no_longterm_state=False)
    ablate_env = (
        AdaptiveTutorEnv(tracker=tracker, total_budget=total_budget, no_longterm_state=True)
        if include_ablation3
        else None
    )
    by_seed = {}
    summary = {}
    for pname, pmodels in policies.items():
        target_env = ablate_env if pname == "ppo_no_longterm_state" else eval_env
        by_seed[pname] = [
            evaluate_policy(target_env, m, test_learners, mechanism=mech, long_gaps=lg, rng_seed=rng_seed)
            for m in pmodels
        ]
        summary[pname] = _aggregate_seed_reports(by_seed[pname])

    ppo_exp = np.mean([r["per_learner_delayed_expected"] for r in by_seed["delayed_retention_ppo"]], axis=0)
    ppo_obs = np.mean([r["per_learner_delayed_observed"] for r in by_seed["delayed_retention_ppo"]], axis=0)
    ppo_d30 = np.mean([r["per_learner_day_30_observed"] for r in by_seed["delayed_retention_ppo"]], axis=0)
    paired = {}
    ref_names = ["random_valid", "fixed_curriculum", "threshold_heuristic", "myopic_rl_bandit"]
    if include_ablation3 and "ppo_no_longterm_state" in by_seed:
        ref_names.append("ppo_no_longterm_state")
    for ref in ref_names:
        ref_exp = np.mean([r["per_learner_delayed_expected"] for r in by_seed[ref]], axis=0)
        ref_obs = np.mean([r["per_learner_delayed_observed"] for r in by_seed[ref]], axis=0)
        ref_d30 = np.mean([r["per_learner_day_30_observed"] for r in by_seed[ref]], axis=0)
        paired[f"delayed_retention_ppo minus {ref}"] = {
            "expected_delayed": _bootstrap_diff(ppo_exp, ref_exp, seed=rng_seed, draws=bootstrap_draws),
            "observed_delayed": _bootstrap_diff(ppo_obs, ref_obs, seed=rng_seed, draws=bootstrap_draws),
            "observed_day_30": _bootstrap_diff(ppo_d30, ref_d30, seed=rng_seed, draws=bootstrap_draws),
        }

    th_exp = np.mean([r["per_learner_delayed_expected"] for r in by_seed["threshold_heuristic"]], axis=0)
    th_obs = np.mean([r["per_learner_delayed_observed"] for r in by_seed["threshold_heuristic"]], axis=0)
    th_d30 = np.mean([r["per_learner_day_30_observed"] for r in by_seed["threshold_heuristic"]], axis=0)
    for cand in ("myopic_rl_bandit", "ppo_no_longterm_state", "fixed_curriculum", "random_valid"):
        if cand not in by_seed:
            continue
        c_exp = np.mean([r["per_learner_delayed_expected"] for r in by_seed[cand]], axis=0)
        c_obs = np.mean([r["per_learner_delayed_observed"] for r in by_seed[cand]], axis=0)
        c_d30 = np.mean([r["per_learner_day_30_observed"] for r in by_seed[cand]], axis=0)
        paired[f"{cand} minus threshold_heuristic"] = {
            "expected_delayed": _bootstrap_diff(c_exp, th_exp, seed=rng_seed, draws=bootstrap_draws),
            "observed_delayed": _bootstrap_diff(c_obs, th_obs, seed=rng_seed, draws=bootstrap_draws),
            "observed_day_30": _bootstrap_diff(c_d30, th_d30, seed=rng_seed, draws=bootstrap_draws),
        }

    clean_by_seed = {
        pname: [
            {k: v for k, v in rep.items() if not k.startswith("per_learner_")}
            for rep in reps
        ]
        for pname, reps in by_seed.items()
    }
    behavioral_by_seed = [
        evaluate_ppo_behavioral_suite(rep, summary)
        for rep in by_seed["delayed_retention_ppo"]
    ]
    return {
        "total_budget": float(total_budget),
        "summary": summary,
        "by_seed": clean_by_seed,
        "paired_intervals": paired,
        "behavioral_by_seed": behavioral_by_seed,
        "all_seeds_passed_behavioral": all(b["all_passed"] for b in behavioral_by_seed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--episodes-per-update", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--test-seed-start", type=int)
    parser.add_argument("--test-learners", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--budgets", type=float, nargs="+", default=[TOTAL_BUDGET])
    parser.add_argument("--run-ablation3", "--include-ablation3", dest="include_ablation3", action="store_true")
    parser.add_argument("--run-classification", default="retrospective")
    args = parser.parse_args()

    if args.mode == "smoke":
        seeds = tuple(args.seeds) if args.seeds else SEEDS[:1]
        updates = args.updates or 4
        ep_per_upd = args.episodes_per_update or 4
        val_learners = list(range(VAL_SEED_START, VAL_SEED_START + 8))
        test_start = args.test_seed_start if args.test_seed_start is not None else TEST_SEED_START
        n_test = args.test_learners or 10
        test_learners = list(range(test_start, test_start + n_test))
        cohorts = (("main_test", "exponential", False),)
    else:
        seeds = tuple(args.seeds) if args.seeds else SEEDS
        updates = args.updates or 100
        ep_per_upd = args.episodes_per_update or 24
        val_learners = list(range(VAL_SEED_START, VAL_SEED_START + 50))
        test_start = args.test_seed_start if args.test_seed_start is not None else TEST_SEED_START
        n_test = args.test_learners or 200
        test_learners = list(range(test_start, test_start + n_test))
        cohorts = (
            ("main_test", "exponential", False),
            ("power_law_test", "power_law", False),
            ("long_gap_test", "exponential", True),
        )

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    tracker = FrozenMemoryTracker()
    started = time.monotonic()

    policies = {
        "random_valid": [RandomValidPolicy()],
        "fixed_curriculum": [FixedCurriculumPolicy()],
        "threshold_heuristic": [ThresholdHeuristicPolicy()],
        "myopic_rl_bandit": [],
        "delayed_retention_ppo": [],
    }
    rl_modes = ["myopic_rl_bandit", "delayed_retention_ppo"]
    if args.include_ablation3:
        policies["ppo_no_longterm_state"] = []
        rl_modes.append("ppo_no_longterm_state")
    fit_logs = {}

    prior_training = {}
    run_json_path = args.output / "run.json"
    if run_json_path.exists():
        try:
            prior_run = json.loads(run_json_path.read_text(encoding="utf-8"))
            prior_training = prior_run.get("training", {})
        except Exception:
            prior_training = {}

    for rl_name in rl_modes:
        fit_logs[rl_name] = []
        prior_by_seed = {
            int(entry["seed"]): entry
            for entry in prior_training.get(rl_name, [])
            if isinstance(entry, dict) and "seed" in entry
        }
        folder = args.output / rl_name
        folder.mkdir(exist_ok=True)
        obs_dim = ABLATED_OBS_DIM if rl_name == "ppo_no_longterm_state" else OBS_DIM
        for seed in seeds:
            ckpt_path = folder / f"seed_{seed}.pt"
            if ckpt_path.exists():
                print(f"Loading existing {rl_name} seed={seed} from {ckpt_path}", flush=True)
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model = MaskedActorCritic(obs_dim=int(payload.get("obs_dim", obs_dim)))
                model.load_state_dict(payload["weights"])
                model.eval()
                prev = prior_by_seed.get(int(seed))
                if prev and not prev.get("reused"):
                    log = dict(prev)
                    log["obs_dim"] = model.obs_dim
                else:
                    log = {"mode": rl_name, "seed": seed, "obs_dim": model.obs_dim, "reused": True}
            else:
                print(f"Training {rl_name} seed={seed} (obs_dim={obs_dim})", flush=True)
                model, log = train_rl_policy(
                    tracker,
                    mode=rl_name,
                    seed=seed,
                    updates=updates,
                    episodes_per_update=ep_per_upd,
                    val_learners=val_learners,
                )
                log["obs_dim"] = obs_dim
                torch.save(
                    {"mode": rl_name, "seed": seed, "obs_dim": obs_dim, "weights": model.state_dict()},
                    ckpt_path,
                )
            fit_logs[rl_name].append(log)
            policies[rl_name].append(model)

    primary_budget = float(TOTAL_BUDGET if TOTAL_BUDGET in args.budgets else args.budgets[0])
    evaluation = {}
    for cohort_name, mech, lg in cohorts:
        print(f"Evaluating all policies on {cohort_name} ({len(test_learners)} learners, budget={primary_budget})", flush=True)
        evaluation[cohort_name] = _evaluate_cohort_policies(
            tracker,
            policies,
            test_learners,
            mech=mech,
            lg=lg,
            rng_seed=seeds[0],
            total_budget=primary_budget,
            bootstrap_draws=args.bootstrap_draws,
            include_ablation3=args.include_ablation3,
        )

    budget_curves = {}
    extra_budgets = [float(b) for b in args.budgets]
    if len(extra_budgets) > 1 or extra_budgets[0] != primary_budget:
        curve_policies = dict(policies)
        for b in extra_budgets:
            b_key = f"budget_{b:g}"
            budget_curves[b_key] = {}
            for cohort_name, mech, lg in cohorts:
                if b == primary_budget and cohort_name in evaluation:
                    src = evaluation[cohort_name]
                    budget_curves[b_key][cohort_name] = {
                        "total_budget": float(b),
                        "summary": src["summary"],
                        "paired_intervals": src["paired_intervals"],
                    }
                else:
                    print(f"Evaluating budget curve B={b:g} on {cohort_name} ({len(test_learners)} learners)", flush=True)
                    res_b = _evaluate_cohort_policies(
                        tracker,
                        curve_policies,
                        test_learners,
                        mech=mech,
                        lg=lg,
                        rng_seed=seeds[0],
                        total_budget=b,
                        bootstrap_draws=args.bootstrap_draws,
                        include_ablation3=args.include_ablation3,
                    )
                    budget_curves[b_key][cohort_name] = {
                        "total_budget": float(b),
                        "summary": res_b["summary"],
                        "paired_intervals": res_b["paired_intervals"],
                    }
                b_label = f"{b:.1f}"
                b_data = budget_curves[b_key][cohort_name]
                for pname, psum in evaluation[cohort_name]["summary"].items():
                    if pname not in b_data["summary"]:
                        continue
                    curve_entry = psum.setdefault("budget_curve", {})
                    b_psum = b_data["summary"][pname]
                    diff_key = f"{pname} minus threshold_heuristic"
                    curve_entry[b_label] = {
                        "delayed_mean": b_psum["delayed_mean"],
                        "day_30": b_psum["day_30"],
                        "latent_mean_floor": b_psum["latent_mean_floor"],
                        "paired_diff_vs_threshold_heuristic": (
                            b_data["paired_intervals"][diff_key]
                            if diff_key in b_data["paired_intervals"]
                            else None
                        ),
                    }

    run = {
        "status": "completed",
        "run_classification": args.run_classification,
        "frozen_memory_checkpoint": str(CHECKPOINT),
        "mode": args.mode,
        "seeds": list(seeds),
        "updates": updates,
        "episodes_per_update": ep_per_upd,
        "val_learners": len(val_learners),
        "test_seed_start": test_start,
        "test_learners": len(test_learners),
        "test_learner_seed_range": [test_start, test_start + len(test_learners) - 1],
        "bootstrap_draws": args.bootstrap_draws,
        "budget": primary_budget,
        "primary_budget": primary_budget,
        "budgets": extra_budgets,
        "include_ablation3": bool(args.include_ablation3),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "training": fit_logs,
        "evaluation": evaluation,
        "budget_curves": budget_curves,
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.output, run)
    print(f"Completed PPO benchmark in {run['elapsed_seconds']}s -> {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
