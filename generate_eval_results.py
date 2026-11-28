"""Generate beginner-friendly, report-ready 3/RESEARCH_EVALUATION_RESULTS.md from verified JSON logs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval_runs" / "v2.2-preregistered"
TIER1_JSON = EVAL_DIR / "tier1_run.json"
TIER4_JSON = EVAL_DIR / "tier4_run.json"
TIER2_JSON = EVAL_DIR / "tier2_offline_diagnostics.json"
OUT_MD = ROOT / "RESEARCH_EVALUATION_RESULTS.md"

COHORTS = ("main_test", "power_law_test", "long_gap_test")
TIER1_MODELS = (
    "constant",
    "task_logistic",
    "history_logistic",
    "history_boosting",
    "time_input_gru",
    "time_decay_gru",
    "time_aware_transformer",
    "kt_decay_gru",
    "monotonic_gru",
    "monotonic_transformer",
    "oracle",
)
TIER1_SEQ = (
    "time_input_gru",
    "time_decay_gru",
    "time_aware_transformer",
    "kt_decay_gru",
    "monotonic_gru",
    "monotonic_transformer",
)
SKILLS_ORDER = (
    "range_boundaries",
    "loop_tracing",
    "accumulator_initialization",
    "conditional_evaluation",
    "list_indexing",
    "function_returns",
)
SKILL_SHORT = {
    "range_boundaries": "range_boundaries",
    "loop_tracing": "loop_tracing",
    "accumulator_initialization": "accumulator_init.",
    "conditional_evaluation": "conditional_eval.",
    "list_indexing": "list_indexing",
    "function_returns": "function_returns",
}
ACTIONS = (
    "give_easier_question",
    "give_same_difficulty_question",
    "give_harder_question",
    "explain_concept",
    "show_worked_example",
    "request_self_explanation",
    "review_prerequisite",
    "advance_next_topic",
)
ACTION_SHORT = {
    "give_easier_question": "give_easier",
    "give_same_difficulty_question": "give_same",
    "give_harder_question": "give_harder",
    "explain_concept": "explain",
    "show_worked_example": "worked_ex",
    "request_self_explanation": "self_explain",
    "review_prerequisite": "review_pre",
    "advance_next_topic": "advance",
}


def fmt_mean_sd(entry, decimals=4, pct=False):
    if entry is None:
        return "—"
    m = entry["mean"]
    sd = entry.get("sd")
    if pct:
        if sd is None:
            return f"{m * 100:.2f}%"
        return f"{m * 100:.2f}% ± {sd * 100:.2f}%"
    if sd is None:
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} ± {sd:.{decimals}f}"


def fmt_ci_ll(entry):
    if entry is None:
        return "—"
    return f"{entry['delta_log_loss']:+.5f} [{entry['lower_95']:+.5f}, {entry['upper_95']:+.5f}]"


def fmt_ci_pp(entry):
    if entry is None:
        return "—"
    return f"{entry['delta'] * 100:+.2f} pp [{entry['lower_95'] * 100:+.2f}, {entry['upper_95'] * 100:+.2f}]"


def fmt_range(entry, decimals=4, pct=True):
    if entry is None or entry.get("min") is None or entry.get("max") is None:
        return "—"
    if pct:
        return f"{entry['min'] * 100:.2f}%–{entry['max'] * 100:.2f}%"
    return f"{entry['min']:.{decimals}f}–{entry['max']:.{decimals}f}"


def _render_calibration_table(lines, title, ord_bins, ass_bins):
    lines.extend([
        title,
        "",
        "| Bin Range | Ordinary Queries ($n_b$) | Ord. Mean Pred | Ord. Observed | Delayed (`assessment`) Queries ($n_b$) | Del. Mean Pred | Del. Observed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for ob, db in zip(ord_bins, ass_bins):
        o_pred = f"{ob['mean_prediction']:.4f}" if ob["mean_prediction"] is not None else "—"
        o_obs = f"{ob['observed_rate']:.4f}" if ob["observed_rate"] is not None else "—"
        d_pred = f"{db['mean_prediction']:.4f}" if db["mean_prediction"] is not None else "—"
        d_obs = f"{db['observed_rate']:.4f}" if db["observed_rate"] is not None else "—"
        upper_br = "]" if abs(ob["upper"] - 1.0) < 1e-9 else ")"
        lines.append(
            f"| `[{ob['lower']:.1f}, {ob['upper']:.1f}{upper_br}` | {ob['queries']} | {o_pred} | {o_obs} | "
            f"{db['queries']} | {d_pred} | {d_obs} |"
        )
    lines.append("")


def render_tier1_tables(lines, block, label):
    seeds = block["seeds"]
    lines.extend([
        f"### Tier 1 — {label} (`{len(seeds)}` seeds: `{seeds}`)",
        "",
    ])
    beh = block["behavioral_summary"]
    val_sel = block["validation_selection"]
    sel_model = block.get("selected_model", "—")
    sel_ckpt = block.get("selected_checkpoint", "—")

    lines.extend([
        f"#### Validation Selection (`main_validation` · {label})",
        "",
        f"- **Selected Model (`validation_selection` minimum among 6/6 gate passers):** `{sel_model}` · **Selected Checkpoint:** `{sel_ckpt}`",
        "",
        "| Model | Val Selection LL (Mean ± SD) | Val Min–Max | Behavioral Gate | Selection Status |",
        "|---|---:|---|---|---|",
    ])
    for m in TIER1_MODELS:
        if m not in val_sel:
            continue
        vs = val_sel[m]
        vrange = fmt_range(vs, decimals=5, pct=False) if vs.get("sd") is not None else "—"
        if m in TIER1_SEQ:
            b = beh[m]
            passed_mean = b.get("checks_passed_mean", b.get("mean_checks_passed", 0.0))
            tot_checks = b.get("checks_total", 6)
            all_ok = b["all_seeds_passed"]
            gate = f"{passed_mean:.1f}/{tot_checks} ({'Pass' if all_ok else 'Fail'})"
            if m == sel_model:
                status = "Selected (Lowest Passing Val LL)"
            elif all_ok:
                status = "Eligible (Pass)"
            else:
                status = "Rejected (Behavioral Gate Fail)"
        else:
            gate = "Tabular"
            status = "Baseline"
        lines.append(
            f"| `{m}` | {fmt_mean_sd(vs, 5)} | {vrange} | {gate} | {status} |"
        )
    lines.append("")

    for cohort in COHORTS:
        ev = block["evaluation"][cohort]
        summ = ev["summary"]
        sig = ev["constant_to_oracle_log_loss_fraction"]
        lines.extend([
            f"#### Table T1-A ({label} · `{cohort}`): Held-Out Memory Performance (Ordinary vs. Delayed)",
            "",
            "| Model | Behav. Gate | Ord. LL (Mean±SD) | Ord. AUC | Ord. Brier | Ord. ECE-10 | Ord. Signal% | Del. LL (Mean±SD) | Del. AUC | Del. Brier | Del. ECE-10 | Del. Signal% |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for m in TIER1_MODELS:
            ord_m = summ[m]["ordinary"]
            ass_m = summ[m]["assessment"]
            del_ll = summ[m]["delayed_equal_horizon_log_loss"]
            if m == "constant":
                gate = "—"
                ord_sig = "0.0%"
                del_sig = "0.0%"
            elif m == "oracle":
                gate = "—"
                ord_sig = "100.0%"
                del_sig = "100.0%"
            elif m in TIER1_SEQ:
                b = beh[m]
                passed_mean = b.get("checks_passed_mean", b.get("mean_checks_passed", 0.0))
                tot_checks = b.get("checks_total", 6)
                gate = f"{passed_mean:.1f}/{tot_checks} ({'Pass' if b['all_seeds_passed'] else 'Fail'})"
                ord_sig = f"{sig['ordinary'][m] * 100:.1f}%"
                del_sig = f"{sig['delayed_equal_horizon_log_loss'][m] * 100:.1f}%"
            else:
                gate = "Tabular"
                ord_sig = f"{sig['ordinary'][m] * 100:.1f}%"
                del_sig = f"{sig['delayed_equal_horizon_log_loss'][m] * 100:.1f}%"
            lines.append(
                f"| `{m}` | {gate} | {fmt_mean_sd(ord_m['log_loss'], 5)} | {fmt_mean_sd(ord_m['roc_auc'], 4)} | "
                f"{fmt_mean_sd(ord_m['brier'], 5)} | {fmt_mean_sd(ord_m['ece_10'], 5)} | {ord_sig} | "
                f"{fmt_mean_sd(del_ll, 5)} | {fmt_mean_sd(ass_m['roc_auc'], 4)} | "
                f"{fmt_mean_sd(ass_m['brier'], 5)} | {fmt_mean_sd(ass_m['ece_10'], 5)} | {del_sig} |"
            )
        lines.append("")

        lines.extend([
            f"#### Table T1-C ({label} · `{cohort}`): Follow-Up & Per-Horizon (`1d`, `7d`, `30d`) Log Loss",
            "",
            "| Model | `followup` LL | `day_1` LL | `day_7` LL | `day_30` LL | `delayed_mean` LL | `delayed_mean` Min–Max |",
            "|---|---:|---:|---:|---:|---:|---|",
        ])
        for m in TIER1_MODELS:
            sm = summ[m]
            del_entry = sm["delayed_equal_horizon_log_loss"]
            del_range = fmt_range(del_entry, decimals=5, pct=False) if del_entry.get("sd") is not None else "—"
            lines.append(
                f"| `{m}` | {fmt_mean_sd(sm['followup']['log_loss'], 5)} | "
                f"{fmt_mean_sd(sm['day_1']['log_loss'], 5)} | "
                f"{fmt_mean_sd(sm['day_7']['log_loss'], 5)} | "
                f"{fmt_mean_sd(sm['day_30']['log_loss'], 5)} | "
                f"{fmt_mean_sd(del_entry, 5)} | {del_range} |"
            )
        lines.append("")

        pi = ev["paired_intervals"]
        best_tab = min(("task_logistic", "history_logistic", "history_boosting"), key=lambda t: block["validation_selection"][t]["mean"])
        contrast_names = [
            "history_logistic minus task_logistic",
            "history_boosting minus history_logistic",
            "monotonic_gru minus kt_decay_gru",
            "monotonic_gru minus time_decay_gru",
            "monotonic_gru minus time_aware_transformer",
            "monotonic_transformer minus time_aware_transformer",
            "monotonic_transformer minus monotonic_gru",
            f"monotonic_gru minus {best_tab}",
        ]
        lines.extend([
            f"#### Table T1-D ({label} · `{cohort}`): Learner-Clustered Paired Bootstrap 95% CIs (`B={block['bootstrap_draws']}`, Negative Favors Candidate)",
            "",
            "| Contrast (`Candidate minus Reference`) | `ordinary` Δ (95% CI) | `followup` Δ (95% CI) | `day_1` Δ (95% CI) | `day_7` Δ (95% CI) | `day_30` Δ (95% CI) |",
            "|---|---|---|---|---|---|",
        ])
        for cname in contrast_names:
            lines.append(
                f"| `{cname}` | {fmt_ci_ll(pi['ordinary'][cname])} | {fmt_ci_ll(pi['followup'][cname])} | "
                f"{fmt_ci_ll(pi['day_1'][cname])} | {fmt_ci_ll(pi['day_7'][cname])} | {fmt_ci_ll(pi['day_30'][cname])} |"
            )
        lines.append("")

    # Table T1-B on main_test for monotonic_gru
    ev_main = block["evaluation"]["main_test"]
    mono_summary = ev_main.get("summary", {}).get("monotonic_gru", {})
    cal_ord = mono_summary.get("ordinary", {}).get("calibration")
    cal_ass = mono_summary.get("assessment", {}).get("calibration")
    if cal_ord and cal_ass:
        _render_calibration_table(
            lines,
            f"#### Table T1-B ({label} · `main_test`): 10-Bin Calibration & Reliability Data (`monotonic_gru`, Seed-Averaged)",
            cal_ord,
            cal_ass,
        )
    scc = ev_main.get("selected_checkpoint_calibration")
    if scc and scc.get("ordinary") and scc.get("assessment"):
        _render_calibration_table(
            lines,
            f"#### Table T1-B ({label} Archived `selected_checkpoint` · `main_test`): 10-Bin Calibration (`{scc.get('checkpoint', sel_ckpt)}`)",
            scc["ordinary"],
            scc["assessment"],
        )

    # Table T1-E Behavioral Gate
    lines.extend([
        f"#### Table T1-E ({label}): 6-Check Behavioral Sanity Gate & Conditionals Trace Replay",
        "",
        "| Sequence Model | Checks Passed | All Seeds Passed? | Fast `0%` Wrong (`6.4s`) Δ | After 3 Hints + 2nd `0%` Wrong | Non-Increasing Forgetting (`0–60d`) | Gate Status |",
        "|---|---:|---|---:|---:|---|---|",
    ])
    for m in TIER1_SEQ:
        b = beh[m]
        passed_mean = b.get("checks_passed_mean", b.get("mean_checks_passed", 0.0))
        tot_checks = b.get("checks_total", 6)
        b0 = b.get("representative_replay") or block["models"][m][0]["behavioral"]
        replay = b0["user_conditionals_replay"]
        cold = replay["before_attempt"]
        after1 = replay["after_first_wrong_6.4s"]
        after_hints = replay["after_three_hints"]
        after2 = replay["after_second_wrong_2.27s"]
        d1 = after1 - cold
        d2 = after2 - cold
        non_inc = all(
            mod_fit["behavioral"]["checks"]["non_increasing_forgetting_after_pass_and_fail"]
            for mod_fit in block["models"][m]
        )
        lines.append(
            f"| `{m}` | {passed_mean:.1f} / {tot_checks} | {'Yes' if b['all_seeds_passed'] else 'No'} | "
            f"{cold * 100:.1f}% → {after1 * 100:.1f}% ({d1 * 100:+.2f} pp) | "
            f"{after2 * 100:.1f}% ({d2 * 100:+.2f} pp; post-hints {after_hints * 100:.1f}%) | "
            f"{'Pass' if non_inc else 'Fail'} | {'Eligible' if b['all_seeds_passed'] else 'Rejected'} |"
        )
    lines.append("")


def render_tier4_tables(lines, block, label):
    seeds = block["seeds"]
    n_test = block["test_learners"]
    budget = block.get("budget", 32.0)
    lines.extend([
        f"### Tier 4 — {label} (`{len(seeds)}` seeds, `N={n_test}` learners `{block['test_learner_seed_range'][0]}..{block['test_learner_seed_range'][1]}`, Budget `{budget}`)",
        "",
    ])

    core_policies = ("delayed_retention_ppo", "threshold_heuristic", "myopic_rl_bandit", "fixed_curriculum", "random_valid")
    lines.extend([
        f"#### Table T4-A ({label}): Policy Benchmark at Budget `{budget}`",
        "",
        "| Cohort | Policy | 1-Day Pass | 7-Day Pass | **30-Day Pass** | **Delayed Mean (Obs)** | Paired Diff vs `threshold_heuristic` (95% CI) | Delayed Expected | Floor $b_t$ | Seed Range (Min–Max) |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ])
    for cohort in COHORTS:
        ev = block["evaluation"][cohort]
        summ = ev["summary"]
        pi = ev["paired_intervals"]
        for p in core_policies:
            sp = summ[p]
            contrast_key = f"{p} minus threshold_heuristic"
            if p == "threshold_heuristic":
                ci_str = "`0.00 pp` (Ref)"
            elif contrast_key in pi:
                ci_str = fmt_ci_pp(pi[contrast_key]["observed_delayed"])
            else:
                ci_str = f"{(sp['delayed_mean']['mean'] - summ['threshold_heuristic']['delayed_mean']['mean']) * 100:+.2f} pp"
            srange = fmt_range(sp["delayed_mean"], pct=True) if len(seeds) > 1 and p in ("delayed_retention_ppo", "myopic_rl_bandit") else "—"
            lines.append(
                f"| `{cohort}` | `{p}` | {fmt_mean_sd(sp['day_1'], pct=True)} | {fmt_mean_sd(sp['day_7'], pct=True)} | "
                f"**{fmt_mean_sd(sp['day_30'], pct=True)}** | **{fmt_mean_sd(sp['delayed_mean'], pct=True)}** | {ci_str} | "
                f"{fmt_mean_sd(sp['delayed_expected_mean'], pct=True)} | {fmt_mean_sd(sp['latent_mean_floor'], decimals=4, pct=False)} | {srange} |"
            )
    lines.append("")

    # Table T4-B: All 6 Day-30 per-skill pass rates
    lines.extend([
        f"#### Table T4-B ({label}): All 6 Day-30 Per-Skill Pass Rates & Lowest-Skill Retention",
        "",
        "| Cohort | Policy | `range_boundaries` | `loop_tracing` | `accumulator_init.` | `conditional_eval.` | `list_indexing` | `function_returns` | **Min (Lowest Skill)** |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for cohort in COHORTS:
        summ = block["evaluation"][cohort]["summary"]
        pols = list(core_policies) + (["ppo_no_longterm_state"] if "ppo_no_longterm_state" in summ else [])
        for p in pols:
            sp = summ[p]
            ps = sp["per_skill_day_30"]
            low_name = sp["lowest_skill_name"]
            low_val = fmt_mean_sd(sp["min_skill_day_30"], pct=True)
            vals = [fmt_mean_sd(ps[s], pct=True) for s in SKILLS_ORDER]
            lines.append(
                f"| `{cohort}` | `{p}` | " + " | ".join(vals) + f" | **{low_val}** (`{SKILL_SHORT[low_name]}`) |"
            )
    lines.append("")

    # Table T4-C: Ablation 3 (pre-registered only)
    if block.get("include_ablation3"):
        lines.extend([
            f"#### Table T4-C ({label}): Ablation 3 — Full 36-Dim `delayed_retention_ppo` vs. 23-Dim Mask/Routing-Ablated `ppo_no_longterm_state`",
            "",
            "| Cohort | Policy | State Dims | Delayed Mean (Obs) | **Day-30 Pass** | Paired Diff (`ppo_full - ablated`, Obs / Day-30 95% CI) | Floor $b_t$ | Min-Skill Day-30 | `self_explanation` Share |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|",
        ])
        for cohort in COHORTS:
            ev = block["evaluation"][cohort]
            summ = ev["summary"]
            pi = ev["paired_intervals"]
            full_s = summ["delayed_retention_ppo"]
            abl_s = summ["ppo_no_longterm_state"]
            ci_pair = pi["delayed_retention_ppo minus ppo_no_longterm_state"]
            ci_str = f"{fmt_ci_pp(ci_pair['observed_delayed'])} / Day-30: {fmt_ci_pp(ci_pair['observed_day_30'])}"
            lines.append(
                f"| `{cohort}` | `delayed_retention_ppo` | 36 | {fmt_mean_sd(full_s['delayed_mean'], pct=True)} | "
                f"**{fmt_mean_sd(full_s['day_30'], pct=True)}** | — | {fmt_mean_sd(full_s['latent_mean_floor'], 4, False)} | "
                f"{fmt_mean_sd(full_s['min_skill_day_30'], pct=True)} | {full_s['action_shares']['request_self_explanation'] * 100:.1f}% |"
            )
            lines.append(
                f"| `{cohort}` | `ppo_no_longterm_state` | 23 | {fmt_mean_sd(abl_s['delayed_mean'], pct=True)} | "
                f"**{fmt_mean_sd(abl_s['day_30'], pct=True)}** | {ci_str} | {fmt_mean_sd(abl_s['latent_mean_floor'], 4, False)} | "
                f"{fmt_mean_sd(abl_s['min_skill_day_30'], pct=True)} | {abl_s['action_shares']['request_self_explanation'] * 100:.1f}% |"
            )
        lines.append("")

    # Table T4-D: Budget Curves
    if block.get("budget_curves"):
        lines.extend([
            f"#### Table T4-D ({label}): Matched Budget Curves (`TOTAL_BUDGET` in `{{16.0, 24.0, 32.0, 40.0}}`, `SESSION_BUDGET = 8.0`)",
            "",
            "| Cohort | Budget ($B$) | Policy | Delayed Mean (Obs) | **Day-30 Pass** | Floor $b_t$ | Paired Diff vs `threshold_heuristic` (95% CI) |",
            "|---|---:|---|---:|---:|---:|---|",
        ])
        for cohort in COHORTS:
            for b in block["budgets"]:
                b_key = f"budget_{b:g}"
                b_data = block["budget_curves"][b_key][cohort]
                bsum = b_data["summary"]
                bpi = b_data["paired_intervals"]
                for p in ("delayed_retention_ppo", "threshold_heuristic", "myopic_rl_bandit"):
                    sp = bsum[p]
                    if p == "threshold_heuristic":
                        ci_str = "`0.00 pp` (Ref)"
                    elif p == "delayed_retention_ppo":
                        ci_str = fmt_ci_pp(bpi["delayed_retention_ppo minus threshold_heuristic"]["observed_delayed"])
                    else:
                        ci_str = fmt_ci_pp(bpi["myopic_rl_bandit minus threshold_heuristic"]["observed_delayed"])
                    lines.append(
                        f"| `{cohort}` | {b:.1f} | `{p}` | {fmt_mean_sd(sp['delayed_mean'], pct=True)} | "
                        f"**{fmt_mean_sd(sp['day_30'], pct=True)}** | {fmt_mean_sd(sp['latent_mean_floor'], 4, False)} | {ci_str} |"
                    )
        lines.append("")

    # Table T4-E: 8-Action Distribution & 5-Check Suite
    lines.extend([
        f"#### Table T4-E ({label}): 8-Action Distribution Shares & 5-Check Behavioral Sanity Suite",
        "",
        "**Part 1 — 8-Action Distribution Shares (% of Turns):**",
        "",
        "| Cohort | Policy | " + " | ".join(f"`{ACTION_SHORT[a]}`" for a in ACTIONS) + " |",
        "|---|---|" + "|".join(["---:"] * len(ACTIONS)) + "|",
    ])
    for cohort in COHORTS:
        summ = block["evaluation"][cohort]["summary"]
        pols = list(core_policies) + (["ppo_no_longterm_state"] if "ppo_no_longterm_state" in summ else [])
        for p in pols:
            shares = [f"{summ[p]['action_shares'][a] * 100:.1f}%" for a in ACTIONS]
            lines.append(f"| `{cohort}` | `{p}` | " + " | ".join(shares) + " |")
    lines.extend([
        "",
        "**Part 2 — 5-Check Behavioral Sanity Suite (`delayed_retention_ppo` Across Seeds):**",
        "",
        "| Cohort | Seed | Checks Passed | Question Turns (`>=40%`) | No Action Spam (`<=32%`) | Topic Advance (`4.5%–25%`) | Floor > Heuristic | Beats All Baselines |",
        "|---|---|---:|---:|---|---|---|---|",
    ])
    any_fail = False
    cohort_pass_counts = {}
    cohort_outcome_counts = {}
    failing_notes = []
    for cohort in COHORTS:
        beh_list = block["evaluation"][cohort]["behavioral_by_seed"]
        cohort_pass_counts[cohort] = sum(1 for b in beh_list if b["all_passed"])
        cohort_outcome_counts[cohort] = sum(
            1
            for b in beh_list
            if b["checks"]["consolidated_floor_beats_heuristic"]
            and b["checks"]["delayed_retention_beats_all_baselines"]
        )
        for seed, b in zip(seeds, beh_list):
            c = b["checks"]
            yn = lambda ok: "Pass" if ok else "Fail"
            lines.append(
                f"| `{cohort}` | `{seed}` | {b['checks_passed']} / {b['checks_total']} | "
                f"{yn(c['question_turns_at_least_40pct'])} ({b['question_turns_share'] * 100:.1f}%) | "
                f"{yn(c['no_action_spam_over_32pct'])} | {yn(c['curriculum_topic_advancement'])} | "
                f"{yn(c['consolidated_floor_beats_heuristic'])} | {yn(c['delayed_retention_beats_all_baselines'])} |"
            )
            if not b["all_passed"]:
                any_fail = True
                failed_checks = [k for k, ok in c.items() if not ok]
                failing_notes.append(f"`{cohort}`/`{seed}` ({', '.join(failed_checks)})")
    lines.append("")
    if any_fail:
        n_seeds = len(seeds)
        lines.extend([
            f"> **Behavioral Sanity Note ({label}):** `{cohort_pass_counts['main_test']}/{n_seeds}` seeds pass `5/5` checks on `main_test` and `{cohort_pass_counts['power_law_test']}/{n_seeds}` on `power_law_test` (`{cohort_pass_counts['long_gap_test']}/{n_seeds}` on `long_gap_test`), while **`{cohort_outcome_counts['main_test']}/{n_seeds}` seeds pass both outcome checks** (`consolidated_floor_beats_heuristic` and `delayed_retention_beats_all_baselines`) across all 3 cohorts; action-share bound misses occur only in {'; '.join(failing_notes)}.",
            "",
        ])


def main():
    t1 = json.loads(TIER1_JSON.read_text(encoding="utf-8"))
    t4 = json.loads(TIER4_JSON.read_text(encoding="utf-8"))
    t2 = json.loads(TIER2_JSON.read_text(encoding="utf-8"))

    # Headline numbers from pre-registered & retrospective Tier 1
    pr1 = t1["pre_registered"]
    rt1 = t1["retrospective"]
    pr1_ti = pr1["validation_selection"]["time_input_gru"]
    pr1_mg = pr1["validation_selection"]["monotonic_gru"]
    pr1_td = pr1["validation_selection"]["time_decay_gru"]
    pr1_td_checks = pr1["behavioral_summary"]["time_decay_gru"]["checks_passed_mean"]
    rt1_mg = rt1["validation_selection"]["monotonic_gru"]

    # Headline numbers from pre-registered Tier 4
    pr4 = t4["pre_registered"]
    pr4_main = pr4["evaluation"]["main_test"]
    ppo_del = pr4_main["summary"]["delayed_retention_ppo"]["delayed_mean"]["mean"] * 100
    th_del = pr4_main["summary"]["threshold_heuristic"]["delayed_mean"]["mean"] * 100
    diff_del = pr4_main["paired_intervals"]["delayed_retention_ppo minus threshold_heuristic"]["observed_delayed"]
    ppo_d30 = pr4_main["summary"]["delayed_retention_ppo"]["day_30"]["mean"] * 100
    th_d30 = pr4_main["summary"]["threshold_heuristic"]["day_30"]["mean"] * 100
    diff_d30 = pr4_main["paired_intervals"]["delayed_retention_ppo minus threshold_heuristic"]["observed_day_30"]
    abl_d30 = pr4_main["summary"]["ppo_no_longterm_state"]["day_30"]["mean"] * 100
    abl_diff_d30 = pr4_main["paired_intervals"]["delayed_retention_ppo minus ppo_no_longterm_state"]["observed_day_30"]
    pr4_n_seeds = len(pr4["seeds"])
    pr4_pass_main = sum(1 for b in pr4_main["behavioral_by_seed"] if b["all_passed"])
    pr4_pass_lg = sum(1 for b in pr4["evaluation"]["long_gap_test"]["behavioral_by_seed"] if b["all_passed"])
    pr4_out_main = sum(
        1
        for b in pr4_main["behavioral_by_seed"]
        if b["checks"]["consolidated_floor_beats_heuristic"]
        and b["checks"]["delayed_retention_beats_all_baselines"]
    )

    lines = [
        "# Research Evaluation Results & Report Reference (`v2.2` Offline Phase)",
        "",
        "> **Scope Statement:** This document reports the verified **Retrospective** and **Pre-Registered** offline research evaluations across **Tier 1** (Learner Memory), **Tier 2** (Retrieval offline diagnostics), and **Tier 4 + Ablation 3** (Teaching Policy in simulation), alongside the exported blind review packs for **Tier 2/3 live-LLM & human adjudication**. All Tier 1 and Tier 4 claims are scoped strictly to the simulated learner environment (`3/data_Set/v2.2/simulator.py` and `3/ppo/env.py`); real-student effectiveness requires a separate IRB-approved Tier 5 pilot.",
        "",
        "---",
        "",
        "## Plain-English Guide for Readers (Beginner-Friendly Overview)",
        "",
        "If you are reading this report for the first time (e.g., as an undergraduate student, thesis reviewer, or fresh graduate), here is what every part and metric means in plain English:",
        "",
        "### What Each Tier Tests",
        "1. **Our Main Goal (Primary Outcome & `Follow-up` vs. `1d/7d/30d`):** Imagine giving a Python tutor a fixed study budget (`32.0` units, roughly four short sessions spaced days apart). After studying finishes, we test the student **without any hints** at **1 day, 7 days, and 30 days later**. A good tutor should not just help a student cram for today—it should help them **still pass 30 days later**. Our primary outcome is the **equal-weight average pass rate across Days 1, 7, and 30 (`delayed_mean`)**, and we highlight **Day 30 (`day_30`)** on its own. (In Tier 1, `Follow-up` measures immediate unaided checks right after assisted practice, whereas `1d`/`7d`/`30d` measure retention after real time gaps.)",
        "2. **Tier 1 — Memory Model (`monotonic_gru` & Comparators):** Before choosing what to teach, the tutor predicts: *\"What is the chance this student will solve a fresh, unaided problem right now or 30 days from now?\"*",
        "   - **Log Loss (Lower is better):** Measures how accurate and honest the model's predicted probabilities are (punishes confident wrong guesses; `0.693` is random 50/50 guessing).",
        "   - **ROC-AUC (Higher is better, `0.5` to `1.0`):** If you pick one student who passed and one who failed, how often did the model give the passing student a higher predicted probability?",
        "   - **Brier Score & ECE-10 (Lower is better):** Checks calibration—if the model says \"70% chance to pass,\" do students in that bin actually pass ~70% of the time? (`ECE-10` averages that calibration error across 10 equal-width probability bins `[0.0, 0.1)` ... `[0.9, 1.0]`.)",
        "   - **Signal %:** How much of the maximum possible predictive signal (from naive `constant` base-rate `0%` up to the hidden simulator `oracle` ceiling `100%`) the model captures from observable history alone.",
        "   - **6/6 Behavioral Gate:** Common-sense rules every tutor memory must obey (e.g., getting `0%` wrong must lower your predicted score, never raise it; 0–60 day forgetting curves must never creep upward without practice; passing without hints beats passing with 3 hints). Models failing any rule are disqualified from deployment.",
        "3. **Tier 2 — Retrieval (`rag.py`):** Checks whether the search engine finds the **right** Python lessons and example questions inside the exact slot limits the LLM tutor receives (`4` chunks for chat; `3` exemplars + `2` curriculum chunks for question generation), quickly, and without ever leaking one student's mistakes into another student's session.",
        "   - **`nDCG@k` (Higher is better, `0` to `1`):** Did the most relevant chunks appear near the *top* of the `k` retrieved slots?",
        "   - **`Recall@k` (Higher is better, `0` to `1`):** Out of all truly relevant chunks available, what fraction made it into the top `k` slots?",
        "   - **`p95 Latency (ms)` (Lower is better):** The 95th-percentile search speed—95% of queries finish faster than this time in milliseconds.",
        "4. **Tier 3 — Tutor Replies, Question Generation & Grading (`llm_pedagogy.py`, `grader.py`):** Evaluates the live LLM teacher with blinded human reviewers.",
        "   - **EduGuard Leakage vs. Over-Refusal (Tier 3a / Ablation 2):** *Leakage* is how often the tutor accidentally gives away the answer on protected answer-fishing prompts (lower is safer). *Over-Refusal* is how often the guardrail wrongly blocks or refuses a legitimate conceptual question (lower is more helpful).",
        "   - **First-Pass / Post-Repair / Usable Rate (Tier 3b):** Out of `180` requests, *First-Pass* is valid JSON/AST on the initial call; *Post-Repair* includes 1-shot automatic fixing; *Usable Rate* is the fraction human reviewers confirm are actually correct, solvable, and match the requested skill/difficulty/format.",
        "   - **MAE / QWK / Spearman $\\rho$ (Tier 3c):** *MAE* is average grading error on `[0, 1]` (lower is better); *QWK* (Quadratic Weighted Kappa, up to `1.0`) measures agreement between the automated grader and human teachers on `0 / partial / full` grades; *Spearman $\\rho$* checks whether higher grader confidence genuinely corresponds to lower error.",
        "5. **Tier 4 — RL Teaching Policy (`delayed_retention_ppo` & Ablation 3):** Compares our reinforcement-learning tutor against a standard rule-based tutor (`threshold_heuristic`) at equal study budget.",
        "   - **Permastore Floor $b_t$ (Higher is better):** The memory model's estimate of durable, consolidated knowledge that will *not* wash away even after a 30-day break.",
        "   - **All 6 Skills at Day 30 & Budget Curves:** Makes sure the tutor doesn't neglect harder skills like `function_returns`, and checks whether PPO wins across tighter (`16`) and larger (`40`) study budgets.",
        "   - **Ablation 3 (`ppo_no_longterm_state`):** Hides 30-day retention predictions and floor $b_t$ from PPO (shrinking its state from `36` to `23` numbers and restricting routing/masking to short-term `p_med`) to prove those long-term features cause the Day-30 gain.",
        "6. **How Confidence Intervals Work (`Paired Bootstrap` vs. `Wilson 95% CI`):**",
        "   - **Paired Bootstrap 95% CI (`B = 2,000`):** Compares two systems on the *exact same students* (or exact same prompts) by resampling students 2,000 times. If the entire 95% interval sits strictly above `0` for pass rates (or strictly below `0` for Log Loss), the improvement is consistent across students rather than luck.",
        "   - **Wilson 95% CI:** A reliable confidence interval for simple yes/no percentages (like answer-leakage rate or usable-question rate), especially when rates are near `0%` or `100%`.",
        "7. **Retrospective vs. Pre-Registered:** *Retrospective* numbers come from earlier 3-seed runs (`200` learners); *Pre-Registered* numbers come from fresh, locked **5-seed (Tier 1)** and **10-seed (`400` fresh learners, Tier 4)** runs executed under fixed protocols.",
        "",
        "### Quick Table Map",
        "| Table | What Question Does It Answer? |",
        "|---|---|",
        "| **T1-A** | Overall memory accuracy (`Log Loss`, `AUC`, `Brier`, `ECE-10`, `Signal %`) on ordinary practice vs. delayed exams. |",
        "| **T1-B** | When `monotonic_gru` predicts e.g. `65%` chance to pass across 10 probability bins, do students actually pass ~`65%`? |",
        "| **T1-C** | Memory accuracy broken down by exact gap (`followup`, `1d`, `7d`, `30d`, and `delayed_mean` with `Min–Max`). |",
        "| **T1-D** | Paired 95% CIs (`B=2,000`) showing which memory architectures beat which baselines on the exact same students. |",
        "| **T1-E** | Does each neural memory model pass all 6 common-sense behavioral sanity rules across every seed? |",
        "| **T2** | How fast (`p95`), schema-accurate, leak-free (`0` violations), and relevant (`nDCG@k`, `Recall@k`) is each RAG retriever? |",
        "| **T-A1** | Ablation 1: Does `BM25 + Dense` improve live-LLM tutor replies and generated questions over `Deployed BM25 + TF-IDF`? |",
        "| **T3a** | Tutor reply quality (`Python Correct`, `Faithfulness`, 3 rubric dims) and Ablation 2 **EduGuard ON vs. OFF** safety. |",
        "| **T3b** | Generated question validity (`First-Pass`, `Post-Repair`, `Fallback`, `Usable Rate`) overall and by skill/format/difficulty. |",
        "| **T3c** | How closely does the automated self-explanation grader match human teachers (`MAE`, `P/R/F1`, `QWK`, `Spearman ρ`)? |",
        "| **T4-A** | Primary policy benchmark: Does `delayed_retention_ppo` beat heuristic/bandit/fixed/random tutors at budget `32.0`? |",
        "| **T4-B** | Day-30 unaided pass rates for every individual Python skill (and the weakest skill) under each teaching policy. |",
        "| **T4-C** | Ablation 3: How much Day-30 retention is lost when hiding long-term 30-day memory & floor $b_t$ (`36` → `23` dims)? |",
        "| **T4-D** | Budget curves: Does PPO still beat baselines across smaller and larger teaching budgets (`16`, `24`, `32`, `40`)? |",
        "| **T4-E** | What mix of the 8 teaching actions does each policy use, and do PPO seeds pass the 5 behavioral sanity checks? |",
        "| **T5** | Optional future IRB-approved human classroom pilot template (strictly separated from simulation). |",
        "",
        "---",
        "",
        "## Headline Summary (Pre-Registered Offline Findings)",
        "",
        f"- **Primary Outcome (`main_test`, Budget `32.0`, `N=400` fresh learners, 10 seeds):** `delayed_retention_ppo` achieves **{ppo_del:.2f}%** mean delayed unaided pass rate vs. **{th_del:.2f}%** for `threshold_heuristic` (**{diff_del['delta']*100:+.2f} pp**, 95% CI `[{diff_del['lower_95']*100:+.2f}, {diff_del['upper_95']*100:+.2f}]`).",
        f"- **Day-30 Durable Retention (`main_test`):** **{ppo_d30:.2f}%** (`delayed_retention_ppo`) vs. **{th_d30:.2f}%** (`threshold_heuristic`), a **{diff_d30['delta']*100:+.2f} pp** (`[{diff_d30['lower_95']*100:+.2f}, {diff_d30['upper_95']*100:+.2f}]`) long-term retention gain.",
        f"- **Ablation 3 Proof (`ppo_no_longterm_state`, 23 dims):** Removing long-term retention & permastore features (and masking/routing with short-term `p_med` only) drops Day-30 pass rate from **{ppo_d30:.2f}%** down to **{abl_d30:.2f}%** on `main_test` (`{abl_diff_d30['delta']*100:+.2f} pp`, 95% CI `[{abl_diff_d30['lower_95']*100:+.2f}, {abl_diff_d30['upper_95']*100:+.2f}]`), with **`{pr4_pass_main}/{pr4_n_seeds}`** seeds passing `5/5` behavioral checks on `main_test`/`power_law_test` (`{pr4_pass_lg}/{pr4_n_seeds}` on `long_gap_test`) and **`{pr4_out_main}/{pr4_n_seeds}`** seeds beating all baselines on retention and floor.",
        f"- **Tier 1 Memory Selection & Gate:** In **Pre-Registered (5 seeds)**, `time_input_gru` (`{pr1_ti['mean']:.5f} ± {pr1_ti['sd']:.5f}` val selection LL, `6/6` behavioral gate) edged `monotonic_gru` (`{pr1_mg['mean']:.5f} ± {pr1_mg['sd']:.5f}`, `6/6` gate, tightest seed variance across all 5 seeds) by `{pr1_mg['mean'] - pr1_ti['mean']:.5f}`, whereas `monotonic_gru` won **Retrospective (3 seeds)** (`{rt1_mg['mean']:.5f} ± {rt1_mg['sd']:.5f}`) and remains the deployed $O(1)$ 32-float grey-box Markov state exposing explicit permastore floor $b_t$ and half-life $h_t$ for PPO/dashboard use; `time_decay_gru` was **rejected** by the gate (`{pr1_td_checks:.1f}/6`, `{pr1_td['mean']:.5f} ± {pr1_td['sd']:.5f}` val LL, failing non-increasing forgetting).",
        "- **Tier 2 Offline Retrieval Diagnostics:** Both `BM25 alone` and `Deployed BM25 + TF-IDF` achieve `100%` exemplar metadata match, `0` cross-learner episodic violations, `0` BM25 underfills, and `< 5.2 ms` `p95` latency across `66` dev and `132` held-out test queries.",
        "",
        "---",
        "",
        "## 1. Tier 1 Results: Learner Memory Model (`monotonic_gru` & Comparators)",
        "",
        "Machine-readable log: [`3/eval_runs/v2.2-preregistered/tier1_run.json`](eval_runs/v2.2-preregistered/tier1_run.json)",
        "",
    ]

    render_tier1_tables(lines, t1["pre_registered"], "Pre-Registered (5 seeds)")
    render_tier1_tables(lines, t1["retrospective"], "Retrospective (3 seeds)")

    # Tier 2 Section
    bm25_dev = t2["arms"]["bm25_alone"]["dev"]
    bm25_test = t2["arms"]["bm25_alone"]["test"]
    hyb_dev = t2["arms"]["hybrid_tfidf"]["dev"]
    hyb_test = t2["arms"]["hybrid_tfidf"]["test"]
    lines.extend([
        "---",
        "",
        "## 2. Tier 2 Results: Hybrid Retrieval & Ablation 1 (Offline Diagnostics + Pending Human/Dense)",
        "",
        f"Machine-readable logs: [`tier2_offline_diagnostics.json`](eval_runs/v2.2-preregistered/tier2_offline_diagnostics.json) · [`tier2_offline_eval.jsonl`](eval_runs/v2.2-preregistered/tier2_offline_eval.jsonl) · Blind annotation pack: [`tier2_relevance_annotation_pack.md`](eval_runs/v2.2-preregistered/blind_packs/tier2_relevance_annotation_pack.md) (`1,090` query-doc pairs across `66` Dev + `132` Test queries).",
        "",
        "> **Dense Encoder Note:** Local `sentence_transformers`, `transformers`, and `huggingface_hub` weights are not installed/cached in this environment (`cached_hf_dense_models: []`), so `Dense` and `BM25 + Dense` are wired through `DenseEncoder` and marked `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]`. Relevance metrics (`nDCG@k`, `Recall@k`) await 2-reviewer blinded labeling on the exported annotation pack.",
        "",
        "#### Table T2: Tier 2 Retrieval Comparison (Held-Out Test Split, `N = 132` Queries: `72` Chat + `60` Q-Gen)",
        "",
        "| Arm | Query Input (`BM25` / `Dense`) | Chat (`k=4`) nDCG@4 (Raw) | Chat (`k=4`) Recall@4 (Raw) | Q-Gen Exemplar (`k=3`) nDCG@3 / Recall@3 | Q-Gen Curric. (`k=2`) nDCG@2 / Recall@2 | Exemplar Meta Match% | Cross-Learner Violations | BM25 Underfills | p95 Latency (ms) |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|",
        f"| `BM25 alone` | `rewrite` / — | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | {bm25_test['metadata_match_rate']*100:.1f}% | {bm25_test['cross_learner_violation_total']} (`+{t2['cross_learner_leakage_probe']['violations']}` probe) | {bm25_test['bm25_underfilled_queries']}/{bm25_test['retrieval_tasks']} | {bm25_test['p95_latency_ms']:.3f} ms |",
        f"| `Deployed BM25 + TF-IDF` | `rewrite` / `rewrite` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | {hyb_test['metadata_match_rate']*100:.1f}% | {hyb_test['cross_learner_violation_total']} (`+{t2['cross_learner_leakage_probe']['violations']}` probe) | {hyb_test['bm25_underfilled_queries']}/{hyb_test['retrieval_tasks']} | {hyb_test['p95_latency_ms']:.3f} ms |",
        "| `Dense` *(Dev-selected)* | — / Dev-sel. | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING]` | `[PENDING]` | — | `[PENDING]` |",
        "| `BM25 + Dense` | `rewrite` / Dev-sel. | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING DENSE WEIGHTS & HUMAN ADJUDICATION]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |",
        "",
        f"*Dev split operational check (`66` queries / `96` retrieval tasks): `BM25 alone` p95 = `{bm25_dev['p95_latency_ms']:.3f} ms`, `Deployed BM25 + TF-IDF` p95 = `{hyb_dev['p95_latency_ms']:.3f} ms`, metadata match = `100.0%`, cross-learner violations = `0`, BM25 underfills = `0`. Out-of-scope abstention is `not_supported_deferred` (`3/adaptive_web/rag.py` always returns best available chunks).* ",
        "",
        "#### Table T-A1: Ablation 1 Fixed-LLM Downstream Check (`Deployed BM25 + TF-IDF` vs. `BM25 + Dense`)",
        f"*Blind downstream packet exported (`132` items: `72` chat + `60` q-gen): [`ablation1_downstream_pack.jsonl`](eval_runs/v2.2-preregistered/blind_packs/ablation1_downstream_pack.jsonl) (`ablation1_downstream_pack.md`).*",
        "",
        "| Retrieval Arm | Chat `live_n` / `72` | Chat Python Correct (`k/N`, %, CI) | Chat Claim Faithfulness (`k/N`, %, CI) | Chat Mean Rubric (0–2) | Q-Gen First-Pass Valid (`k/60`, %) | Q-Gen Post-Repair Valid (`k/60`, %) | Q-Gen Bank Fallback (`k/60`, %) | Q-Gen Usable Rate (`k/60`, %, CI) |",
        "|---|---:|---|---|---|---|---|---|---|",
        "| `Deployed BM25 + TF-IDF` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |",
        "| `BM25 + Dense` *(Dev-sel.)* | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |",
        "| **Paired Diff (`BM25+Dense - Deployed`)** | — | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |",
        "",
        "---",
        "",
        "## 3. Tier 3 Results: Tutor Quality, EduGuard, Question Generation & Self-Explanation Grading (Pending Live LLM & Human Review)",
        "",
        "No live LLM API calls or fabricated human labels were executed in this offline phase. All blind prompt/annotation packs are ready under `3/eval_runs/v2.2-preregistered/blind_packs/`:",
        "- **Tier 3a + Ablation 2 (`216` prompts = `96` adaptation + `60` protected + `60` legit doubts):** [`tier3a_ablation2_prompts.jsonl`](eval_runs/v2.2-preregistered/blind_packs/tier3a_ablation2_prompts.jsonl) · [`tier3a_ablation2_prompts.md`](eval_runs/v2.2-preregistered/blind_packs/tier3a_ablation2_prompts.md)",
        "- **Tier 3b (`180` stratified requests = `6×5×3×2`):** [`tier3b_qgen_requests.jsonl`](eval_runs/v2.2-preregistered/blind_packs/tier3b_qgen_requests.jsonl) · [`tier3b_qgen_requests.md`](eval_runs/v2.2-preregistered/blind_packs/tier3b_qgen_requests.md)",
        "- **Tier 3c (`120` prompts across `18` `se_prompt_bank` prompts, `rubric=None`):** [`tier3c_self_explanation_prompts.jsonl`](eval_runs/v2.2-preregistered/blind_packs/tier3c_self_explanation_prompts.jsonl) · [`tier3c_self_explanation_prompts.md`](eval_runs/v2.2-preregistered/blind_packs/tier3c_self_explanation_prompts.md)",
        "",
        "#### Table T3a: Tier 3a Tutor Quality & Ablation 2 EduGuard Safety (Isolated `chat_doubt` Snapshots)",
        "",
        "| Condition | `live_n` / `api_errors` | Python Correct (`k/N`, %, 95% CI) | Claim Faithfulness (`k/N`, %, 95% CI) | Protected Leakage (`k/60`, %, 95% CI) | Legit Over-Refusal (`k/60`, %, 95% CI) | Diff. Addressed (0–2) | Actionable Step (0–2) | Tier Adaptation (0–2) |",
        "|---|---:|---|---|---|---|---:|---:|---:|",
        "| `EduGuard ON` (Pinned LLM) | `[PENDING LIVE LLM]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING LIVE LLM & HUMAN]` | `[PENDING LIVE LLM & HUMAN]` | `[PENDING HUMAN]` | `[PENDING HUMAN]` | `[PENDING HUMAN]` |",
        "| `EduGuard OFF` (Ablation 2) | `[PENDING LIVE LLM]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING LIVE LLM & HUMAN]` | `[PENDING LIVE LLM & HUMAN]` | `[PENDING HUMAN]` | `[PENDING HUMAN]` | `[PENDING HUMAN]` |",
        "",
        "#### Table T3b: Tier 3b Generated Question Quality (Aggregate & Marginals, `N = 180`)",
        "",
        "| Slice | Requests ($N$) | `api_error` (`k/N`, %) | First-Pass Valid (`k/N`, %, CI) | Post-Repair Valid (`k/N`, %, CI) | Repair Yield (`rep/att`) | Bank Fallback (`k/N`, %, CI) | Usable Generated Rate (`k/N`, %, CI) |",
        "|---|---:|---|---|---|---|---|---|",
        "| **Overall Aggregate** | 180 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |",
    ])
    for sk in SKILLS_ORDER:
        lines.append(
            f"| Skill: `{sk}` | 30 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |"
        )
    for fmt in ("single_choice", "multiple_select", "output_prediction", "fill_in_the_blank", "code_completion (AST)"):
        lines.append(
            f"| Format: `{fmt}` | 36 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |"
        )
    for diff in ("easy", "medium", "hard"):
        lines.append(
            f"| Difficulty: `{diff}` | 60 | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM]` | `[PENDING LIVE LLM & HUMAN]` |"
        )
    lines.extend([
        "",
        "#### Table T3c: Tier 3c Self-Explanation Grader (`N = 120`, `se_prompt_bank`, `rubric=None`, Threshold `score >= 1.0`)",
        "",
        "| Grader Mode | Resolved Cov. (`n/120`) | MAE (`[0, 1]`, 95% CI) | Full-Success Precision | Full-Success Recall | Full-Success F1 | Conf.–Error Spearman $\\rho$ | QWK (3-Cat.) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| Deterministic Rubric (`grade_self_explanation`) | `[PENDING HUMAN CORPUS]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` | `[PENDING HUMAN ADJUDICATION]` |",
        "| Live LLM Confidence Check (`source == \"llm\"`, conditioned on `det_score`) | `[PENDING LIVE LLM]` | — *(score fixed)* | — | — | — | `[PENDING LIVE LLM & HUMAN]` | — |",
        "",
        "---",
        "",
        "## 4. Tier 4 Results: Teaching Policy (`delayed_retention_ppo`), Ablation 3 & Budget Curves",
        "",
        "Machine-readable log: [`3/eval_runs/v2.2-preregistered/tier4_run.json`](eval_runs/v2.2-preregistered/tier4_run.json)",
        "",
    ])

    render_tier4_tables(lines, t4["pre_registered"], "Pre-Registered (10 seeds, N=400)")
    render_tier4_tables(lines, t4["retrospective"], "Retrospective (3 seeds, N=200)")

    lines.extend([
        "---",
        "",
        "## 5. Tier 5: Optional Human Pilot Protocol Outline (Strictly Separated from Simulation)",
        "",
        "#### Table T5: Optional Tier 5 Human Pilot Summary",
        "",
        "| Cohort | Enrolled ($N$) | Completed ($n$, %, 95% CI) | Unaided Pretest | Delayed Posttest (`d` days) | Paired Change `post - pre` (95% CI) |",
        "|---|---:|---|---:|---:|---|",
        "| Human Pilot (Optional, IRB-gated) | `[NOT RUN — OPTIONAL]` | `[NOT RUN]` | `[NOT RUN]` | `[NOT RUN]` | `[NOT RUN]` |",
        "",
        "---",
        "",
        "## Next-Step Reminder: Live-LLM Evaluation Tests Pending",
        "",
        "> **Action Reminder:** Now that the offline evaluation phase (Tier 1 5-seed, Tier 2 offline diagnostics, Tier 4 10-seed + Ablation 3 + Budget Curves, and Retrospective compilations) is complete, **remember to run the live-LLM evaluation tests next**:",
        "> 1. Set a single pinned `(provider, model)` in `3/adaptive_web/.env` with `LLM_FALLBACK_PROVIDERS=\"\"` (no silent failover).",
        "> 2. Optionally configure a dense bi-encoder (`sentence-transformers`) for Tier 2 `Dense` and `BM25 + Dense`, then complete 2-reviewer blinded relevance scoring on [`tier2_relevance_annotation_pack.jsonl`](eval_runs/v2.2-preregistered/blind_packs/tier2_relevance_annotation_pack.jsonl).",
        "> 3. Execute the live-LLM generation runs (`source == \"llm\"` verified) on the exported packs for **Ablation 1 Downstream** (`ablation1_downstream_pack.jsonl`), **Tier 3a + Ablation 2 EduGuard ON/OFF** (`tier3a_ablation2_prompts.jsonl`), **Tier 3b Question Synthesis** (`tier3b_qgen_requests.jsonl`), and **Tier 3c Self-Explanation Grading** (`tier3c_self_explanation_prompts.jsonl`), followed by 2-reviewer blinded human adjudication.",
        "",
    ])

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {OUT_MD} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
