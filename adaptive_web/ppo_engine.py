"""Generalized 8-Action Gated Option B PPO Engine backed by frozen 3-seed delayed_retention_ppo."""
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

from . import CODE_3_ROOT, catalogue_bridge

MODEL_ROOT = CODE_3_ROOT / "model"
for _path in (str(CODE_3_ROOT), str(MODEL_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from ppo.agent import MaskedActorCritic  # noqa: E402
from ppo.env import (  # noqa: E402
    ACTIONS,
    ACTION_COSTS,
    DIFFICULTIES,
    OBS_DIM,
    PREREQUISITES,
    SKILLS_ORDER,
    TOTAL_BUDGET,
)

if tuple(SKILLS_ORDER) != catalogue_bridge.PPO_SKILLS_ORDER:
    raise RuntimeError(
        f"Skill order mismatch: ppo.env.SKILLS_ORDER={SKILLS_ORDER!r} != {catalogue_bridge.PPO_SKILLS_ORDER!r}"
    )

PPO_RUN_DIR = CODE_3_ROOT / "ppo" / "runs" / "v2.2-ppo-main" / "delayed_retention_ppo"
DEFAULT_CHECKPOINTS: tuple[Path, ...] = (
    PPO_RUN_DIR / "seed_20261120.pt",
    PPO_RUN_DIR / "seed_20261121.pt",
    PPO_RUN_DIR / "seed_20261122.pt",
)

ACTION_META: dict[int, dict[str, str]] = {
    0: {
        "name": "give_easier_question",
        "label": "Give easier question",
        "short": "Easier question",
        "category": "question",
        "badge": "Step down difficulty",
        "description": "Steps difficulty down one level (Hard -> Medium -> Easy) to rebuild foundational accuracy.",
    },
    1: {
        "name": "give_same_difficulty_question",
        "label": "Give same-difficulty question",
        "short": "Same difficulty",
        "category": "question",
        "badge": "Practice / Verify",
        "description": "Drills at current difficulty; when the last pass used hints, administers an unaided verification check.",
    },
    2: {
        "name": "give_harder_question",
        "label": "Give harder question",
        "short": "Harder question",
        "category": "question",
        "badge": "Step up challenge",
        "description": "Steps difficulty up one level (Easy -> Medium -> Hard) for higher learning dose and permastore consolidation.",
    },
    3: {
        "name": "explain_concept",
        "label": "Explain concept",
        "short": "Concept lesson",
        "category": "lesson",
        "badge": "Conceptual priming",
        "description": "Delivers a concise conceptual lesson and primes the next question (Gated Option B forces a follow-up question).",
    },
    4: {
        "name": "show_worked_example",
        "label": "Show worked example",
        "short": "Worked example",
        "category": "lesson",
        "badge": "Procedural priming",
        "description": "Walks through a step-by-step trace example and primes the next question (Gated Option B forces a follow-up question).",
    },
    5: {
        "name": "request_self_explanation",
        "label": "Request self-explanation",
        "short": "Self-explanation",
        "category": "consolidation",
        "badge": "Consolidate floor b_t",
        "description": "Post-question active unaided check that locks transient fluency k_t into durable permastore memory b_t.",
    },
    6: {
        "name": "review_prerequisite",
        "label": "Review prerequisite",
        "short": "Review prerequisite",
        "category": "curriculum",
        "badge": "Unblock bottleneck",
        "description": "Switches to a weak prerequisite or decaying earlier skill for a targeted refresher question.",
    },
    7: {
        "name": "advance_next_topic",
        "label": "Advance to next topic",
        "short": "Advance topic",
        "category": "curriculum",
        "badge": "Curriculum progress",
        "description": "Advances to the next unvisited Python skill (or lowest 30-day retention skill) once current topic is practiced.",
    },
}

OBS_GROUPS_META: tuple[dict[str, Any], ...] = (
    {"key": "memory_retention", "label": "Memory & Retention (monotonic_gru)", "start": 0, "end": 8},
    {"key": "recent_attempt_context", "label": "Recent Attempt & Assistance", "start": 8, "end": 13},
    {"key": "pedagogical_gating_state", "label": "Gated Option B & Difficulty", "start": 13, "end": 19},
    {"key": "curriculum_and_budget", "label": "Prerequisites, Budget & Global Min/Mean", "start": 19, "end": 24},
    {"key": "per_skill_p30d", "label": "Per-Skill 30-Day Retention (PPO Order)", "start": 24, "end": 30},
    {"key": "active_topic_one_hot", "label": "Active Topic One-Hot (PPO Order)", "start": 30, "end": 36},
)

OBS_FEATURE_SPEC: tuple[tuple[int, str, str, str], ...] = (
    (0, "p_easy", "P(Easy, 0d)", "memory_retention"),
    (1, "p_med", "P(Medium, 0d)", "memory_retention"),
    (2, "p_hard", "P(Hard, 0d)", "memory_retention"),
    (3, "p_7d", "P(Medium, 7d)", "memory_retention"),
    (4, "p_30d", "P(Medium, 30d)", "memory_retention"),
    (5, "fragility", "Fragility (P_med - P_30d)", "memory_retention"),
    (6, "floor_b", "Permastore floor b_t", "memory_retention"),
    (7, "active_k", "Active skill channel k_t", "memory_retention"),
    (8, "unverified_assisted", "Unverified assisted pass flag", "recent_attempt_context"),
    (9, "last_score", "Last resolved score", "recent_attempt_context"),
    (10, "last_hints_norm", "Last hints used / 3", "recent_attempt_context"),
    (11, "last_leakage", "Last hint leakage l_t", "recent_attempt_context"),
    (12, "attempts_on_topic_norm", "Topic attempts / 10", "recent_attempt_context"),
    (13, "awaiting_question", "Awaiting follow-up question", "pedagogical_gating_state"),
    (14, "primed_concept", "Primed: concept lesson", "pedagogical_gating_state"),
    (15, "primed_example", "Primed: worked example", "pedagogical_gating_state"),
    (16, "diff_easy", "Difficulty = Easy", "pedagogical_gating_state"),
    (17, "diff_medium", "Difficulty = Medium", "pedagogical_gating_state"),
    (18, "diff_hard", "Difficulty = Hard", "pedagogical_gating_state"),
    (19, "prereq_min_p_med", "Min prerequisite P(Med)", "curriculum_and_budget"),
    (20, "prereq_min_floor_b", "Min prerequisite floor b_t", "curriculum_and_budget"),
    (21, "budget_remaining_ratio", "Remaining budget ratio", "curriculum_and_budget"),
    (22, "min_p30_all", "Min P(30d) across 6 skills", "curriculum_and_budget"),
    (23, "mean_p30_all", "Mean P(30d) across 6 skills", "curriculum_and_budget"),
    *(
        (24 + i, f"p30d_{s}", f"P(30d) {catalogue_bridge.SKILL_META[s]['name']}", "per_skill_p30d")
        for i, s in enumerate(SKILLS_ORDER)
    ),
    *(
        (30 + i, f"topic_{s}", f"Active: {catalogue_bridge.SKILL_META[s]['name']}", "active_topic_one_hot")
        for i, s in enumerate(SKILLS_ORDER)
    ),
)


def format_obs_groups(obs_vec: Sequence[float]) -> list[dict[str, Any]]:
    if len(obs_vec) != OBS_DIM:
        raise ValueError(f"Expected {OBS_DIM}-dim observation vector, got {len(obs_vec)}")
    groups: list[dict[str, Any]] = []
    for meta in OBS_GROUPS_META:
        s_idx, e_idx = meta["start"], meta["end"]
        feats = [
            {
                "index": idx,
                "key": key,
                "label": label,
                "value": float(obs_vec[idx]),
            }
            for idx, key, label, gkey in OBS_FEATURE_SPEC
            if gkey == meta["key"]
        ]
        groups.append(
            {
                "key": meta["key"],
                "label": meta["label"],
                "start": s_idx,
                "end": e_idx,
                "features": feats,
            }
        )
    return groups


def _estimate_hint_leakage(hint_count: int, revealed: bool = False) -> float:
    if revealed:
        return 0.95
    if hint_count <= 0:
        return 0.0
    if hint_count == 1:
        return 0.25
    if hint_count == 2:
        return 0.55
    return 0.80


def _extract_item_leakage(item: dict[str, Any], hints_used: int, revealed: bool) -> float:
    hints = item.get("hints", [])
    explicit_vals = [
        float(h.get("leakage_l_t", h.get("leakage")))
        for h in hints
        if isinstance(h, dict) and (h.get("leakage_l_t") is not None or h.get("leakage") is not None)
    ]
    if revealed:
        return max(0.95, max(explicit_vals, default=0.95))
    if explicit_vals:
        return float(max(0.0, min(1.0, max(explicit_vals))))
    return _estimate_hint_leakage(hints_used, revealed=False)


class PPOEngine:
    """Evaluates the 36-dim observation vector and 8-action Gated Option B 3-seed PPO ensemble."""

    def __init__(
        self,
        memory_adapter: Any,
        checkpoint_paths: Sequence[str | Path] = DEFAULT_CHECKPOINTS,
    ):
        self.memory_adapter = memory_adapter
        self.device = memory_adapter.device
        paths = [Path(p) for p in checkpoint_paths if Path(p).is_file()]
        if not paths:
            raise FileNotFoundError(f"No trained PPO checkpoints found in {PPO_RUN_DIR}")
        self.checkpoint_paths = tuple(paths)
        self.models: list[MaskedActorCritic] = []
        self.seeds: list[int] = []

        hasher = hashlib.sha256()
        for path in self.checkpoint_paths:
            hasher.update(path.read_bytes())
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = MaskedActorCritic(obs_dim=OBS_DIM, act_dim=len(ACTIONS)).to(self.device)
            model.load_state_dict(payload["weights"])
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.models.append(model)
            self.seeds.append(int(payload.get("seed", 0)))
        self.version = hasher.hexdigest()

    def reconstruct_context(
        self,
        store: Any,
        interactions: Sequence[dict[str, Any]],
        all_events: Sequence[dict[str, Any]],
        focus_skill: str | None = None,
        focus_difficulty: str | None = None,
    ) -> dict[str, Any]:
        attempts_on_topic = {s: 0 for s in SKILLS_ORDER}
        last_score = {s: 0.5 for s in SKILLS_ORDER}
        last_hints = {s: 0.0 for s in SKILLS_ORDER}
        last_leakage = {s: 0.0 for s in SKILLS_ORDER}
        unverified_assisted = {s: False for s in SKILLS_ORDER}
        can_self_explain = {s: False for s in SKILLS_ORDER}
        last_difficulty_by_skill = {s: "easy" for s in SKILLS_ORDER}
        visited_topics: set[str] = set()
        budget_spent = 0.0
        last_active_skill = SKILLS_ORDER[0]

        for item in interactions:
            skill = item.get("skill")
            if skill not in attempts_on_topic:
                continue

            # Generalized lookup covering BOTH v2.2 catalog and SQLite generated_tasks
            task_id = item.get("task")
            task = store.get_task(task_id) if (store is not None and task_id) else None
            diff = (task.get("difficulty") if task else None) or item.get("difficulty")
            if diff in DIFFICULTIES:
                last_difficulty_by_skill[skill] = diff

            if item.get("ppo_action") == "request_self_explanation":
                can_self_explain[skill] = False

            resolved = [r for r in item.get("results", []) if r.get("grading_status") == "resolved"]
            if not resolved:
                continue

            last_active_skill = skill
            visited_topics.add(skill)
            attempts_on_topic[skill] += 1
            final_res = resolved[-1]
            score = float(final_res["score"] or 0.0)
            full = bool(final_res["full_success"])
            hints_used = len(item.get("hints", []))
            revealed = bool(item.get("revealed", False))
            leak = _extract_item_leakage(item, hints_used, revealed)

            last_score[skill] = score
            if item.get("ppo_action") == "request_self_explanation":
                budget_spent += ACTION_COSTS[5]
                can_self_explain[skill] = False
                # Match 3/ppo/env.py:728-729 self-explanation reset of last_hints/last_leakage
                last_hints[skill] = 0.0
                last_leakage[skill] = 0.0
                if full and hints_used == 0 and not revealed:
                    unverified_assisted[skill] = False
            else:
                budget_spent += ACTION_COSTS[1]
                can_self_explain[skill] = True
                last_hints[skill] = float(hints_used)
                last_leakage[skill] = float(leak)
                if hints_used == 0 and not revealed:
                    if full:
                        unverified_assisted[skill] = False
                else:
                    if full or leak > 0.25:
                        unverified_assisted[skill] = True

        resolved_focus = focus_skill if focus_skill in SKILLS_ORDER else last_active_skill
        visited_topics.add(resolved_focus)
        topic_idx = SKILLS_ORDER.index(resolved_focus)

        diff_name = (
            focus_difficulty
            if focus_difficulty in DIFFICULTIES
            else last_difficulty_by_skill[resolved_focus]
        )
        difficulty_idx = DIFFICULTIES.index(diff_name)

        for ev in all_events:
            ev_skill = ev.get("skill")
            kind = ev.get("kind")
            if kind == "instruction":
                act_name = ev.get("ppo_action")
                p_mode = ev.get("primed_mode")
                if act_name == "explain_concept" or p_mode == "concept":
                    budget_spent += ACTION_COSTS[3]
                else:
                    budget_spent += ACTION_COSTS[4]
                if ev_skill in SKILLS_ORDER:
                    visited_topics.add(ev_skill)
            elif kind == "advance":
                budget_spent += ACTION_COSTS[7]
                if ev_skill in SKILLS_ORDER:
                    visited_topics.add(ev_skill)
                target_sk = ev.get("target_skill")
                if target_sk in SKILLS_ORDER:
                    visited_topics.add(target_sk)

        skill_events = [e for e in all_events if e.get("skill") == resolved_focus]
        awaiting_question = False
        primed_mode: str | None = None
        if skill_events and skill_events[-1].get("kind") == "instruction":
            last_ev = skill_events[-1]
            awaiting_question = True
            primed_mode = last_ev.get("primed_mode") or (
                "concept" if last_ev.get("ppo_action") == "explain_concept" else "example"
            )

        budget_used = budget_spent % TOTAL_BUDGET if budget_spent >= TOTAL_BUDGET else budget_spent
        budget_remaining_ratio = float(max(0.0, (TOTAL_BUDGET - budget_used) / TOTAL_BUDGET))

        return {
            "focus_skill": resolved_focus,
            "topic_idx": topic_idx,
            "difficulty": diff_name,
            "difficulty_idx": difficulty_idx,
            "last_difficulty_by_skill": last_difficulty_by_skill,
            "attempts_on_topic": attempts_on_topic,
            "last_score": last_score,
            "last_hints": last_hints,
            "last_leakage": last_leakage,
            "unverified_assisted": unverified_assisted,
            "can_self_explain": can_self_explain,
            "visited_topics": visited_topics,
            "awaiting_question": awaiting_question,
            "primed_mode": primed_mode,
            "budget_used": float(budget_used),
            "budget_total": float(TOTAL_BUDGET),
            "budget_remaining_ratio": budget_remaining_ratio,
        }

    def weak_prereqs_or_decayed(
        self,
        skill: str,
        metrics_by_skill: dict[str, dict[str, Any]],
        visited_topics: set[str],
    ) -> tuple[list[str], str]:
        prereqs = [
            p
            for p in PREREQUISITES[skill]
            if metrics_by_skill[p]["p_30d"] < 0.56 or metrics_by_skill[p]["floor_b"] < 0.40
        ]
        if prereqs:
            return prereqs, "prerequisite"
        decayed = [
            s
            for s in SKILLS_ORDER
            if s in visited_topics
            and s != skill
            and (metrics_by_skill[s]["p_30d"] < 0.52 or metrics_by_skill[s]["floor_b"] < 0.36)
        ]
        return decayed, "decayed_visited"

    def valid_action_mask_and_reasons(
        self,
        ctx: dict[str, Any],
        metrics_by_skill: dict[str, dict[str, Any]],
    ) -> tuple[np.ndarray, dict[int, str]]:
        skill = ctx["focus_skill"]
        diff_idx = ctx["difficulty_idx"]
        mask = np.zeros(len(ACTIONS), dtype=bool)
        reasons: dict[int, str] = {}

        if diff_idx > 0:
            mask[0] = True
        else:
            reasons[0] = "Already at Easy difficulty."
        mask[1] = True
        if diff_idx < 2:
            mask[2] = True
        else:
            reasons[2] = "Already at Hard difficulty."

        if ctx["awaiting_question"]:
            for aid in (3, 4, 5, 6, 7):
                reasons[aid] = (
                    "Gated Option B: Lesson delivered — choose follow-up question difficulty "
                    "(Easier, Same, or Harder) first."
                )
            return mask, reasons

        if ctx["primed_mode"] is None:
            mask[3] = True
            mask[4] = True
        else:
            reasons[3] = "Already primed by an instruction lesson."
            reasons[4] = "Already primed by an instruction lesson."

        if ctx["can_self_explain"][skill]:
            mask[5] = True
        else:
            reasons[5] = "Unlocks right after completing a question turn on this skill (locks after one use)."

        weak_targets, _ = self.weak_prereqs_or_decayed(skill, metrics_by_skill, ctx["visited_topics"])
        if weak_targets:
            mask[6] = True
        else:
            skill_label = catalogue_bridge.SKILL_META[skill]["name"]
            reasons[6] = (
                f"No prerequisites for {skill_label} (and no decayed visited skills yet)."
                if not PREREQUISITES[skill]
                else "All prerequisites and visited skills exceed target retention/floor thresholds."
            )

        if ctx["attempts_on_topic"][skill] >= 1:
            mask[7] = True
        else:
            reasons[7] = "Complete at least 1 graded question on this topic before advancing."

        return mask, reasons

    def build_observation(
        self,
        ctx: dict[str, Any],
        metrics_by_skill: dict[str, dict[str, Any]],
    ) -> tuple[np.ndarray, float, float]:
        skill = ctx["focus_skill"]
        m = metrics_by_skill[skill]
        prereqs = PREREQUISITES[skill]
        if prereqs:
            pre_med = min(metrics_by_skill[p]["p_med"] for p in prereqs)
            pre_floor = min(metrics_by_skill[p]["floor_b"] for p in prereqs)
        else:
            pre_med = 1.0
            pre_floor = 1.0

        p30_all = [metrics_by_skill[s]["p_30d"] for s in SKILLS_ORDER]
        diff_one_hot = [float(ctx["difficulty_idx"] == i) for i in range(3)]
        topic_one_hot = [float(ctx["topic_idx"] == i) for i in range(len(SKILLS_ORDER))]

        vec = np.array(
            [
                m["p_easy"],
                m["p_med"],
                m["p_hard"],
                m["p_7d"],
                m["p_30d"],
                m["fragility"],
                m["floor_b"],
                m["active_k"],
                float(ctx["unverified_assisted"][skill]),
                ctx["last_score"][skill],
                min(1.0, ctx["last_hints"][skill] / 3.0),
                min(1.0, max(0.0, ctx["last_leakage"][skill])),
                min(1.0, ctx["attempts_on_topic"][skill] / 10.0),
                float(ctx["awaiting_question"]),
                float(ctx["primed_mode"] == "concept"),
                float(ctx["primed_mode"] == "example"),
                *diff_one_hot,
                pre_med,
                pre_floor,
                ctx["budget_remaining_ratio"],
                min(p30_all),
                float(np.mean(p30_all)),
                *p30_all,
                *topic_one_hot,
            ],
            dtype=np.float32,
        )
        if vec.shape != (OBS_DIM,):
            raise RuntimeError(f"Unexpected observation shape {vec.shape}")
        vec = np.clip(vec, 0.0, 1.0)
        return vec, float(pre_med), float(pre_floor)

    def resolve_action_target(
        self,
        action_id: int,
        ctx: dict[str, Any],
        metrics_by_skill: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        skill = ctx["focus_skill"]
        diff_idx = ctx["difficulty_idx"]
        if action_id == 0:
            return {
                "kind": "question",
                "skill": skill,
                "difficulty": DIFFICULTIES[max(0, diff_idx - 1)],
                "mode": "practice",
            }
        if action_id == 1:
            mode = "assessment" if ctx["unverified_assisted"][skill] else "practice"
            return {
                "kind": "question",
                "skill": skill,
                "difficulty": DIFFICULTIES[diff_idx],
                "mode": mode,
            }
        if action_id == 2:
            return {
                "kind": "question",
                "skill": skill,
                "difficulty": DIFFICULTIES[min(2, diff_idx + 1)],
                "mode": "practice",
            }
        if action_id == 3:
            return {
                "kind": "lesson",
                "skill": skill,
                "difficulty": DIFFICULTIES[diff_idx],
                "mode": "lesson",
                "primed_mode": "concept",
                "lesson": catalogue_bridge.DEFAULT_LESSONS[skill]["concept"],
            }
        if action_id == 4:
            return {
                "kind": "lesson",
                "skill": skill,
                "difficulty": DIFFICULTIES[diff_idx],
                "mode": "lesson",
                "primed_mode": "example",
                "lesson": catalogue_bridge.DEFAULT_LESSONS[skill]["worked_example"],
            }
        if action_id == 5:
            return {
                "kind": "question",
                "skill": skill,
                "difficulty": "medium",
                "mode": "assessment",
                "self_explanation_prompt": catalogue_bridge.DEFAULT_LESSONS[skill]["self_explanation_prompt"],
            }
        if action_id == 6:
            weak_targets, reason_kind = self.weak_prereqs_or_decayed(
                skill, metrics_by_skill, ctx["visited_topics"]
            )
            target_skill = (
                min(weak_targets, key=lambda s: metrics_by_skill[s]["p_30d"])
                if weak_targets
                else skill
            )
            target_diff = "easy" if metrics_by_skill[target_skill]["p_med"] < 0.48 else "medium"
            return {
                "kind": "question",
                "skill": target_skill,
                "difficulty": target_diff,
                "mode": "practice",
                "review_reason": reason_kind,
            }
        if action_id == 7:
            unvisited = [
                s
                for s in SKILLS_ORDER
                if s not in ctx["visited_topics"] or ctx["attempts_on_topic"][s] == 0
            ]
            if unvisited:
                target_skill = unvisited[0]
            else:
                others = [s for s in SKILLS_ORDER if s != skill]
                target_skill = min(others, key=lambda s: metrics_by_skill[s]["p_30d"])
            target_diff = "easy" if metrics_by_skill[target_skill]["p_med"] < 0.55 else "medium"
            return {
                "kind": "question",
                "skill": target_skill,
                "difficulty": target_diff,
                "mode": "practice",
            }
        raise ValueError(f"Unknown PPO action ID: {action_id}")

    def _build_rationales(
        self,
        action_id: int,
        target: dict[str, Any],
        ctx: dict[str, Any],
        metrics_by_skill: dict[str, dict[str, Any]],
        pre_med: float,
        pre_floor: float,
        ensemble_prob: float,
        critic_value: float,
    ) -> tuple[str, str]:
        skill = ctx["focus_skill"]
        skill_name = catalogue_bridge.SKILL_META[skill]["name"]
        target_skill_name = catalogue_bridge.SKILL_META[target["skill"]]["name"]
        target_diff_cap = target["difficulty"].capitalize()
        m = metrics_by_skill[skill]

        if action_id == 0:
            plain = (
                f"Let's step down to an {target_diff_cap} question on {skill_name} so you can "
                f"build solid confidence before tackling harder problems."
            )
        elif action_id == 1:
            if ctx["unverified_assisted"][skill]:
                plain = (
                    f"Your last pass on {skill_name} used hints, so let's try an unaided "
                    f"{target_diff_cap} check to confirm you have it down independently."
                )
            else:
                plain = (
                    f"You're right in the productive zone for {skill_name} — let's practice "
                    f"another {target_diff_cap} question to strengthen recall."
                )
        elif action_id == 2:
            primed_note = " with your lesson priming active" if ctx["primed_mode"] else ""
            plain = (
                f"You're ready for a stronger challenge{primed_note}: stepping up to a "
                f"{target_diff_cap} {skill_name} question."
            )
        elif action_id == 3:
            plain = (
                f"Before testing {skill_name}, let's review the core concept rules first so "
                f"the next question sticks better."
            )
        elif action_id == 4:
            plain = (
                f"Let's walk through a step-by-step worked example for {skill_name} before "
                f"your next practice question."
            )
        elif action_id == 5:
            plain = (
                f"Pause and explain the reasoning behind {skill_name} in your own words — "
                f"this locks short-term practice into durable long-term memory."
            )
        elif action_id == 6:
            plain = (
                f"Let's briefly reinforce {target_skill_name} ({target_diff_cap}) first, since "
                f"strengthening that foundation will make {skill_name} much smoother."
            )
        else:
            plain = (
                f"Great work on {skill_name}! Let's advance to {target_skill_name} "
                f"({target_diff_cap}) to keep building across the full Python DAG."
            )

        technical = (
            f"Policy=delayed_retention_ppo action={ACTIONS[action_id]} (p={ensemble_prob:.3f}, "
            f"V(s_t)={critic_value:.3f}); focus={skill}[{ctx['difficulty']}], "
            f"P(Med)={m['p_med']:.3f}, P(30d)={m['p_30d']:.3f}, fragility={m['fragility']:.3f}, "
            f"k_t={m['active_k']:.3f}, b_t={m['floor_b']:.3f}, "
            f"unverified_assisted={ctx['unverified_assisted'][skill]}, "
            f"last_leakage={ctx['last_leakage'][skill]:.2f}, primed={ctx['primed_mode']}, "
            f"prereq_min_p_med={pre_med:.3f}, prereq_min_b={pre_floor:.3f}, "
            f"budget={ctx['budget_used']:.1f}/{TOTAL_BUDGET:.1f}."
        )
        return plain, technical

    @torch.inference_mode()
    def decide(
        self,
        store: Any,
        learner: str,
        focus_skill: str | None = None,
        now: float | None = None,
        focus_difficulty: str | None = None,
    ) -> dict[str, Any]:
        query_now = float(time.time() if now is None else now)
        all_events = store.get_events(learner) if store is not None else []
        interactions = store.get_interactions(learner) if store is not None else []

        metrics_by_skill: dict[str, dict[str, Any]] = {}
        for skill in SKILLS_ORDER:
            skill_ev = [e for e in all_events if e.get("skill") == skill]
            metrics_by_skill[skill] = self.memory_adapter.skill_summary(
                learner,
                skill,
                skill_ev,
                now=query_now,
            )

        ctx = self.reconstruct_context(
            store,
            interactions,
            all_events,
            focus_skill=focus_skill,
            focus_difficulty=focus_difficulty,
        )
        obs_vec, pre_med, pre_floor = self.build_observation(ctx, metrics_by_skill)
        valid_mask, mask_reasons = self.valid_action_mask_and_reasons(ctx, metrics_by_skill)

        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        seed_probs: list[np.ndarray] = []
        seed_values: list[float] = []
        for model in self.models:
            logits = model.actor(obs_t).masked_fill(~mask_t, -1e9)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            val = float(model.critic(obs_t).squeeze(-1).item())
            seed_probs.append(probs)
            seed_values.append(val)

        mean_probs = np.mean(seed_probs, axis=0)
        mean_probs = np.where(valid_mask, mean_probs, 0.0)
        total_p = float(mean_probs.sum())
        if total_p > 0.0:
            mean_probs = mean_probs / total_p

        best_action_id = int(np.argmax(mean_probs))
        best_target = self.resolve_action_target(best_action_id, ctx, metrics_by_skill)
        critic_value = float(np.mean(seed_values))
        rationale_plain, rationale_technical = self._build_rationales(
            best_action_id,
            best_target,
            ctx,
            metrics_by_skill,
            pre_med,
            pre_floor,
            float(mean_probs[best_action_id]),
            critic_value,
        )

        actions_list: list[dict[str, Any]] = []
        for aid in range(len(ACTIONS)):
            meta = ACTION_META[aid]
            is_valid = bool(valid_mask[aid])
            target = self.resolve_action_target(aid, ctx, metrics_by_skill) if is_valid else None
            actions_list.append(
                {
                    "id": aid,
                    "name": meta["name"],
                    "label": meta["label"],
                    "short": meta["short"],
                    "category": meta["category"],
                    "badge": meta["badge"],
                    "description": meta["description"],
                    "cost": float(ACTION_COSTS[aid]),
                    "valid": is_valid,
                    "probability": float(mean_probs[aid]),
                    "per_seed_probabilities": [float(sp[aid]) for sp in seed_probs],
                    "recommended": aid == best_action_id,
                    "mask_reason": None if is_valid else mask_reasons.get(aid, "Masked in current state."),
                    "target": (
                        {
                            "kind": target["kind"],
                            "skill": target["skill"],
                            "skill_name": catalogue_bridge.SKILL_META[target["skill"]]["name"],
                            "difficulty": target["difficulty"],
                            "mode": target.get("mode", "practice"),
                        }
                        if target
                        else None
                    ),
                }
            )

        per_seed_probs = {
            str(seed): [float(p) for p in probs]
            for seed, probs in zip(self.seeds, seed_probs)
        }
        obs_list = [float(x) for x in obs_vec.tolist()]

        return {
            "policy_name": "delayed_retention_ppo",
            "seeds": list(self.seeds),
            "ppo_sha256": self.version,
            "obs_vector_36": obs_list,
            "obs_groups": format_obs_groups(obs_list),
            "valid_mask": [bool(x) for x in valid_mask.tolist()],
            "mask_reasons": dict(mask_reasons),
            "ensemble_probs": [float(x) for x in mean_probs.tolist()],
            "per_seed_probs": per_seed_probs,
            "critic_value": critic_value,
            "action_id": best_action_id,
            "action_name": ACTIONS[best_action_id],
            "target_skill": best_target["skill"],
            "target_difficulty": best_target["difficulty"],
            "mode": best_target.get("mode", "practice"),
            "target": best_target,
            "awaiting_question": bool(ctx["awaiting_question"]),
            "primed_mode": ctx["primed_mode"],
            "unverified_assisted": bool(ctx["unverified_assisted"][ctx["focus_skill"]]),
            "unverified_assisted_by_skill": dict(ctx["unverified_assisted"]),
            "can_self_explain": bool(ctx["can_self_explain"][ctx["focus_skill"]]),
            "can_self_explain_by_skill": dict(ctx["can_self_explain"]),
            "budget_used": float(ctx["budget_used"]),
            "budget_total": float(TOTAL_BUDGET),
            "budget_remaining_ratio": float(ctx["budget_remaining_ratio"]),
            "focus_skill": ctx["focus_skill"],
            "focus_difficulty": ctx["difficulty"],
            "last_difficulty_by_skill": dict(ctx["last_difficulty_by_skill"]),
            "metrics_by_skill": metrics_by_skill,
            "rationale_plain": rationale_plain,
            "rationale_technical": rationale_technical,
            "actions": actions_list,
        }
