"""Live 8-Action Gated Option B PPO Policy Service backed by frozen v2.2 monotonic_gru."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

from catalogue import SKILLS, TASKS, LESSONS, features
from memory.features import DAY, task_features

ROOT = Path(__file__).resolve().parents[1]
PPO_ROOT = ROOT / "ppo"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

PPO_RUN_DIR = PPO_ROOT / "runs/v2.2-ppo-main/delayed_retention_ppo"
DEFAULT_CHECKPOINTS = (
    PPO_RUN_DIR / "seed_20261120.pt",
    PPO_RUN_DIR / "seed_20261121.pt",
    PPO_RUN_DIR / "seed_20261122.pt",
)

ACTION_META = {
    0: {
        "name": "give_easier_question",
        "label": "Give easier question",
        "short": "Easier question",
        "category": "question",
        "badge": "Step down difficulty",
        "description": "Steps difficulty down one level (Hard → Medium → Easy) to rebuild foundational accuracy.",
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
        "description": "Steps difficulty up one level (Easy → Medium → Hard) for higher learning dose and permastore consolidation.",
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


def _estimate_hint_leakage(hint_count, revealed=False):
    if revealed:
        return 0.95
    if hint_count <= 0:
        return 0.0
    if hint_count == 1:
        return 0.25
    if hint_count == 2:
        return 0.55
    return 0.80


class PPOTutorService:
    """Evaluates the 36-dim observation state and 8-action Gated Option B PPO policy on live dashboard history."""

    def __init__(self, device="cpu", checkpoint_paths=DEFAULT_CHECKPOINTS):
        self.device = device
        paths = [Path(p) for p in checkpoint_paths if Path(p).exists()]
        if not paths:
            raise FileNotFoundError(f"No trained PPO checkpoints found in {PPO_RUN_DIR}")
        self.checkpoint_paths = paths
        self.models = []
        self.seeds = []
        hasher = hashlib.sha256()
        for path in paths:
            raw = path.read_bytes()
            hasher.update(raw)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = MaskedActorCritic(obs_dim=OBS_DIM, act_dim=len(ACTIONS)).to(self.device)
            model.load_state_dict(payload["weights"])
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.models.append(model)
            self.seeds.append(int(payload.get("seed", 0)))
        self.version = hasher.hexdigest()

        # Reference task tensors matching FrozenMemoryTracker in 3/ppo/env.py
        refs = {}
        for diff in DIFFICULTIES:
            match = next(
                t for t in TASKS.values()
                if t["skill_id"] == "range_boundaries"
                and t["difficulty"] == diff
                and t["task_type"] == "output_prediction"
                and t["pool"] == "assessment"
            )
            refs[diff] = task_features(features(match))
        self.ref_tasks_tensor = torch.tensor(
            [refs["easy"], refs["medium"], refs["hard"], refs["medium"], refs["medium"]],
            dtype=torch.float32,
            device=self.device,
        )
        self.ref_gaps_tensor = torch.tensor([0.0, 0.0, 0.0, 7.0, 30.0], dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def query_skill_metrics(self, memory_service, events, now, key=None):
        saved = memory_service.state(events, key=key)
        anchor = saved["anchor"]
        if anchor is None or not events:
            elapsed_days = 0.0
        else:
            elapsed_days = max(0.0, (now - events[0]["at"] - anchor) / DAY)
        state = saved["state"]
        states = state.expand(5, -1)
        gaps = self.ref_gaps_tensor + elapsed_days
        logits = memory_service.model.predict_logit(states, self.ref_tasks_tensor, gaps)
        p_easy, p_med, p_hard, p_7d, p_30d = torch.sigmoid(logits).tolist()

        decayed, _ = memory_service.model._at_time(
            state,
            torch.tensor([elapsed_days], dtype=torch.float32, device=self.device),
        )
        m = memory_service.model.skill_channels
        active_k = float(decayed[0, :m].mean().item())
        floor_b = float(decayed[0, m:2 * m].mean().item())
        return {
            "p_easy": float(p_easy),
            "p_med": float(p_med),
            "p_hard": float(p_hard),
            "p_7d": float(p_7d),
            "p_30d": float(p_30d),
            "fragility": float(max(0.0, p_med - p_30d)),
            "floor_b": floor_b,
            "active_k": active_k,
        }

    def reconstruct_context(self, interactions, all_events, focus_skill=None, focus_difficulty=None):
        attempts_on_topic = {s: 0 for s in SKILLS_ORDER}
        last_score = {s: 0.5 for s in SKILLS_ORDER}
        last_hints = {s: 0.0 for s in SKILLS_ORDER}
        last_leakage = {s: 0.0 for s in SKILLS_ORDER}
        unverified_assisted = {s: False for s in SKILLS_ORDER}
        can_self_explain = {s: False for s in SKILLS_ORDER}
        last_difficulty_by_skill = {s: "easy" for s in SKILLS_ORDER}
        visited_topics = set()
        budget_spent = 0.0

        last_active_skill = SKILLS_ORDER[0]
        for item in interactions:
            skill = item["skill"]
            if skill not in attempts_on_topic:
                continue
            task = TASKS.get(item["task"])
            if task:
                last_difficulty_by_skill[skill] = task["difficulty"]
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
            leak = _estimate_hint_leakage(hints_used, revealed=revealed)

            last_score[skill] = score
            last_hints[skill] = float(hints_used)
            last_leakage[skill] = float(leak)
            ppo_act = item.get("ppo_action")
            if ppo_act == "request_self_explanation":
                budget_spent += ACTION_COSTS[5]
                can_self_explain[skill] = False
                if full and hints_used == 0 and not revealed:
                    unverified_assisted[skill] = False
            else:
                budget_spent += 1.0
                can_self_explain[skill] = True
                if hints_used == 0 and not revealed:
                    if full:
                        unverified_assisted[skill] = False
                else:
                    if full or leak > 0.25:
                        unverified_assisted[skill] = True

        if focus_skill not in SKILLS_ORDER:
            focus_skill = last_active_skill
        visited_topics.add(focus_skill)
        topic_idx = SKILLS_ORDER.index(focus_skill)

        diff_name = focus_difficulty if focus_difficulty in DIFFICULTIES else last_difficulty_by_skill[focus_skill]
        difficulty_idx = DIFFICULTIES.index(diff_name)

        # Check if the latest event on focus_skill is an unconsumed PPO instruction lesson (Actions 3 or 4)
        awaiting_question = False
        primed_mode = None
        skill_events = [e for e in all_events if e["skill"] == focus_skill]
        for ev in all_events:
            if ev["kind"] == "instruction":
                act_name = ev.get("ppo_action")
                budget_spent += ACTION_COSTS[3] if act_name == "explain_concept" else ACTION_COSTS[4]
        if skill_events and skill_events[-1]["kind"] == "instruction":
            awaiting_question = True
            act_name = skill_events[-1].get("ppo_action")
            primed_mode = "concept" if act_name == "explain_concept" else "example"

        # Cycle budget modulo TOTAL_BUDGET (32.0) so open-ended profiles keep calibrated budget fraction
        budget_used = budget_spent % TOTAL_BUDGET if budget_spent >= TOTAL_BUDGET else budget_spent

        return {
            "focus_skill": focus_skill,
            "topic_idx": topic_idx,
            "difficulty": diff_name,
            "difficulty_idx": difficulty_idx,
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
            "budget_remaining_ratio": float(max(0.0, (TOTAL_BUDGET - budget_used) / TOTAL_BUDGET)),
        }

    def weak_prereqs_or_decayed(self, skill, metrics_by_skill, visited_topics):
        prereqs = [
            p for p in PREREQUISITES[skill]
            if metrics_by_skill[p]["p_30d"] < 0.56 or metrics_by_skill[p]["floor_b"] < 0.40
        ]
        if prereqs:
            return prereqs, "prerequisite"
        decayed = [
            s for s in SKILLS_ORDER
            if s in visited_topics
            and s != skill
            and (metrics_by_skill[s]["p_30d"] < 0.52 or metrics_by_skill[s]["floor_b"] < 0.36)
        ]
        return decayed, "decayed_visited"

    def valid_action_mask_and_reasons(self, ctx, metrics_by_skill):
        skill = ctx["focus_skill"]
        diff_idx = ctx["difficulty_idx"]
        mask = np.zeros(len(ACTIONS), dtype=bool)
        reasons = {}

        if ctx["awaiting_question"]:
            for a in (3, 4, 5, 6, 7):
                reasons[a] = "Gated Option B: Lesson delivered — choose follow-up question difficulty (Easier, Same, or Harder) first."
            if diff_idx > 0:
                mask[0] = True
            else:
                reasons[0] = "Already at Easy difficulty."
            mask[1] = True
            if diff_idx < 2:
                mask[2] = True
            else:
                reasons[2] = "Already at Hard difficulty."
            return mask, reasons

        if diff_idx > 0:
            mask[0] = True
        else:
            reasons[0] = "Already at Easy difficulty."
        mask[1] = True
        if diff_idx < 2:
            mask[2] = True
        else:
            reasons[2] = "Already at Hard difficulty."

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
            reasons[6] = (
                "No prerequisites for Range boundaries (and no decayed visited skills yet)."
                if not PREREQUISITES[skill]
                else "All prerequisites and visited skills exceed target retention/floor thresholds."
            )

        if ctx["attempts_on_topic"][skill] >= 1:
            mask[7] = True
        else:
            reasons[7] = "Complete at least 1 graded question on this topic before advancing."

        return mask, reasons

    def build_observation(self, ctx, metrics_by_skill):
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

        vec = np.array([
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
            ctx["last_leakage"][skill],
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
        ], dtype=np.float32)
        return vec, float(pre_med), float(pre_floor)

    def resolve_action_target(self, action_id, ctx, metrics_by_skill):
        skill = ctx["focus_skill"]
        diff_idx = ctx["difficulty_idx"]
        if action_id == 0:
            target_diff = DIFFICULTIES[max(0, diff_idx - 1)]
            return {"kind": "question", "skill": skill, "difficulty": target_diff, "mode": "practice"}
        if action_id == 1:
            target_diff = DIFFICULTIES[diff_idx]
            mode = "assessment" if ctx["unverified_assisted"][skill] else "practice"
            return {"kind": "question", "skill": skill, "difficulty": target_diff, "mode": mode}
        if action_id == 2:
            target_diff = DIFFICULTIES[min(2, diff_idx + 1)]
            return {"kind": "question", "skill": skill, "difficulty": target_diff, "mode": "practice"}
        if action_id == 3:
            return {
                "kind": "lesson",
                "skill": skill,
                "difficulty": DIFFICULTIES[diff_idx],
                "primed_mode": "concept",
                "lesson": LESSONS[skill]["concept"],
            }
        if action_id == 4:
            return {
                "kind": "lesson",
                "skill": skill,
                "difficulty": DIFFICULTIES[diff_idx],
                "primed_mode": "example",
                "lesson": LESSONS[skill]["worked_example"],
            }
        if action_id == 5:
            return {
                "kind": "question",
                "skill": skill,
                "difficulty": "medium",
                "mode": "assessment",
                "self_explanation_prompt": LESSONS[skill]["self_explanation_prompt"],
            }
        if action_id == 6:
            weak_targets, reason_kind = self.weak_prereqs_or_decayed(skill, metrics_by_skill, ctx["visited_topics"])
            target_skill = min(weak_targets, key=lambda s: metrics_by_skill[s]["p_30d"]) if weak_targets else skill
            target_diff = "easy" if metrics_by_skill[target_skill]["p_med"] < 0.48 else "medium"
            return {
                "kind": "question",
                "skill": target_skill,
                "difficulty": target_diff,
                "mode": "practice",
                "review_reason": reason_kind,
            }
        if action_id == 7:
            unvisited = [s for s in SKILLS_ORDER if s not in ctx["visited_topics"] or ctx["attempts_on_topic"][s] == 0]
            if unvisited:
                target_skill = unvisited[0]
            else:
                others = [s for s in SKILLS_ORDER if s != skill]
                target_skill = min(others, key=lambda s: metrics_by_skill[s]["p_30d"])
            target_diff = "easy" if metrics_by_skill[target_skill]["p_med"] < 0.55 else "medium"
            return {"kind": "question", "skill": target_skill, "difficulty": target_diff, "mode": "practice"}
        raise ValueError(f"Unknown PPO action ID: {action_id}")

    def _explain_choice(self, action_id, target, ctx, metrics_by_skill, pre_med, pre_floor):
        skill = ctx["focus_skill"]
        skill_name = SKILLS[skill][0]
        m = metrics_by_skill[skill]
        target_skill_name = SKILLS[target["skill"]][0]
        if action_id == 0:
            return (
                f"On {skill_name}, immediate P(Med) is {m['p_med']*100:.0f}% (last score {ctx['last_score'][skill]*100:.0f}%). "
                f"PPO steps down to an {target['difficulty'].capitalize()} question to build confident accuracy before increasing challenge."
            )
        if action_id == 1:
            if ctx["unverified_assisted"][skill]:
                return (
                    f"Last attempt on {skill_name} relied on assistance (estimated leakage {ctx['last_leakage'][skill]*100:.0f}%). "
                    f"PPO triggers a same-difficulty ({target['difficulty'].capitalize()}) Independent Check to verify real unaided recall."
                )
            return (
                f"On {skill_name}, P({target['difficulty'].capitalize()}) is in the productive practice zone "
                f"(P(Med)={m['p_med']*100:.0f}%, 30d={m['p_30d']*100:.0f}%, Floor b_t={m['floor_b']:.2f}). "
                f"PPO recommends another {target['difficulty'].capitalize()} question."
            )
        if action_id == 2:
            return (
                f"With P(Easy)={m['p_easy']*100:.0f}% and P(Med)={m['p_med']*100:.0f}% on {skill_name}"
                f"{' (plus active lesson priming)' if ctx['primed_mode'] else ''}, PPO steps up to a "
                f"{target['difficulty'].capitalize()} question for a stronger learning dose and faster permastore floor b_t growth."
            )
        if action_id == 3:
            return (
                f"On {skill_name} (active k_t={m['active_k']:.2f}, Floor b_t={m['floor_b']:.2f}), "
                f"PPO delivers a Concept Explanation first to prime conceptual understanding (1.45× learning dose on the follow-up question)."
            )
        if action_id == 4:
            return (
                f"On {skill_name} (P(Med)={m['p_med']*100:.0f}%, P(30d)={m['p_30d']*100:.0f}%), "
                f"PPO shows a Step-by-Step Worked Example to scaffold procedural trace fluency before testing you on the follow-up question."
            )
        if action_id == 5:
            return (
                f"Post-question consolidation opportunity on {skill_name}: P(Med) is {m['p_med']*100:.0f}% vs. 30-day retention "
                f"{m['p_30d']*100:.0f}% (Fragility gap {m['fragility']*100:.1f}%, Floor b_t={m['floor_b']:.2f}). "
                f"PPO requests an unaided Self-Explanation check to lock transient fluency into durable permastore memory."
            )
        if action_id == 6:
            return (
                f"Prerequisite / earlier-skill bottleneck detected for {skill_name} (min prerequisite P(Med)={pre_med*100:.0f}%, "
                f"Floor b_t={pre_floor:.2f}). PPO switches to {target_skill_name} ({target['difficulty'].capitalize()}) to strengthen the foundation."
            )
        return (
            f"You have practiced {skill_name} (P(30d)={m['p_30d']*100:.0f}%, Floor b_t={m['floor_b']:.2f}). "
            f"PPO advances the curriculum to {target_skill_name} ({target['difficulty'].capitalize()}) to maximize 6-skill delayed retention."
        )

    @torch.inference_mode()
    def recommend(self, memory_service, learner, interactions, all_events, now, focus_skill=None, focus_difficulty=None):
        metrics_by_skill = {}
        for skill in SKILLS_ORDER:
            ev = [e for e in all_events if e["skill"] == skill]
            metrics_by_skill[skill] = self.query_skill_metrics(
                memory_service,
                ev,
                now,
                key=(learner, skill, memory_service.version),
            )
            metrics_by_skill[skill]["assessed"] = any(
                e["kind"] == "grade" and e["features"]["grading_status"] == "resolved" for e in ev
            )

        ctx = self.reconstruct_context(
            interactions,
            all_events,
            focus_skill=focus_skill,
            focus_difficulty=focus_difficulty,
        )
        obs_vec, pre_med, pre_floor = self.build_observation(ctx, metrics_by_skill)
        valid_mask, mask_reasons = self.valid_action_mask_and_reasons(ctx, metrics_by_skill)

        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        seed_probs = []
        seed_values = []
        for model in self.models:
            logits = model.actor(obs_t).masked_fill(~mask_t, -1e9)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            val = float(model.critic(obs_t).squeeze(-1).item())
            seed_probs.append(probs)
            seed_values.append(val)

        mean_probs = np.mean(seed_probs, axis=0)
        # Re-normalize over valid actions for numerical cleanliness
        mean_probs = np.where(valid_mask, mean_probs, 0.0)
        total_p = float(mean_probs.sum())
        if total_p > 0:
            mean_probs = mean_probs / total_p

        best_action_id = int(np.argmax(mean_probs))
        best_target = self.resolve_action_target(best_action_id, ctx, metrics_by_skill)
        rationale = self._explain_choice(best_action_id, best_target, ctx, metrics_by_skill, pre_med, pre_floor)

        actions_list = []
        for aid in range(len(ACTIONS)):
            meta = ACTION_META[aid]
            is_valid = bool(valid_mask[aid])
            target = self.resolve_action_target(aid, ctx, metrics_by_skill) if is_valid else None
            actions_list.append({
                "id": aid,
                "name": meta["name"],
                "label": meta["label"],
                "short": meta["short"],
                "category": meta["category"],
                "badge": meta["badge"],
                "description": meta["description"],
                "cost": ACTION_COSTS[aid],
                "valid": is_valid,
                "probability": float(mean_probs[aid]),
                "per_seed_probabilities": [float(sp[aid]) for sp in seed_probs],
                "recommended": aid == best_action_id,
                "mask_reason": None if is_valid else mask_reasons.get(aid, "Masked in current state."),
                "target": (
                    {
                        "kind": target["kind"],
                        "skill": target["skill"],
                        "skill_name": SKILLS[target["skill"]][0],
                        "difficulty": target["difficulty"],
                        "mode": target.get("mode", "practice"),
                    }
                    if target
                    else None
                ),
            })

        focus = ctx["focus_skill"]
        fm = metrics_by_skill[focus]
        return {
            "policy_name": "delayed_retention_ppo",
            "seeds": self.seeds,
            "ppo_sha256": self.version,
            "obs_dim": OBS_DIM,
            "action_dim": len(ACTIONS),
            "focus_skill": focus,
            "focus_skill_name": SKILLS[focus][0],
            "focus_difficulty": ctx["difficulty"],
            "recommended_action": {
                **actions_list[best_action_id],
                "rationale": rationale,
            },
            "critic_value": float(np.mean(seed_values)),
            "actions": actions_list,
            "state_summary": {
                "p_easy": fm["p_easy"],
                "p_med": fm["p_med"],
                "p_hard": fm["p_hard"],
                "p_7d": fm["p_7d"],
                "p_30d": fm["p_30d"],
                "fragility": fm["fragility"],
                "floor_b": fm["floor_b"],
                "active_k": fm["active_k"],
                "assessed": fm["assessed"],
                "attempts_on_topic": ctx["attempts_on_topic"][focus],
                "last_score": ctx["last_score"][focus],
                "last_hints": int(ctx["last_hints"][focus]),
                "last_leakage": ctx["last_leakage"][focus],
                "unverified_assisted": ctx["unverified_assisted"][focus],
                "can_self_explain": ctx["can_self_explain"][focus],
                "awaiting_question": ctx["awaiting_question"],
                "primed_mode": ctx["primed_mode"],
                "prereq_med": pre_med,
                "prereq_floor": pre_floor,
                "budget_used": ctx["budget_used"],
                "budget_total": TOTAL_BUDGET,
                "budget_remaining_ratio": ctx["budget_remaining_ratio"],
                "min_p30_all_skills": float(min(metrics_by_skill[s]["p_30d"] for s in SKILLS_ORDER)),
                "mean_p30_all_skills": float(np.mean([metrics_by_skill[s]["p_30d"] for s in SKILLS_ORDER])),
            },
            "skills_retention": [
                {
                    "id": s,
                    "name": SKILLS[s][0],
                    "assessed": metrics_by_skill[s]["assessed"],
                    "attempts": ctx["attempts_on_topic"][s],
                    "p_med": metrics_by_skill[s]["p_med"],
                    "p_30d": metrics_by_skill[s]["p_30d"],
                    "fragility": metrics_by_skill[s]["fragility"],
                    "floor_b": metrics_by_skill[s]["floor_b"],
                    "active_k": metrics_by_skill[s]["active_k"],
                    "unverified_assisted": ctx["unverified_assisted"][s],
                }
                for s in SKILLS_ORDER
            ],
            "observation_vector": [float(x) for x in obs_vec.tolist()],
        }
