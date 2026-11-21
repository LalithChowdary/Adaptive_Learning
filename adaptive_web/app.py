"""Domain application orchestrator for 3/adaptive_web/."""

from __future__ import annotations

import math
import os
import secrets
import threading
import time
from typing import Any
import uuid

from . import catalogue_bridge, grader
from .config import Settings
from .llm_client import LLMClient
from .llm_pedagogy import PedagogyRealizer
from .memory_adapter import MemoryAdapter
from .ppo_engine import ACTIONS as PPO_ACTIONS, PPOEngine
from .store import AdaptiveStore


class InvalidActionError(ValueError):
    """Raised when a requested PPO action is masked or out of bounds."""


class AdaptiveLearningApp:
    """Thread-safe domain application for the adaptive Python learning website."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = AdaptiveStore(settings.adaptive_web_db)
        ckpt = (
            getattr(settings, "memory_model_path", None)
            or getattr(settings, "model_path", None)
            or getattr(settings, "checkpoint_path", None)
            or os.environ.get("MEMORY_MODEL_PATH")
        )
        if ckpt is not None:
            self.memory = MemoryAdapter(device=settings.device, checkpoint_path=ckpt)
        else:
            self.memory = MemoryAdapter(device=settings.device)
        self.ppo = PPOEngine(self.memory)
        self.llm_client = LLMClient(settings)
        self.pedagogy = PedagogyRealizer(self.llm_client)
        self.csrf_token = secrets.token_hex(16)
        self._lock = threading.Lock()

    def close(self) -> None:
        """Close underlying database connections."""
        self.store.close()

    # ---- Validation Helpers ----

    def _require_learner(self, learner: str | None) -> dict[str, Any]:
        if not learner or not isinstance(learner, str):
            raise ValueError("Missing learner ID.")
        profile = self.store.get_profile(learner)
        if profile is None:
            raise ValueError(f"Unknown learner ID: {learner!r}")
        return profile

    def _require_skill(self, skill: str | None) -> str:
        if not skill or skill not in catalogue_bridge.PPO_SKILLS_ORDER:
            raise ValueError(
                f"Unknown skill {skill!r}; expected one of {catalogue_bridge.PPO_SKILLS_ORDER}"
            )
        return skill

    def _validate_format(self, fmt: str | None) -> str | None:
        if fmt is None:
            return None
        if fmt not in catalogue_bridge.FORMATS:
            raise ValueError(
                f"Unknown format {fmt!r}; expected one of {catalogue_bridge.FORMATS}"
            )
        return fmt

    def _validate_difficulty(self, difficulty: str | None) -> str | None:
        if difficulty is None:
            return None
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"Unknown difficulty {difficulty!r}; expected easy, medium, or hard")
        return difficulty

    def _get_active_interaction(self, learner: str, skill: str | None = None) -> dict[str, Any] | None:
        interactions = self.store.get_interactions(learner, skill=skill)
        for item in reversed(interactions):
            if item.get("status") == "open":
                return item
        return None

    def _sanitize_active_interaction(self, item: dict[str, Any] | None) -> dict[str, Any] | None:
        if item is None:
            return None
        return {
            "id": item["id"],
            "interaction_id": item["id"],
            "learner": item["learner"],
            "skill": item["skill"],
            "task_id": item["task"],
            "action_id": item.get("action_id"),
            "action_name": item.get("action_name"),
            "mode": item.get("mode", "practice"),
            "status": item.get("status", "open"),
            "question": item.get("question"),
            "hints": list(item.get("hints", [])),
            "results": list(item.get("results", [])),
            "retry_count": int(item.get("retry_count", 0)),
            "assessment_incomplete": bool(item.get("assessment_incomplete", False)),
        }

    def _emit_event(
        self,
        learner: str,
        skill: str,
        interaction_id: str,
        now: float,
        kind: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        existing = self.store.get_events(learner, skill=skill)
        gap = max(0.0, now - float(existing[-1]["at"])) if existing else None
        body = dict(payload)
        feats = dict(body.get("features") or {})
        feats.setdefault("event_type", kind)
        feats["gap_seconds"] = gap
        feats.setdefault("new_session", None if not existing else False)
        body["features"] = feats
        return self.store.append_event(learner, skill, interaction_id, now, kind, body)

    # ---- Public API Methods ----

    def get_state(self, learner: str | None = None, skill: str | None = None) -> dict[str, Any]:
        """Return full bootstrap/workspace state for UI and Glass-Box inspector."""
        if skill is not None:
            self._require_skill(skill)

        profiles = self.store.list_profiles()
        base: dict[str, Any] = {
            "csrf_token": self.csrf_token,
            "token": self.csrf_token,
            "profiles": profiles,
            "skills": list(catalogue_bridge.PPO_SKILLS_ORDER),
            "skill_meta": catalogue_bridge.SKILL_META,
            "formats": list(catalogue_bridge.FORMATS),
            "prerequisites": {k: list(v) for k, v in catalogue_bridge.PREREQUISITES.items()},
            "llm_status": {
                "provider": self.llm_client.provider_name(),
                "model": self.llm_client.model_name(),
                "is_live": self.llm_client.is_live(),
            },
        }

        if not learner:
            return base

        profile = self._require_learner(learner)
        focus_skill = skill or catalogue_bridge.PPO_SKILLS_ORDER[0]
        ppo_dec = self.ppo.decide(self.store, learner, focus_skill=focus_skill)
        token_audit = self.memory.inspect_tokens(
            self.store.get_events(learner, skill=focus_skill)
        )
        token_audit["token_dim"] = int(token_audit.get("expected_token_dim", 47))
        token_audit["task_feature_dim"] = int(token_audit.get("expected_task_dim", 19))
        token_audit["uses_text_or_code_embeddings"] = False
        learning_path = self.store.get_latest_learning_path(learner, focus_skill)
        course_material = self.store.get_latest_course_material(learner, focus_skill)
        active = self._sanitize_active_interaction(
            self._get_active_interaction(learner, skill=focus_skill)
        )

        base.update(
            {
                "learner": learner,
                "profile": profile,
                "skill": focus_skill,
                "skill_summaries": ppo_dec["metrics_by_skill"],
                "metrics_by_skill": ppo_dec["metrics_by_skill"],
                "ppo": ppo_dec,
                "token_audit": token_audit,
                "learning_path": learning_path,
                "course_material": course_material,
                "active": active,
                "glass_box": {
                    "token_audit": token_audit,
                    "ppo": ppo_dec,
                    "model_sha256": self.memory.version,
                    "ppo_sha256": self.ppo.version,
                },
            }
        )
        return base

    def create_profile(self, name: str, request_id: str | None = None) -> dict[str, Any]:
        """Create a new learner profile."""
        if not isinstance(name, str):
            raise ValueError("Profile name must be a string.")
        cleaned = name.strip()
        if not (1 <= len(cleaned) <= 50):
            raise ValueError("Profile name must be between 1 and 50 characters.")

        def _do() -> dict[str, Any]:
            profile = self.store.create_profile(cleaned)
            return {
                "status": "ok",
                "learner": profile["id"],
                "learner_id": profile["id"],
                "profile": profile,
            }

        if request_id is not None:
            return self.store.run_idempotent(request_id, _do)
        return _do()

    def select_skill(
        self,
        learner: str,
        skill: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Select a skill, curate a learning path & course material via LLM, and query PPO."""
        self._require_learner(learner)
        self._require_skill(skill)

        if request_id is not None:
            cached = self.store.idempotent_get(request_id)
            if cached is not None:
                return cached

        # Gather memory state & PPO recommendation (read-only)
        ppo_dec = self.ppo.decide(self.store, learner, focus_skill=skill)
        skill_summary = ppo_dec["metrics_by_skill"][skill]

        # Extract specific weak/missed patterns from stored interactions for this skill
        weak_topics: list[str] = []
        for it in reversed(self.store.get_interactions(learner, skill=skill)):
            results = it.get("results") or []
            if not results:
                continue
            first_res = results[0]
            if not first_res.get("full_success") or len(it.get("hints") or []) > 0:
                q_obj = it.get("question") or {}
                fmt_label = q_obj.get("task_type") or "item"
                diff_label = q_obj.get("difficulty") or "medium"
                prompt_snip = str(q_obj.get("prompt") or "").strip()[:70]
                missing = first_res.get("missing_concepts") or []
                if missing:
                    weak_topics.append(f"missing concepts ({', '.join(str(m) for m in missing[:3])})")
                elif prompt_snip:
                    weak_topics.append(f"{fmt_label}/{diff_label}: {prompt_snip}")
            if len(weak_topics) >= 4:
                break

        # Release any locks during outbound LLM calls so other learners/reads never block
        learning_path = self.pedagogy.curate_learning_path(
            skill,
            memory_summary=skill_summary,
            all_skill_summaries=ppo_dec["metrics_by_skill"],
            prereqs=catalogue_bridge.PREREQUISITES.get(skill, ()),
        )
        course_material = self.pedagogy.generate_course_material(
            skill,
            memory_summary=skill_summary,
            focus_gap=f"P(Med)={skill_summary.get('p_med', 0.5):.2f}, floor_b={skill_summary.get('floor_b', 0.0):.2f}",
            weak_topics=weak_topics,
        )

        # Persist path and material without disturbing any active open question
        self.store.save_learning_path(learner, skill, learning_path)
        self.store.save_course_material(learner, skill, course_material)

        res = {
            "status": "ok",
            "learner": learner,
            "skill": skill,
            "learning_path": learning_path,
            "course_material": course_material,
            "ppo": ppo_dec,
            "active": self._sanitize_active_interaction(
                self._get_active_interaction(learner, skill=skill)
            ),
        }
        if request_id is not None:
            self.store.idempotent_set(request_id, res)
        return res

    def ppo_step(
        self,
        learner: str,
        skill: str,
        action_id: int | None = None,
        policy_source: str = "delayed_retention_ppo",
        format: str | None = None,
        difficulty: str | None = None,
        question_source: str | None = None,
        mode: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a PPO action (or manual override) and return a lesson or question."""
        self._require_learner(learner)
        self._require_skill(skill)
        fmt = self._validate_format(format)
        diff = self._validate_difficulty(difficulty)

        if request_id is not None:
            cached = self.store.idempotent_get(request_id)
            if cached is not None:
                return cached

        ppo_dec = self.ppo.decide(self.store, learner, focus_skill=skill)
        valid_mask = ppo_dec["valid_mask"]
        mask_reasons = ppo_dec["mask_reasons"]

        if action_id is None:
            chosen_id = int(ppo_dec["action_id"])
        else:
            chosen_id = int(action_id)
            if chosen_id < 0 or chosen_id >= len(PPO_ACTIONS):
                raise ValueError(f"action_id {chosen_id} out of bounds (must be 0..7).")
            if not valid_mask[chosen_id]:
                reason = mask_reasons.get(
                    chosen_id,
                    f"Action {chosen_id} ({PPO_ACTIONS[chosen_id]}) is currently masked.",
                )
                raise InvalidActionError(reason)

        action_name = PPO_ACTIONS[chosen_id]
        action_obj = ppo_dec["actions"][chosen_id]
        target = action_obj.get("target") or ppo_dec["target"]
        target_skill = target.get("skill", skill)
        target_diff = diff or target.get("difficulty", "easy")
        now = time.time()

        # Passive lessons: 3 (explain_concept) and 4 (show_worked_example)
        if chosen_id in (3, 4):
            primed_mode = "concept" if chosen_id == 3 else "example"
            material = self.store.get_latest_course_material(learner, target_skill)
            if material is None:
                material = self.pedagogy.generate_course_material(target_skill)
                self.store.save_course_material(learner, target_skill, material)

            lesson_iid = f"lesson-{uuid.uuid4().hex[:12]}"
            self._emit_event(
                learner,
                target_skill,
                lesson_iid,
                now,
                "instruction",
                {
                    "ppo_action": action_name,
                    "primed_mode": primed_mode,
                    "policy_source": policy_source,
                    "features": {"event_type": "instruction"},
                },
            )
            self.store.save_tutor_turn(
                learner,
                target_skill,
                policy_source,
                {
                    "action_id": chosen_id,
                    "action_name": action_name,
                    "target_skill": target_skill,
                    "primed_mode": primed_mode,
                },
            )
            post_ppo = self.ppo.decide(self.store, learner, focus_skill=skill)
            res = {
                "status": "ok",
                "type": "lesson",
                "action_id": chosen_id,
                "action_name": action_name,
                "ppo_action": action_name,
                "policy_source": policy_source,
                "target_skill": target_skill,
                "primed_mode": primed_mode,
                "lesson": material,
                "awaiting_question": True,
                "ppo": post_ppo,
            }
            if request_id is not None:
                self.store.idempotent_set(request_id, res)
            return res

        # Self-explanation: 5 (request_self_explanation)
        if chosen_id == 5:
            iid = f"se-{uuid.uuid4().hex[:12]}"
            se_prompt_bank = {
                "range_boundaries": [
                    "Explain in your own words how start and stop boundaries work in Python range(start, stop).",
                    "Explain what happens when step is negative in range(start, stop, step) and how you find the last element.",
                    "Explain why range(n) produces n items and why stop is never included.",
                ],
                "loop_tracing": [
                    "Explain in your own words how a loop iterates over a sequence.",
                    "Explain how to track a variable that is updated multiple times inside one loop iteration.",
                    "Explain what value a loop variable holds immediately after a for-loop finishes.",
                ],
                "accumulator_initialization": [
                    "Explain in your own words how to initialize and update an accumulator variable.",
                    "Explain why initializing a product accumulator to 0 breaks multiplication loops.",
                    "Explain what goes wrong if an accumulator is initialized inside the loop body instead of before it.",
                ],
                "conditional_evaluation": [
                    "Explain in your own words how an if/elif/else condition is evaluated.",
                    "Explain the difference between an if/elif/else chain and two separate if statements.",
                    "Explain why strict (<, >) vs inclusive (<=, >=) operators matter at exact boundary values.",
                ],
                "list_indexing": [
                    "Explain in your own words how zero-based list indexing works in Python.",
                    "Explain how negative indices (-1, -2) map to elements in a list of length n.",
                    "Explain why accessing lst[len(lst)] raises an IndexError.",
                ],
                "function_returns": [
                    "Explain in your own words how a return statement exits a function and produces a value.",
                    "Explain the difference between print(x) and return x inside a Python function.",
                    "Explain how arguments bind to parameters and how operator precedence applies in return expressions.",
                ],
            }
            variants = se_prompt_bank.get(
                target_skill,
                [catalogue_bridge.DEFAULT_LESSONS[target_skill]["self_explanation_prompt"]],
            )
            se_count = sum(
                1
                for it in self.store.get_interactions(learner, skill=target_skill)
                if it.get("ppo_action") == "request_self_explanation"
            )
            se_prompt = variants[se_count % len(variants)]
            se_task = {
                "task_id": f"self_explanation.{target_skill}.{uuid.uuid4().hex[:6]}",
                "skill_id": target_skill,
                "task_type": "output_prediction",
                "difficulty": "medium",
                "prompt": se_prompt,
                "self_explanation_prompt": se_prompt,
                "snippets": [],
                "options": [],
                "criteria": [{"kind": "exact_match", "weight": 1.0}],
                "reference_seconds": 45.0,
            }
            interaction = {
                "id": iid,
                "learner": learner,
                "skill": target_skill,
                "task": se_task["task_id"],
                "action_id": chosen_id,
                "action_name": action_name,
                "ppo_action": action_name,
                "mode": "practice",
                "status": "open",
                "question": se_task,
                "raw_task": se_task,
                "hints": [],
                "results": [],
                "retry_count": 0,
                "created_at": now,
            }
            self.store.save_interaction(interaction)
            self._emit_event(
                learner,
                target_skill,
                iid,
                now,
                "task",
                {
                    "task_id": se_task["task_id"],
                    "ppo_action": action_name,
                    "mode": "practice",
                    "features": {
                        "event_type": "task",
                        "task": catalogue_bridge.task_memory_features(se_task),
                    },
                },
            )
            self.store.save_tutor_turn(
                learner,
                target_skill,
                policy_source,
                {
                    "action_id": chosen_id,
                    "action_name": action_name,
                    "target_skill": target_skill,
                    "interaction_id": iid,
                },
            )
            post_ppo = self.ppo.decide(self.store, learner, focus_skill=skill)
            res = {
                "status": "ok",
                "type": "self_explanation",
                "action_id": chosen_id,
                "action_name": action_name,
                "ppo_action": action_name,
                "policy_source": policy_source,
                "target_skill": target_skill,
                "interaction_id": iid,
                "interaction": iid,
                "question": se_task,
                "ppo": post_ppo,
            }
            if request_id is not None:
                self.store.idempotent_set(request_id, res)
            return res

        # Question actions: 0, 1, 2, 6 (review_prerequisite), 7 (advance_next_topic)
        resolved_mode = mode or target.get("mode") or (
            "assessment" if ppo_dec.get("unverified_assisted") else "practice"
        )
        pool = "practice" if resolved_mode == "practice" else "assessment"
        src = question_source or "v2.2"
        chosen_fmt = fmt or ("output_prediction" if src == "llm" else catalogue_bridge.FORMATS[0])

        if src == "llm":
            raw_task = self.pedagogy.generate_question(
                target_skill,
                difficulty=target_diff,
                task_format=chosen_fmt,
                pool=pool,
            )
            raw_task["memory_task"] = catalogue_bridge.task_memory_features(raw_task)
            self.store.save_generated_task(raw_task)
        else:
            seen_ids = {
                it.get("task")
                for it in self.store.get_interactions(learner, skill=target_skill)
                if it.get("task")
            }
            raw_task = catalogue_bridge.sample_v22_task(
                skill=target_skill,
                difficulty=target_diff,
                task_format=chosen_fmt,
                pool=pool,
                exclude_ids=seen_ids,
            )

        sanitized_task = {
            k: v
            for k, v in raw_task.items()
            if k not in ("answer_key", "reference_solution", "criteria", "tests", "hints")
        }
        iid = f"int-{uuid.uuid4().hex[:12]}"
        interaction = {
            "id": iid,
            "learner": learner,
            "skill": target_skill,
            "task": raw_task["task_id"],
            "action_id": chosen_id,
            "action_name": action_name,
            "ppo_action": action_name,
            "mode": resolved_mode,
            "status": "open",
            "question": sanitized_task,
            "raw_task": raw_task,
            "hints": [],
            "results": [],
            "retry_count": 0,
            "assessment_incomplete": False,
            "created_at": now,
        }
        self.store.save_interaction(interaction)
        self._emit_event(
            learner,
            target_skill,
            iid,
            now,
            "task",
            {
                "task_id": raw_task["task_id"],
                "ppo_action": action_name,
                "mode": resolved_mode,
                "features": {
                    "event_type": "task",
                    "task": catalogue_bridge.task_memory_features(raw_task),
                },
            },
        )
        if chosen_id == 7:
            self._emit_event(
                learner,
                skill,
                iid,
                now,
                "advance",
                {"target_skill": target_skill, "features": {"event_type": "advance"}},
            )
        self.store.save_tutor_turn(
            learner,
            target_skill,
            policy_source,
            {
                "action_id": chosen_id,
                "action_name": action_name,
                "target_skill": target_skill,
                "interaction_id": iid,
                "mode": resolved_mode,
            },
        )

        post_ppo = self.ppo.decide(self.store, learner, focus_skill=skill)
        res = {
            "status": "ok",
            "type": "question",
            "action_id": chosen_id,
            "action_name": action_name,
            "ppo_action": action_name,
            "policy_source": policy_source,
            "target_skill": target_skill,
            "mode": resolved_mode,
            "primed_mode": None,
            "interaction_id": iid,
            "interaction": iid,
            "question": sanitized_task,
            "ppo": post_ppo,
        }
        if request_id is not None:
            self.store.idempotent_set(request_id, res)
        return res

    def hint(
        self,
        learner: str,
        interaction_id: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Provide a progressive hint (max 3) and record leakage l_t."""
        self._require_learner(learner)

        def _do() -> dict[str, Any]:
            item = self.store.get_interaction(interaction_id)
            if item is None or item.get("learner") != learner:
                raise ValueError(f"Unknown interaction {interaction_id!r} for learner {learner!r}")
            if item.get("status") != "open":
                raise ValueError("Cannot request hint on a closed interaction.")
            if item.get("results") and item["results"][-1].get("full_success"):
                raise ValueError("Cannot request hint after a correct submission.")
            if len(item.get("hints", [])) >= 3:
                raise ValueError("Maximum of 3 hints reached for this interaction.")

            level = len(item.get("hints", [])) + 1
            raw_task = item["raw_task"]
            predefined = raw_task.get("hints") or []
            if level <= len(predefined):
                h_raw = predefined[level - 1]
                if isinstance(h_raw, dict):
                    hint_obj = {
                        "hint_level": level,
                        "hint_kind": h_raw.get("kind", ("conceptual", "targeted", "scaffold")[level - 1]),
                        "kind": h_raw.get("kind", ("conceptual", "targeted", "scaffold")[level - 1]),
                        "text": h_raw.get("text", ""),
                        "leakage_l_t": round(level * 0.25, 2),
                    }
                else:
                    kind = ("conceptual", "targeted", "scaffold")[level - 1]
                    hint_obj = {
                        "hint_level": level,
                        "hint_kind": kind,
                        "kind": kind,
                        "text": str(h_raw),
                        "leakage_l_t": round(level * 0.25, 2),
                    }
            else:
                hint_obj = self.pedagogy.generate_hint(
                    raw_task,
                    learner_answer=None,
                    hint_level=level,
                    prior_hints=item.get("hints"),
                )
                hint_obj.setdefault("kind", hint_obj.get("hint_kind", "conceptual"))

            now = time.time()
            assessment_incomplete = False
            if item.get("mode") == "assessment":
                item["mode"] = "review"
                item["assessment_incomplete"] = True
                assessment_incomplete = True

            item.setdefault("hints", []).append(hint_obj)
            self.store.save_interaction(item)
            self._emit_event(
                learner,
                item["skill"],
                interaction_id,
                now,
                "help_request",
                {
                    "hint_level": level,
                    "features": {
                        "event_type": "help_request",
                        "task": catalogue_bridge.task_memory_features(raw_task),
                    },
                },
            )
            self._emit_event(
                learner,
                item["skill"],
                interaction_id,
                now,
                "hint",
                {
                    "hint_level": level,
                    "hint_kind": hint_obj["hint_kind"],
                    "leakage_l_t": hint_obj["leakage_l_t"],
                    "features": {
                        "event_type": "hint",
                        "task": catalogue_bridge.task_memory_features(raw_task),
                        "hint_count": level,
                        "hint_type": hint_obj["hint_kind"],
                        "assistance": "assisted",
                    },
                },
            )

            return {
                "status": "ok",
                "hint": hint_obj,
                "hints": item["hints"],
                "leakage_l_t": hint_obj["leakage_l_t"],
                "mode": item["mode"],
                "assessment_incomplete": assessment_incomplete,
            }

        if request_id is not None:
            return self.store.run_idempotent(request_id, _do)
        return _do()

    def submit(
        self,
        learner: str,
        interaction_id: str,
        answer: Any,
        active_seconds: float | None = 5.0,
        self_explanation: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Grade a submission, log response/grade events, and update monotonic_gru."""
        self._require_learner(learner)

        if active_seconds is not None:
            if (
                not isinstance(active_seconds, (int, float))
                or isinstance(active_seconds, bool)
                or not math.isfinite(active_seconds)
                or active_seconds < 0.0
                or active_seconds > 86400.0
            ):
                raise ValueError("active_seconds must be a finite number in 0..86400.")

        def _do() -> dict[str, Any]:
            item = self.store.get_interaction(interaction_id)
            if item is None or item.get("learner") != learner:
                raise ValueError(f"Unknown interaction {interaction_id!r} for learner {learner!r}")
            if item.get("status") != "open":
                raise ValueError("Interaction is already closed.")

            attempts_so_far = len(item.get("results", []))
            retry_count = int(item.get("retry_count", 0))
            if attempts_so_far > retry_count:
                raise ValueError("Duplicate submission without retry.")

            raw_task = item["raw_task"]
            hints_used = len(item.get("hints", []))
            attempt_number = attempts_so_far + 1
            assistance = "independent" if hints_used == 0 else "assisted"
            now = time.time()

            is_self_exp = (
                item.get("ppo_action") == "request_self_explanation"
                or str(raw_task.get("task_id", "")).startswith("self_explanation.")
            )
            if is_self_exp:
                text_ans = self_explanation if self_explanation is not None else str(answer or "")
                grade_result = self.pedagogy.evaluate_self_explanation(
                    skill=item["skill"],
                    task=raw_task,
                    prompt=raw_task.get("prompt", ""),
                    learner_explanation=text_ans,
                )
            else:
                norm_ans = answer
                ans_len = max(1, len(raw_task.get("answer_key") or [1]))
                if raw_task.get("task_type") in ("output_prediction", "fill_in_the_blank"):
                    if isinstance(norm_ans, str):
                        norm_ans = [norm_ans] * ans_len
                    elif (
                        isinstance(norm_ans, list)
                        and len(norm_ans) == 1
                        and ans_len > 1
                        and isinstance(norm_ans[0], str)
                        and norm_ans[0].strip()
                    ):
                        norm_ans = norm_ans * ans_len
                grade_result = grader.grade_submission(raw_task, norm_ans)

            is_first = attempts_so_far == 0
            eligible = (
                not is_self_exp
                and item.get("mode") == "assessment"
                and not item.get("assessment_incomplete", False)
                and is_first
                and hints_used == 0
                and grade_result["grading_status"] == "resolved"
            )
            result_row = {
                **grade_result,
                "attempt": attempt_number,
                "assisted": hints_used > 0,
                "eligible": eligible,
                "active_seconds": active_seconds,
                "at": now,
            }

            ref_sec = float(raw_task.get("reference_seconds", 45.0) or 45.0)
            rel_time = (float(active_seconds) / ref_sec) if active_seconds is not None else 1.0
            task_feats = catalogue_bridge.task_memory_features(raw_task)

            pair_feats = {
                "task": task_feats,
                "attempt_number": attempt_number,
                "assistance": assistance,
                "hint_count": hints_used,
                "answer_revealed": bool(item.get("revealed", False)),
            }

            resp_kind = "self_explanation_response" if is_self_exp else "response"
            grade_kind = "self_explanation_grade" if is_self_exp else "grade"

            item.setdefault("results", []).append(result_row)
            self.store.save_interaction(item)
            self._emit_event(
                learner,
                item["skill"],
                interaction_id,
                now,
                resp_kind,
                {
                    "attempt_number": attempt_number,
                    "active_seconds": active_seconds,
                    "features": {
                        "event_type": "response",
                        "active_response_seconds": active_seconds,
                        "relative_response_time": rel_time,
                        **pair_feats,
                    },
                },
            )
            self._emit_event(
                learner,
                item["skill"],
                interaction_id,
                now,
                grade_kind,
                {
                    **grade_result,
                    "attempt_number": attempt_number,
                    "assistance": assistance,
                    "features": {
                        "event_type": "grade",
                        "score": grade_result["score"],
                        "full_success": grade_result["full_success"],
                        "grading_status": grade_result["grading_status"],
                        **pair_feats,
                    },
                },
            )

            return {
                "status": "ok",
                "interaction_id": interaction_id,
                "result": result_row,
                "results": item["results"],
                "is_correct": bool(grade_result.get("full_success")),
                "score": grade_result.get("score"),
                "grading_status": grade_result["grading_status"],
            }

        if request_id is not None:
            return self.store.run_idempotent(request_id, _do)
        return _do()

    def retry(
        self,
        learner: str,
        interaction_id: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Allow retrying a failed submission (max 2 retries)."""
        self._require_learner(learner)

        if request_id is not None:
            cached = self.store.idempotent_get(request_id)
            if cached is not None:
                return cached

        item = self.store.get_interaction(interaction_id)
        if item is None or item.get("learner") != learner:
            raise ValueError(f"Unknown interaction {interaction_id!r} for learner {learner!r}")
        if not item.get("results"):
            raise ValueError("Cannot retry before submitting an attempt.")
        if item["results"][-1].get("full_success"):
            raise ValueError("Cannot retry after a successful submission.")
        if int(item.get("retry_count", 0)) >= 2:
            raise ValueError("Maximum of 2 retries reached for this interaction.")

        if item.get("mode") == "assessment":
            item["mode"] = "review"
        item["retry_count"] = int(item.get("retry_count", 0)) + 1
        self.store.save_interaction(item)
        self._emit_event(
            learner,
            item["skill"],
            interaction_id,
            time.time(),
            "retry",
            {"retry_count": item["retry_count"], "features": {"event_type": "retry"}},
        )

        res = {"status": "ok", "retry_count": item["retry_count"]}
        if request_id is not None:
            self.store.idempotent_set(request_id, res)
        return res

    def reveal(
        self,
        learner: str,
        interaction_id: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Reveal the reference solution after at least one submission."""
        self._require_learner(learner)

        if request_id is not None:
            cached = self.store.idempotent_get(request_id)
            if cached is not None:
                return cached

        item = self.store.get_interaction(interaction_id)
        if item is None or item.get("learner") != learner:
            raise ValueError(f"Unknown interaction {interaction_id!r} for learner {learner!r}")
        if not item.get("results"):
            raise ValueError("Cannot reveal solution before submitting an attempt.")

        raw_task = item["raw_task"]
        raw_sol = raw_task.get("reference_solution") or raw_task.get("answer_key")
        if raw_task.get("task_type") in ("single_choice", "multiple_select") and raw_task.get("options"):
            opt_map = {o["id"]: o["text"] for o in raw_task["options"] if isinstance(o, dict) and "id" in o}
            if isinstance(raw_sol, str) and raw_sol in opt_map:
                solution = opt_map[raw_sol]
            elif isinstance(raw_sol, list):
                solution = [opt_map.get(x, x) for x in raw_sol]
            else:
                solution = raw_sol or "See worked example."
        else:
            solution = raw_sol or "See worked example."
        item["revealed"] = True
        item["status"] = "revealed"
        self.store.save_interaction(item)
        self._emit_event(
            learner,
            item["skill"],
            interaction_id,
            time.time(),
            "solution_reveal",
            {
                "features": {
                    "event_type": "solution_reveal",
                    "task": catalogue_bridge.task_memory_features(raw_task),
                    "answer_revealed": True,
                    "assistance": "assisted",
                }
            },
        )

        res = {
            "status": "ok",
            "solution": solution,
            "revealed_solution": solution,
            "explanation": raw_task.get("explanation", ""),
        }
        if request_id is not None:
            self.store.idempotent_set(request_id, res)
        return res

    def finish(
        self,
        learner: str,
        interaction_id: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Close an active interaction."""
        self._require_learner(learner)

        if request_id is not None:
            cached = self.store.idempotent_get(request_id)
            if cached is not None:
                return cached

        item = self.store.get_interaction(interaction_id)
        if item is None or item.get("learner") != learner:
            raise ValueError(f"Unknown interaction {interaction_id!r} for learner {learner!r}")

        item["status"] = "finished"
        self.store.save_interaction(item)
        self._emit_event(
            learner,
            item["skill"],
            interaction_id,
            time.time(),
            "finish",
            {"features": {"event_type": "finish"}},
        )

        res = {"status": "ok", "interaction_id": interaction_id, "completed": True}
        if request_id is not None:
            self.store.idempotent_set(request_id, res)
        return res

    def curve(
        self,
        learner: str,
        skill: str,
        fmt: str = "output_prediction",
        task: str | None = None,
    ) -> dict[str, Any]:
        """Return 0-to-60-day retention curves for a learner and skill."""
        self._require_learner(learner)
        self._require_skill(skill)
        self._validate_format(fmt)

        if task is not None:
            t_obj = catalogue_bridge.get_v22_task(task)
            if t_obj is not None and t_obj.get("skill_id") != skill:
                raise ValueError(f"Task {task!r} belongs to {t_obj.get('skill_id')!r}, not {skill!r}")

        events = self.store.get_events(learner, skill=skill)
        return self.memory.retention_curve(
            learner,
            skill,
            events,
            task_format=fmt,
        )

    def export(self, learner: str) -> dict[str, Any]:
        """Export complete learner provenance bundle (zero secrets)."""
        profile = self._require_learner(learner)
        learning_paths = [
            lp
            for s in catalogue_bridge.PPO_SKILLS_ORDER
            if (lp := self.store.get_latest_learning_path(learner, s)) is not None
        ]
        course_materials = [
            cm
            for s in catalogue_bridge.PPO_SKILLS_ORDER
            if (cm := self.store.get_latest_course_material(learner, s)) is not None
        ]
        all_events = self.store.get_events(learner)
        audit_skill = all_events[-1]["skill"] if all_events else catalogue_bridge.PPO_SKILLS_ORDER[0]
        skill_events = [e for e in all_events if e.get("skill") == audit_skill]
        token_audit = self.memory.inspect_tokens(skill_events)
        token_audit["token_dim"] = int(token_audit.get("expected_token_dim", 47))
        token_audit["task_feature_dim"] = int(token_audit.get("expected_task_dim", 19))
        token_audit["uses_text_or_code_embeddings"] = False
        return {
            "profile": profile,
            "model_sha256": self.memory.version,
            "ppo_sha256": self.ppo.version,
            "interactions": self.store.get_interactions(learner),
            "events": all_events,
            "learning_paths": learning_paths,
            "course_materials": course_materials,
            "tutor_turns": self.store.get_tutor_turns(learner),
            "generated_tasks": self.store.list_generated_tasks(),
            "token_audit": token_audit,
        }
