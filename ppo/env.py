"""Closed-loop 8-action teaching environment backed by the frozen v2.2 monotonic_gru."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model"
V22_ROOT = ROOT / "data_Set" / "v2.2"
CHECKPOINT = MODEL_ROOT / "runs/v2.2-mps-main/monotonic_gru/seed_20261122/model.pt"

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from memory.features import DAY, EvidenceStream, task_features  # noqa: E402
from memory.recurrent import LearnerGRU  # noqa: E402


def _load_v22_module(name):
    sys.path.insert(0, str(V22_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(f"ppo_v22_{name}", V22_ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


_CATALOG = _load_v22_module("catalog")
_SIM = _load_v22_module("simulator")

SKILLS_ORDER = (
    "range_boundaries",
    "loop_tracing",
    "accumulator_initialization",
    "conditional_evaluation",
    "list_indexing",
    "function_returns",
)
DIFFICULTIES = ("easy", "medium", "hard")
PREREQUISITES = {
    "range_boundaries": (),
    "loop_tracing": ("range_boundaries",),
    "accumulator_initialization": ("range_boundaries", "loop_tracing"),
    "conditional_evaluation": ("loop_tracing",),
    "list_indexing": ("range_boundaries",),
    "function_returns": ("conditional_evaluation", "accumulator_initialization"),
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
ACTION_COSTS = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 0.6,
    4: 0.7,
    5: 0.9,
    6: 1.0,
    7: 0.2,
}

TOTAL_BUDGET = 32.0
SESSION_BUDGET = 8.0
OBS_DIM = 36


def _task_model_features(task):
    return {
        "task_type": task["task_type"],
        "difficulty": task["difficulty"],
        "reference_seconds": task["reference_seconds"],
        "snippet_count": len(task["snippets"]),
        "option_count": len(task["options"]),
        "criterion_count": len(task["criteria"]),
        "credit_weights": sorted(
            c["weight"] for c in task["criteria"] if c["kind"] != "incorrect_selection_penalty"
        ),
        "penalty_weights": sorted(
            c["weight"] for c in task["criteria"] if c["kind"] == "incorrect_selection_penalty"
        ),
    }


def _load_reference_tasks():
    catalog_tasks = json.loads((V22_ROOT / "dataset/task_catalog.json").read_text())
    refs = {}
    for diff in DIFFICULTIES:
        match = next(
            t for t in catalog_tasks
            if t["skill_id"] == "range_boundaries"
            and t["difficulty"] == diff
            and t["task_type"] == "output_prediction"
            and t["pool"] == "assessment"
        )
        refs[diff] = task_features(_task_model_features(match))
    return refs, catalog_tasks


class FrozenMemoryTracker:
    """Shared frozen monotonic_gru inference wrapper for per-skill learner records."""

    def __init__(self, checkpoint_path=CHECKPOINT):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload["dataset_version"] != "2.2.0" or payload["mode"] != "monotonic":
            raise ValueError("Expected v2.2 monotonic_gru checkpoint")
        self.model = LearnerGRU(mode=payload["mode"], hidden_size=payload["hidden_size"])
        self.model.load_state_dict(payload["weights"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        refs, catalog_tasks = _load_reference_tasks()
        self.catalog_tasks = catalog_tasks
        self.ref_tasks_tensor = torch.tensor(
            [refs["easy"], refs["medium"], refs["hard"], refs["medium"], refs["medium"]],
            dtype=torch.float32,
        )
        self.ref_gaps_tensor = torch.tensor([0.0, 0.0, 0.0, 7.0, 30.0], dtype=torch.float32)

    @torch.inference_mode()
    def initial_record(self):
        return {
            "state": self.model.initial_state(1),
            "stream": EvidenceStream(),
            "anchor": None,
            "first_clock": None,
            "last_clock": None,
        }

    @torch.inference_mode()
    def observe_event(self, record, clock_sec, event_dict):
        row = dict(event_dict)
        if record["first_clock"] is None:
            record["first_clock"] = clock_sec
            row["gap_seconds"] = None
            row["new_session"] = None
        else:
            row["gap_seconds"] = max(0.0, clock_sec - record["last_clock"])
        record["last_clock"] = clock_sec
        evidence = record["stream"].push(row)
        if evidence is not None:
            gap_days = 0.0 if record["anchor"] is None else max(0.0, (evidence.time - record["anchor"]) / DAY)
            token_t = torch.tensor([evidence.vector()], dtype=torch.float32)
            gap_t = torch.tensor([gap_days], dtype=torch.float32)
            record["state"] = self.model.observe(record["state"], token_t, gap_t)
            record["anchor"] = evidence.time

    @torch.inference_mode()
    def query_metrics(self, record, clock_sec):
        if record["anchor"] is None or record["first_clock"] is None:
            elapsed_days = 0.0
        else:
            elapsed_days = max(0.0, ((clock_sec - record["first_clock"]) - record["anchor"]) / DAY)
        state = record["state"]
        states = state.expand(5, -1)
        gaps = self.ref_gaps_tensor + elapsed_days
        logits = self.model.predict_logit(states, self.ref_tasks_tensor, gaps)
        probs = torch.sigmoid(logits).tolist()

        # Extract active skill channels k and consolidated floor b at current time
        decayed, _ = self.model._at_time(state, torch.tensor([elapsed_days], dtype=torch.float32))
        m = self.model.skill_channels
        active_k = float(decayed[0, :m].mean().item())
        floor_b = float(decayed[0, m:2 * m].mean().item())
        p_easy, p_med, p_hard, p_7d, p_30d = probs
        return {
            "p_easy": p_easy,
            "p_med": p_med,
            "p_hard": p_hard,
            "p_7d": p_7d,
            "p_30d": p_30d,
            "fragility": max(0.0, p_med - p_30d),
            "floor_b": floor_b,
            "active_k": active_k,
        }


def _build_pools(catalog_tasks):
    pools = {}
    for skill in SKILLS_ORDER:
        for diff in DIFFICULTIES:
            for pool in ("practice", "assessment"):
                pools[(skill, diff, pool)] = [
                    t for t in catalog_tasks
                    if t["skill_id"] == skill and t["difficulty"] == diff and t["pool"] == pool
                ]
    return pools


def _leakage_to_hint_kind(leakage):
    if leakage < 0.35:
        return "conceptual"
    if leakage < 0.75:
        return "targeted"
    return "scaffold"


class AdaptiveTutorEnv:
    """Closed-loop 8-action pedagogical environment using Gated Option B and frozen monotonic_gru."""

    def __init__(self, tracker=None, mechanism="exponential", long_gaps=False):
        self.tracker = tracker or FrozenMemoryTracker()
        self.mechanism = mechanism
        self.long_gaps = long_gaps
        self.pools = _build_pools(self.tracker.catalog_tasks)
        self.student_seed = 0
        self.rng = random.Random(0)
        self.clock = 0.0
        self.session = 1
        self.budget_used = 0.0
        self.next_session_threshold = SESSION_BUDGET
        self.topic_idx = 0
        self.difficulty_idx = 1
        self.primed_mode = None
        self.awaiting_question = False
        self.done = False
        self.params = {}
        self.latent = {}
        self.records = {}
        self.metrics_cache = {}
        self.unverified_assisted = {}
        self.last_score = {}
        self.last_hints = {}
        self.last_leakage = {}
        self.attempts_on_topic = {}
        self.visited_topics = set()
        self.used_tasks = set()

    def reset(self, seed=0, mechanism=None, long_gaps=None):
        if mechanism is not None:
            self.mechanism = mechanism
        if long_gaps is not None:
            self.long_gaps = long_gaps
        self.student_seed = int(seed)
        self.rng = random.Random(self.student_seed)
        self.clock = 0.0
        self.session = 1
        self.budget_used = 0.0
        self.next_session_threshold = SESSION_BUDGET
        self.topic_idx = 0
        self.difficulty_idx = 0
        self.primed_mode = None
        self.awaiting_question = False
        self.done = False
        self.used_tasks = set()

        # Sample varied v2.2-compatible student profiles across the 6 Python skills
        base_strength = self.rng.betavariate(2.0, 2.2)
        base_rate = self.rng.uniform(0.10, 0.28)
        base_hl = math.exp(self.rng.uniform(math.log(5.0), math.log(45.0)))
        base_speed = self.rng.lognormvariate(0.0, 0.22)
        help_depend = self.rng.uniform(0.25, 0.90)

        self.params = {}
        self.latent = {}
        self.records = {}
        self.metrics_cache = {}
        self.unverified_assisted = {}
        self.last_score = {}
        self.last_hints = {}
        self.last_leakage = {}
        self.attempts_on_topic = {}
        self.visited_topics = {SKILLS_ORDER[0]}

        for skill in SKILLS_ORDER:
            k0 = _SIM.clamp(base_strength + self.rng.uniform(-0.18, 0.18), 0.04, 0.85)
            # Deliberately leave some downstream skills lower so adaptive curriculum matters
            if skill in ("accumulator_initialization", "function_returns"):
                k0 = _SIM.clamp(k0 * 0.78, 0.03, 0.72)
            self.params[skill] = {
                "initial_strength": k0,
                "learning_rate": _SIM.clamp(base_rate * self.rng.lognormvariate(0.0, 0.22), 0.06, 0.36),
                "half_life_days": _SIM.clamp(base_hl * self.rng.lognormvariate(0.0, 0.30), 2.5, 90.0),
                "speed": _SIM.clamp(base_speed * self.rng.lognormvariate(0.0, 0.15), 0.5, 2.2),
                "help_effect": self.rng.uniform(0.7, 1.7),
                "help_dependence": help_depend,
            }
            self.latent[skill] = {
                "k": k0,
                "floor": 0.60 * k0,
                "n_right": 0.0,
                "n_wrong": 0.0,
                "last_time": self.clock,
            }
            self.records[skill] = self.tracker.initial_record()
            self.metrics_cache[skill] = self.tracker.query_metrics(self.records[skill], self.clock)
            self.unverified_assisted[skill] = False
            self.last_score[skill] = 0.5
            self.last_hints[skill] = 0.0
            self.last_leakage[skill] = 0.0
            self.attempts_on_topic[skill] = 0

        return self._observation(), self.valid_action_mask()

    def _current_skill(self):
        return SKILLS_ORDER[self.topic_idx]

    def _latent_half_life(self, skill):
        st = self.latent[skill]
        h0 = self.params[skill]["half_life_days"]
        exp_term = 0.35 * (math.sqrt(1.0 + st["n_right"]) - 1.0) - 0.15 * (math.sqrt(1.0 + st["n_wrong"]) - 1.0)
        return _SIM.clamp(h0 * (2.0 ** exp_term), 2.0, 240.0)

    def _latent_strength_at(self, skill, clock_sec):
        st = self.latent[skill]
        days = max(0.0, (clock_sec - st["last_time"]) / DAY)
        floor = min(st["k"], st["floor"])
        return _SIM.retained(st["k"], days, self._latent_half_life(skill), self.mechanism, floor=floor)

    def _prereq_factor(self, skill):
        prereqs = PREREQUISITES[skill]
        if not prereqs:
            return 0.0, 1.0
        min_pre = min(self._latent_strength_at(p, self.clock) for p in prereqs)
        if min_pre >= 0.55:
            return 0.0, 1.0
        deficit = 0.45 * (0.55 - min_pre)
        lr_scale = _SIM.clamp(0.45 + 0.55 * (min_pre / 0.55), 0.35, 1.0)
        return deficit, lr_scale

    def _refresh_metrics(self, skill):
        self.metrics_cache[skill] = self.tracker.query_metrics(self.records[skill], self.clock)

    def _refresh_all_metrics(self):
        for skill in SKILLS_ORDER:
            self._refresh_metrics(skill)

    def valid_action_mask(self):
        mask = np.zeros(len(ACTIONS), dtype=bool)
        if self.done:
            mask[1] = True
            return mask
        if self.awaiting_question:
            if self.difficulty_idx > 0:
                mask[0] = True
            mask[1] = True
            if self.difficulty_idx < 2:
                mask[2] = True
            return mask

        if self.difficulty_idx > 0:
            mask[0] = True
        mask[1] = True
        if self.difficulty_idx < 2:
            mask[2] = True
        if self.primed_mode is None:
            mask[3] = True
            mask[4] = True
        skill = self._current_skill()
        if self.attempts_on_topic[skill] >= 1:
            mask[5] = True
            mask[7] = True
        prereqs = PREREQUISITES[skill]
        has_decayed_visited = any(
            s != skill and self.metrics_cache[s]["p_30d"] < 0.58 for s in self.visited_topics
        )
        if prereqs or has_decayed_visited:
            mask[6] = True
        return mask

    def _observation(self):
        skill = self._current_skill()
        m = self.metrics_cache[skill]
        prereqs = PREREQUISITES[skill]
        if prereqs:
            pre_med = min(self.metrics_cache[p]["p_med"] for p in prereqs)
            pre_floor = min(self.metrics_cache[p]["floor_b"] for p in prereqs)
        else:
            pre_med = 1.0
            pre_floor = 1.0

        p30_all = [self.metrics_cache[s]["p_30d"] for s in SKILLS_ORDER]
        diff_one_hot = [float(self.difficulty_idx == i) for i in range(3)]
        topic_one_hot = [float(self.topic_idx == i) for i in range(len(SKILLS_ORDER))]

        vec = np.array([
            m["p_easy"],
            m["p_med"],
            m["p_hard"],
            m["p_7d"],
            m["p_30d"],
            m["fragility"],
            m["floor_b"],
            m["active_k"],
            float(self.unverified_assisted[skill]),
            self.last_score[skill],
            min(1.0, self.last_hints[skill] / 3.0),
            self.last_leakage[skill],
            min(1.0, self.attempts_on_topic[skill] / 10.0),
            float(self.awaiting_question),
            float(self.primed_mode == "concept"),
            float(self.primed_mode == "example"),
            *diff_one_hot,
            pre_med,
            pre_floor,
            max(0.0, (TOTAL_BUDGET - self.budget_used) / TOTAL_BUDGET),
            min(p30_all),
            float(np.mean(p30_all)),
            *p30_all,
            *topic_one_hot,
        ], dtype=np.float32)
        return vec

    def _pick_task(self, skill, difficulty, pool="practice"):
        candidates = self.pools[(skill, difficulty, pool)]
        unused = [t for t in candidates if t["task_id"] not in self.used_tasks]
        chosen = self.rng.choice(unused if unused else candidates)
        if pool == "practice":
            self.used_tasks.add(chosen["task_id"])
        return chosen

    def _emit_question_turn(self, skill, difficulty, *, allow_help=True):
        task = self._pick_task(skill, difficulty, "practice")
        task_feats = _task_model_features(task)
        self.tracker.observe_event(
            self.records[skill],
            self.clock,
            {"event_type": "task", "new_session": False, "task": task_feats},
        )

        k_now = self._latent_strength_at(skill, self.clock)
        prereq_def, lr_scale = self._prereq_factor(skill)
        primed_boost = 0.09 if self.primed_mode == "concept" else (0.11 if self.primed_mode == "example" else 0.0)
        k_eff = _SIM.clamp(k_now - prereq_def + primed_boost, 0.01, 0.99)

        hints_used = 0
        leakages = []
        last_result = None
        last_attempt_num = 1
        first_attempt_success = False
        first_attempt_score = 0.0

        for attempt_num in range(1, 4):
            last_attempt_num = attempt_num
            # Simulate response via v2.2 rubric mechanics
            fmt = task["task_type"]
            rules = _SIM.FORMAT_RULES[fmt]
            diff_demand = (0.25, 0.45, 0.65)[DIFFICULTIES.index(difficulty)]
            help_bonus = rules["help_multiplier"] * self.params[skill]["help_effect"] * (
                sum(0.55 + 1.15 * lk for lk in leakages) if leakages else 0.0
            )
            z = 5.0 * (k_eff - diff_demand) + 0.65 + self.rng.gauss(0.0, 0.55) + help_bonus
            qs = [0.97 * _SIM.sigmoid(z - 0.12 * j) for j in range(len(task["criteria"]))]
            if fmt == "single_choice":
                ev = task["answer_key"] if self.rng.random() < qs[0] else self.rng.choice(task["options"])["id"]
            elif fmt == "multiple_select":
                correct = set(task["answer_key"])
                ev = [
                    o["id"] for o, q in zip(task["options"], qs)
                    if (o["id"] in correct) == (self.rng.random() < (0.5 + 0.5 * q))
                ]
            else:
                probs = [rules["guess"] + (1.0 - rules["guess"]) * q for q in qs]
                ev = [self.rng.random() < p for p in probs]
            truth = _CATALOG.grade(task, ev)
            last_result = truth

            if not truth["full_success"] and self.rng.random() < 0.35:
                duration = _SIM.clamp(task["reference_seconds"] * self.rng.uniform(0.05, 0.25), 3.0, 900.0)
            else:
                duration = _SIM.clamp(
                    task["reference_seconds"] * self.params[skill]["speed"] * (1.15 - 0.35 * k_now)
                    * self.rng.lognormvariate(0.0, 0.32) / (1.0 + 0.12 * hints_used),
                    3.0,
                    900.0,
                )
            self.clock += duration
            assistance = "assisted" if hints_used > 0 else "independent"
            max_leak = max(leakages, default=0.0)
            revealed = bool(max_leak >= 0.90)

            self.tracker.observe_event(
                self.records[skill],
                self.clock,
                {
                    "event_type": "response",
                    "new_session": False,
                    "task": task_feats,
                    "attempt_number": attempt_num,
                    "assistance": assistance,
                    "hint_count": hints_used,
                    "answer_revealed": revealed,
                    "active_response_seconds": duration,
                    "relative_response_time": duration / task["reference_seconds"],
                },
            )
            self.clock += 1.0
            self.tracker.observe_event(
                self.records[skill],
                self.clock,
                {
                    "event_type": "grade",
                    "new_session": False,
                    "task": task_feats,
                    "attempt_number": attempt_num,
                    "assistance": assistance,
                    "hint_count": hints_used,
                    "answer_revealed": revealed,
                    "score": float(truth["score"]),
                    "full_success": bool(truth["full_success"]),
                    "grading_status": "resolved",
                },
            )

            if attempt_num == 1:
                first_attempt_success = bool(truth["full_success"])
                first_attempt_score = float(truth["score"])

            if truth["full_success"] or not allow_help or attempt_num == 3:
                break

            # Student asks LLM for 1 hint; Evaluator measures leakage l_t in [0, 1]
            if self.rng.random() > self.params[skill]["help_dependence"] * 0.90 + 0.10:
                break
            leakage = _SIM.clamp(0.18 + 0.25 * hints_used + self.rng.uniform(0.0, 0.45), 0.05, 0.95)
            leakages.append(leakage)
            hints_used += 1
            self.clock += 10.0
            self.tracker.observe_event(
                self.records[skill],
                self.clock,
                {
                    "event_type": "hint",
                    "new_session": False,
                    "task": task_feats,
                    "hint_count": hints_used,
                    "hint_type": _leakage_to_hint_kind(leakage),
                    "assistance": "assisted",
                },
            )
            self.clock += 3.0

        # Apply latent cognitive learning + consolidation update
        before = self._latent_strength_at(skill, self.clock)
        st = self.latent[skill]
        cur_floor = min(before, st["floor"])
        diff_level = DIFFICULTIES.index(difficulty)
        diff_dose = (0.38, 0.54, 0.72)[diff_level]
        diff_cons = (0.50, 0.62, 0.72)[diff_level]
        prime_mult = 1.45 if self.primed_mode is not None else 1.0
        score = float(last_result["score"])
        full = bool(last_result["full_success"])
        max_leak = max(leakages, default=0.0)

        if hints_used == 0:
            factor = score * (1.0 if full else 0.45)
            dose = diff_dose * factor * prime_mult * lr_scale
            after = _SIM.learned(before, self.params[skill]["learning_rate"], dose)
            cons = diff_cons if full else 0.25
            new_floor = min(after, cur_floor + cons * (after - before))
            n_right = st["n_right"] + (1.0 if full else 0.0)
            n_wrong = st["n_wrong"] + (0.0 if full else 1.0)
            if full:
                self.unverified_assisted[skill] = False
        else:
            leak_discount = max(0.15, 1.0 - 0.75 * max_leak)
            factor = (0.15 + 0.55 * score) * leak_discount
            dose = diff_dose * factor * prime_mult * lr_scale / max(1, last_attempt_num)
            after = _SIM.learned(before, self.params[skill]["learning_rate"], dose)
            cons = 0.16 * max(0.10, 1.0 - 0.80 * max_leak)
            new_floor = min(after, cur_floor + cons * (after - before))
            n_right = st["n_right"] + (0.25 * leak_discount if full else 0.0)
            n_wrong = st["n_wrong"] + (0.25 if full else 0.8)
            if full or max_leak > 0.25:
                self.unverified_assisted[skill] = True

        self.latent[skill] = {
            "k": after,
            "floor": new_floor,
            "n_right": n_right,
            "n_wrong": n_wrong,
            "last_time": self.clock,
        }

        self.attempts_on_topic[skill] += 1
        self.last_score[skill] = score
        self.last_hints[skill] = float(hints_used)
        self.last_leakage[skill] = float(max_leak)
        self.primed_mode = None
        self.awaiting_question = False
        self._refresh_metrics(skill)

        return {
            "first_attempt_success": first_attempt_success,
            "first_attempt_score": first_attempt_score,
            "final_score": score,
            "final_full_success": full,
            "hints_used": hints_used,
            "max_leakage": max_leak,
        }

    def _deliver_instruction(self, skill, action_name):
        self.clock += 25.0
        self.tracker.observe_event(
            self.records[skill],
            self.clock,
            {"event_type": "instruction", "new_session": False, "task": None},
        )
        k_now = self._latent_strength_at(skill, self.clock)
        eta = self.params[skill]["learning_rate"]
        st = self.latent[skill]
        cur_floor = min(k_now, st["floor"])
        if action_name == "explain_concept":
            # Strongest for beginners (1 - k)^2
            gain = 0.14 * eta * ((1.0 - k_now) ** 2)
            self.primed_mode = "concept"
        else:
            # Worked example aids procedural fluency
            gain = 0.10 * eta * (1.0 - k_now)
            self.primed_mode = "example"
        k_new = _SIM.clamp(k_now + gain, 0.01, 0.99)
        self.latent[skill] = {
            "k": k_new,
            "floor": min(k_new, cur_floor + 0.25 * gain),
            "n_right": st["n_right"],
            "n_wrong": st["n_wrong"],
            "last_time": self.clock,
        }
        self.awaiting_question = True
        self._refresh_metrics(skill)

    def _run_self_explanation(self, skill):
        task = self._pick_task(skill, "medium", "practice")
        task_feats = _task_model_features(task)
        self.tracker.observe_event(
            self.records[skill],
            self.clock,
            {"event_type": "task", "new_session": False, "task": task_feats},
        )
        k_now = self._latent_strength_at(skill, self.clock)
        prereq_def, lr_scale = self._prereq_factor(skill)
        primed_bonus = 0.08 if self.primed_mode is not None else 0.0
        k_eff = _SIM.clamp(k_now - prereq_def + primed_bonus, 0.01, 0.99)
        p_pass = _SIM.sigmoid(6.0 * (k_eff - 0.38))
        full = self.rng.random() < p_pass
        score = 1.0 if full else (0.5 if self.rng.random() < 0.45 * p_pass else 0.0)
        duration = _SIM.clamp(45.0 * self.params[skill]["speed"] * self.rng.lognormvariate(0.0, 0.25), 8.0, 300.0)
        self.clock += duration

        self.tracker.observe_event(
            self.records[skill],
            self.clock,
            {
                "event_type": "response",
                "new_session": False,
                "task": task_feats,
                "attempt_number": 1,
                "assistance": "independent",
                "hint_count": 0,
                "answer_revealed": False,
                "active_response_seconds": duration,
                "relative_response_time": duration / 45.0,
            },
        )
        self.clock += 1.0
        self.tracker.observe_event(
            self.records[skill],
            self.clock,
            {
                "event_type": "grade",
                "new_session": False,
                "task": task_feats,
                "attempt_number": 1,
                "assistance": "independent",
                "hint_count": 0,
                "answer_revealed": False,
                "score": score,
                "full_success": full,
                "grading_status": "resolved",
            },
        )

        # Self-explanation strongly consolidates existing transient fluency into permastore floor & half-life
        st = self.latent[skill]
        before = self._latent_strength_at(skill, self.clock)
        cur_floor = min(before, st["floor"])
        if full:
            after = _SIM.learned(before, self.params[skill]["learning_rate"], 0.48 * lr_scale)
            transient = max(0.0, after - cur_floor)
            new_floor = min(after, cur_floor + 0.62 * transient + 0.08 * (1.0 - cur_floor))
            n_right = st["n_right"] + 1.5
            n_wrong = st["n_wrong"]
            self.unverified_assisted[skill] = False
        else:
            after = _SIM.learned(before, self.params[skill]["learning_rate"], 0.15 * score * lr_scale)
            new_floor = min(after, cur_floor + 0.20 * (after - before))
            n_right = st["n_right"]
            n_wrong = st["n_wrong"] + 0.6

        self.latent[skill] = {
            "k": after,
            "floor": new_floor,
            "n_right": n_right,
            "n_wrong": n_wrong,
            "last_time": self.clock,
        }
        self.attempts_on_topic[skill] += 1
        self.last_score[skill] = score
        self.last_hints[skill] = 0.0
        self.last_leakage[skill] = 0.0
        self.primed_mode = None
        self.awaiting_question = False
        self._refresh_metrics(skill)
        return full, score

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode already finished; call reset()")
        mask = self.valid_action_mask()
        if not mask[action]:
            raise ValueError(f"Invalid masked action {action} ({ACTIONS[action]})")

        prev_p30 = float(np.mean([self.metrics_cache[s]["p_30d"] for s in SKILLS_ORDER]))
        prev_floor = float(np.mean([self.metrics_cache[s]["floor_b"] for s in SKILLS_ORDER]))
        skill = self._current_skill()
        cleared_unverified = False
        premature_advance = False
        immediate_score = None

        if action in (0, 1, 2):
            if action == 0:
                self.difficulty_idx = max(0, self.difficulty_idx - 1)
            elif action == 2:
                self.difficulty_idx = min(2, self.difficulty_idx + 1)
            was_unverified = self.unverified_assisted[skill]
            # Choosing same difficulty when unverified triggers an unaided verification check
            allow_help = not (action == 1 and was_unverified)
            outcome = self._emit_question_turn(skill, DIFFICULTIES[self.difficulty_idx], allow_help=allow_help)
            immediate_score = outcome["first_attempt_score"]
            if was_unverified and not self.unverified_assisted[skill]:
                cleared_unverified = True
        elif action in (3, 4):
            self._deliver_instruction(skill, ACTIONS[action])
        elif action == 5:
            was_unverified = self.unverified_assisted[skill]
            _, se_score = self._run_self_explanation(skill)
            immediate_score = se_score
            if was_unverified and not self.unverified_assisted[skill]:
                cleared_unverified = True
        elif action == 6:
            prereqs = PREREQUISITES[skill]
            if prereqs:
                target = min(prereqs, key=lambda p: self.metrics_cache[p]["p_30d"])
            else:
                target = min(self.visited_topics, key=lambda s: self.metrics_cache[s]["p_30d"])
            self.topic_idx = SKILLS_ORDER.index(target)
            self.visited_topics.add(target)
            self.difficulty_idx = 0 if self.metrics_cache[target]["p_med"] < 0.48 else 1
            self.primed_mode = "concept"
            outcome = self._emit_question_turn(target, DIFFICULTIES[self.difficulty_idx], allow_help=True)
            immediate_score = outcome["first_attempt_score"]
        elif action == 7:
            # Check if advancing while current topic still has high fragility or unverified flag
            cur_m = self.metrics_cache[skill]
            if self.unverified_assisted[skill] or cur_m["p_30d"] < 0.45:
                premature_advance = True
            unvisited = [s for s in SKILLS_ORDER if s not in self.visited_topics]
            if unvisited:
                next_skill = unvisited[0]
            else:
                others = [s for s in SKILLS_ORDER if s != skill]
                next_skill = min(others, key=lambda s: self.metrics_cache[s]["p_30d"])
            self.topic_idx = SKILLS_ORDER.index(next_skill)
            self.visited_topics.add(next_skill)
            self.difficulty_idx = 0
            self.primed_mode = None
            self.awaiting_question = False

        self.budget_used += ACTION_COSTS[action]
        if self.budget_used >= self.next_session_threshold and self.budget_used < TOTAL_BUDGET:
            gap_days = 4.0 if self.long_gaps else 1.5
            self.clock += gap_days * DAY
            self.session += 1
            self.next_session_threshold += SESSION_BUDGET
            self._refresh_all_metrics()

        new_p30 = float(np.mean([self.metrics_cache[s]["p_30d"] for s in SKILLS_ORDER]))
        new_floor = float(np.mean([self.metrics_cache[s]["floor_b"] for s in SKILLS_ORDER]))
        shaping_reward = (
            2.0 * (new_p30 - prev_p30)
            + 1.5 * (new_floor - prev_floor)
            + (0.08 if cleared_unverified else 0.0)
            - (0.04 if premature_advance else 0.0)
        )

        self.done = self.budget_used >= TOTAL_BUDGET - 1e-6
        info = {
            "action_name": ACTIONS[action],
            "immediate_score": immediate_score,
            "mean_p30_model": new_p30,
            "mean_floor_model": new_floor,
        }
        reward = shaping_reward
        if self.done:
            delayed = self.evaluate_delayed_retention()
            reward += 10.0 * delayed["delayed_mean"]
            info["delayed"] = delayed
        return self._observation(), float(reward), self.done, self.valid_action_mask(), info

    def evaluate_delayed_retention(self, seed_offset=99991):
        """Evaluate cloned end-of-episode states across independent 1d, 7d, and 30d checks."""
        eval_rng = random.Random(self.student_seed * 100003 + seed_offset)
        horizon_scores = {}
        horizon_expected = {}
        latent_floors = [min(self._latent_strength_at(s, self.clock), self.latent[s]["floor"]) for s in SKILLS_ORDER]
        latent_strengths = [self._latent_strength_at(s, self.clock) for s in SKILLS_ORDER]

        for delay in (1, 7, 30):
            target_clock = self.clock + delay * DAY
            scores = []
            expected_probs = []
            for skill in SKILLS_ORDER:
                k_d = self._latent_strength_at(skill, target_clock)
                prereqs = PREREQUISITES[skill]
                pre_def = 0.0
                if prereqs:
                    min_pre = min(self._latent_strength_at(p, target_clock) for p in prereqs)
                    if min_pre < 0.55:
                        pre_def = 0.45 * (0.55 - min_pre)
                k_eff = _SIM.clamp(k_d - pre_def, 0.01, 0.99)
                for diff_idx, diff in enumerate(DIFFICULTIES):
                    demand = (0.25, 0.45, 0.65)[diff_idx]
                    p_pass = _SIM.sigmoid(5.0 * (k_eff - demand) + 0.65)
                    expected_probs.append(p_pass)
                    scores.append(1.0 if eval_rng.random() < p_pass else 0.0)
            horizon_scores[f"day_{delay}"] = float(np.mean(scores))
            horizon_expected[f"day_{delay}_expected"] = float(np.mean(expected_probs))

        return {
            **horizon_scores,
            **horizon_expected,
            "delayed_mean": float(np.mean([horizon_scores[f"day_{d}"] for d in (1, 7, 30)])),
            "delayed_expected_mean": float(np.mean([horizon_expected[f"day_{d}_expected"] for d in (1, 7, 30)])),
            "latent_mean_floor": float(np.mean(latent_floors)),
            "latent_mean_strength": float(np.mean(latent_strengths)),
        }
