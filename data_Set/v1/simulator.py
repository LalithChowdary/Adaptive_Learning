"""Numerical learner simulator, not a model of actual Python reasoning or Jev.

Only observations go to replay.py; private parameters and traces are separate.
Standard library only. See DATASET_DOCUMENTATION.md for equations and assumptions.
"""
from __future__ import annotations
import copy
import hashlib
import math
import random
from dataclasses import dataclass, asdict
from catalog import SKILLS, DIFFICULTIES, PREREQUISITES, make_catalog

VERSION = "1.0.0"
DAY = 86400.0
ORIGIN = 1794009600.0  # fixed synthetic Unix seconds, not real learner dates
TASKS = make_catalog()
TASK_INDEX = {t["task_id"]: t for t in TASKS}
POOLS = {(s, p): [t for t in TASKS if t["skill_id"] == s and t["pool"] == p]
         for s in SKILLS for p in ("practice", "assessment")}


def rng_for(seed, identity, stream):
    raw = hashlib.sha256(f"{VERSION}|{seed}|{identity}|{stream}".encode()).digest()
    return random.Random(int.from_bytes(raw[:16], "big"))


def clamp(x, low, high):
    return max(low, min(high, x))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-clamp(x, -40, 40)))


def retained(strength, elapsed_days, half_life, mechanism="exponential"):
    if not (0 <= strength <= 1 and elapsed_days >= 0 and half_life > 0):
        raise ValueError("invalid retention arguments")
    if mechanism == "exponential":
        return strength * 2 ** (-elapsed_days / half_life)
    if mechanism == "power_law":
        return strength / (1 + elapsed_days / half_life)
    raise ValueError("unknown retention mechanism")


def learned(strength, rate, dose):
    if not (0 <= strength <= 1 and rate >= 0 and dose >= 0):
        raise ValueError("invalid learning arguments")
    return strength + (1 - strength) * (-math.expm1(-rate * dose))


@dataclass(frozen=True)
class Config:
    seed: int = 20261107
    tasks_per_skill: int = 16
    tasks_per_session: int = 12
    mechanism: str = "exponential"
    long_gaps: bool = False
    unresolved_probability: float = 0.03
    missing_time_probability: float = 0.04
    missing_reference_probability: float = 0.04
    missing_assistance_probability: float = 0.02
    assessment_days: tuple = (1, 7, 30)

    def __post_init__(self):
        if self.tasks_per_skill <= 0 or self.tasks_per_session <= 0:
            raise ValueError("positive task budgets required")
        if self.mechanism not in ("exponential", "power_law"):
            raise ValueError("unknown mechanism")
        for p in (self.unresolved_probability, self.missing_time_probability,
                  self.missing_reference_probability, self.missing_assistance_probability):
            if not 0 <= p <= 1:
                raise ValueError("probability out of range")
        if any(d <= 0 for d in self.assessment_days):
            raise ValueError("positive assessment delays required")


class Student:
    def __init__(self, learner_id, config=None):
        self.learner_id = learner_id
        self.config = config or Config()
        self.rng = {s: rng_for(self.config.seed, learner_id, s) for s in
                    ("parameters", "policy", "tasks", "outcomes", "timing", "missing", "gaps")}
        self.clock = ORIGIN
        self.branch = "practice"
        self.session = 0
        self.events = []
        self.trace = []
        self.params = {}
        self.states = {}
        self.opportunity = 0
        self.used_tasks = set()
        r = self.rng["parameters"]
        base = dict(strength=r.betavariate(2, 2), learning=r.uniform(.08, .28),
                    half_life=math.exp(r.uniform(math.log(5), math.log(60))),
                    speed=r.lognormvariate(0, .25))
        for skill in SKILLS:
            p = dict(initial_strength=clamp(base["strength"] + r.uniform(-.2, .2), .02, .95),
                     learning_rate=clamp(base["learning"] * r.lognormvariate(0, .25), .04, .4),
                     half_life_days=clamp(base["half_life"] * r.lognormvariate(0, .35), 2, 120),
                     speed=clamp(base["speed"] * r.lognormvariate(0, .15), .4, 2.5),
                     help_effect=r.uniform(.6, 1.8),
                     retained_help=0.0 if r.random() < .25 else r.uniform(.05, .5),
                     action_doses={a: r.uniform(.35, 1.1) for a in
                                   ("explain", "worked_example", "prerequisite_review", "self_explanation")})
            self.params[skill] = p
            self.states[skill] = (p["initial_strength"], self.clock)

    def advance(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("time must advance by a finite nonnegative amount")
        self.clock += seconds

    def strength(self, skill):
        k, when = self.states[skill]
        return retained(k, (self.clock - when) / DAY, self.params[skill]["half_life_days"],
                        self.config.mechanism)

    def learn(self, skill, dose):
        before = self.strength(skill)
        after = learned(before, self.params[skill]["learning_rate"], dose)
        self.states[skill] = after, self.clock
        return before, after

    def emit(self, kind, skill, interaction_id, **fields):
        event = dict(schema_version=VERSION, event_id=f"{self.learner_id}/{self.branch}/e{len(self.events):05d}",
                     learner_id=self.learner_id, skill_id=skill, course_id="python_basics_v1",
                     session_id=f"s{self.session}", branch=self.branch, interaction_id=interaction_id,
                     event_type=kind, occurred_at=self.clock, available_at=self.clock)
        event.update(fields)
        self.events.append(event)
        return event

    def instruct(self, skill, action, interaction_id):
        if action not in self.params[skill]["action_doses"]:
            raise ValueError("not a standalone instruction action")
        self.advance(45)
        before, after = self.learn(skill, self.params[skill]["action_doses"][action])
        self.emit("instruction", skill, interaction_id, teaching_action=action)
        self.trace.append(dict(kind="instruction", skill_id=skill, at=self.clock,
                               before=before, after=after, action=action))

    def assistance(self, hint_count, kind, revealed):
        # Unknown assistance remains unknown, never independent by default.
        if self.rng["missing"].random() < self.config.missing_assistance_probability:
            return dict(hint_count=None, hint_type=None, scaffold=None, answer_revealed=None,
                        assistance="unknown")
        return dict(hint_count=hint_count, hint_type=kind, scaffold=kind == "scaffold",
                    answer_revealed=revealed,
                    assistance="assisted" if hint_count or revealed else "independent")

    def attempt(self, task, interaction_id, attempt, help_kind=None, assessment=False,
                forced_criteria=None):
        skill = task["skill_id"]
        p = self.params[skill]
        k = self.strength(skill)
        level = DIFFICULTIES.index(task["difficulty"])
        # Task demand is hidden but stable per task; task identity never enters memory.
        task_offset = rng_for(self.config.seed, task["task_id"], "demand").uniform(-.12, .12)
        demand = (.25, .45, .65)[level] + task_offset
        help_multiplier = {None: 0.0, "conceptual": 1.0, "scaffold": 1.6, "reveal": 0.0}[help_kind]
        shared_noise = self.rng["outcomes"].gauss(0, .65)
        logits = [5 * (k - demand) + .65 + help_multiplier * p["help_effect"] + shared_noise + offset
                  for offset in (.15, -.15)]
        probabilities = [.03 + .94 * sigmoid(z) for z in logits]
        criteria = [self.rng["outcomes"].random() < q for q in probabilities]
        if help_kind == "reveal":
            criteria = [True, True]
        if forced_criteria is not None:
            criteria = list(forced_criteria)  # controlled fixtures only, never stochastic cohorts
        duration = clamp(task["reference_seconds"] * p["speed"] * (1.25 - .5 * k)
                         * self.rng["timing"].lognormvariate(0, .35)
                         * (.75 if help_kind else 1.0), 3, 600)
        self.advance(duration)
        help_count = int(help_kind is not None)
        assistance = self.assistance(help_count, help_kind, help_kind == "reveal")
        active = None if self.rng["missing"].random() < self.config.missing_time_probability else duration
        response = self.emit("response", skill, interaction_id, task_id=task["task_id"],
                             attempt_number=attempt, active_response_seconds=active, **assistance)
        before, after = (self.strength(skill), self.strength(skill))
        if not assessment:
            dose = .45 if help_kind is None else .45 * p["retained_help"]
            before, after = self.learn(skill, dose)
        self.advance(self.rng["timing"].choice((.1, 2.0, 30.0)))
        resolved = self.rng["missing"].random() >= self.config.unresolved_probability
        grade = self.emit("grade", skill, interaction_id, task_id=task["task_id"],
                          response_id=response["event_id"], grading_status="resolved" if resolved else "unresolved",
                          criteria=criteria if resolved else None,
                          score=sum(criteria) / 2 if resolved else None,
                          full_success=all(criteria) if resolved else None,
                          rubric_id="two_required_outputs_v1", grader="simulated_rubric_v1",
                          grader_confidence=None)
        self.trace.append(dict(kind="response", response_id=response["event_id"], skill_id=skill,
                               probabilities=probabilities, criteria_truth=criteria, hidden_demand=demand,
                               strength_before=before, strength_after=after))
        return response, grade

    def task(self, skill, task=None, assessment=False, forced_first=None, forced_retry=None):
        self.opportunity += 1
        interaction_id = f"{self.learner_id}/{self.branch}/q{self.opportunity:04d}"
        if task is None:
            candidates = [t for t in POOLS[skill, "assessment" if assessment else "practice"]
                          if t["task_id"] not in self.used_tasks]
            if not candidates:
                raise ValueError("task bank exhausted; add reviewed tasks before increasing per-skill budget")
            task = self.rng["tasks"].choice(candidates)
        if task["skill_id"] != skill:
            raise ValueError("task skill mismatch")
        self.used_tasks.add(task["task_id"])
        reference = None if self.rng["missing"].random() < self.config.missing_reference_probability else task["reference_seconds"]
        self.emit("task", skill, interaction_id, task_id=task["task_id"], difficulty=task["difficulty"],
                  task_type=task["task_type"], reference_seconds=reference)
        response, grade = self.attempt(task, interaction_id, 1, assessment=assessment, forced_criteria=forced_first)
        if not assessment and grade["full_success"] is False and (forced_retry is not None or self.rng["policy"].random() < .65):
            kind = "scaffold" if forced_retry is not None else self.rng["policy"].choices(
                ["conceptual", "scaffold", "reveal"], [.5, .4, .1])[0]
            self.advance(15)
            before, after = self.learn(skill, .25 * self.params[skill]["retained_help"])
            self.emit("help", skill, interaction_id, task_id=task["task_id"], hint_count=1,
                      hint_type=kind, scaffold=kind == "scaffold", answer_revealed=kind == "reveal",
                      teaching_action="partial_scaffold" if kind == "scaffold" else None)
            self.trace.append(dict(kind="help", skill_id=skill, before=before, after=after))
            self.attempt(task, interaction_id, 2, help_kind=kind, forced_criteria=forced_retry)

    def practice(self, schedule):
        skills = list(SKILLS)
        r = self.rng["policy"]
        order = skills[:]
        r.shuffle(order)
        if schedule == "blocked":
            sequence = [s for s in order for _ in range(self.config.tasks_per_skill)]
        elif schedule == "mixed":
            sequence = []
            for _ in range(self.config.tasks_per_skill):
                block = skills[:]
                r.shuffle(block)
                sequence.extend(block)
        else:
            raise ValueError("unknown practice schedule")
        for i, skill in enumerate(sequence):
            if i % self.config.tasks_per_session == 0:
                self.session += 1
                if i:
                    gaps = (.25, 1, 3, 14, 60) if self.config.long_gaps else (.25, 1, 3)
                    self.advance(self.rng["gaps"].choice(gaps) * DAY)
            if r.random() < .65:
                action = r.choice(("explain", "worked_example", "self_explanation", "prerequisite_review"))
                taught_skill = skill
                if action == "prerequisite_review":
                    if skill in PREREQUISITES:
                        taught_skill = PREREQUISITES[skill]
                    else:
                        action = "explain"
                self.instruct(taught_skill, action, f"lesson_{i}")
            self.task(skill)
        return self

    def assessment(self, delay_days, skill):
        """Independent fork from end of practice: no earlier probe teaches this fork."""
        clone = copy.copy(self)
        clone.states = self.states.copy()
        clone.used_tasks = self.used_tasks.copy()
        clone.branch = f"assessment_{delay_days}d_{skill}"
        clone.events, clone.trace = [], []
        clone.rng = {s: rng_for(self.config.seed, self.learner_id + clone.branch, s) for s in self.rng}
        clone.advance(delay_days * DAY)
        clone.session += 1
        bank = POOLS[skill, "assessment"][:]
        rng_for(self.config.seed, self.learner_id + skill, "assessment_bank").shuffle(bank)
        index = list(self.config.assessment_days).index(delay_days)
        clone.task(skill, task=bank[index], assessment=True)
        return clone

    def private_record(self):
        return dict(learner_id=self.learner_id, config=asdict(self.config), parameters=self.params,
                    final_states=self.states, trace=self.trace)
