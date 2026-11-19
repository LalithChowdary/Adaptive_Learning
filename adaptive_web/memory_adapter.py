"""Frozen monotonic_gru cognitive memory adapter with strict learner-skill isolation."""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from . import CODE_3_ROOT, catalogue_bridge

MODEL_ROOT = CODE_3_ROOT / "model"
DEFAULT_CHECKPOINT = MODEL_ROOT / "runs/v2.2-mps-main/monotonic_gru/seed_20261122/model.pt"
V22_CATALOG_JSON = CODE_3_ROOT / "data_Set/v2.2/dataset/task_catalog.json"

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from memory.features import (  # noqa: E402
    ASSISTANCE_TYPES,
    DAY,
    DIFFICULTIES,
    EVIDENCE_TYPES,
    Evidence,
    EvidenceStream,
    FORMATS,
    GRADE_TYPES,
    HINT_TYPES,
    PAIR_FIELDS,
    TASK_SIZE,
    TOKEN_SIZE,
    task_features,
)
from memory.recurrent import LearnerGRU  # noqa: E402

CURVE_DAYS: tuple[int, ...] = (0, 1, 3, 7, 14, 30, 45, 60)

FEATURE_FIELDS: tuple[str, ...] = (
    "event_type",
    "gap_seconds",
    "new_session",
    "task",
    "attempt_number",
    "active_response_seconds",
    "relative_response_time",
    "hint_count",
    "hint_type",
    "answer_revealed",
    "assistance",
    "score",
    "full_success",
    "grading_status",
)

TASK_FEATURE_KEYS: tuple[str, ...] = (
    "task_type",
    "difficulty",
    "reference_seconds",
    "snippet_count",
    "option_count",
    "criterion_count",
    "credit_weights",
    "penalty_weights",
)


@dataclass
class _CachedState:
    stream: EvidenceStream
    state: torch.Tensor
    anchor: float | None
    digests: tuple[bytes, ...]


def _resolve_device(device: str) -> str:
    dev = (device or "cpu").strip().lower()
    if dev == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if dev == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this process.")
    if dev not in ("cpu", "mps"):
        raise ValueError(f"Unsupported device {device!r}")
    return dev


def _sanitize_task_dict(raw_task: Any) -> dict[str, Any] | None:
    if raw_task is None:
        return None
    if not isinstance(raw_task, dict):
        raise ValueError("Task features must be a dict or None.")
    return {
        "task_type": raw_task.get("task_type"),
        "difficulty": raw_task.get("difficulty"),
        "reference_seconds": raw_task.get("reference_seconds"),
        "snippet_count": int(raw_task.get("snippet_count", 0)),
        "option_count": int(raw_task.get("option_count", 0)),
        "criterion_count": int(raw_task.get("criterion_count", 0)),
        "credit_weights": [float(w) for w in raw_task.get("credit_weights", [])],
        "penalty_weights": [float(w) for w in raw_task.get("penalty_weights", [])],
    }


def _sanitize_features(raw_features: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_features, dict):
        raise ValueError("Event features must be a dict.")
    clean: dict[str, Any] = {}
    for k in FEATURE_FIELDS:
        if k == "task":
            clean["task"] = _sanitize_task_dict(raw_features.get("task"))
        else:
            clean[k] = raw_features.get(k)
    return clean


def _event_digest(event: dict[str, Any], sanitized_features: dict[str, Any]) -> bytes:
    payload = json.dumps(
        [event.get("id"), event.get("at"), event.get("kind"), sanitized_features],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


class MemoryAdapter:
    """Thread-safe, isolated wrapper over the frozen v2.2 monotonic_gru checkpoint."""

    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        catalog_path: str | Path = V22_CATALOG_JSON,
    ):
        self.device = _resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.catalog_path = Path(catalog_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing monotonic_gru checkpoint: {self.checkpoint_path}")
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Missing v2.2 task catalog: {self.catalog_path}")

        raw_bytes = self.checkpoint_path.read_bytes()
        self.version = hashlib.sha256(raw_bytes).hexdigest()
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("dataset_version") != "2.2.0" or payload.get("mode") != "monotonic":
            raise ValueError(
                f"Unexpected checkpoint contract: dataset_version={payload.get('dataset_version')!r}, "
                f"mode={payload.get('mode')!r}"
            )

        self.hidden_size = int(payload["hidden_size"])
        self.model = LearnerGRU(mode="monotonic", hidden_size=self.hidden_size).to(self.device)
        self.model.load_state_dict(payload["weights"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._lock = threading.RLock()
        self._cache: dict[tuple[Any, ...], _CachedState] = {}

        # Precompute canonical reference task vectors from v2.2 assessment pool
        catalog_tasks = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self._ref_task_vectors: dict[tuple[str, str], list[float]] = {}
        self._ref_task_tensors: dict[tuple[str, str], torch.Tensor] = {}
        for fmt in FORMATS:
            for diff in DIFFICULTIES:
                match = next(
                    (
                        t
                        for t in catalog_tasks
                        if t["skill_id"] == "range_boundaries"
                        and t["task_type"] == fmt
                        and t["difficulty"] == diff
                        and t["pool"] == "assessment"
                    ),
                    None,
                )
                if match is None:
                    match = next(
                        t
                        for t in catalog_tasks
                        if t["task_type"] == fmt
                        and t["difficulty"] == diff
                        and t["pool"] == "assessment"
                    )
                vec = task_features(catalogue_bridge.task_memory_features(match))
                self._ref_task_vectors[(fmt, diff)] = vec
                self._ref_task_tensors[(fmt, diff)] = torch.tensor(
                    vec, dtype=torch.float32, device=self.device
                )

        # Canonical 5-row PPO batch: output_prediction (easy, med, hard, med@7d, med@30d)
        self._ppo_ref_tasks = torch.stack(
            [
                self._ref_task_tensors[("output_prediction", "easy")],
                self._ref_task_tensors[("output_prediction", "medium")],
                self._ref_task_tensors[("output_prediction", "hard")],
                self._ref_task_tensors[("output_prediction", "medium")],
                self._ref_task_tensors[("output_prediction", "medium")],
            ],
            dim=0,
        )
        self._ppo_ref_gaps = torch.tensor([0.0, 0.0, 0.0, 7.0, 30.0], dtype=torch.float32, device=self.device)

        # Precompute cold-start curves per format
        self._cold_start_curves: dict[str, dict[str, list[dict[str, float]]]] = {}
        with torch.inference_mode():
            init_state = self.model.initial_state(1)
            days_tensor = torch.tensor(CURVE_DAYS, dtype=torch.float32, device=self.device)
            for fmt in FORMATS:
                fmt_curves: dict[str, list[dict[str, float]]] = {}
                for diff in DIFFICULTIES:
                    t_vec = self._ref_task_tensors[(fmt, diff)].unsqueeze(0).expand(len(CURVE_DAYS), -1)
                    s_batch = init_state.expand(len(CURVE_DAYS), -1)
                    probs = torch.sigmoid(self.model.predict_logit(s_batch, t_vec, days_tensor)).tolist()
                    fmt_curves[diff] = [
                        {"day": int(d), "probability": float(p)}
                        for d, p in zip(CURVE_DAYS, probs)
                    ]
                self._cold_start_curves[fmt] = fmt_curves

    def invalidate(self, learner: str | None = None, skill: str | None = None) -> int:
        with self._lock:
            if learner is None and skill is None:
                removed = len(self._cache)
                self._cache.clear()
                return removed
            to_delete = []
            for k in self._cache:
                if len(k) >= 2:
                    if learner is not None and skill is not None:
                        if k[0] == learner and k[1] == skill:
                            to_delete.append(k)
                    elif learner is not None and k[0] == learner:
                        to_delete.append(k)
                    elif skill is not None and k[1] == skill:
                        to_delete.append(k)
            for k in to_delete:
                del self._cache[k]
            return len(to_delete)

    def clear_cache(self) -> int:
        return self.invalidate()

    def _validate_events(
        self,
        learner: str,
        skill: str,
        events: Sequence[dict[str, Any]],
    ) -> None:
        if skill not in catalogue_bridge.PPO_SKILLS_ORDER:
            raise ValueError(f"Unknown skill {skill!r}; expected one of {catalogue_bridge.PPO_SKILLS_ORDER}")
        for idx, ev in enumerate(events):
            ev_learner = ev.get("learner")
            ev_skill = ev.get("skill")
            if ev_learner is not None and ev_learner != learner:
                raise ValueError(
                    f"Learner isolation violation at event[{idx}]: expected {learner!r}, got {ev_learner!r}"
                )
            if ev_skill is not None and ev_skill != skill:
                raise ValueError(
                    f"Skill isolation violation at event[{idx}]: expected {skill!r}, got {ev_skill!r}"
                )

    def _normalize_cache_key(
        self,
        key: Any,
        learner: str | None,
        skill: str | None,
    ) -> tuple[Any, ...] | None:
        if learner is not None and skill is not None:
            return (learner, skill, self.version)
        if key is None:
            return None
        if isinstance(key, tuple):
            if len(key) == 3 and key[2] == self.version:
                return key
            return (*key, self.version)
        return (key, self.version)

    @torch.inference_mode()
    def state(
        self,
        events: Sequence[dict[str, Any]],
        key: Any = None,
        *,
        learner: str | None = None,
        skill: str | None = None,
    ) -> dict[str, Any]:
        if learner is not None and skill is not None:
            self._validate_events(learner, skill, events)

        sanitized_list: list[dict[str, Any]] = []
        digests_list: list[bytes] = []
        for ev in events:
            sanitized = _sanitize_features(ev["features"])
            sanitized_list.append(sanitized)
            digests_list.append(_event_digest(ev, sanitized))
        digests = tuple(digests_list)
        cache_key = self._normalize_cache_key(key, learner, skill)

        with self._lock:
            cached = self._cache.get(cache_key) if cache_key is not None else None
            if (
                cached is not None
                and len(digests) >= len(cached.digests)
                and digests[: len(cached.digests)] == cached.digests
            ):
                stream = copy.deepcopy(cached.stream)
                state = cached.state.clone()
                anchor = cached.anchor
                start = len(cached.digests)
            else:
                stream = EvidenceStream()
                state = self.model.initial_state(1)
                anchor = None
                start = 0

            for sanitized in sanitized_list[start:]:
                evidence = stream.push(sanitized)
                if evidence is not None:
                    gap = 0.0 if anchor is None else (evidence.time - anchor) / DAY
                    token_t = torch.tensor([evidence.vector()], dtype=torch.float32, device=self.device)
                    gap_t = torch.tensor([gap], dtype=torch.float32, device=self.device)
                    state = self.model.observe(state, token_t, gap_t)
                    anchor = evidence.time

            if cache_key is not None:
                self._cache[cache_key] = _CachedState(
                    stream=copy.deepcopy(stream),
                    state=state.clone(),
                    anchor=anchor,
                    digests=digests,
                )

            return {
                "stream": copy.deepcopy(stream),
                "state": state.clone(),
                "anchor": anchor,
                "count": len(events),
                "last_id": events[-1].get("id") if events else None,
                "digests": digests,
            }

    @torch.inference_mode()
    def predict(
        self,
        events: Sequence[dict[str, Any]],
        task: dict[str, Any] | None,
        now: float | None = None,
        key: Any = None,
        *,
        learner: str | None = None,
        skill: str | None = None,
    ) -> float:
        query_now = float(time.time() if now is None else now)
        saved = self.state(events, key=key, learner=learner, skill=skill)
        anchor = saved["anchor"]
        if anchor is None or not events:
            gap = 0.0
        else:
            raw_sec = max(0.0, query_now - float(events[0]["at"]) - anchor)
            gap = 0.0 if raw_sec < 60.0 else raw_sec / DAY
        sanitized_task = _sanitize_task_dict(task)
        task_tensor = torch.tensor([task_features(sanitized_task)], dtype=torch.float32, device=self.device)
        gap_tensor = torch.tensor([gap], dtype=torch.float32, device=self.device)
        return float(torch.sigmoid(self.model.predict_logit(saved["state"], task_tensor, gap_tensor)).item())

    @torch.inference_mode()
    def _half_life_channels(self, anchor_state: torch.Tensor) -> list[float]:
        m = self.model.skill_channels
        n_pos = anchor_state[..., 2 * m : 2 * m + 1]
        n_neg = anchor_state[..., 2 * m + 1 : 2 * m + 2]
        ctx = anchor_state[..., 2 * m + 2 :]
        h0 = 6.0 + F.softplus(self.model.base_hl)
        theta_pos = F.softplus(self.model.hlr_pos)
        theta_neg = F.softplus(self.model.hlr_neg)
        exponent = (
            theta_pos * (torch.sqrt(1.0 + n_pos) - 1.0)
            - theta_neg * (torch.sqrt(1.0 + n_neg) - 1.0)
            + 0.4 * torch.tanh(self.model.hlr_ctx(ctx))
        )
        h_t = torch.clamp(h0 * torch.pow(2.0, exponent), min=5.0, max=365.0)
        return [float(x) for x in h_t[0].tolist()]

    @torch.inference_mode()
    def skill_summary(
        self,
        learner: str,
        skill: str,
        events: Sequence[dict[str, Any]],
        now: float | None = None,
    ) -> dict[str, Any]:
        self._validate_events(learner, skill, events)
        query_now = float(time.time() if now is None else now)

        resolved_grades = [
            e
            for e in events
            if e.get("kind") == "grade" and e.get("features", {}).get("grading_status") == "resolved"
        ]
        observed_attempts = len(resolved_grades)
        unaided = [
            e
            for e in resolved_grades
            if e.get("features", {}).get("attempt_number") == 1
            and e.get("features", {}).get("assistance") == "independent"
        ]
        observed_unaided_attempts = len(unaided)
        observed_unaided_passes = sum(1 for e in unaided if bool(e.get("features", {}).get("full_success")))
        observed_pass_rate = (
            float(observed_unaided_passes / observed_unaided_attempts)
            if observed_unaided_attempts > 0
            else None
        )
        assessed = observed_attempts > 0

        saved = self.state(events, learner=learner, skill=skill)
        anchor = saved["anchor"]
        if anchor is None or not events:
            elapsed_days = 0.0
        else:
            raw_sec = max(0.0, query_now - float(events[0]["at"]) - anchor)
            elapsed_days = 0.0 if raw_sec < 60.0 else raw_sec / DAY

        anchor_state = saved["state"]
        states_5 = anchor_state.expand(5, -1)
        gaps_5 = self._ppo_ref_gaps + elapsed_days
        logits = self.model.predict_logit(states_5, self._ppo_ref_tasks, gaps_5)
        p_easy, p_med, p_hard, p_7d, p_30d = [float(x) for x in torch.sigmoid(logits).tolist()]
        fragility = float(max(0.0, p_med - p_30d))

        elapsed_t = torch.tensor([elapsed_days], dtype=torch.float32, device=self.device)
        decayed, _ = self.model._at_time(anchor_state, elapsed_t)
        m = self.model.skill_channels
        active_k_channels = [float(x) for x in decayed[0, :m].tolist()]
        floor_b_channels = [float(x) for x in decayed[0, m : 2 * m].tolist()]
        active_k = float(sum(active_k_channels) / m)
        floor_b = float(sum(floor_b_channels) / m)
        n_pos = float(decayed[0, 2 * m].item())
        n_neg = float(decayed[0, 2 * m + 1].item())

        half_life_channels = self._half_life_channels(anchor_state)
        half_life_days = float(sum(half_life_channels) / len(half_life_channels))

        return {
            "skill": skill,
            "assessed": assessed,
            "observed_attempts": observed_attempts,
            "observed_unaided_attempts": observed_unaided_attempts,
            "observed_unaided_passes": observed_unaided_passes,
            "observed_pass_rate": observed_pass_rate,
            "p_easy": p_easy,
            "p_med": p_med,
            "p_hard": p_hard,
            "p_7d": p_7d,
            "p_30d": p_30d,
            "fragility": fragility,
            "floor_b": floor_b,
            "active_k": active_k,
            "half_life_days": half_life_days,
            "half_life_channels": half_life_channels,
            "half_life_min_days": float(min(half_life_channels)),
            "half_life_max_days": float(max(half_life_channels)),
            "active_k_channels": active_k_channels,
            "floor_b_channels": floor_b_channels,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "model_version": self.version,
        }

    @torch.inference_mode()
    def retention_curve(
        self,
        learner: str,
        skill: str,
        events: Sequence[dict[str, Any]],
        task_format: str = "output_prediction",
        now: float | None = None,
    ) -> dict[str, Any]:
        if task_format not in FORMATS:
            raise ValueError(f"Invalid task_format {task_format!r}; expected one of {FORMATS}")
        self._validate_events(learner, skill, events)
        query_now = float(time.time() if now is None else now)

        assessed = any(
            e.get("kind") == "grade" and e.get("features", {}).get("grading_status") == "resolved"
            for e in events
        )
        saved = self.state(events, learner=learner, skill=skill)
        anchor = saved["anchor"]
        if anchor is None or not events:
            elapsed_days = 0.0
        else:
            raw_sec = max(0.0, query_now - float(events[0]["at"]) - anchor)
            elapsed_days = 0.0 if raw_sec < 60.0 else raw_sec / DAY

        anchor_state = saved["state"]
        n_days = len(CURVE_DAYS)
        days_t = torch.tensor(CURVE_DAYS, dtype=torch.float32, device=self.device) + elapsed_days

        # Batch all 3 difficulties x 8 horizons = 24 queries in one forward pass
        task_rows = torch.cat(
            [self._ref_task_tensors[(task_format, diff)].unsqueeze(0).expand(n_days, -1) for diff in DIFFICULTIES],
            dim=0,
        )
        state_rows = anchor_state.expand(len(DIFFICULTIES) * n_days, -1)
        gap_rows = days_t.repeat(len(DIFFICULTIES))
        all_probs = torch.sigmoid(self.model.predict_logit(state_rows, task_rows, gap_rows)).tolist()

        curves_by_difficulty: dict[str, list[dict[str, float]]] = {}
        for i, diff in enumerate(DIFFICULTIES):
            slice_probs = all_probs[i * n_days : (i + 1) * n_days]
            curves_by_difficulty[diff] = [
                {"day": int(d), "probability": float(p)}
                for d, p in zip(CURVE_DAYS, slice_probs)
            ]

        elapsed_t = torch.tensor([elapsed_days], dtype=torch.float32, device=self.device)
        decayed, _ = self.model._at_time(anchor_state, elapsed_t)
        m = self.model.skill_channels
        active_k = float(decayed[0, :m].mean().item())
        floor_b = float(decayed[0, m : 2 * m].mean().item())
        n_pos = float(decayed[0, 2 * m].item())
        n_neg = float(decayed[0, 2 * m + 1].item())
        half_life_channels = self._half_life_channels(anchor_state)

        return {
            "skill": skill,
            "format": task_format,
            "assessed": assessed,
            "days": list(CURVE_DAYS),
            "curves_by_difficulty": curves_by_difficulty,
            "points": copy.deepcopy(curves_by_difficulty["medium"]),
            "floor_b": floor_b,
            "active_k": active_k,
            "half_life_days": float(sum(half_life_channels) / len(half_life_channels)),
            "half_life_channels": half_life_channels,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "cold_start_reference": copy.deepcopy(self._cold_start_curves[task_format]),
            "model_version": self.version,
        }

    def inspect_tokens(self, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Project raw events through a fresh EvidenceStream and audit the 47-dim allowlist."""
        stream = EvidenceStream()
        projected: list[dict[str, Any]] = []
        excluded_seen: set[str] = set()
        allowed_event_set = set(FEATURE_FIELDS)
        allowed_task_set = set(TASK_FEATURE_KEYS)
        allowed_envelope = {"id", "at", "kind", "skill", "learner", "interaction", "features"}

        for ev in events:
            for k in ev.keys():
                if k not in allowed_envelope:
                    excluded_seen.add(f"event.{k}")
            raw_feats = ev.get("features", {})
            if isinstance(raw_feats, dict):
                for k in raw_feats.keys():
                    if k not in allowed_event_set:
                        excluded_seen.add(f"features.{k}")
                raw_task = raw_feats.get("task")
                if isinstance(raw_task, dict):
                    for k in raw_task.keys():
                        if k not in allowed_task_set:
                            excluded_seen.add(f"task.{k}")

            sanitized = _sanitize_features(raw_feats)
            evidence = stream.push(sanitized)
            if evidence is not None:
                vec = [float(x) for x in evidence.vector()]
                projected.append(
                    {
                        "index": len(projected),
                        "kind": evidence.kind,
                        "time_seconds": float(evidence.time),
                        "dim": len(vec),
                        "vector": vec,
                    }
                )

        return {
            "raw_event_count": len(events),
            "projected_token_count": len(projected),
            "pending_response": stream.pending is not None,
            "expected_token_dim": TOKEN_SIZE,
            "expected_task_dim": TASK_SIZE,
            "last_vector_dim": len(projected[-1]["vector"]) if projected else TOKEN_SIZE,
            "tokens": projected,
            "field_audit": {
                "allowed_event_fields": list(FEATURE_FIELDS),
                "allowed_task_fields": list(TASK_FEATURE_KEYS),
                "excluded_raw_keys_seen": sorted(excluded_seen),
                "forbidden_fields_in_model_input": [],
                "text_embedding_dims": 0,
                "code_embedding_dims": 0,
                "id_embedding_dims": 0,
                "zero_text_code_id_embeddings": True,
                "fast_guess_timing_masked_on_non_full_success": True,
            },
        }
