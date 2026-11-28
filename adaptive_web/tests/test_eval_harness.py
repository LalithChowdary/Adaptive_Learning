"""Unit tests for Tier 2/3 evaluation harness, 4-arm RAG benchmark, snapshots, and EduGuard parity."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import numpy as np

CODE_3_DIR = Path(__file__).resolve().parents[2]
E2E_DIR = Path(__file__).resolve().parent / "e2e"
for p in (str(CODE_3_DIR), str(E2E_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptive_web import catalogue_bridge  # noqa: E402
from adaptive_web.app import SELF_EXPLANATION_PROMPT_BANK, AdaptiveLearningApp  # noqa: E402
from adaptive_web.config import Settings  # noqa: E402
from adaptive_web.eval_harness import (  # noqa: E402
    build_eval_settings,
    build_tier2_splits,
    export_tier3_blind_packs,
    run_tier2_offline,
    run_tier3_live,
    seed_tier3_base_app,
    verify_live_llm_call,
)
from adaptive_web.llm_client import LLMClient  # noqa: E402
from adaptive_web.llm_pedagogy import PedagogyRealizer, _resolve_mastery_tier  # noqa: E402
from adaptive_web.rag import HybridRAGEngine  # noqa: E402
from adaptive_web.store import AdaptiveStore  # noqa: E402
import harness  # noqa: E402


class _StubDenseEncoder:
    """Deterministic unit-test-only DenseEncoder projecting character n-grams to L2-normalized vectors."""

    model_name = "stub-test-dense-v1"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        low = str(text or "").lower()
        for idx, ch in enumerate(low[:256]):
            vec[(ord(ch) + idx * 7) % self.dim] += 1.0
        for tok in low.split():
            h = sum((i + 1) * ord(c) for i, c in enumerate(tok)) % self.dim
            vec[h] += 2.5
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        return vec

    def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._encode_one(t) for t in texts], axis=0)

    def encode_queries(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        return self.encode_documents(texts)


class RagBenchmarkHarnessTests(unittest.TestCase):
    def test_hybrid_tfidf_matches_retrieve_bit_for_bit(self) -> None:
        store = AdaptiveStore(":memory:")
        try:
            rag = HybridRAGEngine(store)
            rag.record_episodic(
                learner="L_parity",
                skill="range_boundaries",
                event_kind="mistake",
                summary="Included stop value in range(1, 5).",
                difficulty="easy",
                created=1700000001.0,
            )

            cases = [
                {
                    "query": "why off by one in range(1, 5)?",
                    "purpose": "tutor_chat",
                    "skill": "range_boundaries",
                    "learner": "L_parity",
                    "difficulty": None,
                    "task_type": None,
                    "source_types": ("curriculum", "episodic"),
                    "include_prerequisites": True,
                    "mastery_tier": "lacking",
                    "top_k": 4,
                    "exclude_task_ids": (),
                },
                {
                    "query": "loop_tracing medium output_prediction stale variable",
                    "purpose": "question_exemplars",
                    "skill": "loop_tracing",
                    "learner": None,
                    "difficulty": "medium",
                    "task_type": "output_prediction",
                    "source_types": ("v22_exemplar",),
                    "include_prerequisites": False,
                    "mastery_tier": None,
                    "top_k": 3,
                    "exclude_task_ids": ("loop_tracing.medium.output_prediction.00",),
                },
                {
                    "query": "range_boundaries easy pitfalls edge cases",
                    "purpose": "question_curriculum",
                    "skill": "range_boundaries",
                    "learner": "L_parity",
                    "difficulty": None,
                    "task_type": None,
                    "source_types": ("curriculum", "episodic"),
                    "include_prerequisites": False,
                    "mastery_tier": "lacking",
                    "top_k": 2,
                    "exclude_task_ids": (),
                },
            ]
            for kw in cases:
                prod = rag.retrieve(**kw, record_audit=False)
                bench = rag.retrieve_benchmark_arm(**kw, arm="hybrid_tfidf", record_audit=False)
                self.assertEqual(bench["retrieved_doc_ids"], [c["doc_id"] for c in prod["chunks"]])
                self.assertEqual(bench["metadata_match_rate"], 1.0)
                self.assertEqual(bench["cross_learner_violation"], 0)
        finally:
            store.close()

    def test_dense_arms_raise_runtime_error_without_encoder(self) -> None:
        store = AdaptiveStore(":memory:")
        try:
            rag = HybridRAGEngine(store)
            for arm in ("dense", "bm25_dense"):
                with self.assertRaises(RuntimeError):
                    rag.retrieve_benchmark_arm("range stop", arm=arm, skill="range_boundaries", dense_encoder=None)
            with self.assertRaises(RuntimeError):
                rag.retrieve_all_arms("range stop", arms=("bm25_alone", "dense"), skill="range_boundaries")
        finally:
            store.close()

    def test_bm25_underfilled_curriculum_retention_and_dense_modes(self) -> None:
        store = AdaptiveStore(":memory:")
        try:
            rag = HybridRAGEngine(store)
            rag.record_episodic(
                learner="L_a",
                skill="range_boundaries",
                event_kind="mistake",
                summary="xyzzysparse_token boundary mistake on range",
                created=1700000010.0,
            )
            rag.record_episodic(
                learner="L_other",
                skill="range_boundaries",
                event_kind="mistake",
                summary="xyzzysparse_token CANARY_LEAK",
                created=1700000020.0,
            )

            # Sparse/unmatched token: bm25_alone underfills without 0.0-score fallback
            sparse_q = "qqq_unmatched_xyzzy_token_98765"
            bm25_zero = rag.retrieve_benchmark_arm(
                sparse_q,
                arm="bm25_alone",
                skill=None,
                source_types=("curriculum",),
                top_k=4,
            )
            self.assertTrue(bm25_zero["bm25_underfilled"])
            self.assertEqual(len(bm25_zero["chunks"]), 0)

            bm25_skill = rag.retrieve_benchmark_arm(
                sparse_q,
                arm="bm25_alone",
                skill="range_boundaries",
                source_types=("curriculum",),
                top_k=6,
            )
            self.assertTrue(bm25_skill["bm25_underfilled"])
            self.assertEqual(len(bm25_skill["chunks"]), 4)

            encoder = _StubDenseEncoder()
            all_res = rag.retrieve_all_arms(
                "range stop exclusive boundary",
                skill="range_boundaries",
                learner="L_a",
                source_types=("curriculum", "episodic"),
                mastery_tier="lacking",
                top_k=4,
                dense_encoder=encoder,
                dense_query_mode="raw_meta",
            )
            self.assertEqual(set(all_res.keys()), {"bm25_alone", "hybrid_tfidf", "dense", "bm25_dense"})
            for arm_name, arm_res in all_res.items():
                self.assertEqual(arm_res["cross_learner_violation"], 0, msg=arm_name)
                self.assertEqual(arm_res["metadata_match_rate"], 1.0, msg=arm_name)
                self.assertGreaterEqual(len(arm_res["unreranked_doc_ids"]), 1)
                for ch in arm_res["chunks"]:
                    self.assertNotIn("CANARY_LEAK", ch["content"])

            dense_rw = rag.retrieve_benchmark_arm(
                "range stop exclusive",
                arm="dense",
                skill="range_boundaries",
                learner="L_a",
                source_types=("curriculum", "episodic"),
                top_k=4,
                dense_encoder=encoder,
                dense_query_mode="rewrite_query",
            )
            self.assertEqual(dense_rw["query_input_mode"], "rewrite_query")
            self.assertEqual(dense_rw["dense_query_input_mode"], "rewrite_query")
        finally:
            store.close()

    def test_malformed_dense_encoder_raises_runtime_error_and_single_space_dynamic(self) -> None:
        class _MalformedDenseEncoder:
            def __init__(self, mode: str) -> None:
                self.mode = mode
                self.model_name = f"malformed-{mode}"

            def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
                if self.mode == "wrong_rows":
                    return np.ones((3, 16), dtype=np.float32)
                if self.mode == "1d":
                    return np.ones((16,), dtype=np.float32)
                if self.mode == "nan":
                    arr = np.ones((len(texts), 16), dtype=np.float32)
                    arr[0, 0] = np.nan
                    return arr
                return np.ones((len(texts), 16), dtype=np.float32)

            def encode_queries(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
                if self.mode == "wrong_query_dim":
                    return np.ones((1, 8), dtype=np.float32)
                if self.mode == "empty_query":
                    return np.zeros((0, 16), dtype=np.float32)
                if self.mode == "query_nan":
                    arr = np.ones((1, 16), dtype=np.float32)
                    arr[0, 0] = np.inf
                    return arr
                return np.ones((len(texts), 16), dtype=np.float32)

        for bad_mode in ("wrong_rows", "1d", "nan", "wrong_query_dim", "empty_query", "query_nan"):
            with self.subTest(bad_mode=bad_mode):
                store = AdaptiveStore(":memory:")
                try:
                    rag = HybridRAGEngine(store)
                    rag.record_episodic(
                        learner="L_bad",
                        skill="range_boundaries",
                        event_kind="mistake",
                        summary="Boundary mistake",
                        difficulty="easy",
                        created=1700000001.0,
                    )
                    bad_enc = _MalformedDenseEncoder(bad_mode)
                    with self.assertRaises(RuntimeError):
                        rag.retrieve_benchmark_arm(
                            "range stop",
                            arm="dense",
                            skill="range_boundaries",
                            learner="L_bad",
                            source_types=("curriculum", "episodic"),
                            dense_encoder=bad_enc,
                        )
                finally:
                    store.close()

        # Dynamic rows with wrong shape/NaN on second call must also raise RuntimeError,
        # and dynamic text formatting must use single spaces (matching KnowledgeChunk.searchable_text).
        seen_dyn_texts: list[str] = []

        class _RecordingDynFailEncoder(_StubDenseEncoder):
            model_name = "recording-dyn-fail"

            def __init__(self) -> None:
                super().__init__(dim=16)
                self.doc_calls = 0

            def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
                self.doc_calls += 1
                if self.doc_calls == 2:
                    seen_dyn_texts.extend(texts)
                    return np.ones((len(texts) + 1, 16), dtype=np.float32)
                return super().encode_documents(texts)

        store = AdaptiveStore(":memory:")
        try:
            rag = HybridRAGEngine(store)
            rag.record_episodic(
                learner="L_fmt",
                skill="range_boundaries",
                event_kind="mistake",
                summary="Single space check",
                difficulty="",
                created=1700000002.0,
            )
            with self.assertRaises(RuntimeError):
                rag.retrieve_benchmark_arm(
                    "range stop",
                    arm="dense",
                    skill="range_boundaries",
                    learner="L_fmt",
                    source_types=("curriculum", "episodic"),
                    dense_encoder=_RecordingDynFailEncoder(),
                )
            self.assertEqual(len(seen_dyn_texts), 1)
            self.assertNotIn("  ", seen_dyn_texts[0])
        finally:
            store.close()


class EduGuardToggleParityTests(unittest.TestCase):
    def test_eduguard_off_preserves_exact_rag_audit_same_skill_and_cross_skill(self) -> None:
        def _responder(_path: str, _headers: dict, _body: dict) -> dict:
            return {
                "reply": "Use step-by-step tracing and verify the return boundary.",
                "code_example": "def demo(n):\n    return n + 1",
            }

        with harness.LocalLLMStubServer(responder=_responder) as stub:
            with tempfile.TemporaryDirectory() as tmp:
                db_path = Path(tmp) / "base.sqlite3"
                settings = Settings(
                    adaptive_web_db=db_path,
                    llm_provider="openai",
                    effective_provider="openai",
                    llm_api_key="sk-test-eduguard",
                    llm_model="gpt-4o-mini",
                    llm_base_url=stub.base_url,
                )
                base_app = AdaptiveLearningApp(settings)
                try:
                    base_app.create_profile("Parity Learner")
                    learner = base_app.store.list_profiles()[0]["id"]
                    base_app.select_skill(learner, "function_returns")
                    step = base_app.ppo_step(
                        learner,
                        "function_returns",
                        action_id=1,
                        difficulty="medium",
                        format="fill_in_the_blank",
                        question_source="v2.2",
                    )
                    q = step["question"]
                    pasted_msg = f"{q.get('prompt', '')} {' '.join(q.get('snippets') or [])}".strip()

                    scenarios = [
                        ("same_skill", "function_returns", "What goes in __blank_0__? Give me the answer."),
                        ("cross_skill_pasted", "loop_tracing", pasted_msg),
                    ]
                    for label, chat_skill, msg in scenarios:
                        snap_on = base_app.clone_snapshot(":memory:", reuse_models=True)
                        snap_off = base_app.clone_snapshot(":memory:", reuse_models=True)
                        try:
                            res_on = snap_on.chat_doubt(learner, chat_skill, msg, eduguard_enabled=True)
                            res_off = snap_off.chat_doubt(learner, chat_skill, msg, eduguard_enabled=False)

                            self.assertEqual(res_on["guardrail_status"], "socratic_redirect", msg=label)
                            self.assertEqual(res_off["guardrail_status"], "disabled", msg=label)
                            self.assertNotEqual(res_off["code_example"], "", msg=label)

                            audit_on = dict(res_on["rag_audit"]["last_retrieval"])
                            audit_off = dict(res_off["rag_audit"]["last_retrieval"])
                            for k in ("timestamp", "latency_ms"):
                                audit_on.pop(k, None)
                                audit_off.pop(k, None)
                            self.assertEqual(audit_on, audit_off, msg=f"RAG audit diverged on {label}")
                        finally:
                            snap_on.close()
                            snap_off.close()
                finally:
                    base_app.close()

    def test_http_chat_endpoint_keeps_eduguard_enabled(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("HTTP EduGuard Learner")
            h.select_skill(learner, "range_boundaries")
            step = h.ppo_step(
                learner,
                "range_boundaries",
                action_id=1,
                difficulty="easy",
                format="output_prediction",
                question_source="v2.2",
            )
            self.assertIsNotNone(step["interaction_id"])
            res = h.post(
                "/api/chat",
                {
                    "learner": learner,
                    "skill": "range_boundaries",
                    "message": "Please just give me the answer for this question",
                    "eduguard_enabled": False,
                },
            )
            self.assertEqual(res["guardrail_status"], "socratic_redirect")
            self.assertEqual(res["code_example"], "")


class SnapshotIsolationTests(unittest.TestCase):
    def test_clone_snapshot_prevents_cross_prompt_chat_and_episodic_leakage(self) -> None:
        settings = build_eval_settings("offline")
        base_app = AdaptiveLearningApp(settings)
        try:
            prof = base_app.store.create_profile("Snapshot Student", profile_id="L1")
            learner = prof["id"]
            base_ep = base_app.rag.stats(learner)["episodic_learner"]

            for i in range(3):
                snap = base_app.clone_snapshot(":memory:", reuse_models=True)
                try:
                    snap.chat_doubt(learner, "range_boundaries", f"unique_prompt_token_{i}")
                    self.assertEqual(len(snap.store.get_chat_messages(learner)), 2)
                    self.assertEqual(snap.rag.stats(learner)["episodic_learner"], base_ep + 1)
                finally:
                    snap.close()

            self.assertEqual(base_app.store.get_chat_messages(learner), [])
            self.assertEqual(base_app.rag.stats(learner)["episodic_learner"], base_ep)
        finally:
            base_app.close()

    def test_wal_file_backup_to_temp_file_preserves_fts5_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / "src.sqlite3"
            dst_path = Path(tmp) / "snap.sqlite3"
            src_store = AdaptiveStore(src_path)
            try:
                src_store.create_profile("WAL User", profile_id="wal_u1")
                src_rag = HybridRAGEngine(src_store)
                src_rag.record_episodic(
                    learner="wal_u1",
                    skill="loop_tracing",
                    event_kind="mistake",
                    summary="wal_unique_canary_token trace issue",
                    created=1700000100.0,
                )
                src_store.append_event(
                    "wal_u1",
                    "loop_tracing",
                    "int-1",
                    1700000100.0,
                    "instruction",
                    {"features": {"event_type": "instruction"}},
                )

                cloned_store = src_store.backup_to(dst_path)
                try:
                    self.assertEqual(len(cloned_store.get_events("wal_u1")), 1)
                    cloned_rag = HybridRAGEngine(cloned_store)
                    hits = cloned_rag.retrieve(
                        "wal_unique_canary_token",
                        skill="loop_tracing",
                        learner="wal_u1",
                        source_types=("episodic",),
                        top_k=2,
                        record_audit=False,
                    )
                    self.assertEqual(len(hits["chunks"]), 1)

                    # Mutate clone; source must remain untouched
                    cloned_store.save_chat_message("wal_u1", "loop_tracing", "user", "clone only message")
                    self.assertEqual(len(src_store.get_chat_messages("wal_u1")), 0)
                finally:
                    cloned_store.close()
            finally:
                src_store.close()


class SelfExplanationAndQuestionSourceTests(unittest.TestCase):
    def test_evaluate_self_explanation_preserves_unresolved_and_null_confidence(self) -> None:
        def _responder(_path: str, _headers: dict, _body: dict) -> dict:
            return {
                "feedback": "Could not tell.",
                "grader_confidence": None,
                "key_concepts_hit": None,
                "missing_concepts": "not_a_list",
            }

        with harness.LocalLLMStubServer(responder=_responder) as stub:
            cfg = Settings(
                llm_provider="openai",
                effective_provider="openai",
                llm_api_key="sk-se-test",
                llm_model="gpt-4o-mini",
                llm_base_url=stub.base_url,
            )
            realizer = PedagogyRealizer(LLMClient(cfg))
            res = realizer.evaluate_self_explanation("range_boundaries", None, "", "idk")
            self.assertEqual(res["grading_status"], "unresolved")
            self.assertIsNone(res["score"])
            self.assertIsNone(res["full_success"])
            self.assertIsInstance(res["grader_confidence"], float)
            self.assertIsInstance(res["key_concepts_hit"], list)
            self.assertIsInstance(res["missing_concepts"], list)
            self.assertEqual(res["source"], "llm")

    def test_generate_question_source_field_all_paths(self) -> None:
        def _responder(_path: str, _headers: dict, _body: dict) -> dict:
            return {
                "prompt": "Determine the exact output of the snippet below.",
                "snippets": ["x = 7\ny = 3\nprint(list(range(y, x + 2, 2)))"],
                "options": [],
                "answer_key": ["[3, 5, 7]"],
                "reference_solution": "[3, 5, 7]",
                "tests": [],
                "criteria": [{"kind": "exact_match", "weight": 1.0}],
                "reference_seconds": 25.0,
            }

        with harness.LocalLLMStubServer(responder=_responder) as stub:
            cfg_live = Settings(
                llm_provider="openai",
                effective_provider="openai",
                llm_api_key="sk-qgen-test",
                llm_model="gpt-4o-mini",
                llm_base_url=stub.base_url,
            )
            live_realizer = PedagogyRealizer(LLMClient(cfg_live))
            q_live = live_realizer.generate_question("range_boundaries", "easy", "output_prediction")
            self.assertEqual(q_live["source"], "llm")

        mock_realizer = PedagogyRealizer(LLMClient(build_eval_settings("offline")))
        q_clone = mock_realizer.generate_question("range_boundaries", "easy", "output_prediction")
        self.assertEqual(q_clone["source"], "v22_clone")
        ok, errs = catalogue_bridge.validate_task_schema(q_clone)
        self.assertTrue(ok, msg=f"Invalid v22_clone schema: {errs}")

        # Exhaust all practice items in (range_boundaries, easy, practice) across formats -> falls back to synth
        all_practice = [
            t["task_id"]
            for t in catalogue_bridge.list_v22_tasks("range_boundaries", "easy", pool="practice")
        ]
        q_synth = mock_realizer.generate_question(
            "range_boundaries",
            "easy",
            "output_prediction",
            pool="practice",
            exclude_task_ids=all_practice,
        )
        self.assertEqual(q_synth["source"], "synth")


class EvalHarnessOfflineAndVerificationTests(unittest.TestCase):
    def test_verify_live_llm_call_rejects_fallbacks_and_failover(self) -> None:
        client = LLMClient(build_eval_settings("offline"))
        client.complete_json("sys", '{"k": 1}', schema_name="test")
        v_mock = verify_live_llm_call({"source": "llm"}, client)
        self.assertFalse(v_mock["verified_live"])

        # Simulate true live telemetry without network
        client._last_telemetry = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "failover_used": False,
            "attempts": [{"provider": "openai", "model": "gpt-4o-mini", "status": "ok"}],
        }
        self.assertTrue(verify_live_llm_call({"source": "llm", "task_id": "llm.1"}, client)["verified_live"])
        self.assertFalse(verify_live_llm_call({"source": "fallback"}, client)["verified_live"])
        self.assertFalse(verify_live_llm_call({"source": "deterministic_double"}, client)["verified_live"])
        self.assertFalse(verify_live_llm_call({"source": "llm", "task_id": "llm.clone.x"}, client)["verified_live"])

        client._last_telemetry["failover_used"] = True
        self.assertFalse(verify_live_llm_call({"source": "llm"}, client)["verified_live"])

    def test_seed_tier3_base_app_mastery_tiers_distinct(self) -> None:
        app = seed_tier3_base_app(build_eval_settings("offline"))
        try:
            for pid, expected_tier in (
                ("t3_unassessed", "unassessed"),
                ("t3_lacking", "lacking"),
                ("t3_fragile", "fragile"),
                ("t3_mastered", "mastered"),
            ):
                dec = app.ppo.decide(app.store, pid, focus_skill="range_boundaries")
                resolved = _resolve_mastery_tier(dec["metrics_by_skill"]["range_boundaries"])["mastery_tier"]
                self.assertEqual(resolved, expected_tier, msg=f"Profile {pid} resolved to {resolved}")
        finally:
            app.close()

    def test_tier2_offline_benchmark_and_stub_dense_encoder(self) -> None:
        splits = build_tier2_splits()
        self.assertEqual(len(splits["dev"]["logical"]), 66)
        self.assertEqual(len(splits["dev"]["tasks"]), 96)
        self.assertEqual(len(splits["test"]["logical"]), 132)
        self.assertEqual(len(splits["test"]["tasks"]), 192)

        # Ensure zero (skill, format, difficulty) overlap between Dev and Test qgen
        dev_cells = {
            (x["skill"], x["task_format"], x["difficulty"])
            for x in splits["dev"]["logical"]
            if x["request_kind"] == "qgen"
        }
        test_cells = {
            (x["skill"], x["task_format"], x["difficulty"])
            for x in splits["test"]["logical"]
            if x["request_kind"] == "qgen"
        }
        self.assertEqual(len(dev_cells), 30)
        self.assertEqual(len(test_cells), 60)
        self.assertEqual(dev_cells & test_cells, set())

        with tempfile.TemporaryDirectory() as tmp:
            diag = run_tier2_offline(tmp)
            self.assertEqual(diag["cross_learner_leakage_probe"]["violations"], 0)
            self.assertEqual(diag["arms"]["bm25_alone"]["status"], "executed_offline")
            self.assertEqual(diag["arms"]["hybrid_tfidf"]["status"], "executed_offline")
            self.assertEqual(diag["arms"]["dense"]["status"], "pending_dense_weights_and_human_adjudication")
            self.assertEqual(diag["arms"]["bm25_dense"]["status"], "pending_dense_weights_and_human_adjudication")
            self.assertEqual(diag["arms"]["bm25_alone"]["test"]["metadata_match_rate"], 1.0)
            self.assertEqual(diag["arms"]["hybrid_tfidf"]["test"]["metadata_match_rate"], 1.0)
            self.assertEqual(diag["arms"]["bm25_alone"]["test"]["cross_learner_violation_total"], 0)
            self.assertEqual(diag["arms"]["hybrid_tfidf"]["test"]["cross_learner_violation_total"], 0)

        with tempfile.TemporaryDirectory() as tmp_dense:
            diag_dense = run_tier2_offline(tmp_dense, dense_encoder=_StubDenseEncoder())
            self.assertEqual(diag_dense["arms"]["dense"]["status"], "executed_offline")
            self.assertEqual(diag_dense["arms"]["bm25_dense"]["status"], "executed_offline")
            self.assertEqual(diag_dense["arms"]["dense"]["test"]["cross_learner_violation_total"], 0)
            self.assertEqual(diag_dense["arms"]["bm25_dense"]["test"]["cross_learner_violation_total"], 0)

    def test_blind_pack_exports_counts_and_zero_fabricated_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            run_tier2_offline(out_dir)
            counts = export_tier3_blind_packs(out_dir)
            self.assertEqual(counts["tier3a_ablation2_prompts"], 216)
            self.assertEqual(counts["tier3b_qgen_requests"], 180)
            self.assertEqual(counts["tier3c_self_explanation_prompts"], 120)

            blind_dir = out_dir / "blind_packs"
            t2_rows = [
                json.loads(ln)
                for ln in (blind_dir / "tier2_relevance_annotation_pack.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            self.assertGreater(len(t2_rows), 0)
            for row in t2_rows:
                self.assertIsNone(row["human_relevance_grade"])
                self.assertIsNone(row["annotator_1_grade"])
                self.assertNotIn("arm", row)

            ab1_rows = [
                json.loads(ln)
                for ln in (blind_dir / "ablation1_downstream_pack.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            self.assertEqual(len(ab1_rows), 132)

            t3a_rows = [
                json.loads(ln)
                for ln in (blind_dir / "tier3a_ablation2_prompts.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            self.assertEqual(len(t3a_rows), 216)
            self.assertEqual(sum(1 for r in t3a_rows if r["suite"] == "adaptation"), 96)
            self.assertEqual(sum(1 for r in t3a_rows if r["suite"] == "eduguard_protected"), 60)
            self.assertEqual(sum(1 for r in t3a_rows if r["suite"] == "eduguard_legit"), 60)

            # Cross-skill pasted open questions must genuinely trigger PedagogyRealizer._is_pasted_active_question
            cross_rows = [r for r in t3a_rows if r["archetype_or_style"] == "pasted_open_question_cross_skill"]
            self.assertEqual(len(cross_rows), 30)
            for cr in cross_rows:
                task_obj = catalogue_bridge.get_v22_task(cr["snapshot_spec"]["active_task_id"])
                self.assertTrue(
                    PedagogyRealizer._is_pasted_active_question(cr["message"], {"raw_task": task_obj}),
                    msg=f"Cross-skill pasted question did not trigger _is_pasted_active_question: {cr['prompt_id']}",
                )

            t3c_rows = [
                json.loads(ln)
                for ln in (blind_dir / "tier3c_self_explanation_prompts.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            self.assertEqual(len(t3c_rows), 120)
            seen_prompts = {r["prompt_text"] for r in t3c_rows}
            expected_prompts = {p for ps in SELF_EXPLANATION_PROMPT_BANK.values() for p in ps}
            self.assertEqual(seen_prompts, expected_prompts)
            for r in t3c_rows:
                self.assertIsNone(r["rubric"])
                self.assertIsNone(r["learner_explanation"])
                self.assertIsNone(r["human_reference"]["human_score_0_1"])
                self.assertIsNone(r["human_reference"]["human_category_0_1_2"])


class _SequenceLLMStubServer(harness.LocalLLMStubServer):
    """LocalLLMStubServer extension allowing per-request (status_code, payload) tuples."""

    def start(self) -> "_SequenceLLMStubServer":
        parent = self

        class SequenceStubHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b""
                try:
                    parsed_body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except Exception:
                    parsed_body = {"_raw": raw_body.decode("utf-8", errors="replace")}
                headers_lower = {k.lower(): v for k, v in self.headers.items()}
                with parent._lock:
                    parent.received_requests.append(
                        {"path": self.path, "headers": headers_lower, "body": parsed_body}
                    )
                    responder = parent.responder
                    default_status = parent.status_code

                status = default_status
                payload_obj: object
                if responder is not None:
                    res = responder(self.path, headers_lower, parsed_body)
                    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], int):
                        status, payload_obj = res
                    else:
                        payload_obj = res
                else:
                    payload_obj = parent._default_payload(json.dumps(parsed_body))

                if status != 200:
                    err_bytes = json.dumps(
                        payload_obj if isinstance(payload_obj, dict) else {"error": {"message": f"Stub HTTP {status}"}}
                    ).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_bytes)))
                    self.end_headers()
                    self.wfile.write(err_bytes)
                    return

                content_text = payload_obj if isinstance(payload_obj, str) else json.dumps(payload_obj)
                envelope = {
                    "id": "chatcmpl_seq_stub",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content_text},
                            "finish_reason": "stop",
                        }
                    ],
                }
                out_bytes = json.dumps(envelope).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out_bytes)))
                self.end_headers()
                self.wfile.write(out_bytes)

        self._server = harness._StubThreadingHTTPServer(("127.0.0.1", 0), SequenceStubHandler)
        self.port = int(self._server.server_port)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"aw-seq-llm-stub-{self.port}",
            daemon=True,
        )
        self._thread.start()
        return self


def _write_minimal_live_packs(
    out_dir: Path,
    *,
    t3a_rows: list[dict] | None = None,
    t3b_rows: list[dict] | None = None,
    t3c_rows: list[dict] | None = None,
) -> None:
    blind_dir = out_dir / "blind_packs"
    blind_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("tier3a_ablation2_prompts.jsonl", t3a_rows or []),
        ("tier3b_qgen_requests.jsonl", t3b_rows or []),
        ("tier3c_self_explanation_prompts.jsonl", t3c_rows or []),
    ):
        text = "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")
        (blind_dir / name).write_text(text, encoding="utf-8")


def _write_stub_env(tmp_dir: Path, base_url: str) -> Path:
    env_file = tmp_dir / "stub_live.env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai",
                "OPENAI_API_KEY=sk-live-stub-test",
                "LLM_MODEL=gpt-4o-mini",
                f"LLM_BASE_URL={base_url}",
            ]
        ),
        encoding="utf-8",
    )
    return env_file


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


class EvalHarnessLiveStubTests(unittest.TestCase):
    def _run_with_clean_llm_env(self, fn):
        saved = {
            k: os.environ.pop(k)
            for k in list(os.environ.keys())
            if k.startswith(("LLM_", "GROQ_", "GEMINI_", "OPENAI_", "ANTHROPIC_", "OLLAMA_"))
        }
        try:
            return fn()
        finally:
            os.environ.update(saved)

    def test_run_tier3_live_separates_det_and_llm_confidence(self) -> None:
        def _responder(_path: str, _headers: dict, _body: dict) -> dict:
            return {
                "feedback": "Good explanation of exclusive stop.",
                "grader_confidence": 0.42,
                "key_concepts_hit": ["stop", "excluded"],
                "missing_concepts": [],
                "next_tip": "Try negative steps next.",
            }

        with _SequenceLLMStubServer(responder=_responder) as stub:
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp) / "eval_out"
                env_file = _write_stub_env(Path(tmp), stub.base_url)
                _write_minimal_live_packs(
                    out_dir,
                    t3c_rows=[
                        {
                            "item_id": "t3c_001",
                            "prompt_id": "se_range_boundaries_0",
                            "skill": "range_boundaries",
                            "target_quality_band": "full_conceptual",
                            "prompt_text": SELF_EXPLANATION_PROMPT_BANK["range_boundaries"][0],
                            "learner_explanation": (
                                "In range(start, stop), start is included and stop is strictly excluded, "
                                "so iteration stops before reaching stop because Python uses half-open intervals."
                            ),
                            "human_reference": {"human_score_0_1": 1.0, "human_category_0_1_2": 2},
                        }
                    ],
                )
                summary = self._run_with_clean_llm_env(
                    lambda: run_tier3_live(out_dir, env_path=env_file, max_retries=2, retry_base_delay=0.0)
                )
                self.assertEqual(summary["tier3c_runs"], 1)
                rows = _read_jsonl(out_dir / "tier3c_eval.jsonl")
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["source"], "llm")
                self.assertIsNone(row["api_error"])
                self.assertEqual(row["llm_confidence"], 0.42)
                self.assertIsInstance(row["det_confidence"], float)
                self.assertNotEqual(row["det_confidence"], row["llm_confidence"])

    def test_run_tier3_live_transient_503_retry_recovery_and_clean_snapshot(self) -> None:
        counts = {"chat": 0, "qgen": 0, "se": 0}
        chat_payloads: list[dict] = []

        def _responder(_path: str, _headers: dict, body: dict):
            user_txt = ""
            for m in body.get("messages", []):
                if m.get("role") == "user":
                    user_txt = str(m.get("content", ""))
            parsed_user = json.loads(user_txt) if user_txt.startswith("{") else {}
            intent = parsed_user.get("intent", "")
            if "student_question" in parsed_user:
                counts["chat"] += 1
                chat_payloads.append(parsed_user)
                if counts["chat"] == 1:
                    return 503, {"error": "temporary overload"}
                return {
                    "reply": "Stop is strictly excluded in range(start, stop).",
                    "code_example": "print(list(range(1, 4)))",
                }
            if intent == "generate_question":
                counts["qgen"] += 1
                if counts["qgen"] == 1:
                    return 503, {"error": "temporary overload"}
                return {
                    "prompt": "Determine the exact printed output of this range snippet.",
                    "snippets": ["nums = list(range(3, 9, 2))\nprint(nums)"],
                    "options": [],
                    "answer_key": ["[3, 5, 7]"],
                    "reference_solution": "[3, 5, 7]",
                    "tests": [],
                    "criteria": [{"kind": "exact_match", "weight": 1.0}],
                    "reference_seconds": 25.0,
                }
            if intent == "evaluate_self_explanation":
                counts["se"] += 1
                if counts["se"] == 1:
                    return 503, {"error": "temporary overload"}
                return {
                    "feedback": "Solid explanation after transient retry.",
                    "grader_confidence": 0.61,
                    "key_concepts_hit": ["stop", "excluded"],
                    "missing_concepts": [],
                    "next_tip": "Proceed to harder ranges.",
                }
            # Course material generation during seed_tier3_base_app()
            return stub._default_payload(json.dumps(body))

        with _SequenceLLMStubServer(responder=_responder) as stub:
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp) / "eval_out"
                env_file = _write_stub_env(Path(tmp), stub.base_url)
                _write_minimal_live_packs(
                    out_dir,
                    t3a_rows=[
                        {
                            "prompt_id": "t3a_retry_1",
                            "blind_item_id": "b_t3a_1",
                            "suite": "adaptation",
                            "ablation2_paired": False,
                            "skill": "range_boundaries",
                            "chat_skill": "range_boundaries",
                            "mastery_tier": "unassessed",
                            "format": "output_prediction",
                            "message": "Why does range(1, 5) stop at 4?",
                            "snapshot_spec": {"profile_key": "t3_unassessed", "active_task_id": None},
                        }
                    ],
                    t3b_rows=[
                        {
                            "request_id": "t3b_retry_1",
                            "blind_item_id": "b_t3b_1",
                            "skill": "range_boundaries",
                            "format": "output_prediction",
                            "difficulty": "easy",
                            "pool": "practice",
                            "recent_mistakes": ["treats stop as inclusive"],
                            "snapshot_profile_key": "t3_lacking",
                            "exclude_task_ids": ["range_boundaries.easy.output_prediction.00"],
                        }
                    ],
                    t3c_rows=[
                        {
                            "item_id": "t3c_retry_1",
                            "prompt_id": "se_range_boundaries_0",
                            "skill": "range_boundaries",
                            "target_quality_band": "full_conceptual",
                            "prompt_text": SELF_EXPLANATION_PROMPT_BANK["range_boundaries"][0],
                            "learner_explanation": "Stop is excluded so range(1, 5) yields 1, 2, 3, 4.",
                            "human_reference": {"human_score_0_1": 0.9, "human_category_0_1_2": 2},
                        }
                    ],
                )
                summary = self._run_with_clean_llm_env(
                    lambda: run_tier3_live(out_dir, env_path=env_file, max_retries=2, retry_base_delay=0.0)
                )
                self.assertEqual(summary, {"tier3a_runs": 1, "tier3b_runs": 1, "tier3c_runs": 1})
                self.assertEqual(counts, {"chat": 2, "qgen": 2, "se": 2})

                # Second chat attempt must use a fresh clean snapshot (no fallback history or episodic leak from attempt 1)
                self.assertEqual(len(chat_payloads), 2)
                self.assertEqual(chat_payloads[1]["history"], [])
                for rag_ctx in chat_payloads[1]["rag_context"]:
                    self.assertNotIn("Why does range(1, 5) stop at 4?", rag_ctx.get("excerpt", ""))

                t3a_row = _read_jsonl(out_dir / "tier3a_eval.jsonl")[0]
                t3b_row = _read_jsonl(out_dir / "tier3b_eval.jsonl")[0]
                t3c_row = _read_jsonl(out_dir / "tier3c_eval.jsonl")[0]
                self.assertEqual(t3a_row["source"], "llm")
                self.assertIsNone(t3a_row["api_error"])
                self.assertEqual(t3a_row["attempts"], 2)
                self.assertEqual(t3b_row["source"], "llm")
                self.assertIsNone(t3b_row["api_error"])
                self.assertTrue(t3b_row["post_repair_valid"])
                self.assertEqual(t3b_row["attempts"], 2)
                self.assertEqual(t3c_row["source"], "llm")
                self.assertIsNone(t3c_row["api_error"])
                self.assertEqual(t3c_row["llm_confidence"], 0.61)
                self.assertEqual(t3c_row["attempts"], 2)

    def test_run_tier3_live_tier3b_schema_fallback_and_ast_similarity_metrics(self) -> None:
        def _responder(_path: str, _headers: dict, body: dict) -> dict:
            user_txt = ""
            for m in body.get("messages", []):
                if m.get("role") == "user":
                    user_txt = str(m.get("content", ""))
            parsed_user = json.loads(user_txt) if user_txt.startswith("{") else {}
            if parsed_user.get("intent") != "generate_question" and parsed_user.get("intent") != "repair_question_schema":
                return stub._default_payload(json.dumps(body))
            if parsed_user.get("task_format") == "code_completion":
                return {
                    "prompt": "Complete solve(x) to return the sum of odd integers in range(1, x, 2) plus 4.",
                    "snippets": ["def solve(x):\n    acc = 4\n    for v in range(1, x, 2):\n        acc += v\n    return ___"],
                    "options": [],
                    "answer_key": [],
                    "reference_solution": "def solve(x):\n    acc = 4\n    for v in range(1, x, 2):\n        acc += v\n    return acc",
                    "tests": [
                        {"input": 6, "expected": 13},
                        {"input": 8, "expected": 20},
                    ],
                    "criteria": [
                        {"kind": "behavior_test", "weight": 0.5},
                        {"kind": "behavior_test", "weight": 0.5},
                    ],
                    "reference_seconds": 35.0,
                }
            # Invalid output_prediction schema on both initial and repair attempts -> v22_clone fallback
            return {"prompt": "Broken question missing snippets and answer_key"}

        with _SequenceLLMStubServer(responder=_responder) as stub:
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp) / "eval_out"
                env_file = _write_stub_env(Path(tmp), stub.base_url)
                _write_minimal_live_packs(
                    out_dir,
                    t3b_rows=[
                        {
                            "request_id": "t3b_code_ok",
                            "blind_item_id": "b_code_ok",
                            "skill": "range_boundaries",
                            "format": "code_completion",
                            "difficulty": "easy",
                            "pool": "practice",
                            "recent_mistakes": ["treats stop as inclusive"],
                            "snapshot_profile_key": "t3_lacking",
                            "exclude_task_ids": ["range_boundaries.easy.code_completion.00"],
                        },
                        {
                            "request_id": "t3b_schema_fail",
                            "blind_item_id": "b_schema_fail",
                            "skill": "range_boundaries",
                            "format": "output_prediction",
                            "difficulty": "easy",
                            "pool": "practice",
                            "recent_mistakes": ["treats stop as inclusive"],
                            "snapshot_profile_key": "t3_lacking",
                            "exclude_task_ids": ["range_boundaries.easy.output_prediction.00"],
                        },
                    ],
                )
                summary = self._run_with_clean_llm_env(
                    lambda: run_tier3_live(out_dir, env_path=env_file, max_retries=2, retry_base_delay=0.0)
                )
                self.assertEqual(summary["tier3b_runs"], 2)
                rows = _read_jsonl(out_dir / "tier3b_eval.jsonl")
                self.assertEqual(len(rows), 2)

                r_ok, r_fail = rows
                self.assertEqual(r_ok["source"], "llm")
                self.assertIsNone(r_ok["api_error"])
                self.assertIsNone(r_ok["rejection_reason"])
                self.assertIs(r_ok["ast_tests_passed"], True)
                self.assertIsInstance(r_ok["max_exemplar_similarity"], float)
                self.assertGreaterEqual(r_ok["max_exemplar_similarity"], 0.0)
                self.assertLessEqual(r_ok["max_exemplar_similarity"], 1.0)

                self.assertTrue(r_fail["fell_back_to_bank"])
                self.assertEqual(r_fail["source"], "v22_clone")
                self.assertIsNone(r_fail["api_error"])
                self.assertTrue(str(r_fail["rejection_reason"]).startswith("schema_validation_failed:bank_fallback:"))
                self.assertIsNone(r_fail["ast_tests_passed"])
                self.assertIsNone(r_fail["max_exemplar_similarity"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
