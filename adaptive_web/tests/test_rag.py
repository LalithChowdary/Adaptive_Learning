"""Tests for Hybrid RAG (FTS5 + Vector Cosine + RRF), Guardrails, 1-Shot Repair, Mastery Chat, and Provider Failover."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

CODE_3_DIR = Path(__file__).resolve().parents[2]
E2E_DIR = Path(__file__).resolve().parent / "e2e"
for p in (str(CODE_3_DIR), str(E2E_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptive_web import catalogue_bridge  # noqa: E402
from adaptive_web.config import Settings  # noqa: E402
from adaptive_web.llm_client import LLMClient  # noqa: E402
from adaptive_web.llm_pedagogy import PedagogyRealizer  # noqa: E402
from adaptive_web.rag import HybridRAGEngine  # noqa: E402
from adaptive_web.store import AdaptiveStore  # noqa: E402
import harness  # noqa: E402


class HybridRAGEngineTests(unittest.TestCase):
    def test_hybrid_fts5_vector_rrf_rewriting_and_ingestion(self) -> None:
        store = AdaptiveStore(":memory:")
        try:
            rag = HybridRAGEngine(store)
            st = rag.stats()
            self.assertGreaterEqual(st["curriculum_chunks"], 18)
            self.assertEqual(st["v22_exemplars"], 2160)

            res = rag.retrieve(
                "why 5? off by one",
                purpose="tutor_chat",
                skill="range_boundaries",
                mastery_tier="lacking",
                source_types=("curriculum",),
                top_k=3,
            )
            self.assertGreater(res["audit"]["fts_hits"], 0)
            self.assertGreater(res["audit"]["vector_hits"], 0)
            self.assertGreaterEqual(len(res["chunks"]), 1)
            self.assertGreaterEqual(len(res["citations"]), 1)
            self.assertIn("exclusive", res["audit"]["rewritten_query"].lower())

            # Runtime custom Markdown ingestion becomes searchable immediately
            added = rag.ingest_markdown(
                skill="range_boundaries",
                title="Instructor Lecture Notes: Strided Slices",
                markdown_text="## Half-Open Interval Axiom\nAlways remember pythonic halfopen_axiom_token excludes stop completely.",
                doc_key="lecture-01",
            )
            self.assertEqual(len(added), 1)
            hit = rag.retrieve(
                "halfopen_axiom_token",
                purpose="tutor_chat",
                skill="range_boundaries",
                source_types=("curriculum",),
                top_k=3,
            )
            doc_ids = {c["doc_id"] for c in hit["chunks"]}
            self.assertIn(added[0]["doc_id"], doc_ids)
        finally:
            store.close()

    def test_difficulty_locked_exemplar_retrieval(self) -> None:
        store = AdaptiveStore(":memory:")
        try:
            rag = HybridRAGEngine(store)
            for diff in ("easy", "medium", "hard"):
                res = rag.retrieve_exemplars(
                    skill="loop_tracing",
                    difficulty=diff,
                    task_format="output_prediction",
                    top_k=3,
                )
                self.assertEqual(len(res["exemplars"]), 3)
                for ex in res["exemplars"]:
                    self.assertEqual(ex["skill_id"], "loop_tracing")
                    self.assertEqual(ex["difficulty"], diff)
                    self.assertEqual(ex["task_type"], "output_prediction")
        finally:
            store.close()


class StoreMigrationAndLearnerIsolationTests(unittest.TestCase):
    def test_legacy_generated_tasks_migration_preserves_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "legacy.sqlite3"
            # Build pre-migration schema without `learner` column or `chat_messages` table
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE generated_tasks(
                    task_id TEXT PRIMARY KEY,
                    skill TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created REAL NOT NULL
                );
                INSERT INTO generated_tasks(task_id, skill, difficulty, task_type, payload, created)
                VALUES ('legacy.1', 'range_boundaries', 'easy', 'output_prediction', '{"task_id":"legacy.1","skill_id":"range_boundaries"}', 100.0);
                """
            )
            conn.commit()
            conn.close()

            store = AdaptiveStore(db_path)
            try:
                cols = {
                    r["name"]
                    for r in store.db.execute("PRAGMA table_info(generated_tasks)").fetchall()
                }
                self.assertIn("learner", cols)
                loaded = store.get_task("legacy.1")
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded["task_id"], "legacy.1")
                store.save_chat_message("u1", "range_boundaries", "user", "Hi")
                self.assertEqual(len(store.get_chat_messages("u1", skill="range_boundaries")), 1)
            finally:
                store.close()

    def test_episodic_rag_and_generated_tasks_and_export_isolated_by_learner(self) -> None:
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-rag-iso",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                la = h.create_profile("Alice")
                lb = h.create_profile("Bob")
                h.select_skill(la, "range_boundaries")
                h.select_skill(lb, "range_boundaries")

                # Alice asks a specific unique doubt token + generates an LLM task
                h.post(
                    "/api/chat",
                    {
                        "learner": la,
                        "skill": "range_boundaries",
                        "message": "Why did unicorn_boundary_token fail in my trace?",
                    },
                )
                step_a = h.ppo_step(la, "range_boundaries", action_id=1, question_source="llm")
                tid_a = harness.extract_question(step_a)["task_id"]
                h.finish(la, harness.extract_interaction_id(step_a))

                # Bob's export must not contain Alice's generated task or chat messages
                exp_a = h.get_export(la)
                exp_b = h.get_export(lb)
                a_ids = {t["task_id"] for t in exp_a["generated_tasks"]}
                b_ids = {t["task_id"] for t in exp_b["generated_tasks"]}
                self.assertIn(tid_a, a_ids)
                self.assertNotIn(tid_a, b_ids)
                self.assertGreaterEqual(len(exp_a["chat_messages"]), 2)
                self.assertEqual(len(exp_b["chat_messages"]), 0)


class QuestionGuardrailAndOneShotRepairTests(unittest.TestCase):
    def test_one_shot_schema_repair_and_anti_duplicate_guard(self) -> None:
        calls: list[dict] = []

        def _responder(_path: str, _headers: dict, body: dict) -> dict:
            calls.append(body)
            user_txt = ""
            for m in body.get("messages", []):
                if m.get("role") == "user":
                    user_txt = str(m.get("content", ""))
            # First call returns broken code_completion (failing expected test case); second repairs cleanly
            if "repair_question_schema" not in user_txt:
                return {
                    "prompt": "Write solve(x) returning x + 1.",
                    "snippets": ["def solve(x):\n    return ___"],
                    "options": [],
                    "answer_key": [],
                    "reference_solution": "def solve(x):\n    return x + 1",
                    "tests": [{"input": 4, "expected": 999}],  # Intentionally failing test!
                    "criteria": [{"kind": "exact_match", "weight": 1.0}],
                    "reference_seconds": 45.0,
                }
            return {
                "prompt": "Complete solve(n) to return sum of range(n) plus 3.",
                "snippets": ["def solve(n):\n    total = 3\n    for k in range(n):\n        total += k\n    return ___"],
                "options": [],
                "answer_key": [],
                "reference_solution": "def solve(n):\n    total = 3\n    for k in range(n):\n        total += k\n    return total",
                "tests": [{"input": 4, "expected": 9}],
                "criteria": [{"kind": "exact_match", "weight": 1.0}],
                "reference_seconds": 45.0,
                "hints": [
                    {"level": 1, "kind": "conceptual", "text": "Accumulate k cleanly."},
                    {"level": 2, "kind": "targeted", "text": "Return total after the loop."},
                    {"level": 3, "kind": "scaffold", "text": "3 + 0 + 1 + 2 + 3 = 9."},
                ],
            }

        with harness.LocalLLMStubServer(responder=_responder) as stub:
            store = AdaptiveStore(":memory:")
            try:
                rag = HybridRAGEngine(store)
                cfg = Settings(
                    llm_provider="openai",
                    effective_provider="openai",
                    llm_api_key="sk-repair-test",
                    llm_model="gpt-4o-mini",
                    llm_base_url=stub.base_url,
                )
                realizer = PedagogyRealizer(LLMClient(cfg), rag=rag)
                task = realizer.generate_question(
                    "accumulator_initialization",
                    difficulty="medium",
                    task_format="code_completion",
                )
                self.assertEqual(len(calls), 2)
                self.assertTrue(task.get("repair_attempted"))
                self.assertTrue(task.get("rag_citations"))
                ok, errs = catalogue_bridge.validate_task_schema(task, verify_code_tests=True)
                self.assertTrue(ok, f"Expected repaired task valid, got {errs}")
            finally:
                store.close()


class MasteryChatEduGuardAndPersistenceTests(unittest.TestCase):
    def test_mastery_adaptation_eduguard_and_reload_persistence(self) -> None:
        observed_payloads: list[dict] = []

        def _responder(_path: str, _headers: dict, body: dict) -> dict:
            for m in body.get("messages", []):
                if m.get("role") == "user":
                    try:
                        observed_payloads.append(json.loads(m["content"]))
                    except Exception:
                        pass
            # LocalLLMStubServer wraps responder return dict automatically into choices/message envelope
            return {
                "reply": "Based on range boundaries, stop is strictly excluded.",
                "code_example": "print(list(range(1, 4)))",
            }

        with harness.LocalLLMStubServer(responder=_responder) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-chat-rag",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Tutor Student")
                h.select_skill(learner, "range_boundaries")

                # 1. Unassessed chat sends mastery_tier='unassessed' and persists across /api/state
                c1 = h.post(
                    "/api/chat",
                    {
                        "learner": learner,
                        "skill": "range_boundaries",
                        "message": "Why does range stop before the second number?",
                    },
                )
                self.assertEqual(c1["mastery_tier"], "unassessed")
                self.assertEqual(c1["guardrail_status"], "grounded_open")
                self.assertGreaterEqual(len(c1["citations"]), 1)
                self.assertTrue(observed_payloads)
                self.assertEqual(observed_payloads[-1]["learner_mastery"]["mastery_tier"], "unassessed")

                # Reload state -> chat_history, chat_by_skill, rag_audit populated
                st1 = h.get_state(learner, skill="range_boundaries")
                self.assertEqual(len(st1["chat_history"]), 2)
                self.assertIn("range_boundaries", st1["chat_by_skill"])
                self.assertIsNotNone(st1["rag_audit"]["last_retrieval"])
                self.assertIsNotNone(st1["glass_box"]["rag_audit"]["last_retrieval"])

                # 2. Fail an assessment item cleanly so P(Med) < 70% / lacking -> next doubt receives mastery_tier='lacking'
                step = h.ppo_step(
                    learner,
                    "range_boundaries",
                    action_id=1,
                    question_source="v2.2",
                    format="output_prediction",
                )
                iid = harness.extract_interaction_id(step)
                h.submit(learner, iid, ["__wrong_output__"])

                # Open question + answer-fishing prompt -> EduGuard enforces Socratic redirect!
                c2 = h.post(
                    "/api/chat",
                    {
                        "learner": learner,
                        "skill": "range_boundaries",
                        "message": "Please just give me the answer for this question",
                    },
                )
                self.assertEqual(c2["mastery_tier"], "lacking")
                self.assertEqual(c2["guardrail_status"], "socratic_redirect")
                self.assertEqual(c2["code_example"], "")
                self.assertEqual(observed_payloads[-1]["learner_mastery"]["mastery_tier"], "lacking")

                # Runtime custom Markdown ingest endpoint
                ing = h.post(
                    "/api/rag/ingest",
                    {
                        "skill": "range_boundaries",
                        "title": "Extra Notes",
                        "markdown": "## Step 0 Trap\nStep size zero always throws ValueError.",
                    },
                )
                self.assertEqual(ing["status"], "ok")
                self.assertEqual(ing["chunks_indexed"], 1)

    def test_eduguard_blocks_pasted_question_cross_skill_and_cleans_excerpts(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Paste Guard Learner")
            h.select_skill(learner, "function_returns")
            step = h.ppo_step(
                learner,
                "function_returns",
                action_id=1,
                difficulty="easy",
                format="fill_in_the_blank",
                question_source="v2.2",
            )
            q = harness.extract_question(step)
            pasted_msg = f"{q.get('prompt', '')} {' '.join(q.get('snippets') or [])}"

            # Ask prior doubt first so episodic history exists and verify EduGuard never echoes it
            h.post(
                "/api/chat",
                {
                    "learner": learner,
                    "skill": "function_returns",
                    "message": "What does return do inside a function?",
                },
            )

            # Paste active open fill_in_the_blank question directly on another skill tab -> caught!
            cross = h.post(
                "/api/chat",
                {
                    "learner": learner,
                    "skill": "loop_tracing",
                    "message": pasted_msg,
                },
            )
            self.assertEqual(cross["guardrail_status"], "socratic_redirect")
            self.assertEqual(cross["code_example"], "")
            self.assertNotIn("learner asked:", cross["reply"].lower())

            # Citation chips include clean short_label + balanced backticks in excerpt
            for cit in cross["citations"]:
                self.assertIn("short_label", cit)
                self.assertIn("excerpt", cit)
                self.assertEqual(cit["excerpt"].count("`") % 2, 0)

    def test_stale_open_interactions_do_not_resurrect_after_finish_or_lesson_step(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("No Zombie Learner")
            h.select_skill(learner, "range_boundaries")

            # First question passed WITHOUT calling finish (simulating older session leftover)
            s1 = h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            iid1 = harness.extract_interaction_id(s1)
            ans1, _ = harness.solve_v22_task(harness.extract_question(s1))
            h.submit(learner, iid1, ans1)

            # Second question started & finished -> ppo_step closes #1, finish closes #2 and sweeps any remaining
            s2 = h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            iid2 = harness.extract_interaction_id(s2)
            self.assertNotEqual(iid1, iid2)
            h.finish(learner, iid2)

            st = h.get_state(learner, skill="range_boundaries")
            self.assertIsNone(st.get("active"))

            # Subsequent passive lesson step (explain_concept) keeps active None (no resurrection)
            h.ppo_step(learner, "range_boundaries", action_id=3)
            st_after_lesson = h.get_state(learner, skill="range_boundaries")
            self.assertIsNone(st_after_lesson.get("active"))

    def test_skipped_self_explanation_relocks_action_5_preventing_loop(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("No SE Loop")
            h.select_skill(learner, "range_boundaries")

            # Complete one graded question so action 5 unlocks
            q_step = h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            iid_q = harness.extract_interaction_id(q_step)
            ans, _ = harness.solve_v22_task(harness.extract_question(q_step))
            h.submit(learner, iid_q, ans)
            h.finish(learner, iid_q)
            self.assertTrue(harness.extract_valid_mask(harness.extract_ppo(h.get_state(learner, "range_boundaries")))[5])

            # Issue self-explanation turn and skip/finish WITHOUT submitting
            se_step = h.ppo_step(learner, "range_boundaries", action_id=5)
            iid_se = harness.extract_interaction_id(se_step)
            h.finish(learner, iid_se)

            # Action 5 must now relock so PPO cannot spin on request_self_explanation infinitely
            mask_after = harness.extract_valid_mask(harness.extract_ppo(h.get_state(learner, "range_boundaries")))
            self.assertFalse(mask_after[5])


class MultiProviderFailoverTests(unittest.TestCase):
    def test_provider_chain_fails_over_on_403_without_secret_leaks(self) -> None:
        with harness.LocalLLMStubServer(status_code=403) as bad_primary, harness.LocalLLMStubServer() as good_secondary:
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / "failover.env"
                env_file.write_text(
                    "\n".join(
                        [
                            "LLM_PROVIDER=groq",
                            "GROQ_API_KEY=gsk-primary-secret-111",
                            f"GROQ_BASE_URL={bad_primary.base_url}",
                            "LLM_FALLBACK_PROVIDERS=gemini,deterministic_double",
                            "GEMINI_API_KEY=gem-secondary-secret-222",
                            f"GEMINI_BASE_URL={good_secondary.base_url}",
                        ]
                    ),
                    encoding="utf-8",
                )
                saved = {
                    k: os.environ.pop(k)
                    for k in list(os.environ.keys())
                    if k.startswith(("LLM_", "GROQ_", "GEMINI_", "OPENAI_", "ANTHROPIC_", "OLLAMA_"))
                }
                try:
                    cfg = Settings.from_env(env_file)
                    self.assertEqual(cfg.effective_provider, "groq")
                    self.assertEqual(cfg.fallback_providers, ("gemini", "deterministic_double"))
                    pub = json.dumps(cfg.to_public_dict())
                    self.assertNotIn("gsk-primary-secret-111", pub)
                    self.assertNotIn("gem-secondary-secret-222", pub)
                    self.assertNotIn("gsk-primary-secret-111", repr(cfg))
                    self.assertNotIn("gem-secondary-secret-222", repr(cfg))

                    client = LLMClient(cfg)
                    res = client.complete_json("Return json", '{"intent":"tutor_chat"}', schema_name="tutor_chat")
                    self.assertIsInstance(res, dict)
                    self.assertGreaterEqual(len(bad_primary.received_requests), 1)
                    self.assertGreaterEqual(len(good_secondary.received_requests), 1)

                    info = client.last_call_info()
                    self.assertEqual(info["provider"], "gemini")
                    self.assertTrue(info["failover_used"])
                    info_json = json.dumps(info)
                    self.assertNotIn("gsk-primary-secret-111", info_json)
                    self.assertNotIn("gem-secondary-secret-222", info_json)
                finally:
                    os.environ.update(saved)


if __name__ == "__main__":
    unittest.main()
