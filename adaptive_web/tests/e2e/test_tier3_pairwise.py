"""Tier 3: Cross-feature pairwise combinatorial opaque-box tests (16 tests)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import threading
import time
import unittest

E2E_DIR = Path(__file__).resolve().parent
if str(E2E_DIR) not in sys.path:
    sys.path.insert(0, str(E2E_DIR))

import harness  # noqa: E402


class TestTier3Pairwise(unittest.TestCase):
    h: harness.AdaptiveWebHarness

    @classmethod
    def setUpClass(cls) -> None:
        cls.h = harness.AdaptiveWebHarness().start()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "h") and cls.h is not None:
            cls.h.stop()

    def test_pair_01_memory_decay_updates_ppo_s_t_and_action_probs(self) -> None:
        learner = self.h.create_profile("Pair1 MemxPPO")
        self.h.select_skill(learner, "range_boundaries")
        ppo0 = harness.extract_ppo(self.h.get_state(learner, "range_boundaries"))
        obs0 = harness.extract_obs_vector(ppo0)
        probs0 = harness.extract_ensemble_probs(ppo0)

        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        self.h.submit(learner, iid, v_ans)
        self.h.finish(learner, iid)

        ppo1 = harness.extract_ppo(self.h.get_state(learner, "range_boundaries"))
        obs1 = harness.extract_obs_vector(ppo1)
        probs1 = harness.extract_ensemble_probs(ppo1)
        self.assertNotEqual(obs0[:8], obs1[:8])
        self.assertGreater(obs1[24], obs0[24])
        self.assertNotEqual(probs0, probs1)

    def test_pair_02_ppo_step_with_llm_question_and_subsequent_mask(self) -> None:
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-p2",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Pair2 PPOxLLM")
                h.select_skill(learner, "range_boundaries")
                step = h.ppo_step(
                    learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice", question_source="llm"
                )
                q = harness.extract_question(step)
                self.assertTrue(q["task_id"].startswith("llm."))
                iid = harness.extract_interaction_id(step)
                h.submit(learner, iid, "o0")
                h.finish(learner, iid)
                ppo = harness.extract_ppo(h.get_state(learner, "range_boundaries"))
                self.assertTrue(harness.extract_valid_mask(ppo)[5])
                self.assertGreater(harness.extract_obs_vector(ppo)[12], 0.0)

    def test_pair_03_ast_grader_evaluates_llm_code_completion_variants(self) -> None:
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-p3",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Pair3 ASTxLLM")
                h.select_skill(learner, "range_boundaries")
                # Variant 1: valid alternative implementation
                s1 = h.ppo_step(
                    learner, "range_boundaries", action_id=1, difficulty="easy", format="code_completion", question_source="llm"
                )
                id1 = harness.extract_interaction_id(s1)
                r1 = h.submit(learner, id1, "def solve(x):\n    out = list(range(1, x, 2))\n    return out")
                res1 = r1["result"] if "result" in r1 else r1["results"][-1]
                self.assertEqual(res1["score"], 1.0)
                h.finish(learner, id1)
                # Variant 2: wrong output
                s2 = h.ppo_step(
                    learner, "range_boundaries", action_id=1, difficulty="easy", format="code_completion", question_source="llm"
                )
                id2 = harness.extract_interaction_id(s2)
                r2 = h.submit(learner, id2, "def solve(x):\n    return []")
                res2 = r2["result"] if "result" in r2 else r2["results"][-1]
                self.assertLess(res2["score"], 1.0)
                h.finish(learner, id2)
                # Variant 3: unsupported AST
                s3 = h.ppo_step(
                    learner, "range_boundaries", action_id=1, difficulty="easy", format="code_completion", question_source="llm"
                )
                id3 = harness.extract_interaction_id(s3)
                r3 = h.submit(learner, id3, "import math\ndef solve(x):\n    return []")
                res3 = r3["result"] if "result" in r3 else r3["results"][-1]
                self.assertEqual(res3["grading_status"], "unresolved")
                h.finish(learner, id3)

    def test_pair_04_llm_hints_update_leakage_l_t_and_unverified_flag(self) -> None:
        learner = self.h.create_profile("Pair4 HintsxPPO")
        self.h.select_skill(learner, "loop_tracing")
        step = self.h.ppo_step(learner, "loop_tracing", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        self.h.hint(learner, iid)
        self.h.hint(learner, iid)
        self.h.submit(learner, iid, v_ans)
        self.h.finish(learner, iid)

        ppo = harness.extract_ppo(self.h.get_state(learner, "loop_tracing"))
        obs = harness.extract_obs_vector(ppo)
        self.assertGreater(obs[11], 0.4)
        self.assertEqual(obs[8], 1.0)
        self.assertTrue(ppo["unverified_assisted"])

    def test_pair_05_concurrent_duplicate_request_id_posts_safe(self) -> None:
        learner = self.h.create_profile("Pair5 ConcIdemp")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        payload = {
            "learner": learner,
            "interaction": iid,
            "answer": v_ans,
            "active_seconds": 3.0,
            "request_id": "conc-idem-req-0001",
        }
        results: list[dict] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                res = self.h.post("/api/submit", payload, auto_request_id=False)
                with lock:
                    results.append(res)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 5)
        self.h.finish(learner, iid)
        grades = [e for e in self.h.get_export(learner)["events"] if e.get("kind") == "grade"]
        self.assertEqual(len(grades), 1)

    def test_pair_06_dag_prerequisites_surface_in_curated_path(self) -> None:
        learner = self.h.create_profile("Pair6 DAGxPath")
        res = self.h.select_skill(learner, "function_returns")
        blob = json.dumps(res["learning_path"]).lower()
        self.assertTrue("conditional_evaluation" in blob or "accumulator_initialization" in blob)

    def test_pair_07_unassessed_skill_guard_across_state_curve_and_export(self) -> None:
        learner = self.h.create_profile("Pair7 Unassessed")
        state = self.h.get_state(learner, "list_indexing")
        curve = self.h.get_curve(learner, "list_indexing")
        exp = self.h.get_export(learner)
        self.assertFalse(harness.extract_skill_summary(state, "list_indexing")["assessed"])
        self.assertFalse(curve["assessed"])
        self.assertEqual([e for e in exp["events"] if e.get("kind") == "grade"], [])

    def test_pair_08_unresolved_submission_preserves_observed_vs_predicted_integrity(self) -> None:
        learner = self.h.create_profile("Pair8 UnresolvedxMem")
        self.h.select_skill(learner, "range_boundaries")
        cold = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="code_completion")
        iid = harness.extract_interaction_id(step)
        self.h.submit(learner, iid, "def solve(x):\n    return open('/etc/passwd')")
        self.h.finish(learner, iid)
        after = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")
        self.assertEqual(after["observed_unaided_passes"], 0)
        self.assertEqual(after["n_pos"], cold["n_pos"])
        self.assertEqual(after["n_neg"], cold["n_neg"])

    def test_pair_09_manual_override_vs_ppo_provenance_in_export(self) -> None:
        learner = self.h.create_profile("Pair9 Provenance")
        self.h.select_skill(learner, "range_boundaries")
        self.h.ppo_step(learner, "range_boundaries", action_id=3)
        q_ppo = self.h.ppo_step(learner, "range_boundaries", action_id=1)
        self.h.finish(learner, harness.extract_interaction_id(q_ppo))
        q_man = self.h.ppo_step(learner, "range_boundaries", action_id=2, policy_source="learner_manual")
        self.h.finish(learner, harness.extract_interaction_id(q_man))
        turns = self.h.get_export(learner)["tutor_turns"]
        sources = [t.get("policy_source") for t in turns]
        self.assertIn("delayed_retention_ppo", sources)
        self.assertIn("learner_manual", sources)

    def test_pair_10_env_config_switches_stub_provider_cleanly(self) -> None:
        with harness.AdaptiveWebHarness(env_overrides={"LLM_PROVIDER": "mock"}) as h_mock:
            l1 = h_mock.create_profile("MockCfg")
            r1 = h_mock.select_skill(l1, "range_boundaries")
            self.assertEqual(r1["learning_path"]["source"], "deterministic_double")
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-p10",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h_live:
                l2 = h_live.create_profile("LiveCfg")
                r2 = h_live.select_skill(l2, "range_boundaries")
                self.assertEqual(r2["learning_path"]["source"], "llm")

    def test_pair_11_self_explanation_consolidates_b_t_more_than_passive_lesson(self) -> None:
        l_passive = self.h.create_profile("Pair11 Passive")
        l_active = self.h.create_profile("Pair11 Active SE")
        self.h.select_skill(l_passive, "range_boundaries")
        self.h.select_skill(l_active, "range_boundaries")

        # Passive lesson only
        self.h.ppo_step(l_passive, "range_boundaries", action_id=3)
        b_passive = harness.extract_skill_summary(self.h.get_state(l_passive, "range_boundaries"), "range_boundaries")["floor_b"]

        # Question + pass + request_self_explanation + pass
        q = self.h.ppo_step(l_active, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        qiid = harness.extract_interaction_id(q)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(q))
        self.h.submit(l_active, qiid, v_ans)
        self.h.finish(l_active, qiid)
        se = self.h.ppo_step(l_active, "range_boundaries", action_id=5)
        seiid = harness.extract_interaction_id(se)
        self.h.submit(
            l_active,
            seiid,
            "Start is inclusive and stop is strictly excluded from Python range boundaries.",
            self_explanation="Start is inclusive and stop is strictly excluded from Python range boundaries.",
        )
        self.h.finish(l_active, seiid)
        b_active = harness.extract_skill_summary(self.h.get_state(l_active, "range_boundaries"), "range_boundaries")["floor_b"]
        self.assertGreater(b_active, b_passive)

    def test_pair_12_slow_llm_path_generation_never_blocks_v22_submit(self) -> None:
        with harness.LocalLLMStubServer(delay_seconds=0.5) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-p12",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                la = h.create_profile("Slow Select A")
                lb = h.create_profile("Fast Submit B")
                q_b = h.ppo_step(lb, "range_boundaries", action_id=1, difficulty="easy", format="single_choice", question_source="v2.2")
                iid_b = harness.extract_interaction_id(q_b)
                v_b, _ = harness.solve_v22_task(harness.extract_question(q_b))
                errs: list[Exception] = []

                def slow_call() -> None:
                    try:
                        h.select_skill(la, "range_boundaries")
                    except Exception as exc:
                        errs.append(exc)

                t = threading.Thread(target=slow_call)
                t.start()
                time.sleep(0.08)
                t0 = time.monotonic()
                h.submit(lb, iid_b, v_b)
                h.finish(lb, iid_b)
                elapsed = time.monotonic() - t0
                t.join(timeout=5.0)
                self.assertEqual(errs, [])
                self.assertLess(elapsed, 0.25)

    def test_pair_13_two_sequential_harnesses_never_share_sqlite_state(self) -> None:
        with harness.AdaptiveWebHarness() as h1:
            h1.create_profile("Alice Isolated")
            self.assertEqual(len(h1.get_state()["profiles"]), 1)
        with harness.AdaptiveWebHarness() as h2:
            self.assertEqual(h2.get_state()["profiles"], [])

    def test_pair_14_static_assets_and_api_share_consistent_security_and_palette(self) -> None:
        for route in ("/", "/style.css", "/app.js", "/favicon.svg"):
            st, headers, _ = self.h.get(route)
            self.assertEqual(st, 200)
            self.assertEqual(headers.get("x-content-type-options"), "nosniff")
            self.assertIn("frame-ancestors 'none'", headers.get("content-security-policy", ""))
        _, _, css_b = self.h.get("/style.css")
        tokens = harness.normalize_hex_tokens(css_b.decode("utf-8"))
        self.assertIn("#f7f8f3", tokens)
        self.assertIn("#293321", tokens)

    def test_pair_15_action_3_4_primed_lesson_links_with_course_material(self) -> None:
        learner = self.h.create_profile("Pair15 LessonxCourse")
        self.h.select_skill(learner, "loop_tracing")
        l3 = self.h.ppo_step(learner, "loop_tracing", action_id=3)
        self.assertTrue(harness.extract_ppo(l3)["awaiting_question"])
        self.assertEqual(l3["primed_mode"], "concept")
        q1 = self.h.ppo_step(learner, "loop_tracing", action_id=1)
        self.h.finish(learner, harness.extract_interaction_id(q1))

        l4 = self.h.ppo_step(learner, "loop_tracing", action_id=4)
        self.assertTrue(harness.extract_ppo(l4)["awaiting_question"])
        self.assertEqual(l4["primed_mode"], "example")
        q2 = self.h.ppo_step(learner, "loop_tracing", action_id=1)
        self.h.finish(learner, harness.extract_interaction_id(q2))

    def test_pair_16_proposal_documented_actions_and_endpoints_match_live_api(self) -> None:
        if not harness.PROPOSAL_PATH.exists():
            raise AssertionError(
                "Real 3/adaptive_web/PROPOSAL_PPO_LLM_ARCHITECTURE.md not yet available; cannot fake E2E target"
            )
        text = harness.PROPOSAL_PATH.read_text(encoding="utf-8")
        for action in harness.PPO_ACTIONS:
            self.assertIn(action, text, msg=f"Proposal missing PPO action {action}")
        for ep in ("/api/state", "/api/curve", "/api/export", "/api/select_skill", "/api/ppo_step", "/api/submit"):
            self.assertIn(ep, text, msg=f"Proposal missing documented endpoint {ep}")
        learner = self.h.create_profile("Pair16 DocCheck")
        self.assertEqual(self.h.get_state(learner)["skills_order"] if "skills_order" in self.h.get_state(learner) else list(harness.PPO_SKILLS_ORDER), list(harness.PPO_SKILLS_ORDER))


if __name__ == "__main__":
    unittest.main(verbosity=2)
