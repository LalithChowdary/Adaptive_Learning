"""Tier 4: Realistic multi-step learner & research presenter E2E workloads (8 tests)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

E2E_DIR = Path(__file__).resolve().parent
if str(E2E_DIR) not in sys.path:
    sys.path.insert(0, str(E2E_DIR))

import harness  # noqa: E402


class TestTier4Scenarios(unittest.TestCase):
    def test_scenario_01_cold_start_beginner_full_learn_code_remember_loop(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Scenario1 Beginner")
            state0 = h.get_state(learner)
            for sk in harness.PPO_SKILLS_ORDER:
                self.assertFalse(harness.extract_skill_summary(state0, sk)["assessed"])

            sel = h.select_skill(learner, "range_boundaries")
            self.assertIn("learning_path", sel)
            self.assertIn("course_material", sel)

            # Learn it: Action 3 (explain_concept) -> forces follow-up question
            lesson = h.ppo_step(learner, "range_boundaries", action_id=3)
            ppo_lesson = harness.extract_ppo(lesson)
            self.assertTrue(ppo_lesson["awaiting_question"])
            valid_after = [i for i, ok in enumerate(harness.extract_valid_mask(ppo_lesson)) if ok]
            self.assertEqual(valid_after, [1, 2])

            # Code/Practice it: Action 1 -> pass unaided
            q_step = h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            qiid = harness.extract_interaction_id(q_step)
            v_ans, _ = harness.solve_v22_task(harness.extract_question(q_step))
            h.submit(learner, qiid, v_ans, active_seconds=8.0)
            h.finish(learner, qiid)

            pre_se = harness.extract_skill_summary(h.get_state(learner, "range_boundaries"), "range_boundaries")
            ppo_after_q = harness.extract_ppo(h.get_state(learner, "range_boundaries"))
            self.assertTrue(harness.extract_valid_mask(ppo_after_q)[5])

            # Remember it: Action 5 (request_self_explanation) -> consolidate floor_b & p_30d
            se = h.ppo_step(learner, "range_boundaries", action_id=5)
            seiid = harness.extract_interaction_id(se)
            h.submit(
                learner,
                seiid,
                "Python range(start, stop) includes start and stops strictly before stop.",
                self_explanation="Python range(start, stop) includes start and stops strictly before stop.",
                active_seconds=12.0,
            )
            h.finish(learner, seiid)

            post_se = harness.extract_skill_summary(h.get_state(learner, "range_boundaries"), "range_boundaries")
            self.assertEqual(post_se["observed_unaided_passes"], 1)
            self.assertGreater(post_se["floor_b"], pre_se["floor_b"])
            self.assertGreater(post_se["p_30d"], pre_se["p_30d"])
            curve = h.get_curve(learner, "range_boundaries")
            med_pts = [pt["probability"] for pt in curve["curves_by_difficulty"]["medium"]]
            for a, b in zip(med_pts, med_pts[1:]):
                self.assertLessEqual(b, a + 1e-7)

    def test_scenario_02_struggling_learner_3hints_assisted_pass_and_unaided_verification(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Scenario2 Struggler")
            h.select_skill(learner, "loop_tracing")

            q1 = h.ppo_step(learner, "loop_tracing", action_id=1, difficulty="easy", format="single_choice")
            iid1 = harness.extract_interaction_id(q1)
            v_ans, bad_ans = harness.solve_v22_task(harness.extract_question(q1))
            h.submit(learner, iid1, bad_ans, active_seconds=10.0)
            kinds: list[str] = []
            for _ in range(3):
                h_res = h.hint(learner, iid1)
                h_obj = h_res.get("hint") or h_res
                kinds.append(h_obj.get("kind", h_obj.get("hint_kind")))
            self.assertEqual(kinds, ["conceptual", "targeted", "scaffold"])
            h.retry(learner, iid1)
            h.submit(learner, iid1, v_ans, active_seconds=15.0)
            h.finish(learner, iid1)

            ppo1 = harness.extract_ppo(h.get_state(learner, "loop_tracing"))
            obs1 = harness.extract_obs_vector(ppo1)
            self.assertTrue(ppo1["unverified_assisted"])
            self.assertGreaterEqual(obs1[11], 0.75)
            summ1 = harness.extract_skill_summary(h.get_state(learner, "loop_tracing"), "loop_tracing")
            self.assertEqual(summ1["observed_unaided_passes"], 0)

            # Action 1 triggers unaided assessment check
            q2 = h.ppo_step(learner, "loop_tracing", action_id=1, format="single_choice")
            self.assertEqual(q2["mode"], "assessment")
            iid2 = harness.extract_interaction_id(q2)
            v2, _ = harness.solve_v22_task(harness.extract_question(q2))
            h.submit(learner, iid2, v2, active_seconds=9.0)
            h.finish(learner, iid2)

            ppo2 = harness.extract_ppo(h.get_state(learner, "loop_tracing"))
            self.assertFalse(ppo2["unverified_assisted"])
            summ2 = harness.extract_skill_summary(h.get_state(learner, "loop_tracing"), "loop_tracing")
            self.assertEqual(summ2["observed_unaided_passes"], 1)

    def test_scenario_03_prerequisite_weakness_and_decay_detour_action_6(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Scenario3 PrereqDetour")
            h.select_skill(learner, "function_returns")
            ppo0 = harness.extract_ppo(h.get_state(learner, "function_returns"))
            self.assertTrue(harness.extract_valid_mask(ppo0)[6])

            detour = h.ppo_step(learner, "function_returns", action_id=6)
            self.assertEqual(detour["ppo_action"], "review_prerequisite")
            target = detour["target_skill"]
            self.assertIn(target, harness.PREREQUISITES["function_returns"])
            self.assertIsNone(detour.get("primed_mode"))
            iid = harness.extract_interaction_id(detour)
            v_ans, _ = harness.solve_v22_task(harness.extract_question(detour))
            h.submit(learner, iid, v_ans, active_seconds=7.0)
            h.finish(learner, iid)

            state = h.get_state(learner, "function_returns")
            self.assertTrue(harness.extract_skill_summary(state, target)["assessed"])
            self.assertFalse(harness.extract_skill_summary(state, "function_returns")["assessed"])

    def test_scenario_04_hybrid_v22_and_fresh_llm_5format_mastery_with_ast_code(self) -> None:
        def smart_responder(path: str, headers: dict, body: dict) -> dict:
            blob = json.dumps(body).lower()
            if "output_prediction" in blob:
                return {
                    "task_id": "llm.range_boundaries.easy.output_prediction.sc04",
                    "skill_id": "range_boundaries",
                    "task_type": "output_prediction",
                    "difficulty": "easy",
                    "prompt": "Predict printed output.",
                    "snippets": ["print(list(range(2, 7, 2)))"],
                    "options": [],
                    "answer_key": ["[2, 4, 6]"],
                    "reference_solution": None,
                    "tests": [],
                    "criteria": [{"kind": "exact_match", "weight": 1.0}],
                    "reference_seconds": 35.0,
                    "hints": ["Start at 2.", "Step by 2.", "Stop before 7."],
                }
            if "code_completion" in blob:
                return {
                    "task_id": "llm.range_boundaries.easy.code_completion.sc04",
                    "skill_id": "range_boundaries",
                    "task_type": "code_completion",
                    "difficulty": "easy",
                    "prompt": "Return list(range(0, x, 2)).",
                    "snippets": ["def solve(x):\n    pass"],
                    "options": [],
                    "answer_key": None,
                    "reference_solution": "def solve(x):\n    return list(range(0, x, 2))",
                    "tests": [
                        {"input": 5, "expected": [0, 2, 4]},
                        {"input": 6, "expected": [0, 2, 4]},
                    ],
                    "criteria": [
                        {"kind": "behavior_test", "weight": 0.5},
                        {"kind": "behavior_test", "weight": 0.5},
                    ],
                    "reference_seconds": 60.0,
                    "hints": ["Start 0.", "Step 2.", "Use list(range(0, x, 2))."],
                }
            return harness.LocalLLMStubServer()._default_payload(blob)

        with harness.LocalLLMStubServer(responder=smart_responder) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-sc04",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Scenario4 Hybrid5")
                h.select_skill(learner, "range_boundaries")

                # 3 v2.2 items
                for fmt in ("single_choice", "multiple_select", "fill_in_the_blank"):
                    st = h.ppo_step(
                        learner, "range_boundaries", action_id=1, difficulty="easy", format=fmt, question_source="v2.2"
                    )
                    iid = harness.extract_interaction_id(st)
                    v_ans, _ = harness.solve_v22_task(harness.extract_question(st))
                    sub = h.submit(learner, iid, v_ans)
                    res = sub["result"] if "result" in sub else sub["results"][-1]
                    self.assertTrue(res["full_success"])
                    h.finish(learner, iid)

                # 2 LLM items (output_prediction + code_completion)
                for fmt, ans in (
                    ("output_prediction", ["[2, 4, 6]"]),
                    ("code_completion", "def solve(x):\n    res = list(range(0, x, 2))\n    return res"),
                ):
                    st = h.ppo_step(
                        learner, "range_boundaries", action_id=1, difficulty="easy", format=fmt, question_source="llm"
                    )
                    q = harness.extract_question(st)
                    self.assertTrue(q["task_id"].startswith("llm."))
                    iid = harness.extract_interaction_id(st)
                    sub = h.submit(learner, iid, ans)
                    res = sub["result"] if "result" in sub else sub["results"][-1]
                    self.assertEqual(res["grading_status"], "resolved")
                    self.assertTrue(res["full_success"])
                    h.finish(learner, iid)

                state = h.get_state(learner, "range_boundaries")
                summ = harness.extract_skill_summary(state, "range_boundaries")
                self.assertEqual(summ["observed_attempts"], 5)
                self.assertEqual(summ["observed_unaided_passes"], 5)
                self.assertEqual(state["token_audit"]["token_dim"], 47)
                self.assertFalse(state["token_audit"]["uses_text_or_code_embeddings"])

    def test_scenario_05_assessment_integrity_help_before_submit_and_immutable_first_grade(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Scenario5 Integrity")
            h.select_skill(learner, "range_boundaries")

            # Part A: hint before submit in assessment mode -> assessment_incomplete + review mode
            a1 = h.ppo_step(
                learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice", mode="assessment"
            )
            iid1 = harness.extract_interaction_id(a1)
            v1, _ = harness.solve_v22_task(harness.extract_question(a1))
            hint_res = h.hint(learner, iid1)
            self.assertTrue(hint_res.get("assessment_incomplete") or hint_res.get("mode") == "review")
            sub1 = h.submit(learner, iid1, v1)
            res1 = sub1["result"] if "result" in sub1 else sub1["results"][-1]
            self.assertFalse(res1.get("eligible", False))
            h.finish(learner, iid1)

            # Part B: wrong first submission in assessment mode -> immutable under retry/pass
            a2 = h.ppo_step(
                learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice", mode="assessment"
            )
            iid2 = harness.extract_interaction_id(a2)
            v2, bad2 = harness.solve_v22_task(harness.extract_question(a2))
            s_wrong = h.submit(learner, iid2, bad2, active_seconds=4.0)
            r0 = s_wrong["results"][0] if "results" in s_wrong else s_wrong["result"]
            self.assertTrue(r0["eligible"])
            self.assertFalse(r0["full_success"])
            h.hint(learner, iid2)
            h.retry(learner, iid2)
            s_retry = h.submit(learner, iid2, v2, active_seconds=6.0)
            r1 = s_retry["results"][-1] if "results" in s_retry else s_retry["result"]
            self.assertFalse(r1["eligible"])
            h.finish(learner, iid2)

            exp = h.get_export(learner)
            first_match = [i for i in exp["interactions"] if i["id"] == iid2][0]
            self.assertFalse(first_match["results"][0]["full_success"])
            self.assertTrue(first_match["results"][0]["eligible"])

    def test_scenario_06_server_restart_persistence_and_multi_learner_isolation(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="aw_sc06_")
        db_path = Path(tmp.name) / "sc06.sqlite3"
        try:
            with harness.AdaptiveWebHarness(db_path=db_path, tmp_dir=tmp, keep_tmp=True) as h1:
                alice = h1.create_profile("Alice")
                bob = h1.create_profile("Bob")
                h1.select_skill(alice, "range_boundaries")
                h1.select_skill(bob, "list_indexing")
                s_a = h1.ppo_step(alice, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
                ia = harness.extract_interaction_id(s_a)
                va, _ = harness.solve_v22_task(harness.extract_question(s_a))
                h1.submit(alice, ia, va)
                h1.finish(alice, ia)

                s_b = h1.ppo_step(bob, "list_indexing", action_id=1, difficulty="easy", format="single_choice")
                ib = harness.extract_interaction_id(s_b)
                _, bad_b = harness.solve_v22_task(harness.extract_question(s_b))
                h1.submit(bob, ib, bad_b)
                h1.finish(bob, ib)

                snap_a = h1.get_state(alice, "range_boundaries")
                snap_b = h1.get_state(bob, "list_indexing")
                curve_a = h1.get_curve(alice, "range_boundaries")

            with harness.AdaptiveWebHarness(db_path=db_path, tmp_dir=tmp, keep_tmp=True) as h2:
                re_a = h2.get_state(alice, "range_boundaries")
                re_b = h2.get_state(bob, "list_indexing")
                self.assertEqual(
                    harness.extract_obs_vector(harness.extract_ppo(re_a)),
                    harness.extract_obs_vector(harness.extract_ppo(snap_a)),
                )
                self.assertEqual(
                    harness.extract_obs_vector(harness.extract_ppo(re_b)),
                    harness.extract_obs_vector(harness.extract_ppo(snap_b)),
                )
                self.assertEqual(h2.get_curve(alice, "range_boundaries")["curves_by_difficulty"], curve_a["curves_by_difficulty"])
                self.assertFalse(harness.extract_skill_summary(re_a, "list_indexing")["assessed"])
                self.assertFalse(harness.extract_skill_summary(re_b, "range_boundaries")["assessed"])
        finally:
            tmp.cleanup()

    def test_scenario_07_presenter_showcase_glass_box_audit_manual_override_and_palette(self) -> None:
        if not harness.PROPOSAL_PATH.exists():
            raise AssertionError(
                "Real 3/adaptive_web/PROPOSAL_PPO_LLM_ARCHITECTURE.md not yet available; cannot fake E2E target"
            )
        proposal = harness.PROPOSAL_PATH.read_text(encoding="utf-8")
        self.assertIn("Glass-Box", proposal)

        with harness.AdaptiveWebHarness() as h:
            _, _, css_b = h.get("/style.css")
            css_text = css_b.decode("utf-8")
            tokens = harness.normalize_hex_tokens(css_text)
            for hex_c in harness.PALETTE_HEX:
                self.assertIn(hex_c, tokens)
            for pat in harness.FORBIDDEN_GRADIENT_PATTERNS:
                self.assertNotIn(pat, css_text.lower())

            learner = h.create_profile("Scenario7 Presenter")
            h.select_skill(learner, "range_boundaries")
            t1 = h.ppo_step(learner, "range_boundaries", action_id=3)
            self.assertEqual(t1.get("policy_source", "delayed_retention_ppo"), "delayed_retention_ppo")
            t2 = h.ppo_step(learner, "range_boundaries", action_id=1, policy_source="learner_manual")
            h.finish(learner, harness.extract_interaction_id(t2))

            state = h.get_state(learner, "range_boundaries")
            ppo = harness.extract_ppo(state)
            self.assertEqual(len(harness.extract_obs_vector(ppo)), 36)
            self.assertEqual(len(ppo["per_seed_probs"]), 3)
            self.assertEqual(state["token_audit"]["token_dim"], 47)

            exp = h.get_export(learner)
            sources = {t["policy_source"] for t in exp["tutor_turns"]}
            self.assertEqual(sources, {"delayed_retention_ppo", "learner_manual"})

    def test_scenario_08_concurrent_learners_with_slow_loopback_llm_stub(self) -> None:
        with harness.LocalLLMStubServer(delay_seconds=0.45) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-sc08",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                s1 = h.create_profile("Slow 1")
                s2 = h.create_profile("Slow 2")
                f1 = h.create_profile("Fast 1")
                f2 = h.create_profile("Fast 2")

                q1 = h.ppo_step(f1, "range_boundaries", action_id=1, difficulty="easy", format="single_choice", question_source="v2.2")
                iid1 = harness.extract_interaction_id(q1)
                v1, _ = harness.solve_v22_task(harness.extract_question(q1))

                errors: list[Exception] = []
                fast_latencies: list[float] = []

                def slow_worker(lrn: str, sk: str) -> None:
                    try:
                        h.select_skill(lrn, sk)
                    except Exception as exc:
                        errors.append(exc)

                def fast_worker_read() -> None:
                    try:
                        t0 = time.monotonic()
                        h.get_state(f2, "range_boundaries")
                        h.get_curve(f2, "range_boundaries")
                        fast_latencies.append(time.monotonic() - t0)
                    except Exception as exc:
                        errors.append(exc)

                def fast_worker_submit() -> None:
                    try:
                        t0 = time.monotonic()
                        h.submit(f1, iid1, v1)
                        h.finish(f1, iid1)
                        fast_latencies.append(time.monotonic() - t0)
                    except Exception as exc:
                        errors.append(exc)

                ts = [
                    threading.Thread(target=slow_worker, args=(s1, "range_boundaries")),
                    threading.Thread(target=slow_worker, args=(s2, "loop_tracing")),
                ]
                for t in ts:
                    t.start()
                time.sleep(0.08)
                tf = [
                    threading.Thread(target=fast_worker_read),
                    threading.Thread(target=fast_worker_submit),
                ]
                for t in tf:
                    t.start()
                for t in ts + tf:
                    t.join(timeout=8.0)

                self.assertEqual(errors, [])
                self.assertEqual(len(fast_latencies), 2)
                self.assertTrue(all(lat < 0.25 for lat in fast_latencies), msg=f"Blocked fast calls: {fast_latencies}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
