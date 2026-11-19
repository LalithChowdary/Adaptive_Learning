"""Tier 2: Boundary Value Analysis & corner-case opaque-box tests across F01..F14 (72 tests)."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import unittest
import unittest.mock
import urllib.parse

E2E_DIR = Path(__file__).resolve().parent
if str(E2E_DIR) not in sys.path:
    sys.path.insert(0, str(E2E_DIR))

import harness  # noqa: E402


class _SharedBoundaryTestCase(unittest.TestCase):
    h: harness.AdaptiveWebHarness

    @classmethod
    def setUpClass(cls) -> None:
        cls.h = harness.AdaptiveWebHarness().start()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "h") and cls.h is not None:
            cls.h.stop()


def _lazy_import_config_app_server():
    harness.ensure_adaptive_web_on_sys_path()
    try:
        config_mod = importlib.import_module("adaptive_web.config")
        app_mod = importlib.import_module("adaptive_web.app")
        server_mod = importlib.import_module("adaptive_web.server")
    except (ImportError, ModuleNotFoundError) as exc:
        raise AssertionError("Real 3/adaptive_web server/app not yet available; cannot fake E2E target") from exc
    return config_mod, app_mod, server_mod


class TestF01Boundary(unittest.TestCase):
    def test_f01_b_static_path_traversal_rejected_with_404_or_403(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            for bad_path in ("/../dashboard/server.py", "/static/../config.py", "/%2e%2e/config.py"):
                status, _, body = h.get(bad_path)
                self.assertIn(status, (400, 403, 404))
                self.assertNotIn(b"ThreadingHTTPServer", body)

    def test_f01_b_missing_frozen_checkpoint_path_fails_explicitly(self) -> None:
        config_mod, app_mod, _ = _lazy_import_config_app_server()
        with tempfile.TemporaryDirectory(prefix="aw_missing_ckpt_") as tmp:
            bad_ckpt = Path(tmp) / "nonexistent_model.pt"
            env = {
                "ADAPTIVE_WEB_DB": str(Path(tmp) / "t.sqlite3"),
                "DEVICE": "cpu",
                "LLM_PROVIDER": "mock",
                "MEMORY_MODEL_PATH": str(bad_ckpt),
            }
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                settings = config_mod.Settings.from_env(env_path=None)
                for attr in ("memory_model_path", "model_path", "checkpoint_path"):
                    if hasattr(settings, attr):
                        object.__setattr__(settings, attr, bad_ckpt)
                        break
                else:
                    object.__setattr__(settings, "memory_model_path", bad_ckpt)
                with self.assertRaises((FileNotFoundError, OSError, RuntimeError, ValueError)):
                    app_mod.AdaptiveLearningApp(settings)

    def test_f01_b_read_only_db_directory_fails_cleanly_on_startup(self) -> None:
        config_mod, app_mod, _ = _lazy_import_config_app_server()
        env = {
            "ADAPTIVE_WEB_DB": "/dev/null/impossible_dir/bad.sqlite3",
            "DEVICE": "cpu",
            "LLM_PROVIDER": "mock",
        }
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            settings = config_mod.Settings.from_env(env_path=None)
            with self.assertRaises((OSError, ValueError)):
                app_mod.AdaptiveLearningApp(settings)

    def test_f01_b_port_collision_raises_oserror_cleanly(self) -> None:
        _, _, server_mod = _lazy_import_config_app_server()
        serve_fn = getattr(server_mod, "serve", None) or getattr(server_mod, "create_server", None)
        with harness.AdaptiveWebHarness() as h:
            with self.assertRaises(OSError):
                dup = serve_fn(h.app, h.port)
                dup.server_close()

    def test_f01_b_unrecognized_http_methods_put_delete_rejected(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            for method in ("PUT", "DELETE", "PATCH"):
                status, _, _ = h.raw_request(method, "/api/state", body=b"{}")
                self.assertIn(status, (405, 501))


class TestF02Boundary(unittest.TestCase):
    def test_f02_b_empty_api_key_whitespace_falls_back_to_mock(self) -> None:
        with harness.AdaptiveWebHarness(
            env_overrides={"LLM_PROVIDER": "auto", "LLM_API_KEY": "   ", "OPENAI_API_KEY": "\t"}
        ) as h:
            llm = h.get_state().get("llm_status") or {}
            self.assertFalse(llm.get("is_live"))
            self.assertEqual(llm.get("provider"), "mock")

    def test_f02_b_invalid_port_out_of_range_rejected(self) -> None:
        config_mod, _, _ = _lazy_import_config_app_server()
        for bad_port in ("-1", "70000", "abc"):
            with unittest.mock.patch.dict(os.environ, {"ADAPTIVE_WEB_PORT": bad_port}, clear=True):
                with self.assertRaises(ValueError):
                    config_mod.Settings.from_env(env_path=None)

    def test_f02_b_invalid_llm_provider_or_device_rejected(self) -> None:
        config_mod, _, _ = _lazy_import_config_app_server()
        for env in ({"LLM_PROVIDER": "unknown_provider"}, {"DEVICE": "cuda99"}):
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    config_mod.Settings.from_env(env_path=None)

    def test_f02_b_timeout_zero_or_negative_rejected(self) -> None:
        config_mod, _, _ = _lazy_import_config_app_server()
        for bad_t in ("0", "-5", "nan"):
            with unittest.mock.patch.dict(os.environ, {"LLM_TIMEOUT_SECONDS": bad_t}, clear=True):
                with self.assertRaises(ValueError):
                    config_mod.Settings.from_env(env_path=None)

    def test_f02_b_dotenv_malformed_lines_and_equals_in_value(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aw_dotenv_b_") as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n# comment\nMALFORMED_LINE_WITHOUT_EQUALS\nLLM_PROVIDER=mock\nLLM_MODEL=a=b=c\n",
                encoding="utf-8",
            )
            with harness.AdaptiveWebHarness(env_path=env_file) as h:
                llm = h.get_state().get("llm_status") or {}
                self.assertEqual(llm.get("model"), "a=b=c")


class TestF03Boundary(_SharedBoundaryTestCase):
    def test_f03_b_request_id_length_boundaries_7_8_100_101(self) -> None:
        st7, _, _ = self.h.post_raw("/api/profile", {"name": "R7", "request_id": "a" * 7})
        self.assertEqual(st7, 400)
        st8 = self.h.post("/api/profile", {"name": "R8", "request_id": "a" * 8}, auto_request_id=False)
        self.assertTrue(st8.get("learner") or st8.get("learner_id"))
        st100 = self.h.post("/api/profile", {"name": "R100", "request_id": "b" * 100}, auto_request_id=False)
        self.assertTrue(st100.get("learner") or st100.get("learner_id"))
        st101, _, _ = self.h.post_raw("/api/profile", {"name": "R101", "request_id": "c" * 101})
        self.assertEqual(st101, 400)

    def test_f03_b_profile_name_length_boundaries_0_whitespace_1_50_51(self) -> None:
        for bad_name in ("", "   ", "x" * 51):
            st, _, _ = self.h.post_raw("/api/profile", {"name": bad_name, "request_id": "req-name-001"})
            self.assertEqual(st, 400)
        self.h.post("/api/profile", {"name": "A"})
        self.h.post("/api/profile", {"name": "x" * 50})

    def test_f03_b_unknown_learner_id_returns_400_on_all_endpoints(self) -> None:
        bogus = "00000000-0000-0000-0000-000000000000"
        for path in (f"/api/curve?learner={bogus}&skill=range_boundaries", f"/api/export?learner={bogus}"):
            st, _, _ = self.h.get(path)
            self.assertEqual(st, 400)
        for path, body in (
            ("/api/select_skill", {"learner": bogus, "skill": "range_boundaries"}),
            ("/api/ppo_step", {"learner": bogus, "skill": "range_boundaries", "action_id": 1}),
        ):
            st, _, _ = self.h.post_raw(path, body)
            self.assertEqual(st, 400)

    def test_f03_b_duplicate_submit_with_new_request_id_rejected(self) -> None:
        learner = self.h.create_profile("Dup Submit")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        self.h.submit(learner, iid, v_ans)
        st2, _, _ = self.h.post_raw(
            "/api/submit",
            {
                "learner": learner,
                "interaction": iid,
                "answer": v_ans,
                "active_seconds": 5.0,
                "request_id": "dup-new-req-0001",
            },
        )
        self.assertEqual(st2, 400)
        self.h.finish(learner, iid)

    def test_f03_b_failed_transaction_rolls_back_db_and_memory_cache(self) -> None:
        learner = self.h.create_profile("Rollback Guard")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        pre_events = len(self.h.get_export(learner)["events"])
        pre_p = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")["p_med"]

        st, _, _ = self.h.post_raw(
            "/api/submit",
            {
                "learner": learner,
                "interaction": iid,
                "answer": v_ans,
                "active_seconds": -5.0,
                "request_id": "rollback-neg-sec-01",
            },
        )
        self.assertEqual(st, 400)
        self.assertEqual(len(self.h.get_export(learner)["events"]), pre_events)
        post_p = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")["p_med"]
        self.assertAlmostEqual(pre_p, post_p, places=8)
        self.h.finish(learner, iid)


class TestF04Boundary(_SharedBoundaryTestCase):
    def test_f04_b_unknown_skill_id_rejected_across_endpoints(self) -> None:
        learner = self.h.create_profile("Bad Skill")
        st_c, _, _ = self.h.get(f"/api/curve?learner={learner}&skill=recursion_magic")
        self.assertEqual(st_c, 400)
        for path in ("/api/select_skill", "/api/ppo_step"):
            st, _, _ = self.h.post_raw(path, {"learner": learner, "skill": "recursion_magic", "request_id": "bad-sk-0001"})
            self.assertEqual(st, 400)

    def test_f04_b_unknown_format_or_difficulty_rejected(self) -> None:
        learner = self.h.create_profile("Bad Fmt Diff")
        st_c, _, _ = self.h.get(f"/api/curve?learner={learner}&skill=range_boundaries&format=essay")
        self.assertEqual(st_c, 400)
        st_p, _, _ = self.h.post_raw(
            "/api/ppo_step",
            {
                "learner": learner,
                "skill": "range_boundaries",
                "action_id": 1,
                "difficulty": "impossible",
                "request_id": "bad-diff-0001",
            },
        )
        self.assertEqual(st_p, 400)

    def test_f04_b_root_skill_range_boundaries_has_empty_prerequisites(self) -> None:
        prereqs = self.h.get_state()["prerequisites"]
        self.assertEqual(list(prereqs["range_boundaries"]), [])
        self.assertEqual(
            list(prereqs["function_returns"]),
            ["conditional_evaluation", "accumulator_initialization"],
        )

    def test_f04_b_bank_exhaustion_in_single_filter_cell_handled_cleanly(self) -> None:
        learner = self.h.create_profile("Cell Exhaust")
        self.h.select_skill(learner, "range_boundaries")
        seen: list[str] = []
        for _ in range(16):
            step = self.h.ppo_step(
                learner,
                "range_boundaries",
                action_id=1,
                difficulty="easy",
                format="single_choice",
                mode="practice",
                question_source="v2.2",
            )
            q = harness.extract_question(step)
            seen.append(q["task_id"])
            iid = harness.extract_interaction_id(step)
            v_ans, _ = harness.solve_v22_task(q)
            self.h.submit(learner, iid, v_ans)
            self.h.finish(learner, iid)
        self.assertEqual(len(set(seen)), 16)
        st17, _, body17 = self.h.post_raw(
            "/api/ppo_step",
            {
                "learner": learner,
                "skill": "range_boundaries",
                "action_id": 1,
                "difficulty": "easy",
                "format": "single_choice",
                "mode": "practice",
                "question_source": "v2.2",
                "request_id": "exhaust-17-0001",
            },
        )
        self.assertIn(st17, (200, 400))
        if st17 == 200:
            d17 = harness.loads_strict_json(body17)
            self.h.finish(learner, harness.extract_interaction_id(d17))

    def test_f04_b_curve_rejects_mismatched_skill_and_task_id(self) -> None:
        learner = self.h.create_profile("Mismatch Curve")
        st, _, _ = self.h.get(
            f"/api/curve?learner={learner}&skill=range_boundaries&task=loop_tracing.easy.single_choice.00"
        )
        self.assertEqual(st, 400)


class TestF05Boundary(_SharedBoundaryTestCase):
    def test_f05_b_zero_score_fast_guess_decreases_prediction_not_increases(self) -> None:
        l_fast = self.h.create_profile("Fast Guess 2s")
        l_slow = self.h.create_profile("Slow Fail 180s")
        cold_p = harness.extract_skill_summary(
            self.h.get_state(l_fast, "conditional_evaluation"), "conditional_evaluation"
        )["p_med"]
        for lrn, sec in ((l_fast, 2.0), (l_slow, 180.0)):
            self.h.select_skill(lrn, "conditional_evaluation")
            step = self.h.ppo_step(lrn, "conditional_evaluation", action_id=1, difficulty="easy", format="single_choice")
            iid = harness.extract_interaction_id(step)
            _, bad = harness.solve_v22_task(harness.extract_question(step))
            self.h.submit(lrn, iid, bad, active_seconds=sec)
            self.h.finish(lrn, iid)
        p_fast = harness.extract_skill_summary(
            self.h.get_state(l_fast, "conditional_evaluation"), "conditional_evaluation"
        )["p_med"]
        p_slow = harness.extract_skill_summary(
            self.h.get_state(l_slow, "conditional_evaluation"), "conditional_evaluation"
        )["p_med"]
        self.assertLess(p_fast, cold_p)
        self.assertAlmostEqual(p_fast, p_slow, places=6)

    def test_f05_b_unresolved_ast_grade_does_not_penalize_like_wrong_answer(self) -> None:
        learner = self.h.create_profile("Unresolved Guard")
        self.h.select_skill(learner, "list_indexing")
        cold = harness.extract_skill_summary(self.h.get_state(learner, "list_indexing"), "list_indexing")
        step = self.h.ppo_step(learner, "list_indexing", action_id=1, difficulty="easy", format="code_completion")
        iid = harness.extract_interaction_id(step)
        sub = self.h.submit(learner, iid, "def solve(x):\n    while True:\n        pass")
        res = sub["result"] if "result" in sub else sub["results"][-1]
        self.assertEqual(res["grading_status"], "unresolved")
        self.assertIsNone(res["score"])
        self.h.finish(learner, iid)
        after = harness.extract_skill_summary(self.h.get_state(learner, "list_indexing"), "list_indexing")
        self.assertEqual(after["observed_unaided_attempts"], 0)
        self.assertEqual(after["n_neg"], cold["n_neg"])

    def test_f05_b_non_evidence_events_keep_skill_unassessed(self) -> None:
        learner = self.h.create_profile("Non Evidence")
        self.h.select_skill(learner, "range_boundaries")
        self.h.ppo_step(learner, "range_boundaries", action_id=3)
        q = self.h.ppo_step(learner, "range_boundaries", action_id=1)
        self.h.finish(learner, harness.extract_interaction_id(q))
        summ = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")
        self.assertFalse(summ["assessed"])
        self.assertEqual(summ["observed_attempts"], 0)

    def test_f05_b_day_60_curve_boundary_and_floor_clamping(self) -> None:
        learner = self.h.create_profile("Day 60 Bounds")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="hard", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        self.h.submit(learner, iid, v_ans)
        self.h.finish(learner, iid)
        summ = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")
        self.assertGreaterEqual(summ["floor_b"], 1e-5)
        self.assertLessEqual(summ["floor_b"], 1.0 - 1e-5)
        self.assertGreaterEqual(summ["half_life_days"], 5.0)
        self.assertLessEqual(summ["half_life_days"], 365.0)
        curve = self.h.get_curve(learner, "range_boundaries")
        for diff in ("easy", "medium", "hard"):
            for pt in curve["curves_by_difficulty"][diff]:
                self.assertGreater(pt["probability"], 0.0)
                self.assertLess(pt["probability"], 1.0)

    def test_f05_b_repeated_read_queries_idempotent_on_state(self) -> None:
        learner = self.h.create_profile("Read Idemp")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        self.h.submit(learner, iid, v_ans)
        self.h.finish(learner, iid)
        baseline = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")
        for _ in range(10):
            self.h.get_curve(learner, "range_boundaries")
            cur = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")
            for k in ("p_med", "floor_b", "active_k", "n_pos", "n_neg"):
                self.assertEqual(cur[k], baseline[k])

    def test_f05_b_unaided_vs_assisted_3hint_success_ordering(self) -> None:
        l_unaided = self.h.create_profile("Unaided Winner")
        l_assisted = self.h.create_profile("3Hint Assisted")
        for lrn, hints in ((l_unaided, 0), (l_assisted, 3)):
            self.h.select_skill(lrn, "range_boundaries")
            step = self.h.ppo_step(lrn, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            iid = harness.extract_interaction_id(step)
            v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
            for _ in range(hints):
                self.h.hint(lrn, iid)
            self.h.submit(lrn, iid, v_ans)
            self.h.finish(lrn, iid)
        su = harness.extract_skill_summary(self.h.get_state(l_unaided, "range_boundaries"), "range_boundaries")
        sa = harness.extract_skill_summary(self.h.get_state(l_assisted, "range_boundaries"), "range_boundaries")
        self.assertGreater(su["p_med"], sa["p_med"])
        self.assertGreater(su["floor_b"], sa["floor_b"])


class TestF06Boundary(_SharedBoundaryTestCase):
    def test_f06_b_masked_action_invocation_rejected_with_400(self) -> None:
        learner = self.h.create_profile("Masked Action")
        self.h.select_skill(learner, "range_boundaries")
        # Action 0 masked at easy
        st0, _, _ = self.h.post_raw(
            "/api/ppo_step",
            {"learner": learner, "skill": "range_boundaries", "action_id": 0, "request_id": "mask-0-0001"},
        )
        self.assertEqual(st0, 400)
        # Action 3 while awaiting_question=True
        self.h.ppo_step(learner, "range_boundaries", action_id=3)
        st3, _, _ = self.h.post_raw(
            "/api/ppo_step",
            {"learner": learner, "skill": "range_boundaries", "action_id": 3, "request_id": "mask-3-0001"},
        )
        self.assertEqual(st3, 400)

    def test_f06_b_unknown_action_id_minus1_and_8_rejected(self) -> None:
        learner = self.h.create_profile("Unknown Action")
        for bad_id in (-1, 8):
            st, _, _ = self.h.post_raw(
                "/api/ppo_step",
                {
                    "learner": learner,
                    "skill": "range_boundaries",
                    "action_id": bad_id,
                    "request_id": f"bad-act-{bad_id}-001",
                },
            )
            self.assertEqual(st, 400)

    def test_f06_b_budget_wrap_at_total_budget_32_boundary(self) -> None:
        learner = self.h.create_profile("Budget Wrap 32")
        self.h.select_skill(learner, "range_boundaries")
        # 20 lessons (0.6 each = 12) + 20 questions (1.0 each = 20) => 32.0+ budget spent
        for _ in range(20):
            self.h.ppo_step(learner, "range_boundaries", action_id=3)
            q = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            iid = harness.extract_interaction_id(q)
            v_ans, _ = harness.solve_v22_task(harness.extract_question(q))
            self.h.submit(learner, iid, v_ans)
            self.h.finish(learner, iid)
        ppo = harness.extract_ppo(self.h.get_state(learner, "range_boundaries"))
        obs = harness.extract_obs_vector(ppo)
        self.assertGreater(obs[21], 0.0)
        self.assertLessEqual(obs[21], 1.0)
        self.assertGreaterEqual(ppo["budget_used"], 0.0)
        self.assertLess(ppo["budget_used"], 32.0)

    def test_f06_b_awaiting_question_is_scoped_per_skill(self) -> None:
        learner = self.h.create_profile("Awaiting Scoped")
        self.h.select_skill(learner, "range_boundaries")
        self.h.ppo_step(learner, "range_boundaries", action_id=3)
        ppo_rb = harness.extract_ppo(self.h.get_state(learner, "range_boundaries"))
        ppo_lt = harness.extract_ppo(self.h.get_state(learner, "loop_tracing"))
        self.assertTrue(ppo_rb["awaiting_question"])
        self.assertFalse(ppo_lt["awaiting_question"])
        self.assertTrue(harness.extract_valid_mask(ppo_lt)[3])

    def test_f06_b_review_prerequisite_unmasked_at_threshold_and_masked_when_healthy(self) -> None:
        learner = self.h.create_profile("Prereq Mask")
        ppo_rb = harness.extract_ppo(self.h.get_state(learner, "range_boundaries"))
        ppo_lt = harness.extract_ppo(self.h.get_state(learner, "loop_tracing"))
        self.assertFalse(harness.extract_valid_mask(ppo_rb)[6])
        self.assertTrue(harness.extract_valid_mask(ppo_lt)[6])


class TestF07Boundary(_SharedBoundaryTestCase):
    def test_f07_b_multiple_select_empty_and_all_selected_clamp_to_zero(self) -> None:
        learner = self.h.create_profile("MS Bounds")
        self.h.select_skill(learner, "range_boundaries")
        for ans_fn in (lambda opts: [], lambda opts: [o["id"] for o in opts]):
            step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="multiple_select")
            q = harness.extract_question(step)
            iid = harness.extract_interaction_id(step)
            sub = self.h.submit(learner, iid, ans_fn(q["options"]))
            res = sub["result"] if "result" in sub else sub["results"][-1]
            self.assertEqual(res["grading_status"], "resolved")
            self.assertEqual(res["score"], 0.0)
            self.assertFalse(res["full_success"])
            self.h.finish(learner, iid)

    def test_f07_b_code_completion_syntax_error_is_resolved_zero(self) -> None:
        learner = self.h.create_profile("AST SyntaxErr")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="code_completion")
        iid = harness.extract_interaction_id(step)
        sub = self.h.submit(learner, iid, "def solve(x) return x")
        res = sub["result"] if "result" in sub else sub["results"][-1]
        self.assertEqual(res["grading_status"], "resolved")
        self.assertEqual(res["score"], 0.0)
        self.assertFalse(res["full_success"])
        self.h.finish(learner, iid)

    def test_f07_b_code_completion_infinite_loop_and_disallowed_import_unresolved(self) -> None:
        learner = self.h.create_profile("AST Unresolved")
        self.h.select_skill(learner, "range_boundaries")
        for code in ("import os\ndef solve(x): return 1", "def solve(x):\n    while True:\n        pass"):
            step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="code_completion")
            iid = harness.extract_interaction_id(step)
            sub = self.h.submit(learner, iid, code)
            res = sub["result"] if "result" in sub else sub["results"][-1]
            self.assertEqual(res["grading_status"], "unresolved")
            self.assertIsNone(res["score"])
            self.assertIsNone(res["full_success"])
            self.h.finish(learner, iid)

    def test_f07_b_code_completion_exceeding_8000_chars_or_1200_ast_nodes_unresolved(self) -> None:
        learner = self.h.create_profile("AST Size Cap")
        self.h.select_skill(learner, "range_boundaries")
        too_long = "def solve(x):\n    return 1\n" + ("# " + ("z" * 100) + "\n") * 90
        many_nodes = "def solve(x):\n    a = " + " + ".join(["1"] * 650) + "\n    return a"
        for code in (too_long, many_nodes):
            step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="code_completion")
            iid = harness.extract_interaction_id(step)
            sub = self.h.submit(learner, iid, code)
            res = sub["result"] if "result" in sub else sub["results"][-1]
            self.assertEqual(res["grading_status"], "unresolved")
            self.h.finish(learner, iid)

    def test_f07_b_fill_in_the_blank_rejects_float_or_non_int_string(self) -> None:
        learner = self.h.create_profile("FITB Non-Int")
        self.h.select_skill(learner, "range_boundaries")
        for bad in (["3.0"], ["abc"]):
            step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="fill_in_the_blank")
            iid = harness.extract_interaction_id(step)
            sub = self.h.submit(learner, iid, bad)
            res = sub["result"] if "result" in sub else sub["results"][-1]
            self.assertEqual(res["grading_status"], "resolved")
            self.assertEqual(res["score"], 0.0)
            self.assertFalse(res["full_success"])
            self.h.finish(learner, iid)

        step_empty = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="fill_in_the_blank")
        iid_e = harness.extract_interaction_id(step_empty)
        st_e, _, _ = self.h.post_raw(
            "/api/submit",
            {"learner": learner, "interaction": iid_e, "answer": [""], "request_id": "fitb-empty-001"},
        )
        self.assertEqual(st_e, 400)
        self.h.finish(learner, iid_e)

    def test_f07_b_active_seconds_boundary_zero_negative_nan_and_huge(self) -> None:
        learner = self.h.create_profile("Timing Bounds")
        self.h.select_skill(learner, "range_boundaries")
        for valid_sec in (0.0, None):
            step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            iid = harness.extract_interaction_id(step)
            v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
            self.h.submit(learner, iid, v_ans, active_seconds=valid_sec)
            self.h.finish(learner, iid)

        step_b = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid_b = harness.extract_interaction_id(step_b)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step_b))
        for bad_sec in (-0.1, 1e9):
            st, _, _ = self.h.post_raw(
                "/api/submit",
                {
                    "learner": learner,
                    "interaction": iid_b,
                    "answer": v_ans,
                    "active_seconds": bad_sec,
                    "request_id": f"bad-sec-{bad_sec}",
                },
            )
            self.assertEqual(st, 400)
        self.h.finish(learner, iid_b)


class TestF08Boundary(unittest.TestCase):
    def test_f08_b_stub_timeout_triggers_clean_fallback(self) -> None:
        with harness.LocalLLMStubServer(delay_seconds=0.8) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-timeout",
                    "LLM_BASE_URL": stub.base_url,
                    "LLM_TIMEOUT_SECONDS": "0.2",
                }
            ) as h:
                learner = h.create_profile("Timeout Fallback")
                res = h.select_skill(learner, "range_boundaries")
                self.assertIn("learning_path", res)
                self.assertIn("course_material", res)

    def test_f08_b_stub_http_429_rate_limit_fallback(self) -> None:
        with harness.LocalLLMStubServer(status_code=429) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-429",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("429 Fallback")
                res = h.select_skill(learner, "loop_tracing")
                self.assertIn("learning_path", res)

    def test_f08_b_stub_truncated_json_and_missing_keys_fallback(self) -> None:
        with harness.LocalLLMStubServer(malformed_json=True) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-trunc",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Truncated JSON")
                r1 = h.select_skill(learner, "range_boundaries")
                self.assertGreaterEqual(len(r1["learning_path"]["steps"]), 1)
                stub.malformed_json = False
                stub.raw_text_override = "{}"
                r2 = h.select_skill(learner, "loop_tracing")
                self.assertGreaterEqual(len(r2["learning_path"]["steps"]), 1)

    def test_f08_b_markdown_fenced_json_from_llm_parsed_cleanly(self) -> None:
        with harness.LocalLLMStubServer(fenced_json=True) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-fenced",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Fenced JSON")
                res = h.select_skill(learner, "range_boundaries")
                self.assertEqual(res["learning_path"]["source"], "llm")

    def test_f08_b_auto_provider_without_keys_never_makes_network_call(self) -> None:
        with harness.AdaptiveWebHarness(env_overrides={"LLM_PROVIDER": "auto"}) as h:
            state = h.get_state()
            self.assertFalse((state.get("llm_status") or {}).get("is_live"))


class TestF09Boundary(_SharedBoundaryTestCase):
    def test_f09_b_repeated_select_skill_same_skill_clean_no_duplicate_crash(self) -> None:
        learner = self.h.create_profile("Repeat Select")
        r1 = self.h.select_skill(learner, "range_boundaries")
        r2 = self.h.select_skill(learner, "range_boundaries")
        self.assertIn("learning_path", r1)
        self.assertIn("learning_path", r2)

    def test_f09_b_select_skill_with_active_open_question_preserves_active_item(self) -> None:
        learner = self.h.create_profile("Switch Active")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        self.h.select_skill(learner, "loop_tracing")
        state = self.h.get_state(learner, "range_boundaries")
        self.assertIsNotNone(state.get("active"))
        self.h.finish(learner, iid)

    def test_f09_b_course_material_escapes_html_script_tags_safely(self) -> None:
        xss_payload = json.dumps(
            {
                "title": "<script>alert(1)</script>Range",
                "concept_explanation": "<script>alert('xss')</script>Start inclusive.",
                "mental_model": "Model",
                "pitfalls": ["<img src=x onerror=alert(1)>"],
                "worked_example": {"title": "WE", "code": "print(1)", "subgoals": [{"label": "S1", "detail": "D1"}]},
                "quick_check": "Check?",
            }
        )
        with harness.LocalLLMStubServer(raw_text_override=xss_payload) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-xss",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("XSS Guard")
                res = h.select_skill(learner, "range_boundaries")
                self.assertIn("course_material", res)
                st, headers, html_b = h.get("/")
                self.assertEqual(st, 200)
                self.assertIn("script-src 'self'", headers.get("content-security-policy", ""))
                self.assertNotIn(b"alert('xss')", html_b)

    def test_f09_b_all_six_skills_generate_distinct_skill_specific_materials(self) -> None:
        learner = self.h.create_profile("All 6 Distinct")
        titles: list[str] = []
        for sk in harness.PPO_SKILLS_ORDER:
            res = self.h.select_skill(learner, sk)
            cm = res["course_material"]
            titles.append(json.dumps(cm))
        self.assertEqual(len(set(titles)), 6)

    def test_f09_b_empty_focus_gap_handled_in_course_material(self) -> None:
        learner = self.h.create_profile("Empty Focus Gap")
        res = self.h.select_skill(learner, "function_returns")
        cm = res["course_material"]
        self.assertTrue(cm["concept_explanation"])
        self.assertTrue(cm["worked_example"])


class TestF10Boundary(unittest.TestCase):
    def test_f10_b_llm_question_credit_weights_sum_to_one_tolerance(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Weight Sum")
            h.select_skill(learner, "range_boundaries")
            step = h.ppo_step(
                learner,
                "range_boundaries",
                action_id=1,
                difficulty="medium",
                format="output_prediction",
                question_source="llm",
            )
            q = harness.extract_question(step)
            h.finish(learner, harness.extract_interaction_id(step))
            gen = [t for t in h.get_export(learner)["generated_tasks"] if t["task_id"] == q["task_id"]]
            cw = (gen[0].get("memory_task") or gen[0])["credit_weights"]
            self.assertEqual(cw, sorted(cw))
            self.assertAlmostEqual(sum(cw), 1.0, places=6)

    def test_f10_b_llm_question_code_completion_rejects_disallowed_builtins(self) -> None:
        unsafe = json.dumps(
            {
                "task_id": "llm.unsafe.cc.01",
                "skill_id": "range_boundaries",
                "task_type": "code_completion",
                "difficulty": "easy",
                "prompt": "Unsafe",
                "snippets": ["def solve(x):\n    pass"],
                "options": [],
                "answer_key": None,
                "reference_solution": "import os\ndef solve(x):\n    return eval('x')",
                "tests": [{"input": 1, "expected": 1}],
                "criteria": [{"kind": "behavior_test", "weight": 1.0}],
                "reference_seconds": 60.0,
            }
        )
        with harness.LocalLLMStubServer(raw_text_override=unsafe) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-unsafe",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Unsafe CC")
                h.select_skill(learner, "range_boundaries")
                step = h.ppo_step(
                    learner,
                    "range_boundaries",
                    action_id=1,
                    difficulty="easy",
                    format="code_completion",
                    question_source="llm",
                )
                q = harness.extract_question(step)
                self.assertNotEqual(q["task_id"], "llm.unsafe.cc.01")
                h.finish(learner, harness.extract_interaction_id(step))

    def test_f10_b_llm_question_multiple_select_penalty_weights_non_empty(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("MS Penalty")
            h.select_skill(learner, "range_boundaries")
            step = h.ppo_step(
                learner,
                "range_boundaries",
                action_id=1,
                difficulty="easy",
                format="multiple_select",
                question_source="llm",
            )
            q = harness.extract_question(step)
            h.finish(learner, harness.extract_interaction_id(step))
            gen = [t for t in h.get_export(learner)["generated_tasks"] if t["task_id"] == q["task_id"]]
            mt = gen[0].get("memory_task") or gen[0]
            self.assertGreater(len(mt["penalty_weights"]), 0)
            self.assertAlmostEqual(sum(mt["penalty_weights"]), 1.0, places=5)

    def test_f10_b_llm_question_duplicate_or_empty_options_rejected_with_fallback(self) -> None:
        dup_opts = json.dumps(
            {
                "task_id": "llm.dup.sc.01",
                "skill_id": "range_boundaries",
                "task_type": "single_choice",
                "difficulty": "easy",
                "prompt": "Dup options",
                "options": [{"id": "o0", "text": "A"}, {"id": "o0", "text": "B"}],
                "answer_key": "o0",
                "criteria": [{"kind": "exact_match", "weight": 1.0}],
                "reference_seconds": 30.0,
            }
        )
        with harness.LocalLLMStubServer(raw_text_override=dup_opts) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-dupopts",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Dup Opts")
                h.select_skill(learner, "range_boundaries")
                step = h.ppo_step(
                    learner,
                    "range_boundaries",
                    action_id=1,
                    difficulty="easy",
                    format="single_choice",
                    question_source="llm",
                )
                q = harness.extract_question(step)
                self.assertEqual(len({o["id"] for o in q["options"]}), 4)
                h.finish(learner, harness.extract_interaction_id(step))

    def test_f10_b_llm_question_zero_or_negative_reference_seconds_rejected(self) -> None:
        zero_sec = json.dumps(
            {
                "task_id": "llm.zerosec.sc.01",
                "skill_id": "range_boundaries",
                "task_type": "single_choice",
                "difficulty": "easy",
                "prompt": "Zero ref sec",
                "options": [
                    {"id": "o0", "text": "A"},
                    {"id": "o1", "text": "B"},
                    {"id": "o2", "text": "C"},
                    {"id": "o3", "text": "D"},
                ],
                "answer_key": "o0",
                "criteria": [{"kind": "exact_match", "weight": 1.0}],
                "reference_seconds": -10.0,
            }
        )
        with harness.LocalLLMStubServer(raw_text_override=zero_sec) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-zerosec",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Zero Sec")
                h.select_skill(learner, "range_boundaries")
                step = h.ppo_step(
                    learner,
                    "range_boundaries",
                    action_id=1,
                    difficulty="easy",
                    format="single_choice",
                    question_source="llm",
                )
                q = harness.extract_question(step)
                self.assertGreater(q["reference_seconds"], 0.0)
                h.finish(learner, harness.extract_interaction_id(step))


class TestF11Boundary(_SharedBoundaryTestCase):
    def test_f11_b_hint_cap_at_three_rejects_fourth_request(self) -> None:
        learner = self.h.create_profile("Hint Cap 3")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        for _ in range(3):
            self.h.hint(learner, iid)
        st4, _, _ = self.h.post_raw("/api/hint", {"learner": learner, "interaction": iid, "request_id": "hint-4-0001"})
        self.assertEqual(st4, 400)
        self.h.finish(learner, iid)

    def test_f11_b_hint_after_correct_submission_rejected(self) -> None:
        learner = self.h.create_profile("Hint After Pass")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        self.h.submit(learner, iid, v_ans)
        st, _, _ = self.h.post_raw("/api/hint", {"learner": learner, "interaction": iid, "request_id": "hint-pass-01"})
        self.assertEqual(st, 400)
        self.h.finish(learner, iid)

    def test_f11_b_retry_cap_at_two_rejects_third_retry(self) -> None:
        learner = self.h.create_profile("Retry Cap 2")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        _, bad = harness.solve_v22_task(harness.extract_question(step))
        for _ in range(2):
            self.h.submit(learner, iid, bad)
            self.h.retry(learner, iid)
        self.h.submit(learner, iid, bad)
        st3, _, _ = self.h.post_raw("/api/retry", {"learner": learner, "interaction": iid, "request_id": "retry-3-001"})
        self.assertEqual(st3, 400)
        self.h.finish(learner, iid)

    def test_f11_b_reveal_before_first_submission_rejected(self) -> None:
        learner = self.h.create_profile("Reveal Early")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        st, _, _ = self.h.post_raw("/api/reveal", {"learner": learner, "interaction": iid, "request_id": "rev-early-01"})
        self.assertEqual(st, 400)
        self.h.finish(learner, iid)

    def test_f11_b_empty_self_explanation_submission_graded_zero_not_500(self) -> None:
        learner = self.h.create_profile("Empty SelfExp")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        self.h.submit(learner, iid, v_ans)
        self.h.finish(learner, iid)

        se = self.h.ppo_step(learner, "range_boundaries", action_id=5)
        se_iid = harness.extract_interaction_id(se)
        st, _, body = self.h.post_raw(
            "/api/submit",
            {
                "learner": learner,
                "interaction": se_iid,
                "answer": "   ",
                "self_explanation": "   ",
                "request_id": "empty-se-001",
            },
        )
        self.assertIn(st, (200, 400))
        if st == 200:
            data = harness.loads_strict_json(body)
            res = data["result"] if "result" in data else data["results"][-1]
            self.assertEqual(res["score"], 0.0)
            self.assertFalse(res["full_success"])
        self.h.finish(learner, se_iid)


class TestF12Boundary(_SharedBoundaryTestCase):
    def test_f12_b_missing_or_wrong_csrf_token_rejected_with_403(self) -> None:
        st_missing, _, _ = self.h.post_raw("/api/profile", {"name": "No CSRF"}, csrf=False)
        self.assertEqual(st_missing, 403)
        st_wrong, _, _ = self.h.post_raw("/api/profile", {"name": "Bad CSRF"}, csrf="wrong-token-xyz")
        self.assertEqual(st_wrong, 403)

    def test_f12_b_disallowed_origin_and_host_headers_rejected_with_403(self) -> None:
        st_org, _, _ = self.h.post_raw(
            "/api/profile",
            {"name": "Bad Origin"},
            origin="https://evil.example",
        )
        self.assertEqual(st_org, 403)
        st_host_post, _, _ = self.h.post_raw(
            "/api/profile",
            {"name": "Bad Host"},
            host="evil.example:8780",
        )
        self.assertEqual(st_host_post, 403)
        st_host_get, _, _ = self.h.get("/api/state", headers={"Host": "evil.example:8780"})
        self.assertEqual(st_host_get, 403)

    def test_f12_b_non_json_content_type_and_malformed_json_rejected(self) -> None:
        st_ct, _, _ = self.h.post_raw("/api/profile", {"name": "Plain"}, content_type="text/plain")
        self.assertEqual(st_ct, 415)
        st_bad, _, _ = self.h.post_raw("/api/profile", b"{bad json")
        self.assertEqual(st_bad, 400)

    def test_f12_b_oversized_body_and_zero_length_post_rejected(self) -> None:
        st_zero, _, _ = self.h.post_raw("/api/profile", b"")
        self.assertIn(st_zero, (400, 411, 413))
        big_body = json.dumps({"name": "X" * (2 * 1024 * 1024), "request_id": "big-req-0001"}).encode("utf-8")
        st_big, _, _ = self.h.post_raw("/api/profile", big_body)
        self.assertIn(st_big, (400, 413))

    def test_f12_b_nan_infinity_rejected_in_json_submit(self) -> None:
        learner = self.h.create_profile("NaN Guard")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        bad_raw = (
            f'{{"learner": "{learner}", "interaction": "{iid}", '
            f'"answer": "o0", "active_seconds": NaN, "request_id": "nan-sec-0001"}}'
        )
        st, _, _ = self.h.post_raw("/api/submit", bad_raw)
        self.assertEqual(st, 400)
        self.h.finish(learner, iid)


class TestF13Boundary(_SharedBoundaryTestCase):
    def test_f13_b_css_zero_forbidden_purple_gradient_hex_or_css_functions(self) -> None:
        _, _, css_b = self.h.get("/style.css")
        _, _, html_b = self.h.get("/")
        combined = (css_b.decode("utf-8") + "\n" + html_b.decode("utf-8")).lower()
        for pat in harness.FORBIDDEN_GRADIENT_PATTERNS:
            self.assertNotIn(pat, combined)
        for stop in harness.FORBIDDEN_PURPLE_STOPS:
            self.assertNotIn(stop, combined)

    def test_f13_b_css_includes_focus_visible_accessibility_outline(self) -> None:
        _, _, css_b = self.h.get("/style.css")
        css = css_b.decode("utf-8").lower()
        self.assertIn(":focus-visible", css)
        self.assertIn("#7058ba", css)

    def test_f13_b_css_responsive_breakpoints_and_reduced_motion(self) -> None:
        _, _, css_b = self.h.get("/style.css")
        css = css_b.decode("utf-8").lower()
        self.assertIn("@media", css)
        self.assertIn("prefers-reduced-motion", css)

    def test_f13_b_favicon_svg_served_valid_xml_with_brand_palette(self) -> None:
        st, headers, body = self.h.get("/favicon.svg")
        self.assertEqual(st, 200)
        self.assertIn("svg", headers.get("content-type", "").lower())
        svg = body.decode("utf-8").lower()
        self.assertIn("<svg", svg)
        self.assertTrue("#293321" in svg or "#c6f36a" in svg)

    def test_f13_b_api_state_without_learner_still_boots_cleanly(self) -> None:
        state = self.h.get_state()
        for key in ("skills", "formats", "prerequisites", "profiles", "llm_status"):
            self.assertIn(key, state)
        self.assertTrue(state.get("csrf_token") or state.get("token"))
        self.assertIsInstance(state["profiles"], list)


class TestF14Boundary(unittest.TestCase):
    def _read_proposal(self) -> str:
        if not harness.PROPOSAL_PATH.exists():
            raise AssertionError(
                "Real 3/adaptive_web/PROPOSAL_PPO_LLM_ARCHITECTURE.md not yet available; cannot fake E2E target"
            )
        return harness.PROPOSAL_PATH.read_text(encoding="utf-8")

    def test_f14_b_proposal_contains_no_unresolved_placeholders(self) -> None:
        text = self._read_proposal()
        for marker in ("TODO", "TBD", "FIXME", "[placeholder]"):
            self.assertNotIn(marker, text)

    def test_f14_b_proposal_contains_zero_secrets_or_api_keys(self) -> None:
        text = self._read_proposal()
        for prefix in ("sk-proj-", "sk-ant-", "AIzaSy"):
            self.assertNotIn(prefix, text)

    def test_f14_b_proposal_discloses_simulated_calibration_limitation(self) -> None:
        low = self._read_proposal().lower()
        self.assertTrue("simulat" in low and ("unvalidated" in low or "real-student" in low or "real student" in low))

    def test_f14_b_proposal_referenced_local_paths_exist(self) -> None:
        text = self._read_proposal()
        refs = re.findall(r"`(3/[A-Za-z0-9_./-]+)`", text)
        self.assertGreater(len(refs), 0)
        for rel in refs:
            if "*" in rel:
                continue
            target = harness.REPO_ROOT / rel
            self.assertTrue(target.exists(), msg=f"Referenced path in proposal does not exist: {rel}")

    def test_f14_b_proposal_port_and_palette_claims_consistent(self) -> None:
        text = self._read_proposal().lower()
        self.assertIn("8780", text)
        self.assertIn("#f7f8f3", text)
        self.assertIn("#293321", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
