"""Tier 1: Category-partition opaque-box feature tests across F01..F14 (73 tests)."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock

E2E_DIR = Path(__file__).resolve().parent
if str(E2E_DIR) not in sys.path:
    sys.path.insert(0, str(E2E_DIR))

import harness  # noqa: E402


class _SharedHarnessTestCase(unittest.TestCase):
    h: harness.AdaptiveWebHarness

    @classmethod
    def setUpClass(cls) -> None:
        cls.h = harness.AdaptiveWebHarness().start()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "h") and cls.h is not None:
            cls.h.stop()


def _complete_one_v22_question(
    h: harness.AdaptiveWebHarness,
    learner: str,
    skill: str,
    *,
    action_id: int = 1,
    difficulty: str = "easy",
    fmt: str | None = None,
    pass_answer: bool = True,
    active_seconds: float = 5.0,
) -> tuple[str, str, dict]:
    extra = {"difficulty": difficulty, "question_source": "v2.2"}
    if fmt is not None:
        extra["format"] = fmt
    step = h.ppo_step(learner, skill, action_id=action_id, **extra)
    q = harness.extract_question(step)
    iid = harness.extract_interaction_id(step)
    valid_ans, invalid_ans = harness.solve_v22_task(q)
    sub = h.submit(learner, iid, valid_ans if pass_answer else invalid_ans, active_seconds=active_seconds)
    h.finish(learner, iid)
    return iid, q["task_id"], sub


class TestF01Isolation(unittest.TestCase):
    def test_f01_default_port_8780_and_separate_db_path(self) -> None:
        harness.ensure_adaptive_web_on_sys_path()
        try:
            config_mod = importlib.import_module("adaptive_web.config")
        except (ImportError, ModuleNotFoundError) as exc:
            raise AssertionError("Real 3/adaptive_web server/app not yet available; cannot fake E2E target") from exc
        clean = {k: v for k, v in os.environ.items() if k not in harness.LIVE_SECRET_KEYS}
        with unittest.mock.patch.dict(os.environ, clean, clear=True):
            settings = config_mod.Settings.from_env(env_path=None)
        port = getattr(settings, "adaptive_web_port", getattr(settings, "port", None))
        db_path = str(getattr(settings, "adaptive_web_db", getattr(settings, "db_path", "")))
        self.assertEqual(port, 8780)
        self.assertNotEqual(port, 8765)
        self.assertTrue(db_path.endswith("adaptive_web/data/adaptive_web.sqlite3"))
        self.assertNotIn("sessions.sqlite3", db_path)

    def test_f01_reference_artifacts_unchanged_after_sessions(self) -> None:
        before = harness.snapshot_reference_hashes()
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Ref Check")
            h.select_skill(learner, "range_boundaries")
            _complete_one_v22_question(h, learner, "range_boundaries", fmt="single_choice")
        after = harness.snapshot_reference_hashes()
        self.assertEqual(before, after)

    def test_f01_cli_server_startup_from_neutral_cwd(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aw_cli_") as tmp:
            db_path = Path(tmp) / "cli_e2e.sqlite3"
            proc, port, _ = harness.spawn_cli_server(
                env_overrides={"ADAPTIVE_WEB_DB": str(db_path)},
                cwd="/tmp",
            )
            try:
                import http.client

                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
                conn.request("GET", "/api/state", headers={"Host": f"127.0.0.1:{port}"})
                resp = conn.getresponse()
                data = harness.loads_strict_json(resp.read())
                conn.close()
                self.assertEqual(resp.status, 200)
                self.assertIn("skills", data)
            finally:
                proc.terminate()
                rc = proc.wait(timeout=8.0)
                self.assertIn(rc, (0, -15))

    def test_f01_dashboard_sqlite_untouched_during_e2e_writes(self) -> None:
        dash_db = harness.CODE_3_DIR / "dashboard" / "data" / "sessions.sqlite3"
        before_exists = dash_db.exists()
        before_stat = dash_db.stat() if before_exists else None
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Isolation User")
            h.select_skill(learner, "loop_tracing")
            _complete_one_v22_question(h, learner, "loop_tracing", fmt="single_choice")
        self.assertEqual(dash_db.exists(), before_exists)
        if before_stat is not None:
            after_stat = dash_db.stat()
            self.assertEqual(before_stat.st_size, after_stat.st_size)
            self.assertEqual(before_stat.st_mtime_ns, after_stat.st_mtime_ns)

    def test_f01_static_routes_serve_adaptive_web_not_dashboard(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            status, headers, body = h.get("/")
            self.assertEqual(status, 200)
            self.assertIn("text/html", headers.get("content-type", ""))
            html = body.decode("utf-8")
            self.assertNotIn("RECALL · Python memory lab", html)
            self.assertIn("style.css", html)
            self.assertIn("app.js", html)


class TestF02ConfigEnv(unittest.TestCase):
    def test_f02_env_var_overrides_port_db_device_provider(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aw_env_") as tmp:
            custom_db = Path(tmp) / "custom_env.sqlite3"
            with harness.AdaptiveWebHarness(
                db_path=custom_db,
                env_overrides={
                    "DEVICE": "cpu",
                    "LLM_PROVIDER": "mock",
                    "LLM_MODEL": "e2e-custom-model",
                    "LLM_TIMEOUT_SECONDS": "12",
                },
            ) as h:
                actual_db = str(getattr(h.settings, "adaptive_web_db", getattr(h.settings, "db_path", "")))
                self.assertEqual(actual_db, str(custom_db))
                state = h.get_state()
                llm = state.get("llm_status") or {}
                self.assertEqual(llm.get("provider"), "mock")
                self.assertFalse(llm.get("is_live"))

    def test_f02_dotenv_file_loaded_without_external_dotenv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aw_dotenv_") as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                '# E2E comment\n\nLLM_PROVIDER="mock"\nLLM_MODEL=\'gpt-4o-mini-e2e\'\n',
                encoding="utf-8",
            )
            with harness.AdaptiveWebHarness(env_path=env_file) as h:
                state = h.get_state()
                llm = state.get("llm_status") or {}
                self.assertEqual(llm.get("provider"), "mock")
                self.assertEqual(llm.get("model"), "gpt-4o-mini-e2e")

    def test_f02_os_environ_precedence_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aw_prec_") as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("LLM_MODEL=from-dotenv\nLLM_PROVIDER=mock\n", encoding="utf-8")
            with harness.AdaptiveWebHarness(
                env_path=env_file,
                env_overrides={"LLM_MODEL": "from-os-environ"},
            ) as h:
                state = h.get_state()
                self.assertEqual((state.get("llm_status") or {}).get("model"), "from-os-environ")

    def test_f02_provider_specific_api_keys_auto_detected(self) -> None:
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "auto",
                    "OPENAI_API_KEY": "sk-e2e-auto-openai",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                state = h.get_state()
                llm = state.get("llm_status") or {}
                self.assertEqual(llm.get("provider"), "openai")
                self.assertTrue(llm.get("is_live"))

    def test_f02_api_keys_never_exposed_in_http_payloads_or_export(self) -> None:
        secret = "sk-secret-e2e-999-must-not-leak"
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": secret,
                    "OPENAI_API_KEY": secret,
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Secret Audit")
                s1 = json.dumps(h.get_state(learner))
                s2 = json.dumps(h.select_skill(learner, "range_boundaries"))
                s3 = json.dumps(h.ppo_step(learner, "range_boundaries", action_id=3))
                s4 = json.dumps(h.get_export(learner))
                for blob in (s1, s2, s3, s4):
                    self.assertNotIn(secret, blob)


class TestF03StoreIdempotency(unittest.TestCase):
    def test_f03_sqlite_wal_tables_and_persistence_across_restart(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="aw_restart_")
        db_path = Path(tmp.name) / "persist.sqlite3"
        try:
            with harness.AdaptiveWebHarness(db_path=db_path, tmp_dir=tmp, keep_tmp=True) as h1:
                learner = h1.create_profile("Restart Alice")
                h1.select_skill(learner, "range_boundaries")
                _complete_one_v22_question(h1, learner, "range_boundaries", fmt="single_choice")
                before_export = h1.get_export(learner)

            with sqlite3.connect(str(db_path)) as conn:
                mode = conn.execute("PRAGMA journal_mode;").fetchone()[0].lower()
                self.assertEqual(mode, "wal")
                tables = {
                    row[0]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                }
                for tbl in (
                    "profiles",
                    "interactions",
                    "events",
                    "requests",
                    "generated_tasks",
                    "learning_paths",
                    "course_materials",
                    "tutor_turns",
                ):
                    self.assertIn(tbl, tables)

            with harness.AdaptiveWebHarness(db_path=db_path, tmp_dir=tmp, keep_tmp=True) as h2:
                state2 = h2.get_state(learner, "range_boundaries")
                summ2 = harness.extract_skill_summary(state2, "range_boundaries")
                self.assertTrue(summ2["assessed"])
                self.assertEqual(summ2["observed_attempts"], 1)
                after_export = h2.get_export(learner)
                self.assertEqual(len(after_export["events"]), len(before_export["events"]))
        finally:
            tmp.cleanup()

    def test_f03_post_idempotency_same_request_id_returns_cached(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Idemp Learner")
            h.select_skill(learner, "range_boundaries")
            step = h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            iid = harness.extract_interaction_id(step)
            valid_ans, _ = harness.solve_v22_task(harness.extract_question(step))
            req_id = "idem-req-000001"
            payload = {
                "learner": learner,
                "interaction": iid,
                "answer": valid_ans,
                "active_seconds": 4.0,
                "request_id": req_id,
            }
            r1 = h.post("/api/submit", payload, auto_request_id=False)
            r2 = h.post("/api/submit", payload, auto_request_id=False)
            self.assertEqual(r1, r2)
            exp = h.get_export(learner)
            grade_events = [e for e in exp["events"] if e.get("kind") == "grade"]
            self.assertEqual(len(grade_events), 1)

    def test_f03_profile_isolation_across_two_learners(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            a = h.create_profile("Learner A")
            b = h.create_profile("Learner B")
            h.select_skill(a, "range_boundaries")
            _complete_one_v22_question(h, a, "range_boundaries", fmt="single_choice")
            _complete_one_v22_question(h, a, "range_boundaries", fmt="single_choice")

            state_b = h.get_state(b)
            for sk in harness.PPO_SKILLS_ORDER:
                sb = harness.extract_skill_summary(state_b, sk)
                self.assertFalse(sb["assessed"])
                self.assertEqual(sb["observed_attempts"], 0)
            self.assertEqual(h.get_export(b)["events"], [])

    def test_f03_skill_isolation_within_same_learner(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Skill Iso")
            h.select_skill(learner, "range_boundaries")
            _complete_one_v22_question(h, learner, "range_boundaries", fmt="single_choice")
            state = h.get_state(learner, "range_boundaries")
            self.assertTrue(harness.extract_skill_summary(state, "range_boundaries")["assessed"])
            self.assertFalse(harness.extract_skill_summary(state, "loop_tracing")["assessed"])
            self.assertFalse(h.get_curve(learner, "loop_tracing")["assessed"])

    def test_f03_export_schema_contains_all_provenance_artifacts(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Export Full")
            h.select_skill(learner, "range_boundaries")
            h.ppo_step(learner, "range_boundaries", action_id=3)
            _complete_one_v22_question(h, learner, "range_boundaries", fmt="single_choice")
            exp = h.get_export(learner)
            for key in (
                "profile",
                "model_sha256",
                "ppo_sha256",
                "interactions",
                "events",
                "learning_paths",
                "course_materials",
                "tutor_turns",
                "generated_tasks",
            ):
                self.assertIn(key, exp)
            self.assertGreaterEqual(len(exp["learning_paths"]), 1)
            self.assertGreaterEqual(len(exp["course_materials"]), 1)
            self.assertGreaterEqual(len(exp["tutor_turns"]), 2)


class TestF04BankDAG(_SharedHarnessTestCase):
    def test_f04_all_six_canonical_skills_in_ppo_order_and_dag(self) -> None:
        state = self.h.get_state()
        skills = state.get("skills_order") or [
            s if isinstance(s, str) else (s.get("skill_id") or s.get("skill") or s.get("id"))
            for s in state.get("skills", [])
        ]
        self.assertEqual(tuple(skills), harness.PPO_SKILLS_ORDER)
        prereqs = state.get("prerequisites") or {}
        for sk, expected in harness.PREREQUISITES.items():
            self.assertEqual(tuple(prereqs.get(sk, ())), expected)

    def test_f04_all_five_task_formats_deliverable_from_v22(self) -> None:
        learner = self.h.create_profile("All 5 Formats")
        self.h.select_skill(learner, "range_boundaries")
        for fmt in harness.TASK_FORMATS:
            step = self.h.ppo_step(
                learner,
                "range_boundaries",
                action_id=1,
                difficulty="easy",
                format=fmt,
                question_source="v2.2",
            )
            q = harness.extract_question(step)
            iid = harness.extract_interaction_id(step)
            self.assertEqual(q["task_type"], fmt)
            self.assertIn(q["task_id"], harness.load_v22_catalog())
            self.h.finish(learner, iid)

    def test_f04_practice_vs_assessment_pool_routing(self) -> None:
        learner = self.h.create_profile("Pool Routing")
        self.h.select_skill(learner, "range_boundaries")
        catalog = harness.load_v22_catalog()
        p_step = self.h.ppo_step(
            learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice", mode="practice"
        )
        p_task = catalog[harness.extract_question(p_step)["task_id"]]
        self.assertEqual(p_task["pool"], "practice")
        self.h.finish(learner, harness.extract_interaction_id(p_step))

        a_step = self.h.ppo_step(
            learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice", mode="assessment"
        )
        a_task = catalog[harness.extract_question(a_step)["task_id"]]
        self.assertEqual(a_task["pool"], "assessment")
        self.h.finish(learner, harness.extract_interaction_id(a_step))

    def test_f04_unseen_item_selection_avoids_repeats(self) -> None:
        learner = self.h.create_profile("Unseen Dedupe")
        self.h.select_skill(learner, "range_boundaries")
        seen_ids: list[str] = []
        for _ in range(3):
            _, tid, _ = _complete_one_v22_question(
                self.h, learner, "range_boundaries", fmt="single_choice", difficulty="easy"
            )
            seen_ids.append(tid)
        self.assertEqual(len(set(seen_ids)), 3)

    def test_f04_v22_task_answer_key_and_criteria_hidden_before_submit(self) -> None:
        learner = self.h.create_profile("No Leak")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(
            learner, "range_boundaries", action_id=1, difficulty="easy", format="code_completion", question_source="v2.2"
        )
        q = harness.extract_question(step)
        for forbidden in ("answer_key", "reference_solution", "criteria", "tests", "hints"):
            self.assertNotIn(forbidden, q)
        self.h.finish(learner, harness.extract_interaction_id(step))


class TestF05MemoryAdapter(_SharedHarnessTestCase):
    def test_f05_unassessed_cold_start_guard_and_priors(self) -> None:
        learner = self.h.create_profile("Cold Start")
        state = self.h.get_state(learner, "range_boundaries")
        summ = harness.extract_skill_summary(state, "range_boundaries")
        self.assertFalse(summ["assessed"])
        self.assertEqual(summ["observed_attempts"], 0)
        self.assertIsNone(summ.get("observed_pass_rate"))
        self.assertGreater(summ["p_easy"], summ["p_med"])
        self.assertGreater(summ["p_med"], summ["p_hard"])
        self.assertNotAlmostEqual(summ["p_med"], 0.5, places=2)

    def test_f05_observed_grades_strictly_separated_from_predictions(self) -> None:
        learner = self.h.create_profile("Obs vs Pred")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        valid_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        self.h.hint(learner, iid)
        self.h.submit(learner, iid, valid_ans)
        self.h.finish(learner, iid)
        _complete_one_v22_question(self.h, learner, "range_boundaries", fmt="single_choice", pass_answer=True)

        summ = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")
        self.assertTrue(summ["assessed"])
        self.assertEqual(summ["observed_attempts"], 2)
        self.assertEqual(summ["observed_unaided_attempts"], 1)
        self.assertEqual(summ["observed_unaided_passes"], 1)
        self.assertAlmostEqual(summ["observed_pass_rate"], 1.0, places=5)
        self.assertLess(summ["p_med"], 0.99)
        self.assertIn("floor_b", summ)
        self.assertIn("active_k", summ)

    def test_f05_0_to_60_day_retention_curves_monotonic_and_difficulty_ordered(self) -> None:
        learner = self.h.create_profile("Curve Monotonic")
        self.h.select_skill(learner, "function_returns")
        _complete_one_v22_question(self.h, learner, "function_returns", difficulty="hard", fmt="single_choice")
        curve = self.h.get_curve(learner, "function_returns", fmt="output_prediction")
        self.assertTrue(curve["assessed"])
        by_diff = curve["curves_by_difficulty"]
        days = [pt["day"] for pt in by_diff["medium"]]
        self.assertEqual(days, [0, 1, 3, 7, 14, 30, 45, 60])
        for diff in ("easy", "medium", "hard"):
            probs = [pt["probability"] for pt in by_diff[diff]]
            for a, b in zip(probs, probs[1:]):
                self.assertLessEqual(b, a + 1e-7)
        for i in range(len(days)):
            self.assertGreater(by_diff["easy"][i]["probability"], by_diff["medium"][i]["probability"])
            self.assertGreater(by_diff["medium"][i]["probability"], by_diff["hard"][i]["probability"])

    def test_f05_curve_supports_all_five_task_formats(self) -> None:
        learner = self.h.create_profile("Curve Formats")
        p_meds: list[float] = []
        for fmt in harness.TASK_FORMATS:
            curve = self.h.get_curve(learner, "range_boundaries", fmt=fmt)
            self.assertEqual(curve["format"], fmt)
            p_meds.append(curve["curves_by_difficulty"]["medium"][0]["probability"])
        self.assertEqual(len(set(round(p, 4) for p in p_meds)), len(harness.TASK_FORMATS))

    def test_f05_structured_token_audit_zero_text_or_code_embeddings(self) -> None:
        learner = self.h.create_profile("Token Audit")
        self.h.select_skill(learner, "range_boundaries")
        _complete_one_v22_question(self.h, learner, "range_boundaries", fmt="code_completion")
        state = self.h.get_state(learner, "range_boundaries")
        audit = state.get("token_audit") or state.get("glass_box", {}).get("token_audit")
        self.assertIsInstance(audit, dict)
        self.assertEqual(audit.get("token_dim"), 47)
        self.assertEqual(audit.get("task_feature_dim"), 19)
        self.assertFalse(audit.get("uses_text_or_code_embeddings"))

    def test_f05_clean_token_speed_deconfounding_zero_score_invariant(self) -> None:
        fast_learner = self.h.create_profile("Fast Guess 0%")
        slow_learner = self.h.create_profile("Slow Fail 0%")
        cold_p = harness.extract_skill_summary(
            self.h.get_state(fast_learner, "range_boundaries"), "range_boundaries"
        )["p_med"]

        self.h.select_skill(fast_learner, "range_boundaries")
        self.h.select_skill(slow_learner, "range_boundaries")
        _complete_one_v22_question(
            self.h, fast_learner, "range_boundaries", fmt="single_choice", pass_answer=False, active_seconds=5.0
        )
        _complete_one_v22_question(
            self.h, slow_learner, "range_boundaries", fmt="single_choice", pass_answer=False, active_seconds=180.0
        )
        sf = harness.extract_skill_summary(self.h.get_state(fast_learner, "range_boundaries"), "range_boundaries")
        ss = harness.extract_skill_summary(self.h.get_state(slow_learner, "range_boundaries"), "range_boundaries")
        self.assertLess(sf["p_med"], cold_p)
        self.assertAlmostEqual(sf["p_med"], ss["p_med"], places=6)
        self.assertAlmostEqual(sf["floor_b"], ss["floor_b"], places=6)


class TestF06PPOEngine(_SharedHarnessTestCase):
    def test_f06_36_dim_observation_bounds_and_3seed_ensemble(self) -> None:
        learner = self.h.create_profile("PPO Telemetry")
        state = self.h.get_state(learner, "range_boundaries")
        ppo = harness.extract_ppo(state)
        obs = harness.extract_obs_vector(ppo)
        probs = harness.extract_ensemble_probs(ppo)
        self.assertEqual(len(obs), 36)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in obs))
        self.assertEqual(len(probs), 8)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertEqual(len(ppo["per_seed_probs"]), 3)
        self.assertIsInstance(ppo["critic_value"], (int, float))

    def test_f06_cold_start_valid_mask_actions_1_2_3_4(self) -> None:
        learner = self.h.create_profile("Cold Mask")
        state = self.h.get_state(learner, "range_boundaries")
        mask = harness.extract_valid_mask(harness.extract_ppo(state))
        valid_ids = [idx for idx, ok in enumerate(mask) if ok]
        self.assertEqual(valid_ids, [1, 2, 3, 4])

    def test_f06_gated_option_b_after_passive_lesson_forces_question(self) -> None:
        learner = self.h.create_profile("Gated Option B")
        self.h.select_skill(learner, "range_boundaries")
        lesson = self.h.ppo_step(learner, "range_boundaries", action_id=3)
        ppo = harness.extract_ppo(lesson)
        self.assertTrue(ppo["awaiting_question"])
        self.assertEqual(ppo["primed_mode"], "concept")
        mask = harness.extract_valid_mask(ppo)
        self.assertEqual([i for i, ok in enumerate(mask) if ok], [1, 2])
        q_resp = self.h.ppo_step(learner, "range_boundaries", action_id=1)
        self.assertTrue(harness.extract_question(q_resp))
        self.h.finish(learner, harness.extract_interaction_id(q_resp))

    def test_f06_self_explanation_unlocks_after_question_and_relocks(self) -> None:
        learner = self.h.create_profile("SelfExp Gate")
        self.h.select_skill(learner, "range_boundaries")
        _complete_one_v22_question(self.h, learner, "range_boundaries", fmt="single_choice")
        ppo1 = harness.extract_ppo(self.h.get_state(learner, "range_boundaries"))
        self.assertTrue(harness.extract_valid_mask(ppo1)[5])

        se_step = self.h.ppo_step(learner, "range_boundaries", action_id=5)
        iid = harness.extract_interaction_id(se_step)
        self.h.submit(
            learner,
            iid,
            "Range starts at the inclusive start and stops strictly before stop.",
            self_explanation="Range starts at the inclusive start and stops strictly before stop.",
        )
        self.h.finish(learner, iid)
        ppo2 = harness.extract_ppo(self.h.get_state(learner, "range_boundaries"))
        self.assertFalse(harness.extract_valid_mask(ppo2)[5])

    def test_f06_generalized_reconstruct_context_tracks_llm_generated_tasks(self) -> None:
        learner = self.h.create_profile("LLM Context Track")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(
            learner,
            "range_boundaries",
            action_id=2,
            difficulty="hard",
            question_source="llm",
        )
        q = harness.extract_question(step)
        self.assertTrue(q["task_id"].startswith("llm."))
        iid = harness.extract_interaction_id(step)
        self.h.submit(learner, iid, "WRONG_OR_FALLBACK")
        self.h.finish(learner, iid)

        ppo = harness.extract_ppo(self.h.get_state(learner, "range_boundaries"))
        obs = harness.extract_obs_vector(ppo)
        self.assertGreater(obs[12], 0.0)
        self.assertEqual(obs[18], 1.0)
        self.assertFalse(harness.extract_valid_mask(ppo)[2])

    def test_f06_manual_override_logs_learner_manual_provenance(self) -> None:
        learner = self.h.create_profile("Provenance")
        self.h.select_skill(learner, "range_boundaries")
        l1 = self.h.ppo_step(learner, "range_boundaries", action_id=3)
        self.assertEqual(l1.get("policy_source", "delayed_retention_ppo"), "delayed_retention_ppo")
        q1 = self.h.ppo_step(learner, "range_boundaries", action_id=1, policy_source="learner_manual")
        self.h.finish(learner, harness.extract_interaction_id(q1))
        turns = self.h.get_export(learner)["tutor_turns"]
        sources = {t.get("policy_source") for t in turns}
        self.assertIn("delayed_retention_ppo", sources)
        self.assertIn("learner_manual", sources)


class TestF07Grader(_SharedHarnessTestCase):
    def test_f07_single_choice_correct_and_wrong_grading(self) -> None:
        learner = self.h.create_profile("SC Grade")
        self.h.select_skill(learner, "range_boundaries")
        _, _, sub_ok = _complete_one_v22_question(self.h, learner, "range_boundaries", fmt="single_choice", pass_answer=True)
        res_ok = sub_ok["result"] if "result" in sub_ok else sub_ok["results"][-1]
        self.assertEqual(res_ok["grading_status"], "resolved")
        self.assertEqual(res_ok["score"], 1.0)
        self.assertTrue(res_ok["full_success"])

        _, _, sub_bad = _complete_one_v22_question(self.h, learner, "range_boundaries", fmt="single_choice", pass_answer=False)
        res_bad = sub_bad["result"] if "result" in sub_bad else sub_bad["results"][-1]
        self.assertEqual(res_bad["grading_status"], "resolved")
        self.assertEqual(res_bad["score"], 0.0)
        self.assertFalse(res_bad["full_success"])

    def test_f07_multiple_select_partial_and_penalty_grading(self) -> None:
        learner = self.h.create_profile("MS Grade")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="multiple_select")
        q = harness.extract_question(step)
        iid = harness.extract_interaction_id(step)
        valid_ans, _ = harness.solve_v22_task(q)
        partial = self.h.submit(learner, iid, [valid_ans[0]])
        res = partial["result"] if "result" in partial else partial["results"][-1]
        self.assertEqual(res["grading_status"], "resolved")
        self.assertGreater(res["score"], 0.0)
        self.assertLess(res["score"], 1.0)
        self.assertFalse(res["full_success"])
        self.h.finish(learner, iid)

    def test_f07_output_prediction_normalization_and_partial_credit(self) -> None:
        learner = self.h.create_profile("OP Grade")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="medium", format="output_prediction")
        q = harness.extract_question(step)
        iid = harness.extract_interaction_id(step)
        valid_ans, _ = harness.solve_v22_task(q)
        spaced = [valid_ans[0].replace(", ", ",") + "\n", "WRONG"]
        sub = self.h.submit(learner, iid, spaced)
        res = sub["result"] if "result" in sub else sub["results"][-1]
        self.assertEqual(res["grading_status"], "resolved")
        self.assertGreater(res["score"], 0.0)
        self.assertLess(res["score"], 1.0)
        self.assertFalse(res["full_success"])
        self.h.finish(learner, iid)

    def test_f07_fill_in_the_blank_exact_int_grading(self) -> None:
        learner = self.h.create_profile("FITB Grade")
        self.h.select_skill(learner, "range_boundaries")
        _, _, sub = _complete_one_v22_question(self.h, learner, "range_boundaries", fmt="fill_in_the_blank", pass_answer=True)
        res = sub["result"] if "result" in sub else sub["results"][-1]
        self.assertEqual(res["grading_status"], "resolved")
        self.assertEqual(res["score"], 1.0)
        self.assertTrue(res["full_success"])

    def test_f07_code_completion_ast_interpreter_grades_valid_solve(self) -> None:
        learner = self.h.create_profile("AST Grade")
        self.h.select_skill(learner, "list_indexing")
        step = self.h.ppo_step(learner, "list_indexing", action_id=1, difficulty="easy", format="code_completion")
        q = harness.extract_question(step)
        iid = harness.extract_interaction_id(step)
        valid_ans, _ = harness.solve_v22_task(q)
        alt_ans = valid_ans.replace("return ", "res = ") + "\n    return res"
        sub = self.h.submit(learner, iid, alt_ans)
        res = sub["result"] if "result" in sub else sub["results"][-1]
        self.assertEqual(res["grading_status"], "resolved")
        self.assertEqual(res["score"], 1.0)
        self.assertTrue(res["full_success"])
        self.h.finish(learner, iid)

    def test_f07_self_explanation_grader_separates_confidence_from_score(self) -> None:
        learner = self.h.create_profile("SelfExp Grader")
        self.h.select_skill(learner, "range_boundaries")
        _complete_one_v22_question(self.h, learner, "range_boundaries", fmt="single_choice")
        se = self.h.ppo_step(learner, "range_boundaries", action_id=5)
        iid = harness.extract_interaction_id(se)
        sub = self.h.submit(
            learner,
            iid,
            "Because range(start, stop) includes start and excludes stop so iteration terminates before stop.",
            self_explanation="Because range(start, stop) includes start and excludes stop so iteration terminates before stop.",
        )
        res = sub["result"] if "result" in sub else sub["results"][-1]
        self.assertIn("grader_confidence", res)
        self.assertIsInstance(res["grader_confidence"], float)
        self.assertEqual(res["grading_status"], "resolved")
        self.assertGreater(res["score"], 0.0)
        self.h.finish(learner, iid)


class TestF08LLMClient(unittest.TestCase):
    def test_f08_mock_provider_deterministic_double_reproducible(self) -> None:
        with harness.AdaptiveWebHarness(env_overrides={"LLM_PROVIDER": "mock"}) as h:
            l1 = h.create_profile("Mock 1")
            l2 = h.create_profile("Mock 2")
            r1 = h.select_skill(l1, "loop_tracing")
            r2 = h.select_skill(l2, "loop_tracing")
            self.assertEqual(r1["learning_path"]["steps"], r2["learning_path"]["steps"])
            self.assertEqual(r1["course_material"]["title"], r2["course_material"]["title"])
            self.assertEqual(r1["learning_path"].get("source"), "deterministic_double")

    def test_f08_openai_compatible_http_client_via_loopback_stub(self) -> None:
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-openai-stub",
                    "LLM_BASE_URL": stub.base_url,
                    "LLM_MODEL": "gpt-4o-mini",
                }
            ) as h:
                learner = h.create_profile("OpenAI Stub")
                res = h.select_skill(learner, "range_boundaries")
                self.assertEqual(res["learning_path"]["source"], "llm")
                self.assertGreaterEqual(len(stub.received_requests), 1)
                req0 = stub.received_requests[0]
                self.assertIn("/chat/completions", req0["path"])
                self.assertEqual(req0["headers"].get("authorization"), "Bearer sk-e2e-openai-stub")

    def test_f08_anthropic_http_client_via_loopback_stub(self) -> None:
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "anthropic",
                    "ANTHROPIC_API_KEY": "sk-ant-e2e-stub",
                    "LLM_BASE_URL": stub.base_url,
                    "LLM_MODEL": "claude-3-5-haiku-latest",
                }
            ) as h:
                learner = h.create_profile("Anthropic Stub")
                res = h.select_skill(learner, "range_boundaries")
                self.assertEqual(res["learning_path"]["source"], "llm")
                req0 = stub.received_requests[0]
                self.assertIn("/messages", req0["path"])
                self.assertEqual(req0["headers"].get("x-api-key"), "sk-ant-e2e-stub")
                self.assertIn("anthropic-version", req0["headers"])

    def test_f08_gemini_http_client_via_loopback_stub(self) -> None:
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "gemini",
                    "GEMINI_API_KEY": "AIza-e2e-gemini-stub",
                    "LLM_BASE_URL": stub.base_url,
                    "LLM_MODEL": "gemini-2.0-flash",
                }
            ) as h:
                learner = h.create_profile("Gemini Stub")
                res = h.select_skill(learner, "range_boundaries")
                self.assertEqual(res["learning_path"]["source"], "llm")
                req0 = stub.received_requests[0]
                self.assertIn("generatecontent", req0["path"].lower())

    def test_f08_http_error_or_malformed_json_falls_back_gracefully(self) -> None:
        with harness.LocalLLMStubServer(status_code=500) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-err",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Fallback 500")
                res = h.select_skill(learner, "range_boundaries")
                self.assertIn("learning_path", res)
                self.assertIn("course_material", res)
                stub.status_code = 200
                stub.malformed_json = True
                step = h.ppo_step(learner, "range_boundaries", action_id=1, question_source="llm")
                self.assertTrue(harness.extract_question(step))
                h.finish(learner, harness.extract_interaction_id(step))


class TestF09PathCourse(_SharedHarnessTestCase):
    def test_f09_select_skill_generates_curated_path_and_material(self) -> None:
        learner = self.h.create_profile("Path & Course")
        res = self.h.select_skill(learner, "conditional_evaluation")
        lp = res["learning_path"]
        cm = res["course_material"]
        self.assertGreaterEqual(len(lp["steps"]), 3)
        for key in ("concept_explanation", "mental_model", "pitfalls", "worked_example", "quick_check"):
            self.assertIn(key, cm)
            self.assertTrue(cm[key])

    def test_f09_curated_path_reflects_execute_program_cadence(self) -> None:
        learner = self.h.create_profile("Cadence")
        res = self.h.select_skill(learner, "loop_tracing")
        blob = json.dumps(res["learning_path"]).lower()
        self.assertTrue(any(w in blob for w in ("learn", "concept", "diagnostic")))
        self.assertTrue(any(w in blob for w in ("code", "worked", "practice")))
        self.assertTrue(any(w in blob for w in ("remember", "self", "consolidation", "spaced")))

    def test_f09_path_adapts_for_assessed_weak_vs_strong_memory(self) -> None:
        weak = self.h.create_profile("Weak Memory")
        strong = self.h.create_profile("Strong Memory")
        self.h.select_skill(weak, "range_boundaries")
        self.h.select_skill(strong, "range_boundaries")
        for _ in range(2):
            _complete_one_v22_question(self.h, weak, "range_boundaries", fmt="single_choice", pass_answer=False)
            _complete_one_v22_question(self.h, strong, "range_boundaries", fmt="single_choice", pass_answer=True)
        res_weak = self.h.select_skill(weak, "range_boundaries")
        res_strong = self.h.select_skill(strong, "range_boundaries")
        self.assertNotEqual(res_weak["learning_path"], res_strong["learning_path"])

    def test_f09_worked_examples_include_subgoal_labels(self) -> None:
        learner = self.h.create_profile("Subgoals")
        res = self.h.select_skill(learner, "accumulator_initialization")
        we = res["course_material"]["worked_example"]
        subgoals = we.get("subgoals") or we.get("steps") or []
        self.assertGreaterEqual(len(subgoals), 2)

    def test_f09_path_and_material_persisted_per_learner_and_skill(self) -> None:
        learner = self.h.create_profile("Persist Path")
        r1 = self.h.select_skill(learner, "range_boundaries")
        self.h.select_skill(learner, "loop_tracing")
        state_rb = self.h.get_state(learner, "range_boundaries")
        self.assertEqual(state_rb["learning_path"]["steps"], r1["learning_path"]["steps"])
        self.assertEqual(state_rb["course_material"]["title"], r1["course_material"]["title"])


class TestF10LLMQuestions(unittest.TestCase):
    def test_f10_generate_fresh_llm_question_across_formats(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("LLM Formats")
            h.select_skill(learner, "range_boundaries")
            for fmt in ("single_choice", "output_prediction", "code_completion"):
                step = h.ppo_step(
                    learner,
                    "range_boundaries",
                    action_id=1,
                    difficulty="easy",
                    format=fmt,
                    question_source="llm",
                )
                q = harness.extract_question(step)
                self.assertTrue(q["task_id"].startswith("llm."))
                self.assertEqual(q["task_type"], fmt)
                h.finish(learner, harness.extract_interaction_id(step))

    def test_f10_llm_question_emits_exact_8key_memory_task_metadata(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("LLM Meta 8")
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
            h.finish(learner, harness.extract_interaction_id(step))
            exp = h.get_export(learner)
            gen = [t for t in exp["generated_tasks"] if t.get("task_id") == q["task_id"]]
            self.assertEqual(len(gen), 1)
            mt = gen[0].get("memory_task") or gen[0]
            expected_keys = {
                "task_type",
                "difficulty",
                "reference_seconds",
                "snippet_count",
                "option_count",
                "criterion_count",
                "credit_weights",
                "penalty_weights",
            }
            self.assertTrue(expected_keys.issubset(set(mt.keys())))
            self.assertAlmostEqual(sum(mt["credit_weights"]), 1.0, places=5)

    def test_f10_llm_code_completion_executable_in_ast_interpreter(self) -> None:
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-cc",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("LLM AST Exec")
                h.select_skill(learner, "range_boundaries")
                step = h.ppo_step(
                    learner,
                    "range_boundaries",
                    action_id=1,
                    difficulty="easy",
                    format="code_completion",
                    question_source="llm",
                )
                iid = harness.extract_interaction_id(step)
                sub = h.submit(learner, iid, "def solve(x):\n    return list(range(1, x, 2))")
                res = sub["result"] if "result" in sub else sub["results"][-1]
                self.assertEqual(res["grading_status"], "resolved")
                self.assertEqual(res["score"], 1.0)
                self.assertTrue(res["full_success"])
                h.finish(learner, iid)

    def test_f10_llm_question_updates_monotonic_gru_identically_to_v22(self) -> None:
        with harness.LocalLLMStubServer() as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-gru",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("LLM GRU Parity")
                h.select_skill(learner, "range_boundaries")
                cold = harness.extract_skill_summary(h.get_state(learner, "range_boundaries"), "range_boundaries")
                step = h.ppo_step(
                    learner,
                    "range_boundaries",
                    action_id=1,
                    difficulty="easy",
                    format="single_choice",
                    question_source="llm",
                )
                iid = harness.extract_interaction_id(step)
                h.submit(learner, iid, "o0")
                h.finish(learner, iid)
                after = harness.extract_skill_summary(h.get_state(learner, "range_boundaries"), "range_boundaries")
                self.assertTrue(after["assessed"])
                self.assertEqual(after["observed_attempts"], 1)
                self.assertGreater(after["p_med"], cold["p_med"])
                self.assertGreater(after["n_pos"], cold["n_pos"])

    def test_f10_invalid_llm_question_payload_falls_back_cleanly(self) -> None:
        bad_payload = json.dumps(
            {
                "task_id": "llm.bad.01",
                "skill_id": "range_boundaries",
                "task_type": "single_choice",
                "difficulty": "easy",
                "prompt": "Broken weights",
                "options": [{"id": "o0", "text": "only one option"}],
                "answer_key": "o0",
                "criteria": [{"kind": "exact_match", "weight": 0.2}],
                "reference_seconds": 30.0,
            }
        )
        with harness.LocalLLMStubServer(raw_text_override=bad_payload) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-badq",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                learner = h.create_profile("Invalid Q Fallback")
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
                self.assertEqual(len(q["options"]), 4)
                h.finish(learner, harness.extract_interaction_id(step))


class TestF11BoundedPPOTurns(_SharedHarnessTestCase):
    def test_f11_all_eight_ppo_actions_realizable_through_api(self) -> None:
        learner = self.h.create_profile("All 8 Actions")
        self.h.select_skill(learner, "loop_tracing")

        # 3: explain_concept -> 1: same -> submit -> 5: self_explanation -> submit
        l3 = self.h.ppo_step(learner, "loop_tracing", action_id=3)
        self.assertEqual(l3["ppo_action"], "explain_concept")
        _complete_one_v22_question(self.h, learner, "loop_tracing", action_id=1, fmt="single_choice")
        se = self.h.ppo_step(learner, "loop_tracing", action_id=5)
        self.assertEqual(se["ppo_action"], "request_self_explanation")
        se_iid = harness.extract_interaction_id(se)
        self.h.submit(learner, se_iid, "Loop iterates over each item once.", self_explanation="Loop iterates over each item once.")
        self.h.finish(learner, se_iid)

        # 4: show_worked_example -> 2: harder -> submit -> 0: easier -> submit
        l4 = self.h.ppo_step(learner, "loop_tracing", action_id=4)
        self.assertEqual(l4["ppo_action"], "show_worked_example")
        _complete_one_v22_question(self.h, learner, "loop_tracing", action_id=2, difficulty="medium", fmt="single_choice")
        _complete_one_v22_question(self.h, learner, "loop_tracing", action_id=0, difficulty="easy", fmt="single_choice")

        # 6: review_prerequisite (range_boundaries is unassessed/weak) -> submit
        r6 = self.h.ppo_step(learner, "loop_tracing", action_id=6)
        self.assertEqual(r6["ppo_action"], "review_prerequisite")
        self.assertEqual(r6["target_skill"], "range_boundaries")
        r6_iid = harness.extract_interaction_id(r6)
        v6, _ = harness.solve_v22_task(harness.extract_question(r6))
        self.h.submit(learner, r6_iid, v6)
        self.h.finish(learner, r6_iid)

        # 7: advance_next_topic
        a7 = self.h.ppo_step(learner, "loop_tracing", action_id=7)
        self.assertEqual(a7["ppo_action"], "advance_next_topic")
        if "question" in a7 or "interaction_id" in a7 or "id" in a7:
            self.h.finish(learner, harness.extract_interaction_id(a7))

    def test_f11_progressive_3tier_hints_emit_leakage_and_kinds(self) -> None:
        learner = self.h.create_profile("3 Tier Hints")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        kinds: list[str] = []
        leakages: list[float] = []
        for expected_lvl in (1, 2, 3):
            res = self.h.hint(learner, iid)
            h_obj = res.get("hint") or res
            self.assertEqual(h_obj.get("level", h_obj.get("hint_level")), expected_lvl)
            kinds.append(h_obj.get("kind", h_obj.get("hint_kind")))
            leakages.append(float(h_obj["leakage_l_t"]))
        self.assertEqual(kinds, ["conceptual", "targeted", "scaffold"])
        self.assertTrue(leakages[0] < leakages[1] < leakages[2] <= 1.0)
        self.h.finish(learner, iid)

    def test_f11_assisted_pass_sets_unverified_and_triggers_unaided_check(self) -> None:
        learner = self.h.create_profile("Unverified Check")
        self.h.select_skill(learner, "loop_tracing")
        step = self.h.ppo_step(learner, "loop_tracing", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        v_ans, _ = harness.solve_v22_task(harness.extract_question(step))
        self.h.hint(learner, iid)
        self.h.submit(learner, iid, v_ans)
        self.h.finish(learner, iid)

        ppo1 = harness.extract_ppo(self.h.get_state(learner, "loop_tracing"))
        self.assertTrue(ppo1["unverified_assisted"])

        verify = self.h.ppo_step(learner, "loop_tracing", action_id=1, format="single_choice")
        self.assertEqual(verify["mode"], "assessment")
        viid = harness.extract_interaction_id(verify)
        vv, _ = harness.solve_v22_task(harness.extract_question(verify))
        self.h.submit(learner, viid, vv)
        self.h.finish(learner, viid)

        ppo2 = harness.extract_ppo(self.h.get_state(learner, "loop_tracing"))
        self.assertFalse(ppo2["unverified_assisted"])

    def test_f11_self_explanation_submission_consolidates_permastore_floor(self) -> None:
        learner = self.h.create_profile("Permastore Floor")
        self.h.select_skill(learner, "range_boundaries")
        _complete_one_v22_question(self.h, learner, "range_boundaries", fmt="single_choice")
        pre = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")

        se = self.h.ppo_step(learner, "range_boundaries", action_id=5)
        iid = harness.extract_interaction_id(se)
        self.h.submit(
            learner,
            iid,
            "Start is inclusive and stop is exclusive, so values stay strictly below stop.",
            self_explanation="Start is inclusive and stop is exclusive, so values stay strictly below stop.",
        )
        self.h.finish(learner, iid)
        post = harness.extract_skill_summary(self.h.get_state(learner, "range_boundaries"), "range_boundaries")
        self.assertGreater(post["floor_b"], pre["floor_b"])
        self.assertGreater(post["p_30d"], pre["p_30d"])

    def test_f11_retry_and_reveal_emit_clean_events_and_close_correctly(self) -> None:
        learner = self.h.create_profile("Retry Reveal")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        _, bad_ans = harness.solve_v22_task(harness.extract_question(step))
        self.h.submit(learner, iid, bad_ans)
        self.h.retry(learner, iid)
        self.h.submit(learner, iid, bad_ans)
        rev = self.h.reveal(learner, iid)
        self.assertTrue(rev.get("solution") or rev.get("revealed_solution"))
        exp = self.h.get_export(learner)
        kinds = [e["kind"] for e in exp["events"]]
        self.assertIn("retry", kinds)
        self.assertIn("solution_reveal", kinds)


class TestF12HTTPAPI(unittest.TestCase):
    def test_f12_all_rest_endpoints_happy_path_status_and_json(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("REST All")
            self.assertIn("skills", h.get_state(learner))
            self.assertIn("curves_by_difficulty", h.get_curve(learner, "range_boundaries"))
            self.assertIn("events", h.get_export(learner))
            self.assertIn("learning_path", h.select_skill(learner, "range_boundaries"))
            step = h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            iid = harness.extract_interaction_id(step)
            _, bad_ans = harness.solve_v22_task(harness.extract_question(step))
            self.assertIn("hint", h.hint(learner, iid))
            h.submit(learner, iid, bad_ans)
            h.retry(learner, iid)
            h.submit(learner, iid, bad_ans)
            h.reveal(learner, iid)
            step2 = h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
            h.finish(learner, harness.extract_interaction_id(step2))

    def test_f12_security_headers_csp_nosniff_and_no_store(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            for route in ("/", "/api/state"):
                status, headers, _ = h.get(route)
                self.assertEqual(status, 200)
                self.assertEqual(headers.get("cache-control"), "no-store")
                self.assertEqual(headers.get("x-content-type-options"), "nosniff")
                csp = headers.get("content-security-policy", "")
                self.assertIn("frame-ancestors 'none'", csp)
                self.assertIn("script-src 'self'", csp)

    def test_f12_non_blocking_reads_during_slow_outbound_llm_call(self) -> None:
        with harness.LocalLLMStubServer(delay_seconds=0.45) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-e2e-nonblock",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                la = h.create_profile("Slow A")
                lb = h.create_profile("Fast B")
                errors: list[Exception] = []

                def slow_select() -> None:
                    try:
                        h.select_skill(la, "range_boundaries")
                    except Exception as exc:
                        errors.append(exc)

                t = threading.Thread(target=slow_select)
                t.start()
                time.sleep(0.08)
                t0 = time.monotonic()
                h.get_state(lb)
                h.get_curve(lb, "range_boundaries")
                elapsed = time.monotonic() - t0
                t.join(timeout=5.0)
                self.assertEqual(errors, [])
                self.assertLess(elapsed, 0.25)

    def test_f12_glass_box_telemetry_fields_complete_in_api_state(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Glass Box Telemetry")
            state = h.get_state(learner, "range_boundaries")
            ppo = harness.extract_ppo(state)
            for key in (
                "obs_vector_36",
                "valid_mask",
                "mask_reasons",
                "ensemble_probs",
                "per_seed_probs",
                "critic_value",
            ):
                self.assertIn(key, ppo)
            self.assertIn("skill_summaries", state)
            self.assertIn("token_audit", state)

    def test_f12_large_code_or_course_json_post_accepted_cleanly(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            learner = h.create_profile("Large Code")
            h.select_skill(learner, "range_boundaries")
            step = h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="code_completion")
            iid = harness.extract_interaction_id(step)
            valid_ans, _ = harness.solve_v22_task(harness.extract_question(step))
            padded = "# " + ("a" * 4800) + "\n" + valid_ans
            sub = h.submit(learner, iid, padded)
            res = sub["result"] if "result" in sub else sub["results"][-1]
            self.assertEqual(res["grading_status"], "resolved")
            self.assertTrue(res["full_success"])
            h.finish(learner, iid)


class TestF13FrontendUX(_SharedHarnessTestCase):
    def test_f13_css_matches_exact_dashboard_palette_tokens(self) -> None:
        status, headers, body = self.h.get("/style.css")
        self.assertEqual(status, 200)
        self.assertIn("text/css", headers.get("content-type", ""))
        tokens = harness.normalize_hex_tokens(body.decode("utf-8"))
        for expected in harness.PALETTE_HEX:
            self.assertIn(expected, tokens, msg=f"Missing dashboard palette token {expected}")

    def test_f13_css_has_zero_purple_or_linear_radial_gradients(self) -> None:
        _, _, css_body = self.h.get("/style.css")
        _, _, html_body = self.h.get("/")
        combined = (css_body.decode("utf-8") + "\n" + html_body.decode("utf-8")).lower()
        for pat in harness.FORBIDDEN_GRADIENT_PATTERNS:
            self.assertNotIn(pat, combined)

    def test_f13_html_and_js_expose_dag_workspace_and_glass_box_toggle(self) -> None:
        st_h, _, html_b = self.h.get("/")
        st_j, _, js_b = self.h.get("/app.js")
        self.assertEqual(st_h, 200)
        self.assertEqual(st_j, 200)
        html = html_b.decode("utf-8")
        js = js_b.decode("utf-8")
        self.assertIn("/style.css", html)
        self.assertIn("/app.js", html)
        self.assertIn("/favicon.svg", html)
        combined = (html + "\n" + js).lower()
        self.assertTrue("dag" in combined or "prerequisite" in combined)
        self.assertIn("glass", combined)

    def test_f13_frontend_js_renders_0_to_60d_svg_and_unassessed_guard(self) -> None:
        _, _, js_b = self.h.get("/app.js")
        js = js_b.decode("utf-8")
        low = js.lower()
        self.assertIn("unassessed", low)
        self.assertIn("svg", low)
        self.assertIn("floor", low)
        self.assertIn("/api/curve", js)

    def test_f13_active_workspace_state_persists_across_reload_bootstrap(self) -> None:
        learner = self.h.create_profile("Reload Recovery")
        self.h.select_skill(learner, "range_boundaries")
        step = self.h.ppo_step(learner, "range_boundaries", action_id=1, difficulty="easy", format="single_choice")
        iid = harness.extract_interaction_id(step)
        q = harness.extract_question(step)
        _, bad_ans = harness.solve_v22_task(q)
        self.h.submit(learner, iid, bad_ans)
        self.h.hint(learner, iid)

        reloaded = self.h.get_state(learner, "range_boundaries")
        self.assertIsNotNone(reloaded.get("active"))
        self.assertIsNotNone(reloaded.get("learning_path"))
        self.assertIsNotNone(reloaded.get("course_material"))
        self.assertGreaterEqual(len(reloaded["active"].get("hints", [])), 1)
        self.assertGreaterEqual(len(reloaded["active"].get("results", [])), 1)
        self.h.finish(learner, iid)


class TestF14ProposalDoc(unittest.TestCase):
    def _read_proposal(self) -> str:
        if not harness.PROPOSAL_PATH.exists():
            raise AssertionError(
                "Real 3/adaptive_web/PROPOSAL_PPO_LLM_ARCHITECTURE.md not yet available; cannot fake E2E target"
            )
        return harness.PROPOSAL_PATH.read_text(encoding="utf-8")

    def test_f14_proposal_file_exists_and_has_structured_sections(self) -> None:
        text = self._read_proposal()
        lines = text.splitlines()
        self.assertGreaterEqual(len(lines), 100)
        headings = [ln for ln in lines if ln.startswith("#")]
        self.assertGreaterEqual(len(headings), 4)

    def test_f14_proposal_compares_background_suggestion_vs_hard_ppo_control(self) -> None:
        low = self._read_proposal().lower()
        self.assertIn("suggestion", low)
        self.assertTrue("hard ppo" in low or "hard action control" in low or "gated option b" in low)
        self.assertTrue("alternative" in low or "comparison" in low)

    def test_f14_proposal_grounds_ux_in_all_five_research_systems(self) -> None:
        low = self._read_proposal().lower()
        for name in ("math academy", "execute program", "duolingo", "datacamp", "codecademy", "oatutor"):
            self.assertIn(name, low)

    def test_f14_proposal_documents_glass_box_inspector_and_provenance(self) -> None:
        low = self._read_proposal().lower()
        self.assertIn("glass-box", low)
        self.assertIn("learner_manual", low)
        self.assertIn("delayed_retention_ppo", low)

    def test_f14_proposal_matches_exact_model_and_ppo_dimensions(self) -> None:
        text = self._read_proposal()
        self.assertIn("monotonic_gru", text)
        self.assertIn("36", text)
        self.assertIn("47", text)
        self.assertIn("8", text)
        for sk in harness.PPO_SKILLS_ORDER:
            self.assertIn(sk, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
