"""Unit and HTTP tests for Ask the Tutor (`answer_learner_doubt`, `chat_doubt`, `POST /api/chat`)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

CODE_3_DIR = Path(__file__).resolve().parents[2]
E2E_DIR = Path(__file__).resolve().parent / "e2e"
for p in (str(CODE_3_DIR), str(E2E_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptive_web.config import Settings  # noqa: E402
from adaptive_web.llm_client import LLMClient  # noqa: E402
from adaptive_web.llm_pedagogy import PedagogyRealizer  # noqa: E402
import harness  # noqa: E402


class AnswerLearnerDoubtUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.realizer = PedagogyRealizer(LLMClient(Settings(llm_provider="mock")))

    def test_offline_fallback_branches_example_pitfall_concept(self) -> None:
        ex_ans = self.realizer.answer_learner_doubt("range_boundaries", "Show me the trace step example")
        self.assertEqual(ex_ans["source"], "fallback")
        self.assertIn("worked example", ex_ans["reply"].lower())
        self.assertTrue(ex_ans["code_example"])

        pit_ans = self.realizer.answer_learner_doubt("range_boundaries", "What common mistake or bug should I watch for?")
        self.assertEqual(pit_ans["source"], "fallback")
        self.assertIn("pitfalls", pit_ans["reply"].lower())
        self.assertEqual(pit_ans["code_example"], "")

        concept_ans = self.realizer.answer_learner_doubt("loop_tracing", "Why does a for-loop work this way?")
        self.assertEqual(concept_ans["source"], "fallback")
        self.assertIn("Loop tracing", concept_ans["reply"])

    def test_rejects_unknown_skill_or_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            self.realizer.answer_learner_doubt("unknown_skill", "Why?")
        for bad_q in ("", "   "):
            with self.assertRaises(ValueError):
                self.realizer.answer_learner_doubt("range_boundaries", bad_q)


class ChatEndpointHTTPTests(unittest.TestCase):
    def test_chat_offline_with_and_without_learner_validation_csrf_and_idempotency(self) -> None:
        with harness.AdaptiveWebHarness() as h:
            # Without learner ID
            anon = h.post(
                "/api/chat",
                {"skill": "range_boundaries", "message": "Why is stop excluded?"},
            )
            self.assertEqual(anon["status"], "ok")
            self.assertEqual(anon["skill"], "range_boundaries")
            self.assertEqual(anon["source"], "fallback")
            self.assertTrue(anon["reply"])

            # With learner + stored course_material + exact 1000-char boundary
            learner = h.create_profile("Chat Learner")
            h.select_skill(learner, "accumulator_initialization")
            exact_1000 = "x" * 1000
            req_id = "chat-idem-req-0001"
            res1 = h.post(
                "/api/chat",
                {
                    "learner": learner,
                    "skill": "accumulator_initialization",
                    "message": exact_1000,
                    "history": [{"role": "user", "text": "Hi"}],
                    "request_id": req_id,
                },
                auto_request_id=False,
            )
            self.assertEqual(res1["status"], "ok")
            self.assertEqual(res1["source"], "fallback")

            # Idempotent replay returns exact cached response
            res2 = h.post(
                "/api/chat",
                {
                    "learner": learner,
                    "skill": "accumulator_initialization",
                    "message": "different message ignored on replay",
                    "request_id": req_id,
                },
                auto_request_id=False,
            )
            self.assertEqual(res1, res2)

            # Validation boundaries -> 400
            for bad_payload in (
                {"skill": "range_boundaries", "message": ""},
                {"skill": "range_boundaries", "message": "   "},
                {"skill": "range_boundaries", "message": "a" * 1001},
                {"skill": "nonexistent_skill", "message": "Help?"},
                {"learner": "00000000-0000-0000-0000-000000000000", "skill": "range_boundaries", "message": "Help?"},
            ):
                st, _, _ = h.post_raw("/api/chat", bad_payload)
                self.assertEqual(st, 400, msg=f"Expected 400 for {bad_payload}")

            # CSRF enforcement -> 403
            st_no_csrf, _, _ = h.post_raw("/api/chat", {"skill": "range_boundaries", "message": "Help"}, csrf=False)
            self.assertEqual(st_no_csrf, 403)
            st_bad_csrf, _, _ = h.post_raw("/api/chat", {"skill": "range_boundaries", "message": "Help"}, csrf="bad-token")
            self.assertEqual(st_bad_csrf, 403)

    def test_chat_live_llm_stub_idempotency_and_error_fallback(self) -> None:
        def stub_responder(path: str, headers: dict, body: dict) -> dict:
            return {
                "reply": "Start is included and stop is strictly excluded.",
                "code_example": "print(list(range(1, 4)))",
            }

        with harness.LocalLLMStubServer(responder=stub_responder) as stub:
            with harness.AdaptiveWebHarness(
                env_overrides={
                    "LLM_PROVIDER": "openai",
                    "LLM_API_KEY": "sk-chat-stub",
                    "LLM_BASE_URL": stub.base_url,
                }
            ) as h:
                req_id = "chat-live-idem-001"
                r1 = h.post(
                    "/api/chat",
                    {
                        "skill": "range_boundaries",
                        "message": "Why does range(1, 4) stop at 3?",
                        "history": [{"role": "user", "text": "Hi"}, {"role": "assistant", "text": "Ask away!"}],
                        "request_id": req_id,
                    },
                    auto_request_id=False,
                )
                self.assertEqual(r1["source"], "llm")
                self.assertIn("excluded", r1["reply"])
                self.assertEqual(r1["code_example"], "print(list(range(1, 4)))")
                calls_before = len(stub.received_requests)

                # Idempotent replay must NOT make a second outbound LLM call
                r2 = h.post(
                    "/api/chat",
                    {
                        "skill": "range_boundaries",
                        "message": "Why does range(1, 4) stop at 3?",
                        "request_id": req_id,
                    },
                    auto_request_id=False,
                )
                self.assertEqual(r1, r2)
                self.assertEqual(len(stub.received_requests), calls_before)

                # Live LLM HTTP 500 error gracefully falls back to offline curriculum answer
                stub.status_code = 500
                r_err = h.post(
                    "/api/chat",
                    {
                        "skill": "range_boundaries",
                        "message": "What common mistake happens with range?",
                    },
                )
                self.assertEqual(r_err["source"], "fallback")
                self.assertTrue(r_err["reply"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
