"""Unit and contract tests for LLM orchestration and pedagogy realization."""
from pathlib import Path
import sys
import unittest

CODE_3_DIR = Path(__file__).resolve().parents[2]
if str(CODE_3_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_3_DIR))

from adaptive_web.catalogue_bridge import validate_task_schema
from adaptive_web.config import Settings
from adaptive_web.llm_client import LLMClient
from adaptive_web.llm_pedagogy import PedagogyRealizer


class PedagogyRealizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(llm_provider="mock")
        self.client = LLMClient(self.settings)
        self.realizer = PedagogyRealizer(self.client)

    def test_curate_learning_path_unassessed_vs_fragile(self) -> None:
        # Unassessed learner
        p_unassessed = self.realizer.curate_learning_path("range_boundaries", None)
        self.assertEqual(len(p_unassessed["steps"]), 3)
        self.assertEqual(p_unassessed["focus_skill"], "range_boundaries")
        self.assertEqual(p_unassessed["steps"][0]["phase"], "learn")
        self.assertEqual(p_unassessed["steps"][0]["recommended_action"], "explain_concept")

        # Fragile learner (high P_med, low P_30d)
        fragile_mem = {"assessed": True, "p_med": 0.75, "p_30d": 0.40, "floor_b": 0.32}
        p_fragile = self.realizer.curate_learning_path("range_boundaries", fragile_mem)
        self.assertEqual(len(p_fragile["steps"]), 3)
        self.assertEqual(p_fragile["steps"][0]["recommended_action"], "show_worked_example")
        self.assertEqual(p_fragile["steps"][2]["recommended_action"], "request_self_explanation")

    def test_generate_course_material_structure(self) -> None:
        mat = self.realizer.generate_course_material("loop_tracing")
        self.assertEqual(mat["skill"], "loop_tracing")
        self.assertTrue(mat["title"])
        self.assertTrue(mat["concept_explanation"])
        self.assertIn("worked_example", mat)
        self.assertIn("code", mat["worked_example"])
        self.assertIn("subgoals", mat["worked_example"])

    def test_generate_question_valid_schema(self) -> None:
        for fmt in ("single_choice", "multiple_select", "output_prediction", "fill_in_the_blank", "code_completion"):
            with self.subTest(fmt=fmt):
                q = self.realizer.generate_question("conditional_evaluation", "medium", fmt)
                self.assertTrue(validate_task_schema(q))
                self.assertEqual(q["skill_id"], "conditional_evaluation")
                self.assertEqual(q["task_type"], fmt)
                self.assertEqual(q["difficulty"], "medium")
                self.assertGreaterEqual(len(q["hints"]), 3)

    def test_generate_hints_progressive_leakage(self) -> None:
        q = self.realizer.generate_question("accumulator_initialization", "easy", "output_prediction")
        h1 = self.realizer.generate_hint(q, "0", 1)
        h2 = self.realizer.generate_hint(q, "0", 2)
        h3 = self.realizer.generate_hint(q, "0", 3)

        self.assertEqual(h1["hint_kind"], "conceptual")
        self.assertEqual(h2["hint_kind"], "targeted")
        self.assertEqual(h3["hint_kind"], "scaffold")

        self.assertLess(h1["leakage_l_t"], h2["leakage_l_t"])
        self.assertLess(h2["leakage_l_t"], h3["leakage_l_t"])
        self.assertTrue(all(0.0 < h["leakage_l_t"] < 1.0 for h in (h1, h2, h3)))

    def test_evaluate_self_explanation_scoring(self) -> None:
        prompt = "Explain why range(2, 7, 2) excludes the stop value 7."
        # Good explanation
        good = self.realizer.evaluate_self_explanation(
            "range_boundaries", None, prompt, "The stop parameter is strictly excluded, so values above or equal to stop do not run."
        )
        self.assertTrue(good["full_success"])
        self.assertGreaterEqual(good["score"], 0.8)

        # Poor explanation
        poor = self.realizer.evaluate_self_explanation(
            "range_boundaries", None, prompt, "I guessed number 7."
        )
        self.assertFalse(poor["full_success"])
        self.assertLessEqual(poor["score"], 0.5)

    def test_realize_ppo_turns(self) -> None:
        # Lesson
        t_concept = self.realizer.realize_ppo_turn("explain_concept", "list_indexing", "easy")
        self.assertEqual(t_concept["kind"], "lesson")
        self.assertEqual(t_concept["primed_mode"], "concept")

        # Worked example
        t_example = self.realizer.realize_ppo_turn("show_worked_example", "list_indexing", "medium")
        self.assertEqual(t_example["kind"], "lesson")
        self.assertEqual(t_example["primed_mode"], "example")

        # Self-explanation
        t_se = self.realizer.realize_ppo_turn("request_self_explanation", "list_indexing", "medium")
        self.assertEqual(t_se["kind"], "question")
        self.assertEqual(t_se["mode"], "assessment")
        self.assertTrue(t_se.get("self_explanation_prompt"))

        # Practice question
        t_q = self.realizer.realize_ppo_turn("give_harder_question", "list_indexing", "hard")
        self.assertEqual(t_q["kind"], "question")
        self.assertEqual(t_q["difficulty"], "hard")


if __name__ == "__main__":
    unittest.main()
