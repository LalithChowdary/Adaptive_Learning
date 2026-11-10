"""Regression tests for v2 causality, time anchoring, and recurrent equivalence."""
import copy
import os
import unittest

import numpy as np
import torch

from memory.data import Partition, encode_record
from memory.features import DAY, EvidenceStream, HistorySummary
from memory.evaluation import validation_losses
from memory.oracle import full_success_probability
from memory.recurrent import LearnerGRU
from memory.training import make_batches, training_device


TASK = {
    "task_type": "single_choice", "difficulty": "medium", "reference_seconds": 30,
    "snippet_count": 1, "option_count": 4, "criterion_count": 1,
    "credit_weights": [1], "penalty_weights": [],
}


def event(kind, gap, **fields):
    return {
        "event_type": kind, "gap_seconds": gap, "task": TASK, "attempt_number": None,
        "assistance": None, "hint_count": None, "answer_revealed": None,
        "active_response_seconds": None, "relative_response_time": None, **fields,
    }


def attempt_events():
    evidence = dict(attempt_number=1, assistance="independent", hint_count=0, answer_revealed=False)
    return [
        event("response", 60, active_response_seconds=60, relative_response_time=2, **evidence),
        event("grade", 1, grading_status="resolved", score=0.0, full_success=False, **evidence),
    ]


def query(identifier, cursor, gap, *, last_response=None):
    return {
        "query_id": identifier, "history_end": cursor, "phase": "practice", "purpose": "ordinary",
        "target": {"full_success": 0},
        "input": {
            "target_task": TASK, "since_last_event_seconds": gap,
            "since_last_response_seconds": last_response,
        },
    }


def record():
    return {
        "learner_id": "test-learner", "skill_id": "loop_tracing",
        "events": [
            event("task", None), *attempt_events(), event("help_request", DAY),
            event("hint", DAY, hint_type="conceptual"), event("retry", DAY, attempt_number=2),
        ],
        "queries": [
            query("first", 0, None),
            query("pending", 2, 0, last_response=0),
            query("later", 6, 4 * DAY, last_response=7 * DAY + 1),
        ],
    }


class EncodingTests(unittest.TestCase):
    def test_causal_pairing_and_elapsed_time_through_skipped_events(self):
        encoded = encode_record(record(), training=True)
        np.testing.assert_array_equal(encoded.positions, [0, 0, 2])
        np.testing.assert_allclose(encoded.gaps, [0, 2])
        np.testing.assert_allclose(encoded.query_gaps, [0, 0, 5])

    def test_future_outcome_changes_neither_early_inputs_nor_labels(self):
        original = record()
        changed = copy.deepcopy(original)
        changed["events"][2].update(full_success=True, score=1.0)
        before, after = encode_record(original), encode_record(changed)
        np.testing.assert_array_equal(before.tasks, after.tasks)
        np.testing.assert_array_equal(before.summaries[:2], after.summaries[:2])
        np.testing.assert_array_equal(before.labels, after.labels)
        self.assertFalse(np.array_equal(before.tokens, after.tokens))

    def test_unknown_grade_is_distinct_from_failure(self):
        stream = EvidenceStream()
        stream.push(event("task", None))
        response, grade = attempt_events()
        self.assertIsNone(stream.push(response))
        grade.update(grading_status="unresolved", full_success=None, score=None)
        unknown = stream.push(grade)
        self.assertEqual(unknown.kind, "attempt")
        self.assertEqual(unknown.fields["active_response_seconds"], 60)
        self.assertIsNone(unknown.fields["score"])
        summary = HistorySummary()
        summary.update(unknown)
        self.assertEqual(summary.unaided_success.count, 0)
        self.assertEqual(summary.unresolved, 1)

    def test_mismatched_grade_fails_instead_of_guessing(self):
        broken = record()
        broken["events"][2]["attempt_number"] = 2
        with self.assertRaisesRegex(ValueError, "Cannot pair"):
            encode_record(broken)

    def test_identifiers_do_not_enter_features(self):
        original = record()
        changed = copy.deepcopy(original)
        changed["learner_id"] = "another-person"
        changed["skill_id"] = "another-skill"
        a, b = encode_record(original), encode_record(changed)
        for field in ("tokens", "gaps", "tasks", "query_gaps", "summaries"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))

    def test_training_rejects_assessment_queries(self):
        invalid = record()
        invalid["queries"][0]["phase"] = "assessment"
        with self.assertRaisesRegex(ValueError, "Assessment query"):
            encode_record(invalid, training=True)

    def test_v21_training_probes_share_the_frozen_practice_state(self):
        example = record()
        for delay in (1, 7, 30):
            probe = query(f"probe-{delay}", len(example["events"]), delay * DAY)
            probe.update(phase="assessment", purpose="delayed", delay_days=delay)
            example["queries"].append(probe)
        encoded = encode_record(example, training=True, include_training_assessments=True)
        np.testing.assert_array_equal(encoded.positions[-3:], [2, 2, 2])
        np.testing.assert_array_equal(encoded.query_gaps[-3:], [2, 8, 31])
        example["queries"][-1]["history_end"] -= 1
        with self.assertRaisesRegex(ValueError, "frozen practice prefix"):
            encode_record(example, training=True, include_training_assessments=True)


class EvaluationTests(unittest.TestCase):
    def test_validation_score_weights_practice_and_delayed_equally(self):
        labels = np.array([1, 1, 1, 1, 1])
        probabilities = np.array([0.8, 0.8, 0.4, 0.4, 0.4])
        masks = {
            "practice": np.array([True, True, False, False, False]),
            **{f"day_{delay}": np.arange(5) == index for index, delay in enumerate((1, 7, 30), 2)},
        }
        result = validation_losses(labels, probabilities, masks, balanced=True)
        self.assertAlmostEqual(result["validation_selection_log_loss"], -(np.log(0.8) + np.log(0.4)) / 2)

    def test_oracle_probability_respects_format_and_strength(self):
        task = {"task_type": "single_choice", "criteria": [None], "options": [None] * 4}
        low = full_success_probability(0.0, 0.6, task)
        high = full_success_probability(1.0, 0.6, task)
        self.assertTrue(0.25 < low < high < 1.0)
        with_two_checks = {"task_type": "output_prediction", "criteria": [None] * 2}
        with_one_check = dict(with_two_checks, criteria=[None])
        self.assertLess(
            full_success_probability(0.5, 0.6, with_two_checks),
            full_success_probability(0.5, 0.6, with_one_check),
        )


class RecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = training_device(os.environ.get("MEMORY_TEST_DEVICE", "cpu"))

    def test_batched_replay_matches_incremental_including_repeated_queries(self):
        encoded = encode_record(record())
        batch = make_batches(Partition([encoded]), 1, self.device)[0]
        tokens, gaps, positions, tasks, query_gaps = batch.inputs
        for mode in ("time_input", "time_decay"):
            model = LearnerGRU(mode=mode).to(self.device)
            with torch.no_grad():
                logits = model(*batch.inputs)
                state = model.initial_state()
                states = [state]
                for step in range(tokens.shape[1]):
                    state = model.observe(state, tokens[:, step], gaps[:, step])
                    states.append(state)
                for index in reversed(range(tasks.shape[1])):
                    state = states[int(positions[0, index].item())]
                    saved = state.clone()
                    first = model.predict_logit(state, tasks[:, index], query_gaps[:, index])
                    second = model.predict_logit(state, tasks[:, index], query_gaps[:, index])
                    torch.testing.assert_close(first, logits[:, index])
                    torch.testing.assert_close(first, second)
                    torch.testing.assert_close(state, saved)

    def test_decay_is_identity_at_zero_and_converges_toward_prior(self):
        model = LearnerGRU(mode="time_decay").to(self.device)
        state = torch.ones((1, model.hidden_size), device=self.device)
        zero, _ = model._at_time(state, torch.zeros(1, device=self.device))
        long, _ = model._at_time(state, torch.full((1,), 100_000.0, device=self.device))
        torch.testing.assert_close(zero, state)
        torch.testing.assert_close(long, model.initial_state())

    def test_decay_and_recurrent_parameters_receive_gradients(self):
        batch = make_batches(Partition([encode_record(record())]), 1, self.device)[0]
        model = LearnerGRU(mode="time_decay").to(self.device)
        model(*batch.inputs).square().sum().backward()
        for parameter in (model.rate.bias, model.cell.weight_hh):
            gradient = parameter.grad
            self.assertEqual(gradient.device.type, self.device.type)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
