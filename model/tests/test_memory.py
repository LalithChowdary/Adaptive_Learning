"""Regression tests for v2 causality, time anchoring, and recurrent equivalence."""
import copy
import csv
import gzip
import io
import os
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from memory.behavioral import evaluate_behavioral_suite
from memory.data import Partition, encode_record
from memory.features import DAY, EvidenceStream, HistorySummary
from memory.evaluation import groups, paired_intervals, probability_report, validation_losses
from memory.oracle import full_success_probability
from memory.recurrent import LearnerGRU
from memory.tabular import fit_tabular
from memory.training import make_batches, training_device
import run_v22


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

    def test_probability_report_includes_roc_auc_and_ten_padded_bins(self):
        queries = [
            SimpleNamespace(learner="l1", purpose="ordinary", delay_days=None, task_type="single_choice"),
            SimpleNamespace(learner="l2", purpose="ordinary", delay_days=None, task_type="single_choice"),
            SimpleNamespace(learner="l1", purpose="delayed", delay_days=1, task_type="output_prediction"),
            SimpleNamespace(learner="l2", purpose="delayed", delay_days=1, task_type="output_prediction"),
        ]
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        probs = np.array([0.15, 0.85, 0.05, 0.95], dtype=np.float64)
        partition = SimpleNamespace(queries=queries, labels=labels)
        masks = groups(queries)
        report = probability_report(partition, probs, masks)
        for group_name in ("all", "ordinary", "assessment"):
            self.assertEqual(len(report[group_name]["calibration"]), 10)
            for idx, entry in enumerate(report[group_name]["calibration"]):
                self.assertAlmostEqual(entry["lower"], idx / 10)
                self.assertAlmostEqual(entry["upper"], (idx + 1) / 10)
        self.assertAlmostEqual(report["all"]["roc_auc"], float(roc_auc_score(labels, probs)))
        self.assertEqual(report["ordinary"]["calibration"][0]["queries"], 0)
        self.assertIsNone(report["ordinary"]["calibration"][0]["mean_prediction"])
        self.assertIsNone(report["ordinary"]["calibration"][0]["observed_rate"])
        self.assertEqual(report["ordinary"]["calibration"][1]["queries"], 1)

        single_class = SimpleNamespace(queries=queries[:2], labels=np.array([1, 1], dtype=np.int64))
        single_report = probability_report(single_class, np.array([0.2, 0.8]), groups(queries[:2]))
        self.assertIsNone(single_report["all"]["roc_auc"])

    def test_paired_intervals_defaults_to_2000_draws_with_provenance_and_multiseed(self):
        queries = [
            SimpleNamespace(learner="l1", purpose="ordinary", delay_days=None, task_type="single_choice"),
            SimpleNamespace(learner="l2", purpose="ordinary", delay_days=None, task_type="single_choice"),
            SimpleNamespace(learner="l3", purpose="ordinary", delay_days=None, task_type="single_choice"),
            SimpleNamespace(learner="l4", purpose="ordinary", delay_days=None, task_type="single_choice"),
            SimpleNamespace(learner="l1", purpose="delayed", delay_days=1, task_type="single_choice"),
            SimpleNamespace(learner="l2", purpose="delayed", delay_days=1, task_type="single_choice"),
            SimpleNamespace(learner="l3", purpose="delayed", delay_days=1, task_type="single_choice"),
            SimpleNamespace(learner="l4", purpose="delayed", delay_days=1, task_type="single_choice"),
        ]
        labels = np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int64)
        partition = SimpleNamespace(queries=queries, labels=labels)
        masks = groups(queries)
        p_a = np.array([0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4])
        p_b1 = np.array([0.2, 0.8, 0.2, 0.8, 0.3, 0.7, 0.7, 0.3])
        p_b2 = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.8, 0.2])
        intervals = paired_intervals(
            partition, {"single_a": [p_a], "multi_b": [p_b1, p_b2]}, [("multi_b", "single_a")], masks, seed=20261201,
        )
        entry = intervals["ordinary"]["multi_b minus single_a"]
        self.assertEqual(entry["bootstrap_draws"], 2000)
        self.assertEqual(entry["bootstrap_seed"], 20261201)
        self.assertLessEqual(entry["lower_95"], entry["delta_log_loss"])
        self.assertLessEqual(entry["delta_log_loss"], entry["upper_95"])
        loss_a = -np.where(labels == 1, np.log(p_a), np.log1p(-p_a))
        loss_b1 = -np.where(labels == 1, np.log(p_b1), np.log1p(-p_b1))
        loss_b2 = -np.where(labels == 1, np.log(p_b2), np.log1p(-p_b2))
        expected = float((0.5 * (loss_b1 + loss_b2) - loss_a)[masks["ordinary"]].mean())
        self.assertAlmostEqual(entry["delta_log_loss"], expected)


def _make_record(learner_id, *, ordinary_label=0, followup_label=1, assessment_label=1):
    ev = [
        event("task", None),
        *attempt_events(),
        event("help_request", DAY),
        event("hint", DAY, hint_type="conceptual"),
        event("retry", DAY, attempt_number=2),
    ]
    queries = [
        query(f"{learner_id}-ord", 0, None),
        {
            "query_id": f"{learner_id}-fol", "history_end": 3, "phase": "practice", "purpose": "followup",
            "target": {"full_success": followup_label},
            "input": {"target_task": TASK, "since_last_event_seconds": DAY, "since_last_response_seconds": DAY},
        },
    ]
    queries[0]["target"]["full_success"] = ordinary_label
    for delay in (1, 7, 30):
        probe = query(f"{learner_id}-probe-{delay}", len(ev), delay * DAY)
        probe.update(phase="assessment", purpose="delayed", delay_days=delay)
        probe["target"]["full_success"] = assessment_label
        queries.append(probe)
    return {"learner_id": learner_id, "skill_id": "loop_tracing", "events": ev, "queries": queries}


class HarnessTests(unittest.TestCase):
    def test_summarize_includes_min_max_and_handles_single_and_none(self):
        multi = run_v22.summarize([0.5, 0.7, 0.6])
        self.assertAlmostEqual(multi["mean"], 0.6)
        self.assertAlmostEqual(multi["sd"], 0.1)
        self.assertAlmostEqual(multi["min"], 0.5)
        self.assertAlmostEqual(multi["max"], 0.7)

        single = run_v22.summarize([0.42])
        self.assertAlmostEqual(single["mean"], 0.42)
        self.assertIsNone(single["sd"])
        self.assertAlmostEqual(single["min"], 0.42)
        self.assertAlmostEqual(single["max"], 0.42)

        empty = run_v22.summarize([None])
        self.assertEqual(empty, {"mean": None, "sd": None, "min": None, "max": None})

    def test_evaluation_summary_has_learners_roc_auc_and_seed_averaged_calibration(self):
        queries = [
            SimpleNamespace(learner="l1", purpose="ordinary", delay_days=None, task_type="single_choice"),
            SimpleNamespace(learner="l2", purpose="ordinary", delay_days=None, task_type="single_choice"),
            SimpleNamespace(learner="l1", purpose="delayed", delay_days=1, task_type="single_choice"),
            SimpleNamespace(learner="l2", purpose="delayed", delay_days=7, task_type="single_choice"),
            SimpleNamespace(learner="l1", purpose="delayed", delay_days=30, task_type="single_choice"),
            SimpleNamespace(learner="l2", purpose="delayed", delay_days=30, task_type="single_choice"),
        ]
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        partition = SimpleNamespace(queries=queries, labels=labels)
        masks = groups(queries)
        p1 = np.array([0.15, 0.85, 0.25, 0.75, 0.35, 0.65])
        p2 = np.array([0.35, 0.85, 0.25, 0.75, 0.35, 0.65])
        reports = [probability_report(partition, p, masks) for p in (p1, p2)]
        summary = run_v22.evaluation_summary(reports, partition=partition, scores=[p1, p2], masks=masks)
        self.assertEqual(summary["ordinary"]["learners"], 2)
        self.assertEqual(summary["assessment"]["learners"], 2)
        for stat in ("mean", "sd", "min", "max"):
            self.assertIn(stat, summary["ordinary"]["roc_auc"])
            self.assertIn(stat, summary["delayed_equal_horizon_log_loss"])
        # (0.15 + 0.35) / 2 = 0.25 -> bin 2 ([0.2, 0.3)), not bin 1 ([0.1, 0.2))
        self.assertEqual(len(summary["ordinary"]["calibration"]), 10)
        self.assertEqual(len(summary["assessment"]["calibration"]), 10)
        self.assertEqual(summary["ordinary"]["calibration"][2]["queries"], 1)
        self.assertEqual(summary["ordinary"]["calibration"][1]["queries"], 0)

    def test_fit_tabular_seed_and_multiseed_history_boosting_pipeline(self):
        train_records = [
            encode_record(_make_record("tr1", ordinary_label=0, followup_label=1, assessment_label=0), training=True, include_training_assessments=True),
            encode_record(_make_record("tr2", ordinary_label=1, followup_label=0, assessment_label=1), training=True, include_training_assessments=True),
        ]
        val_records = [
            encode_record(_make_record("va1", ordinary_label=0, followup_label=1, assessment_label=0), training=False),
            encode_record(_make_record("va2", ordinary_label=1, followup_label=0, assessment_label=1), training=False),
        ]
        train = Partition(train_records)
        validation = Partition(val_records)
        masks = groups(validation.queries)

        _, tab_fit = fit_tabular("history_boosting", train, validation, masks, seed=20261201, balanced=True)
        self.assertEqual(tab_fit["seed"], 20261201)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            args = run_v22.parse_args([
                "--output", str(tmpdir), "--mode", "smoke", "--device", "cpu",
                "--seeds", "20261201", "20261202",
            ])
            run = {
                "status": "running", "mode": args.mode, "run_classification": args.run_classification,
                "dataset_version": run_v22.VERSION, "reused_from": None, "recurrent_device": "cpu",
                "mps_cpu_fallback_enabled": False, "seeds": args.seeds, "epochs": args.epochs,
                "batch_size": args.batch_size, "bootstrap_draws": args.bootstrap_draws,
                "training": {"base_rate": float(train.labels.mean())},
                "models": {}, "evaluation": {},
            }
            device = torch.device("cpu")
            models, batches = run_v22.fit_models(args, run, train, validation, device)
            run_v22.select_model(run)
            with patch(
                "memory.oracle.oracle_predictions",
                side_effect=lambda dataset, cohort, split, part: np.full(len(part.labels), 0.5),
            ):
                run_v22.evaluate(args, run, models, validation, batches, "pilot", "validation")

            self.assertTrue((tmpdir / "task_logistic" / "model.joblib").exists())
            self.assertTrue((tmpdir / "history_logistic" / "model.joblib").exists())
            self.assertTrue((tmpdir / "history_boosting" / "seed_20261201" / "model.joblib").exists())
            self.assertTrue((tmpdir / "history_boosting" / "seed_20261202" / "model.joblib").exists())
            self.assertEqual([f["seed"] for f in run["models"]["history_boosting"]], [20261201, 20261202])
            self.assertTrue((tmpdir / run["selected_checkpoint"]).exists())

            eval_val = run["evaluation"]["pilot_validation"]
            self.assertEqual(
                sorted(eval_val["by_seed"]["history_boosting"].keys()),
                ["history_boosting_seed_20261201", "history_boosting_seed_20261202"],
            )
            self.assertEqual(len(eval_val["summary"]["history_boosting"]["ordinary"]["calibration"]), 10)
            self.assertEqual(len(eval_val["summary"]["history_boosting"]["assessment"]["calibration"]), 10)
            with gzip.open(tmpdir / "pilot_validation_predictions.csv.gz", "rt", encoding="utf-8") as fh:
                header = next(csv.reader(fh))
            self.assertIn("history_boosting_seed_20261201", header)
            self.assertIn("history_boosting_seed_20261202", header)
            first_contrast = next(iter(eval_val["paired_intervals"]["ordinary"].values()))
            self.assertEqual(first_contrast["bootstrap_draws"], 2000)
            self.assertEqual(first_contrast["bootstrap_seed"], 20261201)

    def test_cli_parse_args_seeds_and_classification(self):
        self.assertEqual(run_v22.parse_args(["--output", "out", "--mode", "smoke"]).seeds, [20261120])
        self.assertEqual(
            run_v22.parse_args(["--output", "out", "--mode", "smoke", "--seeds", "11", "22"]).seeds,
            [11, 22],
        )
        self.assertEqual(
            run_v22.parse_args(["--output", "out", "--mode", "full"]).seeds,
            [20261120, 20261121, 20261122],
        )
        full_5 = run_v22.parse_args([
            "--output", "out", "--mode", "full",
            "--seeds", "20261201", "20261202", "20261203", "20261204", "20261205",
        ])
        self.assertEqual(full_5.seeds, [20261201, 20261202, 20261203, 20261204, 20261205])
        self.assertEqual(full_5.bootstrap_draws, 2000)
        self.assertEqual(full_5.run_classification, "pre_registered")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_v22.parse_args(["--output", "out", "--bootstrap-draws", "0"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_v22.parse_args(["--output", "out", "--seeds", "-1"])


class RecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = training_device(os.environ.get("MEMORY_TEST_DEVICE", "cpu"))

    def test_batched_replay_matches_incremental_including_repeated_queries(self):
        encoded = encode_record(record())
        batch = make_batches(Partition([encoded]), 1, self.device)[0]
        tokens, gaps, positions, tasks, query_gaps = batch.inputs
        for mode in (
            "time_input",
            "time_decay",
            "kt_decay",
            "monotonic",
            "time_aware_transformer",
            "monotonic_transformer",
        ):
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
                    torch.testing.assert_close(first, logits[:, index], atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(first, second)
                    torch.testing.assert_close(state, saved)

    def test_monotonic_and_kt_decay_satisfy_all_behavioral_sanity_checks(self):
        for mode in ("kt_decay", "monotonic", "monotonic_transformer"):
            model = LearnerGRU(mode=mode).to(self.device)
            report = evaluate_behavioral_suite(model)
            self.assertTrue(report["all_passed"], f"{mode} failed checks: {report['checks']}")

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
