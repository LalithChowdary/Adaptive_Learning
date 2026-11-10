import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from catalog import SKILLS, make_catalog
from simulator import Config, Student, DAY, POOLS, retained, learned
from replay import validate_events, prepare, materialize_query
from build_dataset import assignment, build_cohort, export_records
from build_fixtures import fixtures
from audit_dataset import audit
from loader import load_training_examples, load_evaluation_examples
from storage import digest, read_rows, writer, write_row, save_json


def clean_config(**kw):
    values = dict(tasks_per_skill=2, unresolved_probability=0, missing_time_probability=0,
                  missing_reference_probability=0, missing_assistance_probability=0)
    values.update(kw)
    return Config(**values)


class RetentionTests(unittest.TestCase):
    def test_half_life_is_half_strength(self):
        self.assertAlmostEqual(retained(.8, 4, 4), .4)

    def test_zero_gap_preserves_strength(self):
        self.assertEqual(retained(.6, 0, 10), .6)

    def test_power_law_agrees_at_half_life_but_differs_later(self):
        self.assertEqual(retained(.8, 4, 4, "power_law"), .4)
        self.assertGreater(retained(.8, 20, 4, "power_law"), retained(.8, 20, 4))

    def test_invalid_time_or_half_life_rejected(self):
        for args in ((.5, -1, 4), (.5, 1, 0), (2, 1, 4)):
            with self.assertRaises(ValueError): retained(*args)

    def test_learning_is_bounded_and_zero_dose_has_no_gain(self):
        self.assertEqual(learned(.4, .2, 0), .4)
        self.assertLess(learned(.4, .2, 1), 1)
        self.assertGreater(learned(.4, .2, 1), .4)

    def test_repeated_queries_do_not_accelerate_forgetting(self):
        s = Student("clock")
        s.advance(DAY)
        state = s.states.copy()
        first = s.strength("loop_tracing")
        for _ in range(10): self.assertEqual(first, s.strength("loop_tracing"))
        self.assertEqual(state, s.states)

    def test_partitioned_time_matches_single_advance(self):
        a, b = Student("same"), Student("same")
        a.advance(7 * DAY)
        for _ in range(7): b.advance(DAY)
        self.assertEqual(a.strength("range_boundaries"), b.strength("range_boundaries"))

    def test_instruction_does_not_update_unrelated_skill(self):
        a, b = Student("isolation"), Student("isolation")
        a.instruct("range_boundaries", "explain", "lesson")
        b.advance(45)
        self.assertGreater(a.strength("range_boundaries"), b.strength("range_boundaries"))
        self.assertEqual(a.strength("function_returns"), b.strength("function_returns"))

    def test_parameters_vary_between_learners_and_skills(self):
        a, b = Student("a"), Student("b")
        self.assertNotEqual(a.params, b.params)
        self.assertNotEqual(a.params["range_boundaries"], a.params["loop_tracing"])

    def test_invalid_config_rejected(self):
        for kw in ({"tasks_per_skill": 0}, {"mechanism": "unknown"}, {"unresolved_probability": 2}):
            with self.assertRaises(ValueError): Config(**kw)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.s = Student("evidence", clean_config())
        self.s.task("range_boundaries", forced_first=[False, False], forced_retry=[True, True])

    def test_assisted_success_is_not_unaided_target(self):
        record = prepare(self.s.events)["range_boundaries"]
        self.assertEqual(record["queries"][0]["target"]["full_success"], 0)
        self.assertTrue(self.s.events[-1]["full_success"])
        self.assertEqual(self.s.events[-2]["assistance"], "assisted")

    def test_unknown_assistance_excluded(self):
        s = Student("unknown", clean_config(missing_assistance_probability=1))
        s.task("range_boundaries", forced_first=[True, True])
        self.assertIsNone(prepare(s.events)["range_boundaries"]["queries"][0]["target"])

    def test_unresolved_grade_has_no_target(self):
        s = Student("unresolved", clean_config(unresolved_probability=1))
        s.task("range_boundaries")
        validate_events(s.events)
        self.assertIsNone(prepare(s.events)["range_boundaries"]["queries"][0]["target"])
        self.assertIsNone(s.events[-1]["score"])

    def test_partial_credit_is_not_full_success(self):
        s = Student("partial", clean_config())
        task = POOLS["range_boundaries", "practice"][0]
        s.emit("task", "range_boundaries", "q", task_id=task["task_id"], difficulty=task["difficulty"],
               task_type=task["task_type"], reference_seconds=task["reference_seconds"])
        s.attempt(task, "q", 1, forced_criteria=[True, False])
        validate_events(s.events)
        self.assertEqual(s.events[-1]["score"], .5)
        self.assertFalse(s.events[-1]["full_success"])

    def test_future_result_cannot_change_prior_input(self):
        s = Student("causal", clean_config())
        s.task("range_boundaries", forced_first=[True, True])
        record = prepare(s.events)["range_boundaries"]
        old = materialize_query(record, record["queries"][0])
        s.events[-1]["criteria"], s.events[-1]["score"], s.events[-1]["full_success"] = [False, False], 0, False
        changed = prepare(s.events)["range_boundaries"]
        self.assertEqual(old, materialize_query(changed, changed["queries"][0]))

    def test_first_query_has_empty_history(self):
        record = prepare(self.s.events)["range_boundaries"]
        self.assertEqual(materialize_query(record, record["queries"][0])["history"], [])

    def test_missing_duration_and_reference_stay_missing(self):
        s = Student("missing", clean_config(missing_time_probability=1, missing_reference_probability=1))
        s.task("range_boundaries", forced_first=[True, True])
        response = prepare(s.events)["range_boundaries"]["events"][1]
        self.assertIsNone(response["active_response_seconds"])
        self.assertIsNone(response["relative_response_time"])

    def test_raw_events_reject_hidden_field(self):
        self.s.events[0]["hidden_strength"] = .8
        with self.assertRaises(ValueError): validate_events(self.s.events)

    def test_model_features_reject_hidden_field(self):
        self.s.task("range_boundaries", forced_first=[True, True])
        rec = prepare(self.s.events)["range_boundaries"]
        rec["events"][0]["hidden_strength"] = .8
        with self.assertRaises(ValueError): materialize_query(rec, rec["queries"][-1])

    def test_query_rejects_identity_feature(self):
        rec = prepare(self.s.events)["range_boundaries"]
        rec["queries"][0]["input"]["learner_id"] = "leak"
        with self.assertRaises(ValueError): materialize_query(rec, rec["queries"][0])

    def test_duplicate_event_rejected(self):
        self.s.events[1]["event_id"] = self.s.events[0]["event_id"]
        with self.assertRaises(ValueError): validate_events(self.s.events)

    def test_false_independence_rejected(self):
        self.s.events[-2]["assistance"] = "independent"
        with self.assertRaises(ValueError): validate_events(self.s.events)

    def test_wrong_rubric_score_rejected(self):
        self.s.events[-1]["score"] = .5
        with self.assertRaises(ValueError): validate_events(self.s.events)

    def test_future_grade_availability_reordering_rejected(self):
        self.s.events[1]["available_at"] = self.s.events[-1]["available_at"] + 1
        with self.assertRaises(ValueError): validate_events(self.s.events)

    def test_new_task_resets_help_and_attempt(self):
        self.s.task("range_boundaries", forced_first=[True, True])
        response = self.s.events[-2]
        self.assertEqual((response["attempt_number"], response["hint_count"], response["assistance"]), (1, 0, "independent"))

    def test_reveal_does_not_create_perfect_hidden_knowledge(self):
        s = Student("reveal", clean_config())
        skill = "range_boundaries"
        s.states[skill] = (.02, s.clock)
        s.params[skill]["retained_help"] = 0
        task = POOLS[skill, "practice"][0]
        s.emit("task", skill, "q", task_id=task["task_id"], difficulty=task["difficulty"],
               task_type=task["task_type"], reference_seconds=task["reference_seconds"])
        _, grade = s.attempt(task, "q", 1, help_kind="reveal")
        self.assertTrue(grade["full_success"])
        self.assertLess(s.strength(skill), .02)

    def test_cross_course_reuses_same_skill_record(self):
        events = fixtures()["independent_and_cross_course"]
        records = prepare(events)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records["range_boundaries"]["queries"]), 2)

    def test_intervening_skill_does_not_erase_elapsed_gap(self):
        records = prepare(fixtures()["return_after_gap"])
        q = records["function_returns"]["queries"][-1]
        self.assertGreaterEqual(q["input"]["since_last_response_seconds"], 30 * DAY)

    def test_student_id_renaming_does_not_change_features(self):
        a = prepare(self.s.events)
        events = copy.deepcopy(self.s.events)
        for e in events: e["learner_id"] = "renamed"
        b = prepare(events)
        self.assertEqual(a["range_boundaries"]["events"], b["range_boundaries"]["events"])


class GenerationTests(unittest.TestCase):
    def test_no_repeated_task_ids_per_learner(self):
        s = Student("unique", clean_config()).practice("mixed")
        ids = [e["task_id"] for e in s.events if e["event_type"] == "task"]
        for delay in s.config.assessment_days:
            for skill in SKILLS:
                ids.append(s.assessment(delay, skill).events[0]["task_id"])
        self.assertEqual(len(ids), len(set(ids)))

    def test_higher_strength_increases_probability_under_matched_randomness(self):
        a, b = Student("matched", clean_config()), Student("matched", clean_config())
        a.states["range_boundaries"] = (.1, a.clock)
        b.states["range_boundaries"] = (.9, b.clock)
        a.task("range_boundaries", forced_first=[True, True])
        b.task("range_boundaries", forced_first=[True, True])
        self.assertTrue(all(x < y for x, y in zip(a.trace[0]["probabilities"], b.trace[0]["probabilities"])))

    def test_help_improves_current_probability_under_matched_randomness(self):
        a, b = Student("matched_help", clean_config()), Student("matched_help", clean_config())
        task = POOLS["range_boundaries", "practice"][0]
        a.attempt(task, "a", 1)
        b.attempt(task, "b", 1, help_kind="conceptual")
        self.assertTrue(all(x < y for x, y in zip(a.trace[-1]["probabilities"], b.trace[-1]["probabilities"])))

    def test_assessments_do_not_gain_strength_from_response(self):
        s = Student("no_probe_learning", clean_config()).practice("mixed")
        probe = s.assessment(7, "range_boundaries")
        self.assertEqual(probe.trace[-1]["strength_before"], probe.trace[-1]["strength_after"])

    def test_catalog_answers_match_trusted_snippet_execution(self):
        # Only snippets authored by catalog.py, never untrusted student code.
        tasks = make_catalog()
        self.assertEqual(len(tasks), 432)
        for task in tasks:
            for code, answer in zip(task["snippets"], task["reference_outputs"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output): exec(code, {})
                self.assertEqual(output.getvalue().strip(), answer)

    def test_three_controlled_histories_validate(self):
        histories = fixtures()
        self.assertEqual(len(histories), 3)
        for events in histories.values(): self.assertTrue(validate_events(events))

    def test_seed_reproduces_all_observations(self):
        a = Student("same", clean_config()).practice("mixed")
        b = Student("same", clean_config()).practice("mixed")
        self.assertEqual(a.events, b.events)
        self.assertEqual(a.private_record(), b.private_record())

    def test_schedule_has_exact_budget_per_skill(self):
        from collections import Counter
        for schedule in ("mixed", "blocked"):
            s = Student(schedule, clean_config()).practice(schedule)
            self.assertEqual(Counter(e["skill_id"] for e in s.events if e["event_type"] == "task"), Counter({k: 2 for k in SKILLS}))

    def test_split_is_disjoint_and_balanced(self):
        a = assignment(200, 1, "test")
        self.assertEqual(len({x[0] for x in a}), 200)
        for split, size in (("train", 160), ("validation", 20), ("test", 20)):
            self.assertEqual(sum(x[1] == split for x in a), size)
            self.assertEqual(sum(x[1] == split and x[2] == "mixed" for x in a), size // 2)

    def test_assessment_cannot_modify_training_state(self):
        s = Student("fork", clean_config()).practice("mixed")
        before = copy.deepcopy(s.private_record())
        a = s.assessment(7, "range_boundaries")
        s.assessment(1, "range_boundaries")
        b = s.assessment(7, "range_boundaries")
        self.assertEqual(before, s.private_record())
        self.assertEqual(a.events, b.events)

    def test_assessment_history_has_no_other_probe(self):
        s = Student("prefix", clean_config()).practice("mixed")
        probes = [dict(skill_id="range_boundaries", delay_days=d, events=s.assessment(d, "range_boundaries").events) for d in (1, 7, 30)]
        rec = next(r for r in export_records(s.events, probes, True) if r["skill_id"] == "range_boundaries")
        self.assertTrue(all(q["history_end"] == len(rec["events"]) for q in rec["queries"] if q["phase"] == "assessment"))


class DiskAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = build_cohort(Path(cls.temp.name), "testcohort", 20, clean_config())

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_full_audit_passes(self):
        self.assertEqual(audit(self.root)["status"], "passed")

    def test_training_loader_only_exposes_permitted_input(self):
        examples = list(load_training_examples(self.root))
        self.assertGreater(len(examples), 0)
        for x, y in examples:
            self.assertEqual(set(x), {"history", "target_task", "since_last_event_seconds", "since_last_response_seconds"})
            self.assertIn(y, (0, 1))

    def test_delayed_labels_absent_from_training_records(self):
        for r in read_rows(self.root / "model_data/train.jsonl.gz"):
            self.assertTrue(all(q["phase"] == "practice" for q in r["queries"]))

    def test_evaluation_loader_has_delayed_targets(self):
        self.assertEqual(len(list(load_evaluation_examples(self.root, "test"))), 36)

    def test_overwrite_is_refused(self):
        with self.assertRaises(FileExistsError): build_cohort(Path(self.temp.name), "testcohort", 20, clean_config())

    def test_complete_regeneration_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            second = build_cohort(Path(temp), "testcohort", 20, clean_config())
            first_manifest = json.loads((self.root / "manifest.json").read_text())
            second_manifest = json.loads((second / "manifest.json").read_text())
            self.assertEqual(first_manifest, second_manifest)

    def test_modified_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            import shutil
            copy_root = Path(temp) / "copy"
            shutil.copytree(self.root, copy_root)
            path = copy_root / "model_data/train.jsonl.gz"
            with path.open("ab") as f: f.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"): audit(copy_root)

    def test_causal_audit_rejects_future_features_even_with_new_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            import shutil
            copy_root = Path(temp) / "copy"
            shutil.copytree(self.root, copy_root)
            path = copy_root / "model_data/train.jsonl.gz"
            rows = list(read_rows(path))
            rows[0]["queries"][0]["history_end"] = len(rows[0]["events"])
            with writer(path) as f:
                for r in rows: write_row(f, r)
            manifest = json.loads((copy_root / "manifest.json").read_text())
            manifest["files"]["model_data/train.jsonl.gz"] = digest(path)
            save_json(copy_root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "causal replay"): audit(copy_root)


if __name__ == "__main__": unittest.main()
