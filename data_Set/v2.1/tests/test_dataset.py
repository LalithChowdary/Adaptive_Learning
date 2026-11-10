import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from collections import Counter

from catalog import SKILLS, FORMATS, LESSONS, make_catalog, grade
from simulator import Config, Student, TASKS, TASK_INDEX, POOLS, DAY, retained, learned
from replay import validate_events, prepare, materialize_query
from build_dataset import build_cohort, export_records, assignment
from build_fixtures import fixtures, task_for, clean_config
from storage import read_rows, writer, write_row, digest, save_json
from audit_dataset import audit
from loader import load_training_examples, load_evaluation_examples


class CatalogTests(unittest.TestCase):
    def test_catalog_covers_all_skill_format_difficulty_pool_combinations(self):
        self.assertEqual(len(TASKS),2160)
        self.assertEqual(len({t['task_id'] for t in TASKS}),len(TASKS))
        self.assertEqual(len({(t['skill_id'],t['task_type'],t['difficulty'],t['pool']) for t in TASKS}),180)

    def test_snippets_and_criteria_are_variable(self):
        self.assertEqual({len(t['snippets']) for t in TASKS},{0,1,2,3})
        self.assertEqual({len(t['criteria']) for t in TASKS},{1,2,3,4,5,6})
        self.assertTrue(any(len({c['weight'] for c in t['criteria']})>1 for t in TASKS))

    def test_weight_normalization_and_positive_penalty_totals(self):
        for t in TASKS:
            self.assertAlmostEqual(sum(c['weight'] for c in t['criteria'] if c['kind']!='incorrect_selection_penalty'),1.)
            if t['task_type']=='multiple_select': self.assertAlmostEqual(sum(c['weight'] for c in t['criteria'] if c['kind']=='incorrect_selection_penalty'),1.)

    def test_single_choice_exact_scoring(self):
        for t in TASKS:
            if t['task_type']=='single_choice':
                for o in t['options']:
                    g=grade(t,o['id']); self.assertEqual(g['score'],float(o['id']==t['answer_key']))

    def test_multiple_select_all_and_none_earn_zero(self):
        for t in TASKS:
            if t['task_type']=='multiple_select':
                for response in ([],[o['id'] for o in t['options']]):
                    self.assertEqual(grade(t,response)['score'],0); self.assertFalse(grade(t,response)['full_success'])

    def test_multiple_select_wrong_selection_reduces_credit(self):
        t=task_for('multiple_select'); key=t['answer_key']; wrong=next(o['id'] for o in t['options'] if o['id'] not in key)
        self.assertLess(grade(t,key+[wrong])['score'],grade(t,key)['score'])
        self.assertTrue(grade(t,key)['full_success']); self.assertFalse(grade(t,key[:1])['full_success'])

    def test_weighted_partial_credit_does_not_imply_full_success(self):
        t=task_for(); g=grade(t,[True,False]); self.assertAlmostEqual(g['score'],1/3,places=9); self.assertFalse(g['full_success'])

    def test_invalid_grade_evidence_rejected(self):
        with self.assertRaises(ValueError): grade(task_for(),[True])
        with self.assertRaises(ValueError): grade(task_for('multiple_select'),['unknown'])
        with self.assertRaises(ValueError): grade(task_for('multiple_select'),['o0','o0'])

    def test_output_keys_match_trusted_reference_execution(self):
        for t in TASKS:
            if t['task_type']=='output_prediction':
                for code,answer in zip(t['snippets'],t['answer_key']):
                    out=io.StringIO()
                    with contextlib.redirect_stdout(out): exec(code,{})
                    self.assertEqual(out.getvalue().strip(),answer)
            if t['task_type']=='single_choice' and t['snippets']:
                out=io.StringIO()
                with contextlib.redirect_stdout(out): exec(t['snippets'][0],{})
                self.assertEqual(out.getvalue().strip(),next(o['text'] for o in t['options'] if o['id']==t['answer_key']))

    def test_fill_keys_reconstruct_reference_code(self):
        for t in TASKS:
            if t['task_type']=='fill_in_the_blank':
                code=t['snippets'][0]
                for j,answer in enumerate(t['answer_key']): code=code.replace(f'__BLANK_{j+1}__',answer)
                self.assertEqual(code,t['reference_solution'])
                with contextlib.redirect_stdout(io.StringIO()): exec(code,{})

    def test_completion_reference_passes_every_authored_case(self):
        for t in TASKS:
            if t['task_type']=='code_completion':
                namespace={}; exec(t['reference_solution'],namespace)
                for case in t['tests']: self.assertEqual(namespace['solve'](case['input']),case['expected'])


class SimulationTests(unittest.TestCase):
    def test_no_routine_instruction_and_question_first(self):
        s=Student('default',clean_config()).practice('mixed')
        self.assertEqual(s.events[0]['event_type'],'task')
        self.assertFalse(any(e['event_type']=='instruction' for e in s.events))
        validate_events(s.events)

    def test_initial_strength_varies_and_is_not_assumed_zero(self):
        a,b=Student('a'),Student('b')
        self.assertNotEqual(a.params,b.params)
        self.assertTrue(all(.02<=p['initial_strength']<=.95 for p in a.params.values()))

    def test_three_hints_and_two_retries_are_chronological(self):
        events=fixtures()['fixture_three_hints']['events']; first=events[0]['interaction_id']
        rows=[e for e in events if e['interaction_id']==first]
        self.assertEqual([e['hint_number'] for e in rows if e['event_type']=='hint'],[1,2,3])
        self.assertEqual([e['attempt_number'] for e in rows if e['event_type']=='retry'],[2,3])
        self.assertEqual([e['hint_count'] for e in rows if e['event_type']=='response'],[0,2,3])

    def test_four_hint_configuration(self):
        s=Student('four',clean_config(max_hints=4)); t=task_for(); bad=[False]*len(t['criteria'])
        s.task(t['skill_id'],task=t,forced_attempts=[bad,bad,bad],force_hints=[2,2])
        validate_events(s.events); self.assertEqual(len([e for e in s.events if e['event_type']=='hint']),4)

    def test_zero_help_and_retry_caps(self):
        s=Student('zero',clean_config(max_hints=0,max_retries=0)).practice('mixed')
        self.assertFalse(any(e['event_type'] in ('hint','retry') for e in s.events)); validate_events(s.events)

    def test_each_assisted_practice_has_fresh_unaided_followup(self):
        s=Student('followup',clean_config(tasks_per_skill=5)).practice('mixed')
        helped={e['interaction_id'] for e in s.events if e['event_type']=='hint'}
        follows=[e for e in s.events if e['event_type']=='task' and e['purpose']=='followup']
        self.assertEqual({e['parent_interaction_id'] for e in follows},helped)
        for t in follows:
            rows=[e for e in s.events if e['interaction_id']==t['interaction_id']]
            self.assertEqual([e['event_type'] for e in rows],['task','response','grade'])
            self.assertEqual(rows[1]['hint_count'],0)

    def test_requested_lesson_has_no_direct_learning_gain(self):
        a,b=Student('lesson'),Student('lesson'); skill='range_boundaries'
        a.requested_lesson(skill); b.advance(45)
        self.assertEqual(a.states,b.states); self.assertEqual(a.strength(skill),b.strength(skill)); validate_events(a.events)

    def test_requested_lesson_content_exists(self):
        s=Student('lesson_content'); s.requested_lesson('range_boundaries')
        self.assertIn(s.events[-1]['content_ref'],LESSONS)
        self.assertTrue(LESSONS[s.events[-1]['content_ref']])

    def test_hint_event_has_no_retained_gain(self):
        a=Student('hint',clean_config()); t=task_for(); state=copy.deepcopy(a.states)
        a.help(t,'test',0,'practice','learner_request')
        self.assertEqual(state,a.states)

    def test_help_changes_current_probability(self):
        a,b=Student('paired'),Student('paired'); t=task_for('code_completion')
        p=a.sample_evidence(t,.4,0)[1]; q=b.sample_evidence(t,.4,3)[1]
        self.assertTrue(all(x<y for x,y in zip(p,q)))

    def test_strength_changes_current_probability(self):
        a,b=Student('paired'),Student('paired'); t=task_for()
        self.assertTrue(all(x<y for x,y in zip(a.sample_evidence(t,.1,0)[1],b.sample_evidence(t,.9,0)[1])))

    def test_formats_have_distinct_guessing_time_and_help_rules(self):
        from catalog import FORMAT_RULES
        self.assertEqual(len({v['base_seconds'] for v in FORMAT_RULES.values()}),5)
        self.assertEqual(len({v['help_multiplier'] for v in FORMAT_RULES.values()}),5)
        s=Student('guess'); t=next(t for t in TASKS if t['task_type']=='single_choice')
        self.assertGreaterEqual(s.sample_evidence(t,0,0)[1][0],.25)
        t=task_for('multiple_select'); self.assertTrue(all(p>=.5 for p in s.sample_evidence(t,0,0)[1]))

    def test_retention_half_life_and_alternative_tail(self):
        self.assertEqual(retained(.8,4,4),.4); self.assertEqual(retained(.8,4,4,'power_law'),.4)
        self.assertGreater(retained(.8,20,4,'power_law'),retained(.8,20,4))

    def test_querying_strength_does_not_change_state(self):
        s=Student('query'); s.advance(DAY); state=copy.deepcopy(s.states)
        first=s.strength('list_indexing')
        self.assertEqual(first,s.strength('list_indexing')); self.assertEqual(state,s.states)

    def test_learning_is_bounded_and_zero_dose_is_noop(self):
        self.assertEqual(learned(.4,.2,0),.4); self.assertTrue(.4<learned(.4,.2,1)<1)

    def test_invalid_configuration_rejected(self):
        for kw in ({'max_hints':5},{'max_retries':-1},{'pre_answer_request_probability':2},{'tasks_per_skill':0},{'mechanism':'bad'}):
            with self.assertRaises(ValueError): Config(**kw)

    def test_seed_reproduces_full_history_and_hidden_state(self):
        a=Student('same',clean_config()).practice('mixed'); b=Student('same',clean_config()).practice('mixed')
        self.assertEqual(a.events,b.events); self.assertEqual(a.private_record(),b.private_record())

    def test_assessment_branch_does_not_modify_practice_or_learn(self):
        s=Student('fork',clean_config()).practice('mixed'); before=copy.deepcopy(s.private_record())
        a=s.assessment(7,'range_boundaries'); s.assessment(1,'range_boundaries'); b=s.assessment(7,'range_boundaries')
        self.assertEqual(before,s.private_record()); self.assertEqual(a.events,b.events)
        self.assertTrue(all(t['strength_before']==t['strength_after'] for t in a.trace))

    def test_spaced_blocked_schedule_revisits_skills_with_long_gaps(self):
        student = Student("spaced", clean_config(
            tasks_per_skill=16, tasks_per_session=12, spaced_practice=True, long_gaps=True,
        )).practice("blocked")
        ordinary = [e for e in student.events if e["event_type"] == "task" and e["purpose"] == "ordinary"]
        for skill in SKILLS:
            tasks = [e for e in ordinary if e["skill_id"] == skill]
            self.assertEqual(len(tasks), 16)
            self.assertEqual(len({e["session_id"] for e in tasks}), 4)
        self.assertTrue(any(
            q["input"]["since_last_event_seconds"] > 30 * DAY
            for record in prepare(student.events).values() for q in record["queries"]
            if q["target"] is not None and q["input"]["since_last_event_seconds"] is not None
        ))


class EvidenceTests(unittest.TestCase):
    def test_assisted_retry_keeps_original_independent_failure_label(self):
        r=fixtures()['fixture_three_hints']['records'][0]
        self.assertEqual(r['queries'][0]['target']['full_success'],0)
        self.assertEqual(len(r['queries']),2)

    def test_hint_before_first_answer_has_no_independent_label(self):
        r=fixtures()['fixture_requested_help']['records'][0]
        self.assertIsNone(r['queries'][0]['target'])

    def test_assessment_request_before_answer_is_incomplete(self):
        f=fixtures()['fixture_interrupted_assessment']; self.assertIsNone(f['records'][0]['queries'][0]['target'])
        self.assertEqual(next(e for e in f['events'] if e['event_type']=='assessment_end')['status'],'incomplete')
        self.assertEqual(next(e for e in f['events'] if e['event_type']=='response')['phase'],'review')

    def test_assessment_help_after_answer_does_not_replace_label(self):
        f=fixtures()['fixture_assessment_then_review']; self.assertEqual(f['records'][0]['queries'][0]['target']['full_success'],0)
        self.assertTrue(f['events'][-1]['full_success'])

    def test_unknown_assistance_excludes_target(self):
        s=Student('unknown',clean_config(missing_assistance_probability=1)); s.task('range_boundaries')
        self.assertIsNone(prepare(s.events)['range_boundaries']['queries'][0]['target'])

    def test_unresolved_grade_excludes_target(self):
        s=Student('unresolved',clean_config(unresolved_probability=1)); s.task('range_boundaries')
        self.assertIsNone(prepare(s.events)['range_boundaries']['queries'][0]['target']); validate_events(s.events)

    def test_missing_times_remain_null(self):
        s=Student('missing',clean_config(missing_time_probability=1,missing_reference_probability=1)); s.task('range_boundaries')
        r=prepare(s.events)['range_boundaries']; e=next(e for e in r['events'] if e['event_type']=='response')
        self.assertIsNone(e['active_response_seconds']); self.assertIsNone(e['relative_response_time'])

    def test_prediction_cannot_see_current_response_or_grade(self):
        f=fixtures()['fixture_three_hints']; r=f['records'][0]; q=r['queries'][0]
        x=materialize_query(r,q); self.assertEqual(x['history'],[])
        modified=copy.deepcopy(f['events']); g=next(e for e in modified if e['event_type']=='grade'); g['score']=1; g['full_success']=True
        r2=prepare(modified)['range_boundaries']; self.assertEqual(x,materialize_query(r2,r2['queries'][0]))

    def test_other_skill_events_do_not_erase_elapsed_gap(self):
        s=Student('gap',clean_config()); s.task('range_boundaries'); s.advance(30*DAY); s.task('list_indexing'); s.task('range_boundaries')
        self.assertGreaterEqual(prepare(s.events)['range_boundaries']['queries'][-1]['input']['since_last_response_seconds'],30*DAY)

    def test_renaming_learner_does_not_change_network_inputs(self):
        f=fixtures()['fixture_three_hints']; events=copy.deepcopy(f['events'])
        for e in events: e['learner_id']='other'
        a,b=f['records'][0],prepare(events)['range_boundaries']
        self.assertEqual(materialize_query(a,a['queries'][-1]),materialize_query(b,b['queries'][-1]))

    def test_hidden_or_text_fields_in_features_rejected(self):
        r=fixtures()['fixture_three_hints']['records'][0]
        r['events'][0]['hidden_strength']=.9
        with self.assertRaises(ValueError): materialize_query(r,r['queries'][-1])

    def test_query_identity_feature_rejected(self):
        r=fixtures()['fixture_three_hints']['records'][0]; q=r['queries'][0]; q['input']['target_task']['task_id']='leak'
        with self.assertRaises(ValueError): materialize_query(r,q)

    def test_invalid_event_mutations_rejected(self):
        source=fixtures()['fixture_three_hints']['events']
        mutations=[('hint','hint_number',4),('response','assistance','assisted'),('grade','score',.123),('task','hidden_strength',.8)]
        for kind,key,value in mutations:
            rows=copy.deepcopy(source); next(e for e in rows if e['event_type']==kind)[key]=value
            with self.assertRaises(ValueError): validate_events(rows)

    def test_retry_without_retry_event_rejected(self):
        rows=fixtures()['fixture_three_hints']['events']; rows=[e for e in rows if e['event_type']!='retry']
        with self.assertRaises(ValueError): validate_events(rows)

    def test_unjustified_instruction_rejected(self):
        rows=fixtures()['fixture_requested_lesson']['events'][1:]
        with self.assertRaises(ValueError): validate_events(rows)

    def test_delayed_queries_share_only_original_practice_prefix(self):
        s=Student('prefix',clean_config()).practice('mixed')
        probes=[dict(skill_id='range_boundaries',delay_days=d,events=s.assessment(d,'range_boundaries').events) for d in (1,7,30)]
        r=next(r for r in export_records(s.events,probes,True) if r['skill_id']=='range_boundaries')
        self.assertTrue(all(q['history_end']==len(r['events']) for q in r['queries'] if q['phase']=='assessment'))


class DiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.root=build_cohort(Path(cls.temp.name),'testcohort',20,clean_config())
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_full_audit_passes(self): self.assertEqual(audit(self.root)['status'],'passed')

    def test_training_loader_returns_only_input_and_binary_label(self):
        examples=list(load_training_examples(self.root)); self.assertGreater(len(examples),0)
        for x,y in examples:
            self.assertEqual(set(x),{'history','target_task','since_last_event_seconds','since_last_response_seconds'}); self.assertIn(y,(0,1))

    def test_training_records_include_only_frozen_delayed_prefixes(self):
        delayed = 0
        for record in read_rows(self.root / 'model_data/train.jsonl.gz'):
            for query in record['queries']:
                if query['phase'] == 'assessment':
                    self.assertEqual(query['history_end'], len(record['events']))
                    self.assertIsNotNone(query['target'])
                    delayed += 1
        self.assertEqual(delayed, 16 * 18)

    def test_evaluation_loader_has_all_clean_delayed_labels(self): self.assertEqual(len(list(load_evaluation_examples(self.root,'test'))),36)

    def test_splits_disjoint_and_balanced(self):
        rows=assignment(200,1,'check'); self.assertEqual(len({a[0] for a in rows}),200)
        self.assertEqual(Counter(a[1] for a in rows),{'train':160,'validation':20,'test':20})

    def test_overwrite_refused(self):
        with self.assertRaises(FileExistsError): build_cohort(Path(self.temp.name),'testcohort',20,clean_config())

    def test_regeneration_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            second=build_cohort(Path(tmp),'testcohort',20,clean_config())
            self.assertEqual(json.loads((self.root/'manifest.json').read_text()),json.loads((second/'manifest.json').read_text()))

    def test_checksum_tampering_detected(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'copy'; shutil.copytree(self.root,root)
            for name in ('task_catalog.json','lesson_catalog.json'): shutil.copy2(self.root.parent/name,root.parent/name)
            with (root/'model_data/train.jsonl.gz').open('ab') as f: f.write(b'tamper')
            with self.assertRaisesRegex(ValueError,'checksum'): audit(root)

    def test_future_prefix_tampering_detected_even_with_new_checksum(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'copy'; shutil.copytree(self.root,root)
            for name in ('task_catalog.json','lesson_catalog.json'): shutil.copy2(self.root.parent/name,root.parent/name)
            path=root/'model_data/train.jsonl.gz'
            rows=list(read_rows(path)); rows[0]['queries'][0]['history_end']=len(rows[0]['events'])
            with writer(path) as f:
                for r in rows: write_row(f,r)
            m=json.loads((root/'manifest.json').read_text()); m['files']['model_data/train.jsonl.gz']=digest(path); save_json(root/'manifest.json',m)
            with self.assertRaisesRegex(ValueError,'causal replay'): audit(root)


if __name__=='__main__': unittest.main()
