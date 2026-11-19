import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from catalogue import TASKS,SKILLS,features,hint
from grading import grade,Interpreter,Unsupported
from memory_service import MemoryService,CHECKPOINT
from service import Dashboard
from server import serve

class GradingTests(unittest.TestCase):
    def test_all_2160_reference_answers(self):
        for task in TASKS.values():
            answer=task['reference_solution'] if task['task_type']=='code_completion' else task['answer_key']
            with self.subTest(task=task['task_id']):
                result=grade(task,answer)
                self.assertEqual(result['grading_status'],'resolved')
                self.assertTrue(result['full_success'])
                self.assertAlmostEqual(result['score'],1)
    def test_catalogue_structure(self):
        self.assertEqual(len(TASKS),2160)
        for task in TASKS.values():
            self.assertEqual(len(task['criteria']),len(task['tests']) if task['task_type']=='code_completion' else len(task['answer_key']) if task['task_type'] in ('fill_in_the_blank','output_prediction') else len(task['options']) if task['task_type']=='multiple_select' else 1)
            self.assertEqual(len({hint(task,n)['text'] for n in (1,2,3)}),3)
            self.assertTrue(all(hint(task,n)['text'] for n in (1,2,3)))
    def test_multiple_select_penalties(self):
        t=next(t for t in TASKS.values() if t['task_type']=='multiple_select')
        self.assertFalse(grade(t,[])['full_success'])
        self.assertEqual(grade(t,[o['id'] for o in t['options']])['score'],0)
        self.assertGreater(grade(t,[t['answer_key'][0]])['score'],0)
        self.assertFalse(grade(t,[t['answer_key'][0]])['full_success'])
    def test_open_partial_and_format_normalization(self):
        t=next(t for t in TASKS.values() if t['task_type']=='output_prediction' and len(t['criteria'])==2)
        a=t['answer_key'][:];a[0]='wrong'
        self.assertAlmostEqual(grade(t,a)['score'],t['criteria'][1]['weight'])
        a=[v.replace(', ', ',')+'\n' for v in t['answer_key']]
        self.assertTrue(grade(t,a)['full_success'])
    def test_code_alternative_solution_and_partial(self):
        t=next(t for t in TASKS.values() if t['task_type']=='code_completion' and t['skill_id']=='loop_tracing')
        from catalogue import source
        original=Interpreter(t['reference_solution'])
        start=original.run(0);step=original.run(1)-start
        self.assertTrue(grade(t,f'def solve(x):\n    return {start} + x * {step}')['full_success'])
        self.assertFalse(grade(t,'def solve(x):\n    return 0')['full_success'])
    def test_unsupported_never_marked_wrong(self):
        t=next(t for t in TASKS.values() if t['task_type']=='code_completion')
        for code in ['import os\ndef solve(x): return 1','def solve(x): return open("/tmp/a")','def solve(x): return x.__class__','def solve(x):\n    while True: pass','def solve(x): return [0] * 1000000000','def solve(x): return list(range(1000000000))','def solve(x): return 2 ** 99999999','def solve(x): return list(range(999999999999999999999999999999999))']:
            with self.subTest(code=code):
                r=grade(t,code);self.assertEqual(r['grading_status'],'unresolved');self.assertIsNone(r['score'])
    def test_syntax_error_is_resolved_failure(self):
        t=next(t for t in TASKS.values() if t['task_type']=='code_completion')
        self.assertEqual(grade(t,'def solve(x) return x')['score'],0)

class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model=MemoryService('cpu')
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.app=Dashboard(Path(self.tmp.name)/'test.sqlite3','cpu')
        self.learner=self.command('profile',name='Test learner')['learner']
    def tearDown(self):self.app.store.db.close();self.tmp.cleanup()
    def command(self,action,**kw):return self.app.command(action,dict(learner=getattr(self,'learner',None),request_id=str(uuid.uuid4()),**kw))
    def start(self,**kw):return self.command('start',**dict(skill='range_boundaries',format='single_choice',difficulty='easy',mode='practice',**kw))
    def test_prediction_hidden_and_no_answer_leak(self):
        item=self.start();text=json.dumps(item)
        self.assertNotIn('answer_key',text);self.assertNotIn('prediction_before',item);self.assertNotIn('prediction',item)
        self.assertNotIn('hints',item['question'])
        self.assertNotIn('criteria',item['question'])
    def test_submit_hint_retry_and_fresh_target(self):
        i=self.start();t=TASKS[i['task']];wrong=next(o['id'] for o in t['options'] if o['id']!=t['answer_key'])
        i=self.command('submit',interaction=i['id'],answer=wrong,active_seconds=None)
        self.assertTrue(i['results'][0]['eligible']);self.assertFalse(i['results'][0]['full_success'])
        for n in range(3):i=self.command('hint',interaction=i['id'])
        with self.assertRaises(ValueError):self.command('hint',interaction=i['id'])
        i=self.command('retry',interaction=i['id']);i=self.command('submit',interaction=i['id'],answer=t['answer_key'],active_seconds=None)
        self.assertTrue(i['results'][-1]['assisted']);self.assertFalse(i['results'][-1]['eligible'])
        kinds=[e['kind'] for e in self.app.store.events(self.learner)]
        self.assertEqual(kinds,['task','response','grade','help_request','hint','help_request','hint','help_request','hint','retry','response','grade'])
        self.command('finish',interaction=i['id']);next_i=self.start();self.assertNotEqual(next_i['task'],i['task'])
    def test_assessment_help_before_submission_incomplete(self):
        i=self.command('start',skill='range_boundaries',format='single_choice',difficulty='easy',mode='assessment')
        i=self.command('hint',interaction=i['id']);self.assertTrue(i['assessment_incomplete']);self.assertEqual(i['mode'],'review')
        i=self.command('submit',interaction=i['id'],answer=TASKS[i['task']]['answer_key']);self.assertFalse(i['results'][0]['eligible'])
    def test_saved_assessment_not_overwritten(self):
        i=self.command('start',skill='range_boundaries',format='single_choice',difficulty='easy',mode='assessment')
        t=TASKS[i['task']];wrong=next(o['id'] for o in t['options'] if o['id']!=t['answer_key'])
        i=self.command('submit',interaction=i['id'],answer=wrong)
        original=copy.deepcopy(i['results'][0]);i=self.command('hint',interaction=i['id']);i=self.command('retry',interaction=i['id']);i=self.command('submit',interaction=i['id'],answer=t['answer_key'])
        self.assertEqual(i['results'][0],original);self.assertFalse(i['results'][1]['eligible'])
    def test_retry_cap(self):
        i=self.start();t=TASKS[i['task']];wrong=next(o['id'] for o in t['options'] if o['id']!=t['answer_key'])
        for attempt in range(3):
            i=self.command('submit',interaction=i['id'],answer=wrong)
            if attempt<2:i=self.command('retry',interaction=i['id'])
        with self.assertRaises(ValueError):self.command('retry',interaction=i['id'])
    def test_idempotent_request_and_duplicate_submit(self):
        i=self.start();body=dict(learner=self.learner,interaction=i['id'],answer=TASKS[i['task']]['answer_key'],request_id=str(uuid.uuid4()))
        a=self.app.command('submit',body);b=self.app.command('submit',body);self.assertEqual(a,b)
        self.assertEqual(len(self.app.store.events(self.learner)),3)
        with self.assertRaises(ValueError):self.command('submit',interaction=i['id'],answer=body['answer'])
    def test_restart_and_profile_skill_isolation(self):
        i=self.start();self.command('submit',interaction=i['id'],answer=TASKS[i['task']]['answer_key'])
        other=self.command('profile',name='Another')['learner'];p=self.app.progress(other)
        self.assertEqual(p['total_attempts'],0)
        p=self.app.progress(self.learner);self.assertEqual(sum(c['attempts'] for c in p['cards']),1)
        model_hash=self.app.memory.version
        second=Dashboard(Path(self.tmp.name)/'test.sqlite3','cpu')
        try:
            self.assertEqual(second.progress(self.learner)['total_attempts'],1)
            self.assertEqual(second.memory.version,model_hash)
            self.assertEqual(second.bootstrap(self.learner)['active']['id'],i['id'])
        finally:second.store.db.close()
    def test_queries_do_not_change_events_or_state(self):
        i=self.start();self.command('submit',interaction=i['id'],answer=TASKS[i['task']]['answer_key'])
        ev=self.app.store.events(self.learner,'range_boundaries');m=self.app.memory;now=time.time()
        p=m.predict(ev,features(TASKS[i['task']]),now,key='test')
        h=m.cache['test']['state'].clone()
        m.predict(ev,features(TASKS[i['task']]),now+60*86400,key='test')
        self.assertEqual(p,m.predict(ev,features(TASKS[i['task']]),now,key='test'))
        self.assertTrue(h.equal(m.cache['test']['state']))
        self.assertEqual(ev,self.app.store.events(self.learner,'range_boundaries'))
    def test_feature_contract_matches_dataset_projection(self):
        from catalogue import source
        for t in TASKS.values():
            f=features(t)
            self.assertEqual(f['credit_weights'],sorted(c['weight'] for c in t['criteria'] if c['kind']!='incorrect_selection_penalty'))
    def test_replay_matches_original_model_queries(self):
        import torch
        from memory.data import encode_record
        path=ROOT.parent/'data_Set/v2.2/dataset/main/model_data/test.jsonl.gz'
        with gzip.open(path,'rt') as f:record=json.loads(next(f))
        encoded=encode_record(record)
        with torch.inference_mode():
            def t(a,dtype=None):return torch.tensor(a,dtype=dtype).unsqueeze(0)
            expected=torch.sigmoid(self.model.model(t(encoded.tokens),t(encoded.gaps),t(encoded.positions),t(encoded.tasks),t(encoded.query_gaps)))[0].tolist()
        index=0
        for query in record['queries']:
            if not query['target']: continue
            prefix=record['events'][:query['history_end']];events=[];at=0
            for n,e in enumerate(prefix):
                at+=e['gap_seconds'] or 0;events.append(dict(id=n+1,at=at,features=e))
            now=at+(query['input']['since_last_event_seconds'] or 0)
            p=self.model.predict(events,query['input']['target_task'],now)
            self.assertAlmostEqual(p,expected[index],places=6)
            index+=1
        self.assertGreater(index,10)
    def test_fast_wrong_answer_decreases_prediction_and_curve_non_increasing(self):
        unassessed_curve=self.app.curve(self.learner,'conditional_evaluation')
        self.assertFalse(unassessed_curve['assessed'])
        p_before=unassessed_curve['points'][0]['probability']
        i=self.command('start',skill='conditional_evaluation',format='output_prediction',difficulty='medium',mode='practice')
        t=TASKS[i['task']]
        wrong=['__wrong_output__' for _ in t['answer_key']]
        i=self.command('submit',interaction=i['id'],answer=wrong,active_seconds=2.0)
        self.assertFalse(i['results'][0]['full_success'])
        self.assertEqual(i['results'][0]['score'],0.0)
        assessed_curve=self.app.curve(self.learner,'conditional_evaluation')
        self.assertTrue(assessed_curve['assessed'])
        probs=[pt['probability'] for pt in assessed_curve['points']]
        self.assertLess(probs[0],p_before)
        for prev_p,next_p in zip(probs,probs[1:]):
            self.assertLessEqual(next_p,prev_p+1e-7)
    def test_two_hard_successes_retain_realistic_long_horizon_curve(self):
        cold_p=self.app.curve(self.learner,'function_returns')['points'][0]['probability']
        for fmt in ('fill_in_the_blank','single_choice'):
            i=self.command('start',skill='function_returns',format=fmt,difficulty='hard',mode='practice')
            t=TASKS[i['task']]
            i=self.command('submit',interaction=i['id'],answer=t['answer_key'],active_seconds=2.0)
            self.assertTrue(i['results'][0]['full_success'])
            self.command('finish',interaction=i['id'])
        curve=self.app.curve(self.learner,'function_returns')
        self.assertTrue(curve['assessed'])
        probs=[pt['probability'] for pt in curve['points']]
        for prev_p,next_p in zip(probs,probs[1:]):
            self.assertLessEqual(next_p,prev_p+1e-7)
        self.assertGreater(probs[0],0.65)
        self.assertGreater(probs[-2],0.55)
        self.assertGreater(probs[-1],cold_p)
    def test_export_no_synthetic_events(self):
        i=self.start();data=self.app.export(self.learner)
        self.assertEqual(data['source'],'real_local_session');self.assertTrue(all(e['source']=='real_local_session' for e in data['events']))
    def test_http_security_and_static(self):
        server=serve(self.app,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base+'/') as r:self.assertIn(b'recall',r.read())
            with urllib.request.urlopen(base+'/api/state') as r:state=json.load(r)
            request=urllib.request.Request(base+'/api/profile',data=json.dumps(dict(name='HTTP',request_id=str(uuid.uuid4()))).encode(),headers={'Content-Type':'application/json'},method='POST')
            with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(request)
            self.assertEqual(e.exception.code,403)
            request.add_header('X-Dashboard-Token',state['token'])
            with urllib.request.urlopen(request) as r:learner_id=json.load(r)['learner']
            with urllib.request.urlopen(f'{base}/api/ppo?learner={learner_id}&skill=range_boundaries') as r:
                ppo_payload=json.load(r)
                self.assertEqual(len(ppo_payload['observation_vector']),36)
                self.assertEqual(len(ppo_payload['actions']),8)
        finally:server.shutdown();server.server_close();thread.join()
    def test_ppo_state_observation_and_initial_mask(self):
        ppo=self.app.ppo_state(self.learner,skill='range_boundaries',difficulty='easy')
        self.assertEqual(ppo['obs_dim'],36)
        self.assertEqual(len(ppo['observation_vector']),36)
        self.assertTrue(all(0.0<=v<=1.0 for v in ppo['observation_vector']))
        valid_ids=[a['id'] for a in ppo['actions'] if a['valid']]
        self.assertEqual(valid_ids,[1,2,3,4])
        self.assertAlmostEqual(sum(a['probability'] for a in ppo['actions']),1.0,places=5)
        self.assertTrue(all(a['probability']==0.0 for a in ppo['actions'] if not a['valid']))
        self.assertIn(ppo['recommended_action']['id'],valid_ids)
    def test_ppo_step_lesson_gated_option_b_and_followup_question(self):
        lesson=self.command('ppo_step',skill='range_boundaries',difficulty='easy',action_id=3)
        self.assertEqual(lesson['kind'],'lesson')
        self.assertEqual(lesson['ppo_action'],'explain_concept')
        self.assertEqual(lesson['primed_mode'],'concept')
        self.assertTrue(lesson['ppo']['state_summary']['awaiting_question'])
        valid_after_lesson=[a['id'] for a in lesson['ppo']['actions'] if a['valid']]
        self.assertEqual(valid_after_lesson,[1,2])
        with self.assertRaises(ValueError):
            self.command('ppo_step',skill='range_boundaries',difficulty='easy',action_id=3)
        q=self.command('ppo_step',skill='range_boundaries',difficulty='easy',action_id=1)
        self.assertIn('question',q)
        self.assertEqual(q['ppo_action'],'give_same_difficulty_question')
        ppo_during_q=self.app.ppo_state(self.learner,skill='range_boundaries',difficulty='easy')
        self.assertFalse(ppo_during_q['state_summary']['awaiting_question'])
    def test_ppo_self_explanation_unlocks_after_question_and_locks_after_use(self):
        i=self.start();t=TASKS[i['task']]
        self.command('submit',interaction=i['id'],answer=t['answer_key'])
        self.command('finish',interaction=i['id'])
        ppo_after_q=self.app.ppo_state(self.learner,skill='range_boundaries',difficulty='easy')
        self.assertTrue(ppo_after_q['state_summary']['can_self_explain'])
        self.assertTrue(ppo_after_q['actions'][5]['valid'])
        se_q=self.command('ppo_step',skill='range_boundaries',difficulty='easy',format='single_choice',action_id=5)
        self.assertEqual(se_q['ppo_action'],'request_self_explanation')
        self.assertEqual(se_q['mode'],'assessment')
        self.assertTrue(se_q.get('self_explanation_prompt'))
        t_se=TASKS[se_q['task']]
        self.command('submit',interaction=se_q['id'],answer=t_se['answer_key'])
        self.command('finish',interaction=se_q['id'])
        ppo_after_se=self.app.ppo_state(self.learner,skill='range_boundaries',difficulty='easy')
        self.assertFalse(ppo_after_se['state_summary']['can_self_explain'])
        self.assertFalse(ppo_after_se['actions'][5]['valid'])
    def test_ppo_assisted_hint_sets_unverified_and_prereq_or_advance(self):
        i=self.command('start',skill='loop_tracing',format='single_choice',difficulty='easy',mode='practice')
        t=TASKS[i['task']]
        self.command('hint',interaction=i['id'])
        self.command('submit',interaction=i['id'],answer=t['answer_key'])
        self.command('finish',interaction=i['id'])
        ppo=self.app.ppo_state(self.learner,skill='loop_tracing',difficulty='easy')
        self.assertTrue(ppo['state_summary']['unverified_assisted'])
        self.assertGreater(ppo['state_summary']['last_leakage'],0.0)
        self.assertTrue(ppo['actions'][6]['valid'])
        self.assertTrue(ppo['actions'][7]['valid'])
        verify_q=self.command('ppo_step',skill='loop_tracing',difficulty='easy',format='single_choice',action_id=1)
        self.assertEqual(verify_q['mode'],'assessment')
        self.command('submit',interaction=verify_q['id'],answer=TASKS[verify_q['task']]['answer_key'])
        self.command('finish',interaction=verify_q['id'])
        ppo_cleared=self.app.ppo_state(self.learner,skill='loop_tracing',difficulty='easy')
        self.assertFalse(ppo_cleared['state_summary']['unverified_assisted'])
    def test_curve_format_selector_and_difficulty_monotonicity(self):
        c=self.app.curve(self.learner,'range_boundaries',fmt='single_choice')
        self.assertEqual(c['format'],'single_choice')
        self.assertIn('curves_by_difficulty',c)
        self.assertIn('diagnostics',c)
        for idx in range(len(c['points'])):
            pe=c['curves_by_difficulty']['easy'][idx]['probability']
            pm=c['curves_by_difficulty']['medium'][idx]['probability']
            ph=c['curves_by_difficulty']['hard'][idx]['probability']
            self.assertGreater(pe,pm)
            self.assertGreater(pm,ph)

if __name__=='__main__':unittest.main(verbosity=2)
