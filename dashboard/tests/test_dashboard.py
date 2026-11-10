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
        path=ROOT.parent/'data_Set/v2.1/dataset/main/model_data/test.jsonl.gz'
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
            with urllib.request.urlopen(request) as r:self.assertIn('learner',json.load(r))
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main(verbosity=2)
