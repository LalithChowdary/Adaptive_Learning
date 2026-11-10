"""Application rules: question first, separate attempts, causal predictions."""
import json
import math
import random
import time
import uuid
from catalogue import TASKS, SKILLS, FORMATS, features, public_task, hint, solution
from grading import grade
from memory_service import MemoryService
from store import Store

FEATURE_FIELDS=('event_type','gap_seconds','new_session','task','attempt_number','active_response_seconds','relative_response_time','hint_count','hint_type','answer_revealed','assistance','score','full_success','grading_status')

class Dashboard:
    def __init__(self,path,device='auto',max_hints=3,max_retries=2):
        if not 0<=max_hints<=3 or not 0<=max_retries<=10: raise ValueError('Hints: 0–3; retries: 0–10')
        self.store=Store(path); self.memory=MemoryService(device); self.max_hints=max_hints; self.max_retries=max_retries
        self.references={s:[next(t for t in TASKS.values() if t['skill_id']==s and t['difficulty']=='medium' and t['task_type']==f and t['pool']=='assessment') for f in FORMATS] for s in SKILLS}

    def profile(self,learner):
        row=self.store.db.execute('SELECT * FROM profiles WHERE id=?',(learner,)).fetchone()
        if row is None: raise ValueError('Choose a learner profile first.')
        return dict(row)

    def emit(self,item,kind,now,**extra):
        events=self.store.events(item['learner'],item['skill'])
        fields={k:None for k in FEATURE_FIELDS}
        fields.update(event_type=kind,gap_seconds=None if not events else max(0,now-events[-1]['at']),
                      new_session=None if not events else False,task=features(TASKS[item['task']]))
        if kind in ('response','grade'):
            fields.update(attempt_number=item['attempt'],hint_count=len(item['hints']),answer_revealed=False,
                          assistance='assisted' if item['hints'] else 'independent')
        if kind=='hint': fields.update(hint_count=len(item['hints']),hint_type=item['hints'][-1]['kind'],assistance='assisted')
        if kind=='solution_reveal': fields.update(answer_revealed=True,assistance='assisted')
        if kind=='retry': fields['attempt_number']=item['attempt']
        fields.update(extra.pop('feature_values',{}))
        payload=dict(features=fields,source='real_local_session',schema_version='dashboard-1',model_sha256=self.memory.version,**extra)
        self.store.db.execute('INSERT INTO events(learner,skill,interaction,at,kind,payload) VALUES(?,?,?,?,?,?)',
                             (item['learner'],item['skill'],item['id'],now,kind,json.dumps(payload)))

    def predict(self,learner,skill,task,now):
        return self.memory.predict(self.store.events(learner,skill),features(task),now,key=(learner,skill,self.memory.version))

    def public(self,item):
        t=TASKS[item['task']]
        data={k:v for k,v in item.items() if k not in ('prediction','order')}
        data['question']=public_task(t,item['order']); data['max_hints']=self.max_hints; data['max_retries']=self.max_retries
        if item['results']: data['prediction_before']=item['prediction']
        if item.get('revealed'): data['solution']=solution(t)
        return data

    def bootstrap(self,learner=None):
        profiles=[dict(r) for r in self.store.db.execute('SELECT * FROM profiles ORDER BY created')]
        result=dict(profiles=profiles,skills=[dict(id=k,name=v[0],description=v[1],number=v[2]) for k,v in SKILLS.items()],formats=list(FORMATS),device=self.memory.device,model='Time-decayed GRU · v2.1',model_sha256=self.memory.version)
        if learner:
            self.profile(learner); result['learner']=learner
            items=self.store.interactions(learner)
            active=next((i for i in reversed(items) if not i['closed']),None)
            result['active']=self.public(active) if active else None
            result['progress']=self.progress(learner)
        return result

    def progress(self,learner):
        self.profile(learner); now=time.time(); items=self.store.interactions(learner); all_events=self.store.events(learner)
        cards=[]
        for skill in SKILLS:
            ev=[e for e in all_events if e['skill']==skill]
            attempts=[e for e in ev if e['kind']=='grade' and e['features']['grading_status']=='resolved']
            independent=[e for e in attempts if e['features']['attempt_number']==1 and e['features']['assistance']=='independent']
            cards.append(dict(id=skill,name=SKILLS[skill][0],attempts=len(attempts),independent=len(independent),
                successes=sum(bool(e['features']['full_success']) for e in independent),hints=sum(e['kind']=='hint' for e in ev),
                last_at=ev[-1]['at'] if ev else None,
                prediction=sum(self.predict(learner,skill,t,now) for t in self.references[skill])/5 if attempts else None))
        eligible=[r for i in items for r in i['results'] if r['eligible']]
        return dict(cards=cards,total_attempts=sum(c['attempts'] for c in cards),independent=len(eligible),
            successes=sum(r['full_success'] for r in eligible),
            brier=sum((r['prediction']-float(r['full_success']))**2 for r in eligible)/len(eligible) if eligible else None,
            history=[dict(id=i['id'],skill=SKILLS[i['skill']][0],format=TASKS[i['task']]['task_type'],at=i['created'],
                          mode=i['mode'],hints=len(i['hints']),results=i['results'],status=i['status']) for i in reversed(items)])

    def curve(self,learner,skill,task_id=None):
        self.profile(learner)
        if skill not in SKILLS: raise ValueError('Unknown skill')
        task=TASKS[task_id] if task_id else self.references[skill][2]
        if task['skill_id']!=skill: raise ValueError('Task does not match skill')
        now=time.time()
        return dict(skill=skill,task=dict(task_id=task['task_id'],difficulty=task['difficulty'],task_type=task['task_type']),
                    points=[dict(day=d,probability=self.predict(learner,skill,task,now+d*86400)) for d in (0,1,3,7,14,30,45,60)])

    def command(self,action,data):
        request=data.get('request_id')
        if not isinstance(request,str) or not 8<=len(request)<=100: raise ValueError('A request ID is required')
        previous=self.store.db.execute('SELECT result FROM requests WHERE id=?',(request,)).fetchone()
        if previous: return json.loads(previous[0])
        try:
            with self.store.db:
                result=self._command(action,data)
                self.store.db.execute('INSERT INTO requests VALUES(?,?)',(request,json.dumps(result)))
            return result
        except Exception:
            # Predictions during a rolled-back transaction must not retain its evidence.
            self.memory.cache.clear()
            raise

    def _command(self,action,data):
        now=time.time()
        if action=='profile':
            name=str(data.get('name','')).strip()
            if not 1<=len(name)<=50: raise ValueError('Use a name from 1 to 50 characters.')
            learner=str(uuid.uuid4())
            self.store.db.execute('INSERT INTO profiles VALUES(?,?,?)',(learner,name,now))
            return dict(learner=learner)
        learner=data.get('learner'); self.profile(learner)
        items=self.store.interactions(learner)
        if action=='start':
            if any(not i['closed'] for i in items): raise ValueError('Finish or leave the current question first.')
            skill=data.get('skill'); fmt=data.get('format'); difficulty=data.get('difficulty','easy'); mode=data.get('mode','practice')
            if mode not in ('practice','assessment'): raise ValueError('Unknown mode')
            pool='practice' if mode=='practice' else 'assessment'
            used={i['task'] for i in items}
            candidates=[t for t in TASKS.values() if t['skill_id']==skill and t['difficulty']==difficulty and t['pool']==pool and t['task_id'] not in used and (fmt=='mixed' or t['task_type']==fmt)]
            if not candidates: raise ValueError('No unseen questions remain for these filters. Choose another format or difficulty.')
            task=random.SystemRandom().choice(candidates); order=[o['id'] for o in task['options']]; random.SystemRandom().shuffle(order)
            prediction=self.predict(learner,skill,task,now)
            item=dict(id=str(uuid.uuid4()),learner=learner,skill=skill,task=task['task_id'],created=now,attempt_started=now,
                      mode=mode,original_mode=mode,status='answering',attempt=1,hints=[],results=[],closed=False,revealed=False,
                      order=order,prediction=prediction,assessment_incomplete=False)
            self.emit(item,'task',now,prediction_before=prediction,prediction_task=features(task),mode=mode)
            self.store.save(item); return self.public(item)
        item=next((i for i in items if i['id']==data.get('interaction')),None)
        if not item: raise ValueError('Question not found for this learner.')
        if item['closed']: raise ValueError('This question is already closed.')
        task=TASKS[item['task']]
        if action=='hint':
            if item['status'] not in ('answering','feedback') or len(item['hints'])>=self.max_hints: raise ValueError('No more hints are available for this question.')
            if item['results'] and item['results'][-1]['full_success']: raise ValueError('This question is already correct. Try a fresh one.')
            self.emit(item,'help_request',now)
            if item['mode']=='assessment':
                if not item['results']:
                    item['assessment_incomplete']=True; self.emit(item,'assessment_incomplete',now,reason='help_before_submission')
                item['mode']='review'
            h=hint(task,len(item['hints'])+1); item['hints'].append(h); self.emit(item,'hint',now,content=h)
        elif action=='submit':
            if item['status']!='answering': raise ValueError('This attempt has already been submitted. Choose retry or a fresh question.')
            result=grade(task,data.get('answer'))
            active=data.get('active_seconds')
            if active is not None and (type(active) not in (int,float) or not math.isfinite(active) or not 0<=active<=now-item['attempt_started']+3): raise ValueError('Invalid response duration. Reload and submit again.')
            self.emit(item,'response',now,answer=data.get('answer'),wall_seconds=now-item['attempt_started'],timing_source='browser_visible_focused_estimate' if active is not None else 'unavailable',
                      feature_values=dict(active_response_seconds=active,relative_response_time=active/task['reference_seconds'] if active is not None else None))
            self.emit(item,'grade',now,grade=result,feature_values={k:result[k] for k in ('score','full_success','grading_status')})
            eligible=item['attempt']==1 and not item['hints'] and result['grading_status']=='resolved' and item['mode']!='review'
            row=dict(**result,at=now,attempt=item['attempt'],assisted=bool(item['hints']),eligible=eligible,prediction=item['prediction'],active_seconds=active)
            item['results'].append(row); item['status']='feedback'
        elif action=='retry':
            if item['status']!='feedback' or item['attempt']>self.max_retries: raise ValueError('Retry limit reached.')
            if item['results'][-1]['full_success']: raise ValueError('Try a fresh question after success.')
            if item['mode']=='assessment': item['mode']='review'
            item['attempt']+=1; item['attempt_started']=now; item['status']='answering'; self.emit(item,'retry',now)
        elif action=='reveal':
            if item['status']!='feedback': raise ValueError('Submit an attempt before revealing the answer.')
            item['revealed']=True; item['closed']=True; item['status']='completed'; self.emit(item,'solution_reveal',now)
        elif action=='finish':
            item['closed']=True; item['status']='completed' if item['results'] else 'skipped'; self.emit(item,'close',now,reason=item['status'])
        else: raise ValueError('Unknown action')
        self.store.save(item)
        return self.public(item)

    def export(self,learner):
        return dict(schema='dashboard-1',source='real_local_session',profile=self.profile(learner),model_sha256=self.memory.version,
                    interactions=self.store.interactions(learner),events=self.store.events(learner))
