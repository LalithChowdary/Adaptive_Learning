"""Causal replay with explicit field allowlists and raw-event contract checks."""
from copy import deepcopy
import math
from catalog import SKILLS, grade
from simulator import TASK_INDEX, VERSION

COMMON={'schema_version','event_id','learner_id','skill_id','course_id','session_id','branch','interaction_id','event_type','phase','occurred_at','available_at'}
FIELDS={
 'task': {'task_id','purpose','parent_interaction_id','reference_seconds','max_hints','max_retries'},
 'help_request': {'task_id','reason'},
 'hint': {'task_id','hint_number','hint_type','reason','content_ref'},
 'retry': {'task_id','attempt_number','reason'},
 'response': {'task_id','attempt_number','active_response_seconds','hint_count','answer_revealed','assistance'},
 'grade': {'task_id','response_id','grading_status','rubric_id','grader','grader_confidence','evidence','criteria','score','full_success'},
 'solution_reveal': {'task_id','reason','content_ref'},
 'assessment_end': {'task_id','status','reason'},
 'mode_switch': {'task_id','reason'},
 'lesson_request': {'reason'},
 'instruction': {'reason','content_ref','teaching_action'},
}
TASK_FIELDS={'difficulty','task_type','reference_seconds','snippet_count','option_count','criterion_count','credit_weights','penalty_weights'}
FEATURE_FIELDS={'event_type','gap_seconds','new_session','task','attempt_number','active_response_seconds','relative_response_time','hint_count','hint_type','answer_revealed','assistance','score','full_success','grading_status'}


def task_features(task,reference):
    return dict(difficulty=task['difficulty'],task_type=task['task_type'],reference_seconds=reference,
                snippet_count=len(task['snippets']),option_count=len(task['options']),criterion_count=len(task['criteria']),
                credit_weights=sorted(c['weight'] for c in task['criteria'] if c['kind']!='incorrect_selection_penalty'),
                penalty_weights=sorted(c['weight'] for c in task['criteria'] if c['kind']=='incorrect_selection_penalty'))


def validate_events(events):
    seen, interactions, responses, lessons = set(),{}, {},set()
    last=float('-inf'); learner=events[0]['learner_id'] if events else None
    for e in events:
        kind=e['event_type']
        if kind not in FIELDS or set(e)!=COMMON|FIELDS[kind]: raise ValueError('unexpected or missing event fields')
        if e['schema_version']!=VERSION or e['skill_id'] not in SKILLS or e['learner_id']!=learner: raise ValueError('invalid routing/version')
        if e['event_id'] in seen: raise ValueError('duplicate event')
        seen.add(e['event_id'])
        if not all(math.isfinite(e[k]) for k in ('occurred_at','available_at')) or not last<=e['occurred_at']<=e['available_at']: raise ValueError('noncausal event times')
        last=e['available_at']; key=(e['branch'],e['interaction_id'])
        if kind=='lesson_request':
            if e['reason']!='learner_requested_intro': raise ValueError('invalid lesson request')
            lessons.add(key); continue
        if kind=='instruction':
            if key not in lessons or e['reason']!='learner_requested_intro' or not e['content_ref'] or e['teaching_action']!='explain': raise ValueError('unjustified instruction')
            lessons.remove(key); continue
        task=TASK_INDEX.get(e['task_id'])
        if task is None or task['skill_id']!=e['skill_id']: raise ValueError('task attribution mismatch')
        if kind=='task':
            if key in interactions: raise ValueError('duplicate presentation')
            if e['phase'] not in ('practice','assessment') or e['purpose'] not in ('ordinary','followup','delayed'): raise ValueError('invalid task phase')
            if (e['phase']=='assessment')!=(e['purpose']=='delayed'): raise ValueError('purpose mismatch')
            if e['reference_seconds'] not in (None,task['reference_seconds']): raise ValueError('wrong reference time')
            if not (type(e['max_hints']) is int and 0<=e['max_hints']<=len(task['hints']) and type(e['max_retries']) is int and 0<=e['max_retries']<=10): raise ValueError('invalid caps')
            if e['purpose']=='followup':
                parent=interactions.get((e['branch'],e['parent_interaction_id']))
                if not parent or parent['task']['skill_id']!=e['skill_id'] or parent['task']['task_id']==e['task_id'] or not parent['hints']: raise ValueError('invalid fresh followup')
            elif e['parent_interaction_id'] is not None: raise ValueError('unexpected parent')
            interactions[key]=dict(task=task,presentation=e,phase=e['phase'],attempts=0,hints=0,request=False,retry=False,pending_grade=None,last_grade=None,closed=None,revealed=False)
            continue
        if key not in interactions or interactions[key]['task']['task_id']!=e['task_id']: raise ValueError('event without matching task')
        state=interactions[key]; presentation=state['presentation']
        if kind=='mode_switch':
            if state['phase']!='assessment' or state['closed'] is None or e['phase']!='review': raise ValueError('invalid assessment transition')
            if e['reason'] not in ('requested_help','post_assessment_review'): raise ValueError('invalid transition reason')
            state['phase']='review'
        if e['phase']!=state['phase']: raise ValueError('phase mismatch')
        if state['pending_grade'] is not None and kind!='grade': raise ValueError('grade must precede next action')
        if kind=='help_request':
            if e['reason']!='learner_request': raise ValueError('invalid help request')
            state['request']=True
        elif kind=='hint':
            if state['phase']=='assessment' or presentation['purpose']=='followup': raise ValueError('hint in independent phase')
            if state['hints']>=presentation['max_hints'] or e['hint_number']!=state['hints']+1: raise ValueError('hint cap/order')
            spec=task['hints'][state['hints']]
            if e['hint_type']!=spec['kind'] or e['content_ref']!=f"{task['task_id']}/hint/{e['hint_number']}": raise ValueError('invalid hint content')
            if e['reason']=='learner_request':
                if not state['request']: raise ValueError('missing explicit request')
            elif e['reason']=='observed_difficulty':
                if not state['last_grade'] or state['last_grade']['full_success'] is not False: raise ValueError('help without observed difficulty')
            else: raise ValueError('invalid help reason')
            state['request']=False; state['hints']+=1
        elif kind=='retry':
            if state['phase']=='assessment' or presentation['purpose']=='followup': raise ValueError('retry in independent phase')
            if state['attempts']<1 or state['attempts']>presentation['max_retries'] or e['attempt_number']!=state['attempts']+1 or state['retry']: raise ValueError('retry cap/order')
            if not state['last_grade'] or state['last_grade']['full_success'] is not False: raise ValueError('retry without observed difficulty')
            state['retry']=True
        elif kind=='response':
            if state['revealed'] or (state['phase']=='assessment' and state['closed'] is not None): raise ValueError('response after close/reveal')
            if e['attempt_number']!=state['attempts']+1 or (state['attempts']>0 and not state['retry']): raise ValueError('response attempt ordering')
            if state['phase']=='assessment' and (state['hints'] or state['attempts']): raise ValueError('assisted scored assessment')
            state['attempts']+=1; state['retry']=False
            if e['active_response_seconds'] is not None and not 0<e['active_response_seconds']<=900: raise ValueError('duration range')
            if e['assistance']=='unknown':
                if e['hint_count'] is not None or e['answer_revealed'] is not None: raise ValueError('unknown assistance must be null')
            elif e['assistance']!=('assisted' if state['hints'] else 'independent') or e['hint_count']!=state['hints'] or e['answer_revealed'] is not False: raise ValueError('false independence or hint snapshot')
            responses[e['event_id']]=e; state['pending_grade']=e['event_id']
        elif kind=='grade':
            if state['pending_grade']!=e['response_id']: raise ValueError('invalid grade linkage')
            if e['rubric_id']!=task['rubric_id'] or e['grader']!='simulated_rubric_v2' or e['grader_confidence'] is not None: raise ValueError('invalid grading provenance')
            if e['grading_status']=='unresolved':
                if any(e[k] is not None for k in ('evidence','criteria','score','full_success')): raise ValueError('unresolved outcome carries label')
            elif e['grading_status']=='resolved':
                actual=grade(task,e['evidence'])
                if any(e[k]!=v for k,v in actual.items()): raise ValueError('rubric mismatch')
            else: raise ValueError('invalid grading status')
            state['pending_grade']=None; state['last_grade']=e
        elif kind=='assessment_end':
            if state['phase']!='assessment' or state['closed'] is not None: raise ValueError('invalid assessment end')
            if e['status']=='complete':
                if state['attempts']!=1 or state['last_grade'] is None or e['reason']!='first_attempt_recorded': raise ValueError('incomplete scored attempt')
            elif e['status']=='incomplete':
                if state['attempts'] or not state['request'] or e['reason']!='help_requested_before_submission': raise ValueError('invalid assessment interruption')
            else: raise ValueError('invalid assessment status')
            state['closed']=e['status']
        elif kind=='solution_reveal':
            if state['phase']=='assessment' or presentation['purpose']=='followup' or not state['hints'] or not state['last_grade'] or state['last_grade']['full_success'] is not False: raise ValueError('invalid reveal')
            if e['content_ref']!=task['task_id']+'/reference': raise ValueError('invalid reveal reference')
            state['revealed']=True
    if any(s['pending_grade'] is not None or not s['attempts'] or (s['presentation']['phase']=='assessment' and s['closed'] is None) for s in interactions.values()): raise ValueError('unfinished interaction')
    return True


def prepare(events):
    records, tasks, responses, queries = {},{},{},{}
    last_event,last_response,last_session={},{},{}
    for e in events:
        skill=e['skill_id']; kind=e['event_type']; interaction=e['interaction_id']
        rec=records.setdefault(skill,dict(learner_id=e['learner_id'],skill_id=skill,events=[],queries=[]))
        if kind=='task':
            tasks[interaction]=e
            q=dict(query_id=e['event_id'],phase=e['phase'],purpose=e['purpose'],history_end=len(rec['events']),
                   input=dict(target_task=task_features(TASK_INDEX[e['task_id']],e['reference_seconds']),since_last_event_seconds=None if skill not in last_event else e['available_at']-last_event[skill],since_last_response_seconds=None if skill not in last_response else e['available_at']-last_response[skill]),
                   target=None,target_exclusion='no_eligible_first_response')
            queries[interaction]=q; rec['queries'].append(q)
        task=tasks.get(interaction)
        features={k:None for k in FEATURE_FIELDS}
        features.update(event_type=kind,gap_seconds=None if skill not in last_event else e['available_at']-last_event[skill],new_session=None if skill not in last_session else e['session_id']!=last_session[skill])
        if task: features['task']=task_features(TASK_INDEX[task['task_id']],task['reference_seconds'])
        if kind=='response': responses[e['event_id']]=e; last_response[skill]=e['occurred_at']
        response=responses.get(e.get('response_id')) if kind=='grade' else e if kind=='response' else None
        for field in ('score','full_success','grading_status','hint_type'):
            if field in e: features[field]=e[field]
        if kind=='hint': features['hint_count']=e['hint_number']; features['assistance']='assisted'
        if kind=='solution_reveal': features['answer_revealed']=True; features['assistance']='assisted'
        if kind=='retry': features['attempt_number']=e['attempt_number']
        if response:
            for field in ('attempt_number','hint_count','answer_revealed','assistance'): features[field]=response[field]
            if kind=='response':
                duration=response['active_response_seconds']; ref=task['reference_seconds']
                features['active_response_seconds']=duration; features['relative_response_time']=duration/ref if duration is not None and ref else None
        if kind=='grade' and response['attempt_number']==1:
            q=queries[interaction]
            reason=('assisted_or_review_first_attempt' if response['phase']=='review' or response['assistance']=='assisted' else 'unknown_assistance' if response['assistance']=='unknown' else 'unresolved_grade' if e['grading_status']!='resolved' else None)
            q['target_exclusion']=reason
            if reason is None: q['target']=dict(full_success=int(e['full_success']),score=e['score'])
        rec['events'].append(features); last_event[skill]=e['available_at']; last_session[skill]=e['session_id']
    return records


def materialize_query(record,query):
    end=query['history_end']
    if type(end) is not int or not 0<=end<=len(record['events']): raise ValueError('invalid prefix')
    history=record['events'][:end]
    if any(set(e)!=FEATURE_FIELDS or (e['task'] is not None and set(e['task'])!=TASK_FIELDS) for e in history): raise ValueError('feature schema mismatch')
    if set(query['input'])!={'target_task','since_last_event_seconds','since_last_response_seconds'} or set(query['input']['target_task'])!=TASK_FIELDS: raise ValueError('query schema mismatch')
    return deepcopy(dict(history=history,**query['input']))


def training_examples(records):
    for record in records:
        for q in record['queries']:
            if q['phase']=='practice' and q['target'] is not None:
                yield materialize_query(record,q),q['target']['full_success']
