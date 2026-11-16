"""Readable deterministic edge cases, excluded from stochastic cohorts."""
from pathlib import Path
from simulator import Student, Config, POOLS
from replay import validate_events, prepare
from storage import save_json


def clean_config(**kw):
    defaults=dict(tasks_per_skill=2,unresolved_probability=0,missing_time_probability=0,missing_reference_probability=0,missing_assistance_probability=0,pre_answer_request_probability=0,assessment_review_probability=0)
    defaults.update(kw); return Config(**defaults)


def task_for(fmt='output_prediction',skill='range_boundaries'):
    return next(t for t in POOLS[skill,'practice'] if t['task_type']==fmt and len(t['criteria'])>=2)


def fixtures():
    task=task_for(); bad=[False]*len(task['criteria']); good=[True]*len(task['criteria'])
    a=Student('fixture_three_hints',clean_config())
    parent,_=a.task(task['skill_id'],task=task,forced_attempts=[bad,bad,good],force_hints=[2,1])
    a.advance(60); a.task(task['skill_id'],purpose='followup',parent=parent,allow_help=False)
    b=Student('fixture_requested_help',clean_config())
    b.task(task['skill_id'],task=task,force_request=True,forced_attempts=[good])
    c=Student('fixture_interrupted_assessment',clean_config())
    c.task(task['skill_id'],task=task,assessment=True,force_request=True,forced_attempts=[good])
    d=Student('fixture_requested_lesson',clean_config())
    d.requested_lesson(task['skill_id']); d.task(task['skill_id'],task=task,forced_attempts=[good])
    e=Student('fixture_assessment_then_review',clean_config(assessment_review_probability=1))
    e.task(task['skill_id'],task=task,assessment=True,forced_attempts=[bad,good],force_hints=[1])
    result={s.learner_id:dict(events=s.events,records=list(prepare(s.events).values())) for s in (a,b,c,d,e)}
    for row in result.values(): validate_events(row['events'])
    return result


if __name__=='__main__':
    save_json(Path(__file__).parent/'examples/controlled_histories.json',fixtures())
