"""Numerical v2.1 simulation. Latent state never enters memory exports."""
import copy
import hashlib
import math
import random
from dataclasses import dataclass, asdict
from catalog import SKILLS, FORMATS, DIFFICULTIES, FORMAT_RULES, make_catalog, grade

VERSION = "2.1.0"
DAY = 86400.
PRACTICE_BLOCK_SIZE = 4
ORIGIN = 1794009600.
TASKS = make_catalog()
TASK_INDEX = {t['task_id']: t for t in TASKS}
POOLS = {(s,p): [t for t in TASKS if t['skill_id']==s and t['pool']==p] for s in SKILLS for p in ('practice','assessment')}


def rng_for(seed, identity, stream):
    return random.Random(int.from_bytes(hashlib.sha256(f'{VERSION}|{seed}|{identity}|{stream}'.encode()).digest()[:16],'big'))


def clamp(x,a,b): return max(a,min(b,x))
def sigmoid(x): return 1/(1+math.exp(-clamp(x,-40,40)))


def retained(k, days, half_life, mechanism='exponential'):
    if not (0<=k<=1 and days>=0 and half_life>0): raise ValueError('invalid retention')
    if mechanism=='exponential': return k*2**(-days/half_life)
    if mechanism=='power_law': return k/(1+days/half_life)
    raise ValueError('unknown mechanism')


def learned(k, rate, dose):
    if not (0<=k<=1 and rate>=0 and dose>=0): raise ValueError('invalid learning')
    return k+(1-k)*(-math.expm1(-rate*dose))


@dataclass(frozen=True)
class Config:
    seed: int = 2026110901
    tasks_per_skill: int = 16
    tasks_per_session: int = 12
    max_hints: int = 3
    max_retries: int = 2
    mechanism: str = 'exponential'
    long_gaps: bool = False
    spaced_practice: bool = False
    unresolved_probability: float = .03
    missing_time_probability: float = .04
    missing_reference_probability: float = .04
    missing_assistance_probability: float = .02
    pre_answer_request_probability: float = .03
    help_after_difficulty_probability: float = .75
    assessment_review_probability: float = .35
    assessment_days: tuple = (1,7,30)

    def __post_init__(self):
        if self.tasks_per_skill<=0 or self.tasks_per_session<=0: raise ValueError('positive budgets required')
        if type(self.max_hints) is not int or not 0<=self.max_hints<=4: raise ValueError('hint cap must be 0..4; author more hints to extend')
        if type(self.max_retries) is not int or not 0<=self.max_retries<=10: raise ValueError('retry cap must be 0..10')
        if self.mechanism not in ('exponential','power_law'): raise ValueError('unknown mechanism')
        for k,v in asdict(self).items():
            if k.endswith('_probability') and not 0<=v<=1: raise ValueError('invalid probability')
        if len(set(self.assessment_days))!=len(self.assessment_days) or any(d<=0 for d in self.assessment_days): raise ValueError('invalid assessment days')


class Student:
    def __init__(self, learner_id, config=None):
        self.learner_id, self.config = learner_id, config or Config()
        self.rng = {s:rng_for(self.config.seed,learner_id,s) for s in ('parameters','policy','tasks','outcomes','timing','missing','gaps')}
        self.clock, self.branch, self.session, self.opportunity = ORIGIN, 'practice', 0, 0
        self.events, self.trace, self.used_tasks, self.params, self.states = [], [], set(), {}, {}
        self.format_visits = {s:0 for s in SKILLS}
        self.format_offset = self.rng['tasks'].randrange(5)
        r = self.rng['parameters']
        base = dict(strength=r.betavariate(2,2), rate=r.uniform(.08,.28), half_life=math.exp(r.uniform(math.log(5),math.log(60))), speed=r.lognormvariate(0,.25))
        for skill in SKILLS:
            p = dict(initial_strength=clamp(base['strength']+r.uniform(-.2,.2),.02,.95), learning_rate=clamp(base['rate']*r.lognormvariate(0,.25),.04,.4), half_life_days=clamp(base['half_life']*r.lognormvariate(0,.35),2,120), speed=clamp(base['speed']*r.lognormvariate(0,.15),.4,2.5), help_effect=r.uniform(.6,1.8), retained_practice_fraction=0. if r.random()<.25 else r.uniform(.05,.5))
            self.params[skill]=p; self.states[skill]=(p['initial_strength'],self.clock)

    def advance(self,seconds):
        if not math.isfinite(seconds) or seconds<0: raise ValueError('invalid time')
        self.clock+=seconds

    def strength(self,skill):
        k,t=self.states[skill]
        return retained(k,(self.clock-t)/DAY,self.params[skill]['half_life_days'],self.config.mechanism)

    def emit(self,kind,skill,interaction,phase='practice',**fields):
        row=dict(schema_version=VERSION,event_id=f'{self.learner_id}/{self.branch}/e{len(self.events):05d}',learner_id=self.learner_id,skill_id=skill,course_id='python_basics_v2',session_id=f's{self.session}',branch=self.branch,interaction_id=interaction,event_type=kind,phase=phase,occurred_at=self.clock,available_at=self.clock)
        row.update(fields); self.events.append(row); return row

    def requested_lesson(self,skill):
        """Explicit entry point only; never called automatically by practice()."""
        interaction=f'lesson_{len(self.events)}'
        self.emit('lesson_request',skill,interaction,reason='learner_requested_intro')
        self.advance(45)
        self.emit('instruction',skill,interaction,reason='learner_requested_intro',content_ref=f'python_intro_v2/{skill}',teaching_action='explain')
        # No retained-state update from delivering content.

    def choose_task(self,skill,pool,fmt=None):
        bank=[t for t in POOLS[skill,pool] if t['task_id'] not in self.used_tasks and (fmt is None or t['task_type']==fmt)]
        if not bank: raise ValueError('task bank exhausted')
        return self.rng['tasks'].choice(bank)

    def sample_evidence(self,task,k,hints):
        fmt=task['task_type']; rules=FORMAT_RULES[fmt]; r=self.rng['outcomes']
        demand=(.25,.45,.65)[DIFFICULTIES.index(task['difficulty'])]+rng_for(self.config.seed,task['task_id'],'demand').uniform(-.12,.12)
        # Shared attempt noise correlates the criteria; each format has its own guessing process.
        z=5*(k-demand)+.65+r.gauss(0,.65)+rules['help_multiplier']*self.params[task['skill_id']]['help_effect']*math.log1p(hints)
        qs=[.97*sigmoid(z-.12*j) for j in range(len(task['criteria']))]
        if fmt=='single_choice':
            evidence=task['answer_key'] if r.random()<qs[0] else r.choice(task['options'])['id']
            probs=[qs[0]+(1-qs[0])/len(task['options'])]
        elif fmt=='multiple_select':
            correct=set(task['answer_key']); evidence=[]; probs=[]
            for o,q in zip(task['options'],qs):
                correct_decision=r.random()<(.5+.5*q); probs.append(.5+.5*q)
                if (o['id'] in correct)==correct_decision: evidence.append(o['id'])
        else:
            probs=[rules['guess']+(1-rules['guess'])*q for q in qs]
            evidence=[r.random()<p for p in probs]
        return evidence,probs,demand

    def attempt(self,task,interaction,number,hints,phase,learn=True,forced=None):
        skill=task['skill_id']; k=self.strength(skill)
        evidence,probs,demand=self.sample_evidence(task,k,hints)
        if forced is not None: evidence=forced
        truth=grade(task,evidence)
        duration=clamp(task['reference_seconds']*self.params[skill]['speed']*(1.25-.5*k)*self.rng['timing'].lognormvariate(0,.35)/(1+.12*hints),3,900)
        self.advance(duration)
        unknown=self.rng['missing'].random()<self.config.missing_assistance_probability
        response=self.emit('response',skill,interaction,phase,task_id=task['task_id'],attempt_number=number,
                           active_response_seconds=None if self.rng['missing'].random()<self.config.missing_time_probability else duration,
                           hint_count=None if unknown else hints,answer_revealed=None if unknown else False,
                           assistance='unknown' if unknown else ('assisted' if hints else 'independent'))
        before=self.strength(skill); after=before
        if learn:
            dose=FORMAT_RULES[task['task_type']]['practice_dose']*(self.params[skill]['retained_practice_fraction'] if hints else 1.)/number
            after=learned(before,self.params[skill]['learning_rate'],dose); self.states[skill]=(after,self.clock)
        self.advance(self.rng['timing'].choice((.1,2.,30.)))
        resolved=self.rng['missing'].random()>=self.config.unresolved_probability
        result=self.emit('grade',skill,interaction,phase,task_id=task['task_id'],response_id=response['event_id'],grading_status='resolved' if resolved else 'unresolved',
                         rubric_id=task['rubric_id'],grader='simulated_rubric_v2',grader_confidence=None,
                         evidence=evidence if resolved else None,**(truth if resolved else dict(criteria=None,score=None,full_success=None)))
        self.trace.append(dict(response_id=response['event_id'],skill_id=skill,probabilities=probs,hidden_demand=demand,evidence_truth=evidence,strength_at_attempt=k,strength_before=before,strength_after=after))
        return result

    def help(self,task,interaction,hints,phase,reason,requested=True):
        if hints>=self.config.max_hints: raise ValueError('hint budget exceeded')
        if requested: self.emit('help_request',task['skill_id'],interaction,phase,task_id=task['task_id'],reason='learner_request')
        spec=task['hints'][hints]
        self.advance(8+4*hints)
        self.emit('hint',task['skill_id'],interaction,phase,task_id=task['task_id'],hint_number=hints+1,hint_type=spec['kind'],reason='learner_request' if requested else reason,content_ref=f"{task['task_id']}/hint/{hints+1}")
        return hints+1

    def task(self,skill,task=None,purpose='ordinary',parent=None,assessment=False,allow_help=True,force_request=None,forced_attempts=None,force_hints=None):
        """Force arguments are deterministic test fixtures only."""
        if task is None:
            fmt=FORMATS[(self.format_visits[skill]+self.format_offset)%len(FORMATS)]
            self.format_visits[skill]+=1
            task=self.choose_task(skill,'assessment' if assessment else 'practice',fmt)
        if task['skill_id']!=skill or task['task_id'] in self.used_tasks: raise ValueError('wrong/reused task')
        self.used_tasks.add(task['task_id']); self.opportunity+=1
        interaction=f'{self.learner_id}/{self.branch}/q{self.opportunity:04d}'
        phase='assessment' if assessment else 'practice'
        reference=None if self.rng['missing'].random()<self.config.missing_reference_probability else task['reference_seconds']
        self.emit('task',skill,interaction,phase,task_id=task['task_id'],purpose='delayed' if assessment else purpose,parent_interaction_id=parent,reference_seconds=reference,max_hints=self.config.max_hints,max_retries=self.config.max_retries)
        hints=0; r=self.rng['policy']; requested=allow_help and self.config.max_hints>0 and (r.random()<self.config.pre_answer_request_probability if force_request is None else force_request)
        if requested:
            self.emit('help_request',skill,interaction,phase,task_id=task['task_id'],reason='learner_request')
            if assessment:
                self.emit('assessment_end',skill,interaction,phase,task_id=task['task_id'],status='incomplete',reason='help_requested_before_submission')
                phase='review'
                self.emit('mode_switch',skill,interaction,phase,task_id=task['task_id'],reason='requested_help')
            hints=self.help(task,interaction,hints,phase,'learner_request',requested=False)
        first_forced=forced_attempts[0] if forced_attempts else None
        result=self.attempt(task,interaction,1,hints,phase,learn=not assessment,forced=first_forced)
        if assessment and phase=='assessment':
            self.emit('assessment_end',skill,interaction,phase,task_id=task['task_id'],status='complete',reason='first_attempt_recorded')
            if allow_help and result['full_success'] is False and r.random()<self.config.assessment_review_probability:
                phase='review'; self.emit('mode_switch',skill,interaction,phase,task_id=task['task_id'],reason='post_assessment_review')
            else: return interaction,False
        for retry in range(1,self.config.max_retries+1):
            if not allow_help or result['full_success'] is not False: break
            if forced_attempts is None and r.random()>=self.config.help_after_difficulty_probability: break
            count=(force_hints[retry-1] if force_hints is not None and retry-1<len(force_hints) else r.choice((1,1,2)))
            for _ in range(min(count,self.config.max_hints-hints)):
                hints=self.help(task,interaction,hints,phase,'observed_difficulty',requested=r.random()<.5)
            self.emit('retry',skill,interaction,phase,task_id=task['task_id'],attempt_number=retry+1,reason='retry_after_difficulty')
            forced=forced_attempts[retry] if forced_attempts and retry<len(forced_attempts) else None
            result=self.attempt(task,interaction,retry+1,hints,phase,learn=not assessment,forced=forced)
        if allow_help and hints and result['full_success'] is False:
            self.emit('solution_reveal',skill,interaction,phase,task_id=task['task_id'],reason='practice_ended_after_difficulty',content_ref=task['task_id']+'/reference')
        return interaction, bool(hints)

    def practice(self,schedule):
        skills=list(SKILLS); self.rng['policy'].shuffle(skills)
        if schedule == 'blocked':
            size = PRACTICE_BLOCK_SIZE if self.config.spaced_practice else self.config.tasks_per_skill
            sequence = [
                skill
                for start in range(0, self.config.tasks_per_skill, size)
                for skill in skills
                for _ in range(min(size, self.config.tasks_per_skill - start))
            ]
        elif schedule=='mixed':
            sequence=[]
            for _ in range(self.config.tasks_per_skill):
                block=skills[:]; self.rng['policy'].shuffle(block); sequence+=block
        else: raise ValueError('unknown schedule')
        for i,skill in enumerate(sequence):
            if i%self.config.tasks_per_session==0:
                self.session+=1
                if i:
                    gap_rng = self.rng['gaps']
                    if self.config.long_gaps:
                        gap = gap_rng.choice((.25, 1, 3, 14, 60))
                    elif self.config.spaced_practice and gap_rng.random() < .30:
                        gap = math.exp(gap_rng.uniform(math.log(5), math.log(60)))
                    else:
                        gap = gap_rng.choice((.25, 1, 3))
                    self.advance(gap * DAY)
            interaction,assisted=self.task(skill)
            if assisted:
                self.advance(60)
                # New item, same skill, zero hints/retries; it does not recursively spawn more checks.
                self.task(skill,purpose='followup',parent=interaction,allow_help=False)
        return self

    def assessment(self,delay_days,skill):
        clone=copy.copy(self); clone.states=self.states.copy(); clone.used_tasks=self.used_tasks.copy(); clone.format_visits=self.format_visits.copy()
        clone.branch=f'assessment_{delay_days}d_{skill}'; clone.events=[]; clone.trace=[]
        clone.rng={s:rng_for(self.config.seed,self.learner_id+clone.branch,s) for s in self.rng}
        clone.advance(delay_days*DAY); clone.session+=1
        bank=POOLS[skill,'assessment'][:]; rng_for(self.config.seed,self.learner_id+skill,'assessment_bank').shuffle(bank)
        clone.task(skill,task=bank[list(self.config.assessment_days).index(delay_days)],assessment=True)
        return clone

    def private_record(self):
        return dict(learner_id=self.learner_id,config=asdict(self.config),parameters=self.params,final_states=self.states,trace=self.trace)
