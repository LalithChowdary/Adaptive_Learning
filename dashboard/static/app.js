'use strict';
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (p) => `${Math.round(p * 100)}%`;
const fmt = (s) => ({single_choice:'Single choice',multiple_select:'Multiple select',output_prediction:'Output prediction',fill_in_the_blank:'Fill in the blank',code_completion:'Code completion',mixed:'Mixed formats'}[s] || s);
const icons = ['↦','↻','Σ','⋔','ƒ','[ ]'];
let state, view='practice', learner=localStorage.getItem('recall.learner'), active=null, selected='range_boundaries', busy=false;
let settings={format:'mixed',difficulty:'easy',mode:'practice'};
let clockKey=null, clockSeconds=0, clockLast=performance.now();
function tick(){
  const now=performance.now();
  if(clockKey && view==='practice' && active?.status==='answering' && !active.closed && document.visibilityState==='visible' && document.hasFocus() && !busy){
    clockSeconds+=Math.min((now-clockLast)/1000,2);
    localStorage.setItem(clockKey,String(clockSeconds));
  }
  clockLast=now;
}
setInterval(tick,500);
function syncClock(){
  tick();
  const key=active && !active.closed && active.status==='answering' ? `recall.timer.${active.id}.${active.attempt}` : null;
  if(key!==clockKey){clockKey=key;clockSeconds=key?Number(localStorage.getItem(key)||0):0;clockLast=performance.now();}
}
async function api(path,body){
  const r=await fetch(`/api/${path}`,body?{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':state.token},body:JSON.stringify({...body,request_id:crypto.randomUUID()})}:{});
  const result=await r.json();if(!r.ok)throw new Error(result.error||'Something went wrong.');return result;
}
function error(e){$('#error').textContent=e.message||String(e);$('#error').hidden=false;}
function clearError(){$('#error').hidden=true;}
async function load(){
  state=await api(`state${learner?'?learner='+encodeURIComponent(learner):''}`);
  active=state.active||null;
  if(!learner && state.profiles.length){learner=state.profiles[0].id;localStorage.setItem('recall.learner',learner);return load();}
  $('#profile').innerHTML=state.profiles.length?state.profiles.map(p=>`<option value="${esc(p.id)}" ${p.id===learner?'selected':''}>${esc(p.name)}</option>`).join(''):'<option value="">Choose a profile</option>';
  render();
}
function heading(eyebrow,title,description,badge=''){return `<div class="page-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${description}</p></div>${badge?`<span class="micro-pill">${badge}</span>`:''}</div>`;}
function render(){
  syncClock();document.querySelectorAll('.nav').forEach(n=>n.classList.toggle('active',n.dataset.view===view));
  if(!learner){$('#content').innerHTML=`<div class="onboard"><div class="hero-art" aria-hidden="true">✳</div><div class="eyebrow">WELCOME TO RECALL</div><h1>Make learning<br>a little more visible.</h1><p>Answer real Python questions. Get a nudge when you need it. Explore what your memory model learns about your progress.</p><button class="primary" id="get-started">Create your profile ↗</button><div class="under-note">No account. No API key. Everything stays here.</div></div>`;$('#get-started').onclick=openProfile;return;}
  if(view==='practice')renderPractice();else if(view==='progress')renderProgress();else renderMemory();
}
function cards(interactive=true){return `<div class="skill-grid">${state.skills.map((s,i)=>{
  const p=state.progress.cards.find(c=>c.id===s.id);return `<${interactive?'button':'div'} class="skill-card ${interactive&&selected===s.id?'selected':''}" ${interactive?`data-skill="${s.id}" aria-pressed="${selected===s.id}"`:''}><div class="skill-top"><span class="skill-icon">${icons[i]}</span><span class="skill-number">${s.number}</span></div><h3>${esc(s.name)}</h3><p>${esc(s.description)}</p><div class="card-status"><span>${p.attempts?p.attempts+' graded attempts':'No recorded attempts'}</span><strong>${p.prediction===null?'Unassessed':pct(p.prediction)+' predicted'}</strong></div>${!interactive&&p.prediction!==null?`<div class="progress-bar"><span style="width:${p.prediction*100}%"></span></div><div class="quiet">${p.successes}/${p.independent} first unaided answers correct · ${p.hints} hints</div>`:''}</${interactive?'button':'div'}>`;
 }).join('')}</div>`;}
function renderPractice(){
  if(active){renderQuestion();return;}
  const name=state.profiles.find(p=>p.id===learner)?.name||'there';
  $('#content').innerHTML=heading('A LITTLE PROGRESS, EVERY DAY',`Ready when you are, ${esc(name)}.`,`Pick a skill. Try it on your own. See what you learn along the way.`,'6 skills to explore')+
  `<div class="hero"><div><div class="eyebrow">SMALL STEPS. REAL SIGNALS.</div><h2>Your next “I get it” starts here.</h2><p>Every independent attempt adds a little more evidence.<br>Hints are here when you need a different angle.</p></div><div class="hero-art" aria-hidden="true">✳</div></div><div class="section-head"><h3>What would you like to practice?</h3><span>01 — CHOOSE YOUR SKILL</span></div>${cards()}<div class="setup"><div class="field"><label for="format">Question format</label><select id="format"><option value="mixed">Mixed formats</option>${state.formats.map(f=>`<option value="${f}">${fmt(f)}</option>`).join('')}</select></div><div class="field"><label for="difficulty">Difficulty</label><select id="difficulty"><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option></select></div><div class="field"><label for="mode">Session type</label><select id="mode"><option value="practice">Practice · hints available</option><option value="assessment">Independent check</option></select></div><button class="primary" id="start">Start a question ↗</button></div><p class="under-note">No lessons before your first try. Each new question is one you haven't opened in this profile.</p>`;
  document.querySelectorAll('[data-skill]').forEach(b=>b.onclick=()=>{saveSettings();selected=b.dataset.skill;renderPractice();});
  for(const k of ['format','difficulty','mode'])$('#'+k).value=settings[k];
  $('#start').onclick=()=>act('start',{skill:selected,...saveSettings()});
}
function saveSettings(){for(const k of ['format','difficulty','mode'])if($('#'+k))settings[k]=$('#'+k).value;return settings;}
function draftKey(){return `recall.draft.${active.id}.${active.attempt}`;}
function readAnswer(){
  const q=active.question;
  if(q.task_type==='single_choice')return $('input[name="answer"]:checked')?.value??null;
  if(q.task_type==='multiple_select')return Array.from(document.querySelectorAll('input[name="answer"]:checked'),x=>x.value);
  if(q.task_type==='code_completion')return $('#code')?.value||'';
  return Array.from(document.querySelectorAll('.answer-field'),x=>x.value);
}
function saveDraft(){if($('#answer-form'))localStorage.setItem(draftKey(),JSON.stringify(readAnswer()));}
function restoreDraft(){
  let d;try{d=JSON.parse(localStorage.getItem(draftKey()));}catch{return;}if(d===null)return;
  if(active.question.task_type==='code_completion')$('#code').value=d;
  else if(['single_choice','multiple_select'].includes(active.question.task_type)){document.querySelectorAll('input[name="answer"]').forEach(e=>e.checked=Array.isArray(d)?d.includes(e.value):d===e.value);}
  else document.querySelectorAll('.answer-field').forEach((e,i)=>e.value=d?.[i]||'');
}
function renderQuestion(){
  const q=active.question;const skill=state.skills.find(s=>s.id===q.skill_id);const result=active.results.at(-1);const answering=active.status==='answering'&&!active.closed;
  let answers='';
  if(['single_choice','multiple_select'].includes(q.task_type))answers=`<div class="options">${q.options.map((o,i)=>`<label class="option"><input type="${q.task_type==='single_choice'?'radio':'checkbox'}" name="answer" value="${esc(o.id)}" ${answering?'':'disabled'}><span><span class="option-letter">${String.fromCharCode(65+i)}</span>${esc(o.text)}</span></label>`).join('')}</div>`;
  else if(q.task_type==='code_completion')answers=`<label class="answer-label" for="code">Your Python function</label><textarea class="code-editor" id="code" spellcheck="false" ${answering?'':'disabled'}>def solve(x):\n    # Write your solution here\n    pass</textarea><details class="quiet"><summary>Supported Python for these exercises</summary>One solve(x) function; assignments, arithmetic, comparisons, if/else, for loops, indexing and slicing. Calls: range, list, len, sum, min, max, abs. No imports or attribute access. Unsupported syntax is marked “not graded”.</details>`;
  else answers=Array.from({length:q.answer_count},(_,i)=>`<label class="answer-label" for="answer-${i}">${q.task_type==='output_prediction'?'Output for snippet':'Integer for blank'} ${i+1}</label><input class="answer-field" id="answer-${i}" placeholder="${q.task_type==='output_prediction'?'Enter the printed value…':'Enter the missing integer…'}" ${answering?'required':'disabled'} autocomplete="off">`).join('');
  let feedback='';
  if(result && !answering){
    const unresolved=result.grading_status==='unresolved';
    feedback=`<div class="feedback ${unresolved?'unresolved':result.full_success?'':'partial'}"><div class="eyebrow">${unresolved?'GRADING LIMITATION':result.assisted?'ASSISTED ATTEMPT':result.attempt===1?'FIRST UNAIDED ATTEMPT':'RETRY'}</div><h3>${unresolved?'This answer wasn’t graded.':result.full_success?'You got it.':result.score>0?'You’re part of the way there.':'A useful place to pause.'}</h3><p>${unresolved?esc(result.message):`${pct(result.score)} rubric credit. ${result.full_success?'Try a different question to check your understanding.':'Review your reasoning, use a hint, or try again.'}`}</p>${result.criteria?.length?`<div class="criteria">${result.criteria.map((p,i)=>`<span class="criterion ${p?'':'fail'}">${p?'✓':'○'} Criterion ${i+1}</span>`).join('')}</div>`:''}${result.details?.map(d=>`<div class="quiet">${esc(d.message)}</div>`).join('')||''}${active.assessment_incomplete?'<div class="quiet">Help was requested before submission. This independent check is incomplete; this result belongs to assisted review.</div>':''}</div>`;
  }
  $('#content').innerHTML=heading('ONE QUESTION AT A TIME',esc(skill.name),'Take a moment. Work through it. Your first attempt matters.',active.mode==='assessment'?'Independent check':active.mode==='review'?'Review practice':'Practice')+
  `<div class="question-grid"><section class="question-card"><div class="question-meta"><span class="tag">${fmt(q.task_type)}</span><span class="tag purple">${q.difficulty}</span><span class="tag">Attempt ${active.attempt} of ${active.max_retries+1}</span></div><h2>${esc(q.prompt)}</h2>${q.task_type!=='code_completion'?q.snippets.map((s,i)=>`<div class="quiet">${q.snippets.length>1?'Snippet '+(i+1):'PYTHON'}</div><pre><code>${esc(s)}</code></pre>`).join(''):''}<form id="answer-form">${answers}<div class="question-actions">${answering?'<button class="primary" type="submit">Check my answer ↗</button>':!active.closed?`${!result?.full_success&&active.attempt<=active.max_retries?'<button type="button" class="primary" id="retry">Try again ↻</button>':''}<button type="button" class="secondary" id="reveal">View solution</button><button type="button" class="primary" id="next">New question ↗</button>`:'<button type="button" class="primary" id="next">New question ↗</button>'}${answering?'<button type="button" class="text-button" id="leave">Leave question</button>':''}</div></form>${feedback}${active.revealed?`<div class="support-note">Reference solution · seeing this is recorded as assistance.</div><pre class="solution">${esc(active.solution)}</pre>`:''}</section>
  <aside class="side-stack"><section class="panel"><div class="panel-title">A nudge, when you need it <span>${active.hints.length}/${active.max_hints}</span></div>${active.hints.length?active.hints.map(h=>`<div class="hint-item"><small>HINT ${h.level}</small><p>${esc(h.text)}</p></div>`).join(''):'<div class="hint-empty" aria-hidden="true">✧</div><p>Try your own approach first. A small hint is here if you get stuck.</p>'}<button class="secondary hint-button" id="hint" ${active.closed||active.hints.length>=active.max_hints||result?.full_success?'disabled':''}>${active.hints.length?'Get the next hint':'Get a hint'} +</button><div class="quiet">${active.mode==='assessment'&&!result?'Requesting help switches this check to review. It will not count as an independent result.':'Hints are prepared for this skill. Each one is recorded separately.'}</div></section><section class="panel"><div class="panel-title">Before this question <span>◷</span></div>${active.prediction_before!==undefined?`<div class="prediction-number">${pct(active.prediction_before)}</div><p>Predicted chance of full success on the first unaided attempt.</p>`:'<p>Your prediction stays hidden until you submit, so you can focus on the question.</p>'}<div class="quiet">A prediction from the frozen memory model, trained on simulated learners. Not a mastery score.</div></section><section class="panel"><div class="panel-title">Your pace is your own</div><p>Time is estimated while this tab is visible and focused. Take the time you need.</p></section></aside></div>`;
  restoreDraft();$('#answer-form').addEventListener('input',saveDraft);
  $('#answer-form').onsubmit=(e)=>{e.preventDefault();saveDraft();tick();act('submit',{answer:readAnswer(),active_seconds:clockSeconds});};
  if($('#code'))$('#code').onkeydown=e=>{if(e.key==='Tab'){e.preventDefault();const t=e.target;const s=t.selectionStart;t.setRangeText('    ',s,t.selectionEnd,'end');saveDraft();}};
  $('#hint').onclick=()=>{saveDraft();act('hint');};
  if($('#retry'))$('#retry').onclick=()=>act('retry');
  if($('#reveal'))$('#reveal').onclick=()=>act('reveal');
  if($('#leave'))$('#leave').onclick=()=>act('finish',{},true);
  if($('#next'))$('#next').onclick=()=>{if(active.closed){selected=active.skill;active=null;render();}else act('finish',{},true);};
}
async function act(action,extra={},clear=false){
  if(busy)return;tick();busy=true;clearError();document.querySelectorAll('#content button, .nav, #profile, #new-profile').forEach(b=>b.disabled=true);
  try{
    const result=await api(action,{learner,interaction:active?.id,...extra});
    if(active)selected=active.skill;
    active=clear?null:result;
    const updated=await api('state?learner='+encodeURIComponent(learner));state=updated;render();if(action==='start'||action==='finish')window.scrollTo({top:0,behavior:'instant'});
  }catch(e){error(e);render();}
  finally{busy=false;document.querySelectorAll('.nav,#profile,#new-profile').forEach(b=>b.disabled=false);clockLast=performance.now();}
}
function renderProgress(){
  const p=state.progress;
  $('#content').innerHTML=heading('EVIDENCE, NOT GUESSWORK','Your progress, made visible.','What you’ve done and what the model predicts—kept clearly separate.')+
  `<div class="metrics"><div class="metric"><div class="eyebrow">GRADED ATTEMPTS</div><div class="value">${p.total_attempts}</div><small>Including assisted attempts and retries</small></div><div class="metric"><div class="eyebrow">FIRST UNAIDED SUCCESS</div><div class="value">${p.independent?`${p.successes}<span style="font-size:18px;color:#a2ab96"> / ${p.independent}</span>`:'—'}</div><small>Fresh questions answered without hints</small></div><div class="metric"><div class="eyebrow">SKILLS EXPLORED</div><div class="value">${p.cards.filter(c=>c.attempts).length}<span style="font-size:18px;color:#a2ab96"> / 6</span></div><small>Based on recorded graded attempts</small></div></div><div class="section-head"><h3>Your skill map</h3><span>OBSERVED + PREDICTED</span></div>${cards(false)}<p class="under-note">Predictions average five fixed medium-difficulty reference tasks per skill, one per format. They are not measured mastery.</p><section class="panel history-list"><div class="section-head"><h3>Your recent practice</h3><span>${p.history.length} QUESTIONS OPENED</span></div>${p.history.length?p.history.map(h=>`<div class="history-row"><div><strong>${esc(h.skill)}</strong><p>${fmt(h.format)}</p></div><div>${h.results.length?`${h.results.at(-1).score===null?'Not graded':pct(h.results.at(-1).score)+' credit'}`:h.status==='skipped'?'Skipped':'In progress'}<p>${h.hints} hints · ${h.results.length} attempts</p></div><div class="history-date">${new Date(h.at*1000).toLocaleDateString(undefined,{month:'short',day:'numeric'})}</div><span class="tag">${h.mode==='assessment'?'Independent check':h.mode==='review'?'Review':'Practice'}</span></div>`).join(''):'<div class="empty"><div class="empty-symbol">↗</div><h3>Your first attempt starts the story.</h3><p>Answer a question to see your real results here. Nothing is pre-filled or simulated.</p><button class="primary" id="go-practice">Try a question ↗</button></div>'}</section><button class="secondary export-button" id="export">Export my interaction data ↓</button><div class="quiet">Includes your answers, event times, grades, hints, and original pre-answer predictions. Saved as JSON.</div>`;
  if($('#go-practice'))$('#go-practice').onclick=()=>switchView('practice');$('#export').onclick=download;
}
async function download(){try{const data=await api('export?learner='+encodeURIComponent(learner));const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='recall-interactions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){error(e);}}
async function renderMemory(){
  $('#content').innerHTML=heading('MAKE TIME VISIBLE','What happens after a break?','Explore the same question at different time gaps. Your saved history stays untouched.','What-if exploration')+
  `<div class="memory-layout"><section class="panel"><div class="section-head"><h3>Predicted independent success</h3><span>0–60 DAYS</span></div><div class="settings-row"><label for="curve-skill" class="quiet">Skill</label><select id="curve-skill">${state.skills.map(s=>`<option value="${s.id}" ${s.id===selected?'selected':''}>${esc(s.name)}</option>`).join('')}</select></div><div id="curve"><div class="loading">Reading the memory state…</div></div></section><aside class="side-stack"><section class="panel"><div class="panel-title">One question. Different gaps.</div><p>The model uses your current evidence to predict a first unaided answer after each break.</p><div class="quiet">Fixed reference: medium difficulty, output prediction. No additional practice is assumed.</div></section><section class="panel"><div class="panel-title">A forecast, not a promise</div><p>The curve comes from a model trained on simulated histories. Real-world retention has not been validated.</p><div class="quiet">The architecture does not guarantee that every predicted curve decreases monotonically.</div></section></aside></div>`;
  $('#curve-skill').onchange=()=>{selected=$('#curve-skill').value;drawCurve();};await drawCurve();
}
async function drawCurve(){try{
  const skill=$('#curve-skill').value;const data=await api(`curve?learner=${encodeURIComponent(learner)}&skill=${skill}`);if(view!=='memory'||$('#curve-skill')?.value!==skill)return;
  const W=660,H=270,x=d=>45+d/60*585,y=p=>225-p*190;
  const line=data.points.map((p,i)=>`${i?'L':'M'}${x(p.day)},${y(p.probability)}`).join(' ');
  $('#curve').innerHTML=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted success over 60 days"><title>Predicted success over 60 days</title>${[0,.25,.5,.75,1].map(p=>`<line class="chart-grid" x1="45" x2="630" y1="${y(p)}" y2="${y(p)}"/><text x="32" y="${y(p)+4}" text-anchor="end">${Math.round(p*100)}%</text>`).join('')}<path d="${line} L630,225 L45,225 Z" fill="#dff1c4" opacity=".4"/><path class="chart-line" d="${line}"/>${data.points.map(p=>`<circle class="chart-point" cx="${x(p.day)}" cy="${y(p.probability)}" r="4"><title>Day ${p.day}: ${pct(p.probability)}</title></circle>`).join('')}${[0,7,14,30,45,60].map(d=>`<text x="${x(d)}" y="250" text-anchor="middle">${d===0?'Now':d+'d'}</text>`).join('')}</svg><div class="curve-values">${data.points.filter(p=>[0,7,30,60].includes(p.day)).map(p=>`<div class="curve-value"><small>${p.day?'After '+p.day+' days':'Right now'}</small><strong>${pct(p.probability)}</strong></div>`).join('')}</div><div class="quiet">${state.progress.cards.find(c=>c.id===skill).attempts?'Based on your recorded evidence.':'Unassessed skill: this is the shared model prior, not an estimate grounded in your answers.'} No events are added by viewing this chart.</div>`;
}catch(e){error(e);}}
function switchView(v){saveDraft();tick();view=v;clearError();render();window.scrollTo({top:0,behavior:'instant'});}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>switchView(b.dataset.view));
function openProfile(){$('#profile-dialog').showModal();$('#profile-name').focus();}
$('#new-profile').onclick=openProfile;$('#cancel-profile').onclick=()=>$('#profile-dialog').close();
$('#profile-form').onsubmit=async e=>{e.preventDefault();try{const r=await api('profile',{name:$('#profile-name').value});learner=r.learner;localStorage.setItem('recall.learner',learner);$('#profile-dialog').close();$('#profile-name').value='';view='practice';await load();}catch(e){error(e);}};
$('#profile').onchange=async()=>{saveDraft();learner=$('#profile').value;localStorage.setItem('recall.learner',learner);clearError();await load();};
window.addEventListener('beforeunload',()=>{saveDraft();tick();});
load().catch(async e=>{if(learner){learner=null;localStorage.removeItem('recall.learner');try{await load();return;}catch(err){error(err);}}else error(e);});
