(() => {
  "use strict";

  const ALL_FORMATS = [
    "single_choice",
    "multiple_select",
    "output_prediction",
    "fill_in_the_blank",
    "code_completion",
  ];

  const FORMAT_LABELS = {
    single_choice: "Single Choice",
    multiple_select: "Multiple Select",
    output_prediction: "Output Prediction",
    fill_in_the_blank: "Fill in the Blank",
    code_completion: "Code Completion",
    self_explanation: "Self-Explanation",
  };

  const state = {
    csrfToken: null,
    learner: null,
    skill: "range_boundaries",
    activeTab: "learn",
    data: null,
    curve: null,
    glassBoxOpen: false,
    loadingSkill: null,
    loadingStep: false,
    revealedInfo: null,
    workspaceError: null,
    workspaceNotice: null,
    activeLesson: null,
    chatBySkill: {},
    sendingChat: false,
    formatIndex: 0,
    questionStartMs: null,
  };

  function pickQuestionFormat() {
    const sel = document.getElementById("question-format-select");
    const val = sel ? sel.value : "auto";
    if (val && val !== "auto") return val;
    const fmt = ALL_FORMATS[state.formatIndex % ALL_FORMATS.length];
    state.formatIndex = (state.formatIndex + 1) % ALL_FORMATS.length;
    return fmt;
  }

  function esc(val) {
    return String(val ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function uuid() {
    if (window.crypto && window.crypto.randomUUID) {
      return window.crypto.randomUUID();
    }
    return "req-" + Math.random().toString(36).slice(2) + "-" + Date.now();
  }

  function getActiveInteraction(data) {
    return (data && (data.active || data.active_interaction)) || null;
  }

  function setActiveTab(tab) {
    if (!["learn", "practice", "progress"].includes(tab)) return;
    state.activeTab = tab;
    renderTabs();
  }

  function renderTabs() {
    ["learn", "practice", "progress"].forEach((t) => {
      const btn = document.getElementById(`tab-btn-${t}`);
      const panel = document.getElementById(`tab-panel-${t}`);
      const isActive = state.activeTab === t;
      if (btn) {
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-selected", isActive ? "true" : "false");
      }
      if (panel) {
        panel.classList.toggle("hidden", !isActive);
      }
    });
  }

  async function apiGet(path) {
    const res = await fetch(path, { headers: { Accept: "application/json" } });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
    return json;
  }

  async function apiPost(path, body) {
    if (!state.csrfToken) {
      const boot = await apiGet("/api/state");
      state.csrfToken = boot.csrf_token || boot.token || state.csrfToken;
    }
    const payload = Object.assign({ request_id: uuid() }, body || {});
    const res = await fetch(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": state.csrfToken || "",
      },
      body: JSON.stringify(payload),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
    return json;
  }

  async function refreshState() {
    const params = new URLSearchParams();
    if (state.learner) params.set("learner", state.learner);
    if (state.skill) params.set("skill", state.skill);
    const qs = params.toString() ? `?${params.toString()}` : "";
    const data = await apiGet(`/api/state${qs}`);
    state.csrfToken = data.csrf_token || data.token || state.csrfToken;
    state.data = data;

    if (state.learner && state.skill) {
      try {
        state.curve = await apiGet(
          `/api/curve?learner=${encodeURIComponent(state.learner)}&skill=${encodeURIComponent(state.skill)}`
        );
      } catch (_) {
        state.curve = null;
      }
    } else {
      state.curve = null;
    }
    render();
  }

  async function requestNextStep({ finishInteractionId = null, actionId = null, policySource = null } = {}) {
    if (!state.learner || state.loadingStep) return;
    state.loadingStep = true;
    state.workspaceError = null;
    state.workspaceNotice = null;
    state.revealedInfo = null;
    state.activeLesson = null;
    render();

    try {
      if (finishInteractionId) {
        await apiPost("/api/finish", { learner: state.learner, interaction: finishInteractionId });
        state.questionStartMs = null;
      }

      const qSource = (document.getElementById("question-source-select") || {}).value || "v2.2";
      const qFormat = pickQuestionFormat();
      const body = {
        learner: state.learner,
        skill: state.skill,
        question_source: qSource,
        format: qFormat,
      };
      if (actionId !== null && actionId !== undefined) {
        body.action_id = Number(actionId);
      }
      if (policySource) {
        body.policy_source = policySource;
      }

      const res = await apiPost("/api/ppo_step", body);
      if (res && res.target_skill && res.target_skill !== state.skill) {
        state.skill = res.target_skill;
        await apiPost("/api/select_skill", { learner: state.learner, skill: state.skill });
      }
      if (res && res.type === "lesson") {
        state.activeLesson = {
          mode: res.primed_mode || (res.action_id === 4 ? "example" : "concept"),
          material: res.lesson || (state.data && state.data.course_material) || null,
          actionName: res.action_name || "",
        };
      }
      state.questionStartMs = Date.now();
      await refreshState();
    } catch (err) {
      state.workspaceError = err.message || String(err);
    } finally {
      state.loadingStep = false;
      render();
    }
  }

  function render() {
    const d = state.data;
    if (!d) return;

    renderTabs();

    // Profiles dropdown
    const sel = document.getElementById("profile-select");
    sel.innerHTML = `<option value="">Select Learner...</option>` +
      (d.profiles || [])
        .map((p) => `<option value="${esc(p.id)}" ${p.id === state.learner ? "selected" : ""}>${esc(p.name)}</option>`)
        .join("");

    // Skill sidebar
    const dagEl = document.getElementById("skill-dag-list");
    const summaries = d.skill_summaries || {};
    const prereqs = d.prerequisites || {};
    const meta = d.skill_meta || {};

    const skillNameOf = (sk) => (meta[sk] && (meta[sk].title || meta[sk].name)) || sk;

    dagEl.innerHTML = (d.skills || [])
      .map((sk) => {
        const s = summaries[sk];
        const assessed = s ? Boolean(s.assessed) : false;
        const pMed = s ? `${Math.round((s.p_med || 0) * 100)}%` : "Not started";
        const preNames = (prereqs[sk] || []).map((p) => skillNameOf(p)).join(", ") || "Starting topic";
        return `
          <button type="button" class="skill-node ${sk === state.skill ? "active" : ""}" data-skill="${esc(sk)}">
            <div class="skill-node-header">
              <span>${esc(skillNameOf(sk))}</span>
              <span class="badge ${assessed ? "badge-lime" : ""}">${assessed ? pMed : "New"}</span>
            </div>
            <div class="skill-node-meta">${assessed ? "Mastery estimate" : "Prereq"}: ${assessed ? pMed : esc(preNames)}</div>
          </button>`;
      })
      .join("");

    dagEl.querySelectorAll(".skill-node").forEach((btn) => {
      btn.disabled = Boolean(state.loadingSkill);
      btn.addEventListener("click", async () => {
        if (state.loadingSkill) return;
        const chosen = btn.getAttribute("data-skill");
        state.skill = chosen;
        state.loadingSkill = chosen;
        state.workspaceError = null;
        state.workspaceNotice = null;
        state.revealedInfo = null;
        state.activeLesson = null;
        render();
        try {
          if (state.learner) {
            await apiPost("/api/select_skill", { learner: state.learner, skill: state.skill });
          }
          await refreshState();
          const activeQ = getActiveInteraction(state.data);
          setActiveTab(activeQ && activeQ.question ? "practice" : "learn");
        } finally {
          state.loadingSkill = null;
          render();
        }
      });
    });

    // Active skill header
    const activeSummary = summaries[state.skill];
    const activeAssessed = activeSummary ? Boolean(activeSummary.assessed) : false;
    const activePct = activeSummary ? `${Math.round((activeSummary.p_med || 0) * 100)}% Ready` : "Not started";
    document.getElementById("active-skill-title").textContent = skillNameOf(state.skill);
    document.getElementById("active-skill-subtitle").textContent = state.learner
      ? (activeAssessed ? "Continue practicing or review your retention progress." : "Read the quick lesson below, then start practicing.")
      : "Select or create a learner profile in the top bar to begin.";

    const skillBadge = document.getElementById("active-skill-badge");
    skillBadge.textContent = activeAssessed ? activePct : "Not started";
    skillBadge.className = `badge ${activeAssessed ? "badge-lime" : ""}`;

    // LLM / Study Path badge
    const lp = d.learning_path;
    const llm = d.llm_status || {};
    const llmBadge = document.getElementById("llm-status-badge");
    if (lp && lp.source === "llm") {
      llmBadge.textContent = `AI Tutor (${llm.model || "cloud"})`;
    } else {
      llmBadge.textContent = "Offline Curriculum";
    }

    // Learning Path & Course Material
    const cm = d.course_material;
    const cmBody = document.getElementById("course-material-body");

    if (state.loadingSkill) {
      const skName = skillNameOf(state.loadingSkill);
      document.getElementById("path-title").textContent = `Loading: ${skName}`;
      document.getElementById("path-summary").textContent = "";
      document.getElementById("path-steps").innerHTML = `
        <div class="loading-banner" role="status" aria-live="polite">
          <span class="loading-dot"></span>
          <span>Preparing personalized study plan for ${esc(skName)}...</span>
        </div>`;
      cmBody.innerHTML = `
        <div class="loading-banner" role="status" aria-live="polite">
          <span class="loading-dot"></span>
          <span>Loading lesson, common pitfalls, and worked example for ${esc(skName)}...</span>
        </div>`;
    } else {
      document.getElementById("path-title").textContent = lp ? lp.title : "Study Plan";
      document.getElementById("path-summary").textContent = lp ? lp.summary : "Select a learner and topic to begin.";
      document.getElementById("path-steps").innerHTML = lp && Array.isArray(lp.steps)
        ? lp.steps
            .map(
              (st) => `
              <div class="path-step phase-${esc(st.phase)}">
                <div class="badge">${esc(st.phase)}</div>
                <div><strong>${esc(st.title)}</strong></div>
                <div class="brand-sub">${esc(st.description)}</div>
              </div>`
            )
            .join("")
        : "";

      if (cm) {
        const we = cm.worked_example || {};
        const subgoals = (we.subgoals || [])
          .map((sg) => `<div class="subgoal-item"><strong>${esc(sg.label)}:</strong> ${esc(sg.detail)}</div>`)
          .join("");
        const theoryHtml = Array.isArray(cm.theory_sections) && cm.theory_sections.length
          ? `<div class="theory-grid">${cm.theory_sections
              .map(
                (sec) => `
                <div class="theory-section">
                  <h4>${esc(sec.heading)}</h4>
                  <p>${esc(sec.body)}</p>
                </div>`
              )
              .join("")}</div>`
          : "";
        const pitfallsHtml = Array.isArray(cm.pitfalls) && cm.pitfalls.length
          ? `<p><strong>Common Pitfalls to Avoid:</strong></p><ul class="pitfalls-list">${cm.pitfalls
              .map((p) => `<li>${esc(p)}</li>`)
              .join("")}</ul>`
          : "";
        const quickCheckHtml = cm.quick_check
          ? `<div class="hint-box"><strong>Quick Self-Check:</strong> ${esc(cm.quick_check)}</div>`
          : "";
        const adaptBanner = cm.adaptation_note
          ? `<div class="subgoal-item"><strong>Study Tip:</strong> ${esc(cm.adaptation_note)}</div>`
          : "";
        cmBody.innerHTML = `
          ${cm.title ? `<h3 style="margin-top:0">${esc(cm.title)}</h3>` : ""}
          ${adaptBanner}
          <p>${esc(cm.concept_explanation)}</p>
          ${theoryHtml}
          <p><strong>Mental Model:</strong> ${esc(cm.mental_model)}</p>
          ${pitfallsHtml}
          ${we.title ? `<p><strong>Worked Example:</strong> ${esc(we.title)}</p>` : ""}
          ${we.code ? `<pre class="code-block">${esc(we.code)}</pre>` : ""}
          ${subgoals}
          ${quickCheckHtml}
        `;
      } else {
        cmBody.innerHTML = `<p class="brand-sub">Select a learner profile and a Python topic to view its lesson and worked example.</p>`;
      }
    }

    renderTutorChat();
    renderWorkspace(getActiveInteraction(d));
    renderCurve();
    renderGlassBox();
  }

  function renderTutorChat() {
    const box = document.getElementById("tutor-chat-messages");
    const sendBtn = document.getElementById("tutor-chat-send-btn");
    if (!box) return;
    const msgs = state.chatBySkill[state.skill] || [];
    if (sendBtn) {
      sendBtn.disabled = Boolean(state.sendingChat);
      sendBtn.textContent = state.sendingChat ? "Thinking..." : "Ask Tutor";
    }
    if (!msgs.length) {
      box.innerHTML = `
        <div class="chat-msg chat-msg-assistant">
          <div class="chat-msg-role">AI Tutor</div>
          <div>Have any doubts about this lesson or worked example? Ask me here before you jump into practice!</div>
        </div>`;
      return;
    }
    box.innerHTML = msgs
      .map((m) => {
        const isUser = m.role === "user";
        const codeHtml = !isUser && m.code
          ? `<pre class="code-block" style="margin:8px 0 0">${esc(m.code)}</pre>`
          : "";
        return `
          <div class="chat-msg ${isUser ? "chat-msg-user" : "chat-msg-assistant"}">
            <div class="chat-msg-role">${isUser ? "You" : "AI Tutor"}</div>
            <div>${esc(m.text)}</div>
            ${codeHtml}
          </div>`;
      })
      .join("");
    box.scrollTop = box.scrollHeight;
  }

  function renderWorkspace(active) {
    const qArea = document.getElementById("question-area");
    const fArea = document.getElementById("feedback-area");
    const modeBadge = document.getElementById("workspace-mode-badge");

    if (!state.learner) {
      modeBadge.textContent = "Profile needed";
      qArea.innerHTML = `
        <div class="practice-empty">
          <p>Select or create a learner profile in the top bar to start practicing.</p>
        </div>`;
      fArea.innerHTML = "";
      return;
    }

    if (state.loadingStep) {
      modeBadge.textContent = "Loading...";
      qArea.innerHTML = `
        <div class="loading-banner" role="status" aria-live="polite">
          <span class="loading-dot"></span>
          <span>Preparing your next activity...</span>
        </div>`;
      fArea.innerHTML = "";
      return;
    }

    if (!active || !active.question) {
      if (state.activeLesson && state.activeLesson.material) {
        const mat = state.activeLesson.material;
        const isExample = state.activeLesson.mode === "example";
        modeBadge.textContent = isExample ? "Worked Example" : "Concept Lesson";

        let lessonHtml = "";
        if (isExample && mat.worked_example) {
          const we = mat.worked_example;
          const subgoals = (we.subgoals || [])
            .map((sg) => `<div class="subgoal-item"><strong>${esc(sg.label)}:</strong> ${esc(sg.detail)}</div>`)
            .join("");
          lessonHtml = `
            <p class="brand-sub">Review this step-by-step example before your next question:</p>
            ${we.title ? `<h3 style="margin:8px 0">${esc(we.title)}</h3>` : ""}
            ${we.code ? `<pre class="code-block">${esc(we.code)}</pre>` : ""}
            ${subgoals}
          `;
        } else {
          const theoryHtml = Array.isArray(mat.theory_sections) && mat.theory_sections.length
            ? `<div class="theory-grid">${mat.theory_sections
                .map(
                  (sec) => `
                  <div class="theory-section">
                    <h4>${esc(sec.heading)}</h4>
                    <p>${esc(sec.body)}</p>
                  </div>`
                )
                .join("")}</div>`
            : "";
          const pitfallsHtml = Array.isArray(mat.pitfalls) && mat.pitfalls.length
            ? `<p><strong>Watch out for:</strong></p><ul class="pitfalls-list">${mat.pitfalls.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>`
            : "";
          lessonHtml = `
            <p class="brand-sub">Concept lesson recommended before your next question:</p>
            ${mat.title ? `<h3 style="margin:8px 0">${esc(mat.title)}</h3>` : ""}
            <p>${esc(mat.concept_explanation)}</p>
            ${theoryHtml}
            ${mat.mental_model ? `<p><strong>Mental Model:</strong> ${esc(mat.mental_model)}</p>` : ""}
            ${pitfallsHtml}
          `;
        }

        qArea.innerHTML = `
          ${lessonHtml}
          <div class="practice-actions">
            <button id="lesson-continue-btn" class="btn btn-primary" type="button">Got it, next question →</button>
            <button id="lesson-full-btn" class="btn" type="button">Open Reference</button>
          </div>`;
        fArea.innerHTML = state.workspaceError
          ? `<div class="hint-box"><strong>Notice:</strong> ${esc(state.workspaceError)}</div>`
          : "";
        document.getElementById("lesson-continue-btn").addEventListener("click", () => requestNextStep());
        document.getElementById("lesson-full-btn").addEventListener("click", () => {
          const ref = document.getElementById("course-material-card");
          if (ref) ref.open = true;
          setActiveTab("learn");
        });
        return;
      }

      modeBadge.textContent = "Ready";
      let extraMsg = "";
      if (state.revealedInfo) {
        const sol = Array.isArray(state.revealedInfo.solution)
          ? state.revealedInfo.solution.join(", ")
          : state.revealedInfo.solution;
        extraMsg += `<div class="hint-box"><strong>Solution:</strong> <code>${esc(sol)}</code>${state.revealedInfo.explanation ? `<br/><span>${esc(state.revealedInfo.explanation)}</span>` : ""}</div>`;
      }
      if (state.workspaceNotice) {
        extraMsg += `<div class="hint-box">${esc(state.workspaceNotice)}</div>`;
      }
      if (state.workspaceError) {
        extraMsg += `<div class="hint-box"><strong>Notice:</strong> ${esc(state.workspaceError)}</div>`;
      }

      const ctaLabel = state.workspaceNotice ? "Continue to Question →" : "Start Practice →";
      qArea.innerHTML = `
        <div class="practice-empty">
          <p>${state.workspaceNotice ? "Ready for your follow-up question?" : "Ready to practice this topic with an adaptive question."}</p>
          <div class="practice-actions" style="justify-content:center">
            <button id="empty-start-btn" class="btn btn-primary" type="button">${ctaLabel}</button>
            <button id="empty-learn-btn" class="btn" type="button">Review Lesson</button>
          </div>
        </div>`;
      fArea.innerHTML = extraMsg;

      document.getElementById("empty-start-btn").addEventListener("click", () => requestNextStep());
      document.getElementById("empty-learn-btn").addEventListener("click", () => setActiveTab("learn"));
      return;
    }

    if (!state.questionStartMs) state.questionStartMs = Date.now();
    const q = active.question;
    modeBadge.textContent = FORMAT_LABELS[q.task_type] || q.task_type;

    let inputHtml = "";
    const isMultiSnippetText =
      (q.task_type === "output_prediction" || q.task_type === "fill_in_the_blank") &&
      Array.isArray(q.snippets) &&
      q.snippets.length > 0;

    if (q.task_type === "single_choice") {
      inputHtml = `<div class="option-list">` +
        (q.options || [])
          .map(
            (o) => `
            <label class="option-item">
              <input type="radio" name="sc-opt" value="${esc(o.id)}" />
              <span>${esc(o.text)}</span>
            </label>`
          )
          .join("") +
        `</div>`;
    } else if (q.task_type === "multiple_select") {
      inputHtml = `<div class="option-list">` +
        (q.options || [])
          .map(
            (o) => `
            <label class="option-item">
              <input type="checkbox" name="ms-opt" value="${esc(o.id)}" />
              <span>${esc(o.text)}</span>
            </label>`
          )
          .join("") +
        `</div>`;
    } else if (q.task_type === "code_completion") {
      const starter = (q.snippets && q.snippets[0]) || "def solve(x):\n    pass";
      inputHtml = `<textarea id="code-answer" class="answer-input code-editor">${esc(starter)}</textarea>`;
    } else if (isMultiSnippetText) {
      inputHtml = q.snippets
        .map(
          (snip, idx) => `
          <div style="margin-bottom:10px">
            <pre class="code-block">${esc(snip)}</pre>
            <input class="answer-input multi-text-answer" data-idx="${idx}" type="text" placeholder="Enter output for snippet ${idx + 1}..." />
          </div>`
        )
        .join("");
    } else {
      inputHtml = `<input id="text-answer" class="answer-input" type="text" placeholder="Enter your answer..." />`;
    }

    const snippetsHtml =
      q.task_type !== "code_completion" && !isMultiSnippetText && Array.isArray(q.snippets) && q.snippets.length
        ? `<pre class="code-block">${esc(q.snippets.join("\n"))}</pre>`
        : "";

    const hintsHtml = (active.hints || [])
      .map((h) => `<div class="hint-box"><strong>Hint ${esc(h.hint_level)} (${esc(h.hint_kind || h.kind)}):</strong> ${esc(h.text)}</div>`)
      .join("");

    const results = active.results || [];
    const lastRes = results.slice(-1)[0];
    const hasPassed = Boolean(lastRes && lastRes.full_success);
    const hasSubmitted = results.length > 0;
    const retryCount = Number(active.retry_count || 0);
    const hintsCount = (active.hints || []).length;
    const canSubmit = results.length <= retryCount;
    const canRetry = hasSubmitted && !hasPassed && retryCount < 2 && !canSubmit;

    qArea.innerHTML = `
      <p><strong>${esc(q.prompt)}</strong></p>
      ${snippetsHtml}
      ${inputHtml}
      <div class="practice-actions">
        ${canSubmit ? `<button id="submit-ans-btn" class="btn btn-primary" type="button">Submit</button>` : ""}
        ${(!hasPassed && hintsCount < 3) ? `<button id="hint-btn" class="btn" type="button">Hint (${hintsCount}/3)</button>` : ""}
        ${canRetry ? `<button id="retry-btn" class="btn" type="button">Retry (${retryCount}/2)</button>` : ""}
        ${(hasSubmitted && !hasPassed && !state.revealedInfo) ? `<button id="reveal-btn" class="btn" type="button">Reveal Solution</button>` : ""}
        ${hasSubmitted ? `<button id="next-question-btn" class="btn ${hasPassed || state.revealedInfo || !canRetry ? "btn-primary" : "btn-lime"}" type="button">Next →</button>` : ""}
        <button id="finish-btn" class="btn" type="button">Stop</button>
      </div>
      ${hintsHtml}
    `;

    let feedbackHtml = "";
    if (lastRes) {
      const cls = lastRes.full_success ? "feedback-pass" : "feedback-fail";
      const msg = lastRes.full_success
        ? "Nice job! Click Next → to continue."
        : (canRetry ? "Not quite — try again with Retry, use a Hint, or Reveal Solution." : "Review the feedback or solution, then click Next → to continue.");
      feedbackHtml += `<div class="feedback-banner ${cls}"><strong>${lastRes.full_success ? "Correct" : "Needs Review"}</strong> (Score: ${esc(lastRes.score)}) — ${msg}</div>`;
    }
    if (state.revealedInfo) {
      const sol = Array.isArray(state.revealedInfo.solution)
        ? state.revealedInfo.solution.join(", ")
        : state.revealedInfo.solution;
      feedbackHtml += `<div class="hint-box"><strong>Solution:</strong> <code>${esc(sol)}</code>${state.revealedInfo.explanation ? `<br/><span>${esc(state.revealedInfo.explanation)}</span>` : ""}</div>`;
    }
    if (state.workspaceError) {
      feedbackHtml += `<div class="hint-box"><strong>Notice:</strong> ${esc(state.workspaceError)}</div>`;
    }
    fArea.innerHTML = feedbackHtml;

    const submitBtn = document.getElementById("submit-ans-btn");
    if (submitBtn) {
      submitBtn.addEventListener("click", async () => {
        state.workspaceError = null;
        state.revealedInfo = null;
        let answer = null;
        if (q.task_type === "single_choice") {
          const checked = qArea.querySelector('input[name="sc-opt"]:checked');
          answer = checked ? checked.value : "";
        } else if (q.task_type === "multiple_select") {
          answer = Array.from(qArea.querySelectorAll('input[name="ms-opt"]:checked')).map((el) => el.value);
        } else if (q.task_type === "code_completion") {
          answer = document.getElementById("code-answer").value;
        } else if (isMultiSnippetText) {
          answer = Array.from(qArea.querySelectorAll(".multi-text-answer")).map((el) => el.value);
        } else {
          const val = document.getElementById("text-answer").value;
          answer = q.task_type === "self_explanation" ? val : [val];
        }
        const elapsed = Math.max(1.0, (Date.now() - (state.questionStartMs || Date.now())) / 1000);
        try {
          await apiPost("/api/submit", {
            learner: state.learner,
            interaction: active.id,
            answer,
            active_seconds: elapsed,
            self_explanation: q.task_type === "self_explanation" ? answer : undefined,
          });
          await refreshState();
        } catch (err) {
          state.workspaceError = err.message || String(err);
          render();
        }
      });
    }

    const hintBtn = document.getElementById("hint-btn");
    if (hintBtn) {
      hintBtn.addEventListener("click", async () => {
        state.workspaceError = null;
        try {
          await apiPost("/api/hint", { learner: state.learner, interaction: active.id });
          await refreshState();
        } catch (err) {
          state.workspaceError = err.message || String(err);
          render();
        }
      });
    }

    const retryBtn = document.getElementById("retry-btn");
    if (retryBtn) {
      retryBtn.addEventListener("click", async () => {
        state.workspaceError = null;
        try {
          await apiPost("/api/retry", { learner: state.learner, interaction: active.id });
          await refreshState();
        } catch (err) {
          state.workspaceError = err.message || String(err);
          render();
        }
      });
    }

    const revealBtn = document.getElementById("reveal-btn");
    if (revealBtn) {
      revealBtn.addEventListener("click", async () => {
        state.workspaceError = null;
        try {
          const res = await apiPost("/api/reveal", { learner: state.learner, interaction: active.id });
          state.revealedInfo = res;
          state.questionStartMs = null;
          await refreshState();
        } catch (err) {
          state.workspaceError = err.message || String(err);
          render();
        }
      });
    }

    const nextQBtn = document.getElementById("next-question-btn");
    if (nextQBtn) {
      nextQBtn.addEventListener("click", () => requestNextStep({ finishInteractionId: active.id }));
    }

    document.getElementById("finish-btn").addEventListener("click", async () => {
      state.workspaceError = null;
      state.revealedInfo = null;
      try {
        await apiPost("/api/finish", { learner: state.learner, interaction: active.id });
        state.questionStartMs = null;
        await refreshState();
      } catch (err) {
        state.workspaceError = err.message || String(err);
        render();
      }
    });
  }

  function renderCurve() {
    const el = document.getElementById("curve-container");
    const badge = document.getElementById("curve-status-badge");
    const c = state.curve;
    if (!c || !c.assessed) {
      badge.textContent = "Not started";
      badge.className = "badge";
      el.innerHTML = `<p class="brand-sub">Complete at least one practice question on this topic to view your 0–60 day retention forecast.</p>`;
      return;
    }

    badge.textContent = `Floor: ${Math.round((c.floor_b || 0) * 100)}%`;
    badge.className = "badge badge-lime";
    const w = 540;
    const h = 180;
    const pad = 28;
    const xOf = (d) => pad + ((d / 60) * (w - 2 * pad));
    const yOf = (p) => h - pad - (p * (h - 2 * pad));

    const pathFor = (pts) =>
      pts.map((pt, idx) => `${idx === 0 ? "M" : "L"}${xOf(pt.day).toFixed(1)},${yOf(pt.probability).toFixed(1)}`).join(" ");

    const byDiff = c.curves_by_difficulty || {};
    const floorY = yOf(Math.min(1, Math.max(0, c.floor_b || 0))).toFixed(1);

    el.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" width="100%" height="180" role="img" aria-label="0 to 60 day retention curve">
        <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#e0e5d9" stroke-width="1"/>
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="#e0e5d9" stroke-width="1"/>
        <line x1="${pad}" y1="${floorY}" x2="${w - pad}" y2="${floorY}" stroke="#7058ba" stroke-dasharray="4 3" stroke-width="1.5"/>
        <path d="${pathFor(byDiff.easy || [])}" fill="none" stroke="#839f50" stroke-width="2"/>
        <path d="${pathFor(byDiff.medium || [])}" fill="none" stroke="#293321" stroke-width="2.5"/>
        <path d="${pathFor(byDiff.hard || [])}" fill="none" stroke="#687064" stroke-width="2"/>
      </svg>
      <div class="curve-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:#839f50"></span>Easy</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#293321"></span>Medium</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#687064"></span>Hard</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#7058ba"></span>Long-term floor</span>
      </div>
    `;
  }

  function renderGlassBox() {
    const panel = document.getElementById("glass-box-panel");
    const layout = document.getElementById("main-layout");
    if (!state.glassBoxOpen) {
      panel.classList.add("hidden");
      layout.classList.remove("with-inspector");
      return;
    }
    panel.classList.remove("hidden");
    layout.classList.add("with-inspector");

    const body = document.getElementById("glass-box-body");
    const gb = (state.data && state.data.glass_box) || {};
    const ppo = gb.ppo || {};
    const audit = gb.token_audit || {};
    body.innerHTML = `
      <div class="kv-row"><span>Token Dim</span><strong>${esc(audit.token_dim || 47)}</strong></div>
      <div class="kv-row"><span>Text/Code Embeddings</span><strong>${audit.uses_text_or_code_embeddings ? "YES" : "ZERO"}</strong></div>
      <div class="kv-row"><span>PPO Critic V(s_t)</span><strong>${ppo.critic_value !== undefined ? ppo.critic_value.toFixed(3) : "-"}</strong></div>
      <div class="kv-row"><span>Recommended Action</span><strong>${esc(ppo.action_name || "-")}</strong></div>
      <div class="kv-row"><span>Awaiting Question</span><strong>${ppo.awaiting_question ? "True" : "False"}</strong></div>
      ${ppo.rationale_plain ? `<p class="brand-sub" style="margin-top:10px">${esc(ppo.rationale_plain)}</p>` : ""}
    `;
  }

  // Tab bindings
  ["learn", "practice", "progress"].forEach((t) => {
    const btn = document.getElementById(`tab-btn-${t}`);
    if (btn) btn.addEventListener("click", () => setActiveTab(t));
  });

  // Start Practice CTA buttons on Learn tab
  const startFromLearn = async () => {
    setActiveTab("practice");
    if (!state.learner) return;
    const activeQ = getActiveInteraction(state.data);
    if (!activeQ || !activeQ.question) {
      await requestNextStep();
    }
  };
  document.getElementById("start-practice-from-learn-btn").addEventListener("click", startFromLearn);
  const startAfterChatBtn = document.getElementById("start-practice-after-chat-btn");
  if (startAfterChatBtn) startAfterChatBtn.addEventListener("click", startFromLearn);

  async function handleSendChat() {
    const input = document.getElementById("tutor-chat-input");
    if (!input || state.sendingChat) return;
    const text = input.value.trim();
    if (!text) {
      input.focus();
      return;
    }
    const sk = state.skill;
    if (!state.chatBySkill[sk]) state.chatBySkill[sk] = [];
    const history = state.chatBySkill[sk].slice(-6);
    state.chatBySkill[sk].push({ role: "user", text });
    input.value = "";
    state.sendingChat = true;
    renderTutorChat();

    try {
      const res = await apiPost("/api/chat", {
        learner: state.learner || undefined,
        skill: sk,
        message: text,
        history,
      });
      state.chatBySkill[sk].push({
        role: "assistant",
        text: res.reply || "Here is a clarification based on the lesson.",
        code: res.code_example || "",
      });
    } catch (err) {
      state.chatBySkill[sk].push({
        role: "assistant",
        text: `Could not reach tutor: ${err.message || err}`,
      });
    } finally {
      state.sendingChat = false;
      renderTutorChat();
    }
  }

  const chatSendBtn = document.getElementById("tutor-chat-send-btn");
  if (chatSendBtn) chatSendBtn.addEventListener("click", handleSendChat);
  const chatInput = document.getElementById("tutor-chat-input");
  if (chatInput) {
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleSendChat();
      }
    });
  }

  document.getElementById("toggle-glass-box-btn").addEventListener("click", () => {
    state.glassBoxOpen = !state.glassBoxOpen;
    renderGlassBox();
  });

  async function handleCreateProfile() {
    const input = document.getElementById("new-profile-name");
    const btn = document.getElementById("create-profile-btn");
    const name = input.value.trim();
    if (!name) {
      input.focus();
      return;
    }
    btn.disabled = true;
    const origText = btn.textContent;
    btn.textContent = "Creating...";
    try {
      const res = await apiPost("/api/profile", { name });
      state.learner = res.learner;
      input.value = "";
      state.loadingSkill = state.skill;
      setActiveTab("learn");
      render();
      await apiPost("/api/select_skill", { learner: state.learner, skill: state.skill });
      await refreshState();
    } catch (err) {
      alert(`Could not create profile: ${err.message || err}`);
    } finally {
      state.loadingSkill = null;
      btn.disabled = false;
      btn.textContent = origText;
      render();
    }
  }

  document.getElementById("create-profile-btn").addEventListener("click", handleCreateProfile);
  document.getElementById("new-profile-name").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleCreateProfile();
    }
  });

  document.getElementById("profile-select").addEventListener("change", async (e) => {
    state.learner = e.target.value || null;
    state.workspaceError = null;
    state.workspaceNotice = null;
    state.revealedInfo = null;
    state.activeLesson = null;
    await refreshState();
    const activeQ = getActiveInteraction(state.data);
    setActiveTab(activeQ && activeQ.question ? "practice" : "learn");
  });

  document.getElementById("ppo-next-btn").addEventListener("click", () => requestNextStep());

  document.getElementById("manual-step-btn").addEventListener("click", () => {
    const val = document.getElementById("manual-action-select").value;
    if (val === "") return;
    requestNextStep({ actionId: Number(val), policySource: "learner_manual" });
  });

  refreshState().catch(console.error);
})();
