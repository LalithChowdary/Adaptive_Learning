(() => {
  "use strict";

  const state = {
    csrfToken: null,
    learner: null,
    skill: "range_boundaries",
    data: null,
    curve: null,
    glassBoxOpen: false,
    loadingSkill: null,
    revealedInfo: null,
    workspaceError: null,
    questionStartMs: null,
  };

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

  function render() {
    const d = state.data;
    if (!d) return;

    // Profiles dropdown
    const sel = document.getElementById("profile-select");
    sel.innerHTML = `<option value="">Select Learner...</option>` +
      (d.profiles || [])
        .map((p) => `<option value="${esc(p.id)}" ${p.id === state.learner ? "selected" : ""}>${esc(p.name)}</option>`)
        .join("");

    // LLM badge
    const llm = d.llm_status || {};
    document.getElementById("llm-status-badge").textContent =
      `LLM: ${llm.provider || "mock"} (${llm.is_live ? "live" : "offline"})`;

    // Skill DAG
    const dagEl = document.getElementById("skill-dag-list");
    const summaries = d.skill_summaries || {};
    const prereqs = d.prerequisites || {};
    const meta = d.skill_meta || {};
    dagEl.innerHTML = (d.skills || [])
      .map((sk) => {
        const s = summaries[sk];
        const assessed = s ? Boolean(s.assessed) : false;
        const pMed = s ? `${Math.round((s.p_med || 0) * 100)}%` : "unassessed";
        const floorB = s ? (s.floor_b || 0).toFixed(2) : "0.00";
        const preList = (prereqs[sk] || []).join(", ") || "Root";
        const floorText = assessed ? `floor b_t: ${floorB}` : `prior b_t: ${floorB} (unassessed)`;
        return `
          <button type="button" class="skill-node ${sk === state.skill ? "active" : ""}" data-skill="${esc(sk)}">
            <div class="skill-node-header">
              <span>${esc((meta[sk] && meta[sk].name) || sk)}</span>
              <span class="badge ${assessed ? "badge-lime" : ""}">${assessed ? pMed : "Unassessed"}</span>
            </div>
            <div class="skill-node-meta">Prereq: ${esc(preList)} · ${esc(floorText)}</div>
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
        render();
        try {
          if (state.learner) {
            await apiPost("/api/select_skill", { learner: state.learner, skill: state.skill });
          }
          await refreshState();
        } finally {
          state.loadingSkill = null;
          render();
        }
      });
    });

    // Learning Path & Course Material (with explicit loading screen while curating)
    const lp = d.learning_path;
    const cm = d.course_material;
    const cmBody = document.getElementById("course-material-body");
    document.getElementById("active-skill-badge").textContent = state.skill;

    if (state.loadingSkill) {
      const skName = (meta[state.loadingSkill] && (meta[state.loadingSkill].title || meta[state.loadingSkill].name)) || state.loadingSkill;
      document.getElementById("path-title").textContent = `Curating Study Path: ${skName}`;
      document.getElementById("path-summary").textContent = "";
      document.getElementById("path-steps").innerHTML = `
        <div class="loading-banner" role="status" aria-live="polite">
          <span class="loading-dot"></span>
          <span>Reading monotonic_gru memory state &amp; curating personalized Learn → Code → Remember path for ${esc(skName)}...</span>
        </div>`;
      cmBody.innerHTML = `
        <div class="loading-banner" role="status" aria-live="polite">
          <span class="loading-dot"></span>
          <span>Generating complete theory, pitfalls, and subgoal-labeled worked example for ${esc(skName)}...</span>
        </div>`;
    } else {
      document.getElementById("path-title").textContent = lp ? lp.title : "Personalized Learning Path";
      document.getElementById("path-summary").textContent = lp ? lp.summary : "Select a learner and skill to curate a path.";
      if (lp && lp.source) {
        document.getElementById("llm-status-badge").textContent =
          lp.source === "llm"
            ? `Study Path: Live LLM (${llm.model || "cloud"})`
            : `Study Path: Curriculum Engine`;
      }
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
        cmBody.innerHTML = `
          ${cm.title ? `<h3 style="margin-top:0">${esc(cm.title)}</h3>` : ""}
          <p>${esc(cm.concept_explanation)}</p>
          ${theoryHtml}
          <p><strong>Mental Model:</strong> ${esc(cm.mental_model)}</p>
          ${pitfallsHtml}
          ${we.code ? `<p><strong>${esc(we.title || "Worked Example")}:</strong></p><pre class="code-block">${esc(we.code)}</pre>` : ""}
          ${subgoals}
          ${quickCheckHtml}
        `;
      } else {
        cmBody.innerHTML = `<p class="brand-sub">Click a skill in the DAG to generate tailored course material.</p>`;
      }
    }

    // Active Question / Workspace
    renderWorkspace(d.active);

    // 0-60d Curve SVG
    renderCurve();

    // Glass-Box Inspector
    renderGlassBox();
  }

  function renderWorkspace(active) {
    const qArea = document.getElementById("question-area");
    const fArea = document.getElementById("feedback-area");
    const modeBadge = document.getElementById("workspace-mode-badge");
    if (!active || !active.question) {
      modeBadge.textContent = "Ready";
      qArea.innerHTML = `<p class="brand-sub">Click "Next PPO Step" to let delayed_retention_ppo choose the next pedagogical action.</p>`;
      let extraMsg = "";
      if (state.revealedInfo) {
        const sol = Array.isArray(state.revealedInfo.solution)
          ? state.revealedInfo.solution.join(", ")
          : state.revealedInfo.solution;
        extraMsg += `<div class="hint-box"><strong>Revealed Solution:</strong> <code>${esc(sol)}</code>${state.revealedInfo.explanation ? `<br/><span>${esc(state.revealedInfo.explanation)}</span>` : ""}</div>`;
      }
      if (state.workspaceNotice) {
        extraMsg += `<div class="hint-box">${esc(state.workspaceNotice)}</div>`;
      }
      if (state.workspaceError) {
        extraMsg += `<div class="hint-box"><strong>Notice:</strong> ${esc(state.workspaceError)}</div>`;
      }
      fArea.innerHTML = extraMsg;
      return;
    }

    if (!state.questionStartMs) state.questionStartMs = Date.now();
    const q = active.question;
    modeBadge.textContent = `${active.mode || "practice"} · ${q.task_type}`;

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

    qArea.innerHTML = `
      <p><strong>${esc(q.prompt)}</strong></p>
      ${snippetsHtml}
      ${inputHtml}
      <div class="topbar-controls">
        <button id="submit-ans-btn" class="btn btn-primary" type="button" ${canSubmit ? "" : "disabled"}>Submit</button>
        <button id="hint-btn" class="btn" type="button" ${(hasPassed || hintsCount >= 3) ? "disabled" : ""}>Hint (${hintsCount}/3)</button>
        <button id="retry-btn" class="btn" type="button" ${(!hasSubmitted || hasPassed || retryCount >= 2 || canSubmit) ? "disabled" : ""}>Retry (${retryCount}/2)</button>
        <button id="reveal-btn" class="btn" type="button" ${!hasSubmitted ? "disabled" : ""}>Reveal</button>
        <button id="finish-btn" class="btn btn-lime" type="button">Finish</button>
      </div>
      ${hintsHtml}
    `;

    let feedbackHtml = lastRes
      ? `<div class="hint-box"><strong>Score:</strong> ${esc(lastRes.score)} · <strong>Status:</strong> ${esc(lastRes.grading_status)} · ${lastRes.full_success ? "Passed" : "Needs Review"}</div>`
      : "";
    if (state.workspaceError) {
      feedbackHtml += `<div class="hint-box"><strong>Notice:</strong> ${esc(state.workspaceError)}</div>`;
    }
    fArea.innerHTML = feedbackHtml;

    document.getElementById("submit-ans-btn").addEventListener("click", async () => {
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

    document.getElementById("hint-btn").addEventListener("click", async () => {
      state.workspaceError = null;
      try {
        await apiPost("/api/hint", { learner: state.learner, interaction: active.id });
        await refreshState();
      } catch (err) {
        state.workspaceError = err.message || String(err);
        render();
      }
    });
    document.getElementById("retry-btn").addEventListener("click", async () => {
      state.workspaceError = null;
      try {
        await apiPost("/api/retry", { learner: state.learner, interaction: active.id });
        await refreshState();
      } catch (err) {
        state.workspaceError = err.message || String(err);
        render();
      }
    });
    document.getElementById("reveal-btn").addEventListener("click", async () => {
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
      badge.textContent = "Unassessed";
      el.innerHTML = `<p class="brand-sub">Skill is unassessed. Complete an item to view personalized 0–60d retention curves and permastore floor b_t.</p>`;
      return;
    }

    badge.textContent = `Assessed · floor b_t: ${(c.floor_b || 0).toFixed(3)}`;
    const w = 540;
    const h = 180;
    const pad = 28;
    const days = c.days || [0, 1, 3, 7, 14, 30, 45, 60];
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
    await refreshState();
  });

  document.getElementById("ppo-next-btn").addEventListener("click", async () => {
    if (!state.learner) return;
    state.workspaceError = null;
    state.workspaceNotice = null;
    state.revealedInfo = null;
    const qSource = (document.getElementById("question-source-select") || {}).value || "v2.2";
    try {
      const res = await apiPost("/api/ppo_step", {
        learner: state.learner,
        skill: state.skill,
        question_source: qSource,
      });
      if (res && res.type === "lesson") {
        state.workspaceNotice = `Primed Lesson Delivered (${res.action_name}). Review the Course Material above, then click Next PPO Step for your follow-up question.`;
      }
      state.questionStartMs = Date.now();
      await refreshState();
    } catch (err) {
      state.workspaceError = err.message || String(err);
      render();
    }
  });

  document.getElementById("manual-step-btn").addEventListener("click", async () => {
    if (!state.learner) return;
    const val = document.getElementById("manual-action-select").value;
    if (val === "") return;
    state.workspaceError = null;
    state.workspaceNotice = null;
    state.revealedInfo = null;
    const qSource = (document.getElementById("question-source-select") || {}).value || "v2.2";
    try {
      const res = await apiPost("/api/ppo_step", {
        learner: state.learner,
        skill: state.skill,
        action_id: Number(val),
        policy_source: "learner_manual",
        question_source: qSource,
      });
      if (res && res.type === "lesson") {
        state.workspaceNotice = `Primed Lesson Delivered (${res.action_name}). Review the Course Material above, then click Next PPO Step for your follow-up question.`;
      }
      state.questionStartMs = Date.now();
      await refreshState();
    } catch (err) {
      state.workspaceError = err.message || String(err);
      render();
    }
  });

  refreshState().catch(console.error);
})();
