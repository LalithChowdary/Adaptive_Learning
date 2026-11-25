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
    tutorDrawerOpen: false,
    openCitationKey: null,
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
    renderTutorChat();
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

    if (data && data.chat_by_skill && typeof data.chat_by_skill === "object") {
      const hydrated = {};
      Object.keys(data.chat_by_skill).forEach((sk) => {
        const list = Array.isArray(data.chat_by_skill[sk]) ? data.chat_by_skill[sk] : [];
        hydrated[sk] = list.map((m) => ({
          role: m.role,
          text: m.message || m.text || "",
          code: m.code_example || m.code || "",
          citations: Array.isArray(m.citations) ? m.citations : [],
          mastery_tier: m.mastery_tier || "",
          guardrail_status: m.guardrail_status || "",
          source: m.source || "",
        }));
      });
      state.chatBySkill = hydrated;
    } else if (!state.learner) {
      state.chatBySkill = {};
    }

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

      const prevSkill = state.skill;
      const res = await apiPost("/api/ppo_step", body);
      if (res && res.target_skill && res.target_skill !== prevSkill) {
        state.skill = res.target_skill;
        const meta = (state.data && state.data.skill_meta) || {};
        const toTitle = (meta[res.target_skill] && (meta[res.target_skill].title || meta[res.target_skill].name)) || res.target_skill;
        const fromTitle = (meta[prevSkill] && (meta[prevSkill].title || meta[prevSkill].name)) || prevSkill;
        state.workspaceNotice = Number(res.action_id) === 6
          ? `Reviewing prerequisite: ${toTitle} (to strengthen ${fromTitle})`
          : `Moved to topic: ${toTitle}`;
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
      setActiveTab("practice");
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
        const cmCitationsHtml = renderCitationsHtml(cm.citations, `cm-${state.skill}`);
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
          ${cmCitationsHtml}
        `;
      } else {
        cmBody.innerHTML = `<p class="brand-sub">Select a learner profile and a Python topic to view its lesson and worked example.</p>`;
      }
    }

    syncManualActionSelect(d);
    renderTutorChat();
    renderWorkspace(getActiveInteraction(d));
    renderCurve();
    renderGlassBox();
  }

  function syncManualActionSelect(d) {
    const sel = document.getElementById("manual-action-select");
    if (!sel) return;
    const baseLabels = {
      "0": "0: Give Easier Question",
      "1": "1: Give Same Difficulty",
      "2": "2: Give Harder Question",
      "3": "3: Explain Concept (Tutor Explains)",
      "4": "4: Show Worked Example (Tutor Trace)",
      "5": "5: Request Self-Explanation (You Explain)",
      "6": "6: Review Prerequisite",
      "7": "7: Advance Next Topic",
    };
    const ppo = (d && d.ppo) || {};
    const mask = Array.isArray(ppo.valid_mask) ? ppo.valid_mask : null;
    const reasons = ppo.mask_reasons || {};
    const actions = Array.isArray(ppo.actions) ? ppo.actions : [];
    const meta = (d && d.skill_meta) || {};
    Array.from(sel.options).forEach((opt) => {
      if (!opt.value || !(opt.value in baseLabels)) return;
      const aid = Number(opt.value);
      const isValid = mask ? Boolean(mask[aid]) : true;
      let label = baseLabels[opt.value];
      const actObj = actions[aid];
      const tgt = actObj && actObj.target && actObj.target.skill;
      if (isValid && (aid === 6 || aid === 7) && tgt) {
        const tgtName = (meta[tgt] && (meta[tgt].title || meta[tgt].name)) || tgt;
        label = `${label} (→ ${tgtName})`;
      } else if (!isValid) {
        const rsn = String(reasons[aid] || reasons[String(aid)] || "Unavailable").split(".")[0];
        label = `${label} — Locked (${rsn})`;
      }
      opt.textContent = label;
      opt.disabled = !isValid;
    });
    if (sel.value) {
      const selectedOpt = sel.options[sel.selectedIndex];
      if (selectedOpt && selectedOpt.disabled) sel.value = "";
    }
  }

  function renderInlineMd(escapedLine) {
    let s = escapedLine.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');
    s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    return s;
  }

  function renderMarkdown(rawText) {
    const text = String(rawText ?? "").replace(/\r\n/g, "\n");
    if (!text.trim()) return "";
    const parts = text.split(/```(?:python|py)?\n?([\s\S]*?)```/gi);
    const out = [];
    parts.forEach((seg, idx) => {
      if (idx % 2 === 1) {
        out.push(`<pre class="code-block" style="margin:6px 0">${esc(seg.trim())}</pre>`);
        return;
      }
      const lines = seg.split("\n");
      let listMode = null;
      const paraBuf = [];
      const flushPara = () => {
        if (paraBuf.length) {
          out.push(`<p>${paraBuf.map((l) => renderInlineMd(esc(l))).join("<br>")}</p>`);
          paraBuf.length = 0;
        }
      };
      const closeList = () => {
        if (listMode) {
          out.push(`</${listMode}>`);
          listMode = null;
        }
      };
      lines.forEach((line) => {
        const t = line.trim();
        if (!t) {
          flushPara();
          closeList();
          return;
        }
        const ulMatch = t.match(/^[-*•]\s+(.+)$/);
        const olMatch = t.match(/^\d+[.)]\s+(.+)$/);
        if (ulMatch) {
          flushPara();
          if (listMode !== "ul") {
            closeList();
            out.push("<ul>");
            listMode = "ul";
          }
          out.push(`<li>${renderInlineMd(esc(ulMatch[1]))}</li>`);
        } else if (olMatch) {
          flushPara();
          if (listMode !== "ol") {
            closeList();
            out.push("<ol>");
            listMode = "ol";
          }
          out.push(`<li>${renderInlineMd(esc(olMatch[1]))}</li>`);
        } else {
          closeList();
          paraBuf.push(t);
        }
      });
      flushPara();
      closeList();
    });
    return out.join("");
  }

  function renderCitationsHtml(citations, scopeKey) {
    if (!Array.isArray(citations) || !citations.length) return "";
    const shown = citations.slice(0, 4);
    let previewHtml = "";
    const chips = shown
      .map((c, idx) => {
        if (!c) return "";
        const shortLabel = c.short_label || c.section || c.label || c.title || "";
        if (!shortLabel) return "";
        const citKey = `${scopeKey || "chat"}:${idx}`;
        const isOpen = state.openCitationKey === citKey;
        if (isOpen) {
          const fullTitle = c.label || c.title || shortLabel;
          const excerpt = c.excerpt ? `<div style="margin-top:3px">${esc(c.excerpt)}</div>` : "";
          previewHtml = `<div class="citation-preview"><strong>${esc(fullTitle)}</strong>${excerpt}</div>`;
        }
        return `<button type="button" class="citation-chip ${isOpen ? "active" : ""}" data-cit-key="${esc(citKey)}" title="${esc(c.label || shortLabel)}">${esc(shortLabel)}</button>`;
      })
      .filter(Boolean)
      .join("");
    if (!chips) return "";
    return `<div class="citation-list" aria-label="Retrieved knowledge citations"><span class="citation-prefix">Sources:</span>${chips}</div>${previewHtml}`;
  }

  function getStarterPrompts() {
    const activeQ = getActiveInteraction(state.data);
    const hasOpenQ = Boolean(activeQ && activeQ.status === "open" && activeQ.question);
    if (hasOpenQ || state.activeTab === "practice") {
      return [
        "Walk me through the core rule step by step",
        "Give me a small toy example",
        "What common pitfalls trip students up here?",
        "How should I trace boundary values without giving the answer?",
      ];
    }
    return [
      "Walk me through the worked example step by step",
      "Summarize the mental model simply",
      "What common mistakes should I avoid here?",
    ];
  }

  function renderChatContainer(boxEl, chipsEl, msgs, scopePrefix) {
    if (chipsEl) {
      chipsEl.innerHTML = getStarterPrompts()
        .map((p) => `<button type="button" class="starter-chip" data-prompt="${esc(p)}" ${state.sendingChat ? "disabled" : ""}>${esc(p)}</button>`)
        .join("");
    }
    if (!boxEl) return;
    if (!msgs.length) {
      boxEl.innerHTML = `
        <div class="chat-msg chat-msg-assistant">
          <div class="chat-msg-role"><span>AI Tutor</span></div>
          <div class="chat-msg-body"><p>Have doubts on this topic? Pick a quick question above or ask anything below — explanations adapt automatically to your mastery level.</p></div>
        </div>`;
      return;
    }
    boxEl.innerHTML = msgs
      .map((m, idx) => {
        const isUser = m.role === "user";
        const codeHtml = !isUser && m.code
          ? `<pre class="code-block" style="margin:8px 0 0">${esc(m.code)}</pre>`
          : "";
        const tierBadge = !isUser && m.mastery_tier
          ? `<span class="tutor-meta-badge">tier: ${esc(m.mastery_tier)}</span>`
          : "";
        const sourceBadge = !isUser && m.source
          ? `<span class="tutor-meta-badge">${m.source === "llm" ? "Live LLM" : "Offline Guide"}</span>`
          : "";
        const guardBadge = !isUser && m.guardrail_status === "socratic_redirect"
          ? `<span class="tutor-meta-badge tutor-meta-guard">EduGuard: Socratic</span>`
          : "";
        const citHtml = !isUser ? renderCitationsHtml(m.citations, `${scopePrefix}-${state.skill}-${idx}`) : "";
        const bodyHtml = isUser ? `<p>${esc(m.text)}</p>` : renderMarkdown(m.text);
        return `
          <div class="chat-msg ${isUser ? "chat-msg-user" : "chat-msg-assistant"}">
            <div class="chat-msg-role">
              <span>${isUser ? "You" : "AI Tutor"}</span>
              ${tierBadge}
              ${sourceBadge}
              ${guardBadge}
            </div>
            <div class="chat-msg-body">${bodyHtml}</div>
            ${codeHtml}
            ${citHtml}
          </div>`;
      })
      .join("");
    boxEl.scrollTop = boxEl.scrollHeight;
  }

  function renderTutorChat() {
    const msgs = state.chatBySkill[state.skill] || [];
    const sendBtn = document.getElementById("tutor-chat-send-btn");
    const dSendBtn = document.getElementById("drawer-tutor-send-btn");
    [sendBtn, dSendBtn].forEach((b) => {
      if (b) {
        b.disabled = Boolean(state.sendingChat);
        b.textContent = state.sendingChat ? "Thinking..." : "Ask Tutor";
      }
    });

    const drawer = document.getElementById("tutor-drawer");
    const toggleBtn = document.getElementById("toggle-tutor-drawer-btn");
    if (drawer) drawer.classList.toggle("hidden", !state.tutorDrawerOpen);
    if (toggleBtn) toggleBtn.classList.toggle("btn-primary", Boolean(state.tutorDrawerOpen));
    const drawerSub = document.getElementById("drawer-tutor-subtitle");
    if (drawerSub && state.data) {
      const meta = (state.data.skill_meta || {})[state.skill] || {};
      const title = meta.title || meta.name || state.skill;
      const summ = ((state.data.skill_summaries || {})[state.skill]) || {};
      const tierStr = !summ.assessed
        ? "unassessed"
        : (summ.p_med || 0) < 0.70
          ? `lacking (${Math.round((summ.p_med || 0) * 100)}%)`
          : `mastered (${Math.round((summ.p_med || 0) * 100)}%)`;
      drawerSub.textContent = `${title} · Mastery: ${tierStr}`;
    }

    renderChatContainer(
      document.getElementById("tutor-chat-messages"),
      document.getElementById("tutor-starter-chips"),
      msgs,
      "inline"
    );
    renderChatContainer(
      document.getElementById("drawer-tutor-messages"),
      document.getElementById("drawer-starter-chips"),
      msgs,
      "drawer"
    );
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

    const topNotices = [
      state.workspaceError
        ? `<div class="feedback-banner feedback-fail" style="margin-bottom:12px"><strong>Notice:</strong> ${esc(state.workspaceError)}</div>`
        : "",
      state.workspaceNotice
        ? `<div class="hint-box" style="margin-bottom:12px">${esc(state.workspaceNotice)}</div>`
        : "",
    ]
      .filter(Boolean)
      .join("");

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
        ${topNotices}
        ${lessonHtml}
        <div class="practice-actions">
          <button id="lesson-continue-btn" class="btn btn-primary" type="button">Got it, next question →</button>
          <button id="lesson-full-btn" class="btn" type="button">Open Reference</button>
        </div>`;
      fArea.innerHTML = "";
      document.getElementById("lesson-continue-btn").addEventListener("click", () => requestNextStep());
      document.getElementById("lesson-full-btn").addEventListener("click", () => {
        const ref = document.getElementById("course-material-card");
        if (ref) ref.open = true;
        setActiveTab("learn");
      });
      return;
    }

    if (!active || !active.question) {
      modeBadge.textContent = "Ready";
      let extraMsg = "";
      if (state.revealedInfo) {
        const sol = Array.isArray(state.revealedInfo.solution)
          ? state.revealedInfo.solution.join(", ")
          : state.revealedInfo.solution;
        extraMsg += `<div class="hint-box"><strong>Solution:</strong> <code>${esc(sol)}</code>${state.revealedInfo.explanation ? `<br/><span>${esc(state.revealedInfo.explanation)}</span>` : ""}</div>`;
      }

      const ctaLabel = state.workspaceNotice ? "Continue to Question →" : "Start Practice →";
      qArea.innerHTML = `
        ${topNotices}
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
      ${topNotices}
      <p><strong>${esc(q.prompt)}</strong></p>
      ${snippetsHtml}
      ${inputHtml}
      <div class="practice-actions">
        ${(canSubmit || canRetry) ? `<button id="submit-ans-btn" class="btn btn-primary" type="button">${canSubmit ? "Submit" : `Submit Retry (${retryCount + 1}/2)`}</button>` : ""}
        ${(!hasPassed && hintsCount < 3) ? `<button id="hint-btn" class="btn" type="button">Hint (${hintsCount}/3)</button>` : ""}
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
        : (canRetry ? "Not quite — edit your answer and click Submit Retry, use a Hint, or Reveal Solution." : "Review the feedback or solution, then click Next → to continue.");
      feedbackHtml += `<div class="feedback-banner ${cls}"><strong>${lastRes.full_success ? "Correct" : "Needs Review"}</strong> (Score: ${esc(lastRes.score)}) — ${msg}</div>`;
    }
    if (state.revealedInfo) {
      const sol = Array.isArray(state.revealedInfo.solution)
        ? state.revealedInfo.solution.join(", ")
        : state.revealedInfo.solution;
      feedbackHtml += `<div class="hint-box"><strong>Solution:</strong> <code>${esc(sol)}</code>${state.revealedInfo.explanation ? `<br/><span>${esc(state.revealedInfo.explanation)}</span>` : ""}</div>`;
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
          if (!canSubmit && canRetry) {
            await apiPost("/api/retry", {
              learner: state.learner,
              interaction: active.id,
            });
          }
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
      el.innerHTML = `<p class="brand-sub" data-state="unassessed">This topic is currently unassessed. Complete at least one practice question to view your 0–60 day retention forecast.</p>`;
      return;
    }

    const byDiff = c.curves_by_difficulty || {};
    const easyPts = Array.isArray(byDiff.easy) ? byDiff.easy : [];
    const medPts = Array.isArray(byDiff.medium) ? byDiff.medium : [];
    const hardPts = Array.isArray(byDiff.hard) ? byDiff.hard : [];

    const probAtDay = (pts, targetDay, fallbackProb = 0.5) => {
      if (!pts.length) return fallbackProb;
      let best = pts[0];
      let bestDist = Math.abs(Number(best.day || 0) - targetDay);
      pts.forEach((p) => {
        const d = Math.abs(Number(p.day || 0) - targetDay);
        if (d < bestDist) {
          best = p;
          bestDist = d;
        }
      });
      return Number(best.probability ?? fallbackProb);
    };

    const summ = ((state.data && state.data.skill_summaries) || {})[state.skill] || {};
    const pNowEasy = Math.round(probAtDay(easyPts, 0, c.p_easy || 0.7) * 100);
    const pNowMed = Math.round(probAtDay(medPts, 0, summ.p_med || 0.5) * 100);
    const pNowHard = Math.round(probAtDay(hardPts, 0, c.p_hard || 0.35) * 100);
    const p30Med = Math.round(probAtDay(medPts, 30, summ.p_30d || 0.4) * 100);
    const p60Easy = Math.round(probAtDay(easyPts, 60, pNowEasy / 100) * 100);
    const p60Med = Math.round(probAtDay(medPts, 60, p30Med / 100) * 100);
    const p60Hard = Math.round(probAtDay(hardPts, 60, pNowHard / 100) * 100);
    const floorPct = Math.round((c.floor_b || 0) * 100);
    const drop30 = Math.max(0, pNowMed - p30Med);

    const unaidedPass = Number(summ.observed_unaided_passes || 0);
    const unaidedTot = Number(summ.observed_unaided_attempts || 0);
    const totalAtt = Number(summ.observed_attempts || 0);
    const obsMain = unaidedTot > 0 ? `${unaidedPass}/${unaidedTot} unaided` : `${totalAtt} attempt${totalAtt === 1 ? "" : "s"}`;
    const statusText = pNowMed >= 80 && floorPct >= 65
      ? "Strong & durable"
      : pNowMed < 70
        ? "Needs remediation (< 70%)"
        : drop30 > 15
          ? "Fragile — review after gap"
          : "Consolidating";

    badge.textContent = `Durable Floor: ${floorPct}%`;
    badge.className = "badge badge-lime";

    const w = 620;
    const h = 240;
    const padL = 44;
    const padR = 92;
    const padT = 20;
    const padB = 34;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const xOf = (d) => padL + ((d / 60) * plotW);
    const yOf = (p) => padT + ((1 - Math.min(1, Math.max(0, p))) * plotH);

    const pathFor = (pts) =>
      pts.map((pt, idx) => `${idx === 0 ? "M" : "L"}${xOf(pt.day).toFixed(1)},${yOf(pt.probability).toFixed(1)}`).join(" ");

    const floorY = yOf(c.floor_b || 0);
    const yTicks = [100, 75, 50, 25, 0]
      .map((pct) => {
        const y = yOf(pct / 100).toFixed(1);
        return `
          <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="#e0e5d9" stroke-width="${pct === 0 ? "1.2" : "0.8"}" stroke-dasharray="${pct === 0 ? "" : "2 3"}"/>
          <text x="${padL - 8}" y="${(Number(y) + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#687064">${pct}%</text>
        `;
      })
      .join("");

    const xTicks = [0, 15, 30, 45, 60]
      .map((day) => {
        const x = xOf(day).toFixed(1);
        const lbl = day === 0 ? "Today" : `Day ${day}`;
        return `
          <line x1="${x}" y1="${padT}" x2="${x}" y2="${h - padB}" stroke="#e0e5d9" stroke-width="0.7" opacity="0.6"/>
          <text x="${x}" y="${h - 10}" text-anchor="${day === 0 ? "start" : day === 60 ? "end" : "middle"}" font-size="11" fill="#687064">${lbl}</text>
        `;
      })
      .join("");

    const dotsFor = (pts, color) =>
      [0, 30, 60]
        .map((day) => {
          const p = probAtDay(pts, day, 0);
          return `<circle cx="${xOf(day).toFixed(1)}" cy="${yOf(p).toFixed(1)}" r="3.2" fill="${color}"/>`;
        })
        .join("");

    // De-overlap right-edge labels so nearby values stay legible
    const rawLabels = [
      { id: "easy", text: `Easy ${p60Easy}%`, y: yOf(p60Easy / 100), color: "#839f50" },
      { id: "floor", text: `Floor ${floorPct}%`, y: floorY, color: "#7058ba" },
      { id: "med", text: `Med ${p60Med}%`, y: yOf(p60Med / 100), color: "#293321" },
      { id: "hard", text: `Hard ${p60Hard}%`, y: yOf(p60Hard / 100), color: "#687064" },
    ].sort((a, b) => a.y - b.y);
    for (let i = 1; i < rawLabels.length; i++) {
      if (rawLabels[i].y - rawLabels[i - 1].y < 14) {
        rawLabels[i].y = rawLabels[i - 1].y + 14;
      }
    }
    const rightLabelsHtml = rawLabels
      .map(
        (lb) =>
          `<text x="${w - padR + 8}" y="${Math.min(h - padB, Math.max(padT + 8, lb.y + 4)).toFixed(1)}" font-size="11" font-weight="700" fill="${lb.color}">${lb.text}</text>`
      )
      .join("");

    const advice = pNowMed < 70
      ? "Medium recall is below 70%. Try Explain Concept or Easy/Medium unaided practice before moving on."
      : drop30 > 15
        ? `Retention drops ${drop30}% over 30 days. Complete a Self-Explanation or delayed check to raise your long-term floor.`
        : `Your retention stays steady across 60 days (long-term floor ${floorPct}%). Keep challenging yourself on Hard items or Advance Next Topic.`;

    el.innerHTML = `
      <div class="progress-kpis">
        <div class="kpi-card">
          <div class="kpi-label">Today (Medium)</div>
          <div class="kpi-value">${pNowMed}%</div>
          <div class="kpi-sub">Easy ${pNowEasy}% · Hard ${pNowHard}%</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">In 30 Days</div>
          <div class="kpi-value">${p30Med}%</div>
          <div class="kpi-sub">Estimated ${drop30}% forgetting drop</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Long-Term Floor</div>
          <div class="kpi-value">${floorPct}%</div>
          <div class="kpi-sub">Durable recall baseline (60d)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Observed Practice</div>
          <div class="kpi-value">${esc(obsMain)}</div>
          <div class="kpi-sub">${esc(statusText)}</div>
        </div>
      </div>

      <svg viewBox="0 0 ${w} ${h}" width="100%" style="max-height:260px;height:auto;display:block" role="img" aria-label="0 to 60 day retention curve">
        ${yTicks}
        ${xTicks}
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="#e0e5d9" stroke-width="1.2"/>
        <line x1="${padL}" y1="${floorY.toFixed(1)}" x2="${w - padR}" y2="${floorY.toFixed(1)}" stroke="#7058ba" stroke-dasharray="5 4" stroke-width="1.8"/>
        <path d="${pathFor(easyPts)}" fill="none" stroke="#839f50" stroke-width="2.2"/>
        <path d="${pathFor(medPts)}" fill="none" stroke="#293321" stroke-width="2.8"/>
        <path d="${pathFor(hardPts)}" fill="none" stroke="#687064" stroke-width="2.2"/>
        ${dotsFor(easyPts, "#839f50")}
        ${dotsFor(medPts, "#293321")}
        ${dotsFor(hardPts, "#687064")}
        ${rightLabelsHtml}
      </svg>

      <div class="curve-legend">
        <span class="legend-item"><span class="legend-swatch" style="background-color:#839f50"></span>Easy questions (${pNowEasy}% → ${p60Easy}%)</span>
        <span class="legend-item"><span class="legend-swatch" style="background-color:#293321"></span>Medium questions (${pNowMed}% → ${p60Med}%)</span>
        <span class="legend-item"><span class="legend-swatch" style="background-color:#687064"></span>Hard questions (${pNowHard}% → ${p60Hard}%)</span>
        <span class="legend-item"><span class="legend-swatch-dashed"></span>Long-term floor (${floorPct}%)</span>
      </div>

      <div class="progress-guide">
        <strong>How to read this chart:</strong> Each line predicts your probability of solving an unaided question correctly if you return <strong>0 to 60 days later</strong> without extra practice. <strong>What to do next:</strong> ${esc(advice)}
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
    const ragAudit = gb.rag_audit || (state.data && state.data.rag_audit) || {};
    const corpus = ragAudit.corpus || {};
    const lastRet = ragAudit.last_retrieval || null;
    const llmStatus = (state.data && state.data.llm_status) || {};
    const fallbacks = Array.isArray(llmStatus.fallback_providers) && llmStatus.fallback_providers.length
      ? llmStatus.fallback_providers.join(" -> ")
      : "none";
    body.innerHTML = `
      <div class="kv-row"><span>Token Dim</span><strong>${esc(audit.token_dim || 47)}</strong></div>
      <div class="kv-row"><span>Text/Code Embeddings</span><strong>${audit.uses_text_or_code_embeddings ? "YES" : "ZERO"}</strong></div>
      <div class="kv-row"><span>PPO Critic V(s_t)</span><strong>${ppo.critic_value !== undefined ? ppo.critic_value.toFixed(3) : "-"}</strong></div>
      <div class="kv-row"><span>Recommended Action</span><strong>${esc(ppo.action_name || "-")}</strong></div>
      <div class="kv-row"><span>Awaiting Question</span><strong>${ppo.awaiting_question ? "True" : "False"}</strong></div>
      <div class="kv-row"><span>LLM Primary / Fallback</span><strong>${esc(llmStatus.provider || "mock")} / ${esc(fallbacks)}</strong></div>
      ${ppo.rationale_plain ? `<p class="brand-sub" style="margin-top:10px">${esc(ppo.rationale_plain)}</p>` : ""}
      <div class="rag-audit-section">
        <div class="rag-audit-title">RAG &amp; Guardrails Audit</div>
        <div class="kv-row"><span>Curriculum / v2.2 / Episodic</span><strong>${esc(corpus.curriculum_chunks || 0)} / ${esc(corpus.v22_exemplars || 0)} / ${esc(corpus.episodic_learner || 0)}</strong></div>
        <div class="kv-row"><span>Last Retrieval</span><strong>${esc(lastRet ? `${lastRet.purpose} (${lastRet.returned || 0} hits, ${lastRet.latency_ms || 0}ms)` : "none yet")}</strong></div>
        <div class="kv-row"><span>FTS5 / Vector / RRF</span><strong>${lastRet ? `${lastRet.fts_hits} / ${lastRet.vector_hits} / ${lastRet.fused_candidates}` : "-"}</strong></div>
        ${lastRet && lastRet.rewritten_query ? `<p class="brand-sub" style="margin-top:6px"><strong>Rewritten Query:</strong> ${esc(lastRet.rewritten_query)}</p>` : ""}
      </div>
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

  async function handleSendChat(sourceInputId = "tutor-chat-input", promptOverride = null) {
    const input = document.getElementById(sourceInputId) || document.getElementById("tutor-chat-input");
    if (state.sendingChat) return;
    const text = (promptOverride !== null ? String(promptOverride) : (input ? input.value : "")).trim();
    if (!text) {
      if (input) input.focus();
      return;
    }
    const sk = state.skill;
    if (!state.chatBySkill[sk]) state.chatBySkill[sk] = [];
    const history = state.chatBySkill[sk].slice(-6);
    state.chatBySkill[sk].push({ role: "user", text });
    if (promptOverride === null && input) {
      input.value = "";
      input.style.height = "auto";
    }
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
        citations: Array.isArray(res.citations) ? res.citations : [],
        mastery_tier: res.mastery_tier || "",
        guardrail_status: res.guardrail_status || "",
        source: res.source || "",
      });
      if (res.rag_audit && state.data) {
        state.data.rag_audit = res.rag_audit;
        if (state.data.glass_box) state.data.glass_box.rag_audit = res.rag_audit;
        renderGlassBox();
      }
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

  function bindTutorInput(inputEl, btnEl, inputId) {
    if (btnEl) btnEl.addEventListener("click", () => handleSendChat(inputId));
    if (inputEl) {
      inputEl.addEventListener("input", () => {
        inputEl.style.height = "auto";
        inputEl.style.height = `${Math.min(inputEl.scrollHeight, 130)}px`;
      });
      inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          handleSendChat(inputId);
        }
      });
    }
  }

  bindTutorInput(document.getElementById("tutor-chat-input"), document.getElementById("tutor-chat-send-btn"), "tutor-chat-input");
  bindTutorInput(document.getElementById("drawer-tutor-input"), document.getElementById("drawer-tutor-send-btn"), "drawer-tutor-input");

  ["tutor-starter-chips", "drawer-starter-chips"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("click", (e) => {
      const btn = e.target.closest(".starter-chip");
      if (!btn || state.sendingChat) return;
      const prompt = btn.getAttribute("data-prompt");
      if (prompt) handleSendChat(id === "drawer-starter-chips" ? "drawer-tutor-input" : "tutor-chat-input", prompt);
    });
  });

  ["tutor-chat-messages", "drawer-tutor-messages", "course-material-body"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("click", (e) => {
      const chip = e.target.closest(".citation-chip");
      if (!chip) return;
      const key = chip.getAttribute("data-cit-key");
      state.openCitationKey = state.openCitationKey === key ? null : key;
      if (id === "course-material-body") {
        render();
      } else {
        renderTutorChat();
      }
    });
  });

  const toggleTutorBtn = document.getElementById("toggle-tutor-drawer-btn");
  if (toggleTutorBtn) {
    toggleTutorBtn.addEventListener("click", () => {
      state.tutorDrawerOpen = !state.tutorDrawerOpen;
      renderTutorChat();
    });
  }
  const closeTutorBtn = document.getElementById("close-tutor-drawer-btn");
  if (closeTutorBtn) {
    closeTutorBtn.addEventListener("click", () => {
      state.tutorDrawerOpen = false;
      renderTutorChat();
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
    state.chatBySkill = {};
    state.workspaceError = null;
    state.workspaceNotice = null;
    state.revealedInfo = null;
    state.activeLesson = null;
    await refreshState();
    const activeQ = getActiveInteraction(state.data);
    setActiveTab(activeQ && activeQ.question ? "practice" : "learn");
  });

  document.getElementById("ppo-next-btn").addEventListener("click", () => requestNextStep());

  const manualSelEl = document.getElementById("manual-action-select");
  const manualStepBtnEl = document.getElementById("manual-step-btn");
  if (manualSelEl && manualStepBtnEl) {
    manualSelEl.addEventListener("change", () => {
      manualStepBtnEl.classList.toggle("btn-primary", manualSelEl.value !== "");
    });
  }

  document.getElementById("manual-step-btn").addEventListener("click", () => {
    const val = manualSelEl ? manualSelEl.value : "";
    if (val === "") return;
    if (manualSelEl) manualSelEl.value = "";
    if (manualStepBtnEl) manualStepBtnEl.classList.remove("btn-primary");
    requestNextStep({ actionId: Number(val), policySource: "learner_manual" });
  });

  refreshState().catch(console.error);
})();
