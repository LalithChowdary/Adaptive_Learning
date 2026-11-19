# Adaptive Python Tutor (`3/adaptive_web/`)

State-of-the-art adaptive Python learning website combining the frozen **3-seed `delayed_retention_ppo` RL policy**, the frozen **`monotonic_gru` cognitive memory model** (47-dim structured evidence tokens + 19-dim task features), the **`v2.2` 2,160-item bank**, and **bounded LLM pedagogical realization** with a toggleable **Glass-Box Research Inspector**.

## Quick Start

```bash
PYTHONPATH=3 3/model/.venv/bin/python -m adaptive_web.server --port 8780
```

Then open `http://127.0.0.1:8780` in your browser.

## Optional Cloud LLM Configuration

By default, `adaptive_web` runs deterministically offline (`LLM_PROVIDER=mock` or `auto` without API keys). To enable live LLM path curation, course material generation, and fresh question authoring, copy `.env.example` to `.env` or set environment variables:

```bash
export LLM_PROVIDER=openai        # openai | anthropic | gemini | mock | auto
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini
```

## Architecture & Research Proposal

See [`PROPOSAL_PPO_LLM_ARCHITECTURE.md`](file:///Users/lalith/snu/sem7/minor_pro/code/3/adaptive_web/PROPOSAL_PPO_LLM_ARCHITECTURE.md) for the paper presentation comparison between Background PPO Suggestion and Hard PPO Action Control + Bounded LLM Realization (Gated Option B).
