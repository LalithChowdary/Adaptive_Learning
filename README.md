# Adaptive Learning

Adaptive Learning System codebase combining the `monotonic_gru` cognitive memory model, `delayed_retention_ppo` reinforcement learning policy, and the `adaptive_web` interactive Python tutor.

## Quick Start (`adaptive_web`)

See [`adaptive_web/README.md`](adaptive_web/README.md) for full setup instructions, environment configuration, and a complete guide to navigating the web interface.

```bash
python -m adaptive_web.server --port 8780
```

Then open `http://127.0.0.1:8780` in your browser.

## Project Structure
- `adaptive_web/`: Interactive Adaptive Python Tutor web server, UI, and SQLite persistence
- `dashboard/`: Frontend/dashboard components
- `data_Set/`: Datasets and preprocessing (`v2.2` task catalog)
- `model/`: `monotonic_gru` model training, inference logic, and `requirements.txt`
- `ppo/`: `delayed_retention_ppo` RL environment, policy, and trained checkpoints
