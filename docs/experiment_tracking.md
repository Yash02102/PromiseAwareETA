# Experiment Tracking Options

## Candidates Considered
- **MLflow:** local-first runs, lightweight for offline experimentation, integrates with Python API.
- **Weights & Biases (W&B):** hosted SaaS, strong collaboration dashboards, requires account/network.
- **None:** rely on CSV logs and version control only (baseline fallback).

## Evaluation Criteria
| Criterion | MLflow | W&B | Notes |
|-----------|--------|-----|-------|
| Offline support | ? Local file store | ? Requires online unless self-hosted | Need offline option for restricted environments |
| Setup effort | ? Minimal (`pip install mlflow`, file path URI) | ? Requires account, API key | |
| Visualization | ? Basic UI | ? Rich dashboards | W&B excels, but MLflow acceptable |
| Cost | ? Open source | ? Usage-based plans | |
| Privacy | ? Full control on local | ? Data leaves machine | |

## Decision
- **Default tracker:** MLflow with local artifact store at `mlruns/`.
- Provide optional integration hooks for W&B when collaborators require richer dashboards.

## Action Items
1. Keep `DEFAULT_TRACKER` pointing to MLflow in `promise_aware_eta/experiment_tracking.py`.
2. Document environment variables to switch to W&B (e.g., `PROMISE_TRACKER=wandb`).
3. Add MLflow dependency (already in `pyproject.toml`), create setup instructions in docs when enabling experiments.
