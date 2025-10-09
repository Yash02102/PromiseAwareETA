# Promise-Aware ETA Project Plan

## Objectives
- Produce a reproducible public-data pipeline that quantifies delivery-time uncertainty per order.
- Evaluate calibrated quantile models that deliver reliable promise bands and actionable intervals.
- Optimize promise policies that minimize late-promise penalties subject to acceptable promise lengths.
- Provide artifacts (code, reports, simulations) suitable for a master's thesis and public release.

## Key Research Questions
- RQ1 (Calibration): Can calibrated quantile methods achieve nominal coverage with competitive interval widths?
- RQ2 (Policy Optimality): What promise quantiles minimize the composite late versus slow promise cost?
- RQ3 (Trade-offs): At fixed late-promise rates, do calibrated quantiles shorten average promise days compared with point estimates?
- RQ4 (Robustness): How resilient are calibration and policy choices to temporal drift and outliers?
- RQ5 (Fairness): Do policies introduce disparate lateness across regions or sellers, and can segment-aware policies mitigate this?

## Workstreams and Milestones
The timeline assumes a 12 week thesis schedule; adjust as constraints evolve.

### Phase 0 - Foundation (Week 0-1)
- Finalize literature scan, refine evaluation metrics, confirm hardware and tooling stack.
- Set up repository structure, coding standards, and data governance checklist.
- Deliverable: repository skeleton, project charter, initial task backlog.

### Phase 1 - Data Intake and Profiling (Week 1-2)
- Acquire Olist dataset, document provenance, validate schema, add data dictionary.
- Build reproducible ingestion scripts and run an exploratory analysis script for the target distribution.
- Deliverable: validated raw dataset snapshot and initial EDA report.

### Phase 2 - Feature Engineering and Label Construction (Week 3-4)
- Implement purchase-time features covering basket, geography, seller behavior, and seasonality signals.
- Produce leakage-safe historical aggregates and unit tests for feature builders.
- Deliverable: feature store snapshot, feature documentation, baseline label QA results.

### Phase 3 - Modeling and Calibration (Week 4-6)
- Train baseline quantile models (linear, hist gradient boosting, LightGBM) over temporal splits.
- Implement split-conformal and conformalized quantile regression calibration wrappers.
- Deliverable: experiment logs, calibrated quantile checkpoints, evaluation plots.

### Phase 4 - Policy Simulation and Optimization (Week 6-8)
- Build offline simulator computing late promise rate, average promise days, and composite cost.
- Optimize global and segment-aware quantiles for corridor distance, category, and seller dispatch behavior.
- Deliverable: policy selection report, recommended quantiles per segment, reproducible scripts.

### Phase 5 - Fairness, Drift, and Ablations (Week 8-10)
- Run disparity analysis for regional and seller lateness; test constrained optimization variants.
- Execute rolling-origin calibration updates, feature family ablations, and outlier handling study.
- Deliverable: fairness memo, robustness appendix, updated simulator artifacts.

### Phase 6 - Packaging and Write-up (Week 10-12)
- Consolidate findings into thesis chapters, craft reproducibility guide, prepare slides and poster.
- Finalize documentation, release candidate on GitHub, archive datasets and experiment configs.
- Deliverable: thesis draft sections, final code release, reproducibility checklist sign-off.

## Experiment Plan

| Stage | Window | Goal | Experiments | Outputs |
|-------|--------|------|-------------|---------|
| Baseline Quantiles | Week 4-5 | Establish benchmark pinball loss/coverage for LightGBM and simpler quantile models | 1. quantile_baseline_lightgbm (current config) 2. Linear quantile regression baseline 3. HistGradientBoosting quantile variant | Metrics table, MLflow runs, feature importance summary |
| Calibration Sweep | Week 5-6 | Achieve target coverage using conformalized quantile regression | 1. Split-conformal per t 2. Rolling-window calibration sensitivity 3. Residual-based recalibration ablation | Coverage vs. target plots, conformal interval artifacts |
| Policy Simulation | Week 6-7 | Translate model quantiles into promise policies | 1. Global t grid search 2. Segment-aware t by corridor distance/seller decile 3. Cost sensitivity lambda_late vs. lambda_slow | Simulator outputs (LPR, APD, Cost), policy recommendation memo |
| Robustness & Fairness | Week 7-8 | Stress test models against drift/outliers and measure disparity | 1. Rolling-origin training runs 2. Feature family ablations (drop basket, drop seller stats) 3. Fairness metrics by region/seller | Robustness appendix, disparity dashboard |

**Experiment Workflow**
1. Materialize features (make features) and record snapshot version/hash.
2. Launch experiment via trainer (make train-lightgbm or custom config) with MLflow logging enabled.
3. Capture metrics, artifacts, and config hash in experiments/ (YAML + README entry).
4. Summarize results in progress notes and update TASKS.md with follow-up actions.

**Key Metrics & Logging**
- Pinball loss per t, coverage error, interval width (stored in MLflow).
- Promise policy metrics: Late Promise Rate (LPR), Average Promise Days (APD), composite cost.
- Fairness gaps: ?LPR/?APD across geography, seller decile, category.
- Experiment metadata: runtime, feature snapshot version, git commit SHA.

**Dependencies & Tooling**
- Feature snapshot version recorded via FEATURE_SNAPSHOT_VERSION.
- MLflow local tracking URI (mlruns/), optional W&B toggle for collaborative runs.
- configs/experiments/*.yaml serve as canonical run configs; add new files per experiment stage.

## Cross Cutting Work
- **Experiment Tracking:** adopt a lightweight tracker (for example MLflow or Weights and Biases) with reproducible configs.
- **Testing and QA:** enforce unit tests for feature generation, calibration logic, and policy scoring.
- **Automation:** create make or just commands or CI jobs for linting, tests, and data validation.
- **Documentation:** maintain changelog, decision log, and regular progress notes in `docs/`.

## Task Backlog Snapshot
| Priority | Workstream | Task | Notes |
|----------|------------|------|-------|
| P0 | Phase 0 | Set up Python environment (`pyproject`, linting, formatter) | Needed before analysis scripts. |
| P0 | Phase 1 | Download and checksum Olist data into `data/raw/` | Ensure licensing text is stored. |
| P1 | Phase 1 | Draft data dictionary and EDA script template | Capture assumptions on missing values. |
| P1 | Phase 2 | Design seller dispatch feature aggregations with leakage guardrails | Requires temporal windows. |
| P1 | Phase 3 | Prototype quantile LightGBM baseline | Reuse scikit-learn API wrappers. |
| P2 | Phase 4 | Implement policy simulator skeleton with cost hooks | Align with evaluation metrics. |
| P2 | Phase 5 | Outline fairness metrics and reporting format | Coordinate with policy simulator outputs. |
| P3 | Phase 6 | Assemble reproducibility checklist | Finalize near release. |

Backlog items should migrate into GitHub issues or a task tracker once implementation begins.

## Tooling and Infrastructure Assumptions
- Language: Python 3.11 or newer managed via uv.
- Core libraries: pandas, numpy, scikit-learn, lightgbm, statsmodels, matplotlib or plotly, mlflow.
- Analysis scripts: organize Python modules and CLI entry points; use scripts or make targets for reproducible runs.
- CI: GitHub Actions with smoke tests for lint and unit tests once the environment is defined.

## Evaluation Metrics
- Predictive: pinball loss per quantile, optional CRPS, coverage versus target, interval width.
- Policy: late promise rate, average promise days, composite cost, service level constraint adherence.
- Fairness: disparity in late promise rate and average promise days across defined segments with statistical checks.

## Risks and Mitigations
- **Data Quality:** missing geolocation or inconsistent timestamps; add validation rules and fallback proxies.
- **Model Complexity:** high variance quantile estimates; add regularization, ensembling, and conformal smoothing.
- **Computation:** LightGBM training cost; leverage batch script runs on cloud compute or local GPU and cache features.
- **Reproducibility:** drift in dependencies; lock environments and store experiment configs.
- **Fairness Metrics:** sparse segments causing noisy disparity estimates; aggregate segments or apply shrinkage.

## Immediate Next Steps
1. Initialize the project environment using uv (`uv venv` + `uv pip install -e .[dev]`).
2. Import the Olist dataset and add a README in `data/` describing acquisition steps.
3. Stand up an initial EDA analysis script that profiles the delivery-time target and key covariates.
4. Configure lint and test tooling (for example Ruff plus pytest) and a baseline CI workflow.




