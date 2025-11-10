## F1 Race Outcome Predictor – v0 Delivery Plan

### 1. Product Slice (v0)
- **Input**: structured data for a given Grand Prix weekend (Q1/Q2/Q3 session summaries, final qualifying order, actual starting grid, driver/team/circuit metadata).
- **Outputs**  
  1. Predicted race classification before lights out (full 1–20 ranking).  
  2. Actual classification after the checkered flag.  
  3. Automatic error report (metric deltas, biggest misses, feature impact, to-fix backlog).  
  4. Updated model artifact and dataset snapshot to use for the next round.

### 2. Guardrails & Assumptions
- Season replay only progresses forward in time—no leakage from future GPs.
- No weather/tyre/pit/piecewise-safety-car signals in v0; rely exclusively on qualifying-derived metrics plus static metadata.
- Exclude laps flagged invalid by FastF1; ignore late penalties/SC incidents except when they show up in final classification.
- Baseline comparator = starting grid order (MAE, Spearman ρ, Top-10 hit rate, points MAE).

### 3. Logical Architecture
```
┌────────────┐  ingest   ┌──────────────┐  features  ┌───────────┐  train   ┌────────────┐
│ FastF1 API │ ────────▶ │ Bronze Store │ ─────────▶ │ Feature   │ ───────▶ │ Model       │
└────────────┘           │ (raw parquet)│            │ Store     │          │ Registry    │
                         └──────────────┘            └───────────┘          └────────────┘
                                 ▲                                         │
                                 │            eval/report                   │
                                 └──────────▶ Orchestration ◀───────────────┘
                                                (per GP)
```
- **Bronze store**: raw FastF1 pull per session, timestamped snapshot.
- **Feature store (silver)**: engineered qualifying-derived dataset keyed by `season_gp + driver_id`.
- **Model registry (gold)**: serialized regressors + metadata (hyperparams, metrics, training window).
- **Orchestrator**: CLI/NB script that runs pre-race prediction and post-race learning/reporting.

### 4. Data Ingestion
1. `fastf1.get_session(season, gp_name, "Q")` and `"R"` to fetch qualifying and race sessions.  
2. Normalize laps table → filter `Lap.isValid`. Aggregate per driver:
   - Best lap per session (Q1/Q2/Q3).
   - Session rank (FastF1’s `Session.results` or derived).  
3. Extract official starting grid (after penalties) via race session metadata.  
4. Race results: final classified order, points, DNFs (FastF1 status).  
5. Persist raw pulls to `data/raw/{season}/{gp}_{session}.parquet` for reproducibility.

### 5. Feature Engineering (v0)
Columns per `driver-weekend`:
- `session_rank_q1/q2/q3`, `best_lap_q1/q2/q3` (seconds).
- `delta_qX_median`, `delta_qX_fastest`.
- `grid_position` (after penalties).
- `quali_position` (official).  
- Driver & constructor categorical encodings (target or ordinal).  
- `track_id` (one-hot or embedding).  
- Rolling seasonal aggregates (prior races only): average qual rank, finish rank, DNFs, points — computed per driver/team.
- Optional interaction: `track_driver_perf = rolling_delta_on_track`.

Implementation detail: assemble via pandas, store engineered dataset at `data/processed/features_<season>.parquet`.

### 6. Modeling Strategy
- **Baseline**: predicted order equals starting grid order.
- **Model v0**: regression predicting finishing position (float). Candidates:
  - Gradient Boosted Trees (LightGBM or XGBoost via `lightgbm.sklearn.LGBMRegressor`).
  - Regularized linear model (Ridge/Lasso) for interpretable baseline.
- Post-prediction: sort predicted positions ascending to form ranking; enforce unique order by tie-breaking with predicted delta.
- Feature importance from tree model (gain/SHAP) to feed error report.
- Persist artifacts under `models/{season}/model_v{n}.pkl` plus JSON metadata.

### 7. Evaluation
Per GP (rolling test set of size 1):
- `MAE_position`: mean |pred_pos - actual_pos|.
- `Spearman_rho`: ranking correlation between predicted and actual.
- `Top10_hit_rate`: proportion of drivers predicted inside actual Top10.
- `MAE_points`: convert predicted ranking to points using FIA scale, compare to actual.
- Always compute same metrics for baseline grid order to track uplift.

Cross-season dashboard: accumulate metrics per GP and cumulative averages, stored in `reports/metrics_{season}.csv`.

### 8. Automated Post-Mortem
- Highlight top 5 largest absolute errors with context (driver, team, predicted vs actual, DNF flag).
- Feature importance snapshot for the GP (SHAP or permutation on hold-out).
- Buckets: `DNF`, `SC-proxy`, `penalty` when status codes indicate unusual finish.
- “Next steps” queue derived from recurring issues (e.g., chronic underestimation on street circuits).
- Output Markdown + JSON at `reports/{season}/{gp_slug}_report.(md|json)`.

### 9. Incremental Learning Loop
1. **Pre-race**: train on all completed GPs → freeze model → emit predictions for upcoming weekend.
2. **Post-race**: append new GP data to processed dataset, recompute rolling aggregates, retrain (or warm-start boosting model), log new version.
3. Track version metadata (dataset hash, training window) for reproducibility.
4. Tooling suggestion: `mlflow` or lightweight YAML ledger in `models/registry.yaml`.

### 10. Tooling & Stack
- Python 3.11 + Poetry for deps.
- Core libs: `fastf1`, `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `scipy`, `shap`, `mlflow` (optional).
- Storage: parquet files via `pyarrow`.
- Orchestration: CLI entrypoints under `src/pipeline/`:
  - `prepare_gp.py` (ingest + feature build for new GP).
  - `predict_gp.py` (load latest model and emit ranking CSV).
  - `evaluate_gp.py` (compare predictions vs actual, update report, trigger retrain).
- Testing: `pytest` covering feature builders and evaluation metrics.

### 11. Implementation Roadmap
1. **Scaffold project**: Poetry init, src layout (`src/f1ml/...`), config module, logging, data dirs.
2. **Data ingestion module**: wrappers around FastF1 with caching + CLI to dump raw parquet.
3. **Feature pipeline**: transform raw parquet → engineered dataset with rolling aggregates.
4. **Baseline + model training**: implement grid baseline metrics, add LightGBM model, serialization.
5. **Evaluation & reports**: metric computation + Markdown generator vs baseline.
6. **Incremental training loop**: CLI orchestrator that runs predict → evaluate → retrain.
7. **Automation & docs**: README with workflow, make targets (e.g., `make predict GP=Monza2024`), add future-work backlog (track factors, reliability, weather, tyres, SC probabilities).

### 12. Future Enhancements (v1 → v2)
- Learnable circuit embeddings (“track factors”).
- Reliability priors per driver/team (lambda for DNFs).
- Weather categorical input; pit-loss heuristics; SC probability model.
- Monte Carlo race simulator feeding reinforcement learning ranking adjustments.

