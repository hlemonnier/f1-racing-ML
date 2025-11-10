# F1 Race Outcome Predictor (MVP)

Pipeline that ingests F1 qualifying/race data via FastF1, engineers features, trains ranking models, and evaluates against a grid-order baseline.

## Quickstart

1. **Install Python 3.11/3.12 + Poetry** (pyarrow wheels are not available for 3.13 yet)
   ```bash
   # Example with pyenv
   pyenv install 3.12.2
   pyenv local 3.12.2
   poetry install
   ```
   If multiple Python versions are installed, point Poetry to the right one with `poetry env use 3.12`.
2. **Download a Grand Prix weekend**
   ```bash
   # By default FP1/FP2/FP3/(Sprint)/Quali/Race are all pulled
   poetry run f1ml ingest weekend --season 2023 --round 5
   # To limit to pre-race sessions (e.g. upcoming Las Vegas weekend)
   poetry run f1ml ingest weekend --season 2024 --event "Las Vegas" \
     --session FP1 --session FP2 --session FP3 --session SQ --session Q
   ```
3. **Inspect stored data**
   - Raw parquet files under `data/raw/{season}/rXX-event-name/`.
   - Processed features (WIP) under `data/processed/`.

4. **Build basic features once race data is available**
   ```bash
   poetry run f1ml features basic-weekend --season 2023 --round 5
   ```
   The resulting parquet includes best-lap metrics for FP/Sprint/Quali sessions plus long-run features from FP/Sprint race simulations (mean pace, variance, degradation slope, reliability streaks). After the race, the table gains the actual finishing positions for training.

5. **Train/update the model after each completed race**
   ```bash
   poetry run f1ml model train --season 2024 --until-round 6
   ```
   This fits a LightGBM regressor on all rounds ≤ `until-round` and stores the artifact under `models/{season}/round_XX_model.pkl`.

6. **Predict the next race once qualifying is done**
   ```bash
   # Uses the latest trained artifact by default (round-1)
   poetry run f1ml model predict --season 2024 --round 7 --top-k 5
   ```
   A full ranking CSV lands in `reports/{season}/rXX_predictions.csv`, and the CLI prints the requested Top-K snapshot.

7. **(Optional) Bulk-sync an entire season**
   ```bash
   poetry run f1ml pipeline sync-season --season 2025 --round-start 1 --round-end 21
   ```
   This loops through each round, pulls the requested sessions (FP/Sprint/Q/R by default), and builds the processed feature files so training can consume the whole season without manual repetition.

8. **Backtest round-by-round improvements**
   ```bash
   poetry run f1ml pipeline backtest --season 2025 --round-start 5 --round-end 21
   ```
   For every GP it trains on all prior races, predicts the current one, and logs MAE/Spearman/top-K metrics to `reports/{season}/backtest_round_metrics.csv` so you can track performance drift across the season.

### Optional data sources
- **Previous season auto-fetch**: `pipeline sync-season` automatically downloads season `N-1` when you fetch season `N`, so the feature builder can inject prior-year Quali/Race metrics without extra manual steps. Disable with `--no-include-prev-year` if needed.
- **Weather forecasts**: set `OPENWEATHER_API_KEY` in your environment to let the pipeline cache forecasts per Grand Prix (`data/external/weather/...`). Without a key, neutral default values are used.

See `docs/mvp_plan.md` for the v0 scope and roadmap. More CLI commands will be added as the feature and modeling layers come online.
