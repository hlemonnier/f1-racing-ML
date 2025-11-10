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
   poetry run f1ml ingest weekend --season 2023 --round 5
   ```
3. **Inspect stored data**
   - Raw parquet files under `data/raw/{season}/rXX-event-name/`.
   - Processed features (WIP) under `data/processed/`.

4. **Build basic features once race data is available**
   ```bash
   poetry run f1ml features basic-weekend --season 2023 --round 5
   ```

See `docs/mvp_plan.md` for the v0 scope and roadmap. More CLI commands will be added as the feature and modeling layers come online.
