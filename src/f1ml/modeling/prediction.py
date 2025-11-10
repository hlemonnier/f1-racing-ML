"""Prediction helpers using trained artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from f1ml.config import get_project_paths
from f1ml.datasets import load_round_features
from f1ml.modeling.artifacts import load_model_artifact
from f1ml.modeling.preprocessing import prepare_feature_matrix


def _artifact_path(season: int, model_round: int) -> Path:
    return get_project_paths().models / str(season) / f"round_{model_round:02d}_model.pkl"


def predict_round_results(
    season: int,
    round_number: int,
    model_round: int,
    top_k: int = 5,
) -> Dict[str, Path | pd.DataFrame]:
    """Generate predictions for a round using a stored artifact."""
    artifact = load_model_artifact(_artifact_path(season, model_round))
    raw_features = load_round_features(season, round_number)

    feature_df = raw_features.drop(columns=["target_position"], errors="ignore")
    X, _, _ = prepare_feature_matrix(
        feature_df, categorical_mappings=artifact.categorical_mappings, expected_columns=artifact.feature_columns
    )
    predictions = artifact.model.predict(X)

    results = raw_features[["DriverNumber", "DriverId", "TeamName"]].copy()
    results["predicted_position"] = predictions
    results = results.sort_values("predicted_position").reset_index(drop=True)
    results["predicted_rank"] = results.index + 1

    reports_dir = get_project_paths().reports / str(season)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / f"r{round_number:02d}_predictions.csv"
    results.to_csv(output_path, index=False)

    summary = results.head(top_k).copy()
    return {"full_path": output_path, "summary": summary, "full_results": results}
