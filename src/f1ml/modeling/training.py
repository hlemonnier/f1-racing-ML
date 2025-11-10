"""Model training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from f1ml.config import get_project_paths
from f1ml.datasets import load_training_dataframe
from f1ml.modeling.artifacts import ModelArtifact, save_model_artifact
from f1ml.modeling.preprocessing import prepare_feature_matrix


def _default_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
    )


def train_season_model(
    season: int,
    upto_round: int,
    train_rounds: Optional[List[int]] = None,
) -> Tuple[Path, Dict[str, float]]:
    """Train a LightGBM regressor on specified rounds up to `upto_round`."""
    df = load_training_dataframe(season, upto_round, include_rounds=train_rounds)
    if "target_position" not in df.columns:
        raise ValueError("Training dataframe missing target_position column.")

    y = df["target_position"].astype(float).to_numpy()
    feature_df = df.drop(columns=["target_position"])
    X, categorical_mappings, feature_columns = prepare_feature_matrix(feature_df)

    model = _default_model()
    model.fit(X, y)

    preds = model.predict(X)
    mae = float(mean_absolute_error(y, preds))
    spearman = float(spearmanr(y, preds).correlation)

    training_rounds = sorted(df["round"].unique())
    artifact = ModelArtifact(
        season=season,
        trained_round=upto_round,
        feature_columns=feature_columns,
        categorical_mappings=categorical_mappings,
        training_rounds=training_rounds,
        n_samples=len(y),
        model=model,
    )

    models_dir = get_project_paths().models / str(season)
    artifact_path = models_dir / f"round_{upto_round:02d}_model.pkl"
    save_model_artifact(artifact, artifact_path)

    metrics = {
        "mae": mae,
        "spearman": spearman,
        "n_samples": float(len(y)),
    }
    return artifact_path, metrics
