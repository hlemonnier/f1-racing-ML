"""Model artifact definitions and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib


@dataclass
class ModelArtifact:
    season: int
    trained_round: int
    feature_columns: List[str]
    categorical_mappings: Dict[str, Dict[str, int]]
    training_rounds: List[int]
    n_samples: int
    model: Any


def save_model_artifact(artifact: ModelArtifact, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_model_artifact(path: Path) -> ModelArtifact:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return joblib.load(path)
