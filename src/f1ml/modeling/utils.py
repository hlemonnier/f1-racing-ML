"""Modeling helpers shared between training and prediction."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


_GRID_BASELINE_COLUMNS: Iterable[str] = ("GridPosition", "qualifying_position", "best_q_rank")


def baseline_grid_positions(df: pd.DataFrame) -> pd.Series:
    """Return a Series representing the best available starting order per driver."""
    if df.empty:
        raise ValueError("Cannot compute baseline grid positions for an empty dataframe.")

    baseline = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    for col in _GRID_BASELINE_COLUMNS:
        if col not in df.columns:
            continue
        candidate = pd.to_numeric(df[col], errors="coerce")
        baseline = baseline.fillna(candidate)

    if baseline.isna().all():
        baseline = pd.Series(range(1, len(df) + 1), index=df.index, dtype="float64")
    else:
        fallback = pd.Series(range(1, len(df) + 1), index=df.index, dtype="float64")
        baseline = baseline.fillna(fallback)

    return baseline.astype("float64")


def apply_grid_delta_predictions(
    baseline: pd.Series | np.ndarray,
    deltas: np.ndarray,
    max_position: int | None = None,
) -> np.ndarray:
    """Convert predicted deltas back into absolute finishing positions."""
    base_values = np.asarray(baseline, dtype="float64")
    if base_values.shape[0] != deltas.shape[0]:
        raise ValueError("Baseline and delta predictions must have the same length.")

    predictions = base_values + deltas
    upper_bound = max_position or max(int(base_values.max()), len(base_values))
    predictions = np.clip(predictions, 1.0, float(upper_bound))
    return predictions
