"""Evaluation utilities for predictions vs actual race results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error


@dataclass
class RoundMetrics:
    season: int
    round_number: int
    mae: float
    spearman: float
    top5_hit_rate: float
    baseline_mae: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "season": self.season,
            "round": self.round_number,
            "mae": self.mae,
            "spearman": self.spearman,
            "top5_hit_rate": self.top5_hit_rate,
            "baseline_mae": self.baseline_mae,
        }


def _top5_hit_rate(pred_ranks: pd.Series, actual_positions: pd.Series) -> float:
    if pred_ranks.isna().all() or actual_positions.isna().all():
        return float("nan")
    pred_top5 = set(pred_ranks.nsmallest(5).index)
    actual_top5 = set(actual_positions.nsmallest(5).index)
    if not actual_top5:
        return float("nan")
    return len(pred_top5 & actual_top5) / 5.0


def evaluate_round_predictions(
    season: int,
    round_number: int,
    prediction_df: pd.DataFrame,
    actual_df: pd.DataFrame,
) -> RoundMetrics:
    merged = prediction_df.merge(
        actual_df[["DriverNumber", "target_position", "GridPosition"]],
        on="DriverNumber",
        how="inner",
    )
    merged = merged.dropna(subset=["target_position"])
    if merged.empty:
        raise ValueError(f"Missing actual results for season {season} round {round_number}.")

    mae = float(mean_absolute_error(merged["target_position"], merged["predicted_position"]))
    spearman = float(
        spearmanr(merged["target_position"], merged["predicted_position"]).correlation
    )
    hit_rate = _top5_hit_rate(merged["predicted_position"], merged["target_position"])

    if "GridPosition" in merged:
        baseline_mae = float(mean_absolute_error(merged["target_position"], merged["GridPosition"]))
    else:
        baseline_mae = float("nan")

    return RoundMetrics(
        season=season,
        round_number=round_number,
        mae=mae,
        spearman=spearman,
        top5_hit_rate=hit_rate,
        baseline_mae=baseline_mae,
    )
