"""Championship standings utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from f1ml.datasets import iter_feature_files


@dataclass
class StandingsSnapshot:
    driver_points: Dict[str, float]
    driver_positions: Dict[str, int]
    team_points: Dict[str, float]
    team_positions: Dict[str, int]


def _sort_points(points: Dict[str, float]) -> Dict[str, int]:
    sorted_entries = sorted(points.items(), key=lambda item: (-item[1], item[0]))
    ranks: Dict[str, int] = {}
    for idx, (entity, _) in enumerate(sorted_entries, start=1):
        ranks[entity] = idx
    return ranks


def build_standings(season: int, upto_round: int) -> StandingsSnapshot:
    """Aggregate driver/team points up to a given round (exclusive)."""
    driver_points: Dict[str, float] = {}
    team_points: Dict[str, float] = {}

    try:
        feature_files = list(iter_feature_files(season))
    except FileNotFoundError:
        feature_files = []

    for round_number, path in feature_files:
        if round_number >= upto_round:
            continue
        df = pd.read_parquet(path)
        if "Points" not in df.columns:
            continue

        for _, row in df.iterrows():
            driver_id = row.get("DriverId") or row.get("Abbreviation") or row.get("DriverNumber")
            team_name = row.get("TeamName")
            pts = float(row.get("Points", 0.0) or 0.0)

            if driver_id:
                driver_points[driver_id] = driver_points.get(driver_id, 0.0) + pts
            if team_name:
                team_points[team_name] = team_points.get(team_name, 0.0) + pts

    driver_positions = _sort_points(driver_points)
    team_positions = _sort_points(team_points)

    return StandingsSnapshot(
        driver_points=driver_points,
        driver_positions=driver_positions,
        team_points=team_points,
        team_positions=team_positions,
    )
