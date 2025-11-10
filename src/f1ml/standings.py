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
    driver_form_last3: Dict[str, float]
    team_points: Dict[str, float]
    team_positions: Dict[str, int]
    team_form_last3: Dict[str, float]


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

    driver_points_history: Dict[str, List[Tuple[int, float]]] = {}
    team_points_history: Dict[str, List[Tuple[int, float]]] = {}

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
                driver_points_history.setdefault(driver_id, []).append((round_number, pts))
            if team_name:
                team_points[team_name] = team_points.get(team_name, 0.0) + pts
                team_points_history.setdefault(team_name, []).append((round_number, pts))

    driver_positions = _sort_points(driver_points)
    team_positions = _sort_points(team_points)

    def _last_three_sum(history: Dict[str, List[Tuple[int, float]]]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, values in history.items():
            values = sorted(values, key=lambda x: x[0])
            last_three = values[-3:]
            result[key] = float(sum(pt for _, pt in last_three))
        return result

    driver_form = _last_three_sum(driver_points_history)
    team_form = _last_three_sum(team_points_history)

    return StandingsSnapshot(
        driver_points=driver_points,
        driver_positions=driver_positions,
        driver_form_last3=driver_form,
        team_points=team_points,
        team_positions=team_positions,
        team_form_last3=team_form,
    )
