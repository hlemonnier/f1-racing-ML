"""Weekend-level feature builders."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from f1ml.config import get_project_paths
from f1ml.io import write_parquet


@dataclass(frozen=True)
class WeekendRawPaths:
    base: Path
    q_laps: Path
    q_results: Path
    r_results: Path


def _find_weekend_dir(season: int, round_number: int) -> Path:
    paths = get_project_paths()
    pattern = str(paths.data_raw / str(season) / f"r{round_number:02d}-*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No raw data found for season {season} round {round_number}.")
    if len(matches) > 1:
        raise ValueError(f"Multiple weekends match round {round_number}: {matches}")
    return Path(matches[0])


def _weekend_raw_paths(season: int, round_number: int) -> WeekendRawPaths:
    base_dir = _find_weekend_dir(season, round_number)
    return WeekendRawPaths(
        base=base_dir,
        q_laps=base_dir / "Q_laps.parquet",
        q_results=base_dir / "Q_results.parquet",
        r_results=base_dir / "R_results.parquet",
    )


def _best_lap_per_driver(q_laps: pd.DataFrame) -> pd.DataFrame:
    df = q_laps.copy()
    if "LapTime" not in df:
        raise ValueError("LapTime column missing from qualifying laps.")

    df = df[df["LapTime"].notna()]
    for flag in ("IsAccurate", "Deleted"):
        if flag in df:
            if flag == "Deleted":
                df = df[df[flag].eq(False)]
            else:
                df = df[df[flag].eq(True)]

    best = (
        df.groupby("DriverNumber", as_index=False)
        .agg({"LapTime": "min"})
        .rename(columns={"LapTime": "best_q_laptime"})
    )
    best["best_q_laptime_sec"] = pd.to_timedelta(best["best_q_laptime"]).dt.total_seconds()
    best["best_q_rank"] = best["best_q_laptime_sec"].rank(method="dense")
    return best


def build_basic_weekend_features(season: int, round_number: int) -> Path:
    """Combine qualifying/race aggregates into a basic feature table."""
    raw_paths = _weekend_raw_paths(season, round_number)
    q_laps = pd.read_parquet(raw_paths.q_laps)
    q_results = pd.read_parquet(raw_paths.q_results)
    r_results = pd.read_parquet(raw_paths.r_results)

    best_laps = _best_lap_per_driver(q_laps)
    q_columns = [
        "DriverNumber",
        "Abbreviation",
        "DriverId",
        "TeamName",
        "TeamId",
        "Position",
        "Q1",
        "Q2",
        "Q3",
    ]
    q_subset = q_results[[col for col in q_columns if col in q_results.columns]].rename(
        columns={"Position": "qualifying_position"}
    )

    r_columns = [
        "DriverNumber",
        "Position",
        "GridPosition",
        "Points",
        "Status",
    ]
    race_subset = r_results[[col for col in r_columns if col in r_results.columns]].rename(
        columns={"Position": "race_position"}
    )

    features = (
        q_subset.merge(best_laps, on="DriverNumber", how="left")
        .merge(race_subset, on="DriverNumber", how="left", suffixes=("", "_race"))
    )

    for phase in ("Q1", "Q2", "Q3"):
        if phase in features:
            features[f"{phase}_seconds"] = pd.to_timedelta(features[phase]).dt.total_seconds()

    if "GridPosition" in features and "qualifying_position" in features:
        features["grid_delta_vs_quali"] = features["GridPosition"] - features["qualifying_position"]
    features["target_position"] = features["race_position"]

    paths = get_project_paths()
    out_path = paths.data_processed / str(season) / f"r{round_number:02d}_features.parquet"
    write_parquet(features, out_path)
    return out_path
