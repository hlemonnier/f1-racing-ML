"""Estimate per driver/team rain uplift based on historical race laps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from f1ml.config import get_project_paths
from f1ml.weather import get_weather_features


RAIN_THRESHOLD = 0.5


def _race_dirs(season: int) -> list[Path]:
    base = get_project_paths().data_raw / str(season)
    if not base.exists():
        return []
    return sorted(base.glob("r*-*"))


def _load_metadata(race_dir: Path) -> Dict:
    meta_path = race_dir / "R_meta.json"
    if not meta_path.exists():
        # fallback: try any meta file
        for candidate in race_dir.glob("*meta.json"):
            meta_path = candidate
            break
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _prepare_laps(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "LapTime" not in df:
        return None
    df = df[df["LapTime"].notna()]
    if "IsAccurate" in df:
        df = df[df["IsAccurate"].fillna(False)]
    if "Deleted" in df:
        df = df[df["Deleted"].ne(True)]
    df["lap_seconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    df = df[df["lap_seconds"].notna()]
    if "PitInTime" in df:
        df = df[df["PitInTime"].isna()]
    if "PitOutTime" in df:
        df = df[df["PitOutTime"].isna()]
    if "LapNumber" in df:
        df = df[df["LapNumber"] > 1]
    return df


def _lap_medians(df: pd.DataFrame) -> pd.Series:
    positions = df.get("Position")
    if positions is not None:
        top_mask = positions <= 10
        lap_medians = df[top_mask].groupby("LapNumber")["lap_seconds"].median()
    else:
        top_mask = pd.Series(True, index=df.index)
        lap_medians = pd.Series(dtype="float64")
    medians = lap_medians
    fallback = df.groupby("LapNumber")["lap_seconds"].median()
    medians = medians.combine_first(fallback)
    return df["LapNumber"].map(medians)


def _teammate_means(df: pd.DataFrame) -> pd.Series:
    if "Team" not in df.columns:
        return pd.Series(0.0, index=df.index)
    team_medians = df.groupby(["Team", "LapNumber"])["lap_seconds"].median()
    idx = list(zip(df["Team"], df["LapNumber"]))
    mapped = [team_medians.get(key, float("nan")) for key in idx]
    series = pd.Series(mapped, index=df.index)
    return series


def _is_wet_lap(df: pd.DataFrame, weather_prob: float) -> pd.Series:
    comp = df["Compound"].astype(str).str.upper() if "Compound" in df else ""
    wet_comp = comp.isin({"WET", "INTERMEDIATE"})
    if wet_comp.any():
        return wet_comp
    if weather_prob >= RAIN_THRESHOLD:
        return pd.Series(True, index=df.index)
    return pd.Series(False, index=df.index)


def compute_rain_uplift(season: int) -> Path:
    paths = get_project_paths()
    derived_dir = paths.base_dir / "data" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    output_path = derived_dir / f"rain_uplift_{season}.parquet"

    records = []
    for race_dir in _race_dirs(season):
        lap_path = race_dir / "R_laps.parquet"
        laps = _prepare_laps(lap_path)
        if laps is None or laps.empty:
            continue
        meta = _load_metadata(race_dir)
        event_name = meta.get("event_name")
        event_slug = race_dir.name.split('-', 1)[1] if '-' in race_dir.name else race_dir.name
        weather = get_weather_features(event_slug, season, meta.get("round", 0) or 0)
        wet_flag = _is_wet_lap(laps, weather.get("weather_rain_probability", 0.0))

        lap_median = _lap_medians(laps)
        laps["delta_vs_top10"] = laps["lap_seconds"] - lap_median
        teammate_mean = _teammate_means(laps)
        laps["delta_vs_team"] = laps["lap_seconds"] - teammate_mean.fillna(laps["lap_seconds"]).to_numpy()

        laps["is_wet"] = wet_flag.values
        driver_ids = laps["Driver"].astype(str).str.upper() if "Driver" in laps.columns else laps["DriverNumber"].astype(str)
        team_names = laps.get("Team", pd.Series("", index=laps.index)).astype(str)
        laps["driver_code"] = driver_ids
        laps["team_name"] = team_names

        wet = laps[laps["is_wet"]]
        dry = laps[~laps["is_wet"]]
        if wet.empty or dry.empty:
            continue

        group_cols = ["driver_code", "team_name"]
        wet_mean = wet.groupby(group_cols)["delta_vs_top10"].mean()
        dry_mean = dry.groupby(group_cols)["delta_vs_top10"].mean()
        wet_count = wet.groupby(group_cols)["delta_vs_top10"].count()
        dry_count = dry.groupby(group_cols)["delta_vs_top10"].count()

        frame = pd.DataFrame({
            "wet_mean": wet_mean,
            "dry_mean": dry_mean,
            "wet_count": wet_count,
            "dry_count": dry_count,
        }).dropna()
        frame["rain_uplift"] = frame["wet_mean"] - frame["dry_mean"]
        frame.reset_index(inplace=True)
        frame.rename(columns={"level_0": "driver_id", "level_1": "team_name"}, inplace=True)
        frame["season"] = season
        frame["round"] = meta.get("round")
        records.append(frame)

    if not records:
        empty = pd.DataFrame(columns=[
            "driver_id",
            "team_name",
            "rain_uplift",
            "wet_mean",
            "dry_mean",
            "wet_count",
            "dry_count",
            "season",
            "round",
        ])
        empty.to_parquet(output_path, index=False)
        return output_path

    df = pd.concat(records, ignore_index=True)
    agg = df.groupby(["driver_id", "team_name"], as_index=False).agg(
        rain_uplift=("rain_uplift", "mean"),
        wet_count=("wet_count", "sum"),
        dry_count=("dry_count", "sum"),
    )
    agg.to_parquet(output_path, index=False)
    return output_path


def load_rain_uplift(season: int) -> Optional[pd.DataFrame]:
    path = get_project_paths().base_dir / "data" / "derived" / f"rain_uplift_{season}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def ensure_rain_uplift(season: int) -> Optional[pd.DataFrame]:
    df = load_rain_uplift(season)
    if df is not None:
        return df
    compute_rain_uplift(season)
    return load_rain_uplift(season)
