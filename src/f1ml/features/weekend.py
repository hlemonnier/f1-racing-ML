"""Weekend-level feature builders."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from f1ml.config import get_project_paths
from f1ml.io import write_parquet
from f1ml.standings import build_standings
from f1ml.weather import get_weather_features
from f1ml.rain_uplift import ensure_rain_uplift

OPTIONAL_LAP_SESSIONS: Dict[str, str] = {
    "FP1": "fp1",
    "FP2": "fp2",
    "FP3": "fp3",
    "SQ": "sq",
    "SR": "sr",
}

LONG_RUN_SESSIONS: Dict[str, str] = {
    "FP1": "fp1",
    "FP2": "fp2",
    "FP3": "fp3",
    "SR": "sr",
}

MIN_LONG_RUN_LAPS = 5
MIN_CLEAN_STINT_LAPS = 6
RESIDUAL_THRESHOLD = 0.6  # seconds
FUEL_START_KG = 105.0
FUEL_CONSUMPTION_KG = 1.8
FUEL_REF_KG = 70.0
STINT_REF_LAP = 5
ALPHA_FUEL = 0.03  # sec per kg
DEGRADATION_PER_LAP = 0.03  # sec per lap


def _find_weekend_dir(season: int, round_number: int) -> Path:
    paths = get_project_paths()
    pattern = str(paths.data_raw / str(season) / f"r{round_number:02d}-*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No raw data found for season {season} round {round_number}.")
    if len(matches) > 1:
        raise ValueError(f"Multiple weekends match round {round_number}: {matches}")
    return Path(matches[0])


def _event_slug_from_dir(base_dir: Path) -> str:
    name = base_dir.name
    parts = name.split('-', 1)
    if len(parts) == 2:
        return parts[1]
    return name


def _session_file(base_dir: Path, session_code: str, suffix: str) -> Path:
    return base_dir / f"{session_code}_{suffix}.parquet"


def _find_prev_season_dir(event_slug: str, season: int) -> Optional[Path]:
    prev_season = season - 1
    if prev_season < 1950:
        return None
    paths = get_project_paths()
    pattern = str(paths.data_raw / str(prev_season) / f"r*-{event_slug}")
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Prefer same round index if multiple
    matches.sort()
    return Path(matches[0])


def _load_session_parquet(base_dir: Path, session_code: str, suffix: str) -> pd.DataFrame | None:
    path = _session_file(base_dir, session_code, suffix)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _best_lap_per_driver(laps: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = laps.copy()
    if "LapTime" not in df:
        raise ValueError("LapTime column missing from session laps.")

    df = df[df["LapTime"].notna()]
    if "IsAccurate" in df:
        df = df[df["IsAccurate"].eq(True)]

    best = (
        df.groupby("DriverNumber", as_index=False)
        .agg({"LapTime": "min"})
        .rename(columns={"LapTime": f"best_{prefix}_laptime"})
    )
    best[f"best_{prefix}_laptime_sec"] = pd.to_timedelta(best[f"best_{prefix}_laptime"]).dt.total_seconds()
    best[f"best_{prefix}_rank"] = best[f"best_{prefix}_laptime_sec"].rank(method="dense")
    return best


def _convert_timedelta_columns(df: pd.DataFrame, columns: List[str], suffix: str = "_sec") -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result:
            result[f"{col}{suffix}"] = pd.to_timedelta(result[col]).dt.total_seconds()
    return result


def _series_to_seconds(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    try:
        return pd.to_timedelta(series).dt.total_seconds()
    except (ValueError, TypeError):
        return pd.to_numeric(series, errors="coerce")


def _maybe_merge_session_best_laps(
    features: pd.DataFrame, base_dir: Path, session_code: str, prefix: str
) -> pd.DataFrame:
    laps_df = _load_session_parquet(base_dir, session_code, "laps")
    if laps_df is None or "LapTime" not in laps_df:
        return features
    session_bests = _best_lap_per_driver(laps_df, prefix)
    return features.merge(session_bests, on="DriverNumber", how="left")


def _prepare_valid_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()
    df = df[df["LapTime"].notna()]
    if "IsAccurate" in df:
        df = df[df["IsAccurate"].fillna(False)]
    if "Deleted" in df:
        df = df[df["Deleted"].ne(True)]
    df["lap_seconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    return df[df["lap_seconds"].notna()]


def _stint_lap_numbers(df: pd.DataFrame) -> pd.Series:
    if "LapNumber" in df.columns:
        return df.groupby(["DriverNumber", "Stint"]) ["LapNumber"].rank(method="first")
    return df.groupby(["DriverNumber", "Stint"]).cumcount() + 1


def _filter_clean_air_laps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    filtered["stint_lap"] = _stint_lap_numbers(filtered)
    stint_lengths = filtered.groupby(["DriverNumber", "Stint"]) ["stint_lap"].transform("max")
    filtered = filtered[stint_lengths >= MIN_CLEAN_STINT_LAPS]
    filtered = filtered[filtered["stint_lap"] > 1]

    if "TrackStatus" in filtered.columns:
        filtered = filtered[filtered["TrackStatus"].astype(str).isin({"1", "0", ""})]

    gap_mask = None
    for gap_col in ("GapToAhead", "GapAhead", "GapToNext", "GapToLeader"):
        if gap_col in filtered.columns:
            seconds = _series_to_seconds(filtered[gap_col])
            mask = seconds > 2.5
            gap_mask = mask if gap_mask is None else (gap_mask & mask)
    if gap_mask is not None:
        filtered = filtered[gap_mask.fillna(False)]

    speed_col = None
    for col in ("SpeedST", "SpeedFL", "SpeedI1"):
        if col in filtered.columns:
            speed_col = col
            break
    if speed_col:
        if "Compound" in filtered.columns:
            med = filtered.groupby("Compound")[speed_col].transform("median")
            std = filtered.groupby("Compound")[speed_col].transform("std").fillna(0)
        else:
            med_val = filtered[speed_col].median()
            std_val = filtered[speed_col].std() or 0
            med = pd.Series(med_val, index=filtered.index)
            std = pd.Series(std_val, index=filtered.index)
        filtered = filtered[filtered[speed_col] <= med + 2 * std]

    median = filtered.groupby(["DriverNumber", "Stint"])["lap_seconds"].transform("median")
    mad = (
        (filtered["lap_seconds"] - median)
        .abs()
        .groupby([filtered["DriverNumber"], filtered["Stint"]])
        .transform("median")
    )
    mad = mad.replace(0, 1e-6)
    resid = (filtered["lap_seconds"] - median).abs()
    filtered = filtered[resid <= (mad + RESIDUAL_THRESHOLD)]
    return filtered


def _compute_long_run_features(laps: pd.DataFrame, prefix: str) -> Optional[pd.DataFrame]:
    valid = _prepare_valid_laps(laps)
    if valid.empty:
        return None

    results: List[Dict[str, float]] = []

    for driver, driver_laps in valid.groupby("DriverNumber"):
        driver_laps = driver_laps.sort_values("LapNumber")
        lap_diff = driver_laps["LapNumber"].diff().fillna(1)
        streak_groups = (lap_diff != 1).cumsum()
        group_sizes = streak_groups.groupby(streak_groups).size()
        max_consec = float(group_sizes.max()) if not group_sizes.empty else np.nan

        best_entry: Optional[Dict[str, float]] = None
        for _, stint_df in driver_laps.groupby("Stint"):
            stint_df = stint_df.sort_values("LapNumber")
            if len(stint_df) < MIN_LONG_RUN_LAPS:
                continue
            secs = stint_df["lap_seconds"].to_numpy()
            if len(secs) < MIN_LONG_RUN_LAPS:
                continue

            mean_val = float(np.mean(secs))
            std_val = float(np.std(secs, ddof=0))
            if len(secs) >= 2:
                slope = float(np.polyfit(np.arange(len(secs)), secs, 1)[0])
            else:
                slope = np.nan

            entry = {
                "DriverNumber": driver,
                f"best_{prefix}_longrun_mean": mean_val,
                f"best_{prefix}_longrun_std": std_val,
                f"best_{prefix}_longrun_slope": slope,
                f"best_{prefix}_longrun_laps": float(len(secs)),
                f"{prefix}_longrun_consistency": std_val / mean_val if mean_val > 0 else np.nan,
            }
            if best_entry is None or entry[f"best_{prefix}_longrun_mean"] < best_entry[f"best_{prefix}_longrun_mean"]:
                best_entry = entry

        if best_entry is not None:
            best_entry[f"{prefix}_max_consec_valid"] = float(max_consec)
            results.append(best_entry)

    if not results:
        return None

    df = pd.DataFrame(results)
    session_min = df[f"best_{prefix}_longrun_mean"].min()
    df[f"{prefix}_longrun_delta"] = df[f"best_{prefix}_longrun_mean"] - session_min
    return df


def _compute_clean_air_features(laps: pd.DataFrame, prefix: str) -> Optional[pd.DataFrame]:
    valid = _prepare_valid_laps(laps)
    clean = _filter_clean_air_laps(valid)
    if clean.empty:
        return None

    results: List[Dict[str, float]] = []

    for driver, driver_laps in clean.groupby("DriverNumber"):
        driver_laps = driver_laps.sort_values(["Stint", "LapNumber"])
        best_entry: Optional[Dict[str, float]] = None
        for _, stint_df in driver_laps.groupby("Stint"):
            if len(stint_df) < MIN_CLEAN_STINT_LAPS:
                continue
            secs = stint_df["lap_seconds"].to_numpy()
            if len(secs) < MIN_CLEAN_STINT_LAPS:
                continue
            mean_val = float(np.mean(secs))
            std_val = float(np.std(secs, ddof=0))
            slope = float(np.polyfit(np.arange(len(secs)), secs, 1)[0]) if len(secs) >= 2 else np.nan
            stint_laps = stint_df["stint_lap"].to_numpy()
            fuel_est = FUEL_START_KG - FUEL_CONSUMPTION_KG * (stint_laps - 1)
            ref_adjusted = secs - ALPHA_FUEL * (fuel_est - FUEL_REF_KG) - DEGRADATION_PER_LAP * (stint_laps - STINT_REF_LAP)
            ref_mean = float(np.mean(ref_adjusted))
            ref_best = float(np.min(ref_adjusted))
            entry = {
                "DriverNumber": driver,
                f"best_{prefix}_clean_mean": mean_val,
                f"best_{prefix}_clean_std": std_val,
                f"best_{prefix}_clean_slope": slope,
                f"best_{prefix}_clean_laps": float(len(secs)),
                f"best_{prefix}_clean_best": float(np.min(secs)),
                f"best_{prefix}_clean_ref_mean": ref_mean,
                f"best_{prefix}_clean_ref_best": ref_best,
            }
            if best_entry is None or entry[f"best_{prefix}_clean_mean"] < best_entry[f"best_{prefix}_clean_mean"]:
                best_entry = entry
        if best_entry is not None:
            results.append(best_entry)

    if not results:
        return None

    df = pd.DataFrame(results)
    session_min = df[f"best_{prefix}_clean_mean"].min()
    df[f"{prefix}_clean_delta"] = df[f"best_{prefix}_clean_mean"] - session_min
    session_ref_min = df[f"best_{prefix}_clean_ref_mean"].min()
    df[f"{prefix}_clean_ref_delta"] = df[f"best_{prefix}_clean_ref_mean"] - session_ref_min
    return df


def _maybe_merge_long_run_features(
    features: pd.DataFrame, base_dir: Path, session_code: str, prefix: str
) -> pd.DataFrame:
    laps_df = _load_session_parquet(base_dir, session_code, "laps")
    if laps_df is None:
        return features
    long_run_df = _compute_long_run_features(laps_df, prefix)
    if long_run_df is None:
        return features
    return features.merge(long_run_df, on="DriverNumber", how="left")


def _merge_on_driver(features: pd.DataFrame, addon: pd.DataFrame) -> pd.DataFrame:
    for key in ("DriverId", "DriverNumber", "Abbreviation"):
        if key in features.columns and key in addon.columns:
            return features.merge(addon, on=key, how="left")
    return features


def _prev_year_quali_features(prev_dir: Path) -> Optional[pd.DataFrame]:
    q_results = _load_session_parquet(prev_dir, "Q", "results")
    if q_results is None:
        return None
    cols = [
        col
        for col in (
            "DriverId",
            "DriverNumber",
            "Abbreviation",
            "Position",
            "Q1",
            "Q2",
            "Q3",
        )
        if col in q_results.columns
    ]
    data = q_results[cols].copy()
    data = _convert_timedelta_columns(data, [c for c in ("Q1", "Q2", "Q3") if c in data], suffix="_sec")
    rename_map = {"Position": "prev_year_quali_position"}
    for col in ("Q1", "Q2", "Q3"):
        if col in data.columns:
            rename_map[col] = f"prev_year_{col.lower()}"
            rename_map[f"{col}_sec"] = f"prev_year_{col.lower()}_sec"
    data = data.rename(columns=rename_map)
    return data


def _prev_year_race_features(prev_dir: Path, session_code: str, prefix: str) -> Optional[pd.DataFrame]:
    laps = _load_session_parquet(prev_dir, session_code, "laps")
    if laps is None or "LapTime" not in laps:
        return None

    df = _prepare_valid_laps(laps)
    if df.empty:
        return None

    for col in ("Sector1Time", "Sector2Time", "Sector3Time"):
        if col in df:
            df[f"{col}_sec"] = pd.to_timedelta(df[col]).dt.total_seconds()

    agg = df.groupby("DriverNumber").agg(
        lap_mean=("lap_seconds", "mean"),
        lap_min=("lap_seconds", "min"),
        sector1_mean=("Sector1Time_sec", "mean"),
        sector2_mean=("Sector2Time_sec", "mean"),
        sector3_mean=("Sector3Time_sec", "mean"),
    )

    agg = agg.rename(
        columns={
            "lap_mean": f"prev_{prefix}_lap_mean",
            "lap_min": f"prev_{prefix}_lap_best",
            "sector1_mean": f"prev_{prefix}_sector1_mean",
            "sector2_mean": f"prev_{prefix}_sector2_mean",
            "sector3_mean": f"prev_{prefix}_sector3_mean",
        }
    ).reset_index()
    results_map = _load_session_parquet(prev_dir, session_code, "results")
    if results_map is not None and "DriverNumber" in results_map.columns:
        id_cols = [col for col in ("DriverId", "Abbreviation") if col in results_map.columns]
        if id_cols:
            mapping = results_map[["DriverNumber", *id_cols]].drop_duplicates("DriverNumber")
            agg = agg.merge(mapping, on="DriverNumber", how="left")
    return agg


def _prev_year_clean_air_features(prev_dir: Path) -> Optional[pd.DataFrame]:
    laps = _load_session_parquet(prev_dir, "R", "laps")
    if laps is None:
        return None
    return _compute_clean_air_features(laps, "prev_race")


def _inject_prev_year_features(features: pd.DataFrame, season: int, base_dir: Path) -> pd.DataFrame:
    event_slug = _event_slug_from_dir(base_dir)
    prev_dir = _find_prev_season_dir(event_slug, season)
    if prev_dir is None:
        return features

    prev_q = _prev_year_quali_features(prev_dir)
    if prev_q is not None:
        features = _merge_on_driver(features, prev_q)

    prev_race = _prev_year_race_features(prev_dir, "R", "race")
    if prev_race is not None:
        features = _merge_on_driver(features, prev_race)

    prev_sr = _prev_year_race_features(prev_dir, "SR", "sprint")
    if prev_sr is not None:
        features = _merge_on_driver(features, prev_sr)

    prev_clean = _prev_year_clean_air_features(prev_dir)
    if prev_clean is not None:
        features = _merge_on_driver(features, prev_clean)

    return features


def _inject_standings_features(features: pd.DataFrame, season: int, round_number: int) -> pd.DataFrame:
    snapshot = build_standings(season, round_number)

    driver_key = "DriverId" if "DriverId" in features else "Abbreviation"
    team_key = "TeamName" if "TeamName" in features else None

    if driver_key:
        features["champ_points_to_date"] = (
            features[driver_key].map(snapshot.driver_points).astype("Float64")
        )
        features["champ_position"] = (
            features[driver_key].map(snapshot.driver_positions).astype("Float64")
        )
        features["driver_form_last3"] = (
            features[driver_key].map(snapshot.driver_form_last3).astype("Float64")
        )
    else:
        features["champ_points_to_date"] = pd.Series([pd.NA] * len(features), dtype="Float64")
        features["champ_position"] = pd.Series([pd.NA] * len(features), dtype="Float64")
        features["driver_form_last3"] = pd.Series([pd.NA] * len(features), dtype="Float64")

    if team_key:
        features["team_points_to_date"] = (
            features[team_key].map(snapshot.team_points).astype("Float64")
        )
        features["team_champ_position"] = (
            features[team_key].map(snapshot.team_positions).astype("Float64")
        )
        features["team_form_last3"] = (
            features[team_key].map(snapshot.team_form_last3).astype("Float64")
        )
    else:
        features["team_points_to_date"] = pd.Series([pd.NA] * len(features), dtype="Float64")
        features["team_champ_position"] = pd.Series([pd.NA] * len(features), dtype="Float64")
        features["team_form_last3"] = pd.Series([pd.NA] * len(features), dtype="Float64")

    return features


def _ensure_optional_session_columns(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features

    n_rows = len(features)
    for prefix in OPTIONAL_LAP_SESSIONS.values():
        time_col = f"best_{prefix}_laptime"
        time_sec_col = f"best_{prefix}_laptime_sec"
        rank_col = f"best_{prefix}_rank"

        if time_col not in features:
            features[time_col] = pd.Series([pd.NaT] * n_rows, dtype="timedelta64[ns]")
        if time_sec_col not in features:
            features[time_sec_col] = pd.Series([pd.NA] * n_rows, dtype="Float64")
        if rank_col not in features:
            features[rank_col] = pd.Series([pd.NA] * n_rows, dtype="Float64")

        # Long-run placeholders
        lr_cols = [
            f"best_{prefix}_longrun_mean",
            f"best_{prefix}_longrun_std",
            f"best_{prefix}_longrun_slope",
            f"best_{prefix}_longrun_laps",
            f"{prefix}_longrun_consistency",
            f"{prefix}_longrun_delta",
            f"{prefix}_max_consec_valid",
            f"best_{prefix}_clean_mean",
            f"best_{prefix}_clean_std",
            f"best_{prefix}_clean_slope",
            f"best_{prefix}_clean_laps",
            f"best_{prefix}_clean_best",
            f"{prefix}_clean_delta",
            f"best_{prefix}_clean_ref_mean",
            f"best_{prefix}_clean_ref_best",
            f"{prefix}_clean_ref_delta",
        ]
        for col in lr_cols:
            if col not in features:
                features[col] = pd.Series([pd.NA] * n_rows, dtype="Float64")

    return features


def _apply_weather_penalty(features: pd.DataFrame, weather: Dict[str, float]) -> pd.DataFrame:
    rain_prob = weather.get("weather_rain_probability", 0.0) or 0.0
    if rain_prob < 0.2:
        features["qualifying_wet_factor"] = 1.0
        return features
    factor = 1 + 0.3 * min(1.0, float(rain_prob))
    for col in ("best_q_laptime_sec", "Q1_seconds", "Q2_seconds", "Q3_seconds"):
        if col in features:
            features[col] = features[col] * factor
    features["qualifying_wet_factor"] = factor
    return features


def _inject_rain_uplift_features(features: pd.DataFrame, season: int) -> pd.DataFrame:
    uplift_df = ensure_rain_uplift(season)
    if uplift_df is None or uplift_df.empty:
        features["driver_rain_uplift"] = pd.Series([pd.NA] * len(features), dtype="Float64")
        features["team_rain_uplift"] = pd.Series([pd.NA] * len(features), dtype="Float64")
        return features

    driver_map = uplift_df.groupby("driver_id")["rain_uplift"].mean().to_dict()
    team_map = uplift_df.groupby("team_name")["rain_uplift"].mean().to_dict()

    driver_key = "Abbreviation" if "Abbreviation" in features else "DriverId"
    driver_series = features[driver_key].astype(str).str.upper()
    features["driver_rain_uplift"] = driver_series.map(driver_map).astype("Float64")
    if "TeamName" in features:
        features["team_rain_uplift"] = features["TeamName"].astype(str).map(team_map).astype("Float64")
    else:
        features["team_rain_uplift"] = pd.Series([pd.NA] * len(features), dtype="Float64")
    return features


def build_basic_weekend_features(season: int, round_number: int) -> Path:
    """Combine session aggregates into a basic feature table."""
    base_dir = _find_weekend_dir(season, round_number)
    event_slug = _event_slug_from_dir(base_dir)
    q_laps = _load_session_parquet(base_dir, "Q", "laps")
    q_results = _load_session_parquet(base_dir, "Q", "results")
    if q_laps is None or q_results is None:
        raise FileNotFoundError("Qualifying data (laps/results) not found. Run ingestion first.")

    r_results = _load_session_parquet(base_dir, "R", "results")

    best_laps_q = _best_lap_per_driver(q_laps, "q")
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

    features = q_subset.merge(best_laps_q, on="DriverNumber", how="left")

    if r_results is not None:
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
        features = features.merge(race_subset, on="DriverNumber", how="left", suffixes=("", "_race"))

    for phase in ("Q1", "Q2", "Q3"):
        if phase in features:
            features[f"{phase}_seconds"] = pd.to_timedelta(features[phase]).dt.total_seconds()

    if "GridPosition" in features and "qualifying_position" in features:
        features["grid_delta_vs_quali"] = features["GridPosition"] - features["qualifying_position"]

    for session_code, prefix in OPTIONAL_LAP_SESSIONS.items():
        features = _maybe_merge_session_best_laps(features, base_dir, session_code, prefix)

    for session_code, prefix in LONG_RUN_SESSIONS.items():
        features = _maybe_merge_long_run_features(features, base_dir, session_code, prefix)
        laps_df = _load_session_parquet(base_dir, session_code, "laps")
        clean_df = _compute_clean_air_features(laps_df, prefix) if laps_df is not None else None
        if clean_df is not None:
            features = features.merge(clean_df, on="DriverNumber", how="left")

    features = _inject_prev_year_features(features, season, base_dir)
    features = _inject_standings_features(features, season, round_number)
    features = _inject_rain_uplift_features(features, season)
    weather = get_weather_features(event_slug, season, round_number)
    for key, value in weather.items():
        features[key] = value
    features = _apply_weather_penalty(features, weather)
    features = _ensure_optional_session_columns(features)

    if "race_position" in features:
        features["target_position"] = features["race_position"]

    paths = get_project_paths()
    out_path = paths.data_processed / str(season) / f"r{round_number:02d}_features.parquet"
    write_parquet(features, out_path)
    return out_path
