"""Weekend-level feature builders."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from f1ml.config import get_project_paths
from f1ml.io import write_parquet

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


def _find_weekend_dir(season: int, round_number: int) -> Path:
    paths = get_project_paths()
    pattern = str(paths.data_raw / str(season) / f"r{round_number:02d}-*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No raw data found for season {season} round {round_number}.")
    if len(matches) > 1:
        raise ValueError(f"Multiple weekends match round {round_number}: {matches}")
    return Path(matches[0])


def _session_file(base_dir: Path, session_code: str, suffix: str) -> Path:
    return base_dir / f"{session_code}_{suffix}.parquet"


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
        ]
        for col in lr_cols:
            if col not in features:
                features[col] = pd.Series([pd.NA] * n_rows, dtype="Float64")

    return features


def build_basic_weekend_features(season: int, round_number: int) -> Path:
    """Combine session aggregates into a basic feature table."""
    base_dir = _find_weekend_dir(season, round_number)
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

    features = _ensure_optional_session_columns(features)

    if "race_position" in features:
        features["target_position"] = features["race_position"]

    paths = get_project_paths()
    out_path = paths.data_processed / str(season) / f"r{round_number:02d}_features.parquet"
    write_parquet(features, out_path)
    return out_path
