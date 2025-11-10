"""Utilities to load processed feature datasets for training and prediction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from f1ml.config import get_project_paths

ROUND_FILE_PATTERN = re.compile(r"r(\d+)_features\.parquet$")


def _processed_dir(season: int) -> Path:
    paths = get_project_paths()
    season_dir = paths.data_processed / str(season)
    if not season_dir.exists():
        raise FileNotFoundError(f"No processed features found for season {season}. Expected at {season_dir}")
    return season_dir


def iter_feature_files(season: int) -> Iterable[Tuple[int, Path]]:
    """Yield (round, path) pairs for processed feature files of a season."""
    base_dir = _processed_dir(season)
    for path in sorted(base_dir.glob("r*_features.parquet")):
        match = ROUND_FILE_PATTERN.match(path.name)
        if not match:
            continue
        round_number = int(match.group(1))
        yield round_number, path


def load_round_features(season: int, round_number: int) -> pd.DataFrame:
    """Load the processed feature parquet for a single round."""
    season_dir = _processed_dir(season)
    path = season_dir / f"r{round_number:02d}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed features found at {path}. Did you run the features CLI?")
    df = pd.read_parquet(path)
    df["season"] = season
    df["round"] = round_number
    return df


def load_training_dataframe(
    season: int,
    upto_round: int,
    include_rounds: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Concatenate feature tables for rounds <= upto_round (only rows with targets)."""
    round_filter = set(include_rounds) if include_rounds is not None else None
    frames: List[pd.DataFrame] = []
    for round_number, path in iter_feature_files(season):
        if round_number > upto_round:
            continue
        if round_filter is not None and round_number not in round_filter:
            continue
        df = pd.read_parquet(path)
        if "target_position" not in df:
            continue
        df = df.dropna(subset=["target_position"]).copy()
        if df.empty:
            continue
        df["season"] = season
        df["round"] = round_number
        frames.append(df)
    if not frames:
        raise ValueError(
            f"No training data found for season {season} up to round {upto_round}"
            + (
                f" for rounds {sorted(round_filter)}"
                if round_filter
                else ""
            )
            + ". Ensure features exist and include target_position."
        )
    return pd.concat(frames, ignore_index=True)
