"""Feature preprocessing utilities for model training and inference."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS: Tuple[str, ...] = ("Abbreviation", "DriverId", "TeamName", "TeamId")
LEAKAGE_COLUMNS: Tuple[str, ...] = ("target_position", "race_position", "Points", "Status")


def _build_category_mapping(series: pd.Series) -> Dict[str, int]:
    categories = series.astype("string").fillna("<NA>").unique().tolist()
    return {value: idx for idx, value in enumerate(categories)}


def _apply_category_mapping(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    return series.astype("string").map(lambda value: mapping.get(value, -1)).astype("int64")


def _convert_timedeltas(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.select_dtypes(include=["timedelta64[ns]"]).columns:
        sec_col = f"{col}_seconds"
        if sec_col not in result:
            result[sec_col] = result[col].dt.total_seconds()
        result = result.drop(columns=[col])
    return result


def _convert_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "DriverNumber" in result:
        result["DriverNumber"] = pd.to_numeric(result["DriverNumber"], errors="coerce")
    return result


def prepare_feature_matrix(
    df: pd.DataFrame,
    categorical_mappings: Optional[Dict[str, Dict[str, int]]] = None,
    expected_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], List[str]]:
    """Return a numeric feature matrix plus category metadata."""
    work = df.copy()
    work = work.drop(columns=[col for col in LEAKAGE_COLUMNS if col in work], errors="ignore")
    work = _convert_numeric_strings(work)
    work = _convert_timedeltas(work)

    fitted_mappings: Dict[str, Dict[str, int]] = {} if categorical_mappings is None else dict(categorical_mappings)
    for col in CATEGORICAL_COLUMNS:
        if categorical_mappings is None:
            if col in work:
                mapping = _build_category_mapping(work[col])
                fitted_mappings[col] = mapping
                work[col] = _apply_category_mapping(work[col], mapping)
        else:
            mapping = categorical_mappings.get(col)
            if mapping is not None:
                if col not in work:
                    work[col] = -1
                work[col] = _apply_category_mapping(work[col], mapping)
            else:
                # No mapping exists; drop column if present to avoid inconsistent encoding.
                if col in work:
                    work = work.drop(columns=[col])

    # Remove any remaining non-numeric columns
    non_numeric = work.select_dtypes(include=["object", "string"])
    if not non_numeric.empty:
        work = work.drop(columns=non_numeric.columns)

    if expected_columns is not None:
        for col in expected_columns:
            if col not in work:
                work[col] = np.nan
        work = work[expected_columns]
        feature_order = expected_columns
    else:
        feature_order = list(work.columns)

    work = work.astype("float64")

    return work, fitted_mappings, feature_order
