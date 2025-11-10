"""Configuration helpers for F1ML project paths and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path
    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path


def get_project_paths(base_dir: Optional[Path] = None) -> ProjectPaths:
    """Return the canonical project directories, creating them if missing."""
    root = base_dir or Path(__file__).resolve().parents[2]

    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    models_dir = root / "models"
    reports_dir = root / "reports"

    for path in (data_raw, data_processed, models_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        base_dir=root,
        data_raw=data_raw,
        data_processed=data_processed,
        models=models_dir,
        reports=reports_dir,
    )
