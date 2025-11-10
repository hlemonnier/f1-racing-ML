"""FastF1-powered ingestion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import fastf1
import pandas as pd
from fastf1.core import Session
from rich.console import Console
from rich.progress import track

from f1ml.config import get_project_paths
from f1ml.io import write_json, write_parquet

console = Console()

ALLOWED_SESSION_CODES = ("FP1", "FP2", "FP3", "SQ", "SR", "Q", "R")
DEFAULT_SESSION_CODES = ("FP1", "FP2", "FP3", "SQ", "SR", "Q", "R")


@dataclass
class SessionArtifactPaths:
    laps_path: Path
    results_path: Path
    metadata_path: Path


def enable_fastf1_cache(cache_dir: Optional[Path] = None) -> Path:
    """Enable FastF1 on-disk cache to avoid repeated downloads."""
    cache_root = cache_dir or get_project_paths().base_dir / ".fastf1_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_root))
    return cache_root


def _resolve_session(season: int, round_number: Optional[int], event_name: Optional[str], session_code: str) -> Session:
    if round_number is None and not event_name:
        raise ValueError("Either round_number or event_name must be provided.")
    selector = round_number if round_number is not None else event_name  # type: ignore[arg-type]
    session = fastf1.get_session(season, selector, session_code)
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


def _sanitize_event_name(event_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in event_name).strip("-")


def _build_session_paths(session: Session, session_code: str, round_number: Optional[int]) -> SessionArtifactPaths:
    paths = get_project_paths()
    round_num = round_number or int(session.event["RoundNumber"])
    event_date = session.event.get("EventDate")
    if event_date is not None:
        season = event_date.year
    elif session.date is not None:
        season = session.date.year
    else:
        season = int(session.event.get("OfficialEventName", "0").split()[-1])
    event_slug = _sanitize_event_name(session.event["EventName"])
    base = paths.data_raw / str(season) / f"r{round_num:02d}-{event_slug}"
    return SessionArtifactPaths(
        laps_path=base / f"{session_code}_laps.parquet",
        results_path=base / f"{session_code}_results.parquet",
        metadata_path=base / f"{session_code}_meta.json",
    )


def _session_metadata(session: Session, session_code: str) -> Dict:
    event = session.event
    event_date = event.get("EventDate")
    if event_date is not None:
        season = int(event_date.year)
    elif session.date is not None:
        season = int(session.date.year)
    else:
        season = int(event.get("OfficialEventName", "0").split()[-1])
    return {
        "season": season,
        "round": int(event["RoundNumber"]),
        "event_name": event["EventName"],
        "country": event.get("Country"),
        "session_code": session_code,
        "session_name": session.name,
        "total_drivers": len(session.drivers),
    }


def _results_dataframe(session: Session) -> pd.DataFrame:
    results = session.results
    if results is None:
        return pd.DataFrame()
    df = results.reset_index(drop=True)
    return df


def ingest_weekend(
    season: int,
    round_number: Optional[int] = None,
    event_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    session_codes: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Download FastF1 session data for a weekend and persist to parquet."""
    enable_fastf1_cache(cache_dir)
    summary: Dict[str, Dict[str, str]] = {}
    requested_codes: Iterable[str] = session_codes or DEFAULT_SESSION_CODES

    for session_code in track(list(requested_codes), description="Downloading sessions"):
        try:
            session = _resolve_session(season, round_number, event_name, session_code)
        except Exception as exc:  # FastF1 raises several custom errors; treat all as skip-worthy
            console.print(f"[yellow]Skipping {session_code}[/yellow]: {exc}")
            continue

        artifact_paths = _build_session_paths(session, session_code, round_number)

        laps_df = session.laps.reset_index(drop=True)
        results_df = _results_dataframe(session)
        metadata = _session_metadata(session, session_code)

        write_parquet(laps_df, artifact_paths.laps_path)
        write_parquet(results_df, artifact_paths.results_path)
        write_json(metadata, artifact_paths.metadata_path)

        console.print(
            f"[green]Saved {session_code}[/green] laps -> {artifact_paths.laps_path.relative_to(get_project_paths().base_dir)}"
        )
        summary[session_code] = {
            "laps": str(artifact_paths.laps_path),
            "results": str(artifact_paths.results_path),
            "metadata": str(artifact_paths.metadata_path),
        }

    if not summary:
        raise RuntimeError("No sessions were downloaded. Check the requested sessions or availability.")

    return summary
