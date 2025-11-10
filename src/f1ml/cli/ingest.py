"""Ingestion CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from f1ml.fastf1_ingest import ingest_weekend

console = Console()


@click.group(help="Download FastF1 data for race weekends.")
def ingest():
    pass


@ingest.command("weekend")
@click.option("--season", type=int, required=True, help="Championship year (e.g., 2023).")
@click.option("--round", "round_number", type=int, help="Round number (1-indexed).")
@click.option("--event", "event_name", type=str, help="Event/Grand Prix name (e.g., 'Monaco').")
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Override FastF1 cache directory.",
)
def ingest_weekend_command(
    season: int,
    round_number: Optional[int],
    event_name: Optional[str],
    cache_dir: Optional[Path],
) -> None:
    """Download qualifying + race data for a weekend."""
    if round_number is None and not event_name:
        raise click.BadParameter("Either --round or --event must be provided.", param_hint="--round/--event")

    summary = ingest_weekend(
        season=season,
        round_number=round_number,
        event_name=event_name,
        cache_dir=cache_dir,
    )

    console.print(f"[bold green]Download complete[/bold green] for season {season}. Stored files:")
    for session_code, paths in summary.items():
        console.print(f"  [cyan]{session_code}[/cyan]")
        for label, path in paths.items():
            console.print(f"    - {label}: {path}")
