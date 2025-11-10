"""Pipeline automation commands."""

from __future__ import annotations

from typing import Optional, Sequence

import click
from rich.console import Console

from f1ml.cli.ingest import DEFAULT_SESSION_CODES
from f1ml.fastf1_ingest import ALLOWED_SESSION_CODES, ingest_weekend
from f1ml.features import build_basic_weekend_features

console = Console()


@click.group(help="Higher-level orchestration commands.")
def pipeline():
    pass


@pipeline.command("sync-season")
@click.option("--season", type=int, required=True, help="Season to sync (e.g., 2025).")
@click.option("--round-start", type=int, default=1, show_default=True, help="First round to process.")
@click.option(
    "--round-end",
    type=int,
    default=None,
    help="Last round to process (defaults to F1 calendar max or until ingestion fails).",
)
@click.option(
    "--session",
    "sessions",
    type=click.Choice(ALLOWED_SESSION_CODES, case_sensitive=False),
    multiple=True,
    default=DEFAULT_SESSION_CODES,
    show_default=True,
    help="Session codes to fetch for each round.",
)
@click.option(
    "--skip-features",
    is_flag=True,
    default=False,
    help="Only download raw sessions, do not build processed features.",
)
def sync_season(
    season: int,
    round_start: int,
    round_end: Optional[int],
    sessions: Sequence[str],
    skip_features: bool,
) -> None:
    """Download & process multiple rounds in one go."""
    if round_start < 1:
        raise click.BadParameter("--round-start must be >= 1.")

    # Assume modern calendar max 25 rounds unless user specifies otherwise.
    final_round = round_end or 25
    if final_round < round_start:
        raise click.BadParameter("--round-end must be >= --round-start.")

    requested_sessions = tuple(code.upper() for code in sessions)
    successful_rounds = 0

    for round_number in range(round_start, final_round + 1):
        console.rule(f"Season {season} – Round {round_number}")
        try:
            ingest_weekend(
                season=season,
                round_number=round_number,
                session_codes=requested_sessions,
            )
            successful_rounds += 1
        except Exception as exc:
            console.print(f"[red]Failed ingest for round {round_number}[/red]: {exc}")
            continue

        if skip_features:
            continue

        try:
            build_basic_weekend_features(season=season, round_number=round_number)
        except FileNotFoundError as exc:
            console.print(f"[yellow]Skipping feature build (missing race data)[/yellow]: {exc}")
        except Exception as exc:
            console.print(f"[red]Feature build failed for round {round_number}[/red]: {exc}")

    console.print(
        f"[bold green]Season sync complete[/bold green]: processed {successful_rounds} rounds (season {season}, R{round_start}–R{final_round})."
    )
