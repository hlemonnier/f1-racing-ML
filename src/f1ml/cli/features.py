"""Feature CLI commands."""

from __future__ import annotations

import click
from rich.console import Console

from f1ml.features import build_basic_weekend_features

console = Console()


@click.group(help="Feature engineering commands.")
def features():
    pass


@features.command("basic-weekend")
@click.option("--season", required=True, type=int, help="Championship year (e.g., 2023).")
@click.option("--round", "round_number", required=True, type=int, help="Round number (1-indexed).")
def build_basic(season: int, round_number: int) -> None:
    """Build basic qualifying/race features for a completed weekend."""
    output_path = build_basic_weekend_features(season=season, round_number=round_number)
    console.print(f"[green]Features written:[/green] {output_path}")
