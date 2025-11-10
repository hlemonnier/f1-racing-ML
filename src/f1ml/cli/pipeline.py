"""Pipeline automation commands."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import click
import pandas as pd
from rich.console import Console

from f1ml.cli.ingest import DEFAULT_SESSION_CODES
from f1ml.datasets import iter_feature_files, load_round_features
from f1ml.evaluation import evaluate_round_predictions
from f1ml.fastf1_ingest import ALLOWED_SESSION_CODES, ingest_weekend
from f1ml.features import build_basic_weekend_features
from f1ml.modeling.prediction import predict_round_results
from f1ml.modeling.training import train_season_model
from f1ml.config import get_project_paths

console = Console()


def _extract_event_context(summary: dict) -> Tuple[Optional[Path], Optional[str]]:
    for session_code in ("Q", "R", "FP1", "SQ"):
        info = summary.get(session_code)
        if not info:
            continue
        meta_path = Path(info["metadata"])
        if not meta_path.exists():
            continue
        data = json.loads(meta_path.read_text())
        return meta_path.parent, data.get("event_name")
    return None, None


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
@click.option(
    "--include-prev-year/--no-include-prev-year",
    default=True,
    show_default=True,
    help="Automatically fetch the previous season for matching rounds.",
)
def sync_season(
    season: int,
    round_start: int,
    round_end: Optional[int],
    sessions: Sequence[str],
    skip_features: bool,
    include_prev_year: bool,
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
            summary = ingest_weekend(
                season=season,
                round_number=round_number,
                session_codes=requested_sessions,
            )
            base_dir, event_name = _extract_event_context(summary)
            successful_rounds += 1
        except Exception as exc:
            console.print(f"[red]Failed ingest for round {round_number}[/red]: {exc}")
            continue

        if include_prev_year and season > 1950:
            paths = get_project_paths()
            prev_season = season - 1
            event_slug = None
            if base_dir is not None:
                event_slug = base_dir.name.split('-', 1)[1] if '-' in base_dir.name else base_dir.name

            need_event = True
            if event_slug:
                prev_event_pattern = str(paths.data_raw / str(prev_season) / f"*-{event_slug}")
                need_event = not glob.glob(prev_event_pattern)

            if need_event and event_name:
                console.print(
                    f"[cyan]Fetching previous season data[/cyan] for {event_name} (season {prev_season})."
                )
                try:
                    ingest_weekend(
                        season=prev_season,
                        round_number=None,
                        event_name=event_name,
                        session_codes=requested_sessions,
                    )
                    if not skip_features:
                        # Try to infer round from newly created directory
                        glob_pattern = str(paths.data_raw / str(prev_season) / f"*-{event_slug}") if event_slug else None
                        glob_path = glob.glob(glob_pattern) if glob_pattern else []
                        if glob_path:
                            inferred_round = int(Path(glob_path[0]).name.split('-')[0][1:3])
                            build_basic_weekend_features(season=prev_season, round_number=inferred_round)
                except Exception as exc:
                    console.print(f"[yellow]Prev-season ingestion failed[/yellow]: {exc}")

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


@pipeline.command("backtest")
@click.option("--season", type=int, required=True, help="Season to backtest (e.g., 2025).")
@click.option("--round-start", type=int, default=2, show_default=True, help="First round to predict.")
@click.option(
    "--round-end",
    type=int,
    default=None,
    help="Last round to include (defaults to last processed round).",
)
@click.option("--top-k", type=int, default=5, show_default=True, help="Top-K summary saved per round.")
def backtest(season: int, round_start: int, round_end: Optional[int], top_k: int) -> None:
    """Simulate the season round-by-round: train on past GPs, predict the next one."""
    available_rounds: List[int] = [rnd for rnd, _ in iter_feature_files(season)]
    if not available_rounds:
        raise click.ClickException(
            f"No processed features found for season {season}. Run 'pipeline sync-season' first."
        )

    max_round = max(available_rounds)
    final_round = round_end or max_round
    if final_round > max_round:
        console.print(
            f"[yellow]Only rounds up to {max_round} are available; capping round_end to {max_round}.[/yellow]"
        )
        final_round = max_round

    reports_dir = get_project_paths().reports / str(season)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[dict] = []

    for round_number in range(round_start, final_round + 1):
        training_rounds = [rnd for rnd in available_rounds if rnd < round_number]
        if not training_rounds:
            console.print(f"[yellow]Skipping round {round_number}: no prior races available for training.[/yellow]")
            continue

        console.rule(f"Backtest – Season {season} Round {round_number}")
        last_train_round = training_rounds[-1]
        try:
            artifact_path, train_metrics = train_season_model(
                season=season,
                upto_round=last_train_round,
                train_rounds=training_rounds,
            )
        except Exception as exc:
            console.print(f"[red]Training failed for round {round_number}[/red]: {exc}")
            continue

        console.print(
            f"Trained on rounds <= {last_train_round} (n={int(train_metrics['n_samples'])}) – MAE {train_metrics['mae']:.2f}"
        )

        try:
            prediction_output = predict_round_results(
                season=season,
                round_number=round_number,
                model_round=last_train_round,
                top_k=top_k,
            )
            prediction_df = prediction_output["full_results"]
        except Exception as exc:
            console.print(f"[red]Prediction failed for round {round_number}[/red]: {exc}")
            continue

        try:
            actual_df = load_round_features(season, round_number)
            metrics = evaluate_round_predictions(
                season=season,
                round_number=round_number,
                prediction_df=prediction_df,
                actual_df=actual_df,
            )
            metrics_rows.append(metrics.to_dict())
            console.print(
                f"[green]Round {round_number}[/green] → MAE {metrics.mae:.2f}, Spearman {metrics.spearman:.2f}, baseline MAE {metrics.baseline_mae:.2f}"
            )
        except Exception as exc:
            console.print(f"[yellow]Could not evaluate round {round_number}[/yellow]: {exc}")

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = reports_dir / "backtest_round_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        console.print(f"[bold green]Backtest metrics saved to {metrics_path}[/bold green]")
    else:
        console.print("[yellow]No metrics were recorded during backtest.[/yellow]")
