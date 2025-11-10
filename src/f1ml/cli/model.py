"""Model training and prediction CLI."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from f1ml.modeling.prediction import predict_round_results
from f1ml.modeling.training import train_season_model

console = Console()


@click.group(help="Train models and generate race predictions.")
def model():
    pass


@model.command("train")
@click.option("--season", type=int, required=True, help="Season to train on (e.g., 2024).")
@click.option("--until-round", type=int, required=True, help="Use data up to and including this round.")
def train_command(season: int, until_round: int) -> None:
    """Train a LightGBM regressor on all completed rounds."""
    artifact_path, metrics = train_season_model(season=season, upto_round=until_round)
    console.print(
        f"[bold green]Model trained[/bold green] for season {season} up to round {until_round}. Saved to {artifact_path}"
    )
    console.print(f"Training samples: {int(metrics['n_samples'])}")
    console.print(f"MAE (train): {metrics['mae']:.3f}")
    console.print(f"Spearman ρ (train): {metrics['spearman']:.3f}")


@model.command("predict")
@click.option("--season", type=int, required=True, help="Season to predict.")
@click.option("--round", "round_number", type=int, required=True, help="Round to predict.")
@click.option(
    "--model-round",
    type=int,
    default=None,
    help="Model trained up to this round (defaults to round-1).",
)
@click.option("--top-k", type=int, default=5, show_default=True, help="Number of top drivers to display.")
def predict_command(season: int, round_number: int, model_round: int | None, top_k: int) -> None:
    """Predict finishing order for a round and print the Top-K ranking."""
    if model_round is None:
        model_round = round_number - 1
        if model_round < 1:
            raise click.BadParameter("Cannot infer model_round for the first race. Specify --model-round explicitly.")

    outputs = predict_round_results(
        season=season,
        round_number=round_number,
        model_round=model_round,
        top_k=top_k,
    )

    console.print(
        f"[bold green]Predictions written[/bold green] to {outputs['full_path']} (season {season}, round {round_number})."
    )

    summary: Table = Table(title=f"Top {top_k} Projection (model up to round {model_round})")
    summary.add_column("Rank", justify="right")
    summary.add_column("Driver", justify="left")
    summary.add_column("Team", justify="left")
    summary.add_column("Predicted Position", justify="right")

    summary_df = outputs["summary"]
    for _, row in summary_df.iterrows():
        driver_label = row.get("DriverId") or row.get("DriverNumber")
        summary.add_row(
            str(int(row["predicted_rank"])),
            str(driver_label),
            str(row.get("TeamName", "")),
            f"{row['predicted_position']:.2f}",
        )
    console.print(summary)
