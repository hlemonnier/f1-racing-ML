"""CLI entrypoints for the F1ML toolkit."""

from __future__ import annotations

import click

from f1ml.cli.features import features
from f1ml.cli.ingest import ingest
from f1ml.cli.model import model
from f1ml.cli.pipeline import pipeline


@click.group(help="Utilities for the F1 ML pipeline.")
def cli():
    pass


cli.add_command(ingest)
cli.add_command(features)
cli.add_command(model)
cli.add_command(pipeline)


def run():
    """Entry point used by the Poetry script hook."""
    cli()
