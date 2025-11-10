"""CLI entrypoints for the F1ML toolkit."""

from __future__ import annotations

import click

from f1ml.cli.features import features
from f1ml.cli.ingest import ingest


@click.group(help="Utilities for the F1 ML pipeline.")
def cli():
    pass


cli.add_command(ingest)
cli.add_command(features)


def run():
    """Entry point used by the Poetry script hook."""
    cli()
