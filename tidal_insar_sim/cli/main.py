"""tidal-insar-sim CLI entry point."""

from __future__ import annotations

import click

from tidal_insar_sim import __version__
from tidal_insar_sim.cli.analyze import analyze_command
from tidal_insar_sim.cli.batch import batch_command
from tidal_insar_sim.cli.plan import plan_command
from tidal_insar_sim.cli.setup_cats import setup_cats_command
from tidal_insar_sim.cli.synthesize import synthesize_command


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="tidal-insar-sim")
def cli() -> None:
    """tidal-insar-sim — DDInSAR fringe-count simulator for grounding zones.

    See `tidal-insar-sim <COMMAND> --help` for detailed options.
    """


cli.add_command(setup_cats_command)
cli.add_command(analyze_command)
cli.add_command(plan_command)
cli.add_command(synthesize_command)
cli.add_command(batch_command)


if __name__ == "__main__":  # pragma: no cover
    cli()
