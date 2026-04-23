"""`tidal-insar-sim setup-cats` — verify and install CATS2008 from a zip."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel

from tidal_insar_sim.cli._common import CONSOLE, ERR_CONSOLE
from tidal_insar_sim.tides.cats2008 import DEFAULT_DATA_DIR
from tidal_insar_sim.tides.download import (
    install_cats2008_from_zip,
    manual_download_instructions,
)


@click.command("setup-cats")
@click.option(
    "--path", "zip_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Path to the user-downloaded CATS2008.zip (from USAP-DC).",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_DATA_DIR,
    show_default=True,
    help="Install root. CATS2008/ will be created here.",
)
@click.option(
    "--skip-md5", is_flag=True, default=False,
    help="Skip the MD5 verification (for recreated/truncated zips).",
)
def setup_cats_command(
    zip_path: Path | None,
    data_dir: Path,
    skip_md5: bool,
) -> None:
    """Install and verify a downloaded CATS2008 tide model.

    The dataset must be fetched manually from USAP-DC (reCAPTCHA-gated);
    run `tidal-insar-sim setup-cats` with no --path to see instructions.
    """
    if zip_path is None:
        CONSOLE.print(Panel.fit(
            manual_download_instructions(),
            title="CATS2008 — manual download required",
            style="cyan",
        ))
        return

    CONSOLE.print(f"Installing CATS2008 from [bold]{zip_path}[/] ...")
    try:
        report = install_cats2008_from_zip(
            zip_path=zip_path, data_dir=data_dir, skip_md5=skip_md5,
        )
    except Exception as exc:
        ERR_CONSOLE.print(f"Install failed: {exc}")
        raise click.ClickException(str(exc)) from exc

    lines = [
        f"zip:         {report.zip_path}",
        f"md5:         {report.zip_md5} ({'OK' if report.zip_md5_ok else 'MISMATCH'})",
        f"data_dir:    {report.data_dir}",
        f"already:     {report.already_installed}",
        f"files:       {len(report.installed_files)} extracted",
        f"pyTMD probe: {'OK' if report.pytmd_probe_ok else 'n/a'}",
    ]
    CONSOLE.print(Panel.fit("\n".join(lines), title="CATS2008 install report", style="green"))
