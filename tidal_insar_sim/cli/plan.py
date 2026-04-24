"""`tidal-insar-sim plan` — generate a top-N acquisition plan."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.table import Table

if TYPE_CHECKING:
    import pandas as pd

from tidal_insar_sim.cli._common import (
    CONSOLE,
    constellation_options,
    parse_iso_datetime,
    resolve_constellation,
    resolve_sensor,
    resolve_site,
)


@click.command("plan")
@click.option("--sensor", required=True, help="Sensor band (X-BAND, C-BAND, L-BAND).")
@click.option("--incidence", type=float, default=None,
              help="Override incidence angle (degrees).")
@click.option("--preset", default=None, help="Site preset (e.g. THWAITES).")
@click.option("--lat", type=float, default=None)
@click.option("--lon", type=float, default=None)
@click.option("--name", default=None)
@click.option("--ice-thickness", "ice_thickness_m", type=float, default=None)
@click.option("--start", "start_raw", required=True,
              help="Start datetime, ISO 8601 (e.g. 2026-06-01T00:00).")
@click.option("--end", "end_raw", required=True,
              help="End datetime, ISO 8601.")
@click.option("-n", "--top-n", "n", type=int, default=10, show_default=True,
              help="Top-N triplets to return.")
@click.option("--min-fringes", type=float, default=3.0, show_default=True,
              help="Minimum predicted fringes to consider.")
@click.option("--output", "output_prefix", required=True,
              type=click.Path(path_type=Path),
              help="Output prefix; writes {prefix}.csv and {prefix}.ics.")
@constellation_options
def plan_command(
    sensor: str,
    incidence: float | None,
    preset: str | None,
    lat: float | None,
    lon: float | None,
    name: str | None,
    ice_thickness_m: float | None,
    start_raw: str,
    end_raw: str,
    n: int,
    min_fringes: float,
    output_prefix: Path,
    constellation_preset: str | None,
    n_sats: int | None,
    repeat_per_sat: float | None,
    satellites_file: Path | None,
) -> None:
    """Top-N acquisition plan in a date window, with confidence and .ics export."""
    from tidal_insar_sim.planner import plan_acquisitions, to_ics
    from tidal_insar_sim.simulator import Simulator

    sensor_obj = resolve_sensor(sensor, incidence_deg=incidence)
    site_obj = resolve_site(
        preset=preset, lat=lat, lon=lon, name=name, ice_thickness_m=ice_thickness_m,
    )
    constellation = resolve_constellation(
        name=constellation_preset, n_sats=n_sats,
        repeat_per_sat=repeat_per_sat, satellites_file=satellites_file,
    )
    start_dt = parse_iso_datetime(start_raw, param="start")
    end_dt = parse_iso_datetime(end_raw, param="end")

    sim = Simulator(sensor=sensor_obj, site=site_obj, constellation=constellation)
    df = plan_acquisitions(
        sim, start_date=start_dt, end_date=end_dt,
        n=n, min_fringes=min_fringes,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    ics_path = output_prefix.with_suffix(".ics")
    df.to_csv(csv_path, index=False)
    if not df.empty:
        to_ics(df, ics_path)
    _render_plan_table(df, site_obj.name, sensor_obj.name, csv_path, ics_path)


def _render_plan_table(
    df: pd.DataFrame, site_name: str, sensor_name: str, csv: Path, ics: Path,
) -> None:
    if df.empty:
        CONSOLE.print("[yellow]No triplets >= min_fringes in the window.[/]")
        CONSOLE.print(f"Empty plan written to [bold]{csv}[/].")
        return
    t = Table(title=f"Top-{len(df)} acquisition plan  —  {site_name} / {sensor_name}")
    t.add_column("#", justify="right")
    t.add_column("t1", style="cyan")
    t.add_column("t2")
    t.add_column("t3")
    t.add_column("fringes", justify="right")
    t.add_column("confidence", justify="right")
    for _, row in df.iterrows():
        t.add_row(
            str(int(row["rank"])),
            str(row["t1_date"]),
            str(row["t2_date"]),
            str(row["t3_date"]),
            f"{float(row['predicted_fringes']):.2f}",
            f"{float(row['confidence']):.3f}",
        )
    CONSOLE.print(t)
    CONSOLE.print(f"CSV -> [bold]{csv}[/]  ·  ICS -> [bold]{ics}[/]")
