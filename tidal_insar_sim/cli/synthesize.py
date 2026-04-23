"""`tidal-insar-sim synthesize` — render a synthetic DDInSAR GeoTIFF."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click

from tidal_insar_sim.cli._common import (
    CONSOLE,
    parse_iso_datetime,
    resolve_sensor,
    resolve_site,
)
from tidal_insar_sim.planner import REFERENCE_EPOCH


@click.command("synthesize")
@click.option("--sensor", required=True)
@click.option("--preset", default=None)
@click.option("--lat", type=float, default=None)
@click.option("--lon", type=float, default=None)
@click.option("--name", default=None)
@click.option("--ice-thickness", "ice_thickness_m", type=float, default=None)
@click.option("--triplet-start-date", "triplet_start_raw", required=True,
              help="First-acquisition datetime, ISO 8601 (e.g. 2026-07-15T00:00Z).")
@click.option("--size", "size_km", nargs=2, type=float, default=(15.0, 10.0),
              show_default=True, help="Scene width height in km (space-separated pair).")
@click.option("--pixel-m", type=float, default=15.0, show_default=True)
@click.option("--gamma-grounded", type=float, default=0.88, show_default=True)
@click.option("--gamma-shelf", type=float, default=0.60, show_default=True)
@click.option("--multi-look", type=int, default=8, show_default=True)
@click.option("--seed", "noise_seed", type=int, default=1, show_default=True)
@click.option("--output", required=True, type=click.Path(path_type=Path),
              help="Output GeoTIFF path (or prefix; .tif added if missing).")
def synthesize_command(
    sensor: str,
    preset: str | None,
    lat: float | None,
    lon: float | None,
    name: str | None,
    ice_thickness_m: float | None,
    triplet_start_raw: str,
    size_km: tuple[float, float],
    pixel_m: float,
    gamma_grounded: float,
    gamma_shelf: float,
    multi_look: int,
    noise_seed: int,
    output: Path,
) -> None:
    """Build a 3-band DDInSAR GeoTIFF (wrapped_phase, coherence, h_DD)."""
    from tidal_insar_sim.simulator import Simulator

    sensor_obj = resolve_sensor(sensor)
    site_obj = resolve_site(
        preset=preset, lat=lat, lon=lon, name=name, ice_thickness_m=ice_thickness_m,
    )
    start_dt = parse_iso_datetime(triplet_start_raw, param="triplet-start-date")
    dt_hours = _hours_since_epoch(start_dt)

    sim = Simulator(sensor=sensor_obj, site=site_obj)
    width_m = size_km[0] * 1000.0
    height_m = size_km[1] * 1000.0
    fmap = sim.synthesize_ddinsar(
        triplet_start_hours=dt_hours,
        size_m=(width_m, height_m),
        pixel_m=pixel_m,
        gamma_grounded=gamma_grounded,
        gamma_shelf=gamma_shelf,
        multi_look=multi_look,
        noise_seed=noise_seed,
    )
    if output.suffix == "":
        output = output.with_suffix(".tif")
    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.to_geotiff(output)
    CONSOLE.print(
        f"Synthesized [bold]{fmap.site_name}[/] / [bold]{fmap.sensor_name}[/]: "
        f"h_DD={fmap.h_dd_m:+.2f} m, peak={fmap.n_fringes_peak:.2f} fringes  "
        f"-> {output}"
    )


def _hours_since_epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - REFERENCE_EPOCH).total_seconds() / 3600.0
