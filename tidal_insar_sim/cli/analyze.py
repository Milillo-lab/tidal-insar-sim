"""`tidal-insar-sim analyze` — single-site DDInSAR sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from tidal_insar_sim.cli._common import CONSOLE, resolve_sensor, resolve_site


@click.command("analyze")
@click.option("--sensor", required=True, help="Sensor preset, e.g. NISAR-L.")
@click.option("--preset", default=None, help="Site preset (e.g. THWAITES). Overrides --lat/--lon.")
@click.option("--lat", type=float, default=None, help="WGS-84 latitude (-90..90).")
@click.option("--lon", type=float, default=None, help="WGS-84 longitude (-180..180).")
@click.option("--name", default=None, help="Human-readable site name (for --lat/--lon mode).")
@click.option("--ice-thickness", "ice_thickness_m", type=float, default=None,
              help="Ice thickness in metres (default 400).")
@click.option("--duration", "duration_days", type=float, default=29.53, show_default=True,
              help="Sweep duration in days.")
@click.option("--step-hours", type=float, default=1.0, show_default=True,
              help="Triplet sweep step in hours.")
@click.option("--output", "output_dir", required=True,
              type=click.Path(path_type=Path, file_okay=False),
              help="Output directory; summary.json, sweep.csv, site.geojson, site.kml.")
@click.option("--no-geojson", is_flag=True, default=False, help="Skip GeoJSON export.")
@click.option("--no-kml", is_flag=True, default=False, help="Skip KML export.")
def analyze_command(
    sensor: str,
    preset: str | None,
    lat: float | None,
    lon: float | None,
    name: str | None,
    ice_thickness_m: float | None,
    duration_days: float,
    step_hours: float,
    output_dir: Path,
    no_geojson: bool,
    no_kml: bool,
) -> None:
    """Single-site sweep. Writes summary.json, sweep.csv, and optionally
    site.geojson + site.kml to --output.
    """
    from tidal_insar_sim.simulator import Simulator

    sensor_obj = resolve_sensor(sensor)
    site_obj = resolve_site(
        preset=preset, lat=lat, lon=lon, name=name, ice_thickness_m=ice_thickness_m,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = Simulator(sensor=sensor_obj, site=site_obj)
    report = sim.sweep_triplets(step_hours=step_hours, duration_days=duration_days)
    summary = report.summary()

    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "sweep.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report.to_csv(csv_path)

    if not no_geojson:
        report.to_geojson(output_dir / "site.geojson")
    if not no_kml:
        report.to_kml(output_dir / "site.kml")

    _render_summary_table(summary, site_obj.name, sensor_obj.name, output_dir)


def _render_summary_table(
    summary: dict[str, Any], site_name: str, sensor_name: str, out: Path,
) -> None:
    t = Table(title=f"tidal-insar-sim analyze  -  {site_name} / {sensor_name}")
    t.add_column("metric", style="cyan")
    t.add_column("value", justify="right")

    def _pct(k: str) -> str:
        return f"{float(summary[k]):.1%}"

    def _num(k: str, fmt: str = ".2f") -> str:
        return format(float(summary[k]), fmt)

    rows = [
        ("verdict", str(summary["verdict"])),
        ("P(>= 3 fringes) usable", _pct("P_usable_ge_3fr")),
        ("P(>= 5 fringes) robust", _pct("P_robust_ge_5fr")),
        ("P(< 0.5 fr) null",       _pct("P_null_lt_0p5fr")),
        ("fringe mean",            _num("fringe_mean")),
        ("fringe median",          _num("fringe_median")),
        ("fringe max",             _num("fringe_max")),
        ("hDD range (m)",          f"{summary['hDD_range_m']}"),
        ("sweep triplets",         f"{summary['n_triplets']}"),
    ]
    for m, v in rows:
        t.add_row(m, v)
    CONSOLE.print(t)
    CONSOLE.print(f"Outputs written to [bold]{out}[/].")
