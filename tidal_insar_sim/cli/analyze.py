"""`tidal-insar-sim analyze` — single-site DDInSAR sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd
from rich.table import Table

from tidal_insar_sim.cli._common import (
    CONSOLE,
    constellation_options,
    resolve_constellation,
    resolve_sensor,
    resolve_site,
)


@click.command("analyze")
@click.option("--sensor", required=True, help="Sensor band (X-BAND, C-BAND, L-BAND).")
@click.option("--incidence", type=float, default=None,
              help="Override incidence angle (degrees).")
@click.option("--preset", default=None, help="Site preset (e.g. THWAITES). Overrides --lat/--lon.")
@click.option("--lat", type=float, default=None, help="WGS-84 latitude (-90..0 in v0.1).")
@click.option("--lon", type=float, default=None, help="WGS-84 longitude (-180..180).")
@click.option("--name", default=None, help="Human-readable site name (for --lat/--lon mode).")
@click.option("--ice-thickness", "ice_thickness_m", type=float, default=None,
              help="Ice thickness in metres (default 400).")
@click.option("--duration", "duration_days", type=float, default=29.53, show_default=True,
              help="Sweep duration in days.")
@click.option("--step-hours", type=float, default=1.0, show_default=True,
              help="Triplet sweep step in hours.")
@click.option("--mode",
              type=click.Choice(["rigid", "multi_baseline", "any_triplet"]),
              default="rigid", show_default=True,
              help=("rigid: one B = constellation effective. "
                    "multi_baseline: one sweep per valid B. "
                    "any_triplet: every (t1,t2,t3) from schedule."))
@click.option("--output", "output_dir", required=True,
              type=click.Path(path_type=Path, file_okay=False),
              help="Output directory; writes summary.json, sweep.csv, site.geojson, site.kml.")
@click.option("--no-geojson", is_flag=True, default=False, help="Skip GeoJSON export.")
@click.option("--no-kml", is_flag=True, default=False, help="Skip KML export.")
@constellation_options
def analyze_command(
    sensor: str,
    incidence: float | None,
    preset: str | None,
    lat: float | None,
    lon: float | None,
    name: str | None,
    ice_thickness_m: float | None,
    duration_days: float,
    step_hours: float,
    mode: str,
    output_dir: Path,
    no_geojson: bool,
    no_kml: bool,
    constellation_preset: str | None,
    n_sats: int | None,
    repeat_per_sat: float | None,
    satellites_file: Path | None,
) -> None:
    """Single-site sweep. Writes summary.json + sweep.csv and optionally
    site.geojson + site.kml. With --mode=multi_baseline, writes one CSV per
    valid baseline; with --mode=any_triplet, writes all-triplets.csv.
    """
    from tidal_insar_sim.simulator import Simulator

    sensor_obj = resolve_sensor(sensor, incidence_deg=incidence)
    site_obj = resolve_site(
        preset=preset, lat=lat, lon=lon, name=name, ice_thickness_m=ice_thickness_m,
    )
    constellation = resolve_constellation(
        name=constellation_preset, n_sats=n_sats,
        repeat_per_sat=repeat_per_sat, satellites_file=satellites_file,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = Simulator(sensor=sensor_obj, site=site_obj, constellation=constellation)

    if mode == "rigid":
        report = sim.sweep_triplets(step_hours=step_hours, duration_days=duration_days)
        summary = report.summary()
        summary["constellation_n_sats"] = constellation.n_satellites
        summary["constellation_B_eff_days"] = constellation.effective_repeat_days()
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
        report.to_csv(output_dir / "sweep.csv")
        if not no_geojson:
            report.to_geojson(output_dir / "site.geojson")
        if not no_kml:
            report.to_kml(output_dir / "site.kml")
        _render_summary_table(summary, site_obj.name, sensor_obj.name,
                              constellation, output_dir)

    elif mode == "multi_baseline":
        import numpy as np

        sweeps = sim.multi_baseline_sweep(
            step_hours=step_hours, duration_days=duration_days,
        )
        rows = []
        for B, sw in sorted(sweeps.items()):
            rows.append({
                "B_days": B,
                "P_usable_ge_3fr": float(np.mean(sw.fringes >= 3.0)),
                "P_robust_ge_5fr": float(np.mean(sw.fringes >= 5.0)),
                "P_null_lt_0p5fr": float(np.mean(sw.fringes < 0.5)),
                "fringe_mean": float(np.mean(sw.fringes)),
                "fringe_max": float(np.max(sw.fringes)),
            })
            pd.DataFrame({
                "delta_t_hours": sw.delta_t_hours,
                "h1_m": sw.h1_m,
                "h2_m": sw.h2_m,
                "h3_m": sw.h3_m,
                "h_dd_m": sw.h_dd_m,
                "fringes": sw.fringes,
            }).to_csv(output_dir / f"sweep_B{B:05.2f}.csv", index=False)
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_dir / "multi_baseline_summary.csv", index=False)
        _render_multi_baseline_table(summary_df, site_obj.name, sensor_obj.name,
                                      constellation, output_dir)

    elif mode == "any_triplet":
        import numpy as np

        ats = sim.any_triplet_sweep(duration_days=duration_days)
        pd.DataFrame({
            "t1_days": ats.t1_days,
            "t2_days": ats.t2_days,
            "t3_days": ats.t3_days,
            "span_days": ats.t3_days - ats.t1_days,
            "h_dd_m": ats.h_dd_m,
            "fringes": ats.fringes,
        }).to_csv(output_dir / "any_triplets.csv", index=False)
        summary = {
            "mode": "any_triplet",
            "n_triplets": len(ats),
            "P_usable_ge_3fr": float(np.mean(ats.fringes >= 3.0)),
            "P_robust_ge_5fr": float(np.mean(ats.fringes >= 5.0)),
            "fringe_mean": float(np.mean(ats.fringes)),
            "fringe_max": float(np.max(ats.fringes)),
            "constellation_n_sats": constellation.n_satellites,
            "constellation_B_eff_days": constellation.effective_repeat_days(),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
        CONSOLE.print(
            f"any_triplet: [bold]{len(ats)}[/] triplets, "
            f"P(>=3fr)={summary['P_usable_ge_3fr']:.1%}, "
            f"fringe_mean={summary['fringe_mean']:.2f}"
        )
        CONSOLE.print(f"Wrote [bold]{output_dir}[/].")


def _render_summary_table(
    summary: dict[str, Any], site_name: str, sensor_name: str,
    constellation: Any, out: Path,
) -> None:
    t = Table(title=f"tidal-insar-sim analyze  -  {site_name} / {sensor_name}")
    t.add_column("metric", style="cyan")
    t.add_column("value", justify="right")

    def _pct(k: str) -> str: return f"{float(summary[k]):.1%}"
    def _num(k: str, fmt: str = ".2f") -> str: return format(float(summary[k]), fmt)

    rows = [
        ("verdict", str(summary["verdict"])),
        ("P(>= 3 fringes) usable", _pct("P_usable_ge_3fr")),
        ("P(>= 5 fringes) robust", _pct("P_robust_ge_5fr")),
        ("P(< 0.5 fr) null",       _pct("P_null_lt_0p5fr")),
        ("fringe mean",            _num("fringe_mean")),
        ("fringe median",          _num("fringe_median")),
        ("fringe max",             _num("fringe_max")),
        ("hDD range (m)",          f"{summary['hDD_range_m']}"),
        ("constellation",
         f"{constellation.n_satellites} sat(s), "
         f"B_eff={constellation.effective_repeat_days():.2f} d"),
        ("sweep triplets",         f"{summary['n_triplets']}"),
    ]
    for m, v in rows:
        t.add_row(m, v)
    CONSOLE.print(t)
    CONSOLE.print(f"Outputs written to [bold]{out}[/].")


def _render_multi_baseline_table(
    df: pd.DataFrame, site_name: str, sensor_name: str, constellation: Any, out: Path,
) -> None:
    t = Table(title=f"Multi-baseline  -  {site_name} / {sensor_name}")
    for col in ("B_days", "P_usable_ge_3fr", "P_robust_ge_5fr",
                "P_null_lt_0p5fr", "fringe_mean", "fringe_max"):
        t.add_column(col, justify="right")
    for _, row in df.iterrows():
        t.add_row(
            f"{float(row['B_days']):.2f}",
            f"{float(row['P_usable_ge_3fr']):.1%}",
            f"{float(row['P_robust_ge_5fr']):.1%}",
            f"{float(row['P_null_lt_0p5fr']):.1%}",
            f"{float(row['fringe_mean']):.2f}",
            f"{float(row['fringe_max']):.2f}",
        )
    CONSOLE.print(t)
    CONSOLE.print(
        f"Constellation: {constellation.n_satellites} sat(s), "
        f"B_eff={constellation.effective_repeat_days():.2f} d"
    )
    CONSOLE.print(f"Outputs written to [bold]{out}[/].")
