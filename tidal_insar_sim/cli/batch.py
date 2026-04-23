"""`tidal-insar-sim batch` — compare many sensors at many sites."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.table import Table

if TYPE_CHECKING:
    import pandas as pd

    from tidal_insar_sim.site import Site

from tidal_insar_sim.cli._common import CONSOLE, resolve_sensor, resolve_site


@click.command("batch")
@click.option("--sites", "sites_yaml", type=click.Path(path_type=Path, exists=True),
              default=None,
              help="YAML with a 'sites:' list (preset names or {name,lat,lon,...} dicts).")
@click.option("--site-preset", "site_presets", multiple=True,
              help="Alternative to --sites: list site presets (--site-preset THWAITES ...).")
@click.option("--sensor", "sensor_names", multiple=True, required=True,
              help="Sensor preset(s). Pass multiple times for multiple sensors.")
@click.option("--output", "output_dir", required=True,
              type=click.Path(path_type=Path, file_okay=False),
              help="Output dir; writes batch.parquet + heatmap.csv per metric.")
@click.option("--metric", default="P_usable_ge_3fr", show_default=True,
              help="Metric for the printed heatmap.")
def batch_command(
    sites_yaml: Path | None,
    site_presets: tuple[str, ...],
    sensor_names: tuple[str, ...],
    output_dir: Path,
    metric: str,
) -> None:
    """Run the sweep over a grid of (site, sensor) pairs."""
    from tidal_insar_sim.batch import (
        batch_compare,
        batch_to_parquet,
        pivot_heatmap,
    )

    sites = _load_sites(sites_yaml, site_presets)
    if not sites:
        msg = "No sites provided. Pass --sites YAML or one or more --site-preset NAME."
        raise click.UsageError(msg)
    sensors = [resolve_sensor(s) for s in sensor_names]

    output_dir.mkdir(parents=True, exist_ok=True)
    df = batch_compare(sites, sensors)
    parquet_path = output_dir / "batch.parquet"
    batch_to_parquet(df, parquet_path)
    heatmap = pivot_heatmap(df, metric=metric)
    heatmap_path = output_dir / f"heatmap_{metric}.csv"
    heatmap.to_csv(heatmap_path)
    _render_heatmap(heatmap, metric, parquet_path, heatmap_path)


def _load_sites(
    sites_yaml: Path | None, site_presets: tuple[str, ...],
) -> list[Site]:
    from tidal_insar_sim.site import Site

    if sites_yaml is not None:
        import yaml

        data = yaml.safe_load(sites_yaml.read_text(encoding="utf-8"))
        if "sites" not in data:
            msg = f"{sites_yaml} must have a top-level 'sites:' list."
            raise click.BadParameter(msg, param_hint="--sites")
        out = []
        for entry in data["sites"]:
            if isinstance(entry, str):
                out.append(resolve_site(
                    preset=entry, lat=None, lon=None, name=None, ice_thickness_m=None,
                ))
                continue
            out.append(Site.from_coords(
                lat=float(entry["lat"]),
                lon=float(entry["lon"]),
                name=entry.get("name"),
                ice_thickness_m=float(entry.get("ice_thickness_m", 400.0)),
            ))
        return out
    return [
        resolve_site(preset=p, lat=None, lon=None, name=None, ice_thickness_m=None)
        for p in site_presets
    ]


def _render_heatmap(
    heatmap: pd.DataFrame, metric: str, parquet_path: Path, heatmap_path: Path,
) -> None:
    t = Table(title=f"Batch {metric}")
    t.add_column("site", style="cyan")
    for col in heatmap.columns:
        t.add_column(str(col), justify="right")
    for row_name, row in heatmap.iterrows():
        t.add_row(str(row_name), *(f"{v:.3f}" for v in row.values))
    CONSOLE.print(t)
    CONSOLE.print(
        f"Parquet -> [bold]{parquet_path}[/]  ·  heatmap CSV -> [bold]{heatmap_path}[/]"
    )
