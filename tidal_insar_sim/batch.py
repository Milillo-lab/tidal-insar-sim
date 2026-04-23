"""Batch comparison across (site, sensor) grids (M6).

Given a list of sites and sensors, run `Simulator.sweep_triplets()` for each pair
and collect the summary statistics into a `pandas.DataFrame` — one row per
(site, sensor). Useful for operational planning ("which sensor for which glacier")
and for the batch page of the web UI.

Use cases
---------
- `df = batch_compare([Site.THWAITES, Site.RUTFORD], [Sensor.NISAR_L, Sensor.SENTINEL_1_DUAL])`
- `batch_to_parquet(df, "batch.parquet")` for persistence
- `pivot_heatmap(df, metric="P_usable_ge_3fr")` for visual comparison
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tidal_insar_sim.sensor import Sensor
from tidal_insar_sim.simulator import Simulator
from tidal_insar_sim.site import Site

if TYPE_CHECKING:
    pass

TideFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class BatchRow:
    """One entry in a batch comparison: the inputs and the resulting summary."""

    site: Site
    sensor: Sensor
    summary: dict[str, object]


def batch_compare(
    sites: Iterable[Site],
    sensors: Iterable[Sensor],
    *,
    tide_fn_factory: Callable[[Site], TideFn | None] | None = None,
    step_hours: float = 1.0,
    duration_days: float = 29.53,
) -> pd.DataFrame:
    """Return a DataFrame with one row per (site, sensor) pair.

    Parameters
    ----------
    sites, sensors : iterables of Site and Sensor objects.
    tide_fn_factory : optional callable `(site) -> tide_fn`. If None, each
        Simulator auto-loads CATS2008 (requires CATS2008 on disk).
    """
    rows: list[dict[str, object]] = []
    sites_list = list(sites)
    sensors_list = list(sensors)
    for site in sites_list:
        tide_fn = tide_fn_factory(site) if tide_fn_factory is not None else None
        for sensor in sensors_list:
            sim = Simulator(sensor=sensor, site=site, tide_fn=tide_fn)
            report = sim.sweep_triplets(
                step_hours=step_hours, duration_days=duration_days
            )
            row: dict[str, object] = {
                "site": site.name,
                "lat": site.lat,
                "lon": site.lon,
                "ice_thickness_m": site.ice_thickness_m,
                "sensor": sensor.name,
                "wavelength_m": sensor.wavelength_m,
                "repeat_days": sensor.repeat_days,
                "incidence_deg": sensor.incidence_deg,
            }
            row.update(report.summary())
            rows.append(row)
    return pd.DataFrame(rows)


def batch_to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write the batch DataFrame to Parquet. Requires `pyarrow` installed."""
    df.to_parquet(path, engine="pyarrow", index=False)


def batch_from_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def pivot_heatmap(
    df: pd.DataFrame,
    metric: str = "P_usable_ge_3fr",
) -> pd.DataFrame:
    """Pivot the (site, sensor) long-format table into a site-by-sensor matrix
    for the chosen metric. Each cell is the metric value at that pair.

    The result can be passed directly to `matplotlib.pyplot.imshow` or
    `pandas.DataFrame.style.background_gradient` for visual inspection.
    """
    if metric not in df.columns:
        msg = f"metric {metric!r} not in DataFrame columns: {list(df.columns)}"
        raise KeyError(msg)
    return df.pivot(index="site", columns="sensor", values=metric)
