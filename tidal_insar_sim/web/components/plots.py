"""Matplotlib figure builders. Pure helpers — no Streamlit import here."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from tidal_insar_sim.report import FRINGE_THRESHOLDS

if TYPE_CHECKING:
    from tidal_insar_sim.report import TripletReport
    from tidal_insar_sim.synthesis import FringeMap


def plot_sweep_curve(report: TripletReport) -> Figure:
    """Fringes vs triplet-start offset, with threshold markers."""
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=110)
    days = report.delta_t_hours / 24.0
    ax.plot(days, report.fringes, color="#1f4e79", lw=1.2)
    ax.axhline(FRINGE_THRESHOLDS["null"], color="#c72b2b", ls=":", lw=0.8,
               label="null (<0.5 fr)")
    ax.axhline(FRINGE_THRESHOLDS["marginal"], color="#2e8b57", ls="--", lw=0.8,
               label="usable (>=3 fr)")
    ax.axhline(FRINGE_THRESHOLDS["good"], color="#b279a2", ls="-.", lw=0.8,
               label="robust (>=5 fr)")
    ax.set_xlabel("Triplet start offset (days)")
    ax.set_ylabel("|fringes| (LOS)")
    ax.set_title(f"{report.site_name}  -  {report.sensor_name}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_triplet_histogram(report: TripletReport, bins: int = 40) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=110)
    ax.hist(report.fringes, bins=bins, color="#1f4e79", alpha=0.8, edgecolor="white")
    ax.axvline(FRINGE_THRESHOLDS["null"], color="#c72b2b", ls=":", lw=0.8)
    ax.axvline(FRINGE_THRESHOLDS["marginal"], color="#2e8b57", ls="--", lw=0.8)
    ax.axvline(FRINGE_THRESHOLDS["good"], color="#b279a2", ls="-.", lw=0.8)
    summary = report.summary()
    ax.axvline(float(summary["fringe_mean"]), color="k", lw=1.5,
               label=f"mean={summary['fringe_mean']:.2f}")
    ax.set_xlabel("|fringes|")
    ax.set_ylabel("# triplets")
    ax.set_title("Distribution over triplet phases (1 month)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_fringe_map(fmap: FringeMap, title: str | None = None) -> Figure:
    """Wrapped-phase field with HSV colormap."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
    h, w = fmap.shape
    pix_x, _, origin_x, _, pix_y, origin_y = fmap.transform
    extent = (
        origin_x / 1000.0,
        (origin_x + w * pix_x) / 1000.0,
        (origin_y + h * pix_y) / 1000.0,
        origin_y / 1000.0,
    )
    ax.imshow(
        fmap.wrapped_phase,
        extent=extent, origin="upper",
        cmap="hsv", vmin=-np.pi, vmax=np.pi,
        interpolation="bilinear",
    )
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    title_str = title or f"{fmap.site_name} / {fmap.sensor_name}"
    ax.set_title(
        f"{title_str}\n"
        f"h_DD = {fmap.h_dd_m:+.2f} m  |  peak = {fmap.n_fringes_peak:.2f} fringes"
    )
    fig.tight_layout()
    return fig


def plot_tide_time_series(
    delta_t_hours: NDArray[np.float64],
    h1_m: NDArray[np.float64],
    h_dd_m: NDArray[np.float64],
    *,
    site_name: str = "",
) -> Figure:
    """Tide at t1 plus DD residual in one plot — gives the user a feel for
    the tidal regime driving the fringe count."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5), dpi=110, sharex=True)
    days = delta_t_hours / 24.0
    ax1.plot(days, h1_m, color="#1f4e79", lw=1.0)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("h(t1)  (m)")
    ax1.set_title(f"Tide at first acquisition  -  {site_name}")
    ax1.grid(alpha=0.3)

    ax2.plot(days, h_dd_m, color="#c72b2b", lw=1.0)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("h_DD  (m)")
    ax2.set_xlabel("Triplet start offset (days)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig
