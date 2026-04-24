"""Matplotlib / Folium helpers for the Streamlit pages."""

from tidal_insar_sim.web.components.plots import (
    plot_fringe_map,
    plot_sweep_curve,
    plot_tide_time_series,
    plot_triplet_histogram,
)

__all__ = [
    "plot_fringe_map",
    "plot_sweep_curve",
    "plot_tide_time_series",
    "plot_triplet_histogram",
]
