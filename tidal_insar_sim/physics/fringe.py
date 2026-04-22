"""Rigid-triplet phase sweep.

For a 3-date InSAR triplet with fixed temporal baseline B (the sensor repeat interval),
slide the whole triplet in time by Delta_t and, at each step, compute:

    h1 = h(Delta_t)
    h2 = h(Delta_t + B)
    h3 = h(Delta_t + 2B)
    h_DD   = h1 - 2 h2 + h3
    nfringe = |h_DD| cos(theta) / (lambda / 2)

This is not an uncertainty analysis — it is a *mission-planning sensitivity*: for every
possible triplet phase, what fringe count do we get? It is what converts a tide time
series into operational statistics like P(>= 3 fringes) and P(< 1 fringe).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

TideFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class TripletSweep:
    """Raw output of `rigid_triplet_sweep`."""

    step_hours: float
    duration_days: float
    repeat_days: float
    wavelength_m: float
    incidence_deg: float

    delta_t_hours: NDArray[np.float64]
    h1_m: NDArray[np.float64]
    h2_m: NDArray[np.float64]
    h3_m: NDArray[np.float64]
    h_dd_m: NDArray[np.float64]
    fringes: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.delta_t_hours.size)


def rigid_triplet_sweep(
    tide_fn: TideFn,
    step_hours: float,
    duration_days: float,
    repeat_days: float,
    wavelength_m: float,
    incidence_deg: float,
) -> TripletSweep:
    """Sweep a rigid 3-date triplet across `duration_days` in `step_hours` steps.

    Parameters
    ----------
    tide_fn : callable
        `tide_fn(t_hours) -> h(t_hours)` vectorised over ndarray input.
    step_hours : float
        Sweep step (hours).
    duration_days : float
        Sweep span (days). Default use is one synodic month (29.53 d).
    repeat_days : float
        Sensor temporal baseline between adjacent acquisitions. NISAR = 12.
    wavelength_m : float
        SAR wavelength.
    incidence_deg : float
        Nominal incidence angle at scene centre.
    """
    if step_hours <= 0:
        msg = f"step_hours must be > 0, got {step_hours}"
        raise ValueError(msg)
    if duration_days <= 0:
        msg = f"duration_days must be > 0, got {duration_days}"
        raise ValueError(msg)
    if repeat_days <= 0:
        msg = f"repeat_days must be > 0, got {repeat_days}"
        raise ValueError(msg)

    total_hours = duration_days * 24.0
    delta_t = np.arange(0.0, total_hours + step_hours, step_hours, dtype=np.float64)

    repeat_hours = repeat_days * 24.0
    h1 = tide_fn(delta_t)
    h2 = tide_fn(delta_t + repeat_hours)
    h3 = tide_fn(delta_t + 2.0 * repeat_hours)
    h_dd = h1 - 2.0 * h2 + h3

    theta = np.deg2rad(incidence_deg)
    half_lam = wavelength_m / 2.0
    fringes = np.abs(h_dd) * np.cos(theta) / half_lam

    return TripletSweep(
        step_hours=step_hours,
        duration_days=duration_days,
        repeat_days=repeat_days,
        wavelength_m=wavelength_m,
        incidence_deg=incidence_deg,
        delta_t_hours=delta_t,
        h1_m=h1,
        h2_m=h2,
        h3_m=h3,
        h_dd_m=h_dd,
        fringes=fringes,
    )
