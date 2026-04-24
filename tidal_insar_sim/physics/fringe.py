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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tidal_insar_sim.constellation import Constellation

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


def multi_baseline_sweep(
    tide_fn: TideFn,
    constellation: Constellation,
    step_hours: float,
    duration_days: float,
    wavelength_m: float,
    incidence_deg: float,
    *,
    max_baseline_days: float | None = None,
) -> dict[float, TripletSweep]:
    """One rigid-triplet sweep per valid baseline in the constellation schedule.

    Returns {B_days: TripletSweep}. The user then picks the best B (or compares
    all) — this is how a multi-satellite constellation with uneven phasing
    offers the planner multiple DDInSAR cadence options.

    Parameters
    ----------
    tide_fn : callable(t_hours) -> h(t_hours).
    constellation : Constellation instance.
    step_hours, duration_days : same as `rigid_triplet_sweep`.
    wavelength_m, incidence_deg : same as `rigid_triplet_sweep`.
    max_baseline_days : optional cap on B. Defaults to 1.1 x the longest
        single-satellite repeat (captures re-visits; drops cross-cycle pairs).
    """
    baselines = constellation.valid_baselines_days(
        window_days=duration_days, max_days=max_baseline_days,
    )
    out: dict[float, TripletSweep] = {}
    for B in baselines:
        if 3.0 * float(B) > duration_days * 24.0 * 24.0:  # sanity: need 3 acq in window
            continue
        sweep = rigid_triplet_sweep(
            tide_fn=tide_fn,
            step_hours=step_hours,
            duration_days=duration_days,
            repeat_days=float(B),
            wavelength_m=wavelength_m,
            incidence_deg=incidence_deg,
        )
        out[float(B)] = sweep
    return out


@dataclass(frozen=True)
class AnyTripletSweep:
    """Raw output of `any_triplet_sweep`.

    Unlike `TripletSweep`, the acquisition offsets are *not* equally spaced:
    each row is an arbitrary `(t1, t2, t3)` triplet from the constellation
    schedule. Use this for constellations where the physical DDInSAR
    assumption (equal temporal baselines) is relaxed.
    """

    constellation_name: str
    duration_days: float
    wavelength_m: float
    incidence_deg: float

    t1_days: NDArray[np.float64]
    t2_days: NDArray[np.float64]
    t3_days: NDArray[np.float64]
    h1_m: NDArray[np.float64]
    h2_m: NDArray[np.float64]
    h3_m: NDArray[np.float64]
    h_dd_m: NDArray[np.float64]
    fringes: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.t1_days.size)


def any_triplet_sweep(
    tide_fn: TideFn,
    constellation: Constellation,
    duration_days: float,
    wavelength_m: float,
    incidence_deg: float,
    *,
    max_span_days: float | None = None,
    require_equal_baseline: bool = False,
    equal_baseline_tol_days: float = 0.1,
    constellation_name: str = "constellation",
) -> AnyTripletSweep:
    """Enumerate every valid `(t1 < t2 < t3)` triplet from the constellation.

    Parameters
    ----------
    constellation : Constellation.
    duration_days : window to schedule acquisitions in.
    max_span_days : drop triplets with `t3 - t1 > max_span_days`. Default is
        2.2 x the longest single-satellite repeat.
    require_equal_baseline : if True, filter to rigid-B triplets only
        (|(t2-t1) - (t3-t2)| <= equal_baseline_tol_days). Useful to keep the
        classical flow-cancellation property while still exploring the full
        schedule.

    Returns
    -------
    `AnyTripletSweep` — one row per valid triplet.
    """
    acq = constellation.acquisition_times(window_days=duration_days)
    n = int(acq.size)
    if n < 3:
        msg = (
            f"Constellation produces only {n} acquisitions in {duration_days} d; "
            f"need at least 3 for a triplet."
        )
        raise ValueError(msg)

    if max_span_days is None:
        max_span_days = float(
            max(sat.repeat_days for sat in constellation.satellites)
        ) * 2.2

    t1_list: list[float] = []
    t2_list: list[float] = []
    t3_list: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                span = float(acq[k] - acq[i])
                if span > max_span_days:
                    break
                if require_equal_baseline:
                    b1 = float(acq[j] - acq[i])
                    b2 = float(acq[k] - acq[j])
                    if abs(b1 - b2) > equal_baseline_tol_days:
                        continue
                t1_list.append(float(acq[i]))
                t2_list.append(float(acq[j]))
                t3_list.append(float(acq[k]))

    t1_arr = np.asarray(t1_list, dtype=np.float64)
    t2_arr = np.asarray(t2_list, dtype=np.float64)
    t3_arr = np.asarray(t3_list, dtype=np.float64)

    h1 = tide_fn(t1_arr * 24.0)
    h2 = tide_fn(t2_arr * 24.0)
    h3 = tide_fn(t3_arr * 24.0)
    h_dd = h1 - 2.0 * h2 + h3
    theta = np.deg2rad(incidence_deg)
    half_lam = wavelength_m / 2.0
    fringes = np.abs(h_dd) * np.cos(theta) / half_lam

    return AnyTripletSweep(
        constellation_name=constellation_name,
        duration_days=duration_days,
        wavelength_m=wavelength_m,
        incidence_deg=incidence_deg,
        t1_days=t1_arr,
        t2_days=t2_arr,
        t3_days=t3_arr,
        h1_m=h1,
        h2_m=h2,
        h3_m=h3,
        h_dd_m=h_dd,
        fringes=fringes,
    )
