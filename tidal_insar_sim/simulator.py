"""Simulator — binds a Sensor, a Constellation, and a Site to produce
DDInSAR fringe-count predictions under CATS2008 tide forcing.

Two sweep modes are exposed:

- `sweep_triplets()` — rigid-B (classical DDInSAR). The baseline B defaults
  to `constellation.effective_repeat_days()` but can be overridden.
- `multi_baseline_sweep()` — one rigid-B sweep per unique gap that can be
  realised from the constellation's acquisition schedule. Lets the planner
  compare "use the 6-day sub-sampling or the 12-day one?" at a glance.
- `any_triplet_sweep()` — enumerate every valid `(t1<t2<t3)` triplet from
  the schedule (optionally filtered to equal-baseline ones). Breaks the
  rigid-B constraint — use when simulating a constellation whose acquisition
  phasing is operationally important.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from tidal_insar_sim.physics.fringe import (
    AnyTripletSweep,
    TripletSweep,
    any_triplet_sweep,
    multi_baseline_sweep,
    rigid_triplet_sweep,
)
from tidal_insar_sim.report import TripletReport
from tidal_insar_sim.sensor import Sensor
from tidal_insar_sim.site import Site

if TYPE_CHECKING:
    from tidal_insar_sim.constellation import Constellation
    from tidal_insar_sim.synthesis import FringeMap

TideFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class Simulator:
    """Bind a Sensor, Constellation, and Site; run sweeps / Monte Carlo.

    If `tide_fn` is not supplied, CATS2008 is auto-loaded at the site's
    (lat, lon) via pyTMD on first use. Set `tide_fn=` explicitly to inject a
    mock tide (tests) or a custom backend.

    If `constellation` is not supplied, a 1-satellite 12-day constellation
    is used (NISAR-like default).
    """

    sensor: Sensor
    site: Site
    constellation: Constellation | None = None
    tide_fn: TideFn | None = None

    DEFAULT_SWEEP_DAYS: float = 29.53
    DEFAULT_STEP_HOURS: float = 1.0

    _resolved_tide_fn: TideFn | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.constellation is None:
            from tidal_insar_sim.constellation import Constellation

            self.constellation = Constellation.single(repeat_days=12.0, name="default")

    # ---------------------------------------------------------------- tide

    def _require_tide_fn(self) -> TideFn:
        if self.tide_fn is not None:
            return self.tide_fn
        if self._resolved_tide_fn is not None:
            return self._resolved_tide_fn
        from tidal_insar_sim.tides.cats2008 import CATSBackend, make_cats_tide_fn

        backend = CATSBackend()
        self._resolved_tide_fn = make_cats_tide_fn(
            backend=backend, lat=self.site.lat, lon=self.site.lon
        )
        return self._resolved_tide_fn

    @property
    def tide_source(self) -> str:
        if self.tide_fn is not None:
            return "user-supplied"
        if self._resolved_tide_fn is not None:
            return f"CATS2008@{self.site.lat:+.2f},{self.site.lon:+.2f}"
        return "unresolved"

    # ---------------------------------------------------------------- sweeps

    @property
    def _effective_baseline_days(self) -> float:
        assert self.constellation is not None  # set in __post_init__
        return self.constellation.effective_repeat_days()

    def sweep_triplets(
        self,
        step_hours: float | None = None,
        duration_days: float | None = None,
        *,
        baseline_days: float | None = None,
    ) -> TripletReport:
        """Rigid-triplet phase sweep. Returns a `TripletReport`.

        `baseline_days` defaults to the constellation's effective repeat.
        Pass explicitly (e.g. `baseline_days=12.0`) to simulate a specific
        DDInSAR cadence.
        """
        tide_fn = self._require_tide_fn()
        B = float(baseline_days if baseline_days is not None else self._effective_baseline_days)
        sweep: TripletSweep = rigid_triplet_sweep(
            tide_fn=tide_fn,
            step_hours=step_hours if step_hours is not None else self.DEFAULT_STEP_HOURS,
            duration_days=duration_days if duration_days is not None else self.DEFAULT_SWEEP_DAYS,
            repeat_days=B,
            wavelength_m=self.sensor.wavelength_m,
            incidence_deg=self.sensor.incidence_deg,
        )
        return TripletReport(
            sweep=sweep,
            site_name=self.site.name,
            sensor_name=self.sensor.name,
            simulator=self,
        )

    def multi_baseline_sweep(
        self,
        step_hours: float | None = None,
        duration_days: float | None = None,
        *,
        max_baseline_days: float | None = None,
    ) -> dict[float, TripletSweep]:
        """One sweep per valid baseline from the constellation's schedule.
        Returns {B_days: TripletSweep}."""
        assert self.constellation is not None
        tide_fn = self._require_tide_fn()
        return multi_baseline_sweep(
            tide_fn=tide_fn,
            constellation=self.constellation,
            step_hours=step_hours if step_hours is not None else self.DEFAULT_STEP_HOURS,
            duration_days=duration_days if duration_days is not None else self.DEFAULT_SWEEP_DAYS,
            wavelength_m=self.sensor.wavelength_m,
            incidence_deg=self.sensor.incidence_deg,
            max_baseline_days=max_baseline_days,
        )

    def any_triplet_sweep(
        self,
        duration_days: float | None = None,
        *,
        max_span_days: float | None = None,
        require_equal_baseline: bool = False,
    ) -> AnyTripletSweep:
        """Enumerate every `(t1<t2<t3)` triplet from the constellation schedule."""
        assert self.constellation is not None
        tide_fn = self._require_tide_fn()
        return any_triplet_sweep(
            tide_fn=tide_fn,
            constellation=self.constellation,
            duration_days=duration_days if duration_days is not None else self.DEFAULT_SWEEP_DAYS,
            wavelength_m=self.sensor.wavelength_m,
            incidence_deg=self.sensor.incidence_deg,
            max_span_days=max_span_days,
            require_equal_baseline=require_equal_baseline,
            constellation_name=self._constellation_label(),
        )

    # ---------------------------------------------------------------- fringe maps

    def synthesize_ddinsar(
        self,
        triplet_start_hours: float,
        *,
        size_m: tuple[float, float] = (15_000.0, 10_000.0),
        pixel_m: float = 15.0,
        gamma_grounded: float = 0.88,
        gamma_shelf: float = 0.60,
        multi_look: int = 8,
        noise_seed: int | None = 1,
        baseline_days: float | None = None,
    ) -> FringeMap:
        """Build a 2-D synthetic fringe map for a rigid triplet starting at
        `triplet_start_hours`.  Uses `baseline_days` (default: constellation
        effective repeat) for the (t2-t1, t3-t2) interval.
        """
        from tidal_insar_sim.synthesis import synthesize_fringe_map

        tide_fn = self._require_tide_fn()
        B_hours = float(
            baseline_days if baseline_days is not None else self._effective_baseline_days
        ) * 24.0
        triplet_times = np.array(
            [triplet_start_hours,
             triplet_start_hours + B_hours,
             triplet_start_hours + 2.0 * B_hours],
            dtype=np.float64,
        )
        heights = tide_fn(triplet_times)
        h1, h2, h3 = float(heights[0]), float(heights[1]), float(heights[2])

        return synthesize_fringe_map(
            sensor=self.sensor,
            site=self.site,
            h1_m=h1,
            h2_m=h2,
            h3_m=h3,
            size_m=size_m,
            pixel_m=pixel_m,
            gamma_grounded=gamma_grounded,
            gamma_shelf=gamma_shelf,
            multi_look=multi_look,
            noise_seed=noise_seed,
        )

    # ---------------------------------------------------------------- confidence

    def confidence(
        self,
        triplet_start_hours: float,
        *,
        jitter_hours: float = 1.0,
        n_realisations: int = 10_000,
        rng: np.random.Generator | None = None,
        min_fringes_threshold: float = 1.0,
        baseline_days: float | None = None,
    ) -> float:
        """±jitter_hours acquisition-time Monte Carlo confidence.

        confidence = 1 - P(fringes < `min_fringes_threshold`).
        """
        tide_fn = self._require_tide_fn()
        rng = rng if rng is not None else np.random.default_rng()
        B_hours = float(
            baseline_days if baseline_days is not None else self._effective_baseline_days
        ) * 24.0
        d1 = rng.uniform(-jitter_hours, jitter_hours, n_realisations)
        d2 = rng.uniform(-jitter_hours, jitter_hours, n_realisations)
        d3 = rng.uniform(-jitter_hours, jitter_hours, n_realisations)
        h1 = tide_fn(triplet_start_hours + d1)
        h2 = tide_fn(triplet_start_hours + B_hours + d2)
        h3 = tide_fn(triplet_start_hours + 2.0 * B_hours + d3)
        h_dd = h1 - 2.0 * h2 + h3
        theta = np.deg2rad(self.sensor.incidence_deg)
        fringes = np.abs(h_dd) * np.cos(theta) / (self.sensor.wavelength_m / 2.0)
        return float(1.0 - np.mean(fringes < min_fringes_threshold))

    # ---------------------------------------------------------------- helpers

    def _constellation_label(self) -> str:
        assert self.constellation is not None
        return f"{self.sensor.band}/{self.constellation.n_satellites}sats"
