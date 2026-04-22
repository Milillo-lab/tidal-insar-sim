"""Simulator: binds a Sensor and Site to produce DDInSAR fringe-count predictions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tidal_insar_sim.physics.fringe import TripletSweep, rigid_triplet_sweep
from tidal_insar_sim.report import TripletReport
from tidal_insar_sim.sensor import Sensor
from tidal_insar_sim.site import Site

TideFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class Simulator:
    """Bind a Sensor and Site and run sweeps / Monte Carlo against a tide function.

    For Checkpoint 1 the `tide_fn` is a mock synthesizer (see `tidal_insar_sim.tides.mock`).
    Checkpoint 2 wires this to a CATS2008-backed `tide_fn` via pyTMD.
    """

    sensor: Sensor
    site: Site
    tide_fn: TideFn | None = None

    DEFAULT_SWEEP_DAYS: float = 29.53
    DEFAULT_STEP_HOURS: float = 1.0

    def _require_tide_fn(self) -> TideFn:
        if self.tide_fn is None:
            msg = (
                "Simulator.tide_fn is unset. In Checkpoint 1 you must supply a mock tide "
                "callable; pass e.g. `Simulator(sensor=..., site=..., "
                "tide_fn=mixed_m2_k1_tide())`. CATS2008 lookup will be wired in Checkpoint 2."
            )
            raise RuntimeError(msg)
        return self.tide_fn

    def sweep_triplets(
        self,
        step_hours: float | None = None,
        duration_days: float | None = None,
    ) -> TripletReport:
        """Rigid-triplet phase sweep at 1-h step over one synodic month by default."""
        tide_fn = self._require_tide_fn()
        sweep: TripletSweep = rigid_triplet_sweep(
            tide_fn=tide_fn,
            step_hours=step_hours if step_hours is not None else self.DEFAULT_STEP_HOURS,
            duration_days=duration_days if duration_days is not None else self.DEFAULT_SWEEP_DAYS,
            repeat_days=self.sensor.repeat_days,
            wavelength_m=self.sensor.wavelength_m,
            incidence_deg=self.sensor.incidence_deg,
        )
        return TripletReport(
            sweep=sweep,
            site_name=self.site.name,
            sensor_name=self.sensor.name,
        )

    def confidence(
        self,
        triplet_start_hours: float,
        *,
        jitter_hours: float = 1.0,
        n_realisations: int = 10_000,
        rng: np.random.Generator | None = None,
        min_fringes_threshold: float = 1.0,
    ) -> float:
        """±jitter_hours acquisition-time Monte Carlo confidence.

        Definition (per user Apr 2026 Q&A):
            confidence = 1 - P(fringes < `min_fringes_threshold`)

        That is, the fraction of realisations that still clear the detectability floor
        under uniform acquisition-time jitter at each of the three epochs. 1.0 = rock
        solid; ~0 = triplet is operationally brittle.
        """
        tide_fn = self._require_tide_fn()
        rng = rng if rng is not None else np.random.default_rng()
        repeat_hours = self.sensor.repeat_days * 24.0
        d1 = rng.uniform(-jitter_hours, jitter_hours, n_realisations)
        d2 = rng.uniform(-jitter_hours, jitter_hours, n_realisations)
        d3 = rng.uniform(-jitter_hours, jitter_hours, n_realisations)
        h1 = tide_fn(triplet_start_hours + d1)
        h2 = tide_fn(triplet_start_hours + repeat_hours + d2)
        h3 = tide_fn(triplet_start_hours + 2.0 * repeat_hours + d3)
        h_dd = h1 - 2.0 * h2 + h3
        theta = np.deg2rad(self.sensor.incidence_deg)
        fringes = np.abs(h_dd) * np.cos(theta) / (self.sensor.wavelength_m / 2.0)
        return float(1.0 - np.mean(fringes < min_fringes_threshold))
