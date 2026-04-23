"""Simulator: binds a Sensor and Site to produce DDInSAR fringe-count predictions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from tidal_insar_sim.physics.fringe import TripletSweep, rigid_triplet_sweep
from tidal_insar_sim.report import TripletReport
from tidal_insar_sim.sensor import Sensor
from tidal_insar_sim.site import Site

if TYPE_CHECKING:
    from tidal_insar_sim.synthesis import FringeMap

TideFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class Simulator:
    """Bind a Sensor and Site and run sweeps / Monte Carlo against a tide function.

    If `tide_fn` is not supplied, CATS2008 is auto-loaded at the site's (lat, lon)
    via pyTMD on first use. Set `tide_fn=` explicitly to inject a mock tide (tests)
    or a custom backend.
    """

    sensor: Sensor
    site: Site
    tide_fn: TideFn | None = None

    DEFAULT_SWEEP_DAYS: float = 29.53
    DEFAULT_STEP_HOURS: float = 1.0

    _resolved_tide_fn: TideFn | None = field(default=None, init=False, repr=False)

    def _require_tide_fn(self) -> TideFn:
        if self.tide_fn is not None:
            return self.tide_fn
        if self._resolved_tide_fn is not None:
            return self._resolved_tide_fn
        # Lazy-build a CATS2008-backed tide_fn at the site's lat/lon.
        from tidal_insar_sim.tides.cats2008 import CATSBackend, make_cats_tide_fn

        backend = CATSBackend()
        self._resolved_tide_fn = make_cats_tide_fn(
            backend=backend, lat=self.site.lat, lon=self.site.lon
        )
        return self._resolved_tide_fn

    @property
    def tide_source(self) -> str:
        """Human-readable description of where tides are coming from."""
        if self.tide_fn is not None:
            return "user-supplied"
        if self._resolved_tide_fn is not None:
            return f"CATS2008@{self.site.lat:+.2f},{self.site.lon:+.2f}"
        return "unresolved"

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
            simulator=self,
        )

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
    ) -> FringeMap:
        """Build a 2D synthetic fringe map for a triplet starting at
        `triplet_start_hours` hours after the CATS2008 reference epoch.

        Delegates to `tidal_insar_sim.synthesis.synthesize_fringe_map`.
        """
        from tidal_insar_sim.synthesis import synthesize_fringe_map

        tide_fn = self._require_tide_fn()
        repeat_hours = self.sensor.repeat_days * 24.0
        triplet_times = np.array(
            [triplet_start_hours,
             triplet_start_hours + repeat_hours,
             triplet_start_hours + 2.0 * repeat_hours],
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
