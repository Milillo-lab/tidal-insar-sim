"""Output model for a `Simulator.sweep_triplets()` run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from tidal_insar_sim.physics.fringe import TripletSweep

if TYPE_CHECKING:
    import pandas as pd

    from tidal_insar_sim.simulator import Simulator

# Operational thresholds from prototype_sweep.py
FRINGE_THRESHOLDS = {
    "null": 0.5,
    "weak": 2.0,
    "marginal": 3.0,
    "good": 5.0,
    "strong": 7.0,
}

VERDICT_EXCELLENT = 0.85
VERDICT_GOOD = 0.60
VERDICT_MARGINAL = 0.35


def _verdict(p_usable: float) -> str:
    if p_usable >= VERDICT_EXCELLENT:
        return "excellent"
    if p_usable >= VERDICT_GOOD:
        return "good"
    if p_usable >= VERDICT_MARGINAL:
        return "marginal"
    return "poor"


@dataclass
class TripletReport:
    """Statistics of a rigid-triplet sweep plus planner-ready accessors."""

    sweep: TripletSweep
    site_name: str
    sensor_name: str
    simulator: Simulator | None = field(default=None, repr=False)

    @property
    def fringes(self) -> NDArray[np.float64]:
        return self.sweep.fringes

    @property
    def h_dd_m(self) -> NDArray[np.float64]:
        return self.sweep.h_dd_m

    @property
    def delta_t_hours(self) -> NDArray[np.float64]:
        return self.sweep.delta_t_hours

    def best_triplet_h(self) -> float:
        """Delta_t (hours from epoch) of the maximum-fringe triplet."""
        idx = int(np.argmax(self.fringes))
        return float(self.delta_t_hours[idx])

    def worst_triplet_h(self) -> float:
        """Delta_t of the minimum-fringe (NULL) triplet."""
        idx = int(np.argmin(self.fringes))
        return float(self.delta_t_hours[idx])

    def p_at_least(self, n_fringes: float) -> float:
        return float(np.mean(self.fringes >= n_fringes))

    def p_below(self, n_fringes: float) -> float:
        return float(np.mean(self.fringes < n_fringes))

    def summary(self) -> dict[str, Any]:
        fr = self.fringes
        hdd = self.h_dd_m
        p_usable = self.p_at_least(FRINGE_THRESHOLDS["marginal"])
        p_robust = self.p_at_least(FRINGE_THRESHOLDS["good"])
        p_null = self.p_below(FRINGE_THRESHOLDS["null"])
        verdict = _verdict(p_usable)
        return {
            "site": self.site_name,
            "sensor": self.sensor_name,
            "n_triplets": int(fr.size),
            "P_usable_ge_3fr": p_usable,
            "P_robust_ge_5fr": p_robust,
            "P_null_lt_0p5fr": p_null,
            "fringe_mean": float(np.mean(fr)),
            "fringe_median": float(np.median(fr)),
            "fringe_max": float(np.max(fr)),
            "fringe_p05": float(np.percentile(fr, 5)),
            "fringe_p95": float(np.percentile(fr, 95)),
            "hDD_range_m": [float(np.min(hdd)), float(np.max(hdd))],
            "hDD_std_m": float(np.std(hdd)),
            "verdict": verdict,
        }

    def to_csv(self, path: str | Path) -> None:
        """Per-triplet sweep data + summary as a CSV (one row per triplet)."""
        import pandas as pd

        df = pd.DataFrame({
            "delta_t_hours": self.sweep.delta_t_hours,
            "h1_m": self.sweep.h1_m,
            "h2_m": self.sweep.h2_m,
            "h3_m": self.sweep.h3_m,
            "h_dd_m": self.sweep.h_dd_m,
            "fringes": self.sweep.fringes,
        })
        df["site"] = self.site_name
        df["sensor"] = self.sensor_name
        df.to_csv(path, index=False)

    def to_geojson(self, path: str | Path, *, include_polygon: bool = True) -> None:
        from tidal_insar_sim.io.geojson_export import report_to_geojson

        report_to_geojson(self, path, include_polygon=include_polygon)

    def to_kml(self, path: str | Path, *, include_polygon: bool = True) -> None:
        from tidal_insar_sim.io.kml_export import report_to_kml

        report_to_kml(self, path, include_polygon=include_polygon)

    def recommended_triplets(
        self,
        start_date: datetime,
        end_date: datetime,
        n: int = 10,
        min_fringes: float = 3.0,
        *,
        step_hours: float = 1.0,
        jitter_hours: float = 1.0,
        n_mc_realisations: int = 4_000,
        rng_seed: int | None = 2026,
    ) -> pd.DataFrame:
        """Top-`n` triplets in [start_date, end_date] with confidence. See
        `tidal_insar_sim.planner.plan_acquisitions`."""
        if self.simulator is None:
            msg = (
                "TripletReport.recommended_triplets requires a bound Simulator. "
                "Call via `sim.sweep_triplets().recommended_triplets(...)`, or "
                "use `tidal_insar_sim.planner.plan_acquisitions(sim, ...)` directly."
            )
            raise RuntimeError(msg)
        from tidal_insar_sim.planner import plan_acquisitions

        return plan_acquisitions(
            self.simulator,
            start_date=start_date,
            end_date=end_date,
            n=n,
            min_fringes=min_fringes,
            step_hours=step_hours,
            jitter_hours=jitter_hours,
            n_mc_realisations=n_mc_realisations,
            rng_seed=rng_seed,
        )

    def class_fractions(self) -> dict[str, float]:
        """Fraction of triplet phases in each named fringe class."""
        fr = self.fringes
        masks = {
            "NULL": fr < FRINGE_THRESHOLDS["null"],
            "WEAK": (fr >= FRINGE_THRESHOLDS["null"]) & (fr < FRINGE_THRESHOLDS["weak"]),
            "MARGINAL": (fr >= FRINGE_THRESHOLDS["weak"]) & (fr < FRINGE_THRESHOLDS["marginal"]),
            "GOOD": (fr >= FRINGE_THRESHOLDS["marginal"]) & (fr < FRINGE_THRESHOLDS["good"]),
            "STRONG": (fr >= FRINGE_THRESHOLDS["good"]) & (fr < FRINGE_THRESHOLDS["strong"]),
            "PEAK": fr >= FRINGE_THRESHOLDS["strong"],
        }
        return {k: float(np.mean(v)) for k, v in masks.items()}
