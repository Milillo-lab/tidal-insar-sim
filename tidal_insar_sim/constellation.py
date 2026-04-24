"""Satellite constellation model.

A `Constellation` is an ordered collection of `Satellite` descriptors. Each
satellite has a phase offset (days into the orbit cycle at t=0) and an orbit
repeat interval (days). The constellation exposes:

- `effective_repeat_days()` — the smallest inter-acquisition gap, which is the
  natural rigid-B temporal baseline for DDInSAR triplets.
- `acquisition_times(window_days)` — the full sorted schedule of acquisitions.
- `valid_baselines_days(window_days, max_days)` — every unique rigid B that
  can be realised from the schedule.

Triplet sweeps then come in two flavours:

1. **Rigid-B** — pick a single baseline B, sweep the start phase. This is
   the classical DDInSAR assumption and what `Simulator.sweep_triplets()`
   uses.
2. **Any-3** — enumerate every `(t_i < t_j < t_k)` combination from the
   constellation's schedule with `t_k - t_i <= max_span_days`. Breaks the
   equal-baseline constraint but captures real-world constellation geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Satellite:
    """A single SAR satellite with a fixed orbit-repeat interval.

    `phase_offset_days` places the first acquisition at `t = phase_offset_days`
    (days since epoch). Subsequent acquisitions are at
    `phase_offset_days + k * repeat_days` for k = 0, 1, 2, ….
    """

    name: str
    phase_offset_days: float
    repeat_days: float


@dataclass(frozen=True)
class Constellation:
    """Ordered tuple of `Satellite`s sharing a SAR band."""

    satellites: tuple[Satellite, ...]

    def __post_init__(self) -> None:
        if not self.satellites:
            msg = "Constellation must contain at least one Satellite."
            raise ValueError(msg)
        for sat in self.satellites:
            if sat.repeat_days <= 0:
                msg = f"Satellite {sat.name!r} has non-positive repeat_days={sat.repeat_days}"
                raise ValueError(msg)
            if sat.phase_offset_days < 0:
                msg = (
                    f"Satellite {sat.name!r} has negative phase_offset_days="
                    f"{sat.phase_offset_days}; phases must be >= 0"
                )
                raise ValueError(msg)

    @property
    def n_satellites(self) -> int:
        return len(self.satellites)

    @classmethod
    def single(cls, repeat_days: float, name: str = "Sat-1") -> Constellation:
        """A 1-satellite constellation."""
        return cls(satellites=(Satellite(name=name, phase_offset_days=0.0,
                                          repeat_days=float(repeat_days)),))

    @classmethod
    def equally_phased(cls, n: int, repeat_days: float,
                       name_prefix: str = "Sat") -> Constellation:
        """`n` satellites with the same orbit repeat, phased evenly.

        Phase offsets: 0, repeat/n, 2*repeat/n, ..., (n-1)*repeat/n.
        """
        if n < 1:
            msg = f"n must be >= 1, got {n}"
            raise ValueError(msg)
        step = float(repeat_days) / n
        sats = tuple(
            Satellite(name=f"{name_prefix}-{i + 1}",
                      phase_offset_days=i * step,
                      repeat_days=float(repeat_days))
            for i in range(n)
        )
        return cls(satellites=sats)

    def acquisition_times(self, window_days: float) -> NDArray[np.float64]:
        """All scheduled acquisition times in `[0, window_days)`, sorted."""
        if window_days <= 0:
            msg = f"window_days must be > 0, got {window_days}"
            raise ValueError(msg)
        times: list[float] = []
        for sat in self.satellites:
            k = 0
            while True:
                t = sat.phase_offset_days + k * sat.repeat_days
                if t >= window_days:
                    break
                times.append(t)
                k += 1
        return np.sort(np.asarray(times, dtype=np.float64))

    def effective_repeat_days(self, window_days: float = 30.0) -> float:
        """Smallest gap between consecutive acquisitions in the first
        `window_days` (days). For an equally-phased constellation this is
        `repeat / n`. For arbitrary phasing it is the minimum consecutive gap.
        """
        acq = self.acquisition_times(window_days=window_days)
        if acq.size < 2:
            return float(self.satellites[0].repeat_days)
        return float(np.min(np.diff(acq)))

    def valid_baselines_days(
        self,
        window_days: float = 365.0,
        max_days: float | None = None,
    ) -> NDArray[np.float64]:
        """Unique positive inter-acquisition intervals realisable from the
        schedule, up to `max_days`. Used by the multi-baseline sweep."""
        acq = self.acquisition_times(window_days=window_days)
        if acq.size < 2:
            return np.asarray([], dtype=np.float64)
        if max_days is None:
            max_days = float(max(sat.repeat_days for sat in self.satellites)) * 1.1
        gaps: list[float] = []
        n = acq.size
        for i in range(n):
            for j in range(i + 1, n):
                d = float(acq[j] - acq[i])
                if d > max_days:
                    break
                gaps.append(d)
        # Round to 0.01 day to collapse floating-point duplicates
        arr = np.round(np.asarray(gaps, dtype=np.float64), 2)
        return np.unique(arr)

    def describe(self) -> str:
        rows = [
            f"  {s.name:<10s}  phase={s.phase_offset_days:6.2f} d  "
            f"repeat={s.repeat_days:5.2f} d"
            for s in self.satellites
        ]
        eff = self.effective_repeat_days()
        return (
            f"Constellation of {self.n_satellites} satellite(s), "
            f"effective repeat = {eff:.2f} d:\n" + "\n".join(rows)
        )

    NISAR: ClassVar[Constellation]
    ALOS: ClassVar[Constellation]
    SENTINEL_1_SINGLE: ClassVar[Constellation]
    SENTINEL_1_DUAL: ClassVar[Constellation]
    RCM: ClassVar[Constellation]


Constellation.NISAR = Constellation.single(repeat_days=12.0, name="NISAR")
Constellation.ALOS = Constellation.single(repeat_days=14.0, name="ALOS")
Constellation.SENTINEL_1_SINGLE = Constellation.single(repeat_days=12.0, name="S1A")
Constellation.SENTINEL_1_DUAL = Constellation.equally_phased(n=2, repeat_days=12.0,
                                                              name_prefix="S1")
Constellation.RCM = Constellation.equally_phased(n=3, repeat_days=12.0, name_prefix="RCM")


CONSTELLATION_PRESETS: dict[str, Constellation] = {
    "NISAR": Constellation.NISAR,
    "ALOS": Constellation.ALOS,
    "SENTINEL-1-SINGLE": Constellation.SENTINEL_1_SINGLE,
    "SENTINEL-1-DUAL": Constellation.SENTINEL_1_DUAL,
    "RCM": Constellation.RCM,
}
