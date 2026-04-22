"""Mock tide synthesizers for testing and development.

These are *not* intended for scientific use. They reproduce the analytic tide signal
from the validated prototype scripts so the Simulator can be wired up and exercised
before the CATS2008 backend lands at Checkpoint 2.

`mixed_m2_k1_tide` matches `reference/prototype_v1_three_date.py` and
`reference/prototype_sweep.py`.

`multi_constituent_tide` matches the per-site synthesizer in
`reference/prototype_sites.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

TideFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]

CONSTITUENT_PERIODS_H: dict[str, float] = {
    "M2": 12.4206,
    "S2": 12.0000,
    "N2": 12.6583,
    "K2": 11.9672,
    "K1": 23.9345,
    "O1": 25.8193,
    "P1": 24.0659,
    "Q1": 26.8684,
}


@dataclass(frozen=True)
class MixedTide:
    """Deterministic mixed-semidiurnal tide parameterised by M2 and K1 only.

    Matches prototype_v1 / prototype_sweep's `tide(t_hours, ...)` signature.
    """

    amp_m2_m: float = 0.80
    amp_k1_m: float = 0.50
    phase_m2_rad: float = 0.0
    phase_k1_rad: float = 1.1

    def __call__(self, t_hours: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t_hours, dtype=np.float64)
        t_m2 = CONSTITUENT_PERIODS_H["M2"]
        t_k1 = CONSTITUENT_PERIODS_H["K1"]
        out: NDArray[np.float64] = (
            self.amp_m2_m * np.cos(2.0 * np.pi * t / t_m2 + self.phase_m2_rad)
            + self.amp_k1_m * np.cos(2.0 * np.pi * t / t_k1 + self.phase_k1_rad)
        )
        return out


def mixed_m2_k1_tide(
    amp_m2_m: float = 0.80,
    amp_k1_m: float = 0.50,
    phase_m2_rad: float = 0.0,
    phase_k1_rad: float = 1.1,
) -> MixedTide:
    """Return a `MixedTide` callable matching prototype_v1_three_date defaults."""
    return MixedTide(
        amp_m2_m=amp_m2_m,
        amp_k1_m=amp_k1_m,
        phase_m2_rad=phase_m2_rad,
        phase_k1_rad=phase_k1_rad,
    )


def multi_constituent_tide(
    amplitudes_m: Mapping[str, float],
    phases_deg: Mapping[str, float],
) -> TideFn:
    """Return a tide callable summing the listed constituents at amplitude / phase.

    Parameters
    ----------
    amplitudes_m : mapping
        `{"M2": 0.8, "K1": 0.5, ...}` — amplitudes in metres.
    phases_deg : mapping
        `{"M2": 0.0, "K1": 63.0, ...}` — phases in degrees. The convention is
        h(t) = sum_i A_i cos(omega_i t - phi_i), matching prototype_sites.py.

    Unknown constituents raise KeyError immediately (fail loudly — no silent drop).
    """
    for name in amplitudes_m:
        if name not in CONSTITUENT_PERIODS_H:
            msg = (
                f"Unknown tidal constituent {name!r}. "
                f"Known: {sorted(CONSTITUENT_PERIODS_H)}"
            )
            raise KeyError(msg)

    periods = np.array(
        [CONSTITUENT_PERIODS_H[c] for c in amplitudes_m],
        dtype=np.float64,
    )
    amps = np.array(list(amplitudes_m.values()), dtype=np.float64)
    phases = np.deg2rad(
        np.array([phases_deg.get(c, 0.0) for c in amplitudes_m], dtype=np.float64)
    )

    def _tide(t_hours: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t_hours, dtype=np.float64)
        omega = 2.0 * np.pi / periods
        # Broadcast across constituents then sum. Works for scalar and ndarray inputs.
        arg = omega[:, None] * t.reshape(-1)[None, :] - phases[:, None]
        summed = (amps[:, None] * np.cos(arg)).sum(axis=0)
        result: NDArray[np.float64] = summed.reshape(t.shape)
        return result

    return _tide
