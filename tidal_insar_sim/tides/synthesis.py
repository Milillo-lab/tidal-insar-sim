"""Harmonic tide synthesis and least-squares harmonic analysis.

The direct path for production use is `CATSBackend.predict_tide_m()`, which
leans on pyTMD's nodal-correction-aware predictor. This module exists to:

    (a) let the Simulator synthesize tides from a dict of `{constituent: (amp, phase)}`
        without re-reading the CATS2008 grid on every call, and
    (b) provide a simple harmonic analyzer for the `test_tide_validation` self-consistency
        gate (synthesize -> analyze must recover the input amplitudes).

Formulation
-----------
    h(t) = Sum_i  A_i * cos(omega_i * t - phi_i)    # OTIS convention
where `t` is hours from the reference epoch, omega_i = 2*pi / T_i, T_i in hours,
phi_i in radians. This matches what pyTMD's OTIS reader returns for amplitudes
and phases (see `CATSBackend.extract_constants`).

Nodal corrections are not applied here; when the Simulator needs high-fidelity
multi-year synthesis it should call `CATSBackend.predict_tide_m` directly.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tidal_insar_sim.tides.mock import CONSTITUENT_PERIODS_H


def synthesize_from_constants(
    t_hours: ArrayLike,
    constants: Mapping[str, tuple[float, float]],
) -> NDArray[np.float64]:
    """Synthesize a tide time series from `{name: (amp_m, phase_deg)}` constants.

    Parameters
    ----------
    t_hours : array-like
        Hours from the reference epoch.
    constants : mapping
        `{name: (amplitude_m, phase_deg)}`. Names must be in
        `CONSTITUENT_PERIODS_H`.
    """
    t = np.asarray(t_hours, dtype=np.float64)
    h = np.zeros_like(t)
    for name, (amp_m, phase_deg) in constants.items():
        if np.isnan(amp_m) or np.isnan(phase_deg):
            continue
        key = name.upper()
        if key not in CONSTITUENT_PERIODS_H:
            continue
        period_h = CONSTITUENT_PERIODS_H[key]
        omega = 2.0 * np.pi / period_h
        phi = np.deg2rad(phase_deg)
        h = h + amp_m * np.cos(omega * t - phi)
    return h


def analyze_harmonic(
    t_hours: ArrayLike,
    h_m: ArrayLike,
    constituents: tuple[str, ...] = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"),
) -> dict[str, tuple[float, float]]:
    """Least-squares harmonic fit: recover (amp, phase) per constituent.

    Fit model: h(t) ~ Sum_i A_i cos(omega_i t) + B_i sin(omega_i t).
    Then amplitude = sqrt(A^2 + B^2), phase = atan2(B, A) in degrees.

    This is used by the validation gate: synthesize a 1-year tide, analyze it,
    and recovered amplitudes must match the input amplitudes within floating
    precision. It is NOT a production-quality nodal-correction analysis.
    """
    t = np.asarray(t_hours, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)
    if t.shape != h.shape:
        msg = f"t_hours and h_m must have the same shape: got {t.shape} vs {h.shape}"
        raise ValueError(msg)

    keys = [c.upper() for c in constituents if c.upper() in CONSTITUENT_PERIODS_H]
    n_const = len(keys)
    # Design matrix: [cos(om_i t)  sin(om_i t)] for each constituent
    matrix = np.empty((len(t), 2 * n_const), dtype=np.float64)
    for j, key in enumerate(keys):
        omega = 2.0 * np.pi / CONSTITUENT_PERIODS_H[key]
        matrix[:, 2 * j] = np.cos(omega * t)
        matrix[:, 2 * j + 1] = np.sin(omega * t)

    coeffs, *_ = np.linalg.lstsq(matrix, h, rcond=None)
    out: dict[str, tuple[float, float]] = {}
    for j, key in enumerate(keys):
        a = coeffs[2 * j]
        b = coeffs[2 * j + 1]
        amp = float(np.hypot(a, b))
        phase_deg = float(np.degrees(np.arctan2(b, a))) % 360.0
        out[key] = (amp, phase_deg)
    return out
