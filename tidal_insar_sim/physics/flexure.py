"""Elastic plate flexure of a clamped ice shelf under a uniform ocean-tide load.

References
----------
Rignot, E., Mouginot, J., Scheuchl, B. (2011). GRL 38.
Fricker, H.A., Padman, L. (2006). GRL 33.
Walker, R.T., Christianson, K., Parizek, B.R., et al. (2013). EPSL 395.

Formulation
-----------
Flexural rigidity:        D = E H^3 / [12 (1 - nu^2)]
Flexural parameter:       beta = (rho_w g / 4D)^(1/4)
Deflection (clamped GL):  w(s, t) = h(t) * [1 - exp(-beta s)(cos beta s + sin beta s)]  for s >= 0
                          w(s, t) = 0 on grounded ice (s < 0)
Limit of flexure:         L_flex = pi / beta

Validated numerics (prototype_v1_three_date.py, H=400 m, E=0.88 GPa, nu=0.30):
    beta   = 8.362 x 10^-4 m^-1
    L_flex = 3757 m
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

G_STANDARD = 9.81  # m/s^2


def flexural_rigidity_D(  # noqa: N802  (D is the textbook symbol for flexural rigidity)
    young_modulus_pa: float,
    ice_thickness_m: float,
    poisson_ratio: float,
) -> float:
    """Return flexural rigidity D = E H^3 / [12 (1 - nu^2)]  [N m]."""
    if ice_thickness_m <= 0:
        msg = f"ice_thickness_m must be > 0, got {ice_thickness_m}"
        raise ValueError(msg)
    if young_modulus_pa <= 0:
        msg = f"young_modulus_pa must be > 0, got {young_modulus_pa}"
        raise ValueError(msg)
    if not -1.0 < poisson_ratio < 0.5:
        msg = f"poisson_ratio must be in (-1, 0.5), got {poisson_ratio}"
        raise ValueError(msg)
    return young_modulus_pa * ice_thickness_m**3 / (12.0 * (1.0 - poisson_ratio**2))


def flexural_parameter_beta(
    young_modulus_pa: float,
    ice_thickness_m: float,
    poisson_ratio: float,
    rho_water: float = 1028.0,
    gravity: float = G_STANDARD,
) -> float:
    """Return flexural parameter beta = (rho_w g / 4D)^(1/4)  [1/m]."""
    d_rigidity = flexural_rigidity_D(young_modulus_pa, ice_thickness_m, poisson_ratio)
    return float((rho_water * gravity / (4.0 * d_rigidity)) ** 0.25)


def limit_of_flexure(beta: float) -> float:
    """Return L_flex = pi / beta  [m]."""
    if beta <= 0:
        msg = f"beta must be > 0, got {beta}"
        raise ValueError(msg)
    return float(np.pi / beta)


def flexure_profile(
    s_m: ArrayLike,
    h_tide_m: float,
    beta: float,
) -> NDArray[np.float64]:
    """Elastic deflection w(s, t) of a clamped ice-shelf plate under uniform water load h_tide.

    Parameters
    ----------
    s_m : array-like of float
        Signed cross-grounding-line distance in metres. s >= 0 is the floating side
        (clamped at s = 0), s < 0 is grounded.
    h_tide_m : float
        Instantaneous ocean tide height (m). The floating ice rises and falls with it.
    beta : float
        Flexural parameter (1/m). From `flexural_parameter_beta`.

    Returns
    -------
    w : ndarray of float, same shape as s_m
        Vertical deflection of the ice surface (m). Zero on grounded ice.
    """
    s_arr = np.asarray(s_m, dtype=np.float64)
    s_pos = np.maximum(s_arr, 0.0)
    w = h_tide_m * (
        1.0 - np.exp(-beta * s_pos) * (np.cos(beta * s_pos) + np.sin(beta * s_pos))
    )
    w = np.where(s_arr < 0.0, 0.0, w)
    return w
