"""Double-difference InSAR geometry at a grounding line.

Geometry
--------
    IFG_A = phi(t2) - phi(t1)
    IFG_B = phi(t3) - phi(t2)
    DD    = IFG_A - IFG_B = phi(t1) - 2 phi(t2) + phi(t3)

For a tidal signal h(t):
    h_DD(x) = w(x, t1) - 2 w(x, t2) + w(x, t3)             # vertical DD disp [m]
    phi_DD(x) = -(4 pi / lambda) cos(theta) h_DD(x)        # unwrapped DD phase [rad]
    n_fringes_peak = |h_DD_peak| cos(theta) / (lambda / 2) # fringe count along LOS

References
----------
Rignot, E., Mouginot, J., Scheuchl, B. (2011). GRL 38.
Milillo, P., Rignot, E., et al. (2017, 2019).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def double_difference_h(h1: float, h2: float, h3: float) -> float:
    """Return h_DD = h1 - 2 h2 + h3 (tidal residual of a 3-date triplet)."""
    return h1 - 2.0 * h2 + h3


def dd_phase(
    h_dd_field_m: ArrayLike,
    wavelength_m: float,
    incidence_deg: float,
) -> NDArray[np.float64]:
    """Unwrapped DD phase from a vertical DD displacement field.

    phi_DD = -(4 pi / lambda) * cos(theta) * h_DD

    The sign convention follows InSAR practice (positive LOS away from sensor
    yields negative phase). Flip upstream if your convention differs.
    """
    theta = np.deg2rad(incidence_deg)
    field = np.asarray(h_dd_field_m, dtype=np.float64)
    out: NDArray[np.float64] = -4.0 * np.pi / wavelength_m * np.cos(theta) * field
    return out


def dd_fringes_peak(
    h_dd_peak_m: float,
    wavelength_m: float,
    incidence_deg: float,
) -> float:
    """|fringes| at the flexure peak, as LOS count.

    One LOS fringe = lambda / 2. The vertical-to-LOS projection is cos(theta).
    """
    theta = np.deg2rad(incidence_deg)
    return float(abs(h_dd_peak_m) * np.cos(theta) / (wavelength_m / 2.0))
