"""Tide backend error types. `Fail loudly` is a project non-negotiable (brief §7.4)."""

from __future__ import annotations


class TidalDataUnavailable(RuntimeError):  # noqa: N818  (name required by brief §7.4)
    """Raised when the tide model files are missing or pyTMD cannot read them.

    Remediation: install CATS2008 via `tidal-insar-sim setup-cats --path ...`
    and verify files at `~/.tidal_insar_sim/tides/CATS2008/`.
    """


class OutsideDomain(ValueError):  # noqa: N818  (name required by brief §7.4)
    """Raised when (lat, lon) is outside the CATS2008 model domain (> ~30 deg S)."""


class OnLand(RuntimeError):  # noqa: N818  (name required by brief §7.4)
    """Raised when the nearest CATS2008 ocean cell to (lat, lon) is too far.

    CATS2008 masks grounded ice as 'land' — many grounding-line points fall on the
    wrong side of the mask by a grid cell or two. Either (a) shift the point slightly
    toward the open ocean, or (b) pass a larger `extrapolate_cutoff_km` to the
    backend to pull from the nearest ocean cell.
    """
