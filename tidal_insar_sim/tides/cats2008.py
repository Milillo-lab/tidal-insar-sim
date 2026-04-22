"""CATS2008 tide backend via pyTMD.

Wraps `pyTMD.compute.tide_elevations` so the Simulator can pull a tide time series
from the CATS2008 harmonic constants at a lat/lon. Also exposes per-constituent
amplitude extraction (needed by the validation tests).

The model files are expected at `{data_dir}/CATS2008/{grid_CATS2008, hf.CATS2008.out}`
in the original OTIS binary format. Install with `tidal-insar-sim setup-cats`.

Domain notes
------------
- CATS2008 covers the circum-Antarctic ocean, polar stereographic grid, 4 km spacing.
- Points at lat > ~-30 are outside the domain (`OutsideDomain`).
- CATS2008 masks grounded ice. Many grounding-line points fall on the 'land' side of
  the mask. Use `extrapolate_cutoff_km` (default 10 km) to pull from the nearest
  ocean cell; raise `OnLand` if no ocean cell within cutoff.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tidal_insar_sim.tides.errors import OnLand, OutsideDomain, TidalDataUnavailable

DEFAULT_DATA_DIR = Path.home() / ".tidal_insar_sim" / "tides"
CATS_SUBDIR = "CATS2008"
CATS_FILES = ("grid_CATS2008", "hf.CATS2008.out")
CATS_DOMAIN_LAT_MAX = -30.0  # CATS2008 ends around here (Antarctic Circle is -66.5)

DEFAULT_CONSTITUENTS = ("m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1")
DEFAULT_EXTRAPOLATE_CUTOFF_KM = 10.0

# pyTMD's time axis is seconds since this epoch (matches brief §3.4 EPOCH_2000).
EPOCH_2000 = (2000, 1, 1, 0, 0, 0)


@dataclass
class CATSBackend:
    """Lazy wrapper around a CATS2008 install. Call `predict_tide_m_hours()` to
    get an ndarray of ocean-tide heights at a given lat/lon over arbitrary times.

    Use `data_dir=` to override the default `~/.tidal_insar_sim/tides`.
    """

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    model: str = "CATS2008"
    extrapolate_cutoff_km: float = DEFAULT_EXTRAPOLATE_CUTOFF_KM

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self._check_install()

    # --- install / metadata ------------------------------------------------

    def _check_install(self) -> None:
        model_dir = self.data_dir / CATS_SUBDIR
        missing = [f for f in CATS_FILES if not (model_dir / f).exists()]
        if missing:
            msg = (
                f"CATS2008 data files missing in {model_dir}: {missing}. "
                f"Run `tidal-insar-sim setup-cats --path /path/to/CATS2008.zip` "
                f"to install, or set `data_dir=` to an existing CATS2008 install."
            )
            raise TidalDataUnavailable(msg)

    def describe(self) -> dict[str, str]:
        return {
            "model": self.model,
            "data_dir": str(self.data_dir),
            "version": "CATS2008 (Padman et al. 2002; Oct 2009 release)",
        }

    # --- per-point API -----------------------------------------------------

    def predict_tide_m(
        self,
        lat: float,
        lon: float,
        delta_time_seconds: ArrayLike,
    ) -> NDArray[np.float64]:
        """Ocean-tide elevation (m) at (lat, lon) for each `delta_time_seconds`
        since 2000-01-01 00:00:00 UTC. Raises OnLand / OutsideDomain on failure.
        """
        self._check_domain(lat)
        from pyTMD.compute import tide_elevations

        t = np.asarray(delta_time_seconds, dtype=np.float64).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da = tide_elevations(
                x=np.array([lon], dtype=np.float64),
                y=np.array([lat], dtype=np.float64),
                delta_time=t,
                directory=str(self.data_dir),
                model=self.model,
                epoch=EPOCH_2000,
                standard="UTC",
                type="time series",
                infer_minor=True,
                extrapolate=True,
                cutoff=self.extrapolate_cutoff_km,
                crs=4326,
                method="linear",
            )
        arr = np.asarray(da).astype(np.float64).ravel()
        if np.all(np.isnan(arr)):
            self._raise_on_land(lat, lon)
        if np.any(np.isnan(arr)):
            msg = (
                f"CATS2008 returned NaN for {np.sum(np.isnan(arr))}/{arr.size} "
                f"samples at lat={lat}, lon={lon}. Check the point is in the ocean domain."
            )
            raise RuntimeError(msg)
        return arr

    def extract_constants(
        self,
        lat: float,
        lon: float,
        constituents: tuple[str, ...] = DEFAULT_CONSTITUENTS,
    ) -> dict[str, tuple[float, float]]:
        """Return `{constituent: (amplitude_m, phase_deg)}` at (lat, lon).

        Amplitudes and phases follow the OTIS convention
        `h(t) = Re(H * exp(-i * omega * t))`, so the synthesized tide at a point is
        `Sum_i A_i * cos(omega_i * t - phi_i)` with `phi_i = arg(H_i)` in degrees.

        If the point falls on a masked (grounded-ice) cell, falls back to the
        nearest ocean cell within `extrapolate_cutoff_km`. Raises OnLand if no
        ocean cell is within range.
        """
        self._check_domain(lat)
        import pyproj
        from pyTMD.io import OTIS

        model_dir = self.data_dir / CATS_SUBDIR
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = OTIS.open_otis_dataset(
                model_file=str(model_dir / "hf.CATS2008.out"),
                grid_file=str(model_dir / "grid_CATS2008"),
                group="z",
            )
        wgs84 = pyproj.CRS("EPSG:4326")
        cats_crs = pyproj.CRS(
            "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=-70 +datum=WGS84 +units=km"
        )
        transformer = pyproj.Transformer.from_crs(wgs84, cats_crs, always_xy=True)
        x_km, y_km = transformer.transform(lon, lat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interp = ds.tmd.interp(
                np.array([x_km], dtype=np.float64),
                np.array([y_km], dtype=np.float64),
                method="linear",
            )

        # First try strict interpolation; fall back to nearest ocean cell if NaN.
        out: dict[str, tuple[float, float]] = {}
        any_valid = False
        xs = ds.x.values
        ys = ds.y.values
        for name in constituents:
            key = name.lower()
            if key not in interp.data_vars:
                continue
            z_val: complex | None = complex(interp[key].values.flatten()[0])
            assert z_val is not None  # satisfy mypy narrowing below
            if np.isnan(z_val.real) or np.isnan(z_val.imag):
                fallback, dist_km = self._nearest_ocean_cell(
                    ds[key].values, xs, ys, x_km, y_km,
                )
                if fallback is None or dist_km > self.extrapolate_cutoff_km:
                    out[name.upper()] = (float("nan"), float("nan"))
                    continue
                z_val = fallback
            any_valid = True
            amp = float(abs(z_val))
            phase = float(np.degrees(np.angle(z_val))) % 360.0
            out[name.upper()] = (amp, phase)
        if not any_valid:
            self._raise_on_land(lat, lon)
        return out

    @staticmethod
    def _nearest_ocean_cell(
        z_grid: NDArray[np.complex128],
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        x_km: float,
        y_km: float,
    ) -> tuple[complex | None, float]:
        """Nearest non-NaN cell to (x_km, y_km) in the CATS2008 native grid.

        Returns (value, distance_km). (None, inf) if the whole grid is NaN.
        """
        valid = ~(np.isnan(z_grid.real) | np.isnan(z_grid.imag))
        if not np.any(valid):
            return None, float("inf")
        yi, xi = np.nonzero(valid)
        # Bounding-box prefilter to avoid O(N) distance for a huge grid
        max_k = 20  # search radius in grid cells (4 km spacing -> ~80 km radius)
        ix0 = int(np.searchsorted(xs, x_km))
        iy0 = int(np.searchsorted(ys, y_km))
        ix_min = max(0, ix0 - max_k)
        ix_max = min(len(xs), ix0 + max_k)
        iy_min = max(0, iy0 - max_k)
        iy_max = min(len(ys), iy0 + max_k)
        mask = (xi >= ix_min) & (xi < ix_max) & (yi >= iy_min) & (yi < iy_max)
        if not np.any(mask):
            return None, float("inf")
        yi, xi = yi[mask], xi[mask]
        dx = xs[xi] - x_km
        dy = ys[yi] - y_km
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        return complex(z_grid[yi[j], xi[j]]), float(np.sqrt(d2[j]))

    # --- helpers -----------------------------------------------------------

    def _check_domain(self, lat: float) -> None:
        if lat > CATS_DOMAIN_LAT_MAX:
            msg = (
                f"lat={lat} is outside the CATS2008 domain (south of ~{CATS_DOMAIN_LAT_MAX} "
                f"deg). Use a global tide model for sub-polar sites."
            )
            raise OutsideDomain(msg)

    def _raise_on_land(self, lat: float, lon: float) -> None:
        msg = (
            f"(lat={lat}, lon={lon}) is masked in CATS2008 and no ocean cell was "
            f"reachable within {self.extrapolate_cutoff_km} km. Either shift the "
            f"point slightly toward open ocean, or increase `extrapolate_cutoff_km`."
        )
        raise OnLand(msg)


def make_cats_tide_fn(
    backend: CATSBackend,
    lat: float,
    lon: float,
) -> TideFn:
    """Close over a CATSBackend + site to produce a tide_fn(t_hours) -> h.

    The Simulator consumes `tide_fn(t_hours)` as its input API, where `t_hours`
    are hours from the 2000-01-01 UTC epoch.
    """

    def _tide(t_hours: ArrayLike) -> NDArray[np.float64]:
        t_s = np.asarray(t_hours, dtype=np.float64) * 3600.0
        return backend.predict_tide_m(lat=lat, lon=lon, delta_time_seconds=t_s)

    return _tide


# ---- type re-export to avoid circular import ----
from tidal_insar_sim.tides.mock import TideFn  # noqa: E402

__all__ = [
    "CATS_DOMAIN_LAT_MAX",
    "CATS_FILES",
    "CATS_SUBDIR",
    "DEFAULT_DATA_DIR",
    "EPOCH_2000",
    "CATSBackend",
    "make_cats_tide_fn",
]
