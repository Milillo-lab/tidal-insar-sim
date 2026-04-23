"""Synthetic DDInSAR fringe-map synthesis on a 2D grid over a grounding zone.

Given three tide heights (h1, h2, h3) and a site's flexure parameter, produce a
wrapped-phase fringe field on a local projected grid centred on the GL. A
two-zone coherence model adds speckle (grounded gamma vs shelf gamma).

Default coordinate systems:
- Antarctic sites (lat < 0): EPSG:3031 polar stereographic (South)
- Arctic sites (lat > 0):    EPSG:3413 polar stereographic (North) — reserved for v0.2

The grid is centred on the site's (lat, lon), dimensions in metres.

Usage
-----
    fmap = Simulator(...).synthesize_ddinsar(triplet_start_hours=...)
    fmap.to_geotiff("strong.tif")
    fmap.to_png("strong.png", cmap="hsv")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tidal_insar_sim.sensor import Sensor
    from tidal_insar_sim.site import Site

GLCurveFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# Antarctic polar stereographic (MEaSUREs standard). Arctic reserved for v0.2.
EPSG_ANTARCTIC = 3031
EPSG_ARCTIC = 3413


@dataclass
class FringeMap:
    """A 2D synthetic DDInSAR fringe map centred on a grounding-zone site.

    Attributes
    ----------
    array : complex64[H, W]
        Complex DD signal. abs() = coherence magnitude; angle() = wrapped phase.
    wrapped_phase : float64[H, W]
        np.angle(array) — wrapped DD phase in radians, in [-pi, pi].
    coherence : float64[H, W]
        np.abs(array) — magnitude, in [0, 1].
    transform : tuple[float, float, float, float, float, float]
        GDAL/rasterio affine coefficients (a, b, c, d, e, f) for the grid.
    crs_epsg : int
        EPSG code of the grid projection (3031 Antarctic).
    """

    array: NDArray[np.complex64]
    wrapped_phase: NDArray[np.float64]
    coherence: NDArray[np.float64]
    h_dd_field_m: NDArray[np.float64]
    transform: tuple[float, float, float, float, float, float]
    crs_epsg: int
    site_name: str
    sensor_name: str
    h_dd_m: float
    n_fringes_peak: float

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.wrapped_phase.shape
        return (h, w)

    def to_geotiff(self, path: str | Path) -> None:
        """Write a 3-band GeoTIFF: wrapped phase (rad), coherence, h_DD (m)."""
        import rasterio

        path = Path(path)
        h, w = self.shape
        a, b, c, d, e, f = self.transform
        transform = rasterio.transform.Affine(a, b, c, d, e, f)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 3,
            "width": w,
            "height": h,
            "crs": f"EPSG:{self.crs_epsg}",
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(self.wrapped_phase.astype(np.float32), 1)
            dst.write(self.coherence.astype(np.float32), 2)
            dst.write(self.h_dd_field_m.astype(np.float32), 3)
            dst.set_band_description(1, "wrapped_phase_rad")
            dst.set_band_description(2, "coherence")
            dst.set_band_description(3, "h_DD_m")
            dst.update_tags(
                site=self.site_name,
                sensor=self.sensor_name,
                h_DD_m=f"{self.h_dd_m:.4f}",
                n_fringes_peak=f"{self.n_fringes_peak:.3f}",
            )

    def to_png(self, path: str | Path, cmap: str = "hsv") -> None:
        """Write a single-band PNG of the wrapped phase with the chosen colormap."""
        import matplotlib.cm as mcm
        import matplotlib.image as mpimg

        phase_norm = (self.wrapped_phase + np.pi) / (2.0 * np.pi)  # [0, 1]
        rgba = mcm.get_cmap(cmap)(phase_norm)
        mpimg.imsave(path, rgba)


def synthesize_fringe_map(
    *,
    sensor: Sensor,
    site: Site,
    h1_m: float,
    h2_m: float,
    h3_m: float,
    size_m: tuple[float, float] = (15_000.0, 10_000.0),
    pixel_m: float = 15.0,
    gl_geometry: GLGeometry | None = None,
    gamma_grounded: float = 0.88,
    gamma_shelf: float = 0.60,
    multi_look: int = 8,
    noise_seed: int | None = 1,
) -> FringeMap:
    """Generate a 2D synthetic DDInSAR fringe map for a (sensor, site, triplet).

    Parameters
    ----------
    sensor, site : Sensor, Site
    h1_m, h2_m, h3_m : three tide heights (m), typically evaluated at the triplet
        acquisition times by the Simulator.
    size_m : (width_m, height_m) of the synthesised scene (default 15 km x 10 km).
    pixel_m : ground pixel spacing (default 15 m, matches brief §4.3).
    gl_geometry : optional `GLGeometry` describing the grounding line polyline.
        If None, a gently sinuous reference GL (matches prototype_v1) is used.
    gamma_grounded, gamma_shelf : two-zone coherence (brief §4, user decision).
    multi_look : integer, for the noise amplitude model.
    noise_seed : seed for reproducible speckle. Set to None for fresh RNG.
    """
    from tidal_insar_sim.physics.ddinsar import dd_phase, double_difference_h
    from tidal_insar_sim.physics.flexure import flexure_profile

    beta = site.flexural_parameter_beta()
    width_m, height_m = size_m
    nx = round(width_m / pixel_m)
    ny = round(height_m / pixel_m)

    # Local scene axes centred at the site (x = range, y = azimuth).
    x = (np.arange(nx) - nx / 2.0) * pixel_m
    y = (np.arange(ny) - ny / 2.0) * pixel_m
    x_grid, y_grid = np.meshgrid(x, y)

    # Signed cross-GL distance field s(x, y) in metres.
    if gl_geometry is None:
        gl_geometry = GLGeometry.default_sinuous(nx=nx, pixel_m=pixel_m)
    s_field = gl_geometry.signed_distance_m(x_grid, y_grid)

    # Per-epoch flexure, DD = f(h1) - 2 f(h2) + f(h3).
    w1 = flexure_profile(s_field, h1_m, beta=beta)
    w2 = flexure_profile(s_field, h2_m, beta=beta)
    w3 = flexure_profile(s_field, h3_m, beta=beta)
    w_dd = w1 - 2.0 * w2 + w3  # vertical DD disp (m)
    h_dd_scalar = double_difference_h(h1_m, h2_m, h3_m)

    # Unwrapped DD phase.
    phase_unwrapped = dd_phase(
        h_dd_field_m=w_dd,
        wavelength_m=sensor.wavelength_m,
        incidence_deg=sensor.incidence_deg,
    )

    # Two-zone coherence: shelf where s >= 0, grounded where s < 0.
    coherence = np.where(s_field >= 0.0, gamma_shelf, gamma_grounded).astype(np.float64)

    # Speckle: sigma_phi ~ sqrt((1 - gamma^2) / (2 gamma^2 NL)) for multi-look NL.
    rng = np.random.default_rng() if noise_seed is None else np.random.default_rng(noise_seed)
    sigma = np.sqrt(
        np.clip(1.0 - coherence**2, 1e-9, None)
        / (2.0 * np.maximum(coherence, 1e-3) ** 2 * float(multi_look))
    )
    phase_noisy = phase_unwrapped + rng.normal(0.0, sigma)
    wrapped = np.angle(np.exp(1j * phase_noisy))

    # Complex signal: magnitude = coherence, phase = wrapped
    complex_field = (coherence * np.exp(1j * wrapped)).astype(np.complex64)

    # Affine transform (projected metres, EPSG:3031/3413).
    crs_epsg = EPSG_ARCTIC if site.lat > 0 else EPSG_ANTARCTIC
    site_xy = _project_lat_lon_m(site.lat, site.lon, crs_epsg)
    # GDAL affine: origin at upper-left, pixel_width + rotation + origin_x, etc.
    origin_x = site_xy[0] - (nx / 2.0) * pixel_m
    origin_y = site_xy[1] + (ny / 2.0) * pixel_m  # row 0 at top (positive y)
    transform = (pixel_m, 0.0, origin_x, 0.0, -pixel_m, origin_y)

    from tidal_insar_sim.physics.ddinsar import dd_fringes_peak

    n_peak = dd_fringes_peak(
        h_dd_peak_m=float(np.nanmax(np.abs(w_dd))),
        wavelength_m=sensor.wavelength_m,
        incidence_deg=sensor.incidence_deg,
    )

    return FringeMap(
        array=complex_field,
        wrapped_phase=wrapped,
        coherence=coherence,
        h_dd_field_m=w_dd,
        transform=transform,
        crs_epsg=crs_epsg,
        site_name=site.name,
        sensor_name=sensor.name,
        h_dd_m=float(h_dd_scalar),
        n_fringes_peak=float(n_peak),
    )


def _project_lat_lon_m(lat: float, lon: float, epsg: int) -> tuple[float, float]:
    import pyproj

    transformer = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return (float(x), float(y))


@dataclass
class GLGeometry:
    """A 1-D grounding-line curve, returning signed distance from any (x, y).

    s > 0 is the floating side (shelf), s < 0 is grounded. The default is a
    gentle sinusoid matching `reference/prototype_v1_three_date.py`, which
    serves as a physically reasonable placeholder until MEaSUREs GL v2 auto-
    clip lands (roadmap: `tidal-insar-sim setup-measures`).
    """

    curve_fn: GLCurveFn

    @classmethod
    def default_sinuous(cls, nx: int, pixel_m: float) -> GLGeometry:
        """GL = y(x) = 250 * sin(2*pi*x/7000) + 0.03*x, s = y_grid - y_GL(x)."""
        amp = 250.0
        wavelength = 7000.0
        tilt = 0.03

        def _curve(x: NDArray[np.float64]) -> NDArray[np.float64]:
            out: NDArray[np.float64] = amp * np.sin(2.0 * np.pi * x / wavelength) + tilt * x
            return out

        return cls(curve_fn=_curve)

    def signed_distance_m(
        self,
        x_grid: NDArray[np.float64],
        y_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        y_gl = self.curve_fn(x_grid)
        return y_grid - y_gl


