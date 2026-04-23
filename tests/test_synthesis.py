"""Tests for the 2D DDInSAR fringe-map synthesizer (M4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tidal_insar_sim import Sensor, Simulator, Site
from tidal_insar_sim.synthesis import (
    EPSG_ANTARCTIC,
    FringeMap,
    GLGeometry,
    synthesize_fringe_map,
)
from tidal_insar_sim.tides.mock import mixed_m2_k1_tide

# ---------------------------------------------------------------------------
# Grid / geometry
# ---------------------------------------------------------------------------


def _make_fringe_map(**kwargs: float) -> FringeMap:
    defaults: dict[str, object] = {
        "sensor": Sensor.NISAR_L,
        "site": Site.THWAITES,
        "h1_m": 0.30,
        "h2_m": -0.10,
        "h3_m": 0.20,
        "size_m": (3000.0, 2000.0),
        "pixel_m": 50.0,
        "noise_seed": 1,
    }
    defaults.update(kwargs)
    return synthesize_fringe_map(**defaults)  # type: ignore[arg-type]


def test_fringe_map_has_expected_shape() -> None:
    fmap = _make_fringe_map(size_m=(3000.0, 2000.0), pixel_m=50.0)
    assert fmap.shape == (40, 60)  # (height, width) = (2000/50, 3000/50)


def test_fringe_map_complex_field_matches_phase_and_coherence() -> None:
    fmap = _make_fringe_map()
    assert np.allclose(np.abs(fmap.array), fmap.coherence, atol=1e-5)
    assert np.allclose(
        np.angle(fmap.array.astype(np.complex128)), fmap.wrapped_phase, atol=1e-5
    )


def test_fringe_map_wrapped_phase_in_range() -> None:
    fmap = _make_fringe_map()
    assert fmap.wrapped_phase.min() >= -np.pi - 1e-6
    assert fmap.wrapped_phase.max() <= np.pi + 1e-6


def test_two_zone_coherence_separates_grounded_and_shelf() -> None:
    """s >= 0 cells should carry gamma_shelf, s < 0 cells gamma_grounded."""
    gamma_g, gamma_s = 0.88, 0.45
    fmap = _make_fringe_map(gamma_grounded=gamma_g, gamma_shelf=gamma_s)
    vals = set(np.round(np.unique(fmap.coherence), 5))
    assert round(gamma_g, 5) in vals
    assert round(gamma_s, 5) in vals


def test_flat_tide_gives_zero_fringes() -> None:
    """h1 = h2 = h3 => h_DD = 0 => the 'ideal' DD signal is identically zero."""
    fmap = _make_fringe_map(h1_m=0.5, h2_m=0.5, h3_m=0.5, noise_seed=None)
    # We cannot use noise_seed=None to get deterministic test; use noiseless ideal
    # by checking the h_DD scalar and the flexure field:
    assert fmap.h_dd_m == 0.0
    assert np.allclose(fmap.h_dd_field_m, 0.0)
    assert fmap.n_fringes_peak == 0.0


def test_peak_fringe_count_scales_with_h_dd() -> None:
    """Double h_DD -> double the peak fringe count (up to noise)."""
    fmap_small = _make_fringe_map(
        h1_m=0.1, h2_m=0.0, h3_m=0.1, noise_seed=1,
    )
    fmap_large = _make_fringe_map(
        h1_m=0.2, h2_m=0.0, h3_m=0.2, noise_seed=1,
    )
    # Use flexure-field max (noiseless) for the ratio check.
    peak_small = float(np.nanmax(np.abs(fmap_small.h_dd_field_m)))
    peak_large = float(np.nanmax(np.abs(fmap_large.h_dd_field_m)))
    assert peak_large == pytest.approx(2.0 * peak_small, rel=1e-6)


def test_affine_transform_is_origin_aware() -> None:
    """Grid origin corresponds to the site's projected (x, y) (EPSG:3031)."""
    fmap = _make_fringe_map(size_m=(3000.0, 2000.0), pixel_m=50.0)
    pix_x, _, origin_x, _, pix_y, origin_y = fmap.transform
    assert pix_x == 50.0
    assert pix_y == -50.0
    # Origin should be upper-left; site projected coord is at grid centre.
    h, w = fmap.shape
    site_x = origin_x + (w / 2.0) * pix_x
    site_y = origin_y + (h / 2.0) * pix_y
    # Round-trip Site.THWAITES via pyproj to compare
    import pyproj
    tr = pyproj.Transformer.from_crs(4326, EPSG_ANTARCTIC, always_xy=True)
    x_ref, y_ref = tr.transform(Site.THWAITES.lon, Site.THWAITES.lat)
    assert site_x == pytest.approx(x_ref, rel=1e-6)
    assert site_y == pytest.approx(y_ref, rel=1e-6)


def test_antarctic_site_gets_epsg_3031() -> None:
    fmap = _make_fringe_map()
    assert fmap.crs_epsg == EPSG_ANTARCTIC


# ---------------------------------------------------------------------------
# GL geometry
# ---------------------------------------------------------------------------


def test_default_sinuous_gl_signed_distance_sign() -> None:
    gl = GLGeometry.default_sinuous(nx=400, pixel_m=15.0)
    x = np.array([[0.0]])
    # y(0) = 250*sin(0) + 0 = 0 -> s(0, 1000) = 1000 (positive, shelf)
    s_shelf = gl.signed_distance_m(x, np.array([[1000.0]]))
    s_grounded = gl.signed_distance_m(x, np.array([[-1000.0]]))
    assert s_shelf[0, 0] > 0
    assert s_grounded[0, 0] < 0


# ---------------------------------------------------------------------------
# GeoTIFF export
# ---------------------------------------------------------------------------


def test_fringe_map_geotiff_round_trip(tmp_path: Path) -> None:
    import rasterio

    fmap = _make_fringe_map()
    out = tmp_path / "test.tif"
    fmap.to_geotiff(out)
    with rasterio.open(out) as ds:
        assert ds.count == 3
        assert ds.shape == fmap.shape
        assert ds.crs.to_epsg() == EPSG_ANTARCTIC
        assert ds.descriptions == (
            "wrapped_phase_rad",
            "coherence",
            "h_DD_m",
        )
        phase_read = ds.read(1)
        assert np.allclose(phase_read, fmap.wrapped_phase.astype(np.float32), atol=1e-5)
        # Tag metadata preserved
        tags = ds.tags()
        assert "h_DD_m" in tags
        assert "n_fringes_peak" in tags


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------


def test_simulator_synthesize_ddinsar_uses_mock_tide() -> None:
    sim = Simulator(
        sensor=Sensor.NISAR_L,
        site=Site.THWAITES,
        tide_fn=mixed_m2_k1_tide(),
    )
    fmap = sim.synthesize_ddinsar(
        triplet_start_hours=128.0,  # known "PEAK" scenario from prototype_monte_carlo
        size_m=(3000.0, 2000.0),
        pixel_m=50.0,
    )
    assert fmap.shape == (40, 60)
    assert fmap.site_name == "Thwaites_GL"
    assert fmap.sensor_name == "NISAR-L"
    assert fmap.h_dd_m != 0.0  # peak scenario must not be null
