"""Integration tests for the full Simulator + CATS2008 path.

Verifies the auto-load wiring in `Simulator(sensor, site)` with no explicit
`tide_fn` — a Site preset should resolve to a CATSBackend-backed tide, produce
a real sweep, and yield a physically plausible report.

Skipped if CATS2008 is not installed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from tidal_insar_sim import Sensor, Simulator, Site
from tidal_insar_sim.tides import DEFAULT_DATA_DIR
from tidal_insar_sim.tides.cache import LAT_LON_PRECISION_DEG, TideCache
from tidal_insar_sim.tides.cats2008 import CATS_FILES, CATS_SUBDIR
from tidal_insar_sim.tides.download import (
    CATS2008_ZIP_MD5,
    manual_download_instructions,
)

_CATS_MODEL_DIR = DEFAULT_DATA_DIR / CATS_SUBDIR
_CATS_OK = all((_CATS_MODEL_DIR / f).exists() for f in CATS_FILES)

cats_only = pytest.mark.skipif(
    not _CATS_OK,
    reason=f"CATS2008 not installed at {_CATS_MODEL_DIR}",
)


# -----------------------------------------------------------------------------
# Simulator auto-loads CATS when no tide_fn supplied
# -----------------------------------------------------------------------------


@cats_only
def test_simulator_auto_loads_cats_at_thwaites() -> None:
    sim = Simulator(sensor=Sensor.L_BAND, site=Site.THWAITES)
    report = sim.sweep_triplets()
    s = report.summary()
    assert s["n_triplets"] > 0
    assert s["fringe_mean"] > 0.0
    assert s["verdict"] in {"excellent", "good", "marginal", "poor"}
    assert not np.isnan(s["fringe_mean"])
    assert not np.isnan(s["fringe_max"])
    assert sim.tide_source.startswith("CATS2008@")


@cats_only
def test_simulator_regime_switches_between_sites() -> None:
    """Rutford (semidiurnal, big amplitude) should yield more fringes than
    Thwaites (diurnal, ~1 m envelope) for NISAR's 12-day repeat."""
    rep_rutford = Simulator(Sensor.L_BAND, Site.RUTFORD).sweep_triplets()
    rep_thwaites = Simulator(Sensor.L_BAND, Site.THWAITES).sweep_triplets()
    assert (
        rep_rutford.summary()["fringe_mean"] > rep_thwaites.summary()["fringe_mean"]
    ), "Rutford should dominate Thwaites in fringe count under NISAR-L"


@cats_only
def test_explicit_tide_fn_short_circuits_cats() -> None:
    """If user supplies `tide_fn=`, Simulator must NOT call CATS."""
    from tidal_insar_sim.tides.mock import mixed_m2_k1_tide

    sim = Simulator(Sensor.L_BAND, Site.THWAITES, tide_fn=mixed_m2_k1_tide())
    sim.sweep_triplets()
    assert sim.tide_source == "user-supplied"


# -----------------------------------------------------------------------------
# TideCache
# -----------------------------------------------------------------------------


def test_tide_cache_round_trips(tmp_path: Path) -> None:
    cache = TideCache(db_path=tmp_path / "site_cache.db")
    constants = {"M2": (0.8, 90.0), "K1": (0.3, 45.0)}
    cache.put(lat=-75.0, lon=-106.0, model_version="CATS2008", constants=constants)
    out = cache.get(lat=-75.0, lon=-106.0, model_version="CATS2008")
    assert out is not None
    assert out["M2"] == pytest.approx((0.8, 90.0))
    assert out["K1"] == pytest.approx((0.3, 45.0))


def test_tide_cache_is_version_scoped(tmp_path: Path) -> None:
    cache = TideCache(db_path=tmp_path / "site_cache.db")
    cache.put(-75.0, -106.0, "CATS2008", {"M2": (0.8, 0.0)})
    assert cache.get(-75.0, -106.0, "CATS2008-v2023") is None


def test_tide_cache_rounds_to_0p01_deg(tmp_path: Path) -> None:
    """Cache keys are rounded to 0.01 deg (brief §3.3)."""
    cache = TideCache(db_path=tmp_path / "site_cache.db")
    cache.put(-75.00, -106.00, "CATS2008", {"M2": (0.8, 0.0)})
    # Same rounded bin
    assert cache.get(-75.0001, -106.0001, "CATS2008") is not None
    # Outside rounding bin
    assert cache.get(-75.1, -106.1, "CATS2008") is None
    # Precision constant matches brief
    assert LAT_LON_PRECISION_DEG == 0.01


def test_tide_cache_invalidate(tmp_path: Path) -> None:
    cache = TideCache(db_path=tmp_path / "site_cache.db")
    cache.put(-75.0, -106.0, "CATS2008", {"M2": (0.8, 0.0)})
    cache.put(-75.0, -106.0, "CATS2008-v2023", {"M2": (0.9, 0.0)})
    n_dropped = cache.invalidate("CATS2008")
    assert n_dropped == 1
    assert cache.get(-75.0, -106.0, "CATS2008") is None
    assert cache.get(-75.0, -106.0, "CATS2008-v2023") is not None


def test_tide_cache_creates_table(tmp_path: Path) -> None:
    db_path = tmp_path / "site_cache.db"
    TideCache(db_path=db_path)
    with sqlite3.connect(db_path) as con:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
    assert "site_cache" in tables


# -----------------------------------------------------------------------------
# setup-cats helper
# -----------------------------------------------------------------------------


def test_setup_cats_md5_is_published_constant() -> None:
    """The hardcoded MD5 must match the one published at USAP-DC (DOI 10.15784/601235)."""
    # This is a self-documenting sanity check: if the constant changes, the
    # install helper is silently broken for anyone following the README.
    assert CATS2008_ZIP_MD5 == "008a30cd08142cb6acc7f7687e22c4a3"


def test_setup_cats_download_instructions_include_usapdc_url() -> None:
    text = manual_download_instructions()
    assert "usap-dc.org" in text
    assert "CATS2008.zip" in text
    assert CATS2008_ZIP_MD5 in text
