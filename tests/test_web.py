"""Smoke tests for the Streamlit app (M9).

Two layers:
1. Pure helpers (`web.state`, `web.components.plots`) — regular unit tests.
2. Streamlit pages — via `streamlit.testing.v1.AppTest`. These start a headless
   Streamlit session and run the page script in-process; we assert no uncaught
   exceptions and that expected widgets render.

AppTest is intentionally lenient: it catches expected Streamlit calls (like
`st.error(...)` when CATS2008 is missing) without tripping the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tidal_insar_sim import Sensor, Simulator, Site
from tidal_insar_sim.tides.mock import mixed_m2_k1_tide
from tidal_insar_sim.web.components.plots import (
    plot_fringe_map,
    plot_sweep_curve,
    plot_tide_time_series,
    plot_triplet_histogram,
)
from tidal_insar_sim.web.state import (
    DEFAULTS,
    K_REPORT,
    K_SENSOR_NAME,
    K_SITE_LAT,
    K_SITE_LON,
    K_SITE_PRESET,
    Snapshot,
    init_state,
    invalidate_computed,
    resolve_sensor,
    resolve_site,
)

_REPO = Path(__file__).resolve().parent.parent
_APP_PATH = _REPO / "tidal_insar_sim" / "web" / "app.py"
_PAGES_DIR = _REPO / "tidal_insar_sim" / "web" / "pages"


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------


def test_init_state_populates_defaults() -> None:
    store: dict[str, object] = {}
    init_state(store)
    for key, default in DEFAULTS.items():
        assert store[key] == default


def test_init_state_is_idempotent() -> None:
    store: dict[str, object] = {K_SENSOR_NAME: "ALOS-4"}
    init_state(store)
    assert store[K_SENSOR_NAME] == "ALOS-4"  # existing values preserved


def test_resolve_site_preset() -> None:
    store: dict[str, object] = {}
    init_state(store)
    site = resolve_site(store)
    assert site.name == "Thwaites_GL"


def test_resolve_site_custom() -> None:
    store: dict[str, object] = {}
    init_state(store)
    store[K_SITE_PRESET] = "Custom"
    store[K_SITE_LAT] = -77.0
    store[K_SITE_LON] = 150.0
    site = resolve_site(store)
    assert site.lat == -77.0
    assert site.lon == 150.0


def test_resolve_sensor_preset() -> None:
    store: dict[str, object] = {}
    init_state(store)
    store[K_SENSOR_NAME] = "L-BAND"
    assert resolve_sensor(store).band == "L"


def test_invalidate_computed_clears_cached_results() -> None:
    store: dict[str, object] = {}
    init_state(store)
    store[K_REPORT] = "some cached report"
    invalidate_computed(store)
    assert store[K_REPORT] is None


def test_snapshot_from_store() -> None:
    store: dict[str, object] = {}
    init_state(store)
    snap = Snapshot.from_store(store)
    assert snap.site_name == "Thwaites_GL"
    assert snap.sensor_name == "L-BAND"
    assert snap.site_lat == -75.00


# ---------------------------------------------------------------------------
# plot builders (no Streamlit needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    sim = Simulator(
        sensor=Sensor.L_BAND, site=Site.RUTFORD,
        tide_fn=mixed_m2_k1_tide(amp_m2_m=1.4, amp_k1_m=0.3),
    )
    return sim.sweep_triplets()


def test_plot_sweep_curve_returns_figure(report) -> None:
    fig = plot_sweep_curve(report)
    assert fig is not None
    assert len(fig.axes) == 1


def test_plot_triplet_histogram(report) -> None:
    fig = plot_triplet_histogram(report)
    assert fig is not None


def test_plot_tide_time_series(report) -> None:
    fig = plot_tide_time_series(
        report.delta_t_hours, report.sweep.h1_m, report.h_dd_m,
        site_name="Rutford",
    )
    assert fig is not None
    assert len(fig.axes) == 2


def test_plot_fringe_map() -> None:
    sim = Simulator(
        sensor=Sensor.L_BAND, site=Site.RUTFORD,
        tide_fn=mixed_m2_k1_tide(amp_m2_m=1.4, amp_k1_m=0.3),
    )
    # Small map, fast
    fmap = sim.synthesize_ddinsar(
        triplet_start_hours=128.0, size_m=(3000.0, 2000.0), pixel_m=50.0,
    )
    fig = plot_fringe_map(fmap)
    assert fig is not None


# ---------------------------------------------------------------------------
# AppTest smoke tests
# ---------------------------------------------------------------------------


def _run_page(path: Path, timeout: float = 30.0):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(path), default_timeout=timeout)
    at.run()
    return at


def test_app_entry_does_not_error() -> None:
    at = _run_page(_APP_PATH)
    assert not at.exception, f"app.py raised: {at.exception}"


def test_setup_page_renders() -> None:
    """The merged Setup page loads with Antarctic map + constellation form."""
    at = _run_page(_PAGES_DIR / "1_Setup.py", timeout=60.0)
    assert not at.exception, f"Setup raised: {at.exception}"


def test_batch_page_renders_without_running() -> None:
    at = _run_page(_PAGES_DIR / "3_Batch.py")
    assert not at.exception
    assert len(at.multiselect) >= 2


def test_export_page_noop_without_artifacts() -> None:
    at = _run_page(_PAGES_DIR / "4_Export.py")
    assert not at.exception


def test_single_site_page_renders() -> None:
    at = _run_page(_PAGES_DIR / "2_Single_Site.py", timeout=60.0)
    assert not at.exception
