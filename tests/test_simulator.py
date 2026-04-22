"""End-to-end tests for Sensor + Site + Simulator + TripletReport against mock tides."""

from __future__ import annotations

import numpy as np
import pytest

from tidal_insar_sim import Sensor, Simulator, Site, TripletReport
from tidal_insar_sim.report import FRINGE_THRESHOLDS
from tidal_insar_sim.site import SITE_PRESETS
from tidal_insar_sim.tides.mock import MixedTide, mixed_m2_k1_tide, multi_constituent_tide


def _make_sim_thwaites_like() -> Simulator:
    """NISAR-L + a Thwaites-like MixedTide (A_M2=0.8, A_K1=0.5)."""
    return Simulator(
        sensor=Sensor.NISAR_L,
        site=Site.THWAITES,
        tide_fn=mixed_m2_k1_tide(),
    )


# ---------------------------------------------------------------------------
# Sensor / Site sanity
# ---------------------------------------------------------------------------


def test_sensor_nisar_l_parameters() -> None:
    s = Sensor.NISAR_L
    assert s.wavelength_m == pytest.approx(0.2360)
    assert s.repeat_days == 12.0
    assert s.incidence_deg == pytest.approx(39.0)
    assert s.fringe_los_cm == pytest.approx(11.8, rel=1e-3)


def test_sensor_presets_are_all_distinct_instances() -> None:
    presets = [
        Sensor.NISAR_L,
        Sensor.SENTINEL_1_SINGLE,
        Sensor.SENTINEL_1_DUAL,
        Sensor.ALOS_2,
        Sensor.ALOS_4,
        Sensor.COSMO_SKYMED,
        Sensor.TERRASAR_X,
        Sensor.RADARSAT_CONSTELLATION,
        Sensor.UMBRA_X,
    ]
    assert len({p.name for p in presets}) == len(presets)


def test_site_flexural_parameter_matches_physics_module() -> None:
    site = Site.THWAITES
    beta = site.flexural_parameter_beta()
    l_flex = site.limit_of_flexure_m()
    # THWAITES uses H=450; physical sanity: 500-5000 m limit of flexure is expected range.
    assert 500.0 < l_flex < 5000.0
    assert beta > 0.0


def test_site_from_coords_builds_named_site() -> None:
    s = Site.from_coords(lat=-77.0, lon=130.0, ice_thickness_m=800.0)
    assert s.name.startswith("site_")
    assert s.lat == -77.0
    assert s.lon == 130.0
    assert s.ice_thickness_m == 800.0


def test_no_greenland_presets_in_v0_1() -> None:
    """Greenland is deferred (brief §v0.1; user decision Apr 2026)."""
    for name, site in SITE_PRESETS.items():
        assert site.lat < 0, f"Site {name} has non-Antarctic lat {site.lat}"


# ---------------------------------------------------------------------------
# Simulator + report
# ---------------------------------------------------------------------------


def test_simulator_requires_tide_fn_in_c1() -> None:
    sim = Simulator(sensor=Sensor.NISAR_L, site=Site.THWAITES)
    with pytest.raises(RuntimeError, match="tide_fn"):
        sim.sweep_triplets()


def test_sweep_report_summary_keys() -> None:
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    summary = report.summary()
    expected_keys = {
        "site", "sensor", "n_triplets",
        "P_usable_ge_3fr", "P_robust_ge_5fr", "P_null_lt_0p5fr",
        "fringe_mean", "fringe_median", "fringe_max", "fringe_p05", "fringe_p95",
        "hDD_range_m", "hDD_std_m", "verdict",
    }
    assert expected_keys <= summary.keys()
    assert summary["sensor"] == "NISAR-L"
    assert summary["site"] == "Thwaites_GL"


def test_sweep_report_probabilities_sum_consistently() -> None:
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    # P_below(x) + P_at_least(x) must equal 1 for every threshold.
    for thr in (0.5, 1.0, 2.0, 3.0, 5.0):
        assert report.p_below(thr) + report.p_at_least(thr) == pytest.approx(1.0)


def test_sweep_report_class_fractions_partition_unity() -> None:
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    frac = report.class_fractions()
    assert sum(frac.values()) == pytest.approx(1.0)


def test_sweep_fringes_match_ddinsar_formula() -> None:
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    theta = np.deg2rad(sim.sensor.incidence_deg)
    expected = (
        np.abs(report.h_dd_m) * np.cos(theta) / (sim.sensor.wavelength_m / 2.0)
    )
    assert np.allclose(report.fringes, expected)


def test_sweep_best_worst_accessors() -> None:
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    best = report.best_triplet_h()
    worst = report.worst_triplet_h()
    assert report.fringes[np.argmax(report.fringes)] >= report.fringes[np.argmin(report.fringes)]
    assert 0.0 <= best <= 29.53 * 24.0 + 1.0
    assert 0.0 <= worst <= 29.53 * 24.0 + 1.0


def test_sweep_produces_physically_plausible_fringe_statistics() -> None:
    """For a Thwaites-like M2+K1 tide with NISAR-L, fringes should be O(1-5).

    Loose bounds — we're checking for sign/sanity, not exact numbers.
    """
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    s = report.summary()
    assert 0.0 < s["fringe_mean"] < 10.0
    assert 0.0 < s["fringe_median"] < 10.0
    assert s["fringe_max"] > s["fringe_mean"]
    assert abs(s["hDD_range_m"][0]) < 10.0
    assert abs(s["hDD_range_m"][1]) < 10.0
    assert s["verdict"] in {"excellent", "good", "marginal", "poor"}


def test_sweep_with_semidiurnal_tide_dominates_fringes() -> None:
    """Rutford-like (large M2, small K1) gives more fringes than Thwaites-like on average."""
    rutford_like = multi_constituent_tide(
        amplitudes_m={"M2": 1.40, "S2": 0.95, "K1": 0.30, "O1": 0.30},
        phases_deg={"M2": 340.0, "S2": 60.0, "K1": 170.0, "O1": 150.0},
    )
    thwaites_like = multi_constituent_tide(
        amplitudes_m={"M2": 0.20, "S2": 0.10, "K1": 0.55, "O1": 0.50},
        phases_deg={"M2": 80.0, "S2": 120.0, "K1": 220.0, "O1": 200.0},
    )
    sim_r = Simulator(sensor=Sensor.NISAR_L, site=Site.RUTFORD, tide_fn=rutford_like)
    sim_t = Simulator(sensor=Sensor.NISAR_L, site=Site.THWAITES, tide_fn=thwaites_like)
    rep_r = sim_r.sweep_triplets()
    rep_t = sim_t.sweep_triplets()
    assert rep_r.summary()["fringe_mean"] > rep_t.summary()["fringe_mean"]


# ---------------------------------------------------------------------------
# Confidence (±1 h jitter Monte Carlo)
# ---------------------------------------------------------------------------


def test_confidence_is_in_unit_interval() -> None:
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    c = sim.confidence(triplet_start_hours=report.best_triplet_h(), rng=np.random.default_rng(2026))
    assert 0.0 <= c <= 1.0


def test_confidence_zero_jitter_degenerates_to_single_triplet() -> None:
    """With `jitter_hours=0` all realisations are identical → confidence is 0 or 1."""
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    # BEST triplet has high fringe count; should clear 1-fringe floor -> confidence = 1.
    c_best = sim.confidence(
        triplet_start_hours=report.best_triplet_h(),
        jitter_hours=0.0,
        n_realisations=100,
    )
    assert c_best == pytest.approx(1.0)
    # WORST triplet (NULL) has < 1 fringe -> confidence = 0.
    c_worst = sim.confidence(
        triplet_start_hours=report.worst_triplet_h(),
        jitter_hours=0.0,
        n_realisations=100,
    )
    assert c_worst == pytest.approx(0.0)


def test_confidence_monotone_in_nominal_fringe_count() -> None:
    """Triplets with higher nominal fringe count are more robust under jitter."""
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    rng = np.random.default_rng(2026)
    c_best = sim.confidence(triplet_start_hours=report.best_triplet_h(), rng=rng)
    c_worst = sim.confidence(triplet_start_hours=report.worst_triplet_h(), rng=rng)
    assert c_best > c_worst


def test_fringe_thresholds_canonicalize_to_brief() -> None:
    """Brief §5.1 defines the named thresholds."""
    assert FRINGE_THRESHOLDS["null"] == 0.5
    assert FRINGE_THRESHOLDS["marginal"] == 3.0
    assert FRINGE_THRESHOLDS["good"] == 5.0


# ---------------------------------------------------------------------------
# Mock tide sanity
# ---------------------------------------------------------------------------


def test_mock_tide_shape_matches_input() -> None:
    tide = MixedTide()
    t = np.linspace(0, 100, 50)
    out = tide(t)
    assert out.shape == t.shape
    assert out.dtype == np.float64


def test_multi_constituent_rejects_unknown_constituent() -> None:
    with pytest.raises(KeyError):
        multi_constituent_tide(
            amplitudes_m={"M2": 0.5, "BOGUS": 0.1},
            phases_deg={"M2": 0.0, "BOGUS": 0.0},
        )


def test_triplet_report_dataclass() -> None:
    sim = _make_sim_thwaites_like()
    report = sim.sweep_triplets()
    assert isinstance(report, TripletReport)
    assert report.sensor_name == "NISAR-L"
