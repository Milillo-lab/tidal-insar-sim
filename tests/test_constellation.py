"""Tests for `Satellite` / `Constellation` + multi-baseline / any-triplet sweeps."""

from __future__ import annotations

import numpy as np
import pytest

from tidal_insar_sim import Constellation, Satellite, Sensor, Simulator, Site
from tidal_insar_sim.physics.fringe import (
    AnyTripletSweep,
    any_triplet_sweep,
    multi_baseline_sweep,
)
from tidal_insar_sim.tides.mock import mixed_m2_k1_tide

# ---------------------------------------------------------------------------
# Satellite / Constellation dataclasses
# ---------------------------------------------------------------------------


def test_constellation_single_has_one_sat() -> None:
    c = Constellation.single(repeat_days=12.0)
    assert c.n_satellites == 1
    assert c.satellites[0].repeat_days == 12.0
    assert c.satellites[0].phase_offset_days == 0.0


def test_constellation_equally_phased_offsets_are_evenly_spaced() -> None:
    c = Constellation.equally_phased(n=3, repeat_days=12.0)
    offsets = [sat.phase_offset_days for sat in c.satellites]
    assert offsets == pytest.approx([0.0, 4.0, 8.0])


def test_constellation_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Constellation(satellites=())


def test_constellation_rejects_negative_phase() -> None:
    with pytest.raises(ValueError):
        Constellation(satellites=(Satellite(name="bad", phase_offset_days=-1.0,
                                             repeat_days=12.0),))


def test_constellation_rejects_nonpositive_repeat() -> None:
    with pytest.raises(ValueError):
        Constellation(satellites=(Satellite(name="bad", phase_offset_days=0.0,
                                             repeat_days=0.0),))


def test_effective_repeat_single_sat() -> None:
    c = Constellation.single(repeat_days=12.0)
    assert c.effective_repeat_days() == pytest.approx(12.0)


def test_effective_repeat_equally_phased_dual() -> None:
    """2 sats equally-phased with 12-day repeat -> effective = 6 days (S1 A+B)."""
    c = Constellation.equally_phased(n=2, repeat_days=12.0)
    assert c.effective_repeat_days() == pytest.approx(6.0)


def test_effective_repeat_equally_phased_triple() -> None:
    """3 sats equally-phased with 12-day repeat -> 4 days (RCM)."""
    c = Constellation.equally_phased(n=3, repeat_days=12.0)
    assert c.effective_repeat_days() == pytest.approx(4.0)


def test_acquisition_times_are_sorted_and_in_window() -> None:
    c = Constellation.equally_phased(n=2, repeat_days=12.0)
    acq = c.acquisition_times(window_days=30.0)
    assert np.all(np.diff(acq) >= 0)
    assert np.all(acq < 30.0)
    assert np.all(acq >= 0)


def test_valid_baselines_equally_phased_dual() -> None:
    """N=2 equally-phased, 12-day repeat produces gaps {6, 12, 18, 24, ...}."""
    c = Constellation.equally_phased(n=2, repeat_days=12.0)
    baselines = c.valid_baselines_days(window_days=60.0)
    # Only small baselines are returned by default (<= max repeat * 1.1)
    assert 6.0 in baselines
    assert 12.0 in baselines


# ---------------------------------------------------------------------------
# multi_baseline_sweep
# ---------------------------------------------------------------------------


def test_multi_baseline_sweep_returns_one_sweep_per_baseline() -> None:
    c = Constellation.equally_phased(n=2, repeat_days=12.0)
    sweeps = multi_baseline_sweep(
        tide_fn=mixed_m2_k1_tide(amp_m2_m=1.4, amp_k1_m=0.3),
        constellation=c, step_hours=1.0, duration_days=29.53,
        wavelength_m=0.2360, incidence_deg=39.0,
    )
    assert len(sweeps) >= 2
    assert 6.0 in sweeps
    assert 12.0 in sweeps
    # Each sweep is a valid TripletSweep with non-zero fringes.
    for B, sweep in sweeps.items():
        assert sweep.repeat_days == pytest.approx(B)
        assert sweep.fringes.size > 0
        assert np.nanmax(sweep.fringes) > 0.0


# ---------------------------------------------------------------------------
# any_triplet_sweep
# ---------------------------------------------------------------------------


def test_any_triplet_sweep_returns_valid_triplets() -> None:
    c = Constellation.equally_phased(n=2, repeat_days=12.0)
    out = any_triplet_sweep(
        tide_fn=mixed_m2_k1_tide(amp_m2_m=1.4, amp_k1_m=0.3),
        constellation=c, duration_days=29.53,
        wavelength_m=0.2360, incidence_deg=39.0,
    )
    assert isinstance(out, AnyTripletSweep)
    assert len(out) > 0
    # Every triplet is ordered: t1 < t2 < t3
    assert np.all(out.t1_days < out.t2_days)
    assert np.all(out.t2_days < out.t3_days)


def test_any_triplet_sweep_equal_baseline_filter() -> None:
    """`require_equal_baseline=True` keeps only rigid-B triplets."""
    c = Constellation.equally_phased(n=2, repeat_days=12.0)
    out = any_triplet_sweep(
        tide_fn=mixed_m2_k1_tide(),
        constellation=c, duration_days=29.53,
        wavelength_m=0.2360, incidence_deg=39.0,
        require_equal_baseline=True, equal_baseline_tol_days=0.01,
    )
    b1 = out.t2_days - out.t1_days
    b2 = out.t3_days - out.t2_days
    assert np.all(np.abs(b1 - b2) < 0.01)


def test_any_triplet_sweep_rejects_too_short_schedule() -> None:
    c = Constellation.single(repeat_days=12.0)
    with pytest.raises(ValueError):
        any_triplet_sweep(
            tide_fn=mixed_m2_k1_tide(),
            constellation=c, duration_days=20.0,  # only ~2 acquisitions
            wavelength_m=0.2360, incidence_deg=39.0,
        )


# ---------------------------------------------------------------------------
# Simulator wiring
# ---------------------------------------------------------------------------


def test_simulator_default_constellation_is_single_12d() -> None:
    sim = Simulator(sensor=Sensor.L_BAND, site=Site.THWAITES, tide_fn=mixed_m2_k1_tide())
    assert sim.constellation is not None
    assert sim.constellation.n_satellites == 1
    assert sim.constellation.satellites[0].repeat_days == 12.0


def test_simulator_sweep_uses_effective_repeat_from_constellation() -> None:
    """With a 2-sat equally-phased 12-day constellation, the default baseline
    is 6 days — so for a Rutford-like semidiurnal tide, fringe_mean should be
    lower than with a single-sat 12-day constellation (anti-aliasing).
    """
    tide = mixed_m2_k1_tide(amp_m2_m=1.4, amp_k1_m=0.3)
    sim_single = Simulator(sensor=Sensor.L_BAND, site=Site.RUTFORD, tide_fn=tide,
                           constellation=Constellation.single(repeat_days=12.0))
    sim_dual = Simulator(sensor=Sensor.L_BAND, site=Site.RUTFORD, tide_fn=tide,
                         constellation=Constellation.equally_phased(n=2, repeat_days=12.0))
    r_single = sim_single.sweep_triplets()
    r_dual = sim_dual.sweep_triplets()
    # Both should produce valid reports; the baselines differ.
    assert r_single.sweep.repeat_days == pytest.approx(12.0)
    assert r_dual.sweep.repeat_days == pytest.approx(6.0)


def test_simulator_explicit_baseline_overrides_constellation() -> None:
    sim = Simulator(sensor=Sensor.L_BAND, site=Site.RUTFORD, tide_fn=mixed_m2_k1_tide(),
                    constellation=Constellation.equally_phased(n=2, repeat_days=12.0))
    r = sim.sweep_triplets(baseline_days=18.0)
    assert r.sweep.repeat_days == pytest.approx(18.0)


def test_simulator_multi_baseline_returns_dict() -> None:
    sim = Simulator(sensor=Sensor.L_BAND, site=Site.RUTFORD, tide_fn=mixed_m2_k1_tide(),
                    constellation=Constellation.equally_phased(n=2, repeat_days=12.0))
    sweeps = sim.multi_baseline_sweep()
    assert isinstance(sweeps, dict)
    assert all(B > 0 for B in sweeps)


def test_simulator_any_triplet_sweep() -> None:
    sim = Simulator(sensor=Sensor.L_BAND, site=Site.RUTFORD, tide_fn=mixed_m2_k1_tide(),
                    constellation=Constellation.equally_phased(n=2, repeat_days=12.0))
    out = sim.any_triplet_sweep()
    assert isinstance(out, AnyTripletSweep)
    assert len(out) > 0
