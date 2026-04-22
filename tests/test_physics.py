"""Unit tests for the physics layer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tidal_insar_sim.physics import (
    dd_fringes_peak,
    dd_phase,
    double_difference_h,
    flexural_parameter_beta,
    flexural_rigidity_D,
    flexure_profile,
    limit_of_flexure,
    rigid_triplet_sweep,
)
from tidal_insar_sim.physics.flexure import G_STANDARD
from tidal_insar_sim.tides.mock import MixedTide, mixed_m2_k1_tide

# ---------------------------------------------------------------------------
# Flexure
# ---------------------------------------------------------------------------

PROTOTYPE_H = 400.0
PROTOTYPE_E = 0.88e9
PROTOTYPE_NU = 0.30
PROTOTYPE_RHO = 1028.0
PROTOTYPE_BETA_EXPECTED = 8.362e-4
PROTOTYPE_LFLEX_EXPECTED = 3757.0


def test_flexural_rigidity_D_matches_closed_form() -> None:
    d_val = flexural_rigidity_D(PROTOTYPE_E, PROTOTYPE_H, PROTOTYPE_NU)
    expected = PROTOTYPE_E * PROTOTYPE_H**3 / (12.0 * (1.0 - PROTOTYPE_NU**2))
    assert d_val == pytest.approx(expected, rel=1e-12)


def test_flexural_parameter_beta_matches_prototype_to_0p1pct() -> None:
    """Gate test from COWORK_AGENT_BRIEF §4.1."""
    beta = flexural_parameter_beta(
        young_modulus_pa=PROTOTYPE_E,
        ice_thickness_m=PROTOTYPE_H,
        poisson_ratio=PROTOTYPE_NU,
        rho_water=PROTOTYPE_RHO,
        gravity=G_STANDARD,
    )
    assert beta == pytest.approx(PROTOTYPE_BETA_EXPECTED, rel=1e-3)


def test_limit_of_flexure_matches_prototype_to_0p1pct() -> None:
    """Gate test from COWORK_AGENT_BRIEF §4.1."""
    beta = flexural_parameter_beta(PROTOTYPE_E, PROTOTYPE_H, PROTOTYPE_NU)
    l_flex = limit_of_flexure(beta)
    assert l_flex == pytest.approx(PROTOTYPE_LFLEX_EXPECTED, rel=1e-3)


def test_flexure_profile_is_clamped_at_grounding_line() -> None:
    """At s = 0 the flexure deflection must equal zero (clamped boundary)."""
    beta = flexural_parameter_beta(PROTOTYPE_E, PROTOTYPE_H, PROTOTYPE_NU)
    w_at_gl = flexure_profile(np.array([0.0]), h_tide_m=1.0, beta=beta)[0]
    assert abs(w_at_gl) < 1e-12


def test_flexure_profile_zero_on_grounded_ice() -> None:
    beta = flexural_parameter_beta(PROTOTYPE_E, PROTOTYPE_H, PROTOTYPE_NU)
    s = np.array([-1000.0, -500.0, -1.0])
    w = flexure_profile(s, h_tide_m=1.0, beta=beta)
    assert np.allclose(w, 0.0)


def test_flexure_profile_asymptotes_to_tide_height() -> None:
    """Far from the GL the floating ice should rise/fall with the ocean tide."""
    beta = flexural_parameter_beta(PROTOTYPE_E, PROTOTYPE_H, PROTOTYPE_NU)
    l_flex = limit_of_flexure(beta)
    s = np.array([3.0 * l_flex, 5.0 * l_flex, 10.0 * l_flex])
    for h in (-0.7, 0.0, 0.5, 1.3):
        w = flexure_profile(s, h_tide_m=h, beta=beta)
        assert np.allclose(w, h, atol=0.01 * max(abs(h), 1e-6) + 1e-6)


def test_flexure_profile_has_forebulge() -> None:
    """The forebulge (overshoot past h_tide) peaks at s = pi/beta = L_flex."""
    beta = flexural_parameter_beta(PROTOTYPE_E, PROTOTYPE_H, PROTOTYPE_NU)
    s_peak = math.pi / beta  # d/du [1 - e^{-u}(cos u + sin u)] = 2 e^{-u} sin u, zero at u=pi
    h_tide = 1.0
    w_peak = flexure_profile(np.array([s_peak]), h_tide_m=h_tide, beta=beta)[0]
    # Analytic: w(L_flex)/h = 1 + exp(-pi) ~ 1.0432
    assert w_peak == pytest.approx(1.0 + math.exp(-math.pi), rel=1e-6)
    assert w_peak > h_tide


def test_flexural_parameter_beta_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        flexural_rigidity_D(young_modulus_pa=1e9, ice_thickness_m=0.0, poisson_ratio=0.3)
    with pytest.raises(ValueError):
        flexural_rigidity_D(young_modulus_pa=-1.0, ice_thickness_m=400.0, poisson_ratio=0.3)
    with pytest.raises(ValueError):
        flexural_rigidity_D(young_modulus_pa=1e9, ice_thickness_m=400.0, poisson_ratio=0.7)


# ---------------------------------------------------------------------------
# Double-difference InSAR
# ---------------------------------------------------------------------------


def test_double_difference_h_formula() -> None:
    assert double_difference_h(0.0, 0.0, 0.0) == 0.0
    assert double_difference_h(1.0, 0.0, 0.0) == 1.0
    assert double_difference_h(0.0, 1.0, 0.0) == -2.0
    assert double_difference_h(0.5, 0.0, -0.5) == 0.0


def test_dd_fringes_peak_formula() -> None:
    """One LOS fringe = lambda/2 in vertical-projection terms."""
    lam = 0.2360
    theta = 39.0
    # one fringe: |h_DD| cos(theta) = lambda/2  =>  |h_DD| = lambda / (2 cos theta)
    h_dd = lam / (2.0 * np.cos(np.deg2rad(theta)))
    n = dd_fringes_peak(h_dd_peak_m=h_dd, wavelength_m=lam, incidence_deg=theta)
    assert n == pytest.approx(1.0, rel=1e-12)


def test_dd_phase_sign_and_magnitude() -> None:
    lam = 0.2360
    theta = 39.0
    # 1 fringe peak -> |phase| = 2 pi
    h_dd_field = np.array([lam / (2.0 * np.cos(np.deg2rad(theta)))])
    phase = dd_phase(h_dd_field, wavelength_m=lam, incidence_deg=theta)
    # sign = -1 for positive displacement (LOS away from sensor convention)
    assert phase[0] == pytest.approx(-2.0 * np.pi, rel=1e-12)


# ---------------------------------------------------------------------------
# Rigid-triplet sweep
# ---------------------------------------------------------------------------


def test_rigid_triplet_sweep_reproduces_prototype_sweep_shape() -> None:
    """Sweep over 29.53 days at 1 h step — matches prototype_sweep's 710 triplets."""
    tide = mixed_m2_k1_tide()
    sweep = rigid_triplet_sweep(
        tide_fn=tide,
        step_hours=1.0,
        duration_days=29.53,
        repeat_days=12.0,
        wavelength_m=0.2360,
        incidence_deg=39.0,
    )
    # 0, 1, ..., floor(29.53*24) — inclusive endpoint via `arange(0, N+step, step)`
    assert len(sweep) == round(29.53 * 24.0) + 1
    assert sweep.fringes.shape == sweep.h_dd_m.shape == sweep.delta_t_hours.shape
    # Fringe formula self-consistency
    theta = np.deg2rad(39.0)
    expected_fringes = np.abs(sweep.h_dd_m) * np.cos(theta) / (0.2360 / 2.0)
    assert np.allclose(sweep.fringes, expected_fringes)


def test_rigid_triplet_sweep_is_deterministic() -> None:
    tide = MixedTide(amp_m2_m=0.8, amp_k1_m=0.5, phase_m2_rad=0.0, phase_k1_rad=1.1)
    kwargs = {
        "tide_fn": tide,
        "step_hours": 1.0,
        "duration_days": 29.53,
        "repeat_days": 12.0,
        "wavelength_m": 0.2360,
        "incidence_deg": 39.0,
    }
    a = rigid_triplet_sweep(**kwargs)
    b = rigid_triplet_sweep(**kwargs)
    assert np.array_equal(a.fringes, b.fringes)


def test_rigid_triplet_sweep_rejects_bad_inputs() -> None:
    tide = mixed_m2_k1_tide()
    with pytest.raises(ValueError):
        rigid_triplet_sweep(tide, step_hours=0.0, duration_days=10,
                             repeat_days=12, wavelength_m=0.236, incidence_deg=39)
    with pytest.raises(ValueError):
        rigid_triplet_sweep(tide, step_hours=1.0, duration_days=-1,
                             repeat_days=12, wavelength_m=0.236, incidence_deg=39)
    with pytest.raises(ValueError):
        rigid_triplet_sweep(tide, step_hours=1.0, duration_days=10,
                             repeat_days=0, wavelength_m=0.236, incidence_deg=39)
