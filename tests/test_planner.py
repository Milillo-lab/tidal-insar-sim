"""Tests for the acquisition planner (M5)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tidal_insar_sim import Sensor, Simulator, Site
from tidal_insar_sim.planner import (
    REFERENCE_EPOCH,
    _local_maxima,
    plan_acquisitions,
    to_ics,
)
from tidal_insar_sim.tides.mock import mixed_m2_k1_tide


@pytest.fixture(scope="module")
def sim() -> Simulator:
    return Simulator(
        sensor=Sensor.L_BAND,
        site=Site.RUTFORD,
        tide_fn=mixed_m2_k1_tide(amp_m2_m=1.4, amp_k1_m=0.3),  # Rutford-like
    )


# ---------------------------------------------------------------------------
# local maxima
# ---------------------------------------------------------------------------


def test_local_maxima_flags_interior_peaks() -> None:
    x = np.array([0.0, 1.0, 3.0, 2.0, 4.0, 1.0])
    mask = _local_maxima(x)
    # Index 2 (value 3) and index 4 (value 4) are strict local maxima.
    assert mask.tolist() == [False, False, True, False, True, False]


def test_local_maxima_ignores_edges_and_plateaus() -> None:
    x = np.array([5.0, 5.0, 5.0, 5.0])
    assert not _local_maxima(x).any()
    assert not _local_maxima(np.array([3.0, 1.0])).any()


# ---------------------------------------------------------------------------
# plan_acquisitions
# ---------------------------------------------------------------------------


def test_planner_returns_sorted_top_n(sim: Simulator) -> None:
    plan = plan_acquisitions(
        sim,
        start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        n=5,
        min_fringes=3.0,
    )
    assert len(plan) <= 5
    assert list(plan.columns) == [
        "rank", "triplet_start_utc", "t1_date", "t2_date", "t3_date",
        "predicted_fringes", "tide_h1_m", "tide_h2_m", "tide_h3_m", "confidence",
    ]
    # Fringe count is monotonically non-increasing by rank.
    assert plan["predicted_fringes"].is_monotonic_decreasing


def test_planner_skips_below_threshold(sim: Simulator) -> None:
    """min_fringes=100 cannot be reached under the mock Rutford-like tide."""
    plan = plan_acquisitions(
        sim,
        start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        n=10,
        min_fringes=100.0,
    )
    assert plan.empty


def test_planner_confidence_is_in_unit_interval(sim: Simulator) -> None:
    plan = plan_acquisitions(
        sim,
        start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        n=3,
        min_fringes=3.0,
    )
    assert (plan["confidence"] >= 0).all()
    assert (plan["confidence"] <= 1).all()


def test_planner_dates_are_12_days_apart_for_nisar_l(sim: Simulator) -> None:
    """NISAR-L repeat = 12 days; t2-t1 and t3-t2 must match."""
    plan = plan_acquisitions(
        sim,
        start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        n=3,
        min_fringes=3.0,
    )
    for _, row in plan.iterrows():
        t1 = datetime.fromisoformat(row["t1_date"])
        t2 = datetime.fromisoformat(row["t2_date"])
        t3 = datetime.fromisoformat(row["t3_date"])
        assert (t2 - t1) == timedelta(days=12)
        assert (t3 - t2) == timedelta(days=12)


def test_planner_rejects_inverted_date_range(sim: Simulator) -> None:
    with pytest.raises(ValueError, match="end_date must be > start_date"):
        plan_acquisitions(
            sim,
            start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )


def test_planner_triplet_start_matches_dataframe_dates(sim: Simulator) -> None:
    plan = plan_acquisitions(
        sim,
        start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        n=3,
        min_fringes=3.0,
    )
    row = plan.iloc[0]
    ts = datetime.fromisoformat(row["triplet_start_utc"])
    # triplet_start_utc is the exact t1 timestamp; t1_date is that date (UTC).
    assert ts.date().isoformat() == row["t1_date"]


def test_report_recommended_triplets_delegates_to_planner(sim: Simulator) -> None:
    report = sim.sweep_triplets()
    plan = report.recommended_triplets(
        start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        n=3,
        min_fringes=3.0,
    )
    assert isinstance(plan, pd.DataFrame)
    assert not plan.empty


def test_report_recommended_triplets_needs_simulator() -> None:
    """If TripletReport is constructed without a Simulator, planner call fails."""
    from tidal_insar_sim.physics.fringe import rigid_triplet_sweep
    from tidal_insar_sim.report import TripletReport

    sweep = rigid_triplet_sweep(
        tide_fn=mixed_m2_k1_tide(),
        step_hours=1.0, duration_days=29.53, repeat_days=12.0,
        wavelength_m=0.2360, incidence_deg=39.0,
    )
    report = TripletReport(sweep=sweep, site_name="x", sensor_name="y")
    with pytest.raises(RuntimeError, match="bound Simulator"):
        report.recommended_triplets(
            start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )


# ---------------------------------------------------------------------------
# .ics export
# ---------------------------------------------------------------------------


def test_ics_export_has_three_vevents_per_row(sim: Simulator, tmp_path: Path) -> None:
    plan = plan_acquisitions(
        sim,
        start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
        n=2,
        min_fringes=3.0,
    )
    if plan.empty:
        pytest.skip("no plan to export")
    out = tmp_path / "plan.ics"
    to_ics(plan, out)
    text = out.read_text()
    assert "BEGIN:VCALENDAR" in text
    assert "END:VCALENDAR" in text
    assert text.count("BEGIN:VEVENT") == 3 * len(plan)
    assert "tidal-insar-sim" in text


# ---------------------------------------------------------------------------
# reference epoch sanity
# ---------------------------------------------------------------------------


def test_reference_epoch_is_2000_utc() -> None:
    assert datetime(2000, 1, 1, tzinfo=timezone.utc) == REFERENCE_EPOCH
