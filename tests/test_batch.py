"""Tests for batch comparison (M6)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tidal_insar_sim import Sensor, Site
from tidal_insar_sim.batch import (
    batch_compare,
    batch_from_parquet,
    batch_to_parquet,
    pivot_heatmap,
)
from tidal_insar_sim.constellation import Constellation
from tidal_insar_sim.tides.mock import mixed_m2_k1_tide, multi_constituent_tide

SITES = [Site.THWAITES, Site.RUTFORD]
SENSORS = [Sensor.L_BAND, Sensor.C_BAND]
CONSTELLATIONS = [
    Constellation.single(repeat_days=12.0, name="NISAR-like"),
    Constellation.equally_phased(n=2, repeat_days=12.0, name_prefix="S1"),
]


def _mock_tide_factory(site: Site):
    """Use site-distinct mock tides so batch outputs differ by site."""
    if site.name == "Rutford_GL":
        return multi_constituent_tide(
            amplitudes_m={"M2": 1.4, "K1": 0.3},
            phases_deg={"M2": 340.0, "K1": 170.0},
        )
    return mixed_m2_k1_tide()


def test_batch_compare_shape() -> None:
    df = batch_compare(SITES, SENSORS, tide_fn_factory=_mock_tide_factory)
    assert len(df) == len(SITES) * len(SENSORS)
    assert set(df["site"].unique()) == {"Thwaites_GL", "Rutford_GL"}
    assert {b.split(" ")[0] for b in df["sensor"].unique()} == {"L-band", "C-band"}


def test_batch_compare_has_expected_summary_columns() -> None:
    df = batch_compare(SITES, SENSORS, tide_fn_factory=_mock_tide_factory)
    expected = {
        "site", "sensor", "lat", "lon", "wavelength_m",
        "incidence_deg", "ice_thickness_m", "n_triplets",
        "P_usable_ge_3fr", "P_robust_ge_5fr", "P_null_lt_0p5fr",
        "fringe_mean", "fringe_median", "fringe_max",
        "verdict", "hDD_range_m", "effective_baseline_days",
    }
    assert expected <= set(df.columns)


def test_batch_rutford_beats_thwaites_on_fringe_mean() -> None:
    df = batch_compare(SITES, SENSORS, tide_fn_factory=_mock_tide_factory)
    rutford = df[df["site"] == "Rutford_GL"]
    thwaites = df[df["site"] == "Thwaites_GL"]
    for sensor_name in df["sensor"].unique():
        fr_r = rutford[rutford["sensor"] == sensor_name]["fringe_mean"].iloc[0]
        fr_t = thwaites[thwaites["sensor"] == sensor_name]["fringe_mean"].iloc[0]
        assert fr_r > fr_t, f"{sensor_name}: Rutford={fr_r:.2f} not > Thwaites={fr_t:.2f}"


def test_batch_parquet_round_trip(tmp_path: Path) -> None:
    df = batch_compare(SITES, SENSORS, tide_fn_factory=_mock_tide_factory)
    out = tmp_path / "batch.parquet"
    batch_to_parquet(df, out)
    round_tripped = batch_from_parquet(out)
    # Same shape + same column set
    assert df.shape == round_tripped.shape
    assert set(df.columns) == set(round_tripped.columns)
    # Numeric columns equal (parquet preserves floats exactly)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    pd.testing.assert_frame_equal(
        df[numeric_cols], round_tripped[numeric_cols], check_dtype=False
    )
    # String columns equal verbatim
    string_cols = ["site", "sensor", "verdict"]
    for col in string_cols:
        assert (df[col] == round_tripped[col]).all()
    # List-typed columns (hDD_range_m) equal elementwise within floating precision
    for a, b in zip(df["hDD_range_m"], round_tripped["hDD_range_m"], strict=True):
        assert list(a) == pytest.approx(list(b))


def test_pivot_heatmap_shape() -> None:
    df = batch_compare(SITES, SENSORS, tide_fn_factory=_mock_tide_factory)
    heatmap = pivot_heatmap(df, metric="P_usable_ge_3fr")
    assert heatmap.shape == (len(SITES), len(SENSORS))
    # Values in [0, 1]
    assert (heatmap.values >= 0).all()
    assert (heatmap.values <= 1).all()


def test_pivot_heatmap_rejects_unknown_metric() -> None:
    df = batch_compare(SITES, SENSORS, tide_fn_factory=_mock_tide_factory)
    with pytest.raises(KeyError, match="not in DataFrame columns"):
        pivot_heatmap(df, metric="nonexistent_metric")
