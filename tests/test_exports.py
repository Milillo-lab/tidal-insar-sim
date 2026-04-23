"""Tests for GeoJSON, KML and CSV export (M7)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

from tidal_insar_sim import Sensor, Simulator, Site
from tidal_insar_sim.io.geojson_export import (
    CLASSIFICATION_STYLES,
    FlexureFootprint,
    report_to_geojson,
    report_to_geojson_feature,
)
from tidal_insar_sim.io.kml_export import report_to_kml
from tidal_insar_sim.tides.mock import mixed_m2_k1_tide


@pytest.fixture(scope="module")
def report_and_sim():
    sim = Simulator(
        sensor=Sensor.NISAR_L, site=Site.RUTFORD,
        tide_fn=mixed_m2_k1_tide(amp_m2_m=1.4, amp_k1_m=0.3),
    )
    return sim.sweep_triplets(), sim


# ---------------------------------------------------------------------------
# FlexureFootprint
# ---------------------------------------------------------------------------


def test_flexure_footprint_is_closed_ring() -> None:
    fp = FlexureFootprint(lat=-75.0, lon=-106.0, radius_m=3500.0)
    coords = fp.to_coordinates()
    assert coords[0] == coords[-1], "ring must close"
    assert len(coords) == fp.n_vertices + 1


def test_flexure_footprint_radius_matches_input() -> None:
    """Great-circle distance from centre to any ring point equals radius_m."""
    import pyproj

    fp = FlexureFootprint(lat=-75.0, lon=-106.0, radius_m=3500.0)
    coords = fp.to_coordinates()
    geod = pyproj.Geod(ellps="WGS84")
    for lon, lat in coords[:-1]:  # skip the duplicated last vertex
        _, _, dist = geod.inv(fp.lon, fp.lat, lon, lat)
        assert abs(dist - fp.radius_m) < 1.0  # within 1 m


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------


def test_geojson_structure_is_feature_collection(report_and_sim) -> None:
    report, _sim = report_and_sim
    fc = report_to_geojson_feature(report)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    assert fc["features"][0]["geometry"]["type"] == "Point"
    assert fc["features"][1]["geometry"]["type"] == "Polygon"


def test_geojson_point_coords_are_lon_lat(report_and_sim) -> None:
    report, sim = report_and_sim
    fc = report_to_geojson_feature(report)
    lon, lat = fc["features"][0]["geometry"]["coordinates"]
    assert lon == sim.site.lon
    assert lat == sim.site.lat


def test_geojson_has_classification_and_color(report_and_sim) -> None:
    report, _sim = report_and_sim
    fc = report_to_geojson_feature(report)
    props = fc["features"][0]["properties"]
    assert props["classification"] in CLASSIFICATION_STYLES
    assert props["color_hint"].startswith("#")


def test_geojson_file_round_trip(tmp_path: Path, report_and_sim) -> None:
    report, _sim = report_and_sim
    out = tmp_path / "rutford.geojson"
    report_to_geojson(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection"


def test_geojson_skip_polygon(report_and_sim) -> None:
    report, _sim = report_and_sim
    fc = report_to_geojson_feature(report, include_polygon=False)
    assert len(fc["features"]) == 1
    assert fc["features"][0]["geometry"]["type"] == "Point"


def test_geojson_requires_bound_simulator() -> None:
    from tidal_insar_sim.physics.fringe import rigid_triplet_sweep
    from tidal_insar_sim.report import TripletReport

    sweep = rigid_triplet_sweep(
        tide_fn=mixed_m2_k1_tide(),
        step_hours=1.0, duration_days=29.53, repeat_days=12.0,
        wavelength_m=0.2360, incidence_deg=39.0,
    )
    # No simulator bound
    report = TripletReport(sweep=sweep, site_name="x", sensor_name="y")
    with pytest.raises(RuntimeError, match="no bound Simulator"):
        report_to_geojson_feature(report)


# ---------------------------------------------------------------------------
# KML
# ---------------------------------------------------------------------------


def test_kml_file_is_valid_xml(tmp_path: Path, report_and_sim) -> None:
    report, _sim = report_and_sim
    out = tmp_path / "rutford.kml"
    report_to_kml(report, out)
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag.endswith("kml")


def test_kml_contains_site_point_and_polygon(tmp_path: Path, report_and_sim) -> None:
    report, sim = report_and_sim
    out = tmp_path / "rutford.kml"
    report_to_kml(report, out)
    text = out.read_text(encoding="utf-8")
    assert sim.site.name in text
    assert "<Point>" in text
    assert "<Polygon>" in text
    # Lon, lat, 0 coordinates exist
    assert f"{sim.site.lon},{sim.site.lat},0" in text


def test_kml_color_is_aabbggrr(tmp_path: Path, report_and_sim) -> None:
    """KML color is AABBGGRR (alpha, then BGR) — hex, 8 chars."""
    report, _sim = report_and_sim
    out = tmp_path / "rutford.kml"
    report_to_kml(report, out)
    text = out.read_text(encoding="utf-8")
    import re
    colors = re.findall(r"<color>([0-9a-fA-F]+)</color>", text)
    assert colors, "kml should contain at least one color tag"
    for c in colors:
        assert len(c) == 8


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_report_to_csv(tmp_path: Path, report_and_sim) -> None:
    report, _sim = report_and_sim
    out = tmp_path / "rutford.csv"
    report.to_csv(out)
    df = pd.read_csv(out)
    assert len(df) == len(report.fringes)
    assert set(df.columns) >= {
        "delta_t_hours", "h1_m", "h2_m", "h3_m", "h_dd_m", "fringes",
        "site", "sensor",
    }
