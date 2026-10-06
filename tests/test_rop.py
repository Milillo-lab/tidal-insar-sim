"""ROP ingestion against the archived ASE snapshot used in the GMD paper."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tidal_insar_sim import rop
from tidal_insar_sim.site import Site

SNAP = Path(__file__).resolve().parents[1] / "data" / "rop_snapshot" / "ase_bbox_2026-10-02.geojson"
pytestmark = pytest.mark.skipif(not SNAP.exists(), reason="ROP snapshot not present")


@pytest.fixture(scope="module")
def scenes():
    return rop.scenes_from_geojson(rop.load_snapshot(SNAP))


def test_snapshot_counts(scenes):
    assert len(scenes) == 199
    assert len({s.track for s in scenes}) == 58


def test_mode_parser():
    m = rop.parse_mode(150, "L:SCI:SH:77M+---:HS:B4:D02")
    assert (m.band, m.polarisation, m.bandwidth_mhz, m.swath) == ("L", "SH", 77.0, "HS")
    assert m.is_lband_science
    assert not rop.parse_mode(153, "L:SCI:SV:05W+---:FS:B4:D09").is_lband_science
    assert not rop.parse_mode(246, "S:DB:SH:37W:B4:D10").is_lband_science


def test_incidence_within_nisar_range(scenes):
    start = datetime(2025, 9, 24, tzinfo=timezone.utc)
    acq = rop.site_acquisitions(scenes, Site.THWAITES.lat, Site.THWAITES.lon, start)
    assert acq
    assert all(32.0 <= a.incidence_deg <= 47.5 for a in acq)
    assert all(a.look_side == "L" for a in acq)


def test_one_acquisition_per_pass(scenes):
    acq = rop.site_acquisitions(scenes, Site.PIG.lat, Site.PIG.lon)
    keys = [(a.track_id.split(":")[0], a.ut_time) for a in acq]
    assert len(keys) == len(set(keys))


def test_mode_table_resolved_from_combinations():
    table = rop.mode_table(rop.load_snapshot(SNAP))
    assert table[150].bandwidth_mhz == 77.0 and table[150].swath == "HS"
    assert table[151].bandwidth_mhz == 40.0 and table[151].swath == "FS"
    assert not table[153].is_lband_science          # 5 MHz
    assert table[246].stream_key == table[150].stream_key   # L+S shares the L stream
    assert table[67].band == "S"
