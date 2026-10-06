"""NISAR Reference Observation Plan (ROP) ingestion.

Reads the NISAR merged-observation-plan ArcGIS feature service (live, or a
saved GeoJSON snapshot) and turns every scheduled L-band acquisition over a
site into an :class:`~tidal_insar_sim.orbit.Acquisition`, ready for the
per-track triplet optimizer.

What the ROP gives directly
---------------------------
- ``track``, ``frame``, ``passdirection``
- ``acquisition_time``: UT-of-day of the frame (same every 12-day cycle)
- ``date_mode_popup``: one line per scheduled date, ``YYYY-MM-DD | id,id``,
  listing the radar-mode ids flown on that date
- ``unique_radar_mode_list`` / ``unique_radar_mode_mnemonic_list``: the id to
  mnemonic map, e.g. ``150`` -> ``L:SCI:SH:77M+---:HS:B4:D02``
- the frame footprint polygon

What is derived here
--------------------
- Incidence angle at the site. NISAR is left-looking for the whole mission
  (NISAR Science Users' Handbook, 2019), so the nadir track lies to the right
  of the footprint. The site's cross-track distance from the right (near-range)
  edge of the footprint is converted to incidence on a spherical Earth, with
  the near edge pinned at ``NEAR_INCIDENCE_DEG``. Flight direction comes from
  the frame numbering, which increases along track.

InSAR-compatible streams
------------------------
A triplet must use one L-band mode on all three dates. Each acquisition is
keyed ``"<track>:<L-band mode>"`` (e.g. ``"149:LSH77HS"``), so the optimizer's
group-by-track never mixes modes; joint L+S modes share the stream of their
L-band part. Only L-band science modes with a main band of at least
``MIN_BANDWIDTH_MHZ`` are kept.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from tidal_insar_sim.orbit import Acquisition

ROP_FEATURE_SERVICE = (
    "https://services7.arcgis.com/WSiUmUhlFx4CtMBB/arcgis/rest/services/"
    "nisar_merged_observation_plans/FeatureServer/0"
)

EARTH_RADIUS_KM = 6371.0
NISAR_ALTITUDE_KM = 747.0
NEAR_INCIDENCE_DEG = 33.0
MIN_BANDWIDTH_MHZ = 20.0

_MNEMONIC = re.compile(
    r"^(?P<band>[LS]):(?P<kind>[A-Z-]+):(?P<pol>[A-Z]{2}):"
    r"(?P<bw>\d+)(?P<unit>[MW])(?P<aux>[^:]*):(?P<swath>[A-Z]{2})?"
)


@dataclass(frozen=True)
class RadarMode:
    """Parsed NISAR radar-mode mnemonic (first segment only)."""

    mode_id: int
    mnemonic: str
    band: str
    polarisation: str
    bandwidth_mhz: float
    swath: str

    @property
    def stream_key(self) -> str:
        """InSAR-compatibility key: the L-band segment only (an added S-band
        channel does not change the L-band acquisition)."""
        return f"L{self.polarisation}{self.bandwidth_mhz:g}{self.swath}"

    @property
    def is_lband_science(self) -> bool:
        return self.band == "L" and self.bandwidth_mhz >= MIN_BANDWIDTH_MHZ


def parse_mode(mode_id: int, mnemonic: str) -> RadarMode:
    """Parse e.g. ``L:SCI:SH:77M+---:HS:B4:D02``.

    Main-band bandwidth is the leading number before ``M`` (MHz). A ``W``
    suffix (as in ``05W``) marks a narrow-band mode and is read as that many
    MHz, so it falls under ``MIN_BANDWIDTH_MHZ`` and is excluded.
    """
    first = mnemonic.strip().split()[0]
    m = _MNEMONIC.match(first)
    if m is None:
        return RadarMode(mode_id, mnemonic, "?", "?", 0.0, "?")
    return RadarMode(
        mode_id=mode_id,
        mnemonic=first,
        band=m["band"],
        polarisation=m["pol"],
        bandwidth_mhz=float(m["bw"]),
        swath=m["swath"] or "?",
    )


@dataclass(frozen=True)
class RopScene:
    track: int
    frame: int
    pass_direction: str          # "ASC" / "DESC"
    ut_of_day: timedelta
    dates_modes: tuple[tuple[str, tuple[int, ...]], ...]
    modes: dict[int, RadarMode]
    geometry: dict

    @property
    def scene_id(self) -> str:
        return f"T{self.track:03d}F{self.frame:03d}"


def query_rop(bbox: tuple[float, float, float, float], timeout: float = 120.0) -> dict:
    """Return the live ROP features intersecting ``bbox`` (lon0, lat0, lon1, lat1) as GeoJSON."""
    import requests

    params = {
        "where": "1=1",
        "geometry": ",".join(str(v) for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "geojson",
    }
    r = requests.get(f"{ROP_FEATURE_SERVICE}/query", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def load_snapshot(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _parse_ut(s: str) -> timedelta:
    """UT-of-day; some frames list several times (one per mode), seconds to minutes apart.

    The median is used: a few minutes moves the tide by millimetres.
    """
    secs = []
    for part in str(s).split(","):
        h, m, sec = (float(x) for x in part.strip().split(":"))
        secs.append(3600 * h + 60 * m + sec)
    return timedelta(seconds=float(np.median(secs)))


def mode_table(gj: dict) -> dict[int, RadarMode]:
    """Global radar-mode id -> mode, from the ``radar_mode_combination_N`` fields.

    ``unique_radar_mode_list`` and ``unique_radar_mode_mnemonic_list`` are NOT in
    the same order (in the 2026-10-02 snapshot id 153 sits beside a 77 MHz
    mnemonic on some frames), so they cannot be zipped. The combination fields
    pair ids and mnemonics position by position; an id seen with two different
    mnemonics there raises rather than guessing.
    """
    seen: dict[int, set[str]] = {}
    for f in gj["features"]:
        p = f["properties"]
        for n in range(1, 10):
            ids, mns = p.get(f"radar_mode_combination_{n}"), p.get(f"radar_mode_mnemonic_combination_{n}")
            if not ids or not mns:
                continue
            ids_l = [int(x) for x in str(ids).split(",") if x.strip()]
            mns_l = [x.strip() for x in str(mns).split(",")]
            if len(ids_l) != len(mns_l):
                continue
            for i, mn in zip(ids_l, mns_l):
                seen.setdefault(i, set()).add(mn)
    clash = {i: v for i, v in seen.items() if len(v) > 1}
    if clash:
        msg = f"radar-mode ids map to more than one mnemonic: {clash}"
        raise ValueError(msg)
    return {i: parse_mode(i, next(iter(v))) for i, v in seen.items()}


def scenes_from_geojson(gj: dict) -> list[RopScene]:
    table = mode_table(gj)
    out: list[RopScene] = []
    for f in gj["features"]:
        p = f["properties"]
        ids = [int(x) for x in str(p["unique_radar_mode_list"]).split(",") if x.strip()]
        modes = {i: table[i] for i in ids if i in table}
        dm = []
        for line in str(p["date_mode_popup"]).splitlines():
            if "|" not in line:
                continue
            d, ms = (x.strip() for x in line.split("|", 1))
            dm.append((d, tuple(int(x) for x in ms.split(",") if x.strip())))
        out.append(RopScene(
            track=int(p["track"]),
            frame=int(p["frame"]),
            pass_direction="ASC" if str(p["passdirection"]).startswith("A") else "DESC",
            ut_of_day=_parse_ut(p["acquisition_time"]),
            dates_modes=tuple(dm),
            modes=modes,
            geometry=f["geometry"],
        ))
    return out


def _polar(geom_or_xy):
    import pyproj
    from shapely.geometry import shape
    from shapely.ops import transform

    tr = pyproj.Transformer.from_crs(4326, 3031, always_xy=True).transform
    if isinstance(geom_or_xy, dict):
        return transform(tr, shape(geom_or_xy))
    return np.array(tr(*geom_or_xy))


def incidence_from_ground_range(g_km: float) -> float:
    """Incidence (deg) at ground range ``g_km`` from nadir, spherical Earth."""
    r, rs = EARTH_RADIUS_KM, EARTH_RADIUS_KM + NISAR_ALTITUDE_KM
    gam = g_km / r
    dx, dy = -r * np.sin(gam), rs - r * np.cos(gam)
    return float(np.degrees(np.arccos((rs * np.cos(gam) - r) / np.hypot(dx, dy))))


def _near_ground_range_km() -> float:
    lo, hi = 0.0, 1500.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if incidence_from_ground_range(mid) < NEAR_INCIDENCE_DEG:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def site_incidence_deg(scene: RopScene, along_track_xy: np.ndarray, lat: float, lon: float) -> float:
    """Incidence at the site for a left-looking radar.

    ``along_track_xy`` is the flight direction in EPSG:3031 (any length).
    EPSG:3031 is not mirrored, so the left normal is the vector rotated +90 deg.
    """
    poly = _polar(scene.geometry)
    rect = poly.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    edges = [np.array([xs[i + 1] - xs[i], ys[i + 1] - ys[i]]) for i in range(4)]
    width = min(np.hypot(*e) for e in edges)
    u = along_track_xy / np.hypot(*along_track_xy)
    left = np.array([-u[1], u[0]])
    c = np.array([rect.centroid.x, rect.centroid.y])
    cross = float((_polar((lon, lat)) - c) @ left)      # + toward far range
    d_from_near_km = (cross + width / 2.0) / 1e3
    return incidence_from_ground_range(_near_ground_range_km() + d_from_near_km)


def _along_track(scenes: list[RopScene], scene: RopScene) -> np.ndarray:
    """Flight direction from the neighbouring frame centroids of the same track."""
    same = {s.frame: _polar(s.geometry).centroid for s in scenes if s.track == scene.track}
    f = scene.frame
    ahead = same.get(f + 1)
    behind = same.get(f - 1)
    here = same[f]
    if ahead is not None:
        return np.array([ahead.x - here.x, ahead.y - here.y])
    if behind is not None:
        return np.array([here.x - behind.x, here.y - behind.y])
    # Single frame: fall back on pass direction (ascending moves away from the pole).
    radial = np.array([here.x, here.y]) / np.hypot(here.x, here.y)
    rect = _polar(scene.geometry).minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    edges = [np.array([xs[i + 1] - xs[i], ys[i + 1] - ys[i]]) for i in range(4)]
    long_axis = max(edges, key=lambda e: np.hypot(*e))
    sign = 1.0 if (long_axis @ radial > 0) == (scene.pass_direction == "ASC") else -1.0
    return sign * long_axis


def site_acquisitions(
    scenes: list[RopScene],
    lat: float,
    lon: float,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Acquisition]:
    """Every scheduled L-band science acquisition whose footprint contains the site.

    When several frames of one track contain the site (frame overlap), the
    frame whose centroid is nearest the site is used, so each acquisition is
    counted once.
    """
    from shapely.geometry import Point

    pt = _polar((lon, lat))
    p = Point(*pt)
    best: dict[int, RopScene] = {}
    for s in scenes:
        poly = _polar(s.geometry)
        if not poly.contains(p):
            continue
        prev = best.get(s.track)
        if prev is None or poly.centroid.distance(p) < _polar(prev.geometry).centroid.distance(p):
            best[s.track] = s

    out: list[Acquisition] = []
    for track, s in sorted(best.items()):
        inc = site_incidence_deg(s, _along_track(scenes, s), lat, lon)
        for d, mode_ids in s.dates_modes:
            t = datetime.fromisoformat(d).replace(tzinfo=timezone.utc) + s.ut_of_day
            if (start and t < start) or (end and t >= end):
                continue
            # One pass is one acquisition. Where a date lists several L-band modes
            # (different parts of the frame), keep the widest band.
            lband = [s.modes[m] for m in mode_ids if m in s.modes and s.modes[m].is_lband_science]
            if lband:
                best_mode = max(lband, key=lambda md: (md.bandwidth_mhz, -md.mode_id))
                out.append(Acquisition(
                    ut_time=t,
                    track_id=f"{track}:{best_mode.stream_key}",
                    rev_in_cycle=track,
                    asc_desc=s.pass_direction,
                    look_side="L",
                    incidence_deg=inc,
                    azimuth_deg=float("nan"),
                    sat_lat_deg=float("nan"),
                    sat_lon_deg=float("nan"),
                    range_km=float("nan"),
                ))
    out.sort(key=lambda a: (a.track_id, a.ut_time))
    return out
