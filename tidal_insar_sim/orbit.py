"""Orbit-aware acquisition opportunity model.

Replaces the simple `Constellation(phase_offset_days, repeat_days)` schedule
with a deterministic, ephemeris-driven list of `Acquisition`s.

For each candidate pass we record:
- exact UT timestamp
- track_id (so we can group InSAR-compatible acquisitions)
- incidence angle at the site (constant per track_id by construction)
- ascending/descending + look-side
- tide height at that exact UT (CATS2008)

Triplets must be drawn from a single track_id. Different track_ids represent
non-coherent acquisition geometries — independent DDInSAR products, not
mixable into one triplet.

NISAR-specific defaults (NORAD 65053, 747 km altitude, 12-day repeat) are
shipped in `tidal_insar_sim/data/nisar.tle`. Override via TLE strings or by
swapping the `OrbitProvider`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tidal_insar_sim.site import Site


# NISAR design swath (SweepSAR, dual-look). Values are reasonable defaults
# matching the NISAR Project's published Science Coverage Plan: incidence at
# ground in [33, 47] deg, dual-look swap-able per orbit.
NISAR_INCIDENCE_MIN_DEG = 33.0
NISAR_INCIDENCE_MAX_DEG = 47.0
NISAR_REVS_PER_REPEAT = 173        # 14 rev/day × 12 d = 168, but actual exact repeat is 173 revs
NISAR_REPEAT_DAYS = 12.0


@dataclass(frozen=True)
class Acquisition:
    """A single SAR acquisition opportunity over a site."""

    ut_time: datetime          # exact UT (timezone-aware)
    track_id: str              # InSAR-compatibility key
    rev_in_cycle: int          # 0..N-1 within the exact repeat
    asc_desc: str              # "ASC" or "DESC"
    look_side: str             # "L" or "R"
    incidence_deg: float
    azimuth_deg: float         # azimuth of satellite from site
    sat_lat_deg: float         # sub-satellite latitude at acquisition time
    sat_lon_deg: float
    range_km: float            # slant range from site to satellite
    tide_h_m: float = float("nan")  # filled in by SimulatorOrbit

    def __repr__(self) -> str:
        return (
            f"Acquisition({self.ut_time.isoformat()}  "
            f"track={self.track_id}  "
            f"{self.asc_desc}/{self.look_side}  "
            f"theta={self.incidence_deg:.1f}deg  "
            f"h_tide={self.tide_h_m:+.2f}m)"
        )


def default_nisar_tle() -> tuple[str, str, str]:
    """Return the bundled NISAR TLE as `(name, line1, line2)`."""
    text = files("tidal_insar_sim.data").joinpath("nisar.tle").read_text()
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        msg = f"TLE file must have 3 lines, got {len(lines)}"
        raise ValueError(msg)
    return lines[0], lines[1], lines[2]


@dataclass
class TLEOrbitProvider:
    """Ephemeris-driven acquisition provider using SGP4 propagation.

    For SAR opportunity analysis, a "pass" is when the satellite is within
    the SweepSAR access region — i.e. the site is at incidence angle within
    `[incidence_min_deg, incidence_max_deg]` from the satellite. NISAR-like
    dual-look means we record passes on both sides of nadir.
    """

    tle_name: str
    tle_line1: str
    tle_line2: str
    incidence_min_deg: float = NISAR_INCIDENCE_MIN_DEG
    incidence_max_deg: float = NISAR_INCIDENCE_MAX_DEG
    revs_per_repeat: int = NISAR_REVS_PER_REPEAT

    @classmethod
    def nisar(cls) -> TLEOrbitProvider:
        name, l1, l2 = default_nisar_tle()
        return cls(tle_name=name, tle_line1=l1, tle_line2=l2)

    @classmethod
    def from_tle_file(cls, path: str | Path) -> TLEOrbitProvider:
        text = Path(path).read_text(encoding="utf-8")
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        return cls(tle_name=lines[0], tle_line1=lines[1], tle_line2=lines[2])

    def acquisitions(
        self,
        site: Site,
        start: datetime,
        end: datetime,
        *,
        time_step_seconds: float = 30.0,
    ) -> list[Acquisition]:
        """Compute every acquisition opportunity over the site in [start, end].

        Walks the ephemeris at `time_step_seconds` cadence; each contiguous
        run where the incidence angle from the site to the satellite falls in
        the SAR access band is condensed into one Acquisition (taken at the
        closest approach within the run).
        """
        from skyfield.api import EarthSatellite, load, wgs84

        ts = load.timescale()
        sat = EarthSatellite(self.tle_line1, self.tle_line2,
                             self.tle_name, ts)
        topos = wgs84.latlon(site.lat, site.lon, elevation_m=0.0)
        diff = sat - topos

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        n_steps = int((end - start).total_seconds() / time_step_seconds) + 1
        times = ts.from_datetimes([
            start + timedelta(seconds=time_step_seconds * i) for i in range(n_steps)
        ])

        topo = diff.at(times)
        alt, az, distance = topo.altaz()
        elevation_deg = np.asarray(alt.degrees)
        azimuth_deg = np.asarray(az.degrees)
        range_km = np.asarray(distance.km)

        incidence_deg = 90.0 - elevation_deg
        in_swath = (incidence_deg >= self.incidence_min_deg) & (
            incidence_deg <= self.incidence_max_deg
        )

        # Sub-satellite latitude/longitude + asc/desc indicator
        geocentric = sat.at(times)
        sub_lat_deg, sub_lon_deg = wgs84.latlon_of(geocentric)
        sub_lat_arr = np.asarray(sub_lat_deg.degrees)
        sub_lon_arr = np.asarray(sub_lon_deg.degrees)
        # ascending = latitude is increasing
        d_lat = np.gradient(sub_lat_arr)
        asc_mask = d_lat > 0

        # Track-id arithmetic uses the NOMINAL exact-repeat period rather than
        # the TLE's mean motion — the TLE's mean motion drifts a few seconds
        # per orbit, which accumulates and breaks track-id mod arithmetic.
        # NISAR: 12-day exact repeat = 173 revs => 5994.22 s/orbit nominal.
        ref_epoch = sat.epoch.utc_datetime()
        period_seconds = NISAR_REPEAT_DAYS * 86400.0 / float(self.revs_per_repeat)

        return _condense_passes(
            in_swath=in_swath,
            ut_times=[start + timedelta(seconds=time_step_seconds * i) for i in range(n_steps)],
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg,
            range_km=range_km,
            asc_mask=asc_mask,
            sub_lat=sub_lat_arr,
            sub_lon=sub_lon_arr,
            site=site,
            ref_epoch=ref_epoch,
            period_seconds=period_seconds,
            revs_per_repeat=self.revs_per_repeat,
        )


def _condense_passes(
    *,
    in_swath: NDArray[np.bool_],
    ut_times: list[datetime],
    elevation_deg: NDArray[np.float64],
    azimuth_deg: NDArray[np.float64],
    range_km: NDArray[np.float64],
    asc_mask: NDArray[np.bool_],
    sub_lat: NDArray[np.float64],
    sub_lon: NDArray[np.float64],
    site: Site,
    ref_epoch: datetime,
    period_seconds: float,
    revs_per_repeat: int,
    merge_gap_seconds: float = 300.0,
) -> list[Acquisition]:
    """Cluster in-swath samples into one Acquisition per physical pass.

    Two in-swath windows separated by less than `merge_gap_seconds` are
    considered the same pass (covers the SAR nadir-blanking gap that splits
    high-latitude near-zenith encounters into "before" and "after" windows).

    Each pass is reported at the sample with the LOWEST incidence (closest
    approach to nadir, modulo the nadir-blank limit).
    """
    out: list[Acquisition] = []
    indices = np.flatnonzero(in_swath)
    if indices.size == 0:
        return out

    # Cluster indices that are within merge_gap_seconds of each other
    # (using the actual ut_times, not a fixed step).
    clusters: list[list[int]] = []
    current: list[int] = [int(indices[0])]
    for idx in indices[1:]:
        idx = int(idx)
        gap = (ut_times[idx] - ut_times[current[-1]]).total_seconds()
        if gap <= merge_gap_seconds:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    incidence_arr = 90.0 - elevation_deg

    for cluster in clusters:
        # Closest approach within the cluster: lowest incidence (= highest elev).
        best = cluster[int(np.argmin(incidence_arr[cluster]))]
        ut = ut_times[best]
        is_asc = bool(asc_mask[best])

        # Look side via heading * cross-track sign at sub-satellite point.
        # 1. Heading direction = bearing from sub_sat[best-1] to sub_sat[best+1]
        prev_idx = max(best - 1, 0)
        next_idx = min(best + 1, len(ut_times) - 1)
        head_lat = float(sub_lat[next_idx] - sub_lat[prev_idx])
        head_lon = float(sub_lon[next_idx] - sub_lon[prev_idx])
        # 2. Vector from sub-sat to site
        site_dlat = float(site.lat - sub_lat[best])
        site_dlon = float(site.lon - sub_lon[best])
        # Wrap longitude diffs to [-180, 180]
        site_dlon = (site_dlon + 180.0) % 360.0 - 180.0
        head_lon = (head_lon + 180.0) % 360.0 - 180.0
        # 3. 2D cross product: positive => site is to the LEFT of heading;
        #    negative => to the RIGHT. (Standard right-hand rule with N=+y, E=+x.)
        cross = head_lat * site_dlon - head_lon * site_dlat
        look_side = "L" if cross > 0 else "R"

        # Track id: integer orbit count since TLE epoch, mod revs_per_repeat.
        delta_s = (ut - ref_epoch).total_seconds()
        rev = int(round(delta_s / period_seconds)) % revs_per_repeat
        track_id = f"{'A' if is_asc else 'D'}{rev:03d}{look_side}"

        out.append(Acquisition(
            ut_time=ut,
            track_id=track_id,
            rev_in_cycle=rev,
            asc_desc="ASC" if is_asc else "DESC",
            look_side=look_side,
            incidence_deg=float(incidence_arr[best]),
            azimuth_deg=float(azimuth_deg[best]),
            sat_lat_deg=float(sub_lat[best]),
            sat_lon_deg=float(sub_lon[best]),
            range_km=float(range_km[best]),
        ))
    return out


def fill_tide_heights(
    acquisitions: Iterable[Acquisition],
    *,
    site: Site,
    tide_fn,
    epoch_for_hours: datetime | None = None,
) -> list[Acquisition]:
    """Replace each acquisition with a copy carrying `tide_h_m` filled in.

    `tide_fn(t_hours)` is the tide-from-hours-since-epoch callable that the
    Simulator already uses. `epoch_for_hours` defaults to 2000-01-01 UTC
    (CATS2008 reference epoch — same as the rest of the tool).
    """
    if epoch_for_hours is None:
        epoch_for_hours = datetime(2000, 1, 1, tzinfo=timezone.utc)
    acqs = list(acquisitions)
    if not acqs:
        return []
    hours = np.array([
        (a.ut_time - epoch_for_hours).total_seconds() / 3600.0 for a in acqs
    ], dtype=np.float64)
    heights = tide_fn(hours)
    out: list[Acquisition] = []
    for a, h in zip(acqs, heights, strict=True):
        out.append(Acquisition(
            ut_time=a.ut_time, track_id=a.track_id, rev_in_cycle=a.rev_in_cycle,
            asc_desc=a.asc_desc, look_side=a.look_side,
            incidence_deg=a.incidence_deg, azimuth_deg=a.azimuth_deg,
            sat_lat_deg=a.sat_lat_deg, sat_lon_deg=a.sat_lon_deg,
            range_km=a.range_km, tide_h_m=float(h),
        ))
    return out


def acquisitions_to_dataframe(acqs: Iterable[Acquisition]) -> pd.DataFrame:
    rows = []
    for a in acqs:
        rows.append({
            "ut_time": a.ut_time.isoformat(),
            "track_id": a.track_id,
            "rev_in_cycle": a.rev_in_cycle,
            "asc_desc": a.asc_desc,
            "look_side": a.look_side,
            "incidence_deg": a.incidence_deg,
            "azimuth_deg": a.azimuth_deg,
            "sat_lat_deg": a.sat_lat_deg,
            "sat_lon_deg": a.sat_lon_deg,
            "range_km": a.range_km,
            "tide_h_m": a.tide_h_m,
        })
    return pd.DataFrame(rows)
