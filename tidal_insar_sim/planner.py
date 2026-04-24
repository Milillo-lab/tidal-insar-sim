"""Acquisition planner (M5).

Given a Simulator and a date window, scan all candidate triplet starts at 1-h
resolution, find local maxima in predicted fringe count, compute a per-triplet
confidence from ±1 h acquisition jitter Monte Carlo, and return the top N as a
sortable `pandas.DataFrame`. Also exports to a Google/Outlook-compatible `.ics`
calendar.

References
----------
- brief §4.5 (acquisition planner)
- user Q&A Apr 2026: confidence = 1 - P(fringes < 1) under +/- 1 h jitter
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tidal_insar_sim.simulator import Simulator

# CATS2008 reference epoch (matches EPOCH_2000 in tides/cats2008.py)
REFERENCE_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def plan_acquisitions(
    sim: Simulator,
    start_date: datetime,
    end_date: datetime,
    n: int = 10,
    min_fringes: float = 3.0,
    *,
    step_hours: float = 1.0,
    jitter_hours: float = 1.0,
    n_mc_realisations: int = 4_000,
    rng_seed: int | None = 2026,
) -> pd.DataFrame:
    """Return top `n` candidate triplets between `start_date` and `end_date`.

    The first acquisition `t1` must land inside [start_date, end_date]. The
    planner evaluates the rigid-triplet fringe count at each `Delta_t` on a
    `step_hours` grid, keeps local maxima above `min_fringes`, ranks them by
    predicted fringes, and returns the top `n` with per-row confidence.

    Output columns
    --------------
    rank, triplet_start_utc, t1_date, t2_date, t3_date, predicted_fringes,
    tide_h1_m, tide_h2_m, tide_h3_m, confidence.
    """
    if end_date <= start_date:
        msg = f"end_date must be > start_date, got {start_date=}, {end_date=}"
        raise ValueError(msg)
    start_date = _ensure_utc(start_date)
    end_date = _ensure_utc(end_date)

    # Map user-supplied dates to "hours since epoch" — matches the Simulator's
    # internal clock used by sweep_triplets().
    start_h = _hours_from_epoch(start_date)
    end_h = _hours_from_epoch(end_date)
    if end_h <= start_h:
        return _empty_frame()

    delta_t = np.arange(start_h, end_h + step_hours, step_hours, dtype=np.float64)
    tide_fn = sim._require_tide_fn()
    repeat_hours = sim._effective_baseline_days * 24.0
    h1 = tide_fn(delta_t)
    h2 = tide_fn(delta_t + repeat_hours)
    h3 = tide_fn(delta_t + 2.0 * repeat_hours)
    h_dd = h1 - 2.0 * h2 + h3
    theta = np.deg2rad(sim.sensor.incidence_deg)
    half_lam = sim.sensor.wavelength_m / 2.0
    fringes = np.abs(h_dd) * np.cos(theta) / half_lam

    # Find local maxima (both sides higher) above the threshold.
    max_mask = _local_maxima(fringes) & (fringes >= min_fringes)
    idx = np.nonzero(max_mask)[0]
    if idx.size == 0:
        return _empty_frame()

    # Rank by fringe count, keep top n.
    rank_order = np.argsort(fringes[idx])[::-1][:n]
    chosen = idx[rank_order]

    rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()

    rows: list[dict[str, object]] = []
    for r, i in enumerate(chosen, start=1):
        dt_hours = float(delta_t[i])
        t1 = REFERENCE_EPOCH + timedelta(hours=dt_hours)
        t2 = t1 + timedelta(hours=repeat_hours)
        t3 = t1 + timedelta(hours=2 * repeat_hours)
        conf = sim.confidence(
            triplet_start_hours=dt_hours,
            jitter_hours=jitter_hours,
            n_realisations=n_mc_realisations,
            rng=rng,
        )
        rows.append({
            "rank": r,
            "triplet_start_utc": t1.isoformat(),
            "t1_date": t1.date().isoformat(),
            "t2_date": t2.date().isoformat(),
            "t3_date": t3.date().isoformat(),
            "predicted_fringes": float(fringes[i]),
            "tide_h1_m": float(h1[i]),
            "tide_h2_m": float(h2[i]),
            "tide_h3_m": float(h3[i]),
            "confidence": conf,
        })
    return pd.DataFrame(rows)


def to_ics(plan: pd.DataFrame, path: str | Path, *, calendar_name: str = "tidal-insar-sim") -> None:
    """Export a planner DataFrame to an .ics calendar (importable to Google/Outlook).

    Each row becomes three 1-hour VEVENTS (t1/t2/t3) describing the planned
    acquisition. Confidence + predicted fringes are stored in the DESCRIPTION.
    """
    from icalendar import Calendar, Event

    cal = Calendar()  # type: ignore[no-untyped-call]
    cal.add("prodid", f"-//{calendar_name}//tidal-insar-sim//EN")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", calendar_name)

    for _, row in plan.iterrows():
        for i, col in enumerate(("t1_date", "t2_date", "t3_date"), start=1):
            date_str = str(row[col])
            date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            ev = Event()  # type: ignore[no-untyped-call]
            ev.add("uid", f"{row['triplet_start_utc']}-t{i}@tidal-insar-sim")
            ev.add("summary", f"Rank #{int(row['rank'])} DDInSAR t{i}")
            ev.add(
                "description",
                (
                    f"predicted_fringes={row['predicted_fringes']:.2f}  "
                    f"confidence={row['confidence']:.3f}\n"
                    f"tide_h{i}_m={row[f'tide_h{i}_m']:+.3f}"
                ),
            )
            ev.add("dtstart", date)
            ev.add("dtend", date + timedelta(hours=1))
            cal.add_component(ev)

    Path(path).write_bytes(cal.to_ical())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _local_maxima(x: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Boolean mask of strict local maxima in a 1D array."""
    if x.size < 3:
        return np.zeros_like(x, dtype=bool)
    higher_left = np.r_[False, x[1:] > x[:-1]]
    higher_right = np.r_[x[:-1] > x[1:], False]
    mask: NDArray[np.bool_] = higher_left & higher_right
    return mask


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rank", "triplet_start_utc", "t1_date", "t2_date", "t3_date",
            "predicted_fringes", "tide_h1_m", "tide_h2_m", "tide_h3_m", "confidence",
        ],
    )


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _hours_from_epoch(dt: datetime) -> float:
    return (dt - REFERENCE_EPOCH).total_seconds() / 3600.0
