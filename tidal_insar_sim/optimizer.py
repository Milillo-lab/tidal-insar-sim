"""Per-track DDInSAR triplet optimizer.

Given a list of `Acquisition`s, group by `track_id` (each group is an
InSAR-compatible stream — identical incidence and look geometry) and
enumerate all valid `(t_i < t_j < t_k)` triplets within each group. Rank
by the chosen objective and return the top N.

The optimizer is brute-force — N choose 3 with N ~ 30 acquisitions per
track per year is ~4500 candidates, milliseconds.

Objectives
----------
- ``"max_fringes"`` — strongest tidal DDInSAR signal (default).
- ``"min_fringes"`` — minimum tidal contamination (for non-tidal targets).
- ``"target=K"`` — closest to K fringes (e.g. K=4 for unwrappable triplets).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tidal_insar_sim.orbit import Acquisition


@dataclass(frozen=True)
class Triplet:
    """A ranked DDInSAR triplet from a single track."""

    track_id: str
    a1: Acquisition
    a2: Acquisition
    a3: Acquisition
    h_dd_m: float
    fringes: float
    span_days: float
    b_mean_days: float            # mean of (t2-t1, t3-t2)
    b_imbalance_days: float       # |(t2-t1) - (t3-t2)|

    def to_row(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "asc_desc": self.a1.asc_desc,
            "look_side": self.a1.look_side,
            "incidence_deg": self.a1.incidence_deg,
            "t1_utc": self.a1.ut_time.isoformat(),
            "t2_utc": self.a2.ut_time.isoformat(),
            "t3_utc": self.a3.ut_time.isoformat(),
            "tide_h1_m": self.a1.tide_h_m,
            "tide_h2_m": self.a2.tide_h_m,
            "tide_h3_m": self.a3.tide_h_m,
            "h_dd_m": self.h_dd_m,
            "fringes": self.fringes,
            "span_days": self.span_days,
            "b_mean_days": self.b_mean_days,
            "b_imbalance_days": self.b_imbalance_days,
        }


def group_by_track(
    acquisitions: Iterable[Acquisition],
) -> dict[str, list[Acquisition]]:
    streams: dict[str, list[Acquisition]] = defaultdict(list)
    for a in acquisitions:
        streams[a.track_id].append(a)
    for track in streams.values():
        track.sort(key=lambda a: a.ut_time)
    return dict(streams)


def enumerate_triplets(
    track: list[Acquisition],
    *,
    wavelength_m: float,
    max_span_days: float = 60.0,
    require_equal_baseline: bool = True,
    equal_baseline_tol_days: float = 0.5,
) -> list[Triplet]:
    """All valid (t1<t2<t3) triplets within a single track."""
    if len(track) < 3:
        return []
    out: list[Triplet] = []
    cos_theta = float(np.cos(np.deg2rad(track[0].incidence_deg)))
    half_lambda = wavelength_m / 2.0
    n = len(track)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a1, a2, a3 = track[i], track[j], track[k]
                span = (a3.ut_time - a1.ut_time).total_seconds() / 86400.0
                if span > max_span_days:
                    break
                b1 = (a2.ut_time - a1.ut_time).total_seconds() / 86400.0
                b2 = (a3.ut_time - a2.ut_time).total_seconds() / 86400.0
                if require_equal_baseline and abs(b1 - b2) > equal_baseline_tol_days:
                    continue
                h_dd = a1.tide_h_m - 2.0 * a2.tide_h_m + a3.tide_h_m
                fringes = abs(h_dd) * cos_theta / half_lambda
                out.append(Triplet(
                    track_id=a1.track_id, a1=a1, a2=a2, a3=a3,
                    h_dd_m=float(h_dd), fringes=float(fringes),
                    span_days=float(span),
                    b_mean_days=0.5 * (b1 + b2),
                    b_imbalance_days=abs(b1 - b2),
                ))
    return out


def rank_triplets(
    triplets: list[Triplet],
    objective: str = "max_fringes",
    n_top: int = 20,
    target_fringes: float | None = None,
) -> list[Triplet]:
    """Sort triplets by objective and return the top n."""
    if objective == "max_fringes":
        key = lambda t: -t.fringes
    elif objective == "min_fringes":
        key = lambda t: t.fringes
    elif objective.startswith("target") or target_fringes is not None:
        target = (target_fringes if target_fringes is not None
                  else float(objective.split("=", 1)[1]))
        key = lambda t: abs(t.fringes - target)
    else:
        msg = (f"Unknown objective {objective!r}. "
               "Use 'max_fringes', 'min_fringes', or 'target=K'.")
        raise ValueError(msg)
    return sorted(triplets, key=key)[:n_top]


def optimize(
    acquisitions: Iterable[Acquisition],
    *,
    wavelength_m: float,
    objective: str = "max_fringes",
    target_fringes: float | None = None,
    max_span_days: float = 60.0,
    require_equal_baseline: bool = True,
    equal_baseline_tol_days: float = 0.5,
    n_top: int = 20,
) -> pd.DataFrame:
    """One-shot: group by track, enumerate, rank globally, return DataFrame.

    The returned DataFrame ranks the top-N candidates across *all* tracks.
    Each row carries `track_id`, asc/desc, look_side, incidence, the three
    acquisition UTs, the three tide values, h_DD, fringe count, span, and
    baseline imbalance.
    """
    streams = group_by_track(acquisitions)
    all_triplets: list[Triplet] = []
    for track in streams.values():
        all_triplets.extend(enumerate_triplets(
            track,
            wavelength_m=wavelength_m,
            max_span_days=max_span_days,
            require_equal_baseline=require_equal_baseline,
            equal_baseline_tol_days=equal_baseline_tol_days,
        ))
    ranked = rank_triplets(
        all_triplets, objective=objective, n_top=n_top,
        target_fringes=target_fringes,
    )
    return pd.DataFrame([t.to_row() for t in ranked])


def best_per_track(
    acquisitions: Iterable[Acquisition],
    *,
    wavelength_m: float,
    objective: str = "max_fringes",
    target_fringes: float | None = None,
    max_span_days: float = 60.0,
    require_equal_baseline: bool = True,
    equal_baseline_tol_days: float = 0.5,
) -> pd.DataFrame:
    """One row per track_id: the best triplet on that track."""
    streams = group_by_track(acquisitions)
    rows: list[dict[str, object]] = []
    for track in streams.values():
        triplets = enumerate_triplets(
            track,
            wavelength_m=wavelength_m,
            max_span_days=max_span_days,
            require_equal_baseline=require_equal_baseline,
            equal_baseline_tol_days=equal_baseline_tol_days,
        )
        if not triplets:
            continue
        best = rank_triplets(triplets, objective=objective, n_top=1,
                             target_fringes=target_fringes)[0]
        row = best.to_row()
        row["n_acquisitions_in_track"] = len(track)
        row["n_triplets_in_track"] = len(triplets)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("fringes",
                             ascending=(objective == "min_fringes")).reset_index(drop=True)
    return df
