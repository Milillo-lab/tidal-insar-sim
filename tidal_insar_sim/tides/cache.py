"""SQLite cache for per-site CATS2008 harmonic constants.

Avoids re-running the pyTMD grid interpolation on every Simulator construction.
Keyed by (lat, lon rounded to 0.01 deg, model_version, constituent). The cache
is invalidated when `model_version` changes.

Storage
-------
Default path: `~/.tidal_insar_sim/site_cache.db`. Override with `db_path=`.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".tidal_insar_sim" / "site_cache.db"

LAT_LON_PRECISION_DEG = 0.01  # brief §3.3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS site_cache (
    lat_rounded      REAL    NOT NULL,
    lon_rounded      REAL    NOT NULL,
    model_version    TEXT    NOT NULL,
    constituent      TEXT    NOT NULL,
    amplitude_m      REAL    NOT NULL,
    phase_deg        REAL    NOT NULL,
    extracted_at     TEXT    NOT NULL,
    PRIMARY KEY (lat_rounded, lon_rounded, model_version, constituent)
);
"""


def _round(value: float, precision: float = LAT_LON_PRECISION_DEG) -> float:
    """Round to a fixed precision, e.g. 0.01 deg. `math.nan` pass-through."""
    if math.isnan(value):
        return float("nan")
    return round(value / precision) * precision


@dataclass
class TideCache:
    """SQLite-backed cache of `{lat, lon, model} -> {constituent: (amp, phase)}`."""

    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as con:
            con.executescript(_SCHEMA)
            con.commit()

    def get(
        self,
        lat: float,
        lon: float,
        model_version: str,
    ) -> dict[str, tuple[float, float]] | None:
        """Return cached `{name: (amp, phase)}` or None if not cached."""
        lat_r = _round(lat)
        lon_r = _round(lon)
        with closing(sqlite3.connect(self.db_path)) as con:
            rows = con.execute(
                "SELECT constituent, amplitude_m, phase_deg "
                "FROM site_cache "
                "WHERE lat_rounded = ? AND lon_rounded = ? AND model_version = ?",
                (lat_r, lon_r, model_version),
            ).fetchall()
        if not rows:
            return None
        return {str(name): (float(a), float(p)) for (name, a, p) in rows}

    def put(
        self,
        lat: float,
        lon: float,
        model_version: str,
        constants: dict[str, tuple[float, float]],
    ) -> None:
        lat_r = _round(lat)
        lon_r = _round(lon)
        rows: Iterable[tuple[float, float, str, str, float, float, str]] = [
            (lat_r, lon_r, model_version, name.upper(), amp, phase,
             _utc_now_iso())
            for name, (amp, phase) in constants.items()
        ]
        with closing(sqlite3.connect(self.db_path)) as con:
            con.executemany(
                "INSERT OR REPLACE INTO site_cache "
                "(lat_rounded, lon_rounded, model_version, constituent, "
                " amplitude_m, phase_deg, extracted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()

    def invalidate(self, model_version: str | None = None) -> int:
        """Delete cache entries. With `model_version=None`, drop everything."""
        with closing(sqlite3.connect(self.db_path)) as con:
            if model_version is None:
                cur = con.execute("DELETE FROM site_cache")
            else:
                cur = con.execute(
                    "DELETE FROM site_cache WHERE model_version = ?",
                    (model_version,),
                )
            con.commit()
            return int(cur.rowcount)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
