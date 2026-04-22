"""Tide backends.

- `mock.py`: deterministic M2+K1 and multi-constituent synthesizers (no data dep).
- `cats2008.py`: CATS2008 via pyTMD (requires a local install of CATS2008 data).
- `synthesis.py`: harmonic synthesis / analysis (no nodal corrections).
- `cache.py`: SQLite cache for per-site harmonic constants.
- `download.py`: `setup-cats` verify/extract workflow.
- `errors.py`: TidalDataUnavailable, OutsideDomain, OnLand (fail-loudly contract).
"""

from tidal_insar_sim.tides.cats2008 import (
    DEFAULT_DATA_DIR,
    EPOCH_2000,
    CATSBackend,
    make_cats_tide_fn,
)
from tidal_insar_sim.tides.errors import OnLand, OutsideDomain, TidalDataUnavailable
from tidal_insar_sim.tides.mock import (
    CONSTITUENT_PERIODS_H,
    MixedTide,
    mixed_m2_k1_tide,
    multi_constituent_tide,
)
from tidal_insar_sim.tides.synthesis import (
    analyze_harmonic,
    synthesize_from_constants,
)

__all__ = [
    "CONSTITUENT_PERIODS_H",
    "DEFAULT_DATA_DIR",
    "EPOCH_2000",
    "CATSBackend",
    "MixedTide",
    "OnLand",
    "OutsideDomain",
    "TidalDataUnavailable",
    "analyze_harmonic",
    "make_cats_tide_fn",
    "mixed_m2_k1_tide",
    "multi_constituent_tide",
    "synthesize_from_constants",
]
