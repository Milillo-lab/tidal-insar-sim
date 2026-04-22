"""Tide backends.

In v0.1 the production backend is CATS2008 via pyTMD (Antarctic-only). That lives in
`tidal_insar_sim.tides.cats2008` (added at Checkpoint 2).

For Checkpoint 1 we ship only a simple mock tide synthesizer so the library, Simulator,
and tests can be wired up without a tide dataset on disk.
"""

from tidal_insar_sim.tides.mock import (
    MixedTide,
    mixed_m2_k1_tide,
    multi_constituent_tide,
)

__all__ = ["MixedTide", "mixed_m2_k1_tide", "multi_constituent_tide"]
