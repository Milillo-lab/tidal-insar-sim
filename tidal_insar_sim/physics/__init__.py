"""Physics module: elastic plate flexure and DDInSAR fringe-count math.

All formulas are traced to peer-reviewed work in the individual submodule docstrings.
"""

from tidal_insar_sim.physics.ddinsar import dd_fringes_peak, dd_phase, double_difference_h
from tidal_insar_sim.physics.flexure import (
    flexural_parameter_beta,
    flexural_rigidity_D,
    flexure_profile,
    limit_of_flexure,
)
from tidal_insar_sim.physics.fringe import TripletSweep, rigid_triplet_sweep

__all__ = [
    "TripletSweep",
    "dd_fringes_peak",
    "dd_phase",
    "double_difference_h",
    "flexural_parameter_beta",
    "flexural_rigidity_D",
    "flexure_profile",
    "limit_of_flexure",
    "rigid_triplet_sweep",
]
