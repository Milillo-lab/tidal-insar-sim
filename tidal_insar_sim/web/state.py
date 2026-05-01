"""Session-state helpers for the Streamlit app.

Pure-Python — no `import streamlit` here, so these can be unit-tested without
an AppTest runtime. The `store` argument is any mapping-like object; in
production that's `st.session_state`, in tests it's a plain dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tidal_insar_sim.constellation import Constellation

from tidal_insar_sim.sensor import SENSOR_PRESETS, Sensor
from tidal_insar_sim.site import SITE_PRESETS, Site

# Streamlit's SessionStateProxy has overloaded signatures that defeat Protocol-
# based structural typing, so we fall back to `Any` at the function boundary.
# In practice this is a dict-like mapping indexed by string keys.
StoreLike = Any


# --- keys (string constants, so typos fail tests) ------------------------

K_SITE_PRESET = "tis.site_preset"
K_SITE_NAME   = "tis.site_name"
K_SITE_LAT    = "tis.site_lat"
K_SITE_LON    = "tis.site_lon"
K_SITE_ICE_M  = "tis.site_ice_thickness_m"
K_SENSOR_NAME = "tis.sensor_name"    # "X-BAND" | "C-BAND" | "L-BAND"
K_INCIDENCE   = "tis.incidence_deg"
K_CONSTELLATION = "tis.constellation"  # list[dict] — rows of the phase table
K_SWEEP_MODE  = "tis.sweep_mode"       # "rigid" | "multi_baseline" | "any_triplet"
K_SWEEP_STEP  = "tis.sweep_step_hours"
K_SWEEP_DAYS  = "tis.sweep_duration_days"
K_GAMMA_G     = "tis.gamma_grounded"
K_GAMMA_S     = "tis.gamma_shelf"
K_REPORT      = "tis.report"
K_MULTI_B_REPORTS = "tis.multi_b_reports"
K_ANY_TRIPLET = "tis.any_triplet"
K_PLAN_DF     = "tis.plan_df"
K_FMAP_STRONG = "tis.fmap_strong"
K_FMAP_NULL   = "tis.fmap_null"
K_BATCH_DF    = "tis.batch_df"


# --- defaults -----------------------------------------------------------


DEFAULTS: dict[str, Any] = {
    K_SITE_PRESET: "THWAITES",
    K_SITE_NAME:   "Thwaites_GL",
    K_SITE_LAT:    -75.00,
    K_SITE_LON:   -106.00,
    K_SITE_ICE_M:  450.0,
    K_SENSOR_NAME: "L-BAND",
    K_INCIDENCE:   39.0,
    # Default constellation: 1 satellite, 12-day repeat (NISAR-like)
    K_CONSTELLATION: [{"name": "Sat-1", "phase_offset_days": 0.0, "repeat_days": 12.0}],
    K_SWEEP_MODE:  "rigid",
    K_SWEEP_STEP:  1.0,
    K_SWEEP_DAYS:  29.53,
    K_GAMMA_G:     0.88,
    K_GAMMA_S:     0.60,
    K_REPORT:      None,
    K_MULTI_B_REPORTS: None,
    K_ANY_TRIPLET: None,
    K_PLAN_DF:     None,
    K_FMAP_STRONG: None,
    K_FMAP_NULL:   None,
    K_BATCH_DF:    None,
}


def init_state(store: StoreLike) -> None:
    """Populate missing keys with defaults. Idempotent."""
    for key, default in DEFAULTS.items():
        if key not in store:
            store[key] = default


# --- resolvers -----------------------------------------------------------


def resolve_site(store: StoreLike) -> Site:
    preset = store.get(K_SITE_PRESET)
    if preset and preset != "Custom":
        return SITE_PRESETS[preset]
    return Site.from_coords(
        lat=float(store[K_SITE_LAT]),
        lon=float(store[K_SITE_LON]),
        name=str(store.get(K_SITE_NAME) or "custom_site"),
        ice_thickness_m=float(store[K_SITE_ICE_M]),
    )


def resolve_sensor(store: StoreLike) -> Sensor:
    base = SENSOR_PRESETS[store[K_SENSOR_NAME]]
    # Allow user-overridden incidence angle.
    from dataclasses import replace
    incidence = float(store.get(K_INCIDENCE, base.incidence_deg))
    if incidence == base.incidence_deg:
        return base
    return replace(base, incidence_deg=incidence)


def resolve_constellation(store: StoreLike) -> Constellation:
    """Build a `Constellation` from the K_CONSTELLATION rows in store.

    Rows with `None` / missing `phase_offset_days` or `repeat_days` are
    silently skipped, so the page survives mid-edit states in the data
    editor (where a freshly added row hasn't been filled in yet).
    """
    from tidal_insar_sim.constellation import Constellation, Satellite

    rows = store.get(K_CONSTELLATION) or DEFAULTS[K_CONSTELLATION]
    sats: list[Satellite] = []
    for i, r in enumerate(rows):
        phase = r.get("phase_offset_days")
        repeat = r.get("repeat_days")
        if phase is None or repeat is None:
            continue
        try:
            sats.append(Satellite(
                name=str(r.get("name") or f"Sat-{i + 1}"),
                phase_offset_days=float(phase),
                repeat_days=float(repeat),
            ))
        except (TypeError, ValueError):
            continue
    if not sats:
        # Fall back to default so downstream computations don't blow up.
        return Constellation(satellites=(
            Satellite(name="Sat-1", phase_offset_days=0.0, repeat_days=12.0),
        ))
    return Constellation(satellites=tuple(sats))


def invalidate_computed(store: StoreLike) -> None:
    """Clear cached TripletReport / plan / fringe maps when inputs change."""
    for key in (K_REPORT, K_MULTI_B_REPORTS, K_ANY_TRIPLET, K_PLAN_DF,
                K_FMAP_STRONG, K_FMAP_NULL):
        store[key] = None


# --- summary helpers -----------------------------------------------------


@dataclass
class Snapshot:
    """Everything needed to identify a Simulator configuration in the UI."""

    sensor_name: str
    site_name: str
    site_lat: float
    site_lon: float
    ice_thickness_m: float
    step_hours: float
    duration_days: float
    gamma_grounded: float
    gamma_shelf: float

    @classmethod
    def from_store(cls, store: StoreLike) -> Snapshot:
        return cls(
            sensor_name=str(store[K_SENSOR_NAME]),
            site_name=str(store.get(K_SITE_NAME) or "Custom"),
            site_lat=float(store[K_SITE_LAT]),
            site_lon=float(store[K_SITE_LON]),
            ice_thickness_m=float(store[K_SITE_ICE_M]),
            step_hours=float(store[K_SWEEP_STEP]),
            duration_days=float(store[K_SWEEP_DAYS]),
            gamma_grounded=float(store[K_GAMMA_G]),
            gamma_shelf=float(store[K_GAMMA_S]),
        )
