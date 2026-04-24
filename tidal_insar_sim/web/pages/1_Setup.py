"""Setup — Antarctic-projection map with preset glaciers + full config form.

Merges the former Map and Configure pages. Click anywhere on the polar
stereographic map to pick a site; adjust sensor band, constellation (as a
per-satellite phase/repeat table), sweep params, and coherence.
"""

from __future__ import annotations

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from tidal_insar_sim.site import SITE_PRESETS, Site
from tidal_insar_sim.web.state import (
    K_CONSTELLATION,
    K_GAMMA_G,
    K_GAMMA_S,
    K_INCIDENCE,
    K_SENSOR_NAME,
    K_SITE_ICE_M,
    K_SITE_LAT,
    K_SITE_LON,
    K_SITE_NAME,
    K_SITE_PRESET,
    K_SWEEP_DAYS,
    K_SWEEP_MODE,
    K_SWEEP_STEP,
    init_state,
    invalidate_computed,
    resolve_constellation,
)

st.set_page_config(page_title="Setup", layout="wide")
init_state(st.session_state)

st.title("Setup — site, sensor, constellation")

# ---------------------------------------------------------------------------
# Map (Antarctic polar stereographic, EPSG:3031)
# ---------------------------------------------------------------------------

st.subheader("Site selection")
st.caption("Click anywhere on the map (lat < 0) to pick a custom site, or use a preset below.")

# Folium with EPSG:3031 polar stereographic projection.
# NASA GIBS Blue Marble serves tiles in this projection.
m = folium.Map(
    location=[-90, 0],
    zoom_start=1,
    crs="EPSG3031",
    tiles=None,
    max_zoom=5,
)
folium.TileLayer(
    tiles=(
        "https://gibs.earthdata.nasa.gov/wmts/epsg3031/best/"
        "BlueMarble_NextGeneration/default/500m/{z}/{y}/{x}.jpeg"
    ),
    attr="NASA GIBS | Blue Marble Next Generation",
    name="Blue Marble (EPSG:3031)",
    max_zoom=5,
).add_to(m)

# Preset glacier markers (all Antarctic sites).
for name, site in SITE_PRESETS.items():
    folium.CircleMarker(
        location=[site.lat, site.lon],
        radius=6,
        popup=(
            f"<b>{name}</b><br>lat={site.lat:.2f}, lon={site.lon:.2f}<br>"
            f"H = {site.ice_thickness_m:.0f} m"
        ),
        tooltip=name,
        color="#ffd200",
        weight=2,
        fill=True,
        fill_color="#ffd200",
        fill_opacity=0.85,
    ).add_to(m)

# Highlight the current selection.
folium.Marker(
    location=[st.session_state[K_SITE_LAT], st.session_state[K_SITE_LON]],
    tooltip=st.session_state[K_SITE_NAME] or "current",
    icon=folium.Icon(color="red", icon="info-sign"),
).add_to(m)

click = st_folium(m, height=560, width=None, key="site-map-3031",
                   returned_objects=["last_clicked"])

# ---------------------------------------------------------------------------
# Click handler — show a sub-form for name + ice thickness
# ---------------------------------------------------------------------------

if click and click.get("last_clicked"):
    last = click["last_clicked"]
    clicked_lat = float(last["lat"])
    clicked_lon = float(last["lng"])
    if clicked_lat >= 0:
        st.warning("v0.1 is Antarctic-only (lat < 0). Greenland deferred to v0.2.")
    else:
        with st.form("custom-site-form", clear_on_submit=False):
            st.markdown(
                f"**Custom site at** `lat={clicked_lat:+.3f}, lon={clicked_lon:+.3f}`"
            )
            c1, c2 = st.columns(2)
            new_name = c1.text_input("Site name",
                                     value=f"site_{clicked_lat:+.2f}_{clicked_lon:+.2f}")
            new_ice = c2.number_input(
                "Ice thickness (m)", 100.0, 3500.0, step=50.0, value=500.0,
                help=(
                    "H drives the flexural parameter beta = (rho_w*g / 4D)^(1/4). "
                    "Thicker ice => stiffer plate => wider L_flex (fringes spread over "
                    "more kilometres). Typical values: Thwaites ~450 m, Rutford ~2000 m."
                ),
            )
            if st.form_submit_button("Use this site", type="primary"):
                st.session_state[K_SITE_PRESET] = "Custom"
                st.session_state[K_SITE_NAME] = new_name
                st.session_state[K_SITE_LAT] = clicked_lat
                st.session_state[K_SITE_LON] = clicked_lon
                st.session_state[K_SITE_ICE_M] = float(new_ice)
                invalidate_computed(st.session_state)
                st.rerun()

# Preset selector as a fallback / override.
preset_options = ["Custom", *sorted(SITE_PRESETS)]
chosen = st.selectbox(
    "Or pick a preset:",
    options=preset_options,
    index=preset_options.index(st.session_state[K_SITE_PRESET])
    if st.session_state[K_SITE_PRESET] in preset_options else 0,
)
if chosen != st.session_state[K_SITE_PRESET]:
    st.session_state[K_SITE_PRESET] = chosen
    if chosen != "Custom":
        s = SITE_PRESETS[chosen]
        st.session_state[K_SITE_NAME] = s.name
        st.session_state[K_SITE_LAT] = float(s.lat)
        st.session_state[K_SITE_LON] = float(s.lon)
        st.session_state[K_SITE_ICE_M] = float(s.ice_thickness_m)
    invalidate_computed(st.session_state)
    st.rerun()

# Show the resolved Site's L_flex (physics-driven readout).
site = Site.from_coords(
    lat=float(st.session_state[K_SITE_LAT]),
    lon=float(st.session_state[K_SITE_LON]),
    name=str(st.session_state[K_SITE_NAME] or "current"),
    ice_thickness_m=float(st.session_state[K_SITE_ICE_M]),
)
l_flex_m = site.limit_of_flexure_m()
beta = site.flexural_parameter_beta()

current_col1, current_col2, current_col3 = st.columns(3)
current_col1.metric("Current site", st.session_state[K_SITE_NAME] or "-")
current_col2.metric("Ice thickness", f"{float(st.session_state[K_SITE_ICE_M]):.0f} m")
current_col3.metric(
    "L_flex", f"{l_flex_m:.0f} m",
    help=f"Flexure limit = pi/beta, with beta = {beta:.3e} /m.",
)

st.divider()

# ---------------------------------------------------------------------------
# Sensor band
# ---------------------------------------------------------------------------

st.subheader("Sensor band")
band_cols = st.columns([1, 1])
with band_cols[0]:
    band = st.radio(
        "Band",
        options=["X-BAND", "C-BAND", "L-BAND"],
        index=["X-BAND", "C-BAND", "L-BAND"].index(
            st.session_state[K_SENSOR_NAME]
        ) if st.session_state[K_SENSOR_NAME] in ["X-BAND", "C-BAND", "L-BAND"] else 2,
        format_func=lambda x: {
            "X-BAND": "X-band (~3.1 cm)",
            "C-BAND": "C-band (~5.6 cm)",
            "L-BAND": "L-band (~23.6 cm)",
        }[x],
        horizontal=True,
    )
    if band != st.session_state[K_SENSOR_NAME]:
        st.session_state[K_SENSOR_NAME] = band
        invalidate_computed(st.session_state)

with band_cols[1]:
    new_theta = st.number_input(
        "Incidence angle (deg)", 15.0, 60.0, step=1.0,
        value=float(st.session_state[K_INCIDENCE]),
    )
    if new_theta != st.session_state[K_INCIDENCE]:
        st.session_state[K_INCIDENCE] = float(new_theta)
        invalidate_computed(st.session_state)

st.divider()

# ---------------------------------------------------------------------------
# Constellation — per-satellite phase/repeat table
# ---------------------------------------------------------------------------

st.subheader("Constellation")
st.caption(
    "Each row is one satellite. `phase_offset_days` is the days into orbit at t=0. "
    "`repeat_days` is the orbit repeat interval. The effective triplet baseline is the "
    "smallest inter-acquisition gap in the schedule."
)

# Preset factory buttons
p_col1, p_col2, p_col3, p_col4 = st.columns(4)
def _apply_preset(rows: list[dict[str, object]]) -> None:
    st.session_state[K_CONSTELLATION] = rows
    invalidate_computed(st.session_state)
    st.rerun()

if p_col1.button("1 sat / 12 d (NISAR)"):
    _apply_preset([{"name": "NISAR", "phase_offset_days": 0.0, "repeat_days": 12.0}])
if p_col2.button("2 sats equal / 12 d (S1 A+B)"):
    _apply_preset([
        {"name": "S1A", "phase_offset_days": 0.0, "repeat_days": 12.0},
        {"name": "S1B", "phase_offset_days": 6.0, "repeat_days": 12.0},
    ])
if p_col3.button("3 sats equal / 12 d (RCM)"):
    _apply_preset([
        {"name": "RCM-1", "phase_offset_days": 0.0, "repeat_days": 12.0},
        {"name": "RCM-2", "phase_offset_days": 4.0, "repeat_days": 12.0},
        {"name": "RCM-3", "phase_offset_days": 8.0, "repeat_days": 12.0},
    ])
if p_col4.button("4 sats equal / 14 d (ALOS-4 like)"):
    _apply_preset([
        {"name": f"ALOS-{i+1}", "phase_offset_days": i*14.0/4.0, "repeat_days": 14.0}
        for i in range(4)
    ])

df_const = pd.DataFrame(st.session_state[K_CONSTELLATION])
edited = st.data_editor(
    df_const,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "name": st.column_config.TextColumn("Name", required=True),
        "phase_offset_days": st.column_config.NumberColumn(
            "Phase offset (days)", min_value=0.0, step=0.5, format="%.2f"
        ),
        "repeat_days": st.column_config.NumberColumn(
            "Orbit repeat (days)", min_value=0.1, step=0.5, format="%.2f"
        ),
    },
    width="stretch",
    key="constellation-editor",
)

# Apply edits back to session state.
new_rows = edited.to_dict(orient="records")
if new_rows and new_rows != st.session_state[K_CONSTELLATION]:
    st.session_state[K_CONSTELLATION] = new_rows
    invalidate_computed(st.session_state)

# Derived: effective baseline + acquisition count over 30 d.
try:
    c = resolve_constellation(st.session_state)
    eff_B = c.effective_repeat_days()
    acq = c.acquisition_times(window_days=30.0)
    valid_B = c.valid_baselines_days(window_days=60.0)[:6]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("# satellites", f"{c.n_satellites}")
    mc2.metric("Effective B", f"{eff_B:.2f} d",
               help="Smallest inter-acquisition gap — the default rigid-triplet baseline.")
    mc3.metric("Acq / 30 d", f"{acq.size}")
    st.caption(f"Valid DDInSAR baselines (short list): {[f'{b:.1f} d' for b in valid_B]}")
except Exception as exc:
    st.error(f"Invalid constellation: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Sweep + coherence + mode
# ---------------------------------------------------------------------------

st.subheader("Sweep & coherence")

s1, s2, s3 = st.columns(3)
new_mode = s1.radio(
    "Sweep mode",
    options=["rigid", "multi_baseline", "any_triplet"],
    index=["rigid", "multi_baseline", "any_triplet"].index(
        st.session_state[K_SWEEP_MODE]
    ),
    help=(
        "rigid: single B = effective baseline (classical DDInSAR). "
        "multi_baseline: one sweep per valid B. "
        "any_triplet: every (t1,t2,t3) from the schedule."
    ),
    format_func={
        "rigid": "Rigid-B (classical)",
        "multi_baseline": "Multi-baseline",
        "any_triplet": "Any 3 from schedule",
    }.get,
)
if new_mode != st.session_state[K_SWEEP_MODE]:
    st.session_state[K_SWEEP_MODE] = new_mode
    invalidate_computed(st.session_state)

new_step = s2.number_input(
    "Step (hours)", 0.5, 6.0, step=0.5, value=float(st.session_state[K_SWEEP_STEP]),
)
if new_step != st.session_state[K_SWEEP_STEP]:
    st.session_state[K_SWEEP_STEP] = float(new_step)
    invalidate_computed(st.session_state)

new_days = s3.number_input(
    "Duration (days)", 7.0, 90.0, step=0.5, value=float(st.session_state[K_SWEEP_DAYS]),
)
if new_days != st.session_state[K_SWEEP_DAYS]:
    st.session_state[K_SWEEP_DAYS] = float(new_days)
    invalidate_computed(st.session_state)

cc1, cc2 = st.columns(2)
new_gg = cc1.slider("gamma_grounded (coherence)", 0.4, 0.99, step=0.01,
                    value=float(st.session_state[K_GAMMA_G]))
if new_gg != st.session_state[K_GAMMA_G]:
    st.session_state[K_GAMMA_G] = float(new_gg)
    invalidate_computed(st.session_state)

new_gs = cc2.slider("gamma_shelf (coherence)", 0.2, 0.95, step=0.01,
                    value=float(st.session_state[K_GAMMA_S]))
if new_gs != st.session_state[K_GAMMA_S]:
    st.session_state[K_GAMMA_S] = float(new_gs)
    invalidate_computed(st.session_state)

st.success("All settings apply on-the-fly. Switch to the **Single Site** page to run the sweep.")
