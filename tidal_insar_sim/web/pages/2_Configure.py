"""Configure — sensor, site overrides, sweep parameters, coherence."""

from __future__ import annotations

import streamlit as st

from tidal_insar_sim.sensor import SENSOR_PRESETS
from tidal_insar_sim.web.state import (
    K_GAMMA_G,
    K_GAMMA_S,
    K_SENSOR_NAME,
    K_SITE_ICE_M,
    K_SITE_LAT,
    K_SITE_LON,
    K_SITE_NAME,
    K_SITE_PRESET,
    K_SWEEP_DAYS,
    K_SWEEP_STEP,
    init_state,
    invalidate_computed,
)

st.set_page_config(page_title="Configure", layout="wide")
init_state(st.session_state)

st.title("Configure")

prev = {k: st.session_state[k] for k in (
    K_SENSOR_NAME, K_SITE_LAT, K_SITE_LON, K_SITE_NAME, K_SITE_ICE_M,
    K_SWEEP_STEP, K_SWEEP_DAYS, K_GAMMA_G, K_GAMMA_S,
)}

with st.form("configure-form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sensor")
        st.selectbox(
            "Preset",
            options=sorted(SENSOR_PRESETS),
            key=K_SENSOR_NAME,
            help="SAR sensor — sets wavelength, repeat, and incidence angle.",
        )
        sensor = SENSOR_PRESETS[st.session_state[K_SENSOR_NAME]]
        st.caption(
            f"lambda={sensor.wavelength_m*100:.1f} cm  |  "
            f"repeat={sensor.repeat_days:.0f} d  |  "
            f"theta={sensor.incidence_deg:.0f} deg"
        )

        st.subheader("Site")
        st.text_input("Name", key=K_SITE_NAME)
        c_lat, c_lon = st.columns(2)
        c_lat.number_input("Latitude", -90.0, 0.0, key=K_SITE_LAT, step=0.1, format="%.3f")
        c_lon.number_input("Longitude", -180.0, 180.0, key=K_SITE_LON, step=0.1, format="%.3f")
        st.number_input(
            "Ice thickness (m)", 100.0, 3500.0, step=50.0, key=K_SITE_ICE_M,
            help="Drives the flexural parameter beta. Thicker ice -> broader L_flex.",
        )

    with col2:
        st.subheader("Sweep")
        st.number_input(
            "Step (hours)", 0.5, 6.0, step=0.5, key=K_SWEEP_STEP,
            help="Triplet-start resolution. 1 h matches the prototype brief.",
        )
        st.number_input(
            "Duration (days)", 7.0, 60.0, step=0.5, key=K_SWEEP_DAYS,
            help="Sweep span; 29.53 = one synodic month, good for M2/K1.",
        )

        st.subheader("Coherence (synthetic fringe maps only)")
        st.slider("gamma_grounded", 0.4, 0.99, step=0.01, key=K_GAMMA_G)
        st.slider("gamma_shelf",    0.2, 0.95, step=0.01, key=K_GAMMA_S)

    apply = st.form_submit_button("Apply", type="primary")

if apply:
    # If user edited lat/lon/H manually, demote the preset to Custom so the
    # map page doesn't overwrite their values.
    changed_coords = any(
        st.session_state[k] != prev[k]
        for k in (K_SITE_LAT, K_SITE_LON, K_SITE_ICE_M, K_SITE_NAME)
    )
    if changed_coords:
        st.session_state[K_SITE_PRESET] = "Custom"
    invalidate_computed(st.session_state)
    st.success("Settings applied. Cached results cleared.")
