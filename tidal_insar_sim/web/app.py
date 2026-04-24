"""tidal-insar-sim Streamlit app entry point.

Run with:
    streamlit run tidal_insar_sim/web/app.py

The multi-page layout is provided by `pages/` — Streamlit auto-discovers them
in lexicographic order and adds them to the sidebar.
"""

from __future__ import annotations

import streamlit as st

from tidal_insar_sim import __version__
from tidal_insar_sim.web.state import (
    K_SITE_PRESET,
    Snapshot,
    init_state,
)

st.set_page_config(
    page_title="tidal-insar-sim",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state(st.session_state)

st.title("tidal-insar-sim")
st.caption(f"v{__version__}  |  DDInSAR fringe-count simulator for Antarctic grounding zones")

st.markdown(
    """
This tool predicts how many DDInSAR fringes a 3-date acquisition triplet will
produce at a given grounding-line site, under CATS2008 tidal forcing and
elastic-plate flexure. Built primarily for **NISAR** mission planning.

**Start here:**
1. Go to the :point_right: **Map** page and pick a site (or use a preset).
2. Adjust sensor / ice thickness on **Configure**.
3. Run the analysis on **Single Site** — probabilities, curves, synthetic fringe maps, and
   a top-N acquisition plan with confidence.
4. Use **Batch** to compare sensors and sites side-by-side, and **Export** to download
   GeoJSON / KML / GeoTIFF / CSV / ICS.
"""
)

snap = Snapshot.from_store(st.session_state)
with st.sidebar:
    st.subheader("Current selection")
    st.write(f"**Sensor:** {snap.sensor_name}")
    st.write(f"**Site preset:** {st.session_state[K_SITE_PRESET]}")
    st.write(f"**lat/lon:** {snap.site_lat:+.3f}, {snap.site_lon:+.3f}")
    st.write(f"**Ice H:** {snap.ice_thickness_m:.0f} m")
    st.write(f"**Sweep:** {snap.duration_days:.2f} d @ {snap.step_hours:.1f} h")
    st.divider()
    st.caption("Change on the Configure page. All computed results clear when inputs change.")

st.info(
    "CATS2008 must be installed locally (once) before tide computation can run. "
    "See `tidal-insar-sim setup-cats --help` at the terminal.",
    icon=":material/info:",
)
