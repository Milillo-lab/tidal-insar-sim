"""Map — click to pick lat/lon or use a preset marker."""

from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium

from tidal_insar_sim.site import SITE_PRESETS
from tidal_insar_sim.web.state import (
    K_SITE_ICE_M,
    K_SITE_LAT,
    K_SITE_LON,
    K_SITE_NAME,
    K_SITE_PRESET,
    init_state,
    invalidate_computed,
)

st.set_page_config(page_title="Map", layout="wide")
init_state(st.session_state)

st.title("Site selection")
st.caption("Click anywhere on the map to pick a site, or choose a preset.")

col_preset, col_custom = st.columns([2, 1])

with col_preset:
    preset_options = ["Custom", *sorted(SITE_PRESETS)]
    choice = st.selectbox(
        "Preset",
        options=preset_options,
        index=preset_options.index(st.session_state[K_SITE_PRESET])
        if st.session_state[K_SITE_PRESET] in preset_options else 0,
        help="Antarctic GL presets. Pick 'Custom' to use arbitrary lat/lon.",
    )

    if choice != st.session_state[K_SITE_PRESET]:
        st.session_state[K_SITE_PRESET] = choice
        if choice != "Custom":
            site = SITE_PRESETS[choice]
            st.session_state[K_SITE_NAME] = site.name
            st.session_state[K_SITE_LAT] = float(site.lat)
            st.session_state[K_SITE_LON] = float(site.lon)
            st.session_state[K_SITE_ICE_M] = float(site.ice_thickness_m)
        invalidate_computed(st.session_state)
        st.rerun()

with col_custom:
    if st.button("Reset to Thwaites"):
        st.session_state[K_SITE_PRESET] = "THWAITES"
        site = SITE_PRESETS["THWAITES"]
        st.session_state[K_SITE_NAME] = site.name
        st.session_state[K_SITE_LAT] = float(site.lat)
        st.session_state[K_SITE_LON] = float(site.lon)
        st.session_state[K_SITE_ICE_M] = float(site.ice_thickness_m)
        invalidate_computed(st.session_state)
        st.rerun()

# Folium map centred over the selected lat/lon (Antarctic view by default).
m = folium.Map(
    location=[st.session_state[K_SITE_LAT], st.session_state[K_SITE_LON]],
    zoom_start=4,
    tiles="CartoDB positron",
)
for name, site in SITE_PRESETS.items():
    folium.CircleMarker(
        location=[site.lat, site.lon],
        radius=5,
        popup=f"{name}<br>H = {site.ice_thickness_m:.0f} m",
        color="#1f4e79",
        fill=True,
        fill_opacity=0.7,
    ).add_to(m)

# Highlight current selection
folium.Marker(
    [st.session_state[K_SITE_LAT], st.session_state[K_SITE_LON]],
    popup=st.session_state[K_SITE_NAME] or "current",
    icon=folium.Icon(color="red"),
).add_to(m)

result = st_folium(m, height=560, width=None, key="site-map")
if result and result.get("last_clicked"):
    clicked = result["last_clicked"]
    lat, lon = float(clicked["lat"]), float(clicked["lng"])
    if lat < 0:  # Antarctica only in v0.1
        st.session_state[K_SITE_PRESET] = "Custom"
        st.session_state[K_SITE_NAME] = f"site_{lat:+.2f}_{lon:+.2f}"
        st.session_state[K_SITE_LAT] = lat
        st.session_state[K_SITE_LON] = lon
        invalidate_computed(st.session_state)
        st.success(f"Set custom site at lat={lat:+.3f}, lon={lon:+.3f}")
    else:
        st.warning("v0.1 is Antarctic-only (lat < 0). Greenland deferred to v0.2.")

st.markdown(
    f"**Current:** `{st.session_state[K_SITE_NAME]}`  |  "
    f"lat={st.session_state[K_SITE_LAT]:+.3f}, lon={st.session_state[K_SITE_LON]:+.3f}  |  "
    f"ice H = {st.session_state[K_SITE_ICE_M]:.0f} m"
)
st.caption(
    "MEaSUREs GL v2 underlay is a roadmap item (`tidal-insar-sim setup-measures`); "
    "the analysis uses the coordinates above regardless of coastline display."
)
