"""Batch — multi-site, multi-sensor comparison."""

from __future__ import annotations

import streamlit as st

from tidal_insar_sim.batch import batch_compare, pivot_heatmap
from tidal_insar_sim.sensor import SENSOR_PRESETS
from tidal_insar_sim.site import SITE_PRESETS
from tidal_insar_sim.tides.cats2008 import CATS_FILES, CATS_SUBDIR, DEFAULT_DATA_DIR
from tidal_insar_sim.web.state import K_BATCH_DF, init_state

st.set_page_config(page_title="Batch", layout="wide")
init_state(st.session_state)

st.title("Batch comparison")

cats_installed = all((DEFAULT_DATA_DIR / CATS_SUBDIR / f).exists() for f in CATS_FILES)
if not cats_installed:
    st.warning("CATS2008 not installed — batch uses CATS for tides.",
               icon=":material/warning:")

col_a, col_b = st.columns(2)
selected_sites = col_a.multiselect(
    "Sites", options=sorted(SITE_PRESETS), default=["THWAITES", "RUTFORD"],
)
selected_sensors = col_b.multiselect(
    "Sensors", options=["X-BAND", "C-BAND", "L-BAND"],
    default=["C-BAND", "L-BAND"],
)
metric = st.selectbox(
    "Heatmap metric",
    options=["P_usable_ge_3fr", "P_robust_ge_5fr", "P_null_lt_0p5fr",
             "fringe_mean", "fringe_median", "fringe_max"],
    index=0,
)

if st.button("Run batch", type="primary",
             disabled=not (cats_installed and selected_sites and selected_sensors)):
    sites = [SITE_PRESETS[n] for n in selected_sites]
    sensors = [SENSOR_PRESETS[n] for n in selected_sensors]
    with st.spinner("Running sweeps..."):
        df = batch_compare(sites, sensors)
    st.session_state[K_BATCH_DF] = df

df = st.session_state[K_BATCH_DF]
if df is None:
    st.info("Pick sites and sensors, then click **Run batch**.")
    st.stop()

st.subheader(f"Heatmap — {metric}")
heatmap = pivot_heatmap(df, metric=metric)
st.dataframe(
    heatmap.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.3f}"),
    width="stretch",
)

st.subheader("Full table")
display_cols = [
    "site", "sensor", "verdict", "P_usable_ge_3fr", "P_robust_ge_5fr",
    "P_null_lt_0p5fr", "fringe_mean", "fringe_median", "fringe_max",
]
st.dataframe(df[display_cols], width="stretch")
