"""Export — download buttons for all in-memory artifacts."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import streamlit as st

from tidal_insar_sim.io.geojson_export import report_to_geojson_feature
from tidal_insar_sim.io.kml_export import report_to_kml
from tidal_insar_sim.planner import to_ics
from tidal_insar_sim.web.state import (
    K_BATCH_DF,
    K_FMAP_NULL,
    K_FMAP_STRONG,
    K_PLAN_DF,
    K_REPORT,
    init_state,
)

st.set_page_config(page_title="Export", layout="wide")
init_state(st.session_state)

st.title("Export")

report = st.session_state[K_REPORT]
plan_df = st.session_state[K_PLAN_DF]
fm_strong = st.session_state[K_FMAP_STRONG]
fm_null = st.session_state[K_FMAP_NULL]
batch_df = st.session_state[K_BATCH_DF]

if all(x is None for x in (report, plan_df, fm_strong, fm_null, batch_df)):
    st.info("No artifacts yet. Run the Single Site or Batch page first.")
    st.stop()

# --- Report artifacts ---
if report is not None:
    st.subheader("Single-site report")
    rc1, rc2, rc3 = st.columns(3)

    # sweep CSV
    import pandas as pd
    sweep_df = pd.DataFrame({
        "delta_t_hours": report.delta_t_hours,
        "h1_m": report.sweep.h1_m,
        "h2_m": report.sweep.h2_m,
        "h3_m": report.sweep.h3_m,
        "h_dd_m": report.h_dd_m,
        "fringes": report.fringes,
    })
    sweep_df["site"] = report.site_name
    sweep_df["sensor"] = report.sensor_name
    rc1.download_button(
        "Sweep CSV", sweep_df.to_csv(index=False).encode(),
        file_name=f"{report.site_name}_{report.sensor_name}_sweep.csv",
        mime="text/csv",
    )

    # summary JSON
    summary_json = json.dumps(report.summary(), indent=2)
    rc1.download_button(
        "Summary JSON", summary_json.encode(),
        file_name=f"{report.site_name}_{report.sensor_name}_summary.json",
        mime="application/json",
    )

    # GeoJSON
    try:
        fc = report_to_geojson_feature(report)
        rc2.download_button(
            "GeoJSON", json.dumps(fc, indent=2).encode(),
            file_name=f"{report.site_name}.geojson",
            mime="application/geo+json",
        )
    except RuntimeError as exc:
        rc2.caption(f"GeoJSON unavailable: {exc}")

    # KML — write through a temp file (KML uses ElementTree.write which needs a path)
    try:
        with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmpf:
            tmp_path = Path(tmpf.name)
        report_to_kml(report, tmp_path)
        rc2.download_button(
            "KML", tmp_path.read_bytes(),
            file_name=f"{report.site_name}.kml",
            mime="application/vnd.google-earth.kml+xml",
        )
    except RuntimeError as exc:
        rc2.caption(f"KML unavailable: {exc}")

    # Fringe maps
    for col, fmap, label in [(rc3, fm_strong, "strong"), (rc3, fm_null, "null")]:
        if fmap is None:
            continue
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpf:
            tmp_path = Path(tmpf.name)
        fmap.to_geotiff(tmp_path)
        col.download_button(
            f"Fringe map ({label}) GeoTIFF",
            tmp_path.read_bytes(),
            file_name=f"{report.site_name}_{report.sensor_name}_{label}.tif",
            mime="image/tiff",
        )

# --- Planner artifacts ---
if plan_df is not None and not plan_df.empty:
    st.subheader("Acquisition plan")
    pc1, pc2 = st.columns(2)
    pc1.download_button(
        "Plan CSV", plan_df.to_csv(index=False).encode(),
        file_name="acquisition_plan.csv",
        mime="text/csv",
    )
    ics_buf = io.BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".ics", delete=False) as tmpf:
        tmp_ics_path = Path(tmpf.name)
    to_ics(plan_df, tmp_ics_path)
    pc2.download_button(
        "Plan ICS", tmp_ics_path.read_bytes(),
        file_name="acquisition_plan.ics",
        mime="text/calendar",
    )

# --- Batch artifacts ---
if batch_df is not None:
    st.subheader("Batch results")
    bc1, bc2 = st.columns(2)
    bc1.download_button(
        "Batch CSV", batch_df.to_csv(index=False).encode(),
        file_name="batch.csv",
        mime="text/csv",
    )
    parquet_buf = io.BytesIO()
    batch_df.to_parquet(parquet_buf, engine="pyarrow", index=False)
    bc2.download_button(
        "Batch Parquet", parquet_buf.getvalue(),
        file_name="batch.parquet",
        mime="application/octet-stream",
    )
