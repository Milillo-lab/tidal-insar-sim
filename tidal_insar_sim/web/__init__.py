"""Streamlit multi-page web app for tidal-insar-sim (M9).

Entry point: `streamlit run tidal_insar_sim/web/app.py`. The pages under
`web/pages/` are auto-discovered by Streamlit's file-based routing.

Pure helper logic lives in `web/state.py` and `web/components/`; keep the
page scripts thin so they can be re-rendered cheaply.
"""
