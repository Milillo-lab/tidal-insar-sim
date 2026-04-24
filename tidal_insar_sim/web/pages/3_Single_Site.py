"""Single-site report — probabilities, sweep curve, histogram, fringe maps, planner."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from tidal_insar_sim.simulator import Simulator
from tidal_insar_sim.tides.cats2008 import CATS_FILES, CATS_SUBDIR, DEFAULT_DATA_DIR
from tidal_insar_sim.tides.errors import OnLand, OutsideDomain, TidalDataUnavailable
from tidal_insar_sim.web.components.plots import (
    plot_fringe_map,
    plot_sweep_curve,
    plot_tide_time_series,
    plot_triplet_histogram,
)
from tidal_insar_sim.web.state import (
    K_FMAP_NULL,
    K_FMAP_STRONG,
    K_GAMMA_G,
    K_GAMMA_S,
    K_PLAN_DF,
    K_REPORT,
    K_SWEEP_DAYS,
    K_SWEEP_STEP,
    init_state,
    resolve_sensor,
    resolve_site,
)

st.set_page_config(page_title="Single Site", layout="wide")
init_state(st.session_state)

st.title("Single-site report")

cats_installed = all((DEFAULT_DATA_DIR / CATS_SUBDIR / f).exists() for f in CATS_FILES)
if not cats_installed:
    st.warning(
        "CATS2008 is not installed. Run `tidal-insar-sim setup-cats --path CATS2008.zip` "
        "before clicking Run.",
        icon=":material/warning:",
    )

site = resolve_site(st.session_state)
sensor = resolve_sensor(st.session_state)
st.markdown(
    f"**{site.name}** (lat={site.lat:+.3f}, lon={site.lon:+.3f}, H={site.ice_thickness_m:.0f} m)  "
    f"|  **{sensor.name}** (lambda={sensor.wavelength_m*100:.1f} cm, "
    f"repeat={sensor.repeat_days:.0f} d)"
)

if st.button("Run sweep", type="primary", disabled=not cats_installed):
    sim = Simulator(sensor=sensor, site=site)
    try:
        report = sim.sweep_triplets(
            step_hours=float(st.session_state[K_SWEEP_STEP]),
            duration_days=float(st.session_state[K_SWEEP_DAYS]),
        )
    except (TidalDataUnavailable, OutsideDomain, OnLand) as exc:
        st.error(f"{type(exc).__name__}: {exc}")
    else:
        st.session_state[K_REPORT] = report
        # Generate strong + null fringe maps for illustration
        st.session_state[K_FMAP_STRONG] = sim.synthesize_ddinsar(
            triplet_start_hours=report.best_triplet_h(),
            gamma_grounded=float(st.session_state[K_GAMMA_G]),
            gamma_shelf=float(st.session_state[K_GAMMA_S]),
        )
        st.session_state[K_FMAP_NULL] = sim.synthesize_ddinsar(
            triplet_start_hours=report.worst_triplet_h(),
            gamma_grounded=float(st.session_state[K_GAMMA_G]),
            gamma_shelf=float(st.session_state[K_GAMMA_S]),
        )

report = st.session_state[K_REPORT]
if report is None:
    st.info("Click **Run sweep** to compute the report.")
    st.stop()

summary = report.summary()

# --- gauges ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Verdict", str(summary["verdict"]))
m2.metric("P(>= 3 fringes) usable", f"{float(summary['P_usable_ge_3fr']):.1%}")
m3.metric("P(>= 5 fringes) robust", f"{float(summary['P_robust_ge_5fr']):.1%}")
m4.metric("P(null <0.5 fr)",        f"{float(summary['P_null_lt_0p5fr']):.1%}")

# --- sweep curve + histogram ---
st.pyplot(plot_sweep_curve(report))
c1, c2 = st.columns(2)
c1.pyplot(plot_triplet_histogram(report))
c2.pyplot(plot_tide_time_series(
    report.delta_t_hours, report.sweep.h1_m, report.h_dd_m, site_name=site.name,
))

# --- fringe maps ---
st.subheader("Synthetic fringe maps — best / worst triplets")
fm_cols = st.columns(2)
fm_strong = st.session_state[K_FMAP_STRONG]
fm_null = st.session_state[K_FMAP_NULL]
if fm_strong is not None:
    fm_cols[0].pyplot(plot_fringe_map(fm_strong, title="Best triplet (max fringes)"))
if fm_null is not None:
    fm_cols[1].pyplot(plot_fringe_map(fm_null, title="Worst triplet (near-null)"))

# --- planner ---
st.subheader("Acquisition planner")
p_cols = st.columns([1, 1, 1, 2])
start_d = p_cols[0].date_input("Start", value=datetime(2026, 7, 1))
end_d = p_cols[1].date_input("End", value=datetime(2026, 9, 1))
top_n = p_cols[2].number_input("Top N", 1, 30, value=10, step=1)
min_fringes = p_cols[3].slider("Min fringes", 0.5, 10.0, 3.0, step=0.5)

if st.button("Run planner"):
    start_dt = datetime.combine(start_d, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_d, datetime.min.time(), tzinfo=timezone.utc)
    try:
        plan = report.recommended_triplets(
            start_date=start_dt, end_date=end_dt,
            n=int(top_n), min_fringes=float(min_fringes),
        )
    except Exception as exc:  # broad: surface the error to the user cleanly
        st.error(f"{type(exc).__name__}: {exc}")
    else:
        st.session_state[K_PLAN_DF] = plan

plan_df = st.session_state[K_PLAN_DF]
if plan_df is not None and not plan_df.empty:
    st.dataframe(plan_df, width="stretch")
elif plan_df is not None and plan_df.empty:
    st.warning("No triplets >= min_fringes in that window. Relax the threshold?")
