"""Single-site report — gauges, sweep curve, fringe maps, acquisition plan.

Reacts to the sweep mode picked on Setup:
- **rigid**: one curve + histogram + best/null fringe maps + planner.
- **multi_baseline**: one curve per valid constellation baseline, overlaid,
  plus a per-B summary table.
- **any_triplet**: scatter of fringes vs t1, colored by triplet span.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
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
    K_ANY_TRIPLET,
    K_FMAP_NULL,
    K_FMAP_STRONG,
    K_GAMMA_G,
    K_GAMMA_S,
    K_MULTI_B_REPORTS,
    K_PLAN_DF,
    K_REPORT,
    K_SWEEP_DAYS,
    K_SWEEP_MODE,
    K_SWEEP_STEP,
    init_state,
    resolve_constellation,
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
constellation = resolve_constellation(st.session_state)
mode = str(st.session_state[K_SWEEP_MODE])

st.markdown(
    f"**{site.name}** (lat={site.lat:+.3f}, lon={site.lon:+.3f}, H={site.ice_thickness_m:.0f} m)"
    f" · **{sensor.name}** · "
    f"**Constellation**: {constellation.n_satellites} sat(s), "
    f"B_eff={constellation.effective_repeat_days():.2f} d "
    f"· **Mode**: {mode}"
)

if st.button("Run", type="primary", disabled=not cats_installed):
    sim = Simulator(sensor=sensor, site=site, constellation=constellation)
    try:
        if mode == "rigid":
            report = sim.sweep_triplets(
                step_hours=float(st.session_state[K_SWEEP_STEP]),
                duration_days=float(st.session_state[K_SWEEP_DAYS]),
            )
            st.session_state[K_REPORT] = report
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
            st.session_state[K_MULTI_B_REPORTS] = None
            st.session_state[K_ANY_TRIPLET] = None
        elif mode == "multi_baseline":
            st.session_state[K_MULTI_B_REPORTS] = sim.multi_baseline_sweep(
                step_hours=float(st.session_state[K_SWEEP_STEP]),
                duration_days=float(st.session_state[K_SWEEP_DAYS]),
            )
            st.session_state[K_REPORT] = None
            st.session_state[K_ANY_TRIPLET] = None
        elif mode == "any_triplet":
            st.session_state[K_ANY_TRIPLET] = sim.any_triplet_sweep(
                duration_days=float(st.session_state[K_SWEEP_DAYS]),
            )
            st.session_state[K_REPORT] = None
            st.session_state[K_MULTI_B_REPORTS] = None
    except (TidalDataUnavailable, OutsideDomain, OnLand) as exc:
        st.error(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Render: rigid-B
# ---------------------------------------------------------------------------

if mode == "rigid":
    report = st.session_state[K_REPORT]
    if report is None:
        st.info("Click **Run** to compute the rigid-B sweep.")
        st.stop()

    summary = report.summary()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verdict", str(summary["verdict"]))
    m2.metric("P(>= 3 fringes) usable", f"{float(summary['P_usable_ge_3fr']):.1%}")
    m3.metric("P(>= 5 fringes) robust", f"{float(summary['P_robust_ge_5fr']):.1%}")
    m4.metric("P(null <0.5 fr)", f"{float(summary['P_null_lt_0p5fr']):.1%}")

    st.pyplot(plot_sweep_curve(report))
    c1, c2 = st.columns(2)
    c1.pyplot(plot_triplet_histogram(report))
    c2.pyplot(plot_tide_time_series(
        report.delta_t_hours, report.sweep.h1_m, report.h_dd_m, site_name=site.name,
    ))

    st.subheader("Synthetic fringe maps — best / worst triplets")
    fm_cols = st.columns(2)
    fm_strong = st.session_state[K_FMAP_STRONG]
    fm_null = st.session_state[K_FMAP_NULL]
    if fm_strong is not None:
        fm_cols[0].pyplot(plot_fringe_map(fm_strong, title="Best triplet (max fringes)"))
    if fm_null is not None:
        fm_cols[1].pyplot(plot_fringe_map(fm_null, title="Worst triplet (near-null)"))

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
            st.session_state[K_PLAN_DF] = report.recommended_triplets(
                start_date=start_dt, end_date=end_dt,
                n=int(top_n), min_fringes=float(min_fringes),
            )
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

    plan_df = st.session_state[K_PLAN_DF]
    if plan_df is not None and not plan_df.empty:
        st.dataframe(plan_df, width="stretch")
    elif plan_df is not None:
        st.warning("No triplets >= min_fringes in that window. Relax the threshold?")


# ---------------------------------------------------------------------------
# Render: multi-baseline
# ---------------------------------------------------------------------------

elif mode == "multi_baseline":
    sweeps = st.session_state[K_MULTI_B_REPORTS]
    if not sweeps:
        st.info("Click **Run** to compute one sweep per valid constellation baseline.")
        st.stop()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
    for B, sw in sorted(sweeps.items()):
        ax.plot(sw.delta_t_hours / 24.0, sw.fringes,
                label=f"B = {B:.1f} d", lw=1.0, alpha=0.9)
    ax.axhline(3.0, color="#2e8b57", ls="--", lw=0.6, alpha=0.5)
    ax.axhline(5.0, color="#b279a2", ls="-.", lw=0.6, alpha=0.5)
    ax.set_xlabel("Triplet start offset (days)")
    ax.set_ylabel("|fringes|")
    ax.set_title(f"Multi-baseline sweep — {site.name} / {sensor.name}")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    rows = [
        {
            "B_days": B,
            "P_usable_ge_3fr": float(np.mean(sw.fringes >= 3.0)),
            "P_robust_ge_5fr": float(np.mean(sw.fringes >= 5.0)),
            "P_null_lt_0p5fr": float(np.mean(sw.fringes < 0.5)),
            "fringe_mean": float(np.mean(sw.fringes)),
            "fringe_max": float(np.max(sw.fringes)),
        }
        for B, sw in sorted(sweeps.items())
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(cmap="RdYlGn",
                                      subset=["P_usable_ge_3fr", "P_robust_ge_5fr"])
               .format({"B_days": "{:.2f}",
                        "P_usable_ge_3fr": "{:.1%}",
                        "P_robust_ge_5fr": "{:.1%}",
                        "P_null_lt_0p5fr": "{:.1%}",
                        "fringe_mean": "{:.2f}",
                        "fringe_max": "{:.2f}"}),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Render: any-triplet
# ---------------------------------------------------------------------------

elif mode == "any_triplet":
    ats = st.session_state[K_ANY_TRIPLET]
    if ats is None:
        st.info("Click **Run** to enumerate every valid triplet from the constellation.")
        st.stop()

    import matplotlib.pyplot as plt

    spans = ats.t3_days - ats.t1_days
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=110)
    sc = ax.scatter(ats.t1_days, ats.fringes, c=spans, cmap="viridis",
                    s=14, alpha=0.7, edgecolors="none")
    ax.axhline(3.0, color="#2e8b57", ls="--", lw=0.6)
    ax.axhline(5.0, color="#b279a2", ls="-.", lw=0.6)
    ax.set_xlabel("t1 (days from epoch)")
    ax.set_ylabel("|fringes|")
    ax.set_title(f"Any-triplet fringes — {len(ats)} triplets")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("t3 - t1 (days)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    fringes = ats.fringes
    m1, m2, m3 = st.columns(3)
    m1.metric("# triplets", f"{len(ats)}")
    m2.metric("P(>= 3 fringes)", f"{float(np.mean(fringes >= 3.0)):.1%}")
    m3.metric("max fringes", f"{float(np.max(fringes)):.2f}")
