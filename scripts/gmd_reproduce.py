"""Reproduce every number, table and figure of the GMD paper from one triplet set.

    python scripts/gmd_reproduce.py --snapshot data/rop_snapshot/ase_bbox_2026-10-02.geojson \
        --out gmd_outputs [--gl-gpkg InSAR_GL_Antarctica_v02.1.gpkg]

Pass ``--live`` instead of ``--snapshot`` to query the ROP feature service and
save a fresh snapshot next to the outputs.

Every table and figure below reads ``all_triplets.csv`` and nothing else, so
the figures and the tables cannot disagree about which triplets exist.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tidal_insar_sim import rop
from tidal_insar_sim.optimizer import enumerate_triplets, group_by_track
from tidal_insar_sim.physics.flexure import flexural_parameter_beta
from tidal_insar_sim.site import Site
from tidal_insar_sim.tides.cats2008 import CATSBackend

ASE_BBOX = (-130.0, -78.0, -85.0, -71.0)
SITES = {"Thwaites": Site.THWAITES, "Pine Island": Site.PIG, "Smith": Site.SMITH}
COLORS = {"Thwaites": "#1f9bb4", "Pine Island": "#d98b1a", "Smith": "#7b4fa6"}
LAMBDA_M = 0.238
B_MIN, B_MAX, B_TOL, SPAN_MAX = 11.5, 12.5, 0.5, 30.0
CLASSES = [(0, 2, "unusable"), (2, 3, "marginal"), (3, 5, "operational"), (5, np.inf, "excellent")]
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def build_triplets(scenes, start, end, cats):
    rows = []
    for name, site in SITES.items():
        acqs = rop.site_acquisitions(scenes, site.lat, site.lon, start, end)
        times = sorted({a.ut_time for a in acqs})
        secs = np.array([(t - EPOCH).total_seconds() for t in times])
        h = dict(zip(times, cats.predict_tide_m(site.lat, site.lon, secs)))
        acqs = [type(a)(**{**a.__dict__, "tide_h_m": float(h[a.ut_time])}) for a in acqs]
        for track_id, stream in group_by_track(acqs).items():
            for t in enumerate_triplets(stream, wavelength_m=LAMBDA_M, max_span_days=SPAN_MAX,
                                        equal_baseline_tol_days=B_TOL):
                if not (B_MIN <= t.b_mean_days <= B_MAX):
                    continue
                r = t.to_row()
                r["glacier"] = name
                r["track"] = int(track_id.split(":")[0])
                r["l_mode"] = track_id.split(":")[1]
                rows.append(r)
    df = pd.DataFrame(rows)
    for c in ("t1_utc", "t2_utc", "t3_utc"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["mode"] = np.where(df.h_dd_m < 0, "high-at-t2", "low-at-t2")
    df["n_fr_class"] = pd.cut(df.fringes, [c[0] for c in CLASSES] + [np.inf], right=False,
                              labels=[c[2] for c in CLASSES])
    return df


def yield_table(df):
    rows = []
    for name, g in list(df.groupby("glacier", sort=False)) + [("ASE total", df)]:
        n = len(g)
        rows.append({
            "Glacier": name,
            "Tracks": g.track.nunique() if name != "ASE total" else "",
            "Triplets": n,
            "n_fr>=2": int((g.fringes >= 2).sum()),
            "n_fr>=3": int((g.fringes >= 3).sum()),
            "n_fr>=5": int((g.fringes >= 5).sum()),
            "pct>=3": 100 * (g.fringes >= 3).mean(),
            "pct>=5": 100 * (g.fringes >= 5).mean(),
            "Mean": g.fringes.mean(),
            "Max": g.fringes.max(),
        })
    return pd.DataFrame(rows)


def culling(df, fractions=(0.25, 0.5), n_mc=2000, seed=0):
    """Usable triplets retained when a fraction of the triplets is dropped.

    Random culling is Monte Carlo over uniform draws without replacement;
    ranked culling drops the lowest predicted fringe counts first.
    """
    rng = np.random.default_rng(seed)
    usable = (df.fringes >= 3).to_numpy()
    n = len(df)
    out = []
    for f in fractions:
        keep = n - int(round(f * n))
        mc = [usable[rng.choice(n, keep, replace=False)].sum() for _ in range(n_mc)]
        ranked = np.sort(df.fringes.to_numpy())[::-1][:keep]
        out.append({"cull_fraction": f, "kept": keep, "usable_total": int(usable.sum()),
                    "random_mean_retained": float(np.mean(mc)),
                    "random_p05": float(np.percentile(mc, 5)),
                    "random_p95": float(np.percentile(mc, 95)),
                    "ranked_retained": int((ranked >= 3).sum())})
    max_shed = 1 - usable.sum() / n
    return pd.DataFrame(out), max_shed


def fig_footprints(scenes, df, out, gl_gpkg=None):
    import pyproj
    tr = pyproj.Transformer.from_crs(4326, 3031, always_xy=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    best = df.groupby(["glacier", "track"]).fringes.max()
    cmap = plt.get_cmap("viridis")
    vmax = max(5.0, float(df.fringes.max()))
    if gl_gpkg:
        import geopandas as gpd
        gl = gpd.read_file(gl_gpkg).to_crs(3031)
        gl.plot(ax=ax, color="k", lw=0.6, zorder=3)
    for name, site in SITES.items():
        x, y = tr.transform(site.lon, site.lat)
        for s in scenes:
            poly = rop._polar(s.geometry)
            if not poly.contains(__import__("shapely.geometry", fromlist=["Point"]).Point(x, y)):
                continue
            v = best.get((name, s.track))
            if v is None:
                continue
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color=cmap(v / vmax), alpha=0.12, lw=0)
            ax.plot(xs, ys, color=COLORS[name], lw=0.6, alpha=0.6)
        ax.plot(x, y, "o", ms=9, mfc=COLORS[name], mec="k", zorder=5, label=name)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax, shrink=0.7, label="Maximum predicted $n_{fr}$ on track (12-day rigid triplet)")
    pts = [tr.transform(s.lon, s.lat) for s in SITES.values()]
    cx, cy = np.mean(pts, axis=0)
    ax.set_xlim(cx - 450e3, cx + 450e3)
    ax.set_ylim(cy - 400e3, cy + 400e3)
    ax.set_aspect("equal")
    ax.set_xlabel("Polar stereographic x (m, EPSG:3031)")
    ax.set_ylabel("Polar stereographic y (m, EPSG:3031)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig2_footprints.png", dpi=200)
    plt.close(fig)


def fig_yield(table, out):
    t = table[table.Glacier != "ASE total"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    bottoms = np.zeros(len(t))
    parts = [("unusable", t["Triplets"] - t["n_fr>=2"], "#cccccc"),
             ("marginal", t["n_fr>=2"] - t["n_fr>=3"], "#9ecae1"),
             ("operational", t["n_fr>=3"] - t["n_fr>=5"], "#3182bd"),
             ("excellent", t["n_fr>=5"], "#08306b")]
    for lab, v, c in parts:
        a.bar(t.Glacier, v, bottom=bottoms, color=c, label=lab)
        bottoms += v.to_numpy()
    a.set_ylabel("Rigid 12-day triplets in window")
    a.set_title("(a) Triplets by predicted usability class")
    a.legend(fontsize=8)
    x = np.arange(len(t))
    b.bar(x - 0.2, t["pct>=3"], 0.4, label="$n_{fr}\\geq 3$", color="#3182bd")
    b.bar(x + 0.2, t["pct>=5"], 0.4, label="$n_{fr}\\geq 5$", color="#08306b")
    b.set_xticks(x, t.Glacier)
    b.set_ylabel("Share of triplets (%)")
    b.set_title("(b) Predicted share above threshold")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_yield_per_glacier.png", dpi=200)
    plt.close(fig)


def fig_tides(df, cats, out, half_window_days=20):
    """One panel per top-ranked triplet, each centred on its own t2."""
    fig, axes = plt.subplots(len(SITES), 2, figsize=(11, 7.5))
    for row, (name, site) in zip(axes, SITES.items()):
        top = df[df.glacier == name].sort_values("fringes", ascending=False).head(2)
        for ax, (_, r), c, lab in zip(row, top.iterrows(), ["tab:red", "tab:blue"], ["rank 1", "rank 2"]):
            t0 = r.t2_utc - timedelta(days=half_window_days)
            tt = pd.date_range(t0, r.t2_utc + timedelta(days=half_window_days), freq="15min", tz="UTC")
            secs = np.array([(t - EPOCH).total_seconds() for t in tt.to_pydatetime()])
            ax.plot(tt, cats.predict_tide_m(site.lat, site.lon, secs), color="0.6", lw=0.6)
            ax.plot([r.t1_utc, r.t2_utc, r.t3_utc], [r.tide_h1_m, r.tide_h2_m, r.tide_h3_m],
                    "o", color=c, ms=7)
            ax.set_title(f"{name}, {lab}: track {r.track}, $n_{{fr}}$ = {r.fringes:.2f}, "
                         f"$h_{{DD}}$ = {r.h_dd_m:+.2f} m", fontsize=9, loc="left")
            ax.tick_params(axis="x", labelsize=7)
            ax.set_ylabel("Tide (m)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_tide_top2.png", dpi=200)
    plt.close(fig)


def fig_modes(df, out, glacier="Thwaites"):
    g = df[df.glacier == glacier]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    bins = np.arange(0, np.ceil(g.fringes.max()) + 0.25, 0.25)
    for mode, c in [("high-at-t2", "tab:red"), ("low-at-t2", "tab:blue")]:
        s = g[g["mode"] == mode]
        a.hist(s.fringes, bins=bins, color=c, histtype="step", lw=1.6, label=f"{mode} (n={len(s)})")
    a.axvline(3, color="k", ls="--", lw=0.8)
    a.set_xlabel("Predicted $n_{fr}$")
    a.set_ylabel("Triplets")
    a.set_title(f"(a) {glacier}, {len(g)} triplets")
    a.legend(fontsize=8)
    m = (g.tide_h1_m + g.tide_h3_m) / 2
    b.scatter(m, g.tide_h2_m, c=np.where(g["mode"] == "high-at-t2", "tab:red", "tab:blue"), s=8)
    lim = [min(m.min(), g.tide_h2_m.min()) - 0.1, max(m.max(), g.tide_h2_m.max()) + 0.1]
    b.plot(lim, lim, "k-", lw=0.8)
    b.set_xlabel("$(h_1+h_3)/2$ (m)")
    b.set_ylabel("$h_2$ (m)")
    b.set_title("(b) Distance from diagonal $\\propto |h_{DD}|$")
    fig.tight_layout()
    fig.savefig(out / "fig5_high_vs_low_t2.png", dpi=200)
    plt.close(fig)


def fig_calendar(df, out):
    op = df[df.fringes >= 3].copy()
    op["month"] = op.t2_utc.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.4], width_ratios=[5, 1], hspace=0.08, wspace=0.05)
    a = fig.add_subplot(gs[0, 0])
    m = fig.add_subplot(gs[1, 0], sharex=a)
    r = fig.add_subplot(gs[1, 1], sharey=m)
    piv = op.groupby(["month", "glacier"]).size().unstack(fill_value=0).reindex(columns=list(SITES), fill_value=0)
    bottom = np.zeros(len(piv))
    for name in SITES:
        a.bar(piv.index, piv[name], width=25, bottom=bottom, color=COLORS[name])
        bottom += piv[name].to_numpy()
    a.set_ylabel("Triplets\nper month")
    plt.setp(a.get_xticklabels(), visible=False)
    ypos = {n: i for i, n in enumerate(SITES)}
    for name, g in op.groupby("glacier"):
        m.scatter(g.t2_utc.dt.tz_localize(None), [ypos[name]] * len(g), s=20 * g.fringes,
                  color=COLORS[name], alpha=0.7)
        ex = g[g.fringes >= 5]
        m.scatter(ex.t2_utc.dt.tz_localize(None), [ypos[name]] * len(ex), s=20 * ex.fringes,
                  facecolors="none", edgecolors="red", lw=1.5)
    m.set_yticks(list(ypos.values()), list(ypos))
    m.set_xlabel("Central acquisition $t_2$ (UTC)")
    counts = op.glacier.value_counts().reindex(list(SITES), fill_value=0)
    r.barh(list(ypos.values()), counts, color=[COLORS[n] for n in SITES])
    for i, v in enumerate(counts):
        r.text(v, i, f" {v}", va="center", fontsize=9)
    r.set_xlabel("Total")
    plt.setp(r.get_yticklabels(), visible=False)
    r.set_xlim(0, counts.max() * 1.35)
    fig.savefig(out / "fig6_operational_calendar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
RSLC_COLLECTIONS = ("NISAR_L1_RSLC_BETA_V1", "NISAR_L1_RSLC_PROVISIONAL_V1")


def archive_check(df, out):
    """Which predicted triplets have all three RSLCs public at ASF (CMR query, saved to disk).

    This checks that the scheduled passes were acquired and released. It says
    nothing about whether a grounding line can be retrieved from them.
    """
    import re
    import requests

    pat = re.compile(r"NISAR_L1_\w\w_RSLC_\d{3}_(\d{3})_[AD]_\d{3}_.*?_(\d{8})T")
    granules, have = [], {}
    for name, site in SITES.items():
        keys = set()
        for coll in RSLC_COLLECTIONS:
            for page in range(1, 20):
                r = requests.get(CMR, params={"short_name": coll, "point": f"{site.lon},{site.lat}",
                                              "page_size": 2000, "page_num": page}, timeout=120)
                r.raise_for_status()
                entries = r.json()["feed"]["entry"]
                for e in entries:
                    granules.append({"glacier": name, "collection": coll, "granule": e["title"]})
                    m = pat.match(e["title"])
                    if m:
                        keys.add((int(m.group(1)), m.group(2)))
                if len(entries) < 2000:
                    break
        have[name] = keys
    pd.DataFrame(granules).to_csv(out / "asf_rslc_granules.csv", index=False)
    df = df.copy()
    df["all_three_rslc_public"] = [
        all((r.track, t.strftime("%Y%m%d")) in have[r.glacier] for t in (r.t1_utc, r.t2_utc, r.t3_utc))
        for r in df.itertuples()]
    df.to_csv(out / "all_triplets.csv", index=False)
    summ = df.groupby("glacier").agg(triplets=("fringes", "size"),
                                     rslc_public=("all_three_rslc_public", "sum"))
    op = df[df.fringes >= 3]
    # Consecutive 12-day GUNW pairs: their difference is the DDInSAR of the triplet.
    gpat = re.compile(r"NISAR_L2_\w\w_GUNW_\d{3}_(\d{3})_[AD]_\d{3}_.*?_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{8})T")
    pairs, grows = set(), []
    for name, site in SITES.items():
        for page in range(1, 20):
            r = requests.get(CMR, params={"short_name": "NISAR_L2_GUNW_PROVISIONAL_V1",
                                          "point": f"{site.lon},{site.lat}",
                                          "page_size": 2000, "page_num": page}, timeout=120)
            r.raise_for_status()
            entries = r.json()["feed"]["entry"]
            for e in entries:
                grows.append({"glacier": name, "granule": e["title"],
                              "size_mb": float(e.get("granule_size") or 0)})
                m = gpat.match(e["title"])
                if m:
                    pairs.add((name, int(m.group(1)), m.group(2), m.group(3)))
            if len(entries) < 2000:
                break
    pd.DataFrame(grows).to_csv(out / "asf_gunw_granules.csv", index=False)
    f8 = lambda t: t.strftime("%Y%m%d")
    df["both_gunw_public"] = [
        (r.glacier, r.track, f8(r.t1_utc), f8(r.t2_utc)) in pairs
        and (r.glacier, r.track, f8(r.t2_utc), f8(r.t3_utc)) in pairs
        for r in df.itertuples()]
    df.to_csv(out / "all_triplets.csv", index=False)
    gw = df[df.both_gunw_public]
    gunw = {"triplets_with_both_gunw": int(len(gw)),
            "of_which_nfr_ge3": int((gw.fringes >= 3).sum()),
            "of_which_nfr_lt1": int((gw.fringes < 1).sum()),
            "nfr_min": float(gw.fringes.min()) if len(gw) else None,
            "nfr_max": float(gw.fringes.max()) if len(gw) else None,
            "median_gunw_size_mb": float(pd.DataFrame(grows).size_mb.median())}
    return {"per_glacier": summ.to_dict(orient="index"),
            "gunw": gunw,
            "operational_total": len(op),
            "operational_rslc_public": int(op.all_three_rslc_public.sum()),
            "excellent_rslc_public": int(df[df.fringes >= 5].all_three_rslc_public.sum()),
            "archived_track_dates": {k: len(v) for k, v in have.items()},
            "queried_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("gmd_outputs"))
    ap.add_argument("--gl-gpkg", type=Path)
    ap.add_argument("--cats-dir", type=Path)
    ap.add_argument("--archive-check", action="store_true",
                    help="query NASA CMR for public NISAR RSLCs on every predicted triplet")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.live:
        gj = rop.query_rop(ASE_BBOX)
        snap = out / f"rop_snapshot_{datetime.now(timezone.utc):%Y-%m-%d}.geojson"
        snap.write_text(json.dumps(gj))
    else:
        gj = rop.load_snapshot(args.snapshot)
    scenes = rop.scenes_from_geojson(gj)

    lband_dates = [datetime.fromisoformat(d) for s in scenes for d, ids in s.dates_modes
                   if any(s.modes.get(i) and s.modes[i].is_lband_science for i in ids)]
    start = min(lband_dates).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=365)

    cats = CATSBackend(data_dir=args.cats_dir) if args.cats_dir else CATSBackend()
    df = build_triplets(scenes, start, end, cats)
    df.to_csv(out / "all_triplets.csv", index=False)

    table = yield_table(df)
    table.to_csv(out / "table2_yield.csv", index=False)
    thr = pd.DataFrame([{"threshold": k, **{g: int((d.fringes >= k).sum()) for g, d in df.groupby("glacier")},
                          "ASE total": int((df.fringes >= k).sum()),
                          "share_pct": 100 * float((df.fringes >= k).mean())}
                         for k in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5)])
    thr.to_csv(out / "threshold_sensitivity.csv", index=False)
    cull, max_shed = culling(df)
    cull.to_csv(out / "culling.csv", index=False)

    thw = df[df.glacier == "Thwaites"]
    facts = {
        "rop_scenes_in_bbox": len(scenes),
        "rop_tracks_in_bbox": len({s.track for s in scenes}),
        "window_start": start.date().isoformat(),
        "window_end": end.date().isoformat(),
        "tracks_per_glacier": df.groupby("glacier").track.nunique().to_dict(),
        "incidence_deg_range_per_glacier": {k: [round(v.min(), 1), round(v.max(), 1)]
                                            for k, v in df.groupby("glacier").incidence_deg},
        "triplets_total": len(df),
        "high_at_t2": int((df["mode"] == "high-at-t2").sum()),
        "low_at_t2": int((df["mode"] == "low-at-t2").sum()),
        "max_fr_high_thwaites": float(thw[thw["mode"] == "high-at-t2"].fringes.max()),
        "max_fr_low_thwaites": float(thw[thw["mode"] == "low-at-t2"].fringes.max()),
        "usable_share_rank_cutoff": float((df.fringes >= 3).mean()),
        "max_shed_fraction_no_usable_loss": float(max_shed),
        "operational_per_glacier": df[df.fringes >= 3].groupby("glacier").size().to_dict(),
        "operational_t2_gap_days": {
            k: (lambda d: {"distinct_dates": int(len(d) + 1), "median": float(np.median(d)),
                           "max": float(d.max())})(
                np.diff(np.unique(g.t2_utc.dt.floor("D").dt.tz_localize(None).to_numpy()))
                / np.timedelta64(1, "D"))
            for k, g in df[df.fringes >= 3].groupby("glacier")},
        "incidence_sensitivity_fixed40deg": {
            "n_fr>=3": int((df.fringes * np.cos(np.radians(40)) / np.cos(np.radians(df.incidence_deg)) >= 3).sum()),
            "n_fr>=5": int((df.fringes * np.cos(np.radians(40)) / np.cos(np.radians(df.incidence_deg)) >= 5).sum())},
        "operational_months_by_mode_thwaites": {
            m: sorted({str(t) for t in g.t2_utc.dt.tz_localize(None).dt.to_period("M")})
            for m, g in thw[thw.fringes >= 3].groupby("mode")},
        "operational_by_month": {str(k): int(v) for k, v in df[df.fringes >= 3]
                                 .groupby(df.t2_utc.dt.tz_localize(None).dt.to_period("M")).size().items()},
        "flexure_thwaites_H800_nu041": {
            "beta_per_m": flexural_parameter_beta(0.88e9, 800.0, 0.41),
            "inv_beta_km": 1e-3 / flexural_parameter_beta(0.88e9, 800.0, 0.41),
            "pi_over_beta_km": 1e-3 * np.pi / flexural_parameter_beta(0.88e9, 800.0, 0.41)},
    }
    if args.archive_check:
        facts["asf_archive_check"] = archive_check(df, out)
    (out / "facts.json").write_text(json.dumps(facts, indent=2, default=str))

    fig_footprints(scenes, df, out, args.gl_gpkg)
    fig_yield(table, out)
    fig_tides(df, cats, out)
    fig_modes(df, out)
    fig_calendar(df, out)
    print(table.round(2).to_string(index=False))
    print(cull.to_string(index=False))
    print(json.dumps(facts, indent=2, default=str))


if __name__ == "__main__":
    main()
