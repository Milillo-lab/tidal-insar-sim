"""Figure for Sect. 4.5: observed vs predicted DDInSAR fringe count from NISAR GUNW pairs.

(a) observed step across the grounding-zone fringe belt (median, IQR bars) vs predicted n_fr (Eq. 4),
    one point per triplet, coloured by site; triplets without a coherent belt are marked on the axis.
(b) example: SNAPHU-unwrapped DD at Smith (track 34, t2 = 2026-07-10) with the belt locations and the
    latest Sentinel-1 grounding line.
Also writes out/gl_offsets.csv: distance from each group's belt locations to the latest Sentinel-1 GL.
"""
import glob

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

COL = {"Thwaites": "#1f9bb4", "Pine Island": "#d98b1a", "Smith": "#7b4fa6"}
r = pd.read_csv("out/plateau_results.csv")
gl = gpd.read_file("gl_1992_2025_ase.geojson")
gl["date"] = pd.to_datetime(gl.Date_1, errors="coerce")

# GL offsets: belt locations to the latest Sentinel-1 line of the same glacier
offs = []
for bk in sorted(set(r[r.status == "measured"].belt_from.dropna())):
    b = np.load("out/belt_" + bk.replace("|", "_").replace(" ", "") + ".npz")
    gname = bk.split("|")[0]
    pts = [Point(x, y) for x, y in zip(b["xi"], b["yi"])]
    cand = gl[gl.Glac_Name.astype(str).str.contains(gname) & gl.Sensor.astype(str).str.contains("Sentinel")]
    cand = cand[[g.distance(pts[len(pts) // 2]) < 30e3 for g in cand.geometry]]
    if cand.empty:
        continue
    ref = cand.sort_values("date").iloc[-1]
    dist = np.array([ref.geometry.distance(p) for p in pts]) / 1e3
    offs.append({"belt_from": bk, "s1_line": f"{ref.Sensor} {str(ref.Date_1)[:10]}",
                 "dist_km_median": float(np.median(dist)), "dist_km_p25": float(np.percentile(dist, 25)),
                 "dist_km_p75": float(np.percentile(dist, 75))})
pd.DataFrame(offs).to_csv("out/gl_offsets.csv", index=False)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(18.5, 5.6), gridspec_kw={"width_ratios": [1, 1.12, 1.0]})
ok = r[r.status == "measured"]
for g, d in ok.groupby("glacier"):
    a.errorbar(d.n_pred, d.n_obs, yerr=[d.n_obs - d.n_obs_p25, d.n_obs_p75 - d.n_obs],
               fmt="o", color=COL[g], ms=6, capsize=2, label=f"{g} ({len(d)})")
bad = r[r.status != "measured"]
for g, d in bad.groupby("glacier"):
    a.plot(d.n_pred, np.zeros(len(d)) - 0.3, "x", color=COL[g], ms=7)
a.plot([], [], "x", color="0.3", label=f"not measurable: decorrelated ({len(bad)})")
bgm = r.n_bg.dropna()
if len(bgm):
    a.axhspan(0, float(bgm.max()), color="0.85", zorder=0, label="background step range")
m = max(6, np.nanmax(r.n_pred) + 0.5)
a.plot([0, m], [0, m], "k--", lw=0.8)
a.axvline(3, color="0.7", lw=0.8, ls=":"); a.axhline(3, color="0.7", lw=0.8, ls=":")
a.set_xlim(-0.3, m); a.set_ylim(-0.5, m)
a.set_xlabel("Predicted $n_{fr}$ (Eq. 4, CATS2008)")
a.set_ylabel("Observed step across the fringe belt (fringes)")
a.set_title("(a) NISAR DDInSAR, observed vs predicted", loc="left", fontsize=10)
a.legend(fontsize=8, loc="upper left")

ex = glob.glob("out/unw_Smith_34_20260628.npz")
if ex:
    z = np.load(ex[0]); x, y = z["x"], z["y"]
    im = b.imshow(z["unw"] / (2 * np.pi), extent=[x[0], x[-1], y[-1], y[0]], cmap="RdBu", interpolation="nearest")
    fig.colorbar(im, ax=b, label="Unwrapped DD phase (fringes)")
    bb = glob.glob("out/belt_Smith_34_20260628.npz")
    if bb:
        q = np.load(bb[0]); b.plot(q["xi"], q["yi"], "k.", ms=0.6)
    s1 = gl[gl.Glac_Name.astype(str).str.contains("Smith") & gl.Sensor.astype(str).str.contains("Sentinel")]
    s1.sort_values("date").iloc[-1:].plot(ax=b, color="k", lw=1.0, ls="--")
    b.set_xlim(x[0], x[-1]); b.set_ylim(y[-1], y[0])
    b.set_xlabel("x (m, EPSG:3031)"); b.set_ylabel("y (m, EPSG:3031)")
    b.set_title("(b) Smith, track 34, $t_2$ = 2026-07-10 (predicted 4.16)", loc="left", fontsize=10)
# (c) NISAR grounding lines (10 % threshold, delineate.py) on the wrapped DD, with the Sentinel-1 line
w = glob.glob("out/unw_Smith_34_20260628.npz")
if w:
    z = np.load(w[0]); x, y = z["x"], z["y"]
    c.imshow(np.where(z["coh"].astype(float) > 0.3, z["wrapped"], np.nan), extent=[x[0], x[-1], y[-1], y[0]],
             cmap="hsv", interpolation="nearest", alpha=0.85)
    s1 = gl[gl.Glac_Name.astype(str).str.contains("Smith") & gl.Sensor.astype(str).str.contains("Sentinel")]
    s1.sort_values("date").iloc[-1:].plot(ax=c, color="k", lw=1.2, ls="--")
    for f, col, lab in [("out/nisar_gl_Smith_34_20260628_f10.geojson", "k", "NISAR GL, track 34 ($t_2$ 10 Jul 2026)"),
                        ("out/nisar_gl_Smith_106_20260703_f10.geojson", "w", "NISAR GL, track 106 ($t_2$ 15 Jul 2026)")]:
        g = gpd.read_file(f)
        c.plot(g.geometry.x, g.geometry.y, "o", ms=3.5, mfc=col, mec="k", mew=0.4, label=lab)
    c.plot([], [], "k--", label="Sentinel-1 GL, 30 Jan 2025")
    c.set_xlim(-1.585e6, -1.545e6); c.set_ylim(-0.638e6, -0.612e6)
    c.set_xlabel("x (m, EPSG:3031)"); c.legend(fontsize=7.5, loc="lower left", framealpha=0.85)
    c.set_title("(c) NISAR grounding line vs Sentinel-1, Smith", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig("out/fig7_nisar_dd_test.png", dpi=200)
print(pd.DataFrame(offs).round(2).to_string())
