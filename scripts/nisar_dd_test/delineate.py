"""NISAR grounding-line delineation at Smith from the SNAPHU-unwrapped double difference (GMD RC1, R1.C3).

Definition. The grounding line is the landward limit of tidal flexure (point F; Rignot et al., 2011),
taken on each transect as the first point, going seaward, where the unwrapped DD phase departs from
the grounded plateau by FRAC of the total step to the floating plateau. FRAC = 0.10 is reported;
0.05 and 0.20 give the sensitivity of the line to the threshold.

Transects. Normal to the fringe belt at the ridge locations of unwrap_dd.py (densest 30 % of the belt),
every ~250 m along the belt, from -6 to +6 km. Phase is averaged over +-200 m along the belt at each
point. Plateaus: median phase over [-6, -4] km and [+4, +6] km. A transect is used when both plateaus
are coherent (>= 50 % of samples), the step is >= 1 fringe, and the profile is monotonic enough that
the threshold is crossed once within +-4 km of the ridge.

Orientation. Which end is grounded is taken from the most recent Sentinel-1 grounding line (30 January
2025): the landward end is the one on the side of that line. This is the only use of the Sentinel-1 line
in the delineation; the line position comes from the NISAR phase alone.

Outputs: out/nisar_gl_<key>.geojson (points, EPSG:3031), out/delineation_summary.csv, and the
comparison of each NISAR line with the Sentinel-1 line, the latest CSK line and with the other NISAR
track.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point

KEYS = {"Smith|34|20260628": "2026-06-28/07-10/07-22, track 34",
        "Smith|106|20260703": "2026-07-03/07-15/07-27, track 106"}
FRACS = (0.05, 0.10, 0.20)
S = np.arange(-6.0, 6.0001, 0.04)
ALONG = np.arange(-0.2, 0.2001, 0.04)

gl = gpd.read_file("gl_1992_2025_ase.geojson")
gl["date"] = pd.to_datetime(gl.Date_1, errors="coerce")
smith = gl[gl.Glac_Name.astype(str).str.contains("Smith")]
S1 = smith[smith.Sensor.astype(str).str.contains("Sentinel")].sort_values("date").iloc[-1]
_csk = smith[smith.Sensor.astype(str).str.contains("CSK")]
_csk = _csk[[g.distance(S1.geometry) < 5e3 for g in _csk.geometry]]   # same grounding zone only
CSK = _csk.sort_values("date").iloc[-1] if len(_csk) else None


def kf(k):
    return k.replace("|", "_").replace(" ", "")


def sample(z, x, y, X, Y):
    dx, dy = x[1] - x[0], y[1] - y[0]
    j = np.round((X - x[0]) / dx).astype(int); i = np.round((Y - y[0]) / dy).astype(int)
    ok = (i >= 0) & (i < len(y)) & (j >= 0) & (j < len(x))
    out = np.full(X.shape, np.nan); out[ok] = z[i[ok], j[ok]]
    return out


def delineate(key):
    z = np.load(f"out/unw_{kf(key)}.npz"); b = np.load(f"out/belt_{kf(key)}.npz")
    x, y, unw = z["x"], z["y"], z["unw"].astype(float)
    dens = z["dens"].astype(float)
    dx, dy = x[1] - x[0], y[1] - y[0]
    jj = np.round((b["xi"] - x[0]) / dx).astype(int); ii = np.round((b["yi"] - y[0]) / dy).astype(int)
    d = dens[ii, jj]; sel = d >= np.percentile(d, 70)
    pts = list(zip(b["xi"][sel], b["yi"][sel], b["ux"][sel], b["uy"][sel]))
    out = {f: [] for f in FRACS}
    used = 0
    for xi, yi, ux, uy in pts:
        # orient: +s seaward, i.e. away from the Sentinel-1 line
        near = S1.geometry.interpolate(S1.geometry.project(Point(xi, yi)))
        landward = np.sign((near.x - xi) * ux + (near.y - yi) * uy) or 1.0
        gx, gy = -landward * ux, -landward * uy             # unit vector pointing seaward
        tx, ty = -gy, gx
        prof = np.full(S.shape, np.nan)
        for k, s in enumerate(S):
            v = sample(unw, x, y, xi + s * 1e3 * gx + ALONG * 1e3 * tx, yi + s * 1e3 * gy + ALONG * 1e3 * ty)
            v = v[np.isfinite(v)]
            if v.size >= ALONG.size // 2:
                prof[k] = np.median(v)
        land = prof[(S >= -6) & (S <= -4)]; sea = prof[(S >= 4) & (S <= 6)]
        if np.isfinite(land).mean() < 0.5 or np.isfinite(sea).mean() < 0.5:
            continue
        p0, p1 = np.nanmedian(land), np.nanmedian(sea)
        step = p1 - p0
        if abs(step) < 2 * np.pi:                            # < 1 fringe: no usable flexure
            continue
        u = (prof - p0) / step                               # 0 grounded, 1 floating
        used += 1
        for f in FRACS:
            win = (S >= -4) & (S <= 4) & np.isfinite(u)
            cross = np.where(win[:-1] & win[1:] & (u[:-1] < f) & (u[1:] >= f))[0]
            if cross.size == 0:
                continue
            k = cross[0]
            sc = S[k] + (f - u[k]) / (u[k + 1] - u[k]) * (S[k + 1] - S[k])
            out[f].append((xi + sc * 1e3 * gx, yi + sc * 1e3 * gy, gx, gy))
    return out, len(pts), used


def signed_dist(pt, line, ref_dir_pt):
    return line.distance(Point(pt)) / 1e3


rows, lines = [], {}
for key, label in KEYS.items():
    res, n_ridge, used = delineate(key)
    lines[key] = res
    for f, p in res.items():
        if not p:
            continue
        def signed(line, q):
            n = line.interpolate(line.project(Point(q[0], q[1])))
            return np.sign((q[0] - n.x) * q[2] + (q[1] - n.y) * q[3]) * Point(q[0], q[1]).distance(n) / 1e3
        dS1 = np.array([signed(S1.geometry, q) for q in p])        # + = NISAR GL seaward of the S1 line
        # seaward (+) or landward (-) of the S1 line: compare with the ridge side
        rows.append({"triplet": key, "acquisitions": label, "frac": f, "transects_used": used,
                     "points": len(p), "dist_to_S1_2025_km_median": float(np.median(dS1)),
                     "dist_to_S1_2025_km_p25": float(np.percentile(dS1, 25)),
                     "dist_to_S1_2025_km_p75": float(np.percentile(dS1, 75)),
                     "abs_dist_to_S1_km_median": float(np.median(np.abs(dS1))),
                     "dist_to_CSK_km_median": float(np.median([signed(CSK.geometry, q) for q in p]))
                     if CSK is not None else None})
        gpd.GeoDataFrame({"frac": [f] * len(p)}, geometry=[Point(q[0], q[1]) for q in p], crs=3031).to_file(
            f"out/nisar_gl_{kf(key)}_f{int(f * 100):02d}.geojson", driver="GeoJSON")

# agreement between the two NISAR tracks (10 % lines): nearest-point distances
a = lines["Smith|34|20260628"][0.10]; b2 = lines["Smith|106|20260703"][0.10]
if a and b2:
    mb = MultiPoint([(q[0], q[1]) for q in b2])
    d_ab = np.array([mb.distance(Point(q[0], q[1])) for q in a]) / 1e3
    rows.append({"triplet": "track 34 vs track 106", "frac": 0.10, "points": len(a),
                 "dist_to_S1_2025_km_median": None, "track_to_track_km_median": float(np.median(d_ab)),
                 "track_to_track_km_p75": float(np.percentile(d_ab, 75))})
summ = pd.DataFrame(rows)
summ.to_csv("out/delineation_summary.csv", index=False)
print("Sentinel-1 reference:", S1.Sensor, str(S1.Date_1)[:10], S1.Glac_Name)
print("CSK reference:", None if CSK is None else f"{CSK.Sensor} {str(CSK.Date_1)[:10]}")
print(summ.round(2).to_string())
