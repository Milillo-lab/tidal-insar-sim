"""Observed DDInSAR fringe count across the grounding-zone fringe belt (GMD RC1, R1.C3 / R1.C12).

Pass 1 (per triplet): 40 m complex DD = ifg12 * conj(ifg23) (unit phasors, 2x2 looks of the ~20 m
GUNW wrapped interferograms) in a +-HALF km window around the site; coherence from a 5x5 boxcar of
the unit phasors; 2-D unwrapping with SNAPHU (deformation cost, MCF init) on coherent pixels;
fringe density |grad(unw)| / 2pi (cycles/km) after a 200 m Gaussian smoothing.

Belt: coherent pixels (coh >= COH_BELT) within BELT_RADIUS of the site whose fringe density is
>= DENS_BELT and lies in the largest connected cluster of such pixels. For every (glacier, track,
frame) the belt of the triplet with the most belt pixels defines the measurement locations and the
gradient directions; every triplet of that group is measured at those locations.

Observable: at each belt location p with unit gradient direction g, the step
|unw(p + D g) - unw(p - D g)| / 2pi, D = STEP_HALF km, using pixels coherent at both ends and in one
SNAPHU connected component. n_obs = median over locations; p25/p75 reported. The step includes any
broad (ionospheric, tropospheric) phase gradient over 2D km; its size is estimated the same way at
points 10 km off the belt (n_bg) and reported beside n_obs.

Run with Python 3 with h5py, scipy, pyproj and snaphu-py 0.4.1. Outputs out/unw_results.csv and
out/unw_<key>.npz.
"""
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import snaphu
from pyproj import Transformer
from scipy import ndimage

SITES = {"Thwaites": (-75.00, -106.00), "Pine Island": (-74.95, -100.70), "Smith": (-74.60, -112.00)}
W = "science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH/"
HALF = 25e3
COH_MIN, COH_BELT = 0.25, 0.35
DENS_BELT = 0.4           # cycles/km
BELT_RADIUS = 15e3
STEP_HALF = 6.0           # km: the belt is ~3 km wide, so +-6 km reaches both plateaus from any belt pixel
PIX = 40.0


def window(path, cx, cy):
    with h5py.File(path, "r") as f:
        x = f[W + "xCoordinates"][()]; y = f[W + "yCoordinates"][()]
        jx = np.where(np.abs(x - cx) <= HALF)[0]; iy = np.where(np.abs(y - cy) <= HALF)[0]
        if len(jx) < 50 or len(iy) < 50:
            return None
        sl = (slice(iy[0], iy[-1] + 1), slice(jx[0], jx[-1] + 1))
        return x[sl[1]], y[sl[0]], f[W + "wrappedInterferogram"][sl]


def ml2(a):
    r, c = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    return a[:r, :c].reshape(r // 2, 2, c // 2, 2).mean(axis=(1, 3))


def unit(a):
    m = np.abs(a)
    return np.where(m > 0, a / np.where(m > 0, m, 1), 0)


def pass1(f12, f23, cx, cy):
    w1 = window(f12, cx, cy); w2 = window(f23, cx, cy)
    if w1 is None or w2 is None:
        return None
    assert np.allclose(w1[0], w2[0]) and np.allclose(w1[1], w2[1]), "grids differ"
    dd = ml2(unit(w1[2]) * np.conj(unit(w2[2])))
    x = ml2(w1[0][None, :].repeat(2, 0))[0]; y = ml2(w1[1][:, None].repeat(2, 1))[:, 0]
    ddu = unit(dd)
    coh = np.abs(ndimage.uniform_filter(ddu.real, 5) + 1j * ndimage.uniform_filter(ddu.imag, 5))
    coh = np.nan_to_num(coh).astype(np.float32)
    mask = (coh >= COH_MIN) & (np.abs(dd) > 0)
    unw, cc = snaphu.unwrap(ddu.astype(np.complex64), np.clip(coh, 0, 1), nlooks=20.0, cost="defo",
                            init="mcf", mask=mask)
    unw = np.where(mask & (cc > 0), unw, np.nan).astype(np.float32)
    sm = ndimage.gaussian_filter(np.nan_to_num(unw), 200 / PIX)
    gy, gx = np.gradient(sm, PIX / 1e3)                       # rad/km (array row = y index)
    dens = np.hypot(gx, gy) / (2 * np.pi)
    dens[~np.isfinite(unw)] = 0
    return {"x": x, "y": y, "unw": unw, "cc": cc.astype(np.int32), "coh": coh, "dens": dens.astype(np.float32),
            "gx": gx.astype(np.float32), "gy": gy.astype(np.float32), "wrapped": np.angle(ddu).astype(np.float32)}


def belt(r, cx, cy):
    X, Y = np.meshgrid(r["x"], r["y"])
    cand = (r["dens"] >= DENS_BELT) & (r["coh"] >= COH_BELT) & (np.hypot(X - cx, Y - cy) <= BELT_RADIUS)
    lab, n = ndimage.label(cand)
    if n == 0:
        return None
    sizes = ndimage.sum(cand, lab, range(1, n + 1))
    keep = lab == (1 + int(np.argmax(sizes)))
    ii, jj = np.nonzero(keep)
    sub = slice(None, None, max(1, len(ii) // 400))          # <= ~400 locations
    ii, jj = ii[sub], jj[sub]
    g = np.hypot(r["gx"][ii, jj], r["gy"][ii, jj])
    return {"xi": r["x"][jj], "yi": r["y"][ii], "ux": r["gx"][ii, jj] / g, "uy": r["gy"][ii, jj] / g,
            "npix": int(keep.sum())}


def steps(r, b, offset_km=0.0):
    """Step across 2*STEP_HALF km along the belt direction; optionally at points shifted
    offset_km along the gradient (background estimate away from the belt)."""
    dx, dy = r["x"][1] - r["x"][0], r["y"][1] - r["y"][0]

    def at(X, Y):
        j = np.round((X - r["x"][0]) / dx).astype(int); i = np.round((Y - r["y"][0]) / dy).astype(int)
        ok = (i >= 0) & (i < len(r["y"])) & (j >= 0) & (j < len(r["x"]))
        u = np.full(X.shape, np.nan); c = np.full(X.shape, -1)
        u[ok] = r["unw"][i[ok], j[ok]]; c[ok] = r["cc"][i[ok], j[ok]]
        return u, c
    out = []
    for sgn in ([1.0, -1.0] if offset_km else [0.0]):
        X0 = b["xi"] + sgn * offset_km * 1e3 * b["ux"]; Y0 = b["yi"] + sgn * offset_km * 1e3 * b["uy"]
        up, cp = at(X0 + STEP_HALF * 1e3 * b["ux"], Y0 + STEP_HALF * 1e3 * b["uy"])
        um, cm = at(X0 - STEP_HALF * 1e3 * b["ux"], Y0 - STEP_HALF * 1e3 * b["uy"])
        ok = np.isfinite(up) & np.isfinite(um) & (cp == cm) & (cp > 0)
        out.append(np.abs(up[ok] - um[ok]) / (2 * np.pi))
    return np.concatenate(out)


def main():
    sel = pd.read_csv("gunw_test_selection.csv", parse_dates=["t1_utc", "t2_utc", "t3_utc"])
    files = {os.path.basename(p)[:-3]: p for p in glob.glob("gunw/*.h5")}
    tr = Transformer.from_crs(4326, 3031, always_xy=True)
    os.makedirs("out", exist_ok=True)
    res, belts = {}, {}
    for r in sel.itertuples():
        f8 = lambda t: t.strftime("%Y%m%d")
        pat = lambda a, b: re.compile(rf"GUNW_\d{{3}}_{r.track:03d}_[AD]_{r.frame:03d}_.*_{f8(a)}T\d{{6}}_\d{{8}}T\d{{6}}_{f8(b)}T")
        f12 = [p for n, p in files.items() if pat(r.t1_utc, r.t2_utc).search(n)]
        f23 = [p for n, p in files.items() if pat(r.t2_utc, r.t3_utc).search(n)]
        key = f"{r.glacier}|{r.track}|{f8(r.t1_utc)}"
        cx, cy = tr.transform(SITES[r.glacier][1], SITES[r.glacier][0])
        p1 = pass1(f12[0], f23[0], cx, cy) if f12 and f23 else None
        if p1 is None:
            print(key, "skipped", flush=True); continue
        b = belt(p1, cx, cy)
        res[key] = (r, p1, (cx, cy))
        g = (r.glacier, r.track, r.frame)
        if b is not None and (g not in belts or b["npix"] > belts[g][1]["npix"]):
            belts[g] = (key, b)
        np.savez_compressed(f"out/unw_{key.replace('|', '_').replace(' ', '')}.npz",
                            unw=p1["unw"], wrapped=p1["wrapped"], coh=p1["coh"].astype(np.float16),
                            dens=p1["dens"].astype(np.float16), x=p1["x"], y=p1["y"])
        print(key, "unwrapped; own belt pixels:", None if b is None else b["npix"], flush=True)
    rows = []
    for key, (r, p1, c) in res.items():
        g = (r.glacier, r.track, r.frame)
        if g not in belts:
            rows.append({"key": key, "glacier": r.glacier, "track": r.track, "frame": r.frame,
                         "t2": f"{r.t2_utc:%Y-%m-%d}", "n_pred": r.fringes, "status": "no belt on this track"})
            continue
        bkey, b = belts[g]
        st = steps(p1, b)
        bg = steps(p1, b, offset_km=15.0)
        rows.append({"key": key, "glacier": r.glacier, "track": r.track, "frame": r.frame,
                     "t2": f"{r.t2_utc:%Y-%m-%d}", "n_pred": r.fringes, "h_dd_pred_m": r.h_dd_m,
                     "belt_from": bkey, "belt_pixels": b["npix"], "n_locations": int(st.size),
                     "n_obs_median": float(np.median(st)) if st.size else None,
                     "n_obs_p25": float(np.percentile(st, 25)) if st.size else None,
                     "n_obs_p75": float(np.percentile(st, 75)) if st.size else None,
                     "n_bg_median": float(np.median(bg)) if bg.size else None,
                     "coh_median_site": float(np.median(p1["coh"][p1["coh"] > 0])),
                     "status": "ok" if st.size >= 20 else "too few coherent locations"})
        print(rows[-1], flush=True)
        np.savez_compressed(f"out/belt_{key.replace('|', '_').replace(' ', '')}.npz", **b)
    pd.DataFrame(rows).to_csv("out/unw_results.csv", index=False)


if __name__ == "__main__":
    main()
