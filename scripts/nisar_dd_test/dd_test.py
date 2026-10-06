"""Observed vs predicted DDInSAR fringe count from consecutive NISAR GUNW pairs (GMD RC1, R1.C3/R1.C12).

For each selected triplet (t1, t2, t3) on one track/frame:
  1. DD = exp(i*phi12) * conj(exp(i*phi23)) on the common GUNW grid. Rewrapping the unwrapped
     phase (or using the wrapped layer) makes the double difference immune to unwrapping errors
     in the single 12-day pairs, which over fast ice carry hundreds of flow fringes; steady flow
     cancels pixel by pixel because B12 = B23 = 12 d.
  2. Profiles every 1 km along the most recent reference GL within RADIUS_KM of the site,
     normal to it, from -L_LAND to +L_SEA km. The wrapped DD phase is sampled every 100 m,
     masked by coherence, and unwrapped in 1-D (fringe spacing at the hinge is >1 km for
     n <= 6, well above the sampling).
  3. Each profile is fitted, in both orientations, with
        phi(s) = a + b*s + A * F(beta*(s - s0)),  F(x) = 1 - exp(-x)(cos x + sin x) for x >= 0, else 0
     (elastic plate, Eq. (1); the ramp a + b*s absorbs ionospheric and orbital phase).
     |A|/2pi is the observed fringe count; s0 is the NISAR hinge position relative to the
     reference GL crossing (positive = seaward of the reference line).
  4. Per triplet: median over profiles that pass the fit-quality and coherence tests.

Run with Python 3 (h5py, scipy, pyproj, shapely). Inputs in the working directory: gunw/*.h5, gunw_test_selection.csv,
gl_1992_2025_ase.geojson. Outputs to out/.
"""
import glob
import json
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import least_squares
from shapely.geometry import LineString, MultiLineString, Point, shape
from shapely.ops import nearest_points

RADIUS_KM = 25.0
L_LAND, L_SEA, DS = 15.0, 25.0, 0.1      # km
COH_MIN = 0.15          # |mean of ~200 unit phasors|; random noise gives ~0.06
STRIP_KM = 0.5
PROFILE_STEP_KM = 2.0   # strips +-0.5 km wide, so profiles every 2 km are independent
OBJ_MIN = 0.25          # weighted residual coherence of the best fit
GAIN_MIN = 0.05         # improvement over the best no-flexure (ramp-only) fit
LAMBDA = 0.238
SITES = {"Thwaites": (-75.00, -106.00), "Pine Island": (-74.95, -100.70), "Smith": (-74.60, -112.00)}

GROUP = None  # set from the first file: path of the frequencyA unwrapped-interferogram group


WRAP = "science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH/"
HALF_WIN_M = 55e3       # window half-width around the site
LOOKS = 4               # 4 x 4 multilook of the ~20 m wrapped grid -> ~80 m


def window(path, cx, cy):
    """Wrapped interferogram and coherence in a window centred on (cx, cy), EPSG:3031."""
    with h5py.File(path, "r") as f:
        x = f[WRAP + "xCoordinates"][()]; y = f[WRAP + "yCoordinates"][()]
        epsg = int(f[WRAP + "projection"].attrs["epsg_code"])
        jx = np.where(np.abs(x - cx) <= HALF_WIN_M)[0]; iy = np.where(np.abs(y - cy) <= HALF_WIN_M)[0]
        if len(jx) < 10 or len(iy) < 10:
            return None
        sl = (slice(iy[0], iy[-1] + 1), slice(jx[0], jx[-1] + 1))
        ifg = f[WRAP + "wrappedInterferogram"][sl]
        coh = f[WRAP + "coherenceMagnitude"][sl]
    return x[sl[1]], y[sl[0]], ifg, coh, epsg


def multilook(a, n):
    r, c = (a.shape[0] // n) * n, (a.shape[1] // n) * n
    return a[:r, :c].reshape(r // n, n, c // n, n).mean(axis=(1, 3))


def common_dd(p12, p23, cx, cy):
    w1 = window(p12, cx, cy); w2 = window(p23, cx, cy)
    if w1 is None or w2 is None:
        return None
    x1, y1, i1, c1, e1 = w1; x2, y2, i2, c2, e2 = w2
    assert e1 == e2 == 3031, (e1, e2)
    # same frame grid: align on shared coordinates (identical in practice)
    xs = np.intersect1d(np.round(x1, 3), np.round(x2, 3)); ys = np.intersect1d(np.round(y1, 3), np.round(y2, 3))
    a1 = np.searchsorted(np.round(x1, 3), xs); a2 = np.searchsorted(np.round(x2, 3), xs)
    oy1 = np.round(y1, 3); oy2 = np.round(y2, 3)
    b1 = np.array([np.where(oy1 == v)[0][0] for v in ys]); b2 = np.array([np.where(oy2 == v)[0][0] for v in ys])
    i1 = i1[np.ix_(b1, a1)]; i2 = i2[np.ix_(b2, a2)]
    dd = i1 * np.conj(i2)
    mag = np.abs(dd); dd_n = np.where(mag > 0, dd / np.where(mag > 0, mag, 1), 0)
    ddm = multilook(dd_n, LOOKS)                        # |ddm| is the DD phase coherence
    cohm = multilook(np.minimum(c1[np.ix_(b1, a1)], c2[np.ix_(b2, a2)]), LOOKS)
    xs_m = multilook(xs[None, :].repeat(LOOKS, 0), LOOKS)[0]; ys_m = multilook(ys[:, None].repeat(LOOKS, 1), LOOKS)[:, 0]
    return xs_m, ys_m, ddm, np.minimum(np.abs(ddm), cohm), 3031


def sample(xs, ys, img, X, Y):
    """Nearest-neighbour sample of img at map coords (X, Y); NaN outside."""
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    i = np.round((Y - ys[0]) / dy).astype(int); j = np.round((X - xs[0]) / dx).astype(int)
    ok = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
    out = np.full(X.shape, np.nan + 0j if np.iscomplexobj(img) else np.nan)
    out[ok] = img[i[ok], j[ok]]
    return out


def flex(x):
    return np.where(x >= 0, 1 - np.exp(-x) * (np.cos(x) + np.sin(x)), 0.0)


A_COARSE = np.arange(-8.0, 8.0001, 0.1)        # flexure amplitude, fringes (signed)
S0_GRID = np.arange(-12.0, 12.0001, 0.5)        # hinge position, km
BETA_GRID = np.array([0.25, 0.4, 0.6, 0.9, 1.3])  # 1/km
B_GRID = np.arange(-0.6, 0.6001, 0.04)          # phase ramp, rad/km (ionosphere, orbit)


def _obj(s, wz, W, A, be, s0, Eb):
    f = flex(be * (s - s0))
    EA = np.exp(-1j * 2 * np.pi * np.outer(A, f))
    return np.abs((EA * wz) @ Eb.T) / W                       # (nA, nb)


def fit_profile(s, z, w):
    """Fit the wrapped, strip-averaged DD phasors z(s) with a + b s + 2 pi A F(beta (s - s0)).

    Objective: |sum w z exp(-i model)| / sum w (the offset a is absorbed by the modulus); grid
    search, no unwrapping; both orientations tried (which side floats is not assumed). Coarse
    grid, then the amplitude refined to 0.05 fringe at the best hinge and beta. Returns the best
    fit, the best A = 0 (ramp-only) fit and the A interval keeping >= 3/4 of the gain over it.
    """
    wz = w * z; W = w.sum()
    Eb = np.exp(-1j * np.outer(B_GRID, s))
    null = float(np.max(np.abs(Eb @ wz)) / W)
    best = None
    for sign in (1, -1):
        ss = sign * s
        for be in BETA_GRID:
            for s0 in S0_GRID:
                obj = _obj(ss, wz, W, A_COARSE, be, s0, Eb)
                k = np.unravel_index(np.argmax(obj), obj.shape)
                if best is None or obj[k] > best[0]:
                    best = (float(obj[k]), sign, be, s0, A_COARSE[k[0]])
    _, sign, be, s0, a0 = best
    ss = sign * s
    s0f = np.arange(s0 - 1.0, s0 + 1.0001, 0.25)
    Af = np.arange(a0 - 0.4, a0 + 0.4001, 0.05)
    top = None
    for s0x in s0f:
        obj = _obj(ss, wz, W, Af, be, s0x, Eb)
        k = np.unravel_index(np.argmax(obj), obj.shape)
        if top is None or obj[k] > top["obj"]:
            top = {"obj": float(obj[k]), "A_fr": float(Af[k[0]]), "s0": float(sign * s0x), "beta": float(be),
                   "orient": sign, "ramp": float(B_GRID[k[1]]), "_s0x": s0x}
    wide = np.arange(-8.0, 8.0001, 0.05)
    prof = _obj(ss, wz, W, wide, be, top.pop("_s0x"), Eb).max(axis=1)
    near = wide[prof >= null + 0.75 * (top["obj"] - null)]
    if top["A_fr"] != 0:
        near = near[np.sign(near) == np.sign(top["A_fr"])]
    top["A_lo"] = float(np.abs(near).min()) if near.size else abs(top["A_fr"])
    top["A_hi"] = float(np.abs(near).max()) if near.size else abs(top["A_fr"])
    top["null"] = null
    return top


def profiles_for(site, gl_lines, tr):
    """Reference GL near the site: (line in map coords, metadata)."""
    lat, lon = site
    px, py = tr.transform(lon, lat)
    P = Point(px, py)
    near = [(g, m) for g, m in gl_lines if g.distance(P) < RADIUS_KM * 1e3]
    return P, near


def main():
    sel = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "gunw_test_selection.csv",
                      parse_dates=["t1_utc", "t2_utc", "t3_utc"])
    files = {os.path.basename(p)[:-3]: p for p in glob.glob("gunw/*.h5")}
    gl = json.load(open("gl_1992_2025_ase.geojson"))
    gl_feats = [(shape(f["geometry"]), f["properties"]) for f in gl["features"] if f["geometry"]]
    os.makedirs("out", exist_ok=True)
    rows, prof_rows = [], []
    for r in sel.itertuples():
        d = lambda t: t.strftime("%Y%m%d")
        pat12 = re.compile(rf"GUNW_\d{{3}}_{r.track:03d}_[AD]_{r.frame:03d}_.*_{d(r.t1_utc)}T\d{{6}}_\d{{8}}T\d{{6}}_{d(r.t2_utc)}T")
        pat23 = re.compile(rf"GUNW_\d{{3}}_{r.track:03d}_[AD]_{r.frame:03d}_.*_{d(r.t2_utc)}T\d{{6}}_\d{{8}}T\d{{6}}_{d(r.t3_utc)}T")
        f12 = [p for n, p in files.items() if pat12.search(n)]
        f23 = [p for n, p in files.items() if pat23.search(n)]
        key = f"{r.glacier}|{r.track}|{d(r.t1_utc)}"
        if not f12 or not f23:
            rows.append({"key": key, "status": "missing GUNW"}); continue
        tr = Transformer.from_crs(4326, 3031, always_xy=True)
        cx, cy = tr.transform(SITES[r.glacier][1], SITES[r.glacier][0])
        res = common_dd(f12[0], f23[0], cx, cy)
        if res is None:
            rows.append({"key": key, "status": "site outside frame"}); continue
        xs, ys, dd, coh, epsg = res
        trg = None
        P, _ = profiles_for(SITES[r.glacier], [], tr)
        # reference GL in the GUNW CRS, near the site
        cand = []
        for g, m in gl_feats:
            gx = g                                   # GL file is already EPSG:3031
            if gx.distance(P) < RADIUS_KM * 1e3:
                cand.append((gx, m))
        if not cand:
            rows.append({"key": key, "status": "no reference GL near site"}); continue
        def yr(m):
            try:
                return float(str(m.get("Year"))[:4])
            except Exception:
                return 0.0
        from shapely.geometry import box as _box
        foot = _box(xs.min(), ys.min(), xs.max(), ys.max()).intersection(P.buffer(RADIUS_KM * 1e3))
        cover = lambda g: g.intersection(foot).length / 1e3     # km of this line inside data near the site
        usable = [c for c in cand if cover(c[0]) >= 10.0]
        if not usable:
            rows.append({"key": key, "status": "no reference GL with >=10 km inside the data"}); continue
        ref_any = max(usable, key=lambda c: (yr(c[1]), cover(c[0])))
        s1 = [c for c in cand if "sentinel" in str(c[1].get("Sensor", "")).lower()]
        ref_s1 = max(s1, key=lambda c: (yr(c[1]), str(c[1].get("Date_1")))) if s1 else None
        line = ref_any[0].intersection(foot)
        lines = [g for g in getattr(line, "geoms", [line]) if g.geom_type == "LineString" and g.length > 2e3]
        obs = []
        for ln in lines:
            n = int(ln.length // (PROFILE_STEP_KM * 1e3))
            for k in range(n):
                p0 = ln.interpolate(k * PROFILE_STEP_KM * 1e3); p1 = ln.interpolate(min(k * PROFILE_STEP_KM * 1e3 + 50, ln.length))
                t = np.array([p1.x - p0.x, p1.y - p0.y]); t /= np.hypot(*t) or 1
                nrm = np.array([-t[1], t[0]])
                s = np.arange(-L_LAND, L_SEA + DS, DS)
                X = p0.x + nrm[0] * s * 1e3; Y = p0.y + nrm[1] * s * 1e3
                # Average the complex DD over a strip +-STRIP_KM along the GL at each distance s:
                # the tidal phase is nearly constant along the line, the noise is not.
                acc = np.zeros(s.shape, complex); cnt = np.zeros(s.shape)
                for off in np.arange(-STRIP_KM, STRIP_KM + 1e-6, 0.08):
                    zz = sample(xs, ys, dd, X + t[0] * off * 1e3, Y + t[1] * off * 1e3)
                    ok_ = np.isfinite(zz.real) & (np.abs(zz) > 0)
                    acc[ok_] += zz[ok_]; cnt[ok_] += 1
                z = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan + 0j)
                c = np.abs(z)                         # coherence of the strip-averaged DD
                good = np.isfinite(z.real) & (cnt > 0)
                if good.mean() < 0.8:
                    continue
                zn = z[good] / np.where(c[good] > 0, c[good], 1)   # unit phasors
                fit = fit_profile(s[good], zn, c[good] ** 2)
                d_s1 = None
                if ref_s1 is not None:
                    prof = LineString([(X[0], Y[0]), (X[-1], Y[-1])])
                    inter = prof.intersection(ref_s1[0])
                    if not inter.is_empty:
                        ip = inter if inter.geom_type == "Point" else list(getattr(inter, "geoms", [inter]))[0]
                        s_ref = ((ip.x - p0.x) * nrm[0] + (ip.y - p0.y) * nrm[1]) / 1e3
                        d_s1 = fit["orient"] * (fit["s0"] - s_ref)   # + = NISAR hinge seaward of the S1 line
                prof_rows.append({"key": key, "k": k, **fit, "coh_median": float(np.nanmedian(c[good])),
                                  "n_obs": abs(fit["A_fr"]), "d_s1_km": d_s1})
                obs.append(prof_rows[-1])
        # A profile counts when the flexure fit is coherent and clearly beats the no-flexure fit.
        ok = [o for o in obs if o["obj"] >= OBJ_MIN and o["obj"] - o["null"] >= GAIN_MIN]
        coh_all = [o["coh_median"] for o in obs]
        rows.append({
            "key": key, "glacier": r.glacier, "track": r.track, "t2": d(r.t2_utc),
            "n_pred": r.fringes, "h_dd_pred_m": r.h_dd_m, "status": "ok" if ok else "no fit",
            "n_profiles": len(obs), "n_good": len(ok),
            "strip_coh_median": float(np.median(coh_all)) if coh_all else None,
            "null_obj_median": float(np.median([o["null"] for o in obs])) if obs else None,
            "best_obj_median": float(np.median([o["obj"] for o in obs])) if obs else None,
            "n_obs_median": float(np.median([o["n_obs"] for o in ok])) if ok else None,
            "n_obs_p25": float(np.percentile([o["n_obs"] for o in ok], 25)) if ok else None,
            "n_obs_p75": float(np.percentile([o["n_obs"] for o in ok], 75)) if ok else None,
            "A_sign_median": float(np.sign(np.median([o["A_fr"] for o in ok]))) if ok else None,
            "ref_gl": f"{ref_any[1].get('Sensor')} {ref_any[1].get('Date_1')}",
            "ref_s1": f"{ref_s1[1].get('Sensor')} {ref_s1[1].get('Date_1')}" if ref_s1 else None,
            "d_s1_km_median": float(np.nanmedian([o["d_s1_km"] for o in ok if o["d_s1_km"] is not None]))
            if any(o["d_s1_km"] is not None for o in ok) else None,
        })
        np.savez_compressed(f"out/dd_{key.replace('|','_').replace(' ','')}.npz",
                            phase=np.angle(dd).astype(np.float32), coh=coh.astype(np.float16), x=xs, y=ys)
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv("out/dd_results.csv", index=False)
    pd.DataFrame(prof_rows).to_csv("out/dd_profiles.csv", index=False)


if __name__ == "__main__":
    main()
