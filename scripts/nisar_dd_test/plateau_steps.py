"""Plateau-difference estimator of the observed DD fringe count (replaces the per-pixel step median).

For each (glacier, track, frame) group the reference belt (unwrap_dd.py) gives locations p and
gradient directions g. Ridge locations are those whose fringe density in the reference triplet is in
the top 30 % of the belt. At each ridge location and for every triplet of the group:

    step = | median(unw in a 1.5 km disc at p + 5 km g) - median(unw in a 1.5 km disc at p - 5 km g) | / 2 pi

with each disc at least half filled by coherent, SNAPHU-connected pixels (MIN_FRAC), which
excludes locations on the edge of a coherent patch. n_obs = median over ridge
locations, IQR reported. Background: the same estimator with both discs shifted 15 km along g
(away from the belt), which measures broad ionospheric or atmospheric gradients over 10 km.
A triplet is measurable when >= MIN_LOC ridge locations have both discs valid.
"""
import numpy as np
import pandas as pd

D_KM, R_KM, SHIFT_KM = 5.0, 1.5, 15.0
MIN_FRAC, MIN_LOC = 0.5, 10     # each disc at least half coherent; >= 10 ridge locations
RIDGE_PCT = 70


def key2f(k):
    return k.replace("|", "_").replace(" ", "")


def disc_median(z, X, Y, x, y):
    dx = x[1] - x[0]; dy = y[1] - y[0]
    r = int(round(R_KM * 1e3 / abs(dx)))
    j = int(round((X - x[0]) / dx)); i = int(round((Y - y[0]) / dy))
    if i - r < 0 or j - r < 0 or i + r >= len(y) or j + r >= len(x):
        return np.nan, 0
    sub = z[i - r:i + r + 1, j - r:j + r + 1]
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    inside = xx ** 2 + yy ** 2 <= r * r
    v = sub[inside & np.isfinite(sub)]
    return (float(np.median(v)), v.size / inside.sum()) if v.size else (np.nan, 0.0)


def measure(unw, x, y, pts, shift=0.0):
    out = []
    for xi, yi, ux, uy in pts:
        cx, cy = xi + shift * 1e3 * ux, yi + shift * 1e3 * uy
        a, na = disc_median(unw, cx + D_KM * 1e3 * ux, cy + D_KM * 1e3 * uy, x, y)
        b, nb = disc_median(unw, cx - D_KM * 1e3 * ux, cy - D_KM * 1e3 * uy, x, y)
        if na >= MIN_FRAC and nb >= MIN_FRAC:
            out.append(abs(a - b) / (2 * np.pi))
    return np.array(out)


def main():
    r = pd.read_csv("out/unw_results.csv")
    rows = []
    for _, row in r.iterrows():
        rec = {"key": row.key, "glacier": row.glacier, "track": row.track, "t2": row.t2,
               "n_pred": row.n_pred, "belt_from": row.belt_from}
        if not isinstance(row.belt_from, str):
            rows.append({**rec, "status": "no coherent fringe belt"}); continue
        z = np.load(f"out/unw_{key2f(row.key)}.npz"); ref = np.load(f"out/unw_{key2f(row.belt_from)}.npz")
        b = np.load(f"out/belt_{key2f(row.belt_from)}.npz")
        x, y = ref["x"], ref["y"]
        dx, dy = x[1] - x[0], y[1] - y[0]
        jj = np.round((b["xi"] - x[0]) / dx).astype(int); ii = np.round((b["yi"] - y[0]) / dy).astype(int)
        dens = ref["dens"].astype(float)[ii, jj]
        sel = dens >= np.percentile(dens, RIDGE_PCT)
        pts = list(zip(b["xi"][sel], b["yi"][sel], b["ux"][sel], b["uy"][sel]))
        unw = z["unw"].astype(float)
        st = measure(unw, z["x"], z["y"], pts)
        bg = measure(unw, z["x"], z["y"], pts, shift=SHIFT_KM) if len(pts) else np.array([])
        bg2 = measure(unw, z["x"], z["y"], pts, shift=-SHIFT_KM) if len(pts) else np.array([])
        bgall = np.concatenate([bg, bg2])
        ok = st.size >= MIN_LOC
        rows.append({**rec, "ridge_locations": len(pts), "valid_locations": int(st.size),
                     "n_obs": float(np.median(st)) if ok else None,
                     "n_obs_p25": float(np.percentile(st, 25)) if ok else None,
                     "n_obs_p75": float(np.percentile(st, 75)) if ok else None,
                     "n_bg": float(np.median(bgall)) if bgall.size >= MIN_LOC else None,
                     "status": "measured" if ok else "too few coherent ridge locations"})
    out = pd.DataFrame(rows)
    out.to_csv("out/plateau_results.csv", index=False)
    print(out.round(2).sort_values(["glacier", "n_pred"]).to_string())


if __name__ == "__main__":
    main()
