"""Write the 80 m DD image (phase, coherence of the multilooked DD) around the site for every selected triplet."""
import glob, os, re, sys
import numpy as np, pandas as pd
from pyproj import Transformer
import dd_test as d
sel = pd.read_csv("gunw_test_selection.csv", parse_dates=["t1_utc", "t2_utc", "t3_utc"])
files = {os.path.basename(p)[:-3]: p for p in glob.glob("gunw/*.h5")}
tr = Transformer.from_crs(4326, 3031, always_xy=True)
os.makedirs("out", exist_ok=True)
for r in sel.itertuples():
    f8 = lambda t: t.strftime("%Y%m%d")
    pat = lambda a, b: re.compile(rf"GUNW_\d{{3}}_{r.track:03d}_[AD]_{r.frame:03d}_.*_{f8(a)}T\d{{6}}_\d{{8}}T\d{{6}}_{f8(b)}T")
    f12 = [p for n, p in files.items() if pat(r.t1_utc, r.t2_utc).search(n)]
    f23 = [p for n, p in files.items() if pat(r.t2_utc, r.t3_utc).search(n)]
    key = f"{r.glacier}|{r.track}|{f8(r.t1_utc)}"
    out = f"out/dd_{key.replace('|', '_').replace(' ', '')}.npz"
    if not f12 or not f23 or os.path.exists(out):
        print(key, "skip", bool(f12), bool(f23), os.path.exists(out), flush=True); continue
    cx, cy = tr.transform(d.SITES[r.glacier][1], d.SITES[r.glacier][0])
    res = d.common_dd(f12[0], f23[0], cx, cy)
    if res is None:
        print(key, "site outside frame", flush=True); continue
    xs, ys, dd, coh, _ = res
    np.savez_compressed(out, phase=np.angle(dd).astype(np.float32), coh=np.abs(dd).astype(np.float16),
                        cohm=coh.astype(np.float16), x=xs, y=ys)
    print(key, "ok", dd.shape, flush=True)
