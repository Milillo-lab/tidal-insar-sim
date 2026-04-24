# CATS2008 setup

CATS2008 (Padman et al., 2002/2018) is a 4-km inverse barotropic tide model
covering the circum-Antarctic ocean *including* the cavities under floating
ice shelves. `tidal-insar-sim` uses it as the authoritative tide source for
every Antarctic site — no tide amplitudes are ever made up.

## Why manual download?

The dataset is hosted at the U.S. Antarctic Program Data Center (USAP-DC)
under DOI [10.15784/601235](https://doi.org/10.15784/601235), behind a
reCAPTCHA. There is no programmatic bypass, and redistribution terms
preclude bundling it with the tool. So the install is a two-step manual
download + local verify/extract.

## Step 1 — Download

1. Open <https://www.usap-dc.org/view/dataset/601235>.
2. Complete the reCAPTCHA and download `CATS2008.zip` (~582 MB).

Expected MD5 hash: `008a30cd08142cb6acc7f7687e22c4a3`.

## Step 2 — Install

```bash
tidal-insar-sim setup-cats --path /path/to/CATS2008.zip
```

The CLI will:

1. Compute the MD5 of the zip and compare to the published hash.
2. Extract to `~/.tidal_insar_sim/tides/CATS2008/`.
3. Probe pyTMD to confirm it can load the grid.

On success it prints a `rich`-styled report:

```
╭────────────────── CATS2008 install report ──────────────────╮
│ zip:         /path/to/CATS2008.zip                          │
│ md5:         008a30cd08142cb6acc7f7687e22c4a3 (OK)          │
│ data_dir:    ~/.tidal_insar_sim/tides                       │
│ already:     False                                          │
│ files:       7 extracted                                    │
│ pyTMD probe: OK                                             │
╰─────────────────────────────────────────────────────────────╯
```

## Verifying the install

Round-trip a tide query at Thwaites:

```python
from tidal_insar_sim.tides import CATSBackend

backend = CATSBackend()
constants = backend.extract_constants(lat=-75.0, lon=-106.0)
print(f"M2 at Thwaites: amp={constants['M2'][0]:.3f} m, phase={constants['M2'][1]:.1f} deg")
```

## Running the AntTG validation gate

The validation tests compare our `CATSBackend` extraction against the
[Antarctic Tide Gauge Database v1](https://doi.org/10.15784/601358) at eight
well-calibrated stations (McMurdo, Davis, Casey, Syowa, Rothera, Halley,
Cape Roberts, Scott Base) plus two documented edge cases (Mawson, Rutford
GPS). Station-specific tolerances are 3–10 cm for M2/K1, which matches the
published CATS2008 RMS against tide gauges.

```bash
pytest -m cats
```

## Custom install location

If you already have CATS2008 elsewhere (e.g. a shared institutional directory),
pass `--data-dir` to `setup-cats`, or instantiate `CATSBackend(data_dir=...)`
directly. The expected layout under `data_dir` is:

```
CATS2008/
├── grid_CATS2008
├── hf.CATS2008.out
├── uv.CATS2008.out
├── Model_CATS2008
├── xy_ll_CATS2008.m
├── CATS2008_README.pdf
└── CATS2008_FileFormat.pdf
```

Only `grid_CATS2008` and `hf.CATS2008.out` are required for elevation
predictions; the others round out the distribution.

## Troubleshooting

### "MD5 mismatch for CATS2008.zip"

The zip is partial or corrupted. Re-download from USAP-DC. If you need to
install from a repacked zip (e.g. from your institution), add `--skip-md5`
— but this bypasses a layer of defence, so use sparingly.

### "CATS2008 data files missing in ..."

The `CATS2008/` subdirectory is empty or incomplete. Run `setup-cats --path
CATS2008.zip` again. The command is idempotent and safe to re-run.

### `OnLand` at a coastal site

Many tide-gauge points are literally *on* grounded ice in CATS2008's 4-km
mask. `CATSBackend` automatically searches the nearest ocean cell within
`extrapolate_cutoff_km` (default 10 km). If the point is farther offshore,
bump the cutoff: `CATSBackend(extrapolate_cutoff_km=30)`.

### `OutsideDomain` north of 30°S

CATS2008 is circum-Antarctic. v0.1 of `tidal-insar-sim` is Antarctic-only —
Greenland and global sites are deferred to v0.2 (see [`CHANGELOG.md`](../CHANGELOG.md)).
