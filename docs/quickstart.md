# Quick start

## Install

```bash
git clone https://github.com/pmilillo/tidal-insar-sim
cd tidal-insar-sim
pip install -e .[dev]
```

Python ≥ 3.10 is required. The heavy runtime deps are `pyTMD`, `netCDF4`,
`rasterio`, and `streamlit`.

## Install the tide model (once)

CATS2008 must be downloaded manually from USAP-DC. See
[CATS2008 setup](cats2008_setup.md) for the full walk-through.

```bash
tidal-insar-sim setup-cats --path ~/Downloads/CATS2008.zip
```

## Run the sweep — Python

```python
from tidal_insar_sim import Sensor, Site, Simulator

sim = Simulator(sensor=Sensor.NISAR_L, site=Site.RUTFORD)
report = sim.sweep_triplets()
print(report.summary())
```

Typical output (Rutford, NISAR-L, 29.53-day sweep at 1 h step):

```python
{'site': 'Rutford_GL',
 'sensor': 'NISAR-L',
 'n_triplets': 710,
 'P_usable_ge_3fr': 0.869,
 'P_robust_ge_5fr': 0.769,
 'P_null_lt_0p5fr': 0.017,
 'fringe_mean': 9.12,
 'fringe_median': 9.60,
 'fringe_max': 20.59,
 'hDD_range_m': [-3.13, 3.11],
 'verdict': 'excellent'}
```

## Run the sweep — CLI

```bash
tidal-insar-sim analyze --sensor NISAR-L --preset RUTFORD --output out/rutford
```

```
        tidal-insar-sim analyze  -  Rutford_GL / NISAR-L
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ metric                          ┃ value        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ verdict                         │ excellent    │
│ P(>= 3 fringes) usable          │        86.9% │
│ P(>= 5 fringes) robust          │        76.9% │
│ fringe mean                     │         9.12 │
└─────────────────────────────────┴──────────────┘
Outputs written to out/rutford.
```

The directory contains `summary.json`, `sweep.csv`, `site.geojson`,
`site.kml`.

## Plan acquisitions

```bash
tidal-insar-sim plan \
    --sensor NISAR-L --preset RUTFORD \
    --start 2026-06-01T00:00 --end 2026-09-01T00:00 \
    --top-n 10 --min-fringes 5 \
    --output out/rutford_plan
```

Writes `out/rutford_plan.csv` + `out/rutford_plan.ics`.

## Generate a synthetic fringe map

```bash
tidal-insar-sim synthesize \
    --sensor NISAR-L --preset RUTFORD \
    --triplet-start-date 2026-06-02T00:00 \
    --output out/rutford_best.tif
```

The GeoTIFF has three bands: wrapped phase (radians), coherence, and
vertical h_DD (metres). CRS is EPSG:3031 (Antarctic polar stereographic).

## Open the web UI

```bash
streamlit run tidal_insar_sim/web/app.py
```

Click through to the **Map** page, pick a preset or click the map, adjust
the sensor on **Configure**, then hit **Run sweep** on **Single Site**.
