# tidal-insar-sim

[![tests](https://img.shields.io/badge/tests-133%20passing-brightgreen)](tests/)
[![mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

**DDInSAR fringe-count simulator for tidally flexed grounding zones.**

Given a SAR sensor (NISAR, Sentinel-1, ALOS-2/4, CSG, TerraSAR-X, …) and an
Antarctic grounding-line site, `tidal-insar-sim` predicts how many wrapped
double-difference InSAR fringes a 3-date acquisition triplet will produce,
using CATS2008 tide forcing and elastic-plate flexure physics. Built primarily
for **NISAR mission planning**.

> *"Of all possible NISAR triplet phases at Rutford, 87% yield ≥3 usable
> fringes; the best triplet on 2026-06-02 delivers 21 fringes with 100%
> confidence against ±1 h acquisition jitter."* — output of `tidal-insar-sim
> analyze --sensor NISAR-L --preset RUTFORD`.

## Features

- **Library API** — `Sensor`, `Site`, `Simulator`, `TripletReport`, plus
  synthetic fringe-map and acquisition-planner modules.
- **CLI** — `setup-cats`, `analyze`, `plan`, `synthesize`, `batch` with
  rich terminal output.
- **Streamlit web UI** — 5-page app: Map, Configure, Single-Site,
  Batch, Export.
- **Physics traceability** — every formula is docstring-referenced to
  peer-reviewed literature (Rignot 2011, Padman 2002/2018, Walker 2013,
  Milillo 2017/2019). No invented tide constants.
- **Validation gate** — the CATS2008 integration is checked against 8
  AntTG tide-gauge stations with station-specific tolerances.

## Install

```bash
git clone https://github.com/pmilillo/tidal-insar-sim
cd tidal-insar-sim
pip install -e .[dev,docs]
```

Python ≥ 3.10 required. Heavy deps: `pyTMD`, `netCDF4`, `rasterio`,
`streamlit`. See [`pyproject.toml`](pyproject.toml) for the full list.

### CATS2008 tide data (one-time setup)

CATS2008 must be downloaded manually from USAP-DC (reCAPTCHA-gated, ~582 MB):

1. Open <https://www.usap-dc.org/view/dataset/601235>, complete the reCAPTCHA, download `CATS2008.zip`.
2. Run `tidal-insar-sim setup-cats --path ~/Downloads/CATS2008.zip`.

The CLI verifies MD5 `008a30cd08142cb6acc7f7687e22c4a3`, extracts to
`~/.tidal_insar_sim/tides/CATS2008/`, and probes pyTMD to confirm it loads.

## Quick start — library

```python
from tidal_insar_sim import Sensor, Site, Simulator

sim = Simulator(sensor=Sensor.NISAR_L, site=Site.THWAITES)
report = sim.sweep_triplets()
print(report.summary())
# {'verdict': 'good', 'P_usable_ge_3fr': 0.69, 'fringe_mean': 4.13, ...}

from datetime import datetime, timezone
plan = report.recommended_triplets(
    start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
    end_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
    n=10, min_fringes=3.0,
)
plan.to_csv("thwaites_plan.csv", index=False)

fmap = sim.synthesize_ddinsar(triplet_start_hours=report.best_triplet_h())
fmap.to_geotiff("thwaites_best.tif")  # EPSG:3031, 3 bands
```

## Quick start — CLI

```bash
tidal-insar-sim analyze --sensor NISAR-L --preset RUTFORD --output out/rutford

tidal-insar-sim plan \
    --sensor NISAR-L --preset RUTFORD \
    --start 2026-06-01T00:00 --end 2026-09-01T00:00 \
    --top-n 10 --min-fringes 5 \
    --output out/rutford_plan

tidal-insar-sim synthesize \
    --sensor NISAR-L --preset THWAITES \
    --triplet-start-date 2026-07-15T00:00 \
    --output out/thwaites_ddinsar.tif

tidal-insar-sim batch \
    --site-preset THWAITES --site-preset RUTFORD \
    --sensor NISAR-L --sensor SENTINEL-1-DUAL \
    --output out/batch
```

## Quick start — web

```bash
streamlit run tidal_insar_sim/web/app.py
# Browser opens at http://localhost:8501
```

## Supported sites (v0.1)

Antarctic grounding zones only. Named presets: `THWAITES`, `PIG`,
`RUTFORD`, `ROSS_GZ16`, `TOTTEN`, `POPE`, `SMITH`, `KOHLER`. Arbitrary
`(lat, lon)` with `lat < 0` is also supported. Greenland deferred to v0.2.

## Documentation

`docs/`:

- [`quickstart.md`](docs/quickstart.md)
- [`physics.md`](docs/physics.md) — flexure + DDInSAR derivations
- [`cats2008_setup.md`](docs/cats2008_setup.md) — tide-model install
- [`api.md`](docs/api.md) — public Python API
- [`examples/`](docs/examples/) — four worked examples

Build the site: `mkdocs serve`.

## Citing

See `paper.md` and the Zenodo DOI (to be assigned at v0.1.0 release).

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgements

CATS2008 tide model is distributed by Earth and Space Research under DOI
[10.15784/601235](https://doi.org/10.15784/601235); AntTG tide-gauge
amplitudes under DOI [10.15784/601358](https://doi.org/10.15784/601358).
pyTMD by Tyler C. Sutterley et al. (<https://pytmd.readthedocs.io>).
