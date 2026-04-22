# tidal-insar-sim — Project Brief for AI Coding Agent

**Revision 2** — full-scope version per user selections:
- Deliverables: **library + CLI + Streamlit web UI** (all three, in that order)
- Tide backend: **local CATS2008 grid files via pyTMD**
- Outputs: **fringe probability · synthetic fringe maps · acquisition planner · batch comparison · GeoJSON/KML**

---

## 0. Context for the agent

The user is **Pietro Milillo**, an InSAR researcher at the University of Houston specializing in grounding-line monitoring with SAR. This tool is built to help with **NISAR mission planning over Antarctic grounding zones**, to support his book project on AI-assisted scientific workflows, and to eventually become a citable published tool (JOSS/Zenodo).

The physics and numerical methods have been prototyped and validated in four reference scripts (`reference/prototype_*.py`). The agent **must read all four before writing any code** — the physics there is already validated against peer-reviewed literature.

**Critical rule — do not invent physics or tide constants.** Every tidal amplitude and every formula must trace to either (a) CATS2008 (via pyTMD) or (b) a specific peer-reviewed reference in section 11. If the agent is ever unsure, it flags the uncertainty explicitly to the user rather than guessing.

---

## 1. Functional architecture

The project has three concentric layers, built in strict order:

```
┌─────────────────────────────────────────────────┐
│  Layer 3: Streamlit web UI (tidal_insar_sim.web) │
│  ┌─────────────────────────────────────────────┐ │
│  │  Layer 2: CLI (tidal_insar_sim.cli)          │ │
│  │  ┌────────────────────────────────────────┐  │ │
│  │  │  Layer 1: Library (tidal_insar_sim)    │  │ │
│  │  │  Sensor, Site, Simulator, Report       │  │ │
│  │  │  tides/, physics/, synthesis/          │  │ │
│  │  └────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Layers 2 and 3 may not bypass Layer 1.** The CLI calls the library's public API. The Streamlit app calls the library's public API. No physics lives in the CLI or web layers.

### 1.1 Library API (Layer 1) — primary interface

```python
from tidal_insar_sim import Sensor, Site, Simulator

sensor = Sensor.NISAR_L                          # preset
# or: Sensor(name="Custom", wavelength_m=0.24, repeat_days=12, incidence_deg=39)

site = Site.from_coords(lat=-75.0, lon=-106.0,
                         name="Thwaites_GL",
                         ice_thickness_m=400)

sim = Simulator(sensor=sensor, site=site)
report = sim.sweep_triplets(step_hours=1.0, duration_days=29.53)

print(report.summary())
# {'P_usable': 0.24, 'P_robust': 0.06, 'P_null': 0.11,
#  'fringe_mean': 2.07, 'fringe_median': 1.66, 'verdict': 'marginal', ...}

# Synthetic fringe maps for representative triplets
fringe_map_strong = sim.synthesize_ddinsar(triplet_start_hours=report.best_triplet_h())
fringe_map_null   = sim.synthesize_ddinsar(triplet_start_hours=report.worst_triplet_h())

# Acquisition planner
best_dates = report.recommended_triplets(
    start_date=datetime(2026, 6, 1),
    end_date=datetime(2026, 9, 1),
    n=10,
    min_fringes=3.0,
)

# Batch
from tidal_insar_sim import batch_compare
df = batch_compare(
    sites=[Site.THWAITES, Site.PIG, Site.RUTFORD],
    sensors=[Sensor.NISAR_L, Sensor.SENTINEL_1, Sensor.ALOS_4],
)

# Exports
report.to_csv("thwaites_nisar.csv")
report.to_geojson("thwaites_nisar.geojson")
report.to_kml("thwaites_nisar.kml")
fringe_map_strong.to_geotiff("fringe_strong.tif")   # georeferenced EPSG:3031
```

### 1.2 CLI (Layer 2)

```bash
# Single site
tidal-insar-sim analyze \
    --sensor NISAR-L \
    --lat -75.0 --lon -106.0 \
    --ice-thickness 400 \
    --duration 29.53 \
    --output ./out/thwaites_nisar

# Batch
tidal-insar-sim batch \
    --sites sites.yaml \
    --sensors NISAR-L SENTINEL-1 ALOS-4 \
    --output ./out/batch_comparison

# Acquisition planner only
tidal-insar-sim plan \
    --preset THWAITES --sensor NISAR-L \
    --start 2026-06-01 --end 2026-09-01 \
    --n 10 --min-fringes 3

# Generate synthetic DDInSAR map
tidal-insar-sim synthesize \
    --preset THWAITES --sensor NISAR-L \
    --triplet-start-date 2026-07-15T00:00 \
    --output thwaites_ddinsar.tif

# Setup CATS2008 on first use
tidal-insar-sim setup-cats --source ghcat
```

Use `rich` for pretty console output, `click` for the CLI framework.

### 1.3 Streamlit web UI (Layer 3)

One multi-page app: `streamlit run tidal_insar_sim/web/app.py`

Pages:
1. **Map** — Folium/streamlit-folium map of Antarctica with MEaSUREs GL v2 underlay. User clicks → sets lat/lon.
2. **Configure** — sensor dropdown + custom inputs, ice thickness, sweep parameters.
3. **Single-site report** — probability gauge, sweep curve, histogram, 2 synthetic fringe maps (STRONG + NULL), acquisition planner table with download buttons.
4. **Batch comparison** — multi-site, multi-sensor comparison tables and heatmaps.
5. **Export** — download all results (CSV, GeoJSON, KML, GeoTIFF, .ics, PDF report).

Preserve session state so users can iterate without re-running CATS lookups.

---

## 2. Inputs

### 2.1 `Sensor` class and presets

```python
@dataclass(frozen=True)
class Sensor:
    name: str
    wavelength_m: float           # λ
    repeat_days: float            # temporal baseline
    incidence_deg: float          # nominal θ
    polarization: str = "HH"
    provider: str = ""

    @property
    def half_wavelength_m(self) -> float: return self.wavelength_m / 2
    @property
    def fringe_los_cm(self) -> float: return self.half_wavelength_m * 100
```

Shipped presets (`tidal_insar_sim.sensors.presets`):

| Preset | λ (m) | repeat (d) | θ (°) | notes |
|---|---|---|---|---|
| NISAR_L | 0.2360 | 12 | 39 | primary target |
| SENTINEL_1_SINGLE | 0.0556 | 12 | 39 | post-S1B-loss |
| SENTINEL_1_DUAL | 0.0556 | 6 | 39 | pre-S1B-loss |
| ALOS_2 | 0.2360 | 14 | 34 | JAXA |
| ALOS_4 | 0.2360 | 14 | 34 | JAXA L-band SAR |
| COSMO_SKYMED | 0.0312 | 4 | 32 | CSG constellation |
| TERRASAR_X | 0.0311 | 11 | 36 | DLR |
| RADARSAT_CONSTELLATION | 0.0556 | 4 | 34 | CSA, RCM |
| UMBRA_X | 0.0312 | 0 | 40 | tasked, no fixed repeat |

### 2.2 `Site` class

```python
@dataclass
class Site:
    lat: float                      # WGS-84 decimal degrees
    lon: float                      # WGS-84 decimal degrees
    name: str
    ice_thickness_m: float = 400.0
    E_pa: float = 0.88e9            # Young's modulus
    nu: float = 0.30                # Poisson
    rho_water: float = 1028.0       # kg/m³

    # populated lazily from CATS2008
    tide_constants: dict[str, tuple[float, float]] | None = None
    tide_source: str = "unset"      # e.g. "CATS2008_v2023@-75.00,-106.00"

    @classmethod
    def from_coords(cls, lat, lon, name=None, **kw) -> "Site": ...

    def flexural_parameter_beta(self) -> float: ...
    def limit_of_flexure_m(self) -> float: ...
```

Named presets in `sites_presets.toml`:

```python
THWAITES    = Site(lat=-75.00, lon=-106.00, name="Thwaites_GL",    ice_thickness_m=450)
PIG         = Site(lat=-74.95, lon=-100.70, name="Pine_Island_GL", ice_thickness_m=500)
RUTFORD     = Site(lat=-78.50, lon=-83.00,  name="Rutford_GL",     ice_thickness_m=2000)
ROSS_GZ16   = Site(lat=-84.30, lon=-163.00, name="Whillans_GZ16",  ice_thickness_m=720)
PETERMANN   = Site(lat=+80.50, lon=-60.50,  name="Petermann_GL",   ice_thickness_m=600)
JAKOBSHAVN  = Site(lat=+69.10, lon=-49.55,  name="Jakobshavn_GL",  ice_thickness_m=900)
TOTTEN      = Site(lat=-66.90, lon=116.00,  name="Totten_GL",      ice_thickness_m=1500)
POPE        = Site(lat=-74.70, lon=-113.00, name="Pope_GL",        ice_thickness_m=600)
SMITH       = Site(lat=-74.60, lon=-112.00, name="Smith_GL",       ice_thickness_m=700)
KOHLER      = Site(lat=-75.30, lon=-114.80, name="Kohler_GL",      ice_thickness_m=650)
```

---

## 3. Tide backend — local CATS2008 grids via pyTMD

### 3.1 Acquisition (one-time setup)

```bash
tidal-insar-sim setup-cats --source [esr|ghcat]
```

- **ghcat** (default): fetch from Chad Greene's Tide-Model-Driver GitHub release. No registration, MIT-licensed.
- **esr**: fetch from Earth and Space Research. May require registration — prompt user interactively.

Store in `~/.tidal_insar_sim/tides/CATS2008/`. Verify SHA256 against version-pinned checksums in `data/tide_checksums.toml`.

Expected files after setup:
```
~/.tidal_insar_sim/tides/CATS2008/
├── grid_CATS2008                   # OTIS grid file
├── hf.CATS2008.out                 # harmonic constants (heights)
├── uv.CATS2008.out                 # currents (not needed by us)
└── Model_CATS2008                  # control file listing constituents
```

Reject silently-corrupted downloads. Retry with exponential backoff, max 3 retries, then raise `TidalDataUnavailable`.

### 3.2 Extraction at a point

```python
# in tidal_insar_sim/tides/cats2008.py

DEFAULT_CONSTITUENTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]

def extract_constants(lat: float, lon: float,
                      constituents: list[str] = DEFAULT_CONSTITUENTS,
                      ) -> dict[str, tuple[float, float]]:
    """
    Return {constituent_name: (amplitude_m, phase_deg)} at (lat, lon)
    using bilinear interpolation on the CATS2008 grid, via pyTMD.

    Raises TidalDataUnavailable if CATS2008 not installed.
    Raises OutsideDomain if (lat, lon) is outside CATS2008 domain.
    Raises OnLand if the nearest grid cell is masked as grounded ice.
    """
```

Under the hood, use `pyTMD.io.OTIS.extract_constants(ilon, ilat, grid_file, model_file, type='z')`. The agent must read the pyTMD docs and source for the exact call signature — do not trust this brief for the API details.

### 3.3 Caching

SQLite at `~/.tidal_insar_sim/site_cache.db`:

```sql
CREATE TABLE site_cache (
    lat_rounded      REAL,      -- rounded to 0.01°
    lon_rounded      REAL,
    cats_version     TEXT,
    constituent      TEXT,
    amplitude_m      REAL,
    phase_deg        REAL,
    extracted_at     TIMESTAMP,
    PRIMARY KEY (lat_rounded, lon_rounded, cats_version, constituent)
);
```

Invalidate cache when CATS2008 version changes.

### 3.4 Synthesis

```python
# in tidal_insar_sim/tides/synthesis.py

def synthesize_tide(t_hours: np.ndarray,
                    constants: dict[str, tuple[float, float]],
                    reference_epoch: datetime = EPOCH_2000,
                    nodal_correction: bool = True) -> np.ndarray:
    """
    h(t) = Σᵢ fᵢ(t) · Aᵢ · cos(ωᵢ·t + (V₀+u)ᵢ − φᵢ)

    where fᵢ(t), uᵢ(t) are the 18.6-year nodal amplitude and phase
    corrections (IERS conventions, via pyTMD.arguments).
    """
```

### 3.5 Validation gate (tests/test_tide_validation.py)

Before the library is considered "working", this test must pass:

```python
BENCHMARKS = [
    # name,          lat,     lon,    rms_m,  tol,  source
    ("McMurdo",     -77.85,  166.67,  0.39,   0.08, "UHSLC"),
    ("Rutford_GL",  -78.50,  -83.00,  0.95,   0.15, "King & Padman 2005"),
    ("Cape_Roberts",-77.03,  163.19,  0.42,   0.08, "Pfeffer 1993"),
    ("Halley",      -75.58,  -26.65,  0.55,   0.10, "Holgate 2013 GLOSS"),
    ("Davis",       -68.58,   77.97,  0.75,   0.15, "BAS tide gauge"),
]
```

Each benchmark: synthesize a 1-year tide time series, compute std, compare to `rms_m`. **CI fails if any benchmark misses its tolerance.**

---

## 4. DDInSAR physics core

### 4.1 Flexure
```
D       = E · H³ / [12 (1 − ν²)]
β       = (ρ_w · g / 4D)^(1/4)
w(s, t) = h(t) · [1 − exp(−βs)·(cos βs + sin βs)],   s ≥ 0
w(s, t) = 0,                                          s < 0
L_flex  = π / β
```

Ref: Rignot 2011 GRL; Walker 2013 EPSL; Fricker & Padman 2006 GRL.

Unit test: for H=400 m, E=0.88 GPa, ν=0.30, expect β = 8.362×10⁻⁴ m⁻¹ and L_flex = 3757 m within 0.1%.

### 4.2 Double-difference construction
```
h_DD(x) = w(x, t₁) − 2 w(x, t₂) + w(x, t₃)
φ_DD(x) = −(4π / λ) · cos(θ) · h_DD(x)
n_fringes_peak = |h_DD_max| · cos(θ) / (λ/2)
```

### 4.3 Synthetic fringe map
2D field over a box centered on site, default 15 km × 10 km at 15 m pixel. Output: georeferenced complex DDInSAR (`complex64` + GeoTIFF, EPSG:3031 Antarctic or 3413 Arctic). Coherence-dependent speckle (multi-look 8). Cyclic HSV colormap for wrapped phase.

### 4.4 Rigid triplet sweep
Same as validated prototype. Step Δt from 0 to `duration_days` by `step_hours`.

### 4.5 Acquisition planner
Given `(start_date, end_date, n, min_fringes)`:
1. For each candidate Δt at 1 h resolution, predict fringe count.
2. Identify local maxima above `min_fringes`.
3. Return top `n` as DataFrame: UTC dates, predicted |fringes|, tide values at all 3 epochs, **confidence** (from ±1 h Monte Carlo stability, see `reference/prototype_monte_carlo.py`).
4. Export as `.ics` calendar file (Outlook/Google importable).

---

## 5. Outputs (five deliverables)

### 5.1 Fringe-count probability — `report.summary()`
```python
{
    "P_usable_ge_3fr":   0.24,
    "P_robust_ge_5fr":   0.06,
    "P_null_lt_0p5fr":   0.11,
    "fringe_mean":       2.07,
    "fringe_median":     1.66,
    "fringe_max":        6.88,
    "hDD_range_m":       [-1.04, +0.91],
    "verdict":           "marginal",
    "verdict_rationale": "P(≥3 fr) = 24% — NISAR 12-day repeat aliases "
                         "K1 at this diurnal-dominated site",
}
```

### 5.2 Synthetic fringe maps
`Simulator.synthesize_ddinsar(triplet_start)` → `FringeMap` object:
- `.array` (complex64, H×W)
- `.geotransform` (GDAL 6-tuple)
- `.crs` (EPSG:3031 / 3413)
- `.to_geotiff(path)` — readable by QGIS, ISCE, GMTSAR
- `.to_png(path, cmap="hsv")`
- `.to_kmz(path)` — Google Earth overlay

### 5.3 Acquisition recommendations
`pandas.DataFrame`:

| rank | triplet_start_utc | t1_date | t2_date | t3_date | predicted_fringes | tide_h1_m | tide_h2_m | tide_h3_m | confidence |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-14 08:00 | 2026-07-14 | 2026-07-26 | 2026-08-07 | 6.72 | -0.77 | -1.27 | -0.74 | 0.94 |

Also exports `.ics` calendar file.

### 5.4 Batch comparison
`batch_compare(sites, sensors)` → `pandas.DataFrame` one row per (site, sensor), with all summary fields + metadata. Persist to Parquet. Heatmap: `batch.plot_heatmap(metric="P_usable_ge_3fr")`.

### 5.5 GeoJSON / KML export
GeoJSON FeatureCollection:
- Point at (lat, lon) with properties = summary
- Polygon (64-vertex circle approximation) of radius L_flex
- `classification` ∈ {excellent, good, marginal, poor} with QGIS color hint

KML with same content + Google-Earth-ready styling.

---

## 6. Repository layout

```
tidal-insar-sim/
├── pyproject.toml
├── README.md
├── LICENSE                 # MIT
├── CHANGELOG.md
├── tidal_insar_sim/
│   ├── __init__.py
│   ├── sensor.py
│   ├── site.py
│   ├── simulator.py
│   ├── report.py
│   ├── batch.py
│   ├── planner.py
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── flexure.py
│   │   ├── ddinsar.py
│   │   └── fringe.py
│   ├── tides/
│   │   ├── __init__.py
│   │   ├── cats2008.py     # pyTMD wrapper
│   │   ├── synthesis.py
│   │   ├── cache.py        # SQLite
│   │   └── download.py     # setup-cats
│   ├── io/
│   │   ├── __init__.py
│   │   ├── geojson_export.py
│   │   ├── kml_export.py
│   │   ├── geotiff_export.py
│   │   └── ics_export.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py         # click entry
│   │   ├── analyze.py
│   │   ├── batch.py
│   │   ├── plan.py
│   │   ├── synthesize.py
│   │   └── setup_cats.py
│   ├── web/
│   │   ├── __init__.py
│   │   ├── app.py          # streamlit entry
│   │   ├── pages/
│   │   │   ├── 1_Map.py
│   │   │   ├── 2_Configure.py
│   │   │   ├── 3_Single_Site.py
│   │   │   ├── 4_Batch.py
│   │   │   └── 5_Export.py
│   │   └── components/
│   │       ├── map_picker.py
│   │       └── fringe_viewer.py
│   └── data/
│       ├── sensors_presets.toml
│       ├── sites_presets.toml
│       ├── tide_checksums.toml
│       └── benchmarks.toml
├── reference/                  # the 4 validated prototype scripts
│   ├── prototype_v1_three_date.py
│   ├── prototype_sweep.py
│   ├── prototype_sites.py
│   └── prototype_monte_carlo.py
├── tests/
│   ├── test_physics.py
│   ├── test_tide_validation.py
│   ├── test_cats_integration.py    # skipped if CATS2008 not installed
│   ├── test_cache.py
│   ├── test_simulator.py
│   ├── test_batch.py
│   ├── test_planner.py
│   ├── test_geojson_export.py
│   └── test_cli.py
└── docs/
    ├── index.md
    ├── quickstart.md
    ├── api.md
    ├── physics.md
    ├── cats2008_setup.md
    └── examples/
        ├── 01_single_site.md
        ├── 02_acquisition_planning.md
        ├── 03_batch_comparison.md
        └── 04_web_app_walkthrough.md
```

---

## 7. Non-negotiables

1. **Physics traceability.** Every formula has a docstring reference to peer-reviewed work.
2. **Tide traceability.** Every tidal amplitude comes from CATS2008 (via pyTMD). No exceptions, no "looked reasonable" values.
3. **Validation gate before higher layers.** `tests/test_tide_validation.py` must pass before any CLI work. CLI must work end-to-end before any Streamlit work.
4. **Fail loudly.** If CATS2008 unavailable, raise `TidalDataUnavailable` with remediation steps. If point is on land, raise `OnLand`. If outside domain, raise `OutsideDomain`.
5. **All physics tested.** Each formula → a dedicated test case.
6. **Git commits per milestone**, with descriptive messages. Never commit failing tests. Never commit large binary blobs (grids live in `~/.tidal_insar_sim/`).
7. **Type hints everywhere**, `mypy --strict` in CI.
8. **Pydantic v2** for config validation.
9. **Single-source-of-truth version** in `__init__.py`, read by `pyproject.toml`.

---

## 8. Milestone plan

### M1 — Core physics, no tides (4 h)
`physics/` module works against synthetic tide arrays. Tests passing.

### M2 — Sensor + Site + Simulator skeleton (2 h)
Instantiate objects with mock tides, call `sweep_triplets()`, get a `TripletReport`.

### M3 — CATS2008 integration (6 h) ⚠ highest-risk
`tidal-insar-sim setup-cats` works; `extract_constants(lat, lon)` returns real values. Cache functional. **Must pass 5 benchmarks.**

### M4 — Synthetic fringe maps + GeoTIFF export (3 h)

### M5 — Acquisition planner (3 h)
DataFrame + `.ics` calendar file.

### M6 — Batch comparison (2 h)

### M7 — GeoJSON / KML exports (2 h)

### M8 — CLI (3 h)

### M9 — Streamlit UI (6 h)

### M10 — Packaging + docs + release (3 h)
`pip install`, JOSS draft, Zenodo DOI.

**Total: ~34 hours.** Checkpoint with user after each milestone.

---

## 9. Test coverage

- Core physics: 100%
- Tide synthesis: 100%
- Simulator: ≥90%
- CLI: ≥80%
- Web UI: smoke tests
- CI: Linux + macOS (M-series)

---

## 10. Coding standards

- Python ≥3.10
- `ruff format` + `ruff check --select ALL --ignore ANN,D`
- `mypy --strict`
- `pytest` + `pytest-cov` + `pytest-xdist`
- `mkdocs-material` for docs
- Core deps: `numpy`, `scipy`, `xarray`, `pyTMD>=3.0`, `pyproj`, `rasterio`, `shapely`, `pydantic>=2`, `click`, `rich`
- Extras: `pandas`, `pyarrow` (batch); `streamlit`, `folium`, `streamlit-folium` (web); `matplotlib`
- **No** `seaborn`, `plotly`, or `bokeh`

---

## 11. References

### Primary tide model
- Padman, Fricker, Coleman, Howard, Erofeeva (2002). *Annals of Glaciology* 34.
- Padman, Siegfried, Fricker (2018). *Reviews of Geophysics* 56.
- Greene et al. (2024). Tide Model Driver 3.0. github.com/chadagreene/Tide-Model-Driver

### Grounding-line DDInSAR / flexure
- Rignot, Mouginot, Scheuchl (2011). *GRL* 38.
- Fricker & Padman (2006). *GRL* 33.
- Walker et al. (2013). *EPSL* 395.
- Milillo et al. (2017). *GRL*.
- Milillo et al. (2019). *Science Advances* 5.
- Zhong et al. (2023). *JGR Earth Surface* 128.
- Begeman et al. (2020). *JGR Oceans* 125.
- Rosier et al. (2020). *The Cryosphere* 14.

### Software
- pyTMD: https://pytmd.readthedocs.io
- Tide-Model-Driver: https://github.com/chadagreene/Tide-Model-Driver

### Reference implementations (in `reference/`)
- `prototype_v1_three_date.py` — validated 3-date DDInSAR physics
- `prototype_sweep.py` — rigid-triplet sweep logic
- `prototype_sites.py` — multi-site comparison
- `prototype_monte_carlo.py` — triplet robustness for confidence metric

---

## 12. How the agent should operate

1. **Read the 4 reference scripts first.** Do not skip.
2. **Work milestone-by-milestone.** Commit with message matching milestone.
3. **Run `pytest` after every change.**
4. **Ask the user** before: adding a new runtime dep; deviating from milestones; architectural changes not specified here; downgrading a non-negotiable.
5. **Fail loudly.** Better to raise than silently fall back to bogus values.
6. **Document assumptions inline.** Any unusual constant: `# Source: Author YEAR`.
7. **When CATS2008 returns NaN** at a site, report it. Don't mask.
8. **Keep core library pure.** Plotting, I/O, visualization in separate submodules.

---

## 13. Success criteria

- [ ] `pip install tidal-insar-sim` works from a built wheel
- [ ] `tidal-insar-sim setup-cats` downloads and verifies CATS2008
- [ ] `tidal-insar-sim analyze --preset THWAITES --sensor NISAR-L` reproduces `reference/prototype_sites.py` within 5% for P_usable, P_null, fringe_median
- [ ] Arbitrary Antarctic (lat, lon) query on Streamlit returns report in <10 s (cached) / <60 s (first-time)
- [ ] All 5 exports validate against schema validators
- [ ] `mypy --strict` passes
- [ ] `pytest` passes on Linux + macOS with ≥90% library coverage
- [ ] README has 3 worked examples including NISAR/Thwaites
- [ ] JOSS-submission-ready `paper.md` in repo root
- [ ] Zenodo DOI via v0.1.0 GitHub release

---

## 14. Out of scope for v0.1

- Ionospheric phase modeling (L-band) → v0.2
- Viscoelastic (Maxwell) ice rheology → v0.3
- GL migration dynamics → v0.3
- Global sites beyond Antarctica/Greenland → v1.0
- Real SAR data ingest (this is a simulator)
- ICESat-2 ATL14/15 ice-thickness auto-lookup → v0.2 nice-to-have
