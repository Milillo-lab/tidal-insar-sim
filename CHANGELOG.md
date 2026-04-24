# Changelog

All notable changes to **tidal-insar-sim** are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project targets [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-24

First public release. Three concentric layers, each with its own test suite.

### Library (Layer 1)

- `Sensor`, `Site`, `Simulator`, `TripletReport` core API.
- Physics: elastic plate flexure (`physics/flexure.py`), DDInSAR phase math
  (`physics/ddinsar.py`), rigid-triplet sweep (`physics/fringe.py`).
  Analytic forebulge test `w(L_flex)/h = 1 + exp(-pi)` ≈ 1.0432.
- Tides: `CATSBackend` via pyTMD with nearest-ocean fallback for masked
  GL points; `MixedTide` and `multi_constituent_tide` mock synthesizers;
  harmonic synthesis + analysis round-trip (`tides/synthesis.py`);
  SQLite per-site cache keyed on (lat, lon, model_version).
- Error contract: `TidalDataUnavailable` / `OutsideDomain` / `OnLand`.
- 2-D synthetic fringe maps: `FringeMap` with 3-band LZW-compressed
  GeoTIFF export (EPSG:3031 for Antarctic sites) and HSV PNG export.
  Two-zone coherence model with multi-look speckle.
- Acquisition planner: rigid-triplet scan with strict local-maxima
  extraction, per-triplet confidence from ±1 h Monte Carlo jitter,
  top-N ranking, `.ics` export.
- Batch comparison across (site, sensor) grid; Parquet persistence;
  `pivot_heatmap` for visual comparison.
- GeoJSON / KML export (Point + 64-vertex geodesic `L_flex` polygon)
  with classification color hints.

### CLI (Layer 2)

- `tidal-insar-sim setup-cats --path CATS2008.zip`: MD5-verified install
  of the user-downloaded CATS2008 dataset from USAP-DC (reCAPTCHA-gated
  upstream — no programmatic download).
- `tidal-insar-sim analyze --sensor X [--preset Y | --lat A --lon B] --output DIR`.
  Writes `summary.json`, `sweep.csv`, `site.geojson`, `site.kml`, plus a
  rich-rendered summary table.
- `tidal-insar-sim plan --start ISO --end ISO --top-n N --min-fringes F`.
  CSV + `.ics` calendar output.
- `tidal-insar-sim synthesize --triplet-start-date ISO`: 3-band DDInSAR
  GeoTIFF (wrapped phase / coherence / h_DD).
- `tidal-insar-sim batch [--sites sites.yaml | --site-preset N...] --sensor S...`:
  Parquet + heatmap CSV + rich heatmap table.

### Web UI (Layer 3)

Streamlit multi-page app (`streamlit run tidal_insar_sim/web/app.py`):

- Map page with Folium, preset markers, click-to-pick custom site.
- Configure page: sensor, site overrides, sweep & coherence controls.
- Single-site report: verdict + probability gauges + sweep curve +
  histogram + tide-envelope plot + best/null synthetic fringe maps +
  planner table.
- Batch comparison: multi-select heatmap with RdYlGn gradient.
- Export: download every artifact (CSV, JSON, GeoJSON, KML, GeoTIFF,
  ICS, Parquet).

### Validation

- `tests/test_tide_validation.py`: 8 AntTG tide-gauge stations cross-
  checked against CATS2008-extracted M2 and K1 amplitudes within
  station-specific tolerances (3–10 cm); 2 edge cases (Mawson, Rutford
  GPS) assert CATS's published value despite known CATS-vs-gauge bias.
- Self-consistency: synthesize 1-year -> harmonic refit recovers
  amplitudes within 1 mm.
- 133 tests pass; `mypy --strict` and `ruff check` both clean.

### Known limitations

- v0.1 is Antarctic-only. Greenland presets are intentionally absent;
  CATS2008 does not cover the Arctic.
- No ionospheric-phase modelling (planned v0.2).
- MEaSUREs GL v2 auto-clip for fringe-map geometry is a roadmap item
  (placeholder analytic sinuous GL is used in v0.1).
- Real SAR data ingest is explicitly out of scope (this is a simulator).

[0.1.0]: https://github.com/pmilillo/tidal-insar-sim/releases/tag/v0.1.0
