# Changelog

All notable changes to **tidal-insar-sim** are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project targets [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] — 2026-10-06

### Added
- `scripts/nisar_dd_test/delineate.py`: NISAR grounding-line delineation at Smith from the SNAPHU-unwrapped
  double difference, with the comparison against the January 2025 Sentinel-1 line (GMD paper, Sect. 4.5,
  Fig. 7c); delineated lines and summary included.

## [0.3.1] — 2026-10-06 (tagged, not archived on Zenodo; superseded by 0.3.2)

Companion release for the revised GMD paper (egusphere-2026-2542). It adds what the paper's
Section 3 and Code availability statement described and v0.3.0 did not contain.

### Added
- `tidal_insar_sim/rop.py`: NISAR Reference Observation Plan ingestion from the ArcGIS feature
  service or a saved GeoJSON snapshot; per-site acquisitions, one per pass; L-band mode selection
  (main band >= 20 MHz, wider band kept when a date lists two); streams keyed by track and L-band mode;
  incidence angle at the site from the frame footprint for NISAR's left-looking geometry.
  Radar-mode ids are resolved from the `radar_mode_combination_N` fields: the
  `unique_radar_mode_list` and `unique_radar_mode_mnemonic_list` fields are not in the same order.
- `scripts/gmd_reproduce.py`: regenerates every number, table and figure of the paper's Sections 4
  and 5 from one set of triplets; `--archive-check` repeats the NASA CMR query for public NISAR RSLC
  and GUNW products over the sites.
- `data/rop_snapshot/ase_bbox_2026-10-02.geojson`: the ROP query used in the paper (199 frames, 58 tracks).
- `tests/test_rop.py`.

### Fixed
- `__version__` read "0.2.0.dev0" in the v0.3.0 release; it now matches the release.

### Not added (stated in the revised paper instead)
- No PyPI distribution, no CI workflow, no runtime CATS2008 validation exception, no SHA-256
  recording, no `ConstellationDescriptor`: the discussion paper described these; the revised paper
  no longer does.

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
