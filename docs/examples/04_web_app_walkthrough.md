# Example 4 — Web app walkthrough

```bash
streamlit run tidal_insar_sim/web/app.py
```

opens the app at `http://localhost:8501`. You'll see a **Home** page plus
five sidebar links: Map, Configure, Single Site, Batch, Export.

## 1. Map

![Map page](img/placeholder_map.png)

- The Antarctic basemap (CartoDB positron) shows the eight preset GL
  sites as blue circles.
- Pick a preset from the top-left selector, or click anywhere on the map
  with `lat < 0` to create a custom site.
- The current selection is pinned in red and summarised below the map.
- Click *Reset to Thwaites* if you get lost.

All selection changes automatically invalidate any cached TripletReport,
so the next *Run sweep* will use the updated coordinates.

## 2. Configure

![Configure page](img/placeholder_configure.png)

- **Sensor**: the full preset list (NISAR-L, Sentinel-1 single/dual,
  ALOS-2/-4, CSG, TSX, RCM, Umbra-X). The caption shows the resolved
  wavelength / repeat / incidence triple.
- **Site overrides**: name, lat, lon, ice thickness. Editing any of these
  demotes the "site preset" to "Custom".
- **Sweep**: step (hours) and duration (days) for the rigid-triplet
  sweep. Defaults are 1 h × 29.53 d.
- **Coherence**: two-zone gammas for the synthetic fringe-map renderer
  only. They don't affect the sweep / planner numerics.

Click *Apply* to commit. The form clears cached results so the next run
picks up the new config.

## 3. Single Site

![Single Site page](img/placeholder_single_site.png)

Clicking *Run sweep* auto-loads CATS2008 at the site's lat/lon and
synthesises:

- Four top-line metrics: verdict, P(≥3 fringes), P(≥5 fringes), P(null).
- The fringe-count sweep curve (one lunar month, 1-h resolution), with
  horizontal markers at the 0.5 / 3 / 5-fringe thresholds.
- A histogram of the fringe distribution and a two-panel tide plot
  (h(t₁) vs h_DD).
- Synthetic DDInSAR fringe maps for the *best* and *worst* triplet phase
  — this is the eye-level view of what the sensor will actually see.

The lower half is the **acquisition planner**: pick start/end dates, N,
and min-fringes, hit *Run planner*, and get a sortable DataFrame with
per-triplet confidence.

## 4. Batch

![Batch page](img/placeholder_batch.png)

- Multi-select sites (any subset of the eight presets).
- Multi-select sensors (any subset of the nine presets).
- Pick a heatmap metric — default `P_usable_ge_3fr`.
- Click *Run batch* and wait (~1–2 minutes for a 5×4 grid using CATS2008).

The heatmap uses a red-yellow-green gradient: green = go, red = don't
bother. Below it, the full long-format DataFrame lists every `(site, sensor)`
row with all summary fields.

## 5. Export

![Export page](img/placeholder_export.png)

Every artefact cached in the session appears here as a download button:

- **Single-site**: sweep CSV, summary JSON, GeoJSON, KML, best/null
  fringe-map GeoTIFFs.
- **Planner**: plan CSV, plan ICS.
- **Batch**: batch CSV, batch Parquet.

Filenames include the site and sensor names for at-a-glance provenance.
Nothing is written to disk unless you click a download button — the
browser pulls the bytes straight from memory.

## Keeping state across pages

The app uses Streamlit's `session_state` under the `tis.*` prefix (see
`tidal_insar_sim/web/state.py`). Editing the site on the Map page or the
sweep parameters on Configure automatically clears any cached
TripletReport / planner DataFrame / fringe maps, so the next computation
always reflects your current inputs.

The only thing that persists across a full reload is the CATS2008 on-disk
install; everything else is reset when you refresh the browser.
