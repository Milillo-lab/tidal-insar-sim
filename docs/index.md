# tidal-insar-sim

**DDInSAR fringe-count simulator for tidally flexed grounding zones.**

Given a SAR sensor and an Antarctic grounding-line site, `tidal-insar-sim`
predicts how many wrapped double-difference InSAR fringes a 3-date acquisition
triplet will produce, using CATS2008 tide forcing and elastic-plate flexure
physics. It is purpose-built for **NISAR mission planning** and for
teaching-quality sensitivity analyses.

## Three layers

```
Layer 3: Streamlit web UI (tidal_insar_sim.web)
   |
Layer 2: CLI (tidal-insar-sim ...)
   |
Layer 1: Library (tidal_insar_sim)
   - Sensor, Site, Simulator, TripletReport
   - physics/, tides/, synthesis, planner, batch, io/
```

Layers 2 and 3 never bypass Layer 1. Every computation goes through the
public library API.

## What you get per run

- A verdict (`excellent` / `good` / `marginal` / `poor`) derived from
  the probability of observing ≥3 fringes across all triplet phases in
  one lunar month.
- Fringe-count statistics (mean, median, 5/50/95 percentiles, max) and
  operational probabilities (`P(≥3 fringes)`, `P(<0.5 fringes)`, …).
- Wrapped-phase DDInSAR GeoTIFFs (EPSG:3031) for the best and worst
  triplets, so you can *see* what the sensor will record.
- A top-N acquisition plan with per-triplet confidence under ±1 h
  jitter, exportable as `.ics` for Google/Outlook.
- GeoJSON + KML of the site + a 64-vertex geodesic `L_flex` polygon
  for QGIS / Google Earth.

## Where to go next

| If you want to…                            | See                                         |
|--------------------------------------------|---------------------------------------------|
| …run the tool in five minutes              | [Quick start](quickstart.md)                |
| …install the CATS2008 tide model           | [CATS2008 setup](cats2008_setup.md)         |
| …understand the physics                    | [Physics](physics.md)                       |
| …look up a function or class               | [API reference](api.md)                     |
| …see end-to-end examples                   | [Examples](examples/01_single_site.md)      |

## Citation

A Zenodo DOI and JOSS entry will be published at v0.1.0. In the meantime,
cite the code as:

> Milillo, P. (2026). *tidal-insar-sim: DDInSAR fringe-count simulator for
> tidally flexed grounding zones*. Version 0.1.0. University of Houston.
