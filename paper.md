---
title: "tidal-insar-sim: a DDInSAR fringe-count simulator for tidally flexed grounding zones"
tags:
  - Python
  - InSAR
  - SAR
  - grounding line
  - Antarctica
  - tide model
  - NISAR
  - cryosphere
authors:
  - name: Pietro Milillo
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: "Civil and Environmental Engineering, University of Houston, USA"
    index: 1
date: 24 April 2026
bibliography: paper.bib
---

# Summary

Differential double-difference InSAR (DDInSAR) is the standard tool for mapping
ice-shelf grounding lines (GLs) and quantifying their migration. A DDInSAR
observation combines three SAR acquisitions ($t_1, t_2, t_3$) such that the
steady-state flow signal cancels and what remains is the tidally driven
flexure of the shelf near the GL. Whether a given triplet yields a useful
fringe pattern depends on the instantaneous tide at all three epochs, on the
ice-mechanical parameters of the shelf, and on the sensor's wavelength,
incidence angle, and repeat interval.

`tidal-insar-sim` packages this forward problem into a reusable Python
library, a command-line interface, and a Streamlit web application. Given a
SAR sensor preset (NISAR-L, Sentinel-1, ALOS-2/-4, COSMO-SkyMed, TerraSAR-X,
RADARSAT Constellation, Umbra-X) and a grounding-line site, it predicts:

1. The distribution of DDInSAR fringe counts over all possible triplet
   phases in a lunar month (mean, median, percentiles, and operational
   probabilities such as $P(\geq 3\,\text{fringes})$).
2. Synthetic wrapped-phase DDInSAR maps for the best and null triplets,
   exported as georeferenced GeoTIFFs (EPSG:3031 polar stereographic).
3. A top-$N$ ranked acquisition plan in a user-supplied date window, with
   per-triplet confidence under $\pm 1$ h acquisition-time jitter, exportable
   as an `.ics` calendar compatible with Google Calendar and Outlook.
4. Batch comparisons across (site, sensor) grids, persisted to Parquet and
   rendered as red-yellow-green heatmaps.

The tool is Antarctic-only in v0.1 and uses the CATS2008 regional inverse
tide model [@Padman2002; @Padman2018] via `pyTMD` as its exclusive tide
source.

# Statement of need

Grounding-line monitoring is a central activity in Antarctic cryosphere
research because GL retreat rates modulate future sea-level contributions
[@Rignot2011; @Milillo2019]. DDInSAR has been the workhorse method for two
decades, but campaign planners have lacked a quantitative, reproducible way
to answer basic operational questions:

- *Will NISAR actually resolve the GL at this site, given its 12-day
  repeat?* Padman et al. [-@Padman2018] showed that the NISAR sampling
  aliases the K1 diurnal constituent; `tidal-insar-sim` quantifies how bad
  that aliasing is at each specific site.
- *Which of our 10 candidate acquisition windows should we submit?* The
  planner turns a month of candidate triplet phases into a ranked, confidence-
  scored list with standard calendar output.
- *How does Sentinel-1 (6-day dual, 12-day single) compare to NISAR (12-day)
  and CSG (4-day) at this site?* The batch module produces side-by-side
  heatmaps.
- *What does the fringe pattern actually look like in the best vs. null
  case?* The synthesis module renders georeferenced DDInSAR GeoTIFFs that
  load directly into QGIS, ISCE, and GMTSAR.

Existing tools partially address pieces of this (pyTMD provides the tide
backbone; ISCE provides the DDInSAR back-end; the reference MATLAB TMD GUI
can extract amplitudes) but no single tool chains them into a decision-support
workflow for mission planning. `tidal-insar-sim` fills that gap, and its
three-layer architecture (library → CLI → web UI) lets researchers,
operations engineers, and programme managers each work at the level of
abstraction that suits them.

# Design

The physics stack follows Rignot et al. [-@Rignot2011], Fricker and Padman
[-@Fricker2006], and Walker et al. [-@Walker2013]: an elastic plate clamped
at the GL, with flexural rigidity $D = E H^3 / [12 (1-\nu^2)]$ and parameter
$\beta = (\rho_w g / 4 D)^{1/4}$. The deflection under tidal load $h(t)$ is
$w(s, t) = h(t)[1 - e^{-\beta s}(\cos\beta s + \sin\beta s)]$ for $s \ge 0$
(floating side), zero on grounded ice. The DD vertical displacement is then
$h_{DD}(x) = w(x, t_1) - 2 w(x, t_2) + w(x, t_3)$, projected to DDInSAR
phase via $\phi_{DD} = -(4\pi/\lambda) \cos\theta \cdot h_{DD}$.

Tides are supplied by CATS2008 [@Padman2002], extracted at the site's
lat/lon by a thin wrapper around `pyTMD.compute.tide_elevations`, with a
nearest-ocean-cell fallback for masked grounding-zone points.

The validation gate — a hard prerequisite for any downstream work — cross-
checks the wrapper's extracted M2 and K1 amplitudes against the Antarctic
Tide Gauge database [@Howard2019] at eight well-calibrated stations
(McMurdo, Davis, Casey, Syowa, Rothera, Halley, Cape Roberts, Scott Base)
within station-specific tolerances of 3–10 cm. Two additional stations
(Mawson, Rutford GPS) are documented edge cases where CATS2008 itself has
known biases relative to tide gauges; the tool is verified to reproduce
CATS's published values at those points.

# Acknowledgements

CATS2008 is distributed by Earth and Space Research; the Antarctic Tide
Gauge Database v1 is distributed through the U.S. Antarctic Program Data
Center. `pyTMD` (Sutterley et al.) was indispensable for this work.
Development was supported in part by NASA NISAR Science Team funding.

# References
