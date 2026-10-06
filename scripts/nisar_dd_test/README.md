# NISAR DDInSAR test of the predicted fringe count (GMD paper, Sect. 4.5)

Run on a Linux workstation at the University of Houston, 2026-10-02, with released NISAR L2 GUNW products
(NISAR_L2_GUNW_PROVISIONAL_V1, Alaska Satellite Facility; Earthdata login required).

1. `fetch.sh` downloads the 26 GUNW products in `gunw_test_urls.json` (54 GB).
2. `unwrap_dd.py` (Python with h5py, scipy, pyproj, snaphu-py 0.4.1) forms the 40 m double difference
   for each triplet in `gunw_test_selection.csv`, unwraps it with SNAPHU, detects the grounding-zone
   fringe belt and writes `out/unw_*.npz`, `out/belt_*.npz`, `out/unw_results.csv`.
3. `plateau_steps.py` measures the observed fringe count (plateau difference across the belt) and
   writes `plateau_results.csv` (the numbers in Sect. 4.5).
4. `results_figure.py` draws Fig. 7 and writes `gl_offsets.csv` (distance of the fringe belt from the
   latest Sentinel-1 grounding line; needs the grounding-line file `gl_1992_2025_ase.geojson`, a clip of
   the MEaSUREs/InSAR GL dataset to the Amundsen Sea).

5. `delineate.py` delineates the grounding line at Smith from the two coherent triplets (landward limit of
   tidal flexure, 10 % of the phase step; 5 % and 20 % for sensitivity) and compares it with the latest
   Sentinel-1 line: `delineation_summary.csv`, `nisar_gl_Smith_*_f10.geojson` (EPSG:3031).

`dd_test.py` and `extract_dd.py` are the earlier 80 m profile approach, kept because `unwrap_dd.py`
imports nothing from them but the record of what was tried matters: a 1-D profile fit reads noise as
fringes where the double difference is decorrelated.
