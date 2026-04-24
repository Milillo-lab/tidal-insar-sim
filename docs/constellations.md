# Constellations

Starting in v0.2, `tidal-insar-sim` separates the **sensor** (band,
wavelength, incidence) from the **constellation** (number of satellites,
each with its own phase offset and orbit repeat). This matches how
operational SAR systems actually work:

- **NISAR**: 1 satellite, 12-day repeat — `Constellation.single(12)`.
- **Sentinel-1 A+B (pre-2022)**: 2 sats equally phased, 12-day repeat each
  → 6-day effective baseline.
- **RADARSAT Constellation (RCM)**: 3 sats equally phased, 12-day each →
  4-day effective baseline.
- **Umbra-X / tasked SAR**: any number of satellites, arbitrary phasing.

## The `Satellite` + `Constellation` dataclasses

```python
from tidal_insar_sim import Satellite, Constellation

# A single satellite: name, phase offset (days at t=0), orbit repeat (days).
sat = Satellite(name="S1A", phase_offset_days=0.0, repeat_days=12.0)
sat_b = Satellite(name="S1B", phase_offset_days=6.0, repeat_days=12.0)

const = Constellation(satellites=(sat, sat_b))
print(const.effective_repeat_days())   # 6.0  (6-day "effective baseline")
print(const.acquisition_times(window_days=30.0))
# array([ 0.,  6., 12., 18., 24.])
```

### Factory methods

```python
Constellation.single(repeat_days=12.0)                   # NISAR-like
Constellation.equally_phased(n=2, repeat_days=12.0)      # S1 A+B
Constellation.equally_phased(n=3, repeat_days=12.0)      # RCM
Constellation.equally_phased(n=4, repeat_days=14.0)      # ALOS-4 cluster
```

### Presets

```python
Constellation.NISAR               # single 12 d
Constellation.ALOS                # single 14 d
Constellation.SENTINEL_1_SINGLE   # single 12 d
Constellation.SENTINEL_1_DUAL     # 2 sats, 6 d effective
Constellation.RCM                 # 3 sats, 4 d effective
```

## Three sweep modes

Once a `Simulator` is bound to a (sensor, constellation, site) triple,
three sweep modes are available:

### 1. `sweep_triplets()` — rigid-B (classical DDInSAR)

The baseline `B` defaults to `constellation.effective_repeat_days()`. You
can force any other B with `baseline_days=...`.

```python
from tidal_insar_sim import Sensor, Site, Simulator, Constellation

sim = Simulator(
    sensor=Sensor.L_BAND,
    site=Site.RUTFORD,
    constellation=Constellation.equally_phased(n=2, repeat_days=12.0),  # B=6
)
report = sim.sweep_triplets()
print(report.summary()["verdict"])    # e.g. "excellent"

# Force B=12 d (use only sat-A acquisitions)
report_12 = sim.sweep_triplets(baseline_days=12.0)
```

### 2. `multi_baseline_sweep()` — one sweep per valid B

For a 2-sat equally-phased 12-day constellation, valid rigid baselines are
`{6, 12, 18, 24, ...}` days. The multi-baseline sweep runs a rigid sweep at
each and returns a dict `{B: TripletSweep}`. Use it to find the best
cadence at a given site.

```python
sweeps = sim.multi_baseline_sweep()
for B, sw in sorted(sweeps.items()):
    import numpy as np
    print(f"B={B:.1f} d  P(>=3fr)={np.mean(sw.fringes >= 3):.1%}  "
          f"mean={np.mean(sw.fringes):.2f}")
```

Typical Rutford / L-band / RCM output:

```
B=4.0 d  P(>=3fr)=92.5%  mean=16.19
B=8.0 d  P(>=3fr)=95.0%  mean=27.86    <-- best
B=12.0 d P(>=3fr)=86.8%  mean=9.12
```

The semidiurnal Rutford tide aliases best at B=8 d despite RCM's 4-day
*native* sampling.

### 3. `any_triplet_sweep()` — every `(t1, t2, t3)` from the schedule

Relaxes the rigid-B constraint: enumerates every ordered triple
`(t_i < t_j < t_k)` from the constellation's acquisition schedule, up to a
configurable maximum span. Useful when the constellation has uneven
phasing and you want to explore the full combinatorial space.

```python
ats = sim.any_triplet_sweep(duration_days=60.0)
print(f"{len(ats)} triplets")    # e.g. 2418
# Each is (t1_days, t2_days, t3_days, h_dd_m, fringes). Filter as you like:
import numpy as np
rigid = np.isclose((ats.t2_days - ats.t1_days), (ats.t3_days - ats.t2_days),
                   atol=0.01)
```

Pass `require_equal_baseline=True` to filter to rigid triplets only
(equivalent to enumerating across all valid `B` values from
`multi_baseline_sweep`).

## CLI

All four subcommands (`analyze`, `plan`, `synthesize`, `batch`) accept
these flags:

| Flag | What it does |
|---|---|
| `--n-sats INT` | Number of satellites (equally phased) |
| `--repeat-per-sat FLOAT` | Orbit repeat per satellite (days, default 12) |
| `--constellation NAME` | Preset lookup (NISAR, SENTINEL-1-DUAL, RCM, ALOS) |
| `--satellites FILE` | YAML with per-satellite phase/repeat table |

Precedence: `--satellites` > `--constellation` > `--n-sats + --repeat-per-sat`.
Default is `--n-sats 1 --repeat-per-sat 12` (NISAR-like).

### Equally-phased dual constellation

```bash
tidal-insar-sim analyze \
    --sensor C-BAND --preset RUTFORD \
    --n-sats 2 --repeat-per-sat 12 \
    --output out/rutford_dual
```

### Preset constellation

```bash
tidal-insar-sim analyze \
    --sensor C-BAND --preset RUTFORD \
    --constellation RCM \
    --mode multi_baseline \
    --output out/rcm_multi
```

Produces a `multi_baseline_summary.csv` plus per-B sweep CSVs.

### Arbitrary constellation from YAML

```yaml
# constellation.yaml
satellites:
  - name: S1A
    phase_offset_days: 0.0
    repeat_days: 12.0
  - name: S1B
    phase_offset_days: 6.0
    repeat_days: 12.0
  - name: S1C           # hypothetical 3rd sat, phased at 3 d
    phase_offset_days: 3.0
    repeat_days: 12.0
```

```bash
tidal-insar-sim analyze \
    --sensor C-BAND --preset RUTFORD \
    --satellites constellation.yaml \
    --mode any_triplet --duration 60 \
    --output out/rutford_custom_any
```

Produces an `any_triplets.csv` listing every valid `(t1, t2, t3)` from the
3-sat schedule over 60 days.

### Batch: grid over constellations

```bash
tidal-insar-sim batch \
    --site-preset THWAITES --site-preset RUTFORD \
    --sensor L-BAND \
    --constellation NISAR --constellation SENTINEL-1-DUAL --constellation RCM \
    --output out/constellation_comparison
```

Writes a `batch.parquet` with 2 sites × 1 sensor × 3 constellations = 6
rows, plus a heatmap CSV where each column is labelled
`"L-band (23.6 cm) [B=XX.Xd]"` so different constellations of the same
sensor are still distinguishable.

## Web UI

The **Setup** page renders the constellation as an editable table
(`st.data_editor`). Each row is one satellite — name, phase offset, and
orbit repeat. Four preset buttons populate the table for the common
constellations. A live readout shows `# satellites`, `effective B`, and
the number of acquisitions in the next 30 days.

The **Single Site** page respects the sweep mode (rigid / multi_baseline /
any_triplet) picked on Setup and renders the appropriate plot — one sweep
curve, a per-B overlay, or an `(t1, fringes, span)` scatter.

## When to use each mode

| Situation | Mode |
|---|---|
| Single-satellite mission (NISAR), fixed repeat | **rigid** |
| Multi-sat constellation, want best B | **multi_baseline** |
| Unevenly-phased or ad-hoc constellation (Umbra, tasked SAR) | **any_triplet** |
| Teaching the tidal-aliasing problem | All three, side-by-side |

`multi_baseline` is the workhorse for campaign planning — it tells you at a
glance which cadence wins at a given site. `any_triplet` is the "what's
theoretically possible" view — useful for constellations whose operational
scheduler can mix and match acquisitions arbitrarily.
