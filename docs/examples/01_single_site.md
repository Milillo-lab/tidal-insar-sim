# Example 1 — Single-site analysis at Rutford GL

We'll run the full sensitivity analysis for Rutford Ice Stream (where the
semidiurnal tide dominates) against NISAR L-band.

## Library form

```python
from tidal_insar_sim import Sensor, Site, Simulator

sim = Simulator(sensor=Sensor.NISAR_L, site=Site.RUTFORD)
report = sim.sweep_triplets(step_hours=1.0, duration_days=29.53)
summary = report.summary()
print(f"verdict: {summary['verdict']}")
print(f"P(>=3 fringes): {summary['P_usable_ge_3fr']:.1%}")
print(f"fringe mean:    {summary['fringe_mean']:.2f}")
```

```
verdict: excellent
P(>=3 fringes): 86.9%
fringe mean:    9.12
```

## Inspect the best and worst triplets

```python
best = report.best_triplet_h()
worst = report.worst_triplet_h()
print(f"best Delta_t = {best:7.0f} h   fringes = {report.fringes[report.delta_t_hours == best][0]:.2f}")
print(f"worst Delta_t= {worst:7.0f} h   fringes = {report.fringes[report.delta_t_hours == worst][0]:.2f}")
```

## Visualise the two regimes

```python
import matplotlib.pyplot as plt
from tidal_insar_sim.web.components.plots import plot_fringe_map

fmap_strong = sim.synthesize_ddinsar(triplet_start_hours=best)
fmap_null   = sim.synthesize_ddinsar(triplet_start_hours=worst)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(fmap_strong.wrapped_phase, cmap="hsv", vmin=-3.14, vmax=3.14)
axes[0].set_title(f"best: {fmap_strong.n_fringes_peak:.1f} fringes")
axes[1].imshow(fmap_null.wrapped_phase, cmap="hsv", vmin=-3.14, vmax=3.14)
axes[1].set_title(f"null: {fmap_null.n_fringes_peak:.2f} fringes")
plt.savefig("rutford_best_vs_null.png", dpi=140)
```

## Export everything

```python
report.to_csv("rutford_sweep.csv")
report.to_geojson("rutford.geojson")
report.to_kml("rutford.kml")
fmap_strong.to_geotiff("rutford_best.tif")
fmap_null.to_geotiff("rutford_null.tif")
```

## CLI equivalent

One line:

```bash
tidal-insar-sim analyze --sensor NISAR-L --preset RUTFORD --output out/rutford
```

Outputs identically.

## Try it for Thwaites

Rutford is semidiurnal and excellent for NISAR. Thwaites is diurnal-dominated
with smaller amplitude — NISAR's 12-day aliasing against the diurnal K1
constituent makes it much more triplet-sensitive. Swap the preset and
compare:

```python
sim = Simulator(sensor=Sensor.NISAR_L, site=Site.THWAITES)
print(sim.sweep_triplets().summary()["verdict"])
# 'good' (P(>=3fr) ~ 69%)
```

See [Example 2](02_acquisition_planning.md) for how to convert these
statistics into actual acquisition windows.
