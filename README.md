# tidal-insar-sim

DDInSAR fringe-count simulator for tidally flexed grounding zones.

Predicts how many InSAR fringes a triplet of SAR acquisitions will produce across
an Antarctic grounding line, given realistic ocean-tide forcing and elastic-plate
flexure of the ice shelf. Built for mission planning of NISAR and other L/C/X-band
InSAR missions.

## Status

v0.0 — under construction. The library layer (Checkpoint 1) is implemented against
mock tides. CATS2008 integration is deferred to Checkpoint 2.

See [`docs/COWORK_AGENT_BRIEF.md`](docs/COWORK_AGENT_BRIEF.md) for the full build plan
and [`reference/`](reference/) for the four validated prototype scripts the physics
is distilled from.

## Quick start (library, Checkpoint 1)

```python
from tidal_insar_sim import Sensor, Site, Simulator

sensor = Sensor.NISAR_L
site = Site.THWAITES
sim = Simulator(sensor=sensor, site=site)
report = sim.sweep_triplets(step_hours=1.0, duration_days=29.53)
print(report.summary())
```

## License

MIT (pending).
