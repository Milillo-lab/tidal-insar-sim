# Example 2 — Acquisition planning at Thwaites

Operational question: *"I can request NISAR Thwaites acquisitions any time
between 2026-06-01 and 2026-09-01. Which 10 triplets should I pitch, and
how robust are they under ±1 h timing jitter?"*

```python
from datetime import datetime, timezone
from tidal_insar_sim import Sensor, Site, Simulator

sim = Simulator(sensor=Sensor.NISAR_L, site=Site.THWAITES)
report = sim.sweep_triplets()

plan = report.recommended_triplets(
    start_date=datetime(2026, 6, 1, tzinfo=timezone.utc),
    end_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
    n=10, min_fringes=3.0,
)
print(plan.head())
```

Sample output:

```
   rank    triplet_start_utc     t1_date     t2_date     t3_date  predicted_fringes  ... confidence
0     1   2026-06-14T08:00:00  2026-06-14  2026-06-26  2026-07-08               6.72  ...      0.98
1     2   2026-07-24T04:00:00  2026-07-24  2026-08-05  2026-08-17               6.48  ...      0.95
2     3   2026-08-23T22:00:00  2026-08-23  2026-09-04  2026-09-16               6.31  ...      0.96
...
```

## Why confidence matters

A triplet sitting exactly on a tidal null ramp can have *nominally* many
fringes but collapse to near-zero with a few minutes of jitter. Look for
rows with `predicted_fringes >= 5` AND `confidence >= 0.9`.

```python
robust = plan.query("predicted_fringes >= 5 and confidence >= 0.9")
print(robust[["t1_date", "predicted_fringes", "confidence"]])
```

## Export a calendar

```python
from tidal_insar_sim.planner import to_ics

plan.to_csv("thwaites_plan.csv", index=False)
to_ics(plan, "thwaites_plan.ics")
```

Drag `thwaites_plan.ics` onto Google Calendar or Outlook — each triplet
becomes three 1-hour events (t1, t2, t3), with confidence and predicted
fringes in the event description.

## CLI equivalent

```bash
tidal-insar-sim plan \
    --sensor NISAR-L --preset THWAITES \
    --start 2026-06-01T00:00 --end 2026-09-01T00:00 \
    --top-n 10 --min-fringes 3 \
    --output thwaites_plan
```

Produces `thwaites_plan.csv` + `thwaites_plan.ics` and prints a rich table
to stdout.

## Interpretation guide

- **predicted_fringes ≥ 5, confidence ≥ 0.95**: submit this with high
  priority — even sloppy timing will deliver a usable DDInSAR.
- **predicted_fringes ≥ 5, confidence < 0.9**: the triplet lies near a tidal
  gradient; consider requesting a tight timing tolerance.
- **predicted_fringes 3–5, any confidence**: "marginal" tier — still useful
  for ancillary GL monitoring, but may lack SNR for precision inversion.
- **No rows above threshold**: either the window is too narrow, or you're
  over-constraining `min_fringes`. Inspect `report.class_fractions()`
  for the site's intrinsic distribution.
