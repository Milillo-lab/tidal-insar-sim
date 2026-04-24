# Example 3 — Batch comparison

"Which sensor should I prioritise for monitoring the Amundsen-Sea glaciers?"

## Library form

```python
from tidal_insar_sim import Sensor, Site
from tidal_insar_sim.batch import batch_compare, pivot_heatmap

df = batch_compare(
    sites=[Site.THWAITES, Site.PIG, Site.POPE, Site.SMITH, Site.KOHLER],
    sensors=[Sensor.NISAR_L, Sensor.SENTINEL_1_DUAL, Sensor.ALOS_4, Sensor.COSMO_SKYMED],
)
print(df[["site", "sensor", "verdict", "P_usable_ge_3fr", "fringe_mean"]])
```

## Heatmap

```python
heatmap = pivot_heatmap(df, metric="P_usable_ge_3fr")
print(heatmap.round(3))
```

```
sensor         COSMO-SkyMed  NISAR-L  Sentinel-1 (dual)  ALOS-4
site
Thwaites_GL           0.956    0.079              0.928   0.021
Pine_Island_GL        0.964    0.188              0.971   0.183
Pope_GL               0.941    0.075              0.922   0.028
Smith_GL              0.921    0.098              0.901   0.081
Kohler_GL             0.902    0.079              0.888   0.038
```

Key insights from the numbers:

- **NISAR L-band is poor** at most Amundsen-Sea GL's because its 12-day
  repeat aliases the K1 (diurnal) component of the mixed tide. That's the
  classic sampling problem described in Padman et al. (2018).
- **Sentinel-1 dual** (6-day repeat) avoids the diurnal alias and wins
  everywhere.
- **Cosmo-SkyMed** (X-band, 4-day) is competitive because the shorter
  wavelength means more fringes per metre of DD displacement — but
  coherence will be lower than modelled here on ice shelves.

## Persist the DataFrame

```python
from tidal_insar_sim.batch import batch_to_parquet

batch_to_parquet(df, "amundsen_batch.parquet")
```

Parquet keeps the list-typed `hDD_range_m` intact (unlike CSV).

## CLI equivalent

```bash
tidal-insar-sim batch \
    --site-preset THWAITES --site-preset PIG \
    --site-preset POPE --site-preset SMITH --site-preset KOHLER \
    --sensor NISAR-L --sensor SENTINEL-1-DUAL \
    --sensor ALOS-4 --sensor COSMO-SKYMED \
    --output out/amundsen
```

Writes `batch.parquet` plus a `heatmap_P_usable_ge_3fr.csv` and a rendered
heatmap table in the terminal.

## YAML-driven batches

For a stable, version-controlled campaign plan, write `sites.yaml`:

```yaml
sites:
  - THWAITES                           # preset
  - PIG
  - name: amundsen_central_shelf       # custom coordinates
    lat: -74.5
    lon: -103.0
    ice_thickness_m: 550
```

Then:

```bash
tidal-insar-sim batch --sites sites.yaml --sensor NISAR-L --sensor SENTINEL-1-DUAL \
                      --output out/amundsen
```

Mixing preset strings and `{name, lat, lon, ice_thickness_m}` dicts in the
same YAML is supported.
