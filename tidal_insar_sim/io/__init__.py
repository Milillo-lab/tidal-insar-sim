"""File-format adapters for tidal-insar-sim outputs (M7).

- `geojson_export.py`: Point + radius polygon of the flexure zone.
- `kml_export.py`: Google-Earth-ready twin of the GeoJSON.
- (GeoTIFF/PNG live with the FringeMap dataclass in `synthesis.py`.)
"""

from tidal_insar_sim.io.geojson_export import (
    CLASSIFICATION_STYLES,
    report_to_geojson,
    report_to_geojson_feature,
)
from tidal_insar_sim.io.kml_export import report_to_kml

__all__ = [
    "CLASSIFICATION_STYLES",
    "report_to_geojson",
    "report_to_geojson_feature",
    "report_to_kml",
]
