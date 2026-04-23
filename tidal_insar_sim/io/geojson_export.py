"""Export a TripletReport as a GeoJSON FeatureCollection (RFC 7946).

Each report becomes:
    - 1 Point feature at (lon, lat) with all summary fields in properties.
    - 1 Polygon feature (64-vertex circle, radius = L_flex in metres) carrying
      the site's flexure-limit footprint for QGIS visualisation.
    - classification ∈ {excellent, good, marginal, poor} with QGIS color hint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tidal_insar_sim.report import TripletReport
    from tidal_insar_sim.simulator import Simulator


# Color hints are QGIS-friendly hex values keyed by verdict category.
CLASSIFICATION_STYLES: dict[str, dict[str, str]] = {
    "excellent": {"color": "#00a05c", "description": "P(>=3 fr) >= 85%"},
    "good":      {"color": "#57c584", "description": "60% <= P(>=3 fr) < 85%"},
    "marginal":  {"color": "#e2b63d", "description": "35% <= P(>=3 fr) < 60%"},
    "poor":      {"color": "#c0392b", "description": "P(>=3 fr) < 35%"},
}


@dataclass
class FlexureFootprint:
    """A polygonal footprint of the site's limit of flexure (radius = L_flex)."""

    lat: float
    lon: float
    radius_m: float
    n_vertices: int = 64

    def to_coordinates(self) -> list[list[float]]:
        """Return a closed ring of [lon, lat] pairs on a WGS84 geodesic."""
        import pyproj

        geod = pyproj.Geod(ellps="WGS84")
        angles_deg = np.linspace(0.0, 360.0, self.n_vertices, endpoint=False)
        lons, lats, _ = geod.fwd(
            np.full_like(angles_deg, self.lon),
            np.full_like(angles_deg, self.lat),
            angles_deg,
            np.full_like(angles_deg, self.radius_m),
        )
        ring = [[float(lon), float(lat)] for lon, lat in zip(lons, lats, strict=True)]
        ring.append(ring[0])  # close the ring
        return ring


def report_to_geojson_feature(
    report: TripletReport,
    *,
    simulator: Simulator | None = None,
    include_polygon: bool = True,
) -> dict[str, Any]:
    """Return a GeoJSON FeatureCollection for a single TripletReport."""
    sim = simulator if simulator is not None else report.simulator
    if sim is None:
        msg = "TripletReport has no bound Simulator; pass `simulator=` explicitly."
        raise RuntimeError(msg)

    summary = report.summary()
    classification = summary["verdict"]
    style = CLASSIFICATION_STYLES.get(str(classification), {"color": "#888888", "description": ""})
    properties: dict[str, Any] = {
        **{k: _jsonify(v) for k, v in summary.items()},
        "classification": classification,
        "color_hint": style["color"],
        "color_description": style["description"],
        "ice_thickness_m": sim.site.ice_thickness_m,
        "limit_of_flexure_m": sim.site.limit_of_flexure_m(),
        "beta_per_m": sim.site.flexural_parameter_beta(),
    }

    features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [sim.site.lon, sim.site.lat],
            },
            "properties": {**properties, "role": "site"},
        },
    ]

    if include_polygon:
        footprint = FlexureFootprint(
            lat=sim.site.lat,
            lon=sim.site.lon,
            radius_m=float(sim.site.limit_of_flexure_m()),
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [footprint.to_coordinates()],
                },
                "properties": {
                    **properties,
                    "role": "limit_of_flexure",
                    "radius_m": footprint.radius_m,
                },
            },
        )
    return {"type": "FeatureCollection", "features": features}


def report_to_geojson(
    report: TripletReport,
    path: str | Path,
    *,
    simulator: Simulator | None = None,
    include_polygon: bool = True,
    indent: int = 2,
) -> None:
    fc = report_to_geojson_feature(
        report, simulator=simulator, include_polygon=include_polygon
    )
    Path(path).write_text(json.dumps(fc, indent=indent), encoding="utf-8")


def _jsonify(value: Any) -> Any:
    """Coerce numpy scalars / arrays to plain Python for JSON emission."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value
