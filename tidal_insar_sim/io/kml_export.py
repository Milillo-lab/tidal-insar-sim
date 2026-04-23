"""Export a TripletReport as a Google-Earth-ready KML file.

Same content as the GeoJSON — Point at the site + polygon footprint of the
limit of flexure — but formatted for Google Earth (altitude-relative coordinates,
styled with the classification color).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tidal_insar_sim.io.geojson_export import (
    CLASSIFICATION_STYLES,
    FlexureFootprint,
)

if TYPE_CHECKING:
    from tidal_insar_sim.report import TripletReport
    from tidal_insar_sim.simulator import Simulator


KML_NS = "http://www.opengis.net/kml/2.2"


def _hex_to_kml_color(hex_color: str, alpha: str = "b3") -> str:
    """Convert #RRGGBB to KML's AABBGGRR (alpha, blue, green, red)."""
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return (alpha + b + g + r).lower()


def report_to_kml(
    report: TripletReport,
    path: str | Path,
    *,
    simulator: Simulator | None = None,
    include_polygon: bool = True,
) -> None:
    sim = simulator if simulator is not None else report.simulator
    if sim is None:
        msg = "TripletReport has no bound Simulator; pass `simulator=` explicitly."
        raise RuntimeError(msg)

    summary = report.summary()
    verdict = str(summary["verdict"])
    color_hex = CLASSIFICATION_STYLES.get(verdict, {"color": "#888888"})["color"]
    kml_color = _hex_to_kml_color(color_hex)

    ET.register_namespace("", KML_NS)
    kml = ET.Element(f"{{{KML_NS}}}kml")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = f"{sim.site.name} / {sim.sensor.name}"
    ET.SubElement(doc, "description").text = (
        f"tidal-insar-sim: verdict={verdict}, P(>=3fr)={summary['P_usable_ge_3fr']:.2%}"
    )

    # --- style ---
    style = ET.SubElement(doc, "Style", id="tis-style")
    line = ET.SubElement(style, "LineStyle")
    ET.SubElement(line, "color").text = kml_color
    ET.SubElement(line, "width").text = "2"
    poly = ET.SubElement(style, "PolyStyle")
    ET.SubElement(poly, "color").text = kml_color
    ET.SubElement(poly, "fill").text = "1"
    ET.SubElement(poly, "outline").text = "1"
    icon = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon, "color").text = kml_color

    # --- site point ---
    placemark = ET.SubElement(doc, "Placemark")
    ET.SubElement(placemark, "name").text = sim.site.name
    ET.SubElement(placemark, "styleUrl").text = "#tis-style"
    desc = _summary_html(summary, sim)
    ET.SubElement(placemark, "description").text = desc
    pt = ET.SubElement(placemark, "Point")
    ET.SubElement(pt, "coordinates").text = f"{sim.site.lon},{sim.site.lat},0"

    # --- flexure polygon ---
    if include_polygon:
        footprint = FlexureFootprint(
            lat=sim.site.lat, lon=sim.site.lon,
            radius_m=float(sim.site.limit_of_flexure_m()),
        )
        poly_placemark = ET.SubElement(doc, "Placemark")
        ET.SubElement(poly_placemark, "name").text = "Limit of flexure"
        ET.SubElement(poly_placemark, "styleUrl").text = "#tis-style"
        polygon = ET.SubElement(poly_placemark, "Polygon")
        ET.SubElement(polygon, "tessellate").text = "1"
        outer = ET.SubElement(polygon, "outerBoundaryIs")
        ring = ET.SubElement(outer, "LinearRing")
        coords_str = " ".join(
            f"{lon},{lat},0" for lon, lat in footprint.to_coordinates()
        )
        ET.SubElement(ring, "coordinates").text = coords_str

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _summary_html(summary: dict[str, Any], sim: Simulator) -> str:
    return (
        f"<![CDATA[<b>{sim.site.name}</b> / {sim.sensor.name}<br/>"
        f"verdict: <b>{summary['verdict']}</b><br/>"
        f"P(&ge;3fr) = {summary['P_usable_ge_3fr']:.1%}<br/>"
        f"P(&ge;5fr) = {summary['P_robust_ge_5fr']:.1%}<br/>"
        f"P(&lt;0.5fr) = {summary['P_null_lt_0p5fr']:.1%}<br/>"
        f"fringe_mean = {summary['fringe_mean']:.2f}<br/>"
        f"fringe_max = {summary['fringe_max']:.2f}<br/>"
        f"L_flex = {sim.site.limit_of_flexure_m():.0f} m]]>"
    )
