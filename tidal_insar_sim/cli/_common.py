"""Shared CLI helpers: sensor/site resolution, rich console singletons."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import click
from rich.console import Console

if TYPE_CHECKING:
    from tidal_insar_sim.sensor import Sensor
    from tidal_insar_sim.site import Site

CONSOLE = Console()
ERR_CONSOLE = Console(stderr=True, style="bold red")


def resolve_sensor(name: str) -> Sensor:
    from tidal_insar_sim.sensor import SENSOR_PRESETS

    key = name.upper()
    if key not in SENSOR_PRESETS:
        known = ", ".join(sorted(SENSOR_PRESETS))
        msg = f"Unknown sensor preset {name!r}. Known: {known}"
        raise click.BadParameter(msg, param_hint="--sensor")
    return SENSOR_PRESETS[key]


def resolve_site(
    *,
    preset: str | None,
    lat: float | None,
    lon: float | None,
    name: str | None,
    ice_thickness_m: float | None,
) -> Site:
    from tidal_insar_sim.site import SITE_PRESETS, Site

    if preset is not None:
        key = preset.upper()
        if key not in SITE_PRESETS:
            known = ", ".join(sorted(SITE_PRESETS))
            msg = f"Unknown site preset {preset!r}. Known: {known}"
            raise click.BadParameter(msg, param_hint="--preset")
        return SITE_PRESETS[key]
    if lat is None or lon is None:
        msg = "Must pass either --preset OR both --lat and --lon."
        raise click.UsageError(msg)
    kwargs: dict[str, float] = {}
    if ice_thickness_m is not None:
        kwargs["ice_thickness_m"] = ice_thickness_m
    return Site.from_coords(lat=lat, lon=lon, name=name, **kwargs)


def parse_iso_datetime(raw: str, *, param: str) -> datetime:
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        msg = f"--{param} must be ISO 8601 (e.g. 2026-07-15T00:00), got {raw!r}"
        raise click.BadParameter(msg, param_hint=f"--{param}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
