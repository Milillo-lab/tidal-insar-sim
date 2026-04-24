"""Shared CLI helpers: sensor/site/constellation resolution, rich console singletons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import click
from rich.console import Console

if TYPE_CHECKING:
    from tidal_insar_sim.constellation import Constellation
    from tidal_insar_sim.sensor import Sensor
    from tidal_insar_sim.site import Site

F = TypeVar("F", bound=Callable[..., object])

CONSOLE = Console()
ERR_CONSOLE = Console(stderr=True, style="bold red")


def resolve_sensor(name: str, incidence_deg: float | None = None) -> Sensor:
    from tidal_insar_sim.sensor import SENSOR_PRESETS

    key = name.upper()
    if key not in SENSOR_PRESETS:
        known = ", ".join(sorted(set(SENSOR_PRESETS)))
        msg = f"Unknown sensor preset {name!r}. Known: {known}"
        raise click.BadParameter(msg, param_hint="--sensor")
    sensor = SENSOR_PRESETS[key]
    if incidence_deg is not None and incidence_deg != sensor.incidence_deg:
        return replace(sensor, incidence_deg=float(incidence_deg))
    return sensor


def resolve_constellation(
    *,
    name: str | None = None,
    n_sats: int | None = None,
    repeat_per_sat: float | None = None,
    satellites_file: Path | None = None,
) -> Constellation:
    """Build a Constellation from CLI options, with this precedence:

    1. `satellites_file` — YAML with `satellites: [{name, phase_offset_days, repeat_days}, ...]`
    2. `name` — preset lookup in `CONSTELLATION_PRESETS`.
    3. `n_sats` + `repeat_per_sat` — `Constellation.equally_phased(...)`.
    4. Fallback — single-sat 12-day (NISAR-like).
    """
    from tidal_insar_sim.constellation import (
        CONSTELLATION_PRESETS,
        Constellation,
        Satellite,
    )

    if satellites_file is not None:
        import yaml

        data = yaml.safe_load(Path(satellites_file).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "satellites" not in data:
            msg = f"{satellites_file} must have a top-level 'satellites:' list."
            raise click.BadParameter(msg, param_hint="--satellites")
        sats: list[Satellite] = []
        for i, entry in enumerate(data["satellites"]):
            sats.append(Satellite(
                name=str(entry.get("name", f"Sat-{i + 1}")),
                phase_offset_days=float(entry["phase_offset_days"]),
                repeat_days=float(entry["repeat_days"]),
            ))
        return Constellation(satellites=tuple(sats))

    if name is not None:
        key = name.upper()
        if key not in CONSTELLATION_PRESETS:
            known = ", ".join(sorted(CONSTELLATION_PRESETS))
            msg = f"Unknown constellation preset {name!r}. Known: {known}"
            raise click.BadParameter(msg, param_hint="--constellation")
        return CONSTELLATION_PRESETS[key]

    n = n_sats if n_sats is not None else 1
    repeat = repeat_per_sat if repeat_per_sat is not None else 12.0
    if n == 1:
        return Constellation.single(repeat_days=float(repeat))
    return Constellation.equally_phased(n=int(n), repeat_days=float(repeat))


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


def constellation_options(f: F) -> F:
    """Decorator adding the four constellation-selection flags to a Click command."""
    f = click.option(
        "--constellation", "constellation_preset", default=None,
        help="Constellation preset name (NISAR, SENTINEL-1-DUAL, RCM, ALOS).",
    )(f)
    f = click.option(
        "--n-sats", "n_sats", type=int, default=None,
        help="Number of satellites (if building an equally-phased constellation).",
    )(f)
    f = click.option(
        "--repeat-per-sat", "repeat_per_sat", type=float, default=None,
        help="Orbit repeat per satellite in days (default 12).",
    )(f)
    f = click.option(
        "--satellites", "satellites_file",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=None,
        help="YAML file with a 'satellites:' list for arbitrary phasing.",
    )(f)
    return f


def parse_iso_datetime(raw: str, *, param: str) -> datetime:
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        msg = f"--{param} must be ISO 8601 (e.g. 2026-07-15T00:00), got {raw!r}"
        raise click.BadParameter(msg, param_hint=f"--{param}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
