"""Grounding-zone site: location, ice-mechanical parameters, and tide constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from tidal_insar_sim.physics.flexure import (
    flexural_parameter_beta,
    limit_of_flexure,
)


@dataclass
class Site:
    """A grounding-zone site.

    lat/lon are WGS-84 decimal degrees. Southern hemisphere is negative (Antarctica).
    Ice-mechanical defaults follow Rignot 2011 / Walker 2013 practice.
    `tide_constants` is populated lazily by the tide backend (e.g. CATS2008); until
    wired up, the Simulator accepts a mock tide function instead.
    """

    lat: float
    lon: float
    name: str
    ice_thickness_m: float = 400.0
    young_modulus_pa: float = 0.88e9
    poisson_ratio: float = 0.30
    rho_water: float = 1028.0

    tide_constants: dict[str, tuple[float, float]] | None = field(default=None)
    tide_source: str = "unset"

    @classmethod
    def from_coords(
        cls,
        lat: float,
        lon: float,
        name: str | None = None,
        ice_thickness_m: float = 400.0,
        young_modulus_pa: float = 0.88e9,
        poisson_ratio: float = 0.30,
        rho_water: float = 1028.0,
    ) -> Site:
        return cls(
            lat=lat,
            lon=lon,
            name=name or f"site_{lat:+.2f}_{lon:+.2f}",
            ice_thickness_m=ice_thickness_m,
            young_modulus_pa=young_modulus_pa,
            poisson_ratio=poisson_ratio,
            rho_water=rho_water,
        )

    def flexural_parameter_beta(self) -> float:
        return flexural_parameter_beta(
            young_modulus_pa=self.young_modulus_pa,
            ice_thickness_m=self.ice_thickness_m,
            poisson_ratio=self.poisson_ratio,
            rho_water=self.rho_water,
        )

    def limit_of_flexure_m(self) -> float:
        return limit_of_flexure(self.flexural_parameter_beta())

    THWAITES: ClassVar[Site]
    PIG: ClassVar[Site]
    RUTFORD: ClassVar[Site]
    ROSS_GZ16: ClassVar[Site]
    TOTTEN: ClassVar[Site]
    POPE: ClassVar[Site]
    SMITH: ClassVar[Site]
    KOHLER: ClassVar[Site]


Site.THWAITES = Site(lat=-75.00, lon=-106.00, name="Thwaites_GL", ice_thickness_m=450.0)
Site.PIG = Site(lat=-74.95, lon=-100.70, name="Pine_Island_GL", ice_thickness_m=500.0)
Site.RUTFORD = Site(lat=-78.50, lon=-83.00, name="Rutford_GL", ice_thickness_m=2000.0)
Site.ROSS_GZ16 = Site(lat=-84.30, lon=-163.00, name="Whillans_GZ16", ice_thickness_m=720.0)
Site.TOTTEN = Site(lat=-66.90, lon=116.00, name="Totten_GL", ice_thickness_m=1500.0)
Site.POPE = Site(lat=-74.70, lon=-113.00, name="Pope_GL", ice_thickness_m=600.0)
Site.SMITH = Site(lat=-74.60, lon=-112.00, name="Smith_GL", ice_thickness_m=700.0)
Site.KOHLER = Site(lat=-75.30, lon=-114.80, name="Kohler_GL", ice_thickness_m=650.0)


SITE_PRESETS: dict[str, Site] = {
    "THWAITES": Site.THWAITES,
    "PIG": Site.PIG,
    "PINE_ISLAND": Site.PIG,
    "RUTFORD": Site.RUTFORD,
    "ROSS_GZ16": Site.ROSS_GZ16,
    "TOTTEN": Site.TOTTEN,
    "POPE": Site.POPE,
    "SMITH": Site.SMITH,
    "KOHLER": Site.KOHLER,
}
