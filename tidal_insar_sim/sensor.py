"""SAR sensor model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Sensor:
    """A SAR sensor configuration relevant to DDInSAR fringe-count prediction."""

    name: str
    wavelength_m: float
    repeat_days: float
    incidence_deg: float
    polarization: str = "HH"
    provider: str = ""

    @property
    def half_wavelength_m(self) -> float:
        return self.wavelength_m / 2.0

    @property
    def fringe_los_cm(self) -> float:
        return self.half_wavelength_m * 100.0

    NISAR_L: ClassVar[Sensor]
    SENTINEL_1_SINGLE: ClassVar[Sensor]
    SENTINEL_1_DUAL: ClassVar[Sensor]
    ALOS_2: ClassVar[Sensor]
    ALOS_4: ClassVar[Sensor]
    COSMO_SKYMED: ClassVar[Sensor]
    TERRASAR_X: ClassVar[Sensor]
    RADARSAT_CONSTELLATION: ClassVar[Sensor]
    UMBRA_X: ClassVar[Sensor]


Sensor.NISAR_L = Sensor(
    name="NISAR-L", wavelength_m=0.2360, repeat_days=12.0, incidence_deg=39.0, provider="NASA/ISRO"
)
Sensor.SENTINEL_1_SINGLE = Sensor(
    name="Sentinel-1 (single)", wavelength_m=0.0556, repeat_days=12.0, incidence_deg=39.0,
    provider="ESA",
)
Sensor.SENTINEL_1_DUAL = Sensor(
    name="Sentinel-1 (dual)", wavelength_m=0.0556, repeat_days=6.0, incidence_deg=39.0,
    provider="ESA",
)
Sensor.ALOS_2 = Sensor(
    name="ALOS-2", wavelength_m=0.2360, repeat_days=14.0, incidence_deg=34.0, provider="JAXA"
)
Sensor.ALOS_4 = Sensor(
    name="ALOS-4", wavelength_m=0.2360, repeat_days=14.0, incidence_deg=34.0, provider="JAXA"
)
Sensor.COSMO_SKYMED = Sensor(
    name="COSMO-SkyMed", wavelength_m=0.0312, repeat_days=4.0, incidence_deg=32.0, provider="ASI"
)
Sensor.TERRASAR_X = Sensor(
    name="TerraSAR-X", wavelength_m=0.0311, repeat_days=11.0, incidence_deg=36.0, provider="DLR"
)
Sensor.RADARSAT_CONSTELLATION = Sensor(
    name="RCM", wavelength_m=0.0556, repeat_days=4.0, incidence_deg=34.0, provider="CSA"
)
Sensor.UMBRA_X = Sensor(
    name="Umbra-X", wavelength_m=0.0312, repeat_days=0.0, incidence_deg=40.0, provider="Umbra"
)


SENSOR_PRESETS: dict[str, Sensor] = {
    "NISAR-L": Sensor.NISAR_L,
    "SENTINEL-1": Sensor.SENTINEL_1_SINGLE,
    "SENTINEL-1-SINGLE": Sensor.SENTINEL_1_SINGLE,
    "SENTINEL-1-DUAL": Sensor.SENTINEL_1_DUAL,
    "ALOS-2": Sensor.ALOS_2,
    "ALOS-4": Sensor.ALOS_4,
    "COSMO-SKYMED": Sensor.COSMO_SKYMED,
    "TERRASAR-X": Sensor.TERRASAR_X,
    "RCM": Sensor.RADARSAT_CONSTELLATION,
    "UMBRA-X": Sensor.UMBRA_X,
}
