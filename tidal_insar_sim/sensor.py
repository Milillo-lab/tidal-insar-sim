"""SAR sensor model — band-based (X / C / L)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Sensor:
    """A SAR sensor, identified by its band (X / C / L).

    A `Sensor` encodes the electromagnetic properties (wavelength, LOS
    geometry) but *not* the observation cadence — that lives on a
    `Constellation`. The same band can describe a single spacecraft
    (e.g. NISAR-L) or a multi-satellite constellation (e.g. Sentinel-1
    C-band with two spacecraft).

    Attributes
    ----------
    band : "X" | "C" | "L"
    wavelength_m : float — SAR wavelength in metres.
    incidence_deg : float — centre incidence angle in degrees.
    polarization : str — default "HH".
    """

    band: str
    wavelength_m: float
    incidence_deg: float
    polarization: str = "HH"

    @property
    def half_wavelength_m(self) -> float:
        return self.wavelength_m / 2.0

    @property
    def fringe_los_cm(self) -> float:
        return self.half_wavelength_m * 100.0

    @property
    def name(self) -> str:
        return f"{self.band}-band ({self.wavelength_m*100:.1f} cm)"

    X_BAND: ClassVar[Sensor]
    C_BAND: ClassVar[Sensor]
    L_BAND: ClassVar[Sensor]


# Band defaults — wavelengths drawn from the dominant sensor in each band.
Sensor.X_BAND = Sensor(band="X", wavelength_m=0.0312, incidence_deg=35.0)
Sensor.C_BAND = Sensor(band="C", wavelength_m=0.0556, incidence_deg=39.0)
Sensor.L_BAND = Sensor(band="L", wavelength_m=0.2360, incidence_deg=39.0)


BAND_PRESETS: dict[str, Sensor] = {
    "X": Sensor.X_BAND,
    "C": Sensor.C_BAND,
    "L": Sensor.L_BAND,
}

# Alias dict for CLI back-compat.
SENSOR_PRESETS: dict[str, Sensor] = {
    "X-BAND": Sensor.X_BAND,
    "C-BAND": Sensor.C_BAND,
    "L-BAND": Sensor.L_BAND,
    "X": Sensor.X_BAND,
    "C": Sensor.C_BAND,
    "L": Sensor.L_BAND,
}
