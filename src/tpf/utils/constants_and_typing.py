from typing import Final, Literal

# Type aliases and type variables:
type PHASENAME = Literal["wetting", "nonwetting", "oil", "water", "gas", "co2"]
WETTING: PHASENAME = "wetting"
NONWETTING: PHASENAME = "nonwetting"

REL_PERM_MODEL = Literal[
    "Brooks-Corey",
    "Brooks-Corey-Burdine",
    "Brooks-Corey-Mualem",
    "Corey",
    "power",
    "linear",
    "van Genuchten-Burdine",
    "van Genuchten-Mualem",
]
CAP_PRESS_MODEL = Literal["Brooks-Corey", "linear", "van Genuchten", None]

# region KEYWORDS
# Keywords:
PRESSURE_KEY = Literal["global_pressure", "complimentary_pressure"]
GLOBAL_PRESSURE: Final = "global_pressure"
COMPLIMENTARY_PRESSURE: Final = "complimentary_pressure"

FLUX_KEY = Literal["global_flux", "complimentary_flux"]
GLOBAL_FLUX: Final = "global_flux"
COMPLIMENTARY_FLUX: Final = "complimentary_flux"

# Homotopy continuation:
CONTINUATION_SOLUTIONS: str = "continuation_solutions"
# endregion

# region UNITS
# Each units gives the conversion factor to SI units. E.g.,
# >>> psi: float = 6894.76
# Then we can convert 5.2 [psi] to [Pa] by
# >>> 5.2 * psi
cP: float = 1e-3  # centipoise to [Pa s]
FEET: float = 0.3048  # [ft] to [m]
LB: float = 0.453592  # [lb] to [kg]
PSI: float = 6894.76  # [psi] to [Pa]
# endregion
