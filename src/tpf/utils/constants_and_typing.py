from typing import Final, Literal, TypeAlias

# Type aliases and type variables:
PHASENAME: TypeAlias = Literal["wetting", "nonwetting", "oil", "water", "gas", "co2"]
WETTING: Final[PHASENAME] = "wetting"
NONWETTING: Final[PHASENAME] = "nonwetting"

REL_PERM_MODEL: TypeAlias = Literal[
    "Brooks-Corey",
    "Brooks-Corey-Burdine",
    "Brooks-Corey-Mualem",
    "Corey",
    "power",
    "linear",
    "van Genuchten-Burdine",
    "van Genuchten-Mualem",
]
CAP_PRESS_MODEL: TypeAlias = Literal["Brooks-Corey", "linear", "van Genuchten", None]

# region KEYWORDS
# Keywords:
PRESSURE_KEY: TypeAlias = Literal["global_pressure", "complementary_pressure"]
GLOBAL_PRESSURE: Final[PRESSURE_KEY] = "global_pressure"
COMPLEMENTARY_PRESSURE: Final[PRESSURE_KEY] = "complementary_pressure"

FLUX_NAME: TypeAlias = Literal["total_flux", "wetting_flux", "capillary_flux"]
TOTAL_FLUX: Final[FLUX_NAME] = "total_flux"
WETTING_FLUX: Final[FLUX_NAME] = "wetting_flux"
CAPILLARY_FLUX: Final[FLUX_NAME] = "capillary_flux"

# Homotopy continuation:
CONTINUATION_SOLUTIONS: Final = "continuation_solutions"
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
