from typing import Literal, TypeVar

import porepy as pp

# Type aliases and type variables:
# TODO Phase names should be variable, e.g., "wetting", "nonwetting", "oil", "water",
# etc.
type PHASENAME = Literal["wetting", "nonwetting"]
type REL_PERM_MODEL = Literal[
    "Brooks-Corey",
    "Corey",
    "power",
    "linear",
    "van Genuchten-Burdine",
    "van Genuchten-Mualem",
]
type CAP_PRESS_MODEL = Literal["Brooks-Corey", "linear", "van Genuchten", None]
OperatorType = TypeVar("OperatorType", bound=pp.ad.Operator)

# region KEYWORDS
# Keywords:
GLOBAL_PRESSURE: str = "global_pressure"
COMPLIMENTARY_PRESSURE: str = "complimentary_pressure"

GLOBAL_FLUX: str = "global_flux"
COMPLIMENTARY_FLUX: str = "complimentary_flux"

WETTING: PHASENAME = "wetting"
NONWETTING: PHASENAME = "nonwetting"

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
