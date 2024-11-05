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


# Keywords:
GLOBAL_PRESSURE: str = "global_pressure"
COMPLIMENTARY_PRESSURE: str = "complimentary_pressure"

GLOBAL_FLUX: str = "global_flux"
COMPLIMENTARY_FLUX: str = "complimentary_flux"

WETTING: PHASENAME = "wetting"
NONWETTING: PHASENAME = "nonwetting"

# Homotopy continuation:
CONTINUATION_SOLUTIONS: str = "continuation_solutions"
