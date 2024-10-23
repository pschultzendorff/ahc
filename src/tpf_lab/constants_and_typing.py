from typing import Literal, TypeVar

import porepy as pp

# Type aliases and type variables:
type PHASENAME = Literal["wetting", "nonwetting"]
OperatorType = TypeVar("OperatorType", bound=pp.ad.Operator)

# Keywords:
GLOBAL_PRESSURE: str = "global_pressure"
COMPLIMENTARY_PRESSURE: str = "complimentary_pressure"

WETTING: PHASENAME = "wetting"
NONWETTING: PHASENAME = "nonwetting"
