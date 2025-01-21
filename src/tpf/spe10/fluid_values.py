"""This file contains values for fluid parameters.

For now we provide parameter values for the following fluids:
* Oil (at 8000 psi)
* Water

The dictionary containing parameter values is obtained by, e.g.,
``tpf.spe10.fluid_values.oil.``.
They can be used in a simulation by passing, e.g.,
``NONWETTING: tpf.models.phases.FluidPhase(tpf.spe10.fluid_values.oil)`` as
a material parameter during model initiation.

Dead oil:
---------------

The values (except thermal conductivity) are gathered from:
* https://www.spe.org/web/csp/datasets/set02.htm#dead%20oil
* M. A. Christie and M. J. Blunt, “Tenth SPE Comparative Solution Project: A Comparison
of Upscaling Techniques,” SPE Reservoir Evaluation & Engineering, vol. 4, no. 04, pp.
308–317, Aug. 2001, doi: 10.2118/72469-PA.

Water:
---------------

The values are taken from ``porepy.applications.material_values.fluid_values.water``.

"""

from typing import Any

from porepy.applications.material_values.fluid_values import water as _water
from tpf.utils.constants_and_typing import FEET, LB, PSI, cP

# The values in the paper are given in [cP] and [lb ft^-3]. We convert them to [Pa s]
# and [kg m^-3]. Additionally we convert the density from surface density to reservoir
# density.
oil_surface_density: float = 53 * LB / (FEET**3)
oil_volume_factor: float = 1.01

oil: dict[str, Any] = {
    "name": "oil",
    "density": oil_surface_density
    / oil_volume_factor,  # [kg m^-3], density at reservoir conditions.
    "viscosity": 3 * cP,  # [Pa s], absolute viscosity.
    "residual_saturation": 0.2,  # [-], residual saturation.
}

water: dict[str, Any] = _water.copy()
water.update(
    {
        "name": "water",
        "residual_saturation": 0.2,  # [-], residual saturation.
    }
)

BHP: float = 4000 * PSI  # [psi], bottom hole pressure.
INITIAL_PRESSURE: float = 6000 * PSI  # [psi], initial pressure.
INITIAL_SATURATION: float = 0.3  # [-], initial saturation.
PRODUCTION_WELL_SIZE: float = 100 * FEET
"""Size of the production wells in the corner cells. The boundary in in vicinity of the
corners is prescribed Dirichlet conditions corresponding to a production well, i.e.,
fixed BHP."""
