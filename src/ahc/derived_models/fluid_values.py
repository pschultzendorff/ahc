"""This file contains values for fluid parameters.

For now we provide parameter values for the following fluids:
* Oil (at 8000 psi)
* Water
* CO2 (at atmospheric pressure & 20°C)

The dictionary containing parameter values is obtained by, e.g.,
``ahc.spe10.fluid_values.oil.``.
They can be used in a simulation by passing, e.g.,
``NONWETTING: ahc.models.phases.FluidPhase(ahc.spe10.fluid_values.oil)`` as
a material parameter during model initiation.

Dead oil:
---------------

The values (except thermal conductivity) are gathered from:
- https://www.spe.org/web/csp/datasets/set02.htm#dead%20oil
- M. A. Christie and M. J. Blunt, “Tenth SPE Comparative Solution Project: A Comparison
of Upscaling Techniques,” SPE Reservoir Evaluation & Engineering, vol. 4, no. 04, pp.
308–317, Aug. 2001, doi: 10.2118/72469-PA.

Water:
---------------

The values are taken from ``porepy.applications.material_values.fluid_values.water``.

CO2:
---------------

The values are taken from https://www.peacesoftware.de/einigewerte/calc_co2.php7
at 350 bar and 70° C


"""

from typing import Any

from porepy.applications.material_values.fluid_values import water as _water

from ahc.utils.constants_and_typing import FEET, LB, cP

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
}

water: dict[str, Any] = _water.copy()
water.update(
    {
        "name": "water",
    }
)

# CO2 values calculated with https://www.peacesoftware.de/einigewerte/calc_co2.php7
# Surface conditions: 1 bar and 25° C
co2_surface: dict[str, Any] = {
    "name": "co2",
    "density": 1.7845,  # [kg m^-3]
    "viscosity": 1.493e-05,  # [Pa s]
}

# Reservoir conditions: 350 bar and 70° C
co2_reservoir: dict[str, Any] = {
    "name": "co2",
    "density": 826.3,  # [kg m^-3]
    "viscosity": 7.772e-05,  # [Pa s]
}
