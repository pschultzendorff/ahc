from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import porepy as pp


class PhaseVariables:

    mobility_key: str
    """Keyword to define parameters and discretizations for the phase mobility. Normally
    set by a mixin of instance :class:`two_phase_flow.SolutionStrategyTPF`.

    """
    p: pp.ad.MixedDimensionalVariable
    """Ad representation of the phase pressure. Normally initialized by a mixin of
    instance :class:`~porepy.models.abstract_equations.VariableMixin`.

    """
    s: pp.ad.MixedDimensionalVariable
    """Ad representation of the phase saturation. Normally initialized by a mixin of
    instance :class:`~porepy.models.abstract_equations.VariableMixin`.

    """


class PhaseBC:
    domain_boundary_sides: Callable
    """Normally provided by a mixin of instance
    :class:``~porepy.models.geometry.ModelGeometry``.

    """


class PhaseConstants(pp.MaterialConstants):
    """Fluid constants for a phase."""

    def __init__(self, constants: Optional[dict[str, pp.number | str]] = None):
        default_constants = self.default_constants
        self.verify_constants(constants, default_constants)
        if constants is not None:
            default_constants.update(constants)
        super().__init__(default_constants)

    @property
    def default_constants(self) -> dict[str, pp.number | str]:
        """Default constants of the material.

        Returns:
            Dictionary of constants.

        """
        # Default values, sorted alphabetically
        default_constants: dict[str, pp.number | str] = {
            "name": "unnamed_phase",
            "compressibility": 0.0,
            "density": 1000.0,
            "pressure": 0.0,
            "viscosity": 1.0,
            "residual_saturation": 0.0,
            # The following are not used in the current implementation.
            "thermal_conductivity": 1,
            "thermal_expansion": 0,
            "normal_thermal_conductivity": 1,
            "specific_heat_capacity": 1,
        }
        return default_constants

    @property
    def name(self) -> str:
        """Name of the phase."""
        return self.constants["name"]

    @name.setter
    def name(self, value: str) -> None:
        self.constants["name"] = value

    def compressibility(self) -> pp.number:
        """Compressibility [Pa^-1].

        Returns:
            Compressibility array in converted pressure units.

        """
        return self.convert_units(self.constants["compressibility"], "Pa^-1")

    def density(self) -> pp.number:
        """Density [kg * m^-3].

        Note: For now, we assume incompressible flow, i.e., constant density.

        Returns:
            Density in converted mass and length units.

        """
        return self.convert_units(self.constants["density"], "kg * m^-3")

    def pressure(self) -> pp.number:
        """Pressure [Pa].

        Intended usage: Reference pressure.

        Returns:
            Pressure in converted pressure units.

        """
        return self.convert_units(self.constants["pressure"], "Pa")

    def viscosity(self) -> pp.number:
        """Viscosity [Pa * s].

        Note: For now, we assume incompressible and isothermal flow, i.e., constant
        viscosity.

        Returns:
            Viscosity array in converted pressure and time units.

        """
        return self.convert_units(self.constants["viscosity"], "Pa*s")

    def residual_saturation(self) -> pp.number:
        """Residual saturation [-].

        Returns:
            Residual_saturation array.

        """
        return self.convert_units(self.constants["residual_saturation"], "-")


class FluidPhase(PhaseVariables, PhaseBC, PhaseConstants):

    @classmethod
    def from_PhaseConstants(cls, constants: PhaseConstants) -> FluidPhase:
        return cls(constants.constants)
