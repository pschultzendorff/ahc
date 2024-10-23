from __future__ import annotations

import abc
from typing import Any, Callable, Literal, Optional, Type, TypeAlias, Union, overload

import numpy as np
import porepy as pp
from porepy import MaterialConstants
from tpf_lab.constants_and_typing import NONWETTING, PHASENAME, WETTING

number = pp.number


class Phase:

    mobility_key: str
    """Keyword to define parameters and discretizations for the phase mobility. Normally
    set by a mixin of instance :class:`two_phase_flow.SolutionStrategyTPF`.

    """

    def __init__(self, name: PHASENAME) -> None:
        self.name: PHASENAME = name
        """Name given to the phase at instantiation."""

        # TODO: Should the following be here or in without ``self`` right after the
        # class name?
        self.constants: PhaseConstants
        """Phase constant object that takes care of scaling of phase-related
        quantities. Normally set by a mixin of instance
        :class:`~porepy.models.solution_strategy.SolutionStrategy`.

        """
        self.p: pp.ad.MixedDimensionalVariable
        """Ad representation of the phase pressure. Normally initialized by a mixin of
        instance :class:`~porepy.models.abstract_equations.VariableMixin`.

        """
        self.s: pp.ad.MixedDimensionalVariable
        """Ad representation of the phase saturation. Normally initialized by a mixin of
        instance :class:`~porepy.models.abstract_equations.VariableMixin`.

        """

    def set_constants(self, constants: dict[str, number]) -> None:
        """Set phase constants.

        Args:
            constants: Dictionary of phase constants.

        """
        self.constants = PhaseConstants(constants)


class PhaseBC(pp.BoundaryConditionMixin):
    domain_boundary_sides: Callable
    """Normally provided by a mixin of instance
    :class:``~porepy.models.geometry.ModelGeometry``.

    """

    def _bc_type_pressure(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Wetting pressure boundary conditions.

        Neumann conditions on three sides; Dirichlet on the north side to ensure
        existence of a unique solution.

        """
        # Ignore since MyPy complain that the named tuple has no ``north`` attribute.
        north = self.domain_boundary_sides(g).north  # type: ignore
        return pp.BoundaryCondition(g, north, "dir")

    def _dirichlet_bc_values_pressure(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Dirichlet boundary values. Equals the initial state pressure."""
        array = np.zeros(g.num_faces)
        return array

    def _neumann_bc_values_flux(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Neumann boundary values.

        NOTE: In the wetting pressure-wetting saturation formulation, Neumann bc values
        correspond to ``_flux_w``, i.e. the mobility is already incorporated. In other
        formulations this is less clear, however the bc are not needed there. They are
        not needed for upwinding.

        NOTE: We set the values independently from the Dirichlet bc, because otherwise
        they would be counted twice on the RHS (once by ``mpfa.bound_flux`` and once by
        ``upwind.bound_transport_neu``).

        """
        array = np.zeros(g.num_faces)
        return array

    def _bc_values_mobility(self, g: pp.Grid) -> np.ndarray:
        """Wetting mobility at the boundaries.

        NOTE: These are both for Dirichlet bc and Neumann bc. The latter one is needed
        to compute the fractional flow at the boundaries. However, it does not affect
        the Neumann flux.

        NOTE: For some reason, we must choose the negative of the value to get a
        positive mobility at the boundary.

        """
        array = np.full(g.num_faces, 0.25)
        return array


class PhaseConstants(pp.MaterialConstants):
    """Fluid constants for a phase."""

    def __init__(self, constants: Optional[dict[str, number]] = None):
        default_constants = self.default_constants
        self.verify_constants(constants, default_constants)
        if constants is not None:
            default_constants.update(constants)
        super().__init__(default_constants)

    @property
    def default_constants(self) -> dict[str, number]:
        """Default constants of the material.

        Returns:
            Dictionary of constants.

        """
        # Default values, sorted alphabetically
        default_constants: dict[str, number] = {
            "compressibility": 0.0,
            "density": 1000.0,
            "pressure": 0.0,
            "viscosity": 1.0,
            "residual_saturation": 0.0,
        }
        return default_constants

    def compressibility(self) -> number:
        """Compressibility [Pa^-1].

        Returns:
            Compressibility array in converted pressure units.

        """
        return self.convert_units(self.constants["compressibility"], "Pa^-1")

    def density(self) -> number:
        """Density [kg * m^-3].

        Note: For now, we assume incompressible flow, i.e., constant density.

        Returns:
            Density in converted mass and length units.

        """
        return self.convert_units(self.constants["density"], "kg * m^-3")

    def pressure(self) -> number:
        """Pressure [Pa].

        Intended usage: Reference pressure.

        Returns:
            Pressure in converted pressure units.

        """
        return self.convert_units(self.constants["pressure"], "Pa")

    def viscosity(self) -> number:
        """Viscosity [Pa * s].

        Note: For now, we assume incompressible and isothermal flow, i.e., constant
        viscosity.

        Returns:
            Viscosity array in converted pressure and time units.

        """
        return self.convert_units(self.constants["viscosity"], "Pa*s")

    def residual_saturation(self) -> number:
        """Residual saturation [-].

        Returns:
            Residual_saturation array.

        """
        return self.convert_units(self.constants["residual_saturation"], "-")
