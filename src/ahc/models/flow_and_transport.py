r"""This module contains the base model for two-phase flow problems.

The model is implemented in the fractional flow formulation for phase pressures
.. math::
    -\nabla\cdot\left(\lambda_t\nabla p_n - \lambda_n\nabla p_c
    - \lambda_w\nabla\rho_w\bm{g} - \lambda_n\nabla\rho_n\bm{g}\right) = \bm{q}_t,\\
    \phi\frac{\partial S_w}{\partial t} + \nabla\cdot\left(f_w\bm{u}
    + f_w\lambda_n\nabla(p_c + \Delta\rho\bm{g})\right) = \bm{q}_w,

with nonwetting pressure and wetting saturation as the primary variables.

Various constitutive relations, i.e., capillary pressure and relative permeability
models are implemented in :mod:`ahc.models.constitutive_laws_tpf`.

Currently, the module supports only no-flow Neumann boundary conditions, and outflow
Dirichlet boundary conditions, both constant in time. Importantly, the capillary
boundary flux is zero for both cases. For further cases this is not necessarily true:
- Neumann boundaries with flow: Prescribe either total flux and saturation, or both
    phase fluxes. The former approach induces capillary flux over the boundary. The
    latter approach might over-constrain the system if the phase fluxes do not align
    with saturations and fractional flow inside the domain
- Inflow Dirichlet boundaries: A saturation is prescribed and induces a capillary flux.

To differentiate boundaries with and without capillary flux requires modifications to
PorePy. Furthermore, upwinding of phase mobilities at Neumann boundaries and inflow
Dirichlet boundaries requires some careful considerations. Check the documentation of
:class:`~porepy.numerics.fv.upwind.Upwind` for details.


TODO
- Make sure that pressure gets scaled with units and any other physical quantities
    as well.

Units:
    Collection of the SI units for all parameters.
    saturation: dimensionless
    pressure: pascal=kg/(m*s^2)
    density: kg/m^3
    viscosity: kg/(m*s) (cP)
    permeability: m^2
    porosity: dimensionless
    volumetric flux: m^d/(m^(d-1)*s) (:=m/s)
    mass flux: kg/(m^2*s) -> As we assume incompressibility, the fluxes and the source
    terms are measured in volumetric flux.

It follows, that the length unit for the domain is meters and the time unit is seconds.

"""

from __future__ import annotations

import itertools
import logging
import time
import typing
from typing import Any

import numpy as np
import porepy as pp
from ahc.models.constitutive_laws_tpf import TPFConstitutiveLaws
from ahc.models.phase import FluidPhase
from ahc.models.protocol import TPFProtocol
from ahc.numerics.ad.functions import ad_pow as ad_pow
from ahc.utils.constants_and_typing import NONWETTING, WETTING
from porepy.viz.exporter import DataInput

logger = logging.getLogger(__name__)


# region PDEs
class TPFEquations(TPFProtocol, pp.BalanceEquation):
    """This is a model class for two-phase flow problems.


    All attributes are given natural values at initialization of the class.

    The implementation assumes use of AD.

    """

    def phase_fluid_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
        """Volumetric phase source term. Given as volumetric flux. This
        unmodified base function assumes a zero phase source.

        Note: This is the value for a full grid cell, i.e., it does **NOT** get
            integrated over the cell volume in the equation.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        """
        return np.zeros(g.num_cells)

    def total_fluid_source(self, g: pp.Grid) -> np.ndarray:
        """Volumetric total source; sum of the wetting and nonetting source.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        Note: This is the value for a full grid cell, i.e., it does **NOT** get
            integrated over the cell volume in the equation.

        """
        # Ignore mypy. sum will not return Literal[0] for a nonempty list.
        return sum(  # type: ignore
            [self.phase_fluid_source(g, phase) for phase in self.phases.values()]
        )

    def vector_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
        """Volumetric phase vector source. Corresponds to the phase buoyancy flux. This
        unmodified base function assumes a zero vector source.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[..., -1] = pp.GRAVITY_ACCELERATION * self._w_density

        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        # vals[-1] = pp.GRAVITY_ACCELERATION * self.wetting_phase.density
        # For some reason this needs to be a flat array.
        return vals.ravel()

    # More matrix and phase parameters.
    def permeability(self, g: pp.Grid) -> np.ndarray | dict[str, np.ndarray]:
        """Solid permeability. This unmodified base function assumes homogeneous
        permeability. Value and unit are set by :attr:`self.solid`.

        """
        return np.full(g.num_cells, self.solid.permeability())

    def porosity(self, g: pp.Grid) -> np.ndarray:
        """Solid porosity. This unmodified base function assumes homogeneous
        porosity. Value is set by :attr:`self.solid`."""
        return np.full(g.num_cells, self.solid.porosity())

    @typing.override
    def set_equations(self) -> None:
        """Define equations."""
        try:
            self.equation_system.remove_equation("Flow equation")
            self.equation_system.remove_equation("Transport equation")
        except ValueError:
            ValueError("Equations not found.")

        # Spatial discretization operators.
        div = pp.ad.Divergence([self.g])

        # Time derivatives.
        dt_s = pp.ad.time_derivatives.dt(self.wetting.s, self.ad_time_step)

        # Ad source.
        source_ad_w = pp.ad.DenseArray(self.phase_fluid_source(self.g, self.wetting))
        source_ad_t = pp.ad.DenseArray(self.total_fluid_source(self.g))

        # Ad parameters.
        porosity_ad = pp.ad.DenseArray(self.porosity(self.g))

        # Ad equations
        flux_t = self.total_flux(self.g)
        flux_w = self.wetting_flux(self.g)

        flow_equation = pp.ad.Scalar(self.flow_equation_weight) * (
            div @ flux_t - source_ad_t
        )
        transport_equation = pp.ad.Scalar(self.transport_equation_weight) * (
            porosity_ad * (self.volume_integral(dt_s, [self.g], 1))
            + div @ flux_w
            - source_ad_w
        )

        self.flow_equation = "Flow equation"
        self.transport_equation = "Transport equation"

        flow_equation.set_name(self.flow_equation)
        transport_equation.set_name(self.transport_equation)

        # Update the equation list.
        self.equation_system.set_equation(flow_equation, [self.g], {"cells": 1})
        self.equation_system.set_equation(transport_equation, [self.g], {"cells": 1})

        # Secondary variables.
        secondary_pressure: pp.ad.Operator = self.nonwetting.p - self.cap_press(
            self.wetting.s
        )
        secondary_saturation: pp.ad.Operator = pp.ad.Scalar(1) - self.wetting.s

        self.secondary_pressure_eq = "Secondary pressure equation"
        self.secondary_saturation_eq = "Secondary saturation equation"

        secondary_pressure.set_name(self.secondary_pressure_eq)
        secondary_saturation.set_name(self.secondary_saturation_eq)

        self.equation_system.set_equation(secondary_pressure, [self.g], {"cells": 1})
        self.equation_system.set_equation(secondary_saturation, [self.g], {"cells": 1})


class TPFVariablesMixin(TPFProtocol, pp.VariableMixin):
    def create_variables(self) -> None:
        """Create primary variables (wetting pressure, nonwetting pressure,
        saturation)."""
        subdomains = self.mdg.subdomains()
        for phase in self.phases.values():
            phase.p = self.equation_system.create_variables(
                f"{phase.name} pressure",
                {"cells": 1},
                subdomains,
                tags={"si_units": "Pa"},
            )
            phase.s = self.equation_system.create_variables(
                f"{phase.name} saturation",
                {"cells": 1},
                subdomains,
                tags={"si_units": "-"},
            )

        # NOTE The division into primary/secondary variables is internal to this model
        # only and not connected to ``pp.PRIMARY_VARIABLES``.
        self.primary_pressure_var = f"{self.nonwetting.name} pressure"
        self.primary_saturation_var = f"{self.wetting.name} saturation"
        self.secondary_pressure_var = f"{self.wetting.name} pressure"
        self.secondary_saturation_var = f"{self.nonwetting.name} saturation"

    def normalize_saturation(
        self,
        saturation: pp.ad.Operator,
        phase: FluidPhase | None = None,
    ) -> pp.ad.Operator:
        r"""Normalize a given saturation by the residual saturations.

        .. math::
            \hat{S}_w = \frac{S_w - S_w^{min}}{S_w^{max} - S_w^{min}},

        or in other terms

        .. math::
            \hat{S}_w = \frac{S_w - S_{w,res}}{1 - S_{n,res} - S_{w,res}}.

        Parameters:
            saturation: Saturation to be normalized. Can, e.g., be of instance
                :class:`~porepy.ad.MixedDimensionalVariable` or
                :class:`~porepy.ad.SparseArray` (for saturation boundary values).
            phase: Phase object representing the phase the saturation belongs to.

        Returns:
            s_normalized: Normalized wetting saturation.

        Raises:
            ValueError: If neither ``saturation`` has a ``name`` attribute specifying
            the phase nor ``phase`` is specified.

        """
        if (
            saturation.name.startswith(self.wetting.name)
            or getattr(phase, "name", "") == self.wetting.name
        ):
            name: str = self.wetting.name
            residual_saturation_w = pp.ad.Scalar(self.wetting.residual_saturation)
            residual_saturation_n = pp.ad.Scalar(self.nonwetting.residual_saturation)
            s_normalized: pp.ad.Operator = (saturation - residual_saturation_w) / (
                pp.ad.Scalar(1) - residual_saturation_n - residual_saturation_w
            )

        elif (
            saturation.name.startswith(self.nonwetting.name)
            or getattr(phase, "name", "") == self.nonwetting.name
        ):
            name = self.nonwetting.name
            s_normalized = pp.ad.Scalar(1) - self.normalize_saturation(
                pp.ad.Scalar(1) - saturation,
                phase=self.wetting,
            )
        else:
            raise ValueError(
                "``saturation`` must have either a ``name`` attribute"
                + " specifying the phase or ``phase`` must be specified."
            )

        s_normalized.set_name(f"Normalized {name} saturation")
        return s_normalized

    def normalize_saturation_np(
        self,
        saturation: np.ndarray,
        phase: FluidPhase,
    ) -> np.ndarray:
        r"""Normalize a given saturation by the residual saturations.

        For details, see :meth:`normalize_saturation`.

        Returns:
            s_normalized: Normalized wetting saturation.

        Raises:
            ValueError: If ``phase.name`` is not equal to the wetting or nonwetting
            phase name.

        """
        if getattr(phase, "name", "") == self.wetting.name:
            s_normalized: np.ndarray = (
                saturation - self.wetting.residual_saturation
            ) / (
                1.0
                - self.nonwetting.residual_saturation
                - self.wetting.residual_saturation
            )

        elif getattr(phase, "name", "") == self.nonwetting.name:
            s_normalized = 1.0 - self.normalize_saturation_np(
                1.0 - saturation,
                phase=self.wetting,
            )
        else:
            raise ValueError(
                "``phase.name`` must be equal to the wetting or nonwetting phase name."
            )
        return s_normalized

    def normalize_saturation_deriv(
        self,
        phase: FluidPhase,
    ) -> pp.ad.Operator:
        r"""Derivative of the normalized saturation.

        .. math::
            \hat{S}_w = \frac{S_w - S_w^{min}}{S_w^{max} - S_w^{min}},

        or in other terms

        .. math::
            \hat{S}_w = \frac{S_w - S_{w,res}}{1 - S_{n,res} - S_{w,res}}.

        Parameters:
            saturation: Saturation to be normalized. Can, e.g., be of instance
                :class:`~porepy.ad.MixedDimensionalVariable` or
                :class:`~porepy.ad.SparseArray` (for saturation boundary values).
            phase: Phase object representing the phase the saturation belongs to.

        Returns:
            Normalized wetting saturation.

        Raises:
            ValueError: If ``phase.name`` is not equal to the wetting or nonwetting
            phase name.

        """
        residual_saturation_w: float = self.wetting.residual_saturation
        residual_saturation_n: float = self.nonwetting.residual_saturation
        # TODO This shall return exactly the same type as saturation, but set to 0
        # outside the residual saturation range. How to do this for a general
        # pp.ad.Operator? Is it necessary to return the same type as saturation?
        if phase.name == self.wetting.name:
            return pp.ad.Scalar(1 - residual_saturation_n - residual_saturation_w)
        elif phase.name == self.nonwetting.name:
            return pp.ad.Scalar(residual_saturation_w + residual_saturation_n - 1)
        else:
            raise ValueError(
                "``saturation`` must have either a ``name`` attribute"
                + " specifying the phase or ``phase`` must be specified."
            )

    def normalize_saturation_deriv_np(
        self,
        phase: FluidPhase,
    ) -> float:
        r"""Derivative of the normalized saturation.

        For details, see :meth:`normalize_saturation_deriv`.

        Parameters:
            saturation: Saturation to be normalized.
            phase: Phase object representing the phase the saturation belongs to.

        Returns:
            Normalized wetting saturation.

        Raises:
            ValueError: If ``phase.name`` is not equal to the wetting or nonwetting
            phase name.

        """
        if phase.name == self.wetting.name:
            return (
                1
                - self.wetting.residual_saturation
                - self.nonwetting.residual_saturation
            )
        elif phase.name == self.nonwetting.name:
            return (
                self.wetting.residual_saturation
                + self.nonwetting.residual_saturation
                - 1
            )
        else:
            raise ValueError(
                "``saturation`` must have either a ``name`` attribute"
                + " specifying the phase or ``phase`` must be specified."
            )


# endregion

# region SOLUTION STRATEGY & BC


class TPFBoundaryConditionsMixin(pp.BoundaryConditionMixin):
    """This class provides boundary conditions for two-phase flow problems.

    - Dirichlet boundary: Phase pressure and saturation values are provided.
    - Neumann boundary: Total flux and phase saturation values are provided.

    This unmodified base class provides homogeneous Dirichlet boundary conditions with
    wetting saturation values set to ``1.0`` and nonwetting saturation values set to
    ``0.0``.

    """

    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann)."""
        # Dirichlet conditions for both phases.
        boundary_faces = self.domain_boundary_sides(g).all_bf
        return pp.BoundaryCondition(g, boundary_faces, "dir")

    @typing.final
    def bc_dirichlet_pressure_values(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
        """Phase dependent pressure values on Dirichlet boundaries; nulled on Neumann
        boundaries.

        On Neumann boundaries, the pressure values MUST be 0 to not accidentally
        prescibe a flux! Both ``Mpfa.bound_flux()`` and ``Tpfa.bound_flux()`` provide
        matrices with ``1`` values at Neumann boundaries. Nulling the pressure values,
        ensures that ``*pfa.bound_flux() @ p_..._bc`` is zero.

        Returns:
            p_bc: ``shape=(g.num_faces,)`` Phase pressure boundary values.

        """
        is_neu: np.ndarray = self.bc_type(g).is_neu
        p_bc: np.ndarray = self._bc_dirichlet_pressure_values(g, phase)
        p_bc[is_neu] = 0
        return p_bc

    def _bc_dirichlet_pressure_values(
        self, g: pp.Grid, phase: FluidPhase
    ) -> np.ndarray:
        """Homogeneous phase pressure on Dirichlet boundaries.

        Returns:
            p_bc: ``shape=(g.num_faces,)`` Phase pressure boundary values.

        """
        return np.zeros(g.num_faces)

    @typing.override
    def update_all_boundary_conditions(self) -> None:
        """Set values for the saturation, flux, and  and the darcy flux on
        boundaries."""
        # TODO This does not work as of yet. Replace the direct call of the functions
        # ``self.bc_..._values`` with ``create_boundary_operator`` called on the
        # appropriate key.
        # super().update_all_boundary_conditions()

        # self.update_boundary_condition(
        #     name=self.primary_pressure_variable,
        #     function=self.bc_dirichlet_pressure_values,
        # )
        # self.update_boundary_condition(
        #     name=self.primary_pressure_variable,
        #     function=self.bc_dirichlet_saturation_values,
        # )
        # self.update_boundary_condition(
        #     name=self.bc_data_total_flux, function=self.bc_values_darcy_flux
        # )
        # self.update_boundary_condition(
        #     name=self.bc_data_fluid_flux_key, function=self.bc_values_fluid_flux
        # )


# mypy complains because the type of nonlinear_solver_statistics differs for TPFProtocol
# and pp.SolutionStrategy. This is safe in practice, but ``nonlinear_solver_statistics``
# must be used with care.
class TPFSolutionStrategy(TPFProtocol, pp.SolutionStrategy):  # type: ignore
    @typing.override
    def __init__(self, params: dict | None) -> None:
        super().__init__(params)

        flow_equation_weight = self.params.get("flow_equation_weight", 1.0)
        if isinstance(flow_equation_weight, float):
            self.flow_equation_weight: float = flow_equation_weight
        else:
            raise ValueError(
                f"expected flow_equation_weight to be float, got {flow_equation_weight}"
            )
        """Weighting factor for the flow equation in the residual and Jacobian."""

        transport_equation_weight = self.params.get("transport_equation_weight", 1.0)
        if isinstance(transport_equation_weight, float):
            self.transport_equation_weight: float = transport_equation_weight
        else:
            raise ValueError(
                "expected transport_equation_weight to be float, got "
                + f"{transport_equation_weight}"
            )
        """Weighting factor for the transport equation in the residual and Jacobian."""

        # Initialize fluid phases. NOTE This is already done during initialization and
        # not prepare simulation s.t. the keywords can be defined based on the phase
        # names. See below.
        self.set_phases()

        # Discretizations and parameter keywords.
        self.flux_key: str = "total_flux"
        """Keyword to define define parameters and discretizations for the total flux.

        The corresponding ``tpfa`` and ``tpfa`` discretizations are used to calculate
        phase pressure potentials, i.e., wetting and nonwetting.

        """
        self.cap_potential_key: str = "cap_potential"
        """Keyword to define define parameters and discretizations for the capillary
        pressure potential flux.

        """

        for phase in self.phases.values():
            setattr(phase, "mobility_key", f"{phase.name}_mobility")
            """Keyword to define parameters and discretizations for the phase mobility.

            As phase flows can have opposite signs, independent upwind discretization
            are required to evaluate phase mobilities.

            """

        # Solvers:
        self._use_ad: bool = True
        nl_appleyard_chopping = self.params.get("nl_appleyard_chopping", False)
        if isinstance(nl_appleyard_chopping, bool):
            self._nl_appleyard_chopping: bool = nl_appleyard_chopping
        else:
            raise ValueError(
                "expected nl_appleyard_chopping to be bool, got "
                + f"{nl_appleyard_chopping}"
            )
        """Whether to use the Appleyard chopping strategy for the nonlinear solver.

        If ``True``, chop local saturation changes per nonlinear iteration that are
        larger than :math:`0.2`.

        """

        nl_enforce_physical_saturation = self.params.get(
            "nl_enforce_physical_saturation", False
        )
        if isinstance(nl_enforce_physical_saturation, bool):
            self._nl_enforce_physical_saturation: bool = nl_enforce_physical_saturation
        else:
            raise ValueError(
                "expected nl_enforce_physical_saturation to be bool, got "
                + f"{nl_enforce_physical_saturation}"
            )
        """Whether to enforce physical saturation bounds at each nonlinear iteration.

        If ``True``, the saturation is limited to the range of residual saturations
        :math:`[s_{w,res}, 1 - s_{n,res}]` at each nonlinear iteration.

        Note: Narrows the interval by a small epsilon to avoid nonphysical saturations
            due to floating point errors in the process of chopping.

        """

    @typing.override
    def _is_time_dependent(self) -> bool:
        return True

    @typing.override
    def _is_nonlinear_problem(self) -> bool:
        return True

    @property
    def uses_hc(self) -> bool:
        return False

    def set_phases(self) -> None:
        """Set phase constants from parameters."""
        # Check types to satisfy mypy.
        constants_raw: object = self.params.get("material_constants", {})
        if not isinstance(constants_raw, dict) or not all(
            isinstance(v, pp.MaterialConstants) for v in constants_raw.values()
        ):
            raise TypeError("material_constants must be a dict of MaterialConstants")
        constants: dict[str, pp.MaterialConstants] = constants_raw

        self.phases: dict[str, FluidPhase] = {}

        for phase_name in [WETTING, NONWETTING]:
            if phase_name not in constants:
                # Use standard values for phase constants if not provided.
                phase: FluidPhase = FluidPhase({"name": phase_name})
                logger.info(
                    f"No {phase_name} constants provided in params."
                    + " Using default values."
                )
            else:
                phase_constant = constants[phase_name]
                if not isinstance(phase_constant, FluidPhase):
                    raise TypeError(
                        f"constats[{phase_name}] must be of type FluidPhase."
                    )
                phase = phase_constant

            # Add phase to model.
            phase.set_units(self.units)
            setattr(self, phase_name, phase)
            self.phases[phase_name] = phase

    @typing.override
    def set_materials(self) -> None:
        """Set solid constants from parameters."""
        # Check types to satisfy mypy.
        constants_raw: object = self.params.get("material_constants", {})
        if not isinstance(constants_raw, dict) or not all(
            isinstance(v, pp.MaterialConstants) for v in constants_raw.values()
        ):
            raise TypeError("material_constants must be a dict of MaterialConstants")
        constants: dict[str, pp.MaterialConstants] = constants_raw

        if "solid" in constants:
            if not isinstance(constants["solid"], pp.SolidConstants):
                raise TypeError("solid must be of type SolidConstants.")
            self.solid = constants["solid"]
        else:
            self.solid = pp.SolidConstants()
            logger.info("No solid constants provided in params. Using default values.")

        self.solid.set_units(self.units)
        # 'fluid' is not used in two-phase flow, but initialized to avoid typing errors.
        self.fluid = pp.FluidConstants()

    @typing.override
    def prepare_simulation(self) -> None:
        """This setups the model, s.t. a simulation can be run.

        After setup is finished, the initial simulation values are exported.

        """
        # Set the model parameters.
        self.set_rel_perm_constants()
        self.set_cap_press_constants()
        self.set_materials()
        self.set_geometry()

        self.g, self.g_data = self.mdg.subdomains(return_data=True)[0]

        # Exporter initialization after grid creation, but prior to data initialization.
        self.set_solver_statistics()
        self.initialize_data_saving()

        # Create the numerical aparatus.
        self.set_equation_system_manager()
        self.create_variables()
        self.initial_condition()
        self.set_discretization_parameters()
        self.set_equations()

        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        # Save the initial values.
        self.save_data_time_step()

    @typing.override
    def set_discretization_parameters(self) -> None:
        """Set constant bc and darcy flux based on phase potentials.

        The parameter fields of the data dictionaries are updated for all
        subdomains and interfaces (of codimension 1).


        Upwinding needs to be rediscretized at each nonlinear iteration, as the phase
        flux changes. See the last sentence on page 686 of [Y. Brenier, J. Jaffré,
        Upstream differencing for multiphase flow in reservoir simulation, SIAM J.
        Numer. Anal. 28 (3) (1991) 685–696.] for details.

        To evaluate the phase-mobilities separately, both the wetting, as well as the
        nonwetting flux need to be computed.

        """
        super().set_discretization_parameters()

        # Constant parameters for TPFA discretizations.
        if (
            self.nonlinear_solver_statistics.num_iteration == 0
            and self.time_manager.time_index == 0
        ):
            perm = self.permeability(self.g)
            # Different treatment for scalar and tensor permeability.
            if isinstance(perm, np.ndarray):
                diffusivity = pp.SecondOrderTensor(perm)
            elif isinstance(perm, dict):
                diffusivity = pp.SecondOrderTensor(**perm)
            pp.initialize_data(
                self.g,
                self.g_data,
                self.flux_key,
                {
                    "bc": self.bc_type(self.g),
                    "second_order_tensor": diffusivity,
                    "ambient_dimension": self.g.dim,
                    "darcy_flux": np.ones(self.g.num_faces),
                },
            )
            # All Neumann bc to evaluate cap. press. potential.
            all_neu_bc = pp.BoundaryCondition(
                self.g,
            )
            pp.initialize_data(
                self.g,
                self.g_data,
                self.cap_potential_key,
                {
                    "bc": all_neu_bc,
                    "second_order_tensor": diffusivity,
                    "ambient_dimension": self.g.dim,
                },
            )

        # Update phase potentials for upwinding each nonlinear iteration. Newton starts
        # from the previous timestep, improving the guess each step.
        logger.info(
            "Recalculate Darcy flux for upwind discretization."
            + f" Iteration {self.nonlinear_solver_statistics.num_iteration}"
        )
        for phase in self.phases.values():
            if self.nonlinear_solver_statistics.num_iteration > 0:
                # Ignore mypy. Evaluating the phase potential will always return an
                # array.
                phase_potential: np.ndarray = self.phase_potential(self.g, phase).value(  # type: ignore
                    self.equation_system
                )
            elif self.time_manager.time_index == 0:
                # Use unit values for the potential at the start of the simulation.
                phase_potential = -np.ones(self.g.num_faces)
            else:
                # Use previous values at the start of a new time step.
                return None
            pp.initialize_data(
                self.g,
                self.g_data,
                phase.mobility_key,
                {
                    "bc": self.bc_type(self.g),
                    "darcy_flux": phase_potential,
                },
            )

    @typing.override
    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation.

        Note: The initial pressure values serve only as an initial guess to the
        nonlinear solver but do not influence the solution.

        """
        self.equation_system.set_variable_values(
            np.full(self.g.num_cells * 2, 0.0),
            [self.wetting.p, self.nonwetting.p],
            time_step_index=0,
            iterate_index=0,
        )
        initial_saturation = np.full(self.g.num_cells, 0.5)
        initial_saturation = self.bound_saturation(
            initial_saturation,
            is_increment=False,
        )
        self.equation_system.set_variable_values(
            np.concatenate([initial_saturation, 1 - initial_saturation]),
            [self.wetting.s, self.nonwetting.s],
            time_step_index=0,
            iterate_index=0,
        )

    def bound_saturation(
        self,
        s_w: np.ndarray,
        is_increment: bool = False,
        time_step_index: int | None = None,
        iterate_index: int | None = None,
    ) -> np.ndarray:
        r"""Ensure that the saturation is within physical and solver bounds.

        The saturation is bounded both to physical bounds according to the Appleyard
        chopping. The behavior is defined by the model parameters
        :attr:`nl_appleyard_chopping` and :attr:`nl_enforce_physical_saturation`.

        Note: Appleyard chopping is only applied to increments, while physical bounds
            are applied to both increments and actual saturation values.

        Parameters:
            s_w: Saturation or saturation increment to be bounded.
            is_increment: Whether ``s_w`` is an increment or the actual saturation.
                Default is ``False``.
            time_step_index: Time step index at which the old saturation is stored.
                Default is ``None``.
            iterate_index: Iterate index at which the old saturation is stored.
                Default is ``None``.

        Raises:
            ValueError: If ``is_increment`` is ``True``, but neither ``time_step_index``
                nor ``iterate_index`` is specified.

        """
        if is_increment and (time_step_index is None and iterate_index is None):
            raise ValueError(
                "If ``is_increment`` is ``True``, either ``time_step_index`` or"
                + " ``iterate_index`` must be specified."
            )

        if self._nl_appleyard_chopping and is_increment:
            s_w = np.clip(s_w, -0.2, 0.2)

        if self._nl_enforce_physical_saturation:
            # Enforce physical saturation bounds.

            if is_increment:
                # Compute updated saturation from increment and previous value.
                s_w_prev = self.equation_system.get_variable_values(
                    [self.wetting.s],
                    time_step_index=time_step_index,
                    iterate_index=iterate_index,
                )
                s_w = s_w_prev + s_w

            s_w = np.clip(
                s_w,
                self.wetting.residual_saturation + self.wetting.saturation_epsilon,
                1.0
                - self.nonwetting.residual_saturation
                - self.nonwetting.saturation_epsilon,
            )

            if is_increment:
                # Return the chopped increment.
                s_w = s_w - s_w_prev

        return s_w

    @typing.override
    def discretize(self) -> None:
        """Discretize all terms."""
        t_0 = time.time()
        self.equation_system.discretize()
        for phase in self.phases.values():
            phase_potential = self.phase_potential(self.g, phase)
            phase_potential.discretize(self.mdg)
        logger.debug(f"Discretized in {time.time() - t_0:.2e} seconds")

    @typing.override
    def set_nonlinear_discretizations(self) -> None:
        self.add_nonlinear_discretization(
            self.phase_mobility_discretization(self.g, self.wetting).upwind(),
        )
        self.add_nonlinear_discretization(
            self.phase_mobility_discretization(self.g, self.nonwetting).upwind(),
        )

    @typing.override
    def assemble_linear_system(self) -> None:
        """Assemble the linearized system.

        The linear system is defined by the current state of the model.

        Attributes:
            linear_system is assigned.

        """
        t_0 = time.time()
        self.linear_system = self.equation_system.assemble(
            equations=[self.flow_equation, self.transport_equation],
            variables=[self.primary_saturation_var, self.primary_pressure_var],
        )
        logger.debug(f"Assembled linear system in {time.time() - t_0:.2e} seconds")

    def assemble_residual(self) -> np.ndarray:
        """Assemble the residual."""
        return self.equation_system.assemble(
            evaluate_jacobian=False,
            equations=[self.flow_equation, self.transport_equation],
        )

    # region NONLINEAR LOOP
    @typing.override
    def before_nonlinear_loop(self) -> None:
        """Set the starting estimate to the solution from the previous timestep."""
        # Update time step size and empty statistics.
        self.ad_time_step.set_value(self.time_manager.dt)
        self.nonlinear_solver_statistics.reset()
        self.convergence_status = False

        time_step_values = self.equation_system.get_variable_values(time_step_index=0)
        self.equation_system.set_variable_values(
            time_step_values, iterate_index=0, additive=False
        )

    @typing.override
    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the new trial state, visualize
        the current approximation etc.

        Parameters:
            nonlinear_increment: The new solution, as computed by the non-linear solver.

        """
        # Store the non-chopped saturation.
        if self._nl_appleyard_chopping or self._nl_enforce_physical_saturation:
            self.non_chopped_nonlinear_increment: np.ndarray = (
                nonlinear_increment.copy()
            )

        # Apply Appleyard chopping and/or enforce physical saturation bounds.
        # Saturation comes first in the nonlinear increment.
        nonlinear_increment[: self.g.num_cells] = self.bound_saturation(
            nonlinear_increment[: self.g.num_cells],
            is_increment=True,
            iterate_index=0,
        )

        # Update primary variables.
        self.equation_system.shift_iterate_values(max_index=len(self.iterate_indices))
        self.equation_system.set_variable_values(
            values=nonlinear_increment,
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            additive=True,
            iterate_index=0,
        )
        self.eval_secondary_variables()
        self.nonlinear_solver_statistics.num_iteration += 1

    def eval_secondary_variables(self) -> None:
        """Evaluate and update secondary variables."""
        # Take the negative of the assembled values, as ``equation_system.assemble``
        # returns the negative of the RHS.
        secondary_pressure_sol = -self.equation_system.assemble(
            evaluate_jacobian=False, equations=[self.secondary_pressure_eq]
        )
        secondary_saturation_sol = -self.equation_system.assemble(
            evaluate_jacobian=False, equations=[self.secondary_saturation_eq]
        )
        #  Since the values were computed with the additive value of the primary
        #  pressure and saturation variable, we set ``additive=False``.
        self.equation_system.set_variable_values(
            np.concatenate(
                [np.array(secondary_pressure_sol), np.array(secondary_saturation_sol)]
            ),
            variables=[self.secondary_pressure_var, self.secondary_saturation_var],
            iterate_index=0,
            additive=False,
        )

    @typing.override
    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Implement an additional divergence check that returns true if the nonlinear
        increment is unreasonably large."""
        converged, diverged = super().check_convergence(
            nonlinear_increment, residual, reference_residual, nl_params
        )
        nonlinear_increment_norm: float = self.compute_nonlinear_increment_norm(
            nonlinear_increment
        )
        if nonlinear_increment_norm > nl_params["nl_divergence_tol"]:
            diverged = True
        self.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm=None,
            residual_norm=None,
            time_step_index=self.time_manager.time_index,
            time=self.time_manager.time,
            time_step_size=self.time_manager.dt,
        )

        return converged, diverged

    def compute_nonlinear_increment_norm(
        self, nonlinear_increment: np.ndarray
    ) -> float:
        """Compute the norm based on the update increment for a nonlinear iteration

        Note: The pressure and saturation parts can get scaled independently.
            Depending on the simulation setup, the pressure values might be several
            orders of magnitude larger than the saturation values.
            In the model parameters, pass the following keywords:
            - ``"nl_sat_increment_norm_scaling"`` to scale the saturation increment.
            - ``"nl_press_increment_norm_scaling"`` to scale the pressure increment.

        Parameters:
            nonlinear_increment: Solution to the linearization.

        Returns:
            float: Update increment norm.

        """
        # The saturation comes first in the nonlinear increment.
        nonlinear_increment_sat = nonlinear_increment[: self.g.num_cells]
        nonlinear_increment_press = nonlinear_increment[self.g.num_cells :]

        nl_sat_increment_norm_scaling_raw = self.params.get(
            "nl_sat_increment_norm_scaling", 1.0
        )
        if isinstance(nl_sat_increment_norm_scaling_raw, float):
            nl_sat_increment_norm_scaling: float = nl_sat_increment_norm_scaling_raw
        else:
            raise TypeError(
                "expected params[nl_sat_increment_norm_scaling] to be float"
                f", got {nl_sat_increment_norm_scaling_raw}"
            )

        nonlinear_increment_sat_norm = (
            np.linalg.norm(nonlinear_increment_sat) / nl_sat_increment_norm_scaling
        )
        nonlinear_increment_press_norm = np.linalg.norm(nonlinear_increment_press) / (
            nl_sat_increment_norm_scaling
        )
        return np.sqrt(
            nonlinear_increment_sat_norm**2 + nonlinear_increment_press_norm**2
        ) / np.sqrt(nonlinear_increment.size)

    # endregion

    @typing.override
    def after_simulation(self):
        pass


# endregion


class TPFDataSavingMixin(TPFProtocol, pp.DataSavingMixin):
    @typing.override
    def _evaluate_and_scale(
        self,
        grid: pp.Grid | pp.MortarGrid,
        method_name: str,
        units: str,
    ) -> np.ndarray:
        """Evaluate a method for a derived quantity and scale the result to SI units.

        Parameters:
            grid: Grid or mortar grid for which the method should be evaluated.
            method_name: Name of the method to be evaluated.
            units: Units of the quantity returned by the method. Should be parsable by
                :meth:`porepy.fluid.FluidConstants.convert_units`.

        Returns:
            Array of values for the quantity, scaled to SI units.

        """
        vals_scaled = getattr(self, method_name)([grid]).value(self.equation_system)
        vals = self.solid.convert_units(vals_scaled, units, to_si=True)
        return vals

    @typing.override
    def data_to_export(self) -> list[DataInput]:
        return self._data_to_export(time_step_index=0)

    def data_to_export_iteration(self) -> list[DataInput]:
        return self._data_to_export(iterate_index=0)

    def _data_to_export(
        self,
        time_step_index: int | None = None,
        iterate_index: int | None = None,
    ) -> list[DataInput]:
        """Return data to be exported.

        Return type should comply with :class:`~porepy.exporter.DataInput`.

        Returns:
            List containing all (grid, name, scaled_values) tuples.

        """
        if time_step_index is None and iterate_index is None:
            msg: str = "Either time_step_index or iterate_index must be provided."
            raise ValueError(msg)
        data: list[DataInput] = []
        for phase, var_name in itertools.product(self.phases.values(), ["p", "s"]):
            var = getattr(phase, var_name)
            scaled_values = self.equation_system.get_variable_values(
                variables=[var],
                time_step_index=time_step_index,
                iterate_index=iterate_index,
            )
            units = var.tags["si_units"]
            values = phase.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain[0], var.name, values))

        # Add secondary variables/derived quantities.
        data.append(
            (
                self.g,
                "specific_volume",
                self._evaluate_and_scale(
                    self.g, "specific_volume", f"m^{self.nd - self.g.dim}"
                ),
            )
        )

        return data


# Protocols define different types for ``nonlinear_solver_statistics``, causing mypy
# errors. This is safe in practice, but ``nonlinear_solver_statistics`` must be used
# with care. We ignore the error.
class TwoPhaseFlow(  # type: ignore
    TPFEquations,
    TPFVariablesMixin,
    TPFConstitutiveLaws,
    TPFBoundaryConditionsMixin,
    TPFSolutionStrategy,
    TPFDataSavingMixin,
    pp.ModelGeometry,
): ...
