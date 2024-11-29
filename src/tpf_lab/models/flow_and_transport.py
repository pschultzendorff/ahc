r"""This module contains an implementation of a base model for two-phase flow problems.

Currently, the fractional flow formulation solved for nonwetting pressure and wetting
saturation
.. math::
    -\nabla\cdot\left(\lambda_t\nabla p_n - \lambda_n\nabla p_c
    - \lambda_w\nabla\rho_w\bm{g} - \lambda_n\nabla\rho_n\bm{g}\right) = \bm{q}_t,\\
    \phi\frac{\partial S_w}{\partial t} + \nabla\cdot\left(f_w\bm{u}
    + f_w\lambda_n\nabla(p_c + \Delta\rho\bm{g})\right) = \bm{q}_w,

is implemented.

Furthermore, multiple different models for both the capillary pressure, as well as the
relative permeability are implemented.

TODO
    - Change bc_values to ``ad.BoundaryCondition``
    - Right now alot of the things in this model are typed as multi-grid objects. For
      example, the variables are of type ``MultiGridVariable``. To simplify things, change
      to ``Variable``. Does this make sense? ``EquationSystem`` does not have a function to
      initialize single grid variables.
    - Implement new PorePy functionalities that allow for easier model setup, e.g.,
      everything set by ``set_material`` can be passed as parameters and does not have to
      be hardcoded. Also, make use of ``Units``.
    - Remove or fix the unit documentation. The units can depend on the instance of
      ``porepy.Units`` passed to the simulation.
    - Make use of TypeVars for typing of some functions and in general type everything.
    - Negative phase sources give the wrong results (i.e., no fluid gets pulled). Fix
      this!!!
    - Right now, we assume that the phases are called ``WETTING`` and ``NONWETTING``,
      with the constants having string values "wetting" and "nonwetting". Make
      everything work for different phase names, e.g., "brine" and "co2".

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

It follows, that the domain unit is meters and the time unit is seconds.

"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Any, Callable, Literal, NamedTuple, Optional, TypeVar

import numpy as np
import porepy as pp
from porepy.utils.examples_utils import VerificationUtils
from tpf_lab.constants_and_typing import (
    NONWETTING,
    PHASENAME,
    REL_PERM_MODEL,
    WETTING,
    OperatorType,
)
from tpf_lab.models.constitutive_laws_tpf import CapillaryPressure, RelativePermeability
from tpf_lab.models.phase import Phase, PhaseConstants
from tpf_lab.numerics.ad.functions import ad_pow as ad_pow
from tpf_lab.numerics.ad.functions import minimum
from tpf_lab.visualization.diagnostics import SaveDataTPF

logger = logging.getLogger(__name__)


# region CONSTITUTIVE LAWS


class DarcyFluxes:

    rel_perm: Callable[[pp.ad.Operator, Phase], pp.ad.Operator]
    """Phase relative permeability. Normally provided by a mixin of instance
    :class:`RelativePermeability`.

    """
    cap_press: Callable[[pp.ad.Operator], pp.ad.Operator]
    """Capillary pressure. Normally provided by a mixin of instance
    :class:`CapillaryPressure`.

    """
    vector_source: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase vector sources, i.e., buyoyancy terms. Normally provided by a mixin of
    instance :class:`EquationsTPF`.

    """

    wetting: Phase
    """Wetting phase class, providing phase name and phase constants. Normally provided
    by a mixin of instance :class:`SolutionStrategyTPF`.

    """
    nonwetting: Phase
    """Nonwetting phase class, providing phase name and phase constants. Normally
    provided by a mixin of instance :class:`SolutionStrategyTPF`.

    """
    phases: dict[str, Phase]
    """List of fluid phases, providing phase names and phase constants. Normally
    provided by a mixin of instance :class:`SolutionStrategyTPF`.

    """
    equation_system: pp.ad.EquationSystem
    """Equation system. Normally provided by a mixin of instance
    :class:`SolutionStrategyTPF`.

    """
    flux_key: str
    """Keyword to define parameters and discretizations for the total flux. Normally
    provided by a mixin of instance :class:`SolutionStrategyTPF`.

    """

    bc_type: Callable[[pp.Grid], pp.BoundaryCondition]
    """BC type (Neumann or Dirichlet) for flux and mobility discretization. Normally
    provided by a mixin of instance :class:`BoundaryConditionsTPF`.

    """
    bc_dirichlet_pressure_values: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase dependent pressure bc values. Normally provided by a mixin of instance
    :class:`BoundaryConditionsTPF`.

    """
    bc_dirichlet_saturation_values: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase dependent saturation bc values. Normally provided by a mixin of instance
    :class:`BoundaryConditionsTPF`.

    """
    bc_neumann_flux_values: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase flux bc values. Normally provided by a mixin of instance
    :class:`BoundaryConditionsTPF`.

    """

    def phase_mobility(
        self,
        g: pp.Grid,
        phase: Phase,
    ) -> pp.ad.Operator:
        """

        Parameters:
            g:
            phase:

        Returns:
            Phase mobility.

        """
        # TODO Does it make more sense to get this from the dictionary? Would need to
        # change ``BuckleyLeverett`` then, as the mobilities are updated at iterations
        # via ``self._bc_values_mobility_w``.
        # NOTE At Neumann boundaries, both phase fluxes are prescribed, hence no need
        # to determine any phase mobilities.

        saturation_w = self.wetting.s
        saturation_w_bc = pp.ad.DenseArray(
            self.bc_dirichlet_saturation_values(g, self.wetting),
            name=f"{self.wetting.name}_s_bc",
        )
        viscosity = pp.ad.Scalar(
            phase.constants.viscosity(), name=f"{phase.name}_viscosity"
        )
        upwind = pp.ad.UpwindAd(phase.mobility_key, [g])

        mobility = upwind.upwind() @ (
            self.rel_perm(saturation_w, phase) / viscosity
        ) + upwind.bound_transport_dir() @ (
            self.rel_perm(saturation_w_bc, phase) / viscosity
        )
        mobility.set_name(f"{phase.name} mobility")
        # NOTE Alternatively to the current presciption of both phase fluxes at Neumann
        # boundaries, one could also prescibe both the total flux over boundaries and
        # the saturation. This would require to upwind phase mobilities and capillary
        # pressures at Neumann boundaries to determine phase fluxes. Upwinding direction
        # would also be based on phase fluxes (from the previous time step)
        # This is a bit tricky in PorePy, since ``upwind.bound_transport_neu()`` is ``1`` on
        # **all** Neumann boundary faces while ``upwind.upwind()`` is ``0`` on all Neumann
        # boundary faces, independent of flow direction.
        # This could be solved in the following way:
        # 1. We use one upwind discretization with ``phase.mobility_key`` to obtain
        # ``upwind.bound_transport_neu()`` and manually null all outflow boundary faces.
        # 2. We use a second upwind discretization with ``all_bc_dir_key`` that
        # corresponds to fully Dirichlet bc and manually null all cells adjacent to an
        # inflow boundary.
        # Code proposal:
        # upwind_all_dir = pp.ad.UpwindAd(self.all_bc_dir_key, [g])
        # flux_bc: np.ndarray = self.bc_neumann_flux_values(g)
        # neumann_ind: np.ndarray = np.where(self.bc_type(g).is_neu)[0]
        # neumann_inflow_ind: np.ndarray = np.logical_and(neumann_ind, flux_bc > 0)
        # neumann_outflow_ind: np.ndarray = np.logical_and(neumann_ind, flux_bc < 0)

        return mobility

    def total_mobility(self, g: pp.Grid) -> pp.ad.Operator:
        """We add a small epsilon to avoid division by zero when calculating the
        fractional flow at Neumann boundaries.

        Parameters:
            g:

        Returns:
            Total mobility.

        """
        return pp.ad.sum_operator_list(
            [self.phase_mobility(g, phase) for phase in self.phases.values()],
            name="total mobility",
        ) + pp.ad.Scalar(1e-7)

    def phase_flux(self, g: pp.Grid, phase: Phase) -> pp.ad.Operator:
        """Phase volume flux. Combines advective and buoyancy components.

        SI Units: kg/s -> Depends on the units of the other parameters.

        """
        # Phase data.
        p_bc = pp.ad.DenseArray(self.bc_dirichlet_pressure_values(g, phase))
        vector_source = pp.ad.DenseArray(self.vector_source(g, phase))
        flux_bc_neu = pp.ad.DenseArray(self.bc_neumann_flux_values(g, phase))

        # Discretizations.
        mpfa = pp.ad.MpfaAd(self.flux_key, [g])
        upwind = pp.ad.UpwindAd(self.flux_key, [g])

        # Phase flux terms.
        p_diff: pp.ad.Operator = mpfa.flux() @ phase.p + mpfa.bound_flux() @ p_bc
        flux_buoyancy: pp.ad.Operator = mpfa.vector_source() @ vector_source

        # Phase mobility.
        mobility = self.phase_mobility(g, phase)

        # Add together.
        flux: pp.ad.Operator = (
            mobility * (p_diff - flux_buoyancy)
            + upwind.bound_transport_neu() @ flux_bc_neu
        )
        flux.set_name(f"{phase.name} volume flux")
        return flux

    def phase_potential(self, g: pp.Grid, phase: Phase) -> pp.ad.Operator:
        """Phase potential flux. Combines advective and buoyancy components.

        Note: This is zero at Neumann boundaries.

        """
        # Phase data.
        p_bc = pp.ad.DenseArray(self.bc_dirichlet_pressure_values(g, phase))
        vector_source = pp.ad.DenseArray(self.vector_source(g, phase))

        # Discretization.
        mpfa = pp.ad.MpfaAd(self.flux_key, [g])

        # Phase flux terms.
        p_potential: pp.ad.Operator = mpfa.flux() @ phase.p + mpfa.bound_flux() @ p_bc
        buyoancy_potential: pp.ad.Operator = mpfa.vector_source() @ vector_source

        # Add together:
        total_potential: pp.ad.Operator = p_potential - buyoancy_potential
        total_potential.set_name(f"{phase.name} potential")
        return total_potential

    def total_flux(self, g: pp.Grid) -> pp.ad.Operator:
        """Total volume flux.

        This is always calculated in terms of the nonwetting pressure and the capillary
        pressure (i.e. in terms of the wetting Saturation). Note that, unlike in the
        phase flux functions, the mobilities are already included in this formulation.

        SI Units: kg/s -> Depends on the units of the other parameters.

        """
        # Variables and bc.
        # We want to ensure that no diffusive flux due to nonwetting or capillary
        # pressure contributes to total flux at Neumann boundaries. Thus, we manually
        # null both ``p_n_bc`` and ``p_cap_bc`` at Neumann boundaries. Thus,
        # ``tpfa.bound_flux() @ p_..._bc`` will be zero.
        neumann_ind = np.where(self.bc_type(g).is_neu)[0]
        p_n_bc_values: np.ndarray = self.bc_dirichlet_pressure_values(
            g, self.nonwetting
        )
        p_n_bc_values[neumann_ind] = 0
        p_n_bc = pp.ad.DenseArray(p_n_bc_values)

        # FIXME This is a little inefficient. Would be nicer if we didn't have to
        # recreate the array.
        p_cap_bc: pp.ad.Operator = self.cap_press(
            pp.ad.DenseArray(self.bc_dirichlet_saturation_values(g, self.wetting))
        )
        p_cap_values: np.ndarray = p_cap_bc.value(self.equation_system)
        p_cap_values[neumann_ind] = 0
        p_cap_bc = pp.ad.DenseArray(p_cap_values)

        # Sum phase Neumann boundary fluxes to obtain total Neumann boundary flux.
        flux_t_bc_neu = pp.ad.sum_operator_list(
            [
                pp.ad.DenseArray(self.bc_neumann_flux_values(g, phase))
                for phase in self.phases.values()
            ]
        )

        # Buyoyancy flux.
        vector_source_w = pp.ad.DenseArray(self.vector_source(g, self.wetting))
        vector_source_n = pp.ad.DenseArray(self.vector_source(g, self.nonwetting))

        # Spatial discretization operators.
        # NOTE Some notes on the boundary conditions of the potential discretizations
        # and the total flux:
        # - Neumann boundaries: both phase fluxes are prescribed, the total flux is the
        #   sum of both.
        # - Dirichlet boundaries: the total flux is a function of the mobilities and
        #   phase potentials.

        # NOTE Again, we could also use an alternative formulation with total flux and
        # saturation values at Neumann boundaries. In this case:
        # - Neumann boundaries: the total flux is fixed and independent of the
        #   mobilities. We add the Neumann bc values to the total flux by using an
        #   upwind discretization, but crucially ``p_n_bc`` is zero on Neumann faces s.t.
        #   the Mpfa discretization does not add any phase potential and thus flux here.

        mpfa = pp.ad.MpfaAd(self.flux_key, [g])

        # NOTE We use TPFA for discretization of the capillary flux. Don't know the
        # reason, but apparently this was somehow necessary in 2023. Possibly stability
        # reasons.
        # NOTE The flux key is the same as for the total flux discretization, i.e., the
        # boundary condition types coincide. Same considerations as above apply, except
        # that we explicitly have to set ``p_cap_bc`` to zero on Neumann faces as
        # described above.
        tpfa = pp.ad.TpfaAd(self.flux_key, [g])

        # NOTE Neither 'MpfaAd' nor 'TpfaAd' allow to separate Dirichlet and Neumann
        # conditions. As a workaround, we use an upwind discretization initialized
        # with any discretization key (all have the same bc type). On Neumann boundary
        # faces the ``upwind_n.bound_transport_neu`` matrix takes value ``1`` and we can
        # simply multiply with the total flux bc values to obtain the the inflow/outflow
        # at Neumann faces.
        upwind_t = pp.ad.UpwindAd(self.flux_key, [g])

        # Cap pressure and phase mobilities.
        p_cap = self.cap_press(self.wetting.s)
        mobility_w = self.phase_mobility(g, self.wetting)
        mobility_n = self.phase_mobility(g, self.nonwetting)
        mobility_t = self.total_mobility(g)

        # Compute nonwetting & capillary pressure potential including dirichlet bc.
        p_n_potential: pp.ad.Operator = (
            mpfa.flux() @ self.nonwetting.p + mpfa.bound_flux() @ p_n_bc
        )
        p_cap_potential: pp.ad.Operator = (
            tpfa.flux() @ p_cap + tpfa.bound_flux() @ p_cap_bc
        )

        # Gravity terms.
        buoyancy_w_potential: pp.ad.Operator = mpfa.vector_source() @ vector_source_w
        buoyancy_n_potential: pp.ad.Operator = mpfa.vector_source() @ vector_source_n

        # Finally, we can combine all Darcy and buoyancy fluxes multiplied with phase
        # mobilities to the total flux.
        total_flux = (
            mobility_t * p_n_potential
            - mobility_w * p_cap_potential
            - mobility_w * buoyancy_w_potential
            - mobility_n * buoyancy_n_potential
            # Lastly, we add boundary flux at faces with Neumann bc.
            + upwind_t.bound_transport_neu() @ flux_t_bc_neu
        )
        total_flux.set_name("Total volume flux")
        return total_flux


class ConstitutiveLawsTPF(
    RelativePermeability,
    CapillaryPressure,
    DarcyFluxes,
    pp.constitutive_laws.DimensionReduction,
): ...


# endregion


# region PDEs
class EquationsTPF(pp.BalanceEquation):
    """This is a model class for two-phase flow problems.

    This class is intended to provide a standardized setup, with all discretizations
    in place and reasonable parameter and boundary values. The intended use is to
    inherit from this class, and do the necessary modifications and specifications
    for the problem to be fully defined. The minimal adjustment needed is to
    specify the method create_grid(). The class also serves as parent for other
    model classes (CompressibleFlow).

    Public attributes:
    TODO Update this list!
        primary_pressure_var: Name assigned to the pressure variable in the
            highest-dimensional subdomain. Will be used throughout the simulations,
            including in ParaView export. The default variable name is "wetting
            pressure" or "nonwetting pressure", depending on the chosen two-phase flow
            formulation.
        parameter_key: Keyword used to define parameters and discretizations.
        params: Dictionary of parameters used to control the solution procedure.
            Some frequently used entries are file and folder names for export, mesh
            sizes...
        mdg: Mixed-dimensional grid. Should be set by a method
            create_grid, which should be provided by the user.
        convergence_status: Whether the non-linear iteration has converged.
        linear_solver: Specification of linear solver. Only known permissible
            value is 'direct'.
        exporter: Used for writing files for visualization.

    All attributes are given natural values at initialization of the class.

    The implementation assumes use of AD.

    """

    # Managers and exporter:
    equation_system: pp.ad.EquationSystem
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    time_manager: pp.TimeManager
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""

    # Parameter keys:
    params_key: str
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    flux_key: str
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    formulation: str
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally provided by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    # Fluid phases:
    wetting: Phase
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    nonwetting: Phase
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    phases: dict[str, Phase]
    """List of fluid phases, providing phase names and phase constants. Normally
    provided by a mixin of instance :class:`SolutionStrategyTPF`.

    """

    # Variables:
    primary_pressure_var: str
    """Normally provided by a mixin of instance :class:`VariablesTPF` after calling
    :meth:`VariablesTPF.create_variables()`.

    """
    secondary_pressure_var: str
    """Normally provided by a mixin of instance :class:`VariablesTPF` after calling
    :meth:`VariablesTPF.create_variables()`.

    """

    # Constitutive laws:
    rel_perm: Callable[[pp.ad.Operator, Phase], pp.ad.Operator]
    """Phase relative permeability. Normally provided by a mixin of instance
    :class:`RelativePermeability`.

    """
    cap_press: Callable[[pp.ad.Operator], pp.ad.Operator]
    """Capillary pressure. Normally provided by a mixin of instance
    :class:`CapillaryPressure`.

    """
    phase_mobility: Callable[[pp.Grid, Phase], pp.ad.Operator]
    """Phase mobility. Normally provided by a mixin of instance
    :class:`DarcyFluxes`.

    """
    total_mobility: Callable[[pp.Grid], pp.ad.Operator]
    """Total mobility. Normally provided by a mixin of instance
    :class:`DarcyFluxes`.

    """
    total_flux: Callable[[pp.Grid], pp.ad.Operator]
    """Total flux. Normally provided by a mixin of instance
    :class:`DarcyFluxes`.

    """
    volume_integral: Callable
    """Normally provided by a mixin of instance :class:`~porepy.BalanceEquation`."""

    # Parameters for the error function derivative:
    _yscale: float
    _xscale: float
    _offset: float

    # Grid and boundary conditions
    mdg: pp.MixedDimensionalGrid
    """Provided by a mixin of instance :class:`~porepy.models.geometry.ModelGeometry`."""

    bc_dirichlet_pressure_values: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase dependent pressure bc values. Normally provided by a mixin of instance
    :class:`~porepy.BoundaryConditionMixin`.

    """
    bc_dirichlet_saturation_values: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase dependent saturation bc values. Normally provided by a mixin of instance
    :class:`~porepy.BoundaryConditionMixin`.

    """
    bc_neumann_flux_values: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase flux bc values. Normally provided by a mixin of instance
    :class:`BoundaryConditionsTPF`.

    """

    def phase_fluid_source(self, g: pp.Grid, phase: Phase) -> np.ndarray:
        """Volumetric phase source term. Given as volumetric flux. This
        unmodified base function assumes a zero phase source.

        Note: This is the value for a full grid cell, i.e., it does **NOT** get
            integrated over the cell volume in the equation.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        """
        # TODO Does this has to be a vector instead?
        return np.zeros(g.num_cells)

    def total_fluid_source(self, g: pp.Grid) -> np.ndarray:
        """Volumetric total source; sum of the wetting and nonetting source.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        Note: This is the value for a full grid cell, i.e., it does **NOT** get
            integrated over the cell volume in the equation.

        """
        return sum(
            [self.phase_fluid_source(g, phase) for phase in self.phases.values()]
        )

    def vector_source(self, g: pp.Grid, phase: Phase) -> np.ndarray:
        """Volumetric phase vector source. Corresponds to the phase buoyancy flux. This
        unmodified base function assumes a zero vector source.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[..., -1] = pp.GRAVITY_ACCELERATION * self._w_density

        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        # vals[-1] = pp.GRAVITY_ACCELERATION * self.wetting_phase.density()
        # For some reason this needs to be a flat array.
        return vals.ravel()

    # More matrix and phase parameters.
    def _permeability(self, g: pp.Grid) -> np.ndarray:
        """Solid permeability. This unmodified base function assumes homogeneous
        permeability. Value and unit are set by :attr:`self.solid`."""
        return np.full(g.num_cells, self.solid.permeability())

    def _porosity(self, g: pp.Grid) -> np.ndarray:
        """Solid porosity. This unmodified base function assumes homogeneous
        porosity. Value is set by :attr:`self.solid`."""
        return np.full(g.num_cells, self.solid.porosity())

    def set_equations(self, equation_names: Optional[dict[str, str]] = None) -> None:
        """Define equations."""
        try:
            self.equation_system.remove_equation("Flow equation")
            self.equation_system.remove_equation("Transport equation")
        except:
            ValueError("Equations not found.")

        g = self.mdg.subdomains()[0]

        # Spatial discretization operators.
        div = pp.ad.Divergence([g])
        flux_mpfa = pp.ad.MpfaAd(self.flux_key, [g])
        upwind_w = pp.ad.UpwindAd(self.wetting.mobility_key, [g])

        # Time derivatives.
        dt_s = pp.ad.time_derivatives.dt(self.wetting.s, self.ad_time_step)

        # Ad source.
        source_ad_w = pp.ad.DenseArray(self.phase_fluid_source(g, self.wetting))
        source_ad_t = pp.ad.DenseArray(self.total_fluid_source(g))

        # Ad parameters.
        porosity_ad = pp.ad.DenseArray(self._porosity(g))

        # Compute cap pressure and relative permeabilities.
        p_cap = self.cap_press(self.wetting.s)
        # p_cap_bc = pp.ad.DenseArray(self._bc_values_cap_press(g))

        mobility_w: pp.ad.Operator = self.phase_mobility(g, self.wetting)
        mobility_n: pp.ad.Operator = self.phase_mobility(g, self.nonwetting)
        mobility_t: pp.ad.Operator = self.total_mobility(g)

        # Ad equations
        if self.formulation == "fractional_flow":
            # NOTE For ``flux_t``, the total mobility is already included.
            flux_t = self.total_flux(g)
            fractional_flow_w: pp.ad.Operator = mobility_w / mobility_t
            vector_source_w = pp.ad.DenseArray(self.vector_source(g, self.wetting))
            vector_source_n = pp.ad.DenseArray(self.vector_source(g, self.nonwetting))
            bc_neumann_flux_values_w = pp.ad.DenseArray(
                self.bc_neumann_flux_values(g, self.wetting)
            )

            flow_equation = div @ flux_t - source_ad_t
            transport_equation = (
                porosity_ad * (self.volume_integral(dt_s, [g], 1))
                + div
                @ (
                    fractional_flow_w * flux_t
                    + fractional_flow_w
                    * mobility_n
                    * (
                        flux_mpfa.flux() @ p_cap
                        + flux_mpfa.vector_source() @ vector_source_w
                        - flux_mpfa.vector_source() @ vector_source_n
                    )
                    # NOTE ``phase_mobility(self.wetting)`` and hence
                    # ``fractional_flow_w`` are zero on Neumann boundaries. We add the
                    # wetting flux at Neumann boundaries by using an upwind
                    # discretization. For the total flux, this is already included,
                    # hence it does not need to be added to the flow equation
                    + upwind_w.bound_transport_neu() @ bc_neumann_flux_values_w
                )
                - source_ad_w
            )
        flow_equation.set_name("Flow equation")
        transport_equation.set_name("Transport equation")

        # Update the equation list.
        self.equation_system.set_equation(flow_equation, [g], {"cells": 1})
        self.equation_system.set_equation(transport_equation, [g], {"cells": 1})

    def _error_function_deriv(self) -> pp.ad.Operator:
        """Returns the derivative of the error function w.r.t. the saturation.

        This can be used to simulate perturbations in the cap. pressure and rel. perm.
        models.

        Returns:
            Derivative of the error function in terms of :math:`S_w`.
        """
        s = self.equation_system.md_variable(self.primary_saturation_var)
        yscale = pp.ad.Scalar(self._yscale)
        xscale = pp.ad.Scalar(self._xscale)
        offset = pp.ad.Scalar(self._offset)
        exp_func = pp.ad.Function(pp.ad.functions.exp, "exp")
        square_func = pp.ad.Function(partial(ad_pow, exponent=2), "square")
        return yscale * exp_func(pp.ad.Scalar(-1) * xscale * square_func(s - offset))


class VariablesTPF(pp.VariableMixin):

    equation_system: pp.ad.EquationSystem
    """Normally provided by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`

    """
    wetting: Phase
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    nonwetting: Phase
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    phases: dict[str, Phase]
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    formulation: Literal["fractional_flow"]
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    mdg: pp.MixedDimensionalGrid
    """Normally provided by a mixin of instance
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def create_variables(self) -> None:
        """Create primary variables (wetting pressure, nonwetting pressure,
        saturation)."""
        subdomains = self.mdg.subdomains()
        for phase in self.phases.values():
            phase.p = self.equation_system.create_variables(
                f"{phase.name}_p",
                {"cells": 1},
                subdomains,
                tags={"si_units": "Pa"},
            )
            # TODO Check that the pressure units are correct.
            phase.s = self.equation_system.create_variables(
                f"{phase.name}_s",
                {"cells": 1},
                subdomains,
                tags={"si_units": "-"},
            )

        # NOTE The division into primary/secondary variables is internal to this model
        # only and not connected to ``pp.PRIMARY_VARIABLES``.
        if self.formulation == "fractional_flow":
            self.primary_pressure_var = f"{self.nonwetting.name}_p"
            self.primary_saturation_var = f"{self.wetting.name}_s"
            self.secondary_pressure_var = f"{self.wetting.name}_p"
            self.secondary_saturation_var = f"{self.nonwetting.name}_s"

    def normalize_saturation(
        self,
        saturation: pp.ad.Operator,
        phase: Optional[Phase] = None,
        limit: bool = False,
        epsilon: float = 0.0,
    ) -> pp.ad.Operator:
        # TODO Replace typing with ``OperatorType``?
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
            epsilon: Added/substracted from normalized saturation to avoid values of
                ``0``, ``1``.

        Returns:
            s_normalized: Normalized wetting saturation.

        Raises:
            ValueError: If neither ``saturation`` has a ``name`` attribute specifying
            the phase nor ``phase`` is specified.

        """
        if saturation.name.startswith(self.wetting.name) or getattr(
            phase, "name", ""
        ).startswith(WETTING):
            residual_saturation_w = pp.ad.Scalar(
                self.wetting.constants.residual_saturation()
            )
            residual_saturation_n = pp.ad.Scalar(
                self.nonwetting.constants.residual_saturation()
            )
            s_normalized: pp.ad.Operator = (saturation - residual_saturation_w) / (
                pp.ad.Scalar(1) - residual_saturation_n - residual_saturation_w
            )

            # Cut off irregular saturations and add/substract epsilon s.t. capillary
            # pressure does not grow to infinity.
            if limit:
                maximum_func = pp.ad.Function(
                    partial(pp.ad.functions.maximum, var_1=epsilon),
                    "max",
                )
                minimum_func = pp.ad.Function(
                    partial(minimum, var_1=1 - epsilon),
                    "min",
                )
                s_normalized = minimum_func(maximum_func(s_normalized))

        elif saturation.name.startswith(self.nonwetting.name) or getattr(
            phase, "name", ""
        ).startswith(NONWETTING):
            s_normalized = pp.ad.Scalar(1) - self.normalize_saturation(
                pp.ad.Scalar(1) - saturation,
                phase=self.wetting,
                limit=limit,
                epsilon=epsilon,
            )
        else:
            raise ValueError(
                "Either ``saturation`` must have a ``name`` attribute"
                + " specifying the phase or ``phase`` must be specified."
            )
        return s_normalized

    def normalize_saturation_deriv(
        self,
        phase: Phase,
        # limit: bool = False,
        # epsilon: float = 0.0,
    ) -> OperatorType:
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
            limit: If ``True``, the derivative is 0 outside the residual saturations
                plus/minus epsilon. Default is ``False``.
            epislon: Added/substracted from normalized saturation to avoid values of
                ``0``, ``1``. Default is ``0.0``.

        Returns:
            Normalized wetting saturation.

        Raises:
            ValueError: If neither ``saturation`` has a ``name`` attribute specifying the
            phase nor ``phase`` is specified.

        """
        residual_saturation_w: float = self.wetting.constants.residual_saturation()
        residual_saturation_n: float = self.nonwetting.constants.residual_saturation()
        # FIXME This shall return exactly the same type as saturation, but set to 0
        # outside the residual saturation range. How to do this for a general
        # OperatorType?
        # NOTE For now, this function is only needed when
        # ``reconstructions.GlobalPressure.global_pressure`` or
        # ``reconstructions.GlobalPressure.complimentary_pressure`` are called. They
        # take care of limiting the saturation themselves. Thus, we do not need to care
        # about this.
        if phase.name == WETTING:
            return pp.ad.Scalar(1 - residual_saturation_n - residual_saturation_w)
        elif phase.name == NONWETTING:
            return pp.ad.Scalar(residual_saturation_w + residual_saturation_n - 1)


# endregion

# region SOLUTION STRATEGY & BC


class BoundaryConditionsTPF(pp.BoundaryConditionMixin):
    """This class provides boundary conditions for two-phase flow problems.

    - Dirichlet boundary: Phase pressure and saturation values are provided.
    - Neumann boundary: Total flux and phase saturation values are provided.

    This unmodified base class provides homogeneous Dirichlet boundary conditions with
    wetting saturation values set to ``1.0`` and nonwetting saturation values set to
    ``0.0``.

    """

    dim: int
    """Normally provided by a mixin of instance :class:`ModelGeometry`."""

    # TODO Could use a protocol here to account for optional function arguments.
    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]
    """Normally provided by a mixin of instance :class:`ModelGeometry`."""

    wetting: Phase
    """Normally provided by a mixin of instance :class:`SolutionStrategyTPF`."""
    flux_key: str
    """Keyword to define parameters and discretizations for the total flux. Normally
    provided by a mixin of instance :class:`SolutionStrategyTPF`.

    """

    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann)."""
        # Dirichlet conditions for both phases.
        boundary_faces = self.domain_boundary_sides(g).all_bf
        return pp.BoundaryCondition(g, boundary_faces, "dir")

    def bc_dirichlet_pressure_values(self, g: pp.Grid, phase: Phase) -> np.ndarray:
        """Phase dependent pressure bc values.

        Note: On Neumann boundaries, the pressure values MUST be 0 to not accidentally
        prescibe a flux! Both ``Mpfa.bound_flux()`` and ``Tpfa.bound_flux()`` provide
        matrices with ``1`` values at Neumann boundaries.

        """
        # Homogeneous Dirichlet conditions for both phases.
        return np.zeros(g.num_faces)

    # Ignore Pylance complaining. Function will always return a value.
    def bc_dirichlet_saturation_values(self, g: pp.Grid, phase: Phase) -> np.ndarray:  # type: ignore
        """Phase dependent saturation bc values.

        Note: On Neumann boundaries, the saturation values SHOULD be equal to the
            residual saturation to ensure the capillary pressure is zero and no
            capillary flux is prescibed! As a safety measure, we manually null the
            capillary pressure on Neumann faces in :meth:``total_flux``.

        """
        # Homogeneous Dirichlet conditions for both phases.
        if phase.name == WETTING:
            s_bc: np.ndarray = np.full(g.num_faces, 0.5)
        elif phase.name == NONWETTING:
            s_bc: np.ndarray = np.ones(
                g.num_faces
            ) - self.bc_dirichlet_saturation_values(g, self.wetting)
        neu_ind: np.ndarray = self.bc_type(g).is_neu
        s_bc[neu_ind] = self.wetting.constants.residual_saturation()
        return s_bc

    def bc_neumann_flux_values(self, g: pp.Grid, phase: Phase) -> np.ndarray:
        """Phase flux bc values."""
        return np.zeros(g.num_faces)

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


class SolutionStrategyTPF(pp.SolutionStrategy):
    # Ad discretization:
    _flux_w: Callable
    """Provided by `TwoPhaseFlow`."""
    _flux_n: Callable
    """Provided by `TwoPhaseFlow`."""

    set_rel_perm_constants: Callable
    """Normally provided by a mixin of instance :class:`RelativePermeability`."""
    rel_perm: Callable[[pp.ad.Operator, Phase], pp.ad.Operator]
    """Normally provided by a mixin of instance :class:`RelativePermeability`."""

    set_cap_press_constants: Callable
    """Normally provided by a mixin of instance :class:`CapillaryPressure`."""
    cap_press: Callable[[pp.ad.Operator], pp.ad.Operator]
    """Normally provided by a mixin of instance :class:`CapillaryPressure`."""

    phase_potential: Callable[[pp.Grid, Phase], pp.ad.Operator]
    """Phase potential. Normally provided by a mixin of instance
    :class:`DarcyFluxes`.

    """
    total_flux: Callable[[pp.Grid], pp.ad.Operator]
    """Total flux. Normally provided by a mixin of instance
    :class:`DarcyFluxes`.

    """

    # Matrix properties:
    _permeability: Callable[[pp.Grid], np.ndarray]
    """Normally provided by a mixin of instance :class:`EquationsTPF`."""
    _porosity: Callable[[pp.Grid], np.ndarray]
    """Normally provided by a mixin of instance :class:`EquationsTPF`."""

    # Source:
    phase_fluid_source: Callable[[Phase], np.ndarray]
    """Phase fluid sources. Normally provided by a mixin of instance
    :class:`EquationsTPF`.

    """
    total_fluid_source: Callable[[Phase], np.ndarray]
    """Sum of phase fluid sources. Normally provided by a mixin of instance
    :class:`EquationsTPF`.

    """

    # Equations:
    set_equations: Callable[[], None]
    """Normally provided by a mixin of instance :class:`EquationsTPF`."""

    # Variables:
    primary_pressure_var: str
    """Name of primary pressure variable. Normally provided by a mixin of instance
    :class:`VariablesTPF`.

    """
    primary_saturation_var: str
    """Name of primary saturation variable. Normally provided by a mixin of instance
    :class:`VariablesTPF`.

    """
    secondary_pressure_var: str
    """Name of secondary pressure variable. Normally provided by a mixin of instance
    :class:`VariablesTPF`.

    """
    secondary_saturation_var: str
    """Name of secondary saturation variable. Normally provided by a mixin of instance
    :class:`VariablesTPF`.

    """

    # Boundary conditions:
    bc_type: Callable[[pp.Grid], pp.BoundaryCondition]
    """BC type (Neumann or Dirichlet) for flux and mobility discretization. Normally
    provided by a mixin of instance :class:`BoundaryConditionsTPF`.

    """
    bc_dirichlet_pressure_values: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase dependent pressure bc values. Normally provided by a mixin of instance
    :class:`BoundaryConditionsTPF`.

    """
    bc_dirichlet_saturation_values: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase dependent saturation bc values. Normally provided by a mixin of instance
    :class:`BoundaryConditionsTPF`.

    """
    bc_neumann_flux_values: Callable[[pp.Grid, Phase], np.ndarray]
    """Phase flux bc values. Normally provided by a mixin of instance
    :class:`BoundaryConditionsTPF`.

    """

    save_data_time_step: Callable[[], None]
    """Normally provided by a mixin of instance :class:`~porepy.DataSavingMixing`."""

    def __init__(self, params: Optional[dict]) -> None:
        super().__init__(params)
        if params is None:
            params = {}

        self.formulation: Literal["fractional_flow"] = self.params.get(
            "formulation", "fractional_flow"
        )
        """Choose which formulation of two-phase flow shall be run. Note, that his has
        (!!!) to be passed as a parameter. Changing it after initialization may result
        in wrong results. "
        
        Valid values:
            'fractional_flow':

        """

        # Initialize fluid phases.
        self.set_phases()

        # Discretizations and parameter keywords.
        self.flux_key: str = "total_flux"
        """Keyword to define define parameters and discretizations for the total flux.

        The corresponding ``tpfa`` and ``mpfa`` discretizations are used to calculate
        **all** pressure potentials, i.e., wetting, nonwetting, and capillary.

        """
        # NOTE The following is only relevant when using an alternative formulation
        # for Neumann boundaries in terms of total flux and saturation values.
        # self.all_bc_dir_key: str = "all_bc_dir"
        """Keyword to define upwind discretization with fully Dirichlet bc."""

        for phase in self.phases.values():
            setattr(phase, "mobility_key", f"{phase.name}_mobility")
            """Keyword to define parameters and discretizations for the phase mobility.

            As phase potentials can have opposite signs, independent upwind
            discretization are required to evaluate phase mobilities.
            # TODO I do not think this is true, as this would imply negative fractional
            flow. Check the inline comment in ``set_discretization_parameters()`` when
            changing this.

            """

        # Parameters for the error function derivative:
        self._yscale: float = self.params.get("yscale", 1.0)
        self._xscale: float = self.params.get("xscale", 200)
        self._offset: float = self.params.get("offset", 0.5)

        # Solvers:
        self._use_ad: bool = self.params.get("use_ad", True)
        self.linear_solver: Literal["scipy_sparse", "pypardiso", "umfpack"] = (
            self.params.get("linear_solver", "scipy_sparse")
        )

        # Option to limit the saturation change per timestep.
        self._limit_saturation_change: bool = False
        """If this is set to ``True``, the Newton method fails, if the final solution
        differs from the previous timestep by more than ``self._max_saturation_change``
        in any grid cell. The timestep is then shortened and recalculated.
        """
        self._max_saturation_change: float = 0.2

        # Data saving.
        self.results: list[SaveDataTPF] = []
        """List of stored results from the convergence analysis."""

    def is_time_dependent(self) -> bool:
        return True

    def _is_nonlinear_problem(self) -> bool:
        return True

    def set_phases(self) -> None:
        # TODO Add an option to give the phases different names.
        if self.formulation == "fractional_flow":
            self.wetting: Phase = Phase(WETTING)
            self.nonwetting: Phase = Phase(NONWETTING)
            self.phases: dict[str, Phase] = {
                self.wetting.name: self.wetting,
                self.nonwetting.name: self.nonwetting,
            }

    def set_materials(self) -> None:
        # super().set_materials()
        constants: dict[str, pp.MaterialConstants] = self.params.get(  # type: ignore
            "material_constants", {}
        )
        # Use standard models for fluid and solid constants if not provided.
        if "fluid" not in constants:
            constants["fluid"] = pp.FluidConstants()
        if "solid" not in constants:
            constants["solid"] = pp.SolidConstants()

        # TODO Include some functionality that checks if a similar key was included, to
        # account for wrong user inputs.

        # Loop over fluid, solid constants and set units.
        # TODO This is quite ugly, find a nice way to combine this with the phase
        # constants down below. Ideally, we call super().set_materials() and do nothing
        # else or only set the phase constants.
        for name, const in constants.items():
            # Check that the object is of the correct type
            assert isinstance(const, pp.models.material_constants.MaterialConstants)
            # Impose the units passed to initialization of the model.
            const.set_units(self.units)
            # This is where the constants (fluid, solid) are actually set as attributes
            if name in ["fluid", "solid"]:
                setattr(self, name, const)

        for phase in self.phases.values():
            phase_constants: PhaseConstants = constants.get(
                phase.name, PhaseConstants()
            )
            assert isinstance(phase_constants, pp.MaterialConstants)
            # Impose the units passed to initialization of the model.
            phase_constants.set_units(self.units)
            setattr(phase, "constants", phase_constants)

    def prepare_simulation(self) -> None:
        """This setups the model, s.t. a simulation can be run.

        The initial values are exported.

        """
        # Set the model parameters.
        self.set_rel_perm_constants()
        self.set_cap_press_constants()
        self.set_materials()
        self.set_geometry()
        # Exporter initialization must be done after grid creation,
        # but prior to data initialization.
        self.set_solver_statistics()
        self.initialize_data_saving()
        # Create the numerical aparatus.
        self.set_equation_system_manager()
        self.create_variables()
        self.initial_condition()
        self.set_discretization_parameters()
        self.set_equations()
        self.discretize()
        # Save the initial values.
        self.save_data_time_step()

    def set_discretization_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameter fields of the data dictionaries are updated for all
        subdomains and interfaces (of codimension 1).
        """
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            # Boundary conditions and parameters.
            perm = self._permeability(sd)
            # Different treatment for scalar and tensor permeability.
            if isinstance(perm, np.ndarray):
                diffusivity = pp.SecondOrderTensor(self._permeability(sd))
            elif isinstance(perm, dict):
                diffusivity = pp.SecondOrderTensor(**self._permeability(sd))
            # all_bf, *_ = self._domain_boundary_sides(sd)
            # Parameters that are not used for discretization.
            pp.initialize_data(
                sd,
                data,
                self.flux_key,
                {
                    "bc": self.bc_type(sd),
                    "second_order_tensor": diffusivity,
                    "ambient_dimension": sd.dim,
                    # We initialize the Darcy flux to one just s.t.
                    # ``Upwind.discretize()`` can be called. This does not need to be
                    # updated, as only ``Upwind.bound_transport_neu()`` is used, which
                    # does not depend on the Darcy flux.
                    "darcy_flux": np.ones(sd.num_faces),
                },
            )
            # Upwinding is done for both phases separately, hence we create two different
            # data dictionaries.
            # NOTE This is not really necessary and we could just as well use the flux
            # data, as the ``bc_type`` are identical for all discretizations. The
            # delicaties of boundary conditions for fractional flow in PorePy are dealt
            # with in ``DarcyFluxes.mobility()`` and ``DarcyFluxes.total_flux()``.
            for phase in self.phases.values():
                pp.initialize_data(
                    sd,
                    data,
                    phase.mobility_key,
                    {
                        "bc": self.bc_type(sd),
                        # We initialize the Darcy flux to one just s.t.
                        # ``Upwind.discretize()`` can be called.
                        "darcy_flux": np.ones(sd.num_faces),
                    },
                )
            # NOTE The following is only relevant when using an alternative formulation
            # for Neumann boundaries in terms of total flux and saturation values.
            # To correctly upwind phase mobilities at Neumann boundaries, we need
            # another upwind discretization with full Dirichlet boundaries. Check
            # ``DarcyFluxes.mobility()`` for more details.
            # bc_dir = pp.BoundaryCondition(sd, sd.get_all_boundary_faces(), "dir")
            # pp.initialize_data(
            #     sd,
            #     data,
            #     self.all_bc_dir_key,
            #     {
            #         "bc": bc_dir,
            #     },
            # )

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        g = self.mdg.subdomains()[0]
        self.equation_system.set_variable_values(
            np.full(g.num_cells * 2, 0.0),
            [self.wetting.p, self.nonwetting.p],
            time_step_index=0,
            iterate_index=0,
        )
        self.equation_system.set_variable_values(
            np.full(g.num_cells * 2, 0.5),
            [self.wetting.s, self.nonwetting.s],
            time_step_index=0,
            iterate_index=0,
        )

    def discretize(self) -> None:
        """Discretize all terms."""
        # t_0 = time.time()
        self.equation_system.discretize()
        if self.formulation == "fractional_flow":
            # Phase potential. This needs (?) to be discretized at each nonlinear iteration
            # s.t. the Darcy flux can be computed for both phases s.t. the upwind
            # operators for phase mobilities works correctly.
            for phase in self.phases.values():
                phase_potential = self.phase_potential(self.mdg.subdomains()[0], phase)
                phase_potential.discretize(self.mdg)
        # logger.debug(f"Discretized in {time.time() - t_0:.2e} seconds")

    def rediscretize(self) -> None:
        # TODO Discretize only nonlinear discretizations. Make sure to call this in
        # ``before_nonlinear_iteration`` instead of ``discretize``.
        ...

    def set_nonlinear_discretizations(self) -> None:
        # TODO Collect all nonlinear discretizations in one place. This way, linear
        # discretizations do not get called at each nonlinear iteration.
        ...

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system.

        The linear system is defined by the current state of the model.

        Attributes:
            linear_system is assigned.

        """
        t_0 = time.time()
        if self._use_ad:
            self.linear_system = self.equation_system.assemble(
                variables=[self.primary_pressure_var, self.primary_saturation_var]
            )
        logger.debug(f"Assembled linear system in {time.time() - t_0:.2e} seconds")

    # region NONLINEAR LOOP
    def before_nonlinear_loop(self) -> None:
        """Set the starting estimate to the solution from the previous timestep."""
        # Update time step size and empty statistics.
        self.ad_time_step.set_value(self.time_manager.dt)
        self.nonlinear_solver_statistics.reset()

        assembled_variables = self.equation_system.get_variable_values(
            time_step_index=0
        )
        self.equation_system.set_variable_values(
            assembled_variables, iterate_index=0, additive=False
        )
        # FIXME
        # if self._limit_saturation_change:
        #     self._prev_saturation: np.ndarray = (
        #         self.equation_system.get_variable_values(
        #             variables=[self.primary_saturation_var], time_step_index=0
        #         )
        #     )

    def before_nonlinear_iteration(self) -> None:
        """Compute Darcy flux based on previous pressure solution to determine upstream
        direction.

        # TODO Do we need to do the rediscretization of the upwinding at every
        iteration? Or only at the first one?

        # TODO Is it smarter to calculate and distribute the secondary variables
        **after** the nonlinear iterations?

        To evaluate the phase-mobilities separately, both the wetting, as well as the
        nonwetting flux need to be computed.

        """
        # Compute the Darcy flux for upwinding.
        # -> Needs to happen at each nonlinear iteration, because we are starting with a
        # bad guess (previous timestep) and improve towards the solution. We want to use
        # the better guess of the Darcy flux for discretization.

        # NOTE This should only be done with care at the first iteration of the initial
        # time step, as the upwind direction will not align with the initital guess for
        # upwinding then.
        # -> Might be unwanted or wanted behavior.
        if (
            self.time_manager.time_index >= 2
            or self.nonlinear_solver_statistics.num_iteration >= 1
        ):
            logger.info(
                "Recalculate Darcy flux for upwind discretization."
                + f" Iteration {self.nonlinear_solver_statistics.num_iteration}"
            )
            for sd, data in self.mdg.subdomains(return_data=True):
                # Update Darcy fluxes for both phases.
                for phase in self.phases.values():
                    vals = self.phase_potential(sd, phase).value(self.equation_system)
                    data[pp.PARAMETERS][phase.mobility_key].update({"darcy_flux": vals})

                # NOTE Only needed for the alternative formulation of Neumann bc.
                # data[pp.PARAMETERS][self.all_bc_dir_key].update({"darcy_flux": vals})

        # TODO Do I need to reset discretization parameters as is done by
        # ``pp.solution_strategy.SolutionStrategy``?
        self.discretize()

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the new trial state, visualize
        the current approximation etc.

        Parameters:
            nonlinear_increment: The new solution, as computed by the non-linear solver.

        """
        # Update primary variables.
        self.equation_system.shift_iterate_values(max_index=len(self.iterate_indices))
        self.equation_system.set_variable_values(
            values=nonlinear_increment,
            variables=[self.primary_pressure_var, self.primary_saturation_var],
            additive=True,
            iterate_index=0,
        )

        # Evaluate and update secondary variables.
        # TODO Shift this to a separate method.
        if self.formulation == "fractional_flow":
            secondary_pressure = self.equation_system.get_variables(
                variables=[self.primary_pressure_var]
            )[0] - self.cap_press(
                self.equation_system.get_variables(
                    variables=[self.primary_saturation_var]
                )[0]
            )
        secondary_saturation = (
            pp.ad.Scalar(1)
            - self.equation_system.get_variables(
                variables=[self.primary_saturation_var]
            )[0]
        )
        secondary_pressure_sol = secondary_pressure.value(self.equation_system)
        secondary_saturation_sol = secondary_saturation.value(self.equation_system)
        #  As the values were computed with the additive value of the primary pressure
        #  and saturation variable, we set ``additive=False``.
        self.equation_system.set_variable_values(
            np.concatenate(
                [np.array(secondary_pressure_sol), np.array(secondary_saturation_sol)]
            ),
            variables=[self.secondary_pressure_var, self.secondary_saturation_var],
            iterate_index=0,
            additive=False,
        )

        self.nonlinear_solver_statistics.num_iteration += 1

    # Ignore mypy complaining about uncompatible signature.
    # Ignore mypy complaining about uncompatible signature for ``save_data_time_step``.
    def after_nonlinear_convergence(self) -> None:  # type: ignore
        """Export and move to the next time step.

        When ``self._limit_saturation_change == True``, check if the wetting saturation
        has changed too much

        Parameters:
            solution: _description_
            errors: _description_

        """
        # TODO At the moment this sets **all** iterate variables to the time step
        # solution (also secondary variables). This should be changed.
        # Secondary variables are not updated yet, so this will make things confusing.
        super().after_nonlinear_convergence()
        # If the saturation changes to much, decrease the time step and calculate again.
        if self._limit_saturation_change:
            new_saturation: np.ndarray = self.equation_system.get_variable_values(
                variables=[self.primary_saturation_var], iterate_index=0
            )
            if (
                np.max(np.abs(new_saturation - self._prev_saturation))
                > self._max_saturation_change
            ):
                # This is set to false again in ``before_nonlinear_loop``.
                # NOTE This is not a very nice solution, however, as of now I didn't
                # find a way to pass ``recompute_solution`` to
                # ``time_manager.compute_time_step()`` in ``run_time_dependent_model``
                # without the code getting really messy.
                self.time_manager._recomp_sol = True
                self.convergence_status = False
                logger.debug(
                    "Saturation grew to quickly. Trying again with a smaller time step."
                )
                # TODO Actually try again with a smaller time step.
                return None

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
        return converged, diverged

    # endregion

    def after_simulation(self):
        pass


# endregion


# Ignore mypy complaining about uncompatible signature between
# ``SolutionStrategyTPF`` and ``pp.DataSavingMixin``.
class TwoPhaseFlow(  # type: ignore
    EquationsTPF,
    VariablesTPF,
    ConstitutiveLawsTPF,
    BoundaryConditionsTPF,
    SolutionStrategyTPF,
    pp.ModelGeometry,
    pp.DataSavingMixin,
): ...
