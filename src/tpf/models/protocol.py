from typing import TYPE_CHECKING, Any, Literal, Protocol

from numpy.typing import ArrayLike

from tpf.numerics.quadrature import TriangleQuadrature
from tpf.utils.constants_and_typing import FLUX_NAME
from tpf.viz.solver_statistics import (
    SolverStatisticsANewton,
    SolverStatisticsEst,
    SolverStatisticsHC,
    SolverStatisticsTPF,
)

if not TYPE_CHECKING:
    # This branch is accessed in python runtime.
    # NOTE See Warning in module docstring before attempting anything here.
    class TPFProtocol:
        """This is an empty placeholder of the protocol, used for type hints."""

    class HCProtocol:
        """This is an empty placeholder of the protocol, used for type hints."""

    class ReconstructionProtocol:
        """This is an empty placeholder of the protocol, used for type hints."""

    class EstimatesProtocol:
        """This is an empty placeholder of the protocol, used for type hints."""

    class AdaptiveNewtonProtocol:
        """This is an empty placeholder of the protocol, used for type hints."""

    class SPE11Protocol:
        """This is an empty placeholder of the protocol, used for type hints."""

    class DataSavingMixinExtendedProtocol:
        """This is an empty placeholder of the protocol, used for type hints."""

else:
    # This branch is accessed by mypy and linters.
    import numpy as np
    import porepy as pp
    from porepy.models.protocol import DataSavingProtocol, PorePyModel
    from porepy.viz.exporter import DataInput

    from tpf.models.constitutive_laws_tpf import CapPressConstants, RelPermConstants
    from tpf.models.phase import FluidPhase
    from tpf.utils.constants_and_typing import PRESSURE_KEY

    class TPFProtocol(PorePyModel):
        # Variables & equations
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
        """Name of secondary saturation variable. Normally provided by a mixin of
        instance :class:`VariablesTPF`.

        """
        flow_equation: str
        """Name of the flow equation. Normally provided by a mixin of instance
        :class:`EquationsTPF`.

        """
        transport_equation: str
        """Name of the transport equation. Normally provided by a mixin of instance
        :class:`EquationsTPF`.

        """
        secondary_pressure_eq: str
        """Name of the secondary pressure equation. Normally provided by a mixin of
        instance :class:`EquationsTPF`.

        """
        secondary_saturation_eq: str
        """Name of the secondary saturation equation. Normally provided by a mixin of
        instance :class:`EquationsTPF`.

        """

        # Phases
        wetting: FluidPhase
        """Wetting phase class, providing phase name, constants, variables, and bc.
        Normally set by a mixin of instance :class:`SolutionStrategyTPF`.

        """
        nonwetting: FluidPhase
        """Nonwetting phase class, providing phase name, constants, variables, and bc.
        Normally set by a mixin of instance :class:`SolutionStrategyTPF`.#

        """
        phases: dict[str, FluidPhase]
        """List of fluid phases, providing phase names, constants, variables, and bc.
        Normally set by a mixin of instance :class:`SolutionStrategyTPF`.

        """
        _cap_press_constants: CapPressConstants
        """Capillary pressure constants. Normally set by a mixin of instance
        :class:`CapillaryPressure`.

        """

        formulation: Literal["fractional_flow"]
        """Normally set by a mixin of instance :class:`SolutionStrategyTPF`."""
        flow_equation_weight: float
        """Weighting factor for the flow equation in the residual and Jacobian."""
        transport_equation_weight: float
        """Weighting factor for the transport equation in the residual and Jacobian."""

        # Discretization keywords
        flux_key: str
        """Keyword to define parameters and discretizations for the total flux. Normally
        provided by a mixin of instance :class:`SolutionStrategyTPF`.

        """
        cap_potential_key: str
        """Keyword to define parameters and discretizations for the capillary pressure
        potential flux. Normally provided by a mixin of instance
        :class:`SolutionStrategyTPF`.

        """
        params_key: str
        """Normally set by a mixin of instance :class:`SolutionStrategyTPF`."""

        # Nonlinear solver
        nonlinear_solver_statistics: SolverStatisticsTPF

        _nl_appleyard_chopping: bool
        """Whether to use the Appleyard chopping strategy for the nonlinear solver."""
        _nl_enforce_physical_saturation: bool
        """Whether to enforce physical saturations in the nonlinear solver."""

        # Grid
        g: pp.Grid
        """Single subdomain."""
        g_data: dict
        """Single subdomain data dictionarys."""

        @property
        def uses_hc(self) -> bool:
            """"""
            ...

        def normalize_saturation(
            self,
            saturation: pp.ad.Operator,
            phase: FluidPhase | None = None,
        ) -> pp.ad.Operator:
            """Normallly provided by a mixin of instance :class:`VariablesTPF`."""
            ...

        def normalize_saturation_np(
            self,
            saturation: np.ndarray,
            phase: FluidPhase,
        ) -> np.ndarray:
            """Normallly provided by a mixin of instance :class:`VariablesTPF`."""
            ...

        def normalize_saturation_deriv(
            self,
            phase: FluidPhase,
        ) -> pp.ad.Operator:
            """Normallly provided by a mixin of instance :class:`VariablesTPF`."""
            ...

        def normalize_saturation_deriv_np(
            self,
            phase: FluidPhase,
        ) -> float:
            """Normallly provided by a mixin of instance :class:`VariablesTPF`."""
            ...

        # Constitutive laws
        def rel_perm(
            self,
            saturation_w: pp.ad.Operator,
            phase: FluidPhase,
            rel_perm_constants: RelPermConstants | None = None,
        ) -> pp.ad.Operator:
            """Phase relative permeability. Normally provided by a mixin of instance
            :class:`RelativePermeability`.

            """
            ...

        def rel_perm_np(
            self,
            saturation_w: np.ndarray,
            phase: FluidPhase,
            rel_perm_constants: RelPermConstants | None = None,
        ) -> np.ndarray:
            """Phase relative permeability for saturations of type
            :class:`~numpy.ndarray`. Normally provided by a mixin of instance
            :class:`CapillaryPressure`.

            """
            ...

        def set_rel_perm_constants(self) -> None:
            """Normally provided by a mixin of instance
            :class:`RelativePermeability`.

            """
            ...

        def entry_pressure_np(
            self,
            g: pp.Grid,
            cap_press_constants: CapPressConstants | None = None,
            **kwargs,
        ) -> float | np.ndarray:
            """Entry pressure for capillary pressure. Normally provided by a mixin of
            instance :class:`CapillaryPressure`.

            """
            ...

        def cap_press(
            self,
            saturation_w: pp.ad.Operator,
            cap_press_constants: CapPressConstants | None = None,
            **kwargs: Any,
        ) -> pp.ad.Operator:
            """Capillary pressure. Normally provided by a mixin of instance
            :class:`CapillaryPressure`.

            """
            ...

        def cap_press_np(
            self,
            saturation_w: np.ndarray,
            cap_press_constants: CapPressConstants | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            """Capillary pressure for saturations of type
            :class:`~numpy.ndarray`. Normally provided by a mixin of instance
            :class:`CapillaryPressure`.

            """
            ...

        def cap_press_deriv(
            self,
            saturation_w: pp.ad.Operator,
            cap_press_constants: CapPressConstants | None = None,
            **kwargs: Any,
        ) -> pp.ad.Operator:
            """Capillary pressure derivative. Normally provided by a mixin of instance
            :class:`CapillaryPressure`.

            """
            ...

        def cap_press_deriv_np(
            self,
            saturation_w: np.ndarray,
            cap_press_constants: CapPressConstants | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            """Capillary pressure derivative for saturations of type
            :class:`~numpy.ndarray`. Normally provided by a mixin of instance
            :class:`CapillaryPressure`.

            """
            ...

        def set_cap_press_constants(self) -> None:
            """Normally provided by a mixin of instance :class:`CapillaryPressure`."""
            ...

        # DarcyFluxes attributes and methods:
        def phase_mobility(
            self,
            g: pp.Grid,
            phase: FluidPhase,
        ) -> pp.ad.Operator:
            """Formula for the phase mobility for the given phase."""
            ...

        def total_mobility(self, g: pp.Grid) -> pp.ad.Operator:
            """Formula for the total mobility."""
            ...

        def phase_potential(self, g: pp.Grid, phase: FluidPhase) -> pp.ad.Operator:
            """Formula fo the phase potential."""
            ...

        def total_flux(self, g: pp.Grid) -> pp.ad.Operator:
            """Total volume flux."""
            ...

        def wetting_flux(self, g: pp.Grid) -> pp.ad.Operator:
            """Calculate the wetting flux from the total flux and fractional flow."""
            ...

        # EquationsTPF
        def phase_fluid_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
            """Volumetric phase source term."""
            ...

        def total_fluid_source(self, g: pp.Grid) -> np.ndarray:
            """Volumetric total source."""
            ...

        def vector_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
            """Volumetric phase vector source."""
            ...

        def permeability(self, g: pp.Grid) -> np.ndarray | dict[str, np.ndarray]:
            """Solid permeability."""
            ...

        def porosity(self, g: pp.Grid) -> np.ndarray:
            """Solid porosity."""
            ...

        # BoundaryConditionsTPF attributes and methods:
        def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
            """BC type (Neumann or Dirichlet) for flux and mobility discretization. Normally
            provided by a mixin of instance :class:`BoundaryConditionsTPF`.

            """
            ...

        def bc_dirichlet_pressure_values(
            self, g: pp.Grid, phase: FluidPhase
        ) -> np.ndarray:
            """Phase dependent pressure bc values on Dirichlet boundaries."""
            ...

        def bc_dirichlet_saturation_values(
            self, g: pp.Grid, phase: FluidPhase
        ) -> np.ndarray:  # type: ignore
            """Phase dependent saturation bc values on Dirichlet boundaries."""
            ...

        def bc_neumann_flux_values(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
            """Phase flux bc values on Neumann boundaries."""
            ...

        # SolutionStrategyTPF attributes and methods:
        def bound_saturation(
            self,
            s_w: np.ndarray,
            is_increment: bool = False,
            time_step_index: int | None = None,
            iterate_index: int | None = None,
        ) -> np.ndarray:
            """Bound saturation values to physical limits."""
            ...

        def assemble_residual(self) -> np.ndarray:
            """Assemble the residual."""
            ...

    class HCProtocol(Protocol):
        _rel_perm_constants_1: RelPermConstants
        """Relative permeability constants for the first phase."""
        _rel_perm_constants_2: RelPermConstants
        """Relative permeability constants for the second phase."""

        hc_toggle_fl: float
        """Toggle homotopy continuation on and off.

        The default value is one to use homotopy continuation numerical fluxes.
        To toggle the target numerical fluxes, it is set value to zero. Any
        method that changes this value, is expected to change it back to one after 
        the call.

        """
        hc_toggle_ad: pp.ad.Scalar
        """Toggle homotopy continuation on and off.

        Needs to be updated whenever :attr:`hc_rel_perm_toggle_fl` is changed.

        """

        nonlinear_solver_statistics: SolverStatisticsHC

        hc_is_converged: bool
        """Flag to indicate if the homotopy continuation has converged."""
        hc_is_diverged: bool
        """Flag to indicate if the homotopy continuation has diverged."""
        hc_constant_decay: bool
        hc_init_decay: float
        hc_decay: float
        hc_decay_min_max: tuple[float, float]
        nl_iter_optimal_range: tuple[int, int]
        nl_iter_relax_factors: tuple[float, float]
        hc_decay_recomp_max: int
        hc_decay_recomp_counter: int

        def rel_perm(
            self, saturation_w: pp.ad.Operator, phase: FluidPhase
        ) -> pp.ad.Operator:
            """Phase relative permeability."""
            ...

        @property
        def hc_indices(self) -> list[int]:
            """"""
            ...

        def global_discr_est(self) -> float:
            """Estimate for global discretization error."""
            ...

        def global_spatial_est(self) -> float:
            """Estimate for global spatial discretization error."""
            ...

        def global_temp_est(self) -> float:
            """Estimate for global temporal discretization error."""
            ...

        def global_hc_est(self) -> float:
            """Estimate for global homotopy continuation error."""
            ...

        def global_lin_est(self) -> float:
            """Estimate for global linearization error."""
            ...

        def before_hc_loop(self) -> None:
            """Methods to run for homotopy continuation loop."""
            ...

        def before_hc_iteration(self) -> None:
            """Methods to run for homotopy continuation iteration."""
            ...

        def after_hc_iteration(self) -> None:
            """Methods to run after homotopy continuation iteration."""
            ...

        def hc_check_convergence(
            self,
            nl_is_converged: bool,
            nl_is_diverged: bool,
            hc_params: dict[str, Any],
        ) -> tuple[bool, bool]:
            """Check if homotopy continuation has converged."""
            ...

        def after_hc_convergence(self) -> None:
            """Methods to run after homotopy continuation convergence."""
            ...

        def after_hc_failure(self) -> None:
            """Methods to run after homotopy continuation failure."""
            ...

    class ReconstructionProtocol(Protocol):
        postproc_ad_ops: dict[str, pp.ad.Operator]
        """Operators to be evaluated during post-processing. Normally provided by a
        mixin of instance :class:`SolutionStrategyReconstruction`.

        """

        quadpy_elements: np.ndarray
        """Grid cells in quadpy format."""

        # PressureMixin attributes and methods:
        def setup_glob_compl_pressure(self) -> None:
            """Setup global pressure and global pressure interpolants."""
            ...

        def calc_pressure_interpolants(self) -> None:
            """Calculate interpolants values for the global and complementary pressure."""
            ...

        def eval_glob_compl_pressure(
            self,
            s_w: np.ndarray,
            pressure_key: PRESSURE_KEY,
            entry_pressure: ArrayLike = 1.0,
            p_n: np.ndarray | None = None,
        ) -> np.ndarray:
            """Evaluate the global or complementary pressure field for the given pressure
            and saturation values.

            """
            ...

        def eval_saturation(self, q: np.ndarray) -> np.ndarray:
            """Calculate wetting saturation from global pressure."""
            ...

        def eval_glob_compl_pressure_on_domain(
            self,
            time_step_index: int | None = None,
        ) -> None:
            """Evaluate the global and complementary pressure fields on the full domain and
            store it in the data dictionary.

            """
            ...

        def global_pressure_integral_part(
            self, s_0: np.ndarray, s_1: np.ndarray
        ) -> np.ndarray:
            r"""Compute the integral in the global pressure formula for given integral
            boundaries.

            """
            ...

        def complementary_pressure_integral_part(
            self, s_0: np.ndarray, s_1: np.ndarray
        ) -> np.ndarray:
            r"""Compute complementary pressure from the rel. perm. and capillary pressure
            functions.

            """
            ...

        def set_boundary_pressures(self) -> None:
            """Set boundary pressures for the global and complementary pressure fields."""
            ...

        # PressureReconstructionMixin attributes and methods:
        def setup_pressure_reconstruction(self) -> None:
            """Setup pressure reconstruction."""
            ...

        def postprocess_pressure_vohralik(
            self,
            pressure_key: PRESSURE_KEY,
            flux_specifier: str = "",
            prepare_simulation: bool = False,
        ) -> None:
            """Postprocess pressure into elementwise P2 polynomials."""
            ...

        def reconstruct_pressure_vohralik(
            self,
            pressure_key: PRESSURE_KEY,
            prepare_simulation: bool = False,
        ) -> None:
            r"""Reconstruct pressures in :math:`H^1_0(\Omega)` by applying the Oswald
            interpolator."""
            ...

        @staticmethod
        def _evaluate_poly_at_points(
            coeffs: np.ndarray, x: np.ndarray, y: np.ndarray
        ) -> np.ndarray:
            """Evaluate P2 polynomial defined by `coeffs` at given (x, y)
            coordinates."""
            ...

        # EquilibratedFluxMixin attributes and methods:

        def setup_flux_equilibration(self) -> None:
            """Setup flux equilibration."""
            ...

        def equilibrate_flux_during_Newton(
            self,
            flux_name: FLUX_NAME,
            nonlinear_increment: np.ndarray | None = None,
        ) -> None:
            """Equilibrate an approximate flux solution at a given Newton iteration."""
            ...

        def extend_fv_fluxes(
            self,
            flux_name: FLUX_NAME,
            flux_specifier: str = "",
            prepare_simulation: bool = False,
        ) -> None:
            """Extend flux (eqilibrated or non-equilibrated) using RT0 basis functions."""
            ...

        def equilibrated_flux_mismatch(self) -> dict[str, float]:
            r"""Calculate mismatch of the equilibrated flux from being in :math:`H(div)`
            and being mass conservative.

            """
            ...

        # SolutionStrategyReconstructionsMixin attributes and methods:
        def eval_postproc_qtys(self, time_step_index: int | None = None) -> None:
            """Evaluate additional pressure and flux variables and save in data dictionary
            after each iteration.

            """
            ...

        def postprocess_solution(
            self, nonlinear_increment: np.ndarray, prepare_simulation: bool = False
        ) -> None:
            """Equilibrate fluxes and reconstruct pressures."""
            ...

        def local_energy_norm(self) -> None:
            r"""Calculate the local in space and time energy norm of the numerical
            solution."""
            ...

        def global_energy_norm(self) -> float:
            r"""Calculate the global in space and local in time energy norm of the
            numerical solution."""
            ...

    class EstimatesProtocol(Protocol):
        #  EstimatesMixin attributes and methods:
        quadrature_est_degree: int
        quadrature_est: TriangleQuadrature
        nonlinear_solver_statistics: SolverStatisticsEst

        def setup_estimates(self) -> None:
            """Setup error estimates."""
            ...

        def poincare_constant(self) -> float:
            """Compute the Poincare constant."""
            ...

        def local_residual_est(self, flux_name: FLUX_NAME) -> None:
            """Calculate and store the local residual estimate for each element."""
            ...

        def local_flux_est(self, flux_name: FLUX_NAME) -> None:
            """Calculate and store the local flux estimate for each element."""
            ...

        def local_darcy_est(
            self, flux_name: FLUX_NAME, flux_specifier: str = ""
        ) -> None:
            """Calculate and store the local Darcy estimate for each element."""
            ...

        @staticmethod
        def _evaluate_flux_at_points(
            coeffs: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
        ) -> np.ndarray:
            r"""Helper function to evaluate the RT0 flux defined by the given coefficients
            at the specified points.

            """
            ...

        def global_res_and_flux_est(self) -> float:
            """Sum local flux and residual error estimators, integrate in time, and sum
            total and wetting estimators.

            """
            ...

        def global_darcy_est(self) -> tuple[float, float]:
            """Sum local Darcy error estimators and integrate in time."""
            ...

        def global_sp_est(self) -> tuple[float, float]:
            """Sum local saturation-pressure error estimators and integrate in time."""
            ...

        def global_darcy_and_sp_est(self) -> float:
            """Sum global Darcy and saturation-pressure error estimators."""
            ...

        # SolutionStrategyEstimatesMixin attributes and methods:
        def set_initial_estimators(self) -> None:
            """Initialize time step values for error estimators."""
            ...

    class AdaptiveNewtonProtocol(Protocol):
        nonlinear_solver_statistics: SolverStatisticsANewton

        def global_spatial_est(self) -> float:
            """Estimate for global spatial discretization error."""
            ...

        def global_temp_est(self) -> float:
            """Estimate for global temporal discretization error."""
            ...

        def global_discr_est(self) -> float:
            """Estimate for global discretization error."""
            ...

        def global_lin_est(self) -> float:
            """Estimate for global linearization error."""
            ...

    class SPE11Protocol(Protocol):
        """Protocol for the SPE11 derived model."""

        # SPE11 attributes and methods:
        spe11_case: str
        """SPE11 case name. Normally provided by a mixin of instance
        :class:`SolutionStrategySPE11`.

        """
        spe11_params: dict[str, Any]
        """Parameters for the specificSPE11 case."""

    class DataSavingMixinExtendedProtocol(DataSavingProtocol):
        def data_to_export_iteration(
            self,
        ) -> list[DataInput]:
            """Export data after nonlinear iteration."""
            ...

        def _data_to_export(
            self,
            time_step_index: int | None = None,
            iterate_index: int | None = None,
        ) -> list[DataInput]:
            """Private function to allow easy coexistence of ``data_to_export`` and
            ``data_to_export_iteration``

            """
            ...
