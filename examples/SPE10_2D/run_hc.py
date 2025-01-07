r"""We loosely follow the setup of Wang and Tchelepi (2013) to test the homotopy
continuation. The considered model is similar to the heterogeneous 3D models in the
article (section 4.6.4), but on a 2D domain for now.

X. Wang and H. A. Tchelepi, “Trust-region based solver for nonlinear transport in
   heterogeneous porous media,” Journal of Computational Physics, vol. 253, pp.
   114–137, Nov. 2013, doi: 10.1016/j.jcp.2013.06.041.

Model description:
- 600x1100 ft domain (we just take a quarter of the original SPE 10th CSP domain)
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 10 days
- Solid properties:
    - Porosity: Uppermost layer of the SPE 10th CSP (model 2).
    - Permeability: Uppermost layer of the SPE 10th CSP (model 2).
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.2.
    - Oil: PVT table from the SPE 10th CSP (model 2). We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Pressure: 6000 psi
    - Saturation: residual water saturation (0.2).
- Rel. perm. models:
    - linear
    - Corey with power 2.
- Capillary pressure model:
    - Brooks-Corey

"""

import itertools
import json
import logging
import pathlib
import random
import shutil
import warnings
from typing import Any, Callable, Literal, Optional

import numpy as np
import porepy as pp
from matplotlib import pyplot as plt
from numba import config
from tpf.models.error_estimate import (
    DataSavingMixinEst,
    ErrorEstimateMixin,
    SolutionStrategyEst,
)
from tpf.models.flow_and_transport import (
    BoundaryConditionsTPF,
    ConstitutiveLawsTPF,
    DataSavingTPF,
    EquationsTPF,
    SolutionStrategyTPF,
    VariablesTPF,
)
from tpf.models.homotopy_continuation import (
    DataSavingMixinHC,
    EstimatesHCMixin,
    HCSolver,
    RelativePermeabilityHC,
    SolutionStrategyEstHCMixin,
    SolutionStrategyHCMixin,
    SolverStatisticsHC,
)
from tpf.models.phase import FluidPhase
from tpf.models.reconstruction import (
    DataSavingMixinRec,
    EquilibratedFluxMixin,
    GlobalPressureMixin,
    PressureReconstructionMixin,
)
from tpf.spe10.mixin import SPE10Mixin
from tpf.utils.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    FEET,
    GLOBAL_PRESSURE,
    PHASENAME,
    PSI,
)

# Disable numba JIT for debugging.
config.DISABLE_JIT = True

# Catch numpy warnings.
np.seterr(all="raise")
warnings.filterwarnings("default")

# Fix seed for reproducability.
random.seed(0)
np.random.seed(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# region MODEL
class ModifiedGeometry(pp.ModelGeometry):

    def set_domain(self) -> None:
        r"""Single layer of the SPE10 problem 2 model. Extend of the full domain is
        :math:`\qty{1200 x 2200 x 170}{\feet}`. A single layer is
        :math:`\qty{1200 x 2200}{\feet}`.

        """
        bounding_box: dict[str, pp.number] = {
            "xmin": 0,
            "xmax": 600 * FEET,
            "ymin": 0,
            "ymax": 1100 * FEET,
        }
        self._domain = pp.Domain(bounding_box)


class ModifiedEquations(EquationsTPF):

    def phase_fluid_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
        r"""Volumetric phase source term. Given as volumetric flux.

        Five-spot setup. Water (wetting) injection in the center, oil (nonwetting)
        production in the four corners.

        NOTE: This is the average value per grid cell, i.e., it gets scaled with the
        cell volume in the equation.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        """
        if phase.name == self.wetting.name:
            array: np.ndarray = super().phase_fluid_source(g, phase)
            if np.isclose(
                self.params["meshing_arguments"]["cell_size"], 600 * FEET / 30
            ):
                array[1358] = 87.5 / pp.DAY  # 87.5 m^3/day in [m^3/s]
            return array
        elif phase.name == self.nonwetting.name:
            return super().phase_fluid_source(g, phase)

    def corner_cell_ids(self, g: pp.Grid) -> list[int]:
        """Get the corner cell ids."""
        if np.isclose(self.params["meshing_arguments"]["cell_size"], 600 * FEET / 30):
            return [3616, 3618, 3622, 3624]

    def corner_masks(self, g: pp.Grid) -> tuple[pp.ad.DenseArray, pp.ad.DenseArray]:
        """Create masks that hide and single out the corner cells."""
        corner_cell_ids: list[int] = self.corner_cell_ids(g)
        corner_mask_ndarray: np.ndarray = np.zeros((g.num_cells))
        corner_mask_ndarray[corner_cell_ids] = 1
        corner_mask = pp.ad.DenseArray(corner_mask_ndarray)
        corner_mask_inverse = pp.ad.DenseArray(1 - corner_mask_ndarray)
        corner_mask.set_name("Corner mask")
        corner_mask_inverse.set_name("Corner mask inverse")
        return corner_mask, corner_mask_inverse

    def set_equations(self, equation_names: Optional[dict[str, str]] = None) -> None:
        """Modify the equations s.t. the corner cells get prescibed a pressure and
        saturation explicitly. This simulates production wells.

        """
        super().set_equations(equation_names)

        # Prescibre the corner cell values directly. This resembles Dirichlet
        # conditions.
        g: pp.Grid = self.mdg.subdomains()[0]
        flow_equation: pp.ad.Operator = self.equation_system.equations["Flow equation"]
        transport_equation: pp.ad.Operator = self.equation_system.equations[
            "Transport equation"
        ]

        corner_mask, corner_mask_inverse = self.corner_masks(g)

        # Subdivide new equations in 3 parts for easier debugging.
        old_flow_equation_masked: pp.ad.Operator = corner_mask_inverse * flow_equation
        old_transport_equation_masked: pp.ad.Operator = (
            corner_mask_inverse * transport_equation
        )
        old_flow_equation_masked.set_name("Old flow equation masked")
        old_transport_equation_masked.set_name("Old transport equation masked")

        explicit_pressure: pp.ad.Operator = corner_mask * (
            self.nonwetting.p - pp.ad.Scalar(4000 * PSI)
        )
        explicit_saturation: pp.ad.Operator = corner_mask * (
            self.wetting.s - pp.ad.Scalar(1 - self.wetting.residual_saturation)
        )
        explicit_pressure.set_name("Explicit pressure")
        explicit_saturation.set_name("Explicit saturation")

        # Attach the pressure values to the flow equation and the saturation values to
        # the transport equation.
        new_flow_equation: pp.ad.Operator = old_flow_equation_masked + explicit_pressure
        new_transport_equation: pp.ad.Operator = (
            old_transport_equation_masked + explicit_saturation
        )
        new_flow_equation.set_name("Flow equation")
        new_transport_equation.set_name("Transport equation")

        self.equation_system.equations["Flow equation"] = new_flow_equation
        self.equation_system.equations["Transport equation"] = new_transport_equation


class ModifiedBoundaryConditions(BoundaryConditionsTPF):

    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann).

        We assign Neumann conditions for all faces. The four corner cells get prescribed
        a pressure explicitely, which acts as a Dirichlet condition.

        """
        return pp.BoundaryCondition(g)


class ModifiedConstitutiveLaws(
    RelativePermeabilityHC,
    ConstitutiveLawsTPF,
): ...


class ModifiedEstimates(EstimatesHCMixin):
    """Null the estimates in the corner (well) cells."""

    def local_residual_est(self, flux_name: Literal["total", PHASENAME]) -> None:
        super().local_residual_est(flux_name)

        sd, sd_data = self.mdg.subdomains(return_data=True)[0]
        R_estimate: np.ndarray = pp.get_solution_values(
            f"{flux_name}_R_estimate", sd_data, iterate_index=0
        )
        R_estimate[self.corner_cell_ids(sd)] = 0
        pp.set_solution_values(
            f"{flux_name}_R_estimate",
            R_estimate,
            sd_data,
            iterate_index=0,
        )

    def local_flux_est(self, flux_name: Literal["total", "wetting"]) -> None:
        super().local_flux_est(flux_name)

        sd, sd_data = self.mdg.subdomains(return_data=True)[0]
        F_estimate: np.ndarray = pp.get_solution_values(
            f"{flux_name}_F_estimate", sd_data, iterate_index=0
        )
        F_estimate[self.corner_cell_ids(sd)] = 0
        pp.set_solution_values(
            f"{flux_name}_F_estimate",
            F_estimate,
            sd_data,
            iterate_index=0,
        )

    def local_nonconformity_est(
        self,
        pressure_key: Literal[GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
    ) -> None:
        super().local_nonconformity_est(pressure_key)

        sd, sd_data = self.mdg.subdomains(return_data=True)[0]
        NC_estimate: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_NC_estimate", sd_data, iterate_index=0
        )
        NC_estimate[self.corner_cell_ids(sd)] = 0
        pp.set_solution_values(
            f"{pressure_key}_NC_estimate",
            NC_estimate,
            sd_data,
            iterate_index=0,
        )

        NC_estimate_inner_product_new_old: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_NC_estimate_inner_product_new_old",
            sd_data,
            iterate_index=0,
        )
        NC_estimate_inner_product_new_old[self.corner_cell_ids(sd)] = 0
        pp.set_solution_values(
            f"{pressure_key}_NC_estimate_inner_product_new_old",
            NC_estimate_inner_product_new_old,
            sd_data,
            iterate_index=0,
        )

    def local_hc_est(self, flux_name: Literal["total", "wetting"]) -> None:
        """Set the local estimates for the corner cells to zero."""
        super().local_hc_est(flux_name)

        sd, sd_data = self.mdg.subdomains(return_data=True)[0]
        C_estimate: np.ndarray = pp.get_solution_values(
            f"{flux_name}_C_estimate", sd_data, iterate_index=0
        )
        C_estimate[self.corner_cell_ids(sd)] = 0
        pp.set_solution_values(
            f"{flux_name}_C_estimate",
            C_estimate,
            sd_data,
            iterate_index=0,
        )

    def local_linearization_est(self, flux_name: Literal["total", "wetting"]) -> None:
        """Set the local estimates for the corner cells to zero."""
        super().local_linearization_est(flux_name)

        sd, sd_data = self.mdg.subdomains(return_data=True)[0]
        L_estimate: np.ndarray = pp.get_solution_values(
            f"{flux_name}_L_estimate", sd_data, iterate_index=0
        )
        L_estimate[self.corner_cell_ids(sd)] = 0
        pp.set_solution_values(
            f"{flux_name}_L_estimate",
            L_estimate,
            sd_data,
            iterate_index=0,
        )


class ModifiedSolutionStrategyMixin:

    mdg: pp.MixedDimensionalGrid
    wetting: FluidPhase
    nonwetting: FluidPhase
    corner_cell_ids: Callable[[pp.Grid], list[int]]
    equation_system: pp.EquationSystem

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation.

        The corner cells get prescibed the right values immediately. Inside the
        reservoir, the initial pressure is higher. The initial saturation is set to the
        residual wetting saturation + 0.1 inside the reservoir.

        """
        g: pp.Grid = self.mdg.subdomains()[0]
        corner_cell_ids: list[int] = self.corner_cell_ids(g)

        initial_pressure = np.full(g.num_cells, 6000 * PSI)
        initial_pressure[corner_cell_ids] = 4000 * PSI
        initial_saturation = np.full(g.num_cells, 0.4)
        initial_saturation[corner_cell_ids] = 1 - self.wetting.residual_saturation
        self.equation_system.set_variable_values(
            np.concatenate([initial_pressure, initial_pressure]),
            [self.wetting.p, self.nonwetting.p],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )
        self.equation_system.set_variable_values(
            np.concatenate([initial_saturation, 1 - initial_saturation]),
            [self.wetting.s, self.nonwetting.s],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )


class ModifiedTPF(
    # SPE10:
    SPE10Mixin,
    # Model mixins:
    ModifiedEquations,
    VariablesTPF,
    ModifiedConstitutiveLaws,
    ModifiedGeometry,
    ModifiedBoundaryConditions,
    ModifiedSolutionStrategyMixin,
    # Homotopy continuation mixins:
    EstimatesHCMixin,
    SolutionStrategyHCMixin,
    SolutionStrategyEstHCMixin,
    DataSavingMixinHC,
    # Estimator mixins:
    ErrorEstimateMixin,
    SolutionStrategyEst,
    DataSavingMixinEst,
    # Reconstruction mixins:
    GlobalPressureMixin,
    PressureReconstructionMixin,
    EquilibratedFluxMixin,
    DataSavingMixinRec,
    # Base data saving:
    SolutionStrategyTPF,
    DataSavingTPF,
): ...  # type: ignore


# endregion

# region RUN
spe10_layer: int = 80
cell_size: float = 600 * FEET / 30

params = {
    # Base folder and file name. These will get changed by
    # ``ConvergenceAnalysisExtended``.
    "file_name": "setup",
    "progressbars": True,
    "nonlinear_solver_statistics": SolverStatisticsHC,
    "spe10_layer": spe10_layer - 1,
    "spe10_isotropic_perm": True,
    # HC params:
    "nonlinear_solver": HCSolver,
    "hc_max_iterations": 20,
    "hc_lambda_min": 0.0,
    # HC decay parameters.
    "hc_constant_decay": False,
    "hc_lambda_decay": 0.8,
    "hc_decay_min_max": (0.1, 0.95),
    "nl_iter_optimal_range": (4, 7),
    "nl_iter_relax_factors": (0.7, 1.3),
    "hc_decay_recomp_max": 5,
    # Adaptivity parameters.
    "hc_error_ratio": 0.1,
    "nl_error_ratio": 0.1,
    # Nonlinear params:
    "max_iterations": 20,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e15,
    # Grid and time discretization:
    "grid_type": "simplex",
    # Model:
    "formulation": "fractional_flow",
    "rel_perm_constants": {},
    "cap_press_constants": {},
}
rel_perm_constants_list_1: list[dict[str, Any]] = [
    {
        "model": "linear",
        "limit": False,
        "linear_param_w": 1,
        "linear_param_n": 1,
    },
]
rel_perm_constants_list_2: list[dict[str, Any]] = [
    {
        "model": "Brooks-Corey",
        "limit": False,
        "n1": 2,
        "n2": 2,  # 1 + 2/n_b
        "n3": 1,
    },
]

cap_press_constants_list: list[dict[str, Any]] = [
    # {"model": None}
    {
        "model": "Brooks-Corey",
        "entry_pressure": 5 * PSI,
        "n_b": 2,
    },
]


for i, (rp_model_1, rp_model_2, cp_model) in enumerate(
    itertools.product(
        rel_perm_constants_list_1, rel_perm_constants_list_2, cap_press_constants_list
    )
):
    continue
    logger.info(
        f"Run {i + 1} of {len(rel_perm_constants_list_1) *  len(rel_perm_constants_list_2)* len(cap_press_constants_list)}"
    )
    logger.info(
        f"Cell size: {cell_size:.2f}, RP model 1: {rp_model_1['model']}, RP model 2: {rp_model_2['model']}, CP model: {cp_model['model']}"
    )

    # We have the file name both in the folder name and the filename to make
    # distinguishing different runs in ParaView easier.
    filename: str = (
        f"rp1_{rp_model_1['model']} rp2_{rp_model_2['model']}_cp_{cp_model['model']}"
    )
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / "homotopy continuation"
        / f"lay_{spe10_layer}_cellsz_{int(cell_size)}"
        / filename
    )

    try:
        shutil.rmtree(foldername)
    except Exception:
        pass
    foldername.mkdir(parents=True)

    params.update(
        {
            # Reinitialize the time manager for each run
            "time_manager": pp.TimeManager(
                schedule=np.array([0, 1.0 * pp.DAY]),  # 5 days
                dt_init=1.0 * pp.DAY,  # time step size in days
                dt_min_max=(1e-4 * pp.DAY, 1.0 * pp.DAY),
                constant_dt=False,
                recomp_factor=0.1,
                recomp_max=5,
            ),
            "folder_name": foldername,
            "file_name": filename,
            "solver_statistics_file_name": foldername / "solver_statistics.json",
            "meshing_arguments": {"cell_size": cell_size},
            "rel_perm_constants": {"model_1": rp_model_1, "model_2": rp_model_2},
            "cap_press_constants": cp_model,
        }
    )

    model = ModifiedTPF(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")

# endregion

# region PLOTTING
for i, (rp_model_1, rp_model_2, cp_model) in enumerate(
    itertools.product(
        rel_perm_constants_list_1, rel_perm_constants_list_2, cap_press_constants_list
    )
):
    filename: str = (
        f"rp1_{rp_model_1['model']} rp2_{rp_model_2['model']}_cp_{cp_model['model']}"
    )
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / "homotopy continuation"
        / f"lay_{spe10_layer}_cellsz_{int(cell_size)}"
        / filename
    )
    solver_statistics_file: pathlib.Path = foldername / "solver_statistics.json"
    with open(solver_statistics_file) as f:
        history: dict[str, Any] = json.load(f)
        history_list: list[Any] = list(history.values())[1:]

    discretization_est: list[float] = []
    hc_est: list[float] = []
    linearization_est: list[float] = []
    for time_step in history_list:
        for nl_step in list(time_step.values()):
            if isinstance(nl_step, int) or isinstance(nl_step, float):
                continue
            discretization_est.extend(nl_step["discretization_error_estimates"])
            hc_est.extend(nl_step["hc_error_estimates"])
            linearization_est.extend(nl_step["linearization_error_estimates"])

    fig, ax = plt.subplots()
    ax.semilogy(discretization_est[:100], label="Discretization estimator")
    ax.semilogy(hc_est[:100], label="HC estimator")
    ax.semilogy(linearization_est[:100], label="Linearization estimator")
    ax.set_xlabel("Nonlinear iteration")
    ax.set_ylabel("Estimator")
    ax.set_title(f"Estimator values")
    ax.legend()
    plt.show()
    fig.savefig(foldername / "solver_convergence.png")

    fig, ax = plt.subplots()
    ax.semilogy(discretization_est[:100], label="Discretization estimator")
    ax.semilogy(hc_est[:100], label="HC estimator")
    ax.semilogy(linearization_est[:100], label="Linearization estimator")
    ax.set_ylim([5e-5, 1e5])
    ax.set_xlabel("Nonlinear iteration")
    ax.set_ylabel("Estimator")
    ax.set_title(f"Estimator values")
    ax.legend()
    plt.show()
    fig.savefig(foldername / "solver_convergence_zoomed.png")

# endregion
