"""Compare the accuracy of adaptive Newton, adaptive Newton with Appleyard chopping, and
adaptive homotopy continuation.

We fix the cell size and time step s.t. basic Newton converges and treat this as the
baseline. Next, we run the abovementioned solvers and compare the accuracy during
nonlinear iterations.

Model description:
- 600x1100 ft domain (we just take a quarter of the original SPE10 domain)
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 10 days
- Solid properties:
    - Porosity: Homogeneous 0.3.
    - Permeability: Homogenous; 1e-15 m^2.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Pressure: 6000 psi
    - Saturation: residual water saturation (0.2) or 0.3.
- Rel. perm. models:
    - linear
    - Corey with power 2.
- Capillary pressure model:
    - Brooks-Corey
- Time step size and cell size are kept constant.

"""

import itertools
import json
import logging
import os
import pathlib
import shutil
import typing
import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib.ticker as tck
import numpy as np
import porepy as pp
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tpf.derived_models.spe10 import INITIAL_PRESSURE, SPE10Mixin
from tpf.models.adaptive_newton import TwoPhaseFlowANewton
from tpf.models.flow_and_transport import EquationsTPF
from tpf.models.homotopy_continuation import TwoPhaseFlowAHC
from tpf.models.protocol import TPFProtocol
from tpf.numerics.nonlinear.hc_solver import HCSolver
from tpf.utils.constants_and_typing import FEET
from tpf.viz.pca import biplot, screeplot
from tpf.viz.solver_statistics import SolverStatisticsANewton, SolverStatisticsHC

# region SETUP

# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch all numpy errors except underflow. The latter can appear during estimator
# calculation.
np.seterr(all="raise")
np.seterr(under="ignore")

warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

initial_saturation: float = 0.2
dirname: pathlib.Path = (
    pathlib.Path(__file__).parent.resolve()
    / "solver_accuracy"
    / f"init_{initial_saturation}"
)

# endregion


# region MODEL
@dataclass
class SolverStatisticsANewtonAcc(SolverStatisticsANewton):
    """Include accuracy compared to baseline in solver statistics."""

    accuracies: list[dict[str, float]] = field(default_factory=list)
    """List of accuracy for each non-linear iteration."""

    @typing.override
    def log_error(
        self,
        nonlinear_increment_norm: float | None = None,
        residual_norm: float | None = None,
        **kwargs,
    ) -> None:
        if "accuracy" in kwargs:
            self.accuracies.append(kwargs["accuracy"])
        else:
            super().log_error(nonlinear_increment_norm, residual_norm, **kwargs)

    @typing.override
    def reset(self) -> None:
        """Reset the estimator lists."""
        super().reset()
        self.accuracies.clear()

    @typing.override
    def save(self) -> None:
        """Save the estimator statistics to a JSON file."""
        # This calls ``pp.SolverStatistics.save``, which adds a new entry to the
        # ``data`` dictionary that is found at ``self.path``.
        super().save()
        # Instead of creating a new entry, we load the already created entry and append.
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
            else:
                data = {}

            # Append data.
            ind = len(data)
            # Since data was stored and loaded as json, the keys have turned to strings.
            data[str(ind)].update(
                {
                    "accuracies": self.accuracies,
                }
            )

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


@dataclass
class SolverStatisticsHCAcc(SolverStatisticsHC):
    """Include accuracy compared to baseline in solver statistics."""

    accuracies: list[list[dict[str, float]]] = field(default_factory=list)
    """List of accuracy for each non-linear iteration."""

    @typing.override
    def log_error(
        self,
        nonlinear_increment_norm: float | None = None,
        residual_norm: float | None = None,
        **kwargs,
    ) -> None:
        if "accuracy" in kwargs:
            # When called in :meth:`prepare_simulation`, add an empty list.
            if not self.accuracies:
                self.accuracies.append([])
            self.accuracies[-1].append(kwargs["accuracy"])
        else:
            super().log_error(nonlinear_increment_norm, residual_norm, **kwargs)

    @typing.override
    def reset(self) -> None:
        """Reset the estimator lists."""
        super().reset()
        self.accuracies.append([])

    @typing.override
    def hc_reset(self) -> None:
        """Reset the estimator lists."""
        super().hc_reset()
        self.accuracies.clear()

    @typing.override
    def save(self) -> None:
        """Save the estimator statistics to a JSON file."""
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data: dict[int, Any] = json.load(file)
            else:
                data = {}

            # Append data.
            ind: int = len(data) + 1

            if self.hc_num_iteration > 0:
                # :meth:`reset` is called at the start of each Newton loop, so we have to
                # append some of the last Newton loop data.
                self.nums_iteration.append(self.num_iteration)
                self.nonlinear_increment_norms_hc.append(self.nonlinear_increment_norms)
                self.residual_norms_hc.append(self.residual_norms)

            # The data is organized into dictionaries for each hc step. Each hc step
            # contains lists with values for all Newton steps.
            data[ind] = {
                i: {
                    "num_iteration": n,
                    "nonlinear_increment_norms": nin,
                    "residual_norms": rn,
                    "spatial_error_estimates": se,
                    "temporal_error_estimates": te,
                    "hc_error_estimates": hce,
                    "linearization_error_estimates": le,
                    "global_energy_norm": gen,
                    "equilibrated_flux_mismatch": efm,
                    "accuracy": acc,
                }
                for i, (n, nin, rn, se, te, hce, le, gen, efm, acc) in enumerate(
                    zip(
                        self.nums_iteration,
                        self.nonlinear_increment_norms_hc,
                        self.residual_norms_hc,
                        self.spatial_est,
                        self.temp_est,
                        self.hc_est,
                        self.lin_est,
                        self.global_energy_norm,
                        self.equilibrated_flux_mismatch,
                        self.accuracies,
                    )
                )
            }
            data[ind].update(
                {
                    "time step index": self.time_step_index,
                    "current time": self.time,
                    "time step size": self.time_step_size,
                    # Do not log the latest hc iteration, since it wasn't solved in a
                    # Newton loop. This is because :meth:`after_hc_iteration` is called
                    # before :meth:`after_hc_convergence/failure`
                    "hc_lambdas": self.hc_lambdas[:-1],
                    "hc_num_iterations": self.hc_num_iteration - 1,
                }
            )
            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


class HomogeneousAndAccuracyMixin(TPFProtocol):
    """Override the heterogeneous geometry of the SPE10 model by using methods of
    ``EquationsTPF`` instead.

    """

    def permeability(self, g: pp.Grid) -> dict[str, np.ndarray]:
        """Homogeneous solid permeability. Units are set by
        :attr:`self.solid`."""
        return EquationsTPF.permeability(self, g)  #  type: ignore

    def porosity(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous solid porosity. Chosen layer of the SPE10 model."""
        return EquationsTPF.porosity(self, g)  #  type: ignore

    def load_spe10_model(self, g: pp.Grid) -> None:
        pass

    def add_constant_spe10_data(self) -> None:
        pass

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        initial_pressure = np.full(self.g.num_cells, INITIAL_PRESSURE)
        initial_saturation = np.full(
            self.g.num_cells, self.params["spe10_initial_saturation"]
        )
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

    # Methods to calculate the accuracy of the solution compared to the baseline.
    def prepare_simulation(self) -> None:
        """Prepare the simulation."""
        super().prepare_simulation()  # type: ignore
        if self.params.get("compute_accuracy", False):
            self.baseline_solution: np.ndarray = np.load(
                dirname / "baseline" / "solution_vector.npy"
            )
            self.baseline_solution_sat_norm: float = np.linalg.norm(
                self.baseline_solution[: self.g.num_cells]
            ).item()
            self.baseline_solution_pres_norm: float = np.linalg.norm(
                self.baseline_solution[self.g.num_cells :]
            ).item()

            # Compute initial accuracy. NOTE This will make
            # :attr:`nonlinear_solver_statistics.accuracies` have one more entry on the
            # first time step than the other statistics.
            # saturation_acc, pressure_acc = self.compute_accuracy()
            # self.nonlinear_solver_statistics.log_error(
            #     accuracy={"saturation": saturation_acc, "pressure": pressure_acc}
            # )
            # self.save_solution_vector()

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Check convergence of the nonlinear solver."""
        converged, diverged = super().check_convergence(  # type: ignore
            nonlinear_increment, residual, reference_residual, nl_params
        )
        if self.params.get("compute_accuracy", False):
            saturation_acc, pressure_acc = self.compute_accuracy()
            self.nonlinear_solver_statistics.log_error(
                accuracy={"saturation": saturation_acc, "pressure": pressure_acc}
            )
            self.save_solution_vector()
        return converged, diverged

    def compute_accuracy(self) -> tuple[float, float]:
        """Compute the accuracy of the current solution compared to the baseline."""
        current_saturation: np.ndarray = self.equation_system.get_variable_values(
            [self.primary_saturation_var],
            iterate_index=0,
        )
        current_pressure: np.ndarray = self.equation_system.get_variable_values(
            [self.primary_pressure_var],
            iterate_index=0,
        )
        saturation_acc: float = (
            np.linalg.norm(
                self.baseline_solution[: self.g.num_cells] - current_saturation
            ).item()
            / self.baseline_solution_sat_norm
        )
        pressure_acc: float = (
            np.linalg.norm(
                self.baseline_solution[self.g.num_cells :] - current_pressure
            ).item()
            / self.baseline_solution_pres_norm
        )
        return saturation_acc, pressure_acc

    def save_solution_vector(self) -> None:
        """Save the solution vector."""
        current_solution: np.ndarray = self.equation_system.get_variable_values(
            [self.primary_saturation_var, self.primary_pressure_var],
            iterate_index=0,
        )
        if self.uses_hc:
            filename: str = (
                self.params["file_name"]
                + "solution_vector"
                + f"_{self.nonlinear_solver_statistics.hc_num_iteration}"  # type: ignore
                + f"_{self.nonlinear_solver_statistics.num_iteration}.npy"
            )
        else:
            filename = (
                self.params["file_name"]
                + f"solution_vector_{self.nonlinear_solver_statistics.num_iteration}.npy"
            )
        np.save(self.params["folder_name"] / filename, current_solution)


class HomogeneousSPE10Newton(
    HomogeneousAndAccuracyMixin, SPE10Mixin, TwoPhaseFlowANewton
): ...


class HomogeneousSPE10AHC(HomogeneousAndAccuracyMixin, SPE10Mixin, TwoPhaseFlowAHC): ...


# endregion

# region RUN
params: dict[str, Any] = {
    "folder_name": dirname / "baseline",
    "file_name": "baseline",
    "solver_statistics_file_name": dirname / "baseline" / "solver_statistics.json",
    "progressbars": True,
    # Model:
    "material_constants": {
        "solid": pp.SolidConstants({"porosity": 0.3, "permeability": 1e-15}),
    },
    "rel_perm_constants": {
        "model": "Brooks-Corey",
        "limit": False,
        "n1": 2,
        "n2": 2,  # 1 + 2/n_b
        "n3": 1,
    },
    "cap_press_constants": {"model": None},
    "grid_type": "simplex",
    "meshing_arguments": {"cell_size": 600 * FEET / 15},
    "spe10_quarter_domain": True,
    "spe10_initial_saturation": initial_saturation,
    # Nonlinear params:
    "nonlinear_solver_statistics": SolverStatisticsANewtonAcc,
    "nonlinear_solver": pp.NewtonSolver,
    "max_iterations": 20,
    "nl_convergence_tol": 1e-7,
    "nl_divergence_tol": 1e15,
    "nl_sat_increment_norm_scaling": 1.0,
    "nl_pres_increment_norm_scaling": INITIAL_PRESSURE,
    "nl_adaptive": False,
}

# region BASELINE
# Solve once to get the baseline.

try:
    shutil.rmtree(dirname / "baseline")
except Exception:
    pass
(dirname / "baseline").mkdir(parents=True)

# Try until we find a time step that works.
for time_step in np.logspace(1, -3, 5) * pp.DAY:
    params.update(
        {
            # Reinitialize the time manager for each run.
            "time_manager": pp.TimeManager(
                schedule=np.array([0.0, time_step]),
                dt_init=time_step,
                constant_dt=True,
            ),
        }
    )
    model = HomogeneousSPE10Newton(params)  # type: ignore
    try:
        pp.run_time_dependent_model(model=model, params=params)
        solution_vector: np.ndarray = model.equation_system.get_variable_values(
            [model.primary_saturation_var, model.primary_pressure_var],
            time_step_index=0,
        )
        np.save(dirname / "baseline" / "solution_vector.npy", solution_vector)
        break
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")

# endregion

# region VARYING_SOLVERS
# Solve with different solvers and solver settings and compare the accuracy with the
# baseline.

adaptive_error_ratio: float = 0.01
solvers: list[str] = ["Newton", "AHC"]
appleyard_chopping_list: list[bool] = [False, True]

for i, (
    solver,
    appleyard_chopping,
) in enumerate(
    itertools.product(
        solvers,
        appleyard_chopping_list,
    )
):
    if appleyard_chopping:
        # AHC is not run with Appleyard chopping.
        if solver == "AHC":
            continue
        max_iterations: int = 50
        appleyard_str: str = "_Appleyard"
    else:
        max_iterations = 20
        appleyard_str = ""

    filename: str = f"{solver}{appleyard_str}_{adaptive_error_ratio}"
    foldername: pathlib.Path = dirname / filename

    try:
        shutil.rmtree(foldername)
    except Exception:
        pass
    foldername.mkdir(parents=True)

    params.update(
        {
            "folder_name": foldername,
            "file_name": filename,
            "solver_statistics_file_name": foldername / "solver_statistics.json",
            "nonlinear_solver_statistics": (
                SolverStatisticsANewtonAcc
                if solver == "Newton"
                else SolverStatisticsHCAcc
            ),
            "compute_accuracy": True,
            # Reinitialize the time manager for each run.
            "time_manager": pp.TimeManager(
                schedule=np.array([0.0, time_step]),
                dt_init=time_step,
                constant_dt=True,
            ),
            "spe10_initial_saturation": initial_saturation,
        }
    )
    if solver == "Newton":
        params.update(
            {
                "rel_perm_constants": {
                    "model": "Brooks-Corey",
                    "limit": False,
                    "n1": 2,
                    "n2": 2,  # 1 + 2/n_b
                    "n3": 1,
                },
                "nonlinear_solver_statistics": SolverStatisticsANewtonAcc,
                "nonlinear_solver": pp.NewtonSolver,
                "nl_adaptive": True,
                "nl_adaptive_convergence_tol": 1e3,
                "nl_error_ratio": adaptive_error_ratio,
                "nl_appleyard_chopping": appleyard_chopping,
                "max_iterations": max_iterations,
            }
        )
    elif solver == "AHC":
        params.update(
            {
                "rel_perm_constants": {
                    "model_1": {"model": "linear", "limit": False},
                    "model_2": {
                        "model": "Brooks-Corey",
                        "limit": False,
                        "n1": 2,
                        "n2": 2,  # 1 + 2/n_b
                        "n3": 1,
                    },
                },
                # HC params:
                "nonlinear_solver_statistics": SolverStatisticsHCAcc,
                "nonlinear_solver": HCSolver,
                "hc_max_iterations": 20,
                "hc_adaptive": True,
                # HC decay parameters.
                "hc_constant_decay": False,
                "hc_lambda_decay": 0.9,
                "hc_decay_min_max": (0.1, 0.95),
                "nl_iter_optimal_range": (4, 7),
                "nl_iter_relax_factors": (0.7, 1.3),
                "hc_decay_recomp_max": 5,
                # Adaptivity parameters.
                "hc_error_ratio": adaptive_error_ratio,
                "nl_error_ratio": 0.1,
                "hc_nl_convergence_tol": 1e3,
                # Nonlinear params:
                "nl_adaptive": True,
                "max_iterations": 20,
                "nl_appleyard_chopping": False,
            }
        )

    model = (
        HomogeneousSPE10Newton(params)  # type: ignore
        if solver == "Newton"
        else HomogeneousSPE10AHC(params)  # type: ignore
    )
    pp.run_time_dependent_model(model=model, params=params)

# endregion
# endregion

# region PLOTTING
# Plot the accuracy of the different solvers.
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

for i, (
    solver,
    appleyard_chopping,
) in enumerate(
    itertools.product(
        solvers,
        appleyard_chopping_list,
    )
):
    if appleyard_chopping:
        # AHC is not run with Appleyard chopping.
        if solver == "AHC":
            continue
        appleyard_str: str = "_Appleyard"
    else:
        appleyard_str = ""

    filename: str = f"{solver}{appleyard_str}_{adaptive_error_ratio}"
    foldername: pathlib.Path = dirname / filename

    with (foldername / "solver_statistics.json").open("r") as file:
        data = json.load(file)

    # The first data point corresponds to the initial condition and is empty.
    if solver == "Newton":
        sat_accuracies: list[float] = [
            acc["saturation"] for acc in data["2"]["accuracies"]
        ]
        pres_accuracies: list[float] = [
            acc["pressure"] for acc in data["2"]["accuracies"]
        ]
    elif solver == "AHC":
        sat_accuracies = [
            acc["saturation"]
            for hc_step in list(data["2"].values())[:-5]
            for acc in hc_step["accuracy"]
        ]
        pres_accuracies = [
            acc["pressure"]
            for hc_step in list(data["2"].values())[:-5]
            for acc in hc_step["accuracy"]
        ]

        # Get estimators
        spatial_est: list[float] = [
            est
            for hc_step in list(data["2"].values())[:-5]
            for est in hc_step["spatial_error_estimates"]
        ]
        temp_est: list[float] = [
            est
            for hc_step in list(data["2"].values())[:-5]
            for est in hc_step["temporal_error_estimates"]
        ]
        hc_est: list[float] = [
            est
            for hc_step in list(data["2"].values())[:-5]
            for est in hc_step["hc_error_estimates"]
        ]
        lin_est: list[float] = [
            est
            for hc_step in list(data["2"].values())[:-5]
            for est in hc_step["linearization_error_estimates"]
        ]

        # Add vertical lines for HC iterations and lambda values.
        nl_iterations: list[int] = [
            hc_step["num_iteration"] for hc_step in list(data["2"].values())[:-5]
        ]
        num_iterations: int = 0
        lambdas: list[float] = []
        for i, hc_iter in enumerate(nl_iterations):
            num_iterations += hc_iter
            lambdas.extend([data["2"]["hc_lambdas"][i]] * hc_iter)
            ax1.axvline(x=num_iterations, color="grey", linestyle="--", alpha=0.5)

        # Plot lambda values.
        for ax in [ax1, ax3]:
            ax_twin = ax.twinx()
            ax_twin.plot(
                range(num_iterations), lambdas, color="red", linestyle="--", alpha=0.5
            )
            ax_twin.set_ylabel(r"$\lambda$", color="red")
            ax_twin.tick_params(axis="y", labelcolor="red")
            ax_twin.set_ylim(0.0, 1.1)

        # Plot estimators.
        for est, est_name in zip(
            [spatial_est, temp_est, hc_est, lin_est],
            [
                r"$\eta_{\mathrm{spat}}$",
                r"$\eta_{\mathrm{temp}}$",
                r"$\eta_{\mathrm{HC}}$",
                r"$\eta_{\mathrm{lin}}$",
            ],
        ):
            ax3.plot(est, label=est_name)

    accuracies: np.ndarray = np.array(sat_accuracies) + np.array(pres_accuracies)
    ax1.plot(accuracies, label=filename)
    ax2.plot(sat_accuracies, pres_accuracies, label=filename)
    if solver == "AHC":
        ax3.plot(accuracies, label="accuracy", linewidth=3)

ax1.set_ylim(bottom=0.0)
ax1.set_xlabel("Non-linear iteration")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.xaxis.set_major_locator(tck.MultipleLocator(base=5))
fig1.savefig(dirname / "accuracy.png")

ax2.set_xlabel("Saturation accuracy")
ax2.set_ylabel("Pressure accuracy")
ax2.legend()
fig2.savefig(dirname / "accuracy_more_details.png")
ax2.set_xscale("log")
ax2.set_yscale("log")
fig2.savefig(dirname / "accuracy_more_details_log.png")

ax3.set_ylim(bottom=0.0)
ax3.set_xlabel("Non-linear iteration")
ax3.set_ylabel("Accuracy")
ax3.legend()
ax3.xaxis.set_major_locator(tck.MultipleLocator(base=5))
fig3.savefig(dirname / "estimators.png")

# PCA of the solution paths
solutions_dic: dict[str, list[np.ndarray]] = {}

for i, (
    solver,
    appleyard_chopping,
) in enumerate(
    itertools.product(
        solvers,
        appleyard_chopping_list,
    )
):
    if appleyard_chopping:
        # AHC is not run with Appleyard chopping.
        if solver == "AHC":
            continue
        appleyard_str: str = "_Appleyard"
    else:
        appleyard_str = ""

    filename: str = f"{solver}{appleyard_str}_{adaptive_error_ratio}"
    foldername: pathlib.Path = dirname / filename

    solutions_dic[filename] = []
    for file in foldername.glob("*solution_vector*.npy"):
        solutions_dic[filename].append(np.load(file))

# Normalize the solution vectors. Pressure values are large and saturation values low.
# Not normalizing would give the wrong impression of the importance of the pressure.
solutions: np.ndarray = np.vstack(
    [np.vstack(solutions_dic[solver]) for solver in solutions_dic]
)
solutions_scaled: np.ndarray = StandardScaler().fit_transform(solutions)

# Perform PCA (reduce to 2 dimensions for plotting).
pca = PCA()
reduced_solutions: np.ndarray = pca.fit_transform(solutions_scaled)

# Plot information about the PCA and the solutions in the reduced space.
fig = screeplot(pca, n=10)
fig.savefig(dirname / "scree_plot.png")

start_idx: int = 0  # Start index for each solver.
solver_to_color: dict[str, tuple] = {
    "Newton_0.1": (0.1, 0.1, 1.0, 1.0),
    "Newton_0.01": (0.1, 0.1, 1.0, 0.7),
    "Newton_Appleyard_0.1": (1.0, 0.1, 0.1, 1.0),
    "Newton_Appleyard_0.01": (1.0, 0.1, 0.1, 0.7),
    "AHC_0.1": (0.1, 1.0, 0.1, 1.0),
    "AHC_0.01": (0.1, 1.0, 0.1, 0.7),
}
colors: list[tuple] = [
    solver_to_color[solver]
    for solver in solutions_dic
    for _ in range(len(solutions_dic[solver]))
]
fig = biplot(
    reduced_solutions, pca.components_.T, colors=colors, labels=[""] * 2000, n=2000
)
fig.savefig(dirname / "biplot.png")

# endregion
