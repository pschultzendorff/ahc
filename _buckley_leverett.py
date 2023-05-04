"""Implementation of the Buckley-Leverett model in the two-phase flow model."""

import logging
import math
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import tqdm
from buckley_leverett import (
    analytical_solution,
    functions,
    grid,
    misc,
    numerical_solution,
)

import tpf_lab.visualization.error_curves as error_curves
from tpf_lab.models.buckley_leverett import BuckleyLeverett
from tpf_lab.models.run_models import run_time_dependent_model
from tpf_lab.utils import logging_redirect_tqdm

try:
    # MyPy is not happy with Seaborn since it's not typed. We silence this warning.
    import seaborn as sns  # type: ignore[import]
except ImportError:
    _IS_SEABORN_AVAILABLE: bool = False
else:
    _IS_SEABORN_AVAILABLE = True


plt.rcParams.update(
    {
        "text.latex.preamble": r"\usepackage{lmodern}",
        "text.usetex": True,
        "font.size": 16,
        # "font.family": "serif",
        # "text.latex.unicode": True,
    }
)


# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BuckleyLeverett_perturbed_mobility_w(BuckleyLeverett):
    def _mobility_w(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Add a perturbation to the wetting mobility."""
        mobility_w = super()._mobility_w(subdomains)
        upwind_w = pp.ad.UpwindAd(self.w_flux_key, subdomains)
        return mobility_w + upwind_w.upwind @ self._error_function_deriv()


class DiagnosticsMixin_with_save_functionality(pp.DiagnosticsMixin):
    def plot_diagnostics(
        self, diagnostics_data, key: str, filename: str, **kwargs
    ) -> None:
        if _IS_SEABORN_AVAILABLE:
            plt.figure()
            super().plot_diagnostics(diagnostics_data, key)
            plt.savefig(filename)


class BuckleyLeverett_Analytics(
    DiagnosticsMixin_with_save_functionality, BuckleyLeverett_perturbed_mobility_w
):
    ...


class FractionalFlowSympy_PerturbedMobilityW(functions.FractionalFlowSymPy):
    def lambda_w(self):
        r"""Wetting phase mobility.

        Power model
        .. math::
            k_{r,w}(S_w)=S_w^3 + \epsilon(S_w)

        """
        return self.S_normalized() ** 3 + self.error_function_deriv()


yscales = np.arange(20.0, 25.0, 1.0)
densities = [1.0]
xscale = 50
offset = 0.35

folder_basename: str = os.path.join(
    "results", "buckley_leverett", "perturbed_mobility_w_saturation_normalized"
)
try:
    os.makedirs(folder_basename)
except Exception:
    pass

fh: Optional[logging.FileHandler] = None

for yscale in yscales:
    for density in densities:
        # Set up folder and files for logging/plots/saved time steps.
        foldername = os.path.join(
            folder_basename,
            f"yscale_{yscale}_xscale_{xscale}_offset_{offset}_density_w_{1.0}_density_n_{density}",
        )
        try:
            os.makedirs(foldername)
        except Exception:
            pass
        filename: str = f"yscale_{yscale}_density_w_{density}"
        # Remove old file handler.
        try:
            logger.removeHandler(fh)
        except Exception:
            pass
        log_filename = os.path.join(foldername, "log.txt")
        fh = logging.FileHandler(filename=log_filename)
        logger.removeHandler(fh)
        logger.addHandler(fh)

        # Set up model.
        params: dict[str, Any] = {
            "formulation": "n_pressure_w_saturation",
            "file_name": filename,
            "folder_name": foldername,
        }
        model = BuckleyLeverett_Analytics(params)

        model._grid_size = 200
        model._phys_size = 20

        model._density_w = 1.0
        model._density_n = density

        model._rel_perm_model = "power"
        model._rel_perm_linear_param = 1.0
        model._limit_rel_perm = True

        model._yscale = yscale
        model._xscale = xscale
        model._offset = offset

        model.prepare_simulation()

        # Set up a numerical buckley-leverret instance to calculate the time step size
        # s.t. the CFL condition is satisfied.
        # Set up an analytical buckley-leverett instance to compare the solutions.
        g = model.mdg.subdomains()[0]

        lax_friedrichs_grid = grid.create_grid(
            (model.domain.bounding_box["xmin"], model.domain.bounding_box["xmax"]),
            (model.domain.bounding_box["xmax"] - model.domain.bounding_box["xmin"])
            / model._grid_size,
        )
        initial_condition = np.full_like(
            lax_friedrichs_grid, model._residual_saturation_w
        )
        initial_condition[0 : int(model._grid_size / 2) - 10] = (
            1 - model._residual_saturation_n
        )
        initial_condition[
            int(model._grid_size / 2) - 10 : int(model._grid_size / 2) + 10
        ] = np.linspace(
            1 - model._residual_saturation_n, model._residual_saturation_w, 20
        )

        params = {
            # Negative influx of the model, since the sides are switched
            "influx": -model._influx,
            "porosity": model._porosity(g)[0],
            "density_w": model._density_w,
            "density_n": model._density_n,
            "angle": model._angle,
            "S_M": 1 - model._residual_saturation_n,
            "S_m": model._residual_saturation_w,
            "yscale": model._yscale,
            "xscale": model._xscale,
            "offset": model._offset,
            "rel_perm_model": model._rel_perm_model,
            "grid": lax_friedrichs_grid,
            "initial_condition": initial_condition,
        }
        lax_friedrichs = numerical_solution.BuckleyLeverett(params)
        analytical = analytical_solution.BuckleyLeverett(params)

        # Exchange the fractional flow function to one with perturbed mobility.
        fractionalflow = FractionalFlowSympy_PerturbedMobilityW(params)
        lax_friedrichs.fractionalflow = fractionalflow
        lax_friedrichs.lambdify()
        analytical.fractionalflow = fractionalflow
        analytical.lambdify()

        # Get max time step size.
        time_step: float = lax_friedrichs.cfl_condition()

        # For some reason the end time needs to be multiplied by 10 to get the same
        # result as the Lax-Friedrichs solver.
        # The time step is also multiplied by 10.
        model._time_step = time_step * 10
        model._schedule = np.array(
            [0, 10.0 + (model._time_step - 10.0 % model._time_step)]
        )

        model._create_managers()
        model.prepare_simulation()

        model.before_newton_loop()
        model.before_newton_iteration()

        # Run and plot the fractional flow model.
        try:
            with logging_redirect_tqdm([logger]):
                logger.info(
                    f"set time step size to {time_step} to satisfy CFL condition"
                )
                run_time_dependent_model(
                    model, {"nl_convergence_tol": 1e-10, "max_iterations": 30}
                )
        except Exception:
            pass

        # Plot condition numbers.
        diagnostics_filename = os.path.join(foldername, "diagnostics.png")
        diagnostics_data = model.run_diagnostics(
            default_handlers=("max",),
        )
        model.plot_diagnostics(
            diagnostics_data, key="max", filename=diagnostics_filename
        )

        # Plot error curves after the last time step.
        error_plot_filename = os.path.join(foldername, "error_plot.png")
        errors = error_curves.read_errors_from_log(log_filename)
        error_curves.plot_error_curves(error_plot_filename, errors)

        # Plot solution
        plt.figure()
        saturation = model.equation_system.get_variable_values(
            variables=[model._ad.saturation], time_step_index=0
        )
        # Switch sides of the saturation, as PorePy models it the other way around.
        plt.plot(
            np.linspace(-10, model._phys_size - 10, model._grid_size)[5:-5:],
            saturation[-5:5:-1],
            label="fractional flow solution",
        )

        # Compute and plot the lax friedrichs solution.
        lax_friedrichs.time_step = lax_friedrichs.cfl_condition()
        for _ in tqdm.tqdm(
            list(
                np.arange(
                    model._schedule[0],
                    model._schedule[1] / 10,
                    lax_friedrichs.time_step,
                )
            )
        ):
            lax_friedrichs.solve()
        plt.plot(
            lax_friedrichs_grid,
            lax_friedrichs.previous_solution,
            label="lax friedrich solution",
        )

        # Compute and plot the analytical solution
        concave_hull, f_prime = analytical.concave_hull()
        # Cut on both sides to avoid weird behavior.
        yy = np.arange(
            analytical.S_m, analytical.S_M, (analytical.S_M - analytical.S_m) / 500
        )[10:-10]
        xx = f_prime(yy)
        plt.plot(xx, yy, label="analytical solution")
        plt.xlabel(rf"\(x\)")
        plt.ylabel(rf"\(S_w\)")
        plt.legend()
        plt.savefig(os.path.join(foldername, "compare_solutions") + ".png")
        plt.close()
        misc.map_fractional_flow(
            analytical,
            filename=os.path.join(foldername, filename),
        ),
        logging.info(f"finished run and saved to {os.path.join(foldername, filename)}")
