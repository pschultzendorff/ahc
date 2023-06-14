"""Implementation of the Buckley-Leverett model in the two-phase flow model."""

import logging
import math
import os
from functools import partial
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import scipy as sp
import sympy
import tqdm
from buckley_leverett import (
    analytical_solution,
    functions,
    grid,
    misc,
    numerical_solution,
)

import tpf_lab.visualization.error_curves as error_curves
from tpf_lab.models.buckley_leverett import BuckleyLeverettEquations
from tpf_lab.models.run_models import run_time_dependent_model
from tpf_lab.numerics.ad.functions import ad_pow, minimum
from tpf_lab.utils import logging_redirect_tqdm
from tpf_lab.visualization.diagnostics import DiagnosticsMixinExtended

# plt.rcParams.update(
#     {
#         "text.latex.preamble": r"\usepackage{lmodern}",
#         "text.usetex": True,
#         "font.size": 16,
#         # "font.family": "serif",
#         # "text.latex.unicode": True,
#     }
# )


# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BuckleyLeverett_perturbed_mobility_w(BuckleyLeverettEquations):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # Parameters for the error function derivative:
        self._yscales: list[float] = params.get("yscales", [1.0])
        self._xscales: list[float] = params.get("xscales", [200])
        self._offsets: list[float] = params.get("offsets", [0.5])

    def _mobility_w(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Add a perturbation to the wetting mobility."""
        mobility_w = super()._mobility_w(subdomains)
        upwind_w = pp.ad.UpwindAd(self.w_flux_key, subdomains)
        return mobility_w + upwind_w.upwind @ self._error_function_deriv()

    def _error_function_deriv(self) -> pp.ad.Operator:
        """Returns the derivative of the error function w.r.t. the saturation.

        This can be used to simulate perturbations in the cap. pressure and rel. perm.
        models.

        Returns:
            Derivative of the error function in terms of :math:`S_w`.
        """
        s = self._ad.saturation
        xscales = [pp.ad.Scalar(xscale) for xscale in self._xscales]
        yscales = [pp.ad.Scalar(yscale) for yscale in self._yscales]
        offsets = [pp.ad.Scalar(offset) for offset in self._offsets]
        exp_func = pp.ad.Function(pp.ad.functions.exp, "exp")
        square_func = pp.ad.Function(partial(ad_pow, exponent=2), "square")
        error = pp.ad.Scalar(0) * s
        for xscale, yscale, offset in zip(xscales, yscales, offsets):
            error = error + yscale * exp_func(
                pp.ad.Scalar(-1) * xscale * square_func(s - offset)
            )
        return error


class BuckleyLeverett_Analytics(
    DiagnosticsMixinExtended, BuckleyLeverett_perturbed_mobility_w
):
    ...


class FractionalFlowSympy_PerturbedMobilityW(functions.FractionalFlowSymPy):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        # Parameters for the error function derivative:
        self.yscales: list[float] = params.get("yscales", [1.0])
        self.xscales: list[float] = params.get("xscales", [200])
        self.offsets: list[float] = params.get("offsets", [0.5])

    def lambda_w(self):
        r"""Wetting phase mobility.

        Power model
        .. math::
            k_{r,w}(S_w)=S_w^3 + \epsilon(S_w)

        """
        return self.S_normalized() ** 3 + self.error_function_deriv()

    def error_function_deriv(self):
        return sympy.Add(
            *[
                yscale * sympy.exp(-xscale * (self.S_w - offset) ** 2)
                for xscale, yscale, offset in zip(
                    self.xscales, self.yscales, self.offsets
                )
            ]
        )


# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "buckley_leverett",
    "mobility_w_multiple_perturbations_saturation_normalized",
)
try:
    os.makedirs(foldername)
except Exception:
    pass

fh: Optional[logging.FileHandler] = None

try:
    os.makedirs(foldername)
except Exception:
    pass


# Set up model.
params: dict[str, Any] = {
    "formulation": "n_pressure_w_saturation",
    "folder_name": foldername,
}
model = BuckleyLeverett_Analytics(params)

model._grid_size = 200
model._phys_size = 20

model._density_w = 1.0
model._density_n = 1.0

model._rel_perm_model = "power"
model._rel_perm_linear_param = 1.0
model._limit_rel_perm = True


yscales = np.maximum(np.random.rand(20), 0.5).tolist()
yscales[:5] = np.arange(0.1, 0.5, 0.1)
xscales = [20000.0] * 20
offsets = np.linspace(0.4, 0.6, 20).tolist()

model._yscales = yscales
model._xscales = xscales
model._offsets = offsets

model._time_step = 0.2
model._schedule = np.array([0, 10.0])

model.prepare_simulation()

g = model.mdg.subdomains()[0]

lax_friedrichs_grid = grid.create_grid(
    (model.domain.bounding_box["xmin"], model.domain.bounding_box["xmax"]),
    (model.domain.bounding_box["xmax"] - model.domain.bounding_box["xmin"])
    / model._grid_size,
)
initial_condition = np.full_like(lax_friedrichs_grid, model._residual_saturation_w)
initial_condition[0 : int(model._grid_size / 2) - 10] = 1 - model._residual_saturation_n
initial_condition[
    int(model._grid_size / 2) - 10 : int(model._grid_size / 2) + 10
] = np.linspace(1 - model._residual_saturation_n, model._residual_saturation_w, 20)

params = {
    # Negative influx of the model, since the sides are switched
    "influx": -model._influx,
    "porosity": model._porosity(g)[0],
    "density_w": model._density_w,
    "density_n": model._density_n,
    "angle": model._angle,
    "S_M": 1 - model._residual_saturation_n,
    "S_m": model._residual_saturation_w,
    "yscales": model._yscales,
    "xscales": model._xscales,
    "offsets": model._offsets,
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
max_time_step: float = lax_friedrichs.cfl_condition()

# For some reason the end time needs to be multiplied by 10 to get the same
# result as the Lax-Friedrichs solver.
# The time step is also multiplied by 10.
plt.figure()
# Run different time steps to get convergence order in time.
for time_step in [
    max_time_step
]:  # np.linspace(max_time_step / 10, max_time_step * 10, 20):
    filename = f"timestep_{time_step}"
    # Remove old file handler.
    try:
        logger.removeHandler(fh)  # type: ignore
    except Exception:
        pass
    log_filename = os.path.join(foldername, filename + "_log.txt")
    fh = logging.FileHandler(filename=log_filename)
    logger.removeHandler(fh)
    logger.addHandler(fh)

    model.exporter._file_name = f"timestep_{time_step}"

    model._time_step = time_step * 10
    model._schedule = np.array([0, 10.0 + (model._time_step - 10.0 % model._time_step)])

    model.prepare_simulation()

    # Run and plot the fractional flow model.
    try:
        with logging_redirect_tqdm([logger]):
            run_time_dependent_model(
                model, {"nl_convergence_tol": 1e-10, "max_iterations": 30}
            )
    except Exception as e:
        print(e)
        pass

    # Plot condition numbers.
    diagnostics_filename = os.path.join(
        foldername, f"timestep_{time_step}_diagnostics.png"
    )
    diagnostics_data = model.run_diagnostics(
        default_handlers=("max",),
    )
    model.plot_diagnostics(diagnostics_data, key="max", filename=diagnostics_filename)

    # Plot error curves after the last time step.
    error_plot_filename = os.path.join(
        foldername, f"timestep_{time_step}_error_plot.png"
    )
    # errors = error_curves.read_errors_from_log(log_filename)
    # error_curves.plot_error_curves(error_plot_filename, errors)

    # Plot solution
    saturation = model.equation_system.get_variable_values(
        variables=[model._ad.saturation], time_step_index=0
    )
    # Switch sides of the saturation, as PorePy models it the other way around.
    plt.figure()
    plt.plot(
        np.linspace(-10, model._phys_size - 10, model._grid_size)[5:-5:],
        saturation[-5:5:-1],
        label=f"timestep_{time_step}_fractional flow solution",
    )

# Compute and plot the lax friedrichs solution.
lax_friedrichs.time_step = lax_friedrichs.cfl_condition()
for _ in tqdm.tqdm(list(np.arange(0, 1, lax_friedrichs.time_step))):
    lax_friedrichs.solve()
plt.plot(
    lax_friedrichs_grid,
    lax_friedrichs.previous_solution,
    label="lax friedrich solution",
)

# Compute and plot the analytical solution
concave_hull, f_prime = analytical.concave_hull()
# Cut on both sides to avoid weird behavior.
yy = np.arange(analytical.S_m, analytical.S_M, (analytical.S_M - analytical.S_m) / 500)[
    10:-10
]
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
