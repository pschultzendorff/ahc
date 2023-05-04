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
    def create_grid(self) -> None:
        # 1d grid
        cell_dims: np.ndarray = np.array([self._grid_size])
        phys_dims: np.ndarray = np.array([self._phys_size])
        g_cart: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims)
        g_cart.compute_geometry()
        self.mdg: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_cart]])
        self.domain = pp.Domain(
            bounding_box={
                "xmin": -5,
                "xmax": phys_dims[0] - 5,
                "ymin": 0,
                "ymax": 0,
            }
        )
        logger.debug("Grid created")


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


folder_basename: str = os.path.join("results", "bl-test")
try:
    os.makedirs(folder_basename)
except Exception:
    pass

fh: Optional[logging.FileHandler] = None

# Set up folder and files for logging/plots/saved time steps.
foldername = os.path.join(
    folder_basename,
    f"density_w_{0.0}_density_n_{0.0}",
)
try:
    os.makedirs(foldername)
except Exception:
    pass
filename: str = f"run"
# Remove old file handler.
try:
    # MyPy complains, although we are in a try statement, so we ignore it.
    logger.removeHandler(fh)  # type:ignore
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
model._phys_size = 10.0

model._density_w = 0.0
model._density_n = 0

model._rel_perm_model = "power"
model._rel_perm_linear_param = 1.0
model._limit_rel_perm = True


model.prepare_simulation()

FINAL_TIME = 1.0

# Initiate Lax-Friedrichs and analytical solver.
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
    "yscale": model._yscale,
    "xscale": model._xscale,
    "offset": model._offset,
    "rel_perm_model": model._rel_perm_model,
    "grid": lax_friedrichs_grid,
    "initial_condition": initial_condition,
}
lax_friedrichs = numerical_solution.BuckleyLeverett(params)
analytical = analytical_solution.BuckleyLeverett(params)


# Set time step to satisfy the CFL condition for explicit solvers.
model._time_step = lax_friedrichs.cfl_condition()
model._schedule = np.array([0, FINAL_TIME])
model.prepare_simulation()


with logging_redirect_tqdm([logger]):
    run_time_dependent_model(model, {"nl_convergence_tol": 1e-10, "max_iterations": 30})


# Plot condition numbers.
diagnostics_filename = os.path.join(foldername, "diagnostics.png")
diagnostics_data = model.run_diagnostics(
    default_handlers=("max",),
)
model.plot_diagnostics(diagnostics_data, key="max", filename=diagnostics_filename)

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
    np.linspace(
        model.domain.bounding_box["xmin"],
        model.domain.bounding_box["xmax"],
        model._grid_size,
    )[5:-5:],
    saturation[-5:5:-1],
    label="fractional flow solution",
)

# Compute and plot the lax friedrichs solution. # Comment this
lax_friedrichs.time_step = lax_friedrichs.cfl_condition()
for _ in tqdm.tqdm(list(np.arange(0, FINAL_TIME, lax_friedrichs.time_step))):
    lax_friedrichs.solve()
plt.plot(
    lax_friedrichs_grid,
    lax_friedrichs.previous_solution,
    label="lax friedrich solution",
)

# Compute and plot the analytical solution.
concave_hull, f_prime = analytical.concave_hull()
# Cut on both sides to avoid weird behavior.
yy = np.arange(analytical.S_m, analytical.S_M, (analytical.S_M - analytical.S_m) / 500)[
    10:-10
]
xx = f_prime(yy)
plt.plot(xx, yy, label="analytical solution")

# Finish plotting.
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
