import json
import pathlib
import sys

import porepy as pp
from run import studies

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from SPE10_2D.plotting import _key_varying_rp, plot_study
from utils import (
    SimulationConfig,
    SolverStats,
)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

EXPECTED_FINAL_TIME = 3000.0 * pp.DAY

rel_errors: dict[str, str] = {}


def _key_varying_ref_factor(
    config: SimulationConfig, stats: SolverStats
) -> tuple[str, str]:
    return config.solver_specs(), str(stats.num_grid_cells)


if __name__ == "__main__":
    fig_dir = dirname / "figures"
    fig_dir.mkdir(exist_ok=True)

    for study_name, study in studies.items():
        kwargs = {}
        match study_name:
            case "viscous_varying_rp_init_s_08" | "viscous_varying_rp_init_s_09":
                key_func = _key_varying_rp
                varying_param_name = "Relative permeability model"
            case (
                "viscous_varying_ref_factor_init_s_08"
                | "viscous_varying_ref_factor_init_s_09"
            ):
                key_func = _key_varying_ref_factor
                varying_param_name = "Number of grid cells"
            case _:
                raise ValueError(f"Unknown study: {study_name}")

        fig = plot_study(
            study,
            key_func=key_func,
            varying_param_name=varying_param_name,
            **kwargs,
        )
        fig.savefig(fig_dir / f"nl_iters_{study_name}.png")

    with (fig_dir / "relative_errors.txt").open("w") as f:
        json.dump(rel_errors, f, indent=2)
