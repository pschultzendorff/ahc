import json
import logging
import os
from collections import ChainMap
from copy import deepcopy
from typing import Any, ClassVar, Literal, Optional, Protocol, Type

import matplotlib.pyplot as plt
import numpy as np
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from tpf_lab.utils import save_params_and_run_model
from tpf_lab.visualization.diagnostics import SaveDataBL, SaveDataTPF

# Get module wide logger.
logger = logging.getLogger(__name__)


class IsDataclass(Protocol):
    """For typehinting a ``dataclass``.

    Cf.
    https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass.

    """

    __dataclass_fields__: ClassVar[dict]


def all_annotations(cls) -> ChainMap:
    """Returns a dictionary-like ChainMap that includes annotations for all attributes
    defined in cls or inherited from superclasses.

    Copied from
    https://stackoverflow.com/questions/63903901/how-can-i-access-to-annotations-of-parent-class

    """
    return ChainMap(
        *(c.__annotations__ for c in cls.__mro__ if "__annotations__" in c.__dict__)
    )


class ConvergenceAnalysisExtended(ConvergenceAnalysis):
    def __init__(
        self,
        model_class,
        model_params: dict,
        levels: int = 1,
        spatial_refinement_rate: int = 1,
        temporal_refinement_rate: int = 1,
    ):
        super().__init__(
            model_class,
            model_params,
            levels,
            spatial_refinement_rate,
            temporal_refinement_rate,
        )
        # Run through list of model params for levels and create seperate folder names
        # for each level.
        self.base_folder_name = model_params["folder_name"]
        for params in self.model_params:
            params["folder_name"] = os.path.join(
                self.base_folder_name,
                f"cell_diameter_{params['meshing_arguments']['cell_size']}"
                + f"_dt_{params['time_manager'].dt}",
            )
            params["file_name"] = "data"

    def run_analysis(self) -> list:
        """Run convergence analysis. Changed from the super function s.t. results from
        each time step get saved. Returns a json compatible format.

        Returns:
            List of results (i.e., data classes containing the errors) for each
            refinement level and time step.

        """
        convergence_results: list = []
        for level in range(self.levels):
            level_result: dict = {}
            setup = self.model_class(deepcopy(self.model_params[level]))
            try:
                save_params_and_run_model(setup, deepcopy(self.model_params[level]))
            except Exception as e:
                # The model does not converge.
                logger.info(e)
            # Export the final model data.
            setup._export()
            # Add level information to results.
            level_result["dt"] = setup.time_manager.dt
            level_result["cell_diameter"] = setup.mdg.diameter()
            # Loop over lists of model results. Convert into a json compatible list.
            level_result["results"] = [
                result_to_dict(result) for result in setup.results
            ]
            convergence_results.append(level_result)
        return convergence_results

    def export_results_to_json(
        self,
        list_of_results: list,
        variables_to_export: Optional[list[str]] = None,
        file_name="convergence_analysis.json",
    ) -> None:
        """Write errors into a ``json`` file. Changed from the super function s.t.
        results from each time step get saved.

        The format is the following one:

            - "cell_diameter" contains the cell diameters.
            - "dt" contains the time step sizes (if the model is time-dependent).
            - "results" contains results for each variable in ``variables``.

        Parameters:
            list_of_results: List containing the results of the convergence analysis.
                Typically, the output of :meth:`run_analysis`.
            variables_to_export: names of the variables for which the TXT file will be
                generated. If ``variables`` is not given, all the variables present
                in the txt file will be collected.
            file_name: Name of the output file. Default is "error_analysis.txt".

        """
        # Filter variables from the list of results.
        var_names: list[str] = self._filter_variables_from_list_of_results(
            list_of_results=list_of_results,
            variables=variables_to_export,
        )
        results_to_export: list = []
        # Loop over lists of results. Filter all the wanted information.
        for level_result in list_of_results:
            result_to_export: dict = {"cell_diameter": level_result["cell_diameter"]}
            if self._is_time_dependent:
                result_to_export["dt"] = level_result["dt"]
            # Filter results to be exported.
            result_to_export["results"] = [
                {name: time_step_result[name] for name in var_names}
                for time_step_result in level_result["results"]
            ]
            results_to_export.append(result_to_export)

        # Save into the base folder, not into one of the individual level folders.
        file_name = os.path.join(self.base_folder_name, file_name)

        # Finally, call the function to write into the json.
        with open(file_name, "w") as f:
            json.dump(results_to_export, f, indent=2)

    def import_results_from_json(
        self,
        file_name="convergence_analysis.json",
    ) -> list:
        """Read results from a ``json`` file.

        Parameters:
            file_name: Name of the input file. Default is "error_analysis.txt".

        """
        with open(file_name, "r") as f:
            results = json.load(f)
        return results

    def transform_results_to_classical(
        self,
        list_of_results: list,
        save_data_class: Type[IsDataclass] = SaveDataTPF,
    ) -> list[IsDataclass]:
        """Transform the results from ``run_analysis`` to the format the super class
        uses."""
        new_list_of_results: list[IsDataclass] = []
        try:
            end_time: float = self.model_params[0]["time_manager"].time_final
        except:
            end_time = 1.0
        for level_result in list_of_results:
            # Only get the results of the last time step.
            for time_step_result in level_result["results"]:
                if np.isclose(time_step_result["time"], end_time):
                    kwargs = {name: value for name, value in time_step_result.items()}
                    save_data = save_data_class(**kwargs)
                    # Set refinement level attributes.
                    setattr(save_data, "cell_diameter", level_result["cell_diameter"])
                    if self._is_time_dependent:
                        setattr(save_data, "dt", level_result["dt"])
                    new_list_of_results.append(save_data)
        return new_list_of_results

    def average_number_of_iterations(
        self, list_of_results: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute the average number of nonlinear iterations for each level.

        Warning:
            The model might run up until :math:`1+\Delta t-\epsilon`, i.e., one time
            step longer than expected. The function will also take into account this
            additional time step.

        """
        avg_nonlinear_iterations: list[float] = []
        spatial_refinement_level: list[float] = []
        temporal_refinement_level: list[float] = []
        try:
            end_time: float = self.model_params[0]["time_manager"].time_final
        except:
            end_time = 1.0

        for level_result in list_of_results:
            nonlinear_iterations = [
                time_step_result["iteration_counter"]
                for time_step_result in level_result["results"]
            ]

            # Only append results if the last time step was reached.
            if len(nonlinear_iterations) > 0 and (
                level_result["results"][len(nonlinear_iterations) - 1]["time"]
                >= end_time
            ):
                avg_nonlinear_iterations.append(
                    sum(nonlinear_iterations) / len(nonlinear_iterations)
                )
                # Append refinement levels.
                spatial_refinement_level.append(level_result["cell_diameter"])
                if self._is_time_dependent:
                    temporal_refinement_level.append(level_result["dt"])
        return (
            np.array(avg_nonlinear_iterations),
            np.array(spatial_refinement_level),
            np.array(temporal_refinement_level),
        )

    def final_l2_error(
        self, list_of_results: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract the final l2 error (aka the error at the last time step) for each
        level."""
        final_l2_errors: list[float] = []
        spatial_refinement_level: list[float] = []
        temporal_refinement_level: list[float] = []
        try:
            end_time: float = self.model_params[0]["time_manager"].time_final
        except:
            end_time = 1.0
        for level_result in list_of_results:
            for time_step_result in level_result["results"]:
                # Only get the results of the last time step.
                if np.isclose(time_step_result["time"], end_time):
                    final_l2_errors.append(time_step_result["l2_error"])
                    # Append refinement levels.
                    spatial_refinement_level.append(level_result["cell_diameter"])
                    if self._is_time_dependent:
                        temporal_refinement_level.append(level_result["dt"])
        return (
            np.array(final_l2_errors),
            np.array(spatial_refinement_level),
            np.array(temporal_refinement_level),
        )


def result_to_dict(result: IsDataclass) -> dict:
    """Transform the ``result`` dataclass to a dict."""
    return {name: getattr(result, name) for name in all_annotations(result.__class__)}


def save_convergence_results(
    analysis: ConvergenceAnalysisExtended,
    results: list,
    level_type: Literal["time", "space"] = "time",
    courant_number: float = 0.0,
    max_iterations: int = 30,
    foldername: str = "results",
    save_data_class: Type[IsDataclass] = SaveDataTPF,
) -> None:
    """Small script save the results of a convergence analysis and plot some graphs.

    In detail, this script takes in an analysis that has run and:
        - Calculates the order of convergence and saves it to
          ``f"order_of_convergence_in_{level_type}.txt"``.
        - Calculates the L2 error for each resolution level and plots the results.
        - Calculates the average number of Newton iterations for each resolution level
          and plots the results.

    Parameters:
        analysis: _description_
        results: _description_
        level_type: _description_. Defaults to ``"time"``.
        courant_number: _description_. Defaults to ``0.0``.
        max_iterations: _description_. Defaults to ``30``.
        foldername: _description_. Defaults to ``"results"``.

    """
    # Order of convergence
    try:
        x_axis: Literal["cell_diameter", "time_step"] = (
            "time_step" if level_type == "time" else "cell_diameter"
        )
        ooc = analysis.order_of_convergence(
            analysis.transform_results_to_classical(
                results, save_data_class=save_data_class
            ),
            x_axis=x_axis,
        )
        logger.info(f"Order of convergence: {ooc}")
        with open(
            os.path.join(foldername, f"order_of_convergence_in_{level_type}.txt"), "w"
        ) as f:
            f.write(f"order of convergence: {ooc}")
    # If no refinement level reaches the last time step, ooc will fail.
    except IndexError:
        logger.info("Skipping order of convergence")

    # L2 errors
    try:
        (
            final_l2_errors,
            cell_diameter_levels,
            time_step_levels,
        ) = analysis.final_l2_error(results)
        xx = time_step_levels if level_type == "time" else cell_diameter_levels
        # Plot final error for each parameter.
        fig = plt.figure()
        plt.plot(
            xx,
            final_l2_errors,
            "xb-",
            label=f"final error ({max_iterations} iter)",
        )
        plt.xlabel(r"$\log_{10}(\Delta t)$")
        plt.ylabel(r"$\log_{10}(\|e\|)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.axvline(x=courant_number, color="b", label=r"$\mathcal{C}$")
        plt.title("Convergence Analysis")
        plt.legend()
        fig.subplots_adjust(left=0.2, bottom=0.2)
        plt.savefig(os.path.join(foldername, f"accuracy_in_{level_type}.png"))
    # Catch exceptions for results without L2 errors.
    except (KeyError, ValueError):
        pass

    # Avg. number of Newton iterations
    try:
        (
            avg_nonlinear_iterations,
            cell_diameter_levels,
            time_step_levels,
        ) = analysis.average_number_of_iterations(results)
        xx = time_step_levels if level_type == "time" else cell_diameter_levels
        fig = plt.figure()
        plt.plot(
            xx,
            avg_nonlinear_iterations,
            "xb-",
            label=f"average Newton iterations",
        )
        plt.xlabel(r"$\log_{10}(\Delta t)$")
        plt.ylabel(r"$n_{iterations}$")
        plt.xscale("log")
        plt.axvline(x=courant_number, color="b", label=r"$\mathcal{C}$")
        plt.title("Convergence analysis")
        plt.legend()
        fig.subplots_adjust(left=0.2, bottom=0.2)
        plt.savefig(
            os.path.join(foldername, f"average_newton_iterations_varying_{x_axis}.png")
        )
    except IndexError:
        pass


def plot_convergence_for_timestep(
    results: list[dict[str, Any]],
    foldername: str,
    filename: str = "convergence",
    refinement_level: int = 0,
    time_step: int = 1,
    additional_keys: Optional[list] = None,
    additional_data: Optional[dict[str, Any]] = None,
):
    """Plot convergence for a given refinement level and time step.

    Parameters:
        results:
        foldername: _description_
        filename: _description_. Defaults to "convergence".
        refinement_level: _description_. Defaults to ``0``.
        time_step: _description_. Defaults to ``1``.
        additional_data: Additional keys to obtain data from the results. By default,
        only the norms of the solution updates are plotted. Defaults to ``None``.

    """
    if additional_keys is None:
        additional_keys = []
    if additional_data is None:
        additional_data = {}
    try:
        solution_norms = results[refinement_level]["results"][time_step - 1][
            "solution_norms"
        ]
    except Exception:
        # Older saves.
        solution_norms = results[refinement_level]["results"][time_step - 1][
            "residuals"
        ]

    xx = np.linspace(0, len(solution_norms) - 1, len(solution_norms))
    plt.plot(xx + 1, solution_norms, label=r"$\|\Delta x\|$")
    for key in additional_keys:
        dataset = results[refinement_level]["results"][time_step - 1][key]
        plt.plot(xx + 1, dataset, label=key)
    for data_name, dataset in additional_data.items():
        plt.plot(xx + 1, dataset, label=data_name)
    plt.xlabel("iteration")
    plt.yscale("log")
    plt.legend()
    plt.savefig(
        os.path.join(
            foldername,
            filename
            + f"_refinement_level_{refinement_level}_time_step_{time_step}.png",
        )
    )
