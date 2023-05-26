import json
import logging
from copy import deepcopy
from typing import ClassVar, Literal, Optional, Protocol, Type, Union

import numpy as np
import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.utils.txt_io import TxtData, export_data_to_txt
from scipy import stats

from tpf_lab.visualization.diagnostics import BuckleyLeverettSaveData

logger = logging.getLogger("__name__")


class IsDataclass(Protocol):
    """For typehinting a ``dataclass``.

    Cf.
    https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass.

    """

    __dataclass_fields__: ClassVar[dict]


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
        for params in self.model_params:
            params["file_name"] = (
                f"cell_diameter_{params['meshing_arguments']['cell_size']}"
                + f"_dt_{params['time_manager'].dt}"
            )

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
            if not setup._is_time_dependent():
                # Run stationary model
                try:
                    pp.run_stationary_model(setup, deepcopy(self.model_params[level]))
                except Exception as e:
                    # The model does not converge.
                    logger.info(e)
            else:
                # Run time-dependent model
                try:
                    pp.run_time_dependent_model(
                        setup, deepcopy(self.model_params[level])
                    )
                except Exception as e:
                    # The model does not converge.
                    logger.info(e)
                # Complement information in results
                level_result["dt"] = setup.time_manager.dt
            # Export the model results.
            setup._export()
            level_result["cell_diameter"] = setup.mdg.diameter()
            # Loop over lists of results. Convert into a json compatible list.
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

        # Finally, call the function to write into the json.#
        with open(file_name, "w") as f:
            json.dump(results_to_export, f, indent=2)

    def transform_results_to_classical(
        self,
        list_of_results: list,
        save_data_class: Type[IsDataclass] = BuckleyLeverettSaveData,
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
        """Compute the average number of nonlinear iterations for each level.

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
            if (
                level_result["results"][len(nonlinear_iterations) - 1]["time"]
                >= end_time
            ):
                # Only append results if the last time step was reached.
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
    return {name: getattr(result, name) for name in result.__annotations__}
