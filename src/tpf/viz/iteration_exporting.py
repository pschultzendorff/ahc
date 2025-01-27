"""Mixin to export model states at each iteration. 

Functionally mostly copied from
https://github.com/pmgbergen/porepy/blob/develop/tutorials/exporting_models.ipynb
and adjusted for homotopy continuation.

"""

import numpy as np
import porepy as pp
from tpf.models.protocol import DataSavingMixinExtendedProtocol, HCProtocol, TPFProtocol


class IterationExportingMixin(DataSavingMixinExtendedProtocol, HCProtocol, TPFProtocol):

    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_iterations",
            folder_name=self.params["folder_name"],
            export_constants_separately=self.params.get(
                "export_constants_separately", False
            ),
            length_scale=self.units.m,
        )
        # To make sure the nonlinear iteration index does not interfere with the
        # time part, we multiply the latter by the next power of ten above the
        # maximum number of nonlinear iterations. Default value set to 10 in
        # accordance with the default value used in NewtonSolver.
        # We do the same for the hc iteration index if homotopy continuation is used.
        n_1: int = self.params.get("max_iterations", 10)
        p_1: int = np.ceil(np.log10(n_1))
        self.r_1: int = 10**p_1
        if self.uses_hc:
            n_2: int = self.params.get("hc_max_iterations", 10)
            p_2: int = np.ceil(np.log10(n_2))
            self.r_2: int = 10**p_2
            self.r_1 = self.r_1 * self.r_2

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.

        Having a separate exporter for iterations avoids distinguishing between
        iterations and time steps in the regular exporter's history (used for
        export_pvd).

        """
        if self.uses_hc:
            time_step: int = int(
                self.nonlinear_solver_statistics.num_iteration
                + self.r_2 * self.nonlinear_solver_statistics.hc_num_iteration
                + self.r_1 * self.time_manager.time_index
            )
        else:
            time_step = int(
                self.nonlinear_solver_statistics.num_iteration
                + self.r_1 * self.time_manager.time_index
            )

        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(), time_dependent=True, time_step=time_step
        )

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution to
        iterate subdictionary.

        """
        super().after_nonlinear_iteration(nonlinear_increment)
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()

    def after_nonlinear_failure(self) -> None:
        """Save model state after nonlinear failure."""
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()
        super().after_nonlinear_failure()
