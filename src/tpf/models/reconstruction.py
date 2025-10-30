from __future__ import annotations

import itertools
import logging
import typing
from typing import Any, Literal

import numpy as np
import porepy as pp
import scipy.sparse as sps
from numba import njit
from porepy.viz.exporter import DataInput

from tpf.models.flow_and_transport import (
    DataSavingTPF,
    SolutionStrategyTPF,
    TwoPhaseFlow,
)
from tpf.models.protocol import ReconstructionProtocol, TPFProtocol
from tpf.numerics.quadrature import (
    GaussLegendreQuadrature1D,
    Integral,
    TriangleQuadrature,
    get_quadpy_elements,
)
from tpf.utils.constants_and_typing import (
    COMPLEMENTARY_PRESSURE,
    GLOBAL_PRESSURE,
    PRESSURE_KEY,
)

logger = logging.getLogger(__name__)


class GlobalPressureMixin(TPFProtocol):
    def setup_glob_compl_pressure(self) -> None:
        global_pressure_constants: dict[str, Any] = self.params.get(
            "global_pressure_constants", {}
        )
        self.quadrature_1d = GaussLegendreQuadrature1D(
            global_pressure_constants.get("quadrature_degree", 10)
        )

        # Do not start evaluation at :math:`\hat{s}_w = 0`, where the derivative
        # may be :math:`\infty` (in case the capillary pressure is not limited). Instead
        # add an epsilon.
        self.s_interpol_vals: np.ndarray = np.linspace(
            self.wetting.residual_saturation + self.wetting.saturation_epsilon,
            1
            - self.nonwetting.residual_saturation
            - self.nonwetting.saturation_epsilon,
            global_pressure_constants.get("interpolation_degree", 100),
        )
        self.calc_pressure_interpolants()

    def calc_pressure_interpolants(self) -> None:
        """Calculate interpolants values for the global and complementary pressure."""
        global_pressure_integral_parts: np.ndarray = self.global_pressure_integral_part(
            self.s_interpol_vals[:-1], self.s_interpol_vals[1:]
        )
        self.global_pressure_interpol_vals: np.ndarray = np.cumsum(
            global_pressure_integral_parts
        )
        compl_pressure_integral_parts: np.ndarray = (
            self.complementary_pressure_integral_part(
                self.s_interpol_vals[:-1], self.s_interpol_vals[1:]
            )
        )
        self.compl_pressure_interpol_vals: np.ndarray = np.cumsum(
            compl_pressure_integral_parts
        )
        # Add zero at the start of the arrays, s.t. :math:`P(0) = p_w` and
        # :math:`Q(0) = 0`.
        self.global_pressure_interpol_vals = np.insert(
            self.global_pressure_interpol_vals, 0, 0.0
        )
        self.compl_pressure_interpol_vals = np.insert(
            self.compl_pressure_interpol_vals, 0, 0.0
        )

    def eval_glob_compl_pressure(
        self,
        s_w: np.ndarray,
        pressure_key: PRESSURE_KEY,
        p_n: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the global or complementary pressure field for the given pressure
        and saturation values.

        Note: This is done via interpolation of the global pressure values depending on
        the saturation values.

        """
        # Limit saturation from below to 0 and from above to 1 in case of negative or >1
        # saturation values during Newton.
        s_w[
            s_w
            > 1
            - self.nonwetting.residual_saturation
            - self.nonwetting.saturation_epsilon
        ] = 1 - self.nonwetting.residual_saturation - self.nonwetting.saturation_epsilon
        s_w[
            s_w < self.wetting.residual_saturation + self.wetting.saturation_epsilon
        ] = self.wetting.residual_saturation + self.wetting.saturation_epsilon

        if pressure_key == GLOBAL_PRESSURE:
            if p_n is None:
                raise ValueError(
                    "Wetting pressure must be provided for global" + " pressure."
                )
            p_vals: np.ndarray = p_n - np.interp(
                s_w, self.s_interpol_vals, self.global_pressure_interpol_vals
            )
        elif pressure_key == COMPLEMENTARY_PRESSURE:
            p_vals = np.interp(
                s_w, self.s_interpol_vals, self.compl_pressure_interpol_vals
            )
        return p_vals

    def eval_glob_compl_pressure_on_domain(
        self,
        time_step_index: int | None = None,
    ) -> None:
        """Evaluate the global and complementary pressure fields on the full domain and
        store it in the data dictionary.

        Parameters:
            time_step_index: Save values at this 'time_step_index' in the data
                dictionary, in addition to the current 'iterate_index'. During
                initialization this must be 0. Default is None.

        """
        for pressure_key in [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE]:
            logger.info(f"Evaluating {pressure_key}.")
            p_n: np.ndarray = self.equation_system.get_variable_values(
                [self.nonwetting.p], iterate_index=0
            )
            s_w: np.ndarray = self.equation_system.get_variable_values(
                [self.wetting.s], iterate_index=0
            )
            p_vals: np.ndarray = self.eval_glob_compl_pressure(
                s_w,
                # mypy cannot infer that pressure_key is of type PRESSURE_KEY here.
                pressure_key,  # type: ignore
                p_n=p_n,
            )
            pp.set_solution_values(
                name=pressure_key,
                values=p_vals,
                data=self.g_data,
                time_step_index=time_step_index,
                iterate_index=0,
            )

    def global_pressure_integral_part(
        self, s_0: np.ndarray, s_1: np.ndarray
    ) -> np.ndarray:
        r"""Compute the integral in the global pressure formula for given integral
        boundaries.

        Note: The evaluation works by transforming ``s`` to
            :class:`~porepy.ad.DenseArray`, calling
            :meth:`CapillaryPressure.cap_press_deriv` and
            :meth:`RelativePermeability.rel_perm` and then evaluating the resulting
            :class:`~porepy.ad.Operator`. This is probably quite inefficient, however
            this is only done once at the start of each simulation. During the
            simulation the global pressure is interpolated.

        Cellwise it holds:
        .. math::
            result = \int_{s_0}^{s_1} \frac{\lambda_n}{\lambda_t} p'_c ds.

        Parameters:
            s_0: ``shape=(num_elements,)`` Lower integral boundaries.
            s_1: ``shape=(num_elements,)`` Upper integral boundaries.

        Returns:
            Global pressure values.

        """
        intervals = np.linspace(s_0, s_1, 2)[..., None]

        def integrand(s: np.ndarray) -> np.ndarray:
            """Note: We cannot use ``two_phase_flow.DarcyFluxes.phase_mobility`` and
            ``two_phase_flow.DarcyFluxes.total_mobility`` here, because they require
            upwinding.

            """
            w_mobility: np.ndarray = (
                self.rel_perm_np(s, self.wetting) / self.wetting.viscosity
            )
            n_mobility: np.ndarray = (
                self.rel_perm_np(s, self.nonwetting) / self.nonwetting.viscosity
            )
            t_mobility: np.ndarray = w_mobility + n_mobility
            return w_mobility / t_mobility * self.cap_press_deriv_np(s)

        integral: Integral = self.quadrature_1d.integrate(integrand, intervals)
        return integral.elementwise.squeeze()

    def complementary_pressure_integral_part(
        self, s_0: np.ndarray, s_1: np.ndarray
    ) -> np.ndarray:
        r"""Compute complementary pressure from the rel. perm. and capillary pressure
        functions.

        Note: The evaluation works by transforming ``p`` and ``s`` to
            :class:`~porepy.ad.DenseArray`, calling
            :meth:`CapillaryPressure.cap_press_deriv` and
            :meth:`RelativePermeability.rel_perm` and then evaluating the resulting
            :class:`~porepy.ad.Operator`. This is probably quite inefficient, however
            this is only done once at the start of each simulation. During the
            simulation the complementary pressure is interpolated.

        Cellwise it holds:
        .. math::
            result = - \int_{s_0}^{s_1} \frac{\lambda_n \lambda_w}{\lambda_t} p'_c ds.

        Parameters:
            s_0: ``shape=(num_elements,)`` Lower integral boundaries.
            s_1: ``shape=(num_elements,)`` Upper integral boundaries.

        Returns:
            complementary pressure values.

        """
        intervals = np.linspace(s_0, s_1, 2)[..., None]

        def integrand(s: np.ndarray) -> np.ndarray:
            """Note: We cannot use ``two_phase_flow.DarcyFluxes.phase_mobility`` and
            ``two_phase_flow.DarcyFluxes.total_mobility`` here, because they require
            upwinding.

            """
            w_mobility: np.ndarray = (
                self.rel_perm_np(s, self.wetting) / self.wetting.viscosity
            )
            n_mobility: np.ndarray = (
                self.rel_perm_np(s, self.nonwetting) / self.nonwetting.viscosity
            )
            t_mobility: np.ndarray = w_mobility + n_mobility
            return w_mobility * n_mobility / t_mobility * self.cap_press_deriv_np(s)

        integral: Integral = self.quadrature_1d.integrate(integrand, intervals)
        return -1 * integral.elementwise.squeeze()

    def set_boundary_pressures(self) -> None:
        """Set boundary pressures for the global and complementary pressure fields."""
        # We assume that boundaries are either no-flow Neumann or outflow Dirichlet. In
        # both cases no capillary pressure distribution from the boundaries goes into
        # the global and complementary pressures. At the boundary, the global pressure
        # equals the non-wetting pressure, while the complementary pressure is zero .
        pressure_glob_bc_dir = self.bc_dirichlet_pressure_values(
            self.g, self.nonwetting
        )
        pressure_compl_bc_dir = np.zeros_like(pressure_glob_bc_dir)

        # TODO See change above. If we loop over boundaries, we do not need to do
        # this next construction.
        bg, bg_data = self.mdg.boundaries(return_data=True)[0]
        for pressure_key, values in zip(
            [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE],
            [pressure_glob_bc_dir, pressure_compl_bc_dir],
        ):
            # Boundary values are constant in time. Store them both at the time step and
            # iterate index 0.
            pp.set_solution_values(
                name=pressure_key,
                values=bg.projection() @ values,
                data=bg_data,
                time_step_index=0,
                iterate_index=0,
            )


class PressureReconstructionMixin(TPFProtocol):
    """Code and method are copied from Valera et al. (2024)."""

    def setup_pressure_reconstruction(self) -> None:
        self.quadpy_elements: np.ndarray = get_quadpy_elements(self.g)
        self.quadrature_reconstruction_degree: int = 4
        self.quadrature_reconstruction = TriangleQuadrature(
            degree=self.quadrature_reconstruction_degree
        )
        # Precompute mappings for pressure reconstruction. As the grid is fixed, these
        # are constant during simulation.
        self.cell_faces_map = sps.find(self.g.cell_faces.T)[1]
        self.cell_nodes_map = sps.find(self.g.cell_nodes().T)[1]
        self.faces_of_cell = self.cell_faces_map.reshape(
            self.g.num_cells, self.g.dim + 1
        )
        self.nodes_of_cell = self.cell_nodes_map.reshape(
            self.g.num_cells, self.g.dim + 1
        )

    def postprocess_pressure_vohralik(
        self,
        pressure_key: PRESSURE_KEY,
        flux_specifier: str = "",
        prepare_simulation: bool = False,
    ) -> None:
        """Postprocess pressure into elementwise P2 polynomials.

        Parameters:
            pressure_key: Name of the pressure field to be reconstructed.
            flux_specifier: Specify the name of the flux field used to post-process the
                pressure. Used, e.g., in homotopy continuation, where the fluxes used
                are ``f"{flux_name}_flux_wrt_goal_rel_perm_RT0_coeffs"`` instead of
                ``f"{flux_name}_flux_RT0_coeffs"``.
                Gets appended between ``f"{flux_name}_flux"`` and ``"_RT0_coeffs"``,
                i.e., the values ``f"{flux_name}_flux{flux_specifier}_RT0_coeffs"`` in
                the data dir are accessed.
            prepare_simulation: Set to True if called in :meth:`prepare_simulation`.
                Stores zero coefficients for all pressures. Stores values additionally
                for the time step.

        Returns:
            None

        """
        logger.info(f"Reconstructing {pressure_key} pressure.")

        if prepare_simulation:
            time_step_index: int | None = 0
            # Assume equilibrium with constant pressure values at initial conditions.
            pressure_postprocessed_coeffs = np.zeros((self.g.num_cells, 6))
        else:
            time_step_index = None

            # Retrieve finite volume cell-centered pressures.
            p_cc = pp.get_solution_values(pressure_key, self.g_data, iterate_index=0)
            assert p_cc.size == self.g.num_cells

            # Retrieve RT0 flux coefficients.
            if pressure_key == GLOBAL_PRESSURE:
                coeffs_flux: np.ndarray = pp.get_solution_values(
                    f"total_by_t_mobility_flux{flux_specifier}_RT0_coeffs",
                    self.g_data,
                    iterate_index=0,
                )
            elif pressure_key == COMPLEMENTARY_PRESSURE:
                coeffs_flux: np.ndarray = pp.get_solution_values(
                    f"total_times_fractional_flow_flux{flux_specifier}_RT0_coeffs",
                    self.g_data,
                    iterate_index=0,
                ) - pp.get_solution_values(
                    f"wetting_from_ff_flux{flux_specifier}_RT0_coeffs",
                    self.g_data,
                    iterate_index=0,
                )
            # Multiply by inverse of the permeability and total mobility to obtain
            # pressure potential.
            perm: np.ndarray = self.g_data[pp.PARAMETERS][self.flux_key][
                "second_order_tensor"
            ].values

            pressure_postprocessed_coeffs = compute_pressure_coeffs(
                self.g.num_cells, self.g.dim, perm, coeffs_flux
            )

            # To obtain the constant c_5, we solve  c_5 = p_h - 1/|K| (gamma(x, y), 1)_K,
            # where s(x, y) = gamma(x, y) + c_5.
            # Decorating this with @njit, it gets recompiled at each nonlinear iteration,
            # which makes the whole process super slow. For now, we turn this off.
            # @njit
            def integrand(x):
                int_0 = pressure_postprocessed_coeffs[:, 0][None, ...] * x[..., 0] ** 2
                int_1 = (
                    pressure_postprocessed_coeffs[:, 1][None, ...]
                    * x[..., 0]
                    * x[..., 1]
                )
                int_2 = pressure_postprocessed_coeffs[:, 2][None, ...] * x[..., 0]
                int_3 = pressure_postprocessed_coeffs[:, 3][None, ...] * x[..., 1] ** 2
                int_4 = pressure_postprocessed_coeffs[:, 4][None, ...] * x[..., 1]
                return int_0 + int_1 + int_2 + int_3 + int_4

            integral: Integral = self.quadrature_reconstruction.integrate(
                integrand,
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )

            # Now, we can compute the constant C, one per cell.
            pressure_postprocessed_coeffs[:, 5] = (
                p_cc - integral.elementwise.squeeze() / self.g.cell_volumes
            )

        # Store post-processed but not reconstructed pressure at the nodes.
        pp.set_solution_values(
            f"{pressure_key}_postprocessed_coeffs",
            pressure_postprocessed_coeffs,
            self.g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )

    def reconstruct_pressure_vohralik(
        self,
        pressure_key: PRESSURE_KEY,
        prepare_simulation: bool = False,
    ) -> None:
        r"""Reconstruct pressures in :math:`H^1_0(\Omega)` by applying the Oswald
        interpolator.

        Parameters:
            pressure_key: Name of the pressure field to be reconstructed.
            prepare_simulation: Set to True if called in :meth:`prepare_simulation`.
                Stores zero coefficients for all pressures. Stores values additionally
                for the time step.

        Returns:
            None

        """
        if prepare_simulation:
            time_step_index: int | None = 0
            # Assume equilibrium with constant pressure values at initial conditions.
            pressure_reconstructed_coeffs = np.zeros((self.g.num_cells, 6))

        else:
            time_step_index = None

            bg, bg_data = self.mdg.boundaries(return_data=True)[0]

            pressure_postprocessed_coeffs: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_postprocessed_coeffs",
                self.g_data,
                iterate_index=0,
            )

            # Sanity check
            if not pressure_postprocessed_coeffs.shape == (self.g.num_cells, 6):
                raise ValueError("Wrong shape of P2 polynomial.")

            # Abbreviations
            dim = self.g.dim
            nn = self.g.num_nodes
            nf = self.g.num_faces
            nc = self.g.num_cells

            # Treatment of the nodes
            # Evaluate post-processed pressure at the nodes
            nodes_p = np.zeros([nc, 3])
            nx = self.g.nodes[0][self.nodes_of_cell]  # local node x-coordinates
            ny = self.g.nodes[1][self.nodes_of_cell]  # local node y-coordinates

            # Compute node pressures
            for col in range(dim + 1):
                nodes_p[:, col] = (
                    pressure_postprocessed_coeffs[:, 0] * nx[:, col] ** 2  # c0x^2
                    + pressure_postprocessed_coeffs[:, 1]
                    * nx[:, col]
                    * ny[:, col]  # c1xy
                    + pressure_postprocessed_coeffs[:, 2] * nx[:, col]  # c2x
                    + pressure_postprocessed_coeffs[:, 3] * ny[:, col] ** 2  # c3y^2
                    + pressure_postprocessed_coeffs[:, 4] * ny[:, col]  # c4y
                    + pressure_postprocessed_coeffs[:, 5]  # c5
                )

            # Average nodal pressure
            node_cardinality = np.bincount(self.cell_nodes_map)
            node_pressure = np.zeros(nn)
            for col in range(dim + 1):
                node_pressure += np.bincount(
                    self.nodes_of_cell[:, col], weights=nodes_p[:, col], minlength=nn
                )
            node_pressure /= node_cardinality

            # Treatment of the faces
            # Evaluate post-processed pressure at the face-centers
            faces_p = np.zeros([nc, 3])
            fx = self.g.face_centers[0][
                self.faces_of_cell
            ]  # local face-center x-coordinates
            fy = self.g.face_centers[1][
                self.faces_of_cell
            ]  # local face-center y-coordinates

            for col in range(3):
                faces_p[:, col] = (
                    pressure_postprocessed_coeffs[:, 0] * fx[:, col] ** 2  # c0x^2
                    + pressure_postprocessed_coeffs[:, 1]
                    * fx[:, col]
                    * fy[:, col]  # c1xy
                    + pressure_postprocessed_coeffs[:, 2] * fx[:, col]  # c2x
                    + pressure_postprocessed_coeffs[:, 3] * fy[:, col] ** 2  # c3y^2
                    + pressure_postprocessed_coeffs[:, 4] * fy[:, col]  # c4x
                    + pressure_postprocessed_coeffs[:, 5]  # c5
                )

            # Average face pressure
            face_cardinality = np.bincount(self.cell_faces_map)
            face_pressure = np.zeros(nf)
            for col in range(3):
                face_pressure += np.bincount(
                    self.faces_of_cell[:, col], weights=faces_p[:, col], minlength=nf
                )
            face_pressure /= face_cardinality

            # Treatment of the boundary points.
            bc: pp.BoundaryCondition = self.g_data[pp.PARAMETERS][self.flux_key]["bc"]

            external_dirichlet_boundary = bc.is_dir
            bc_pressure = bg_data[pp.ITERATE_SOLUTIONS][pressure_key][0]
            bg_dir_filter: np.ndarray = (bg.projection() @ bc.is_dir) == 1
            face_pressure[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]

            # Boundary values at the nodes.
            face_vec = np.zeros(nf)
            face_vec[external_dirichlet_boundary] = 1
            num_dir_face_of_node = self.g.face_nodes * face_vec
            is_dir_node = num_dir_face_of_node > 0
            face_vec *= 0
            face_vec[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]
            node_val_dir = self.g.face_nodes * face_vec
            node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
            node_pressure[is_dir_node] = node_val_dir[is_dir_node]

            # # Prepare for exporting.
            point_val = np.column_stack(
                [node_pressure[self.nodes_of_cell], face_pressure[self.faces_of_cell]]
            )
            point_coo = np.empty([dim, nc, 6])
            point_coo[0] = np.column_stack([nx, fx])
            point_coo[1] = np.column_stack([ny, fy])

            # Solve local systems of equation to obtain the coefficients of the
            # reconstructed pressure from the points coordinates and values.
            # TODO Is there an easier way to do this WHILE reconstructing the pressure?
            # TODO Store these at the end of each time step.
            A_elements: np.ndarray = np.stack(
                [
                    point_coo[0] ** 2,  # x^2
                    point_coo[0] * point_coo[1],  # xy
                    point_coo[0],  # x
                    point_coo[1] ** 2,  # y^2
                    point_coo[1],  # y
                    np.ones_like(point_coo[0]),  # constant
                ],
                -1,
            )
            pressure_reconstructed_coeffs = linalg_solve_batch(A_elements, point_val)

        # Store in data dictionary.
        pp.set_solution_values(
            f"{pressure_key}_reconstructed_coeffs",
            pressure_reconstructed_coeffs,
            self.g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )


class EquilibratedFluxMixin(TPFProtocol):
    """Methods to equilibrate fluxes during the Newton iteration.

    Note: If the grid is updated during the simulation, the opposite side nodes and sign
    normals have to be updated as well. This is not supported by the current
    implementation.

    """

    def setup_flux_equilibration(self) -> None:
        """Calculate opposite side nodes, cell faces, and sign normals for the grid."""
        opp_nodes_cell: np.ndarray = get_opposite_side_nodes(self.g)
        self.opp_nodes_coor_cell: np.ndarray = self.g.nodes[:, opp_nodes_cell]
        cell_faces_map = sps.find(self.g.cell_faces.T)[1]
        self.faces_cell = cell_faces_map.reshape(self.g.num_cells, self.g.dim + 1)
        self.sign_normals: np.ndarray = get_sign_normals(self.g)

    def equilibrate_flux_during_Newton(
        self,
        flux_name: Literal["total", "wetting_from_ff"],
        nonlinear_increment: np.ndarray | None = None,
    ) -> None:
        """Equilibrate an approximate flux solution at a given Newton iteration.

        We assume the following sub-dictionaries to be present in the data dictionary:
            ``iterate_dictionary``, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        We assume the following entries to be present in ``iterate_dictionary` at
        ``iterate_index`` 1!!!
            - ``{flux_name}_flux``, storing the flux value from the previous nonlinear
              iteration.
            - ``{flux_name}_flux_jacobian``, storing the Jacobian of the flux value from
              the previous nonlinear iteration.

        The following entries in ``iterate_dictionary` will be updated:
            - ``{flux_name}_flux_equilibrated``, storing the equilibrated flux.

        Parameters:
            flux_field: Name flux field to be equilibrated.
            nonlinear_increment: Nonlinear increment of the primary valuables. If
                already constructed during Newton, we can save some time instead of
                computing it again. **Increment HAS to be in the order
                ``self.primary_saturation_var``, ``self.primary_pressure_var``.**
                Default is ``None``.

        Returns:
            None

        """
        if self._nl_appleyard_chopping and nonlinear_increment is None:
            raise ValueError(
                "The non-chopped nonlinear increment vector has to be provided when"
                + " Newton is run with Appleyard chopping."
            )
        # The functions ``DarcyFluxes.total_flux`` and ``DarcyFluxes.phase_flux``
        # incorporate bc values, hence ``d[f"{flux_name}_flux"]`` includes bc
        # values when set by ``eval_additional_vars`` and hence we do not need to care
        # about bc values here.
        logger.info(f"Equilibrating {flux_name} flux.")

        val: np.ndarray = pp.get_solution_values(
            f"{flux_name}_flux", self.g_data, iterate_index=1
        )
        jac: np.ndarray = pp.get_solution_values(
            f"{flux_name}_flux_jacobian", self.g_data, iterate_index=1
        )

        if nonlinear_increment is None:
            # NOTE The variables are retrieved in the same order as in the Jacobian
            # construction.
            # NOTE This requires the variables to be shifted at each nonlinear
            # iteration. By default, this happens in
            # :meth:`SolutionStrategyTPF.after_nonlinear_iteration`
            var_val: np.ndarray = self.equation_system.get_variable_values(
                [self.primary_saturation_var, self.primary_pressure_var],
                iterate_index=1,
            )
            var_val_new: np.ndarray = self.equation_system.get_variable_values(
                [self.primary_saturation_var, self.primary_pressure_var],
                iterate_index=0,
            )
            nonlinear_increment = var_val_new - var_val

        equilibrated_flux: np.ndarray = val + jac @ nonlinear_increment

        # On Neumann boundaries, the equilibrated flux is set to the boundary value.
        # is_dir: np.ndarray = self.g_data[pp.PARAMETERS][self.flux_key]["bc"].is_dir
        # equilibrated_flux[is_dir] = 0
        pp.set_solution_values(
            f"{flux_name}_flux_equilibrated",
            equilibrated_flux,
            self.g_data,
            iterate_index=0,
        )

    def extend_fv_fluxes(
        self,
        flux_name: Literal[
            "total",
            "wetting_from_ff",
            "total_by_t_mobility",
            "total_times_fractional_flow",
        ],
        flux_specifier: str = "",
        prepare_simulation: bool = False,
    ) -> None:
        """Extend (eqilibrated or non-equilibrated) flux using RT0 basis functions.

        Note:
            The data dictionary of each node of the grid bucket will be updated with the
            field d["estimates"]["recon_sd_flux"], a nd-array of shape
            (g.num_cells x (g.dim+1)) containing the coefficients of the reconstructed
            flux for each element. Each column corresponds to the coefficient a, b, c,
            and so on.

            The coefficients satisfy the following velocity fields depending on the
            dimensionality of the problem:

                q = ax + b                          (for 1d),
                q = (ax + b, ay + c)^T              (for 2d),
                q = (ax + b, ay + c, az + d)^T      (for 3d).

            The reconstructed velocity field inside an element K is given by:

                q = sum_{j=1}^{g.dim+1} q_j psi_j,

            where psi_j are the global basis functions defined on each face,
            and q_j are the normal fluxes.

            The global basis takes the form

                psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i)^T                   (for 1d),
                psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i)^T          (for 2d),
                psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i, z - z_i)^T (for 3d),

            where s(normal_j) is the sign of the normal vector,|K| is the Lebesgue
            measure of the element K, and (x_i, y_i, z_i) are the coordinates of the
            opposite side nodes to the face j. The function s(normal_j) = 1 if the
            signs of the local and global normals are the same, and -1 otherwise.

        Parameters:
            mdg: pp.MixedDimensionalGrid
                Mixed-dimensional grid for the problem.
            flux_name: Name of the flux field to be extended.
            flux_specifier: E.g., "_equilibrated" or "_wrt_goal_rel_perm".
            prepare_simulation: Set to True if called in :meth:`prepare_simulation`.
                Stores values additionally for the time step.


        """
        if self.params["grid_type"] != "simplex":
            raise ValueError("Not implemented for non-simplex grids.")
        logger.info(f"Extending {flux_name}{flux_specifier} flux to RT0 functions.")

        if prepare_simulation:
            time_step_index: int | None = 0
        else:
            time_step_index = None

        # Cell-basis arrays
        faces_cell = self.faces_cell
        opp_nodes_coor_cell = self.opp_nodes_coor_cell
        sign_normals_cell = self.sign_normals
        vol_cell = self.g.cell_volumes

        # Retrieve finite volume fluxes
        flux = pp.get_solution_values(
            f"{flux_name}_flux{flux_specifier}", self.g_data, iterate_index=0
        )

        # Perform actual reconstruction and obtain coefficients
        coeffs = np.empty([self.g.num_cells, self.g.dim + 1])
        alpha = 1 / (self.g.dim * vol_cell)
        coeffs[:, 0] = alpha * np.sum(sign_normals_cell * flux[faces_cell], axis=1)
        for dim in range(self.g.dim):
            coeffs[:, dim + 1] = -alpha * np.sum(
                (sign_normals_cell * flux[faces_cell] * opp_nodes_coor_cell[dim]),
                axis=1,
            )

        # Store coefficients in the data dictionary.
        pp.set_solution_values(
            f"{flux_name}_flux{flux_specifier}_RT0_coeffs",
            coeffs,
            self.g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )

    def equilibrated_flux_mismatch(self) -> dict[str, float]:
        r"""Calculate mismatch of the equilibrated flux from being in :math:`H(div)` and
        being mass conservative.

        Check :meth:`tpf.models.two_phase_flow.EquationsTPF.set_equations` for
        details.

        TODO Make this more efficient, by updating the arrays at each iteration and
        adding the equations to the equation system.
        TODO Calculate the elementwise mismatch.
        TODO This copies 90% of the code from ``set_equations``. Make ``set_equations``
        more flexible and call (with the reconstructed flux) to avoid this.
        TODO The equation does not need to be redefined at each Newton iteration.
        Instead, save it to the equation system and just reevaluate.

        """
        # Spatial discretization operators.
        div = pp.ad.Divergence([self.g])

        # Time derivatives.
        dt = pp.ad.Scalar(self.time_manager.dt)
        dt_s: pp.ad.Operator = pp.ad.time_derivatives.dt(self.wetting.s, dt)

        # Ad source.
        source_ad_w = pp.ad.DenseArray(self.phase_fluid_source(self.g, self.wetting))
        source_ad_t = pp.ad.DenseArray(self.total_fluid_source(self.g))

        # Ad parameters.
        porosity_ad = pp.ad.DenseArray(self.porosity(self.g))

        # Ad equations
        if self.formulation == "fractional_flow":
            # Note, that for ``flux_t``, the total mobility is already included.
            flux_t = pp.ad.DenseArray(
                pp.get_solution_values(
                    "total_flux_equilibrated", self.g_data, iterate_index=0
                )
            )
            flux_w = pp.ad.DenseArray(
                pp.get_solution_values(
                    "wetting_from_ff_flux_equilibrated", self.g_data, iterate_index=0
                )
            )

            flow_equation = div @ flux_t - source_ad_t
            transport_equation = (
                porosity_ad * (self.volume_integral(dt_s, [self.g], 1))
                + div @ flux_w
                - source_ad_w
            )

        flow_equation.set_name("Flow reconstruction mismatch")
        transport_equation.set_name("Transport reconstruction wetting flow mismatch")
        # Mypy complains about wrong types, because ``*.value(*)`` is not necessarily an
        # ``np.ndarray``. We ignore this.
        flow_equation_mismatch: float = np.sum(
            np.abs(flow_equation.value(self.equation_system))  # type: ignore
        )
        transport_equation_mismatch: float = np.sum(
            np.abs(transport_equation.value(self.equation_system))  # type: ignore
        )
        logger.info(f"Flow equation mismatch {flow_equation_mismatch}")
        logger.info(
            f"Transport equation mismatch wetting flow {transport_equation_mismatch}"
        )
        return {
            "flow": flow_equation_mismatch,
            "transport": transport_equation_mismatch,
        }


class EquationsRecMixin(TPFProtocol):
    def set_equations(self) -> None:
        """Set additional equations needed for reconstructions.

        The following equations are set:
        - Total flux
        - Wetting flux
        - Total flux divided by total mobility
        - Total flux times fractional flow
        The former two are required to equilibrate fluxes while the latter two are
        required to post-process pressures into elementwise P2 polynomials.

        """
        super().set_equations()

        total_flux: pp.ad.Operator = self.total_flux(self.g)
        wetting_flux_from_fractional_flow: pp.ad.Operator = (
            self.wetting_flux_from_fractional_flow(self.g)
        )

        self.total_flux_eq = "Total flux"
        self.wetting_flux_from_ff_eq = "Wetting flux from fractional flow"

        total_flux.set_name(self.total_flux_eq)
        wetting_flux_from_fractional_flow.set_name(self.wetting_flux_from_ff_eq)
        self.equation_system.set_equation(total_flux, [self.g], {"cells": 1})
        self.equation_system.set_equation(
            wetting_flux_from_fractional_flow, [self.g], {"cells": 1}
        )

        total_flux_by_total_mobility: pp.ad.Operator = total_flux / self.total_mobility(
            self.g
        )
        total_flux_times_fractional_flow: pp.ad.Operator = (
            total_flux_by_total_mobility * self.phase_mobility(self.g, self.wetting)
        )

        self.total_flux_by_total_mobility_eq = "Total flux by total mobility"
        self.total_flux_times_fractional_flow_eq = "Total flux times fractional flow"

        total_flux_by_total_mobility.set_name(self.total_flux_by_total_mobility_eq)
        total_flux_times_fractional_flow.set_name(
            self.total_flux_times_fractional_flow_eq
        )

        self.equation_system.set_equation(
            total_flux_by_total_mobility, [self.g], {"faces": 1}
        )
        self.equation_system.set_equation(
            total_flux_times_fractional_flow, [self.g], {"faces": 1}
        )


# This could also be a mixin, but by subclassing ``SolutionStrategyTPF``, we avoid
# having to pay attention to the order of the different solution strategy classes.
# ``TPFProtocol`` and ``pp.SolutionStrategy`` define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class SolutionStrategyRec(  # type: ignore
    ReconstructionProtocol,
    SolutionStrategyTPF,
):
    @property
    @typing.override
    def iterate_indices(self) -> np.ndarray:
        """Indices for storing iterate solutions. To equilibrate the fluxes, the
        previous iterate has to be stored."""
        return np.array([0, 1])

    @typing.override
    def prepare_simulation(self) -> None:
        """Set up reconstructions after setting up the base simulation."""
        super().prepare_simulation()

        # Setup reconstructions
        self.setup_glob_compl_pressure()
        self.setup_pressure_reconstruction()
        self.setup_flux_equilibration()

        # Initialize fluxes, global, and complementary pressure from the initial data of
        # the problem.

        # Flux equilibration requires the fluxes at two different iterates, hence we
        # initialize both the current and previous iterate with the same value. Values
        # from 'iterate_index' 0 in the data dictionary are shifted to 'iterate_index' 1
        # in the second call.
        self.eval_val_and_jac_fluxes()
        self.eval_val_and_jac_fluxes()

        # Initalize P0 pressures and fluxes to construct piecewise P2 pressures.
        # Initialize both at current iterate index and time step index 0.
        self.eval_additional_vars(time_step_index=0)
        self.set_boundary_pressures()
        self.postprocess_solution(
            np.zeros(self.g.num_cells * 2), prepare_simulation=True
        )

    @typing.override
    def before_nonlinear_iteration(self) -> None:
        """

        Note: In comparison to the base implementation, the pressure potentials are not
        updated here, but in :meth:`after_nonlinear_iteration`. This is done s.t. the
        total and wetting flux needed for flux equilibration and pressure reconstruction
        are evaluated with the correct upwind values without too much hassle. Similarly,
        the operators are rediscretized after the nonlinear iteration.

        """
        pass

    @typing.override
    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        super().after_nonlinear_iteration(nonlinear_increment)
        # Update pressure potentials.
        self.set_discretization_parameters()
        # Re-discretize nonlinear terms. If none have been added to
        # self.nonlinear_discretizations, this will be a no-op.
        self.rediscretize()

        # Now, the fluxes can be evaluated with the new upwinded mobilities.
        self.eval_val_and_jac_fluxes()
        self.eval_additional_vars()

        # When Newton is run with Appleyard chopping, the non-chopped nonlinear
        # increment has to be used to equilibrate the fluxes.
        if self._nl_appleyard_chopping or self._nl_enforce_physical_saturation:
            if hasattr(self, "non_chopped_nonlinear_increment"):
                nonlinear_increment = self.non_chopped_nonlinear_increment
            else:
                raise AttributeError(
                    "The non-chopped nonlinear increment vector has to be stored when"
                    + " Newton is run with Appleyard chopping or enforced physical"
                    + " saturations."
                )

        # If Newton diverged, postprocessing might fail because
        # ``linalg_solve_batch`` cannot handle infs or nans.
        try:
            self.postprocess_solution(nonlinear_increment)

        except np.linalg.LinAlgError as e:
            logger.warning(
                "Postprocessing failed, likely due to diverged Newton. "
                + "Skipping postprocessing this iteration."
            )
            logger.warning(e)

    @typing.override
    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        converged, diverged = super().check_convergence(
            nonlinear_increment, residual, reference_residual, nl_params
        )
        equilibrated_flux_mismatch: dict[str, float] = self.equilibrated_flux_mismatch()
        self.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm=None,
            residual_norm=None,
            equilibrated_flux_mismatch=equilibrated_flux_mismatch,
        )
        return converged, diverged

    @typing.override
    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        # Save time step values for pressures, postprocessings, and reconstructions.
        for pressure_key, specifier in itertools.product(
            [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE],
            ["", "_postprocessed_coeffs", "_reconstructed_coeffs"],
        ):
            values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}{specifier}", self.g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}{specifier}",
                values,
                self.g_data,
                time_step_index=0,
            )

    def eval_val_and_jac_fluxes(self) -> None:
        """Evaluate residual and Jacobian of fluxes to be equilibrated.

        Note: This is separated from ``eval_additional_vars`` and called **before** the
        Newton update s.t. the fluxes are evaluated with the current iterate values and
        upwinding direction.

        Populates the `iterate_index` in the data dictionary with:
        - 'total_flux_val', 'total_flux_jac': Total flux value and Jacobian.
        - 'wetting_from_ff_flux_val','wetting_from_ff_flux_jac': Wetting flux value and
            Jacobian.


        """
        for flux_name, flux_eq in zip(
            ["total", "wetting_from_ff"],
            [self.total_flux_eq, self.wetting_flux_from_ff_eq],
        ):
            jac, val = self.equation_system.assemble(
                equations=[flux_eq],
                variables=[self.primary_saturation_var, self.primary_pressure_var],
            )
            # Multiply the values with -1 since ``equation_system.assemble`` returns the
            # negative of the RHS.
            val *= -1

            # For flux equilibration, the values at the previous iteration are required,
            # hence they are shifted.
            pp.shift_solution_values(
                f"{flux_name}_flux",
                self.g_data,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"{flux_name}_flux",
                val,
                self.g_data,
                iterate_index=0,
            )
            pp.shift_solution_values(
                f"{flux_name}_flux_jacobian",
                self.g_data,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            # ``jac`` is an ``sps.spmatrix``, which mypy complains about.
            pp.set_solution_values(
                f"{flux_name}_flux_jacobian",
                jac,  # type: ignore
                self.g_data,
                iterate_index=0,
            )

    def eval_additional_vars(self, time_step_index: int | None = None) -> None:
        """Evaluate additional pressure and flux variables and save in data dictionary
        after each iteration.

        Populates the data dictionary with:
        - 'global_pressure': Global pressure value.
        - 'complementary_pressure': complementary pressure value.
        - '{phase.name}_flux': Flux value for both phases.

        Parameters:
            time_step_index: Save values at this 'time_step_index' in the data
                dictionary, in addition to the current 'iterate_index'. During
                initialization this must be 0. Default is None.

        """
        # Save FV P0 pressures.
        self.eval_glob_compl_pressure_on_domain(time_step_index=time_step_index)

        # Calculate fluxes for pressure post-processing.
        # NOTE Take the negative of the values since ``equation_system.assemble``
        # returns the negative of the RHS.
        total_flux_by_total_mobility: np.ndarray = -self.equation_system.assemble(
            evaluate_jacobian=False, equations=[self.total_flux_by_total_mobility_eq]
        )
        total_flux_times_fractional_flow: np.ndarray = -self.equation_system.assemble(
            evaluate_jacobian=False,
            equations=[self.total_flux_times_fractional_flow_eq],
        )
        pp.set_solution_values(
            "total_by_t_mobility_flux",
            total_flux_by_total_mobility,
            self.g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )
        pp.set_solution_values(
            "total_times_fractional_flow_flux",
            total_flux_times_fractional_flow,
            self.g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )

    def postprocess_solution(
        self, nonlinear_increment: np.ndarray, prepare_simulation: bool = False
    ) -> None:
        """Equilibrate fluxes and reconstruct pressures."""
        for flux_name in ["total", "wetting_from_ff"]:
            # Satisfy mypy.
            flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

            # Extend both the nonequilibrated and equilibrated flux to compare in
            # the flux estimator. The nonequilibrated wetting_from_ff flux is also used
            # in the pressure reconstruction.
            self.extend_fv_fluxes(
                flux_name,
                prepare_simulation=prepare_simulation,
            )

            # Equilibration can only be run during Newton.
            if not prepare_simulation:
                # In ``nonlinear_increment``, the saturation variable comes first, then
                # the pressure variable, just as required by
                # ``equilibrate_flux_during_Newton``.
                self.equilibrate_flux_during_Newton(flux_name, nonlinear_increment)

                self.extend_fv_fluxes(
                    flux_name,
                    flux_specifier="_equilibrated",
                )

        # Extend fluxes needed for pressure reconstruction.
        for flux_name in ["total_by_t_mobility", "total_times_fractional_flow"]:
            # Satisfy mypy.
            flux_name = typing.cast(
                Literal["total_by_t_mobility", "total_times_fractional_flow"],
                flux_name,
            )

            self.extend_fv_fluxes(flux_name, prepare_simulation=prepare_simulation)

        # Reconstruct pressures.
        for pressure_key in [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE]:
            # Satisfy mypy.
            pressure_key = typing.cast(PRESSURE_KEY, pressure_key)
            self.postprocess_pressure_vohralik(
                pressure_key, prepare_simulation=prepare_simulation
            )
            self.reconstruct_pressure_vohralik(
                pressure_key, prepare_simulation=prepare_simulation
            )


def r2c(array: np.ndarray) -> np.ndarray:
    """Reshape a 1d array into a column vector"""
    return array.reshape(-1, 1)


def get_opposite_side_nodes(g: pp.Grid) -> np.ndarray:
    """Computes opposite side nodes for each face of each cell in the grid.

    Parameters:
        g: pp.Grid
            Subdomain grid.

    Returns:
        Opposite nodes with rows representing the cell number and columns
        representing the opposite side node index of the face. The size of the array
        is (g.num_cells x (g.dim + 1)).

    """
    dim = g.dim
    nc = g.num_cells
    nf = g.num_faces

    faces_of_cell = sps.find(g.cell_faces.T)[1].reshape(nc, dim + 1)
    nodes_of_cell = sps.find(g.cell_nodes().T)[1].reshape(nc, dim + 1)
    nodes_of_face = sps.find(g.face_nodes.T)[1].reshape(nf, dim)

    opposite_nodes = np.empty_like(faces_of_cell)
    for cell in range(g.num_cells):
        opposite_nodes[cell] = [
            np.setdiff1d(nodes_of_cell[cell], nodes_of_face[face])[0]
            for face in faces_of_cell[cell]
        ]

    return opposite_nodes


def get_sign_normals(g: pp.Grid) -> np.ndarray:
    """Computes sign of the face normals for each cell of the grid.

    Note:
        We have to take care of the sign of the basis functions. The idea is to create
        an array of signs "sign_normals" that will be multiplying each edge basis
        function for the RT0 extension of fluxes.

        To determine this array, we need the following:
            (1) Compute the local outer normal `lon` vector for each cell.
            (2) For every face of each cell, compare if lon == global normal vector.
                If they're not, then we need to flip the sign of lon for that face

    Parameters:
        g: pp.Grid
            Subdomain grid.

    Returns:
        Sign of the face normal. 1 if the signs of the local and global normals are
        the same, -1 otherwise. The size of the np.ndarray is `g.num_faces`.

    """
    # Faces associated to each cell
    faces_cell = sps.find(g.cell_faces.T)[1].reshape(g.num_cells, g.dim + 1)

    # Face centers coordinates for each face associated to each cell
    face_center_cells = g.face_centers[:, faces_cell]

    # Global normals of the faces per cell
    global_normal_faces_cell = g.face_normals[:, faces_cell]

    # Computing the local outer normals of the faces per cell. To do this, we first
    # assume that n_loc = n_glb, and then we fix the sign. To fix the sign,we compare
    # the length of two vectors, the first vector v1 = face_center - cell_center,
    # and the second vector v2 is a prolongation of v1 in the direction of the
    # normal. If ||v2||<||v1||, then the  normal of the face in question is pointing
    # inwards, and we needed to flip the sign.
    local_normal_faces_cell = global_normal_faces_cell.copy()
    v1 = face_center_cells - g.cell_centers[:, :, np.newaxis]
    v2 = v1 + local_normal_faces_cell * 0.001
    # Checking if ||v2|| < ||v1|| or not
    length_v1 = np.linalg.norm(v1, axis=0)
    length_v2 = np.linalg.norm(v2, axis=0)
    swap_sign = 1 - 2 * (length_v2 < length_v1)
    # Swapping the sign of the local normal vectors
    local_normal_faces_cell *= swap_sign

    # Now that we have the local outer normals. We can check if the local
    # and global normals are pointing in the same direction. To do this
    # we compute length_sum_n = || n_glb + n_loc||. If they're the same, then
    # length_sum_n > 0. Otherwise, they're opposite and length_sum_n \approx 0.
    sum_n = local_normal_faces_cell + global_normal_faces_cell
    length_sum_n = np.linalg.norm(sum_n, axis=0)
    sign_normals = 1 - 2 * (length_sum_n < 1e-8)

    return sign_normals


# The loop could be parallelized with njit(parallel=True) and prange. However, on a
# small grids (<16000 cells) the overhead made :meth:`postprocess_pressure_vohralik`
# slower.
@njit
def compute_pressure_coeffs(
    num_cells: int, dim: int, perm: np.ndarray, coeffs_flux: np.ndarray
) -> np.ndarray:
    """Compute all but the constant coefficient of the post-processed pressure from the
    flux

    """
    s = np.zeros((num_cells, 6))
    # Loop through all cells and compute the nonconstant coefficients.
    for ci in range(num_cells):
        # Local permeability tensor
        K = perm[:dim, :dim, ci]
        Kxx = K[0][0]
        Kxy = K[0][1]
        Kyy = K[1][1]

        # Retrieve components of the RT0 local flux field.
        a = coeffs_flux[ci][0]
        b = coeffs_flux[ci][1]
        c = coeffs_flux[ci][2]

        # Compute components of vector post-processed pressure.
        s[ci][0] = (a * Kyy) / (2 * (Kxy**2 - Kxx * Kyy))  # x^2
        # NOTE If K is a scalar, the following term will vanish.
        s[ci][1] = (a * Kxy) / (Kxx * Kyy - Kxy**2)  # xy
        s[ci][2] = (Kxy * c - Kyy * b) / (Kxx * Kyy - Kxy**2)  # x
        s[ci][3] = (a * Kxx) / (2 * (Kxy**2 - Kxx * Kyy))  # y^2
        s[ci][4] = (Kxx * c - Kxy * b) / (Kxy**2 - Kxx * Kyy)  # y
    return s


# The loop could be parallelized with njit(parallel=True) and prange. However, on a
# small grids (<16000 cells), the overhead made :meth:`reconstruct_pressure_vohralik`
# slower.
@njit
def linalg_solve_batch(A_elements: np.ndarray, point_val: np.ndarray) -> np.ndarray:
    ret: np.ndarray = np.empty_like(point_val)
    for i, (A, b) in enumerate(zip(A_elements, point_val)):
        ret[i] = np.linalg.solve(A, b)
    return ret


class DataSavingRec(DataSavingTPF):
    def _data_to_export(
        self, time_step_index: int | None = None, iterate_index: int | None = None
    ) -> list[DataInput]:
        """Append global and complementary pressures to the exported data."""
        data: list[DataInput] = super()._data_to_export(
            time_step_index=time_step_index,
            iterate_index=iterate_index,
        )
        for pressure_key in [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE]:
            # Before simulation, the estimates won't be set yet, due to the order of
            # calls in :meth:`prepare_simulation`. However, after the first time step,
            # :attr:`time_manager.time_step_index` won't be updated yet. Checking for
            # all of this is quite convoluted. Instead we just use try-except.
            try:
                data.append(
                    (
                        self.g,
                        pressure_key,
                        pp.get_solution_values(
                            pressure_key,
                            self.g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
            except KeyError:
                pass
        return data


# ``TPFProtocol`` and ``pp.SolutionStrategy`` define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class TwoPhaseFlowReconstruction(  # type: ignore
    GlobalPressureMixin,
    PressureReconstructionMixin,
    EquilibratedFluxMixin,
    EquationsRecMixin,
    SolutionStrategyRec,
    DataSavingRec,
    TwoPhaseFlow,
): ...  # type: ignore
