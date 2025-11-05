from __future__ import annotations

import itertools
import logging
import typing
from typing import Any, Literal

import numpy as np
import porepy as pp
import scipy.sparse as sps
from numba import njit, prange
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
    TOTAL_FLUX,
    WETTING_FLUX,
)

logger = logging.getLogger(__name__)


class GlobalPressureMixin(TPFProtocol):
    DEFAULT_QUAD_DEGREE_1D = 10
    DEFAULT_INTERP_DEGREE = 100

    def setup_glob_compl_pressure(self) -> None:
        constants: dict[str, Any] = self.params.get("global_pressure_constants", {})

        quadrature_degree = constants.get(
            "quadrature_degree", self.DEFAULT_QUAD_DEGREE_1D
        )
        interp_degree = constants.get(
            "interpolation_degree", self.DEFAULT_INTERP_DEGREE
        )

        self.quadrature_1d = GaussLegendreQuadrature1D(quadrature_degree)

        # Avoid the saturation limits where the capillary derivative may go to infinity.
        s_min = self.wetting.residual_saturation + self.wetting.saturation_epsilon
        s_max = (
            1 - self.nonwetting.residual_saturation - self.nonwetting.saturation_epsilon
        )
        self.s_interpol_vals: np.ndarray = np.linspace(s_min, s_max, interp_degree)

        self.calc_pressure_interpolants()

    def calc_pressure_interpolants(self) -> None:
        """Precompute and store interpolated values for global and complementary
        pressure."""
        s_left, s_right = self.s_interpol_vals[:-1], self.s_interpol_vals[1:]

        # Compute cumulative integrals. Add zero at the start of the arrays, s.t.
        # :math:`P(0) = p_w` and :math:`Q(0) = 0`.
        self.global_pressure_interpol_vals: np.ndarray = np.insert(
            np.cumsum(self.global_pressure_integral(s_left, s_right)), 0, 0.0
        )
        self.compl_pressure_interpol_vals: np.ndarray = np.insert(
            np.cumsum(self.complementary_pressure_integral(s_left, s_right)),
            0,
            0.0,
        )

    def eval_glob_compl_pressure(
        self,
        s_w: np.ndarray,
        pressure_key: PRESSURE_KEY,
        p_n: np.ndarray | None = None,
    ) -> np.ndarray:
        """Interpolate and evaluate global or complementary pressure field.

        Parameters:
            s_w: Wetting phase saturation values.
            pressure_key: Either GLOBAL_PRESSURE or COMPLEMENTARY_PRESSURE.
            p_n: Non-wetting phase pressure (required for global pressure). Default is
                None.

        Returns:
            pressure: Evaluated pressure field.

        """
        # Limit saturation values to :math:`[0 + \epsilon, 1 - \epsilon]`
        s_min = self.wetting.residual_saturation + self.wetting.saturation_epsilon
        s_max = (
            1 - self.nonwetting.residual_saturation - self.nonwetting.saturation_epsilon
        )
        s_clipped = np.clip(s_w, s_min, s_max)

        if pressure_key == GLOBAL_PRESSURE:
            if p_n is None:
                raise ValueError(
                    "Non-wetting pressure `p_n` must be provided for global pressure"
                    + " evaluation."
                )
            interpolated = np.interp(
                s_clipped, self.s_interpol_vals, self.global_pressure_interpol_vals
            )
            return p_n - interpolated

        elif pressure_key == COMPLEMENTARY_PRESSURE:
            return np.interp(
                s_clipped, self.s_interpol_vals, self.compl_pressure_interpol_vals
            )

        raise ValueError(f"Invalid pressure key: {pressure_key}")

    def eval_glob_compl_pressure_on_domain(
        self, time_step_index: int | None = None
    ) -> None:
        """Evaluate and store global and complementary pressures over the entire domain.

        Parameters:
            time_step_index: Store values at 'time_step_index' in addition to
                'iterate_index' 0. During simulation initialization this must be 0.
                Default is None.

        """
        p_n = self.equation_system.get_variable_values(
            [self.nonwetting.p], iterate_index=0
        )
        s_w = self.equation_system.get_variable_values(
            [self.wetting.s], iterate_index=0
        )

        for pressure_key in (GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE):
            logger.info(f"Evaluating {pressure_key}.")
            p_vals = self.eval_glob_compl_pressure(s_w, pressure_key, p_n=p_n)
            pp.set_solution_values(
                name=pressure_key,
                values=p_vals,
                data=self.g_data,
                time_step_index=time_step_index,
                iterate_index=0,
            )

    def _compute_cap_press_integral(
        self, s_0: np.ndarray, s_1: np.ndarray, numerator_fn
    ) -> np.ndarray:
        """Generic helper for computing integrals used in pressure calculations."""
        intervals = np.linspace(s_0, s_1, 2)[..., None]

        def integrand(s: np.ndarray) -> np.ndarray:
            """Note: We cannot use ``two_phase_flow.DarcyFluxes.phase_mobility`` and
            ``two_phase_flow.DarcyFluxes.total_mobility`` here, because they require
            upwinding."""
            w_mobility = self.rel_perm_np(s, self.wetting) / self.wetting.viscosity
            n_mobility = (
                self.rel_perm_np(s, self.nonwetting) / self.nonwetting.viscosity
            )
            t_mobility = w_mobility + n_mobility
            return numerator_fn(
                w_mobility, n_mobility, t_mobility
            ) * self.cap_press_deriv_np(s)

        return self.quadrature_1d.integrate(integrand, intervals).elementwise.squeeze()

    def global_pressure_integral(self, s_0: np.ndarray, s_1: np.ndarray) -> np.ndarray:
        r"""Compute the integral in the global pressure formula for given integral
        boundaries.


        Cellwise it holds:
        .. math::
            result = \int_{s_0}^{s_1} \frac{\lambda_n}{\lambda_t} p'_c ds.

        Parameters:
            s_0: ``shape=(num_elements,)`` Lower integral boundaries.
            s_1: ``shape=(num_elements,)`` Upper integral boundaries.

        Returns:
            Global pressure values.

        """
        return self._compute_cap_press_integral(s_0, s_1, lambda w, n, t: w / t)

    def complementary_pressure_integral(
        self, s_0: np.ndarray, s_1: np.ndarray
    ) -> np.ndarray:
        r"""Compute complementary pressure from the rel. perm. and capillary pressure
        functions.


        Cellwise it holds:
        .. math::
            result = - \int_{s_0}^{s_1} \frac{\lambda_n \lambda_w}{\lambda_t} p'_c ds.

        Parameters:
            s_0: ``shape=(num_elements,)`` Lower integral boundaries.
            s_1: ``shape=(num_elements,)`` Upper integral boundaries.

        Returns:
            Complementary pressure values.

        """
        return -self._compute_cap_press_integral(s_0, s_1, lambda w, n, t: w * n / t)

    def set_boundary_pressures(self) -> None:
        """Set boundary pressures for the global and complementary pressure fields."""
        # We assume that boundaries are either no-flow Neumann or outflow Dirichlet. In
        # both cases no capillary pressure distribution from the boundaries goes into
        # the global and complementary pressures. At the boundary, the global pressure
        # equals the non-wetting pressure, while the complementary pressure is zero .
        glob_bc_dir = self.bc_dirichlet_pressure_values(self.g, self.nonwetting)
        compl_bc_dir = np.zeros_like(glob_bc_dir)

        # TODO See change below. If we loop over boundaries, we do not need to do
        # this next construction.
        bg, bg_data = self.mdg.boundaries(return_data=True)[0]
        for pressure_key, values in zip(
            [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE],
            [glob_bc_dir, compl_bc_dir],
        ):
            # Constant in time values are stored at `time_step_index` and
            # `iterate_index` 0.
            pp.set_solution_values(
                name=pressure_key,
                values=bg.projection() @ values,
                data=bg_data,
                time_step_index=0,
                iterate_index=0,
            )


class PressureReconstructionMixin(TPFProtocol):
    """Code and method are copied from Valera et al. (2024)."""

    DEFAULT_QUAD_DEGREE_REC = 4
    POLY_COEFF_COUNT = 6  # Quadratic polynomial: x^2, xy, x, y^2, y, const

    def setup_pressure_reconstruction(self) -> None:
        """Precompute static mappings and quadrature rules for pressure
        reconstruction."""
        self.quadpy_elements: np.ndarray = get_quadpy_elements(self.g)
        self.quadrature_rec = TriangleQuadrature(degree=self.DEFAULT_QUAD_DEGREE_REC)

        # Precompute mappings (fixed because of static grid).
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
            coeffs = np.zeros((self.g.num_cells, self.POLY_COEFF_COUNT))

        else:
            time_step_index = None

            # Retrieve CCFVM pressures.
            p_cc = pp.get_solution_values(pressure_key, self.g_data, iterate_index=0)
            assert p_cc.size == self.g.num_cells

            # Retrieve RT0 flux coefficients depending on pressure type.
            if pressure_key == GLOBAL_PRESSURE:
                coeffs_flux: np.ndarray = pp.get_solution_values(
                    f"{TOTAL_FLUX}_by_t_mobility{flux_specifier}_RT0_coeffs",
                    self.g_data,
                    iterate_index=0,
                )
            elif pressure_key == COMPLEMENTARY_PRESSURE:
                coeffs_flux: np.ndarray = pp.get_solution_values(
                    f"{TOTAL_FLUX}_times_fractional_flow{flux_specifier}_RT0_coeffs",
                    self.g_data,
                    iterate_index=0,
                ) - pp.get_solution_values(
                    f"{WETTING_FLUX}{flux_specifier}_RT0_coeffs",
                    self.g_data,
                    iterate_index=0,
                )
            else:
                raise ValueError(f"Unknown pressure key: {pressure_key}")

            # Multiply by inverse of the permeability and total mobility to obtain
            # pressure potential.
            perm: np.ndarray = self.g_data[pp.PARAMETERS][self.flux_key][
                "second_order_tensor"
            ].values

            # Compute pressure coefficients without constant term.
            coeffs = compute_pressure_coeffs(
                self.g.num_cells, self.g.dim, perm, coeffs_flux
            )

            # To obtain the constant c_5, we solve
            # :math:`c_5 = p_h - 1/|K| (gamma(x, y), 1)_K`,
            # where :math:`s(x, y) = gamma(x, y) + c_5`.

            def integrand(x: np.ndarray) -> np.ndarray:
                # `coeffs` was initiated with a zero c_5 term.
                return self._evaluate_poly_at_points(coeffs, x[..., 0], x[..., 1])

            integral: Integral = self.quadrature_rec.integrate(
                integrand,
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )

            # Now, we can compute the constant c_5, one per cell.
            coeffs[:, 5] = p_cc - integral.elementwise.squeeze() / self.g.cell_volumes

        # Store post-processed pressure coeffs.
        pp.set_solution_values(
            f"{pressure_key}_postprocessed_coeffs",
            coeffs,
            self.g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )

    def reconstruct_pressure_vohralik(
        self,
        pressure_key: PRESSURE_KEY,
        prepare_simulation: bool = False,
    ) -> None:
        r"""Reconstruct conforming pressures using the Oswald interpolator.

        Converts nonconforming :math:`\mathbb{P}^2(\mathcal{T}_h)` postprocessed
        coefficients into continuous :math:`H^1_0(\Omega)`-conforming ones.

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
            coeffs_rec = np.zeros((self.g.num_cells, self.POLY_COEFF_COUNT))
        else:
            time_step_index = None

            coeffs_postproc: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_postprocessed_coeffs",
                self.g_data,
                iterate_index=0,
            )

            # Sanity check
            if coeffs_postproc.shape != (self.g.num_cells, self.POLY_COEFF_COUNT):
                raise ValueError(
                    "Unexpected P2 polynomial coefficients shape:"
                    + f" {coeffs_postproc.shape}"
                )

            # Abbreviations
            g, dim = self.g, self.g.dim
            nc, nn, nf = g.num_cells, g.num_nodes, g.num_faces

            # Treatment of the nodes:
            # Evaluate post-processed pressure at the nodes.
            nodes_p = np.zeros([nc, 3])
            nx = g.nodes[0][self.nodes_of_cell]  # Local node x-coordinates
            ny = g.nodes[1][self.nodes_of_cell]  # Local node y-coordinates

            # Compute node pressures.
            for col in range(dim + 1):
                nodes_p[:, col] = self._evaluate_poly_at_points(
                    coeffs_postproc, nx[:, col], ny[:, col]
                )

            # Average nodal pressure.
            node_pressure = self._average_over_entities(
                nodes_p, self.cell_nodes_map, self.nodes_of_cell, nn
            )

            # Treatment of the faces:
            # Evaluate post-processed pressure at the face-centers.
            faces_p = np.zeros([nc, 3])
            fx = self.g.face_centers[0][
                self.faces_of_cell
            ]  # local face-center x-coordinates
            fy = self.g.face_centers[1][
                self.faces_of_cell
            ]  # local face-center y-coordinates

            for col in range(3):
                faces_p[:, col] = self._evaluate_poly_at_points(
                    coeffs_postproc, fx[:, col], fy[:, col]
                )

            # Average face pressure.
            face_pressure = self._average_over_entities(
                faces_p, self.cell_faces_map, self.faces_of_cell, nf
            )

            # Treatment of the boundary points:
            bg, bg_data = self.mdg.boundaries(return_data=True)[0]
            bc: pp.BoundaryCondition = self.g_data[pp.PARAMETERS][self.flux_key]["bc"]

            dir_faces = bc.is_dir
            bc_pressure = bg_data[pp.ITERATE_SOLUTIONS][pressure_key][0]
            bg_dir_filter: np.ndarray = (bg.projection() @ bc.is_dir) == 1

            # Set boundary face pressures.
            face_pressure[dir_faces] = bc_pressure[bg_dir_filter]

            # Average Dirichlet node pressures
            # Boundary values at the nodes.
            face_indicator = np.zeros_like(face_pressure)
            face_indicator[dir_faces] = 1

            face_vec = np.zeros(nf)
            face_vec[dir_faces] = 1
            num_dir_face_of_node = self.g.face_nodes * face_vec
            is_dir_node = num_dir_face_of_node > 0

            face_vec[:] = 0
            face_vec[dir_faces] = bc_pressure[bg_dir_filter]
            node_val_dir = self.g.face_nodes * face_vec
            node_val_dir[is_dir_node] /= np.maximum(
                num_dir_face_of_node[is_dir_node], 1
            )
            node_pressure[is_dir_node] = node_val_dir[is_dir_node]

            # Prepare for exporting.
            point_val = np.column_stack(
                [node_pressure[self.nodes_of_cell], face_pressure[self.faces_of_cell]]
            )
            point_coo = np.empty([dim, nc, self.POLY_COEFF_COUNT])
            point_coo[0] = np.column_stack([nx, fx])
            point_coo[1] = np.column_stack([ny, fy])

            # Solve local systems of equation to obtain the coefficients of the
            # reconstructed pressure from the points coordinates and values.
            # TODO Is there an easier way to do this WHILE reconstructing the pressure?
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
            coeffs_rec = linalg_solve_batch(A_elements, point_val)

        # Store in data dictionary.
        pp.set_solution_values(
            f"{pressure_key}_reconstructed_coeffs",
            coeffs_rec,
            self.g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )

    @staticmethod
    def _evaluate_poly_at_points(
        coeffs: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Evaluate P2 polynomial defined by `coeffs` at given (x, y) coordinates."""
        c = coeffs
        return (
            c[:, 0] * x**2
            + c[:, 1] * x * y
            + c[:, 2] * x
            + c[:, 3] * y**2
            + c[:, 4] * y
            + c[:, 5]
        )

    @staticmethod
    def _average_over_entities(
        values: np.ndarray,
        cell_entities_map: np.ndarray,
        entities_of_cell: np.ndarray,
        num_entities: int,
    ) -> np.ndarray:
        """Average per-cell entity values (nodes or faces) over shared entities."""
        cardinality = np.bincount(cell_entities_map)
        result = np.zeros(num_entities)
        for col in range(values.shape[1]):
            result += np.bincount(
                entities_of_cell[:, col],
                weights=values[:, col],
                minlength=num_entities,
            )
        return result / np.maximum(cardinality, 1)


class EquilibratedFluxMixin(ReconstructionProtocol, TPFProtocol):
    """Methods to equilibrate fluxes during the Newton iteration.

    Note: If the grid is updated during the simulation, the opposite side nodes and sign
    normals have to be updated as well. This is not supported by the current
    implementation.

    """

    def setup_flux_equilibration(self) -> None:
        """Precalculate opposite side nodes, cell faces, and sign normals for the
        grid."""
        opp_nodes_cell: np.ndarray = get_opposite_side_nodes(self.g)
        self.opp_nodes_coor_cell: np.ndarray = self.g.nodes[:, opp_nodes_cell]
        cell_faces_map = sps.find(self.g.cell_faces.T)[1]
        self.faces_cell = cell_faces_map.reshape(self.g.num_cells, self.g.dim + 1)
        self.sign_normals: np.ndarray = get_sign_normals(self.g)

    def equilibrate_flux_during_Newton(
        self,
        flux_name: Literal["total", "wetting"],
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
            - ``{flux_name}_flux_jac``, storing the Jacobian of the flux value from
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
        # The operators returned by ``DarcyFluxes.total_flux`` and
        # ``DarcyFluxes.wetting_flux`` include bc values, hence
        # ``f"{flux_name}_flux"`` includes bc values when set by ``eval_jac_and_val_fluxes``, and hence we do not need to care
        # about bc values here.
        logger.info(f"Equilibrating {flux_name} flux.")

        val: np.ndarray = pp.get_solution_values(
            flux_name, self.g_data, iterate_index=1
        )
        jac: np.ndarray = pp.get_solution_values(
            flux_name + "_jac", self.g_data, iterate_index=1
        )

        if nonlinear_increment is None:
            # NOTE The variables are retrieved in the same order as in the Jacobian
            # construction.
            # NOTE This requires the variables to be shifted at each nonlinear
            # iteration. By default, this happens in
            # :meth:`SolutionStrategyTPF.after_nonlinear_iteration`.
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

        pp.set_solution_values(
            f"{flux_name}_equil",
            equilibrated_flux,
            self.g_data,
            iterate_index=0,
        )

    def extend_fv_fluxes(
        self,
        flux_name: str,
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
            flux_name: Name of the flux field to be extended.
            prepare_simulation: Set to True if called in :meth:`prepare_simulation`.
                Stores values additionally for the time step.


        """
        if self.params["grid_type"] != "simplex":
            raise ValueError("Not implemented for non-simplex grids.")
        logger.info(f"Extending {flux_name} to RT0 functions.")

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
        flux = pp.get_solution_values(flux_name, self.g_data, iterate_index=0)

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
            flux_name + "_RT0_coeffs",
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

        TODO Calculate the elementwise mismatch.

        """
        # Calculate mismatches. The equilibrated flux values are stored in the data
        # dictionary were updated by :meth:`eval_postproc_qtys`.
        # Ignore Mypy type check for *.value(*).
        flux_t_equil_mismatch: float = np.sum(
            np.abs(
                self.postproc_ad_ops[TOTAL_FLUX + "_equil_mismatch"].value(
                    self.equation_system
                )  # type: ignore
            )
        )
        flux_w_equil_mismatch: float = np.sum(
            np.abs(
                self.postproc_ad_ops[WETTING_FLUX + "_equil_mismatch"].value(
                    self.equation_system
                )  # type: ignore
            )
        )

        logger.info(f"Total flux equilibration mismatch {flux_t_equil_mismatch}")
        logger.info(f"Wetting flux equilibration mismatch {flux_w_equil_mismatch}")
        return {
            "total flux": flux_t_equil_mismatch,
            "wetting flux": flux_w_equil_mismatch,
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

        self.postproc_ad_ops: dict[str, pp.ad.Operator] = {}

        # Fluxes.
        flux_t: pp.ad.Operator = self.total_flux(self.g)
        flux_w: pp.ad.Operator = self.wetting_flux(self.g)

        flux_t_by_lambda_t: pp.ad.Operator = flux_t / self.total_mobility(self.g)
        flux_t_times_f_w: pp.ad.Operator = flux_t_by_lambda_t * self.phase_mobility(
            self.g, self.wetting
        )

        # Equilibrated fluxes and mismatches.
        # TODO This copies 90% of the code from ``set_equations``. Make
        # ``set_equations`` more flexible and call (with the reconstructed flux) to
        # avoid this.

        # Discretization operators.
        div = pp.ad.Divergence([self.g])
        dt = pp.ad.Scalar(self.time_manager.dt)
        dt_s: pp.ad.Operator = pp.ad.time_derivatives.dt(self.wetting.s, dt)

        # Ad source.
        source_ad_w = pp.ad.DenseArray(self.phase_fluid_source(self.g, self.wetting))
        source_ad_t = pp.ad.DenseArray(self.total_fluid_source(self.g))

        # Ad parameters.
        porosity_ad = pp.ad.DenseArray(self.porosity(self.g))

        # Ad equations. The equilibrations are
        flux_t_equil = pp.ad.TimeDependentDenseArray(TOTAL_FLUX + "_equil", [self.g])
        flux_w_equil = pp.ad.TimeDependentDenseArray(WETTING_FLUX + "_equil", [self.g])

        flux_t_equil_mismatch = div @ flux_t_equil - source_ad_t
        flux_w_equil_mismatch = (
            porosity_ad * (self.volume_integral(dt_s, [self.g], 1))
            + div @ flux_w_equil
            - source_ad_w
        )

        # Store all post-processing equations in a dictionary for easy access.
        def add_to_postproc_ad_ops(name: str, op: pp.ad.Operator) -> None:
            """Helper function to add an operator to the post-processing equations
            dictionary."""
            op.set_name(name)
            self.postproc_ad_ops[name] = op

        for name, op in [
            (TOTAL_FLUX, flux_t),
            (WETTING_FLUX, flux_w),
            (TOTAL_FLUX + "_by_t_mobility", flux_t_by_lambda_t),
            (TOTAL_FLUX + "_times_fractional_flow", flux_t_times_f_w),
            (TOTAL_FLUX + "_equil", flux_t_equil),
            (WETTING_FLUX + "_equil", flux_w_equil),
            (TOTAL_FLUX + "_equil_mismatch", flux_t_equil_mismatch),
            (WETTING_FLUX + "_equil_mismatch", flux_w_equil_mismatch),
        ]:
            add_to_postproc_ad_ops(name, op)


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

        # Initalize P0 pressures and scaled fluxes to construct piecewise P2 pressures
        # at `iterate_index` 0 and `time_step_index` 0.
        self.eval_postproc_qtys(time_step_index=0)
        # Flux equilibration requires values from two iterates, hence we
        # initialize both the current and previous iterate with the same value. Values
        # from `iterate_index` 0 in the data dictionary are shifted to 'iterate_index' 1
        # in the second call.
        self.eval_postproc_qtys()

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
        self.eval_postproc_qtys()

        # When Newton is run with Appleyard chopping, the non-chopped nonlinear
        # increment has to be used to equilibrate the fluxes.
        if self._nl_appleyard_chopping or self._nl_enforce_physical_saturation:
            if hasattr(self, "non_chopped_nonlinear_increment"):
                nonlinear_increment = self.non_chopped_nonlinear_increment
            else:
                raise AttributeError(
                    "The non-chopped nonlinear increment vector has to be stored when"
                    + " Appleyard chopping or enforcing of physical saturations is"
                    + " enabled."
                )

        try:
            self.postprocess_solution(nonlinear_increment)

        except np.linalg.LinAlgError as e:
            # If Newton diverged, postprocessing may fail because ``linalg_solve_batch``
            # cannot handle infs or nans.
            logger.warning(
                "Postprocessing failed, likely due to diverged Newton."
                + " Skipping postprocessing this iteration."
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
            pressure_values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}{specifier}", self.g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}{specifier}",
                pressure_values,
                self.g_data,
                time_step_index=0,
            )

    def eval_postproc_qtys(self, time_step_index: int | None = None) -> None:
        """Evaluate post-processed pressure and scaled flux variables and save in data
        dictionary after each iteration.

        Populates the current `iterate_index` in the data dictionary with:
        - 'total_flux_val', 'total_flux_jac': Total flux value and Jacobian.
        - 'wetting_flux_val','wetting_flux_jac': Wetting flux value and
            Jacobian.

        Populates the current `iterate_index` (and the given `time_step_index`) in the
        data dictionary with:
        - 'global_pressure': Global pressure value.
        - 'complementary_pressure': complementary pressure value.
        - 'total_by_t_mobility_flux': Total flux divided by total mobility.
        - 'total_times_fractional_flow_flux': Total flux times fractional flow.

        Parameters:
            time_step_index: Save values at this 'time_step_index' in the data
                dictionary, in addition to the current 'iterate_index'. During
                initialization this must be 0. Default is None.

        """
        # Evaluate and save cellwise constant pressure values.
        self.eval_glob_compl_pressure_on_domain(time_step_index=time_step_index)

        # Evaluate face fluxes and their Jacobians.
        for flux_name in [TOTAL_FLUX, WETTING_FLUX]:
            flux = self.postproc_ad_ops[flux_name].value_and_jacobian(
                self.equation_system
            )
            val = flux.val
            jac = flux.jac[  # type: ignore
                :, : self.g.num_cells * 2
            ]  # Only primary variables.

            # Flux equilibration requires values from the previous iteration, hence
            # everything is shifted.
            pp.shift_solution_values(
                flux_name,
                self.g_data,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                flux_name,
                val,
                self.g_data,
                iterate_index=0,
            )
            pp.shift_solution_values(
                f"{flux_name}_jac",
                self.g_data,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            # ``jac`` is an ``sps.spmatrix``, which mypy complains about.
            pp.set_solution_values(
                f"{flux_name}_jac",
                jac,  # type: ignore
                self.g_data,
                iterate_index=0,
            )

        # Evaluate scaled fluxes required for pressure post-processing.
        for scaled_flux_name in [
            TOTAL_FLUX + "_by_t_mobility",
            TOTAL_FLUX + "_times_fractional_flow",
        ]:
            scaled_flux = self.postproc_ad_ops[scaled_flux_name].value(
                self.equation_system
            )
            pp.set_solution_values(
                scaled_flux_name,
                scaled_flux,  # type: ignore
                self.g_data,
                time_step_index=time_step_index,
                iterate_index=0,
            )

    def postprocess_solution(
        self, nonlinear_increment: np.ndarray, prepare_simulation: bool = False
    ) -> None:
        """Equilibrate fluxes and reconstruct pressures."""
        for flux_name in [TOTAL_FLUX, WETTING_FLUX]:
            # Extend both the nonequilibrated and equilibrated flux to compare in
            # the flux estimator. The nonequilibrated wetting flux is also used
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
                    flux_name + "_equil",
                )

        # Extend scaled fluxes needed for pressure reconstruction.
        for scaled_flux_name in [
            TOTAL_FLUX + "_by_t_mobility",
            TOTAL_FLUX + "_times_fractional_flow",
        ]:
            self.extend_fv_fluxes(
                scaled_flux_name, prepare_simulation=prepare_simulation
            )

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
    """Compute the opposite side nodes for each face of each cell in the grid.

    Parameters:
        g: Subdomain grid.

    Returns:
        np.ndarray: Opposite node indices with shape ``(g.num_cells, g.dim + 1)``.
            Each entry ``out[c, f]`` gives the node opposite to face ``f`` of cell
            ``c``.

    """
    dim = g.dim
    nc = g.num_cells
    nf = g.num_faces

    faces_of_cell = sps.find(g.cell_faces.T)[1].reshape(nc, dim + 1)
    nodes_of_cell = sps.find(g.cell_nodes().T)[1].reshape(nc, dim + 1)
    nodes_of_face = sps.find(g.face_nodes.T)[1].reshape(nf, dim)

    opposite_nodes = np.empty_like(faces_of_cell)
    for cell in range(nc):
        opposite_nodes[cell] = [
            np.setdiff1d(nodes_of_cell[cell], nodes_of_face[face])[0]
            for face in faces_of_cell[cell]
        ]
    return opposite_nodes


def get_sign_normals(g: pp.Grid) -> np.ndarray:
    """Compute the sign of the face normals for each cell in the grid.

    Note:
        We have to take care of the sign of the basis functions. The idea is to create
        an array of signs "sign_normals" that will be multiplying each edge basis
        function for the RT0 extension of fluxes.

        To determine this array, we need the following:
            (1) Compute the local outer normal `lon` vector for each cell.
            (2) For every face of each cell, compare if lon == global normal vector.
                If they're not, then we need to flip the sign of lon for that face

    Parameters:
        g: Subdomain grid.

    Returns:
        np.ndarray: Array of signs for each face with shape ``(grid.num_faces,)``.
            ``+1`` means the local and global normals point in the same direction,
            and ``-1`` means they point in opposite directions.

    """
    dim = g.dim
    nc = g.num_cells

    # Global face normals and centers associated to each cell.
    faces_cell = sps.find(g.cell_faces.T)[1].reshape(nc, dim + 1)
    face_centers_cells = g.face_centers[:, faces_cell]
    global_normals_cells = g.face_normals[:, faces_cell]

    # Compute the local outer normals of the faces per cell. To do this, we first
    # assume that n_loc = n_glb, and then we fix the sign. To fix the sign, we compare
    # the length of two vectors:
    # - v1 = face_center - cell_center
    # - v2 is a prolongation of v1 in the direction of the normal
    # If ||v2||<||v1||, then the  normal of the face in question is pointing inwards and
    # we needed to flip the sign.
    v1 = face_centers_cells - g.cell_centers[:, :, np.newaxis]
    v2 = v1 + global_normals_cells * 0.001

    # Checking whether ||v2|| < ||v1||.
    length_v1 = np.linalg.norm(v1, axis=0)
    length_v2 = np.linalg.norm(v2, axis=0)

    # Swap the sign of the local normal vectors.
    swap_sign = 1 - 2 * (length_v2 < length_v1)
    local_normals_cells = global_normals_cells * swap_sign

    # Now that we have the local outer normals, we can check if the local and global
    # normals are pointing in the same direction. To do this, we compute length_sum_n =
    # || n_glb + n_loc||. If they're the same, then length_sum_n > 0. Otherwise, they're
    # opposite and length_sum_n \approx 0.
    sum_n = local_normals_cells + global_normals_cells
    length_sum_n = np.linalg.norm(sum_n, axis=0)
    sign_normals = 1 - 2 * (length_sum_n < 1e-8)

    return sign_normals


# The loop could be parallelized with njit(parallel=True) and prange. However, on a
# small grids (<16000 cells) the overhead made :meth:`postprocess_pressure_vohralik`
# slower.
@njit
def compute_pressure_coeffs(
    num_cells: int, dim: int, perm: np.ndarray, flux_coeffs: np.ndarray
) -> np.ndarray:
    """Compute all but the constant coefficient for  the post-processed pressure.

    This function computes the coefficients of the quadratic part of the pressure
    polynomial from local RT0 flux coefficients and the permeability tensor.

    Parameters:
        num_cells: Number of cells in the grid.
        dim: Spatial dimension (2 for 2D grids).
        permeability: ``shape=(dim, dim, num_cells)`` Local permeability tensors.
        flux_coeffs: ``shape=(num_cells, 3)`` Local RT0 flux coefficients.

    Returns:
        np.ndarray: ``shape=(num_cells, 6)`` Pressure coefficients, where columns
            correspond to [x², xy, x, y², y, const]. The constant term is set to zero.

    """
    coeffs = np.zeros((num_cells, 6))

    # Loop through all cells and compute the nonconstant coefficients.
    for ci in prange(num_cells):
        # Local permeability tensor.
        K = perm[:dim, :dim, ci]
        Kxx, Kxy, Kyy = K[0, 0], K[0, 1], K[1, 1]

        # Retrieve components of the RT0 local flux field.
        a, b, c = flux_coeffs[ci]

        denom = Kxy**2 - Kxx * Kyy
        if abs(denom) < 1e-14:
            denom = np.sign(denom) * 1e-14  # numerical safety

        # Compute components of vector post-processed pressure.
        coeffs[ci, 0] = (a * Kyy) / (2 * denom)  # x^2
        # NOTE If K is a scalar, the following term will vanish.
        coeffs[ci, 1] = (a * Kxy) / (Kxx * Kyy - Kxy**2)  # xy
        coeffs[ci, 2] = (Kxy * c - Kyy * b) / (Kxx * Kyy - Kxy**2)  # x
        coeffs[ci, 3] = (a * Kxx) / (2 * denom)  # y^2
        coeffs[ci, 4] = (Kxx * c - Kxy * b) / denom  # y
    return coeffs


# NOTE The loop could be parallelized with njit(parallel=True). However, on a small grid
# (<16000 cells), the overhead made :meth:`reconstruct_pressure_vohralik` slower.
@njit
def linalg_solve_batch(A_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
    """Solve multiple small dense linear systems in batch mode.

    Args:
        A_batch: ``shape=(n, m, m)`` Array of matrices where ``n`` is the number of
            systems.
        b_batch: ``shape=(n, m)``Right-hand sides.

    Returns:
        np.ndarray: ``shape=(n, m)`` Solutions.

    """
    n, m, _ = A_batch.shape
    solutions = np.empty((n, m), dtype=A_batch.dtype)
    for i in prange(n):
        solutions[i] = np.linalg.solve(A_batch[i], b_batch[i])
    return solutions


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
