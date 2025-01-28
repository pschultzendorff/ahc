from __future__ import annotations

import itertools
import logging
import typing
from typing import Any, Literal, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps
from jax import jit
from numba import jit, njit
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
    COMPLIMENTARY_PRESSURE,
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
        self.epsilon: float = global_pressure_constants.get("epsilon", 1e-6)

        # Do not start evaluation at normalized saturation = zero, where the derivative
        # is :math:`\infty`. Instead add an epsilon.
        self.s_interpol_vals: np.ndarray = np.linspace(
            self.wetting.residual_saturation + self.epsilon,
            1 - self.nonwetting.residual_saturation,
            global_pressure_constants.get("interpolation_degree", 100),
        )
        self.calc_pressure_interpolants()
        # Evaluate initial values for global and complimentary pressure.

    def calc_pressure_interpolants(self) -> None:
        """Calculate interpolants values for the global and complimentary pressure."""
        global_pressure_integral_parts: np.ndarray = self.global_pressure_integral_part(
            self.s_interpol_vals[:-1], self.s_interpol_vals[1:]
        )
        self.global_pressure_interpol_vals: np.ndarray = np.cumsum(
            global_pressure_integral_parts
        )
        compl_pressure_integral_parts: np.ndarray = (
            self.complimentary_pressure_integral_part(
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
        p_n: Optional[np.ndarray] = None,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """Evaluate the global or complimentary pressure field for the given pressure
        and saturation values.

        Note: This is done via interpolation of the global pressure values depending on
        the saturation values.

        """
        # Limit saturation from below to 0 and from above to 1 in case of negative or >1
        # saturation values during Newton.
        # FIXME Using epsilon here instead of at the normalized saturation is not the
        # same! Think about this! Also, we add epsilon when calculating the
        # interpolants.
        s_w[s_w > 1 - self.nonwetting.residual_saturation - epsilon] = (
            1 - self.nonwetting.residual_saturation - epsilon
        )
        s_w[s_w < self.wetting.residual_saturation + epsilon] = (
            self.wetting.residual_saturation + epsilon
        )

        if pressure_key == GLOBAL_PRESSURE:
            if p_n is None:
                raise ValueError(
                    "Wetting pressure must be provided for global" + " pressure."
                )
            p_vals: np.ndarray = p_n - np.interp(
                s_w, self.s_interpol_vals, self.global_pressure_interpol_vals
            )
        elif pressure_key == COMPLIMENTARY_PRESSURE:
            p_vals = np.interp(
                s_w, self.s_interpol_vals, self.compl_pressure_interpol_vals
            )
        return p_vals

    def eval_glob_compl_pressure_on_domain(
        self,
        pressure_key: PRESSURE_KEY,
        prepare_simulation: bool = False,
    ) -> None:
        """Evaluate the global or complimentary pressure field on the full domain and
        store it in the data dictionary.

        Parameters:
            pressure_key: Name of the pressure field to be evaluated.
            prepare_simulation: Set to True if called in :meth:`prepare_simulation`.
                Stores values additionally for the time step.

        """
        if prepare_simulation:
            time_step_index: Optional[int] = 0
        else:
            time_step_index = None

        logger.info(f"Evaluating {pressure_key}.")
        g_data = self.mdg.subdomains(return_data=True)[0][1]
        p_n: np.ndarray = self.equation_system.get_variable_values(
            [self.nonwetting.p], iterate_index=0
        )
        s_w: np.ndarray = self.equation_system.get_variable_values(
            [self.wetting.s], iterate_index=0
        )
        p_vals: np.ndarray = self.eval_glob_compl_pressure(s_w, pressure_key, p_n=p_n)
        pp.shift_solution_values(
            name=pressure_key,
            data=g_data,
            location=pp.ITERATE_SOLUTIONS,
            max_index=len(self.iterate_indices),
        )
        pp.set_solution_values(
            name=pressure_key,
            values=p_vals,
            data=g_data,
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
        return integral.elementwise[:, 0]

    def complimentary_pressure_integral_part(
        self, s_0: np.ndarray, s_1: np.ndarray
    ) -> np.ndarray:
        r"""Compute complimentary pressure from the rel. perm. and capillary pressure
        functions.

        Note: The evaluation works by transforming ``p`` and ``s`` to
        :class:`~porepy.ad.DenseArray`, calling
        :meth:`CapillaryPressure.cap_press_deriv` and
        :meth:`RelativePermeability.rel_perm` and then evaluating the resulting
        :class:`~porepy.ad.Operator`. This is probably quite inefficient, however this
        is only done once at the start of each simulation. During the simulation the
        complimentary pressure is interpolated.

        Cellwise it holds:
        .. math::
            result = - \int_{s_0}^{s_1} \frac{\lambda_n \lambda_w}{\lambda_t} p'_c ds.

        Parameters:
            s_0: ``shape=(num_elements,)`` Lower integral boundaries.
            s_1: ``shape=(num_elements,)`` Upper integral boundaries.

        Returns:
            Complimentary pressure values.

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
        return -1 * integral.elementwise[..., 0]

    def set_boundary_pressures(self) -> None:
        """Set boundary pressures for the global and complimentary pressure fields."""
        g = self.mdg.subdomains()[0]
        # TODO: This is a bit of a mess. Ideally, bc_dirichlet_pressure_values gets
        # called with a boundary grid instead of a subdomain grid. Then we can loop
        # through boundaries. Alternatively, we implement
        # ``update_boundary_condition`` and ``create_boundary_operator`` in
        # ``BoundaryConditionsTPF`` and get the pressure and saturation values from
        # the data dictionary instead of calling the functions here.
        p_n_bc = self.bc_dirichlet_pressure_values(g, self.nonwetting)
        s_w_bc = self.bc_dirichlet_saturation_values(g, self.wetting)

        # Compute global and complimentary pressures on boundaries.
        p_g_bc = self.eval_glob_compl_pressure(s_w_bc, GLOBAL_PRESSURE, p_n=p_n_bc)
        p_c_bc = self.eval_glob_compl_pressure(s_w_bc, COMPLIMENTARY_PRESSURE)

        # TODO See change above. If we loop over boundaries, we do not need to do
        # this next construction.
        bg, bg_data = self.mdg.boundaries(return_data=True)[0]
        for pressure_key, values in zip(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE], [p_g_bc, p_c_bc]
        ):
            pp.set_solution_values(
                name=pressure_key,
                values=bg.projection() @ values,
                data=bg_data,
                time_step_index=0,
            )
            pp.set_solution_values(
                name=pressure_key,
                values=bg.projection() @ values,
                data=bg_data,
                iterate_index=0,
            )


class PressureReconstructionMixin(TPFProtocol):
    """Code and method are copied from Valera et al. (2024)."""

    def setup_pressure_reconstruction(self) -> None:
        self.quadpy_elements: np.ndarray = get_quadpy_elements(self.mdg.subdomains()[0])
        self.quadrature_reconstruction_degree: int = 4
        self.quadrature_reconstruction = TriangleQuadrature(
            degree=self.quadrature_reconstruction_degree
        )

    def reconstruct_pressure_vohralik(
        self,
        pressure_key: PRESSURE_KEY,
        flux_specifier: str = "",
        prepare_simulation: bool = False,
    ) -> None:
        """Reconstruct pressure as elementwise P2 polynomials.

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
                Stores values additionally for the time step.

        Returns:
            None

        """
        logger.info(f"Reconstructing {pressure_key} pressure.")

        if prepare_simulation:
            time_step_index: Optional[int] = 0
        else:
            time_step_index = None

        g, g_data = self.mdg.subdomains(return_data=True)[0]
        bg, bg_data = self.mdg.boundaries(return_data=True)[0]

        # Retrieve finite volume cell-centered pressures.
        p_cc = pp.get_solution_values(pressure_key, g_data, iterate_index=0)
        assert p_cc.size == g.num_cells

        # Retrieve RT0 flux coefficients.
        if pressure_key == GLOBAL_PRESSURE:
            coeffs_flux: np.ndarray = pp.get_solution_values(
                f"total_by_t_mobility_flux{flux_specifier}_RT0_coeffs",
                g_data,
                iterate_index=0,
            )
        elif pressure_key == COMPLIMENTARY_PRESSURE:
            coeffs_flux: np.ndarray = pp.get_solution_values(
                f"total_times_fractional_flow_flux{flux_specifier}_RT0_coeffs",
                g_data,
                iterate_index=0,
            ) - pp.get_solution_values(
                f"wetting_from_ff_flux{flux_specifier}_RT0_coeffs",
                g_data,
                iterate_index=0,
            )
        # Multiply by inverse of the permeability and total mobility to obtain
        # pressure potential.
        perm = g_data[pp.PARAMETERS][self.flux_key]["second_order_tensor"].values

        # Loop through all cells and compute the vector r.
        s = np.zeros((g.num_cells, 6))
        for ci in range(g.num_cells):
            # Local permeability tensor
            K = perm[: g.dim, : g.dim, ci]
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

        # To obtain the constant c_5, we solve  c_5 = p_h - 1/|K| (gamma(x, y), 1)_K,
        # where s(x, y) = gamma(x, y) + c_5.
        def integrand(x):
            int_0 = s[:, 0][None, ...] * x[..., 0] ** 2
            int_1 = s[:, 1][None, ...] * x[..., 0] * x[..., 1]
            int_2 = s[:, 2][None, ...] * x[..., 0]
            int_3 = s[:, 3][None, ...] * x[..., 1] ** 2
            int_4 = s[:, 4][None, ...] * x[..., 1]
            return int_0 + int_1 + int_2 + int_3 + int_4

        integral: Integral = self.quadrature_reconstruction.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )

        # Now, we can compute the constant C, one per cell.
        s[:, 5] = p_cc - integral.elementwise / g.cell_volumes

        # Store post-processed but not reconstructed pressure at the nodes.
        pp.shift_solution_values(
            f"{pressure_key}_postprocessed_coeffs",
            g_data,
            pp.ITERATE_SOLUTIONS,
            max_index=len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{pressure_key}_postprocessed_coeffs",
            s,
            g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )

        # The following step is now to apply the Oswald interpolator

        # Sanity check
        if not s.shape == (g.num_cells, 6):
            raise ValueError("Wrong shape of P2 polynomial.")

        # Abbreviations
        dim = g.dim
        nn = g.num_nodes
        nf = g.num_faces
        nc = g.num_cells

        # Mappings
        cell_faces_map = sps.find(g.cell_faces.T)[1]
        cell_nodes_map = sps.find(g.cell_nodes().T)[1]
        faces_of_cell = cell_faces_map.reshape(nc, dim + 1)
        nodes_of_cell = cell_nodes_map.reshape(nc, dim + 1)

        # Treatment of the nodes
        # Evaluate post-processed pressure at the nodes
        nodes_p = np.zeros([nc, 3])
        nx = g.nodes[0][nodes_of_cell]  # local node x-coordinates
        ny = g.nodes[1][nodes_of_cell]  # local node y-coordinates

        # Compute node pressures
        for col in range(dim + 1):
            nodes_p[:, col] = (
                s[:, 0] * nx[:, col] ** 2  # c0x^2
                + s[:, 1] * nx[:, col] * ny[:, col]  # c1xy
                + s[:, 2] * nx[:, col]  # c2x
                + s[:, 3] * ny[:, col] ** 2  # c3y^2
                + s[:, 4] * ny[:, col]  # c4y
                + s[:, 5]  # c5
            )

        # Average nodal pressure
        node_cardinality = np.bincount(cell_nodes_map)
        node_pressure = np.zeros(nn)
        for col in range(dim + 1):
            node_pressure += np.bincount(
                nodes_of_cell[:, col], weights=nodes_p[:, col], minlength=nn
            )
        node_pressure /= node_cardinality

        # Treatment of the faces
        # Evaluate post-processed pressure at the face-centers
        faces_p = np.zeros([nc, 3])
        fx = g.face_centers[0][faces_of_cell]  # local face-center x-coordinates
        fy = g.face_centers[1][faces_of_cell]  # local face-center y-coordinates

        for col in range(3):
            faces_p[:, col] = (
                s[:, 0] * fx[:, col] ** 2  # c0x^2
                + s[:, 1] * fx[:, col] * fy[:, col]  # c1xy
                + s[:, 2] * fx[:, col]  # c2x
                + s[:, 3] * fy[:, col] ** 2  # c3y^2
                + s[:, 4] * fy[:, col]  # c4x
                + s[:, 5]  # c5
            )

        # Average face pressure
        face_cardinality = np.bincount(cell_faces_map)
        face_pressure = np.zeros(nf)
        for col in range(3):
            face_pressure += np.bincount(
                faces_of_cell[:, col], weights=faces_p[:, col], minlength=nf
            )
        face_pressure /= face_cardinality

        # Treatment of the boundary points
        bc: pp.BoundaryCondition = g_data[pp.PARAMETERS][self.flux_key]["bc"]

        external_dirichlet_boundary = bc.is_dir
        bc_pressure = bg_data[pp.ITERATE_SOLUTIONS][pressure_key][0]
        bg_dir_filter: np.ndarray = (bg.projection() @ bc.is_dir) == 1
        face_pressure[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]

        # Boundary values at the nodes
        face_vec = np.zeros(nf)
        face_vec[external_dirichlet_boundary] = 1
        num_dir_face_of_node = g.face_nodes * face_vec
        is_dir_node = num_dir_face_of_node > 0
        face_vec *= 0
        face_vec[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]
        node_val_dir = g.face_nodes * face_vec
        node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
        node_pressure[is_dir_node] = node_val_dir[is_dir_node]

        # Prepare for exporting
        point_val = np.column_stack(
            [node_pressure[nodes_of_cell], face_pressure[faces_of_cell]]
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
        coeffs_reconstructed_pressure = np.array(
            [np.linalg.solve(A, b) for A, b in zip(A_elements, point_val)]
        )

        # Store in data dictionary
        for value, name in zip(
            [point_val, point_coo, coeffs_reconstructed_pressure],
            ["point_val", "point_coo", "coeffs"],
        ):
            pp.shift_solution_values(
                f"{pressure_key}_reconstructed_{name}",
                g_data,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"{pressure_key}_reconstructed_{name}",
                value,
                g_data,
                time_step_index=time_step_index,
                iterate_index=0,
            )

    def gradient_mismatch(self, reconstructed_pressure) -> np.ndarray:
        raise NotImplementedError


class EquilibratedFluxMixin(TPFProtocol):

    def equilibrate_flux_during_Newton(
        self,
        flux_name: Literal["total", "wetting_from_ff"],
        nonlinear_increment: Optional[np.ndarray] = None,
    ) -> None:
        """Equilibrate an approximate flux solution at a given Newton iteration.

        We assume the following sub-dictionaries to be present in the data dictionary:
            iterate_dictionary, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        The following entries in iterate_dictionary will be shifted and updated:
            - {flux_name}_flux_equilibrated, storing the equilibrated flux.

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
        # The functions ``DarcyFluxes.total_flux`` and ``DarcyFluxes.phase_flux``
        # incorporate bc values, hence ``d[f"{flux_name}_flux"]`` includes bc
        # values when set by ``eval_additional_vars`` and hence we do not need to care
        # about bc values here.
        logger.info(f"Equilibrating {flux_name} flux.")

        g_data = self.mdg.subdomains(return_data=True)[0][1]
        val: np.ndarray = pp.get_solution_values(
            f"{flux_name}_flux", g_data, iterate_index=1
        )
        jac: np.ndarray = pp.get_solution_values(
            f"{flux_name}_flux_jacobian", g_data, iterate_index=1
        )

        if nonlinear_increment is None:
            # NOTE The variables are retrieved in the same order as in the Jacobian
            # construction.
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
        # is_dir: np.ndarray = g_data[pp.PARAMETERS][self.flux_key]["bc"].is_dir
        # equilibrated_flux[is_dir] = 0

        pp.shift_solution_values(
            f"{flux_name}_flux_equilibrated",
            g_data,
            pp.ITERATE_SOLUTIONS,
            max_index=len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_flux_equilibrated",
            equilibrated_flux,
            g_data,
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
        """Extend flux (eqilibrated or non-equilibrated) using RT0 basis functions.

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

                psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i)^T                     (for 1d),
                psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i)^T            (for 2d),
                psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i, z - z_i)^T   (for 3d),

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
            time_step_index: Optional[int] = 0
        else:
            time_step_index = None

        # Only one domain is considered here.
        g, g_data = self.mdg.subdomains(return_data=True)[0]

        # Cell-basis arrays
        cell_faces_map = sps.find(g.cell_faces.T)[1]
        faces_cell = cell_faces_map.reshape(g.num_cells, g.dim + 1)
        opp_nodes_cell = get_opposite_side_nodes(g)
        opp_nodes_coor_cell = g.nodes[:, opp_nodes_cell]
        sign_normals_cell = get_sign_normals(g)
        vol_cell = g.cell_volumes

        # Retrieve finite volume fluxes
        flux = pp.get_solution_values(
            f"{flux_name}_flux{flux_specifier}", g_data, iterate_index=0
        )

        # Perform actual reconstruction and obtain coefficients
        coeffs = np.empty([g.num_cells, g.dim + 1])
        alpha = 1 / (g.dim * vol_cell)
        coeffs[:, 0] = alpha * np.sum(sign_normals_cell * flux[faces_cell], axis=1)
        for dim in range(g.dim):
            coeffs[:, dim + 1] = -alpha * np.sum(
                (sign_normals_cell * flux[faces_cell] * opp_nodes_coor_cell[dim]),
                axis=1,
            )

        # Store coefficients in the data dictionary.
        pp.shift_solution_values(
            f"{flux_name}_flux{flux_specifier}_RT0_coeffs",
            g_data,
            pp.ITERATE_SOLUTIONS,
            max_index=len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_flux{flux_specifier}_RT0_coeffs",
            coeffs,
            g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )

    def equilibrated_flux_mismatch(self) -> dict[str, float]:
        r"""Calculate mismatch of the equilibrated flux from being in :math:`H(div)` and
        being mass conservative.

        Check :meth:`tpf.models.two_phase_flow.EquationsTPF.set_equations` for
        details.

        TODO Calculate the elementwise mismatch.
        TODO This copies 90% of the code from ``set_equations``. Make ``set_equations``
        more flexible and call (with the reconstructed flux) to avoid this.
        TODO The equation does not need to be redefined at each Newton iteration.
        Instead, save it to the equation system and just reevaluate.

        """
        # TEST -> Local mass conservation
        # Check if mass conservation is satisfied on a cell basis, in order to do
        # this, we check on a local basis, if the divergence of the flux equals
        # the sum of internal and external source terms
        # full_flux_local_div = (sign_normals_cell * flux[faces_cell]).sum(axis=1)
        # external_src = d[pp.PARAMETERS][self.kw]["source"]
        # np.testing.assert_allclose(
        #     full_flux_local_div,
        #     external_src + mortar_jump,
        #     rtol=1e-6,
        #     atol=1e-3,
        #     err_msg="Error estimates only valid for local mass-conservative methods.",
        # )
        # END OF TEST

        g, g_data = self.mdg.subdomains(return_data=True)[0]

        # Spatial discretization operators.
        div = pp.ad.Divergence([g])

        # Time derivatives.
        dt = pp.ad.Scalar(self.time_manager.dt)
        dt_s: pp.ad.Operator = pp.ad.time_derivatives.dt(self.wetting.s, dt)

        # Ad source.
        source_ad_w = pp.ad.DenseArray(self.phase_fluid_source(g, self.wetting))
        source_ad_t = pp.ad.DenseArray(self.total_fluid_source(g))

        # Ad parameters.
        porosity_ad = pp.ad.DenseArray(self.porosity(g))

        # Ad equations
        if self.formulation == "fractional_flow":
            # Note, that for ``flux_t``, the total mobility is already included.
            flux_t = pp.ad.DenseArray(
                pp.get_solution_values(
                    "total_flux_equilibrated", g_data, iterate_index=0
                )
            )
            flux_w = pp.ad.DenseArray(
                pp.get_solution_values(
                    "wetting_from_ff_flux_equilibrated", g_data, iterate_index=0
                )
            )

            flow_equation = div @ flux_t - source_ad_t
            transport_equation = (
                porosity_ad * (self.volume_integral(dt_s, [g], 1))
                + div @ flux_w
                - source_ad_w
            )

            # flux_tpfa = pp.ad.TpfaAd(self.flux_key, [g])

            # # Compute cap pressure and relative permeabilities.
            # p_cap = self.cap_press(self.wetting.s)

            # mobility_w = self.phase_mobility(g, self.wetting)
            # mobility_n = self.phase_mobility(g, self.nonwetting)
            # mobility_t = self.total_mobility(g)
            # fractional_flow_w = mobility_w / mobility_t
            # vector_source_w = pp.ad.DenseArray(self.vector_source(g, self.wetting))
            # vector_source_n = pp.ad.DenseArray(self.vector_source(g, self.nonwetting))
            # transport_equation_ff = (
            #     porosity_ad * (self.volume_integral(dt_s, [g], 1))
            #     + div
            #     @ (
            #         fractional_flow_w * flux_t
            #         + fractional_flow_w
            #         * mobility_n
            #         * (
            #             flux_tpfa.flux() @ p_cap
            #             # TODO: Plus boundary values here or are they included in the total flux?
            #             + flux_tpfa.vector_source() @ vector_source_w
            #             - flux_tpfa.vector_source() @ vector_source_n
            #         )
            #     )
            #     - source_ad_w
            # )
        flow_equation.set_name("Flow reconstruction mismatch")
        transport_equation.set_name("Transport reconstruction wetting flow mismatch")
        # transport_equation_ff.set_name(
        #     "Transport reconstruction fractional flow mismatch"
        # )
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
        # logger.info(
        #     f"Transport equation mismatch fractional flow {np.sum(np.abs(transport_equation_ff.value(self.equation_system)))}"  # type: ignore
        # )
        return {
            "flow": flow_equation_mismatch,
            "transport": transport_equation_mismatch,
        }


# This could also be a mixin, but by subclassing ``SolutionStrategyTPF``, we avoid
# having to pay attention to the order of the different solution strategy classes.
# ``TPFProtocol`` and ``pp.SolutionStrategy`` define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class SolutionStrategyReconstruction(  # type: ignore
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

        # Initialize fluxes, global, and complimentary pressure from the initial data of
        # the problem.
        self.eval_val_and_jac_fluxes(prepare_simulation=True)
        self.eval_additional_vars(prepare_simulation=True)
        self.set_boundary_pressures()
        self.postprocess_solution(
            np.zeros(self.mdg.subdomains()[0].num_cells * 2), prepare_simulation=True
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
        self.postprocess_solution(nonlinear_increment)

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
        # Shift pressures, postprocessing, and reconstructions to the next time step.
        g_data = self.mdg.subdomains(return_data=True)[0][1]
        for pressure_key, specifier in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            ["", "_postprocessed_coeffs", "_reconstructed_coeffs"],
        ):
            pp.shift_solution_values(
                f"{pressure_key}{specifier}",
                g_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}{specifier}", g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}{specifier}",
                values,
                g_data,
                time_step_index=0,
            )

    def eval_val_and_jac_fluxes(self, prepare_simulation: bool = False) -> None:
        """Evaluate residual and Jacobian of fluxes to be equilibrated.

        Note: This is separated from ``eval_additional_vars`` and called **before** the
        Newton update s.t. the fluxes are evaluated with the current iterate values and
        upwinding direction.

        Populates the data dictionary with:
        - 'total_flux_val', 'total_flux_jac': Total flux value and Jacobian.
        - 'wetting_from_ff_flux_val','wetting_from_ff_flux_jac': Wetting flux value and
        Jacobian.

        Parameters:
            prepare_simulation: Set to True if called in :meth:`prepare_simulation`.
                If True, values are additionally saved at `iterate_index = 1`.

        """
        g, g_data = self.mdg.subdomains(return_data=True)[0]
        # Multiply Jacobians with the the following matrix, s.t., the Jacobian is
        # only w.r.t. the primary variables.
        # FIXME It would be more efficient to NOT assemble the full Jacobian.
        # Possibly, by not setting secondary pressure and saturation as variables,
        # but just evaluating their values after each nonlinear iteration and
        # storing them in the data dict?
        column_projection = self.equation_system.projection_to(
            [self.primary_saturation_var, self.primary_pressure_var]
        ).transpose()
        for flux_name in ["total", "wetting_from_ff"]:
            if flux_name == "total":
                flux: pp.ad.AdArray = self.total_flux(g).value_and_jacobian(
                    self.equation_system
                )
            elif flux_name == "wetting_from_ff":
                flux: pp.ad.AdArray = self.wetting_flux_from_fractional_flow(
                    g
                ).value_and_jacobian(self.equation_system)
            pp.shift_solution_values(
                f"{flux_name}_flux",
                g_data,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"{flux_name}_flux",
                flux.val,
                g_data,
                iterate_index=0,
            )
            pp.shift_solution_values(
                f"{flux_name}_flux_jacobian",
                g_data,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"{flux_name}_flux_jacobian",
                flux.jac * column_projection,
                g_data,
                iterate_index=0,
            )
        if prepare_simulation:
            pp.set_solution_values(
                flux_name,
                flux.val,
                g_data,
                iterate_index=0,
            )
            pp.set_solution_values(
                f"{flux_name}_jacobian",
                flux.jac * column_projection,
                g_data,
                iterate_index=1,
            )

    @typing.override
    def eval_additional_vars(self, prepare_simulation: bool = False) -> None:
        """Evaluate additional pressure and flux variables and save in data dictionary
        after each iteration.

        Populates the data dictionary with:
        - 'global_pressure': Global pressure value.
        - 'complimentary_pressure': Complimentary pressure value.
        - '{phase.name}_flux': Flux value for both phases.

        Parameters:
            prepare_simulation: If True, the function is called during setup and the
                values are saved at the initial time step in addition to the current
                nonlinear iteration. Default is False.

        """
        if prepare_simulation:
            time_step_index: Optional[int] = 0
        else:
            time_step_index = None

        g, g_data = self.mdg.subdomains(return_data=True)[0]

        # Save FV P0 pressures.
        self.eval_glob_compl_pressure_on_domain(
            GLOBAL_PRESSURE, prepare_simulation=prepare_simulation
        )
        self.eval_glob_compl_pressure_on_domain(
            COMPLIMENTARY_PRESSURE, prepare_simulation=prepare_simulation
        )

        # Calculate fluxes divided by mobilities. These are needed for pressure
        # post-processing.
        total_by_t_mobility: np.ndarray = (
            self.total_flux(g) / self.total_mobility(g)
        ).value(self.equation_system)
        total_times_fractional_flow: np.ndarray = (
            total_by_t_mobility
            * self.phase_mobility(g, self.wetting).value(self.equation_system)
        )
        pp.set_solution_values(
            f"total_by_t_mobility_flux",
            total_by_t_mobility,
            g_data,
            time_step_index=time_step_index,
            iterate_index=0,
        )
        pp.set_solution_values(
            "total_times_fractional_flow_flux",
            total_times_fractional_flow,
            g_data,
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
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            # Satisfy mypy.
            pressure_key = typing.cast(PRESSURE_KEY, pressure_key)

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


class DataSavingReconstruction(DataSavingTPF):

    def _data_to_export(
        self, time_step_index: Optional[int] = None, iterate_index: Optional[int] = None
    ) -> list[DataInput]:
        """Append global and complimentary pressures to the exported data."""
        data: list[DataInput] = super()._data_to_export(
            time_step_index=time_step_index,
            iterate_index=iterate_index,
        )
        # Only export for nonzero time steps or nonlinear steps. Otherwise, this causes
        # an error, as the function is called via
        # ``SolutionStrategyTPF.prepare_simulation`` BEFORE the initial values are set
        # by ``SolutionStrategyEst.prepare_simulation``.
        if (time_step_index is not None and self.time_manager.time_index > 0) or (
            iterate_index is not None
        ):
            g, g_data = self.mdg.subdomains(return_data=True)[0]
            for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
                data.append(
                    (
                        g,
                        pressure_key,
                        pp.get_solution_values(
                            pressure_key,
                            g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
        return data


# ``TPFProtocol`` and ``pp.SolutionStrategy`` define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class TwoPhaseFlowReconstruction(  # type: ignore
    GlobalPressureMixin,
    PressureReconstructionMixin,
    EquilibratedFluxMixin,
    SolutionStrategyReconstruction,
    DataSavingReconstruction,
    TwoPhaseFlow,
): ...  # type: ignore
