r"""

FIXME Evaluation of global and complimentary pressure is done super inefficiently!
Instead of evaluating the expressions for each cell, we should evaluate the expressions
once from :math:`\hat{s}_w = 0` to :math:`\hat{s}_w = 1` for sufficiently many
integration points and then interpolate the values for each cell.

"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps
from tpf_lab.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    GLOBAL_PRESSURE,
    OperatorType,
)
from tpf_lab.models.phase import PHASENAME, Phase
from tpf_lab.models.two_phase_flow import SolutionStrategyTPF
from tpf_lab.numerics.quadrature import GaussLegendreQuadrature1D, Integral

logger = logging.getLogger(__name__)


class PressureMixin:

    mdg: pp.MixedDimensionalGrid

    wetting: Phase
    nonwetting: Phase

    cap_press_deriv: Callable[[pp.ad.DenseArray], pp.ad.DenseArray]

    rel_perm: Callable[[pp.ad.Operator, Phase], pp.ad.Operator]

    bc_dirichlet_pressure_values: Callable[[pp.Grid, Phase], np.ndarray]
    bc_dirichlet_saturation_values: Callable[[pp.Grid, Phase], np.ndarray]

    params: dict[str, Any]
    equation_system: pp.ad.EquationSystem
    time_step_indices: list
    iterate_indices: list

    def set_global_pressure_constants(self) -> None:
        global_pressure_constants: dict[str, Any] = self.params.get(
            "global_pressure_constants", {}
        )
        self.quadrature_degree: int = global_pressure_constants.get(
            "quadrature_degree", 10
        )
        self.quadrature = GaussLegendreQuadrature1D(self.quadrature_degree)

    def global_pressure(
        self, p: np.ndarray, s: np.ndarray, epsilon: float = 1e-6
    ) -> np.ndarray:
        r"""Compute the global pressure from the primary variables.

        TODO: Should we use ``pp.ad.DenseArray`` or rather ``np.ndarray`` for faster
        execution? The latter has the disadvantage that ``cap_press`` and ``*_mobility``
        cannot be called on the values. The former has the disadvantage that
        ``quadrature.integrate`` cannot be called on the values. A possibility is to ...

        Cellwise it holds:
        ..math::
            p_G = p_w + \int_{0}^{s} \frac{\lambda_n}{\lambda_t} p'_c ds.

        Parameters:
            p: Wetting pressure field.
            s: Wetting saturation field.

        Returns:
            Global pressure field.

        """
        # TODO: Change this s.t. p and s are ``pp.ad.DenseArray``.
        # Question: What to do with negative saturation values during Newton?
        # Limit s from below and above by residual saturations.
        # FIXME Using epsilon here instead of the normalized saturation is not the same!
        s[s > 1 - self.nonwetting.constants.residual_saturation() - epsilon] = 0
        intervals = np.linspace(
            np.full(s.shape, self.wetting.constants.residual_saturation() + epsilon),
            s,
            2,
        )[..., None]

        def func(s: np.ndarray) -> np.ndarray:
            """Note: We cannot use ``two_phase_flow.DarcyFluxes.phase_mobility`` and
            ``two_phase_flow.DarcyFluxes.total_mobility`` here, because they require
            upwinding.

            """
            # Theoretically, we can evaluate the expression before returning. However,
            # this is probably super inefficient.
            # TODO: This is probably super inefficient!!!
            s_ad = pp.ad.DenseArray(s)
            w_mobility: pp.ad.DenseArray = self.rel_perm(
                s_ad, self.wetting
            ) / pp.ad.Scalar(self.wetting.constants.viscosity())
            n_mobility: pp.ad.DenseArray = self.rel_perm(
                s_ad, self.nonwetting
            ) / pp.ad.Scalar(self.nonwetting.constants.viscosity())
            t_mobility: pp.ad.DenseArray = w_mobility + n_mobility
            p_g: pp.ad.DenseArray = n_mobility / t_mobility * self.cap_press_deriv(s_ad)
            return p_g.value(self.equation_system)

        integral: Integral = self.quadrature.integrate(func, intervals)
        return p + integral.elementwise[:, 0]

    def eval_global_pressure(self) -> None:
        """Evaluate the global pressure field on the grid and store it in the data
        dictionary."""
        logger.info(f"Evaluating global pressure.")
        for _, data in self.mdg.subdomains(return_data=True):
            p_w: np.ndarray = self.equation_system.get_variable_values(
                [self.wetting.p], iterate_index=0
            )
            s_w: np.ndarray = self.equation_system.get_variable_values(
                [self.wetting.s], iterate_index=0
            )
            p_g: np.ndarray = self.global_pressure(p_w, s_w)
            pp.shift_solution_values(
                name=GLOBAL_PRESSURE,
                data=data,
                location=pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                name=GLOBAL_PRESSURE,
                values=p_g,
                data=data,
                iterate_index=0,
            )

    def complimentary_pressure(
        self, s: np.ndarray, epsilon: float = 1e-6
    ) -> np.ndarray:
        r"""Compute the complimentary pressure from the primary variables.

        Cellwise it holds:
        ..math::
            p_G = - \int_{0}^{s} \frac{\lambda_n \lambda_w}{\lambda_t} p'_c ds.

        Parameters:
            s: np.ndarray
                Wetting saturation field.

        Returns:
            Complimentary pressure field.

        """
        # TODO: Change this s.t. p and s are ``pp.ad.DenseArray``.
        # Question: What to do with negative saturation values during Newton?
        # Limit s from below to 0 and from above to 1
        s[s > 1 - self.nonwetting.constants.residual_saturation() - epsilon] = 0
        intervals = np.linspace(
            np.full(s.shape, self.wetting.constants.residual_saturation() + epsilon),
            s,
            2,
        )[..., None]

        def func(s: np.ndarray) -> np.ndarray:
            """Note: We cannot use ``two_phase_flow.DarcyFluxes.phase_mobility`` and
            ``two_phase_flow.DarcyFluxes.total_mobility`` here, because they require
            upwinding.

            """
            # TODO: This is probably super inefficient!!!
            s_ad = pp.ad.DenseArray(s)
            w_mobility: pp.ad.DenseArray = self.rel_perm(
                s_ad, self.wetting
            ) / pp.ad.Scalar(self.wetting.constants.viscosity())
            n_mobility: pp.ad.DenseArray = self.rel_perm(
                s_ad, self.nonwetting
            ) / pp.ad.Scalar(self.nonwetting.constants.viscosity())
            t_mobility: pp.ad.DenseArray = w_mobility + n_mobility
            p_c: pp.ad.DenseArray = (
                w_mobility * n_mobility / t_mobility * self.cap_press_deriv(s_ad)
            )
            return p_c.value(self.equation_system)

        integral: Integral = self.quadrature.integrate(func, intervals)
        return -1 * integral.elementwise[..., 0]

    def eval_complimentary_pressure(self) -> None:
        """Evaluate the complimentary pressure field on the grid and store it in the data
        dictionary."""
        logger.info(f"Evaluating complimentary pressure.")
        for _, data in self.mdg.subdomains(return_data=True):
            s_w: np.ndarray = self.equation_system.get_variable_values(
                [self.wetting.s], iterate_index=0
            )
            p_c: np.ndarray = self.complimentary_pressure(s_w)
            pp.shift_solution_values(
                name=COMPLIMENTARY_PRESSURE,
                data=data,
                location=pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                name=COMPLIMENTARY_PRESSURE,
                values=p_c,
                data=data,
                iterate_index=0,
            )

    def set_boundary_pressures(self) -> None:
        """Set boundary pressures for the global and complimentary pressure fields."""
        for sd, _ in self.mdg.subdomains(return_data=True):
            # TODO: This is a bit of a mess. Ideally, bc_dirichlet_pressure_values gets
            # called with a boundary grid instead of a subdomain grid. Then we can loop
            # through boundaries. Alternatively, we implement
            # ``update_boundary_condition`` and ``create_boundary_operator`` in
            # ``BoundaryConditionsTPF`` and get the pressure and saturation values from
            # the data dictionary instead of calling the functions here.
            p_w_bc = self.bc_dirichlet_pressure_values(sd, self.wetting)
            s_w_bc = self.bc_dirichlet_saturation_values(sd, self.wetting)

            # Compute global and complimentary pressures on boundaries.
            p_g_bc = self.global_pressure(p_w_bc, s_w_bc)
            p_c_bc = self.complimentary_pressure(s_w_bc)

            # TODO See change above. If we loop over boundaries, we do not need to do
            # this next construction.
            _, bd = next(iter(self.mdg.boundaries(return_data=True)))
            for pressure_key, values in zip(
                [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE], [p_g_bc, p_c_bc]
            ):
                # pp.shift_solution_values(
                #     name=pressure_key,
                #     data=bd,
                #     location=pp.TIME_STEP_SOLUTIONS,
                #     max_index=len(self.time_step_indices),
                # )
                pp.set_solution_values(
                    name=pressure_key,
                    values=values,
                    data=bd,
                    time_step_index=0,
                )
                pp.set_solution_values(
                    name=pressure_key,
                    values=values,
                    data=bd,
                    iterate_index=0,
                )


class PressureReconstructionMixin:
    """Code and method are copied from Valera et al. (2024)."""

    mdg: pp.MixedDimensionalGrid
    iterate_indices: list
    bc_type: Callable[[pp.Grid], pp.BoundaryCondition]

    def reconstruct_pressure(
        self,
        pressure_key: Literal[GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
        flux_name: Literal["total", "capillary"],
    ) -> None:
        """Pressure reconstruction using the inverse of the numerical fluxes.

        Note: The flux field and pressure shield should correspond to each other, i.e.,
        if a phase pressure is used, then the phase flux should be used. For global
        pressure, we use the total flux and for complimentary pressure, we use the
        capillary flux.

        TODO For complimentary pressure, this is wrong!

        Note: Not implemented for phase pressures.

        Parameters:
            pressure_key: Name of the pressure field. Corresponds to pressure array
            stored in the data dictionary of the grid.
            flux_name: Name of the flux field. Corresponds to pressure array
            stored in the data dictionary of the grid.

        Stores two numpy arrays containing the values and Lagrangian coordinates of the
        reconstructed pressure for all elements of the subdomain grid in the data dict.

        """
        logger.info(f"Reconstrucing {pressure_key} pressure.")
        # FIXME Does it make sense to combine subdomains and boundaries like this?
        for sd, sd_data, _, bg_data in zip(
            *zip(*self.mdg.subdomains(return_data=True)),
            *zip(*self.mdg.boundaries(return_data=True)),
        ):
            # Retrieve finite volume cell-centered pressures
            p_cc = pp.get_solution_values(pressure_key, sd_data, iterate_index=0)
            assert p_cc.size == sd.num_cells

            # Retrieve topological data
            nc = sd.num_cells
            nf = sd.num_faces
            nn = sd.num_nodes

            # Perform reconstruction
            cell_nodes = sd.cell_nodes()
            cell_node_volumes = cell_nodes * sps.dia_matrix(
                arg1=(sd.cell_volumes, 0), shape=(nc, nc)
            )
            sum_cell_nodes = cell_node_volumes * np.ones(nc)
            cell_nodes_scaled = (
                sps.dia_matrix(arg1=(1.0 / sum_cell_nodes, 0), shape=(nn, nn))
                * cell_node_volumes
            )

            # Retrieve reconstructed velocities
            coeff = pp.get_solution_values(
                f"{flux_name}_flux_equilibrated_RT0_coeffs", sd_data, iterate_index=0
            )
            if sd.dim == 3:
                proj_flux = np.array(
                    [
                        coeff[:, 0] * sd.cell_centers[0] + coeff[:, 1],
                        coeff[:, 0] * sd.cell_centers[1] + coeff[:, 2],
                        coeff[:, 0] * sd.cell_centers[2] + coeff[:, 3],
                    ]
                )
            elif sd.dim == 2:
                proj_flux = np.array(
                    [
                        coeff[:, 0] * sd.cell_centers[0] + coeff[:, 1],
                        coeff[:, 0] * sd.cell_centers[1] + coeff[:, 2],
                    ]
                )
            else:
                proj_flux = np.array(
                    [
                        coeff[:, 0] * sd.cell_centers[0] + coeff[:, 1],
                    ]
                )

            # Obtain local gradients
            loc_grad = np.zeros((sd.dim, nc))
            perm = sd_data[pp.PARAMETERS]["total_flux"]["second_order_tensor"].values
            for ci in range(nc):
                loc_grad[: sd.dim, ci] = -np.linalg.inv(
                    perm[: sd.dim, : sd.dim, ci]
                ).dot(proj_flux[:, ci])

            # Obtaining nodal pressures
            cell_nodes_map = sps.find(sd.cell_nodes().T)[1]
            cell_node_matrix = cell_nodes_map.reshape(
                np.array([sd.num_cells, sd.dim + 1])
            )
            nodal_pressures = np.zeros(nn)

            for col in range(sd.dim + 1):
                nodes = cell_node_matrix[:, col]
                dist = sd.nodes[: sd.dim, nodes] - sd.cell_centers[: sd.dim]
                scaling = cell_nodes_scaled[nodes, np.arange(nc)]
                contribution = (
                    np.asarray(scaling) * (p_cc + np.sum(dist * loc_grad, axis=0))
                ).ravel()
                nodal_pressures += np.bincount(
                    nodes, weights=contribution, minlength=nn
                )

            # Treatment of boundary conditions
            # TODO How to do this?
            bc = sd_data[pp.PARAMETERS]["total_flux"]["bc"]

            bc_dir_values = np.zeros(sd.num_faces)
            external_dirichlet_boundary = np.logical_and(
                bc.is_dir, sd.tags["domain_boundary_faces"]
            )
            bc_pressure = bg_data[pp.ITERATE_SOLUTIONS][pressure_key][0]
            bg_dir_filter: np.ndarray = self.bc_type(sd).is_dir
            # bg_dir_filter = (
            #     bg_data[pp.ITERATE_SOLUTIONS]["bc_values_darcy_filter_dir"][0] == 1
            # )
            bc_dir_values[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]

            face_vec = np.zeros(nf)
            face_vec[external_dirichlet_boundary] = 1
            num_dir_face_of_node = sd.face_nodes * face_vec
            is_dir_node = num_dir_face_of_node > 0
            face_vec *= 0
            face_vec[external_dirichlet_boundary] = bc_dir_values[
                external_dirichlet_boundary
            ]
            node_val_dir = sd.face_nodes * face_vec
            node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
            nodal_pressures[is_dir_node] = node_val_dir[is_dir_node]

            # Export Lagrangian nodes and coordinates
            nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(
                sd.num_cells, sd.dim + 1
            )
            point_val = nodal_pressures[nodes_of_cell]
            point_coo = sd.nodes[:, nodes_of_cell]

            # Store in data dictionary
            for value, name in zip([point_val, point_coo], ["point_val", "point_coo"]):
                pp.shift_solution_values(
                    f"{pressure_key}_reconstructed_{name}",
                    sd_data,
                    pp.ITERATE_SOLUTIONS,
                    max_index=len(self.iterate_indices),
                )
                pp.set_solution_values(
                    f"{pressure_key}_reconstructed_{name}",
                    value,
                    sd_data,
                    iterate_index=0,
                )

    def gradient_mismatch(self, reconstructed_pressure) -> np.ndarray: ...


class EquilibratedFluxMixin:

    mdg: pp.MixedDimensionalGrid
    """Normally provided by a mixin of instance
    :class:`~porepy.models.geometry.ModelGeometry`.

    """
    # Variables:
    primary_pressure_var: str
    """Normally provided by a mixin of instance :class:`VariablesTPF` after calling
    :meth:`VariablesTPF.create_variables()`.

    """
    primary_saturation_var: str
    """Normally provided by a mixin of instance :class:`VariablesTPF` after calling
    :meth:`VariablesTPF.create_variables()`.

    """

    equation_system: pp.ad.EquationSystem
    """Normally provided by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`

    """

    iterate_indices: list

    def equilibrate_flux_during_Newton(
        self,
        flux_name: Literal["total", "capillary", PHASENAME],
        nonlinear_increment: Optional[np.ndarray] = None,
    ) -> None:
        """Equilibrates an approximate flux solution at a given Newton iteration.

        Parameters:
            flux_field: Name flux field to be equilibrated.
            nonlinear_increment: Nonlinear increment of the primary valuables. If
            already constructed during Newton, we can save some time instead of
            computing it again. Default is ``None``.

        Returns:
            equilibrated_flux: The equilibrated flux field.

        """
        # The functions ``DarcyFluxes.total_flux`` and ``DarcyFluxes.phase_flux``
        # incorporate bc values, hence ``d[f"{flux_name}_flux_value"]`` includes bc
        # values when set by ``eval_additional_vars`` and hence we do not need to care
        # about bc values here.
        logger.info(f"Equilibrating {flux_name} flux.")
        for _, d in self.mdg.subdomains(return_data=True):
            val: np.ndarray = pp.get_solution_values(
                f"{flux_name}_flux_value", d, iterate_index=1
            )
            jac: np.ndarray = pp.get_solution_values(
                f"{flux_name}_flux_jacobian", d, iterate_index=1
            )

            if nonlinear_increment is None:
                var_val: np.ndarray = self.equation_system.get_variable_values(
                    [self.primary_pressure_var, self.primary_saturation_var],
                    iterate_index=1,
                )
                var_val_new: np.ndarray = self.equation_system.get_variable_values(
                    [self.primary_pressure_var, self.primary_saturation_var],
                    iterate_index=0,
                )
                nonlinear_increment = var_val_new - var_val

            equilibrated_flux = val + jac @ nonlinear_increment
            pp.shift_solution_values(
                "flux_equilibrated",
                d,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"{flux_name}_flux_equilibrated",
                equilibrated_flux,
                d,
                iterate_index=0,
            )

    def extend_fv_fluxes(
        self,
        flux_name: Literal["total", "capillary", PHASENAME],
    ) -> None:
        """Extend equilibrated flux using RT0 basis functions.

        Parameters:
            mdg: pp.MixedDimensionalGrid
                Mixed-dimensional grid for the problem.

        Note:
            The data dictionary of each node of the grid bucket will be updated with the
            field d["estimates"]["recon_sd_flux"], a nd-array of shape
            (sd.num_cells x (sd.dim+1)) containing the coefficients of the reconstructed
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

        """
        logger.info(f"Extending {flux_name} flux to RT0 functions.")
        # Loop through all the nodes of the grid bucket
        for sd, d in self.mdg.subdomains(return_data=True):
            # Create key if it does not exist
            # TODO: is this necessary?
            # FIXME Should be done for sub dir pp.ITERATE_SOLUTIONS or
            # pp.TIME_STEP_SOLUTIONS!
            if not f"{flux_name}_flux_equilibrated_RT0_coeffs" in d:
                d[f"{flux_name}_flux_equilibrated_RT0_coeffs"] = {}

            # Cell-basis arrays
            cell_faces_map = sps.find(sd.cell_faces.T)[1]
            # TODO: This depends on the shape of the grid. For now, we assume a
            # triangular grid.
            faces_cell = cell_faces_map.reshape(sd.num_cells, sd.dim + 1)
            opp_nodes_cell = get_opposite_side_nodes(sd)
            opp_nodes_coor_cell = sd.nodes[:, opp_nodes_cell]
            sign_normals_cell = get_sign_normals(sd)
            vol_cell = sd.cell_volumes

            # Retrieve finite volume fluxes
            flux = pp.get_solution_values(
                f"{flux_name}_flux_equilibrated", d, iterate_index=0
            )
            # Perform actual reconstruction and obtain coefficients
            coeffs = np.empty([sd.num_cells, sd.dim + 1])
            alpha = 1 / (sd.dim * vol_cell)
            coeffs[:, 0] = alpha * np.sum(sign_normals_cell * flux[faces_cell], axis=1)
            for dim in range(sd.dim):
                coeffs[:, dim + 1] = -alpha * np.sum(
                    (sign_normals_cell * flux[faces_cell] * opp_nodes_coor_cell[dim]),
                    axis=1,
                )

            # Store coefficients in the data dictionary.
            pp.shift_solution_values(
                f"{flux_name}_flux_equilibrated_RT0_coeffs",
                d,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"{flux_name}_flux_equilibrated_RT0_coeffs",
                coeffs,
                d,
                iterate_index=0,
            )

    def divergence_mismatch(
        self, equilibrated_flux: pp.ad.SparseArray
    ) -> pp.ad.SparseArray:
        r"""Calculate mismatch of the equilibrated flux from being in :math:`H(div)` and
        being mass conservative.

        We calculate how far the equilibrated total flux :math:`\sigma_t` is from
        satisfying
        ..math::
            (q_t^n - \nabla \cdot \sigma_t, 1)_K = 0,

        and how far the equilibrated wetting flux :math:`\sigma_w` is from satisfying
        ..math::
            (q_w^n - \partial_t^n(\varphi s_{w, h, \tau)}) - \nabla \cdot
            \sigma_w, 1)_K = 0,

        where :math:`n` is the discrete time step and :math:`K` is a given grid cell.

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

        g = self.mdg.subdomains()[0]

        # Spatial discretization operators.
        div = pp.ad.Divergence([g])
        flux_mpfa = pp.ad.MpfaAd(self.flux_key, [g])

        # Time derivatives.
        dt = pp.ad.Scalar(self.time_manager.dt)
        dt_s: pp.ad.Operator = pp.ad.time_derivatives.dt(self.wetting.s, dt)

        # Ad source.
        source_ad_w = pp.ad.DenseArray(self.phase_fluid_source(g, self.wetting))
        source_ad_t = pp.ad.DenseArray(self.total_fluid_source(g))

        # Ad parameters.
        porosity_ad = pp.ad.DenseArray(self._porosity(g))

        # Compute cap pressure and relative permeabilities.
        p_cap = self.cap_press(self.wetting.s)
        # p_cap_bc = pp.ad.DenseArray(self._bc_values_cap_press(g))

        mobility_w = self.phase_mobility(g, self.wetting)
        mobility_n = self.phase_mobility(g, self.nonwetting)
        mobility_t = self.total_mobility(g)

        # Ad equations
        if self.formulation == "fractional_flow":
            # Note, that for ``flux_t``, the total mobility is already included.
            flux_t = self.total_flux(g)
            fractional_flow_w = mobility_w / mobility_t
            vector_source_w = pp.ad.DenseArray(self.vector_source(g, self.wetting))
            vector_source_n = pp.ad.DenseArray(self.vector_source(g, self.nonwetting))

            flow_equation = div @ flux_t - self.volume_integral(source_ad_t, [g], 1)
            transport_equation = (
                porosity_ad * (self.volume_integral(dt_s, [g], 1))
                + div
                @ (
                    fractional_flow_w * flux_t
                    + fractional_flow_w
                    * mobility_n
                    * (
                        flux_mpfa.flux() @ p_cap
                        # TODO: Plus boundary values here or are they included in the total flux?
                        + flux_mpfa.vector_source() @ vector_source_w
                        - flux_mpfa.vector_source() @ vector_source_n
                    )
                )
                - self.volume_integral(source_ad_w, [g], 1)
            )
        flow_equation.set_name("Flow equation")
        transport_equation.set_name("Transport equation")


class ReconstructionSolutionStrategy(SolutionStrategyTPF):

    global_pressure: Callable[[pp.Grid], pp.ad.Operator]
    eval_global_pressure: Callable[[], None]
    complimentary_pressure: Callable[[pp.Grid], pp.ad.Operator]
    eval_complimentary_pressure: Callable[[], None]
    set_global_pressure_constants: Callable[[], None]
    set_boundary_pressures: Callable[[], None]

    total_flux: Callable[[pp.Grid], pp.ad.Operator]
    phase_flux: Callable[[pp.Grid, Phase], pp.ad.Operator]

    wetting: Phase
    nonwetting: Phase
    phases: list[Phase]

    time_manager: pp.TimeManager

    reconstruct_pressure: Callable[[Literal, Literal], None]
    equilibrate_flux_during_Newton: Callable[
        [Literal["total", "capillary", PHASENAME]], None
    ]
    extend_fv_fluxes: Callable[[Literal["total", "capillary", PHASENAME]], None]
    compute_errors: Callable

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.set_global_pressure_constants()

    @property
    def iterate_indices(self) -> np.ndarray:
        """Indices for storing iterate solutions. To construct the reconstructions we
        need to store the previous iterate as well."""
        return np.array([0, 1])

    def prepare_simulation(self) -> None:
        super().prepare_simulation()
        # Run once to initialize flux, global, and complimentary pressure data.
        self.eval_additional_vars()
        self.set_boundary_pressures()

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        super().after_nonlinear_iteration(nonlinear_increment)
        # Only do this for values that make sense.
        # FIXME Think about how this should be.
        if (
            self.time_manager.time_index > 1
            or self.nonlinear_solver_statistics.num_iteration > 1
        ):
            self.eval_additional_vars()
            self.postprocess_solution()

    def eval_additional_vars(self) -> None:
        """Evaluate additional pressure and flux variables and save in data dictionary
        after each iteration.

        Populates the data dictionary with:
        - 'global_pressure': Global pressure value.
        - 'complimentary_pressure': Complimentary pressure value.
        - 'total_flux_val', 'total_flux_jac': Total flux value and Jacobian.
        - '{phase.name}_flux_val','{phase.name}_flux_jac': Phase Flux value and
        Jacobian.

        """
        for sd, d in self.mdg.subdomains(return_data=True):
            # TODO Ideally, we use ``equation_system.set_solution_values()`` and
            # ``equation_system.shift_solution_values()``.

            # Save FV P0 pressures.
            self.eval_global_pressure()
            self.eval_complimentary_pressure()

            # Multiply Jacobians with the the following matrix, s.t., the Jacobian is
            # only w.r.t. the primary variables.
            # FIXME It would be more efficient to NOT assemble the full Jacobian.
            # Possibly, by not setting secondary pressure and saturation as variables,
            # but just evaluating their values after each nonlinear iteration and
            # storing them in the data dict?
            column_projection = self.equation_system.projection_to(
                [self.primary_pressure_var, self.primary_saturation_var]
            ).transpose()

            # Save FV P0 normal fluxes and Jacobians.
            t_flux: pp.ad.AdArray = self.total_flux(sd).value_and_jacobian(
                self.equation_system
            )
            pp.shift_solution_values(
                "total_flux_value",
                d,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values("total_flux_value", t_flux.val, d, iterate_index=0)
            pp.shift_solution_values(
                "total_flux_jacobian",
                d,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                "total_flux_jacobian",
                t_flux.jac * column_projection,
                d,
                iterate_index=0,
            )

            for phase in self.phases:
                p_flux: pp.ad.AdArray = self.phase_flux(sd, phase).value_and_jacobian(
                    self.equation_system
                )
                pp.shift_solution_values(
                    f"{phase.name}_flux_value",
                    d,
                    pp.ITERATE_SOLUTIONS,
                    max_index=len(self.iterate_indices),
                )
                pp.set_solution_values(
                    f"{phase.name}_flux_value", p_flux.val, d, iterate_index=0
                )
                pp.shift_solution_values(
                    f"{phase.name}_flux_jacobian",
                    d,
                    pp.ITERATE_SOLUTIONS,
                    max_index=len(self.iterate_indices),
                )
                pp.set_solution_values(
                    f"{phase.name}_flux_jacobian",
                    p_flux.jac * column_projection,
                    d,
                    iterate_index=0,
                )

            # TODO When we have buoyancy terms, the capillary flux must NOT include
            # gravity flux!!! Thus, we need to calculate it in a slightly more
            # complicated way
            # FIXME This is wrong!
            p_c_flux_val: np.ndarray = pp.get_solution_values(
                f"{self.nonwetting.name}_flux_value", d, iterate_index=0
            ) - pp.get_solution_values(
                f"{self.wetting.name}_flux_value", d, iterate_index=0
            )
            pp.shift_solution_values(
                f"capillary_flux_value",
                d,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"capillary_flux_value", p_c_flux_val, d, iterate_index=0
            )
            p_c_flux_jac: np.ndarray = pp.get_solution_values(
                f"{self.nonwetting.name}_flux_jacobian", d, iterate_index=0
            ) - pp.get_solution_values(
                f"{self.wetting.name}_flux_jacobian", d, iterate_index=0
            )
            pp.shift_solution_values(
                f"capillary_flux_jacobian",
                d,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"capillary_flux_jacobian",
                p_c_flux_jac,
                d,
                iterate_index=0,
            )

    def postprocess_solution(self) -> None:
        for flux_name in ["total", "capillary"]:
            self.equilibrate_flux_during_Newton(flux_name)
            self.extend_fv_fluxes(flux_name)
        for pressure_key, flux_name in zip(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE], ["total", "capillary"]
        ):
            self.reconstruct_pressure(pressure_key, flux_name)

        # self.compute_errors()


def get_opposite_side_nodes(sd: pp.Grid) -> np.ndarray:
    """Computes opposite side nodes for each face of each cell in the grid.

    Parameters:
        sd: pp.Grid
            Subdomain grid.

    Returns:
        Opposite nodes with rows representing the cell number and columns
        representing the opposite side node index of the face. The size of the array
        is (sd.num_cells x (sd.dim + 1)).

    """
    dim = sd.dim
    nc = sd.num_cells
    nf = sd.num_faces

    faces_of_cell = sps.find(sd.cell_faces.T)[1].reshape(nc, dim + 1)
    nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(nc, dim + 1)
    nodes_of_face = sps.find(sd.face_nodes.T)[1].reshape(nf, dim)

    opposite_nodes = np.empty_like(faces_of_cell)
    for cell in range(sd.num_cells):
        opposite_nodes[cell] = [
            np.setdiff1d(nodes_of_cell[cell], nodes_of_face[face])[0]
            for face in faces_of_cell[cell]
        ]

    return opposite_nodes


def get_sign_normals(sd: pp.Grid) -> np.ndarray:
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
        sd: pp.Grid
            Subdomain grid.

    Returns:
        Sign of the face normal. 1 if the signs of the local and global normals are
        the same, -1 otherwise. The size of the np.ndarray is `sd.num_faces`.

    """
    # Faces associated to each cell
    faces_cell = sps.find(sd.cell_faces.T)[1].reshape(sd.num_cells, sd.dim + 1)

    # Face centers coordinates for each face associated to each cell
    face_center_cells = sd.face_centers[:, faces_cell]

    # Global normals of the faces per cell
    global_normal_faces_cell = sd.face_normals[:, faces_cell]

    # Computing the local outer normals of the faces per cell. To do this, we first
    # assume that n_loc = n_glb, and then we fix the sign. To fix the sign,we compare
    # the length of two vectors, the first vector v1 = face_center - cell_center,
    # and the second vector v2 is a prolongation of v1 in the direction of the
    # normal. If ||v2||<||v1||, then the  normal of the face in question is pointing
    # inwards, and we needed to flip the sign.
    local_normal_faces_cell = global_normal_faces_cell.copy()
    v1 = face_center_cells - sd.cell_centers[:, :, np.newaxis]
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
