from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps
from tpf_lab.models.phase import Phase
from tpf_lab.models.two_phase_flow import SolutionStrategyTPF


def r2c(array: np.ndarray) -> np.ndarray:
    """Reshape a 1d array into a column vector"""
    return array.reshape(-1, 1)


class PressureReconstructionMixin:
    """Code and method are copied from Valera et al. (2024)."""

    def reconstruct_pressure(
        sd: pp.Grid, sd_data: dict, bg_data: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pressure reconstruction using the inverse of the numerical fluxes.

        Parameters:
            sd: pp.Grid
                Subdomain grid.
            sd_data: dict
                Subdomain data dictionary.
            bg_data: dict
                Boundary grid data dictionary.

        Returns:
            2-tuple of numpy arrays containing the values and Lagrangian coordinates of
            the reconstructed pressure for all elements of the subdomain grid.

        """
        # Retrieve finite volume cell-centered pressures
        p_cc = sd_data["estimates"]["fv_sd_pressure"]
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
        coeff = sd_data["estimates"]["recon_sd_flux"]
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
        perm = sd_data[pp.PARAMETERS]["flow"]["second_order_tensor"].values
        for ci in range(nc):
            loc_grad[: sd.dim, ci] = -np.linalg.inv(perm[: sd.dim, : sd.dim, ci]).dot(
                proj_flux[:, ci]
            )

        # Obtaining nodal pressures
        cell_nodes_map = sps.find(sd.cell_nodes().T)[1]
        cell_node_matrix = cell_nodes_map.reshape(np.array([sd.num_cells, sd.dim + 1]))
        nodal_pressures = np.zeros(nn)

        for col in range(sd.dim + 1):
            nodes = cell_node_matrix[:, col]
            dist = sd.nodes[: sd.dim, nodes] - sd.cell_centers[: sd.dim]
            scaling = cell_nodes_scaled[nodes, np.arange(nc)]
            contribution = (
                np.asarray(scaling) * (p_cc + np.sum(dist * loc_grad, axis=0))
            ).ravel()
            nodal_pressures += np.bincount(nodes, weights=contribution, minlength=nn)

        # Treatment of boundary conditions
        bc = sd_data[pp.PARAMETERS]["flow"]["bc"]

        bc_dir_values = np.zeros(sd.num_faces)
        external_dirichlet_boundary = np.logical_and(
            bc.is_dir, sd.tags["domain_boundary_faces"]
        )
        bc_pressure = bg_data[pp.ITERATE_SOLUTIONS]["pressure"][0]
        bg_dir_filter = (
            bg_data[pp.ITERATE_SOLUTIONS]["bc_values_darcy_filter_dir"][0] == 1
        )
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
        nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(sd.num_cells, sd.dim + 1)
        point_val = nodal_pressures[nodes_of_cell]
        point_coo = sd.nodes[:, nodes_of_cell]

        return point_val, point_coo

    def gradient_mismatch(self, reconstructed_pressure) -> np.ndarray: ...


class EquilibratedFluxMixin:

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

    def equilibrate_flux_during_Newton(
        self,
        flux_field: pp.ad.Operator,
        nonlinear_increment: Optional[np.ndarray] = None,
    ) -> None:
        """Equilibrates an approximate flux solution at a given Newton iteration.

        Parameters:
            flux_field: The flux field to be equilibrated.
            nonlinear_increment: Nonlinear increment of the primary valuables. If
            already constructed during Newton, we can save some time instead of
            computing it again. Default is ``None``.

        Returns:
            equilibrated_flux: The equilibrated flux field.

        """
        # Implement the flux equilibration logic here

        flux_field_prev: pp.ad.AdArray = (
            flux_field.previous_iteration().value_and_jacobian(self.equation_system)
        )
        val, jac = flux_field_prev.val, flux_field_prev.jac

        if nonlinear_increment is None:
            vars: list[pp.ad.Variable] = self.equation_system.get_variables(
                [self.primary_pressure_var, self.primary_saturation_var]
            )
            nonlinear_increment = np.concatenate(
                [
                    np.array((v - v.previous_iteration()).value(self.equation_system))
                    for v in vars
                ]
            )

        equilibrated_flux = val + jac @ nonlinear_increment
        return pp.ad.SparseArray(equilibrated_flux)

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

    def extend_fv_fluxes(mdg: pp.MixedDimensionalGrid) -> None:
        """Extend normal fluxes using RT0 basis functions.

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
        # Loop through all the nodes of the grid bucket
        for sd, d in mdg.subdomains(return_data=True):
            # Create key if it does not exist
            if d["estimates"].get("recon_sd_flux") is None:
                d["estimates"]["recon_sd_flux"] = {}

            # Cell-basis arrays
            cell_faces_map = sps.find(sd.cell_faces.T)[1]
            faces_cell = cell_faces_map.reshape(sd.num_cells, sd.dim + 1)
            opp_nodes_cell = get_opposite_side_nodes(sd)
            opp_nodes_coor_cell = sd.nodes[:, opp_nodes_cell]
            sign_normals_cell = get_sign_normals(sd)
            vol_cell = sd.cell_volumes

            # Retrieve finite volume fluxes
            flux = d["estimates"]["fv_sd_flux"]

            # Perform actual reconstruction and obtain coefficients
            coeffs = np.empty([sd.num_cells, sd.dim + 1])
            alpha = 1 / (sd.dim * vol_cell)
            coeffs[:, 0] = alpha * np.sum(sign_normals_cell * flux[faces_cell], axis=1)
            for dim in range(sd.dim):
                coeffs[:, dim + 1] = -alpha * np.sum(
                    (sign_normals_cell * flux[faces_cell] * opp_nodes_coor_cell[dim]),
                    axis=1,
                )

            # Store coefficients in the data dictionary
            d["estimates"]["recon_sd_flux"] = coeffs


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


class ReconstructionSolutionStrategy(SolutionStrategyTPF):

    global_pressure: Callable[[pp.Grid], pp.ad.Operator]
    complimentary_pressure: Callable[[pp.Grid], pp.ad.Operator]

    total_flux: Callable[[pp.Grid], pp.ad.Operator]
    phase_flux: Callable[[pp.Grid, Phase], pp.ad.Operator]

    phases: list[Phase]

    reconstruct_pressure: Callable[[pp.Grid, dict, dict], tuple[np.ndarray, np.ndarray]]
    equilibrate_flux: Callable[[pp.ad.Operator], pp.ad.SparseArray]
    compute_errors: Callable

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        super().after_nonlinear_iteration(nonlinear_increment)

    def transfer_solution(self) -> None:
        """Transfers solution data to subdomains dictionary after each iteration.

        Populates the 'estimates' dictionary with:
        - 'global_pressure': Global pressure value.
        - 'complimentary_pressure': Complimentary pressure value.
        - 'total_flux': Total flux value.
        - '{phase.name}_flux': Flux value for each phase.

        Note:
            Creates 'estimates' key if it does not exist.

        Returns:
            None

        """
        self.equation_system.shift_iterate_values(variables=[])
        for sd, d in self.mdg.subdomains(return_data=True):
            # Create key if it does not exist
            if not "estimates" in d is None:
                d["estimates"] = {}

            # Save FV P0 pressures.
            d["estimates"]["global_pressure"] = self.global_pressure(sd).value(
                self.equation_system
            )
            d["estimates"]["complimentary_pressure"] = self.complimentary_pressure(
                sd
            ).value(self.equation_system)

            # Save FV P0 normal fluxes.
            d["estimates"][f"total_flux"] = self.total_flux(sd).value(
                self.equation_system
            )
            for phase in self.phases:
                d["estimates"][f"{phase.name}_flux"] = self.phase_flux(sd, phase).value(
                    self.equation_system
                )

    def postprocess_solution(self) -> None:
        self.transfer_solution()
        self.reconstruct_pressure()
        self.equilibrate_flux()
        self.compute_errors()
