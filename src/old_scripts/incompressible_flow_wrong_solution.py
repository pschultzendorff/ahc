import numpy as np
import scipy.sparse as sps
import porepy as pp


class ChangedSetup(pp.IncompressibleFlow):
    def create_grid(self) -> None:
        GRID_SIZE: int = 10
        PHYS_SIZE: int = 10
        cell_dims: np.ndarray = np.array(
            [
                GRID_SIZE,
                GRID_SIZE,
            ]
        )
        phys_dims: np.ndarray = np.array(
            [
                PHYS_SIZE,
                PHYS_SIZE,
            ]
        )
        g_cart: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims)
        g_cart.compute_geometry()
        # g_tetra: pp.TetrahedralGrid = pp.TetrahedralGrid(g_cart.nodes)
        # g_tetra.compute_geometry()
        self.mdg: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_cart]])
        self.box: dict = pp.bounding_box.from_points(
            np.array(
                [
                    [
                        0,
                        0,
                    ],
                    [
                        GRID_SIZE,
                        GRID_SIZE,
                    ],
                ]
            ).T
        )

    def _source(self, g: pp.Grid) -> np.ndarray:
        array = np.zeros(g.num_cells)
        array[55] = 1
        array[75] = -1
        # array[189:191] = 1
        # array[209:211] = 1
        return array

    def _bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries."""
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "neu")

    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values."""
        return np.full(g.num_faces, 0)


model = ChangedSetup({"folder_name": "test_run", "file_name": "IncompressibleFlow"})
model.prepare_simulation()
model.assemble_linear_system()
A, b = model.linear_system
print(
    A[
        10:20,
    ]
)
print(b)
sol = sps.linalg.spsolve(A, b)
ones = np.full(100, 1)
print(f"A(1,\dots,1)^T {A @ ones}")
print(f"Az (here z is the solution) {A @ sol}")
print(f"A(z[0],\dots,z[0]) (here z is the solution) {A @ (ones*sol[0])}")
print(f"A(z-(z[0],\dots,z[0])) (here z is the solution) {A @ (sol - ones*sol[0])}")
print((A @ sol)[50:60])
print(sol)
print(b)
# pp.run_stationary_model(model, {})
