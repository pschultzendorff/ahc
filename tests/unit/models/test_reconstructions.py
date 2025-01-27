import numpy as np
import pytest
from tpf.models.reconstruction import EquilibratedFluxMixin, PressureReconstructionMixin


class TestPressureReconstruction(PressureReconstructionMixin):
    def test_integral_of_postprocessed_pressures(self):
        # Create a mock grid and pressure values
        grid = self.create_mock_grid()
        cellwise_pressure = np.random.rand(grid.num_cells)

        # Perform pressure reconstruction
        postprocessed_pressure = self.reconstruct_pressure_vohralik(
            grid, cellwise_pressure
        )

        # Compute integrals of the postprocessed pressures
        integral_postprocessed_pressure = (
            self.compute_integral_of_postprocessed_pressures(
                grid, postprocessed_pressure
            )
        )

        # Check if the integrals equal the cellwise constant pressure values
        np.testing.assert_allclose(
            integral_postprocessed_pressure, cellwise_pressure, rtol=1e-5
        )

    def create_mock_grid(self):
        # Create a mock grid for testing purposes
        class MockGrid:
            def __init__(self):
                self.num_cells = 10
                self.cell_volumes = np.ones(self.num_cells)

        return MockGrid()

    def compute_integral_of_postprocessed_pressures(self, grid, postprocessed_pressure):
        # Compute the integral of the postprocessed pressures
        return np.sum(postprocessed_pressure * grid.cell_volumes, axis=1)


# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
