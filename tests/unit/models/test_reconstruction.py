from typing import cast

import numpy as np
import porepy as pp
import pytest
from ahc.models.reconstruction import EquilibratedFluxMixin
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.abstract_equations import BalanceEquation
from porepy.models.constitutive_laws import DimensionReduction
from porepy.models.geometry import ModelGeometry


# Subclass all classes required to call volume_integral.
class MockModel(
    EquilibratedFluxMixin, DimensionReduction, ModelGeometry, BalanceEquation
):
    def __init__(self, params: dict) -> None:
        # Setup the required attributes to run setup_flux_equilibration.
        self.nd = 2
        domain = nd_cube_domain(self.nd, 1.0)
        fracture_network = pp.create_fracture_network([], domain)
        self.mdg = pp.create_mdg("simplex", {"cell_size": 0.1}, fracture_network)
        self.g = self.mdg.subdomains()[0]
        self.equation_system = pp.EquationSystem(self.mdg)

        self._nl_appleyard_chopping = True
        self._nl_enforce_physical_saturation = True

        self.setup_flux_equilibration()

        # Add a mock saturation variable to the equation system.
        self.equation_system.create_variables(
            "saturation", dof_info={"cells": 1}, subdomains=[self.g]
        )
        self.equation_system.set_variable_values(
            np.zeros(self.g.num_cells),
            ["saturation"],
            time_step_index=0,
            iterate_index=0,
        )

        self.time_manager = pp.TimeManager(
            schedule=[0.0, 1.0], dt_init=1.0, dt_min_max=(0.1, 1.0)
        )

    def porosity(self, grid: pp.Grid) -> np.ndarray:
        rng = np.random.default_rng(0)
        return 0.4 + rng.random(grid.num_cells) * 0.1


@pytest.mark.parametrize("seed", [0, 1, 2])
class TestEquilibratedFluxMixin:
    @pytest.fixture
    def mock_model(self) -> MockModel:
        # Ignore mypy containing about abstract classes.
        return MockModel({})  # type: ignore

    def test_equilibrate_increment_diff(self, seed: int, mock_model: MockModel) -> None:
        rng = np.random.default_rng(seed)
        nonlinear_increment_diff = rng.random(mock_model.g.num_cells)

        equilibrated_flux = mock_model.equilibrate_increment_diff(
            nonlinear_increment_diff
        )
        # Scale the nonlinear increment
        rhs = (
            mock_model.porosity(mock_model.g)
            * mock_model.g.cell_volumes
            * nonlinear_increment_diff
        ) / mock_model.time_manager.dt

        np.testing.assert_allclose(
            mock_model.D @ equilibrated_flux, rhs, rtol=1e-15, atol=1e-15
        )

    def test_equilibrate_increment_diff_in_equation(
        self, seed: int, mock_model: MockModel
    ) -> None:
        rng = np.random.default_rng(seed)
        nonlinear_increment_diff = rng.random(mock_model.g.num_cells)

        flux_w_equil = pp.ad.DenseArray(
            mock_model.equilibrate_increment_diff(nonlinear_increment_diff)
        )

        mock_model.equation_system.set_variable_values(
            nonlinear_increment_diff, ["saturation"], iterate_index=0
        )
        dt_s = pp.ad.time_derivatives.dt(
            mock_model.equation_system.variables[0],
            pp.ad.Scalar(mock_model.time_manager.dt),
        )

        div = pp.ad.Divergence([mock_model.g], dim=1, name="divergence")
        # Ad source.
        source_ad_w = pp.ad.DenseArray(
            np.zeros(mock_model.g.num_cells), name="source_w"
        )

        # Ad parameters.
        porosity_ad = pp.ad.DenseArray(mock_model.porosity(mock_model.g))

        # Ad flux.
        flux_w_equil_mismatch = (
            porosity_ad * (mock_model.volume_integral(dt_s, [mock_model.g], 1))
            - div @ flux_w_equil
            - source_ad_w
        )
        flux_w_equil_mismatch_value = cast(
            np.ndarray,
            flux_w_equil_mismatch.value(mock_model.equation_system),
        )

        np.testing.assert_allclose(
            flux_w_equil_mismatch_value,
            np.zeros_like(flux_w_equil_mismatch_value),
            rtol=1e-15,
            atol=1e-15,
        )


class TestPressureReconstruction:
    def test_integral_of_postprocessed_pressures(self, mock_mdg: pp.Grid) -> None:
        # Create a mock grid and pressure values
        cellwise_pressure = np.random.rand(mock_mdg.num_cells)

        # Perform pressure reconstruction
        postprocessed_pressure = self.reconstruct_pressure_vohralik(
            mock_mdg, cellwise_pressure
        )

        # Compute integrals of the postprocessed pressures
        integral_postprocessed_pressure = (
            self.compute_integral_of_postprocessed_pressures(
                mock_mdg, postprocessed_pressure
            )
        )

        # Check if the integrals equal the cellwise constant pressure values
        np.testing.assert_allclose(
            integral_postprocessed_pressure, cellwise_pressure, rtol=1e-5
        )

    def compute_integral_of_postprocessed_pressures(self, grid, postprocessed_pressure):
        # Compute the integral of the postprocessed pressures
        return np.sum(postprocessed_pressure * grid.cell_volumes, axis=1)
