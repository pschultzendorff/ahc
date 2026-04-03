import functools
import itertools
import logging
import pathlib
import typing
from typing import Literal

import numpy as np
import porepy as pp
from ahc.models.error_estimate import ErrorEstimatesTwoPhaseFlow
from ahc.models.protocol import EstimatesProtocol
from ahc.models.reconstruction import (
    RecDataSavingMixin,
    evaluate_poly_at_points,
)
from ahc.numerics.quadrature import Integral
from ahc.utils.constants_and_typing import (
    COMPLEMENTARY_PRESSURE,
    FLUX_NAME,
    GLOBAL_PRESSURE,
    PRESSURE_KEY,
    TOTAL_FLUX,
    WETTING_FLUX,
)
from ahc.viz.plot_quadratic_pressures import plot_quadratic_pressures
from porepy.viz.exporter import DataInput

logger = logging.getLogger(__name__)

# If the local estimators are very small (~1e-160), taking a square during their
# computation will result in an underflow error. These errors should NOT be raised.
# Treating the local estimators as zero is fine.
np.seterr(under="ignore")


class ErrorEstimateAnalyticsMixin(EstimatesProtocol):
    """Analytics and debugging functionality for :class:`ErrorEstimates`."""

    # NOTE This is purely a mixin, i.e., has no concrete superclass. We ignore mypy
    # complaining about missing methods or calls to abstract methods with trivial body
    # in superclass.

    def local_pressure_potential(self, pressure_key: PRESSURE_KEY) -> None:
        def evaluate_potential_from_coeffs(
            x: np.ndarray,
            pressure_coeffs: np.ndarray,
        ) -> np.ndarray:
            """Calculate the total flux from reconstructed pressures and P0
            mobilities."""
            return self._evaluate_pressure_potential_at_points(
                pressure_coeffs,
                x[..., 0],
                x[..., 1],
            )

        pressure_coeffs_postproc = pp.get_solution_values(
            f"{pressure_key}_coeffs_postproc", self.g_data, iterate_index=0
        )
        pressure_coeffs_rec = pp.get_solution_values(
            f"{pressure_key}_coeffs_rec", self.g_data, iterate_index=0
        )

        def integrand(
            x: np.ndarray,
            pressure_coeffs: np.ndarray,
        ) -> np.ndarray:
            pressure_potential = evaluate_potential_from_coeffs(x, pressure_coeffs)
            return np.sqrt((pressure_potential**2).sum(axis=-1))

        # 5: Finally, integrate in space. Store norm at current time step and inner
        # product of previous and current time step.
        for specifier in ["_postproc", "_rec"]:
            pressure_coeffs = (
                pressure_coeffs_postproc
                if specifier == "_postproc"
                else pressure_coeffs_rec
            )
            integral: Integral = self.quadrature_est.integrate(
                functools.partial(integrand, pressure_coeffs=pressure_coeffs),
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"{pressure_key}{specifier}_potential",
                integral.elementwise.squeeze(),
                self.g_data,
                iterate_index=0,
            )

    def local_darcy_est_from_postproc(self, flux_name: FLUX_NAME) -> None:
        r"""Calculate and store the local Darcy estimator for each element at the
         current time and the inner product between current and previous time step.

        Note: The values stored are the squares of the elementwise estimators, i.e.,

        .. math::
            \|\mathbf{u}_\mathrm{t} 
                - \kappa \lambda_\mathrm{t} \nabla \tilde{P}_{h,\tau}\|_K^2(t^n), \\
            \|\mathbf{u}_\mathrm{w} 
                - \kappa \left(\lambda_\mathrm{w} \nabla \tilde{P}_{h,\tau}
                + \nabla \tilde{Q}_{h,\tau}\right)\|_K^2(t^n).

        and 

        .. math::
            \left(
                (\mathbf{u}_\mathrm{t} - \kappa \lambda_\mathrm{t} \nabla \tilde{P}_{h,\tau})(t^n),
                (\mathbf{u}_\mathrm{t} - \kappa \lambda_\mathrm{t} \nabla \tilde{P}_{h,\tau})(t^{n-1})
            \right), \\
            \left(
                (\mathbf{u}_\mathrm{w} - \kappa (\lambda_\mathrm{w} \nabla \tilde{P}_{h,\tau}
                + \nabla \tilde{Q}_{h,\tau}))(t^n),
                (\mathbf{u}_\mathrm{w} - \kappa (\lambda_\mathrm{w} \nabla \tilde{P}_{h,\tau}
                + \nabla \tilde{Q}_{h,\tau}))(t^{n-1})
            \right).

        """
        # 1. Get a scalar permeability array.
        perm: np.ndarray | dict[str, np.ndarray] = self.permeability(self.g)
        if isinstance(perm, np.ndarray):
            perm_arr = perm
        elif len(perm) == 1:
            perm_arr = perm["kxx"]
        else:
            raise ValueError("Not implemented for tensor permeability.")

        # 2: Get reconstructed pressure coefficients and mobilities.
        if flux_name == TOTAL_FLUX:
            pressure_keys = [GLOBAL_PRESSURE]
            mobility_keys = ["total_mobility"]
        elif flux_name == WETTING_FLUX:
            pressure_keys = [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE]
            mobility_keys = ["total_mobility", "fractional_flow"]
        else:
            raise ValueError(f"Unknown flux name: {flux_name}")

        pressure_coeffs_new: dict[str, np.ndarray] = {}
        pressure_coeffs_old: dict[str, np.ndarray] = {}
        mobilities_new: dict[str, np.ndarray] = {}
        mobilities_old: dict[str, np.ndarray] = {}

        for pressure_key in pressure_keys:
            pressure_coeffs_new[pressure_key] = pp.get_solution_values(
                f"{pressure_key}_coeffs_postproc", self.g_data, iterate_index=0
            )
            pressure_coeffs_old[pressure_key] = pp.get_solution_values(
                f"{pressure_key}_coeffs_postproc", self.g_data, time_step_index=0
            )

        for mobility_key in mobility_keys:
            mobilities_new[mobility_key] = pp.get_solution_values(
                mobility_key, self.g_data, iterate_index=0
            )
            mobilities_old[mobility_key] = pp.get_solution_values(
                mobility_key, self.g_data, time_step_index=0
            )

        # 3: Define helper functions to evaluate fluxes from reconstructed pressures and
        # P0 mobilities.
        if flux_name == TOTAL_FLUX:

            def evaluate_flux_from_reconstructions(
                x: np.ndarray,
                pressure_coeffs: dict[str, np.ndarray],
                phase_mobilities: dict[str, np.ndarray],
            ) -> np.ndarray:
                """Calculate the total flux from reconstructed pressures and P0
                mobilities."""
                global_pressure_pot: np.ndarray = (
                    self._evaluate_pressure_potential_at_points(
                        pressure_coeffs[GLOBAL_PRESSURE], x[..., 0], x[..., 1]
                    )
                )
                total_mobility = phase_mobilities["total_mobility"]
                return (
                    -perm_arr[None, :, None]
                    * total_mobility[None, :, None]
                    * global_pressure_pot
                )

        elif flux_name == WETTING_FLUX:

            def evaluate_flux_from_reconstructions(
                x: np.ndarray,
                pressure_coeffs: dict[str, np.ndarray],
                phase_mobilities: dict[str, np.ndarray],
            ) -> np.ndarray:
                """Calculate the total flux from reconstructed pressures and P0
                mobilities."""
                global_pressure_pot = self._evaluate_pressure_potential_at_points(
                    pressure_coeffs[GLOBAL_PRESSURE], x[..., 0], x[..., 1]
                )
                complementary_pressure_pot = (
                    self._evaluate_pressure_potential_at_points(
                        pressure_coeffs[COMPLEMENTARY_PRESSURE], x[..., 0], x[..., 1]
                    )
                )
                wetting_mobility = (
                    phase_mobilities["fractional_flow"]
                    * phase_mobilities["total_mobility"]
                )

                return -perm_arr[None, :, None] * (
                    wetting_mobility[None, :, None] * global_pressure_pot
                    + complementary_pressure_pot
                )

        # 4: Define integrand that computes either the norm or the inner product of the
        # flux differences.
        def integrand(
            x: np.ndarray,
            specifier: Literal["_norm", "_inner_product"],
        ) -> np.ndarray:
            fv_flux_new = self._evaluate_flux_at_points(
                pp.get_solution_values(
                    f"{flux_name}_RT0_coeffs", self.g_data, iterate_index=0
                ),
                x[..., 0],
                x[..., 1],
            )
            rec_flux_new = evaluate_flux_from_reconstructions(
                x, pressure_coeffs_new, mobilities_new
            )

            # FIXME Not zero for wetting flux.
            flux_diff_new = fv_flux_new - rec_flux_new

            if specifier == "_norm":
                return (flux_diff_new**2).sum(axis=-1)

            elif specifier == "_inner_product":
                fv_flux_old = self._evaluate_flux_at_points(
                    pp.get_solution_values(
                        f"{flux_name}_RT0_coeffs", self.g_data, time_step_index=0
                    ),
                    x[..., 0],
                    x[..., 1],
                )
                rec_flux_old = evaluate_flux_from_reconstructions(
                    x, pressure_coeffs_old, mobilities_old
                )
                flux_diff_old = fv_flux_old - rec_flux_old

                return (flux_diff_new * flux_diff_old).sum(axis=-1)

        # 5: Finally, integrate in space. Store norm at current time step and inner
        # product of previous and current time step.
        for specifier in ["_norm", "_inner_product"]:
            # Make mypy happy.
            specifier = typing.cast(Literal["_norm", "_inner_product"], specifier)
            integral: Integral = self.quadrature_est.integrate(
                functools.partial(integrand, specifier=specifier),
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"{flux_name}_D_estimator_from_postproc{specifier}",
                integral.elementwise.squeeze(),
                self.g_data,
                iterate_index=0,
            )

    def local_saturation_min_max(self, specifier: str) -> None:
        pressure_coeffs = pp.get_solution_values(
            f"{COMPLEMENTARY_PRESSURE}_coeffs_{specifier}", self.g_data, iterate_index=0
        )

        def integrand_min(
            x: np.ndarray,
            pressure_coeffs: np.ndarray,
        ) -> np.ndarray:
            pressure = evaluate_poly_at_points(pressure_coeffs, x[..., 0], x[..., 1])
            saturation = self.eval_saturation(pressure)
            return np.repeat(np.min(saturation, axis=0)[None, ...], 6, axis=0)

        integral_min: Integral = self.quadrature_est.integrate(
            functools.partial(integrand_min, pressure_coeffs=pressure_coeffs),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"s_w_min_{specifier}",
            integral_min.elementwise.squeeze() / self.quadrature_est.volumes,
            self.g_data,
            iterate_index=0,
        )

        def integrand_max(
            x: np.ndarray,
            pressure_coeffs: np.ndarray,
        ) -> np.ndarray:
            pressure = evaluate_poly_at_points(pressure_coeffs, x[..., 0], x[..., 1])
            saturation = self.eval_saturation(pressure)
            return np.repeat(np.max(saturation, axis=0)[None, ...], 6, axis=0)

        integral_max: Integral = self.quadrature_est.integrate(
            functools.partial(integrand_max, pressure_coeffs=pressure_coeffs),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"s_w_max_{specifier}",
            integral_max.elementwise.squeeze() / self.quadrature_est.volumes,
            self.g_data,
            iterate_index=0,
        )

    def local_flux_norm(self, flux_name: FLUX_NAME) -> None:
        def integrand(
            x: np.ndarray,
        ) -> np.ndarray:
            fv_flux = self._evaluate_flux_at_points(
                pp.get_solution_values(
                    f"{flux_name}_RT0_coeffs", self.g_data, iterate_index=0
                ),
                x[..., 0],
                x[..., 1],
            )
            return np.sqrt((fv_flux**2).sum(axis=-1))

        integral: Integral = self.quadrature_est.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"{flux_name}_norm",
            integral.elementwise.squeeze(),
            self.g_data,
            iterate_index=0,
        )

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        super().after_nonlinear_iteration(nonlinear_increment)  # type: ignore
        self.local_pressure_potential(GLOBAL_PRESSURE)
        self.local_pressure_potential(COMPLEMENTARY_PRESSURE)
        self.local_flux_norm(TOTAL_FLUX)
        self.local_flux_norm(WETTING_FLUX)
        self.local_saturation_min_max("rec")
        self.local_saturation_min_max("postproc")
        self.local_darcy_est_from_postproc(TOTAL_FLUX)
        self.local_darcy_est_from_postproc(WETTING_FLUX)

        dirname = pathlib.Path(__file__).parent.parent.parent.parent / "pressure_plots"
        dirname.mkdir(exist_ok=True)

        global_pressure_coeffs = pp.get_solution_values(
            GLOBAL_PRESSURE + "_coeffs_postproc", self.g_data, iterate_index=0
        )
        save_path = (
            dirname
            / f"g_pp_{self.time_manager.time_index}_{self.nonlinear_solver_statistics.num_iteration}.png"
        )

        plot_quadratic_pressures(
            self.g,
            self.domain.bounding_box,
            global_pressure_coeffs,
            save_path=save_path,
        )

        global_pressure_coeffs = pp.get_solution_values(
            GLOBAL_PRESSURE + "_coeffs_rec", self.g_data, iterate_index=0
        )
        save_path = (
            dirname
            / f"g_rec_{self.time_manager.time_index}_{self.nonlinear_solver_statistics.num_iteration}.png"
        )

        plot_quadratic_pressures(
            self.g,
            self.domain.bounding_box,
            global_pressure_coeffs,
            save_path=save_path,
        )

        complementary_pressure_coeffs = pp.get_solution_values(
            COMPLEMENTARY_PRESSURE + "_coeffs_postproc", self.g_data, iterate_index=0
        )
        save_path = (
            dirname
            / f"c_pp_{self.time_manager.time_index}_{self.nonlinear_solver_statistics.num_iteration}.png"
        )
        plot_quadratic_pressures(
            self.g,
            self.domain.bounding_box,
            complementary_pressure_coeffs,
            save_path=save_path,
        )
        complementary_pressure_coeffs = pp.get_solution_values(
            COMPLEMENTARY_PRESSURE + "_coeffs_rec", self.g_data, iterate_index=0
        )
        save_path = (
            dirname
            / f"c_rec_{self.time_manager.time_index}_{self.nonlinear_solver_statistics.num_iteration}.png"
        )
        plot_quadratic_pressures(
            self.g,
            self.domain.bounding_box,
            complementary_pressure_coeffs,
            save_path=save_path,
        )


class EstDataSavingMixin(RecDataSavingMixin):
    def _data_to_export(
        self, time_step_index: int | None = None, iterate_index: int | None = None
    ) -> list[DataInput]:
        """Append error estimators to the exported data."""
        data: list[DataInput] = super()._data_to_export(
            time_step_index=time_step_index,
            iterate_index=iterate_index,
        )
        for flux_name, specifier in itertools.product(
            (TOTAL_FLUX, WETTING_FLUX),
            [
                "norm",
                "D_estimator_from_postproc_norm",
            ],
        ):
            try:
                key = f"{flux_name}_{specifier}"
                data.append(
                    (
                        self.g,
                        key,
                        pp.get_solution_values(
                            key,
                            self.g_data,
                            iterate_index=0,
                        ),
                    )
                )
            except KeyError:
                pass

        for pressure_key, specifier in itertools.product(
            (GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE), ("_postproc", "_rec")
        ):
            try:
                data.append(
                    (
                        self.g,
                        f"{pressure_key}{specifier}_potential",
                        pp.get_solution_values(
                            f"{pressure_key}{specifier}_potential",
                            self.g_data,
                            iterate_index=0,
                        ),
                    )
                )
            except KeyError:
                pass

        for specifier in ["max_postproc", "min_postproc", "max_rec", "min_rec"]:
            try:
                data.append(
                    (
                        self.g,
                        f"s_w_{specifier}",
                        pp.get_solution_values(
                            f"s_w_{specifier}", self.g_data, iterate_index=0
                        ),
                    )
                )
            except KeyError:
                pass

        return data


# Protocols define different types for ``nonlinear_solver_statistics``, causing mypy
# errors. This is safe in practice, but ``nonlinear_solver_statistics`` must be used
# with care. We ignore the error.
class TwoPhaseFlowErrorEstimateAnalytics(  # type: ignore
    ErrorEstimateAnalyticsMixin,
    EstDataSavingMixin,
    ErrorEstimatesTwoPhaseFlow,
): ...
