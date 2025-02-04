import json
import typing
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import porepy as pp


class SolverStatisticsTPF(pp.SolverStatistics):

    time_step_index: int = 0
    """Time step count."""
    time: float = 0.0
    """Current simulation time."""
    time_step_size: float = 0.0
    """Time step size."""

    @typing.override
    def log_error(
        self,
        nonlinear_increment_norm: float | None = None,
        residual_norm: float | None = None,
        **kwargs,
    ) -> None:
        """Log errors produced from convergence criteria.

        Parameters:
            nonlinear_increment_norm (float): Error in the increment.
            residual_norm (float): Error in the residual.
            **kwargs: Additional keyword arguments, for potential extension.

        Raises:
            ValueError: If neither the time step information nor the norms are provided.

        """
        if (
            "time_step_index" in kwargs
            and "time" in kwargs
            and "time_step_size" in kwargs
        ):
            self.time_step_index = kwargs["time_step_index"]
            self.time = kwargs["time"]
            self.time_step_size = kwargs["time_step_size"]
        elif nonlinear_increment_norm is not None and residual_norm is not None:
            super().log_error(nonlinear_increment_norm, residual_norm, **kwargs)
        else:
            raise ValueError("Either provide all time step information or norms.")

    @typing.override
    def save(self) -> None:
        """Save the statistics object to a JSON file."""
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
            else:
                data = {}

            # Append data - assume the index corresponds to time step
            ind = len(data) + 1
            data[ind] = {
                "time step index": self.time_step_index,
                "current time": self.time,
                "time step size": self.time_step_size,
                "num_iteration": self.num_iteration,
                "nonlinear_increment_norms": self.nonlinear_increment_norms,
                "residual_norms": self.residual_norms,
            }

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


@dataclass
class SolverStatisticsRec(SolverStatisticsTPF):

    equilibrated_flux_mismatch: list[dict[str, float]] = field(default_factory=list)
    """List of mismatches of equilibrated fluxes for each non-linear iteration."""

    @typing.override
    def log_error(
        self,
        nonlinear_increment_norm: float | None = None,
        residual_norm: float | None = None,
        **kwargs,
    ) -> None:
        if "equilibrated_flux_mismatch" in kwargs:
            self.equilibrated_flux_mismatch.append(kwargs["equilibrated_flux_mismatch"])
        else:
            super().log_error(nonlinear_increment_norm, residual_norm, **kwargs)

    @typing.override
    def reset(self) -> None:
        """Reset the mismatch list."""
        super().reset()
        self.equilibrated_flux_mismatch.clear()

    @typing.override
    def save(self) -> None:
        """Save the estimator statistics to a JSON file."""
        # This calls ``pp.SolverStatistics.save``, which adds a new entry to the
        # ``data`` dictionary that is found at ``self.path``.
        super().save()
        # Instead of creating a new entry, we load the already created entry and append.
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
            else:
                data = {}

            # Append data.
            ind = len(data)
            # Since data was stored and loaded as json, the keys have turned to strings.
            data[str(ind)].update(
                {
                    "equilibrated_flux_mismatch": self.equilibrated_flux_mismatch,
                }
            )

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


@dataclass
class SolverStatisticsEst(SolverStatisticsRec):
    """

    Note: Things may not be logged correctly during a Newton loop if a
    `FloatingPointError` occurs during computation of the qantities to log in
    :meth:`model.check_convergence`. In this case, `NewtonSolver` catches the error and
    :meth:`model.after_nonlinear_failure` is called without logging the quantities.

    """

    residual_and_flux_est: list[float] = field(default_factory=list)
    """List of residual and flux error estimates for each non-linear iteration."""
    nonconformity_est: list[dict[str, float]] = field(default_factory=list)
    """List of nonconformity error estimates for each non-linear iteration."""
    global_energy_norm: list[float] = field(default_factory=list)
    """List of global energy norms for each non-linear iteration."""

    @typing.override
    def log_error(
        self,
        nonlinear_increment_norm: float | None = None,
        residual_norm: float | None = None,
        **kwargs,
    ) -> None:
        if (
            "residual_and_flux_est" in kwargs
            and "nonconformity_est" in kwargs
            and "global_energy_norm" in kwargs
        ):
            self.residual_and_flux_est.append(kwargs["residual_and_flux_est"])
            self.nonconformity_est.append(
                kwargs["nonconformity_est"],
            )
            self.global_energy_norm.append(kwargs["global_energy_norm"])
        else:
            super().log_error(nonlinear_increment_norm, residual_norm, **kwargs)

    @typing.override
    def reset(self) -> None:
        """Reset the estimator lists."""
        super().reset()
        self.residual_and_flux_est.clear()
        self.nonconformity_est.clear()
        self.global_energy_norm.clear()

    @typing.override
    def save(self) -> None:
        """Save the estimator statistics to a JSON file."""
        # This calls ``pp.SolverStatistics.save``, which adds a new entry to the
        # ``data`` dictionary that is found at ``self.path``.
        super().save()
        # Instead of creating a new entry, we load the already created entry and append.
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
            else:
                data = {}

            # Append data.
            ind = len(data)
            # Since data was stored and loaded as json, the keys have turned to strings.
            data[str(ind)].update(
                {
                    "residual_and_flux_est": self.residual_and_flux_est,
                    "nonconformity_est": self.nonconformity_est,
                    "global_energy_norm": self.global_energy_norm,
                }
            )

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


# Important to subclass SolverStatisticsTPF and not SolverStatisticsEst.
@dataclass
class SolverStatisticsHC(SolverStatisticsTPF):
    hc_lambda_fl: float = 1.0
    """Homotopy continuation lambda."""
    hc_lambda_ad: pp.ad.Scalar = pp.ad.Scalar(1.0)
    """Homotopy continuation lambda for automatic differentiation."""
    hc_lambdas: list[float] = field(default_factory=list)
    """List of homotopy continuation lambda values for the current time step."""
    hc_num_iteration: int = 0
    """Number of homotopy continuation iterations performed for current time step."""
    num_iteration: int = 0
    """Number of non-linear iterations performed for current homotopy continuation step.

    """
    nums_iteration: list[int] = field(default_factory=list)
    """Number of non-linear iterations performed for current homotopy continuation step.

    """
    nonlinear_increment_norms_hc: list[list[float]] = field(default_factory=list)
    """List of list of increment magnitudes for each non-linear iteration. Outer list
    are HC iterations, inner list are non-linear iterations.

    """
    residual_norms_hc: list[list[float]] = field(default_factory=list)
    """List of list of residual norms. Outer list are HC iterations, inner list are
    non-linear iterations.

    """
    discretization_est: list[list[float]] = field(default_factory=list)
    """List of list of discretization error estimates. Outer list are HC iterations,
    inner list are non-linear iterations.

    """
    hc_est: list[list[float]] = field(default_factory=list)
    """List of list of homotopy continuation error estimates. Outer list are HC
    iterations, inner list are non-linear iterations.

    """
    linearization_est: list[list[float]] = field(default_factory=list)
    """List of list of linearization error estimates. Outer list are HC iterations,
    inner list are non-linear iterations.

    """
    global_energy_norm: list[list[float]] = field(default_factory=list)
    """List of global energy norms for each non-linear iteration."""
    equilibrated_flux_mismatch: list[list[dict[str, float]]] = field(
        default_factory=list
    )
    """List of mismatches of equilibrated fluxes for each non-linear iteration."""

    @typing.override
    def log_error(
        self,
        nonlinear_increment_norm: float | None = None,
        residual_norm: float | None = None,
        **kwargs,
    ) -> None:
        if (
            "global_energy_norm" in kwargs
            and "equilibrated_flux_mismatch" in kwargs
            and "discretization_est" in kwargs
            and "hc_est" in kwargs
            and "linearization_est" in kwargs
        ):
            self.global_energy_norm[-1].append(kwargs["global_energy_norm"])
            self.equilibrated_flux_mismatch[-1].append(
                kwargs["equilibrated_flux_mismatch"]
            )
            self.discretization_est[-1].append(kwargs["discretization_est"])
            self.hc_est[-1].append(kwargs["hc_est"])
            self.linearization_est[-1].append(kwargs["linearization_est"])
        else:
            super().log_error(nonlinear_increment_norm, residual_norm, **kwargs)

    @typing.override
    def reset(self) -> None:
        """Reset the homotopy continuation statistics object at the start of a new
        Newton loop.

        """
        # Do no append empty Newton loop data at the start of a new time step.
        if self.hc_num_iteration > 0:
            self.nums_iteration.append(self.num_iteration)
            # Append a deep copy of the lists; otherwise only a reference to the mutable
            # object is appended.
            self.nonlinear_increment_norms_hc.append(
                deepcopy(self.nonlinear_increment_norms)
            )
            self.residual_norms_hc.append(deepcopy(self.residual_norms))
        super().reset()
        self.discretization_est.append([])
        self.hc_est.append([])
        self.linearization_est.append([])
        self.global_energy_norm.append([])
        self.equilibrated_flux_mismatch.append([])

    def hc_reset(self) -> None:
        """Reset the homotopy continuation statistics object at the start of a new
        continuation loop.

        """
        self.nums_iteration.clear()
        self.nonlinear_increment_norms_hc.clear()
        self.residual_norms_hc.clear()
        self.discretization_est.clear()
        self.hc_est.clear()
        self.linearization_est.clear()
        self.global_energy_norm.clear()
        self.equilibrated_flux_mismatch.clear()

        self.hc_num_iteration = 0
        self.hc_lambda_fl = 1.0
        self.hc_lambda_ad.set_value(1.0)
        self.hc_lambdas.clear()
        self.hc_lambdas.append(1.0)

    @typing.override
    def save(self) -> None:
        """Save the estimator statistics to a JSON file."""
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data: dict[int, Any] = json.load(file)
            else:
                data = {}

            # Append data.
            ind: int = len(data) + 1

            if self.hc_num_iteration > 0:
                # :meth:`reset` is called at the start of each Newton loop, so we have to
                # append some of the last Newton loop data.
                self.nums_iteration.append(self.num_iteration)
                self.nonlinear_increment_norms_hc.append(self.nonlinear_increment_norms)
                self.residual_norms_hc.append(self.residual_norms)

            # The data is organized into dictionaries for each hc step. Each hc step
            # contains lists with values for all Newton steps.
            data[ind] = {
                i: {
                    "num_iteration": n,
                    "nonlinear_increment_norms": nin,
                    "residual_norms": rn,
                    "discretization_error_estimates": de,
                    "hc_error_estimates": hce,
                    "linearization_error_estimates": le,
                    "global_energy_norm": gen,
                    "equilibrated_flux_mismatch": efm,
                }
                for i, (n, nin, rn, de, hce, le, gen, efm) in enumerate(
                    zip(
                        self.nums_iteration,
                        self.nonlinear_increment_norms_hc,
                        self.residual_norms_hc,
                        self.discretization_est,
                        self.hc_est,
                        self.linearization_est,
                        self.global_energy_norm,
                        self.equilibrated_flux_mismatch,
                    )
                )
            }
            data[ind].update(
                {
                    "time step index": self.time_step_index,
                    "current time": self.time,
                    "time step size": self.time_step_size,
                    # Do not log the latest hc iteration, since it wasn't solved in a
                    # Newton loop. This is because :meth:`after_hc_iteration` is called
                    # before :meth:`after_hc_convergence/failure`
                    "hc_lambdas": self.hc_lambdas[:-1],
                    "hc_num_iterations": self.hc_num_iteration - 1,
                }
            )
            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)
