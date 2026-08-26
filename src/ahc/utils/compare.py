from ahc.models.protocol import TPFProtocol


class AnalyzerMixin(TPFProtocol):
    def __init__(self, params: dict | None, **kwargs) -> None:
        # Ignore mypy. When mixed in with a concrete class, super().__init__ takes
        # params.
        super().__init__(params)  # type: ignore

        if "reference_solution" in kwargs:
            self.reference_solution = kwargs["reference_solution"]
            self.reference_model = self.__class__(self.params)
            self.reference_model.set_values(self.reference_solution)
        else:
            raise ValueError("Reference solution must be provided for analysis.")

    def compare_primary_variables(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "reference_model"):
            raise ValueError(
                "Reference model is not set. Cannot compare primary variables."
            )

        # Compare the primary variables of the current model with the reference model
        current_primary_vars = self.get_primary_variables()
        reference_primary_vars = self.reference_model.get_primary_variables()

        # Calculate the difference between the two primary variables
        primary_var_difference = current_primary_vars - reference_primary_vars

        return {
            "current_primary_vars": current_primary_vars,
            "reference_primary_vars": reference_primary_vars,
            "primary_var_difference": primary_var_difference,
        }

    def compare_fluxes(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "reference_model"):
            raise ValueError("Reference model is not set. Cannot compare fluxes.")

        # Compare the fluxes of the current model with the reference model
        current_fluxes = self.get_fluxes()
        reference_fluxes = self.reference_model.get_fluxes()

        # Calculate the difference between the two fluxes
        flux_difference = current_fluxes - reference_fluxes

        return {
            "current_fluxes": current_fluxes,
            "reference_fluxes": reference_fluxes,
            "flux_difference": flux_difference,
        }

    def analyze(
        self,
    ):
        # Perform analysis on the model
        second
