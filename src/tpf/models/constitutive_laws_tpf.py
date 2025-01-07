import logging
import typing
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, NamedTuple, Optional, TypeGuard

import numpy as np
import porepy as pp
from tpf.models.phase import FluidPhase
from tpf.models.protocol import TPFProtocol
from tpf.numerics.ad.functions import minimum
from tpf.utils.constants_and_typing import CAP_PRESS_MODEL, REL_PERM_MODEL, OperatorType

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RelPermConstants:
    """:class:`RelPermConstants` is a :class:`~typing.NamedTuple` that holds the
    parameters for various relative permeability models.

    In alignment with the current (version > 3.10) implementation of fluid and material
    values in PorePy, all model constants are stored as `int`/`float` and converted to
    :class:`~porepy.ad.Scalar` when called in equations.

    """

    model: REL_PERM_MODEL = "linear"
    """The relative permeability model."""
    n1: int
    """Model parameter for the Brooks-Corey model."""
    n2: int
    """Model parameter for the Brooks-Corey model."""
    n3: int
    """Model parameter for the Brooks-Corey model."""
    power: int
    """Model parameter for the Corey and linear models."""
    linear_param_w: float
    """Model parameter for the Corey and linear models."""
    linear_param_n: float
    """Model parameter for the Corey and linear models."""
    kappa_g: float
    """Model parameter for the van Genuchten model."""
    n_g: int
    """Model parameter for the van Genuchten model."""
    m_g: float
    """Model parameter for the van Genuchten model."""
    limit: bool
    """Flag to indicate if limits are applied."""
    max_w: float
    min_w: float
    max_n: float
    min_n: float

    def __post_init__(self) -> None:
        if not self.is_rel_perm_model(self.model):
            raise ValueError("Invalid relative permeability model.")
        if self.model == "van Genuchten-Burdine":
            self.m_g: float = 1 - 1 / self.n_g
            logger.info(
                f"van Genuchten-Burdine model is used. Adjusting m_g to {self.m_g}."
            )
        if self.model == "van Genuchten-Mualem":
            self.m_g = 1 - 2 / self.n_g
            logger.info(
                f"van Genuchten-Mualem model is used. Adjusting m_g to {self.m_g}."
            )

    @staticmethod
    def is_rel_perm_model(model: str) -> TypeGuard[REL_PERM_MODEL]:
        return model in typing.get_args(REL_PERM_MODEL)


class RelativePermeability(TPFProtocol):

    _rel_perm_constants: RelPermConstants

    def set_rel_perm_constants(self) -> None:
        self._rel_perm_constants = RelPermConstants(
            **self.params.get("rel_perm_constants", {})
        )

    def _rel_perm_linear_param(
        self, phase: FluidPhase, rel_perm_constants: Optional[RelPermConstants] = None
    ) -> pp.ad.Scalar:
        if rel_perm_constants is None:
            rel_perm_constants = self._rel_perm_constants
        return pp.ad.Scalar(
            (
                rel_perm_constants.linear_param_w
                if phase.name == self.wetting.name
                else rel_perm_constants.linear_param_n
            ),
            name=f"{phase.name} rel. perm. linear_param",
        )

    def _rel_perm_limit(
        self,
        phase: FluidPhase,
        limit: Literal["min", "max"],
        rel_perm_constants: Optional[RelPermConstants] = None,
    ) -> float:
        if rel_perm_constants is None:
            rel_perm_constants = self._rel_perm_constants
        if limit == "min":
            return (
                rel_perm_constants.min_w
                if phase.name == self.wetting.name
                else rel_perm_constants.min_n
            )
        elif limit == "max":
            return (
                rel_perm_constants.max_w
                if phase.name == self.wetting.name
                else rel_perm_constants.max_n
            )

    def rel_perm(
        self,
        saturation_w: OperatorType,
        phase: FluidPhase,
        rel_perm_constants: Optional[RelPermConstants] = None,
    ) -> OperatorType:
        r"""Phase relative permeability.

        The following two models are implemented:

        Brooks-Corey model
        .. math::
            k_{r,w}(\hat{S}_w) = \hat{S}_w^{n_1 + n_2 \cdot n_3}, \\
            k_{r,n}(\hat{S}_w) = (1 - \hat{S}_w)^{n_1}(1 - \hat{S}_w^{n_2})^{n_3}, \\
            \text{where} \\
            \hat{S}_w = \frac{S_w - S_{w,res}}{1 - S_{w,res} - S_{n,res}}

        The default values (Brooks–Corey–Burdine model) are
        .. math::
            n_1 = 2, n_2 = 1 + 2/n_b, n_3 = 1.

        Corey model
        .. math::
            k_{r,w}(S_w) = \hat{S}_w^3, \\
            k_{r,n}(S_w) = (1 - \hat{S}_w)^3.

        To avoid ill-conditioned equation systems and crashing of the Newton solver at
        unphysical saturations (i.e., :math:`S_w\not\in[0,1]`), the nonwetting rel.
        perm. can be limited below and above.

        .. math::
            \hat{k}_{r,\alpha}(S_w) = \min\{\max\{k_{r,\alpha}, k_{r,\alpha}^{max}\}, k_{r,\alpha}^{min}},

        Parameters:
            saturation_w: Wetting phase saturation. Can, e.g., be of instance
                :class:`~porepy.ad.MixedDimensionalVariable` or
                :class:`~porepy.ad.SparseArray` (for saturation boundary values).
            rel_perm_constants: Relative permeability constants. If set, overrides
                ``self._rel_perm_constants``. Default is ``None``.

        Returns:
            Phase relative permeability.

        """
        if rel_perm_constants is None:
            rel_perm_constants = self._rel_perm_constants

        # Normalize saturation and compute both phase saturations.
        s_w_normalized = self.normalize_saturation(saturation_w, phase=self.wetting)
        if phase.name == self.wetting.name:
            s_phase: pp.ad.Operator = s_w_normalized
        elif phase.name == self.nonwetting.name:
            s_phase = pp.ad.Scalar(1) - s_w_normalized
        else:
            raise ValueError("Invalid phase name.")

        # Compute relative permeability based on chosen model.
        if rel_perm_constants.model in ["Corey", "power"]:
            rel_perm: pp.ad.Operator = (
                s_phase ** pp.ad.Scalar(3)
            ) * self._rel_perm_linear_param(
                phase, rel_perm_constants=rel_perm_constants
            )

        elif rel_perm_constants.model == "linear":
            rel_perm = s_phase * self._rel_perm_linear_param(
                phase, rel_perm_constants=rel_perm_constants
            )

        elif rel_perm_constants.model == "Brooks-Corey":
            if phase.name == self.wetting.name:
                rel_perm = s_phase ** pp.ad.Scalar(
                    rel_perm_constants.n1
                    + rel_perm_constants.n2 * rel_perm_constants.n3
                )
            elif phase.name == self.nonwetting.name:
                rel_perm = (s_phase ** pp.ad.Scalar(rel_perm_constants.n1)) * (
                    (
                        pp.ad.Scalar(1)
                        - s_w_normalized ** pp.ad.Scalar(rel_perm_constants.n2)
                    )
                    ** pp.ad.Scalar(rel_perm_constants.n3)
                )

        elif rel_perm_constants.model == "van Genuchten-Mualem":
            if phase.name == self.wetting.name:
                rel_perm = s_phase ** pp.ad.Scalar(rel_perm_constants.kappa_g) * (
                    pp.ad.Scalar(1)
                    - (
                        pp.ad.Scalar(1)
                        - s_phase ** pp.ad.Scalar(1 / rel_perm_constants.m_g)
                    )
                    ** pp.ad.Scalar(rel_perm_constants.m_g)
                ) ** pp.ad.Scalar(2)
            elif phase.name == self.nonwetting.name:
                rel_perm = s_phase ** pp.ad.Scalar(rel_perm_constants.kappa_g) * (
                    pp.ad.Scalar(1)
                    - s_w_normalized ** pp.ad.Scalar(1 / rel_perm_constants.m_g)
                ) ** pp.ad.Scalar(rel_perm_constants.m_g * 2)

        elif rel_perm_constants.model == "van Genuchten-Burdine":
            if phase.name == self.wetting.name:
                rel_perm = s_phase ** pp.ad.Scalar(2) * (
                    pp.ad.Scalar(1)
                    - (
                        pp.ad.Scalar(1)
                        - s_phase ** pp.ad.Scalar(1 / rel_perm_constants.m_g)
                    )
                    ** pp.ad.Scalar(rel_perm_constants.m_g)
                )
            elif phase.name == self.nonwetting.name:
                rel_perm = s_phase ** pp.ad.Scalar(2) * (
                    pp.ad.Scalar(1)
                    - s_w_normalized ** pp.ad.Scalar(1 / rel_perm_constants.m_g)
                ) ** pp.ad.Scalar(rel_perm_constants.m_g)

        # Limit relative permeability above and below.
        if rel_perm_constants.limit:
            maximum_func = pp.ad.Function(
                partial(
                    pp.ad.functions.maximum,
                    var_1=self._rel_perm_limit(
                        phase, "min", rel_perm_constants=rel_perm_constants
                    ),
                ),
                "max",
            )
            minimum_func = pp.ad.Function(
                partial(
                    minimum,
                    var_1=self._rel_perm_limit(
                        phase, "max", rel_perm_constants=rel_perm_constants
                    ),
                ),
                "min",
            )
            rel_perm = minimum_func(maximum_func(rel_perm))

        rel_perm.set_name(f"{phase.name} rel. perm.")
        return rel_perm

    def rel_perm_np(
        self,
        saturation_w: np.ndarray,
        phase: FluidPhase,
        rel_perm_constants: Optional[RelPermConstants] = None,
    ) -> np.ndarray:
        r"""Phase relative permeability for :class:`~numpy.ndarray`.

        For explanation of the implemented models, see :meth:`rel_perm`.


        Parameters:
            saturation_w: Array with wetting phase saturations.
            rel_perm_constants: Relative permeability constants. If set, overrides
                ``self._rel_perm_constants``. Default is ``None``.

        Returns:
            Phase relative permeability.

        """
        if rel_perm_constants is None:
            rel_perm_constants = self._rel_perm_constants

        # Normalize saturation and compute both phase saturations.
        s_w_normalized: np.ndarray = self.normalize_saturation_np(
            saturation_w, phase=self.wetting
        )
        if phase.name == self.wetting.name:
            s_phase: np.ndarray = s_w_normalized
            linear_param: float = rel_perm_constants.linear_param_w
        elif phase.name == self.nonwetting.name:
            s_phase = 1 - s_w_normalized
            linear_param = rel_perm_constants.linear_param_n
        else:
            raise ValueError("Invalid phase name.")

        # Compute relative permeability based on chosen model.
        if rel_perm_constants.model in ["Corey", "power"]:
            rel_perm: np.ndarray = (s_phase**3) * linear_param

        elif rel_perm_constants.model == "linear":
            rel_perm = s_phase * linear_param

        elif rel_perm_constants.model == "Brooks-Corey":
            if phase.name == self.wetting.name:
                rel_perm = s_phase ** (
                    rel_perm_constants.n1
                    + rel_perm_constants.n2 * rel_perm_constants.n3
                )
            elif phase.name == self.nonwetting.name:
                rel_perm = (s_phase**rel_perm_constants.n1) * (
                    (1 - s_w_normalized**rel_perm_constants.n2) ** rel_perm_constants.n3
                )

        elif rel_perm_constants.model == "van Genuchten-Mualem":
            if phase.name == self.wetting.name:
                rel_perm = (
                    s_phase**rel_perm_constants.kappa_g
                    * (
                        1
                        - (1 - s_phase ** (1 / rel_perm_constants.m_g))
                        ** rel_perm_constants.m_g
                    )
                    ** 2
                )
            elif phase.name == self.nonwetting.name:
                rel_perm = s_phase**rel_perm_constants.kappa_g * (
                    1 - s_w_normalized ** (1 / rel_perm_constants.m_g)
                ) ** (rel_perm_constants.m_g * 2)

        elif rel_perm_constants.model == "van Genuchten-Burdine":
            if phase.name == self.wetting.name:
                rel_perm = s_phase**2 * (
                    1
                    - (1 - s_phase ** (1 / rel_perm_constants.m_g))
                    ** rel_perm_constants.m_g
                )
            elif phase.name == self.nonwetting.name:
                rel_perm = (
                    s_phase**2
                    * (1 - s_w_normalized ** (1 / rel_perm_constants.m_g))
                    ** rel_perm_constants.m_g
                )

        # Limit relative permeability above and below.
        if rel_perm_constants.limit:
            rel_perm = np.minimum(
                np.maximum(
                    rel_perm,
                    self._rel_perm_limit(
                        phase, "min", rel_perm_constants=rel_perm_constants
                    ),
                ),
                self._rel_perm_limit(
                    phase, "max", rel_perm_constants=rel_perm_constants
                ),
            )

        return rel_perm


@dataclass(kw_only=True)
class CapPressConstants:
    """:class:`CapPressConstants` is a :class:`~typing.NamedTuple` that holds constants
    for capillary pressure models.

    """

    model: CAP_PRESS_MODEL
    """Model name."""
    entry_pressure: float
    """Entry pressure value for the Brooks-Corey model."""
    n_b: int
    """Exponent for the Brooks-Corey model."""
    n_g: int
    """First exponent for the van Genuchten model."""
    m_g: float
    """Second exponent for the van Genuchten model."""
    beta_g: float
    """Scaling factor for the van Genuchten model."""
    linear_param: float
    """Scaling factor for the linear model."""

    def __post_init__(self) -> None:
        if not self.is_cap_press_model(self.model):
            raise ValueError("Invalid capillary pressure model.")

    @staticmethod
    def is_cap_press_model(model: Optional[str]) -> TypeGuard[CAP_PRESS_MODEL]:
        return model in typing.get_args(CAP_PRESS_MODEL)


class CapillaryPressure(TPFProtocol):

    _cap_press_constants: CapPressConstants

    def set_cap_press_constants(self) -> None:
        self._cap_press_constants = CapPressConstants(
            **self.params.get("cap_press_constants", {})
        )

    def verify_cap_press_rel_perm_constants(self) -> None:
        """Verify that the capillary pressure and relative permeability constants are
        consistent.

        """
        ...

    def cap_press(
        self,
        saturation_w: OperatorType,
        cap_press_constants: Optional[CapPressConstants] = None,
    ) -> OperatorType:
        r"""Capillary pressure function.

        The following three models are implemented:

        Brooks-Corey model
        .. math::
            p_c(\hat{S}_w) = p_e\hat{S}_w^{n_b}

        Linear model
        .. math::
            p_c(\hat{S}_w) = c\hat{S}_w

        van Genuchten model
        .. math::
            p_c(\hat{S}_w) = \frac{(\hat{S}_w^{m_g}-1)^{-n_g}}{\beta_g}

        All three models are computed in terms of the normalized saturation
        .. math::
            \hat{S}_w = \frac{S_w - S_w^{min}}{S_w^{max} - S_w^{min}},

        If none of the models is chosen, the capillary pressure is set to 0.

        Parameters:
            saturation_w: Wetting phase saturation. Can, e.g., be of instance
                :class:`~porepy.ad.MixedDimensionalVariable` or
                :class:`~porepy.ad.SparseArray` (for saturation boundary values).
            cap_press_model: Capillary pressure constants. If set, overrides
                ``self._cap_press_constants``. Default is ``None``.

        Returns:
            Capillary pressure.

        """
        if cap_press_constants is None:
            cap_press_constants = self._cap_press_constants

        s_normalized = self.normalize_saturation(
            saturation_w, self.wetting, limit=False
        )
        s_normalized.set_name(f"{self.wetting.name}_s_normalized")
        if cap_press_constants.model == "Brooks-Corey":
            entry_pressure = pp.ad.Scalar(
                cap_press_constants.entry_pressure, name="entry pressure"
            )
            p_c: OperatorType = entry_pressure * s_normalized ** pp.ad.Scalar(
                -1 / cap_press_constants.n_b
            )
        elif cap_press_constants.model == "linear":
            cap_press_linear_param = pp.ad.Scalar(
                cap_press_constants.linear_param, name="cap. press. linear param"
            )
            p_c = cap_press_linear_param * s_normalized
        elif cap_press_constants.model == "van Genuchten":
            beta_g = pp.ad.Scalar(cap_press_constants.beta_g)
            p_c = (
                (
                    (
                        (s_normalized ** pp.ad.Scalar(-1 / cap_press_constants.m_g))
                        - pp.ad.Scalar(1)
                    )
                )
                ** pp.ad.Scalar(1 / cap_press_constants.n_g)
            ) / beta_g
        else:
            # Return cap. pressure 0.
            p_c = pp.ad.Scalar(0) * s_normalized
        p_c.set_name("cap. press.")
        return p_c

    def cap_press_deriv(
        self,
        saturation_w: OperatorType,
        cap_press_constants: Optional[CapPressConstants] = None,
    ) -> OperatorType:
        if cap_press_constants is None:
            cap_press_constants = self._cap_press_constants

        s_normalized = self.normalize_saturation(
            saturation_w, phase=self.wetting, limit=False
        )
        s_normalized_deriv = self.normalize_saturation_deriv(self.wetting)
        if cap_press_constants.model == "Brooks-Corey":
            entry_pressure = pp.ad.Scalar(cap_press_constants.entry_pressure)
            return (
                entry_pressure
                * pp.ad.Scalar(-1 / cap_press_constants.n_b)
                * s_normalized ** pp.ad.Scalar(-1 / cap_press_constants.n_b - 1)
            ) * s_normalized_deriv
        elif cap_press_constants.model == "linear":
            cap_press_linear_param = pp.ad.Scalar(cap_press_constants.linear_param)
            return cap_press_linear_param * s_normalized_deriv
        elif cap_press_constants.model == "van Genuchten":
            beta_g = pp.ad.Scalar(cap_press_constants.beta_g)
            return (
                pp.ad.Scalar(1 / cap_press_constants.n_g - 1)
                * (
                    (
                        (s_normalized ** pp.ad.Scalar(-1 / cap_press_constants.m_g))
                        - pp.ad.Scalar(1)
                    )
                    ** pp.ad.Scalar(1 / cap_press_constants.n_g - 1)
                )
                / beta_g
                * pp.ad.Scalar(-1 / cap_press_constants.m_g)
                * (s_normalized ** pp.ad.Scalar(-1 / cap_press_constants.m_g - 1))
                * s_normalized_deriv
            )
        else:
            # Return cap. pressure 0.
            return pp.ad.Scalar(0) * s_normalized

    def cap_press_deriv_np(
        self,
        saturation_w: np.ndarray,
        cap_press_constants: Optional[CapPressConstants] = None,
    ) -> np.ndarray:
        r"""Capillary pressure derivative for saturation of type
         :class:`~numpy.ndarray`.

        For explanation of the implemented models, see :meth:`cap_press_deriv`.


        Parameters:
            saturation_w: Array with wetting phase saturations.
            cap_press_constants: Relative permeability constants. If set, overrides
                ``self.cap_press_constants``. Default is ``None``.

        Returns:
            Capillary pressure derivative.

        """

        if cap_press_constants is None:
            cap_press_constants = self._cap_press_constants

        s_normalized: np.ndarray = self.normalize_saturation_np(
            saturation_w, phase=self.wetting, limit=False
        )
        s_normalized_deriv: np.ndarray = self.normalize_saturation_deriv_np(
            self.wetting
        )
        if cap_press_constants.model == "Brooks-Corey":
            return (
                cap_press_constants.entry_pressure
                * (-1 / cap_press_constants.n_b)
                * s_normalized ** (-1 / cap_press_constants.n_b - 1)
            ) * s_normalized_deriv
        elif cap_press_constants.model == "linear":
            cap_press_linear_param = cap_press_constants.linear_param
            return cap_press_linear_param * s_normalized_deriv
        elif cap_press_constants.model == "van Genuchten":
            return (
                (1 / cap_press_constants.n_g - 1)
                * (
                    ((s_normalized ** (-1 / cap_press_constants.m_g)) - 1)
                    ** (1 / cap_press_constants.n_g - 1)
                )
                / cap_press_constants.beta_g
                * (-1 / cap_press_constants.m_g)
                * (s_normalized ** (-1 / cap_press_constants.m_g - 1))
                * s_normalized_deriv
            )
        else:
            # Return cap. pressure 0.
            return np.zeros_like(s_normalized)


def validate_constants() -> bool:
    """Check that the relative permeability and capillary pressure models go in hand."""
    # TODO: Implement this function.
    pass
    pass
