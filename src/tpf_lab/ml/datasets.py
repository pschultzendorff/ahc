"""Provide pytorch datasets for relative permeability and capillary pressure functions. 

So far, the following models are implemented:
- Relative permeability:
    - Brooks-Corey
    - power model (i.e., Corey)

- Capillary pressure:
    - Brooks-Corey

Both a :obj:`~torch.utils.data.Dataset` as well as a
:obj:`~torch.utils.data.IterableDataset` can be initialized with all data generating
models.

Additionally, noise can be added to the datasets, and with custom modifications, this
noise can even be structured (e.g., only apply to some part saturations etc.).

"""

from collections.abc import Iterator
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn


class DatasetWithNoise(torch.utils.data.Dataset):
    def __init__(
        self,
        len: int = 1000,
        model: Literal[
            "Brooks-Corey_w",
            "Brooks-Corey_n",
            "Brooks-Corey_pcap",
            "Corey_w",
            "Corey_n",
            "power_w",
            "power_n",
        ] = "Brooks-Corey_w",
        model_params: Optional[dict] = None,
        mean: float = 1.5,
        std: float = 1.5,
        biased_noise: bool = False,
    ) -> None:
        """_summary_



        Parameters:
            len: _description_. Defaults to 1000.
            model: _description_. Defaults to "Brooks-Corey_w".
            model_params: _description_. Defaults to None.
            mean: _description_. Defaults to 1.5.
            std: _description_. Defaults to 1.5.
            biased_noise: _description_. At the moment, the exact form of the biased
                noise is hard coded. Defaults to False.

        TODO: Fix the biased noise implementation s.t., e.g., a noise function is
            passed. This would allow for much greater flexibility.

        """
        super().__init__()
        if model_params is None:
            model_params = {"residual_saturation_w": 0.3, "residual_saturation_n": 0.3}

        self.len = len
        self.S_w = torch.rand([self.len, 1])
        mean_tensor = torch.tensor([mean] * self.len).unsqueeze(-1)
        std_tensor = torch.tensor([std] * self.len).unsqueeze(-1)
        noise = torch.normal(mean_tensor, std_tensor)

        if biased_noise:
            # Noise only for :math:`0.4\leq S_w\leq0.6`.
            noise = torch.where(
                torch.logical_and(self.S_w >= 0.4, self.S_w <= 0.6),
                noise,
                torch.zeros_like(noise),
            )

        if model == "Brooks-Corey_w":
            gen_func: nn.Module | Callable = RelPermW_BrooksCorey(model_params)
        elif model == "Brooks-Corey_n":
            gen_func = RelPermN_BrooksCorey(model_params)
        elif model == "Brooks-Corey-Cap-Press":
            gen_func = CapPress_BrooksCorey(model_params)
        elif model in ["Corey_w", "power_w"]:
            gen_func = RelPermW_Corey(model_params)
        elif model in ["Corey_n", "power_n"]:
            gen_func = RelPermN_Corey(model_params)

        target: torch.Tensor = gen_func(self.S_w) + noise

        # Sanitize rel. perm. data.
        if not model == "Brooks-Corey_pcap":
            # Cut values < 0 and > 1.
            target = torch.where(target > 1.0, 1.0, target)
            target = torch.where(target < 0.0, 0.0, target)

        # Cut values above/below the residual saturations.
        if model in ["Brooks-Corey_w", "Corey_w", "power_w"]:
            target = torch.where(
                self.S_w > 1 - model_params["residual_saturation_n"], 1.0, target
            )
            target = torch.where(
                self.S_w < model_params["residual_saturation_w"], 0.0, target
            )
        elif model in ["Brooks-Corey_n", "Corey_n", "power_n"]:
            target = torch.where(
                self.S_w > 1 - model_params["residual_saturation_n"], 0.0, target
            )
            target = torch.where(
                self.S_w < model_params["residual_saturation_w"], 1.0, target
            )

        self.target: torch.Tensor = target

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.S_w[index], self.target[index]


class IterableDatasetWithNoise(torch.utils.data.IterableDataset):
    def __init__(
        self,
        model: Literal[
            "Brooks-Corey-W",
            "Brooks-Corey-N",
            "Brooks-Corey-Cap-Press",
            "power_w",
            "power_n",
        ] = "Brooks-Corey-W",
        model_params: Optional[dict] = None,
        mean: float = 1.5,
        std: float = 1.5,
    ) -> None:
        super().__init__()
        self.model: Literal[
            "Brooks-Corey-W",
            "Brooks-Corey-N",
            "Brooks-Corey-Cap-Press",
            "power_w",
            "power_n",
        ] = model
        self.mean = torch.tensor([mean])
        self.std = torch.tensor([std])
        if model == "Brooks-Corey-W":
            self.gen_func: nn.Module | Callable = RelPermW_BrooksCorey(model_params)
        elif model == "Brooks-Corey-N":
            self.gen_func = RelPermN_BrooksCorey(model_params)
        elif model == "Brooks-Corey-Cap-Press":
            self.gen_func = CapPress_BrooksCorey(model_params)
        elif model == "power_w":
            self.gen_func = RelPermW_Corey(model_params)
        elif model == "power_n":
            self.gen_func = RelPermN_Corey(model_params)

    def __iter__(self):
        def iterator(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            while True:
                s_w = torch.rand([1])
                noise = torch.normal(self.mean, self.mean)
                target = self.gen_func(s_w) + noise
                yield s_w, target

        return iterator(self)


class RelPermW_BrooksCorey(nn.Module):
    """Generates wetting rel. perm. with the Brooks-Corey model.

    Default values correspond to the Brooks-Corey-Burdine model.

    The return values are limited above and below.

    """

    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.n_1: int = int(params.get("n_1", 2))
        self.n_2: int = int(params.get("n_3", 3))
        self.n_3: int = int(params.get("n_2", 1))
        self._residual_saturation_w: float = params.get("residual_saturation_w", 0.3)
        """Wetting residual saturation."""
        self._residual_saturation_n: float = params.get("residual_saturation_n", 0.3)
        """Nonwetting residual saturation."""

    def forward(self, S_w: torch.Tensor) -> torch.Tensor:
        S_w_normalized = (S_w - self._residual_saturation_w) / (
            1 - self._residual_saturation_w - self._residual_saturation_n
        )
        out = S_w_normalized ** (self.n_1 + self.n_2 * self.n_3)
        return out


class RelPermN_BrooksCorey(nn.Module):
    """Generates nonwetting rel. perm. with the Brooks-Corey model.

    Default values correspond to the Brooks-Corey-Burdine model.

    The return values are limited above and below.

    """

    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.n_1: int = int(params.get("n_1", 2))
        self.n_2: int = int(params.get("n_3", 3))
        self.n_3: int = int(params.get("n_2", 1))
        self._residual_saturation_w: float = params.get("residual_saturation_w", 0.3)
        """Wetting residual saturation."""
        self._residual_saturation_n: float = params.get("residual_saturation_n", 0.3)
        """Nonwetting residual saturation."""

    def forward(self, S_w: torch.Tensor) -> torch.Tensor:
        S_w_normalized = (S_w - self._residual_saturation_w) / (
            1 - self._residual_saturation_w - self._residual_saturation_n
        )
        out = ((1 - S_w_normalized) ** self.n_1) * (
            (1 - S_w_normalized**self.n_2) ** self.n_3
        )
        return out


class RelPermW_Corey(nn.Module):
    """Generates wetting rel. perm. with the Corey model.

    Default values correspond to a cubic law.

    The return values are limited above and below.

    """

    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.power: int = int(params.get("power", 3))
        self.linear_param: float = params.get("linear_param", 1.0)
        self._residual_saturation_w: float = params.get("residual_saturation_w", 0.3)
        """Wetting residual saturation."""
        self._residual_saturation_n: float = params.get("residual_saturation_n", 0.3)
        """Nonwetting residual saturation."""

    def forward(self, S_w: torch.Tensor) -> torch.Tensor:
        S_w_normalized = (S_w - self._residual_saturation_w) / (
            1 - self._residual_saturation_w - self._residual_saturation_n
        )
        out = (S_w_normalized**self.power) * self.linear_param
        return out


class RelPermN_Corey(nn.Module):
    """Generates nonwetting rel. perm. with the Corey model.

    Default values correspond to a cubic law.

    The return values are limited above and below.

    """

    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.power: int = int(params.get("power", 3))
        self.linear_param: float = params.get("linear_param", 1.0)
        self._residual_saturation_w: float = params.get("residual_saturation_w", 0.3)
        """Wetting residual saturation."""
        self._residual_saturation_n: float = params.get("residual_saturation_n", 0.3)
        """Nonwetting residual saturation."""

    def forward(self, S_w: torch.Tensor) -> torch.Tensor:
        S_w_normalized = (S_w - self._residual_saturation_w) / (
            1 - self._residual_saturation_w - self._residual_saturation_n
        )
        out = ((1 - S_w_normalized) ** self.power) * self.linear_param
        return out


class CapPress_BrooksCorey(nn.Module):
    """Calculates capillary pressure with the Brooks-Corey model."""

    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.w_psi: float = params.get("w_psi", 0.5)
        """Wetting fluid pore size index."""
        self.n_psi: float = params.get("n_psi", 0.5)
        """Nonwetting fluid pore size index."""
        self.w_thresh_press: float = params.get("w_thresh_press", 1.5)
        """Wetting threshold pressure."""
        self.n_thresh_press: float = params.get("n_thresh_press", -1.5)
        """Nonwetting threshold pressure."""
        self._residual_saturation_w: float = params.get("residual_saturation_w", 0.3)
        """Wetting residual saturation."""
        self._residual_saturation_n: float = params.get("residual_saturation_n", 0.3)
        """Nonwetting residual saturation."""

    def forward(self, S_w: torch.Tensor) -> torch.Tensor:
        normalized_S_w = (S_w - self._residual_saturation_w) / (
            1 - self._residual_saturation_w - self._residual_saturation_n
        )
        normalized_S_n = (1 - S_w - self._residual_saturation_w) / (
            1 - self._residual_saturation_w - self._residual_saturation_n
        )
        p_cw = self.w_thresh_press / (normalized_S_w**self.w_psi)
        p_cn = self.n_thresh_press / (normalized_S_n**self.n_psi)
        return p_cw + p_cn
