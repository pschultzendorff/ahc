"""Provide pytorch datasets for relative permeability and capillary pressure functions. 

So far, the following models are implemented:
- Relative permeability:
    - Brooks-Corey

- Capillary pressure:
    - Brooks-Corey

Both a :obj:`~torch.utils.data.Dataset` as well as a
:obj:`~torch.utils.data.IterableDataset` can be initialized with all data generating
models.

Additionally, noise can be added to the datasets, and with custom modifications, this
noise can even be structured (e.g., only apply to some part saturations etc.).

"""

from collections.abc import Iterator
from typing import Optional, Callable

import torch
import torch.nn as nn


class DatasetWithNoise(torch.utils.data.Dataset):
    def __init__(
        self,
        len: int = 1000,
        model: str = "Brooks-Corey-W",
        model_params: Optional[dict] = None,
        mean: float = 1.5,
        std: float = 1.5,
    ) -> None:
        super().__init__()
        self.len = len
        self.S_w = torch.rand([self.len, 1])
        mean_tensor = torch.tensor([mean] * self.len).unsqueeze(-1)
        std_tensor = torch.tensor([std] * self.len).unsqueeze(-1)
        noise = torch.normal(mean_tensor, std_tensor) * self.S_w
        biased_noise = noise
        # biased_noise = torch.where(self.S_w >= 0.5, torch.zeros_like(noise), noise)

        if model == "Brooks-Corey-W":
            gen_func: nn.Module | Callable = RelPermW_BrooksCorey(model_params)
        elif model == "Brooks-Corey-N":
            gen_func = RelPermN_BrooksCorey(model_params)
        elif model == "Brooks-Corey-Cap-Press":
            gen_func = CapPress_BrooksCorey(model_params)
        elif model == "power_w":
            gen_func = power_w
        elif model == "power_n":
            gen_func = power_n

        self.target: torch.Tensor = gen_func(self.S_w) + biased_noise

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.S_w[index], self.target[index]


class IterableDatasetWithNoise(torch.utils.data.IterableDataset):
    def __init__(
        self,
        model: str = "Brooks-Corey-W",
        model_params: Optional[dict] = None,
        mean: float = 1.5,
        std: float = 1.5,
    ) -> None:
        super().__init__()
        self.model: str = model
        self.mean = torch.tensor([mean])
        self.std = torch.tensor([std])
        if model == "Brooks-Corey-W":
            self.gen_func: nn.Module | Callable = RelPermW_BrooksCorey(model_params)
        elif model == "Brooks-Corey-N":
            self.gen_func = RelPermN_BrooksCorey(model_params)
        elif model == "Brooks-Corey-Cap-Press":
            self.gen_func = CapPress_BrooksCorey(model_params)
        elif model == "power_w":
            self.gen_func = power_w
        elif model == "power_n":
            self.gen_func = power_n

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

    """

    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.n_1: int = int(params.get("n_1", 2))
        self.n_2: int = int(params.get("n_3", 3))
        self.n_3: int = int(params.get("n_2", 1))
        self._residual_saturation_w: float = params.get("w_res_sat", 0.3)
        """Wetting residual saturation."""
        self._residual_saturation_n: float = params.get("n_res_sat", 0.3)
        """Nonwetting residual saturation."""

    def forward(self, S_w: torch.Tensor) -> torch.Tensor:
        S_w_normalized = (S_w - self._residual_saturation_w) / (
            1 - self._residual_saturation_w - self._residual_saturation_n
        )
        return S_w_normalized ** (self.n_1 + self.n_2 * self.n_3)


class RelPermN_BrooksCorey(nn.Module):
    """Generates nonwetting rel. perm. with the Brooks-Corey model.

    Default values correspond to the Brooks-Corey-Burdine model.

    """

    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.n_1: int = int(params.get("n_1", 2))
        self.n_2: int = int(params.get("n_3", 3))
        self.n_3: int = int(params.get("n_2", 1))
        self._residual_saturation_w: float = params.get("w_res_sat", 0.3)
        """Wetting residual saturation."""
        self._residual_saturation_n: float = params.get("n_res_sat", 0.3)
        """Nonwetting residual saturation."""

    def forward(self, S_w: torch.Tensor) -> torch.Tensor:
        S_w_normalized = (S_w - self._residual_saturation_w) / (
            1 - self._residual_saturation_w - self._residual_saturation_n
        )
        return S_w_normalized ** (self.n_1 + self.n_2 * self.n_3)


def power_w(S_w: torch.Tensor) -> torch.Tensor:
    """Cubic power law for wetting rel. perm."""
    return S_w**3


def power_n(S_w: torch.Tensor) -> torch.Tensor:
    """Cubic power law for wetting rel. perm."""
    return (1 - S_w) ** 3


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
        self._residual_saturation_w: float = params.get("w_res_sat", 0.3)
        """Wetting residual saturation."""
        self._residual_saturation_n: float = params.get("n_res_sat", 0.3)
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
