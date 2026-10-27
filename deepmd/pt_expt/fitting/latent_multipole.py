# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Any

import torch

from deepmd.dpmodel.fitting.latent_multipole import (
    LatentMultipoleFitting as LatentMultipoleFittingDP,
)
from deepmd.pt_expt.common import torch_module


@torch_module
class LatentMultipoleFitting(LatentMultipoleFittingDP):
    """Latent-multipole readout/update networks (pt_expt)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        value = self.alpha_pol
        if not isinstance(value, torch.nn.Parameter):
            tensor = (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(value)
            )
            if "alpha_pol" in self._buffers:
                del self._buffers["alpha_pol"]
            trainable = self.trainable
            if isinstance(trainable, list):
                trainable = all(trainable)
            self.register_parameter(
                "alpha_pol", torch.nn.Parameter(tensor, requires_grad=bool(trainable))
            )
