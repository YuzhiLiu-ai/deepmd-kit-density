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
        value = self.pol_stiffness
        if not isinstance(value, torch.nn.Parameter):
            tensor = (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(value)
            )
            if "pol_stiffness" in self._buffers:
                del self._buffers["pol_stiffness"]
            trainable = self.trainable
            if isinstance(trainable, list):
                trainable = all(trainable)
            self.register_parameter(
                "pol_stiffness",
                torch.nn.Parameter(tensor, requires_grad=bool(trainable)),
            )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        legacy_key = prefix + "alpha_pol"
        if legacy_key in state_dict:
            # Checkpoints written before the stiffness reparameterization
            # store the polarizability ``alpha_pol`` (energy |p|^2 / (2
            # alpha^2)); the spring form 0.5 * theta^2 * |p|^2 is exactly
            # recovered with theta = 1 / alpha.  Convert once here so old
            # checkpoints load without Missing/Unexpected key errors.
            value = state_dict.pop(legacy_key)
            state_dict[prefix + "pol_stiffness"] = (
                torch.reciprocal(value) if torch.is_tensor(value) else 1.0 / value
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
