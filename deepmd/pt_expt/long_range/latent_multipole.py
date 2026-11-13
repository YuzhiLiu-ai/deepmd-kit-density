# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.dpmodel.long_range.latent_multipole import (
    LatentMultipoleLongRange as LatentMultipoleLongRangeDP,
)
from deepmd.pt_expt.common import torch_module
from deepmd.pt_expt.fitting.latent_multipole import LatentMultipoleFitting

from .base_long_range import BaseLongRange
from .kernel import LESKernel


@BaseLongRange.register("latent_multipole")
@torch_module
class LatentMultipoleLongRange(LatentMultipoleLongRangeDP):
    """Non-periodic direct latent-multipole interaction (pt_expt) with latent electrostatic feedback (LEF)."""

    charge_net: LatentMultipoleFitting
    kernel: LESKernel

    def _kernel_coefficients(
        self, r_sq: torch.Tensor, src: torch.Tensor, n: int
    ) -> dict[str, torch.Tensor]:
        coefficients = self.kernel.coefficients(r_sq)
        if self.training and isinstance(r_sq.shape[0], torch.SymInt):
            return {
                name: value.expand(n, *value.shape)[src]
                for name, value in coefficients.items()
            }
        return coefficients
