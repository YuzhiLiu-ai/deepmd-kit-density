# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.dpmodel.long_range.latent_charge import (
    LatentChargeLongRange as LatentChargeLongRangeDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.fitting.latent_charge import (
    LatentChargeFitting,
)

from .base_long_range import (
    BaseLongRange,
)
from .kernel import (
    LESKernel,
    SOGKernel,
)


@BaseLongRange.register("latent_charge")
@torch_module
class LatentChargeLongRange(LatentChargeLongRangeDP):
    """Non-periodic direct latent-charge interaction (pt_expt)."""

    charge_net: LatentChargeFitting
    kernel: LESKernel | SOGKernel

    def _pair_kernel(
        self, r_sq: torch.Tensor, src: torch.Tensor, n: int
    ) -> torch.Tensor:
        # Keep symbolic training traces' parameter reductions over the node axis.
        if not (self.training and isinstance(r_sq.shape[0], torch.SymInt)):
            return self.kernel(r_sq)
        coefficients = {
            name: value.expand(n, *value.shape)[src]
            for name, value in self.kernel.coefficients(r_sq).items()
        }
        return self.kernel.evaluate(r_sq, coefficients)
