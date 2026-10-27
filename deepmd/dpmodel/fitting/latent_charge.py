# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)

from .invar_fitting import (
    InvarFitting,
)


@fitting_check_output
class LatentChargeFitting(InvarFitting):
    """Per-atom latent charges predicted from invariant descriptor features.

    The constructor is that of :class:`InvarFitting`; ``var_name`` is normally
    ``latent_charge`` and ``dim_out`` is the number of charge channels.  This
    is a readout owned by a long-range module, not a standalone fitting type.
    """

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            ]
        )

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "latent_charge"
        return data
