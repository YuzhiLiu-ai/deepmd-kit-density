# SPDX-License-Identifier: LGPL-3.0-or-later
"""Convert a non-periodic long-range checkpoint to periodic evaluation.

The conversion edits only the stored model config (the ``long_range``
section flips ``periodic: false -> true`` with the requested k-space mode);
every trained weight is copied unchanged, because the LES/SOG kernel and
the charge/dipole networks are identical under both boundary conditions.
"""

from copy import (
    deepcopy,
)
from typing import (
    Any,
)

from deepmd.dpmodel.long_range.migration import (
    convert_long_range_config_to_periodic,
)


def convert_checkpoint_to_periodic(
    INPUT: str,
    OUTPUT: str,
    k_grid_mode: str = "kgrid",
    **kwargs: Any,
) -> None:
    """Convert a pt_expt training checkpoint to periodic long-range evaluation.

    Parameters
    ----------
    INPUT
        The input ``.pt`` training checkpoint (non-periodic long-range
        plugin model).
    OUTPUT
        The output ``.pt`` checkpoint with the converted model config.
    k_grid_mode
        ``"kgrid"`` (default, per-frame adaptive explicit wave numbers) or
        ``"pme"`` (fixed particle-mesh Ewald mesh).
    """
    import torch

    checkpoint = torch.load(INPUT, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    extra = state_dict.get("_extra_state") if isinstance(state_dict, dict) else None
    if not (isinstance(extra, dict) and "model_params" in extra):
        raise ValueError(
            f"Invalid checkpoint '{INPUT}': expected a pt_expt training "
            "checkpoint whose state dict carries '_extra_state' with nested "
            "'model_params'."
        )
    model_params = convert_long_range_config_to_periodic(
        deepcopy(extra["model_params"]), k_grid_mode=k_grid_mode
    )
    extra["model_params"] = model_params
    torch.save(checkpoint, OUTPUT)
