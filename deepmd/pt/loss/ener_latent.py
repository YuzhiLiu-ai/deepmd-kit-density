# SPDX-License-Identifier: LGPL-3.0-or-later
"""Energy/force loss with DPA4-LR latent electrostatic supervision."""

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.loss.reduction import (
    masked_atom_mean,
)
from deepmd.pt.loss.ener import (
    EnergyStdLoss,
)
from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


@TaskLoss.register("ener_latent")
class EnergyLatentLoss(EnergyStdLoss):
    """Combine the standard energy loss with atomic latent charge/dipole loss.

    The supervised tensors are the final electrostatic state returned by
    DPA4-LR, after all configured recurrent polar-update steps.  Labels are
    loaded from ``atom_charge.npy`` and ``atom_dipole.npy`` and appear in
    the training label dictionary as ``atom_charge`` and ``atom_dipole``.
    """

    def __init__(
        self,
        start_pref_lq: float = 0.0,
        limit_pref_lq: float = 0.0,
        start_pref_ld: float = 0.0,
        limit_pref_ld: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        for name, value in (
            ("start_pref_lq", start_pref_lq),
            ("limit_pref_lq", limit_pref_lq),
            ("start_pref_ld", start_pref_ld),
            ("limit_pref_ld", limit_pref_ld),
        ):
            if value < 0.0:
                raise ValueError(f"`{name}` must be non-negative, got {value}.")
        self.start_pref_lq = float(start_pref_lq)
        self.limit_pref_lq = float(limit_pref_lq)
        self.start_pref_ld = float(start_pref_ld)
        self.limit_pref_ld = float(limit_pref_ld)
        self.has_lq = self.start_pref_lq != 0.0 or self.limit_pref_lq != 0.0
        self.has_ld = self.start_pref_ld != 0.0 or self.limit_pref_ld != 0.0
        if not self.has_lq and not self.has_ld:
            raise ValueError(
                "`ener_latent` requires a non-zero latent charge or latent "
                "dipole prefactor."
            )

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float | torch.Tensor,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        model_pred, loss, more_loss = super().forward(
            input_dict,
            model,
            label,
            natoms,
            learning_rate,
            mae=mae,
        )
        coef = learning_rate / self.starter_learning_rate
        pref_lq = self.limit_pref_lq + (self.start_pref_lq - self.limit_pref_lq) * coef
        pref_ld = self.limit_pref_ld + (self.start_pref_ld - self.limit_pref_ld) * coef
        mask = model_pred.get("mask")

        if self.has_lq and "latent_charge" not in model_pred:
            raise RuntimeError(
                "`ener_latent` charge supervision requires a model output named "
                "`latent_charge`; use the DPA4-LR model with `sog_energy` fitting."
            )
        if self.has_ld and "latent_dipole" not in model_pred:
            raise RuntimeError(
                "`ener_latent` dipole supervision requires a model output named "
                "`latent_dipole`; set `lr_kernel='dipole'` on the DPA4-LR "
                "`sog_energy` fitting."
            )

        if self.has_lq and "latent_charge" in model_pred and "atom_charge" in label:
            find_lq = label.get("find_atom_charge", 0.0)
            charge_pred = model_pred["latent_charge"].reshape(-1, natoms, 1)
            charge_label = (
                label["atom_charge"].reshape(-1, natoms, 1).to(charge_pred.dtype)
            )
            charge_error = torch.square(charge_pred - charge_label)
            if mask is not None:
                l2_lq = masked_atom_mean(charge_error, mask, 1)
            else:
                l2_lq = torch.mean(charge_error)
            loss = loss + pref_lq * find_lq * l2_lq
            more_loss["l2_latent_charge_loss"] = self.display_if_exist(
                l2_lq.detach(), find_lq
            )
            more_loss["rmse_latent_charge"] = self.display_if_exist(
                torch.sqrt(l2_lq.detach()), find_lq
            )

        if self.has_ld and "latent_dipole" in model_pred and "atom_dipole" in label:
            find_ld = label.get("find_atom_dipole", 0.0)
            dipole_pred = model_pred["latent_dipole"].reshape(-1, natoms, 3)
            dipole_label = (
                label["atom_dipole"].reshape(-1, natoms, 3).to(dipole_pred.dtype)
            )
            dipole_error = torch.square(dipole_pred - dipole_label)
            if mask is not None:
                l2_ld = masked_atom_mean(dipole_error, mask, 3)
            else:
                l2_ld = torch.mean(dipole_error)
            loss = loss + pref_ld * find_ld * l2_ld
            more_loss["l2_latent_dipole_loss"] = self.display_if_exist(
                l2_ld.detach(), find_ld
            )
            more_loss["rmse_latent_dipole"] = self.display_if_exist(
                torch.sqrt(l2_ld.detach()), find_ld
            )

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        requirements = list(super().label_requirement)
        if self.has_lq:
            requirements.append(
                DataRequirementItem(
                    "atom_charge",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_ld:
            requirements.append(
                DataRequirementItem(
                    "atom_dipole",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        return requirements

    def serialize(self) -> dict:
        data = super().serialize()
        data["@class"] = "EnergyLatentLoss"
        data.update(
            {
                "start_pref_lq": self.start_pref_lq,
                "limit_pref_lq": self.limit_pref_lq,
                "start_pref_ld": self.start_pref_ld,
                "limit_pref_ld": self.limit_pref_ld,
            }
        )
        return data
