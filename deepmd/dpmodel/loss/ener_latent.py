# SPDX-License-Identifier: LGPL-3.0-or-later
"""Energy/force loss with atomic latent charge/dipole direct supervision."""

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.loss.reduction import (
    masked_atom_mean,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class EnergyLatentLoss(EnergyLoss):
    r"""Combine the energy/force loss with atomic latent charge/dipole loss.

    The supervised tensors are the electrostatic state a long-range plugin
    model exposes as ``latent_charge`` and ``latent_dipole`` (the latter only
    for :class:`latent_multipole`).  Labels are loaded from ``atom_charge.npy``
    and ``atom_dipole.npy`` and appear in the label dictionary as
    ``atom_charge`` and ``atom_dipole``.

    The total objective gains two terms,

    .. math::

       L = L_{\mathrm{ener}} + p_{q} L_{q} + p_{d} L_{d},

    where each prefactor is interpolated with the learning rate like the
    energy/force prefactors.  Each supervision branch is enabled only when at
    least one of its prefactors is non-zero; with all four at zero the loss
    degenerates to the plain :class:`EnergyLoss`.

    Parameters
    ----------
    start_pref_lq : float
        The prefactor of the latent charge loss at the start of the training.
    limit_pref_lq : float
        The prefactor of the latent charge loss at the end of the training.
    start_pref_ld : float
        The prefactor of the latent dipole loss at the start of the training.
    limit_pref_ld : float
        The prefactor of the latent dipole loss at the end of the training.
    **kwargs
        Other keyword arguments forwarded to :class:`EnergyLoss`.
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

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        """Calculate the energy/force objective plus latent charge/dipole terms."""
        loss, more_loss = super().call(
            learning_rate, natoms, model_dict, label_dict, mae=mae
        )
        if not (self.has_lq or self.has_ld):
            return loss, more_loss

        xp = array_api_compat.array_namespace(model_dict["energy"])
        lr_ratio = learning_rate / self.starter_learning_rate

        if self.has_lq and "latent_charge" not in model_dict:
            raise RuntimeError(
                "`ener_latent` charge supervision requires a model output named "
                "`latent_charge`; use a long-range plugin model "
                "(`latent_charge`/`latent_multipole`)."
            )
        if self.has_ld and "latent_dipole" not in model_dict:
            raise RuntimeError(
                "`ener_latent` dipole supervision requires a model output named "
                "`latent_dipole`; use the `latent_multipole` long-range plugin."
            )

        maskf = (
            xp.astype(model_dict["mask"], model_dict["energy"].dtype)
            if "mask" in model_dict
            else None
        )
        node_shape = maskf.shape if maskf is not None else None

        if self.has_lq and "atom_charge" in label_dict:
            find_lq = label_dict.get("find_atom_charge", 0.0)
            pref_lq = (
                self.limit_pref_lq
                + (self.start_pref_lq - self.limit_pref_lq) * lr_ratio
            )
            charge_pred = model_dict["latent_charge"]
            charge_label = xp.astype(label_dict["atom_charge"], charge_pred.dtype)
            if maskf is not None:
                charge_pred = xp.reshape(charge_pred, node_shape)
                charge_label = xp.reshape(charge_label, node_shape)
                charge_error = xp.square(charge_pred - charge_label)[..., None]
                l2_lq = masked_atom_mean(charge_error, maskf, 1)
            else:
                l2_lq = xp.mean(xp.square(charge_pred - charge_label))
            loss = loss + pref_lq * find_lq * l2_lq
            more_loss["l2_latent_charge_loss"] = self.display_if_exist(l2_lq, find_lq)
            more_loss["rmse_latent_charge"] = self.display_if_exist(
                xp.sqrt(l2_lq), find_lq
            )

        if self.has_ld and "atom_dipole" in label_dict:
            find_ld = label_dict.get("find_atom_dipole", 0.0)
            pref_ld = (
                self.limit_pref_ld
                + (self.start_pref_ld - self.limit_pref_ld) * lr_ratio
            )
            dipole_pred = model_dict["latent_dipole"]
            dipole_label = xp.astype(label_dict["atom_dipole"], dipole_pred.dtype)
            if maskf is not None:
                dipole_pred = xp.reshape(dipole_pred, (*node_shape, 3))
                dipole_label = xp.reshape(dipole_label, (*node_shape, 3))
                dipole_error = xp.square(dipole_pred - dipole_label)
                l2_ld = masked_atom_mean(dipole_error, maskf, 3)
            else:
                l2_ld = xp.mean(xp.square(dipole_pred - dipole_label))
            loss = loss + pref_ld * find_ld * l2_ld
            more_loss["l2_latent_dipole_loss"] = self.display_if_exist(l2_ld, find_ld)
            more_loss["rmse_latent_dipole"] = self.display_if_exist(
                xp.sqrt(l2_ld), find_ld
            )

        # The base class already published ``rmse`` for its own objective;
        # refresh it so the aggregate reflects the added latent terms.
        more_loss["rmse"] = xp.sqrt(loss)
        self.l2_l = loss
        self.l2_more = more_loss
        return loss, more_loss

    @property
    def training_metric_names(self) -> tuple[str, ...]:
        """Return energy metrics plus the enabled latent charge/dipole metrics."""
        names = super().training_metric_names
        if self.has_lq:
            names += ("rmse_latent_charge",)
        if self.has_ld:
            names += ("rmse_latent_dipole",)
        return names

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
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
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """
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

    @classmethod
    def deserialize(cls, data: dict) -> "EnergyLatentLoss":
        """Deserialize the loss module.

        Parameters
        ----------
        data : dict
            The serialized loss module

        Returns
        -------
        EnergyLatentLoss
            The deserialized loss module
        """
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 5, 1)
        data.pop("@class")
        # Mirror the EnergyLoss backward-compatibility handling.
        if version < 3:
            data.setdefault("intensive_ener_virial", False)
        if version < 5:
            data.setdefault("start_pref_h", 0.0)
            data.setdefault("limit_pref_h", 0.0)
        # Latent supervision is opt-in; default to disabled for older payloads.
        data.setdefault("start_pref_lq", 0.0)
        data.setdefault("limit_pref_lq", 0.0)
        data.setdefault("start_pref_ld", 0.0)
        data.setdefault("limit_pref_ld", 0.0)
        return cls(**data)
