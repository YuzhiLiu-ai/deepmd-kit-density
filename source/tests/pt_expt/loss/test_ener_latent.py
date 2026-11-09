# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the pt_expt/dpmodel EnergyLatentLoss (latent charge/dipole supervision)."""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.loss.ener_latent import (
    EnergyLatentLoss as EnergyLatentLossDP,
)
from deepmd.pt_expt.loss.ener import (
    EnergyLoss,
)
from deepmd.pt_expt.loss.ener_latent import (
    EnergyLatentLoss,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.env import (
    PRECISION_DICT,
)

from ...common.test_mixins import (
    get_tols,
)
from ...seed import (
    GLOBAL_SEED,
)


def _make_data(
    rng: np.random.Generator,
    nframes: int,
    natoms: int,
    dtype: torch.dtype,
    device: torch.device,
    with_dipole: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build model prediction and label dicts including latent charge/dipole."""
    model_pred = {
        "energy": torch.tensor(rng.random((nframes,)), dtype=dtype, device=device),
        "force": torch.tensor(
            rng.random((nframes, natoms, 3)), dtype=dtype, device=device
        ),
        "mask": torch.ones((nframes, natoms), dtype=dtype, device=device),
        "latent_charge": torch.tensor(
            rng.random((nframes, natoms, 1)), dtype=dtype, device=device
        ),
    }
    if with_dipole:
        model_pred["latent_dipole"] = torch.tensor(
            rng.random((nframes, natoms, 3)), dtype=dtype, device=device
        )
    label = {
        "energy": torch.tensor(rng.random((nframes,)), dtype=dtype, device=device),
        "force": torch.tensor(
            rng.random((nframes, natoms, 3)), dtype=dtype, device=device
        ),
        "find_energy": torch.tensor(1.0, dtype=dtype, device=device),
        "find_force": torch.tensor(1.0, dtype=dtype, device=device),
        "atom_charge": torch.tensor(
            rng.random((nframes, natoms, 1)), dtype=dtype, device=device
        ),
        "find_atom_charge": torch.tensor(1.0, dtype=dtype, device=device),
    }
    if with_dipole:
        label["atom_dipole"] = torch.tensor(
            rng.random((nframes, natoms, 3)), dtype=dtype, device=device
        )
        label["find_atom_dipole"] = torch.tensor(1.0, dtype=dtype, device=device)
    return model_pred, label


def _base_kwargs() -> dict:
    return {
        "starter_learning_rate": 1e-3,
        "start_pref_e": 1.0,
        "limit_pref_e": 1.0,
        "start_pref_f": 1.0,
        "limit_pref_f": 1.0,
    }


class TestEnergyLatentLoss:
    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.nframes, self.natoms = 2, 6
        self.learning_rate = 1e-3

    @pytest.mark.parametrize("prec", ["float64", "float32"])
    def test_latent_terms_active_and_serialize(self, prec) -> None:
        """Latent charge/dipole terms raise the loss and appear in more_loss.

        Also serialize/deserialize round-trip and dpmodel consistency.
        """
        rng = np.random.default_rng(GLOBAL_SEED)
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        model_pred, label = _make_data(
            rng, self.nframes, self.natoms, dtype, self.device
        )

        loss_lat = EnergyLatentLoss(
            **{
                **_base_kwargs(),
                "start_pref_lq": 2.0,
                "limit_pref_lq": 2.0,
                "start_pref_ld": 3.0,
                "limit_pref_ld": 3.0,
            },
        )
        l_lat, more_lat = loss_lat(self.learning_rate, self.natoms, model_pred, label)
        assert l_lat.shape == ()
        assert "rmse_latent_charge" in more_lat
        assert "rmse_latent_dipole" in more_lat
        assert "l2_latent_charge_loss" in more_lat
        assert "l2_latent_dipole_loss" in more_lat

        # plain energy loss (no latent prefactors) is strictly smaller
        loss_plain = EnergyLoss(**_base_kwargs())
        l_plain, _ = loss_plain(self.learning_rate, self.natoms, model_pred, label)
        assert float(l_lat) > float(l_plain)

        # serialize / deserialize round-trip
        loss_rt = EnergyLatentLoss.deserialize(loss_lat.serialize())
        l_rt, more_rt = loss_rt(self.learning_rate, self.natoms, model_pred, label)
        np.testing.assert_allclose(
            l_lat.detach().cpu().numpy(),
            l_rt.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
        for key in more_lat:
            np.testing.assert_allclose(
                more_lat[key].detach().cpu().numpy(),
                more_rt[key].detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"key={key}",
            )

        # dpmodel (numpy) consistency
        dp_loss = EnergyLatentLossDP.deserialize(loss_lat.serialize())
        mp_np = {k: v.detach().cpu().numpy() for k, v in model_pred.items()}
        lb_np = {k: v.detach().cpu().numpy() for k, v in label.items()}
        l_dp, _ = dp_loss(self.learning_rate, self.natoms, mp_np, lb_np)
        np.testing.assert_allclose(
            l_lat.detach().cpu().numpy(),
            np.array(l_dp),
            rtol=rtol,
            atol=atol,
            err_msg="pt_expt vs dpmodel",
        )

    def test_label_requirement_and_metric_names(self) -> None:
        loss = EnergyLatentLoss(
            **{
                **_base_kwargs(),
                "start_pref_lq": 1.0,
                "limit_pref_lq": 1.0,
                "start_pref_ld": 1.0,
                "limit_pref_ld": 1.0,
            },
        )
        keys = [item.key for item in loss.label_requirement]
        assert "atom_charge" in keys
        assert "atom_dipole" in keys
        assert "rmse_latent_charge" in loss.training_metric_names
        assert "rmse_latent_dipole" in loss.training_metric_names

        # charge-only supervision requests only atom_charge
        loss_q = EnergyLatentLoss(**{**_base_kwargs(), "start_pref_lq": 1.0})
        keys_q = [item.key for item in loss_q.label_requirement]
        assert "atom_charge" in keys_q and "atom_dipole" not in keys_q

    def test_zero_prefactors_degenerate(self) -> None:
        """All-zero latent prefactors reproduce the plain energy loss exactly."""
        rng = np.random.default_rng(GLOBAL_SEED)
        dtype = PRECISION_DICT["float64"]
        model_pred, label = _make_data(
            rng, self.nframes, self.natoms, dtype, self.device
        )
        loss_lat = EnergyLatentLoss(**_base_kwargs())  # all latent prefs = 0
        l_lat, more_lat = loss_lat(self.learning_rate, self.natoms, model_pred, label)
        loss_plain = EnergyLoss(**_base_kwargs())
        l_plain, _ = loss_plain(self.learning_rate, self.natoms, model_pred, label)
        np.testing.assert_allclose(
            l_lat.detach().cpu().numpy(),
            l_plain.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
        assert "rmse_latent_charge" not in more_lat

    def test_missing_dipole_output_raises(self) -> None:
        """Dipole supervision without a latent_dipole output must raise."""
        rng = np.random.default_rng(GLOBAL_SEED)
        dtype = PRECISION_DICT["float64"]
        model_pred, label = _make_data(
            rng, self.nframes, self.natoms, dtype, self.device, with_dipole=False
        )
        loss = EnergyLatentLoss(**{**_base_kwargs(), "start_pref_ld": 1.0})
        with pytest.raises(RuntimeError, match="latent_dipole"):
            loss(self.learning_rate, self.natoms, model_pred, label)

    def test_negative_prefactor_raises(self) -> None:
        with pytest.raises(ValueError, match="start_pref_lq"):
            EnergyLatentLoss(**{**_base_kwargs(), "start_pref_lq": -1.0})

    def test_latent_only_no_ef_terms(self) -> None:
        """Regression: all energy/force prefactors zero (latent-only branch).

        The base EnergyLoss used to leave ``loss`` as the int ``0`` and crash in
        ``xp.sqrt(loss)``; it must now produce a tensor and still supervise the
        latent charge/dipole terms.
        """
        rng = np.random.default_rng(GLOBAL_SEED)
        dtype = PRECISION_DICT["float64"]
        model_pred, label = _make_data(
            rng, self.nframes, self.natoms, dtype, self.device
        )
        loss = EnergyLatentLoss(
            starter_learning_rate=1e-3,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_lq=5.0,
            limit_pref_lq=5.0,
            start_pref_ld=5.0,
            limit_pref_ld=5.0,
        )
        l, more = loss(self.learning_rate, self.natoms, model_pred, label)
        assert isinstance(l, torch.Tensor) and l.shape == ()
        assert torch.isfinite(l)
        assert "rmse_latent_charge" in more and "rmse_latent_dipole" in more
        assert "rmse" in more and torch.isfinite(more["rmse"])
        # no EF terms enabled -> no EF metrics
        assert "rmse_e" not in more and "rmse_f" not in more

    def test_gradient_flows_to_latent(self) -> None:
        """The latent loss differentiates w.r.t. latent_charge/latent_dipole."""
        rng = np.random.default_rng(GLOBAL_SEED)
        dtype = PRECISION_DICT["float64"]
        model_pred, label = _make_data(
            rng, self.nframes, self.natoms, dtype, self.device
        )
        model_pred["latent_charge"] = model_pred["latent_charge"].requires_grad_(True)
        model_pred["latent_dipole"] = model_pred["latent_dipole"].requires_grad_(True)
        loss = EnergyLatentLoss(
            **{
                **_base_kwargs(),
                "start_pref_lq": 1.0,
                "limit_pref_lq": 1.0,
                "start_pref_ld": 1.0,
                "limit_pref_ld": 1.0,
            },
        )
        l, _ = loss(self.learning_rate, self.natoms, model_pred, label)
        l.backward()
        assert model_pred["latent_charge"].grad is not None
        assert torch.count_nonzero(model_pred["latent_charge"].grad) > 0
        assert model_pred["latent_dipole"].grad is not None
        assert torch.count_nonzero(model_pred["latent_dipole"].grad) > 0
