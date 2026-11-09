# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-task sharing of the long-range plugin (charge/dipole head).

The point of sharing the ``long_range`` plugin across branches is that an
energy/force-only branch (no charge/dipole labels) still trains the shared
charge/dipole head through the long-range energy term, while the labelled
branch supervises it directly. These tests assert exactly that coupling.
"""

import numpy as np
import pytest
import torch

from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils import env


def _model(seed: int):
    cfg = {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "dpa4",
            "rcut": 3.0,
            "sel": 20,
            "channels": 8,
            "lmax": 2,
            "n_radial": 4,
            "precision": "float64",
            "seed": seed,
            "add_chg_spin_ebd": True,
            "default_chg_spin": [0.0, 1.0],
            "use_env_seed": True,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [16, 16],
            "precision": "float64",
            "seed": 19,
        },
        "long_range": {
            "type": "latent_multipole",
            "charge_net": {"neuron": [8, 8], "use_glu": False},
            "kernel": {"type": "les", "alpha": 0.9},
            "charge_constraint": {"source": "neutral"},
            "lr_update_steps": 1,
        },
    }
    return get_model(cfg).to(env.DEVICE)


def _batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    coord = torch.tensor(
        rng.uniform(0.2, 4.0, (2, 5, 3)), dtype=torch.float64, device=env.DEVICE
    )
    atype = torch.tensor(
        rng.integers(0, 2, (2, 5)), dtype=torch.int64, device=env.DEVICE
    )
    cs = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64, device=env.DEVICE)
    return coord, atype, cs


def _share(ef, dipole):
    """Share descriptor + long_range, as the multi-task framework does."""
    dipole.atomic_model.descriptor.share_params(ef.atomic_model.descriptor, 0)
    dipole.atomic_model.long_range.share_params(ef.atomic_model.long_range, 0)


class TestLongRangeSharing:
    def test_plugin_params_shared_as_same_objects(self) -> None:
        ef, dipole = _model(17), _model(117)
        _share(ef, dipole)
        lr_ef = ef.atomic_model.long_range
        lr_dip = dipole.atomic_model.long_range
        assert lr_dip.charge_net is lr_ef.charge_net
        assert lr_dip.kernel is lr_ef.kernel
        pe = dict(lr_ef.named_parameters())
        pd = dict(lr_dip.named_parameters())
        assert pe.keys() == pd.keys()
        for k in pe:
            assert pe[k] is pd[k], f"long_range param {k} not shared"

    def test_shared_head_consistent_output(self) -> None:
        ef, dipole = _model(17), _model(117)
        _share(ef, dipole)
        # also share descriptor so both branches see identical inputs->latent
        ef.eval()
        dipole.eval()
        coord, atype, cs = _batch()
        o_ef = ef(coord, atype, charge_spin=cs)
        o_dip = dipole(coord, atype, charge_spin=cs)
        torch.testing.assert_close(
            o_ef["latent_charge"].detach(), o_dip["latent_charge"].detach()
        )
        torch.testing.assert_close(
            o_ef["latent_dipole"].detach(), o_dip["latent_dipole"].detach()
        )

    def test_energy_only_branch_trains_shared_head(self) -> None:
        """The user's core ask: an EF-only loss must move the shared charge/dipole head."""
        ef, dipole = _model(17), _model(117)
        _share(ef, dipole)
        ef.train(True)
        coord, atype, cs = _batch()
        # ef branch: pure energy/force-style objective, no latent labels
        out = ef(coord, atype, charge_spin=cs)
        loss = out["energy"].square().sum() + out["force"].square().sum()
        loss.backward()
        head = ef.atomic_model.long_range.charge_net
        got = [
            n
            for n, p in head.named_parameters()
            if p.grad is not None and torch.count_nonzero(p.grad) > 0
        ]
        assert got, "EF-only loss should backprop into the shared charge/dipole head"

    def test_labelled_branch_supervises_shared_head(self) -> None:
        """The dipole branch supervises latent_charge/latent_dipole directly."""
        ef, dipole = _model(17), _model(117)
        _share(ef, dipole)
        dipole.train(True)
        coord, atype, cs = _batch()
        out = dipole(coord, atype, charge_spin=cs)
        loss = out["latent_charge"].square().sum() + out["latent_dipole"].square().sum()
        loss.backward()
        head = ef.atomic_model.long_range.charge_net  # shared with dipole
        got = [
            n
            for n, p in head.named_parameters()
            if p.grad is not None and torch.count_nonzero(p.grad) > 0
        ]
        assert got, (
            "latent supervision should backprop into the shared charge/dipole head"
        )

    def test_type_mismatch_raises(self) -> None:
        ef = _model(17)
        with pytest.raises((TypeError, AttributeError)):
            ef.atomic_model.long_range.share_params(object(), 0)

    def test_bad_level_raises(self) -> None:
        ef, dipole = _model(17), _model(117)
        with pytest.raises(NotImplementedError):
            dipole.atomic_model.long_range.share_params(ef.atomic_model.long_range, 3)
