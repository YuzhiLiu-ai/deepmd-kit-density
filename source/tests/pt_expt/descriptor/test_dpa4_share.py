# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for DescrptDPA4 multi-task parameter sharing (share_params)."""

import numpy as np
import pytest
import torch

from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils import env


def _model(seed: int, with_spin: bool = False):
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


class TestDPA4ShareParams:
    def test_level0_shares_all_learnable_params(self) -> None:
        A = _model(17)
        B = _model(117)  # different descriptor seed -> different init
        dA, dB = A.atomic_model.descriptor, B.atomic_model.descriptor

        pa = dict(dA.named_parameters())
        pb = dict(dB.named_parameters())
        assert pa.keys() == pb.keys()
        # sanity: different seeds give different weights before sharing
        assert not all(torch.equal(pa[k], pb[k]) for k in pa)

        dB.share_params(dA, 0)

        pa = dict(dA.named_parameters())
        pb = dict(dB.named_parameters())
        assert pa.keys() == pb.keys()
        # every learnable descriptor parameter is now the SAME object
        for k in pa:
            assert pa[k] is pb[k], f"param {k} not shared"

    def test_branch_local_state_stays_independent(self) -> None:
        A = _model(17)
        B = _model(117)
        dA, dB = A.atomic_model.descriptor, B.atomic_model.descriptor
        dB.share_params(dA, 0)
        # routing / version buffers and the exclusion mask are per-branch
        assert dA.graph_lower_disabled is not dB.graph_lower_disabled
        assert dA.version_tensor is not dB.version_tensor
        assert dA.emask is not dB.emask

    def test_forward_identical_after_sharing(self) -> None:
        A = _model(17).eval()
        B = _model(117).eval()
        B.atomic_model.descriptor.share_params(A.atomic_model.descriptor, 0)
        coord, atype, cs = _batch()
        # force is produced via autograd inside forward, so grad must stay enabled
        oA = A(coord, atype, charge_spin=cs)
        oB = B(coord, atype, charge_spin=cs)
        torch.testing.assert_close(oA["energy"].detach(), oB["energy"].detach())
        torch.testing.assert_close(
            oA["latent_charge"].detach(), oB["latent_charge"].detach()
        )

    def test_gradient_flows_across_shared_params(self) -> None:
        A = _model(17)
        B = _model(117)
        A.train(True)
        B.train(True)
        dA, dB = A.atomic_model.descriptor, B.atomic_model.descriptor
        dB.share_params(dA, 0)
        coord, atype, cs = _batch()
        # loss on branch B only; the shared descriptor lives in both
        loss = B(coord, atype, charge_spin=cs)["energy"].square().sum()
        loss.backward()
        shared = dict(dA.named_parameters())
        got_grad = [
            k
            for k, p in shared.items()
            if p.grad is not None and torch.count_nonzero(p.grad) > 0
        ]
        assert got_grad, (
            "shared descriptor params should receive gradients via branch B"
        )

    def test_type_mismatch_raises(self) -> None:
        A = _model(17)
        dA = A.atomic_model.descriptor
        with pytest.raises((TypeError, AttributeError)):
            dA.share_params(object(), 0)

    def test_bad_level_raises(self) -> None:
        A = _model(17)
        B = _model(117)
        with pytest.raises(NotImplementedError):
            B.atomic_model.descriptor.share_params(A.atomic_model.descriptor, 5)

    def test_structural_mismatch_raises(self) -> None:
        A = _model(17)
        B = _model(117)
        dB = B.atomic_model.descriptor
        dB.channels = 999  # corrupt a structural field
        with pytest.raises(ValueError, match="identical structural"):
            dB.share_params(A.atomic_model.descriptor, 0)
