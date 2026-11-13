# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the external electric field (efield) of latent_multipole + LEF.

The field enters the latent electrostatic feedback (LEF), polarizing the
latent dipoles, and adds a dipole-field energy. It is gated to LEF-only
(``lr_update_steps > 0``); every other configuration raises.
"""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.long_range.latent_multipole import (
    LatentMultipoleLongRange as LatentMultipoleLongRangeDP,
)
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils import env


def _model(efield: bool, steps: int = 2, seed: int = 17):
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
            "lr_update_steps": steps,
            "efield": efield,
        },
    }
    return get_model(cfg).to(env.DEVICE).eval()


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


def _efield(value, seed_frames=2):
    return torch.full(
        (seed_frames, 3), float(value), dtype=torch.float64, device=env.DEVICE
    )


class TestEfieldGating:
    def test_efield_requires_lef(self) -> None:
        with pytest.raises(ValueError, match="latent electrostatic feedback"):
            _model(efield=True, steps=0)

    def test_efield_ok_with_lef(self) -> None:
        m = _model(efield=True, steps=2)
        assert m.atomic_model.long_range.efield is True

    def test_no_efield_default(self) -> None:
        m = _model(efield=False, steps=2)
        assert m.atomic_model.long_range.efield is False

    def test_charge_only_plugin_rejects_efield(self) -> None:
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
                "seed": 17,
                "add_chg_spin_ebd": True,
                "default_chg_spin": [0.0, 1.0],
            },
            "fitting_net": {
                "type": "dpa4_ener",
                "neuron": [16, 16],
                "precision": "float64",
                "seed": 19,
            },
            "long_range": {
                "type": "latent_charge",
                "channels": 1,
                "charge_net": {"neuron": [8, 8]},
                "kernel": {"type": "les", "alpha": 0.9},
                "charge_constraint": {"source": "neutral"},
            },
        }
        m = get_model(cfg).to(env.DEVICE).eval()
        coord, atype, cs = _batch()
        with pytest.raises(ValueError, match="charge-only"):
            m(coord, atype, charge_spin=cs, efield=_efield(0.1))


class TestEfieldSerialization:
    def test_roundtrip(self) -> None:
        m = _model(efield=True, steps=2)
        rec = m.atomic_model.long_range.serialize()
        assert rec.get("efield") is True
        lr = LatentMultipoleLongRangeDP.deserialize(rec)
        assert lr.efield is True

    def test_backward_compatible_old_record(self) -> None:
        m = _model(efield=True, steps=2)
        rec = m.atomic_model.long_range.serialize()
        rec.pop("efield")
        lr = LatentMultipoleLongRangeDP.deserialize(rec)
        assert lr.efield is False

    def test_efield_false_not_serialized(self) -> None:
        m = _model(efield=False, steps=2)
        assert "efield" not in m.atomic_model.long_range.serialize()


class TestEfieldPhysics:
    def test_dipole_polarizes_along_field(self) -> None:
        m = _model(efield=True, steps=2)
        coord, atype, cs = _batch()
        e_axis = torch.zeros(2, 3, dtype=torch.float64, device=env.DEVICE)
        e_axis[:, 0] = 0.5  # field along x
        p0 = m(coord, atype, charge_spin=cs, efield=_efield(0.0))["latent_dipole"]
        pE = m(coord, atype, charge_spin=cs, efield=e_axis)["latent_dipole"]
        dp = (pE - p0).detach().cpu().numpy()
        # polarization is dominantly along the applied (x) direction; the
        # transverse components are nonzero but far smaller (the dipoles also
        # respond to the system's own field, which is not purely along x).
        assert np.abs(dp[..., 0]).mean() > 1e-3
        assert np.abs(dp[..., 1]).mean() < 0.1 * np.abs(dp[..., 0]).mean()
        assert np.abs(dp[..., 2]).mean() < 0.1 * np.abs(dp[..., 0]).mean()

    def test_energy_changes_with_field(self) -> None:
        m = _model(efield=True, steps=2)
        coord, atype, cs = _batch()
        e0 = m(coord, atype, charge_spin=cs, efield=_efield(0.0))["energy"]
        eE = m(coord, atype, charge_spin=cs, efield=_efield(0.5))["energy"]
        assert not torch.allclose(e0, eE)

    def test_efield_disabled_ignores_argument(self) -> None:
        """A field-free model is unaffected even if an efield argument is passed."""
        m = _model(efield=False, steps=2)
        coord, atype, cs = _batch()
        e0 = m(coord, atype, charge_spin=cs)["energy"]
        e_arg = m(coord, atype, charge_spin=cs, efield=_efield(0.5))["energy"]
        torch.testing.assert_close(e0, e_arg)

    def test_force_differs_with_field(self) -> None:
        """The autograd force changes under field (polarization force)."""
        m = _model(efield=True, steps=2)
        coord, atype, cs = _batch()
        f0 = m(coord, atype, charge_spin=cs, efield=_efield(0.0))["force"]
        fE = m(coord, atype, charge_spin=cs, efield=_efield(0.5))["force"]
        assert not torch.allclose(f0, fE)

    def test_direct_field_force_qE_added(self) -> None:
        """The wrapper adds the uniform field force q_i E to the atomic force."""
        m = _model(efield=True, steps=1)
        coord, atype, cs = _batch()
        E = 1.0
        ef = torch.tensor(
            [[E, 0.0, 0.0], [E, 0.0, 0.0]], dtype=torch.float64, device=env.DEVICE
        )
        out_full = m(coord, atype, charge_spin=cs, efield=ef)
        q = out_full["latent_charge"]  # (nf, nloc, 1)
        # Re-run with the efield_force side output stripped: the wrapper then
        # adds nothing, so the difference is exactly q_i E.
        orig = m.atomic_model.long_range.call_graph

        def stripped(*a, **k):
            ret = orig(*a, **k)
            ret.pop("efield_force", None)
            return ret

        m.atomic_model.long_range.call_graph = stripped
        try:
            out_auto = m(coord, atype, charge_spin=cs, efield=ef)
        finally:
            m.atomic_model.long_range.call_graph = orig
        diff = out_full["force"] - out_auto["force"]  # (nf, nloc, 3)
        # q_i E broadcast to (nf, nloc, 3)
        expected = q * ef[:, None, :].expand(-1, q.shape[1], -1)
        torch.testing.assert_close(diff, expected, rtol=1e-10, atol=1e-8)
