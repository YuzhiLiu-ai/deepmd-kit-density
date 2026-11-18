# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the external electric field (efield) of latent_multipole + LEF.

The field enters the latent electrostatic feedback (LEF), polarizing the
latent dipoles, and adds a dipole-field energy. It is gated to LEF-only
(``lr_update_steps > 0``); every other configuration raises. Two transports:
the explicit per-frame ``efield`` input (API/tests, eager-only) and
``fparam[:, :3]`` (training data, compile/export/deploy-friendly).
"""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.long_range.latent_multipole import (
    LatentMultipoleLongRange as LatentMultipoleLongRangeDP,
)
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils import env


def _model_config(efield: bool, steps: int = 2, seed: int = 17) -> dict:
    return {
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


def _model(efield: bool, steps: int = 2, seed: int = 17):
    return get_model(_model_config(efield, steps, seed)).to(env.DEVICE).eval()


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


class TestEfieldTrainingWiring:
    """The training pipeline feeds the field via fparam[:, :3] (V/AA)."""

    def test_data_requirement_registered_when_enabled(self) -> None:
        from deepmd.pt_expt.train.training import get_additional_data_requirement

        reqs = get_additional_data_requirement(_model(efield=True, steps=2))
        # dim_fparam == 0 model: the field transport registers fparam dim 3.
        fparam_reqs = [r for r in reqs if r.key == "fparam"]
        assert len(fparam_reqs) == 1
        req = fparam_reqs[0]
        assert req.ndof == 3
        assert req.atomic is False
        # Optional: a system without fparam.npy trains field-free.
        assert req.must is False
        # No separate efield data key.
        assert all(r.key != "efield" for r in reqs)

    def test_no_data_requirement_when_disabled(self) -> None:
        from deepmd.pt_expt.train.training import get_additional_data_requirement

        reqs = get_additional_data_requirement(_model(efield=False, steps=2))
        assert all(r.key not in ("efield", "fparam") for r in reqs)

    def test_split_batch_routes_efield_to_inputs(self) -> None:
        from deepmd.dpmodel.utils.batch import split_batch

        batch = {
            "coord": np.zeros((2, 15)),
            "type": np.zeros((2, 5), dtype=np.int64),
            "efield": np.ones((2, 3)),
            "energy": np.zeros((2, 1)),
            "find_efield": np.bool_(True),
        }
        input_dict, label_dict = split_batch(batch)
        assert "efield" in input_dict
        assert "efield" not in label_dict
        # The find flag stays with the labels, like find_fparam.
        assert "find_efield" in label_dict

    def test_wrapper_forward_passes_efield(self) -> None:
        from deepmd.pt_expt.train.wrapper import ModelWrapper

        m = _model(efield=True, steps=1)
        wrapper = ModelWrapper(m).eval()
        coord, atype, cs = _batch()
        ef = _efield(0.5)
        pred_e, _, _ = wrapper(coord, atype, charge_spin=cs, efield=ef)
        pred_0, _, _ = wrapper(coord, atype, charge_spin=cs)
        # The field reaches the plugin: polarized energy differs from the
        # field-free evaluation.
        assert not torch.allclose(pred_e["energy"], pred_0["energy"])

    def test_wrapper_forward_ignores_efield_kwarg_when_absent(self) -> None:
        """A field-free model is never called with the efield key."""
        from deepmd.pt_expt.train.wrapper import ModelWrapper

        m = _model(efield=False, steps=1)
        wrapper = ModelWrapper(m).eval()
        coord, atype, cs = _batch()
        pred, _, _ = wrapper(coord, atype, charge_spin=cs)
        direct = m(coord, atype, charge_spin=cs)
        torch.testing.assert_close(pred["energy"], direct["energy"])

    def test_compiled_model_rejects_efield(self) -> None:
        from deepmd.pt_expt.train.training import _CompiledModel

        m = _model(efield=True, steps=1)
        compiled = _CompiledModel(m, structure_key=(0,))
        compiled.train()
        coord, atype, cs = _batch()
        with pytest.raises(NotImplementedError, match="enable_compile"):
            compiled(
                coord,
                atype,
                None,
                charge_spin=cs,
                efield=_efield(0.5),
            )


class TestEfieldFparamTransport:
    """The compile/export-friendly transport: the field rides fparam[:, :3]."""

    def test_fparam_transport_matches_explicit_kwarg(self) -> None:
        m = _model(efield=True, steps=2)
        coord, atype, cs = _batch()
        fp = torch.tensor(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        out_fp = m(coord, atype, charge_spin=cs, fparam=fp)
        out_kw = m(coord, atype, charge_spin=cs, efield=fp)
        torch.testing.assert_close(out_fp["energy"], out_kw["energy"])
        torch.testing.assert_close(out_fp["force"], out_kw["force"])

    def test_fparam_transport_polarizes(self) -> None:
        m = _model(efield=True, steps=2)
        coord, atype, cs = _batch()
        fp = torch.tensor(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        e0 = m(coord, atype, charge_spin=cs)["energy"]
        e_fp = m(coord, atype, charge_spin=cs, fparam=fp)["energy"]
        assert not torch.allclose(e0, e_fp)

    def test_explicit_kwarg_wins_over_fparam(self) -> None:
        m = _model(efield=True, steps=2)
        coord, atype, cs = _batch()
        fp = torch.tensor(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        kw = _efield(0.1)
        out_both = m(coord, atype, charge_spin=cs, fparam=fp, efield=kw)
        out_kw = m(coord, atype, charge_spin=cs, efield=kw)
        torch.testing.assert_close(out_both["energy"], out_kw["energy"])

    def test_short_fparam_rejected(self) -> None:
        m = _model(efield=True, steps=2)
        coord, atype, cs = _batch()
        fp = torch.full((2, 1), 0.5, dtype=torch.float64, device=env.DEVICE)
        with pytest.raises(ValueError, match="fparam"):
            m(coord, atype, charge_spin=cs, fparam=fp)

    def test_field_free_model_ignores_fparam(self) -> None:
        """efield=False: fparam stays invisible (dim_fparam is 0)."""
        m = _model(efield=False, steps=2)
        coord, atype, cs = _batch()
        fp = torch.tensor(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        e0 = m(coord, atype, charge_spin=cs)["energy"]
        e_fp = m(coord, atype, charge_spin=cs, fparam=fp)["energy"]
        torch.testing.assert_close(e0, e_fp)

    def _combo_model(self, fparam_index: int, steps: int = 2):
        """Efield + charge constraint from fparam (numb_fparam=4 layout)."""
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
                "use_env_seed": True,
            },
            "fitting_net": {
                "type": "dpa4_ener",
                "neuron": [16, 16],
                "precision": "float64",
                "seed": 19,
                "numb_fparam": 4,
            },
            "long_range": {
                "type": "latent_multipole",
                "charge_net": {"neuron": [8, 8], "use_glu": False},
                "kernel": {"type": "les", "alpha": 0.9},
                "charge_constraint": {
                    "source": "fparam",
                    "fparam_index": fparam_index,
                },
                "lr_update_steps": steps,
                "efield": True,
            },
        }
        return get_model(cfg).to(env.DEVICE).eval()

    def test_fparam_charge_constraint_overlap_rejected(self) -> None:
        """fparam_index < 3 overlaps the field slice fparam[:3]."""
        with pytest.raises(ValueError, match="fparam_index"):
            self._combo_model(fparam_index=2)

    def test_efield_and_charge_spin_constraint_coexist(self) -> None:
        """The production combo: field via fparam[:3], charge via charge_spin."""
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
                "charge_constraint": {"source": "charge_spin"},
                "lr_update_steps": 2,
                "efield": True,
            },
        }
        m = get_model(cfg).to(env.DEVICE).eval()
        coord, atype, _ = _batch()
        cs = torch.tensor(
            [[-1.0, 2.0], [0.5, 1.0]], dtype=torch.float64, device=env.DEVICE
        )
        fp = torch.tensor(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        out = m(coord, atype, charge_spin=cs, fparam=fp)
        # The field in fparam[:3] is active.
        out_nofield = m(coord, atype, charge_spin=cs, fparam=fp * 0.0)
        assert not torch.allclose(out["energy"], out_nofield["energy"])
        # The charge constraint reads charge_spin[:, 0] (spin channel untouched).
        q_sum = out["latent_charge"].sum(dim=1).squeeze(-1)  # (nf,)
        torch.testing.assert_close(q_sum, cs[:, 0], rtol=1e-8, atol=1e-8)

    def test_efield_and_fparam_charge_constraint_coexist(self) -> None:
        """Fparam = [Ex, Ey, Ez, Q]: field from [:3], charge target from [3]."""
        m = self._combo_model(fparam_index=3)
        coord, atype, cs = _batch()
        fp = torch.tensor(
            [[0.0, 0.0, 0.5, 1.5], [0.0, 0.0, 0.5, -0.5]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        out = m(coord, atype, charge_spin=cs, fparam=fp)
        # 1. The field in fparam[:3] is active: energy matches the explicit
        #    kwarg evaluation with the same field.
        out_kw = m(coord, atype, charge_spin=cs, fparam=fp, efield=fp[:, :3])
        torch.testing.assert_close(out["energy"], out_kw["energy"])
        out_nofield = m(coord, atype, charge_spin=cs, fparam=fp * 0.0)
        assert not torch.allclose(out["energy"], out_nofield["energy"])
        # 2. The charge constraint reads fparam[:, 3]: per-frame latent
        #    charges sum to the target.
        q_sum = out["latent_charge"].sum(dim=1).squeeze(-1)  # (nf,)
        torch.testing.assert_close(q_sum, fp[:, 3], rtol=1e-8, atol=1e-8)


class TestEfieldInference:
    """dp test / DeepEval transport the field via fparam[:, :3]."""

    @staticmethod
    def _save_ckpt(tmp_path) -> str:
        from deepmd.pt_expt.train.wrapper import ModelWrapper

        cfg = _model_config(efield=True, steps=2)
        model = get_model(cfg).to(env.DEVICE).eval()
        wrapper = ModelWrapper(model, model_params=cfg)
        ckpt = str(tmp_path / "model.pt")
        torch.save({"model": wrapper.state_dict()}, ckpt)
        return ckpt

    def test_deep_eval_transports_fparam_field(self, tmp_path) -> None:
        """The report's minimum regression: eval responds to fparam[:, :3]."""
        from deepmd.infer.deep_eval import DeepEval

        ckpt = self._save_ckpt(tmp_path)
        dp = DeepEval(ckpt)
        assert dp.has_efield is True
        assert dp.efield_mode == "uniform_fparam"
        assert dp.get_dim_fparam() == 0

        coord, atype, _ = _batch()
        coord_np = coord.detach().cpu().numpy()
        atype_np = atype[0].detach().cpu().numpy().astype(np.int32)
        field = np.zeros((2, 3))
        field[:, 2] = 0.5
        out0 = dp.eval(coord_np, None, atype_np, fparam=np.zeros((2, 3)))
        out1 = dp.eval(coord_np, None, atype_np, fparam=field)
        assert not np.allclose(out0[0], out1[0])  # energy responds
        assert not np.allclose(out0[1], out1[1])  # force responds
        # And a zero-field evaluation without fparam still works.
        out2 = dp.eval(coord_np, None, atype_np)
        np.testing.assert_allclose(out2[0], out0[0], rtol=1e-10, atol=1e-10)

    @staticmethod
    def _write_system(root, field_value: float) -> None:
        import pathlib

        root = pathlib.Path(root)
        (root / "set.000").mkdir(parents=True, exist_ok=True)
        coord, atype, _ = _batch()
        coord_np = coord.detach().cpu().numpy()
        atype_np = atype[0].detach().cpu().numpy()
        nf = coord_np.shape[0]
        (root / "type.raw").write_text(" ".join(str(int(t)) for t in atype_np))
        np.save(root / "set.000" / "coord.npy", coord_np.reshape(nf, -1))
        np.save(root / "set.000" / "box.npy", np.zeros((nf, 9)))
        np.save(root / "set.000" / "energy.npy", np.zeros((nf, 1)))
        np.save(root / "set.000" / "force.npy", np.zeros_like(coord_np).reshape(nf, -1))
        fparam = np.zeros((nf, 3))
        fparam[:, 2] = field_value
        np.save(root / "set.000" / "fparam.npy", fparam)

    def test_dp_test_end_to_end_field_response(self, tmp_path) -> None:
        """detail.f.out of a zero-field and a field system must differ."""
        from deepmd.entrypoints.test import test as dp_test

        ckpt = self._save_ckpt(tmp_path)
        outs = {}
        for name, e in (("zero", 0.0), ("field", 0.5)):
            sys_dir = tmp_path / name
            self._write_system(sys_dir, e)
            detail = str(tmp_path / f"detail_{name}")
            dp_test(
                model=ckpt,
                system=str(sys_dir),
                datafile=None,
                set_prefix="set",
                numb_test=0,
                rand_seed=None,
                shuffle_test=False,
                detail_file=detail,
                atomic=False,
            )
            outs[name] = np.loadtxt(detail + ".f.out")
        assert not np.allclose(outs["zero"], outs["field"])

    def test_dp_test_without_fparam_file(self, tmp_path) -> None:
        """A zero-field system without fparam.npy still tests (optional)."""
        import os

        from deepmd.entrypoints.test import test as dp_test

        ckpt = self._save_ckpt(tmp_path)
        sys_dir = tmp_path / "zero_nofile"
        self._write_system(sys_dir, 0.0)
        os.remove(sys_dir / "set.000" / "fparam.npy")
        detail = str(tmp_path / "detail_nofile")
        dp_test(
            model=ckpt,
            system=str(sys_dir),
            datafile=None,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail,
            atomic=False,
        )
        assert os.path.exists(detail + ".f.out")

    def test_eval_exposes_latent_outputs(self, tmp_path) -> None:
        """dp.eval surfaces latent_charge/latent_dipole with correct shapes."""
        from deepmd.infer.deep_eval import DeepEval

        ckpt = self._save_ckpt(tmp_path)
        dp = DeepEval(ckpt)
        coord, atype, _ = _batch()
        # The non-mixed-type eval interface shares one atype across frames;
        # align the reference forward to that convention.
        atype = atype[0:1].repeat(2, 1)
        coord_np = coord.detach().cpu().numpy()
        atype_np = atype[0].detach().cpu().numpy().astype(np.int32)

        # Backend eval dict carries the latent outputs under atomic=True,
        # with the declared shapes.
        out = dp.deep_eval.eval(coord_np, None, atype_np, atomic=True)
        assert out["latent_charge"].shape == (2, 5, 1)
        assert out["latent_dipole"].shape == (2, 5, 3)

        # They match the direct model forward exactly.
        m = dp.deep_eval._dpmodel.eval()
        ref = m(coord, atype)
        np.testing.assert_allclose(
            out["latent_charge"],
            ref["latent_charge"].detach().cpu().numpy(),
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            out["latent_dipole"],
            ref["latent_dipole"].detach().cpu().numpy(),
            rtol=1e-8,
            atol=1e-10,
        )

        # atomic=False keeps the standard output set.
        out_na = dp.deep_eval.eval(coord_np, None, atype_np, atomic=False)
        assert "latent_dipole" not in out_na

        # High-level passthrough appends the requested outputs.
        res = dp.eval(
            coord_np, None, atype_np, atomic=True, extra_outputs=["latent_dipole"]
        )
        np.testing.assert_allclose(res[-1], out["latent_dipole"], rtol=1e-10)

        # extra_outputs require atomic=True.
        with pytest.raises(ValueError, match="atomic=True"):
            dp.eval(coord_np, None, atype_np, extra_outputs=["latent_dipole"])
