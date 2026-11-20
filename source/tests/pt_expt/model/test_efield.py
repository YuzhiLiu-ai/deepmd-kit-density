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


class TestEfieldBranch:
    """The continuous-branch charge-field potential: conservative, smooth,
    periodic, and translation invariant for neutral frames.
    """

    @staticmethod
    def _model(
        efield_potential: str = "branch", periodic: bool = False, steps: int = 2
    ):
        cfg = _model_config(efield=True, steps=steps)
        cfg["long_range"]["efield_potential"] = efield_potential
        if periodic:
            cfg["long_range"]["periodic"] = True
            cfg["long_range"]["k_grid_mode"] = "kgrid"
        return get_model(cfg).to(env.DEVICE).eval()

    @staticmethod
    def _field(value, nf=1):
        return torch.tensor(
            [[0.0, 0.0, float(value)]] * nf, dtype=torch.float64, device=env.DEVICE
        )

    def test_conservative_finite_difference(self) -> None:
        """Branch force = autograd energy gradient + the explicit q_i E side
        output (the potential's explicit derivative, unreachable from the
        edge-vector leaf).
        """
        m = self._model()
        coord, atype, cs = _batch()
        fp = self._field(0.1, nf=2)
        coord = coord.requires_grad_(True)
        out = m(coord, atype, charge_spin=cs, fparam=fp)
        # Finite differences of the energy match the model force.
        eps = 1e-6
        for f in range(2):
            for i in range(5):
                for d in range(3):
                    dp = torch.zeros_like(coord)
                    dp[f, i, d] = eps
                    e_plus = m(coord.detach() + dp, atype, charge_spin=cs, fparam=fp)[
                        "energy"
                    ]
                    e_minus = m(coord.detach() - dp, atype, charge_spin=cs, fparam=fp)[
                        "energy"
                    ]
                    fd = -(e_plus - e_minus).flatten()[f] / (2 * eps)
                    assert abs(fd.item() - out["force"][f, i, d].item()) < 1e-5, (
                        f"frame {f} atom {i} dim {d}: FD {fd.item():.6e} vs "
                        f"force {out['force'][f, i, d].item():.6e}"
                    )

    def test_translation_invariant_when_neutral(self) -> None:
        """For the neutral constraint the branch energy is translation invariant."""
        m = self._model()
        coord, atype, cs = _batch()
        fp = self._field(0.1, nf=2)
        shift = torch.tensor([1.3, -2.1, 0.7], dtype=torch.float64, device=env.DEVICE)
        e0 = m(coord, atype, charge_spin=cs, fparam=fp)["energy"]
        e1 = m(coord + shift, atype, charge_spin=cs, fparam=fp)["energy"]
        torch.testing.assert_close(e0, e1, rtol=1e-8, atol=1e-8)

    def test_response_linear_in_field(self) -> None:
        """The branch coupling is linear in E: dF(2E) = 2 dF(E)."""
        m = self._model()
        coord, atype, cs = _batch()
        zero = torch.zeros((2, 3), dtype=torch.float64, device=env.DEVICE)
        f0 = m(coord, atype, charge_spin=cs, fparam=zero)["force"]
        # Dominated by the linear q*E term; the LEF relaxation is mildly
        # nonlinear in the field, so judge the whole-response norm in the
        # linear-response regime (small fields).
        d_half = (
            m(coord, atype, charge_spin=cs, fparam=self._field(0.05, nf=2))["force"]
            - f0
        )
        d_full = (
            m(coord, atype, charge_spin=cs, fparam=self._field(0.1, nf=2))["force"] - f0
        )
        rel = (d_full - 2 * d_half).norm() / (2 * d_half).norm()
        assert rel < 0.05, rel

    def test_branch_energy_enters(self) -> None:
        """With the branch potential the energy responds to the field even
        when the LEF state barely moves (direct check the term is present).
        """
        m = self._model(steps=1)
        coord, atype, cs = _batch()
        zero = torch.zeros((2, 3), dtype=torch.float64, device=env.DEVICE)
        e0 = m(coord, atype, charge_spin=cs, fparam=zero)["energy"]
        e1 = m(coord, atype, charge_spin=cs, fparam=self._field(0.1, nf=2))["energy"]
        assert not torch.allclose(e0, e1)

    def test_periodic_whole_molecule_lattice_translation(self) -> None:
        """Translating a whole neutral molecule by a lattice vector leaves the
        field energy AND the forces unchanged (continuous branch).

        This is the property the wrapped sawtooth violated: per-atom wrapping
        creates artificial dipoles (and ~q E L energy spikes) whenever a
        neutral molecule straddles the cell edge.
        """
        m = self._model(periodic=True)
        L = 8.0
        box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * L)[None]
        coord = torch.tensor(
            [[[2.0, 4.0, 0.40 * L], [2.0, 4.0, 0.55 * L]]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        atype = torch.tensor([[0, 1]], dtype=torch.int64, device=env.DEVICE)
        cs = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device=env.DEVICE)
        fp = self._field(0.1)
        out_a = m(coord, atype, charge_spin=cs, fparam=fp, box=box)
        shift = torch.tensor([[[0.0, 0.0, L]]], dtype=torch.float64, device=env.DEVICE)
        out_b = m(coord + shift, atype, charge_spin=cs, fparam=fp, box=box)
        torch.testing.assert_close(
            out_b["energy"], out_a["energy"], rtol=1e-8, atol=1e-8
        )
        torch.testing.assert_close(out_b["force"], out_a["force"], rtol=1e-8, atol=1e-8)

    def test_periodic_cross_boundary_energy_continuous(self) -> None:
        """Moving one atom across the cell boundary changes the energy
        CONTINUOUSLY and leaves its field force finite -- no wrap spike.

        The energy gain is the linear dipole term -E * q0 * dz on the
        unwrapped branch (the internal MIC geometry is unchanged, and the
        LEF relaxation contributes only a small second-order shift), and the
        force at the crossed configuration still matches finite differences.
        """
        m = self._model(periodic=True)
        L = 8.0
        box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * L)[None]
        # Two atoms near the boundary, close via the minimum image.
        coord = torch.tensor(
            [[[2.0, 4.0, 0.05 * L], [2.0, 4.0, 0.50 * L]]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        atype = torch.tensor([[0, 1]], dtype=torch.int64, device=env.DEVICE)
        cs = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device=env.DEVICE)
        E = 0.1
        fp = self._field(E)
        e_a = m(coord, atype, charge_spin=cs, fparam=fp, box=box)["energy"]
        q0 = m(coord, atype, charge_spin=cs, fparam=fp, box=box)["latent_charge"][
            0, 0, 0
        ]
        # Move atom 0 across the boundary: z = 0.05L -> -0.05L (continuous
        # branch: NO wrap).  The internal (MIC) geometry is unchanged, so the
        # difference is the linear dipole term -E * q0 * dz.
        dz = -0.10 * L
        coord_b = coord.clone()
        coord_b[0, 0, 2] += dz
        out_b = m(coord_b, atype, charge_spin=cs, fparam=fp, box=box)
        e_b = out_b["energy"]
        expected = -E * q0 * dz
        # The dominant term is the linear dipole shift; the LEF state also
        # relaxes slightly between the two configurations (q changes), which
        # adds a small second-order relaxation energy on top.
        torch.testing.assert_close(
            (e_b - e_a).flatten()[0], expected, rtol=0.05, atol=1e-3
        )
        # The force at the crossed configuration is finite and conservative:
        # it matches finite differences of the energy there.
        eps = 1e-6
        for i in range(2):
            for d in range(3):
                dp = torch.zeros_like(coord_b)
                dp[0, i, d] = eps
                e_plus = m(coord_b + dp, atype, charge_spin=cs, fparam=fp, box=box)[
                    "energy"
                ]
                e_minus = m(coord_b - dp, atype, charge_spin=cs, fparam=fp, box=box)[
                    "energy"
                ]
                fd = -(e_plus - e_minus).flatten()[0] / (2 * eps)
                assert abs(fd.item() - out_b["force"][0, i, d].item()) < 1e-5, (
                    f"atom {i} dim {d}: FD {fd.item():.6e} vs force "
                    f"{out_b['force'][0, i, d].item():.6e}"
                )

    def test_periodic_conservative_finite_difference(self) -> None:
        """Periodic branch forces are the exact energy gradient everywhere on
        the unwrapped branch -- including across the former wrap plane.
        """
        m = self._model(periodic=True)
        L = 8.0
        box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * L)[None]
        # One atom close to the boundary: the smooth branch must hold even
        # where the wrapped sawtooth used to jump.
        coord = torch.tensor(
            [[[2.0, 4.0, 0.03 * L], [2.5, 4.0, 0.55 * L]]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        atype = torch.tensor([[0, 1]], dtype=torch.int64, device=env.DEVICE)
        cs = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device=env.DEVICE)
        fp = self._field(0.1)
        out = m(coord, atype, charge_spin=cs, fparam=fp, box=box)
        eps = 1e-6
        for i in range(2):
            for d in range(3):
                dp = torch.zeros_like(coord)
                dp[0, i, d] = eps
                e_plus = m(coord + dp, atype, charge_spin=cs, fparam=fp, box=box)[
                    "energy"
                ]
                e_minus = m(coord - dp, atype, charge_spin=cs, fparam=fp, box=box)[
                    "energy"
                ]
                fd = -(e_plus - e_minus).flatten()[0] / (2 * eps)
                assert abs(fd.item() - out["force"][0, i, d].item()) < 1e-5, (
                    f"atom {i} dim {d}: FD {fd.item():.6e} vs force "
                    f"{out['force'][0, i, d].item():.6e}"
                )

    def test_serialization_round_trip(self) -> None:
        m = self._model()
        data = m.atomic_model.long_range.serialize()
        assert data["efield_potential"] == "branch"
        restored = LatentMultipoleLongRangeDP.deserialize(data)
        assert restored.efield_potential == "branch"

    def test_sawtooth_alias_reads_as_branch(self) -> None:
        """The deprecated 'sawtooth' mode name is accepted and normalized."""
        m = self._model()
        data = m.atomic_model.long_range.serialize()
        data["efield_potential"] = "sawtooth"
        restored = LatentMultipoleLongRangeDP.deserialize(data)
        assert restored.efield_potential == "branch"
        # The alias also flows through the constructor (direct config path).
        m_alias = self._model(efield_potential="sawtooth")
        assert m_alias.atomic_model.long_range.efield_potential == "branch"

    def test_legacy_record_defaults_to_force_only(self) -> None:
        """Records without the key predate the charge-potential coupling."""
        m = self._model(efield_potential="force_only")
        data = m.atomic_model.long_range.serialize()
        data.pop("efield_potential", None)
        restored = LatentMultipoleLongRangeDP.deserialize(data)
        assert restored.efield_potential == "force_only"

    def test_compiled_model_matches_eager(self, tmp_path) -> None:
        """The branch path is compile-safe: compiled energy/force/gradients
        match eager to machine precision (the trace sample must carry the
        field's fparam even though dim_fparam is 0).
        """
        from deepmd.pt_expt.train.training import _CompiledModel

        m = self._model()
        compiled = _CompiledModel(m, structure_key=(0,))
        compiled.train()
        coord, atype, cs = _batch()
        fp = self._field(0.1, nf=2)
        zero = torch.zeros((2, 3), dtype=torch.float64, device=env.DEVICE)

        out_c = compiled(coord, atype, None, charge_spin=cs, fparam=fp)
        m.train()
        out_e = m(coord, atype, charge_spin=cs, fparam=fp)
        torch.testing.assert_close(
            out_c["energy"], out_e["energy"], rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            out_c["force"], out_e["force"], rtol=1e-10, atol=1e-10
        )

        # A zero-field batch through the same compiled graph matches too.
        out_c0 = compiled(coord, atype, None, charge_spin=cs, fparam=zero)
        out_e0 = m(coord, atype, charge_spin=cs, fparam=zero)
        torch.testing.assert_close(
            out_c0["energy"], out_e0["energy"], rtol=1e-10, atol=1e-10
        )

        # Training gradients flow through the compiled graph.
        (out_c["energy"].square().sum() + out_c["force"].square().sum()).backward()
        assert all(p.grad is not None for p in compiled.parameters() if p.requires_grad)

    def test_branch_has_no_manual_side_output(self) -> None:
        """Branch force is fully autograd (coordinate leaf); only the legacy
        ``force_only`` coupling emits the q*E side output for the wrapper.
        The deprecated 'sawtooth' alias behaves like 'branch'.
        """
        captured = {}
        for potential, expected in (
            ("branch", False),
            ("sawtooth", False),
            ("force_only", True),
        ):
            m = self._model(efield_potential=potential)
            plugin = m.atomic_model.long_range
            orig = plugin.call_graph

            def hook(*a, **k):
                ret = orig(*a, **k)
                captured["has_force"] = "efield_force" in ret
                return ret

            plugin.call_graph = hook
            try:
                coord, atype, cs = _batch()
                m(coord, atype, charge_spin=cs, fparam=self._field(0.1, nf=2))
            finally:
                plugin.call_graph = orig
            assert captured["has_force"] is expected, potential


class TestEfieldBec:
    """The BEC (Born-effective-charge) field coupling: branch-free forces
    (rigid q_i E + local charge response + dipole response) that are NOT the
    gradient of the reported energy -- the deployment trade-off for periodic
    MD without unwrapped-coordinate tracking.
    """

    @staticmethod
    def _model(periodic: bool = False, steps: int = 2):
        cfg = _model_config(efield=True, steps=steps)
        cfg["long_range"]["efield_potential"] = "bec"
        if periodic:
            cfg["long_range"]["periodic"] = True
            cfg["long_range"]["k_grid_mode"] = "kgrid"
        return get_model(cfg).to(env.DEVICE).eval()

    @staticmethod
    def _field(value, nf=1):
        return torch.tensor(
            [[0.0, 0.0, float(value)]] * nf, dtype=torch.float64, device=env.DEVICE
        )

    def test_bec_force_matches_effective_energy_fd_on_dimer(self) -> None:
        """Decisive machinery test: on a neutral dimer the BEC force equals
        finite differences of the effective energy
        ``U_eff = U_model - E * q_1 * (r_12 . e_E)``.

        (For a dimer the local pair form IS the exact molecular dipole), so
        signs, factors, scattering and the nested edge gradient are all
        locked to FD precision.
        """
        m = self._model()
        E = 0.1
        fp = self._field(E)
        coord = torch.tensor(
            [[[1.0, 1.0, 1.0], [1.8, 1.2, 1.1]]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        atype = torch.tensor([[0, 1]], dtype=torch.int64, device=env.DEVICE)
        cs = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device=env.DEVICE)

        def u_eff(cc):
            out = m(cc, atype, charge_spin=cs, fparam=fp)
            q1 = out["latent_charge"][0, 0, 0]
            r12_z = (cc[0, 0] - cc[0, 1])[2]
            # U_model already carries -E . sum(mu); add the dimer
            # charge-dipole term -E * q1 * r12 (M = q1 r12 for a neutral
            # dimer), which the BEC force approximates exactly here.
            return out["energy"].flatten()[0] - E * q1 * r12_z

        out = m(coord, atype, charge_spin=cs, fparam=fp)
        eps = 1e-6
        for i in range(2):
            for d in range(3):
                dp = torch.zeros_like(coord)
                dp[0, i, d] = eps
                fd = -(u_eff(coord + dp) - u_eff(coord - dp)).item() / (2 * eps)
                assert abs(fd - out["force"][0, i, d].item()) < 1e-5, (
                    f"atom {i} dim {d}: FD {fd:.6e} vs force "
                    f"{out['force'][0, i, d].item():.6e}"
                )

    def test_bec_energy_excludes_charge_field(self) -> None:
        """BEC must not receive the coordinate potential (no q . phi in the
        energy), and the rigid q_i E force still rides the side output.
        """
        m = self._model()
        plugin = m.atomic_model.long_range
        orig = plugin.call_graph
        captured = {}

        def hook(*a, **k):
            captured["ext_pot"] = k.get("efield_ext_pot", "ABSENT")
            ret = orig(*a, **k)
            captured["has_force"] = "efield_force" in ret
            return ret

        plugin.call_graph = hook
        try:
            coord, atype, cs = _batch()
            m(coord, atype, charge_spin=cs, fparam=self._field(0.1, nf=2))
        finally:
            plugin.call_graph = orig
        assert captured["ext_pot"] in (None, "ABSENT")
        assert captured["has_force"] is True

    def test_bec_force_branch_point_invariant(self) -> None:
        """Branch-freedom made concrete: two configurations related by
        wrapping one atom by a lattice vector have the SAME MIC geometry and
        hence exactly the same BEC forces (the branch energy differs by
        q E L, the BEC force does not care).
        """
        m = self._model(periodic=True)
        L = 8.0
        box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * L)[None]
        coord_a = torch.tensor(
            [[[2.0, 4.0, 0.05 * L], [2.0, 4.0, 0.50 * L]]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        atype = torch.tensor([[0, 1]], dtype=torch.int64, device=env.DEVICE)
        cs = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device=env.DEVICE)
        fp = self._field(0.1)
        coord_b = coord_a.clone()
        coord_b[0, 0, 2] += L  # same MIC image, other side of the boundary
        fa = m(coord_a, atype, charge_spin=cs, fparam=fp, box=box)["force"]
        fb = m(coord_b, atype, charge_spin=cs, fparam=fp, box=box)["force"]
        torch.testing.assert_close(fa, fb, rtol=1e-10, atol=1e-10)

    def test_bec_training_gradients_flow(self) -> None:
        """The force loss is differentiable through the zero quasi-energy
        (second-order path): every trainable parameter receives a grad.
        """
        m = self._model(steps=1)
        m.train()
        coord, atype, cs = _batch()
        out = m(coord, atype, charge_spin=cs, fparam=self._field(0.1, nf=2))
        (out["energy"].square().sum() + out["force"].square().sum()).backward()
        missing = [
            name
            for name, par in m.named_parameters()
            if par.requires_grad and par.grad is None
        ]
        assert not missing, missing

    def test_bec_serialization_round_trip(self) -> None:
        m = self._model()
        data = m.atomic_model.long_range.serialize()
        assert data["efield_potential"] == "bec"
        restored = LatentMultipoleLongRangeDP.deserialize(data)
        assert restored.efield_potential == "bec"
