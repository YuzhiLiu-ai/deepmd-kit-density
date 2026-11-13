# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the delta-learning (SAD reference) path of density_basis."""

import os
import tempfile
import unittest

import numpy as np
import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.density_basis_model import (
    DensityBasisModel,
)
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pt.model.task.density_basis import (
    DensityBasisHead,
)
from deepmd.pt.model.task.density_reference import (
    RadialReferenceDensity,
)
from deepmd.pt.utils import (
    env,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

# state_dict keys of the scalar head without the reference module
LEGACY_KEYS: set = {
    "coef_proj.bias",
    "coef_proj.matrix",
    "radial_basis.adam_freqs",
    "radial_mlp.net.0.matrix",
    "radial_mlp.net.2.matrix",
    "type_embedding.embedding.embedding_net.layers.0.matrix",
}

TYPE_MAP = ["O", "H", "X"]
RCUT = 2.0
N_REF = 201


def _envelope(r: float, rcut: float = RCUT, p: int = 7) -> float:
    """C^3 cutoff envelope E_p(x) = u^4 * sum_k C(k+3,3) x^k."""
    from math import comb

    x = r / rcut
    u = 1.0 - x
    return u**4 * sum(comb(k + 3, 3) * x**k for k in range(p))


def _make_reference_npz(directory: str) -> str:
    """A tiny analytic reference table: rho_t(r) = (t + 1) * exp(-r).

    Covers [0, 6] Angstrom so it works both for the head-level tests
    (rcut = 2) and the model-level tests (rcut = 3).
    """
    r_grid = np.linspace(0.0, 6.0, 601)
    ref_table = np.stack(
        [(t + 1) * np.exp(-r_grid) for t in range(len(TYPE_MAP))]
    )  # X row is exp too; it is never a neighbor
    ref_table[-1] = 0.0
    path = os.path.join(directory, "reference_radial.npz")
    np.savez_compressed(
        path, ref_table=ref_table, r_grid=r_grid, type_map=np.array(TYPE_MAP)
    )
    return path


def _head_params(**kwargs) -> dict:
    params = {
        "ntypes": 3,
        "dim_descrpt": 8,
        "rcut": RCUT,
        "channels": 3,
        "n_radial": 4,
        "radial_neurons": [5],
        "seed": 1,
        "precision": "float64",
        "type_map": TYPE_MAP,
    }
    params.update(kwargs)
    return params


def _inputs(nf: int = 2, nloc: int = 4, ngrid: int = 3, nsel: int = 5):
    generator = torch.Generator().manual_seed(7)
    descriptor = torch.randn(
        nf, nloc, 8, generator=generator, dtype=torch.float64, device="cpu"
    ).to(device)
    latent = torch.randn(
        nf * nloc, 9, 1, 8, generator=generator, dtype=torch.float64, device="cpu"
    ).to(device)
    pair_atype = torch.randint(0, 2, (nf, ngrid, nsel), device=device)
    pair_vec = torch.randn(
        nf, ngrid, nsel, 3, generator=generator, dtype=torch.float64, device="cpu"
    ).to(device)
    pair_dist = pair_vec.norm(dim=-1).clamp(max=RCUT - 0.1)
    pair_mask = torch.ones(nf, ngrid, nsel, dtype=torch.bool, device=device)
    coef_gather_index = torch.randint(0, nloc, (nf, ngrid, nsel), device=device)
    return (
        descriptor,
        pair_atype,
        pair_dist,
        pair_mask,
        coef_gather_index,
        latent,
        pair_vec,
    )


class TestRadialReferenceDensity(unittest.TestCase):
    def test_analytic_interpolation(self) -> None:
        # rho_t(r) = (t + 1) * exp(-r), evaluated by hand
        r_grid = np.linspace(0.0, RCUT, N_REF)
        ref_table = np.stack(
            [(t + 1) * np.exp(-r_grid) for t in range(3)] + [np.zeros(N_REF)]
        )
        ref = RadialReferenceDensity(ref_table, r_grid, RCUT, dtype=torch.float64).to(
            device
        )
        pair_dist = torch.tensor([[[0.3], [1.1]]], dtype=torch.float64, device=device)
        pair_atype = torch.tensor([[[0], [1]]], dtype=torch.long, device=device)
        pair_keep = torch.ones(1, 2, 1, dtype=torch.bool, device=device)
        out = ref(pair_dist, pair_atype, pair_keep)
        expect = torch.tensor(
            [
                [
                    [1 * np.exp(-0.3) * _envelope(0.3)],
                    [2 * np.exp(-1.1) * _envelope(1.1)],
                ]
            ],
            dtype=torch.float64,
            device=device,
        )
        torch.testing.assert_close(out, expect, rtol=1e-3, atol=1e-6)

    def test_padding_row_is_zero(self) -> None:
        r_grid = np.linspace(0.0, RCUT, N_REF)
        ref_table = np.stack([np.exp(-r_grid) for _ in range(3)] + [np.zeros(N_REF)])
        ref = RadialReferenceDensity(ref_table, r_grid, RCUT, dtype=torch.float64).to(
            device
        )
        pair_dist = torch.tensor([[[0.5]]], dtype=torch.float64, device=device)
        pair_atype = torch.tensor([[[3]]], dtype=torch.long, device=device)  # padding
        pair_keep = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
        out = ref(pair_dist, pair_atype, pair_keep)
        torch.testing.assert_close(out, torch.zeros_like(out))


class TestDensityBasisHeadDelta(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ref_path = _make_reference_npz(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _head(self, **kwargs) -> DensityBasisHead:
        params = _head_params(use_delta=True, reference_density=self.ref_path)
        params.update(kwargs)
        return DensityBasisHead(**params).to(device)

    def test_shape_and_finite(self) -> None:
        for kwargs in (
            {},
            {
                "use_equivariant": True,
                "coef_lmax": 2,
                "latent_lmax": 2,
                "latent_channels": 8,
            },
            {
                "use_equivariant": True,
                "coef_lmax": 6,
                "latent_lmax": 2,
                "latent_channels": 8,
            },
        ):
            with self.subTest(**kwargs):
                head = self._head(**kwargs)
                *args, latent, pair_vec = _inputs()
                out = head(*args, latent=latent, pair_vec=pair_vec)
                self.assertEqual(out.shape, (2, 3, 1))
                self.assertTrue(torch.isfinite(out).all())

    def test_zero_delta_gives_reference(self) -> None:
        head = self._head()
        with torch.no_grad():
            for param in head.parameters():
                param.zero_()
        args = _inputs()
        (
            descriptor,
            pair_atype,
            pair_dist,
            pair_mask,
            coef_gather_index,
            latent,
            pair_vec,
        ) = args
        out = head(descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        # with all parameters zero the output must be the reference only
        keep = torch.ones_like(pair_mask)
        ref = head.reference_density(pair_dist, pair_atype, keep)
        torch.testing.assert_close(out, ref.sum(dim=-1).unsqueeze(-1))

    def test_reference_values_match_table(self) -> None:
        # reference of a single O-type pair at distance r must interpolate
        # rho(r) = 1 * exp(-r) (type 0 row) times the cutoff envelope
        head = self._head()
        pair_dist = torch.tensor([[[0.4]]], dtype=torch.float64, device=device)
        pair_atype = torch.tensor([[[0]]], dtype=torch.long, device=device)
        pair_keep = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
        out = head.reference_density(pair_dist, pair_atype, pair_keep)
        self.assertAlmostEqual(
            out.item(), float(np.exp(-0.4)) * _envelope(0.4), places=6
        )

    def test_frozen_no_grad(self) -> None:
        head = self._head()
        param_names = {name for name, _ in head.named_parameters()}
        self.assertFalse(any("reference_density" in name for name in param_names))
        *args, latent, pair_vec = _inputs()
        out = head(*args, latent=latent, pair_vec=pair_vec)
        out.square().sum().backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        # buffers unchanged by backward
        self.assertTrue(torch.isfinite(head.reference_density.ref_table).all())

    def test_serialize_roundtrip_without_file(self) -> None:
        head = self._head()
        *args, latent, pair_vec = _inputs()
        out = head(*args, latent=latent, pair_vec=pair_vec)
        blob = head.serialize()
        # the file may vanish after training; deserialization must not need it
        os.remove(self.ref_path)
        head2 = BaseFitting.deserialize(blob)
        self.assertIsInstance(head2, DensityBasisHead)
        self.assertTrue(head2.use_delta)
        torch.testing.assert_close(head2(*args, latent=latent, pair_vec=pair_vec), out)

    def test_checkpoint_construction_fallback(self) -> None:
        good = self._head()
        *args, latent, pair_vec = _inputs()
        expect = good(*args, latent=latent, pair_vec=pair_vec)
        # construct without the file: zero placeholder table + warning
        with self.assertLogs("deepmd.pt.model.task.density_basis", level="WARNING"):
            fallback = DensityBasisHead(
                **_head_params(use_delta=True, reference_density="/nonexistent.npz")
            ).to(device)
        fallback.load_state_dict(good.state_dict())
        torch.testing.assert_close(
            fallback(*args, latent=latent, pair_vec=pair_vec), expect
        )

    def test_off_mode_state_dict_unchanged(self) -> None:
        head = DensityBasisHead(**_head_params()).to(device)
        self.assertEqual(set(head.state_dict().keys()), LEGACY_KEYS)

    def test_legacy_serialization_loads(self) -> None:
        legacy = {
            "@class": "Fitting",
            "@version": 1,
            "type": "density_basis",
            "ntypes": 3,
            "dim_descrpt": 8,
            "rcut": 2.0,
            "channels": 3,
            "n_radial": 4,
            "basis_type": "bessel",
            "radial_neurons": [5],
            "precision": "float64",
            "seed": 1,
            "trainable": True,
            "exclude_types": [],
            "type_map": TYPE_MAP,
        }
        ref = DensityBasisHead(**_head_params()).to(device)
        legacy["@variables"] = {
            kk: vv.detach().cpu().numpy() for kk, vv in ref.state_dict().items()
        }
        head = BaseFitting.deserialize(legacy)
        self.assertIsInstance(head, DensityBasisHead)
        self.assertFalse(head.use_delta)
        self.assertEqual(set(head.state_dict().keys()), LEGACY_KEYS)

    def test_change_type_map(self) -> None:
        head = self._head()
        old_table = head.reference_density.ref_table.clone()
        head.change_type_map(["H", "O", "X"])
        new_table = head.reference_density.ref_table
        # new row 0 (H) = old row 1, new row 1 (O) = old row 0
        torch.testing.assert_close(new_table[0], old_table[1])
        torch.testing.assert_close(new_table[1], old_table[0])
        torch.testing.assert_close(new_table[-1], torch.zeros_like(new_table[-1]))

    def test_jit_scalar_path(self) -> None:
        head = self._head()
        *args, latent, pair_vec = _inputs()
        out = head(*args)
        scripted = torch.jit.script(head)
        out_s = scripted(*args)
        torch.testing.assert_close(out_s, out)


class TestDensityBasisDeltaModel(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ref_path = _make_reference_npz(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _model_params(self, **fitting_kwargs) -> dict:
        fitting_net = {
            "type": "density_basis",
            "channels": 3,
            "n_radial": 5,
            "radial_neurons": [6],
            "seed": 1,
            "precision": "float64",
            "use_delta": True,
            "reference_density": self.ref_path,
        }
        fitting_net.update(fitting_kwargs)
        return {
            "type_map": TYPE_MAP,
            "descriptor": {
                "type": "dpa4",
                "sel": [8, 8, 0],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": True,
                "random_gamma": False,
                "lmax": 2,
                "mmax": 1,
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "mlp_bias": False,
                "layer_scale": False,
                "use_amp": False,
                "activation_function": "silu",
                "precision": "float64",
                "seed": 7,
            },
            "fitting_net": fitting_net,
        }

    def _frame(self, nf: int = 1, nloc: int = 5, ngrid: int = 7, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        coord = (
            torch.rand(
                nf, nloc * 3, generator=generator, dtype=torch.float64, device="cpu"
            )
            * 6.0
            + 1.0
        ).to(device)
        atype = torch.tensor([[0, 0, 1, 1, 1]] * nf, dtype=torch.long, device=device)
        box = (
            torch.eye(3, dtype=torch.float64, device=device).reshape(1, 9).repeat(nf, 1)
            * 8.0
        )
        grid = (
            torch.rand(
                nf, ngrid, 3, generator=generator, dtype=torch.float64, device="cpu"
            )
            * 6.0
            + 1.0
        ).to(device)
        return coord, atype, box, grid

    def test_rotation_translation_invariance(self) -> None:
        # the reference is purely radial and trivially invariant; this guards
        # against wiring mistakes (e.g. using the wrong pair layout)
        for coef_lmax in (2, 6):
            with self.subTest(coef_lmax=coef_lmax):
                model = get_model(
                    self._model_params(use_equivariant=True, coef_lmax=coef_lmax)
                )
                self.assertIsInstance(model, DensityBasisModel)
                coord, atype, box, grid = self._frame()
                out = model(coord, atype, box=box, grid=grid)["density"]
                rotation = torch.tensor(
                    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float64,
                    device=device,
                )
                translation = torch.tensor(
                    [2.0, 3.0, 1.0], dtype=torch.float64, device=device
                )
                coord_rt = coord.view(1, -1, 3) @ rotation.T + translation
                grid_rt = grid @ rotation.T + translation
                out_rt = model(coord_rt.reshape(1, -1), atype, box=box, grid=grid_rt)[
                    "density"
                ]
                torch.testing.assert_close(out, out_rt, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
