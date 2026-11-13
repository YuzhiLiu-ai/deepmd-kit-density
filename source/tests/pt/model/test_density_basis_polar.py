# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the MACE-POLAR style extensions of density_basis.

Covers the spin channel structure (n_spin), the polarisable field update
(Coulomb features + update head), and the spin-resolved Fukui charge
equilibration, all in coefficient space.
"""

import math
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
from deepmd.pt.utils import (
    env,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

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


def _make_reference_npz(directory: str) -> str:
    """Tiny analytic reference: rho_t(r) = (t + 1) * exp(-r), X row zero."""
    r_grid = np.linspace(0.0, 6.0, 601)
    ref_table = np.stack([(t + 1) * np.exp(-r_grid) for t in range(len(TYPE_MAP))])
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
    # local atoms and the direct-space Coulomb pairs (no images here)
    atom_atype = torch.randint(0, 2, (nf, nloc), device=device)
    atom_coord = torch.randn(
        nf, nloc, 3, generator=generator, dtype=torch.float64, device="cpu"
    ).to(device)
    coul_pair_dist = torch.cdist(atom_coord, atom_coord)
    coul_pair_gather = (
        torch.arange(nloc, device=device)
        .view(1, 1, nloc)
        .expand(nf, nloc, nloc)
        .contiguous()
    )
    coul_pair_mask = coul_pair_dist > 1e-8
    return (
        descriptor,
        pair_atype,
        pair_dist,
        pair_mask,
        coef_gather_index,
        latent,
        pair_vec,
        atom_atype,
        atom_coord,
        coul_pair_dist,
        coul_pair_gather,
        coul_pair_mask,
    )


class TestSpinStructure(unittest.TestCase):
    def test_off_state_dict_unchanged(self) -> None:
        head = DensityBasisHead(**_head_params()).to(device)
        self.assertEqual(set(head.state_dict().keys()), LEGACY_KEYS)

    def test_duplicated_spin_doubles_output(self) -> None:
        """n_spin = 2 with both spin channels identical must give 2x the
        n_spin = 1 output (spin axis layout + shared radial double sum).
        """
        head1 = DensityBasisHead(**_head_params()).to(device)
        head2 = DensityBasisHead(**_head_params(n_spin=2)).to(device)
        with torch.no_grad():
            # duplicate every spin-carrying weight across the spin axis;
            # flat channel index is c * n_spin + s
            for name1, p1 in head1.named_parameters():
                p2 = dict(head2.named_parameters())[name1]
                if p1.shape == p2.shape:
                    p2.copy_(p1)
                else:
                    # output-channel duplication along the last axis
                    c = p1.shape[-1]
                    self.assertEqual(p2.shape[-1], 2 * c)
                    dup = torch.stack([p1, p1], dim=-1).reshape(*p1.shape[:-1], 2 * c)
                    p2.copy_(dup)
        args = _inputs()
        descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index = args[:5]
        out1 = head1(descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        out2 = head2(descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        torch.testing.assert_close(out2, 2.0 * out1)


class TestCoulombFeatures(unittest.TestCase):
    def _head(self, **kwargs) -> DensityBasisHead:
        params = _head_params(use_field_update=True, **kwargs)
        return DensityBasisHead(**params).to(device)

    def test_analytic_kernel(self) -> None:
        head = self._head(coulomb_sigma=1.0)
        # two sources, q = [1, 2], separated by 3 Angstrom
        charges = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64, device=device)
        dist = torch.tensor(
            [[[0.0, 3.0], [3.0, 0.0]]], dtype=torch.float64, device=device
        )
        gather = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long, device=device)
        mask = dist > 1e-8
        v = head._coulomb_features(charges, dist, gather, mask)
        # kernel = erf(r / sqrt(2) sigma) / r * C3-envelope(r / coul_rcut)
        from math import comb

        x = 3.0 / head.coulomb_rcut
        env_val = (1.0 - x) ** 4 * sum(comb(k + 3, 3) * x**k for k in range(7))
        kernel = math.erf(3.0 / math.sqrt(2.0)) / 3.0 * env_val
        expect = torch.tensor(
            [[[2.0 * kernel], [1.0 * kernel]]], dtype=torch.float64, device=device
        )
        torch.testing.assert_close(v, expect)

    def test_masked_pairs_contribute_nothing(self) -> None:
        head = self._head()
        charges = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64, device=device)
        dist = torch.tensor(
            [[[0.0, 3.0], [3.0, 0.0]]], dtype=torch.float64, device=device
        )
        gather = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long, device=device)
        mask = torch.zeros(1, 2, 2, dtype=torch.bool, device=device)
        v = head._coulomb_features(charges, dist, gather, mask)
        torch.testing.assert_close(v, torch.zeros_like(v))

    def test_charge_weights_match_dense_quadrature(self) -> None:
        # the 64-point charge weights must agree with an independent fine
        # quadrature of the same radial branch
        head = self._head()
        w64 = head._radial_charge_weights()
        with torch.no_grad():
            r = torch.linspace(
                1e-6, RCUT, 4096, dtype=torch.float64, device=device
            ).view(-1, 1)
            nt = head.ntypes + 1
            rbf = head.radial_basis(r.repeat(nt, 1))
            tebd = head.type_embedding(head.quad_atype).repeat_interleave(
                r.shape[0], dim=0
            )
            radial = head.radial_mlp(torch.cat([rbf, tebd], dim=-1)) * head.envelope(
                r.repeat(nt, 1)
            )
            radial = radial.view(nt, r.shape[0], 1 + head.coef_lmax, head.channels)
            dr = r[1, 0] - r[0, 0]
            r2 = 4.0 * math.pi * r.view(1, -1) ** 2
            w_fine = (radial[..., 0, :] * r2.unsqueeze(-1)).sum(dim=1) * dr
        torch.testing.assert_close(w64, w_fine, rtol=1e-3, atol=1e-6)

    def test_monopole_charge_vs_density_integral(self) -> None:
        # sum_i q_i must equal the numerical integral of the network density
        # around a single atom (charge-content consistency)
        head = DensityBasisHead(**_head_params(channels=2, n_radial=4)).to(device)
        nf, nloc = 1, 1
        atom_atype = torch.zeros(nf, nloc, dtype=torch.long, device=device)
        descriptor = torch.randn(
            nf,
            nloc,
            8,
            generator=torch.Generator().manual_seed(3),
            dtype=torch.float64,
            device="cpu",
        ).to(device)
        coef = head.coef_proj(descriptor)
        # the public monopole uses the unit-normalized charge convention;
        # rebuild the raw charge content (integral of the density) here
        w = head._radial_charge_weights()[atom_atype]  # (nf, nloc, channels)
        q_raw = (coef.view(nf, nloc, head.channels, head.n_spin) * w.unsqueeze(-1)).sum(
            -2
        )
        # Monte-Carlo integrate the head density in a ball around the atom
        n_pts = 200000
        gen = torch.Generator().manual_seed(11)
        pts = torch.randn(
            n_pts, 3, generator=gen, dtype=torch.float64, device="cpu"
        ).to(device)
        pts = (
            pts
            / pts.norm(dim=-1, keepdim=True)
            * (
                torch.rand(
                    n_pts, 1, generator=gen, dtype=torch.float64, device="cpu"
                ).to(device)
                ** (1.0 / 3.0)
                * (RCUT - 1e-3)
            )
        )
        pair_dist = pts.norm(dim=-1).view(1, n_pts, 1)
        pair_atype = torch.zeros(1, n_pts, 1, dtype=torch.long, device=device)
        pair_mask = torch.ones(1, n_pts, 1, dtype=torch.bool, device=device)
        coef_gather_index = torch.zeros(1, n_pts, 1, dtype=torch.long, device=device)
        rho = head(descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        volume = 4.0 / 3.0 * math.pi * RCUT**3
        integral = rho.mean() * volume
        torch.testing.assert_close(
            integral, q_raw.sum(), rtol=2e-2, atol=2e-2 * abs(float(q_raw.sum())) + 1e-6
        )


class TestFieldUpdateAndFukui(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ref_path = _make_reference_npz(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _head(self, **kwargs) -> DensityBasisHead:
        params = _head_params(use_delta=True, reference_density=self.ref_path, **kwargs)
        return DensityBasisHead(**params).to(device)

    def _call(self, head, args=None, **kw):
        if args is None:
            args = _inputs()
        descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index = args[:5]
        atom_atype, _, dist, gather, mask = args[7:]
        return head(
            descriptor,
            pair_atype,
            pair_dist,
            pair_mask,
            coef_gather_index,
            atom_atype=atom_atype,
            coul_pair_dist=dist,
            coul_pair_gather=gather,
            coul_pair_mask=mask,
            **kw,
        )

    def test_field_update_changes_output(self) -> None:
        head_off = self._head()
        head_on = self._head(use_field_update=True)
        # the update head is zero-initialized (identity at init): perturb it
        # away from zero to make the update path active
        with torch.no_grad():
            for p in head_on.update_mlp.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        with torch.no_grad():
            for (n1, p1), (_, p2) in zip(
                head_off.named_parameters(), head_on.named_parameters(), strict=False
            ):
                if n1 in dict(head_on.named_parameters()) and p1.shape == p2.shape:
                    p2.copy_(p1)
        out_off = self._call(head_off)
        out_on = self._call(head_on)
        self.assertTrue(torch.isfinite(out_on).all())
        self.assertFalse(torch.allclose(out_off, out_on))

    def test_iters_count(self) -> None:
        head1 = self._head(use_field_update=True, field_update_iters=1)
        head2 = self._head(use_field_update=True, field_update_iters=2)
        # zero-initialized update heads are the identity at init, so the
        # iteration count only matters once the update weights are nonzero
        with torch.no_grad():
            for head in (head1, head2):
                for p in head.update_mlp.parameters():
                    p.copy_(torch.randn_like(p) * 0.1)
        out1 = self._call(head1)
        out2 = self._call(head2)
        self.assertTrue(torch.isfinite(out1).all())
        self.assertTrue(torch.isfinite(out2).all())
        self.assertFalse(torch.allclose(out1, out2))

    def test_fukui_constraint(self) -> None:
        head = self._head(use_charge_equilibration=True)
        args = _inputs()
        descriptor, atom_atype = args[0], args[7]
        coef = head.coef_proj(descriptor)
        coef_eq = head._fukui_equilibrate(coef, descriptor, atom_atype)
        q = head._monopole_charges(coef_eq, atom_atype)
        torch.testing.assert_close(
            q.sum(dim=1), torch.zeros_like(q.sum(dim=1)), rtol=0, atol=1e-10
        )

    def test_fukui_requires_delta(self) -> None:
        with self.assertRaises(NotImplementedError):
            DensityBasisHead(**_head_params(use_charge_equilibration=True))

    def test_backward_reaches_update_modules(self) -> None:
        head = self._head(use_field_update=True, use_charge_equilibration=True)
        out = self._call(head)
        out.square().sum().backward()
        grads = {
            name: param.grad
            for name, param in head.named_parameters()
            if ("update_mlp" in name or "fukui_mlp" in name) and param.grad is not None
        }
        self.assertTrue(grads)
        self.assertTrue(any(g.abs().sum() > 0 for g in grads.values()))

    def test_serialization_roundtrip(self) -> None:
        head = self._head(
            use_field_update=True,
            field_update_iters=2,
            use_charge_equilibration=True,
            n_spin=2,
        )
        args = _inputs()
        out = self._call(head, args)
        head2 = BaseFitting.deserialize(head.serialize())
        self.assertIsInstance(head2, DensityBasisHead)
        self.assertEqual(head2.n_spin, 2)
        self.assertEqual(head2.field_update_iters, 2)
        torch.testing.assert_close(self._call(head2, args), out)

    def test_jit_scalar_with_update(self) -> None:
        head = self._head(use_field_update=True)
        args = _inputs()
        descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index = args[:5]
        atom_atype, _, dist, gather, mask = args[7:]
        out = head(
            descriptor,
            pair_atype,
            pair_dist,
            pair_mask,
            coef_gather_index,
            atom_atype=atom_atype,
            coul_pair_dist=dist,
            coul_pair_gather=gather,
            coul_pair_mask=mask,
        )
        scripted = torch.jit.script(head)
        out_s = scripted(
            descriptor,
            pair_atype,
            pair_dist,
            pair_mask,
            coef_gather_index,
            atom_atype=atom_atype,
            coul_pair_dist=dist,
            coul_pair_gather=gather,
            coul_pair_mask=mask,
        )
        torch.testing.assert_close(out_s, out)


class TestPolarModel(unittest.TestCase):
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

    def test_equivariant_update_end_to_end(self) -> None:
        # l = 6 + n_spin = 2 + field update (x2) + Fukui + delta
        model = get_model(
            self._model_params(
                use_equivariant=True,
                coef_lmax=6,
                n_spin=2,
                use_field_update=True,
                field_update_iters=2,
                use_charge_equilibration=True,
            )
        )
        self.assertIsInstance(model, DensityBasisModel)
        coord, atype, box, grid = self._frame()
        out = model(coord, atype, box=box, grid=grid)
        self.assertEqual(out["density"].shape, (1, 7, 1))
        self.assertTrue(torch.isfinite(out["density"]).all())
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
            device=device,
        )
        translation = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64, device=device)
        coord_rt = coord.view(1, -1, 3) @ rotation.T + translation
        grid_rt = grid @ rotation.T + translation
        out_rt = model(coord_rt.reshape(1, -1), atype, box=box, grid=grid_rt)["density"]
        torch.testing.assert_close(out["density"], out_rt, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
