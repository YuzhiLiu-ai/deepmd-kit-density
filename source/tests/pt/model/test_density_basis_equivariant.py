# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the equivariant (l >= 1) density_basis expansion."""

import unittest
from typing import (
    ClassVar,
)

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


def _head_params(**kwargs) -> dict:
    params = {
        "ntypes": 3,
        "dim_descrpt": 8,
        "rcut": 2.0,
        "channels": 3,
        "n_radial": 4,
        "radial_neurons": [5],
        "seed": 1,
        "precision": "float64",
        "type_map": ["O", "H", "X"],
    }
    params.update(kwargs)
    return params


def _model_params(**fitting_kwargs) -> dict:
    fitting_net = {
        "type": "density_basis",
        "channels": 3,
        "n_radial": 5,
        "radial_neurons": [6],
        "seed": 1,
        "precision": "float64",
    }
    fitting_net.update(fitting_kwargs)
    return {
        "type_map": ["O", "H", "X"],
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


def _randomize(model: torch.nn.Module, seed: int = 1234) -> None:
    """Perturb the parameters away from any symmetry of the initialization."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                torch.randn(param.shape, generator=generator, device="cpu").to(param)
                * 0.1
                + param * 0.5
            )


def _frame(
    nf: int = 1,
    nloc: int = 5,
    ngrid: int = 7,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    coord = (
        torch.rand(nf, nloc * 3, generator=generator, dtype=torch.float64, device="cpu")
        * 6.0
        + 1.0
    ).to(device)
    atype = torch.tensor([[0, 0, 1, 1, 1]] * nf, dtype=torch.long, device=device)
    box = (
        torch.eye(3, dtype=torch.float64, device=device).reshape(1, 9).repeat(nf, 1)
        * 8.0
    )
    grid = (
        torch.rand(nf, ngrid, 3, generator=generator, dtype=torch.float64, device="cpu")
        * 6.0
        + 1.0
    ).to(device)
    return coord, atype, box, grid


class TestDensityBasisHeadEquivariant(unittest.TestCase):
    def _head(self, **kwargs) -> DensityBasisHead:
        params = _head_params(
            use_equivariant=True, coef_lmax=2, latent_lmax=2, latent_channels=8
        )
        params.update(kwargs)
        return DensityBasisHead(**params).to(device)

    def _inputs(self, nf: int = 2, nloc: int = 4, ngrid: int = 3, nsel: int = 5):
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
        pair_dist = pair_vec.norm(dim=-1).clamp(max=1.9)
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

    def test_forward_shape(self) -> None:
        head = self._head()
        *args, latent, pair_vec = self._inputs()
        out = head(*args, latent=latent, pair_vec=pair_vec)
        self.assertEqual(out.shape, (2, 3, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_requires_latent_and_pair_vec(self) -> None:
        head = self._head()
        *args, latent, pair_vec = self._inputs()
        with self.assertRaises((AssertionError, TypeError)):
            head(*args)
        with self.assertRaises(AssertionError):
            head(*args, latent=latent)
        with self.assertRaises(AssertionError):
            head(*args, pair_vec=pair_vec)

    def test_equivariant_branch_active(self) -> None:
        # the same head with the equivariant projection zeroed out must fall
        # back to the scalar expansion, so a nonzero projection changes the
        # output
        head = self._head()
        *args, latent, pair_vec = self._inputs()
        out = head(*args, latent=latent, pair_vec=pair_vec)
        with torch.no_grad():
            head.coef_proj_so3.weight.zero_()
        out_zero = head(*args, latent=latent, pair_vec=pair_vec)
        self.assertFalse(torch.allclose(out, out_zero))

    def test_zero_latent_reduces_to_scalar(self) -> None:
        head = self._head()
        *args, latent, pair_vec = self._inputs()
        out = head(*args, latent=torch.zeros_like(latent), pair_vec=pair_vec)
        head_scalar = DensityBasisHead(**_head_params()).to(device)
        # copy the shared scalar-path parameters
        with torch.no_grad():
            head_scalar.coef_proj.load_state_dict(head.coef_proj.state_dict())
            head_scalar.type_embedding.load_state_dict(head.type_embedding.state_dict())
            head_scalar.radial_basis.load_state_dict(head.radial_basis.state_dict())
            # the l = 0 block of the equivariant radial MLP plays the role of
            # the scalar radial MLP only when the weights coincide, which they
            # do not; so instead compare against the same head with the SO(3)
            # projection zeroed
            head.coef_proj_so3.weight.zero_()
        out_zero = head(*args, latent=latent, pair_vec=pair_vec)
        torch.testing.assert_close(out, out_zero)

    def test_serialization_roundtrip(self) -> None:
        head = self._head()
        *args, latent, pair_vec = self._inputs()
        out = head(*args, latent=latent, pair_vec=pair_vec)
        head2 = BaseFitting.deserialize(head.serialize())
        self.assertIsInstance(head2, DensityBasisHead)
        self.assertTrue(head2.use_equivariant)
        self.assertEqual(head2.coef_lmax, 2)
        torch.testing.assert_close(head2(*args, latent=latent, pair_vec=pair_vec), out)


class TestDensityBasisHeadBackwardCompat(unittest.TestCase):
    # state_dict keys of the scalar head before the equivariant upgrade
    # (for the radial_neurons=[5] configuration: two MLP layers)
    LEGACY_KEYS: ClassVar[set] = {
        "coef_proj.bias",
        "coef_proj.matrix",
        "radial_basis.adam_freqs",
        "radial_mlp.net.0.matrix",
        "radial_mlp.net.2.matrix",
        "type_embedding.embedding.embedding_net.layers.0.matrix",
    }

    def test_off_mode_state_dict_unchanged(self) -> None:
        head = DensityBasisHead(**_head_params()).to(device)
        self.assertEqual(set(head.state_dict().keys()), self.LEGACY_KEYS)

    def test_off_mode_forward_signature_unchanged(self) -> None:
        head = DensityBasisHead(**_head_params()).to(device)
        nf, nloc, ngrid, nsel = 2, 4, 3, 5
        descriptor = torch.randn(nf, nloc, 8, dtype=torch.float64, device=device)
        pair_atype = torch.randint(0, 2, (nf, ngrid, nsel), device=device)
        pair_dist = torch.rand(nf, ngrid, nsel, dtype=torch.float64, device=device)
        pair_mask = torch.ones(nf, ngrid, nsel, dtype=torch.bool, device=device)
        coef_gather_index = torch.randint(0, nloc, (nf, ngrid, nsel), device=device)
        out = head(descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        self.assertEqual(out.shape, (nf, ngrid, 1))

    def test_legacy_serialization_loads(self) -> None:
        # a serialization produced by the pre-upgrade head carries no
        # equivariant config keys and no equivariant variables
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
            "type_map": ["O", "H", "X"],
        }
        ref = DensityBasisHead(**_head_params()).to(device)
        legacy["@variables"] = {
            kk: vv.detach().cpu().numpy() for kk, vv in ref.state_dict().items()
        }
        head = BaseFitting.deserialize(legacy)
        self.assertIsInstance(head, DensityBasisHead)
        self.assertFalse(head.use_equivariant)
        self.assertEqual(set(head.state_dict().keys()), self.LEGACY_KEYS)


class TestDensityBasisEquivariantModel(unittest.TestCase):
    def _model(self, **fitting_kwargs) -> DensityBasisModel:
        kwargs = {"use_equivariant": True, "coef_lmax": 2}
        kwargs.update(fitting_kwargs)
        model = get_model(_model_params(**kwargs))
        self.assertIsInstance(model, DensityBasisModel)
        return model

    def test_shape(self) -> None:
        model = self._model()
        coord, atype, box, grid = _frame()
        out = model(coord, atype, box=box, grid=grid)
        self.assertEqual(out["density"].shape, (1, 7, 1))
        self.assertTrue(torch.isfinite(out["density"]).all())

    def test_rotation_translation_invariance(self) -> None:
        # the contraction of the equivariant coefficients with the spherical
        # harmonics of the pair direction must be invariant to rigid motions;
        # the 90-degree rotation maps the cubic box onto itself, keeping PBC
        # exact
        model = self._model()
        _randomize(model)
        coord, atype, box, grid = _frame()
        out = model(coord, atype, box=box, grid=grid)["density"]
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
            device=device,
        )
        translation = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64, device=device)
        coord_rt = coord.view(1, -1, 3) @ rotation.T + translation
        grid_rt = grid @ rotation.T + translation
        out_rt = model(coord_rt.reshape(1, -1), atype, box=box, grid=grid_rt)["density"]
        torch.testing.assert_close(out, out_rt, rtol=1e-8, atol=1e-8)

    def test_backward_reaches_equivariant_projection(self) -> None:
        model = self._model()
        coord, atype, box, grid = _frame()
        out = model(coord, atype, box=box, grid=grid)
        out["density"].square().sum().backward()
        so3_grads = [
            param.grad
            for name, param in model.named_parameters()
            if "coef_proj_so3" in name and param.grad is not None
        ]
        self.assertTrue(so3_grads)
        self.assertTrue(any(grad.abs().sum() > 0 for grad in so3_grads))

    def test_equivariant_changes_output(self) -> None:
        coord, atype, box, grid = _frame()
        model_on = get_model(_model_params(use_equivariant=True, coef_lmax=2))
        model_off = get_model(_model_params())
        _randomize(model_on)
        _randomize(model_off)
        out_on = model_on(coord, atype, box=box, grid=grid)["density"]
        out_off = model_off(coord, atype, box=box, grid=grid)["density"]
        self.assertFalse(torch.allclose(out_on, out_off))

    def test_serialization(self) -> None:
        model = self._model()
        coord, atype, box, grid = _frame()
        out = model(coord, atype, box=box, grid=grid)
        model2 = DensityBasisModel.deserialize(model.serialize())
        out2 = model2(coord, atype, box=box, grid=grid)
        torch.testing.assert_close(out2["density"], out["density"])

    def test_requires_equivariant_descriptor(self) -> None:
        # se_e2_a has no SO(3) latent to project
        params = _model_params(use_equivariant=True)
        params["descriptor"] = {
            "type": "se_e2_a",
            "sel": [8, 8, 0],
            "rcut_smth": 0.5,
            "rcut": 3.0,
            "neuron": [8, 16],
            "resnet_dt": False,
            "axis_neuron": 4,
            "seed": 1,
            "precision": "float64",
        }
        with self.assertRaisesRegex(ValueError, "forward_with_edges"):
            get_model(params)


class TestHigherLConstants(unittest.TestCase):
    """Deterministic SH transforms and CG couplings (see sezm_nn.higher_l)."""

    def test_sh_transform_matches_cartesian_basis(self) -> None:
        # the numerically solved l = 1, 2 transforms must reproduce the
        # packed direction coordinates of the analytic Cartesian basis up to
        # a positive per-degree scale
        from deepmd.pt.model.descriptor.sezm_nn.cartesian import (
            build_cartesian_basis,
            build_edge_cartesian_tensors,
        )
        from deepmd.pt.model.descriptor.sezm_nn.higher_l import (
            build_higher_l_constants,
        )
        from deepmd.pt.model.descriptor.sezm_nn.spherical import (
            SQRT_2,
            build_real_sh_norm,
            real_spherical_harmonics,
        )

        transforms, _ = build_higher_l_constants(3)
        norm = build_real_sh_norm(3, device=torch.device("cpu"))
        generator = torch.Generator().manual_seed(0)
        dirs = torch.randn(
            512, 3, generator=generator, dtype=torch.float64, device="cpu"
        )
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        y = real_spherical_harmonics(3, dirs, norm, SQRT_2)
        cart = build_cartesian_basis(2, dtype=torch.float64, device="cpu")
        a0, s0 = build_edge_cartesian_tensors(dirs)
        import math

        for ll, u_cart in (
            (1, torch.einsum("dij,eij->ed", cart[1:4], a0 / math.sqrt(2.0))),
            (2, torch.einsum("dij,eij->ed", cart[4:9], s0 / math.sqrt(2.0 / 3.0))),
        ):
            y_l = y[:, ll * ll : (ll + 1) * (ll + 1)]
            u_sh = y_l @ transforms[ll].transpose(0, 1)
            scale = (u_sh * u_cart).sum() / (u_cart * u_cart).sum()
            self.assertGreater(scale.item(), 0.0)
            residual = ((u_sh - scale * u_cart).norm() / (scale * u_cart).norm()).item()
            self.assertLess(residual, 1e-12)

    def test_cg_coupling_equivariance(self) -> None:
        # rotating the packed inputs of a coupling by the Wigner-D blocks
        # must rotate the output by the corresponding block
        from deepmd.pt.model.descriptor.sezm_nn.higher_l import (
            build_higher_l_constants,
        )
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            WignerDCalculator,
        )

        _, couplings = build_higher_l_constants(6)
        n_rot = 4
        generator = torch.Generator().manual_seed(4242)
        quats = torch.randn(
            n_rot, 4, generator=generator, dtype=torch.float64, device="cpu"
        )
        quats = quats / quats.norm(dim=-1, keepdim=True)
        calculator = WignerDCalculator(6, dtype=torch.float64).to("cpu")
        with torch.no_grad():
            d_full, _ = calculator(quats)
        feature_gen = torch.Generator().manual_seed(7)
        for l1, l2, l3 in ((1, 2, 3), (2, 2, 4), (3, 2, 5), (4, 2, 6)):
            coupling = couplings[(l1, l2, l3)]
            c1 = torch.randn(
                n_rot,
                2 * l1 + 1,
                6,
                generator=feature_gen,
                dtype=torch.float64,
                device="cpu",
            )
            c2 = torch.randn(
                n_rot,
                2 * l2 + 1,
                6,
                generator=feature_gen,
                dtype=torch.float64,
                device="cpu",
            )
            out = torch.einsum("mno,xmk,xnl->xokl", coupling, c1, c2)
            d1 = d_full[:, l1 * l1 : (l1 + 1) ** 2, l1 * l1 : (l1 + 1) ** 2]
            d2 = d_full[:, l2 * l2 : (l2 + 1) ** 2, l2 * l2 : (l2 + 1) ** 2]
            d3 = d_full[:, l3 * l3 : (l3 + 1) ** 2, l3 * l3 : (l3 + 1) ** 2]
            c1_rot = torch.einsum("xom,xmk->xok", d1, c1)
            c2_rot = torch.einsum("xon,xnl->xol", d2, c2)
            out_rot = torch.einsum("mno,xmk,xnl->xokl", coupling, c1_rot, c2_rot)
            expect = torch.einsum("xon,xnkl->xokl", d3, out)
            err = (out_rot - expect).abs().max().item()
            self.assertLess(
                err,
                1e-12,
                msg=f"CG({l1}, {l2} -> {l3}) equivariance failed: {err:.3e}",
            )


class TestDensityBasisHeadHigherL(unittest.TestCase):
    """Head-level checks for the Clebsch-Gordan (l >= 3) coefficient path."""

    def _head(self, coef_lmax: int = 6, **kwargs) -> DensityBasisHead:
        params = _head_params(
            use_equivariant=True,
            coef_lmax=coef_lmax,
            latent_lmax=2,
            latent_channels=8,
        )
        params.update(kwargs)
        return DensityBasisHead(**params).to(device)

    def _inputs(self, nf: int = 2, nloc: int = 4, ngrid: int = 3, nsel: int = 5):
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
        pair_dist = pair_vec.norm(dim=-1).clamp(max=1.9)
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

    def test_forward_shape(self) -> None:
        for coef_lmax in (3, 4, 6):
            with self.subTest(coef_lmax=coef_lmax):
                head = self._head(coef_lmax)
                *args, latent, pair_vec = self._inputs()
                out = head(*args, latent=latent, pair_vec=pair_vec)
                self.assertEqual(out.shape, (2, 3, 1))
                self.assertTrue(torch.isfinite(out).all())

    def test_backward_reaches_cg_modules(self) -> None:
        head = self._head(6)
        *args, latent, pair_vec = self._inputs()
        out = head(*args, latent=latent, pair_vec=pair_vec)
        out.square().sum().backward()
        grads = [
            param.grad
            for name, param in head.named_parameters()
            if ("cg_proj" in name or "cg_mix" in name) and param.grad is not None
        ]
        self.assertTrue(grads)
        self.assertTrue(any(grad.abs().sum() > 0 for grad in grads))

    def test_serialization_roundtrip(self) -> None:
        for coef_lmax in (3, 6):
            with self.subTest(coef_lmax=coef_lmax):
                head = self._head(coef_lmax)
                *args, latent, pair_vec = self._inputs()
                out = head(*args, latent=latent, pair_vec=pair_vec)
                head2 = BaseFitting.deserialize(head.serialize())
                self.assertIsInstance(head2, DensityBasisHead)
                self.assertTrue(head2.use_equivariant)
                self.assertEqual(head2.coef_lmax, coef_lmax)
                torch.testing.assert_close(
                    head2(*args, latent=latent, pair_vec=pair_vec), out
                )


class TestDensityBasisHigherLModel(unittest.TestCase):
    """End-to-end dpa4 checks for coef_lmax > 2."""

    def test_shape_and_backward(self) -> None:
        for coef_lmax in (3, 6):
            with self.subTest(coef_lmax=coef_lmax):
                model = get_model(
                    _model_params(use_equivariant=True, coef_lmax=coef_lmax)
                )
                coord, atype, box, grid = _frame()
                out = model(coord, atype, box=box, grid=grid)
                self.assertEqual(out["density"].shape, (1, 7, 1))
                self.assertTrue(torch.isfinite(out["density"]).all())
                out["density"].square().sum().backward()

    def test_rotation_translation_invariance(self) -> None:
        # the 90-degree rotation maps the cubic box onto itself, keeping PBC
        # exact; the CG-built higher-degree blocks must contract with the
        # transformed spherical harmonics without breaking invariance
        for coef_lmax in (3, 6):
            with self.subTest(coef_lmax=coef_lmax):
                model = get_model(
                    _model_params(use_equivariant=True, coef_lmax=coef_lmax)
                )
                _randomize(model)
                coord, atype, box, grid = _frame()
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

    def test_serialization(self) -> None:
        model = get_model(_model_params(use_equivariant=True, coef_lmax=3))
        coord, atype, box, grid = _frame()
        out = model(coord, atype, box=box, grid=grid)
        model2 = DensityBasisModel.deserialize(model.serialize())
        out2 = model2(coord, atype, box=box, grid=grid)
        torch.testing.assert_close(out2["density"], out["density"])


if __name__ == "__main__":
    unittest.main()
