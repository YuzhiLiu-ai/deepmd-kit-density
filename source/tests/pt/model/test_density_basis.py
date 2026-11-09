# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

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


def _model_params() -> dict:
    return {
        "type_map": ["O", "H", "X"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [8, 8, 0],
            "rcut_smth": 0.5,
            "rcut": 4.0,
            "neuron": [8, 16],
            "resnet_dt": False,
            "axis_neuron": 4,
            "seed": 1,
            "precision": "float64",
        },
        "fitting_net": {
            "type": "density_basis",
            "channels": 3,
            "n_radial": 5,
            "radial_neurons": [6],
            "seed": 1,
            "precision": "float64",
        },
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
    nf: int = 2,
    nloc: int = 5,
    ngrid: int = 7,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    coord = (
        torch.rand(nf, nloc * 3, generator=generator, dtype=torch.float64, device="cpu")
        * 8.0
    ).to(device)
    atype = torch.tensor([[0, 0, 1, 1, 1]] * nf, dtype=torch.long, device=device)
    box = (
        torch.eye(3, dtype=torch.float64, device=device).reshape(1, 9).repeat(nf, 1)
        * 10.0
    )
    grid = (
        torch.rand(nf, ngrid, 3, generator=generator, dtype=torch.float64, device="cpu")
        * 8.0
    ).to(device)
    return coord, atype, box, grid


class TestDensityBasisHead(unittest.TestCase):
    def _head(self) -> DensityBasisHead:
        return DensityBasisHead(
            ntypes=3,
            dim_descrpt=8,
            rcut=2.0,
            channels=3,
            n_radial=4,
            radial_neurons=[5],
            seed=1,
            precision="float64",
            type_map=["O", "H", "X"],
        ).to(device)

    def test_output_def(self) -> None:
        head = self._head()
        out_def = head.output_def()
        self.assertIn("density", out_def.var_defs)
        self.assertEqual(out_def["density"].shape, [1])
        self.assertFalse(out_def["density"].reducible)

    def test_forward_shape(self) -> None:
        head = self._head()
        nf, nloc, ngrid, nsel = 2, 4, 3, 5
        descriptor = torch.randn(nf, nloc, 8, dtype=torch.float64, device=device)
        pair_atype = torch.randint(0, 2, (nf, ngrid, nsel), device=device)
        pair_dist = torch.rand(nf, ngrid, nsel, dtype=torch.float64, device=device)
        pair_mask = torch.ones(nf, ngrid, nsel, dtype=torch.bool, device=device)
        coef_gather_index = torch.randint(0, nloc, (nf, ngrid, nsel), device=device)
        out = head(descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        self.assertEqual(out.shape, (nf, ngrid, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_cutoff_strictly_zero(self) -> None:
        # pairs at and beyond rcut must contribute exactly zero, whatever the
        # atom type embedding branch produces
        head = self._head()
        descriptor = torch.randn(1, 4, 8, dtype=torch.float64, device=device)
        pair_atype = torch.tensor([[[0, 1, 0]]], device=device)
        pair_dist = torch.tensor(
            [[[1.0, 2.0, 2.5]]], dtype=torch.float64, device=device
        )
        pair_mask = torch.ones(1, 1, 3, dtype=torch.bool, device=device)
        coef_gather_index = torch.tensor([[[0, 1, 2]]], device=device)
        out = head(descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        out_first = head(
            descriptor,
            pair_atype[:, :, :1],
            pair_dist[:, :, :1],
            pair_mask[:, :, :1],
            coef_gather_index[:, :, :1],
        )
        torch.testing.assert_close(out, out_first)

    def test_padding_pairs_ignored(self) -> None:
        head = self._head()
        descriptor = torch.randn(1, 4, 8, dtype=torch.float64, device=device)
        pair_atype = torch.tensor([[[0, 3, 3]]], device=device)
        pair_dist = torch.tensor(
            [[[1.0, 2.0, 1.5]]], dtype=torch.float64, device=device
        )
        pair_mask = torch.tensor([[[True, False, False]]], device=device)
        coef_gather_index = torch.tensor([[[0, 0, 0]]], device=device)
        out = head(descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        # garbage in the padded slots must not leak into the output
        pair_atype_g = torch.tensor([[[0, 3, 1]]], device=device)
        pair_dist_g = torch.tensor(
            [[[1.0, 0.01, 1.9]]], dtype=torch.float64, device=device
        )
        coef_gather_index_g = torch.tensor([[[0, 2, 3]]], device=device)
        out_g = head(
            descriptor, pair_atype_g, pair_dist_g, pair_mask, coef_gather_index_g
        )
        torch.testing.assert_close(out, out_g)

    def test_ghost_rows_share_coefficients(self) -> None:
        # a pair whose coefficient index points at the local row of a ghost
        # atom must produce the same contribution as the local atom itself
        head = self._head()
        descriptor = torch.randn(1, 2, 8, dtype=torch.float64, device=device)
        pair_atype = torch.tensor([[[0]]], device=device)
        pair_dist = torch.tensor([[[1.0]]], dtype=torch.float64, device=device)
        pair_mask = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
        out_local = head(
            descriptor,
            pair_atype,
            pair_dist,
            pair_mask,
            torch.zeros(1, 1, 1, dtype=torch.long, device=device),
        )
        out_ghost = head(
            descriptor,
            pair_atype,
            pair_dist,
            pair_mask,
            torch.ones(1, 1, 1, dtype=torch.long, device=device),
        )
        # rows 0 and 1 differ, so the gathered coefficient matters
        self.assertFalse(torch.allclose(out_local, out_ghost))
        # but the ghost row gathers exactly the row it points at
        out_row1 = head(
            descriptor[:, 1:],
            pair_atype,
            pair_dist,
            pair_mask,
            torch.zeros(1, 1, 1, dtype=torch.long, device=device),
        )
        torch.testing.assert_close(out_ghost, out_row1)

    def test_exclude_types_zeroes_contribution(self) -> None:
        head = self._head()
        descriptor = torch.randn(1, 2, 8, dtype=torch.float64, device=device)
        pair_atype = torch.tensor([[[0]]], device=device)
        pair_dist = torch.tensor([[[1.0]]], dtype=torch.float64, device=device)
        pair_mask = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
        coef_gather_index = torch.zeros(1, 1, 1, dtype=torch.long, device=device)
        out = head(descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        head.reinit_exclude([0])
        try:
            out_excl = head(
                descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index
            )
        finally:
            head.reinit_exclude([])
        self.assertFalse(torch.allclose(out, torch.zeros_like(out)))
        torch.testing.assert_close(out_excl, torch.zeros_like(out_excl))

    def test_head_serialization(self) -> None:
        head = self._head()
        descriptor = torch.randn(1, 2, 8, dtype=torch.float64, device=device)
        pair_atype = torch.tensor([[[0, 1]]], device=device)
        pair_dist = torch.tensor([[[1.0, 1.5]]], dtype=torch.float64, device=device)
        pair_mask = torch.ones(1, 1, 2, dtype=torch.bool, device=device)
        coef_gather_index = torch.tensor([[[0, 1]]], device=device)
        args = (descriptor, pair_atype, pair_dist, pair_mask, coef_gather_index)
        out = head(*args)
        head2 = BaseFitting.deserialize(head.serialize())
        self.assertIsInstance(head2, DensityBasisHead)
        torch.testing.assert_close(head2(*args), out)

    def test_rejects_conditioning(self) -> None:
        with self.assertRaisesRegex(ValueError, "fparam"):
            self._head_params(numb_fparam=1)
        with self.assertRaisesRegex(ValueError, "aparam"):
            self._head_params(numb_aparam=1)
        with self.assertRaisesRegex(ValueError, "case"):
            self._head_params(dim_case_embd=1)

    def _head_params(self, **kwargs) -> DensityBasisHead:
        params = {
            "ntypes": 3,
            "dim_descrpt": 8,
            "rcut": 2.0,
            "type_map": ["O", "H", "X"],
        }
        params.update(kwargs)
        return DensityBasisHead(**params)


class TestDensityBasisModel(unittest.TestCase):
    def _model(self) -> DensityBasisModel:
        model = get_model(_model_params())
        self.assertIsInstance(model, DensityBasisModel)
        return model

    def test_shape_and_mask(self) -> None:
        model = self._model()
        coord, atype, box, grid = _frame()
        out = model(coord, atype, box=box, grid=grid)
        self.assertEqual(out["density"].shape, (2, 7, 1))
        self.assertEqual(out["mask"].shape, (2, 7))
        self.assertTrue(torch.isfinite(out["density"]).all())

    def test_backward_updates_descriptor_and_head(self) -> None:
        model = self._model()
        coord, atype, box, grid = _frame()
        out = model(coord, atype, box=box, grid=grid)
        out["density"].square().sum().backward()
        descriptor_grads = [
            param.grad
            for name, param in model.named_parameters()
            if "descriptor" in name and param.grad is not None
        ]
        head_grads = [
            param.grad
            for name, param in model.named_parameters()
            if "fitting_net" in name and param.grad is not None
        ]
        self.assertTrue(descriptor_grads)
        self.assertTrue(head_grads)
        self.assertTrue(any(grad.abs().sum() > 0 for grad in descriptor_grads))
        self.assertTrue(any(grad.abs().sum() > 0 for grad in head_grads))

    def test_pbc_ghost_coefficient(self) -> None:
        # one atom near the box face, one grid point near the opposite face:
        # only the periodic image of the atom is within the cutoff, so the
        # periodic run must match a non-periodic run with the image placed
        # explicitly (and differ from a non-periodic run without the image)
        model = self._model()
        _randomize(model)
        box_len = 8.0
        coord_pbc = torch.tensor([[0.5, 4.0, 4.0]], dtype=torch.float64, device=device)
        atype = torch.tensor([[0]], dtype=torch.long, device=device)
        box = torch.eye(3, dtype=torch.float64, device=device).reshape(1, 9) * box_len
        grid = torch.tensor([[[7.8, 4.0, 4.0]]], dtype=torch.float64, device=device)
        out_pbc = model(coord_pbc, atype, box=box, grid=grid)["density"]
        # without PBC the only atom is 7.3 A away, outside the cutoff
        out_nobox = model(coord_pbc, atype, box=None, grid=grid)["density"]
        torch.testing.assert_close(out_nobox, torch.zeros_like(out_nobox))
        # with the image placed explicitly the densities coincide
        coord_image = torch.tensor(
            [[0.5, 4.0, 4.0, 0.5 + box_len, 4.0, 4.0]],
            dtype=torch.float64,
            device=device,
        )
        atype_image = torch.tensor([[0, 0]], dtype=torch.long, device=device)
        out_image = model(coord_image, atype_image, box=None, grid=grid)["density"]
        self.assertFalse(torch.allclose(out_pbc, torch.zeros_like(out_pbc)))
        torch.testing.assert_close(out_pbc, out_image)

    def test_rotation_translation_invariance(self) -> None:
        # a scalar radial expansion on a rotation-invariant descriptor must
        # be invariant to rigid motions of the whole system; the 90-degree
        # rotation maps the cubic box onto itself, keeping PBC exact
        model = self._model()
        coord, atype, box, grid = _frame(nf=1, nloc=5, ngrid=7)
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
        torch.testing.assert_close(out, out_rt)

    def test_atom_exclude_types(self) -> None:
        model = self._model()
        coord, atype, box, grid = _frame()
        out_none = model(coord, atype, box=box, grid=grid)["density"]
        model.atomic_model.reinit_atom_exclude([0])
        try:
            out_excl = model(coord, atype, box=box, grid=grid)["density"]
        finally:
            model.atomic_model.reinit_atom_exclude([])
        self.assertTrue(torch.isfinite(out_excl).all())
        self.assertFalse(torch.allclose(out_none, out_excl))

    def test_serialization(self) -> None:
        model = self._model()
        coord, atype, box, grid = _frame()
        out = model(coord, atype, box=box, grid=grid)
        model2 = DensityBasisModel.deserialize(model.serialize())
        out2 = model2(coord, atype, box=box, grid=grid)
        torch.testing.assert_close(out2["density"], out["density"])

    def test_jit_script(self) -> None:
        model = self._model()
        coord, atype, box, grid = _frame()
        out = model(coord, atype, box=box, grid=grid)
        scripted = torch.jit.script(model)
        out_s = scripted(coord, atype, box=box, grid=grid)
        torch.testing.assert_close(out_s["density"], out["density"])
        self.assertTrue(scripted.has_grid())

    def test_type_map_requires_reserved_grid_type(self) -> None:
        params = _model_params()
        params["type_map"] = ["O", "H", "He"]
        with self.assertRaisesRegex(ValueError, "reserved grid type"):
            get_model(params)


if __name__ == "__main__":
    unittest.main()
