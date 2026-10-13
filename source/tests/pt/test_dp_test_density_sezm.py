# SPDX-License-Identifier: LGPL-3.0-or-later
"""Grid-density model on the DPA4/SeZM descriptor.

The density pipeline is descriptor-agnostic: grid points are appended as
pseudo-atoms of the reserved type (the last entry of the type map) and the
merged grid+atom system goes through the descriptor in one call.  These
tests exercise that path with DescrptSeZM, which keeps no input statistics
(``compute_input_stats`` is a no-op) and guards divisions with ``eps``
instead of ``env_protection``.
"""

import os
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.entrypoints.test import (
    test as dp_test,
)
from deepmd.infer.deep_density import (
    DeepDensity,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.model.descriptor.sezm import (
    DescrptSeZM,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.density_model import (
    GridDensityModel,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.utils.argcheck import (
    normalize,
)

model_density_sezm = {
    "type_map": ["O", "H", "X"],
    "descriptor": {
        "type": "dpa4",
        "sel": 8,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "n_blocks": 2,
        "mixing_layers": 2,
        "lmax": 2,
        "mmax": 1,
        "use_env_seed": True,
        "edge_norm": False,
        "random_gamma": False,
        "use_amp": False,
        "precision": "float32",
        "seed": 1,
    },
    "fitting_net": {
        "type": "density",
        "neuron": [8, 8],
        "seed": 1,
    },
}


class TestDPTestDensitySeZM(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        root = Path(cls.tmpdir.name)

        # same synthetic density system as the se_e2_a/dpa2/dpa3 density tests
        rng = np.random.default_rng(42)
        cls.nframes, natoms, cls.ngrid = 4, 5, 8
        cls.system = root / "system"
        set_dir = cls.system / "set.000"
        set_dir.mkdir(parents=True)
        np.save(set_dir / "coord.npy", rng.random((cls.nframes, natoms * 3)) * 8.0)
        np.save(
            set_dir / "box.npy",
            np.tile(np.eye(3).reshape(-1) * 10.0, (cls.nframes, 1)),
        )
        np.save(set_dir / "grid.npy", rng.random((cls.nframes, cls.ngrid, 3)) * 8.0)
        np.save(set_dir / "density.npy", rng.random((cls.nframes, cls.ngrid, 1)))
        np.savetxt(cls.system / "type.raw", [0, 0, 1, 1, 1], fmt="%d")
        np.savetxt(cls.system / "type_map.raw", ["O", "H", "X"], fmt="%s")

        cls.config = {
            "model": deepcopy(model_density_sezm),
            "learning_rate": {
                "type": "exp",
                "start_lr": 0.001,
                "stop_lr": 1e-8,
                "decay_steps": 10,
            },
            "optimizer": {"type": "Adam"},
            "loss": {
                "type": "grid_density",
                "start_pref_d": 1.0,
                "limit_pref_d": 1.0,
            },
            "training": {
                "training_data": {
                    "systems": [str(cls.system)],
                    "batch_size": 1,
                },
                "validation_data": {
                    "systems": [str(cls.system)],
                    "batch_size": 1,
                },
                "numb_steps": 1,
                "seed": 1,
                "disp_file": os.devnull,
                "save_freq": 100,
            },
        }

        # build and freeze a tiny density model through the trainer path
        cls.normalized_config = normalize(deepcopy(cls.config))
        trainer = get_trainer(deepcopy(cls.normalized_config))
        with torch.device("cpu"):
            input_dict, _, _ = trainer.get_data(is_train=False)
        # the density model takes grid instead of spin as the extra input
        input_dict.pop("spin", None)
        trainer.model(**input_dict)
        # keep the live torch model and a data sample for the gradient check
        cls.torch_model = trainer.model
        cls.input_dict = input_dict
        # SeZM carries PEP-604 annotations that TorchScript cannot parse, so
        # unlike the se_e2_a/dpa2/dpa3 density models it is not exported to a
        # frozen TorchScript archive here; it is served from the training
        # checkpoint (.pt) instead, exactly like `dp test -m model.ckpt-*.pt`.
        # DeepEval detects SeZM model params and skips scripting on load.
        wrapper = ModelWrapper(
            trainer.model, model_params=cls.normalized_config["model"]
        )
        tmp_fd, cls.model_path = tempfile.mkstemp(suffix=".pt")
        os.close(tmp_fd)
        torch.save({"model": wrapper.state_dict()}, cls.model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        os.unlink(cls.model_path)
        cls.tmpdir.cleanup()

    def test_model_is_grid_density(self) -> None:
        self.assertIsInstance(self.torch_model, GridDensityModel)
        self.assertTrue(self.torch_model.has_grid())

    def test_forward_shape_and_finite(self) -> None:
        model_out = self.torch_model(**self.input_dict)
        density = model_out["density"]
        nframes = self.input_dict["coord"].shape[0]
        self.assertEqual(density.shape, (nframes, self.ngrid, 1))
        self.assertTrue(torch.isfinite(density).all())

    def test_grid_at_atoms_finite(self) -> None:
        # grid points coincident with atoms are legitimate inputs; SeZM has
        # no 1/r environment matrix, and its eps floor keeps the coincident
        # distances finite
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:2]
        box = np.load(set_dir / "box.npy")[:2]
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        grid_at_atoms = coord.reshape(coord.shape[0], -1, 3)
        out = DeepDensity(self.model_path).eval(coord, box, atype, grid=grid_at_atoms)
        self.assertTrue(np.isfinite(out).all())

        # gradients wrt the grid input and all parameters stay finite
        input_dict = {
            kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
            for kk, vv in self.input_dict.items()
        }
        coord_t = input_dict["coord"]
        grid_t = (
            coord_t.reshape(coord_t.shape[0], -1, 3).detach().clone().requires_grad_()
        )
        input_dict["grid"] = grid_t
        model_out = self.torch_model(**input_dict)
        self.assertTrue(torch.isfinite(model_out["density"]).all())
        model_out["density"].sum().backward()
        self.assertIsNotNone(grid_t.grad)
        self.assertTrue(torch.isfinite(grid_t.grad).all())
        for param in self.torch_model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_no_input_statistics_patch(self) -> None:
        # SeZM keeps no davg/dstd statistics at all; the density model must
        # detect that and skip the grid-type stat patch instead of warning
        descriptor = self.torch_model.atomic_model.descriptor
        self.assertIsInstance(descriptor, DescrptSeZM)
        with self.assertLogs(
            "deepmd.pt.model.atomic_model.density_atomic_model", level="INFO"
        ) as cm:
            self.torch_model.atomic_model.compute_or_load_stat(lambda: [])
        self.assertTrue(
            any("no davg/dstd input statistics" in line for line in cm.output),
        )

    def test_normalize_eps_guard(self) -> None:
        # the density default-protection hook must not inject the foreign
        # `env_protection` key into a SeZM descriptor...
        data = normalize(deepcopy(self.config))
        self.assertNotIn("env_protection", data["model"]["descriptor"])
        self.assertGreater(data["model"]["descriptor"]["eps"], 0.0)
        # ...but an explicit eps: 0 must be lifted to a positive floor
        config = deepcopy(self.config)
        config["model"]["descriptor"]["eps"] = 0.0
        data = normalize(config)
        self.assertEqual(data["model"]["descriptor"]["eps"], 1e-6)

    def test_env_protection_untouched_for_other_descriptors(self) -> None:
        # the eps branch must not change the env_protection behavior of
        # descriptors that carry it
        config = deepcopy(self.config)
        config["model"]["descriptor"] = {
            "type": "se_e2_a",
            "sel": [8, 8, 0],
            "rcut_smth": 0.50,
            "rcut": 4.00,
            "neuron": [8, 16],
            "resnet_dt": False,
            "axis_neuron": 4,
            "seed": 1,
        }
        data = normalize(config)
        self.assertEqual(data["model"]["descriptor"]["env_protection"], 1e-6)
        self.assertNotIn("eps", data["model"]["descriptor"])

    def test_use_compile_matches_eager(self) -> None:
        # the descriptor-level use_compile flag must route the block stack
        # through torch.compile with numerically identical forward/backward
        section = deepcopy(model_density_sezm)
        section_eager = deepcopy(section)
        section["descriptor"]["use_compile"] = True
        model_eager = get_model(deepcopy(section_eager))
        model_compiled = get_model(deepcopy(section))
        model_eager.train()
        model_compiled.train()
        self.assertFalse(model_eager.atomic_model.descriptor.use_compile)
        self.assertTrue(model_compiled.atomic_model.descriptor.use_compile)

        rng = np.random.default_rng(7)

        def _input(nf: int, ngrid: int) -> dict:
            # source/tests/pt sets a fake CUDA default device: every tensor
            # must be created with an explicit device
            return {
                "coord": torch.tensor(
                    rng.random((nf, 15)) * 8.0,
                    dtype=torch.float64,
                    device="cpu",
                ),
                "atype": torch.tensor(
                    [[0, 0, 1, 1, 1]] * nf, dtype=torch.long, device="cpu"
                ),
                "box": torch.tensor(
                    np.tile(np.eye(3).reshape(-1) * 10.0, (nf, 1)),
                    dtype=torch.float64,
                    device="cpu",
                ),
                "grid": torch.tensor(
                    rng.random((nf, ngrid, 3)) * 8.0,
                    dtype=torch.float64,
                    device="cpu",
                ),
            }

        def _forward_backward(model, inp) -> tuple[torch.Tensor, torch.Tensor]:
            model.zero_grad(set_to_none=True)
            out = model(**inp)
            loss = out["density"].square().sum()
            loss.backward()
            grad = next(
                p.grad
                for p in model.atomic_model.descriptor.blocks[0].parameters()
                if p.grad is not None
            )
            return out["density"].detach(), grad.detach().clone()

        inp = _input(nf=3, ngrid=8)
        dens_eager, grad_eager = _forward_backward(model_eager, inp)
        dens_comp, grad_comp = _forward_backward(model_compiled, inp)
        self.assertTrue(torch.isfinite(dens_comp).all())
        torch.testing.assert_close(dens_comp, dens_eager)
        torch.testing.assert_close(grad_comp, grad_eager)

        # dynamic shapes: a differently sized batch must not recompile-error
        inp2 = _input(nf=5, ngrid=11)
        dens2, _ = _forward_backward(model_compiled, inp2)
        dens2_eager, _ = _forward_backward(model_eager, inp2)
        self.assertTrue(torch.isfinite(dens2).all())
        torch.testing.assert_close(dens2, dens2_eager)

    def test_dp_test(self) -> None:
        detail_file = os.path.join(self.tmpdir.name, "detail_sezm")
        dp_test(
            model=self.model_path,
            system=str(self.system),
            datafile=None,
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=False,
        )
        # one detail file per frame, each holding ngrid [label, pred] pairs
        for frame in range(self.nframes):
            detail = np.loadtxt(f"{detail_file}.density.out.{frame}", skiprows=1)
            self.assertEqual(detail.shape, (self.ngrid, 2))


if __name__ == "__main__":
    unittest.main()
