# SPDX-License-Identifier: LGPL-3.0-or-later
"""Non-periodic direct latent-charge plugin tests (pt_expt)."""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    ALL_PAIRS,
    K_GRID,
    PME_MESH,
    build_topologies,
)
from deepmd.pt_expt.long_range import BaseLongRange, LESKernel, SOGKernel
from deepmd.pt_expt.model.edge_transform_output import releaf_topologies
from deepmd.pt_expt.model.get_model import DPAtomicModel


def test_torch_kernel_registration_and_serialization() -> None:
    les = LESKernel(alpha=0.7, trainable=False)
    assert isinstance(les.log_alpha, torch.nn.Parameter)
    assert not les.log_alpha.requires_grad
    r_sq = torch.tensor([0.0, 1.0], dtype=torch.float64)
    restored = LESKernel.deserialize(les.serialize())
    torch.testing.assert_close(restored(r_sq), les(r_sq), rtol=0.0, atol=0.0)

    sog = SOGKernel(amplitude=[0.7, -0.2], width=[0.8, 1.7])
    assert sog.amplitude.requires_grad
    assert SOGKernel.deserialize(sog.serialize()).amplitude.shape == (2,)


def test_releaf_topologies_detaches_geometry_inputs() -> None:
    coord = torch.arange(18, dtype=torch.float64).reshape(6, 3).requires_grad_(True)
    n_node = torch.tensor([2, 4])
    atype = torch.zeros(6, dtype=torch.int64)
    box = torch.stack([torch.eye(3, dtype=torch.float64) * 8.0] * 2)
    names = (ALL_PAIRS, K_GRID, PME_MESH)
    topologies = build_topologies(
        names,
        coord,
        n_node,
        box,
        atype=atype,
        topology_params={K_GRID: {"k_max": 1.0}, PME_MESH: {"spacing": 2.0}},
    )
    releafed = releaf_topologies(topologies)
    for name, topology in releafed.items():
        for field in topology.LEAVES:
            value = getattr(topology, field)
            assert value.is_leaf
            assert value.requires_grad
    assert releafed[ALL_PAIRS].vec.data_ptr() == topologies[ALL_PAIRS].vec.data_ptr()
    assert DPAtomicModel.base_long_range_cls is BaseLongRange


from deepmd.dpmodel.long_range.kernel import COULOMB_CONSTANT
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.model.model import BaseModel
from deepmd.pt_expt.utils import env

from ..compile_utils import REQUIRES_SUPPORTED_COMPILE


def _charge_config(kernel: str = "les", with_long_range: bool = True) -> dict:
    config = {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "dpa4c",
            "rcut": 3.0,
            "channels": 8,
            "lmax": 2,
            "n_radial": 4,
            "precision": "float64",
            "seed": 17,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [16, 16],
            "precision": "float64",
            "seed": 19,
        },
    }
    if with_long_range:
        kernel_config = {"type": "les", "alpha": 0.9}
        if kernel == "sog":
            kernel_config = {
                "type": "sog",
                "amplitude": [0.7, -0.2],
                "width": [0.8, 1.7],
            }
        config["long_range"] = {
            "type": "latent_charge",
            "channels": 1,
            "charge_net": {"neuron": [8, 8]},
            "kernel": kernel_config,
            "charge_constraint": {"source": "neutral", "channel": 0},
        }
    return config


def _charge_model(kernel: str = "les", train: bool = False) -> torch.nn.Module:
    model = get_model(_charge_config(kernel)).to(env.DEVICE)
    model.train(train)
    return model


def _batch(nlocs=(4, 3), seed=0):
    rng = np.random.default_rng(seed)
    coord = np.zeros((len(nlocs), max(nlocs), 3))
    atype = np.full((len(nlocs), max(nlocs)), -1, dtype=np.int64)
    for f, n in enumerate(nlocs):
        coord[f, :n] = rng.uniform(0.2, 4.0, (n, 3))
        atype[f, :n] = rng.integers(0, 2, n)
    return (
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(atype, dtype=torch.int64, device=env.DEVICE),
    )


def _brute_lr_energy(coord, atype, charge, kernel):
    energies = []
    for f in range(coord.shape[0]):
        total = coord.new_zeros(())
        for i in range(coord.shape[1]):
            if atype[f, i] < 0:
                continue
            for j in range(i + 1, coord.shape[1]):
                if atype[f, j] < 0:
                    continue
                r_sq = ((coord[f, i] - coord[f, j]) ** 2).sum().reshape(1)
                total = total + charge[f, i, 0] * charge[f, j, 0] * kernel(r_sq)
        energies.append((total * COULOMB_CONSTANT).reshape(()))
    return torch.stack(energies)


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_direct_energy_matches_pair_reference_and_attribution(kernel) -> None:
    model = _charge_model(kernel)
    short = get_model(_charge_config(kernel, with_long_range=False)).to(env.DEVICE)
    short.eval()
    coord, atype = _batch()
    result = model(coord, atype)
    short_result = short(coord, atype)
    lr = model.atomic_model.long_range
    expected = _brute_lr_energy(coord, atype, result["latent_charge"], lr.kernel)
    torch.testing.assert_close(
        result["energy"] - short_result["energy"],
        expected.reshape(-1, 1),
        rtol=1e-12,
        atol=1e-12,
    )
    count = (atype >= 0).sum(dim=1).to(expected.dtype)
    expected_atom = (expected / count)[:, None, None].expand_as(result["atom_energy"])
    expected_atom = expected_atom * (atype >= 0)[..., None]
    torch.testing.assert_close(
        result["atom_energy"] - short_result["atom_energy"],
        expected_atom,
        rtol=1e-12,
        atol=1e-12,
    )


def test_frame_independence_and_charge_constraint_autograd() -> None:
    model = _charge_model(train=True)
    coord, atype = _batch((4, 3), seed=5)
    batched = model(coord, atype)
    for f, n in enumerate((4, 3)):
        alone = model(coord[f : f + 1, :n], atype[f : f + 1, :n])
        for key in ("energy", "virial"):
            torch.testing.assert_close(batched[key][f], alone[key][0])
        for key in ("force", "latent_charge"):
            torch.testing.assert_close(batched[key][f, :n], alone[key][0])
        torch.testing.assert_close(
            batched["latent_charge"][f, :n].sum(dim=0),
            torch.zeros(1, dtype=torch.float64, device=env.DEVICE),
            atol=1e-12,
            rtol=0.0,
        )
    loss = batched["latent_charge"].square().sum()
    loss.backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in model.atomic_model.long_range.charge_net.parameters()
    )


def test_force_and_virial_match_finite_differences() -> None:
    model = _charge_model()
    coord, atype = _batch((4,), seed=11)
    result = model(coord, atype, do_atomic_virial=True)
    eps = 1e-5

    def energy(value):
        return model(value, atype)["energy"].item()

    force_fd = torch.zeros_like(result["force"])
    for atom in range(4):
        for d in range(3):
            plus, minus = coord.clone(), coord.clone()
            plus[0, atom, d] += eps
            minus[0, atom, d] -= eps
            force_fd[0, atom, d] = -(energy(plus) - energy(minus)) / (2 * eps)
    torch.testing.assert_close(result["force"], force_fd, rtol=2e-6, atol=2e-6)

    virial_fd = torch.zeros_like(result["virial"])
    for a, b in ((0, 0), (0, 1), (1, 2), (2, 2)):
        strain = torch.zeros((3, 3), dtype=torch.float64, device=env.DEVICE)
        strain[a, b] = eps
        ep = energy(
            coord @ (torch.eye(3, dtype=coord.dtype, device=coord.device) + strain).T
        )
        em = energy(
            coord @ (torch.eye(3, dtype=coord.dtype, device=coord.device) - strain).T
        )
        virial_fd[0, 3 * a + b] = -(ep - em) / (2 * eps)
        if a != b:
            virial_fd[0, 3 * b + a] = virial_fd[0, 3 * a + b]
    for a, b in ((0, 0), (0, 1), (1, 2), (2, 2)):
        torch.testing.assert_close(
            result["virial"][0, 3 * a + b],
            virial_fd[0, 3 * a + b],
            rtol=2e-6,
            atol=2e-6,
        )
    torch.testing.assert_close(result["atom_virial"].sum(dim=1), result["virial"])


def test_force_loss_second_order_gradients_and_serialization_round_trip() -> None:
    model = _charge_model(train=True)
    coord, atype = _batch((4,), seed=13)
    result = model(coord, atype)
    (result["force"].square().sum() + result["energy"].square().sum()).backward()
    lr = model.atomic_model.long_range
    assert lr.kernel.log_alpha.grad is not None
    assert torch.isfinite(lr.kernel.log_alpha.grad).all()
    assert torch.count_nonzero(lr.kernel.log_alpha.grad) > 0
    for name, parameter in lr.charge_net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in model.atomic_model.descriptor.parameters()
    )

    model.eval()
    restored = BaseModel.deserialize(model.serialize()).to(env.DEVICE).eval()
    assert restored.atomic_model.serialize()["@version"] == 3
    assert isinstance(restored.atomic_model.long_range, type(lr))
    expected = model(coord, atype, do_atomic_virial=True)
    actual = restored(coord, atype, do_atomic_virial=True)
    for key in (
        "energy",
        "atom_energy",
        "force",
        "virial",
        "atom_virial",
        "latent_charge",
    ):
        torch.testing.assert_close(actual[key], expected[key])


def test_no_long_range_regression_and_periodic_rejection() -> None:
    model = get_model(_charge_config(with_long_range=False)).to(env.DEVICE).eval()
    coord, atype = _batch((4,), seed=17)
    result = model(coord, atype, do_atomic_virial=True)
    for key in ("energy", "force", "virial", "atom_virial"):
        assert torch.isfinite(result[key]).all()
    assert model.atomic_model.required_topologies() == ()
    record = model.serialize()
    assert record["@version"] == 2
    assert "long_range" not in record
    assert "latent_charge" not in result

    long_range_model = _charge_model()
    box = torch.eye(3, dtype=torch.float64, device=env.DEVICE).reshape(1, 9)
    with pytest.raises(NotImplementedError, match="non-periodic"):
        long_range_model(coord, atype, box=box)


@REQUIRES_SUPPORTED_COMPILE
def test_compiled_direct_matches_eager() -> None:
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    model = _charge_model()
    compiled = _CompiledModel(model, _get_model_structure_key(model)).eval()
    coord, atype = _batch((4, 3), seed=19)
    expected = model(coord, atype, do_atomic_virial=True)
    actual = compiled(coord, atype, None, do_atomic_virial=True)
    for key in (
        "energy",
        "atom_energy",
        "force",
        "virial",
        "atom_virial",
        "latent_charge",
    ):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-10, atol=1e-10)


def _old_and_new_charge_models():
    old_config = {
        "type": "dpa4c_lr",
        **_charge_config(with_long_range=False),
    }
    old_config["fitting_net"] = {
        **old_config["fitting_net"],
        "type": "dpa4c_lr",
        "dim_out_lr": 1,
        "neuron_lr": [8, 8],
        "use_charge_constraint": True,
        "lr_kernel": "les",
        "les_alpha": 0.9,
        "use_glu": False,
    }
    old = get_model(old_config).to(env.DEVICE).eval()
    new = _charge_model()
    old_state = old.state_dict()
    new_state = new.state_dict()
    shared = {
        key: value
        for key, value in old_state.items()
        if key in new_state
        and (
            key.startswith("atomic_model.descriptor.")
            or key.startswith("atomic_model.fitting_net.nets.")
            or key
            in {
                "atomic_model.out_bias",
                "atomic_model.out_std",
                "atomic_model.fitting_net.bias_atom_e",
            }
        )
    }
    new.load_state_dict(shared, strict=False)
    charge_weights = {
        key.replace(
            "atomic_model.fitting_net.filter_layers_lr.",
            "atomic_model.long_range.charge_net.nets.",
        ): value
        for key, value in old_state.items()
        if key.startswith("atomic_model.fitting_net.filter_layers_lr.")
    }
    new.load_state_dict(charge_weights, strict=False)
    with torch.no_grad():
        old.atomic_model.fitting_net.bias_atom_q.zero_()
        new.atomic_model.long_range.charge_net.bias_atom_e.zero_()
        new.atomic_model.long_range.kernel.log_alpha.copy_(
            old.atomic_model.fitting_net.les_alpha.log()
        )
    return old, new


def test_new_direct_matches_old_dpa4c_charge_only() -> None:
    old, new = _old_and_new_charge_models()
    coord, atype = _batch((4, 3), seed=23)
    expected = old(coord, atype, do_atomic_virial=True)
    actual = new(coord, atype, do_atomic_virial=True)
    for key in (
        "energy",
        "atom_energy",
        "force",
        "virial",
        "atom_virial",
        "latent_charge",
    ):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-11, atol=1e-11)


def test_long_range_argcheck_normalizes_direct_config() -> None:
    from deepmd.utils.argcheck import normalize

    config = {
        "model": _charge_config(),
        "learning_rate": {"type": "exp", "start_lr": 1e-3},
        "loss": {"type": "ener"},
        "training": {
            "numb_steps": 1,
            "training_data": {"systems": ["unused"], "batch_size": 1},
        },
    }
    normalized = normalize(config)["model"]["long_range"]
    assert normalized["type"] == "latent_charge"
    assert normalized["periodic"] is False
    assert normalized["kernel"]["type"] == "les"


@REQUIRES_SUPPORTED_COMPILE
def test_compiled_force_loss_second_order_gradients_match_eager() -> None:
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    eager = _charge_model(train=True)
    compiled_model = _charge_model(train=True)
    compiled_model.load_state_dict(eager.state_dict())
    compiled = _CompiledModel(
        compiled_model, _get_model_structure_key(compiled_model)
    ).train()
    coord, atype = _batch((4,), seed=29)
    eager_loss = eager(coord, atype)["force"].square().sum()
    compiled_loss = compiled(coord, atype, None)["force"].square().sum()
    eager_loss.backward()
    compiled_loss.backward()
    torch.testing.assert_close(compiled_loss, eager_loss)
    eager_parameters = dict(eager.named_parameters())
    for name, parameter in compiled_model.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, eager_parameters[name].grad, rtol=1e-8, atol=1e-8
        )


def _multipole_config(steps: int = 0, use_self_pol: bool = True) -> dict:
    config = _charge_config(with_long_range=False)
    config["long_range"] = {
        "type": "latent_multipole",
        "charge_net": {"neuron": [8, 8], "use_glu": False},
        "kernel": {"type": "les", "alpha": 0.9},
        "charge_constraint": {"source": "neutral"},
        "use_self_pol": use_self_pol,
        "lr_update_steps": steps,
    }
    return config


def _multipole_model(steps: int = 0, train: bool = False) -> torch.nn.Module:
    model = get_model(_multipole_config(steps)).to(env.DEVICE)
    model.train(train)
    return model


def _old_and_new_multipole_models(steps: int):
    old_config = {"type": "dpa4c_lr", **_charge_config(with_long_range=False)}
    old_config["fitting_net"] = {
        **old_config["fitting_net"],
        "type": "dpa4c_lr",
        "dim_out_lr": 1,
        "neuron_lr": [8, 8],
        "use_charge_constraint": True,
        "lr_kernel": "dipole",
        "les_alpha": 0.9,
        "use_glu": False,
        "lr_update_steps": steps,
    }
    old = get_model(old_config).to(env.DEVICE).eval()
    new = _multipole_model(steps)
    old_state, new_state = old.state_dict(), new.state_dict()
    shared = {
        key: value
        for key, value in old_state.items()
        if key in new_state
        and (
            key.startswith("atomic_model.descriptor.")
            or key.startswith("atomic_model.fitting_net.nets.")
            or key == "atomic_model.fitting_net.bias_atom_e"
        )
    }
    new.load_state_dict(shared, strict=False)
    mapped = {}
    for key, value in old_state.items():
        if key.startswith("atomic_model.fitting_net.filter_layers_lr."):
            mapped[
                key.replace(
                    "atomic_model.fitting_net.filter_layers_lr.",
                    "atomic_model.long_range.charge_net.nets.",
                )
            ] = value
        elif key.startswith("atomic_model.fitting_net.dipole_weight_net."):
            mapped[
                key.replace(
                    "atomic_model.fitting_net.dipole_weight_net.",
                    "atomic_model.long_range.charge_net.dipole_weight_net.",
                )
            ] = value
        elif key.startswith("atomic_model.fitting_net.polar_update_net."):
            mapped[
                key.replace(
                    "atomic_model.fitting_net.polar_update_net.",
                    "atomic_model.long_range.charge_net.polar_update_net.",
                )
            ] = value
    new.load_state_dict(mapped, strict=False)
    with torch.no_grad():
        new.atomic_model.long_range.charge_net.bias_atom_e.copy_(
            old.atomic_model.fitting_net.bias_atom_q
        )
        new.atomic_model.long_range.charge_net.alpha_pol.copy_(
            old.atomic_model.fitting_net.alpha_pol
        )
        new.atomic_model.long_range.kernel.log_alpha.copy_(
            old.atomic_model.fitting_net.les_alpha.log()
        )
        # The legacy fitting includes lr_descriptor in its output statistics;
        # keep it out of this numerical comparison of the physical outputs.
        old.atomic_model.out_bias.zero_()
        new.atomic_model.out_bias.zero_()
        old.atomic_model.out_std.zero_()
        new.atomic_model.out_std.zero_()
    return old, new


@pytest.mark.parametrize("steps", [0, 1, 2])
def test_new_direct_matches_old_dpa4c_multipole(steps: int) -> None:
    old, new = _old_and_new_multipole_models(steps)
    coord, atype = _batch((4, 3), seed=23)
    expected = old(coord, atype, do_atomic_virial=True)
    actual = new(coord, atype, do_atomic_virial=True)
    for key in (
        "energy",
        "atom_energy",
        "force",
        "virial",
        "atom_virial",
        "latent_charge",
        "latent_dipole",
    ):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-11, atol=1e-11)


def test_multipole_rotation_equivariance_and_self_pol_switch() -> None:
    model = _multipole_model(steps=1)
    coord, atype = _batch((4,), seed=31)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=coord.dtype,
        device=coord.device,
    )
    reference = model(coord, atype)
    rotated = model(coord @ rotation.T, atype)
    torch.testing.assert_close(rotated["energy"], reference["energy"])
    torch.testing.assert_close(
        rotated["force"], reference["force"] @ rotation.T, rtol=1e-10, atol=1e-10
    )
    torch.testing.assert_close(
        rotated["latent_dipole"],
        reference["latent_dipole"] @ rotation.T,
        rtol=1e-10,
        atol=1e-10,
    )
    without_self_pol = (
        get_model(_multipole_config(steps=1, use_self_pol=False)).to(env.DEVICE).eval()
    )
    without_self_pol.load_state_dict(model.state_dict())
    assert not torch.allclose(
        without_self_pol(coord, atype)["energy"], reference["energy"]
    )


def test_multipole_serialization_and_auxiliary_gradient() -> None:
    model = _multipole_model(steps=1, train=True)
    coord, atype = _batch((4,), seed=37)
    result = model(coord, atype)
    (
        result["latent_charge"].square().sum() + result["latent_dipole"].square().sum()
    ).backward()
    update_net = model.atomic_model.long_range.charge_net.polar_update_net
    assert update_net is not None
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in update_net.parameters()
    )
    model.eval()
    restored = BaseModel.deserialize(model.serialize()).to(env.DEVICE).eval()
    expected = model(coord, atype, do_atomic_virial=True)
    actual = restored(coord, atype, do_atomic_virial=True)
    for key in (
        "energy",
        "atom_energy",
        "force",
        "virial",
        "atom_virial",
        "latent_charge",
        "latent_dipole",
    ):
        torch.testing.assert_close(actual[key], expected[key])


@REQUIRES_SUPPORTED_COMPILE
def test_compiled_multipole_matches_eager() -> None:
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    model = _multipole_model(steps=1)
    compiled = _CompiledModel(model, _get_model_structure_key(model)).eval()
    coord, atype = _batch((4, 3), seed=19)
    expected = model(coord, atype, do_atomic_virial=True)
    actual = compiled(coord, atype, None, do_atomic_virial=True)
    for key in (
        "energy",
        "atom_energy",
        "force",
        "virial",
        "atom_virial",
        "latent_charge",
        "latent_dipole",
    ):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-10, atol=1e-10)


@REQUIRES_SUPPORTED_COMPILE
def test_compiled_multipole_force_loss_second_order_gradients_match_eager() -> None:
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    eager = _multipole_model(steps=1, train=True)
    compiled_model = _multipole_model(steps=1, train=True)
    compiled_model.load_state_dict(eager.state_dict())
    compiled = _CompiledModel(
        compiled_model, _get_model_structure_key(compiled_model)
    ).train()
    coord, atype = _batch((4,), seed=29)
    eager_loss = eager(coord, atype)["force"].square().sum()
    compiled_loss = compiled(coord, atype, None)["force"].square().sum()
    eager_loss.backward()
    compiled_loss.backward()
    torch.testing.assert_close(compiled_loss, eager_loss)
    eager_parameters = dict(eager.named_parameters())
    for name, parameter in compiled_model.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, eager_parameters[name].grad, rtol=1e-8, atol=1e-8
        )


def test_multipole_argcheck_normalizes_direct_config() -> None:
    from deepmd.utils.argcheck import normalize

    config = {
        "model": _multipole_config(steps=1),
        "learning_rate": {"type": "exp", "start_lr": 1e-3},
        "loss": {"type": "ener"},
        "training": {
            "numb_steps": 1,
            "training_data": {"systems": ["unused"], "batch_size": 1},
        },
    }
    normalized = normalize(config)["model"]["long_range"]
    assert normalized["type"] == "latent_multipole"
    assert normalized["lr_update_steps"] == 1
    assert normalized["kernel"]["type"] == "les"
    assert normalized["periodic"] is False


# ---------------------------------------------------------------------------
# Periodic (k-grid) evaluation
# ---------------------------------------------------------------------------


def _periodic_config(kernel: str = "les", channels: int = 1) -> dict:
    config = _charge_config(kernel, with_long_range=True)
    config["long_range"]["periodic"] = True
    config["long_range"]["channels"] = channels
    return config


def _periodic_model(kernel: str = "les", channels: int = 1, train: bool = False):
    model = get_model(_periodic_config(kernel, channels)).to(env.DEVICE)
    model.train(train)
    return model


def _periodic_batch(nlocs=(5, 4), seed=0):
    rng = np.random.default_rng(seed)
    nf = len(nlocs)
    n_max = max(nlocs)
    coord = np.zeros((nf, n_max, 3))
    atype = np.full((nf, n_max), -1, dtype=np.int64)
    cells = []
    for f, n in enumerate(nlocs):
        cells.append(np.eye(3) * (6.0 + f) + rng.normal(0, 0.2, (3, 3)))
        coord[f, :n] = rng.uniform(0, 5, (n, 3)) @ cells[-1] / 6.0
        atype[f, :n] = rng.integers(0, 2, n)
    return (
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(atype, dtype=torch.int64, device=env.DEVICE),
        torch.tensor(np.stack(cells), dtype=torch.float64, device=env.DEVICE),
    )


def _kspace_reference(coord, cell, charge, kernel) -> float:
    """Independent structure-factor evaluation of the k-space energy."""
    import math

    volume = abs(np.linalg.det(cell))
    if kernel.TYPE == "les":
        alpha = float(np.exp(kernel.log_alpha.detach().cpu().numpy()).reshape(-1)[0])
        k_max = 2 * alpha * math.sqrt(math.log(1 / 1e-8))

        def kspace(k2):
            return 4 * np.pi * np.exp(-k2 / (4 * alpha * alpha)) / k2

        contact = 2 * alpha / np.sqrt(np.pi)
    else:
        amp = kernel.amplitude.detach().cpu().numpy()
        width = np.exp(kernel.log_width.detach().cpu().numpy())
        k_max = math.sqrt(2 * math.log(1 / 1e-8)) / width.min()

        def kspace(k2):
            return np.sum(
                amp[:, None]
                * (2 * np.pi) ** 1.5
                * width[:, None] ** 3
                * np.exp(-0.5 * width[:, None] ** 2 * k2[None, :]),
                axis=0,
            )

        contact = float(amp.sum())
    nd = np.ceil(k_max * np.linalg.norm(cell, axis=1) / (2 * math.pi)).astype(int)
    grids = np.meshgrid(*[np.arange(-m, m + 1) for m in nd], indexing="ij")
    n_grid = np.stack(grids, -1).reshape(-1, 3)
    k = n_grid @ (2 * math.pi * np.linalg.inv(cell)).T
    k2 = (k * k).sum(-1)
    keep = (k2 > 0) & (k2 <= k_max * k_max)
    k, k2 = k[keep], k2[keep]
    phase = k @ coord.T  # (nk, N)
    energy = 0.0
    for c in range(charge.shape[1]):
        rho = (charge[:, c][None, :] * np.exp(1j * phase)).sum(-1)
        energy += 0.5 * (kspace(k2) * np.abs(rho) ** 2).sum() / volume
        energy -= 0.5 * contact * (charge[:, c] ** 2).sum()
    return COULOMB_CONSTANT * energy


def _sr_twin(model):
    """A copy of a periodic model with a zeroed charge net: the SR part alone."""
    twin = BaseModel.deserialize(model.serialize()).to(env.DEVICE).eval()
    for parameter in twin.atomic_model.long_range.charge_net.parameters():
        parameter.detach().zero_()
    return twin


@pytest.mark.parametrize("kernel,channels", [("les", 1), ("sog", 2)])
def test_periodic_energy_matches_kspace_reference(kernel, channels) -> None:
    model = _periodic_model(kernel, channels)
    twin = _sr_twin(model)
    coord, atype, box = _periodic_batch()
    out = model(coord, atype, box=box)
    sr = twin(coord, atype, box=box)
    lr_energy = (out["energy"] - sr["energy"]).detach().cpu().numpy()[:, 0]
    charge = out["latent_charge"].detach().cpu().numpy()
    coord_np = coord.cpu().numpy()
    cells = box.cpu().numpy()
    kernel_obj = model.atomic_model.long_range.kernel
    nlocs = (atype >= 0).sum(dim=1).cpu().numpy()
    start = 0
    for f, n in enumerate(nlocs):
        reference = _kspace_reference(
            coord_np[f, :n], cells[f], charge[f, :n], kernel_obj
        )
        np.testing.assert_allclose(lr_energy[f], reference, rtol=1e-9, atol=1e-9)
        start += n


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_force_virial_match_finite_differences(kernel) -> None:
    model = _periodic_model(kernel)
    coord, atype, box = _periodic_batch((5,), seed=11)
    result = model(coord, atype, box=box, do_atomic_virial=True)
    eps = 1e-5

    def energy(c, b):
        return model(c, atype, box=b)["energy"].item()

    force_fd = torch.zeros_like(result["force"][0])
    for atom in range(5):
        for d in range(3):
            plus, minus = coord.clone(), coord.clone()
            plus[0, atom, d] += eps
            minus[0, atom, d] -= eps
            force_fd[atom, d] = -(energy(plus, box) - energy(minus, box)) / (2 * eps)
    torch.testing.assert_close(result["force"][0], force_fd, rtol=2e-6, atol=2e-6)

    # The virial is -dE/d(strain) at zero strain, for r -> (I + eps) r and
    # B -> B (I + eps)^T (the strain family of the k-space strain leaf).
    virial_fd = torch.zeros(9, dtype=torch.float64, device=env.DEVICE)
    eye = torch.eye(3, dtype=torch.float64, device=env.DEVICE)
    for a in range(3):
        for b in range(3):
            strain = torch.zeros((3, 3), dtype=torch.float64, device=env.DEVICE)
            strain[a, b] = eps
            ep = energy(coord @ (eye + strain).T, box @ (eye + strain).T)
            em = energy(coord @ (eye - strain).T, box @ (eye - strain).T)
            virial_fd[3 * a + b] = -(ep - em) / (2 * eps)
    torch.testing.assert_close(result["virial"][0], virial_fd, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(result["atom_virial"][0].sum(dim=0), result["virial"][0])


def test_periodic_frames_are_independent() -> None:
    model = _periodic_model()
    coord, atype, box = _periodic_batch((5, 4), seed=5)
    batched = model(coord, atype, box=box)
    for f, n in enumerate((5, 4)):
        alone = model(coord[f : f + 1, :n], atype[f : f + 1, :n], box=box[f : f + 1])
        torch.testing.assert_close(batched["energy"][f], alone["energy"][0])
        torch.testing.assert_close(batched["force"][f, :n], alone["force"][0])
        torch.testing.assert_close(batched["virial"][f], alone["virial"][0])


def test_periodic_model_serves_nonperiodic_batches() -> None:
    """A periodic model on a box=None batch takes its direct pair branch."""
    periodic = _periodic_model()
    direct = _charge_model()
    direct.load_state_dict(
        {
            key: value
            for key, value in periodic.state_dict().items()
            if key in direct.state_dict() and not key.startswith("atomic_model.out_")
        },
        strict=False,
    )
    coord, atype = _batch((4, 3), seed=7)
    expected = direct(coord, atype, do_atomic_virial=True)
    actual = periodic(coord, atype, do_atomic_virial=True)
    for key in ("energy", "atom_energy", "force", "virial", "latent_charge"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-11, atol=1e-11)


def test_periodic_serialization_round_trip() -> None:
    model = _periodic_model()
    record = model.serialize()
    assert record["long_range"]["periodic"] is True
    assert record["@version"] == 3
    restored = BaseModel.deserialize(record).to(env.DEVICE).eval()
    coord, atype, box = _periodic_batch((4,), seed=13)
    expected = model(coord, atype, box=box, do_atomic_virial=True)
    actual = restored(coord, atype, box=box, do_atomic_virial=True)
    for key in ("energy", "force", "virial", "latent_charge"):
        torch.testing.assert_close(actual[key], expected[key])


@REQUIRES_SUPPORTED_COMPILE
def test_compiled_periodic_matches_eager() -> None:
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    model = _periodic_model()
    compiled = _CompiledModel(model, _get_model_structure_key(model)).eval()
    coord, atype, box = _periodic_batch((5, 4), seed=19)
    expected = model(coord, atype, box=box, do_atomic_virial=True)
    actual = compiled(coord, atype, box, do_atomic_virial=True)
    for key in (
        "energy",
        "atom_energy",
        "force",
        "virial",
        "atom_virial",
        "latent_charge",
    ):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-9, atol=1e-9)


@REQUIRES_SUPPORTED_COMPILE
def test_compiled_periodic_force_loss_second_order_gradients_match_eager() -> None:
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    eager = _periodic_model(train=True)
    compiled_model = _periodic_model(train=True)
    compiled_model.load_state_dict(eager.state_dict())
    compiled = _CompiledModel(
        compiled_model, _get_model_structure_key(compiled_model)
    ).train()
    coord, atype, box = _periodic_batch((5,), seed=29)
    eager_loss = eager(coord, atype, box=box)["force"].square().sum()
    compiled_loss = compiled(coord, atype, box)["force"].square().sum()
    eager_loss.backward()
    compiled_loss.backward()
    torch.testing.assert_close(compiled_loss, eager_loss)
    eager_parameters = dict(eager.named_parameters())
    for name, parameter in compiled_model.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, eager_parameters[name].grad, rtol=1e-7, atol=1e-7
        )


# ---------------------------------------------------------------------------
# Periodic multipole (k-grid) evaluation
# ---------------------------------------------------------------------------


def _periodic_multipole_model(steps: int = 0, train: bool = False):
    config = _multipole_config(steps=steps)
    config["long_range"]["periodic"] = True
    model = get_model(config).to(env.DEVICE)
    model.train(train)
    return model


def _kspace_multipole_reference(
    coord, cell, charge, dipole, kernel, alpha_pol
) -> float:
    """Structure-factor reference of the periodic multipole energy."""
    import math

    alpha = float(np.exp(kernel.log_alpha.detach().cpu().numpy()).reshape(-1)[0])
    volume = abs(np.linalg.det(cell))
    k_max = 2 * alpha * math.sqrt(math.log(1 / 1e-8))
    nd = np.ceil(k_max * np.linalg.norm(cell, axis=1) / (2 * math.pi)).astype(int)
    grids = np.meshgrid(*[np.arange(-m, m + 1) for m in nd], indexing="ij")
    n_grid = np.stack(grids, -1).reshape(-1, 3)
    k = n_grid @ (2 * math.pi * np.linalg.inv(cell)).T
    k2 = (k * k).sum(-1)
    keep = (k2 > 0) & (k2 <= k_max * k_max)
    k, k2 = k[keep], k2[keep]
    phase = k @ coord.T
    k_dot_p = k @ dipole.T
    struct = ((charge[:, 0] + 1j * k_dot_p) * np.exp(1j * phase)).sum(-1)
    kernel_hat = 4 * np.pi * np.exp(-k2 / (4 * alpha * alpha)) / k2
    return COULOMB_CONSTANT * (
        0.5 * (kernel_hat * np.abs(struct) ** 2).sum() / volume
        - 0.5 * (2 * alpha / np.sqrt(np.pi)) * (charge[:, 0] ** 2).sum()
        + 0.5 * (-4 * alpha**3 / (3 * np.sqrt(np.pi))) * (dipole * dipole).sum()
        + ((dipole * dipole).sum(-1) / (2 * alpha_pol**2)).sum()
    )


def _sr_multipole_twin(model):
    """A copy with the charge/dipole readout zeroed: the short-range part."""
    twin = BaseModel.deserialize(model.serialize()).to(env.DEVICE).eval()
    for name, parameter in twin.atomic_model.long_range.charge_net.named_parameters():
        if "alpha_pol" in name:
            continue
        parameter.detach().zero_()
    return twin


@pytest.mark.parametrize("steps", [0, 1, 2])
def test_periodic_multipole_energy_matches_reference(steps) -> None:
    model = _periodic_multipole_model(steps)
    twin = _sr_multipole_twin(model)
    coord, atype, box = _periodic_batch((5, 4), seed=17)
    out = model(coord, atype, box=box)
    sr = twin(coord, atype, box=box)
    lr_energy = (out["energy"] - sr["energy"]).detach().cpu().numpy()[:, 0]
    charge = out["latent_charge"].detach().cpu().numpy()
    dipole = out["latent_dipole"].detach().cpu().numpy()
    coord_np = coord.cpu().numpy()
    cells = box.cpu().numpy()
    kernel_obj = model.atomic_model.long_range.kernel
    alpha_pol = (
        model.atomic_model.long_range.charge_net.alpha_pol.detach().cpu().numpy()
    )
    for f, n in enumerate((5, 4)):
        reference = _kspace_multipole_reference(
            coord_np[f, :n],
            cells[f],
            charge[f, :n],
            dipole[f, :n],
            kernel_obj,
            alpha_pol[atype[f, :n].cpu().numpy()],
        )
        np.testing.assert_allclose(lr_energy[f], reference, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("steps", [0, 1])
def test_periodic_multipole_force_virial_match_finite_differences(steps) -> None:
    model = _periodic_multipole_model(steps)
    coord, atype, box = _periodic_batch((5,), seed=11)
    result = model(coord, atype, box=box, do_atomic_virial=True)
    eps = 1e-5

    def energy(c, b):
        return model(c, atype, box=b)["energy"].item()

    force_fd = torch.zeros_like(result["force"][0])
    for atom in range(5):
        for d in range(3):
            plus, minus = coord.clone(), coord.clone()
            plus[0, atom, d] += eps
            minus[0, atom, d] -= eps
            force_fd[atom, d] = -(energy(plus, box) - energy(minus, box)) / (2 * eps)
    torch.testing.assert_close(result["force"][0], force_fd, rtol=2e-6, atol=2e-6)

    # The virial is -dE/d(strain) at zero strain, for r -> (I + eps) r and
    # B -> B (I + eps)^T (the strain family of the k-space strain leaf).
    virial_fd = torch.zeros(9, dtype=torch.float64, device=env.DEVICE)
    eye = torch.eye(3, dtype=torch.float64, device=env.DEVICE)
    for a in range(3):
        for b in range(3):
            strain = torch.zeros((3, 3), dtype=torch.float64, device=env.DEVICE)
            strain[a, b] = eps
            ep = energy(coord @ (eye + strain).T, box @ (eye + strain).T)
            em = energy(coord @ (eye - strain).T, box @ (eye - strain).T)
            virial_fd[3 * a + b] = -(ep - em) / (2 * eps)
    torch.testing.assert_close(result["virial"][0], virial_fd, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(result["atom_virial"][0].sum(dim=0), result["virial"][0])


def test_periodic_multipole_frames_independent_and_round_trip() -> None:
    model = _periodic_multipole_model(steps=1)
    restored = BaseModel.deserialize(model.serialize()).to(env.DEVICE).eval()
    coord, atype, box = _periodic_batch((5, 4), seed=5)
    batched = model(coord, atype, box=box)
    for f, n in enumerate((5, 4)):
        alone = restored(coord[f : f + 1, :n], atype[f : f + 1, :n], box=box[f : f + 1])
        torch.testing.assert_close(batched["energy"][f], alone["energy"][0])
        torch.testing.assert_close(batched["force"][f, :n], alone["force"][0])
        torch.testing.assert_close(batched["virial"][f], alone["virial"][0])


@REQUIRES_SUPPORTED_COMPILE
def test_compiled_periodic_multipole_matches_eager() -> None:
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    model = _periodic_multipole_model(steps=1)
    compiled = _CompiledModel(model, _get_model_structure_key(model)).eval()
    coord, atype, box = _periodic_batch((5, 4), seed=19)
    expected = model(coord, atype, box=box, do_atomic_virial=True)
    actual = compiled(coord, atype, box, do_atomic_virial=True)
    for key in (
        "energy",
        "atom_energy",
        "force",
        "virial",
        "atom_virial",
        "latent_charge",
        "latent_dipole",
    ):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-9, atol=1e-9)


@REQUIRES_SUPPORTED_COMPILE
def test_compiled_periodic_multipole_second_order_gradients_match_eager() -> None:
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    eager = _periodic_multipole_model(steps=1, train=True)
    compiled_model = _periodic_multipole_model(steps=1, train=True)
    compiled_model.load_state_dict(eager.state_dict())
    compiled = _CompiledModel(
        compiled_model, _get_model_structure_key(compiled_model)
    ).train()
    coord, atype, box = _periodic_batch((5,), seed=29)
    eager_loss = eager(coord, atype, box=box)["force"].square().sum()
    compiled_loss = compiled(coord, atype, box)["force"].square().sum()
    eager_loss.backward()
    compiled_loss.backward()
    torch.testing.assert_close(compiled_loss, eager_loss)
    eager_parameters = dict(eager.named_parameters())
    for name, parameter in compiled_model.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, eager_parameters[name].grad, rtol=1e-6, atol=1e-6
        )
