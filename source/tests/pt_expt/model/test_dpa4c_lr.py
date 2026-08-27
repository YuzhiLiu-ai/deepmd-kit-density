# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the DPA4C-LR non-periodic latent-charge model."""

import numpy as np
import pytest
import torch

from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils import (
    env,
)

from ..compile_utils import (
    REQUIRES_SUPPORTED_COMPILE,
)


def _config() -> dict:
    """Minimal float64 DPA4C-LR config for unit tests."""
    return {
        "type": "dpa4c_lr",
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
            "type": "dpa4c_lr",
            "neuron": [16, 16],
            "precision": "float64",
            "seed": 19,
            "dim_out_lr": 1,
            "neuron_lr": [8, 8],
            "use_charge_constraint": True,
            "lr_kernel": "les",
            "les_alpha": 1.0,
        },
    }


def _build_model(train: bool = False) -> torch.nn.Module:
    """Build a small DPA4C-LR model on the default test device."""
    model = get_model(_config()).to(env.DEVICE)
    if train:
        model.train()
    else:
        model.eval()
    return model


def _make_batch(
    nlocs: tuple[int, ...],
    seed: int = 0,
    box: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded rectangular batch."""
    rng = np.random.default_rng(seed)
    pad_nloc = max(nlocs)
    padded_coord = np.zeros((len(nlocs), pad_nloc, 3))
    padded_atype = np.full((len(nlocs), pad_nloc), -1, dtype=np.int64)
    for index, nloc in enumerate(nlocs):
        padded_coord[index, :nloc] = rng.uniform(0.0, 6.0, (nloc, 3))
        padded_atype[index, :nloc] = rng.integers(0, 2, nloc)
    return padded_coord, padded_atype


def test_energy_force_are_finite_and_phantom_atoms_are_force_free() -> None:
    """A padded batch produces finite outputs and zero force on phantom atoms."""
    torch.manual_seed(1234)
    model = _build_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    coord, atype = _make_batch((4, 7, 3))
    result = model(
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(atype, dtype=torch.long, device=env.DEVICE),
    )
    for key in ("energy", "force", "atom_energy"):
        assert torch.isfinite(result[key]).all(), key
    phantom = torch.tensor(atype, device=env.DEVICE) < 0
    assert bool(phantom.any()), "fixture must exercise padding"
    assert bool(torch.all(result["force"][phantom] == 0.0))


def test_frames_do_not_interact_across_batch() -> None:
    """The dense LR kernel must not couple atoms in different frames."""
    torch.manual_seed(1234)
    model = _build_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    nlocs = (4, 7, 3)
    coord, atype = _make_batch(nlocs)

    def run(
        batch_coord: np.ndarray, batch_atype: np.ndarray
    ) -> dict[str, torch.Tensor]:
        return model(
            torch.tensor(batch_coord, dtype=torch.float64, device=env.DEVICE),
            torch.tensor(batch_atype, dtype=torch.long, device=env.DEVICE),
        )

    batched = run(coord, atype)
    for index, nloc in enumerate(nlocs):
        alone = run(
            coord[index : index + 1, :nloc],
            atype[index : index + 1, :nloc],
        )
        torch.testing.assert_close(
            batched["energy"].reshape(-1)[index],
            alone["energy"].reshape(-1)[0],
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            batched["force"][index, :nloc],
            alone["force"][0],
            atol=1e-12,
            rtol=1e-12,
        )


def test_force_matches_finite_difference() -> None:
    """The public model force agrees with a central finite-difference stencil."""
    torch.manual_seed(1234)
    model = _build_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    rng = np.random.default_rng(1)
    coord = rng.uniform(0.0, 6.0, (1, 5, 3))
    atype = rng.integers(0, 2, (1, 5)).astype(np.int64)
    atype_t = torch.tensor(atype, dtype=torch.long, device=env.DEVICE)

    def energy(x: torch.Tensor) -> float:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            ret = model(x, atype_t)
        return ret["energy"].item()

    coord_t = torch.tensor(coord, dtype=torch.float64, device=env.DEVICE)
    force = model(coord_t, atype_t)["force"].detach().cpu().numpy().reshape(-1, 3)

    eps = 1e-5
    force_fd = np.zeros_like(force)
    for i in range(coord.shape[1]):
        for d in range(3):
            cp = coord.copy()
            cp[0, i, d] += eps
            cm = coord.copy()
            cm[0, i, d] -= eps
            Ep = energy(torch.tensor(cp, dtype=torch.float64, device=env.DEVICE))
            Em = energy(torch.tensor(cm, dtype=torch.float64, device=env.DEVICE))
            force_fd[i, d] = -(Ep - Em) / (2 * eps)

    np.testing.assert_allclose(force, force_fd, atol=1e-6, rtol=1e-6)


def test_ragged_matches_padded() -> None:
    """Ragged and padded layouts must describe the same batch."""
    torch.manual_seed(1234)
    model = _build_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    nlocs = (4, 7, 3)
    coord, atype = _make_batch(nlocs)

    padded = model(
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(atype, dtype=torch.long, device=env.DEVICE),
    )
    flat_coord = torch.tensor(
        np.concatenate([coord[i, :nloc] for i, nloc in enumerate(nlocs)]),
        dtype=torch.float64,
        device=env.DEVICE,
    )
    flat_atype = torch.tensor(
        np.concatenate([atype[i, :nloc] for i, nloc in enumerate(nlocs)]),
        dtype=torch.long,
        device=env.DEVICE,
    )
    n_node = torch.tensor(nlocs, dtype=torch.long, device=env.DEVICE)
    ragged = model.forward_ragged(flat_coord, flat_atype, n_node)

    torch.testing.assert_close(padded["energy"], ragged["energy"])
    offset = 0
    for index, nloc in enumerate(nlocs):
        torch.testing.assert_close(
            padded["force"][index, :nloc],
            ragged["force"][offset : offset + nloc],
            atol=1e-12,
            rtol=1e-12,
        )
        offset += nloc
    assert offset == ragged["force"].shape[0]


def test_charge_constraint_zero_target_preserves_total_charge() -> None:
    """With no charge_spin or fparam, constraint drives total latent charge to 0."""
    torch.manual_seed(1234)
    model = _build_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    fitting = model.get_fitting_net()
    nf, nloc = 2, 5
    q = torch.randn(nf * nloc, 1, dtype=torch.float64, device=env.DEVICE)
    n_node = torch.tensor([nloc] * nf, dtype=torch.long, device=env.DEVICE)
    from deepmd.dpmodel.utils.neighbor_graph import (
        frame_id_from_n_node,
    )

    frame_id = frame_id_from_n_node(n_node, n_total=q.shape[0])
    q_out = model._constrain_charges_flat(
        q, frame_id, n_node, fparam=None, charge_spin=None
    )
    for f in range(nf):
        assert torch.isclose(
            q_out[frame_id == f, 0].sum(),
            torch.zeros((), dtype=q_out.dtype, device=q_out.device),
        )


def test_periodic_box_is_rejected() -> None:
    """DPA4C-LR must reject non-zero simulation cells."""
    model = _build_model()
    coord = torch.zeros((1, 3, 3), dtype=torch.float64, device=env.DEVICE)
    atype = torch.zeros((1, 3), dtype=torch.long, device=env.DEVICE)
    box = torch.eye(3, dtype=torch.float64, device=env.DEVICE).reshape(1, 9)
    with pytest.raises(NotImplementedError):
        model(coord, atype, box=box)


def test_graph_training_gradient_reaches_lr_parameters() -> None:
    """Backprop through the graph lower reaches descriptor and LR fitting params."""
    torch.manual_seed(1234)
    model = _build_model(train=True)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    coord, atype = _make_batch((4, 5))
    result = model(
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(atype, dtype=torch.long, device=env.DEVICE),
    )
    loss = result["energy"].square().mean() + result["force"].square().mean()
    loss.backward()

    fitting = model.get_fitting_net()
    for name, parameter in fitting.named_parameters():
        assert parameter.grad is not None, f"fitting parameter {name} has no grad"
        assert torch.isfinite(parameter.grad).all(), f"{name} grad non-finite"
    descriptor = model.get_descriptor()
    grads = {name: parameter.grad for name, parameter in descriptor.named_parameters()}
    assert grads, "descriptor has parameters"
    for name, gradient in grads.items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


@REQUIRES_SUPPORTED_COMPILE
def test_compiled_lower_matches_eager() -> None:
    """The compiled graph lower must reproduce the eager energy and force."""
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    torch.manual_seed(0)
    model = _build_model()
    compiled = _CompiledModel(model, _get_model_structure_key(model))
    compiled.eval()

    nlocs = (4, 7, 3)
    coord, atype = _make_batch(nlocs)
    coord_t = torch.tensor(coord, dtype=torch.float64, device=env.DEVICE)
    atype_t = torch.tensor(atype, dtype=torch.long, device=env.DEVICE)

    expected = model(coord_t, atype_t)
    got = compiled(coord_t, atype_t, None)

    torch.testing.assert_close(got["energy"], expected["energy"])
    torch.testing.assert_close(got["force"], expected["force"])
