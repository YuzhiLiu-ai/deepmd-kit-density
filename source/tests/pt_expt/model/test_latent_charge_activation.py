# SPDX-License-Identifier: LGPL-3.0-or-later
"""Activation configuration and checkpoint compatibility of latent readouts."""

import copy

import pytest
import torch

from deepmd.entrypoints.patch_latent_charge_activation import patch_checkpoint
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.train.wrapper import ModelWrapper
from deepmd.pt_expt.utils import env

from .test_efield import _batch, _model_config


def _config(activation=None, *, charge_activation=None, use_glu=False):
    cfg = _model_config(efield=True, steps=1)
    charge_net = cfg["long_range"]["charge_net"]
    charge_net["use_glu"] = use_glu
    if activation is not None:
        charge_net["activation_function"] = activation
    if charge_activation is not None:
        charge_net["charge_activation_function"] = charge_activation
    return cfg


@pytest.mark.parametrize("use_glu", [False, True])
def test_readout_respects_activation_function(use_glu) -> None:
    cfg = _config("silu", use_glu=use_glu)
    model = get_model(cfg).to(env.DEVICE).eval()
    charge_net = model.atomic_model.long_range.charge_net
    assert charge_net.activation_function == "silu"
    assert charge_net.charge_activation_function == "silu"
    assert charge_net.nets[()].activation_function == "silu"

    tanh_model = get_model(_config("tanh", use_glu=use_glu)).to(env.DEVICE).eval()
    coord, atype, cs = _batch()
    out_silu = model(coord, atype, charge_spin=cs)
    out_tanh = tanh_model(coord, atype, charge_spin=cs)
    assert not torch.allclose(out_silu["latent_charge"], out_tanh["latent_charge"])


def test_split_charge_and_dipole_activations() -> None:
    model = get_model(_config("silu", charge_activation="tanh", use_glu=True)).to(
        env.DEVICE
    )
    charge_net = model.atomic_model.long_range.charge_net
    assert charge_net.nets[()].activation_function == "tanh"
    assert charge_net.polar_update_net.activation_function == "silu"


def test_serialization_round_trip_and_legacy_default() -> None:
    charge_net = get_model(_config("silu")).atomic_model.long_range.charge_net
    record = charge_net.serialize()
    assert record["activation_function"] == "silu"
    assert record["charge_activation_function"] == "silu"
    restored = type(charge_net).deserialize(record)
    assert restored.charge_activation_function == "silu"
    assert restored.nets[()].activation_function == "silu"

    legacy = copy.deepcopy(record)
    legacy.pop("charge_activation_function")
    # The old hard-coded-tanh record could still declare silu for dipole/LEF,
    # but its serialized charge-net layers themselves were tanh.
    tanh_record = get_model(
        _config("tanh")
    ).atomic_model.long_range.charge_net.serialize()
    legacy["nets"] = tanh_record["nets"]
    restored_legacy = type(charge_net).deserialize(legacy)
    assert restored_legacy.activation_function == "silu"
    assert restored_legacy.charge_activation_function == "tanh"
    assert restored_legacy.nets[()].activation_function == "tanh"


@pytest.mark.parametrize("activation", [None, "silu"])
def test_wrapper_stores_effective_activations(activation) -> None:
    cfg = _config(activation)
    model = get_model(cfg).to(env.DEVICE)
    wrapper = ModelWrapper(model, model_params=cfg)
    expected = "tanh" if activation is None else activation
    charge_cfg = wrapper.model_params["long_range"]["charge_net"]
    assert charge_cfg["activation_function"] == expected
    assert charge_cfg["charge_activation_function"] == expected
    state_cfg = wrapper.state_dict()["_extra_state"]["model_params"]
    assert (
        state_cfg["long_range"]["charge_net"]["charge_activation_function"] == expected
    )


def test_patch_checkpoint_script_dry_run_write_and_forward(tmp_path) -> None:
    cfg = _config("silu")
    model = get_model(cfg).to(env.DEVICE).eval()
    wrapper = ModelWrapper(model)
    state = wrapper.state_dict()
    legacy_cfg = copy.deepcopy(cfg)
    legacy_charge = legacy_cfg["long_range"]["charge_net"]
    legacy_charge.pop("activation_function")
    legacy_charge.pop("charge_activation_function", None)
    state["_extra_state"]["model_params"] = legacy_cfg
    checkpoint = tmp_path / "legacy_silu.pt"
    torch.save({"model": state}, checkpoint)

    assert patch_checkpoint(checkpoint, "silu") == 1
    dry_run = torch.load(checkpoint, weights_only=False)
    dry_charge = dry_run["model"]["_extra_state"]["model_params"]["long_range"][
        "charge_net"
    ]
    assert "activation_function" not in dry_charge
    assert "charge_activation_function" not in dry_charge

    assert patch_checkpoint(checkpoint, "silu", write=True) == 1
    assert (tmp_path / "legacy_silu.pt.bak").exists()
    patched = torch.load(checkpoint, weights_only=False)
    patched_cfg = patched["model"]["_extra_state"]["model_params"]
    patched_charge = patched_cfg["long_range"]["charge_net"]
    assert patched_charge["activation_function"] == "silu"
    assert patched_charge["charge_activation_function"] == "silu"

    restored_model = get_model(patched_cfg).to(env.DEVICE).eval()
    restored_wrapper = ModelWrapper(restored_model)
    restored_wrapper.load_state_dict(patched["model"])
    coord, atype, cs = _batch()
    expected = model(coord, atype, charge_spin=cs)
    actual = restored_model(coord, atype, charge_spin=cs)
    torch.testing.assert_close(actual["energy"], expected["energy"])
    torch.testing.assert_close(actual["force"], expected["force"])
    torch.testing.assert_close(actual["latent_charge"], expected["latent_charge"])
