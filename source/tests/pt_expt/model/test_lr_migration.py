# SPDX-License-Identifier: LGPL-3.0-or-later
"""Migration of legacy dpa4c_lr configs and records to the long-range plugin."""

import numpy as np
import pytest
import torch

from copy import (
    deepcopy,
)

from deepmd.dpmodel.fitting.dpa4c_lr import (
    SOG_DEFAULT_B,
    SOG_DEFAULT_M,
    SOG_DEFAULT_SIGMA,
)
from deepmd.dpmodel.long_range.migration import (
    migrate_dpa4_lr_config,
    migrate_dpa4_lr_record,
    migrate_dpa4c_lr_config,
    migrate_dpa4c_lr_record,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.utils import (
    env,
)

from .test_long_range import (
    _batch,
    _charge_config,
)


def _legacy_config(
    lr_kernel: str,
    steps: int = 0,
    use_glu: bool = False,
    constraint: bool = True,
    use_self_pol: bool = True,
    bias_atom_q: list | None = None,
) -> dict:
    config = {"type": "dpa4c_lr", **_charge_config(with_long_range=False)}
    config["fitting_net"] = {
        **config["fitting_net"],
        "type": "dpa4c_lr",
        "dim_out_lr": 1,
        "neuron_lr": [8, 8],
        "use_charge_constraint": constraint,
        "lr_kernel": lr_kernel,
        "les_alpha": 0.9,
        "use_glu": use_glu,
        "lr_update_steps": steps,
        "use_self_pol": use_self_pol,
    }
    if bias_atom_q is not None:
        config["fitting_net"]["bias_atom_q"] = bias_atom_q
    if lr_kernel == "sog":
        config["fitting_net"].pop("les_alpha")
    return config


_CASES = [
    ("les", 0, False),
    ("les", 0, True),
    ("sog", 0, False),
    ("dipole", 0, False),
    ("dipole", 1, False),
    ("dipole", 2, True),
]


@pytest.mark.parametrize("lr_kernel,steps,use_glu", _CASES)
def test_record_migration_matches_legacy_model(lr_kernel, steps, use_glu) -> None:
    """A migrated checkpoint record reproduces the legacy model exactly."""
    legacy = get_model(_legacy_config(lr_kernel, steps, use_glu)).to(env.DEVICE).eval()
    record = legacy.serialize()
    assert record["fitting"]["type"] == "dpa4c_lr"
    migrated_record = migrate_dpa4c_lr_record(record)
    assert migrated_record["@version"] == 3
    assert migrated_record["fitting"]["type"] == "dpa4_ener"
    migrated = BaseModel.deserialize(migrated_record).to(env.DEVICE).eval()

    coord, atype = _batch((4, 3), seed=23)
    expected = legacy(coord, atype, do_atomic_virial=True)
    actual = migrated(coord, atype, do_atomic_virial=True)
    keys = ["energy", "atom_energy", "force", "virial", "atom_virial", "latent_charge"]
    if lr_kernel == "dipole":
        keys.append("latent_dipole")
    for key in keys:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("lr_kernel,steps,use_glu", _CASES)
def test_config_migration_matches_record_migration(lr_kernel, steps, use_glu) -> None:
    """A migrated training config builds the model the record migration gives."""
    legacy = get_model(_legacy_config(lr_kernel, steps, use_glu)).to(env.DEVICE).eval()
    via_record = BaseModel.deserialize(migrate_dpa4c_lr_record(legacy.serialize())).to(
        env.DEVICE
    )
    via_config = get_model(
        migrate_dpa4c_lr_config(_legacy_config(lr_kernel, steps, use_glu))
    ).to(env.DEVICE)
    via_config.load_state_dict(via_record.state_dict(), strict=True)
    coord, atype = _batch((4, 3), seed=23)
    expected = legacy(coord, atype)
    actual = via_config(coord, atype)
    for key in ("energy", "force", "latent_charge"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-11, atol=1e-11)


def test_config_migration_maps_fields() -> None:
    migrated = migrate_dpa4c_lr_config(_legacy_config("dipole", steps=2))
    assert "type" not in migrated
    assert migrated["fitting_net"]["type"] == "dpa4_ener"
    for key in (
        "lr_kernel",
        "neuron_lr",
        "les_alpha",
        "use_charge_constraint",
        "lr_update_steps",
    ):
        assert key not in migrated["fitting_net"]
    long_range = migrated["long_range"]
    assert long_range["type"] == "latent_multipole"
    assert long_range["kernel"] == {"type": "les", "alpha": 0.9}
    assert long_range["charge_constraint"] == {
        "source": "neutral",
        "channel": 0,
        "fparam_index": 0,
    }
    assert long_range["lr_update_steps"] == 2
    assert long_range["use_self_pol"] is True


def test_sog_migration_materializes_legacy_default_amplitudes() -> None:
    """The legacy SOG default is 4 pi ln b per Gaussian, not the plugin default."""
    migrated = migrate_dpa4c_lr_config(_legacy_config("sog"))
    kernel = migrated["long_range"]["kernel"]
    assert kernel["type"] == "sog"
    assert len(kernel["width"]) == SOG_DEFAULT_M
    np.testing.assert_allclose(
        kernel["width"],
        SOG_DEFAULT_SIGMA * np.power(SOG_DEFAULT_B, np.arange(SOG_DEFAULT_M)),
    )
    np.testing.assert_allclose(
        kernel["amplitude"],
        np.full(SOG_DEFAULT_M, 4.0 * np.pi * np.log(SOG_DEFAULT_B)),
    )


def test_migration_preserves_nonzero_charge_bias() -> None:
    """The legacy 3*tanh(bias/3) squash survives with a trained nonzero bias."""
    legacy = (
        get_model(_legacy_config("les", bias_atom_q=[0.3, -0.2])).to(env.DEVICE).eval()
    )
    migrated = (
        BaseModel.deserialize(migrate_dpa4c_lr_record(legacy.serialize()))
        .to(env.DEVICE)
        .eval()
    )
    coord, atype = _batch((4, 3), seed=23)
    expected = legacy(coord, atype)
    actual = migrated(coord, atype)
    assert expected["latent_charge"].abs().max() > 0
    torch.testing.assert_close(
        actual["latent_charge"], expected["latent_charge"], rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(actual["energy"], expected["energy"])


def test_migration_rejects_unsupported_models() -> None:
    with pytest.raises(NotImplementedError, match="DPA4"):
        migrate_dpa4c_lr_config({"type": "dpa4_lr"})
    with pytest.raises(ValueError, match="legacy long-range"):
        migrate_dpa4c_lr_config({"type": "standard"})
    record = get_model(_legacy_config("les")).serialize()
    record["fitting"]["case_film_embd"] = True
    with pytest.raises(NotImplementedError, match="case_film_embd"):
        migrate_dpa4c_lr_record(record)


# ---------------------------------------------------------------------------
# Legacy DPA4 (pt ``dpa4_lr`` / ``sog_energy``) migration
# ---------------------------------------------------------------------------


def _legacy_dpa4_config(
    lr_kernel: str,
    steps: int = 0,
    constraint: bool = False,
    use_glu: bool = False,
    bias_atom_q: list | None = None,
    numb_fparam: int = 0,
) -> dict:
    fitting = {
        "type": "sog_energy",
        "neuron": [16],
        "neuron_lr": [8],
        "precision": "float64",
        "seed": 7,
        "lr_kernel": lr_kernel,
        "use_charge_constraint": constraint,
        "use_glu": use_glu,
        "lr_update_steps": steps,
    }
    if lr_kernel in ("les", "dipole"):
        fitting["les_alpha"] = 0.9
    if bias_atom_q is not None:
        fitting["bias_atom_q"] = bias_atom_q
    if numb_fparam:
        fitting["numb_fparam"] = numb_fparam
    return {
        "type": "dpa4_lr",
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "dpa4",
            "sel": 6,
            "rcut": 4.0,
            "channels": 8,
            "n_radial": 4,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "so2_layers": 1,
            "n_atten_head": 1,
            "use_amp": False,
            "precision": "float64",
            "seed": 7,
        },
        "fitting_net": fitting,
    }


def _legacy_dpa4_model(config: dict):
    """A randomized legacy pt model (the descriptor zero-init is degenerate)."""
    from deepmd.pt.model.model import (
        get_model as pt_get_model,
    )

    model = pt_get_model(config).to(env.DEVICE).eval()
    explicit_bias = config["fitting_net"].get("bias_atom_q")
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
        fitting = model.atomic_model.fitting_net
        for name in ("bandwidth", "les_alpha", "alpha_pol"):
            tensor = getattr(fitting, name, None)
            if tensor is not None:
                tensor.abs_().clamp_(min=0.2)
        if explicit_bias is not None:
            # Keep the configured per-type bias instead of the random fill.
            fitting.bias_atom_q.copy_(
                torch.as_tensor(
                    explicit_bias, dtype=fitting.prec, device=env.DEVICE
                ).view(-1, 1)
            )
    return model


_DPA4_CASES = [
    ("les", 0, False, False),
    ("les", 0, True, True),
    ("sog", 0, False, False),
    ("dipole", 0, False, False),
    ("dipole", 1, False, False),
    ("dipole", 2, True, False),
    ("dipole", 2, False, True),
]


@pytest.mark.parametrize("lr_kernel,steps,constraint,use_glu", _DPA4_CASES)
def test_dpa4_record_migration_matches_legacy_model(
    lr_kernel, steps, constraint, use_glu
) -> None:
    """A migrated legacy DPA4 checkpoint reproduces the legacy model exactly."""
    legacy = _legacy_dpa4_model(
        _legacy_dpa4_config(lr_kernel, steps, constraint, use_glu)
    )
    record = legacy.serialize()
    assert record["type"] == "dpa4_lr"
    assert record["atomic_model"]["fitting"]["type"] == "sog_energy"
    migrated_record = migrate_dpa4_lr_record(record)
    assert migrated_record["@version"] == 3
    assert migrated_record["fitting"]["type"] == "dpa4_ener"
    migrated = BaseModel.deserialize(migrated_record).to(env.DEVICE).eval()

    # The legacy pt readout constraint averages over *all* padded slots, so a
    # padded batch pollutes the shift; the plugin constrains the real atoms
    # (its flattened graph has no padding slots at all).  Exactness of the
    # constrained cases is therefore checked on equal-size frames.
    coord, atype = _batch((5, 5) if constraint else (5, 3), seed=29)
    expected = legacy(coord, atype, do_atomic_virial=True)
    actual = migrated(coord, atype, do_atomic_virial=True)
    keys = ["energy", "atom_energy", "force", "virial", "latent_charge"]
    if lr_kernel == "dipole":
        keys.append("latent_dipole")
    for key in keys:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("lr_kernel,steps,constraint,use_glu", _DPA4_CASES)
def test_dpa4_config_migration_matches_record_migration(
    lr_kernel, steps, constraint, use_glu
) -> None:
    """A migrated DPA4 training config builds the model the record migration gives."""
    config = _legacy_dpa4_config(lr_kernel, steps, constraint, use_glu)
    legacy = _legacy_dpa4_model(config)
    via_record = BaseModel.deserialize(migrate_dpa4_lr_record(legacy.serialize())).to(
        env.DEVICE
    )
    via_config = get_model(migrate_dpa4_lr_config(config)).to(env.DEVICE)
    via_config.load_state_dict(via_record.state_dict(), strict=True)
    coord, atype = _batch((5, 5) if constraint else (5, 3), seed=29)
    expected = legacy(coord, atype)
    actual = via_config(coord, atype)
    for key in ("energy", "force", "latent_charge"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-11, atol=1e-11)


def test_dpa4_record_migration_evaluates_type_embed_bias() -> None:
    """The legacy TypeEmbedNet ``bias_atom_q`` becomes a static per-type array."""
    legacy = _legacy_dpa4_model(_legacy_dpa4_config("les", constraint=True))
    migrated_record = migrate_dpa4_lr_record(legacy.serialize())
    bias = migrated_record["long_range"]["charge_net"]["@variables"]["bias_atom_e"]
    assert bias.shape == (2, 1)
    assert np.any(np.abs(bias) > 0)
    legacy_variables = legacy.serialize()["atomic_model"]["fitting"]["@variables"]
    matrix = legacy_variables[
        "bias_atom_q_tebd.embedding.embedding_net.layers.0.matrix"
    ]
    np.testing.assert_allclose(bias, matrix)
    migrated = BaseModel.deserialize(migrated_record).to(env.DEVICE).eval()
    # Equal-size frames: the legacy constraint mean includes padded slots.
    coord, atype = _batch((5, 5), seed=29)
    torch.testing.assert_close(
        migrated(coord, atype)["latent_charge"],
        legacy(coord, atype)["latent_charge"],
        rtol=1e-11,
        atol=1e-11,
    )


def test_dpa4_record_migration_preserves_explicit_bias() -> None:
    legacy = _legacy_dpa4_model(_legacy_dpa4_config("les", bias_atom_q=[0.3, -0.2]))
    migrated_record = migrate_dpa4_lr_record(legacy.serialize())
    bias = migrated_record["long_range"]["charge_net"]["@variables"]["bias_atom_e"]
    np.testing.assert_allclose(bias, [[0.3], [-0.2]])
    migrated = BaseModel.deserialize(migrated_record).to(env.DEVICE).eval()
    coord, atype = _batch((5, 3), seed=29)
    torch.testing.assert_close(
        migrated(coord, atype)["latent_charge"],
        legacy(coord, atype)["latent_charge"],
        rtol=1e-11,
        atol=1e-11,
    )


def test_dpa4_migrated_dipole_is_equivariant() -> None:
    """The migrated SeZM dipole readout keeps the corrected Cartesian decode."""
    legacy = _legacy_dpa4_model(_legacy_dpa4_config("dipole", steps=1))
    migrated = (
        BaseModel.deserialize(migrate_dpa4_lr_record(legacy.serialize()))
        .to(env.DEVICE)
        .eval()
    )
    coord, atype = _batch((5,), seed=31)
    theta = 0.7
    rotation = torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
        device=env.DEVICE,
    )
    base = migrated(coord, atype)
    rotated = migrated(coord @ rotation.T, atype)
    torch.testing.assert_close(
        rotated["energy"], base["energy"], rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        rotated["latent_dipole"],
        base["latent_dipole"] @ rotation.T,
        rtol=1e-9,
        atol=1e-9,
    )


def test_dpa4_config_migration_maps_fields() -> None:
    migrated = migrate_dpa4_lr_config(
        _legacy_dpa4_config("dipole", steps=2, constraint=True)
    )
    assert "type" not in migrated
    assert migrated["fitting_net"]["type"] == "dpa4_ener"
    for key in (
        "lr_kernel",
        "neuron_lr",
        "les_alpha",
        "use_charge_constraint",
        "lr_update_steps",
    ):
        assert key not in migrated["fitting_net"]
    long_range = migrated["long_range"]
    assert long_range["type"] == "latent_multipole"
    assert long_range["kernel"] == {"type": "les", "alpha": 0.9}
    assert long_range["charge_constraint"] == {
        "source": "neutral",
        "channel": 0,
        "fparam_index": 0,
    }
    assert long_range["lr_update_steps"] == 2
    assert long_range["periodic"] is False


def test_dpa4_config_migration_constraint_prefers_fparam() -> None:
    """The legacy DPA4 fitting checks the frame parameter before charge_spin."""
    migrated = migrate_dpa4_lr_config(
        _legacy_dpa4_config("les", constraint=True, numb_fparam=1)
    )
    assert migrated["long_range"]["charge_constraint"]["source"] == "fparam"


def test_dpa4_sog_migration_self_interaction_semantics() -> None:
    """The legacy DPA4 SOG keeps the self-interaction term unless removed."""
    migrated = migrate_dpa4_lr_config(_legacy_dpa4_config("sog"))
    assert migrated["long_range"]["self_interaction"] is True
    config = _legacy_dpa4_config("sog")
    config["fitting_net"]["remove_self_interaction"] = True
    migrated = migrate_dpa4_lr_config(config)
    assert "self_interaction" not in migrated["long_range"]


def test_dpa4_migration_rejects_unsupported_records() -> None:
    legacy = _legacy_dpa4_model(_legacy_dpa4_config("les"))
    record = legacy.serialize()
    with pytest.raises(ValueError, match="legacy DPA4"):
        migrate_dpa4_lr_record({"type": "dpa4c_lr"})
    bad = deepcopy(record)
    bad["atomic_model"]["fitting"]["type"] = "dpa4_ener"
    with pytest.raises(ValueError, match="sog_energy"):
        migrate_dpa4_lr_record(bad)
    bad = deepcopy(record)
    bad["atomic_model"]["dens_fitting"] = {"type": "dens"}
    with pytest.raises(NotImplementedError, match="dens"):
        migrate_dpa4_lr_record(bad)
    bad = deepcopy(record)
    bad["atomic_model"]["active_mode"] = "dens"
    with pytest.raises(NotImplementedError, match="active_mode"):
        migrate_dpa4_lr_record(bad)
    with pytest.raises(ValueError, match="legacy DPA4"):
        migrate_dpa4_lr_config({"type": "dpa4c_lr"})
    bad_config = _legacy_dpa4_config("les")
    bad_config["fitting_net"]["type"] = "dpa4_ener"
    with pytest.raises(ValueError, match="sog_energy"):
        migrate_dpa4_lr_config(bad_config)
