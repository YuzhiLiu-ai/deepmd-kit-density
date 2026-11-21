# SPDX-License-Identifier: LGPL-3.0-or-later
"""Migration of legacy ``dpa4c_lr`` models to the long-range plugin layout.

Two entry points cover the two artifacts users hold:

- :func:`migrate_dpa4c_lr_config` translates the ``model`` section of a
  training config, so an old ``input.json`` trains the plugin architecture;
- :func:`migrate_dpa4c_lr_record` translates a serialized model record (the
  ``model`` field of a checkpoint), so an old checkpoint deserializes into the
  plugin architecture with identical outputs.

Both translations preserve the legacy numerical semantics: the GLU/MLP
readout, the ``3*tanh(bias/3)`` charge-bias squash, the SOG default
amplitudes, the charge-constraint priority chain and the self-polarization
and update settings.  ``dpa4_lr``/``sezm_sog`` (DPA4) models are rejected
until the plugin serves that descriptor.
"""

from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel import (
    PRECISION_DICT,
)
from deepmd.dpmodel.fitting.dpa4c_lr import (
    SOG_DEFAULT_B,
    SOG_DEFAULT_M,
    SOG_DEFAULT_SIGMA,
)

# Constructor/config keys of DPA4CLRFitting that move to the plugin.
_LR_FITTING_KEYS = (
    "dim_out_lr",
    "neuron_lr",
    "bias_atom_q",
    "les_alpha",
    "amp",
    "bandwidth",
    "b",
    "sigma",
    "M",
    "use_charge_constraint",
    "lr_kernel",
    "charge_bound",
    "use_glu",
    "embedding_width",
    "alpha_pol",
    "use_self_pol",
    "lr_update_steps",
    "lr_update_neuron",
    "lr_update_damping",
)

# Keys of the serialized DPA4CLRFitting record that move to the plugin
# (on top of the constructor keys above).
_LR_RECORD_KEYS = ("nets_lr", "dipole_weight_net", "polar_update_net")

# @variables of the legacy fitting record that belong to the plugin.
_LR_VARIABLES = ("bias_atom_q", "les_alpha", "alpha_pol", "amp", "bandwidth")


def _constraint_of(
    lr: dict,
    fitting: dict,
    descriptor: dict,
    charge_spin_first: bool = True,
) -> dict | None:
    """Resolve the legacy runtime constraint chain to one config-time source.

    The legacy dpa4c model read the total charge from ``charge_spin`` when the
    call carried it, then from the first frame parameter, and fell back to
    zero; the legacy DPA4 (``sog_energy``) fitting checks the frame parameter
    first.  The plugin fixes the source at build time: ``charge_spin`` when
    the descriptor takes the condition, ``fparam`` when the fitting reads
    frame parameters, neutral otherwise, ordered by ``charge_spin_first``.
    """
    if not lr.get("use_charge_constraint", False):
        return None
    has_spin = bool(descriptor.get("add_chg_spin_ebd"))
    has_fparam = fitting.get("numb_fparam", 0) > 0
    if has_spin and has_fparam:
        source = "charge_spin" if charge_spin_first else "fparam"
    elif has_spin:
        source = "charge_spin"
    elif has_fparam:
        source = "fparam"
    else:
        source = "neutral"
    return {"source": source, "channel": 0, "fparam_index": 0}


def _sog_kernel_config(lr: dict) -> dict:
    """SOG kernel config, materializing the legacy default amplitudes.

    The legacy fitting initializes ``amp = 4 pi ln b`` shared by every
    Gaussian, which differs from the plugin kernel's own 1/r-discretizing
    default; a migrated model must therefore carry the legacy values
    explicitly.
    """
    width = lr.get("bandwidth")
    if width is None:
        b = float(lr.get("b", SOG_DEFAULT_B))
        sigma = float(lr.get("sigma", SOG_DEFAULT_SIGMA))
        m = int(lr.get("M", SOG_DEFAULT_M))
        width = (sigma * np.power(b, np.arange(m))).tolist()
        b_for_amp = b
    else:
        width = [float(w) for w in width]
        b_for_amp = float(lr.get("b", SOG_DEFAULT_B))
    amp = lr.get("amp")
    if amp is None:
        amp = [4.0 * np.pi * np.log(b_for_amp)] * len(width)
    elif np.isscalar(amp):
        amp = [float(amp)] * len(width)
    return {"type": "sog", "amplitude": list(amp), "width": width}


def _kernel_config(lr_kernel: str, lr: dict) -> dict:
    if lr_kernel in ("les", "dipole"):
        return {"type": "les", "alpha": float(lr.get("les_alpha", 1.0))}
    return _sog_kernel_config(lr)


def _charge_net_config(lr: dict, fitting: dict) -> dict:
    charge_net: dict[str, Any] = {
        "neuron": list(lr.get("neuron_lr") or [64, 64]),
        "use_glu": bool(lr.get("use_glu", True)),
        "bias_out": bool(fitting.get("bias_out", False)),
    }
    if lr.get("charge_bound") is not None:
        charge_net["charge_bound"] = float(lr["charge_bound"])
    if lr.get("bias_atom_q") is not None:
        charge_net["bias_atom"] = lr["bias_atom_q"]
    return charge_net


def _long_range_config(
    lr_kernel: str, lr: dict, fitting: dict, descriptor: dict
) -> dict:
    """The model-level ``long_range`` section of a migrated config."""
    config: dict[str, Any] = {
        "type": "latent_multipole" if lr_kernel == "dipole" else "latent_charge",
        "channels": int(lr.get("dim_out_lr", 1)),
        "charge_net": _charge_net_config(lr, fitting),
        "kernel": _kernel_config(lr_kernel, lr),
        "periodic": False,
    }
    constraint = _constraint_of(lr, fitting, descriptor)
    if constraint is not None:
        config["charge_constraint"] = constraint
    if lr_kernel == "dipole":
        config["embedding_width"] = int(lr.get("embedding_width", 0))
        if lr.get("alpha_pol") is not None:
            config["alpha_pol"] = lr["alpha_pol"]
        config["use_self_pol"] = bool(lr.get("use_self_pol", True))
        config["lr_update_steps"] = int(lr.get("lr_update_steps", 0))
        if lr.get("lr_update_neuron") is not None:
            config["lr_update_neuron"] = list(lr["lr_update_neuron"])
        config["lr_update_damping"] = float(lr.get("lr_update_damping", 0.2))
    return config


def migrate_dpa4c_lr_config(config: dict) -> dict:
    """Translate a legacy ``dpa4c_lr`` model config to the plugin layout.

    Parameters
    ----------
    config
        The ``model`` section of a training config with
        ``type: "dpa4c_lr"``.

    Returns
    -------
    dict
        A standard-model config with a ``long_range`` section.
    """
    config = deepcopy(config)
    model_type = config.pop("type", "dpa4c_lr")
    if model_type == "dpa4c_lr":
        pass
    elif model_type in ("dpa4_lr", "sezm_sog", "sog_sezm"):
        raise NotImplementedError(
            "migrating DPA4/SeZM long-range models requires the plugin to "
            "support the DPA4 descriptor, which is not implemented yet"
        )
    else:
        raise ValueError(
            f"expected a legacy long-range model config, got type {model_type!r}"
        )
    fitting = config.get("fitting_net", {})
    if fitting.get("type", "dpa4c_lr") != "dpa4c_lr":
        raise ValueError(
            "a dpa4c_lr model requires fitting type 'dpa4c_lr', got "
            f"{fitting.get('type')!r}"
        )
    lr = {key: fitting.pop(key) for key in _LR_FITTING_KEYS if key in fitting}
    lr_kernel = str(lr.get("lr_kernel", "les"))
    fitting["type"] = "dpa4_ener"
    config["fitting_net"] = fitting
    config["long_range"] = _long_range_config(
        lr_kernel, lr, fitting, config.get("descriptor", {})
    )
    return config


# ---------------------------------------------------------------------------
# Legacy DPA4 (``dpa4_lr`` / ``sog_energy``) migration
# ---------------------------------------------------------------------------

# LR constructor keys of the legacy pt ``sog_energy`` fitting.
_LR_FITTING_KEYS_DPA4 = (
    "dim_out_lr",
    "neuron_lr",
    "bias_atom_q",
    "les_alpha",
    "amp",
    "bandwidth",
    "b",
    "sigma",
    "M",
    "remove_self_interaction",
    "use_charge_constraint",
    "use_self_pol",
    "lr_kernel",
    "alpha_pol",
    "dipole_lmax",
    "dipole_channels",
    "charge_bound",
    "use_glu",
    "lr_update_steps",
    "lr_update_neuron",
    "lr_update_damping",
)


def migrate_dpa4_lr_config(config: dict) -> dict:
    """Translate a legacy DPA4 ``dpa4_lr``/``sog_energy`` config to the plugin layout.

    Parameters
    ----------
    config
        The ``model`` section of a training config with ``type: "dpa4_lr"``
        (aliases ``sezm_sog``/``sog_sezm``) and fitting type ``sog_energy``.

    Returns
    -------
    dict
        A standard-model config with a ``long_range`` section. The dipole
        readout uses the corrected SeZM l=1 decode (the legacy pt head was not
        rotation-equivariant), so a migrated dipole model's dipoles differ
        from the legacy buggy outputs by design; charge-only models migrate
        exactly.
    """
    config = deepcopy(config)
    model_type = config.pop("type", "dpa4_lr")
    if model_type not in ("dpa4_lr", "sezm_sog", "sog_sezm"):
        raise ValueError(
            f"expected a legacy DPA4 long-range model config, got {model_type!r}"
        )
    fitting = config.get("fitting_net", {})
    if fitting.get("type") != "sog_energy":
        raise ValueError(
            "a dpa4_lr model requires fitting type 'sog_energy', got "
            f"{fitting.get('type')!r}"
        )
    lr = {key: fitting.pop(key) for key in _LR_FITTING_KEYS_DPA4 if key in fitting}
    lr_kernel = str(lr.get("lr_kernel", "les"))
    fitting["type"] = "dpa4_ener"
    config["fitting_net"] = fitting

    charge_net: dict[str, Any] = {
        "neuron": list(lr.get("neuron_lr") or [64, 64]),
        "use_glu": bool(lr.get("use_glu", True)),
        "bias_out": False,
    }
    if lr.get("charge_bound") is not None:
        charge_net["charge_bound"] = float(lr["charge_bound"])
    if lr.get("bias_atom_q") is not None:
        charge_net["bias_atom"] = lr["bias_atom_q"]

    if lr_kernel in ("les", "dipole"):
        kernel = {"type": "les", "alpha": float(lr.get("les_alpha", 1.0))}
    else:
        kernel = _sog_kernel_config(lr)

    long_range: dict[str, Any] = {
        "type": "latent_multipole" if lr_kernel == "dipole" else "latent_charge",
        "channels": int(lr.get("dim_out_lr", 1)),
        "charge_net": charge_net,
        "kernel": kernel,
        "periodic": False,
    }
    if lr_kernel == "sog" and not lr.get("remove_self_interaction", False):
        # The legacy DPA4 SOG default keeps the self-interaction term.
        long_range["self_interaction"] = True
    constraint = _constraint_of(
        lr, fitting, config.get("descriptor", {}), charge_spin_first=False
    )
    if constraint is not None:
        long_range["charge_constraint"] = constraint
    if lr_kernel == "dipole":
        if lr.get("alpha_pol") is not None:
            long_range["alpha_pol"] = lr["alpha_pol"]
        long_range["use_self_pol"] = bool(lr.get("use_self_pol", True))
        long_range["lr_update_steps"] = int(lr.get("lr_update_steps", 0))
        if lr.get("lr_update_neuron") is not None:
            long_range["lr_update_neuron"] = list(lr["lr_update_neuron"])
        long_range["lr_update_damping"] = float(lr.get("lr_update_damping", 0.2))
    config["long_range"] = long_range
    return config


# ---------------------------------------------------------------------------
# Record (checkpoint) migration
# ---------------------------------------------------------------------------


def _fitting_record(config: dict, fitting: dict, lr: dict) -> dict:
    """The energy-fitting record: the legacy record without its LR parts."""
    record = {
        key: value
        for key, value in fitting.items()
        if key not in _LR_FITTING_KEYS and key not in _LR_RECORD_KEYS
    }
    record["type"] = "dpa4_ener"
    variables = {
        key: value
        for key, value in fitting["@variables"].items()
        if key not in _LR_VARIABLES
    }
    record["@variables"] = variables
    return record


def _kernel_record(lr_kernel: str, fitting: dict) -> dict:
    variables = fitting["@variables"]
    precision = fitting.get("precision", "default")
    record: dict[str, Any] = {
        "@class": "InteractionKernel",
        "@version": 1,
        "trainable": True,
        "precision": precision,
    }
    if lr_kernel in ("les", "dipole"):
        record["type"] = "les"
        record["@variables"] = {
            "log_alpha": np.log(np.asarray(variables["les_alpha"], dtype=np.float64))
        }
    else:
        record["type"] = "sog"
        record["@variables"] = {
            "amplitude": np.asarray(variables["amp"], dtype=np.float64),
            "log_width": np.log(np.asarray(variables["bandwidth"], dtype=np.float64)),
        }
    return record


def _charge_net_record(lr_kernel: str, fitting: dict) -> dict:
    """The LatentCharge/Multipole readout record from the legacy fitting record."""
    if fitting.get("case_film_embd"):
        raise NotImplementedError(
            "legacy dpa4c_lr models with `case_film_embd` cannot be migrated: "
            "the plugin readout does not implement FiLM case conditioning"
        )
    variables = fitting["@variables"]
    # The layer bias of the readout lives in the inner network records of the
    # legacy record (``bias_out`` defaults to False for dpa4c_lr).
    nets_lr = fitting["nets_lr"]
    inner_networks = nets_lr.get("networks", [])
    bias_out = bool(inner_networks and inner_networks[0].get("bias_out", False))
    record: dict[str, Any] = {
        "@class": "Fitting",
        "@version": 5,
        "type": "latent_multipole" if lr_kernel == "dipole" else "latent_charge",
        "var_name": "latent_charge",
        "ntypes": fitting["ntypes"],
        "dim_descrpt": fitting["dim_descrpt"],
        "dim_out": fitting["dim_out_lr"],
        "neuron": fitting["neuron_lr"],
        "resnet_dt": fitting["resnet_dt"],
        "numb_fparam": fitting["numb_fparam"],
        "numb_aparam": fitting["numb_aparam"],
        "dim_case_embd": fitting["dim_case_embd"],
        "default_fparam": fitting["default_fparam"],
        "rcond": fitting["rcond"],
        "activation_function": fitting["activation_function"],
        "precision": fitting["precision"],
        "mixed_types": fitting["mixed_types"],
        "exclude_types": fitting["exclude_types"],
        "vacuum_ref": False,
        "nets": fitting["nets_lr"],
        "type_map": fitting["type_map"],
        "use_glu": fitting.get("use_glu", True),
        "charge_bound": fitting.get("charge_bound"),
        "bias_out": bias_out,
        "atom_ener": None,
        "spin": None,
        "layer_name": None,
        "tot_ener_zero": False,
        "trainable": fitting.get("trainable", True),
        "use_aparam_as_mask": fitting.get("use_aparam_as_mask", False),
        "@variables": {
            "bias_atom_e": variables["bias_atom_q"],
            "case_embd": variables.get("case_embd"),
            "fparam_avg": variables.get("fparam_avg"),
            "fparam_inv_std": variables.get("fparam_inv_std"),
            "aparam_avg": variables.get("aparam_avg"),
            "aparam_inv_std": variables.get("aparam_inv_std"),
        },
    }
    if lr_kernel == "dipole":
        record.update(
            {
                "embedding_width": fitting["embedding_width"],
                "lr_update_steps": fitting["lr_update_steps"],
                "lr_update_neuron": fitting["lr_update_neuron"],
                "lr_update_damping": fitting["lr_update_damping"],
                "dipole_weight_net": fitting["dipole_weight_net"],
                "polar_update_net": fitting.get("polar_update_net"),
            }
        )
        record["@variables"]["alpha_pol"] = variables["alpha_pol"]
    return record


def _out_stat_of(record: dict, lr_kernel: str, lr_update_steps: int) -> None:
    """Slice the legacy output statistics to the migrated output definition.

    A legacy dipole record with recurrent updates carries an extra
    ``lr_descriptor`` row and is padded to its width; the migrated model has
    neither.  Rows follow ``fitting_output_def`` order, so the drop is the
    last row.
    """
    if lr_kernel != "dipole" or lr_update_steps == 0:
        return
    max_size = 3
    for key in ("out_bias", "out_std"):
        stat = np.asarray(record["@variables"][key])
        record["@variables"][key] = stat[:3, :, :max_size]


# ---------------------------------------------------------------------------
# Legacy DPA4 record migration
# ---------------------------------------------------------------------------

# @variables prefixes of the legacy pt ``sog_energy`` record that belong to the
# long-range branch.
_LR_VARIABLE_PREFIXES_DPA4 = (
    "bias_atom_q",
    "dipole_head.",
    "alpha_pol",
    "amp",
    "bandwidth",
    "les_alpha",
)


def _dpa4_lr_bias(variables: dict, ntypes: int, dim_out: int) -> np.ndarray:
    """The static per-type charge bias of a legacy DPA4 fitting record.

    The legacy ``bias_atom_q`` is either an explicit parameter or a
    TypeEmbedNet evaluated per type; the plugin readout carries a static
    per-type array, so the embedding net is evaluated once here. The net is a
    single Linear layer on the type one-hot, so the bias of type ``i`` is row
    ``i`` of its matrix.
    """
    explicit = variables.get("bias_atom_q")
    if explicit is not None:
        return np.asarray(explicit, dtype=np.float64).reshape(ntypes, dim_out)
    matrix_key = "bias_atom_q_tebd.embedding.embedding_net.layers.0.matrix"
    bias_key = "bias_atom_q_tebd.embedding.embedding_net.layers.0.bias"
    if matrix_key not in variables:
        raise ValueError(
            "the legacy DPA4 record carries neither `bias_atom_q` nor its "
            "type-embedding net"
        )
    matrix = np.asarray(variables[matrix_key], dtype=np.float64)
    # TypeEmbedNet with one Linear layer: type i -> row i of the matrix (plus
    # the optional layer bias).
    if matrix.shape == (ntypes, dim_out):
        bias = matrix.copy()
    elif matrix.shape == (dim_out, ntypes):
        bias = matrix.T.copy()
    else:
        raise ValueError(
            f"unexpected bias_atom_q embedding matrix shape {matrix.shape}, "
            f"expected ({ntypes}, {dim_out}) or its transpose"
        )
    if bias_key in variables and variables[bias_key] is not None:
        bias += np.asarray(variables[bias_key], dtype=np.float64).reshape(1, dim_out)
    return bias


def _dpa4_dipole_proj_record(fitting: dict) -> dict:
    """The SO3Linear record of the plugin dipole head from the legacy one.

    The legacy ``dipole_head.proj`` weights transfer directly; the packed to
    Cartesian decode is a fixed, unlearned map which the plugin takes in its
    corrected form (the legacy decode was not rotation-equivariant).
    """
    variables = fitting["@variables"]
    precision = np.dtype(PRECISION_DICT[fitting.get("precision", "default")]).name
    return {
        "@class": "SO3Linear",
        "@version": 1,
        "config": {
            "lmax": int(fitting["dipole_lmax"]),
            "in_channels": int(fitting["dipole_channels"]),
            "out_channels": 1,
            "n_focus": 1,
            "precision": precision,
            "mlp_bias": False,
            "trainable": True,
            "seed": None,
        },
        "@variables": {
            "weight": np.asarray(variables["dipole_head.proj.weight"]),
            "expand_index": np.asarray(
                variables["dipole_head.proj.expand_index"], dtype=np.int64
            ),
        },
    }


def migrate_dpa4_lr_record(record: dict) -> dict:
    """Translate a serialized legacy pt ``dpa4_lr`` record to the plugin layout.

    The returned record deserializes into the pt_expt plugin architecture.
    Charge-only (les/sog) models migrate with identical outputs. Dipole
    models carry the corrected packed-to-Cartesian decode, so their dipoles
    and dipole-dependent energies differ from the legacy buggy outputs by
    design.
    """
    record = deepcopy(record)
    if record.get("type") not in ("dpa4_lr", "sezm_sog", "sog_sezm"):
        raise ValueError(
            "expected a serialized legacy DPA4 long-range model record, got "
            f"{record.get('type')!r}"
        )
    atomic = record["atomic_model"]
    fitting = atomic.get("fitting", {})
    if fitting.get("type") != "sog_energy":
        raise ValueError(
            "a legacy DPA4 long-range record requires fitting type "
            f"'sog_energy', got {fitting.get('type')!r}"
        )
    if fitting.get("case_film_embd"):
        raise NotImplementedError(
            "legacy dpa4_lr records with `case_film_embd` cannot be migrated"
        )
    lr_kernel = str(fitting.get("lr_kernel", "les"))
    variables = fitting["@variables"]
    ntypes = int(fitting["ntypes"])
    dim_out_lr = int(fitting.get("dim_out_lr", 1))

    # The new record is the pt_expt atomic-model-level record.  Mirror the
    # pt-checkpoint normalization of ``DPA4EnergyModel._normalize_pt_sezm_atomic``
    # (kept here so the dpmodel layer does not import pt_expt): the pt-only
    # ``dens`` head state is stripped, and only out_bias/out_std survive.
    if atomic.get("dens_fitting") is not None:
        raise NotImplementedError(
            "legacy dpa4_lr records with a `dens` fitting head cannot be migrated"
        )
    active_mode = atomic.get("active_mode")
    if active_mode not in (None, "ener"):
        raise NotImplementedError(
            f"legacy dpa4_lr records in active_mode {active_mode!r} cannot be "
            "migrated (energy only)"
        )
    new = {
        key: deepcopy(value)
        for key, value in atomic.items()
        if key not in ("fitting", "dens_fitting", "active_mode")
    }
    new["type"] = "standard"
    new["@version"] = 3
    atomic_variables = atomic.get("@variables", {})
    # The legacy DPA4 output def is energy + latent_charge (+ latent_dipole),
    # padded to the widest output -- exactly the plugin layout, so the output
    # statistics pass through unchanged.
    new["@variables"] = {
        key: atomic_variables[key]
        for key in ("out_bias", "out_std")
        if key in atomic_variables
    }

    # Energy fitting: the sog_energy record without its LR parts.
    new_fitting = {
        key: deepcopy(value)
        for key, value in fitting.items()
        if key not in _LR_FITTING_KEYS_DPA4
        and key not in ("nets_lr", "polar_update_net")
    }
    new_fitting["type"] = "dpa4_ener"
    new_fitting["@variables"] = {
        key: value
        for key, value in variables.items()
        if not any(key.startswith(prefix) for prefix in _LR_VARIABLE_PREFIXES_DPA4)
    }
    new["fitting"] = new_fitting

    # Charge/dipole readout record.
    precision = fitting.get("precision", "default")
    charge_net: dict[str, Any] = {
        "@class": "Fitting",
        "@version": 5,
        "type": "latent_multipole" if lr_kernel == "dipole" else "latent_charge",
        "var_name": "latent_charge",
        "ntypes": ntypes,
        "dim_descrpt": fitting["dim_descrpt"],
        "dim_out": dim_out_lr,
        "neuron": fitting["neuron_lr"],
        "resnet_dt": fitting["resnet_dt"],
        "numb_fparam": fitting["numb_fparam"],
        "numb_aparam": fitting["numb_aparam"],
        "dim_case_embd": fitting["dim_case_embd"],
        "default_fparam": fitting["default_fparam"],
        "rcond": fitting["rcond"],
        "activation_function": fitting["activation_function"],
        "precision": precision,
        "mixed_types": fitting["mixed_types"],
        "exclude_types": fitting["exclude_types"],
        "vacuum_ref": False,
        "nets": fitting["nets_lr"],
        "type_map": fitting["type_map"],
        "use_glu": fitting.get("use_glu", True),
        "charge_bound": fitting.get("charge_bound"),
        "bias_out": False,
        "atom_ener": None,
        "spin": None,
        "layer_name": None,
        "tot_ener_zero": False,
        "trainable": fitting.get("trainable", True),
        "use_aparam_as_mask": fitting.get("use_aparam_as_mask", False),
        "@variables": {
            "bias_atom_e": _dpa4_lr_bias(variables, ntypes, dim_out_lr),
            "case_embd": variables.get("case_embd"),
            "fparam_avg": variables.get("fparam_avg"),
            "fparam_inv_std": variables.get("fparam_inv_std"),
            "aparam_avg": variables.get("aparam_avg"),
            "aparam_inv_std": variables.get("aparam_inv_std"),
        },
    }
    if lr_kernel == "dipole":
        charge_net.update(
            {
                "dipole_mode": "sezm",
                "latent_lmax": int(fitting["dipole_lmax"]),
                "latent_channels": int(fitting["dipole_channels"]),
                "embedding_width": 0,
                "lr_update_steps": int(fitting.get("lr_update_steps", 0)),
                "lr_update_neuron": fitting.get("lr_update_neuron"),
                "lr_update_damping": float(fitting.get("lr_update_damping", 0.2)),
                "dipole_weight_net": None,
                "dipole_proj": _dpa4_dipole_proj_record(fitting),
                "polar_update_net": fitting.get("polar_update_net"),
            }
        )
        charge_net["@variables"]["alpha_pol"] = variables["alpha_pol"]

    long_range: dict[str, Any] = {
        "@class": "LongRange",
        "@version": 1,
        "type": "latent_multipole" if lr_kernel == "dipole" else "latent_charge",
        "charge_net": charge_net,
        "kernel": _kernel_record(lr_kernel, fitting),
        "charge_constraint": _constraint_of(
            dict(fitting),
            fitting,
            record["atomic_model"].get("descriptor", {}),
            charge_spin_first=False,
        ),
    }
    if lr_kernel == "sog" and not fitting.get("remove_self_interaction", False):
        long_range["self_interaction"] = True
    if lr_kernel == "dipole":
        long_range["use_self_pol"] = bool(fitting.get("use_self_pol", True))
    new["long_range"] = long_range
    return new


def migrate_dpa4c_lr_record(record: dict) -> dict:
    """Translate a serialized legacy ``dpa4c_lr`` model record.

    The returned record deserializes into the plugin architecture with
    identical energy, force, virial and latent outputs.
    """
    record = deepcopy(record)
    fitting = record.get("fitting", {})
    if fitting.get("type") != "dpa4c_lr":
        raise ValueError(
            "expected a serialized dpa4c_lr fitting record, got "
            f"{fitting.get('type')!r}"
        )
    lr_kernel = str(fitting.get("lr_kernel", "les"))
    lr = {key: fitting[key] for key in _LR_FITTING_KEYS if key in fitting}
    long_range: dict[str, Any] = {
        "@class": "LongRange",
        "@version": 1,
        "type": "latent_multipole" if lr_kernel == "dipole" else "latent_charge",
        "charge_net": _charge_net_record(lr_kernel, fitting),
        "kernel": _kernel_record(lr_kernel, fitting),
        "charge_constraint": _constraint_of(lr, fitting, record.get("descriptor", {})),
    }
    if lr_kernel == "dipole":
        long_range["use_self_pol"] = bool(fitting.get("use_self_pol", True))
    record["fitting"] = _fitting_record(record, fitting, lr)
    record["long_range"] = long_range
    record["@version"] = 3
    _out_stat_of(record, lr_kernel, int(fitting.get("lr_update_steps", 0)))
    return record


# ---------------------------------------------------------------------------
# Non-periodic -> periodic evaluation conversion
# ---------------------------------------------------------------------------


def _convert_one_long_range_config(lr: dict, k_grid_mode: str, head: str) -> dict:
    """Flip one ``long_range`` config section to periodic evaluation."""
    lr = deepcopy(lr)
    if lr.get("periodic", False):
        raise ValueError(
            f"the long-range plugin of {head} already evaluates periodic "
            "cells (`periodic: true`); nothing to convert"
        )
    lr["periodic"] = True
    lr["k_grid_mode"] = k_grid_mode
    if k_grid_mode == "kgrid":
        # The explicit k grid is sized from every frame's cell and needs no
        # mesh calibration; drop any stale mesh pinning.
        lr["pme_mesh"] = None
        lr["pme_mesh_spacing"] = None
    return lr


def convert_long_range_config_to_periodic(
    model_params: dict,
    k_grid_mode: str = "kgrid",
) -> dict:
    """Convert a non-periodic long-range model config to periodic evaluation.

    The LES/SOG kernel, the charge/dipole readout, the LEF update networks
    and every trained weight are untouched: the same latent charges and
    dipoles are simply summed with periodic boundary conditions (tin-foil)
    instead of as an isolated cluster.  The converted model still evaluates
    ``box=None`` frames with the direct all-pairs sum (the topology builder
    dispatches on the presence of a cell), so it remains bitwise-compatible
    with the original model on non-periodic data.

    Parameters
    ----------
    model_params
        The model config (``_extra_state["model_params"]`` of a training
        checkpoint): either a single-model config carrying a top-level
        ``long_range`` section, or a multi-task config whose
        ``model_dict`` heads carry one each.
    k_grid_mode
        ``"kgrid"`` (default): a per-frame explicit wave-number grid sized
        from the cell by ``kspace_tolerance`` -- the accuracy reference,
        adaptive to cell changes (NPT-safe).  ``"pme"``: the smooth
        particle-mesh Ewald mesh, faster at large N but fixed-size.

    Returns
    -------
    dict
        A converted deep copy; the input is not mutated.
    """
    if k_grid_mode not in ("kgrid", "pme"):
        raise ValueError(f"`k_grid_mode` must be 'kgrid' or 'pme', got {k_grid_mode!r}")
    model_params = deepcopy(model_params)
    converted = 0
    if isinstance(model_params.get("long_range"), dict):
        model_params["long_range"] = _convert_one_long_range_config(
            model_params["long_range"], k_grid_mode, "the model"
        )
        converted += 1
    for head, head_config in (model_params.get("model_dict") or {}).items():
        if isinstance(head_config, dict) and isinstance(
            head_config.get("long_range"), dict
        ):
            head_config["long_range"] = _convert_one_long_range_config(
                head_config["long_range"], k_grid_mode, f"head {head!r}"
            )
            converted += 1
    if converted == 0:
        raise ValueError(
            "no `long_range` section found in the model config; this "
            "conversion applies to latent-charge/latent-multipole plugin "
            "models only"
        )
    return model_params
