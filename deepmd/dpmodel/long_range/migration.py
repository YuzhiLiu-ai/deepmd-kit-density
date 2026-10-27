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


def _constraint_of(lr: dict, fitting: dict, descriptor: dict) -> dict | None:
    """Resolve the legacy runtime constraint chain to one config-time source.

    The legacy model read the total charge from ``charge_spin`` when the call
    carried it, then from the first frame parameter, and fell back to zero.
    The plugin fixes the source at build time: ``charge_spin`` when the
    descriptor takes the condition, ``fparam`` when the fitting reads frame
    parameters, neutral otherwise.
    """
    if not lr.get("use_charge_constraint", False):
        return None
    if descriptor.get("add_chg_spin_ebd"):
        source = "charge_spin"
    elif fitting.get("numb_fparam", 0) > 0:
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
