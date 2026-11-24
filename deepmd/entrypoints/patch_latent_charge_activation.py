# SPDX-License-Identifier: LGPL-3.0-or-later
"""Explicitly record latent-charge activations in existing ``.pt`` checkpoints.

The default is a dry run. Use ``--write`` to back up and atomically replace the
checkpoint after the patched in-memory copy has been validated and reloaded.
"""

import argparse
import copy
import os
import shutil
from collections.abc import (
    Iterator,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)


def _load_checkpoint(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _iter_wrapper_model_params(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield every wrapper ``model_params`` dictionary in a checkpoint."""
    if not isinstance(payload, dict):
        return
    extra = payload.get("_extra_state")
    if isinstance(extra, dict) and isinstance(extra.get("model_params"), dict):
        yield extra["model_params"]
        return
    for value in payload.values():
        yield from _iter_wrapper_model_params(value)


def _iter_model_configs(
    model_params: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    if isinstance(model_params.get("long_range"), dict):
        yield model_params
    model_dict = model_params.get("model_dict")
    if isinstance(model_dict, dict):
        for config in model_dict.values():
            if isinstance(config, dict) and isinstance(config.get("long_range"), dict):
                yield config


def _patch_model_params(
    model_params: dict[str, Any],
    activation: str,
    charge_activation: str,
    *,
    force: bool,
) -> int:
    """Patch model configs in place and return the number of changed configs."""
    changed = 0
    for config in _iter_model_configs(model_params):
        charge_net = config["long_range"].setdefault("charge_net", {})
        updates = {
            "activation_function": activation,
            "charge_activation_function": charge_activation,
        }
        config_changed = False
        for key, value in updates.items():
            previous = charge_net.get(key)
            if previous is not None and previous != value and not force:
                raise ValueError(
                    f"{key} is already {previous!r}, not {value!r}; "
                    "pass --force to overwrite it"
                )
            if previous != value:
                charge_net[key] = value
                config_changed = True
        changed += int(config_changed)
    return changed


def _validate_model_params(model_params: dict[str, Any]) -> None:
    """Construct every patched model and verify its effective activations."""
    from deepmd.pt_expt.model.get_model import get_model

    for config in _iter_model_configs(model_params):
        model = get_model(copy.deepcopy(config))
        charge_net = model.atomic_model.long_range.charge_net
        expected = config["long_range"]["charge_net"]
        assert charge_net.activation_function == expected["activation_function"]
        assert (
            charge_net.charge_activation_function
            == expected["charge_activation_function"]
        )


def _assert_unchanged(before: Any, after: Any, path: str = "root") -> None:
    """Compare everything except wrapper ``_extra_state`` configuration."""
    if isinstance(before, dict):
        assert isinstance(after, dict), path
        assert before.keys() == after.keys(), path
        for key in before:
            if key != "_extra_state":
                _assert_unchanged(before[key], after[key], f"{path}.{key}")
        return
    if isinstance(before, (list, tuple)):
        assert type(before) is type(after) and len(before) == len(after), path
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            _assert_unchanged(left, right, f"{path}[{index}]")
        return
    import numpy as np
    import torch

    if isinstance(before, torch.Tensor):
        assert isinstance(after, torch.Tensor) and torch.equal(before, after), path
    elif isinstance(before, np.ndarray):
        assert isinstance(after, np.ndarray) and np.array_equal(before, after), path
    else:
        assert before == after, path


def patch_checkpoint(
    path: str | Path,
    activation: str,
    *,
    charge_activation: str | None = None,
    write: bool = False,
    force: bool = False,
    backup: bool = True,
) -> int:
    """Patch one checkpoint and return its number of changed model configs.

    ``activation`` controls dipole/LEF networks; ``charge_activation`` defaults
    to the same value and controls the latent-charge readout. Nothing is written
    unless ``write=True``. A written file is first saved to a sibling temporary
    file and reloaded for validation, then the original is backed up as ``.bak``
    before an atomic replacement.
    """
    import torch

    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError(f"only training .pt checkpoints are supported: {path}")
    charge_activation = activation if charge_activation is None else charge_activation
    original = _load_checkpoint(path)
    patched = copy.deepcopy(original)
    changed = 0
    for model_params in _iter_wrapper_model_params(patched):
        changed += _patch_model_params(
            model_params,
            activation,
            charge_activation,
            force=force,
        )
        _validate_model_params(model_params)
    if changed == 0:
        raise ValueError(f"no latent charge_net configuration needed patching: {path}")
    _assert_unchanged(original, patched)
    if not write:
        return changed

    temporary = path.with_name(path.name + ".tmp")
    backup_path = path.with_name(path.name + ".bak")
    if backup and backup_path.exists():
        raise FileExistsError(f"backup already exists: {backup_path}")
    try:
        torch.save(patched, temporary)
        reloaded = _load_checkpoint(temporary)
        _assert_unchanged(original, reloaded)
        for model_params in _iter_wrapper_model_params(reloaded):
            _validate_model_params(model_params)
        if backup:
            shutil.copy2(path, backup_path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--activation",
        required=True,
        help="dipole/LEF activation, e.g. silu or tanh",
    )
    parser.add_argument(
        "--charge-activation",
        help="charge readout activation; defaults to --activation",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write validated patches (the default is a dry run)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite conflicting existing activation keys",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not keep a .bak copy when --write is used",
    )
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        changed = patch_checkpoint(
            checkpoint,
            args.activation,
            charge_activation=args.charge_activation,
            write=args.write,
            force=args.force,
            backup=not args.no_backup,
        )
        action = "patched" if args.write else "would patch"
        print(f"{checkpoint}: {action} {changed} model config(s)")  # noqa: T201


if __name__ == "__main__":
    main()
