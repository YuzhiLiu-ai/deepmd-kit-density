# SPDX-License-Identifier: LGPL-3.0-or-later
"""Latent-charge readout owned by a long-range plugin.

The readout mirrors the legacy ``dpa4c_lr`` latent-charge branch so that
legacy configs and checkpoints migrate without numerical changes:

- per-type or shared GLU (``use_glu=True``) / MLP fitting networks;
- a per-type bias squashed as ``3 * tanh(bias / 3)``;
- an optional output bound ``charge_bound * tanh(q / charge_bound)``.
"""

from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    array_api_compat,
    xp_asarray_nodetach,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.dpmodel.utils.network import (
    NetworkCollection,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .dpa4_ener import (
    GLUFittingNet,
    SeZMNetworkCollection,
)
from .invar_fitting import (
    InvarFitting,
)

# Bias squash of the legacy dpa4c_lr charge branch.
BIAS_BOUND = 3.0


@fitting_check_output
class LatentChargeFitting(InvarFitting):
    """Per-atom latent charges predicted from invariant descriptor features.

    The constructor is that of :class:`InvarFitting` with the readout options
    below; ``var_name`` is normally ``latent_charge`` and ``dim_out`` is the
    number of charge channels.  This is a readout owned by a long-range
    module, not a standalone fitting type.

    Parameters
    ----------
    use_glu
        Use GLU fitting networks (the legacy default) instead of plain MLPs.
    charge_bound
        Bound the charge as ``charge_bound * tanh(q / charge_bound)``;
        ``None`` (default) leaves the output unbounded.
    charge_activation_function
        Activation of the latent-charge readout. Defaults to
        ``activation_function`` (itself ``tanh`` by default). The separate key
        preserves legacy models with tanh charge readout but silu dipole/LEF
        networks. Old serialized records without this key retain tanh readout.
    """

    def __init__(
        self,
        *args: Any,
        use_glu: bool = False,
        charge_bound: float | None = None,
        bias_out: bool = False,
        charge_activation_function: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.use_glu = bool(use_glu)
        self.charge_bound = None if charge_bound is None else float(charge_bound)
        if self.charge_bound is not None and self.charge_bound <= 0.0:
            raise ValueError(f"`charge_bound` must be positive, got {charge_bound}")
        self.bias_out = bool(bias_out)
        seed = kwargs.get("seed")
        super().__init__(*args, **kwargs)
        # New configs use activation_function for the charge readout too. A
        # separate charge_activation_function allows legacy dipole models to
        # retain their split policy: tanh charge readout + silu dipole/LEF nets.
        self.charge_activation_function = (
            self.activation_function
            if charge_activation_function is None
            else str(charge_activation_function)
        )
        # Rebuild the networks with the legacy dpa4c_lr readout
        # parameterization: the layer bias defaults to off (the per-type bias
        # carries it), unlike the energy fitting's nets.
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + self.dim_case_embd
        )
        n_networks = 1 if self.mixed_types else self.ntypes
        if self.use_glu:
            self.nets = SeZMNetworkCollection(
                0 if self.mixed_types else 1,
                self.ntypes,
                network_type="sezm_fitting_network",
                networks=[
                    GLUFittingNet(
                        in_dim,
                        self.dim_out,
                        self.neuron,
                        activation_function=self.charge_activation_function,
                        resnet_dt=self.resnet_dt,
                        precision=self.precision,
                        bias_out=self.bias_out,
                        seed=child_seed(seed, 10_000 + idx),
                        trainable=self.trainable,
                        descriptor_dim=self.dim_descrpt,
                        dim_case_embd=self.dim_case_embd,
                        case_film_embd=False,
                    )
                    for idx in range(n_networks)
                ],
            )
        else:
            from deepmd.dpmodel.utils.network import (
                FittingNet,
            )

            trainable = (
                all(self.trainable)
                if isinstance(self.trainable, list)
                else self.trainable
            )
            self.nets = NetworkCollection(
                1 if not self.mixed_types else 0,
                self.ntypes,
                network_type="fitting_network",
                networks=[
                    FittingNet(
                        in_dim,
                        self.dim_out,
                        self.neuron,
                        self.charge_activation_function,
                        self.resnet_dt,
                        self.precision,
                        bias_out=self.bias_out,
                        seed=child_seed(seed, 10_000 + idx),
                        trainable=trainable,
                    )
                    for idx in range(n_networks)
                ],
            )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            ]
        )

    def readout(
        self,
        descriptor: Any,
        atype: Any,
        fparam: Any | None = None,
        aparam: Any | None = None,
    ) -> Any:
        """Evaluate the charge of every flat node, with the legacy bounding."""
        xp = array_api_compat.array_namespace(descriptor, atype)
        n = descriptor.shape[0]
        d1 = xp.reshape(descriptor, (n, 1, self.dim_descrpt))
        a1 = xp.reshape(atype, (n, 1))
        ap1 = None if aparam is None else xp.reshape(aparam, (n, 1, -1))
        xx = d1
        cond = self.conditioning_columns(d1, fparam, ap1)
        if cond is not None:
            xx = xp.concat([xx, cond], axis=-1)
        net_dtype = get_xp_precision(xp, self.precision)
        xx = xp.astype(xx, net_dtype)
        if self.mixed_types:
            charge = self.nets[()](xx)
        else:
            charge = xp.zeros(
                (n, 1, self.dim_out),
                dtype=net_dtype,
                device=array_api_compat.device(descriptor),
            )
            for type_i in range(self.ntypes):
                value = self.nets[(type_i,)](xx)
                charge = charge + xp.where(
                    (a1 == type_i)[..., None], value, xp.zeros_like(value)
                )
        bias = xp_asarray_nodetach(
            xp,
            self.bias_atom_e,
            dtype=net_dtype,
            device=array_api_compat.device(descriptor),
        )
        bias = xp.take(bias, xp.reshape(atype, (-1,)), axis=0)
        charge = charge[:, 0, :] + BIAS_BOUND * xp.tanh(bias / BIAS_BOUND)
        exclude = xp.astype(self.emask.build_type_exclude_mask(atype), xp.bool)
        charge = xp.where(exclude[:, None], charge, xp.zeros_like(charge))
        if self.charge_bound is not None:
            charge = self.charge_bound * xp.tanh(charge / self.charge_bound)
        return charge

    def call_graph(
        self,
        descriptor: Any,
        atype: Any,
        gr: Any | None = None,
        g2: Any | None = None,
        h2: Any | None = None,
        fparam: Any | None = None,
        aparam: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return the latent charge of every node."""
        del gr, g2, h2, kwargs
        return {self.var_name: self.readout(descriptor, atype, fparam, aparam)}

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "latent_charge"
        # Persist both policies explicitly. In particular, old migrated models
        # may use silu for dipole/LEF but tanh for the charge readout.
        data["activation_function"] = self.activation_function
        data["charge_activation_function"] = self.charge_activation_function
        data["use_glu"] = self.use_glu
        data["charge_bound"] = self.charge_bound
        data["bias_out"] = self.bias_out
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "LatentChargeFitting":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 5, 1)
        data.pop("@class")
        data.pop("type")
        variables = data.pop("@variables")
        nets = data.pop("nets")
        # Records written before charge_activation_function existed had a
        # hard-coded tanh readout, even when dipole/LEF used activation_function.
        data.setdefault("charge_activation_function", "tanh")
        use_glu = data.get("use_glu", False)
        obj = cls(**data)
        for key, value in variables.items():
            obj[key] = value
        collection = SeZMNetworkCollection if use_glu else NetworkCollection
        obj.nets = collection.deserialize(nets)
        return obj
