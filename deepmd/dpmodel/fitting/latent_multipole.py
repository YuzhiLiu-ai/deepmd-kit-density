# SPDX-License-Identifier: LGPL-3.0-or-later
"""Charge/dipole readout and recurrent updates owned by a long-range plugin."""

from typing import Any

import numpy as np

from deepmd.dpmodel.array_api import array_api_compat, xp_asarray_nodetach
from deepmd.dpmodel.common import get_xp_precision, to_numpy_array
from deepmd.dpmodel.output_def import FittingOutputDef, OutputVariableDef
from deepmd.dpmodel.utils.network import FittingNet
from deepmd.dpmodel.utils.seed import child_seed
from deepmd.utils.version import check_version_compatibility

from .dpa4_ener import GLUFittingNet, SeZMNetworkCollection
from .latent_charge import LatentChargeFitting


class LatentMultipoleFitting(LatentChargeFitting):
    """DPA4C latent-charge/dipole networks with the legacy numerical policy.

    This is not an energy fitting.  It owns only the readout, degree-one
    contraction and optional recurrent polar update used by
    :class:`LatentMultipoleLongRange`.
    """

    def __init__(
        self,
        *args: Any,
        charge_bound: float | None = None,
        use_glu: bool = True,
        embedding_width: int = 0,
        alpha_pol: float | list[float] | None = None,
        lr_update_steps: int = 0,
        lr_update_neuron: list[int] | None = None,
        lr_update_damping: float = 0.2,
        **kwargs: Any,
    ) -> None:
        self.charge_bound = None if charge_bound is None else float(charge_bound)
        if self.charge_bound is not None and self.charge_bound <= 0.0:
            raise ValueError("`charge_bound` must be positive")
        self.use_glu = bool(use_glu)
        self.embedding_width = int(embedding_width)
        if self.embedding_width < 1:
            raise ValueError("`embedding_width` must be positive")
        self.lr_update_steps = int(lr_update_steps)
        if self.lr_update_steps < 0:
            raise ValueError("`lr_update_steps` must be non-negative")
        self.lr_update_neuron = (
            [64, 64] if lr_update_neuron is None else list(lr_update_neuron)
        )
        self.lr_update_damping = float(lr_update_damping)
        if not 0.0 < self.lr_update_damping <= 1.0:
            raise ValueError("`lr_update_damping` must be in (0, 1]")
        seed = kwargs.get("seed")
        super().__init__(*args, **kwargs)
        if self.dim_out != 1:
            raise ValueError("latent multipole interactions require one charge channel")

        trainable = (
            all(self.trainable) if isinstance(self.trainable, list) else self.trainable
        )
        if self.use_glu:
            in_dim = (
                self.dim_descrpt
                + self.numb_fparam
                + (0 if self.use_aparam_as_mask else self.numb_aparam)
                + self.dim_case_embd
            )
            n_networks = 1 if self.mixed_types else self.ntypes
            self.nets = SeZMNetworkCollection(
                0 if self.mixed_types else 1,
                self.ntypes,
                network_type="sezm_fitting_network",
                networks=[
                    GLUFittingNet(
                        in_dim,
                        1,
                        self.neuron,
                        activation_function=self.activation_function,
                        resnet_dt=self.resnet_dt,
                        precision=self.precision,
                        bias_out=True,
                        seed=child_seed(seed, 10_000 + idx),
                        trainable=self.trainable,
                        descriptor_dim=self.dim_descrpt,
                        dim_case_embd=self.dim_case_embd,
                        case_film_embd=False,
                    )
                    for idx in range(n_networks)
                ],
            )
        self.dipole_weight_net = FittingNet(
            self.dim_descrpt,
            self.embedding_width,
            self.neuron,
            self.activation_function,
            self.resnet_dt,
            self.precision,
            bias_out=True,
            seed=child_seed(seed, 10_002),
            trainable=trainable,
        )
        if self.lr_update_steps > 0:
            self.polar_update_net = GLUFittingNet(
                self.dim_descrpt + 5,
                4,
                self.lr_update_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                bias_out=True,
                seed=child_seed(seed, 10_003),
                trainable=self.trainable,
                descriptor_dim=self.dim_descrpt,
                dim_case_embd=0,
                case_film_embd=False,
            )
        else:
            self.polar_update_net = None
        if alpha_pol is None:
            alpha_pol_array = np.ones(self.ntypes, dtype=self.prec)
        else:
            alpha_pol_array = np.asarray(alpha_pol, dtype=self.prec).reshape(-1)
            if alpha_pol_array.size == 1:
                alpha_pol_array = np.full(
                    self.ntypes, alpha_pol_array[0], dtype=self.prec
                )
            if alpha_pol_array.size != self.ntypes:
                raise ValueError(
                    "`alpha_pol` must be scalar or have one value per type"
                )
        if not np.all(np.isfinite(alpha_pol_array)) or np.any(alpha_pol_array <= 0.0):
            raise ValueError("`alpha_pol` must contain finite positive values")
        self.alpha_pol = alpha_pol_array

    def _parameter_array(self, name: str, like: Any) -> Any:
        """Return a parameter in the input namespace without detaching autograd."""
        xp = array_api_compat.array_namespace(like)
        return xp_asarray_nodetach(
            xp,
            getattr(self, name),
            dtype=like.dtype,
            device=array_api_compat.device(like),
        )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                *self.charge_output_def().get_data().values(),
                OutputVariableDef(
                    "latent_dipole",
                    [3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def charge_output_def(self) -> FittingOutputDef:
        return super().output_def()

    def _charge_readout(
        self, descriptor: Any, atype: Any, fparam: Any | None, aparam: Any | None
    ) -> Any:
        """Reproduce DPA4CLRFitting._forward_lr on the flat node axis."""
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
                (n, 1, 1), dtype=net_dtype, device=array_api_compat.device(descriptor)
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
        charge = charge[:, 0, :] + 3.0 * xp.tanh(bias / 3.0)
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
        del g2, h2, kwargs
        charge = self._charge_readout(descriptor, atype, fparam, aparam)
        if gr is None:
            raise RuntimeError("latent multipoles require DPA4C degree-one moments")
        if gr.shape[-2] != self.embedding_width:
            raise ValueError(
                f"degree-one width {gr.shape[-2]} does not match "
                f"embedding_width={self.embedding_width}"
            )
        xp = array_api_compat.array_namespace(descriptor, atype)
        net_dtype = get_xp_precision(xp, self.precision)
        weights = self.dipole_weight_net(xp.astype(descriptor, net_dtype))
        dipole = xp.sum(weights[..., :, None] * xp.astype(gr, net_dtype), axis=-2)
        exclude = xp.astype(self.emask.build_type_exclude_mask(atype), xp.bool)
        dipole = xp.where(exclude[:, None], dipole, xp.zeros_like(dipole))
        return {"latent_charge": charge, "latent_dipole": dipole}

    def update_polar_state(
        self,
        descriptor: Any,
        charge: Any,
        dipole: Any,
        local_dipole: Any,
        potential: Any,
        field: Any,
    ) -> tuple[Any, Any]:
        """Apply one normalized, gated and damped equivariant polar update."""
        if self.polar_update_net is None:
            return charge, dipole
        xp = array_api_compat.array_namespace(descriptor, charge, dipole)
        dtype = get_xp_precision(xp, self.precision)
        descriptor = xp.astype(descriptor, dtype)
        q = xp.astype(charge, dtype)
        p = xp.astype(dipole, dtype)
        p0 = xp.astype(local_dipole, dtype)
        phi = xp.astype(potential, dtype)
        electric = xp.astype(field, dtype)
        p_sq = xp.sum(p * p, axis=-1, keepdims=True)
        e_sq = xp.sum(electric * electric, axis=-1, keepdims=True)
        p_unit = p / xp.sqrt(1.0 + p_sq)
        e_unit = electric / xp.sqrt(1.0 + e_sq)
        update_input = xp.concat(
            [
                descriptor,
                q,
                xp.tanh(phi),
                p_sq / (1.0 + p_sq),
                e_sq / (1.0 + e_sq),
                xp.sum(p_unit * e_unit, axis=-1, keepdims=True),
            ],
            axis=-1,
        )
        update = self.polar_update_net(update_input)
        dq = xp.tanh(update[..., 0:1])
        gates = xp.tanh(update[..., 1:4])
        dmu = gates[..., 0:1] * p + gates[..., 1:2] * e_unit + gates[..., 2:3] * p0
        return q + self.lr_update_damping * dq, p + self.lr_update_damping * dmu

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any = None
    ) -> None:
        from deepmd.utils.finetune import get_index_between_two_maps

        assert self.type_map is not None
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        xp = array_api_compat.array_namespace(self.alpha_pol)
        alpha_pol = self.alpha_pol
        if has_new_type:
            extended = xp.ones(
                (len(type_map),),
                dtype=alpha_pol.dtype,
                device=array_api_compat.device(alpha_pol),
            )
            alpha_pol = xp.concat([alpha_pol, extended[len(self.ntypes) :]], axis=0)
        self.alpha_pol = alpha_pol[remap_index]
        super().change_type_map(type_map, model_with_new_type_stat)

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "latent_multipole"
        data.update(
            {
                "charge_bound": self.charge_bound,
                "use_glu": self.use_glu,
                "embedding_width": self.embedding_width,
                "lr_update_steps": self.lr_update_steps,
                "lr_update_neuron": self.lr_update_neuron,
                "lr_update_damping": self.lr_update_damping,
                "dipole_weight_net": self.dipole_weight_net.serialize(),
                "polar_update_net": (
                    None
                    if self.polar_update_net is None
                    else self.polar_update_net.serialize()
                ),
            }
        )
        data["@variables"]["alpha_pol"] = to_numpy_array(self.alpha_pol)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "LatentMultipoleFitting":
        data = data.copy()
        variables = data.pop("@variables").copy()
        alpha_pol = variables.pop("alpha_pol", None)
        if alpha_pol is not None:
            data["alpha_pol"] = np.asarray(alpha_pol).reshape(-1).tolist()
        nets = data.pop("nets")
        dipole_weight_net = data.pop("dipole_weight_net")
        polar_update_net = data.pop("polar_update_net")
        data.pop("@class")
        data.pop("type")
        check_version_compatibility(data.pop("@version", 1), 5, 1)
        obj = cls(**data)
        for key, value in variables.items():
            obj[key] = value
        if obj.use_glu:
            obj.nets = SeZMNetworkCollection.deserialize(nets)
        else:
            from deepmd.dpmodel.utils.network import NetworkCollection

            obj.nets = NetworkCollection.deserialize(nets)
        obj.dipole_weight_net = FittingNet.deserialize(dipole_weight_net)
        if polar_update_net is not None:
            obj.polar_update_net = GLUFittingNet.deserialize(polar_update_net)
        return obj
