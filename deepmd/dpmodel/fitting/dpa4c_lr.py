# SPDX-License-Identifier: LGPL-3.0-or-later
"""Long-range latent-charge fitting for DPA4C (dpmodel backend).

This fitting extends the GLU energy fitting of ``dpa4_ener`` with a second,
lightweight GLU branch that predicts per-atom latent charges.  The actual
long-range energy/force is evaluated by the model layer
(:class:`deepmd.pt_expt.model.dpa4c_lr_model.DPA4CLREnergyModel`).

Only non-periodic systems are supported by the model layer; this fitting
class itself is backend-agnostic and can run on any array-API backend.
"""

from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.array_api import (
    array_api_compat,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)

from .dpa4_ener import (
    GLUFittingNet,
    SeZMEnergyFittingNet,
    SeZMNetworkCollection,
)
from .invar_fitting import (
    InvarFitting,
)


@InvarFitting.register("dpa4c_lr")
@fitting_check_output
class DPA4CLRFitting(SeZMEnergyFittingNet):
    """DPA4C energy fitting with an independent latent-charge branch.

    Parameters
    ----------
    dim_out_lr : int
        Latent charge dimension.  Must be 1 for the LES kernel.
    neuron_lr : list[int], optional
        Hidden widths of the latent-charge GLU branch.
    bias_atom_q : list[float] | None
        Per-element latent-charge bias.  If ``None`` a trainable array is
        allocated.
    les_alpha : float
        Initial width of the LES kernel ``erf(alpha * r) / r``.
    use_charge_constraint : bool
        Whether to enforce that the sum of latent charges equals the total
        charge.  In the graph path this is applied by the model layer, because
        the per-frame sum is not available inside ``call_graph``.
    lr_kernel : str
        Long-range kernel selector.  ``"les"`` is the only value supported by
        the DPA4C model layer (DPA4C has no equivariant latent, so ``"dipole"``
        is impossible).
    """

    def __init__(
        self,
        *args: Any,
        dim_out_lr: int = 1,
        neuron_lr: list[int] | None = None,
        bias_atom_q: list[float] | None = None,
        les_alpha: float | None = None,
        use_charge_constraint: bool = False,
        lr_kernel: str = "les",
        **kwargs: Any,
    ) -> None:
        # Attributes consulted by output_def() during fitting_check_output must
        # be set before super().__init__().
        self.dim_out_lr = int(dim_out_lr)
        if self.dim_out_lr < 1:
            raise ValueError(f"`dim_out_lr` must be positive, got {dim_out_lr}.")
        self.neuron_lr = [64, 64] if neuron_lr is None else [int(x) for x in neuron_lr]
        self.use_charge_constraint = bool(use_charge_constraint)
        self.bias_atom_q_bound = 3.0
        lr_kernel = str(lr_kernel).lower()
        if lr_kernel not in ("les", "sog", "dipole"):
            raise ValueError(
                f"`lr_kernel` should be 'les', 'sog' or 'dipole', got {lr_kernel!r}."
            )
        if lr_kernel == "dipole":
            raise NotImplementedError(
                "DPA4C has no equivariant latent output, so `lr_kernel='dipole'` "
                "is not supported."
            )
        self.lr_kernel = lr_kernel

        super().__init__(*args, **kwargs)

        # --- latent-charge GLU branch ---
        case_dim = 0 if self.case_film_embd else self.dim_case_embd
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + case_dim
        )
        n_networks = self.ntypes if not self.mixed_types else 1
        self.filter_layers_lr = SeZMNetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="sezm_fitting_network",
            networks=[
                GLUFittingNet(
                    in_dim,
                    self.dim_out_lr,
                    self.neuron_lr,
                    activation_function=self.activation_function,
                    resnet_dt=self.resnet_dt,
                    precision=self.precision,
                    bias_out=self.bias_out,
                    seed=child_seed(self.seed, 10_000 + idx),
                    trainable=self.trainable,
                    descriptor_dim=self.dim_descrpt,
                    dim_case_embd=self.dim_case_embd,
                    case_film_embd=self.case_film_embd,
                )
                for idx in range(n_networks)
            ],
        )

        # --- per-element latent-charge bias ---
        if bias_atom_q is None:
            self.bias_atom_q = np.zeros((self.ntypes, self.dim_out_lr), dtype=self.prec)
        else:
            bias_atom_q_t = np.asarray(bias_atom_q, dtype=self.prec).reshape(
                (self.ntypes, self.dim_out_lr)
            )
            self.bias_atom_q = bias_atom_q_t

        # --- LES width ---
        les_alpha_value = float(les_alpha) if les_alpha is not None else 1.0
        if les_alpha_value <= 0.0:
            raise ValueError("`les_alpha` should be positive.")
        self.les_alpha = np.array(les_alpha_value, dtype=self.prec)

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
                OutputVariableDef(
                    "latent_charge",
                    [self.dim_out_lr],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def call(
        self,
        descriptor: Any,
        atype: Any,
        gr: Any | None = None,
        g2: Any | None = None,
        h2: Any | None = None,
        fparam: Any | None = None,
        aparam: Any | None = None,
    ) -> dict[str, Any]:
        """Return energy and latent charges."""
        energy_ret = super().call(
            descriptor, atype, gr=gr, g2=g2, h2=h2, fparam=fparam, aparam=aparam
        )
        xp = array_api_compat.array_namespace(descriptor, atype)
        nf = descriptor.shape[0]
        if self.numb_fparam > 0 and fparam is None:
            assert self.default_fparam_tensor is not None
            default_fparam_tensor = xp.asarray(
                self.default_fparam_tensor,
                dtype=descriptor.dtype,
                device=array_api_compat.device(descriptor),
            )
            fparam = xp.tile(
                xp.reshape(default_fparam_tensor, (1, self.numb_fparam)), (nf, 1)
            )
        latent_charge = self._forward_lr(
            descriptor, atype, fparam=fparam, aparam=aparam
        )
        return {
            **energy_ret,
            "latent_charge": latent_charge,
        }

    def _forward_lr(
        self,
        descriptor: Any,
        atype: Any,
        fparam: Any | None = None,
        aparam: Any | None = None,
    ) -> Any:
        """Evaluate the latent-charge branch."""
        xp = array_api_compat.array_namespace(descriptor, atype)
        nf, nloc, nd = descriptor.shape
        if nd != self.dim_descrpt:
            raise ValueError(
                f"descriptor dim {nd} inconsistent with {self.dim_descrpt}"
            )

        xx = descriptor
        if self.numb_fparam > 0:
            assert fparam is not None
            fparam = xp.reshape(fparam, (nf, self.numb_fparam))
            fparam_avg = xp.asarray(
                self.fparam_avg,
                dtype=fparam.dtype,
                device=array_api_compat.device(fparam),
            )
            fparam_inv_std = xp.asarray(
                self.fparam_inv_std,
                dtype=fparam.dtype,
                device=array_api_compat.device(fparam),
            )
            fparam = (fparam - fparam_avg) * fparam_inv_std
            fparam = xp.tile(
                xp.reshape(fparam, (nf, 1, self.numb_fparam)), (1, nloc, 1)
            )
            xx = xp.concat([xx, fparam], axis=-1)

        if self.numb_aparam > 0 and not self.use_aparam_as_mask:
            assert aparam is not None
            aparam = xp.reshape(aparam, (nf, nloc, self.numb_aparam))
            aparam_avg = xp.asarray(
                self.aparam_avg,
                dtype=aparam.dtype,
                device=array_api_compat.device(aparam),
            )
            aparam_inv_std = xp.asarray(
                self.aparam_inv_std,
                dtype=aparam.dtype,
                device=array_api_compat.device(aparam),
            )
            aparam = (aparam - aparam_avg) * aparam_inv_std
            xx = xp.concat([xx, aparam], axis=-1)

        if self.dim_case_embd > 0:
            assert self.case_embd is not None
            case_embd = xp.asarray(
                self.case_embd,
                dtype=descriptor.dtype,
                device=array_api_compat.device(descriptor),
            )
            case_embd = xp.tile(xp.reshape(case_embd, (1, 1, -1)), (nf, nloc, 1))
            xx = xp.concat([xx, case_embd], axis=-1)

        net_dtype = get_xp_precision(xp, self.precision)
        xx = xp.astype(xx, net_dtype)
        lr_out = xp.zeros(
            (nf, nloc, self.dim_out_lr),
            dtype=net_dtype,
            device=array_api_compat.device(descriptor),
        )
        if self.mixed_types:
            lr_out = lr_out + self.filter_layers_lr.networks[0](xx)
        else:
            for type_i, ll in enumerate(self.filter_layers_lr.networks):
                mask = xp.astype(atype == type_i, descriptor.dtype)
                atom_property = ll(xx)
                atom_property = xp.where(
                    xp.astype(mask[:, :, None], xp.bool),
                    atom_property,
                    xp.zeros_like(atom_property),
                )
                lr_out = lr_out + atom_property

        lr_out = lr_out + self._get_lr_bias(atype, xp)

        # Total-charge constraint is applied by the model layer
        # (:class:`~deepmd.pt_expt.model.dpa4c_lr_model.DPA4CLREnergyModel`),
        # which has access to the frame-level ``charge_spin`` target.
        exclude_mask = self.emask.build_type_exclude_mask(atype)
        exclude_mask = xp.astype(exclude_mask, xp.bool)
        lr_out = xp.where(exclude_mask[:, :, None], lr_out, xp.zeros_like(lr_out))
        return lr_out

    def _get_lr_bias(self, atype: Any, xp: Any) -> Any:
        """Return per-atom latent-charge bias."""
        atype_flat = xp.reshape(xp.astype(atype, xp.int64), (-1,))
        target_dtype = get_xp_precision(xp, self.precision)
        target_device = array_api_compat.device(atype)
        bias_atom_q = self.bias_atom_q
        if array_api_compat.is_torch_array(bias_atom_q):
            # Keep the autograd graph: ``.to`` preserves grad_fn, while
            # ``xp.asarray`` on a Parameter may return a detached view in some
            # array_api_compat versions.
            bias_atom_q = bias_atom_q.to(dtype=target_dtype, device=target_device)
        else:
            bias_atom_q = xp.asarray(
                bias_atom_q, dtype=target_dtype, device=target_device
            )
        bias = xp.take(bias_atom_q, atype_flat, axis=0).reshape(
            *atype.shape, self.dim_out_lr
        )
        return self.bias_atom_q_bound * xp.tanh(bias / self.bias_atom_q_bound)

    def serialize(self) -> dict:
        """Serialize the fitting."""
        data = super().serialize()
        data["type"] = "dpa4c_lr"
        data["dim_out_lr"] = self.dim_out_lr
        data["neuron_lr"] = self.neuron_lr
        data["nets_lr"] = self.filter_layers_lr.serialize()
        data["use_charge_constraint"] = self.use_charge_constraint
        data["lr_kernel"] = self.lr_kernel
        variables = data.setdefault("@variables", {})
        variables["bias_atom_q"] = to_numpy_array(self.bias_atom_q)
        variables["les_alpha"] = to_numpy_array(self.les_alpha)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DPA4CLRFitting":
        """Deserialize the fitting."""
        data = data.copy()
        variables = data.get("@variables", {}).copy()
        bias_atom_q = variables.pop("bias_atom_q", None)
        les_alpha = variables.pop("les_alpha", None)
        nets_lr = data.pop("nets_lr", None)
        data["@variables"] = variables
        obj = super().deserialize(data)
        if nets_lr is not None:
            obj.filter_layers_lr = SeZMNetworkCollection.deserialize(nets_lr)
        if bias_atom_q is not None:
            new_bias = np.asarray(bias_atom_q, dtype=obj.prec).reshape(
                obj.ntypes, obj.dim_out_lr
            )
            obj.bias_atom_q = new_bias
        if les_alpha is not None:
            new_alpha = float(np.asarray(les_alpha).reshape(-1)[0])
            if new_alpha <= 0.0:
                raise ValueError("`les_alpha` should be positive.")
            obj.les_alpha = np.array(new_alpha, dtype=obj.prec)
        return obj
