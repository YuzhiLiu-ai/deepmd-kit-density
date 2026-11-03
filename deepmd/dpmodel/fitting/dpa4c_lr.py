# SPDX-License-Identifier: LGPL-3.0-or-later
"""Long-range latent charge/dipole fitting for DPA4C (dpmodel backend).

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
from deepmd.dpmodel.utils.network import (
    FittingNet,
    NetworkCollection,
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

SOG_DEFAULT_B = 1.62976708826776469
SOG_DEFAULT_SIGMA = 2.180230445405648
SOG_DEFAULT_M = 12


@InvarFitting.register("dpa4c_lr")
@fitting_check_output
class DPA4CLRFitting(SeZMEnergyFittingNet):
    """DPA4C fitting with latent charge and optional dipole/update branches.

    Parameters
    ----------
    dim_out_lr : int
        Latent charge dimension. Must be 1 for LES/dipole; SOG supports
        multiple latent channels.
    neuron_lr : list[int], optional
        Hidden widths of the latent-charge GLU branch.
    bias_atom_q : list[float] | None
        Per-element latent-charge bias.  If ``None`` a trainable array is
        allocated.
    les_alpha : float
        Initial width of the LES kernel ``erf(alpha * r) / r``.
    amp : float | list[float] | None
        Initial amplitudes of the SOG Gaussian levels.
    bandwidth : float | list[float] | None
        Initial widths of the SOG Gaussian levels.
    b, sigma, M
        Hyperparameters used to initialize geometric SOG widths when
        ``bandwidth`` is not provided.
    use_charge_constraint : bool
        Whether to enforce that the sum of latent charges equals the total
        charge.  In the graph path this is applied by the model layer, because
        the per-frame sum is not available inside ``call_graph``.
    lr_kernel : str
        Long-range kernel selector: ``"les"``, ``"sog"``, or ``"dipole"``.
    """

    def __init__(
        self,
        *args: Any,
        dim_out_lr: int = 1,
        neuron_lr: list[int] | None = None,
        bias_atom_q: list[float] | None = None,
        amp: float | list[float] | None = None,
        bandwidth: float | list[float] | None = None,
        les_alpha: float | None = None,
        b: float | None = None,
        sigma: float | None = None,
        M: int | None = None,
        use_charge_constraint: bool = False,
        lr_kernel: str = "les",
        charge_bound: float | None = None,
        use_glu: bool = False,
        embedding_width: int = 0,
        alpha_pol: float | list[float] | None = None,
        use_self_pol: bool = True,
        lr_update_steps: int = 0,
        lr_update_neuron: list[int] | None = None,
        lr_update_damping: float = 0.2,
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
        self.charge_bound = None if charge_bound is None else float(charge_bound)
        self.use_glu = bool(use_glu)
        self.embedding_width = int(embedding_width)
        self.use_self_pol = bool(use_self_pol)
        self.lr_update_steps = int(lr_update_steps)
        if self.lr_update_steps < 0:
            raise ValueError("`lr_update_steps` must be non-negative.")
        self.lr_update_neuron = (
            [64, 64]
            if lr_update_neuron is None
            else [int(value) for value in lr_update_neuron]
        )
        self.lr_update_damping = float(lr_update_damping)
        if not 0.0 < self.lr_update_damping <= 1.0:
            raise ValueError("`lr_update_damping` must be in (0, 1].")
        lr_kernel = str(lr_kernel).lower()
        if lr_kernel not in ("les", "sog", "dipole"):
            raise ValueError(
                f"`lr_kernel` should be 'les', 'sog' or 'dipole', got {lr_kernel!r}."
            )
        if lr_kernel in ("les", "dipole") and self.dim_out_lr != 1:
            raise ValueError(
                f"`lr_kernel='{lr_kernel}'` requires `dim_out_lr=1`, got "
                f"{self.dim_out_lr}. Use `lr_kernel='sog'` for multiple latent "
                "channels."
            )
        if lr_kernel == "dipole" and self.embedding_width < 1:
            raise ValueError(
                "`lr_kernel='dipole'` requires a positive DPA4C degree-one "
                "embedding width."
            )
        if self.lr_update_steps > 0 and lr_kernel != "dipole":
            raise ValueError("`lr_update_steps > 0` requires `lr_kernel='dipole'`.")
        self.lr_kernel = lr_kernel

        super().__init__(*args, **kwargs)

        # --- latent-charge branch ---
        # use_glu=False (default) uses the standard FittingNet (MLP), which is
        # less expressive but more stable (no unbounded val branch that can
        # blow up the latent charge); use_glu=True opts into GLUFittingNet.
        case_dim = 0 if self.case_film_embd else self.dim_case_embd
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + case_dim
        )
        n_networks = self.ntypes if not self.mixed_types else 1
        if self.use_glu:
            self.filter_layers_lr = SeZMNetworkCollection(
                1 if not self.mixed_types else 0,
                self.ntypes,
                network_type="sezm_fitting_network",
                networks=[
                    GLUFittingNet(
                        in_dim,
                        self.dim_out_lr,
                        self.neuron_lr,
                        # The LR charge readout defaults to tanh, independent
                        # of the main fitting's activation (silu by default).
                        activation_function="tanh",
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
        else:
            self.filter_layers_lr = NetworkCollection(
                1 if not self.mixed_types else 0,
                self.ntypes,
                network_type="fitting_network",
                networks=[
                    FittingNet(
                        in_dim,
                        self.dim_out_lr,
                        self.neuron_lr,
                        "tanh",  # the LR charge readout defaults to tanh
                        self.resnet_dt,
                        self.precision,
                        bias_out=self.bias_out,
                        seed=child_seed(self.seed, 10_000 + idx),
                        trainable=(
                            all(self.trainable)
                            if isinstance(self.trainable, list)
                            else self.trainable
                        ),
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

        # --- SOG Gaussian levels ---
        self.b = SOG_DEFAULT_B if b is None else float(b)
        self.sigma = SOG_DEFAULT_SIGMA if sigma is None else float(sigma)
        self.M = max(1, SOG_DEFAULT_M if M is None else int(M))
        if self.lr_kernel == "sog":
            if not np.isfinite(self.b) or self.b <= 0.0:
                raise ValueError("`b` should be finite and positive.")
            if not np.isfinite(self.sigma) or self.sigma <= 0.0:
                raise ValueError("`sigma` should be finite and positive.")

            if bandwidth is None:
                bandwidth_array = self.sigma * np.power(
                    self.b, np.arange(self.M, dtype=self.prec)
                )
            else:
                bandwidth_array = np.asarray(bandwidth, dtype=self.prec).reshape(-1)
            if bandwidth_array.size < 1 or not np.all(np.isfinite(bandwidth_array)):
                raise ValueError("`bandwidth` should contain finite values.")
            if not np.all(bandwidth_array > 0.0):
                raise ValueError("`bandwidth` should contain positive values.")

            if amp is None:
                amp_array = np.full_like(bandwidth_array, 4.0 * np.pi * np.log(self.b))
            else:
                amp_array = np.asarray(amp, dtype=self.prec).reshape(-1)
            if amp_array.size == 1 and bandwidth_array.size > 1:
                amp_array = np.full_like(bandwidth_array, amp_array[0])
            elif amp_array.size != bandwidth_array.size:
                raise ValueError(
                    "`amp` should be scalar or have the same length as `bandwidth`."
                )
            if not np.all(np.isfinite(amp_array)):
                raise ValueError("`amp` should contain finite values.")

            self.bandwidth = bandwidth_array
            self.amp = amp_array
            self.M = int(bandwidth_array.size)
        else:
            # LES checkpoints created before SOG existed have no such keys.
            # Keep the unused SOG attributes out of the module state entirely.
            self.bandwidth = None
            self.amp = None

        trainable = (
            all(self.trainable)
            if isinstance(self.trainable, list)
            else bool(self.trainable)
        )
        if self.lr_kernel == "dipole":
            # Invariant descriptor-dependent channel weights contract the
            # exposed l=1 moments into one Cartesian dipole per atom.
            self.dipole_weight_net = FittingNet(
                self.dim_descrpt,
                self.embedding_width,
                self.neuron_lr,
                self.activation_function,
                self.resnet_dt,
                self.precision,
                bias_out=True,
                seed=child_seed(self.seed, 10_002),
                trainable=trainable,
            )
        else:
            self.dipole_weight_net = None

        if self.lr_update_steps > 0:
            self.polar_update_net = GLUFittingNet(
                self.dim_descrpt + 5,
                4,
                self.lr_update_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                bias_out=True,
                seed=child_seed(self.seed, 10_003),
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
            if alpha_pol_array.size == 1 and self.ntypes > 1:
                alpha_pol_array = np.full(
                    self.ntypes, alpha_pol_array[0], dtype=self.prec
                )
            elif alpha_pol_array.size != self.ntypes:
                raise ValueError(
                    "`alpha_pol` must be scalar or contain one value per atom type."
                )
        if not np.all(np.isfinite(alpha_pol_array)):
            raise ValueError("`alpha_pol` must contain finite values.")
        if not np.all(alpha_pol_array > 0.0):
            raise ValueError("`alpha_pol` must contain positive values.")
        self.alpha_pol = alpha_pol_array

    def output_def(self) -> FittingOutputDef:
        outputs = [
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
        if self.lr_kernel == "dipole":
            outputs.append(
                OutputVariableDef(
                    "latent_dipole",
                    [3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            )
        if self.lr_update_steps > 0:
            outputs.append(
                OutputVariableDef(
                    "lr_descriptor",
                    [self.dim_descrpt],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            )
        return FittingOutputDef(outputs)

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
        """Return energy, latent charges, and optional latent dipoles."""
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
        result = {
            **energy_ret,
            "latent_charge": latent_charge,
        }
        if self.lr_kernel == "dipole":
            if gr is None:
                raise RuntimeError(
                    "DPA4C dipole fitting requires exposed degree-one moments."
                )
            assert self.dipole_weight_net is not None
            net_dtype = get_xp_precision(xp, self.precision)
            weights = self.dipole_weight_net(xp.astype(descriptor, net_dtype))
            gr_t = xp.astype(gr, net_dtype)
            latent_dipole = xp.sum(weights[..., :, None] * gr_t, axis=-2)
            mask = xp.astype(self.emask.build_type_exclude_mask(atype), xp.bool)
            result["latent_dipole"] = xp.where(
                mask[..., None], latent_dipole, xp.zeros_like(latent_dipole)
            )
        if self.lr_update_steps > 0:
            result["lr_descriptor"] = descriptor
        return result

    def update_polar_state(
        self,
        descriptor: Any,
        charge: Any,
        dipole: Any,
        local_dipole: Any,
        potential: Any,
        field: Any,
    ) -> tuple[Any, Any]:
        """Apply one shared rotation-equivariant charge/dipole update."""
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
        return (
            q + self.lr_update_damping * dq,
            p + self.lr_update_damping * dmu,
        )

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
        if self.charge_bound is not None:
            lr_out = self.charge_bound * xp.tanh(lr_out / self.charge_bound)
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
        data["charge_bound"] = self.charge_bound
        data["use_glu"] = self.use_glu
        data["embedding_width"] = self.embedding_width
        data["use_self_pol"] = self.use_self_pol
        data["lr_update_steps"] = self.lr_update_steps
        data["lr_update_neuron"] = self.lr_update_neuron
        data["lr_update_damping"] = self.lr_update_damping
        data["b"] = self.b
        data["sigma"] = self.sigma
        data["M"] = self.M
        variables = data.setdefault("@variables", {})
        variables["bias_atom_q"] = to_numpy_array(self.bias_atom_q)
        variables["les_alpha"] = to_numpy_array(self.les_alpha)
        variables["alpha_pol"] = to_numpy_array(self.alpha_pol)
        if self.dipole_weight_net is not None:
            data["dipole_weight_net"] = self.dipole_weight_net.serialize()
        if self.polar_update_net is not None:
            data["polar_update_net"] = self.polar_update_net.serialize()
        if self.lr_kernel == "sog":
            variables["amp"] = to_numpy_array(self.amp)
            variables["bandwidth"] = to_numpy_array(self.bandwidth)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DPA4CLRFitting":
        """Deserialize the fitting."""
        data = data.copy()
        variables = data.get("@variables", {}).copy()
        bias_atom_q = variables.pop("bias_atom_q", None)
        les_alpha = variables.pop("les_alpha", None)
        amp = variables.pop("amp", None)
        bandwidth = variables.pop("bandwidth", None)
        alpha_pol = variables.pop("alpha_pol", None)
        nets_lr = data.pop("nets_lr", None)
        dipole_weight_net = data.pop("dipole_weight_net", None)
        polar_update_net = data.pop("polar_update_net", None)
        use_glu = data.get("use_glu", True)  # Default True for backward compatibility
        # Feed LR variables through the constructor. This preserves their
        # Parameter status when ``cls`` is the pt_expt torch wrapper; assigning
        # numpy arrays after construction would replace Parameters with buffers.
        if bias_atom_q is not None:
            data["bias_atom_q"] = np.asarray(bias_atom_q).tolist()
        if les_alpha is not None:
            data["les_alpha"] = float(np.asarray(les_alpha).reshape(-1)[0])
        if amp is not None:
            data["amp"] = np.asarray(amp).reshape(-1).tolist()
        if bandwidth is not None:
            data["bandwidth"] = np.asarray(bandwidth).reshape(-1).tolist()
        if alpha_pol is not None:
            data["alpha_pol"] = np.asarray(alpha_pol).reshape(-1).tolist()
        data["@variables"] = variables
        obj = super().deserialize(data)
        if nets_lr is not None:
            # Use the correct collection type based on use_glu
            if use_glu:
                obj.filter_layers_lr = SeZMNetworkCollection.deserialize(nets_lr)
            else:
                obj.filter_layers_lr = NetworkCollection.deserialize(nets_lr)
        if dipole_weight_net is not None:
            obj.dipole_weight_net = FittingNet.deserialize(dipole_weight_net)
        if polar_update_net is not None:
            obj.polar_update_net = GLUFittingNet.deserialize(polar_update_net)
        return obj
