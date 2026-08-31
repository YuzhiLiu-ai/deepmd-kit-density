# SPDX-License-Identifier: LGPL-3.0-or-later
"""SOG/LES long-range energy fitting for the SeZM/DPA4 model (pt backend).

This fitting extends :class:`SeZMEnergyFittingNet` with a second (long-range)
GLU branch that predicts per-atom latent charges, plus the trainable
long-range kernel parameters (SOG Gaussian levels, LES width, per-element
polarizabilities).  The actual long-range energy is evaluated by the model
layer (:class:`deepmd.pt.model.model.sezm_sog_model.SeZMSOGModel`); this
fitting only owns the readout heads and the kernel parameters.

Only non-periodic systems are supported: the kernel evaluated by the model
layer is the real-space O(N^2) pairwise kernel (SOG Gaussians and/or the
LES ``erf(alpha * r) / r`` kernel, optionally with smeared dipole terms).
"""

import logging
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.model.task.sezm_ener import (
    GLUFittingNet,
    SeZMEnergyFittingNet,
    SeZMNetworkCollection,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

log = logging.getLogger(__name__)

SOG_DEFAULT_B = to_numpy_array(np.array(1.62976708826776469))
SOG_DEFAULT_SIGMA = to_numpy_array(np.array(2.180230445405648))
SOG_DEFAULT_M = 12


@Fitting.register("sog_energy")
@fitting_check_output
class SOGEnergyFittingNet(SeZMEnergyFittingNet):
    """SeZM energy fitting with an additional long-range (latent charge) branch.

    The short-range branch is the standard SeZM GLU energy fitting.  The
    long-range branch predicts per-atom latent charges ``q_i`` (and,
    optionally, per-atom dipole vectors read from the SeZM equivariant
    latent) consumed by the SOG/LES kernel in the model layer.

    Parameters
    ----------
    dim_out_lr : int
        Latent charge dimension.  Must be 1 when ``lr_kernel="les"``.
    neuron_lr : list[int], optional
        Hidden widths of the long-range GLU branch.
    bias_atom_q : torch.Tensor, optional
        Per-element latent-charge bias, compressed with ``3 * tanh(x / 3)``.
        When None, a trainable :class:`TypeEmbedNet` is used instead.
    amp : float | list[float] | torch.Tensor, optional
        SOG amplitude for each Gaussian level. If None, initialized to
        ``4 * pi * log(b)`` for every level.
    bandwidth : list[float] | torch.Tensor, optional
        Base width ``sigma_m`` for each Gaussian level; stored directly and
        squared on-the-fly so it stays positive during training. If None,
        initialized to ``sigma * b**m`` for ``m = 0, ..., M-1``.
    les_alpha : float, optional
        Width parameter of the LES kernel ``erf(alpha * r) / r``. Defaults
        to 1.0.
    b : float
        Geometric base used by the SOG parameterization.
    sigma : float
        Base bandwidth used by the SOG parameterization.
    M : int
        Number of geometric bandwidth levels.
    remove_self_interaction : bool
        If True, remove the SOG self-interaction term.
    use_charge_constraint : bool
        Whether to enforce the sum of latent charges equals the target total
        charge (from fparam column 0).
    lr_kernel : str
        Long-range kernel for non-periodic systems: ``"les"`` applies only
        the LES ``erf(alpha*r)/r`` charge-charge kernel; ``"dipole"``
        additionally predicts a per-atom dipole vector from the SeZM
        equivariant latent (l=1 readout, rotationally covariant) and adds
        the smeared charge-dipole, dipole-dipole, and self-polarization
        terms; ``"sog"`` applies the SOG Gaussian kernel (with its
        self-interaction term unless ``remove_self_interaction=True``)
        without any LES part. ``"les"`` and ``"dipole"`` use physical
        scalar charges and require ``dim_out_lr=1``.
    alpha_pol : float | list[float] | torch.Tensor, optional
        Base polarizability per element for the dipole self-polarization
        term ``sum_i |p_i|^2 / (2 * alpha_pol_i)``. Stored directly and
        squared on-the-fly. A scalar is broadcast to all elements.
        Defaults to 1.0. Only used when ``lr_kernel="dipole"``.
    dipole_lmax : int, optional
        Spherical harmonic degree of the SeZM equivariant latent read by
        the dipole head. Wired automatically from the descriptor; must be
        >= 1. Only used when ``lr_kernel="dipole"``.
    dipole_channels : int, optional
        Channel count of the SeZM equivariant latent read by the dipole
        head. Wired automatically from the descriptor. Only used when
        ``lr_kernel="dipole"``.
    lr_update_steps : int, optional
        Number of recurrent long-range field updates applied to charge and
        dipole before evaluating the final energy. Defaults to zero for
        checkpoint compatibility and currently requires ``lr_kernel="dipole"``.
    lr_update_neuron : list[int], optional
        Hidden widths of the shared polar update GLU network.
    lr_update_damping : float, optional
        Residual damping applied to every charge and dipole update.
    """

    def __init__(
        self,
        *args: Any,
        dim_out_lr: int = 1,
        neuron_lr: list[int] | None = None,
        bias_atom_q: torch.Tensor | list[float] | None = None,
        amp: float | list[float] | torch.Tensor | None = None,
        bandwidth: list[float] | torch.Tensor | None = None,
        les_alpha: float | None = None,
        b: float | torch.Tensor | None = None,
        sigma: float | torch.Tensor | None = None,
        M: int | None = None,
        remove_self_interaction: bool = False,
        use_charge_constraint: bool = False,
        use_self_pol: bool = True,
        lr_kernel: str = "les",
        alpha_pol: float | list[float] | torch.Tensor | None = None,
        dipole_lmax: int | None = None,
        dipole_channels: int | None = None,
        lr_update_steps: int = 0,
        lr_update_neuron: list[int] | None = None,
        lr_update_damping: float = 0.2,
        **kwargs: Any,
    ) -> None:
        # NOTE: the config attributes consumed by ``output_def`` must be set
        # *before* ``super().__init__``: the ``fitting_check_output`` wrapper
        # around the parent classes calls ``self.output_def()`` during its own
        # ``__init__``, which resolves to the override below.
        self.dim_out_lr = int(dim_out_lr)
        if self.dim_out_lr < 1:
            raise ValueError(f"`dim_out_lr` should be positive, got {dim_out_lr}.")
        self.neuron_lr = (
            [120, 120, 120] if neuron_lr is None else [int(x) for x in neuron_lr]
        )
        self.use_charge_constraint = bool(use_charge_constraint)
        self.use_self_pol = bool(use_self_pol)
        self.bias_atom_q_bound = 3.0
        self.lr_update_steps = int(lr_update_steps)
        if self.lr_update_steps < 0:
            raise ValueError("`lr_update_steps` should be non-negative.")
        self.lr_update_neuron = (
            [64, 64]
            if lr_update_neuron is None
            else [int(value) for value in lr_update_neuron]
        )
        self.lr_update_damping = float(lr_update_damping)
        if not 0.0 < self.lr_update_damping <= 1.0:
            raise ValueError("`lr_update_damping` should be in (0, 1].")

        lr_kernel = str(lr_kernel).lower()
        if lr_kernel not in ("les", "dipole", "sog"):
            raise ValueError(
                f"`lr_kernel` should be 'les', 'dipole' or 'sog', got {lr_kernel!r}."
            )
        if lr_kernel in ("les", "dipole") and self.dim_out_lr != 1:
            raise ValueError(
                f"`lr_kernel='{lr_kernel}'` requires `dim_out_lr=1`: the LES "
                "kernel uses physical scalar charges (summing to the total "
                f"charge), but got `dim_out_lr={self.dim_out_lr}`. "
                "Use `lr_kernel='sog'` for multi-dimensional latent charges."
            )
        self.lr_kernel = lr_kernel

        # The dipole terms are selected by ``lr_kernel='dipole'``; the
        # ``use_dipole`` attribute is kept as the readout switch consulted by
        # the model layer (whether to feed the equivariant latent).
        self.use_dipole = lr_kernel == "dipole"
        if self.lr_update_steps > 0 and not self.use_dipole:
            raise ValueError(
                "`lr_update_steps > 0` currently requires `lr_kernel='dipole'`."
            )
        if self.use_dipole:
            if dipole_lmax is None or int(dipole_lmax) < 1:
                raise ValueError(
                    "`lr_kernel='dipole'` requires `dipole_lmax >= 1` (the dipole "
                    "head reads the l=1 block of the SeZM equivariant latent); "
                    f"got {dipole_lmax!r}."
                )
            if dipole_channels is None or int(dipole_channels) < 1:
                raise ValueError(
                    "`lr_kernel='dipole'` requires a positive `dipole_channels`; "
                    f"got {dipole_channels!r}."
                )
            self.dipole_lmax: int | None = int(dipole_lmax)
            self.dipole_channels: int | None = int(dipole_channels)
        else:
            self.dipole_lmax = None
            self.dipole_channels = None

        super().__init__(*args, **kwargs)
        if self.case_film_embd:
            raise NotImplementedError(
                "`sog_energy` fitting does not support `case_film_embd` yet."
            )

        # --- LR (latent charge) GLU branch ---
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + self.dim_case_embd
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
                    case_film_embd=False,
                )
                for idx in range(n_networks)
            ],
        )

        # --- bias_atom_q (LR-specific) ---
        if bias_atom_q is None:
            self.bias_atom_q: torch.nn.Parameter | TypeEmbedNet = TypeEmbedNet(
                type_nums=self.ntypes,
                embed_dim=self.dim_out_lr,
                precision=self.precision,
                seed=child_seed(self.seed, 10_001),
                type_map=self.type_map,
                trainable=self.trainable,
            )
        else:
            bias_atom_q_t = torch.as_tensor(
                bias_atom_q, dtype=self.prec, device=env.DEVICE
            ).view([self.ntypes, self.dim_out_lr])
            self.bias_atom_q = torch.nn.Parameter(
                bias_atom_q_t,
                requires_grad=bool(self.trainable),
            )

        # --- scalar hyper-parameters ---
        b_value = self._init_scalar_param(b, SOG_DEFAULT_B, name="b")
        sigma_value = self._init_scalar_param(sigma, SOG_DEFAULT_SIGMA, name="sigma")
        m_value = max(1, SOG_DEFAULT_M if M is None else int(M))

        # --- bandwidth vector ---
        if bandwidth is None:
            b_base = torch.tensor(b_value, dtype=self.prec, device=env.DEVICE)
            bandwidth_tensor = sigma_value * torch.pow(
                b_base,
                torch.arange(m_value, dtype=self.prec, device=env.DEVICE),
            )
        else:
            bandwidth_tensor = torch.as_tensor(
                bandwidth, dtype=self.prec, device=env.DEVICE
            ).reshape(-1)
        self._validate_1d_tensor(bandwidth_tensor, name="bandwidth", positive=True)

        # --- amplitude vector ---
        if amp is None:
            coef1 = float(4.0 * np.pi * np.log(b_value))
            amp_tensor = torch.full_like(bandwidth_tensor, coef1)
        else:
            amp_tensor = torch.as_tensor(
                amp, dtype=self.prec, device=env.DEVICE
            ).reshape(-1)
        self._validate_1d_tensor(amp_tensor, name="amp", positive=False)

        # Align scalar amp to match bandwidth length
        if amp_tensor.numel() == 1 and bandwidth_tensor.numel() > 1:
            amp_tensor = amp_tensor.expand_as(bandwidth_tensor).clone()
        elif amp_tensor.numel() != bandwidth_tensor.numel():
            raise ValueError(
                "`amp` should be scalar or have the same length as `bandwidth`."
            )

        self.amp = torch.nn.Parameter(amp_tensor, requires_grad=bool(self.trainable))
        self.bandwidth = torch.nn.Parameter(
            bandwidth_tensor, requires_grad=bool(self.trainable)
        )

        # LES (Latent Ewald Summation) width parameter for the non-periodic
        # 1/r kernel.  Stored as a scalar Parameter so it can be trained
        # together with the other SOG parameters.
        les_alpha_value = float(les_alpha) if les_alpha is not None else 1.0
        if les_alpha_value <= 0.0:
            raise ValueError("`les_alpha` should be positive.")
        self.les_alpha = torch.nn.Parameter(
            torch.tensor(les_alpha_value, dtype=self.prec, device=env.DEVICE),
            requires_grad=bool(self.trainable),
        )

        self.b = b_value
        self.sigma = sigma_value
        self.M = m_value
        self.remove_self_interaction = bool(remove_self_interaction)
        # Signal to the model layer that this fitting can consume the
        # frame-level charge/spin condition (charge_spin[:, 0] is the total
        # charge) for charge-constraint and other charge-aware paths.
        self.use_charge_spin = True

        # --- dipole head (optional; config validated above) ---
        if self.use_dipole:
            # Imported lazily: ``sezm_nn.dens`` itself depends on the task
            # package, so a top-level import would create a cycle.
            from deepmd.pt.model.descriptor.sezm_nn.dens import (
                _SeZMVectorHead,
            )

            assert self.dipole_lmax is not None and self.dipole_channels is not None
            self.dipole_head = _SeZMVectorHead(
                lmax=self.dipole_lmax,
                channels=self.dipole_channels,
                precision=self.precision,
                mlp_bias=False,
                trainable=bool(self.trainable),
                seed=child_seed(self.seed, 10_002),
            )
        else:
            self.dipole_head = None

        # A single shared update map is reused at every polarisation step, so
        # the recurrence has a meaningful fixed point.  Its five outputs are
        # dq, the Fukui log-weight, and three equivariant vector gates.
        if self.lr_update_steps > 0:
            self.polar_update_net: GLUFittingNet | None = GLUFittingNet(
                self.dim_descrpt + 5,
                5,
                self.lr_update_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                bias_out=True,
                seed=child_seed(self.seed, 10_003),
                trainable=self.trainable,
            )
            # The recurrent branch starts as the identity, which keeps old
            # predictions stable when fine-tuning a checkpoint.
            with torch.no_grad():
                self.polar_update_net.output_layer.matrix.zero_()
                if self.polar_update_net.output_layer.bias is not None:
                    self.polar_update_net.output_layer.bias.zero_()
        else:
            self.polar_update_net = None

        # --- per-element polarizability for the dipole self term ---
        if alpha_pol is None:
            alpha_pol_tensor = torch.ones(
                self.ntypes, dtype=self.prec, device=env.DEVICE
            )
        else:
            alpha_pol_tensor = torch.as_tensor(
                alpha_pol, dtype=self.prec, device=env.DEVICE
            ).reshape(-1)
            if alpha_pol_tensor.numel() == 1 and self.ntypes > 1:
                alpha_pol_tensor = alpha_pol_tensor.expand(self.ntypes).clone()
            elif alpha_pol_tensor.numel() != self.ntypes:
                raise ValueError(
                    "`alpha_pol` should be a scalar or have one value per "
                    f"element ({self.ntypes}), got {alpha_pol_tensor.numel()}."
                )
        self._validate_1d_tensor(alpha_pol_tensor, name="alpha_pol", positive=False)
        self.alpha_pol = torch.nn.Parameter(
            alpha_pol_tensor, requires_grad=bool(self.trainable)
        )

    # ------------------------------------------------------------------
    # Static helpers for SOG parameter initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_scalar_param(
        value: float | torch.Tensor | None,
        default: np.ndarray,
        name: str,
    ) -> float:
        """Parse a scalar SOG hyper-parameter with validation.

        Uses NumPy only so no tensor is created on the (possibly unset)
        default device; callers materialize tensors with explicit devices.
        """
        arr = np.asarray(default if value is None else value, dtype=np.float64).reshape(
            -1
        )
        if arr.size == 0:
            arr = np.asarray(default, dtype=np.float64).reshape(-1)
        result = float(arr[0])
        if result <= 0.0:
            raise ValueError(f"`{name}` should be positive.")
        return result

    @staticmethod
    def _validate_1d_tensor(
        tensor: torch.Tensor,
        name: str,
        positive: bool = True,
    ) -> None:
        """Validate a 1-D SOG parameter tensor."""
        if tensor.numel() == 0:
            raise ValueError(f"`{name}` should not be empty.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"`{name}` should be finite.")
        if positive and torch.any(tensor <= 0.0):
            raise ValueError(f"`{name}` values should be positive.")

    @staticmethod
    def _restore_param(
        obj: "SOGEnergyFittingNet",
        param_attr: str,
        new_tensor: torch.Tensor | None,
        target_shape: torch.Size,
        positive: bool = True,
    ) -> None:
        """Restore a single SOG parameter from a checkpoint tensor."""
        if new_tensor is None:
            return
        new = new_tensor.to(
            dtype=getattr(obj, param_attr).dtype,
            device=getattr(obj, param_attr).device,
        ).reshape(-1)
        SOGEnergyFittingNet._validate_1d_tensor(new, name=param_attr, positive=positive)

        old = getattr(obj, param_attr)
        if new.numel() == 1 and target_shape.numel() > 1:
            new = new.expand(target_shape.numel()).clone()
        elif new.numel() != target_shape.numel():
            raise ValueError(
                f"`{param_attr}` should be scalar or match the stored shape."
            )

        if old.shape != new.shape:
            setattr(
                obj,
                param_attr,
                torch.nn.Parameter(new, requires_grad=bool(obj.trainable)),
            )
        else:
            old.copy_(new)

    # ------------------------------------------------------------------
    # LR branch helpers
    # ------------------------------------------------------------------

    def _compress_bias_atom_q(self, bias: torch.Tensor) -> torch.Tensor:
        return self.bias_atom_q_bound * torch.tanh(bias / self.bias_atom_q_bound)

    def _get_lr_bias(self, atype: torch.Tensor) -> torch.Tensor:
        atype_long = atype.to(torch.long)
        if isinstance(self.bias_atom_q, torch.nn.Parameter):
            return self._compress_bias_atom_q(
                self.bias_atom_q[atype_long].to(self.prec)
            )
        return self._compress_bias_atom_q(self.bias_atom_q(atype_long).to(self.prec))

    def _forward_lr(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the long-range branch, returning per-atom latent charges.

        Mirrors the input assembly of the SR branch (descriptor + fparam +
        aparam + case embedding) so both GLU branches share the same input
        contract.
        """
        xx = descriptor.to(self.prec)
        nf, nloc, nd = xx.shape

        if self.numb_fparam > 0 and fparam is None:
            assert self.default_fparam_tensor is not None
            fparam = torch.tile(self.default_fparam_tensor.unsqueeze(0), [nf, 1])
        fparam = fparam.to(self.prec) if fparam is not None else None
        aparam = aparam.to(self.prec) if aparam is not None else None

        if self.numb_fparam > 0 and fparam is not None:
            target_total_charge = fparam.view([nf, self.numb_fparam])[:, 0]
        elif charge_spin is not None and charge_spin.shape[-1] > 0:
            target_total_charge = charge_spin[:, 0].to(
                dtype=self.prec, device=xx.device
            )
        else:
            target_total_charge = torch.zeros(nf, dtype=self.prec, device=xx.device)

        if nd != self.dim_descrpt:
            raise ValueError(
                f"get an input descriptor of dim {nd},"
                f"which is not consistent with {self.dim_descrpt}."
            )

        if self.numb_fparam > 0:
            assert fparam is not None
            assert self.fparam_avg is not None
            assert self.fparam_inv_std is not None
            fparam = fparam.view([nf, self.numb_fparam])
            nb, _ = fparam.shape
            t_fparam_avg = self._extend_f_avg_std(self.fparam_avg, nb)
            t_fparam_inv_std = self._extend_f_avg_std(self.fparam_inv_std, nb)
            fparam = (fparam - t_fparam_avg) * t_fparam_inv_std
            fparam = torch.tile(fparam.reshape([nf, 1, -1]), [1, nloc, 1])
            xx = torch.cat([xx, fparam], dim=-1)

        if self.numb_aparam > 0 and not self.use_aparam_as_mask:
            assert aparam is not None
            assert self.aparam_avg is not None
            assert self.aparam_inv_std is not None
            aparam = aparam.view([nf, -1, self.numb_aparam])
            nb, nloc, _ = aparam.shape
            t_aparam_avg = self._extend_a_avg_std(self.aparam_avg, nb, nloc)
            t_aparam_inv_std = self._extend_a_avg_std(self.aparam_inv_std, nb, nloc)
            aparam = (aparam - t_aparam_avg) * t_aparam_inv_std
            xx = torch.cat([xx, aparam], dim=-1)

        if self.dim_case_embd > 0:
            assert self.case_embd is not None
            case_embd = torch.tile(self.case_embd.reshape([1, 1, -1]), [nf, nloc, 1])
            xx = torch.cat([xx, case_embd], dim=-1)

        nf, nloc, _ = xx.shape
        lr_out = torch.zeros(
            (nf, nloc, self.dim_out_lr), dtype=self.prec, device=xx.device
        )
        if self.mixed_types:
            lr_out = lr_out + self.filter_layers_lr.networks[0](xx)
        else:
            for type_i, ll in enumerate(self.filter_layers_lr.networks):
                mask = (atype == type_i).unsqueeze(-1)
                atom_property = ll(xx)
                atom_property = torch.where(
                    mask.expand(-1, -1, self.dim_out_lr), atom_property, 0.0
                )
                lr_out = lr_out + atom_property

        lr_out = lr_out + self._get_lr_bias(atype)

        if self.use_charge_constraint:
            # (nf, 1, dim_out_lr): keepdim keeps the broadcast below clean.
            q_mean = lr_out.mean(dim=1, keepdim=True)
            # Divide by the (possibly symbolic) nloc instead of a Python float.
            q_target_per_atom = target_total_charge / lr_out.shape[1]
            # Only correct the first latent channel; keep the others intact.
            c0 = lr_out[:, :, 0:1] - (
                q_mean[:, :, 0:1] - q_target_per_atom[:, None, None]
            )
            crest = lr_out[:, :, 1:]
            lr_out = torch.cat([c0, crest], dim=-1)

        mask = self.emask(atype).to(torch.bool)
        lr_out = torch.where(mask[:, :, None], lr_out, 0.0)
        return lr_out

    def update_polar_state(
        self,
        descriptor: torch.Tensor,
        charge: torch.Tensor,
        dipole: torch.Tensor,
        local_dipole: torch.Tensor,
        potential: torch.Tensor,
        field: torch.Tensor,
        atom_mask: torch.Tensor,
        target_charge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one shared, rotation-equivariant charge/dipole update."""
        if self.polar_update_net is None:
            return charge, dipole

        # Keep the complete recurrent update in the fitting precision.  In
        # particular, do not cast the GLU output back to the global model
        # precision after every step: that creates a repeated fp32 -> fp64 ->
        # fp32 chain which is both unnecessary and problematic for Inductor's
        # fused GLU backward on CUDA.  The model casts the final state once at
        # the long-range-kernel boundary.
        dtype = self.prec
        # ``descriptor`` and the latent states can be views whose leading
        # stride depends on the padded nloc/nscatter of the current mixed-size
        # batch.  AOTAutograd saves some of these inputs for the force-loss
        # double backward; giving the GLU a canonical layout prevents its
        # compiled backward from retaining the trace batch's leading stride.
        descriptor = descriptor.to(dtype).contiguous()
        q = charge.to(dtype).contiguous()
        p = dipole.to(dtype).contiguous()
        p0 = local_dipole.to(dtype).contiguous()
        phi = potential.to(dtype).contiguous()
        electric = field.to(dtype).contiguous()

        p_sq = torch.sum(p * p, dim=-1, keepdim=True)
        e_sq = torch.sum(electric * electric, dim=-1, keepdim=True)
        p_scale = torch.sqrt(1.0 + p_sq)
        e_scale = torch.sqrt(1.0 + e_sq)
        p_unit = p / p_scale
        e_unit = electric / e_scale
        invariants = torch.cat(
            [
                descriptor,
                q,
                torch.tanh(phi),
                p_sq / (1.0 + p_sq),
                e_sq / (1.0 + e_sq),
                torch.sum(p_unit * e_unit, dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        update = self.polar_update_net(invariants)
        dq = torch.tanh(update[..., 0:1])
        fukui = torch.nn.functional.softplus(update[..., 1:2]) + 1e-8
        gates = torch.tanh(update[..., 2:5])
        dmu = gates[..., 0:1] * p + gates[..., 1:2] * e_unit + gates[..., 2:3] * p0

        mask = atom_mask.unsqueeze(-1)
        q_new = torch.where(
            mask,
            q + self.lr_update_damping * dq,
            torch.zeros_like(q),
        )
        p_new = torch.where(
            mask,
            p + self.lr_update_damping * dmu,
            torch.zeros_like(p),
        )
        if self.use_charge_constraint:
            weights = torch.where(mask, fukui, torch.zeros_like(fukui))
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            deficit = target_charge.to(dtype) - q_new.sum(dim=1, keepdim=True)
            q_new = q_new + weights * deficit
        return q_new, p_new

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def output_def(self) -> FittingOutputDef:
        defs = [
            OutputVariableDef(
                name="energy",
                shape=[1],
                reducible=True,
                r_differentiable=True,
                c_differentiable=True,
            ),
            OutputVariableDef(
                name="latent_charge",
                shape=[self.dim_out_lr],
                reducible=False,
                r_differentiable=False,
                c_differentiable=False,
            ),
        ]
        if self.use_dipole:
            defs.append(
                OutputVariableDef(
                    name="latent_dipole",
                    shape=[3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            )
        return FittingOutputDef(defs)

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        return_atomic_feature: bool = False,
        latent: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self._forward_common(
            descriptor,
            atype,
            gr,
            g2,
            h2,
            fparam,
            aparam,
            return_atomic_feature=return_atomic_feature,
        )
        result = {
            self.var_name: out[self.var_name].to(env.GLOBAL_PT_FLOAT_PRECISION),
        }
        if return_atomic_feature:
            result["atomic_feature"] = out["atomic_feature"]
        result["latent_charge"] = self._forward_lr(
            descriptor,
            atype,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
        ).to(env.GLOBAL_PT_FLOAT_PRECISION)
        if self.use_dipole:
            if latent is None:
                raise RuntimeError(
                    "`lr_kernel='dipole'` requires the SeZM equivariant latent "
                    "passed as `latent` (available in SeZMSOGModel paths)."
                )
            assert self.dipole_head is not None
            nf, nloc = atype.shape[:2]
            dipole = self.dipole_head(latent)  # (nf * nloc, 3)
            result["latent_dipole"] = dipole.view(nf, nloc, 3).to(
                env.GLOBAL_PT_FLOAT_PRECISION
            )
        return result

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "sog_energy"
        data["dim_out_lr"] = self.dim_out_lr
        data["neuron_lr"] = self.neuron_lr
        data["nets_lr"] = self.filter_layers_lr.serialize()
        variables = data["@variables"]
        variables["amp"] = to_numpy_array(self.amp)
        variables["bandwidth"] = to_numpy_array(self.bandwidth)
        variables["les_alpha"] = to_numpy_array(self.les_alpha)
        variables["alpha_pol"] = to_numpy_array(self.alpha_pol)
        if isinstance(self.bias_atom_q, torch.nn.Parameter):
            variables["bias_atom_q"] = to_numpy_array(self.bias_atom_q)
        else:
            for key, value in self.bias_atom_q.state_dict().items():
                variables[f"bias_atom_q_tebd.{key}"] = to_numpy_array(value)
        data["b"] = float(self.b)
        data["sigma"] = float(self.sigma)
        data["M"] = int(self.M)
        data["remove_self_interaction"] = bool(self.remove_self_interaction)
        data["use_charge_constraint"] = bool(self.use_charge_constraint)
        data["use_self_pol"] = bool(self.use_self_pol)
        data["lr_kernel"] = self.lr_kernel
        data["dipole_lmax"] = self.dipole_lmax
        data["dipole_channels"] = self.dipole_channels
        data["lr_update_steps"] = self.lr_update_steps
        data["lr_update_neuron"] = self.lr_update_neuron
        data["lr_update_damping"] = self.lr_update_damping
        if self.polar_update_net is not None:
            data["polar_update_net"] = self.polar_update_net.serialize()
        if self.dipole_head is not None:
            for key, value in self.dipole_head.state_dict().items():
                variables[f"dipole_head.{key}"] = to_numpy_array(value)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "SOGEnergyFittingNet":
        data = data.copy()
        variables = data.get("@variables", {}).copy()

        amp_tensor = to_torch_tensor(variables.pop("amp", None))
        bandwidth_tensor = to_torch_tensor(variables.pop("bandwidth", None))
        les_alpha_tensor = to_torch_tensor(variables.pop("les_alpha", None))
        alpha_pol_tensor = to_torch_tensor(variables.pop("alpha_pol", None))
        bias_atom_q_tensor = to_torch_tensor(variables.pop("bias_atom_q", None))
        tebd_state = {
            key[len("bias_atom_q_tebd.") :]: to_torch_tensor(variables.pop(key))
            for key in list(variables.keys())
            if key.startswith("bias_atom_q_tebd.")
        }
        dipole_head_state = {
            key[len("dipole_head.") :]: to_torch_tensor(variables.pop(key))
            for key in list(variables.keys())
            if key.startswith("dipole_head.")
        }
        polar_update_net = data.pop("polar_update_net", None)
        nets_lr = data.pop("nets_lr", None)
        data["@variables"] = variables

        obj = super().deserialize(data)

        if nets_lr is not None:
            obj.filter_layers_lr = SeZMNetworkCollection.deserialize(nets_lr)

        with torch.no_grad():
            target_shape = obj.bandwidth.shape
            cls._restore_param(obj, "bandwidth", bandwidth_tensor, target_shape)
            # Update M to match the restored bandwidth length.
            obj.M = int(obj.bandwidth.numel())
            cls._restore_param(obj, "amp", amp_tensor, target_shape, positive=False)
            cls._restore_param(
                obj, "alpha_pol", alpha_pol_tensor, obj.alpha_pol.shape, positive=False
            )

            if les_alpha_tensor is not None:
                new_alpha = float(
                    les_alpha_tensor.to(
                        dtype=obj.les_alpha.dtype, device=obj.les_alpha.device
                    )
                    .reshape(-1)[0]
                    .item()
                )
                if new_alpha <= 0.0:
                    raise ValueError("`les_alpha` should be positive.")
                obj.les_alpha.copy_(
                    torch.tensor(
                        new_alpha,
                        dtype=obj.les_alpha.dtype,
                        device=obj.les_alpha.device,
                    )
                )

            if bias_atom_q_tensor is not None:
                new_q = bias_atom_q_tensor.to(
                    dtype=obj.prec,
                    device=obj.bias_atom_q.device
                    if isinstance(obj.bias_atom_q, torch.nn.Parameter)
                    else env.DEVICE,
                ).view(obj.ntypes, obj.dim_out_lr)
                if isinstance(obj.bias_atom_q, torch.nn.Parameter):
                    obj.bias_atom_q.copy_(new_q)
                else:
                    obj.bias_atom_q = torch.nn.Parameter(
                        new_q, requires_grad=bool(obj.trainable)
                    )

        if tebd_state:
            if not isinstance(obj.bias_atom_q, TypeEmbedNet):
                raise ValueError(
                    "Serialized data contains a type-embedding charge bias but "
                    "`bias_atom_q` was given explicitly."
                )
            obj.bias_atom_q.load_state_dict(tebd_state)

        if dipole_head_state:
            if obj.dipole_head is None:
                raise ValueError(
                    "Serialized data contains dipole head weights but "
                    "`lr_kernel` is not 'dipole'."
                )
            obj.dipole_head.load_state_dict(dipole_head_state)

        if polar_update_net is not None:
            if obj.polar_update_net is None:
                raise ValueError(
                    "Serialized data contains a polar update network but "
                    "`lr_update_steps` is zero."
                )
            obj.polar_update_net = GLUFittingNet.deserialize(polar_update_net)

        return obj

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
