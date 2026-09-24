# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4C energy model with a non-periodic LES/SOG/dipole term (pt_expt).

The short-range part is the standard DPA4C graph-native energy model.  The
long-range correction is evaluated from per-atom latent charges and optional
latent dipoles produced by
:class:`deepmd.pt_expt.fitting.dpa4c_lr.DPA4CLRFitting`.

The graph lower consumes two edge sets:

* the short-range edges (within ``rcut``) used by the descriptor;
* the long-range edges (all pairs within each frame) used by the LR kernel.

Both edge sets are built eagerly outside the compiled region and passed in as
inputs, so the compiled graph lower contains no data-dependent Python ints.
The force is computed via two gradient paths:

* ``grad(E_SR + E_LR, edge_vec)`` gives the short-range force plus the
  charge-response term ``∂E_LR/∂q · dq/dR``;
* ``grad(E_LR, lr_edge_vec)`` gives the geometry term ``∂E_LR/∂r|_q``,
  assembled with the ``edge_force_virial`` conventions.

Only non-periodic systems (``box=None``) are supported.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import array_api_compat
import torch
import torch.nn as nn

from deepmd.dpmodel import (
    get_hessian_name,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    compact_nodes,
    edge_force_virial,
    expand_node_values,
    frame_id_from_n_node,
    segment_sum,
)
from deepmd.kernels.utils import (
    cuda_infer_level,
)
from deepmd.pt_expt.fitting.dpa4c_lr import (
    DPA4CLRFitting,
)
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
)
from deepmd.pt_expt.model.make_model import (
    _cal_hessian_ext_graph,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.utils.graph_builder import (
    build_neighbor_graph_for_method,
    build_ragged_neighbor_graph,
    resolve_neighbor_graph_method,
)

from .edge_transform_output import (
    fit_output_to_model_output_graph,
)
from .transform_output import (
    fit_output_to_model_output,
)

E2_PER_ANGSTROM_TO_EV = 14.3996454784255
_SQRT_PI = 1.7724538509055160272981674833411451827975494561224


def _les_radial_coefficients(
    r_sq: torch.Tensor,
    alpha_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stable ``K``, ``K'/r`` and ``(K''-K'/r)/r^2``."""
    alpha = alpha_t.to(dtype=r_sq.dtype, device=r_sq.device)
    r_sq_safe = r_sq.clamp_min(torch.finfo(r_sq.dtype).eps)
    r = torch.sqrt(r_sq_safe)
    x_sq = alpha * alpha * r_sq
    exp_t = torch.exp(-x_sq)
    erf_t = torch.erf(alpha * r)

    k_analytic = erf_t / r
    a_analytic = (2.0 * alpha / _SQRT_PI) * exp_t / r_sq_safe - erf_t / (r_sq_safe * r)
    kpp = (
        -(4.0 * alpha**3 / _SQRT_PI) * exp_t
        - (4.0 * alpha / _SQRT_PI) * exp_t / r_sq_safe
        + 2.0 * erf_t / (r_sq_safe * r)
    )
    c_analytic = (kpp - a_analytic) / r_sq_safe

    alpha2 = alpha * alpha
    alpha3 = alpha * alpha2
    alpha5 = alpha3 * alpha2
    alpha7 = alpha5 * alpha2
    alpha9 = alpha7 * alpha2
    r4 = r_sq * r_sq
    r6 = r4 * r_sq
    k_series = (2.0 / _SQRT_PI) * (
        alpha - alpha3 * r_sq / 3.0 + alpha5 * r4 / 10.0 - alpha7 * r6 / 42.0
    )
    a_series = (
        -4.0 * alpha3 / (3.0 * _SQRT_PI)
        + 4.0 * alpha5 * r_sq / (5.0 * _SQRT_PI)
        - 2.0 * alpha7 * r4 / (7.0 * _SQRT_PI)
        + 2.0 * alpha9 * r6 / (27.0 * _SQRT_PI)
    )
    c_series = (
        8.0 * alpha5 / (5.0 * _SQRT_PI)
        - 8.0 * alpha7 * r_sq / (7.0 * _SQRT_PI)
        + 4.0 * alpha9 * r4 / (9.0 * _SQRT_PI)
    )
    small = x_sq < 0.04
    return (
        torch.where(small, k_series, k_analytic),
        torch.where(small, a_series, a_analytic),
        torch.where(small, c_series, c_analytic),
    )


def _les_kernel_from_squared_distance(
    r_sq: torch.Tensor,
    les_alpha: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``erf(alpha * r) / r`` with a finite derivative at ``r=0``."""
    return _les_radial_coefficients(r_sq, les_alpha)[0]


def _les_potential_and_field_dense(
    r_ij: torch.Tensor,
    pair_mask: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor,
    alpha_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-atom potential and field for dense all-pairs input."""
    r_sq = torch.sum(r_ij * r_ij, dim=-1)
    kernel, a_coeff, c_coeff = _les_radial_coefficients(r_sq, alpha_t)
    q_src = q.squeeze(-1).unsqueeze(-2)
    p_src = p.unsqueeze(-3)
    p_src_dot_r = torch.sum(p_src * r_ij, dim=-1)
    mask = pair_mask.to(r_ij.dtype)
    potential = torch.sum(
        (q_src * kernel - a_coeff * p_src_dot_r) * mask,
        dim=-1,
        keepdim=True,
    )
    field_pair = (
        q_src.unsqueeze(-1) * a_coeff.unsqueeze(-1) * r_ij
        - a_coeff.unsqueeze(-1) * p_src
        - c_coeff.unsqueeze(-1) * p_src_dot_r.unsqueeze(-1) * r_ij
    )
    field = torch.sum(field_pair * mask.unsqueeze(-1), dim=-2)
    return potential, field


def _les_dipole_energy_dense(
    r_ij: torch.Tensor,
    pair_mask: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_pol_atom: torch.Tensor | None,
) -> torch.Tensor:
    """Return dense LES multipole energy, without the unit conversion."""
    r_sq = torch.sum(r_ij * r_ij, dim=-1)
    kernel, a_coeff, c_coeff = _les_radial_coefficients(r_sq, alpha_t)
    q_dst = q.squeeze(-1).unsqueeze(-1)
    q_src = q.squeeze(-1).unsqueeze(-2)
    p_dst_dot_r = torch.sum(p.unsqueeze(-2) * r_ij, dim=-1)
    p_src_dot_r = torch.sum(p.unsqueeze(-3) * r_ij, dim=-1)
    p_dot_p = torch.sum(p.unsqueeze(-2) * p.unsqueeze(-3), dim=-1)
    mask = pair_mask.to(r_ij.dtype)
    pair_energy = (
        q_dst * q_src * kernel
        + (q_src * p_dst_dot_r - q_dst * p_src_dot_r) * a_coeff
        - a_coeff * p_dot_p
        - c_coeff * p_dst_dot_r * p_src_dot_r
    )
    energy = 0.5 * torch.sum(pair_energy * mask, dim=(-2, -1))
    if alpha_pol_atom is not None:
        energy = energy + torch.sum(
            torch.sum(p * p, dim=-1) / (2.0 * alpha_pol_atom), dim=-1
        )
    return energy


def _disable_pointless_cumsum_pattern() -> None:
    """Remove Inductor's ``pointless_cumsum_replacement`` pattern.

    The pattern rewrites ``cumsum(full(shape, fill_value))`` into
    ``arange * fill_value``; it crashes when ``fill_value`` is a symbolic
    shape (``Node``) instead of a scalar.  DPA4C's local-frame construction
    hits exactly that case under symbolic tracing, so we drop the pattern.
    The pattern is a pure optimization; removing it is semantics-preserving.
    """
    try:
        from torch._inductor.fx_passes import (
            post_grad,
        )
    except Exception:
        return
    for pm in post_grad.pass_patterns:
        for key in list(pm.patterns.keys()):
            pm.patterns[key] = [
                p
                for p in pm.patterns[key]
                if getattr(getattr(p, "handler", None), "__name__", "")
                != "pointless_cumsum_replacement"
            ]


class _LESKernel(nn.Module):
    """Batched non-periodic LES charge-charge kernel."""

    def __init__(self, norm_factor: float = E2_PER_ANGSTROM_TO_EV) -> None:
        super().__init__()
        self.norm_factor = float(norm_factor)

    def forward(
        self,
        positions: torch.Tensor,
        latent_charges: torch.Tensor,
        les_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-frame LES energy.

        Parameters
        ----------
        positions
            Local atom coordinates, shape ``(nf, nloc, 3)``.
        latent_charges
            Per-atom latent charges, shape ``(nf, nloc, 1)``.
        les_alpha
            Scalar LES width parameter.

        Returns
        -------
        energy
            Per-frame energy, shape ``(nf,)``.
        """
        nloc = positions.shape[1]
        q = latent_charges.to(positions.dtype)
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(2)  # (nf,nloc,nloc,3)
        r_sq = torch.sum(r_ij * r_ij, dim=-1)
        eye = 1.0 - torch.eye(
            nloc, dtype=positions.dtype, device=positions.device
        ).unsqueeze(0)
        pair_q = q.unsqueeze(2) * q.unsqueeze(1)  # (nf,nloc,nloc,1)

        alpha_t = les_alpha.to(positions.dtype)
        kernel = _les_kernel_from_squared_distance(r_sq, alpha_t)
        kernel = kernel * eye
        pot = 0.5 * torch.sum(pair_q.squeeze(-1) * kernel, dim=(1, 2))  # (nf,)
        return pot * self.norm_factor


class _SOGKernel(nn.Module):
    """Batched non-periodic SOG kernel over distinct atom pairs ``i != j``."""

    def __init__(self, norm_factor: float = E2_PER_ANGSTROM_TO_EV) -> None:
        super().__init__()
        self.norm_factor = float(norm_factor)

    def forward(
        self,
        positions: torch.Tensor,
        latent_charges: torch.Tensor,
        amp: torch.Tensor,
        bandwidth: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-frame SOG energy."""
        nloc = positions.shape[1]
        q = latent_charges.to(dtype=positions.dtype, device=positions.device)
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(2)
        r_sq = torch.sum(r_ij * r_ij, dim=-1, keepdim=True)
        eye = 1.0 - torch.eye(
            nloc, dtype=positions.dtype, device=positions.device
        ).unsqueeze(0)
        pair_q = q.unsqueeze(2) * q.unsqueeze(1)

        amp_t = amp.to(dtype=positions.dtype, device=positions.device).view(1, 1, 1, -1)
        bandwidth_sq = (
            bandwidth.to(dtype=positions.dtype, device=positions.device)
            .square()
            .view(1, 1, 1, -1)
        )
        kernel = torch.sum(amp_t * torch.exp(-0.5 * r_sq / bandwidth_sq), dim=-1)
        kernel = kernel * eye
        pot = 0.5 * torch.sum(pair_q * kernel.unsqueeze(-1), dim=(1, 2, 3))

        return pot * self.norm_factor


@BaseModel.register("dpa4c_lr")
class DPA4CLREnergyModel(EnergyModel):
    """DPA4C model with a non-periodic LES, SOG, or dipole correction."""

    model_type = "dpa4c_lr"
    _needs_long_range_edges = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        fitting = self.get_fitting_net()
        if not isinstance(fitting, DPA4CLRFitting):
            raise TypeError(
                "DPA4CLREnergyModel requires a DPA4CLRFitting fitting net "
                f"(type 'dpa4c_lr'), got {type(fitting).__name__}."
            )
        self.lr_kernel = fitting.lr_kernel
        if self.lr_kernel not in ("les", "sog", "dipole"):
            raise NotImplementedError(
                f"Unsupported DPA4C-LR kernel {self.lr_kernel!r}."
            )
        if self.lr_kernel == "dipole" and getattr(
            self.get_descriptor(), "compress", False
        ):
            raise NotImplementedError(
                "A compressed DPA4C descriptor cannot be used with "
                "`lr_kernel='dipole'` because compression does not expose "
                "degree-one moments."
            )
        self.les_kernel = _LESKernel(norm_factor=E2_PER_ANGSTROM_TO_EV)
        self.sog_kernel = _SOGKernel(norm_factor=E2_PER_ANGSTROM_TO_EV)
        _disable_pointless_cumsum_pattern()

    def enable_compression(
        self,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Enable compression except when degree-one moments are required."""
        if self.lr_kernel == "dipole":
            raise NotImplementedError(
                "DPA4C compression does not expose degree-one moments and is "
                "therefore unavailable for `lr_kernel='dipole'`."
            )
        super().enable_compression(
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        )

    def _to_public_keys(
        self,
        model_ret: dict[str, torch.Tensor],
        do_atomic_virial: bool,
    ) -> dict[str, torch.Tensor]:
        """Carry the latent multipoles through to the public output."""
        model_predict = super()._to_public_keys(model_ret, do_atomic_virial)
        for key in ("latent_charge", "latent_dipole"):
            if key in model_ret:
                model_predict[key] = model_ret[key]
        return model_predict

    def call_common(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Reject periodic cells; both LR kernels are non-periodic."""
        if box is not None:
            xp = array_api_compat.array_namespace(box)
            if bool(xp.any(box != 0)):
                raise NotImplementedError(
                    "DPA4C-LR supports only non-periodic systems (box=None)."
                )
        return super().call_common(coord, atype, box=box, **kwargs)

    # ------------------------------------------------------------------
    # Graph lower with LES/SOG/dipole
    # ------------------------------------------------------------------

    def forward_common_lower_graph(
        self,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        n_local: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        destination_order: torch.Tensor | None = None,
        destination_row_ptr: torch.Tensor | None = None,
        source_order: torch.Tensor | None = None,
        source_row_ptr: torch.Tensor | None = None,
        destination_sorted: bool = False,
        do_atomic_virial: bool = False,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
        comm_dict: dict | None = None,
        lr_edge_index: torch.Tensor | None = None,
        lr_edge_vec: torch.Tensor | None = None,
        lr_edge_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Graph lower with the long-range correction.

        The short-range ``edge_*`` tensors and the long-range ``lr_edge_*``
        tensors are built eagerly by the caller and consumed as inputs, so the
        compiled graph lower contains no data-dependent neighbor-list
        construction.  The force is computed via two gradient paths:

        * ``grad(E_SR + E_LR, edge_vec)`` gives the short-range force plus the
          charge-response term ``∂E_LR/∂q · dq/dR``;
        * ``grad(E_LR, lr_edge_vec)`` gives the geometry term
          ``∂E_LR/∂r|_q``.
        """
        if lr_edge_index is None or lr_edge_vec is None or lr_edge_mask is None:
            raise ValueError(
                "DPA4C-LR requires long-range edges in the graph lower; the caller "
                "must pass lr_edge_index, lr_edge_vec, and lr_edge_mask."
            )

        # Short-range graph (edge_vec is the autograd leaf).
        edge_vec = edge_vec.detach().requires_grad_(True)
        if spin is not None:
            spin = spin.detach().requires_grad_(True)
        graph = NeighborGraph(
            n_node=n_node,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            n_local=n_local,
            destination_order=destination_order,
            destination_row_ptr=destination_row_ptr,
            source_order=source_order,
            source_row_ptr=source_row_ptr,
            destination_sorted=destination_sorted,
        )

        # Descriptor + fitting on the short-range graph.
        atomic_ret = self.atomic_model.forward_common_atomic_graph(
            graph,
            atype,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            spin=spin,
            comm_dict=comm_dict,
        )

        # LR energy on the long-range edge set.
        lr_edge_vec = lr_edge_vec.detach().requires_grad_(True)
        atomic_ret, e_lr = self._apply_lr_correction_graph(
            atomic_ret,
            atype,
            lr_edge_index,
            lr_edge_vec,
            lr_edge_mask,
            n_node,
            fparam=fparam,
            charge_spin=charge_spin,
        )

        # Short-range force + charge-response LR force via the short-range graph.
        model_ret = fit_output_to_model_output_graph(
            atomic_ret,
            self.atomic_output_def(),
            graph,
            do_atomic_virial=do_atomic_virial,
            create_graph=self.training,
            mask=atomic_ret.get("mask"),
            node_capacity=atype.shape[0],
            n_local=n_local,
            force_precision=edge_vec.dtype,
        )

        # Geometry LR force via the long-range graph.
        if e_lr is not None and self.do_grad_r("energy"):
            (g_lr,) = torch.autograd.grad(
                e_lr.sum(),
                lr_edge_vec,
                create_graph=self.training,
                retain_graph=True,
            )
            lr_force, lr_atom_virial, lr_virial = edge_force_virial(
                g_lr,
                lr_edge_vec,
                lr_edge_index,
                lr_edge_mask,
                n_node,
                node_capacity=atype.shape[0],
            )
            model_ret["energy_derv_r"] = model_ret["energy_derv_r"] + lr_force.reshape(
                -1, 1, 3
            ).to(model_ret["energy_derv_r"].dtype)
            if self.do_grad_c("energy"):
                nf = n_node.shape[0]
                model_ret["energy_derv_c_redu"] = model_ret[
                    "energy_derv_c_redu"
                ] + lr_virial.reshape(nf, 1, 9).to(
                    model_ret["energy_derv_c_redu"].dtype
                )
                if do_atomic_virial:
                    model_ret["energy_derv_c"] = model_ret[
                        "energy_derv_c"
                    ] + lr_atom_virial.reshape(-1, 1, 9).to(
                        model_ret["energy_derv_c"].dtype
                    )

        return model_ret

    def _apply_lr_correction_graph(
        self,
        atomic_ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
        lr_edge_index: torch.Tensor,
        lr_edge_vec: torch.Tensor,
        lr_edge_mask: torch.Tensor,
        n_node: torch.Tensor,
        fparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        """Compute the LES, SOG, or dipole energy on long-range edges.

        The long-range edges are built eagerly by the caller and passed in as
        inputs.  They contain only intra-frame all-pairs, so the storage scales
        with ``Σ_f n_f^2`` rather than the full batch ``N^2``.
        """
        if "latent_charge" not in atomic_ret:
            return atomic_ret, None

        fitting = self.get_fitting_net()
        nf = n_node.shape[0]
        N = atomic_ret["latent_charge"].shape[0]
        dtype = lr_edge_vec.dtype

        q = atomic_ret["latent_charge"]  # (N, dim)
        frame_id = frame_id_from_n_node(n_node, n_total=N)  # (N,)
        node_mask = atomic_ret.get("mask")
        if node_mask is None:
            node_mask = torch.ones(N, dtype=torch.bool, device=q.device)
        else:
            node_mask = node_mask.to(torch.bool)

        # Charge constraint (flat, vectorized).
        if fitting.use_charge_constraint:
            q = self._constrain_charges_flat(q, frame_id, n_node, fparam, charge_spin)

        src = lr_edge_index[0]
        dst = lr_edge_index[1]
        r_sq = torch.sum(lr_edge_vec * lr_edge_vec, dim=-1)  # (E_lr,)
        if self.lr_kernel in ("les", "dipole"):
            les_alpha = fitting.les_alpha
            if not isinstance(les_alpha, torch.Tensor):
                les_alpha = torch.tensor(
                    les_alpha, dtype=dtype, device=lr_edge_vec.device
                )
            else:
                les_alpha = les_alpha.to(dtype=dtype, device=lr_edge_vec.device)
            K, a_coeff, c_coeff = _les_radial_coefficients(r_sq, les_alpha)
        else:
            amp = fitting.amp.to(dtype=dtype, device=lr_edge_vec.device)
            bandwidth_sq = fitting.bandwidth.to(
                dtype=dtype, device=lr_edge_vec.device
            ).square()
            K = torch.sum(
                amp.unsqueeze(0)
                * torch.exp(-0.5 * r_sq.unsqueeze(-1) / bandwidth_sq.unsqueeze(0)),
                dim=-1,
            )
        if self.lr_kernel == "dipole":
            p = atomic_ret.get("latent_dipole")
            if p is None:
                raise RuntimeError(
                    "`lr_kernel='dipole'` requires DPA4C degree-one moments."
                )
            descriptor = atomic_ret.pop("lr_descriptor", None)
            local_p = p
            if fitting.lr_update_steps > 0:
                if descriptor is None:
                    raise RuntimeError(
                        "Dipole updates require the invariant LR descriptor."
                    )
                q_output_dtype = q.dtype
                p_output_dtype = p.dtype
                update_dtype = get_xp_precision(torch, fitting.precision)
                q = q.to(update_dtype)
                p = p.to(update_dtype)
                local_p = local_p.to(update_dtype)
                descriptor = descriptor.to(update_dtype)
                update_vec = lr_edge_vec.to(update_dtype)
                update_mask = lr_edge_mask.to(update_dtype)
                update_r_sq = torch.sum(update_vec * update_vec, dim=-1)
                update_alpha = les_alpha.to(update_dtype)
                update_k, update_a, update_c = _les_radial_coefficients(
                    update_r_sq, update_alpha
                )
                for _ in range(fitting.lr_update_steps):
                    q_src_update = q[src, 0]
                    p_src_update = p[src]
                    p_src_dot_r = torch.sum(p_src_update * update_vec, dim=-1)
                    potential_pair = (
                        q_src_update * update_k - update_a * p_src_dot_r
                    ) * update_mask
                    field_pair = (
                        q_src_update.unsqueeze(-1) * update_a.unsqueeze(-1) * update_vec
                        - update_a.unsqueeze(-1) * p_src_update
                        - update_c.unsqueeze(-1)
                        * p_src_dot_r.unsqueeze(-1)
                        * update_vec
                    ) * update_mask.unsqueeze(-1)
                    potential = segment_sum(potential_pair, dst, N).unsqueeze(-1)
                    field = segment_sum(field_pair, dst, N)
                    q, p = fitting.update_polar_state(
                        descriptor,
                        q,
                        p,
                        local_p,
                        potential,
                        field,
                    )
                    q = q * node_mask.unsqueeze(-1).to(q.dtype)
                    p = p * node_mask.unsqueeze(-1).to(p.dtype)
                    if fitting.use_charge_constraint:
                        q = self._constrain_charges_flat(
                            q, frame_id, n_node, fparam, charge_spin
                        )
                q = q.to(q_output_dtype)
                p = p.to(p_output_dtype)
            atomic_ret["latent_charge"] = q
            atomic_ret["latent_dipole"] = p

            q_src = q[src, 0].to(dtype)
            q_dst = q[dst, 0].to(dtype)
            p_src = p[src].to(dtype)
            p_dst = p[dst].to(dtype)
            p_src_dot_r = torch.sum(p_src * lr_edge_vec, dim=-1)
            p_dst_dot_r = torch.sum(p_dst * lr_edge_vec, dim=-1)
            pair_energy = (
                q_src * q_dst * K
                + (q_src * p_dst_dot_r - q_dst * p_src_dot_r) * a_coeff
                - a_coeff * torch.sum(p_src * p_dst, dim=-1)
                - c_coeff * p_src_dot_r * p_dst_dot_r
            )
            e_edge = 0.5 * pair_energy * lr_edge_mask.to(dtype)
        else:
            q_src = q[src]
            q_dst = q[dst]
            e_edge = 0.5 * (q_src * q_dst).sum(-1) * K * lr_edge_mask.to(dtype)
            atomic_ret["latent_charge"] = q

        edge_frame = frame_id[dst]  # (E_lr,) frame of dst
        e_lr_raw = segment_sum(e_edge, edge_frame, nf)
        if self.lr_kernel == "dipole" and fitting.use_self_pol:
            alpha_pol_atom = fitting.alpha_pol[atype].to(dtype).square()
            self_pol_atom = (
                torch.sum(p.to(dtype) * p.to(dtype), dim=-1)
                / (2.0 * alpha_pol_atom)
                * node_mask.to(dtype)
            )
            e_lr_raw = e_lr_raw + segment_sum(self_pol_atom, frame_id, nf)
        e_lr = e_lr_raw * E2_PER_ANGSTROM_TO_EV

        # Spread the frame LR energy evenly over the real atoms.
        nreal = n_node.to(e_lr.dtype).clamp_min(1)
        e_lr_atom = (e_lr / nreal)[frame_id].unsqueeze(-1)  # (N, 1)
        atomic_ret["energy"] = atomic_ret["energy"] + e_lr_atom.to(
            atomic_ret["energy"].dtype
        )
        return atomic_ret, e_lr

    def _constrain_charges_flat(
        self,
        q: torch.Tensor,
        frame_id: torch.Tensor,
        n_node: torch.Tensor,
        fparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shift the first latent-charge channel to match the target total charge.

        Flat (ragged) version: ``q`` is ``(N, dim)``, ``frame_id`` is ``(N,)``.
        """
        nf = n_node.shape[0]
        q_sum = segment_sum(q[:, 0], frame_id, nf)  # (nf,)
        nreal = n_node.to(q.dtype).clamp_min(1)
        if charge_spin is not None and charge_spin.shape[-1] > 0:
            target = charge_spin[:, 0].to(q.dtype)
        elif fparam is not None and self.get_dim_fparam() > 0:
            target = fparam[:, 0].to(q.dtype)
        else:
            target = torch.zeros(nf, dtype=q.dtype, device=q.device)
        correction = (q_sum - target) / nreal  # (nf,)
        q_new = q.clone()
        q_new[:, 0] = q[:, 0] - correction[frame_id]
        return q_new

    # ------------------------------------------------------------------
    # Eager path: pass coord into the graph lower
    # ------------------------------------------------------------------

    def call_common_ragged(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Ragged graph forward with the LR correction."""
        if not (self.mixed_types() and self.atomic_model.uses_graph_lower()):
            raise NotImplementedError(
                "a flat node axis requires a mixed_types descriptor with a graph lower"
            )
        if box is not None and torch.all(box == 0):
            box = None
        if box is not None:
            raise NotImplementedError(
                "DPA4C-LR supports only non-periodic systems (box=None)."
            )
        method = getattr(self, "neighbor_graph_method", None)
        if method is None:
            method = resolve_neighbor_graph_method("auto", coord.device)
        graph = build_ragged_neighbor_graph(
            method,
            coord,
            atype,
            n_node,
            box,
            self.get_rcut(),
            self.atomic_model.pair_excl,
        )
        # Long-range all-pairs graph, built eagerly outside the graph lower.
        lr_graph = build_ragged_neighbor_graph(
            method, coord, atype, n_node, None, 1e6, None
        )
        predict = self.forward_common_lower_graph(
            atype,
            graph.n_node,
            graph.n_node,
            graph.edge_index,
            graph.edge_vec,
            graph.edge_mask,
            graph.destination_order,
            graph.destination_row_ptr,
            graph.source_order,
            graph.source_row_ptr,
            destination_sorted=graph.destination_sorted,
            do_atomic_virial=do_atomic_virial,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            spin=spin,
            lr_edge_index=lr_graph.edge_index,
            lr_edge_vec=lr_graph.edge_vec,
            lr_edge_mask=lr_graph.edge_mask,
        )
        predict["n_node"] = graph.n_node
        return predict

    def _call_common_graph(
        self,
        cc: torch.Tensor,
        atype: torch.Tensor,
        bb: torch.Tensor | None,
        fp: torch.Tensor | None,
        ap: torch.Tensor | None,
        method: str,
        do_atomic_virial: bool = False,
        spin: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Carry-all graph forward with the LR correction.

        Mirrors the base :meth:`_call_common_graph` but builds the long-range
        all-pairs graph eagerly and passes it into
        :meth:`forward_common_lower_graph`.
        """
        if not (self.mixed_types() and self.atomic_model.uses_graph_lower()):
            raise NotImplementedError(
                "neighbor_graph_method requires a mixed_types descriptor with a "
                "graph lower."
            )
        if bb is not None and torch.all(bb == 0):
            bb = None
        if bb is not None:
            raise NotImplementedError(
                "DPA4C-LR supports only non-periodic systems (box=None)."
            )
        rcut = self.get_rcut()
        _desc = getattr(self.atomic_model, "descriptor", None)
        with_csr = (
            not self.training
            and cuda_infer_level() >= 1
            and _desc is not None
            and _desc.get_geo_compress()
        )
        pair_excl = self.atomic_model.pair_excl
        ng = build_neighbor_graph_for_method(
            method, cc, atype, bb, rcut, pair_excl, with_csr=with_csr
        )
        nf, nloc = atype.shape[:2]
        n_padded = nf * nloc
        atype_flat = atype.reshape(n_padded)
        ng, node_index = compact_nodes(ng, atype_flat >= 0)
        atype_flat = atype_flat[node_index]
        ap_flat = (
            ap.reshape(n_padded, ap.shape[-1])[node_index] if ap is not None else None
        )
        spin_flat = spin.reshape(n_padded, 3)[node_index] if spin is not None else None
        coord_flat = cc.reshape(n_padded, 3)[node_index]

        # Long-range all-pairs graph on the compact node axis.
        lr_graph = build_ragged_neighbor_graph(
            method, coord_flat, atype_flat, ng.n_node, None, 1e6, None
        )

        model_predict = self.forward_common_lower_graph(
            atype_flat,
            ng.n_node,
            ng.n_node,
            ng.edge_index,
            ng.edge_vec,
            ng.edge_mask,
            ng.destination_order,
            ng.destination_row_ptr,
            ng.source_order,
            ng.source_row_ptr,
            destination_sorted=ng.destination_sorted,
            do_atomic_virial=do_atomic_virial,
            fparam=fp,
            aparam=ap_flat,
            spin=spin_flat,
            charge_spin=charge_spin,
            lr_edge_index=lr_graph.edge_index,
            lr_edge_vec=lr_graph.edge_vec,
            lr_edge_mask=lr_graph.edge_mask,
        )

        # Unravel flat node-axis outputs back to the rectangular I/O boundary.
        N = node_index.shape[0]
        for k in list(model_predict.keys()):
            v = model_predict[k]
            if (
                v is not None
                and not k.endswith("_redu")
                and v.shape[:1] == torch.Size([N])
            ):
                model_predict[k] = expand_node_values(v, node_index, n_padded).reshape(
                    nf, nloc, *v.shape[1:]
                )

        # Graph-native Hessian, if enabled.
        aod = self.atomic_output_def()
        for kk in aod.keys():
            vdef = aod[kk]
            if vdef.reducible and vdef.r_hessian:
                model_predict[get_hessian_name(kk)] = _cal_hessian_ext_graph(
                    model=self,
                    kk=kk,
                    vdef=vdef,
                    coord=cc,
                    atype=atype,
                    box=bb,
                    fparam=fp,
                    aparam=ap,
                    spin=spin,
                    charge_spin=charge_spin,
                    method=method,
                    pair_excl=pair_excl,
                    rcut=rcut,
                    create_graph=self.training,
                )
        return model_predict

    # ------------------------------------------------------------------
    # Dense lower fallback (neighbor_graph_method="legacy")
    # ------------------------------------------------------------------

    def forward_common_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        extended_coord_corr: torch.Tensor | None = None,
        comm_dict: dict | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Dense lower with the LR correction folded in."""
        atomic_ret = self.atomic_model.forward_common_atomic(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
            charge_spin=charge_spin,
        )
        atomic_ret = self._apply_lr_correction(
            atomic_ret,
            extended_coord,
            extended_atype,
            fparam=fparam,
            charge_spin=charge_spin,
        )
        model_ret = fit_output_to_model_output(
            atomic_ret,
            self.atomic_output_def(),
            extended_coord,
            do_atomic_virial=do_atomic_virial,
            create_graph=self.training,
            mask=atomic_ret.get("mask"),
            extended_coord_corr=extended_coord_corr,
        )
        return model_ret

    def _apply_lr_correction(
        self,
        atomic_ret: dict[str, torch.Tensor],
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        fparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Add the LR energy to the per-atom short-range energy (dense path)."""
        if "latent_charge" not in atomic_ret:
            return atomic_ret

        fitting = self.get_fitting_net()
        nf, nloc = atomic_ret["energy"].shape[:2]
        cc = extended_coord[:, :nloc, :]
        atype = extended_atype[:, :nloc]
        q = atomic_ret["latent_charge"]

        mask = atomic_ret.get("mask")
        if mask is None:
            mask_bool = torch.ones(nf, nloc, dtype=torch.bool, device=cc.device)
        else:
            mask_bool = mask.to(torch.bool)

        if fitting.use_charge_constraint:
            q = self._constrain_charges(q, mask_bool, fparam, charge_spin)

        q = q * mask_bool.unsqueeze(-1).to(q.dtype)
        if self.lr_kernel in ("les", "dipole"):
            les_alpha = fitting.les_alpha
            if not isinstance(les_alpha, torch.Tensor):
                les_alpha = torch.tensor(
                    les_alpha,
                    dtype=cc.dtype,
                    device=cc.device,
                )
            else:
                les_alpha = les_alpha.to(dtype=cc.dtype, device=cc.device)
            if self.lr_kernel == "les":
                e_lr = self.les_kernel(cc, q, les_alpha)
            else:
                p = atomic_ret.get("latent_dipole")
                if p is None:
                    raise RuntimeError(
                        "`lr_kernel='dipole'` requires DPA4C degree-one moments."
                    )
                descriptor = atomic_ret.get("lr_descriptor")
                p = p * mask_bool.unsqueeze(-1).to(p.dtype)
                r_ij = cc.unsqueeze(1) - cc.unsqueeze(2)
                eye = ~torch.eye(nloc, dtype=torch.bool, device=cc.device).unsqueeze(0)
                pair_mask = eye & mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)
                local_p = p
                if fitting.lr_update_steps > 0:
                    if descriptor is None:
                        raise RuntimeError(
                            "Dipole updates require the invariant LR descriptor."
                        )
                    q_output_dtype = q.dtype
                    p_output_dtype = p.dtype
                    update_dtype = get_xp_precision(torch, fitting.precision)
                    q = q.to(update_dtype)
                    p = p.to(update_dtype)
                    local_p = local_p.to(update_dtype)
                    descriptor = descriptor.to(update_dtype)
                    update_r = r_ij.to(update_dtype)
                    for _ in range(fitting.lr_update_steps):
                        potential, field = _les_potential_and_field_dense(
                            update_r,
                            pair_mask,
                            q,
                            p,
                            les_alpha.to(update_dtype),
                        )
                        q, p = fitting.update_polar_state(
                            descriptor,
                            q,
                            p,
                            local_p,
                            potential,
                            field,
                        )
                        q = q * mask_bool.unsqueeze(-1).to(q.dtype)
                        p = p * mask_bool.unsqueeze(-1).to(p.dtype)
                        if fitting.use_charge_constraint:
                            q = self._constrain_charges(
                                q, mask_bool, fparam, charge_spin
                            )
                    q = q.to(q_output_dtype)
                    p = p.to(p_output_dtype)
                alpha_pol_atom = (
                    fitting.alpha_pol[atype].to(cc.dtype).square()
                    if fitting.use_self_pol
                    else None
                )
                e_lr = (
                    _les_dipole_energy_dense(
                        r_ij,
                        pair_mask,
                        q.to(cc.dtype),
                        p.to(cc.dtype),
                        les_alpha,
                        alpha_pol_atom,
                    )
                    * E2_PER_ANGSTROM_TO_EV
                )
                atomic_ret["latent_dipole"] = p
        else:
            e_lr = self.sog_kernel(cc, q, fitting.amp, fitting.bandwidth)

        atomic_ret["latent_charge"] = q

        # Evenly spread the frame LR energy over the real atoms so that the
        # per-atom sum matches the corrected total.
        nreal = mask_bool.sum(dim=1, keepdim=True).clamp_min(1).to(e_lr.dtype)
        e_lr_atom = (e_lr.view(nf, 1, 1) / nreal.view(nf, 1, 1)) * mask_bool.unsqueeze(
            -1
        ).to(e_lr.dtype)
        atomic_ret["energy"] = atomic_ret["energy"] + e_lr_atom.to(
            atomic_ret["energy"].dtype
        )
        return atomic_ret

    def _constrain_charges(
        self,
        q_rect: torch.Tensor,
        mask_rect: torch.Tensor,
        fparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shift the first latent-charge channel to match the target total charge."""
        nf, nloc, dim_out_lr = q_rect.shape
        if charge_spin is not None and charge_spin.shape[-1] > 0:
            target = charge_spin[:, 0].to(q_rect.dtype)
        elif fparam is not None and self.get_dim_fparam() > 0:
            target = fparam[:, 0].to(q_rect.dtype)
        else:
            target = torch.zeros(nf, dtype=q_rect.dtype, device=q_rect.device)

        nreal = mask_rect.sum(dim=1, keepdim=True).clamp_min(1).to(q_rect.dtype)
        q0 = q_rect[:, :, 0]
        q_sum = (q0 * mask_rect.to(q_rect.dtype)).sum(dim=1, keepdim=True)
        q_target_per_atom = target.view(nf, 1) / nreal
        q0 = q0 - (q_sum / nreal - q_target_per_atom) * mask_rect.to(q_rect.dtype)
        if dim_out_lr == 1:
            q_rect = q0.unsqueeze(-1)
        else:
            q_rect = torch.cat([q0.unsqueeze(-1), q_rect[:, :, 1:]], dim=-1)
        return q_rect
