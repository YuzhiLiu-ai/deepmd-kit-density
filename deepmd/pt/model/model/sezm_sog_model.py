# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM/DPA4 model with a non-periodic SOG/LES long-range term (pt backend).

This module ports the long-range mechanism of the reference implementation
(``sog_model.py`` / ``sezm_sog_model.py``) onto the edge-schema force path of
:class:`deepmd.pt.model.model.sezm_model.SeZMModel`:

* :func:`_les_dipole_energy` evaluates the smeared charge-dipole,
  dipole-dipole and self-polarization terms sharing the LES Gaussian width.
* :class:`_SogLRKernel` is the batched, non-periodic real-space kernel
  (SOG Gaussian sum + optional self energy + LES ``erf(alpha r) / r``).
* :class:`SeZMSOGModel` adds the long-range energy/force/virial correction
  on top of the standard SeZM lower outputs.

The SeZM force path differentiates the energy with respect to ``edge_vec``;
the long-range energy instead depends on the absolute coordinates through
all O(N^2) pairs, which are not covered by the neighbor list.  The
correction therefore builds a separate autograd branch with respect to
``coord`` and adds the resulting force/virial onto the lower outputs, in
the same spirit as the reference ``_apply_frame_correction_lower``.

Only non-periodic systems (``box=None``) are supported.
"""

from typing import (
    Any,
)

import torch
import torch.nn as nn

from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.task.sog_energy_fitting import (
    SOGEnergyFittingNet,
    _materialize_compact,
)

from .sezm_model import (
    SeZMModel,
)

E2_PER_ANGSTROM_TO_EV = 14.3996454784255

_SQRT_PI = 1.7724538509055160272981674833411451827975494561224


def _les_radial_coefficients(
    r_sq: torch.Tensor,
    alpha_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stable ``K``, ``K'/r`` and ``(K''-K'/r)/r^2`` coefficients.

    Directly evaluating the derivatives of ``erf(alpha*r)/r`` subtracts
    individually divergent terms near the origin.  The small-distance Taylor
    branch below keeps both the forward field and the force-loss backward
    finite in float32.  The regular analytic branch uses a clamped radius so
    the unselected diagonal branch cannot create NaNs through ``torch.where``.
    """
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


def _les_potential_and_field(
    r_ij: torch.Tensor,
    eye: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor,
    alpha_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-atom LES potential and electric field from ``q``/``p``.

    The source multipoles use the same convention as
    :func:`_les_dipole_energy`.  Returned shapes are ``(..., n, 1)`` and
    ``(..., n, 3)`` and self interactions are excluded.
    """
    r_sq = torch.sum(r_ij * r_ij, dim=-1)
    kernel, a_coeff, c_coeff = _les_radial_coefficients(r_sq, alpha_t)
    mask = eye.to(dtype=r_ij.dtype)
    q_j = q.squeeze(-1).unsqueeze(-2)
    p_j = p.unsqueeze(-3)
    p_j_dot_r = torch.sum(p_j * r_ij, dim=-1)

    potential_pair = q_j * kernel - a_coeff * p_j_dot_r
    potential = torch.sum(potential_pair * mask, dim=-1, keepdim=True)

    field_pair = (
        q_j.unsqueeze(-1) * a_coeff.unsqueeze(-1) * r_ij
        - a_coeff.unsqueeze(-1) * p_j
        - c_coeff.unsqueeze(-1) * p_j_dot_r.unsqueeze(-1) * r_ij
    )
    field = torch.sum(field_pair * mask.unsqueeze(-1), dim=-2)
    return potential, field


def _les_dipole_energy(
    r_ij: torch.Tensor,
    r: torch.Tensor,
    eye: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_pol_atom: torch.Tensor | None = None,
) -> torch.Tensor:
    """LES dipole terms E_qp + E_pp + E_pol (without the unit norm factor).

    Implements the smeared multipole expansion with the shared Gaussian width
    ``alpha_t`` and kernel K(r) = erf(alpha r) / r:

    * charge-dipole:  0.5 * sum_{i!=j} (q_j p_i - q_i p_j) . r_hat K'(r)
    * dipole-dipole: -0.5 * sum_{i!=j} [ (K'/r)(p_i.p_j)
        + (K'' - K'/r)(p_i.r_hat)(p_j.r_hat) ]
    * self-polarization (optional): sum_i |p_i|^2 / (2 alpha_pol_i)

    All leading batch dimensions are preserved; the two trailing pair axes
    are reduced.

    Parameters
    ----------
    r_ij
        Pair displacement r_i - r_j with shape (..., n, n, 3).
    r
        Pair distances with shape (..., n, n), regularized so r > 0.
    eye
        Off-diagonal mask (0 on the diagonal, 1 elsewhere), shape (..., n, n).
    q
        Scalar latent charges with shape (..., n, 1).
    p
        Dipole vectors with shape (..., n, 3).
    alpha_t
        Shared LES Gaussian width (scalar tensor).
    alpha_pol_atom
        Per-atom polarizabilities with shape (..., n), or None to disable
        the self-polarization term.
    """
    if q.shape[-1] != 1:
        raise ValueError(
            "LES dipole terms require scalar latent charges (dim_out_lr=1), "
            f"got latent charge dim {q.shape[-1]}."
        )
    del r  # Stable coefficients are evaluated from squared displacements.
    r_sq = torch.sum(r_ij * r_ij, dim=-1)
    _, a_coeff, c_coeff = _les_radial_coefficients(r_sq, alpha_t)

    q_s = q.squeeze(-1)
    q_i = q_s.unsqueeze(-1)  # (..., n, 1)
    q_j = q_s.unsqueeze(-2)  # (..., 1, n)
    p_i_dot_r = torch.sum(p.unsqueeze(-2) * r_ij, dim=-1)  # p_i . r_ij
    p_j_dot_r = torch.sum(p.unsqueeze(-3) * r_ij, dim=-1)  # p_j . r_ij
    p_dot_p = torch.sum(p.unsqueeze(-2) * p.unsqueeze(-3), dim=-1)  # p_i . p_j

    e_qp = 0.5 * torch.sum(
        (q_j * p_i_dot_r - q_i * p_j_dot_r) * a_coeff * eye,
        dim=(-2, -1),
    )
    e_pp = -0.5 * torch.sum(
        (a_coeff * p_dot_p + c_coeff * p_i_dot_r * p_j_dot_r) * eye,
        dim=(-2, -1),
    )
    e_dip = e_qp + e_pp
    if alpha_pol_atom is not None:
        e_dip = e_dip + torch.sum(
            torch.sum(p * p, dim=-1) / (2.0 * alpha_pol_atom), dim=-1
        )
    return e_dip


class _SogLRKernel(nn.Module):
    """Batched, non-periodic SOG/LES long-range kernel.

    Real-space O(N^2) pairwise kernel with three modes selected by
    ``lr_kernel``:

    * ``"les"``: only the LES ``erf(alpha r) / r`` charge-charge term;
    * ``"dipole"``: the LES charge term plus the smeared charge-dipole,
      dipole-dipole and self-polarization terms (requires
      ``latent_dipoles``);
    * ``"sog"``: only the SOG Gaussian sum (with its self-interaction term
      unless ``remove_self_interaction=True``), no LES part.

    The trainable kernel parameters (``amp``/``bandwidth``/
    ``les_alpha``) are owned by the fitting net and passed at call time so
    they stay on the optimizer's parameter list.
    """

    def __init__(
        self,
        remove_self_interaction: bool = False,
        norm_factor: float = E2_PER_ANGSTROM_TO_EV,
        lr_kernel: str = "les",
    ) -> None:
        super().__init__()
        self.remove_self_interaction = bool(remove_self_interaction)
        self.norm_factor = float(norm_factor)
        lr_kernel = str(lr_kernel).lower()
        if lr_kernel not in ("les", "dipole", "sog"):
            raise ValueError(
                f"`lr_kernel` should be 'les', 'dipole' or 'sog', got {lr_kernel!r}."
            )
        self.lr_kernel = lr_kernel

    def forward(
        self,
        positions: torch.Tensor,
        latent_charges: torch.Tensor,
        amp: torch.Tensor,
        bandwidth: torch.Tensor,
        les_alpha: torch.Tensor,
        latent_dipoles: torch.Tensor | None = None,
        alpha_pol_atom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the long-range energy.

        Parameters
        ----------
        positions
            Local atom coordinates with shape (nf, nloc, 3).
        latent_charges
            Per-atom latent charges with shape (nf, nloc, nq).
        amp, bandwidth, les_alpha
            SOG/LES kernel parameters (trainable, owned by the fitting net).
        latent_dipoles
            Optional per-atom dipole vectors with shape (nf, nloc, 3);
            required when ``lr_kernel == "dipole"``, rejected otherwise.
        alpha_pol_atom
            Optional per-atom polarizabilities with shape (nf, nloc) used by
            the dipole self-polarization term.

        Returns
        -------
        torch.Tensor
            Per-frame long-range energy with shape (nf, 1).
        """
        r_raw = positions
        q = latent_charges.to(dtype=r_raw.dtype, device=r_raw.device)
        nf, nloc = r_raw.shape[:2]
        r_ij = r_raw.unsqueeze(1) - r_raw.unsqueeze(2)  # (nf, nloc, nloc, 3)
        r_sq = torch.sum(r_ij * r_ij, dim=-1, keepdim=True)  # (nf, nloc, nloc, 1)

        eye = 1.0 - torch.eye(nloc, dtype=r_raw.dtype, device=r_raw.device).unsqueeze(0)
        pair_q = q.unsqueeze(2) * q.unsqueeze(1)  # (nf, nloc, nloc, nq)

        if self.lr_kernel in ("les", "dipole"):
            # LES non-periodic charge-charge kernel:
            # E = 0.5 * C * sum_{i!=j} q_i q_j * erf(alpha * r_ij) / r_ij
            alpha_t = les_alpha.to(dtype=r_raw.dtype, device=r_raw.device)
            r = torch.sqrt(r_sq.squeeze(-1) + 1e-12)
            les_kernel, _, _ = _les_radial_coefficients(r_sq.squeeze(-1), alpha_t)
            les_kernel = les_kernel * eye
            pot = 0.5 * torch.sum(pair_q * les_kernel.unsqueeze(-1), dim=(1, 2, 3))
            if self.lr_kernel == "dipole":
                if latent_dipoles is None:
                    raise RuntimeError(
                        "`lr_kernel='dipole'` requires per-atom latent dipoles "
                        "(the `dipole` fitting head must be active)."
                    )
                pot = pot + _les_dipole_energy(
                    r_ij,
                    r,
                    eye,
                    q,
                    latent_dipoles.to(dtype=r_raw.dtype, device=r_raw.device),
                    alpha_t,
                    alpha_pol_atom=(
                        alpha_pol_atom.to(dtype=r_raw.dtype, device=r_raw.device)
                        if alpha_pol_atom is not None
                        else None
                    ),
                )
            elif latent_dipoles is not None:
                raise RuntimeError(
                    "Latent dipoles were provided but `lr_kernel` is 'les'; "
                    "use `lr_kernel='dipole'` for the dipole terms."
                )
            return (pot * self.norm_factor).view(nf, 1)

        # ``lr_kernel == "sog"``: SOG Gaussian sum only (no LES term).
        amp_t = amp.to(dtype=r_raw.dtype, device=r_raw.device).view(1, 1, 1, -1)
        bw2 = (
            bandwidth.to(dtype=r_raw.dtype, device=r_raw.device)
            .square()
            .view(1, 1, 1, -1)
        )
        kernel = amp_t * torch.exp(-0.5 * r_sq / bw2)  # (nf, nloc, nloc, M)
        kernel = kernel.sum(dim=-1)  # (nf, nloc, nloc)

        kernel = kernel * eye

        pot = 0.5 * torch.sum(pair_q * kernel.unsqueeze(-1), dim=(1, 2, 3))  # (nf,)

        if not self.remove_self_interaction:
            k0 = amp_t.sum()
            pot = pot + 0.5 * torch.sum(q * q, dim=(1, 2)) * k0

        return (pot * self.norm_factor).view(nf, 1)


@BaseModel.register("sezm_sog")
@BaseModel.register("dpa4_lr")
@BaseModel.register("sog_sezm")
class SeZMSOGModel(SeZMModel):
    """SeZM/DPA4 energy model with a non-periodic SOG/LES long-range term.

    The short-range part is the standard :class:`SeZMModel`; the long-range
    correction is evaluated from the per-atom latent charges (and optional
    latent dipoles) produced by :class:`SOGEnergyFittingNet` and added to
    the lower outputs through a dedicated coordinate autograd branch.

    Only non-periodic systems (``box=None``) are supported.  Both the eager
    and the ``use_compile`` execution paths are available.
    """

    model_type = "dpa4_lr"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        fitting = self.get_fitting_net()
        if not isinstance(fitting, SOGEnergyFittingNet):
            raise TypeError(
                "SeZMSOGModel requires a SOGEnergyFittingNet fitting "
                f"(fitting type 'sog_energy'), got {type(fitting).__name__}."
            )
        self.lr_kernel = fitting.lr_kernel
        self.sog_kernel = _SogLRKernel(
            remove_self_interaction=bool(fitting.remove_self_interaction),
            norm_factor=E2_PER_ANGSTROM_TO_EV,
            lr_kernel=self.lr_kernel,
        )

    def forward_common(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if box is not None:
            raise NotImplementedError(
                "SeZMSOGModel currently supports only non-periodic systems "
                "(box=None); periodic SOG/LES evaluation is not implemented."
            )
        return super().forward_common(coord, atype, box=box, **kwargs)

    def _lr_energy_for_edge_path(
        self,
        energy_redu: torch.Tensor,
        coord: torch.Tensor,
        atype: torch.Tensor,
        descriptor: torch.Tensor,
        fit_ret: dict[str, torch.Tensor],
        comm_dict: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, Any]:
        """Evaluate the SOG/LES energy and inject it into the edge backward.

        The kernel geometry is evaluated against a dedicated coordinate leaf
        (``coord_lr``) so the explicit coordinate dependence is collected by
        :meth:`_apply_lr_correction`; the latent charges / dipoles stay on
        the ``edge_vec`` graph, so their dependence on the local environment
        is differentiated by the shared ``edge_energy_deriv`` backward.
        """
        if "latent_charge" not in fit_ret:
            return energy_redu, None
        # Python-level check (not a symbolic shape comparison), safe under
        # make_fx tracing: ghost-node parallel inference is not supported.
        if comm_dict is not None:
            raise NotImplementedError(
                "SeZMSOGModel long-range correction currently supports only "
                "the single-domain path (no ghost atoms / comm_dict)."
            )
        fitting = self.atomic_model.fitting_net
        assert isinstance(fitting, SOGEnergyFittingNet)
        nf, nloc = atype.shape[:2]

        q = fit_ret["latent_charge"][:, :nloc, :]
        p = fit_ret.get("latent_dipole")
        p = p[:, :nloc, :] if p is not None else None
        alpha_pol_atom = (
            fitting.alpha_pol[atype].square()
            if p is not None and getattr(fitting, "use_self_pol", True)
            else None
        )

        coord_lr = coord[:, :nloc, :].detach().requires_grad_(True)
        if fitting.lr_update_steps > 0:
            if p is None:
                raise RuntimeError("Long-range polar updates require latent dipoles.")
            # Mirror the regular DPA4 fitting path: keep all recurrent neural
            # network work in fitting precision, then cross the global model
            # precision boundary only once.  Recasting q/p after every update
            # can make Inductor fuse a mixed fp32/fp64 GLU kernel that Triton
            # cannot lower on CUDA.
            q_output_dtype = q.dtype
            p_output_dtype = p.dtype
            update_dtype = fitting.prec
            q = _materialize_compact(q, update_dtype)
            p = _materialize_compact(p, update_dtype)
            coord_update = _materialize_compact(coord_lr, update_dtype)
            r_ij = coord_update.unsqueeze(1) - coord_update.unsqueeze(2)
            eye = 1.0 - torch.eye(
                nloc, dtype=update_dtype, device=coord_lr.device
            ).unsqueeze(0)
            local_p = p
            atom_mask = fit_ret["mask"][:, :nloc].to(torch.bool)
            target_charge = q.sum(dim=1, keepdim=True)
            for _ in range(fitting.lr_update_steps):
                potential, field = _les_potential_and_field(
                    r_ij,
                    eye,
                    q,
                    p,
                    fitting.les_alpha.to(update_dtype),
                )
                q, p = fitting.update_polar_state(
                    descriptor[:, :nloc, :],
                    q,
                    p,
                    local_p,
                    potential,
                    field,
                    atom_mask,
                    target_charge,
                )
            q = _materialize_compact(q, q_output_dtype)
            p = _materialize_compact(p, p_output_dtype)
            fit_ret["latent_charge"] = q
            fit_ret["latent_dipole"] = p
        e_lr = self.sog_kernel(
            coord_lr,
            q,
            fitting.amp,
            fitting.bandwidth,
            fitting.les_alpha,
            latent_dipoles=p,
            alpha_pol_atom=alpha_pol_atom,
        )  # (nf, 1)
        lr_ctx = {"e_lr": e_lr, "coord_lr": coord_lr}
        return energy_redu + e_lr.to(energy_redu.dtype), lr_ctx

    def _apply_lr_correction(
        self,
        model_ret: dict[str, torch.Tensor],
        coord: torch.Tensor,
        atype: torch.Tensor,
        fit_ret: dict[str, torch.Tensor],
        lr_ctx: Any,
        create_graph: bool,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Add the explicit-coordinate SOG/LES force and virial.

        The O(N^2) kernel geometry depends on the absolute coordinates, which
        are not covered by the ``edge_vec`` autograd branch; a dedicated
        ``autograd.grad`` with respect to the coordinate leaf produces this
        part of the force, and the virial follows the ``sum_i F_i (x) r_i``
        convention of the edge path (``W = -sum_e g_e (x) e_e``).
        """
        if lr_ctx is None:
            return model_ret
        e_lr: torch.Tensor = lr_ctx["e_lr"]
        coord_lr: torch.Tensor = lr_ctx["coord_lr"]
        nf, nloc = atype.shape[:2]

        # Per-atom even spread of the long-range energy (masked), so that
        # the ``atom_energy`` output sums to the corrected total.
        atom_mask = fit_ret["mask"]  # (nf, nloc)
        nreal = atom_mask.sum(dim=1, keepdim=True).clamp_min(1).to(e_lr.dtype)
        e_lr_atom = (e_lr.view(nf, 1, 1) / nreal.view(nf, 1, 1)).expand(nf, nloc, 1)
        model_ret["energy"] = model_ret["energy"] + e_lr_atom.to(
            model_ret["energy"].dtype
        ) * atom_mask.unsqueeze(-1).to(model_ret["energy"].dtype)

        # Explicit-coordinate force through the dedicated coordinate leaf.
        force_lr = -torch.autograd.grad(
            e_lr.sum(),
            coord_lr,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        # Virial: W = sum_i F_i (x) r_i.
        vir_lr = torch.einsum("fni,fnj->fnij", force_lr, coord_lr.detach())

        # Frame-keyed virial is shape-safe inside the (possibly traced) graph.
        model_ret["energy_derv_c_redu"] = model_ret["energy_derv_c_redu"] + vir_lr.sum(
            dim=1
        ).reshape(nf, 1, 9).to(model_ret["energy_derv_c_redu"].dtype)
        # The nscatter-keyed force / per-atom virial are merged in
        # ``forward_common_lower`` (outside the traced region): inside the
        # symbolic make_fx graph, ``coord``'s scatter axis and ``atype``'s
        # local axis are distinct symbols even though they are equal at
        # runtime on the single-domain path.
        model_ret["lr_force"] = force_lr
        model_ret["lr_virial_atom"] = vir_lr
        return model_ret
