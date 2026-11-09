# SPDX-License-Identifier: LGPL-3.0-or-later
"""Higher-degree (``l >= 3``) SO(3) constants for the density_basis head.

The SeZM/dpa4 descriptor latent only carries angular degrees ``l <= 2`` and
the hand-built Cartesian direction basis (:func:`build_cartesian_basis`)
likewise stops at ``l = 2``. To extend the equivariant density expansion to
``l = 3..6`` this module derives, numerically and deterministically, the two
families of constants the head needs, both expressed in the SeZM packed
Wigner-D convention:

* ``sh_transform[l]``: orthogonal change of basis mapping the packed real
  spherical harmonics of :func:`real_spherical_harmonics` onto the SeZM
  packed convention, so direction spherical harmonics contract
  rotation-invariantly with descriptor-latent coefficients.
* ``cg[(l1, l2, l3)]``: Clebsch-Gordan coupling tensors mapping packed
  coefficients of degrees ``l1`` and ``l2`` onto packed coefficients of
  degree ``l3``.

Both families are solved from linear conditions (the intertwining relation
and the harmonic product identity) sampled on deterministic grids, in
float64 on the CPU, and cross-validated against the analytic ``l = 1, 2``
Cartesian basis before being returned. Everything is built from fixed-seed
samples, so the results are identical across processes and reruns.

Why the numeric route: the within-degree ``m`` ordering/signs of the SeZM
packed basis are only defined implicitly by ``WignerDCalculator`` (plus the
``l1_perm``/``l1_sign`` calibration for ``l = 1``). Hand-deriving rank-3 to
rank-6 Cartesian basis tensors with matching conventions is error-prone;
solving the intertwining equations against the calculator itself, then
anchoring the solution to the validated ``l = 1, 2`` Cartesian blocks,
removes all convention guessing.
"""

from __future__ import (
    annotations,
)

import math
from functools import (
    cache,
)

import torch

from .cartesian import (
    build_cartesian_basis,
    build_edge_cartesian_tensors,
)
from .spherical import (
    SQRT_2,
    build_real_sh_norm,
    real_spherical_harmonics,
)
from .wignerd import (
    WignerDCalculator,
)

# coupling chain used by the density head to grow degrees from the l <= 2
# descriptor latent: l=3 from (1,2), l=4 from (2,2), then extend by (l-1,2)
_CG_CHAIN: dict[int, tuple[int, int]] = {3: (1, 2), 4: (2, 2), 5: (3, 2), 6: (4, 2)}

_BUILD_SEED = 20261106
_N_DIR = 2048
_N_ROT = 48
_N_CG_DIR = 4096


def _sample_dirs(n: int, generator: torch.Generator) -> torch.Tensor:
    dirs = torch.randn(
        n, 3, generator=generator, dtype=torch.float64, device=torch.device("cpu")
    )
    return dirs / dirs.norm(dim=-1, keepdim=True)


def _solve_intertwiner(
    d_in: list[torch.Tensor],
    d_out: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve ``T A_r = B_r T`` for the orthogonal ``(m, m)`` matrix ``T``.

    The equation is linear in ``T``; stacking it over sampled rotations gives
    an overdetermined homogeneous system whose one-dimensional null space
    (spanned by the intertwiner, unique up to sign for equivalent real
    irreps) is extracted with an SVD and cleaned up by polar
    orthogonalization.
    """
    m = d_in[0].shape[0]
    eye = torch.eye(m * m, dtype=torch.float64, device=torch.device("cpu")).reshape(
        m * m, m, m
    )
    rows = []
    for a_mat, b_mat in zip(d_in, d_out, strict=True):
        lin = torch.einsum("kij,jl->kil", eye, a_mat) - torch.einsum(
            "ij,kjl->kil", b_mat, eye
        )
        rows.append(lin.reshape(m * m, m * m))
    system = torch.cat(rows)
    _, singular, vh = torch.linalg.svd(system, full_matrices=False)
    transform = vh[-1].reshape(m, m)
    u_mat, _, vh2 = torch.linalg.svd(transform)
    transform = u_mat @ vh2
    return transform, singular


def _verify_intertwiner(
    transform: torch.Tensor,
    d_in: list[torch.Tensor],
    d_out: list[torch.Tensor],
) -> float:
    """Worst-case Frobenius residual of ``T A = B T`` over the samples."""
    worst = 0.0
    for a_mat, b_mat in zip(d_in, d_out, strict=True):
        residual = (transform @ a_mat - b_mat @ transform).norm().item()
        denom = a_mat.norm().item()
        worst = max(worst, residual / max(denom, 1e-30))
    return worst


def _cross_check_cartesian(
    ll: int,
    transform: torch.Tensor,
    norm: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Anchor ``T_l`` (``l = 1, 2``) to the validated Cartesian direction basis.

    Both the transformed spherical harmonics ``y @ T_l^T`` and the Cartesian
    projection of the edge tensors must give the same packed direction
    coordinates up to a per-degree scale (the overall scale is free and
    absorbed by the learnable radial weights; the sign is NOT free for
    ``l <= 2`` because the coefficients come from the descriptor latent in
    the fixed packed convention). Returns the sign-pinned transform.
    """
    dirs = _sample_dirs(512, generator)
    cart_basis = build_cartesian_basis(2, dtype=torch.float64, device=dirs.device)
    a0, s0 = build_edge_cartesian_tensors(dirs)
    if ll == 1:
        u_cart = torch.einsum("dij,eij->ed", cart_basis[1:4], a0 / math.sqrt(2.0))
    else:
        u_cart = torch.einsum("dij,eij->ed", cart_basis[4:9], s0 / math.sqrt(2.0 / 3.0))
    y = real_spherical_harmonics(2, dirs, norm, SQRT_2)
    y_l = y[:, ll * ll : (ll + 1) * (ll + 1)]
    u_sh = y_l @ transform.transpose(0, 1)
    cos = torch.nn.functional.cosine_similarity(u_sh.flatten(), u_cart.flatten(), dim=0)
    if cos.item() < 0.0:
        # the intertwiner null space is defined up to sign; pin it to the
        # Cartesian basis convention (the sign is NOT free for l <= 2 since
        # the coefficients come from the descriptor latent)
        transform = -transform
        cos = -cos
    if cos.item() < 1.0 - 1e-10:
        raise RuntimeError(
            f"l = {ll} SH transform does not match the Cartesian direction "
            f"basis (cos = {cos.item():.6e}); the SeZM packed convention is "
            "not matched"
        )
    return transform


def _solve_cg_packed(
    l1: int,
    l2: int,
    l3: int,
    transforms: dict[int, torch.Tensor],
    norm: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Clebsch-Gordan tensor in the SeZM packed convention.

    Solved from the harmonic product identity in the spherical-harmonics
    basis, ``Y_l1m1 * Y_l2m2 = sum_m3 CG[m1, m2, m3] Y_l3m3``, then conjugated
    into the packed basis by the transforms (inputs transform with
    ``T^{-1} = T^T``, outputs with ``T``).
    """
    dirs = _sample_dirs(_N_CG_DIR, generator)
    # the product Y_l1m1 * Y_l2m2 decomposes into every degree from
    # |l1 - l2| to l1 + l2, so the design matrix must span all of them;
    # the target l3 block is then extracted from the packed solution
    lmax_full = l1 + l2
    y = real_spherical_harmonics(lmax_full, dirs, norm, SQRT_2)
    y1 = y[:, l1 * l1 : (l1 + 1) * (l1 + 1)]
    y2 = y[:, l2 * l2 : (l2 + 1) * (l2 + 1)]
    prod = torch.einsum("pm,pn->pmn", y1, y2).reshape(y.shape[0], -1)
    # lstsq gives (sum(2L+1), m1*m2); the l3 rows are the CG coefficients
    sol = torch.linalg.lstsq(y, prod).solution
    lo = l3 * l3
    cg_y = sol[lo : lo + 2 * l3 + 1].T.reshape(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    residual = (y @ sol - prod).abs().max().item()
    if residual > 1e-9:
        raise RuntimeError(
            f"CG product identity for ({l1}, {l2}) -> {l3} unsolvable, "
            f"residual {residual:.3e}"
        )
    return torch.einsum(
        "abc,ma,nb,kc->mnk", cg_y, transforms[l1], transforms[l2], transforms[l3]
    )


@cache
def build_higher_l_constants(
    lmax: int,
) -> tuple[dict[int, torch.Tensor], dict[tuple[int, int, int], torch.Tensor]]:
    """
    Build the SH transforms and CG coupling tensors for degrees up to ``lmax``.

    Parameters
    ----------
    lmax : int
        Highest target degree, between 3 and 6 (the ``l <= 2`` path uses the
        analytic Cartesian basis instead).

    Returns
    -------
    tuple
        ``(transforms, couplings)`` with ``transforms[l]`` of shape
        ``(2*l + 1, 2*l + 1)`` (rows = packed SeZM coords, cols = spherical
        harmonics coords) and ``couplings[(l1, l2, l3)]`` of shape
        ``(2*l1 + 1, 2*l2 + 1, 2*l3 + 1)``. All tensors are float64 on the
        CPU; the caller casts and moves them.
    """
    if not 3 <= lmax <= 6:
        raise ValueError(f"lmax for higher_l constants must be in 3..6, got {lmax}")
    generator = torch.Generator(device="cpu").manual_seed(_BUILD_SEED)
    cpu = torch.device("cpu")
    f64 = torch.float64
    norm = build_real_sh_norm(lmax, device=cpu)

    dirs = _sample_dirs(_N_DIR, generator)
    y = real_spherical_harmonics(lmax, dirs, norm, SQRT_2)

    quats = torch.randn(
        _N_ROT, 4, generator=generator, dtype=f64, device=torch.device("cpu")
    )
    quats = quats / quats.norm(dim=-1, keepdim=True)
    calculator = WignerDCalculator(lmax, dtype=f64)
    # force the dense CPU path (the fused triton/cutile kernels only engage
    # for CUDA inputs) and keep the solve device-independent
    calculator = calculator.to(cpu)
    with torch.no_grad():
        d_full, _ = calculator(quats)
    d_full = d_full.to(cpu)

    cart_basis = build_cartesian_basis(2, dtype=f64, device=cpu)

    # calibrate the packed l = 1 block to a Cartesian rotation. The
    # calculator builds its l = 1 block as
    # ``D1 = diag(sign) P R P^T diag(sign)`` with ``P`` the ``l1_perm``
    # permutation and ``sign`` the ``l1_sign`` vector, so the physical
    # rotation represented by a packed block diagonal is
    # ``R = M^T D1 M`` with ``M = diag(sign) P``.
    l1_perm = calculator.l1_perm.cpu().to(torch.long)
    # the sign vector itself is not recoverable from ``l1_sign_outer`` (only
    # its squares appear there); the two hardcoded values below are pinned
    # instead by the Cartesian cross-checks of both the l = 1 and l = 2
    # blocks further down, which uniquely determine the calibration
    l1_sign = torch.tensor([-1.0, -1.0, 1.0], dtype=f64, device=cpu)
    perm_mat = torch.zeros(3, 3, dtype=f64, device=cpu)
    perm_mat[torch.arange(3, device=cpu), l1_perm] = 1.0
    sign_mat = torch.diag(l1_sign)
    cal_mat = sign_mat @ perm_mat
    d1 = d_full[:, 1:4, 1:4]
    rotations = cal_mat.T @ d1 @ cal_mat
    eye3 = torch.eye(3, dtype=f64, device=cpu)
    if not torch.allclose(
        rotations @ rotations.transpose(-1, -2), eye3, atol=1e-10
    ) or not torch.allclose(
        torch.linalg.det(rotations),
        torch.ones(_N_ROT, dtype=f64, device=cpu),
        atol=1e-10,
    ):
        raise RuntimeError(
            "Wigner-D l = 1 blocks do not calibrate to proper rotations; "
            "convention chain broken"
        )
    # cross-check the calibration against the analytic Cartesian basis:
    # B D = R^ B must hold for both the l = 1 and l = 2 blocks
    for ll in (1, 2):
        rows = slice(ll * ll, (ll + 1) * (ll + 1))
        lhs = torch.einsum("nij,anm->amij", cart_basis[rows], d_full[:, rows, rows])
        rhs = torch.einsum("xia,mab,xjb->xmij", rotations, cart_basis[rows], rotations)
        if not torch.allclose(lhs, rhs, atol=1e-9):
            raise RuntimeError(
                f"l = {ll} Cartesian basis fails the Wigner-D calibration; "
                "convention chain broken"
            )

    transforms: dict[int, torch.Tensor] = {}
    for ll in range(1, lmax + 1):
        sl = slice(ll * ll, (ll + 1) * (ll + 1))
        d_in: list[torch.Tensor] = []
        d_out: list[torch.Tensor] = []
        for rr in range(_N_ROT):
            rot = rotations[rr]
            y_rot = real_spherical_harmonics(lmax, dirs @ rot.T, norm, SQRT_2)
            # the SH basis transforms as y(R r) = D_sh(R) y(r), solved exactly
            # by least squares over the sampled directions
            d_in.append(torch.linalg.lstsq(y[:, sl], y_rot[:, sl]).solution)
            # the SH basis transforms as y(R r) = D_sh(R) y(r) in row
            # convention (an anti-homomorphism, D_sh = S^T); solving
            # ``T D_sh = D^T T`` is the transpose-equivalent of
            # ``M S = D M`` with ``M = T^T``, i.e. the transformed direction
            # coordinates must rotate by ``D`` exactly like the Cartesian
            # basis does; only this hypothesis yields an exact
            # one-dimensional intertwiner null space
            d_out.append(d_full[rr, sl, sl].T.contiguous())
        transform, singular = _solve_intertwiner(d_in, d_out)
        gap = singular[-2].item() / max(singular[-1].item(), 1e-30)
        if gap < 1e6:
            raise RuntimeError(
                f"l = {ll} intertwiner null space is not one-dimensional "
                f"(singular gap {gap:.3e})"
            )
        residual = _verify_intertwiner(transform, d_in, d_out)
        if residual > 1e-9:
            raise RuntimeError(
                f"l = {ll} SH transform fails the intertwining check, "
                f"residual {residual:.3e}"
            )
        if ll <= 2:
            transform = _cross_check_cartesian(ll, transform, norm, generator)
        else:
            # canonical sign for the degrees without a Cartesian anchor: the
            # overall sign cancels between the CG-built coefficients and the
            # direction coordinates, but pinning it keeps the buffers
            # deterministic across builds
            pivot = transform.abs().argmax()
            if transform.flatten()[pivot].item() < 0.0:
                transform = -transform
        transforms[ll] = transform

    couplings: dict[tuple[int, int, int], torch.Tensor] = {}
    for l3 in range(3, lmax + 1):
        l1, l2 = _CG_CHAIN[l3]
        couplings[(l1, l2, l3)] = _solve_cg_packed(
            l1, l2, l3, transforms, norm, generator
        )
    return transforms, couplings
