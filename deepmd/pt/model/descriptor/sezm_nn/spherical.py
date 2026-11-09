# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Real spherical harmonics in the SeZM packed ``(l, m)`` layout.

The packed layout orders the coefficients degree-major, ``m = -l, ..., +l``
inside every degree-``l`` block, so component ``l*l + l + m`` is the order-``m``
harmonic and a degree-``lmax`` expansion has ``(lmax + 1) ** 2`` components.
The convention (z polar axis, ``phi = atan2(y, x)``, negative orders carry
``sin(m * phi)``, positive orders ``cos(m * phi)``) matches the internal
SO(3) basis of the SeZM/DPA4 descriptor, so contractions
``sum_m c_lm * Y_lm`` against descriptor latent coefficients are rotationally
invariant.
"""

import math

import torch

__all__ = [
    "SQRT_2",
    "associated_legendre_all",
    "build_real_sh_norm",
    "real_spherical_harmonics",
]

SQRT_2 = math.sqrt(2.0)


def build_real_sh_norm(lmax: int, *, device: torch.device) -> torch.Tensor:
    """Precompute real-spherical-harmonic normalization factors."""
    norm = torch.zeros(lmax + 1, lmax + 1, dtype=torch.float64, device=device)
    for l in range(lmax + 1):
        for m in range(l + 1):
            norm[l, m] = math.sqrt(
                (2 * l + 1)
                / (4.0 * math.pi)
                * math.exp(math.lgamma(l - m + 1) - math.lgamma(l + m + 1))
            )
    return norm


def associated_legendre_all(
    lmax: int,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate associated Legendre polynomials `P_l^m(x)` up to `lmax`.

    Parameters
    ----------
    lmax
        Maximum angular degree.
    x
        Cosine values with shape `(N,)`.

    Returns
    -------
    torch.Tensor
        Tensor with shape `(lmax + 1, lmax + 1, N)` where the second axis is
        `m`. Entries with `m > l` stay zero.
    """
    n_sample = x.shape[0]
    out = x.new_zeros((lmax + 1, lmax + 1, n_sample))
    out[0, 0] = 1.0
    if lmax == 0:
        return out

    sin_theta = torch.sqrt((1.0 - x * x).clamp_min(0.0))
    for m in range(1, lmax + 1):
        out[m, m] = -(2 * m - 1) * sin_theta * out[m - 1, m - 1]
    for m in range(lmax):
        out[m + 1, m] = (2 * m + 1) * x * out[m, m]
    for m in range(lmax + 1):
        for l in range(m + 2, lmax + 1):
            out[l, m] = (
                (2 * l - 1) * x * out[l - 1, m] - (l + m - 1) * out[l - 2, m]
            ) / float(l - m)
    return out


def real_spherical_harmonics(
    lmax: int,
    unit_vec: torch.Tensor,
    sh_norm: torch.Tensor,
    sqrt_2: torch.Tensor | float,
) -> torch.Tensor:
    """
    Compute packed real spherical harmonics in the SeZM `(l, m)` layout.

    Parameters
    ----------
    lmax
        Maximum angular degree.
    unit_vec
        Unit vectors with shape `(N, 3)`.
    sh_norm
        Normalization factors from :func:`build_real_sh_norm` with shape
        `(lmax + 1, lmax + 1)`.
    sqrt_2
        Precomputed `sqrt(2)` as a scalar tensor or float.

    Returns
    -------
    torch.Tensor
        Packed real spherical harmonics with shape `(N, (lmax + 1) ** 2)`.
    """
    x = unit_vec[:, 0]
    y = unit_vec[:, 1]
    z = unit_vec[:, 2].clamp(-1.0, 1.0)
    phi = torch.atan2(y, x)
    legendre = associated_legendre_all(lmax, z)

    out = unit_vec.new_zeros((unit_vec.shape[0], (lmax + 1) ** 2))
    for l in range(lmax + 1):
        for m in range(l + 1):
            base = legendre[l, m] * sh_norm[l, m]
            zero_idx = l * l + l
            if m == 0:
                out[:, zero_idx] = base
                continue
            sin_term = torch.sin(float(m) * phi)
            cos_term = torch.cos(float(m) * phi)
            out[:, zero_idx - m] = sqrt_2 * base * sin_term
            out[:, zero_idx + m] = sqrt_2 * base * cos_term
    return out
