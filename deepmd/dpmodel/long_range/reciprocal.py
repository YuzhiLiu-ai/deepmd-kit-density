# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Reciprocal-space energy of latent charges and dipoles in periodic cells.

For the charges :math:`q_i^c` and dipoles :math:`p_i` of a periodic cell of
volume :math:`V` and a pair kernel :math:`K` with Fourier transform
:math:`\hat K`, the energy of the lattice of images reads

.. math::
    E = k_e \Big[ \frac{1}{2V} \sum_{\mathbf{k} \neq \mathbf{0}} \sum_c
    \hat K(\mathbf{k}) \, |S_c(\mathbf{k})|^2
    - \frac{1}{2} K(0) \sum_{i,c} (q_i^c)^2
    + \frac{1}{2} K''(0) \sum_i p_i^2 \Big],

with the structure factor :math:`S_c(\mathbf{k}) = \sum_i (q_i^c +
i\,\mathbf{k}\cdot\mathbf{p}_i)\, e^{i \mathbf{k} \cdot \mathbf{r}_i}`: a
dipole is the limit of two opposite charges, which shifts the structure
factor by :math:`i\,\mathbf{k}\cdot\mathbf{p}_i`. The :math:`\mathbf{k} =
\mathbf{0}` mode is dropped (tin-foil). The dipole self term follows from the
self content of the mesh sum,
:math:`\frac{1}{2}\int \hat K(\mathbf{k})(\mathbf{k}\cdot\mathbf{p})^2
d^3k/(2\pi)^3 = -\frac{1}{2}K''(0) p^2`, subtracted from the lattice sum as
the charge self :math:`\frac{1}{2}K(0)q^2` is.

The energy is attributed to the atoms as :math:`E_i = \frac{1}{2}[\sum_c
q_i^c \phi_i^c + \mathbf{p}_i \cdot \mathbf{F}_i]` with the self-corrected
potential :math:`\phi = \partial E / \partial q` and generalized force
:math:`\mathbf{F} = \partial E / \partial \mathbf{p}`; the electric field
driving the polar updates is :math:`-\mathbf{F}`.
"""

from typing import (
    Any,
)

import array_api_compat
import numpy as np

import math

from deepmd.dpmodel.array_api import (
    Array,
    xp_add_at,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    KGrid,
    MeshGrid,
)

from .kernel import (
    COULOMB_CONSTANT,
    BaseInteractionKernel,
)


def _strained_rows(xp: Any, rows: Array, strain: Array) -> Array:
    r"""Wave numbers of the strained cells, to first order in the strain.

    The strain :math:`\varepsilon` of a frame maps its cell :math:`B` to
    :math:`B S^\mathsf{T}`, :math:`S = I + \varepsilon`, and a wave number,
    a row :math:`\mathbf{k}`, to :math:`\mathbf{k} S^{-1}`. The strain of a
    topology is zero, and the energy is differentiated once with respect to
    it, for the virial; the value and the first derivative at zero strain
    are those of :math:`S^{-1} = I - \varepsilon`, which maps the rows here
    and keeps linear algebra out of the energy and its derivatives.
    """
    eps = xp.reshape(strain, (strain.shape[0],) + (1,) * (rows.ndim - 2) + (3, 3))
    return rows - xp.sum(rows[..., :, None] * eps, axis=-2)


def _strained_volume(volume: Array, strain: Array) -> Array:
    r"""Volumes of the strained cells, :math:`V (1 + \mathrm{tr}\,\varepsilon)` to first order."""
    return volume * (1.0 + strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2])


def _frame_slots(
    xp: Any, k_grid: KGrid, dtype: Any
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """The slots, geometry and strained wave numbers of a k-grid evaluation."""
    nf, n_slot = k_grid.slot.shape
    real = k_grid.slot >= 0
    index = xp.reshape(xp.where(real, k_grid.slot, xp.zeros_like(k_grid.slot)), (-1,))
    coord = xp.take(xp.astype(k_grid.coord, dtype), index, axis=0)
    coord = xp.reshape(coord, (nf, n_slot, 3))
    strain = xp.astype(k_grid.strain, dtype)
    k_vec = xp.astype(k_grid.vec, dtype)
    strained_k = _strained_rows(xp, k_vec, strain)
    volume = _strained_volume(xp.astype(k_grid.volume, dtype), strain)
    weight_mask = xp.astype(k_grid.mask, dtype)
    return real, index, coord, k_vec, strained_k, volume, weight_mask


def kspace_energy(
    charge: Array,
    k_grid: KGrid,
    kernel: BaseInteractionKernel,
) -> Array:
    r"""Energy of the periodic frames over explicit wave numbers, per atom.

    The structure factors of the wave numbers of every frame come from the
    phases of its nodes, and the potential at a node from the same phases:
    :math:`\phi_i = \frac{k_e}{V} \sum_{\mathbf{k}} \hat K(\mathbf{k})
    [\cos(\mathbf{k} \cdot \mathbf{r}_i) \mathrm{Re}\,S +
    \sin(\mathbf{k} \cdot \mathbf{r}_i) \mathrm{Im}\,S] - k_e K(0) q_i`.
    Every frame pairs the wave numbers of its row of :attr:`KGrid.vec` with
    the nodes of its row of :attr:`KGrid.slot`, in products batched over the
    frames and the blocks of their wave numbers. The geometry is strained by
    :attr:`KGrid.strain` (see :class:`KGrid`).

    Parameters
    ----------
    charge
        Node charges with shape ``(N, C)``, zero on the nodes that carry none.
    k_grid
        The wave numbers and the nodes of the periodic frames.
    kernel
        The pair kernel.

    Returns
    -------
    Array
        The energy of every node with shape ``(N,)``, in eV; zero in a frame
        without wave numbers.
    """
    xp = array_api_compat.array_namespace(charge)
    device = array_api_compat.device(charge)
    dtype = charge.dtype
    n_node, n_channel = charge.shape
    nf, n_slot = k_grid.slot.shape
    real, index, coord, k_vec, strained_k, volume, weight_mask = _frame_slots(
        xp, k_grid, dtype
    )
    charge_slot = xp.reshape(xp.take(charge, index, axis=0), (nf, n_slot, n_channel))
    charge_slot = charge_slot * xp.astype(real, dtype)[:, :, None]
    # === Structure factors, (nf, nb, kb, C) ===
    # The phase k' . r' = k . r does not change under the strain.
    phase = xp.matmul(k_vec, xp.matrix_transpose(coord)[:, None, :, :])
    cos_phase = xp.cos(phase)
    sin_phase = xp.sin(phase)
    s_real = xp.matmul(cos_phase, charge_slot[:, None, :, :])
    s_imag = xp.matmul(sin_phase, charge_slot[:, None, :, :])
    # === Potential at the nodes of every frame, (nf, ns, C) ===
    weight = COULOMB_CONSTANT * kernel.kspace(xp.sum(strained_k * strained_k, axis=-1))
    weight = weight / volume[:, None, None]
    weight = xp.where(k_grid.mask, weight, xp.zeros_like(weight))[..., None]
    block_potential = xp.matmul(xp.matrix_transpose(cos_phase), weight * s_real)
    block_potential = block_potential + xp.matmul(
        xp.matrix_transpose(sin_phase), weight * s_imag
    )
    potential = xp.sum(block_potential, axis=1)
    # The self term belongs to the lattice sum of a frame with wave numbers;
    # a non-periodic batch evaluated by a periodic module carries none.
    has_points = xp.astype(xp.any(k_grid.mask, axis=(1, 2)), dtype)[:, None, None]
    potential = (
        potential
        - COULOMB_CONSTANT * kernel.contact_value(charge) * charge_slot * has_points
    )
    slot_energy = 0.5 * xp.sum(charge_slot * potential, axis=-1)
    # === Energy on the node axis ===
    # A padding slot carries no charge and adds a zero to the first node.
    energy = xp.zeros((n_node,), dtype=dtype, device=device)
    return xp_add_at(energy, index, xp.reshape(slot_energy, (-1,)))


def kspace_multipole_potential_field(
    charge: Array,
    dipole: Array,
    k_grid: KGrid,
    kernel: BaseInteractionKernel,
) -> tuple[Array, Array, Array]:
    r"""Self-corrected potential, generalized force and energy of multipoles.

    Parameters
    ----------
    charge
        Node charges with shape ``(N, 1)``, zero on the nodes that carry none.
    dipole
        Node dipoles with shape ``(N, 3)``.
    k_grid
        The wave numbers and the nodes of the periodic frames.
    kernel
        The pair kernel.

    Returns
    -------
    potential
        :math:`\phi_i = \partial E / \partial q_i` with shape ``(N, 1)``.
    gen_force
        :math:`\mathbf{F}_i = \partial E / \partial \mathbf{p}_i` with shape
        ``(N, 3)``; the electric field driving the polar updates is
        ``-gen_force``.
    """
    xp = array_api_compat.array_namespace(charge)
    device = array_api_compat.device(charge)
    dtype = charge.dtype
    n_node, n_channel = charge.shape
    if n_channel != 1:
        raise NotImplementedError(
            f"the periodic multipole sum requires one charge channel, got {n_channel}"
        )
    nf, n_slot = k_grid.slot.shape
    real, index, coord, k_vec, strained_k, volume, weight_mask = _frame_slots(
        xp, k_grid, dtype
    )
    charge_slot = xp.reshape(xp.take(charge, index, axis=0), (nf, n_slot, 1))
    dipole_slot = xp.reshape(xp.take(dipole, index, axis=0), (nf, n_slot, 3))
    charge_slot = charge_slot * xp.astype(real, dtype)[:, :, None]
    dipole_slot = dipole_slot * xp.astype(real, dtype)[:, :, None]
    # Phases k . r are invariant under the strain (k' . r' = k . r), but the
    # k . p of the structure factor is not: the dipoles of the graph flow from
    # the descriptor, not from the strain leaf, so the reciprocal vectors of
    # the dipole response carry the first-order strain of the cells.
    phase = xp.matmul(k_vec, xp.matrix_transpose(coord)[:, None, :, :])
    k_dot_p = xp.matmul(strained_k, xp.matrix_transpose(dipole_slot)[:, None, :, :])
    cos_phase = xp.cos(phase)
    sin_phase = xp.sin(phase)
    # S = sum_s (q + i k.p) (cos + i sin): real = q cos - k.p sin
    s_real = xp.matmul(cos_phase, charge_slot[:, None, :, :]) - xp.sum(
        sin_phase * k_dot_p, axis=-1, keepdims=True
    )
    s_imag = xp.matmul(sin_phase, charge_slot[:, None, :, :]) + xp.sum(
        cos_phase * k_dot_p, axis=-1, keepdims=True
    )
    weight = COULOMB_CONSTANT * kernel.kspace(xp.sum(strained_k * strained_k, axis=-1))
    weight = weight / volume[:, None, None]
    weight = xp.where(k_grid.mask, weight, xp.zeros_like(weight))[..., None]
    # phi_s = sum_k w [cos Re(S) + sin Im(S)] - k_e K(0) q_s
    potential = xp.matmul(xp.matrix_transpose(cos_phase), weight * s_real)
    potential = potential + xp.matmul(xp.matrix_transpose(sin_phase), weight * s_imag)
    potential = xp.sum(potential, axis=1)
    has_points = xp.astype(xp.any(k_grid.mask, axis=(1, 2)), dtype)[:, None, None]
    dipole_self = kernel.dipole_contact_value(charge)
    potential = (
        potential
        - COULOMB_CONSTANT * (kernel.contact_value(charge) * charge_slot) * has_points
    )
    # F_s = -sum_k w k [sin Re(S) - cos Im(S)] + k_e K''(0) p_s
    mix = weight * (sin_phase * s_real - cos_phase * s_imag)
    gen_force = []
    for a in range(3):
        force_a = xp.matmul(xp.matrix_transpose(mix), (strained_k[..., a])[..., None])
        gen_force.append(-xp.sum(force_a, axis=1))
    gen_force = xp.concat(gen_force, axis=-1)  # (nf, ns, 3)
    gen_force = gen_force + (COULOMB_CONSTANT * dipole_self * dipole_slot * has_points)
    # Scatter back to the node axis. The potential and field of a padding slot
    # are NOT zero (they read the position of node 0 on the frame's own grid),
    # so they are zeroed first; otherwise they would leak into node 0.
    potential = potential * xp.astype(real, dtype)[:, :, None]
    gen_force = gen_force * xp.astype(real, dtype)[:, :, None]
    potential = xp_add_at(
        xp.zeros((n_node, 1), dtype=dtype, device=device),
        index,
        xp.reshape(potential, (-1, 1)),
    )
    gen_force = xp_add_at(
        xp.zeros((n_node, 3), dtype=dtype, device=device),
        index,
        xp.reshape(gen_force, (-1, 3)),
    )
    return potential, gen_force


def kspace_multipole_energy(
    charge: Array,
    dipole: Array,
    k_grid: KGrid,
    kernel: BaseInteractionKernel,
) -> Array:
    r"""Energy of the periodic multipole frames, per atom.

    Evaluates :math:`E_i = \frac{1}{2}[q_i \phi_i + \mathbf{p}_i \cdot
    \mathbf{F}_i]` from the self-corrected potential and generalized force of
    :func:`kspace_multipole_potential_field`, in eV, on the node axis.
    """
    potential, gen_force = kspace_multipole_potential_field(
        charge, dipole, k_grid, kernel
    )
    return 0.5 * (charge[:, 0] * potential[:, 0] + xp_sum_last(dipole, gen_force))


def xp_sum_last(dipole: Array, gen_force: Array) -> Array:
    """The per-node dipole-force contraction ``p . F``."""
    xp = array_api_compat.array_namespace(dipole, gen_force)
    return xp.sum(dipole * gen_force, axis=-1)


# ---------------------------------------------------------------------------
# Smooth particle-mesh Ewald
# ---------------------------------------------------------------------------


def bspline_integer_weights(order: int) -> np.ndarray:
    r"""The cardinal B-spline of order ``order`` at :math:`1, 2, \dots, p - 1`.

    These are the weights of the assignment factor
    :math:`D(\theta) = \sum_{l=0}^{p-2} M_p(l + 1) e^{i \theta l}` of the
    smooth particle-mesh Ewald method.
    """
    return np.asarray(
        [
            sum(
                (-1.0) ** j * math.comb(order, j) * max(l + 1.0 - j, 0.0) ** (order - 1)
                for j in range(order + 1)
            )
            / math.factorial(order - 1)
            for l in range(order - 1)
        ]
    )


def _bspline(xp: Any, x: Array, order: int) -> Array:
    r"""The cardinal B-spline :math:`M_p(x)` of order ``p``, elementwise."""
    value = None
    for j in range(order + 1):
        term = ((-1.0) ** j * math.comb(order, j)) * xp.maximum(
            x - j, xp.zeros_like(x)
        ) ** (order - 1)
        value = term if value is None else value + term
    return value / math.factorial(order - 1)


def _inverse3(xp: Any, m: Array) -> Array:
    """Inverse of a batch of 3x3 matrices: columns of cross products over det."""
    a, b, c = m[..., 0, :], m[..., 1, :], m[..., 2, :]
    bc = xp.linalg.cross(b, c)
    ca = xp.linalg.cross(c, a)
    ab = xp.linalg.cross(a, b)
    det = xp.sum(a * bc, axis=-1)
    return xp.stack([bc, ca, ab], axis=-1) / det[..., None, None]


def _stencil(
    xp: Any, frac: Array, dims: tuple[Any, Any, Any], order: int
) -> tuple[Array, Array]:
    r"""The mesh points every node spreads to and their B-spline weights.

    Returns the flat mesh indices and the weights, ``(N, order**3)`` each.
    """
    device = array_api_compat.device(frac)
    n_node = frac.shape[0]
    shift = xp.arange(order, dtype=frac.dtype, device=device)[None, :]
    offset = xp.arange(order, dtype=xp.int64, device=device)[None, :]
    index = None
    weight = None
    for d, n_d in enumerate(dims):
        u = frac[:, d] * n_d
        base = xp.floor(u)
        w = _bspline(xp, (u - base)[:, None] + (order - 1) - shift, order)
        m = (xp.astype(base, xp.int64)[:, None] - (order - 1) + offset) % n_d
        if index is None:
            index, weight = m, w
        else:
            index = xp.reshape(index[:, :, None] * n_d + m[:, None, :], (n_node, -1))
            weight = xp.reshape(weight[:, :, None] * w[:, None, :], (n_node, -1))
    return index, weight


def _assignment_factor(
    m_axes: list[Array], dims: tuple[Any, Any, Any], order: int
) -> list[Array]:
    r"""The inverse squared assignment factor :math:`|D_d(m)|^{-2}` of every axis."""
    xp = array_api_compat.array_namespace(m_axes[0])
    weights = bspline_integer_weights(order)
    factors = []
    for m, n_d in zip(m_axes, dims, strict=True):
        theta = (2.0 * math.pi) * m / n_d
        real = xp.zeros_like(theta)
        imag = xp.zeros_like(theta)
        for l, w in enumerate(weights):
            real = real + float(w) * xp.cos(theta * l)
            imag = imag + float(w) * xp.sin(theta * l)
        factors.append(1.0 / (real * real + imag * imag))
    return factors


def _mesh_setup(
    charge: Array,
    mesh: MeshGrid,
    frame_id: Array,
    order: int,
    kernel: BaseInteractionKernel,
) -> tuple:
    """Shared setup of a mesh evaluation: stencils, modes, influence pieces."""
    xp = array_api_compat.array_namespace(charge)
    dtype = charge.dtype
    m_full = [xp.astype(m, dtype) for m in (mesh.mode_1, mesh.mode_2, mesh.mode_3)]
    n1, n2, n3 = (m.shape[0] for m in m_full)
    box = xp.astype(mesh.box, dtype)
    volume = xp.astype(mesh.volume, dtype)
    # Fractional coordinates of the nodes; they do not depend on the strain.
    box_inv = _inverse3(xp, box)
    frac = xp.sum(
        xp.astype(mesh.coord, dtype)[:, :, None] * xp.take(box_inv, frame_id, axis=0),
        axis=1,
    )
    frac = frac - xp.floor(frac)
    index, weight = _stencil(xp, frac, (n1, n2, n3), order)
    # Influence pieces on the half spectrum.
    m3 = m_full[2][: n3 // 2 + 1]
    m_half = [m_full[0], m_full[1], m3]
    strain = xp.astype(mesh.strain, dtype)
    # The rows of A^T = B^-T S^-1 are the reciprocal vectors of the strained
    # cell, strained like wave numbers; W = A^T A.
    a_t = _strained_rows(xp, xp.matrix_transpose(box_inv), strain)
    metric = xp.sum(a_t[:, :, None, :] * a_t[:, None, :, :], axis=-1)
    volume = _strained_volume(volume, strain)
    g1 = m_half[0][:, None, None]
    g2 = m_half[1][None, :, None]
    g3 = m_half[2][None, None, :]
    monomials = (g1 * g1, g2 * g2, g3 * g3, g1 * g2, g1 * g3, g2 * g3)
    entries = (
        metric[:, 0, 0],
        metric[:, 1, 1],
        metric[:, 2, 2],
        2.0 * metric[:, 0, 1],
        2.0 * metric[:, 0, 2],
        2.0 * metric[:, 1, 2],
    )
    k_sq = None
    for entry, monomial in zip(entries, monomials, strict=True):
        term = entry[:, None, None, None] * monomial[None, ...]
        k_sq = term if k_sq is None else k_sq + term
    k_sq = (2.0 * math.pi) ** 2 * k_sq
    # The zero mode and the Nyquist planes carry no weight.
    keep = (
        (2.0 * xp.abs(g1) != n1) & (2.0 * xp.abs(g2) != n2) & (2.0 * xp.abs(g3) != n3)
    )
    keep = keep & ((g1 != 0.0) | (g2 != 0.0) | (g3 != 0.0))
    keep = keep[None, ...] & mesh.mask[:, None, None, None]
    safe_k_sq = xp.where(keep, k_sq, xp.ones_like(k_sq))
    rows = xp.reshape(safe_k_sq, (safe_k_sq.shape[0] * n1, n2 * (n3 // 2 + 1)))
    transform = xp.reshape(kernel.kspace(rows), safe_k_sq.shape)
    f1, f2, f3 = _assignment_factor(m_half, (n1, n2, n3), order)
    factor = f1[:, None, None] * f2[None, :, None] * f3[None, None, :]
    influence = (COULOMB_CONSTANT * transform * factor[None, ...]) / volume[
        :, None, None, None
    ]
    influence = xp.where(keep, influence, xp.zeros_like(influence))
    # k vectors of the half spectrum per frame, for the dipole response.
    k_half = (2.0 * math.pi) * xp.stack(
        [
            g1 * a_t[:, 0, 0, None, None, None]
            + g2 * a_t[:, 1, 0, None, None, None]
            + g3 * a_t[:, 2, 0, None, None, None],
            g1 * a_t[:, 0, 1, None, None, None]
            + g2 * a_t[:, 1, 1, None, None, None]
            + g3 * a_t[:, 2, 1, None, None, None],
            g1 * a_t[:, 0, 2, None, None, None]
            + g2 * a_t[:, 1, 2, None, None, None]
            + g3 * a_t[:, 2, 2, None, None, None],
        ],
        axis=-1,
    )
    return (
        xp,
        index,
        weight,
        influence,
        (n1, n2, n3),
        k_half,
        keep,
        dtype,
    )


def mesh_energy(
    charge: Array,
    mesh: MeshGrid,
    frame_id: Array,
    kernel: BaseInteractionKernel,
    order: int,
) -> Array:
    r"""Energy of the periodic frames by smooth particle-mesh Ewald, per atom.

    The charges spread to the mesh of every frame with B-splines of order
    ``order``; a real FFT gives the half spectrum of the charge mesh, the
    influence function :math:`G(\mathbf{m}) = \frac{k_e}{V}
    \hat K(\mathbf{k}_\mathbf{m}) \prod_d |D_d(m_d)|^{-2}` corrects for the
    spreading, and the potential at a node is the inverse transform
    interpolated by the same splines. The zero mode and the Nyquist planes
    carry no weight. See :func:`kspace_energy` for the lattice sum evaluated.
    """
    xp = array_api_compat.array_namespace(charge)
    device = array_api_compat.device(charge)
    dtype = charge.dtype
    nf = mesh.box.shape[0]
    n_node, n_channel = charge.shape
    n1, n2, n3 = 0, 0, 0
    (
        xp,
        index,
        weight,
        influence,
        (n1, n2, n3),
        k_half,
        keep,
        dtype,
    ) = _mesh_setup(charge, mesh, frame_id, order, kernel)
    del k_half
    n_mesh = n1 * n2 * n3
    # Spread: flat targets on the frame- and channel-major meshes.
    channel = xp.arange(n_channel, dtype=xp.int64, device=device)
    first = (frame_id[:, None] * n_channel + channel[None, :]) * n_mesh
    target = first[:, :, None] + index[:, None, :]
    q_mesh = xp.zeros((nf * n_channel * n_mesh,), dtype=dtype, device=device)
    q_mesh = xp_add_at(
        q_mesh,
        xp.reshape(target, (-1,)),
        xp.reshape(charge[:, :, None] * weight[:, None, :], (-1,)),
    )
    q_mesh = xp.reshape(q_mesh, (nf * n_channel, n1, n2, n3))
    spectrum = xp.fft.rfftn(q_mesh, axes=(-3, -2, -1))
    spectrum = xp.reshape(spectrum, (nf, n_channel, n1, n2, n3 // 2 + 1))
    potential_mesh = xp.fft.irfftn(
        spectrum * influence[:, None, ...],
        s=(n1, n2, n3),
        axes=(-3, -2, -1),
        norm="forward",
    )
    stencil_potential = xp.take(
        xp.reshape(potential_mesh, (-1,)), xp.reshape(target, (-1,)), axis=0
    )
    potential = xp.sum(
        xp.reshape(stencil_potential, (n_node, n_channel, -1)) * weight[:, None, :],
        axis=-1,
    )
    periodic_node = xp.take(mesh.mask, frame_id, axis=0)
    potential = potential - _self_potential(kernel, charge, periodic_node)
    return 0.5 * xp.sum(charge * potential, axis=-1)


def _self_potential(
    kernel: BaseInteractionKernel, charge: Array, periodic_node: Array
) -> Array:
    """The potential of every charge at itself, k_e K(0) q, on periodic nodes."""
    xp = array_api_compat.array_namespace(charge)
    return (
        COULOMB_CONSTANT
        * kernel.contact_value(charge)
        * charge
        * xp.astype(periodic_node, charge.dtype)[:, None]
    )


def mesh_multipole_potential_field(
    charge: Array,
    dipole: Array,
    mesh: MeshGrid,
    frame_id: Array,
    kernel: BaseInteractionKernel,
    order: int,
) -> tuple[Array, Array]:
    r"""Self-corrected potential and generalized force of multipoles on a mesh.

    The charges and the three dipole components spread to meshes with the
    same B-splines; the mesh structure factor is :math:`S(\mathbf{m}) =
    \hat Q(\mathbf{m}) + i\,\mathbf{k_m}\cdot\hat{\mathbf P}(\mathbf{m})`,
    since the dipole is the limit of two opposite charges. The potential is
    the inverse transform of :math:`G S` interpolated to the nodes, and the
    generalized force :math:`\mathbf F = \partial E/\partial \mathbf p`
    interpolates the inverse transform of :math:`-i k_a G S` per axis. The
    self terms are those of :func:`kspace_multipole_potential_field`.
    """
    xp = array_api_compat.array_namespace(charge)
    device = array_api_compat.device(charge)
    dtype = charge.dtype
    nf = mesh.box.shape[0]
    n_node, n_channel = charge.shape
    if n_channel != 1:
        raise NotImplementedError(
            f"the mesh multipole sum requires one charge channel, got {n_channel}"
        )
    (
        xp,
        index,
        weight,
        influence,
        (n1, n2, n3),
        k_half,
        keep,
        dtype,
    ) = _mesh_setup(charge, mesh, frame_id, order, kernel)
    n_mesh = n1 * n2 * n3
    # Spread the charge and the three dipole components, frame-major meshes.
    first = frame_id * n_mesh
    target = first[:, None] + index
    spread_weight = weight
    meshes = [charge[:, 0]]
    for a in range(3):
        meshes.append(dipole[:, a])
    spectra = []
    for values in meshes:
        mesh_flat = xp.zeros((nf * n_mesh,), dtype=dtype, device=device)
        mesh_flat = xp_add_at(
            mesh_flat,
            xp.reshape(target, (-1,)),
            xp.reshape(values[:, None] * spread_weight, (-1,)),
        )
        mesh_shaped = xp.reshape(mesh_flat, (nf, n1, n2, n3))
        spectra.append(xp.fft.rfftn(mesh_shaped, axes=(-3, -2, -1)))
    # S = Q - i k.P on the half spectrum: rfftn transforms with e^{-ik.r},
    # and the dipole charge density -div P transforms to -i k . P with it.
    spectrum = spectra[0]
    for a in range(3):
        spectrum = spectrum - 1j * k_half[..., a] * spectra[a + 1]
    # Potential mesh V = irfft(G S); field meshes X_a = irfft(-i k_a G S).
    convolved = influence * spectrum
    potential_mesh = xp.fft.irfftn(
        convolved, s=(n1, n2, n3), axes=(-3, -2, -1), norm="forward"
    )
    potential = _interpolate_mesh(xp, potential_mesh, target, spread_weight, n_node)
    periodic_node = xp.take(mesh.mask, frame_id, axis=0)
    potential = (
        potential
        - (COULOMB_CONSTANT * kernel.contact_value(charge) * charge)
        * xp.astype(periodic_node, dtype)[:, None]
    )
    gen_force = []
    for a in range(3):
        field_mesh = xp.fft.irfftn(
            (1j * k_half[..., a]) * convolved,
            s=(n1, n2, n3),
            axes=(-3, -2, -1),
            norm="forward",
        )
        force_a = _interpolate_mesh(xp, field_mesh, target, spread_weight, n_node)
        gen_force.append(force_a[:, 0])
    gen_force = xp.stack(gen_force, axis=-1)
    gen_force = (
        gen_force
        + (COULOMB_CONSTANT * kernel.dipole_contact_value(charge) * dipole)
        * xp.astype(periodic_node, dtype)[:, None]
    )
    return potential, gen_force


def _interpolate_mesh(
    xp: Any, mesh_values: Array, target: Array, weight: Array, n_node: int
) -> Array:
    """Interpolate a mesh back to the nodes with the spreading weights."""
    stencil_value = xp.take(
        xp.reshape(mesh_values, (-1,)), xp.reshape(target, (-1,)), axis=0
    )
    return xp.sum(xp.reshape(stencil_value, (n_node, -1)) * weight, axis=-1)[:, None]


def mesh_multipole_energy(
    charge: Array,
    dipole: Array,
    mesh: MeshGrid,
    frame_id: Array,
    kernel: BaseInteractionKernel,
    order: int,
) -> Array:
    r"""Energy of the periodic multipole frames on the mesh, per atom in eV."""
    potential, gen_force = mesh_multipole_potential_field(
        charge, dipole, mesh, frame_id, kernel, order
    )
    return 0.5 * (charge[:, 0] * potential[:, 0] + xp_sum_last(dipole, gen_force))
