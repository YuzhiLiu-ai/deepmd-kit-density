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

from deepmd.dpmodel.array_api import (
    Array,
    xp_add_at,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    KGrid,
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
            "the periodic multipole sum requires one charge channel, got "
            f"{n_channel}"
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
    potential = potential + xp.matmul(
        xp.matrix_transpose(sin_phase), weight * s_imag
    )
    potential = xp.sum(potential, axis=1)
    has_points = xp.astype(xp.any(k_grid.mask, axis=(1, 2)), dtype)[:, None, None]
    dipole_self = kernel.dipole_contact_value(charge)
    potential = potential - COULOMB_CONSTANT * (
        kernel.contact_value(charge) * charge_slot
    ) * has_points
    # F_s = -sum_k w k [sin Re(S) - cos Im(S)] + k_e K''(0) p_s
    mix = weight * (sin_phase * s_real - cos_phase * s_imag)
    gen_force = []
    for a in range(3):
        force_a = xp.matmul(
            xp.matrix_transpose(mix), (strained_k[..., a])[..., None]
        )
        gen_force.append(-xp.sum(force_a, axis=1))
    gen_force = xp.concat(gen_force, axis=-1)  # (nf, ns, 3)
    gen_force = gen_force + (
        COULOMB_CONSTANT * dipole_self * dipole_slot * has_points
    )
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
