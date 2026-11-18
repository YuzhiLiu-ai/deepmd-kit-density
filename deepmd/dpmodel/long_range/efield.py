# SPDX-License-Identifier: LGPL-3.0-or-later
"""External-field helpers for the long-range plugins.

The physical coupling of a uniform applied field to the latent charges is
the sawtooth (Umari--Pasquarello) potential: inside the cell it is a plain
ramp ``phi_ext(r) = -E . r`` and it jumps by the polarization quantum
``-E . L`` at the cell boundary, so it is periodic.  With coordinates
wrapped into the cell the ramp is just the wrapped coordinate itself; the
jump is implicit at the wrap plane and contributes no force (the ramp's
slope is the uniform field everywhere).

The potential is computed INSIDE the model lower from a coordinate leaf so
that the sawtooth force is autograd-exact (the charge-relaxation path
``sum_i phi_ext d q_i / d r`` is only visible when ``phi_ext`` is a
differentiable function of the geometry).  The caller therefore passes
GAUGE-FIXED coordinates:

- non-periodic: per-frame mean-subtracted coordinates (the mean drops out
  of the energy for the neutral charge constraint, and the subtraction
  makes the coupling exactly translation invariant);
- periodic: coordinates wrapped into the (orthorhombic) cell, where the
  wrap discontinuity IS the physical sawtooth jump.

For non-periodic neutral frames this is the pair-vector identity
``sum_i q_i r_i = (1/N) sum_{i<k} (q_i - q_k)(r_i - r_k)`` evaluated in the
centered frame.
"""

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.utils.neighbor_graph import (
    frame_id_from_n_node,
    segment_sum,
)

#: Modes for the charge-field coupling of ``latent_multipole``.
EFIELD_POTENTIAL_MODES = ("force_only", "sawtooth")


def resolve_efield_frame(efield_flag: bool, efield: Any, fparam: Any) -> Any:
    """Resolve the per-frame field vector, mirroring the plugin's rule.

    The explicit per-frame ``efield`` input (``(nf, 3)``) wins; otherwise
    the field rides ``fparam[:, :3]`` when the plugin is efield-enabled.
    Returns ``None`` when no field is active.
    """
    if not efield_flag:
        return None
    if efield is not None:
        return efield
    if fparam is not None:
        if fparam.shape[-1] < 3:
            raise ValueError(
                "`efield` reads the external field from fparam[:, :3], "
                f"but the given fparam has dimension {fparam.shape[-1]} (< 3)."
            )
        return fparam[..., :3]
    return None


def build_efield_coordinate(
    coord: Any,
    n_node: Any,
    box: Any,
    *,
    periodic: bool,
) -> Any:
    """Gauge-fixed coordinates for the sawtooth potential.

    Parameters
    ----------
    coord
        Node coordinates with shape ``(N, 3)``, frame-major over ``n_node``.
    n_node
        Per-frame node counts with shape ``(nf,)``.
    box
        Simulation cells with shape ``(nf, 3, 3)`` (rows are the lattice
        vectors), or ``None`` for non-periodic frames.
    periodic
        Whether the frames are periodic (wrap into the cell) or isolated
        (subtract the per-frame centroid).

    Returns
    -------
    Array
        The gauge-fixed coordinates with shape ``(N, 3)``.
    """
    import numpy as np

    xp = array_api_compat.array_namespace(coord)
    frame_id = frame_id_from_n_node(n_node, n_total=coord.shape[0])
    if periodic:
        if box is None:
            raise ValueError(
                "the sawtooth efield potential requires the simulation cell"
            )
        # Orthorhombicity gate on the host array (a validation, not part of
        # the differentiable graph): the jump plane must be perpendicular to
        # a lattice vector.
        box_np = np.asarray(box.detach().cpu() if hasattr(box, "detach") else box)
        if float(np.max(np.abs(box_np * (1.0 - np.eye(3))))) > 1e-8:
            raise NotImplementedError(
                "the sawtooth efield potential requires an orthorhombic cell: "
                "the jump plane must be perpendicular to a lattice vector; "
                "general triclinic cells are not supported"
            )
        length = xp.reshape(box, (box.shape[0], 9))[:, ::4]
        bcast = xp.reshape(xp.take(length, frame_id, axis=0), (coord.shape[0], 3))
        # The wrap discontinuity IS the physical sawtooth jump; its Jacobian
        # is the identity almost everywhere (the floor is piecewise constant),
        # so no further gauge fixing is applied here.
        return coord - xp.floor(coord / bcast) * bcast
    # Non-periodic: raw coordinates.  The per-frame centroid is subtracted
    # INSIDE the model lower (on the differentiable leaf) -- doing it here on
    # the data would hide the centroid's Jacobian (I - 1/N) from the force.
    return coord


def efield_potential_from_coordinate(
    coord: Any,
    n_node: Any,
    efield_frame: Any,
    *,
    center: bool,
) -> Any:
    """Sawtooth potential ``phi_ext = -E . coord`` per node, shape ``(N, 1)``.

    ``coord`` must be the gauge-fixed coordinates from
    :func:`build_efield_coordinate` (wrapped for periodic).  With
    ``center=True`` (non-periodic) the per-frame centroid is subtracted ON
    THE GRAPH, which makes the coupling exactly translation invariant and
    keeps the centroid's Jacobian inside the autograd force.  ``phi_ext`` in
    volts times a charge in electrons is an energy in eV.
    """
    xp = array_api_compat.array_namespace(coord)
    frame_id = frame_id_from_n_node(n_node, n_total=coord.shape[0])
    if center:
        centroid = (
            segment_sum(coord, frame_id, n_node.shape[0])
            / xp.astype(n_node, coord.dtype)[:, None]
        )
        coord = coord - xp.take(centroid, frame_id, axis=0)
    field_node = xp.take(efield_frame, frame_id, axis=0)
    return -xp.sum(
        xp.astype(coord, efield_frame.dtype) * field_node, axis=-1, keepdims=True
    )
