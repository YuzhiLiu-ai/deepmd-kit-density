# SPDX-License-Identifier: LGPL-3.0-or-later
"""External-field helpers for the long-range plugins.

The physical coupling of a uniform applied field follows the modern
(Berry-phase) theory of polarization in its classical limit: the field
enters the enthalpy through the cell dipole on a CONTINUOUS branch,

    U_field = -E . (sum_i q_i r_i + sum_i p_i),

with the coordinates taken UNWRAPPED -- no cell wrapping anywhere in the
field path.  This is the classical counterpart of the Berry-phase
finite-field functional ``F(E) = E_0 - Omega E . P`` used by CP2K's
``PERIODIC_EFIELD`` (which is a Berry-phase implementation, not a
sawtooth potential).  Keeping the branch continuous makes the field
response conservative across cell crossings: translating an atom or a
whole neutral molecule by a lattice vector changes ``U_field``
continuously, and the force stays the uniform ``q_i E`` everywhere --
no sawtooth discontinuity, no polarization-quantum bookkeeping in the
dynamics, and no artificial dipoles when a neutral molecule straddles
the cell edge (per-atom wrapping would create ``-q_i E L`` energy
spikes for partial crossings).

For a neutral frame (``sum q = 0``, enforced by the charge constraint)
the total is invariant under a global translation; for a charged frame
the coupling is the physical origin-dependent dipole term.  Only the
LEF's charge-equilibration input is gauge-fixed (per-frame mean
removed): a constant potential shift does not change the QEq driving,
and the small values keep the update net responsive.
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
#: "branch" couples the field through the charge potential on the
#: continuous polarization branch (conservative, needs unwrapped
#: coordinates); "bec" adds the charge-field energy on the pair
#: center-of-geometry identity, -(1/N) sum_pairs (E.r_ij)(q_i-q_j),
#: which is exact for neutral frames on the complete pair set and whose
#: autograd carries the rigid q_i E force plus the full charge response
#: (conservative on non-periodic frames, no coordinate leaf, no
#: unwrapped coordinates; rejected for periodic plugins, where the pair
#: topology is empty); "force_only" is the legacy coupling that only
#: adds the uniform force q_i E.
EFIELD_POTENTIAL_MODES = ("force_only", "branch", "bec")


def normalize_efield_potential(mode: str) -> str:
    """Map deprecated mode names to their modern spelling.

    "sawtooth" was the original name of the potential coupling, when the
    coordinates were still wrapped into the cell (Umari-Pasquarello
    sawtooth).  The physics is now the continuous polarization branch on
    unwrapped coordinates; the old name remains accepted for backward
    compatibility of serialized records and configs.
    """
    if mode == "sawtooth":
        return "branch"
    return mode


def uses_efield_potential(mode: str) -> bool:
    """Whether the mode couples the field through the charge potential.

    Only ``branch`` does (and hence needs the unwrapped-coordinate
    conditioning input); ``bec`` computes the field force from pair
    geometry alone.
    """
    return normalize_efield_potential(mode) == "branch"


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


def efield_potential_from_coordinate(coord: Any, n_node: Any, efield_frame: Any) -> Any:
    """Charge-field potential ``phi_ext = -E . coord`` per node, ``(N, 1)``.

    ``coord`` is the coordinates AS PROVIDED (unwrapped, any branch --
    for the neutral constraint the energy is branch-independent, and a
    global translation only shifts it by ``-E . t * Q``).  ``phi_ext``
    in volts times a charge in electrons is an energy in eV.
    """
    xp = array_api_compat.array_namespace(coord)
    frame_id = frame_id_from_n_node(n_node, n_total=coord.shape[0])
    field_node = xp.take(efield_frame, frame_id, axis=0)
    return -xp.sum(
        xp.astype(coord, efield_frame.dtype) * field_node, axis=-1, keepdims=True
    )


def center_efield_potential(potential: Any, frame_id: Any, n_node: Any, xp: Any) -> Any:
    """Per-frame gauge fixing for the LEF input: remove the frame mean.

    A constant potential shift does not change the QEq driving (only
    differences matter), and the smaller values keep the update net
    responsive.  The ENERGY term uses the uncentered potential, so this
    is purely an input-conditioning choice.
    """
    n_total = potential.shape[0]
    mean = segment_sum(
        xp.reshape(potential, (n_total,)), frame_id, n_node.shape[0]
    ) / xp.astype(n_node, potential.dtype)
    return potential - xp.reshape(xp.take(mean, frame_id, axis=0), (n_total, 1))
