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


# ---------------------------------------------------------------------------
# Whole-molecule unwrapping (Route-2 molecular branch for periodic systems)
# ---------------------------------------------------------------------------


def _as_numpy(value: Any) -> Any:
    """Numpy view of an array, migrating accelerator tensors to the host."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


#: Covalent radii in Angstrom for the connectivity heuristic of
#: :func:`infer_molecule_ids` (common elements; unknown types fall back to
#: their own singleton molecule, which is the safe no-unwrap default).
COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Na": 1.66,
    "K": 2.03,
    "Li": 1.28,
}


def infer_molecule_ids(
    coord: Any,
    n_node: Any,
    box: Any,
    atype: Any,
    type_map: list[str],
    scale: float = 1.25,
) -> Any:
    """Infer per-atom molecule (fragment) ids from bonded connectivity.

    Two atoms are bonded when their minimum-image distance is below
    ``scale`` times the sum of their covalent radii
    (:data:`COVALENT_RADII`).  Bonds are searched within a frame only;
    atoms whose type is missing from the table stay singletons.

    Parameters
    ----------
    coord
        Flat coordinates with shape ``(N, 3)``, frame-major over ``n_node``.
    n_node
        Per-frame atom counts with shape ``(nf,)``.
    box
        Simulation cells with shape ``(nf, 3, 3)``, or ``None``.
    atype
        Atom types with shape ``(N,)``.
    type_map
        Element names by type index.
    scale
        Safety factor on the covalent-radius sum.

    Returns
    -------
    Integer molecule ids with shape ``(N,)``; unique per frame (a molecule
    never spans frames).
    """
    import numpy as np

    coord_np = _as_numpy(coord).astype(np.float64)
    n_node_np = _as_numpy(n_node).reshape(-1)
    atype_np = _as_numpy(atype).reshape(-1)
    box_np = None if box is None else _as_numpy(box).astype(np.float64)
    radii = np.array([COVALENT_RADII.get(type_map[int(t)], np.inf) for t in atype_np])
    mol = np.full(coord_np.shape[0], -1, dtype=np.int64)
    next_id = 0
    offset = 0
    for f, n in enumerate(n_node_np):
        n = int(n)
        pos = coord_np[offset : offset + n]
        rad = radii[offset : offset + n]
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        cell = None if box_np is None else box_np[f]
        for i in range(n):
            if not np.isfinite(rad[i]):
                continue
            delta = pos[i + 1 :] - pos[i]
            if cell is not None:
                frac = delta @ np.linalg.inv(cell)
                delta = (frac - np.round(frac)) @ cell
            dist = np.linalg.norm(delta, axis=-1)
            thresh = scale * (rad[i] + rad[i + 1 :])
            for j in np.nonzero(dist < thresh)[0]:
                ri, rj = find(i), find(i + 1 + j)
                if ri != rj:
                    parent[rj] = ri
        for i in range(n):
            r = find(i)
            if mol[offset + i] < 0:
                members = [j for j in range(n) if find(j) == r]
                for j in members:
                    mol[offset + j] = next_id
                next_id += 1
        offset += n
    import array_api_compat

    xp = array_api_compat.array_namespace(coord)
    return xp.asarray(mol)


def maybe_unwrap_molecules(
    coord: Any,
    n_node: Any,
    box: Any,
    atype: Any,
    type_map: list[str],
) -> Any:
    """Whole-molecule unwrap of ``coord`` when bonded fragments exist.

    Infers molecule ids with :func:`infer_molecule_ids`; applies
    :func:`unwrap_molecules` only when at least one multi-atom fragment is
    present (otherwise returns ``coord`` unchanged, so ionic crystals and
    non-molecular systems keep the as-provided coordinates).
    """
    if box is None:
        return coord
    import numpy as np

    mol = infer_molecule_ids(coord, n_node, box, atype, type_map)
    counts = np.bincount(np.asarray(mol).reshape(-1))
    if not bool((counts >= 2).any()):
        return coord
    return unwrap_molecules(coord, n_node, box, mol)


def unwrap_molecules(
    coord: Any,
    n_node: Any,
    box: Any,
    molecule_id: Any,
) -> Any:
    """Whole-molecule unwrapped coordinates for the branch efield coupling.

    Every molecule is unwrapped independently per frame: each atom is moved
    to the nearest image of its molecule's reference atom (molecules must
    be smaller than half the cell, otherwise the minimum-image gauge is
    ambiguous and this function raises), and the whole molecule is then
    shifted by an integer lattice vector so its centre lies in the home
    cell.  The image shifts are integers and locally constant, so the
    unwrap Jacobian w.r.t. the input coordinates is the identity and the
    coordinate-leaf forces map one-to-one onto the atoms.

    Parameters
    ----------
    coord
        Flat coordinates with shape ``(N, 3)``, frame-major over ``n_node``.
    n_node
        Per-frame atom counts with shape ``(nf,)``.
    box
        Simulation cells with shape ``(nf, 3, 3)`` (required).
    molecule_id
        Integer fragment ids with shape ``(N,)`` (e.g. from
        :func:`infer_molecule_ids`); atoms sharing an id form one molecule.

    Returns
    -------
    Unwrapped coordinates with shape ``(N, 3)``: intramolecularly
    continuous, whole-molecule re-centred into the home cell (bounded), and
    continuous under whole-molecule boundary crossings.

    Raises
    ------
    ValueError
        If a molecule's unwrapped extent reaches half the cell in any
        fractional direction (cocycle failure).
    """
    import numpy as np

    if box is None:
        raise ValueError("`unwrap_molecules` requires the simulation cell")
    coord_np = _as_numpy(coord).astype(np.float64).copy()
    n_node_np = _as_numpy(n_node).reshape(-1)
    box_np = _as_numpy(box).astype(np.float64)
    mol_np = _as_numpy(molecule_id).reshape(-1)
    out = coord_np.copy()
    offset = 0
    for f, n in enumerate(n_node_np):
        n = int(n)
        cell = box_np[f]
        inv_cell = np.linalg.inv(cell)
        pos = coord_np[offset : offset + n]
        mols = mol_np[offset : offset + n]
        new_pos = pos.copy()
        for m in np.unique(mols):
            idx = np.nonzero(mols == m)[0]
            ref = pos[idx[0]]
            delta = pos[idx] - ref
            frac = delta @ inv_cell
            shift = -np.round(frac)
            x = pos[idx] + shift @ cell
            frac_res = (x - ref) @ inv_cell
            extent = frac_res.max(axis=0) - frac_res.min(axis=0)
            if np.any(extent > 0.5):
                raise ValueError(
                    f"molecule {m} of frame {f} spans half the cell after "
                    "unwrapping (cocycle failure): the molecular gauge is "
                    "ill-defined for cells smaller than twice the molecule"
                )
            centre = x.mean(axis=0)
            centre_shift = np.floor(centre @ inv_cell)
            new_pos[idx] = x - centre_shift @ cell
        out[offset : offset + n] = new_pos
        offset += n
    import array_api_compat

    if hasattr(coord, "new_tensor"):
        # torch: keep device and dtype of the input.
        return coord.new_tensor(out)
    xp = array_api_compat.array_namespace(coord)
    result = xp.asarray(out)
    return xp.astype(result, coord.dtype)
