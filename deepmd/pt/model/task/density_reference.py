# SPDX-License-Identifier: LGPL-3.0-or-later
"""Frozen superposed-atomic-density (SAD) reference for delta learning.

The reference density of a grid point is the sum over neighbor atoms of a
per-element radial function evaluated at the grid-atom distance:

    rho_ref(g) = sum_j ref_table[t_j](r_gj) * envelope(r_gj)

with ``ref_table`` a frozen per-element radial table (e.g. free-atom DFT
densities shell-averaged onto a uniform grid). The table lives in a
persistent buffer so it is stored in the checkpoint and inference does not
depend on the file it was built from. The module holds no parameters, so no
gradient ever flows into the reference and the delta-learning semantics is
strict: the network only learns the residual on top of it.
"""

from __future__ import (
    annotations,
)

import numpy as np
import torch

from deepmd.pt.model.descriptor.sezm_nn.radial import (
    C3CutoffEnvelope,
)

__all__ = [
    "RadialReferenceDensity",
]


class RadialReferenceDensity(torch.nn.Module):
    """Frozen per-element radial reference density evaluated on grid-atom pairs.

    Parameters
    ----------
    ref_table
        Radial density values with shape ``(ntypes + 1, n_ref)``; row ``t`` is
        the reference radial function of atom type ``t``. The last row is the
        padding row and must be all zeros.
    r_grid
        Uniform radial grid of the table with shape ``(n_ref,)``, strictly
        increasing, starting at or below 0 and ending at or above ``rcut``.
    rcut
        Cutoff radius; the shared C^3 envelope keeps the reference strictly
        zero at and beyond it.
    dtype
        Buffer dtype.
    """

    def __init__(
        self,
        ref_table: np.ndarray,
        r_grid: np.ndarray,
        rcut: float,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        table = np.asarray(ref_table, dtype=np.float64)
        grid = np.asarray(r_grid, dtype=np.float64)
        if table.ndim != 2:
            raise ValueError(f"ref_table must be 2D, got shape {table.shape}")
        if grid.ndim != 1 or grid.shape[0] != table.shape[1]:
            raise ValueError(
                f"r_grid length {grid.shape[0] if grid.ndim == 1 else '?'} does not "
                f"match ref_table columns {table.shape[1]}"
            )
        if grid.shape[0] < 2:
            raise ValueError("r_grid must have at least 2 points")
        dr = np.diff(grid)
        if not np.all(dr > 0):
            raise ValueError("r_grid must be strictly increasing")
        if not np.allclose(dr, dr[0], rtol=1e-5, atol=1e-12):
            raise ValueError("r_grid must be uniform for linear interpolation")
        if grid[-1] < rcut:
            raise ValueError(
                f"r_grid ends at {grid[-1]:.4f} but rcut is {rcut:.4f}; the "
                "reference table must cover the whole cutoff range"
            )
        if not np.allclose(table[-1], 0.0):
            raise ValueError("the last (padding) row of ref_table must be zero")
        self.n_ref = int(table.shape[1])
        self.r0 = float(grid[0])
        self.dr = float(dr[0])
        self.register_buffer(
            "ref_table",
            torch.tensor(table, dtype=dtype, device=torch.device("cpu")),
            persistent=True,
        )
        self.register_buffer(
            "r_grid",
            torch.tensor(grid, dtype=dtype, device=torch.device("cpu")),
            persistent=True,
        )
        # same envelope as the learnable radial branch: strict zero at rcut
        self.envelope = C3CutoffEnvelope(rcut, exponent=7, dtype=dtype)

    def forward(
        self,
        pair_dist: torch.Tensor,
        pair_atype: torch.Tensor,
        pair_keep: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the reference density contribution of every pair.

        Parameters
        ----------
        pair_dist
            Grid-atom distance of every pair. nf x ngrid x nsel.
        pair_atype
            Type of the neighbor atom of every pair. nf x ngrid x nsel.
            Padding positions carry ``ntypes`` (the zero padding row).
        pair_keep
            True for valid, non-excluded pairs. nf x ngrid x nsel.

        Returns
        -------
        torch.Tensor
            Reference density contribution of every pair. nf x ngrid x nsel.
        """
        dtype = self.ref_table.dtype
        # uniform-grid linear interpolation indices
        x = (pair_dist.to(dtype) - self.r0) / self.dr
        idx0 = torch.floor(x).long().clamp(min=0, max=self.n_ref - 2)
        weight = (x - idx0.to(dtype)).clamp(min=0.0, max=1.0)
        # flattened (type, radial bin) gather; padding types hit the zero row
        atype = pair_atype.clamp(min=0, max=self.ref_table.shape[0] - 1)
        flat_index = atype * self.n_ref
        row0 = self.ref_table.reshape(-1)[(flat_index + idx0).reshape(-1)].view_as(
            pair_dist
        )
        row1 = self.ref_table.reshape(-1)[(flat_index + idx0 + 1).reshape(-1)].view_as(
            pair_dist
        )
        interp = row0 * (1.0 - weight) + row1 * weight
        env = self.envelope(pair_dist.reshape(-1, 1).to(dtype)).view_as(pair_dist)
        return interp * env * pair_keep.to(dtype)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Resize the buffers to the checkpoint's table shape before loading.

        The fallback construction path (reference file unavailable) builds a
        placeholder zero table whose shape cannot match the checkpoint; the
        buffers are resized here so a strict load succeeds and the grid
        metadata (n_ref, r0, dr) is recomputed from the loaded grid.
        """
        for name in ("ref_table", "r_grid"):
            key = prefix + name
            current = getattr(self, name)
            if key in state_dict and state_dict[key].shape != current.shape:
                setattr(
                    self,
                    name,
                    torch.empty(
                        state_dict[key].shape,
                        dtype=current.dtype,
                        device=current.device,
                    ),
                )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if prefix + "r_grid" in state_dict:
            grid = self.r_grid.detach().cpu().double().numpy()
            self.n_ref = int(grid.shape[0])
            self.r0 = float(grid[0])
            self.dr = float(grid[1] - grid[0])
