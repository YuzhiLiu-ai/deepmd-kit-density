# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import torch

from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.task.density_basis import (
    DensityBasisHead,
)
from deepmd.pt.utils.nlist import (
    build_directional_neighbor_list,
    extend_input_and_build_neighbor_list,
)
from deepmd.pt_expt.utils.edge_schema import (
    edge_schema_from_extended,
)
from deepmd.pt.utils.region import (
    normalize_coord,
)
from deepmd.utils.model_preset import (
    PERIODIC_TABLE,
)
from deepmd.utils.path import (
    DPPath,
)

from .base_atomic_model import (
    BaseAtomicModel,
)
from .dp_atomic_model import (
    DPAtomicModel,
)

log = logging.getLogger(__name__)


@BaseAtomicModel.register("density_basis")
class DensityBasisAtomicModel(DPAtomicModel):
    """Atomic model expanding the charge density of real atoms onto a grid.

    Unlike :class:`DPDensityAtomicModel`, the grid points never enter the
    descriptor: the descriptor runs on the real extended atoms only and the
    :class:`DensityBasisHead` expands the resulting per-atom density
    coefficients onto the grid through a scalar radial basis. Periodic
    images share the coefficients of their local row through ``mapping``.
    """

    def __init__(
        self,
        descriptor: BaseDescriptor,
        fitting: DensityBasisHead,
        type_map: list[str],
        **kwargs: Any,
    ) -> None:
        assert isinstance(fitting, DensityBasisHead)
        super().__init__(descriptor, fitting, type_map, **kwargs)
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()
        self.nnei = self.descriptor.get_nsel()
        if type_map and type_map[-1] in PERIODIC_TABLE:
            raise ValueError(
                "the last entry of type_map must be a reserved grid type "
                "for a density model, but got the real element "
                f"{type_map[-1]!r}; append a non-element placeholder "
                '(e.g. "X") at the end of type_map'
            )
        if abs(self.fitting_net.rcut - self.rcut) > 1e-10:
            raise ValueError(
                f"the density_basis head cutoff ({self.fitting_net.rcut}) "
                f"must match the descriptor cutoff ({self.rcut}); the grid "
                "neighbor list is built with the descriptor cutoff"
            )
        if getattr(self.fitting_net, "use_equivariant", False) and not hasattr(
            self.descriptor, "forward_with_edges"
        ):
            raise ValueError(
                "use_equivariant requires a descriptor exposing its SO(3) "
                "latent via forward_with_edges (e.g. dpa4), but got "
                f"{type(self.descriptor).__name__}"
            )

    def reinit_atom_exclude(
        self,
        exclude_types: list[int] = [],
    ) -> None:
        super().reinit_atom_exclude(exclude_types)
        # atoms of the excluded types carry no density: the head masks their
        # pair contributions. fitting_net is not set yet when the base
        # constructor runs this; DPAtomicModel.__init__ repeats the call on
        # the fitting once it exists.
        if hasattr(self, "fitting_net"):
            self.fitting_net.reinit_exclude(exclude_types)

    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        grid: torch.Tensor | None = None,
        grid_type: torch.Tensor | None = None,
        grid_nlist: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        return_atomic_feature: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return the density prediction on the grid points.

        Parameters
        ----------
        extended_coord
            coodinates in extended region
        extended_atype
            atomic type in extended region
        nlist
            neighbor list. nf x nloc x nsel
        mapping
            mapps the extended indices to local indices
        grid
            coordinates of the grid points. nf x ngrid x 3
        grid_type
            type of the grid points. nf x ngrid
        grid_nlist
            directional neighbor list from grid points to extended atoms.
            nf x ngrid x nsel

        Returns
        -------
        result_dict
            the result dict, defined by the `FittingOutputDef`.

        """
        del fparam, aparam, grid_type, return_atomic_feature
        nframes, _, _ = nlist.shape
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)
        assert mapping is not None
        assert grid is not None
        assert grid_nlist is not None
        _, ngrid, nsel = grid_nlist.shape

        # Handle default chg_spin if descriptor supports it
        if self.add_chg_spin_ebd and charge_spin is None:
            default_cs_tensor = self.descriptor.get_default_chg_spin()
            if default_cs_tensor is not None:
                default_cs_tensor = default_cs_tensor.to(device=extended_coord.device)
                charge_spin = torch.tile(default_cs_tensor.unsqueeze(0), [nframes, 1])

        extended_coord = extended_coord.view(nframes, -1, 3)
        # the descriptor sees the real extended atoms only; the grid points
        # are expansion centers and are never merged into the atom list
        use_equivariant = self.fitting_net.use_equivariant
        descriptor: torch.Tensor | None = None
        latent = None
        # the equivariant branch is eager-only (the edge-list builder is not
        # TorchScript-compatible); nesting it under is_scripting prunes it
        # under torch.jit.script, keeping the scalar path scriptable
        if not torch.jit.is_scripting():
            if use_equivariant:
                if comm_dict is not None:
                    raise NotImplementedError(
                        "density_basis with use_equivariant does not support "
                        "the parallel comm_dict path yet"
                    )
                nloc = nlist.shape[1]
                # the edge-based descriptor path also returns the SO(3) latent
                edge_schema = edge_schema_from_extended(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                )
                descriptor, latent, _vacuum = self.descriptor.forward_with_edges(
                    extended_coord=extended_coord[:, :nloc, :],
                    extended_atype=extended_atype[:, :nloc],
                    edge_index=edge_schema.edge_index,
                    edge_vec=edge_schema.edge_vec,
                    edge_mask=edge_schema.edge_mask,
                    charge_spin=charge_spin if self.add_chg_spin_ebd else None,
                )
        if not use_equivariant:
            descriptor, _rot_mat, _g2, _h2, _sw = self.descriptor(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                comm_dict=comm_dict,
                charge_spin=charge_spin if self.add_chg_spin_ebd else None,
            )
        if use_equivariant and torch.jit.is_scripting():
            raise RuntimeError(
                "density_basis with use_equivariant=True does not support "
                "torch.jit.script; evaluate the model eagerly"
            )
        assert descriptor is not None

        grid = grid.view(nframes, ngrid, 3)

        # nf x ngrid x nsel
        pair_mask = grid_nlist >= 0
        atom_idx = torch.where(pair_mask, grid_nlist, torch.zeros_like(grid_nlist))
        flat_idx = atom_idx.reshape(nframes, ngrid * nsel)

        # pair vector = extended_coord[atom_idx] - grid[grid_idx]; the
        # neighbor list was built against the extended coordinates, so the
        # images already account for the periodic boundary conditions
        neigh_coord = torch.gather(
            extended_coord, 1, flat_idx.unsqueeze(-1).expand(-1, -1, 3)
        ).view(nframes, ngrid, nsel, 3)
        pair_vec = neigh_coord - grid[:, :, None, :]
        pair_dist = torch.linalg.norm(pair_vec, dim=-1)
        # padding pairs take a distance at the cutoff, where the response is
        # exactly zero
        pair_dist = torch.where(
            pair_mask, pair_dist, torch.full_like(pair_dist, self.rcut)
        )

        pair_atype = torch.gather(extended_atype, 1, flat_idx).view(
            nframes, ngrid, nsel
        )
        # virtual atoms carry no density
        pair_mask = pair_mask & (pair_atype >= 0)
        # padding pairs take the zero padding row of the type embedding
        padding_type = torch.full_like(pair_atype, self.descriptor.get_ntypes())
        pair_atype = torch.where(pair_mask, pair_atype, padding_type)

        # periodic images share the coefficients of their local row
        coef_gather_index = torch.gather(mapping, 1, flat_idx).view(
            nframes, ngrid, nsel
        )

        density = self.fitting_net(
            descriptor,
            pair_atype=pair_atype,
            pair_dist=pair_dist,
            pair_mask=pair_mask,
            coef_gather_index=coef_gather_index,
            latent=latent,
            pair_vec=pair_vec,
        )
        return {"density": density}

    def forward_common_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        grid: torch.Tensor | None = None,
        grid_type: torch.Tensor | None = None,
        grid_nlist: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Common interface for atomic inference.

        This method accept extended coordinates, extended atom typs, neighbor list,
        and predict the density on the grid points.

        Parameters
        ----------
        extended_coord
            extended coodinates, shape: nf x (nall x 3)
        extended_atype
            extended atom typs, shape: nf x nall
            for a type < 0 indicating the atomic is virtual.
        nlist
            neighbor list, shape: nf x nloc x nsel
        mapping
            extended to local index mapping, shape: nf x nall
        grid
            coordinates of the grid points, shape: nf x ngrid x 3
        grid_type
            type of the grid points, shape: nf x ngrid
        grid_nlist
            directional neighbor list from grid points to atoms,
            shape: nf x ngrid x nsel

        Returns
        -------
        ret_dict
            dict of output atomic properties.
            should implement the definition of `fitting_output_def`.
            ret_dict["mask"] of shape nf x ngrid will be provided.
            ret_dict["mask"][ff,ii] == 1 indicating the ii-th grid point of the
            ff-th frame is not excluded.

        """
        assert grid is not None
        assert grid_type is not None
        assert grid_nlist is not None
        nframes, _, _ = nlist.shape
        _, ngrid, nsel = grid_nlist.shape

        if self.pair_excl is not None:
            pair_mask = self.pair_excl(nlist, extended_atype)
            # exclude neighbors in the nlist
            nlist = torch.where(pair_mask == 1, nlist, -1)
            # the directional grid-to-atom list: every center is the grid
            # type, so exclusions involving the grid type drop the neighbor
            reserved_type = self.descriptor.get_ntypes() - 1
            if any(
                reserved_type in pair for pair in self.pair_excl.get_exclude_types()
            ):
                nall = extended_atype.shape[1]
                virtual_type = self.pair_excl.ntypes * torch.ones(
                    [nframes, 1],
                    dtype=extended_atype.dtype,
                    device=extended_atype.device,
                )
                ae = torch.cat([extended_atype, virtual_type], dim=-1)
                index = torch.where(grid_nlist == -1, nall, grid_nlist).view(
                    nframes, ngrid * nsel
                )
                type_j = torch.gather(ae, 1, index)
                type_ij = reserved_type * (self.pair_excl.ntypes + 1) + type_j
                grid_pair_mask = self.pair_excl.type_mask[type_ij].view(
                    nframes, ngrid, nsel
                )
                grid_nlist = torch.where(grid_pair_mask == 1, grid_nlist, -1)

        ext_atom_mask = self.make_atom_mask(extended_atype)
        ret_dict = self.forward_atomic(
            extended_coord,
            torch.where(ext_atom_mask, extended_atype, 0),
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
            grid=grid,
            grid_type=grid_type,
            grid_nlist=grid_nlist,
            charge_spin=charge_spin,
        )
        ret_dict = self.apply_out_stat(ret_dict, grid_type)

        # nf x ngrid
        grid_mask = torch.ones(
            [nframes, ngrid], dtype=torch.int32, device=ext_atom_mask.device
        )
        if self.atom_excl is not None:
            grid_mask *= self.atom_excl(grid_type)

        for kk in ret_dict.keys():
            out_shape = ret_dict[kk].shape
            out_shape2 = 1
            for ss in out_shape[2:]:
                out_shape2 *= ss
            ret_dict[kk] = (
                ret_dict[kk].reshape([out_shape[0], out_shape[1], out_shape2])
                * grid_mask[:, :, None]
            ).view(out_shape)
        ret_dict["mask"] = grid_mask

        return ret_dict

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "DensityBasisAtomicModel.forward requires grid, grid_type and "
            "grid_nlist; use forward_common_atomic with explicit grid inputs "
            "instead."
        )

    def _get_forward_wrapper_func(
        self,
    ) -> Callable[..., dict[str, torch.Tensor]]:
        """Get a forward wrapper of the atomic model for output bias calculation."""

        def model_forward(
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: torch.Tensor | None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            charge_spin: torch.Tensor | None = None,
            spin: torch.Tensor | None = None,
            grid: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            del spin
            with torch.no_grad():
                (
                    extended_coord,
                    extended_atype,
                    mapping,
                    nlist,
                ) = extend_input_and_build_neighbor_list(
                    coord,
                    atype,
                    self.get_rcut(),
                    self.get_sel(),
                    mixed_types=self.mixed_types(),
                    box=box,
                )
                assert grid is not None
                if box is not None:
                    # same wrapping convention as forward_common: grid points
                    # outside the primary cell are periodically equivalent
                    grid = normalize_coord(
                        grid, box.to(grid.device).reshape(box.shape[0], 3, 3)
                    )
                grid_type = torch.full(
                    grid.shape[:-1],
                    self.descriptor.get_ntypes() - 1,
                    device=grid.device,
                    dtype=atype.dtype,
                )
                grid_nlist = build_directional_neighbor_list(
                    grid,
                    grid_type,
                    extended_coord,
                    extended_atype,
                    self.get_rcut(),
                    self.get_sel(),
                    distinguish_types=(not self.mixed_types()),
                )
                atomic_ret = self.forward_common_atomic(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping=mapping,
                    fparam=fparam,
                    aparam=aparam,
                    grid=grid,
                    grid_type=grid_type,
                    grid_nlist=grid_nlist,
                    charge_spin=charge_spin,
                )
                return {kk: vv.detach() for kk, vv in atomic_ret.items()}

        return model_forward

    def change_out_bias(
        self,
        sample_merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        """Change the output bias according to the input data and the pretrained model.

        For density models, this operation is skipped because the output is
        grid-based rather than atomic-based, and the standard atomic bias
        adjustment (change-by-statistic / set-by-statistic) does not apply.
        The head will adapt to the target dataset through normal gradient
        descent during training.
        """
        log.warning("change_out_bias is not supported for density models; skipping.")

    def compute_or_load_out_stat(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
    ) -> None:
        """Compute the output statistics for the head.

        Not implemented for density models: the output lives on grid points,
        so per-type atomic output statistics do not apply.
        """
        log.warning("Not implemented yet for density out stat!")
