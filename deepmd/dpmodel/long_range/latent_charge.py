# SPDX-License-Identifier: LGPL-3.0-or-later
"""Non-periodic direct interaction of descriptor-predicted latent charges."""

from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.fitting.latent_charge import (
    LatentChargeFitting,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    ALL_PAIRS,
    K_GRID,
    PME_MESH,
    frame_id_from_n_node,
    segment_sum,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_long_range import (
    BaseLongRange,
    check_long_range_support,
)
from .kernel import (
    COULOMB_CONSTANT,
    BaseInteractionKernel,
)
from .reciprocal import (
    kspace_energy,
    mesh_energy,
)

CHARGE_TARGET_SOURCES = ("neutral", "charge_spin", "fparam")
CHARGE_NET_NEURON = [64, 64]
CHARGE_NET_SEED_INDEX = 10_000
K_GRID_MODES = ("kgrid", "pme")
PME_ORDER = 8
PME_SPACING_FRACTION = 0.75


@BaseLongRange.register("latent_charge")
class LatentChargeLongRange(NativeOP, BaseLongRange):
    """Direct latent-charge energy over unordered intra-frame ``i < j`` pairs.

    The total interaction of each frame is shared evenly by its real atoms,
    matching the attribution of the existing DPA4C-LR implementation.  Periodic
    cells and reciprocal-space kernels are deliberately outside this stage.
    """

    def __init__(
        self,
        charge_net: LatentChargeFitting,
        kernel: BaseInteractionKernel,
        charge_constraint: dict | None = None,
        periodic: bool = False,
        kspace_tolerance: float = 1e-8,
        k_grid_mode: str = "kgrid",
        pme_order: int = PME_ORDER,
        pme_mesh: list[int] | tuple[int, int, int] | None = None,
        pme_mesh_spacing: float | None = None,
        self_interaction: bool = False,
    ) -> None:
        if charge_net.vacuum_ref:
            raise ValueError("a latent-charge readout cannot use `vacuum_ref`")
        self.charge_net = charge_net
        self.kernel = kernel
        self.charge_constraint = self._normalize_constraint(
            charge_constraint, charge_net
        )
        self.periodic = bool(periodic)
        self.kspace_tolerance = float(kspace_tolerance)
        if (
            not np.isfinite(self.kspace_tolerance)
            or not 0.0 < self.kspace_tolerance < 1.0
        ):
            raise ValueError(
                f"`kspace_tolerance` must be in (0, 1), got {self.kspace_tolerance}"
            )
        self.k_grid_mode = str(k_grid_mode)
        if self.k_grid_mode not in K_GRID_MODES:
            raise ValueError(
                f"`k_grid_mode` must be one of {K_GRID_MODES}, got {self.k_grid_mode!r}"
            )
        self.pme_order = int(pme_order)
        if self.pme_order < 2:
            raise ValueError(f"`pme_order` must be at least 2, got {self.pme_order}")
        self.pme_mesh = None if pme_mesh is None else tuple(int(d) for d in pme_mesh)
        if self.pme_mesh is not None and (
            len(self.pme_mesh) != 3 or any(d < 2 for d in self.pme_mesh)
        ):
            raise ValueError(
                f"`pme_mesh` must hold three dimensions of at least 2, got {pme_mesh!r}"
            )
        # The reference spacing of a fixed mesh is frozen at construction:
        # the kernel parameters may drift during training, and validating
        # later cells against the drifted spacing would spuriously reject the
        # very cells the mesh was sized for.
        if self.pme_mesh is None:
            self.pme_mesh_spacing = None
        elif pme_mesh_spacing is not None:
            self.pme_mesh_spacing = float(pme_mesh_spacing)
        else:
            self.pme_mesh_spacing = (
                PME_SPACING_FRACTION
                * float(np.pi)
                / self.kernel.kspace_cutoff(self.kspace_tolerance)
            )
        self.self_interaction = bool(self_interaction)

    @staticmethod
    def _normalize_constraint(
        constraint: dict | None, charge_net: LatentChargeFitting
    ) -> dict | None:
        if constraint is None:
            return None
        source = str(constraint.get("source", "neutral"))
        channel = int(constraint.get("channel", 0))
        fparam_index = int(constraint.get("fparam_index", 0))
        if source not in CHARGE_TARGET_SOURCES:
            raise ValueError(
                f"charge constraint source must be one of {CHARGE_TARGET_SOURCES}, "
                f"got {source!r}"
            )
        if not 0 <= channel < charge_net.dim_out:
            raise ValueError(f"constrained channel {channel} is out of range")
        if source == "fparam" and not 0 <= fparam_index < charge_net.numb_fparam:
            raise ValueError(f"`fparam_index` {fparam_index} is out of range")
        return {"source": source, "channel": channel, "fparam_index": fparam_index}

    @classmethod
    def from_config(
        cls,
        config: dict,
        descriptor: Any,
        fitting: Any,
        fitting_seed: int | list[int] | None = None,
    ) -> "LatentChargeLongRange":
        check_long_range_support(descriptor)
        if not descriptor.uses_graph_lower():
            raise NotImplementedError(
                "LatentChargeLongRange requires the NeighborGraph lower"
            )
        if config.get("periodic", False):
            pass  # periodic evaluation over the k grid is served below
        charge_net_config = {
            "neuron": CHARGE_NET_NEURON,
            # The LR charge readout defaults to tanh (independent of the main
            # fitting's activation), still overridable via `charge_net`.
            "activation_function": "tanh",
            "resnet_dt": fitting.resnet_dt,
            "precision": fitting.precision,
            "seed": child_seed(fitting_seed, CHARGE_NET_SEED_INDEX),
            **config.get("charge_net", {}),
        }
        kernel_config = {
            "precision": charge_net_config["precision"],
            **config["kernel"],
        }
        kernel_cls = BaseInteractionKernel.get_class_by_type(kernel_config.pop("type"))
        charge_net = LatentChargeFitting(
            var_name="latent_charge",
            ntypes=descriptor.get_ntypes(),
            dim_descrpt=descriptor.get_dim_out(),
            dim_out=int(config.get("channels", 1)),
            numb_fparam=fitting.get_dim_fparam(),
            numb_aparam=fitting.get_dim_aparam(),
            dim_case_embd=fitting.dim_case_embd,
            default_fparam=fitting.get_default_fparam(),
            use_aparam_as_mask=fitting.use_aparam_as_mask,
            mixed_types=descriptor.mixed_types(),
            type_map=fitting.get_type_map(),
            **charge_net_config,
        )
        return cls(
            charge_net,
            kernel_cls(**kernel_config),
            config.get("charge_constraint"),
            periodic=bool(config.get("periodic", False)),
            kspace_tolerance=float(config.get("kspace_tolerance", 1e-8)),
            # The explicit k grid stays the default until the mesh test suite
            # lands; PME is opt-in via `k_grid_mode: "pme"`.
            k_grid_mode=str(config.get("k_grid_mode", "kgrid")),
            pme_order=int(config.get("pme_order", PME_ORDER)),
            pme_mesh=config.get("pme_mesh"),
            self_interaction=bool(config.get("self_interaction", False)),
        )

    def output_def(self) -> FittingOutputDef:
        return self.charge_net.output_def()

    def required_topologies(self) -> tuple[str, ...]:
        if self.periodic:
            if self.k_grid_mode == "pme":
                return (ALL_PAIRS, PME_MESH)
            return (ALL_PAIRS, K_GRID)
        return (ALL_PAIRS,)

    def topology_params(self) -> dict[str, dict[str, Any]]:
        if not self.periodic:
            return {}
        k_max = self.kernel.kspace_cutoff(self.kspace_tolerance)
        if self.k_grid_mode == "pme":
            # B-splines of pme_order at three quarters of the Nyquist spacing
            # of the wave-number cutoff.
            spacing = PME_SPACING_FRACTION * float(np.pi) / k_max
            if self.pme_mesh is not None:
                # A mesh fixed by the training-time cell statistic keeps the
                # compiled graph at one mesh shape for the whole run; the
                # reference spacing stays at its frozen value.
                return {
                    PME_MESH: {
                        "dims": self.pme_mesh,
                        "spacing": self.pme_mesh_spacing,
                    }
                }
            return {PME_MESH: {"spacing": spacing}}
        return {K_GRID: {"k_max": k_max}}

    def uses_charge_spin(self) -> bool:
        return (
            self.charge_constraint is not None
            and self.charge_constraint["source"] == "charge_spin"
        )

    def call(self, *args: Any, **kwargs: Any) -> dict[str, Array]:
        return self.call_graph(*args, **kwargs)

    def call_graph(
        self,
        descriptor: Array,
        atype: Array,
        node_mask: Array,
        n_node: Array,
        topologies: dict[str, Any],
        fparam: Array | None = None,
        aparam: Array | None = None,
        charge_spin: Array | None = None,
        efield: Array | None = None,
        efield_ext_pot: Array | None = None,
        descriptor_context: dict[str, Array] | None = None,
    ) -> dict[str, Array]:
        if efield is not None:
            raise ValueError(
                "an external electric field (`efield`) requires the latent "
                "electrostatic feedback of the multipole plugin; the charge-only "
                "long-range plugin does not support it"
            )
        if efield_ext_pot is not None:
            raise ValueError(
                "the sawtooth efield potential requires the latent-multipole "
                "plugin; the charge-only long-range plugin does not support it"
            )
        xp = array_api_compat.array_namespace(descriptor)
        n = descriptor.shape[0]
        pairs = topologies[ALL_PAIRS]
        frame_id = frame_id_from_n_node(n_node, n_total=n)
        fparam_node = None if fparam is None else xp.take(fparam, frame_id, axis=0)
        charge = self.charge_net.call_graph(
            descriptor, atype, fparam=fparam_node, aparam=aparam
        )["latent_charge"]
        charge = xp.astype(charge, pairs.vec.dtype)
        charge = xp.where(node_mask[:, None], charge, xp.zeros_like(charge))
        if self.charge_constraint is not None:
            charge = self._constrain(
                charge, node_mask, frame_id, n_node, fparam, charge_spin
            )

        src, dst = pairs.index
        pair_charge = xp.sum(
            xp.take(charge, src, axis=0) * xp.take(charge, dst, axis=0), axis=-1
        )
        r_sq = xp.sum(pairs.vec * pairs.vec, axis=-1)
        pair_energy = pair_charge * self._pair_kernel(r_sq, src, n)
        pair_energy = COULOMB_CONSTANT * xp.where(
            pairs.mask, pair_energy, xp.zeros_like(pair_energy)
        )
        pair_frame = xp.take(frame_id, dst, axis=0)
        frame_energy = segment_sum(pair_energy, pair_frame, n_node.shape[0])
        if self.self_interaction:
            # The legacy DPA4 SOG path adds the self term 1/2 K(0) sum_i q_i^2
            # on top of the unordered pair sum.
            self_energy = (
                0.5
                * self.kernel.contact_value(charge)
                * xp.sum(charge * charge, axis=-1)
            )
            frame_energy = frame_energy + COULOMB_CONSTANT * segment_sum(
                self_energy * xp.astype(node_mask, charge.dtype),
                frame_id,
                n_node.shape[0],
            )
        count = xp.astype(n_node, frame_energy.dtype)
        count = xp.maximum(count, xp.ones_like(count))
        energy = xp.take(frame_energy / count, frame_id, axis=0)
        energy = xp.reshape(energy, (n, 1))
        if self.periodic:
            if self.k_grid_mode == "pme":
                # The mesh energy is attributed to the atoms as q.phi/2 (see
                # reciprocal.mesh_energy); its gradient w.r.t. the mesh's
                # coord and strain leaves gives its force and negative virial.
                energy = energy + xp.reshape(
                    mesh_energy(
                        charge,
                        topologies[PME_MESH],
                        frame_id,
                        self.kernel,
                        self.pme_order,
                    ),
                    (n, 1),
                )
            else:
                # The k-space energy is attributed to the atoms as q.phi/2.
                energy = energy + xp.reshape(
                    kspace_energy(charge, topologies[K_GRID], self.kernel), (n, 1)
                )
        return {
            "energy": energy,
            "latent_charge": charge,
        }

    def _pair_kernel(self, r_sq: Array, src: Array, n: int) -> Array:
        return self.kernel(r_sq)

    def _constrain(
        self,
        charge: Array,
        node_mask: Array,
        frame_id: Array,
        n_node: Array,
        fparam: Array | None,
        charge_spin: Array | None,
    ) -> Array:
        xp = array_api_compat.array_namespace(charge)
        nf = n_node.shape[0]
        channel = self.charge_constraint["channel"]
        target = self._total_charge(nf, charge, fparam, charge_spin)
        weight = xp.astype(node_mask, charge.dtype)
        count = segment_sum(weight, frame_id, nf)
        total = segment_sum(charge[:, channel], frame_id, nf)
        shift = (total - target) / xp.maximum(count, xp.ones_like(count))
        constrained = charge[:, channel] - xp.take(shift, frame_id, axis=0) * weight
        return xp.concat(
            [charge[:, :channel], constrained[:, None], charge[:, channel + 1 :]],
            axis=-1,
        )

    def _total_charge(
        self,
        nf: int,
        like: Array,
        fparam: Array | None,
        charge_spin: Array | None,
    ) -> Array:
        xp = array_api_compat.array_namespace(like)
        device = array_api_compat.device(like)
        source = self.charge_constraint["source"]
        if source == "charge_spin":
            if charge_spin is None:
                raise ValueError(
                    "the charge constraint reads `charge_spin`, which was not given"
                )
            target = charge_spin[:, 0]
        elif source == "fparam":
            if fparam is None:
                raise ValueError(
                    "the charge constraint reads `fparam`, which was not given"
                )
            target = fparam[:, self.charge_constraint["fparam_index"]]
        else:
            target = xp.zeros((nf,), dtype=like.dtype, device=device)
        return xp.astype(target, like.dtype)

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        protection: float = 1e-2,
        stat_file_path: DPPath | None = None,
    ) -> None:
        self.charge_net.compute_input_stats(
            merged, protection=protection, stat_file_path=stat_file_path
        )

    def change_type_map(self, type_map: list[str]) -> None:
        self.charge_net.change_type_map(type_map=type_map)

    def set_case_embd(self, case_idx: int) -> None:
        self.charge_net.set_case_embd(case_idx)

    def share_params(
        self,
        base_class: "LatentChargeLongRange",
        shared_level: int,
        model_prob: float = 1.0,
        protection: float = 1e-2,
        resume: bool = False,
    ) -> None:
        """Share the charge/dipole head and kernel with ``base_class``.

        The learnable state of a long-range plugin is the readout head
        (``charge_net`` -- for a multipole also carrying the dipole projection,
        the recurrent polar-update net and the self-polarization strength) and
        the interaction ``kernel``. Rebinding them registers the shared modules
        in this plugin's tables, which produces correct PyTorch parameter
        sharing. Branch-local runtime state -- the neighbor topologies, the PME
        mesh and the charge-constraint configuration -- is deliberately left
        untouched so each branch keeps its own graph-derived topology.

        Parameters
        ----------
        base_class
            The plugin that owns the shared parameters.
        shared_level
            Sharing level. Only level ``0`` (share head and kernel) is supported.
        model_prob, protection, resume
            Accepted by the common multitask ABI; unused.

        Raises
        ------
        TypeError
            If ``base_class`` is not the same plugin class.
        ValueError
            If the structural configuration differs between the replicas.
        NotImplementedError
            If ``shared_level`` is not ``0``.
        """
        del model_prob, protection, resume
        if self.__class__ is not base_class.__class__:
            raise TypeError(
                "Only identical long-range plugins can share parameters, got "
                f"{type(base_class).__name__}."
            )
        if shared_level != 0:
            raise NotImplementedError(
                f"Long-range plugins support only shared_level=0, got {shared_level}."
            )
        # Defensive structural agreement check; two branches built from the same
        # ``shared_dict`` entry are identical by construction.
        for attr in ("periodic", "self_interaction", "k_grid_mode"):
            if getattr(self, attr, None) != getattr(base_class, attr, None):
                raise ValueError(
                    "Long-range parameter sharing requires identical "
                    f"`{attr}`, got {getattr(self, attr, None)!r} vs "
                    f"{getattr(base_class, attr, None)!r}."
                )
        if type(self.kernel) is not type(base_class.kernel):
            raise ValueError(
                "Long-range parameter sharing requires the same kernel type."
            )
        self.charge_net = base_class.charge_net
        self.kernel = base_class.kernel

    def serialize(self) -> dict:
        data = {
            "@class": "LongRange",
            "type": "latent_charge",
            "@version": 1,
            "charge_net": self.charge_net.serialize(),
            "kernel": self.kernel.serialize(),
            "charge_constraint": self.charge_constraint,
        }
        if self.self_interaction:
            data["self_interaction"] = True
        if self.periodic:
            data["periodic"] = True
            data["kspace_tolerance"] = self.kspace_tolerance
            data["k_grid_mode"] = self.k_grid_mode
            data["pme_order"] = self.pme_order
            if self.pme_mesh is not None:
                data["pme_mesh"] = list(self.pme_mesh)
                data["pme_mesh_spacing"] = self.pme_mesh_spacing
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "LatentChargeLongRange":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        periodic = data.get("periodic", False)
        return cls(
            LatentChargeFitting.deserialize(data["charge_net"]),
            BaseInteractionKernel.deserialize(data["kernel"]),
            data["charge_constraint"],
            periodic=periodic,
            kspace_tolerance=data.get("kspace_tolerance", 1e-8),
            # A periodic record that names no mode predates the mesh and
            # evaluates the explicit wave numbers.
            k_grid_mode=data.get("k_grid_mode", "kgrid" if periodic else "pme"),
            pme_order=data.get("pme_order", PME_ORDER),
            pme_mesh=data.get("pme_mesh"),
            pme_mesh_spacing=data.get("pme_mesh_spacing"),
            self_interaction=data.get("self_interaction", False),
        )
