# SPDX-License-Identifier: LGPL-3.0-or-later
"""Direct and reciprocal interaction of descriptor-predicted latent multipoles."""

from typing import Any

import array_api_compat

from deepmd.dpmodel.array_api import Array
from deepmd.dpmodel.common import get_xp_precision
from deepmd.dpmodel.fitting.latent_multipole import LatentMultipoleFitting
from deepmd.dpmodel.utils.neighbor_graph import (
    ALL_PAIRS,
    K_GRID,
    PME_MESH,
    frame_id_from_n_node,
    segment_sum,
)
from deepmd.dpmodel.utils.seed import child_seed
from deepmd.utils.version import check_version_compatibility

from .base_long_range import BaseLongRange
from .efield import (
    EFIELD_POTENTIAL_MODES,
    center_efield_potential,
    normalize_efield_potential,
    resolve_efield_frame,
    uses_efield_potential,
)
from .kernel import COULOMB_CONSTANT, BaseInteractionKernel
from .latent_charge import (
    CHARGE_NET_NEURON,
    CHARGE_NET_SEED_INDEX,
    LatentChargeLongRange,
)
from .reciprocal import (
    kspace_multipole_energy,
    kspace_multipole_potential_field,
    mesh_multipole_energy,
    mesh_multipole_potential_field,
)


@BaseLongRange.register("latent_multipole")
class LatentMultipoleLongRange(LatentChargeLongRange):
    """LES charge--charge, charge--dipole and dipole--dipole direct energy.

    Unordered intra-frame pairs are counted once.  The recurrent updates --
    the latent electrostatic feedback (LEF) -- and the single
    self-polarization term use the final latent state, with no detached
    tensors anywhere in the readout/update/energy path.
    """

    def __init__(
        self,
        charge_net: LatentMultipoleFitting,
        kernel: BaseInteractionKernel,
        charge_constraint: dict | None = None,
        use_self_pol: bool = True,
        efield: bool = False,
        efield_potential: str = "force_only",
        periodic: bool = False,
        kspace_tolerance: float = 1e-8,
        k_grid_mode: str = "kgrid",
        pme_order: int = 8,
        pme_mesh: list[int] | tuple[int, int, int] | None = None,
        pme_mesh_spacing: float | None = None,
    ) -> None:
        if efield and int(getattr(charge_net, "lr_update_steps", 0)) <= 0:
            raise ValueError(
                "the external electric field (`efield`) requires the latent "
                "electrostatic feedback (LEF), i.e. `lr_update_steps > 0`; got "
                f"lr_update_steps={getattr(charge_net, 'lr_update_steps', 0)}. "
                "Set a positive `lr_update_steps` to use an external field."
            )
        if efield and (charge_constraint or {}).get("source") == "fparam":
            _cc_index = int((charge_constraint or {}).get("fparam_index", 0))
            if _cc_index < 3:
                raise ValueError(
                    "the external electric field (`efield`) reads the field "
                    "from fparam[:, :3], which overlaps the charge "
                    f"constraint's fparam_index={_cc_index}; move the "
                    "total-charge target to fparam_index >= 3 (and size "
                    "numb_fparam accordingly)."
                )
        super().__init__(
            charge_net,
            kernel,
            charge_constraint,
            periodic=periodic,
            kspace_tolerance=kspace_tolerance,
            k_grid_mode=k_grid_mode,
            pme_order=pme_order,
            pme_mesh=pme_mesh,
            pme_mesh_spacing=pme_mesh_spacing,
        )
        self.use_self_pol = bool(use_self_pol)
        self.efield = bool(efield)
        efield_potential = normalize_efield_potential(str(efield_potential))
        if efield_potential not in EFIELD_POTENTIAL_MODES:
            raise ValueError(
                f"`efield_potential` must be one of {EFIELD_POTENTIAL_MODES}, "
                f"got {efield_potential!r}"
            )
        self.efield_potential = str(efield_potential)
        if not self.use_self_pol:
            # pol_stiffness is only consumed by the self-polarization energy; with
            # use_self_pol=False it stays a registered trainable tensor that
            # no forward reads, which breaks DDP reduction ("parameters that
            # were not used in producing loss").  Freeze it instead -- it
            # remains a parameter in the state dict, so serialization and
            # checkpoints are unchanged.  Backends without autograd
            # parameters (numpy) skip the duck-typed call.
            stiffness_param = getattr(self.charge_net, "pol_stiffness", None)
            if hasattr(stiffness_param, "requires_grad_"):
                stiffness_param.requires_grad_(False)

    @property
    def multipole_net(self) -> LatentMultipoleFitting:
        return self.charge_net

    def uses_descriptor_context(self) -> bool:
        return True

    @classmethod
    def from_config(
        cls,
        config: dict,
        descriptor: Any,
        fitting: Any,
        fitting_seed: int | list[int] | None = None,
    ) -> "LatentMultipoleLongRange":
        from .base_long_range import check_long_range_support

        check_long_range_support(descriptor)
        if not descriptor.uses_graph_lower():
            raise NotImplementedError(
                "LatentMultipoleLongRange requires the graph lower"
            )
        if getattr(descriptor, "compress", False):
            raise NotImplementedError(
                "compressed DPA4C does not expose degree-one moments for latent multipoles"
            )
        if config.get("periodic", False):
            pass  # periodic evaluation over the k grid is served below
        if int(config.get("channels", 1)) != 1:
            raise ValueError("latent multipoles require `channels=1`")
        charge_net_config = {
            "neuron": CHARGE_NET_NEURON,
            # The LR charge/dipole readout defaults to tanh (independent of
            # the main fitting's activation), still overridable via
            # `charge_net`.
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
        # The dipole head follows the descriptor family: DPA4C contracts its
        # degree-one Cartesian moments with an invariant weight net ("gr");
        # DPA4 projects its SO(3)-packed equivariant latent with an SO3Linear
        # and decodes the l=1 block ("sezm").
        from deepmd.dpmodel.descriptor.dpa4 import (
            DescrptDPA4,
        )

        if isinstance(descriptor, DescrptDPA4):
            dipole_mode = "sezm"
            head_kwargs = {
                "latent_lmax": int(descriptor.mp_init_lmax),
                "latent_channels": int(descriptor.channels),
                "embedding_width": 0,
            }
        else:
            dipole_mode = "gr"
            embedding_width = int(config.get("embedding_width", 0))
            if embedding_width == 0:
                embedding_width = int(descriptor.degree_channels[1])
            head_kwargs = {"embedding_width": embedding_width}
        charge_net = LatentMultipoleFitting(
            var_name="latent_charge",
            ntypes=descriptor.get_ntypes(),
            dim_descrpt=descriptor.get_dim_out(),
            dim_out=1,
            numb_fparam=fitting.get_dim_fparam(),
            numb_aparam=fitting.get_dim_aparam(),
            dim_case_embd=fitting.dim_case_embd,
            default_fparam=fitting.get_default_fparam(),
            use_aparam_as_mask=fitting.use_aparam_as_mask,
            mixed_types=descriptor.mixed_types(),
            type_map=fitting.get_type_map(),
            dipole_mode=dipole_mode,
            pol_stiffness=config.get("pol_stiffness"),
            alpha_pol=config.get("alpha_pol"),
            lr_update_steps=int(config.get("lr_update_steps", 0)),
            lr_update_neuron=config.get("lr_update_neuron"),
            lr_update_damping=float(config.get("lr_update_damping", 0.2)),
            **head_kwargs,
            **charge_net_config,
        )
        return cls(
            charge_net,
            kernel_cls(**kernel_config),
            config.get("charge_constraint"),
            bool(config.get("use_self_pol", True)),
            efield=bool(config.get("efield", False)),
            efield_potential=str(config.get("efield_potential", "force_only")),
            periodic=bool(config.get("periodic", False)),
            kspace_tolerance=float(config.get("kspace_tolerance", 1e-8)),
            k_grid_mode=str(config.get("k_grid_mode", "kgrid")),
            pme_order=int(config.get("pme_order", 8)),
            pme_mesh=config.get("pme_mesh"),
        )

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
        xp = array_api_compat.array_namespace(descriptor)
        need_gr = self.charge_net.dipole_mode == "gr"
        if descriptor_context is None or (
            (need_gr and descriptor_context.get("gr") is None)
            or (not need_gr and descriptor_context.get("equivariant_latent") is None)
        ):
            raise RuntimeError(
                "latent multipoles require DPA4C degree-one moments or the "
                "DPA4 equivariant latent"
            )
        n = descriptor.shape[0]
        pairs = topologies[ALL_PAIRS]
        mesh = topologies.get(PME_MESH) if self.periodic else None
        k_grid = None
        if self.periodic and mesh is None:
            k_grid = topologies.get(K_GRID)
        frame_id = frame_id_from_n_node(n_node, n_total=n)
        fparam_node = None if fparam is None else xp.take(fparam, frame_id, axis=0)
        # Uniform external field, gathered to the node axis. Active only when
        # the plugin was built with `efield=True` (which the constructor gates
        # on LEF, `lr_update_steps > 0`); otherwise the argument is ignored so
        # field-free models/checkpoints are untouched. Translation invariant:
        # it shifts the field each atom responds to, never the geometry (which
        # enters only through the relative pair vectors).
        #
        # Two transports: the explicit `efield` input (per-frame (nf, 3), API
        # and tests) wins; otherwise the field RIDES fparam[:, :3] (V/AA) --
        # the compile/export/deploy-friendly route, because fparam is already
        # part of the traced ABI. With fparam as the transport the charge net
        # (built for the model's own numb_fparam, 0 in this configuration)
        # must NOT see it as conditioning, so it receives None below.
        efield_frame = resolve_efield_frame(self.efield, efield, fparam)
        efield_node = (
            None if efield_frame is None else xp.take(efield_frame, frame_id, axis=0)
        )
        branch_mode = uses_efield_potential(self.efield_potential)
        if branch_mode and efield_node is not None and efield_ext_pot is None:
            raise ValueError(
                "the 'branch' charge-field potential requires the coordinate "
                "potential input (`efield_ext_pot`), which the model wrapper "
                "builds from the as-provided coordinates"
            )
        charge_net_fparam = (
            fparam_node if int(getattr(self.charge_net, "numb_fparam", 0)) > 0 else None
        )
        initial = self.charge_net.call_graph(
            descriptor,
            atype,
            gr=descriptor_context.get("gr"),
            fparam=charge_net_fparam,
            aparam=aparam,
            equivariant_latent=descriptor_context.get("equivariant_latent"),
        )
        q_output_dtype = initial["latent_charge"].dtype
        p_output_dtype = initial["latent_dipole"].dtype
        q = xp.astype(initial["latent_charge"], pairs.vec.dtype)
        p = xp.astype(initial["latent_dipole"], pairs.vec.dtype)
        q = xp.where(node_mask[:, None], q, xp.zeros_like(q))
        p = xp.where(node_mask[:, None], p, xp.zeros_like(p))
        if self.charge_constraint is not None:
            q = self._constrain(q, node_mask, frame_id, n_node, fparam, charge_spin)
        local_p = p
        src, dst = pairs.index
        r_sq = xp.sum(pairs.vec * pairs.vec, axis=-1)
        kernel, a_coeff, c_coeff = self._radial_coefficients(r_sq, src, n)

        if self.charge_net.lr_update_steps > 0:
            # The LEF's QEq input uses the gauge-fixed potential (frame mean
            # removed): a constant shift does not change the driving, and the
            # energy term below consumes the uncentered potential directly.
            efield_pot_lef = (
                None
                if efield_ext_pot is None
                else center_efield_potential(efield_ext_pot, frame_id, n_node, xp)
            )
            update_dtype = get_xp_precision(xp, self.charge_net.precision)
            q = xp.astype(q, update_dtype)
            p = xp.astype(p, update_dtype)
            local_p = xp.astype(local_p, update_dtype)
            update_descriptor = xp.astype(descriptor, update_dtype)
            update_vec = xp.astype(pairs.vec, update_dtype)
            update_mask = xp.astype(pairs.mask, update_dtype)
            if update_dtype == r_sq.dtype:
                # The pair geometry is invariant under the polar updates, so
                # the radial coefficients (erf/exp of every pair) are shared
                # with the final energy instead of recomputed per precision.
                update_k = xp.astype(kernel, update_dtype)
                update_a = xp.astype(a_coeff, update_dtype)
                update_c = xp.astype(c_coeff, update_dtype)
            else:
                update_r_sq = xp.sum(update_vec * update_vec, axis=-1)
                update_k, update_a, update_c = self._radial_coefficients(
                    update_r_sq, src, n
                )
            mask_q = xp.astype(node_mask[:, None], update_dtype)
            sezm_fukui = (
                self.charge_net.dipole_mode == "sezm"
                and self.charge_constraint is not None
            )
            if sezm_fukui:
                # The legacy SeZM recurrence conserves the per-frame,
                # per-channel totals the state had when the iteration
                # started, redistributing each deficit with the learned
                # Fukui weights instead of a uniform shift.
                q_target = segment_sum(q, frame_id, n_node.shape[0])
            for _ in range(self.charge_net.lr_update_steps):
                # Convention note: the LEF drives the dipole update with the
                # GENERALIZED FORCE dU/dp (= -E), the convention of the
                # direct pair sum (`_pair_potential_field`, inherited from
                # the legacy DPA4C-LR recurrence).  The reciprocal-space
                # helpers return +dE/dp and the external field is a physical
                # E, so both enter with the opposite sign below.
                potential, field = self._pair_potential_field(
                    q,
                    p,
                    update_vec,
                    update_mask,
                    update_k,
                    update_a,
                    update_c,
                    src,
                    dst,
                    n,
                    xp,
                )
                if k_grid is not None:
                    # force_k = dE_k/dp; the direct part carries dU/dp = -E,
                    # so the k-space sum enters with a PLUS here.
                    phi_k, force_k = kspace_multipole_potential_field(
                        q, p, k_grid, self.kernel
                    )
                    potential = potential + phi_k
                    field = field + force_k
                if mesh is not None:
                    phi_m, force_m = mesh_multipole_potential_field(
                        q, p, mesh, frame_id, self.kernel, self.pme_order
                    )
                    potential = potential + phi_m
                    field = field + force_m
                if efield_pot_lef is not None:
                    # The external potential drives the charge
                    # equilibration physically: the LEF's QEq-like response
                    # to the applied field is no longer purely learned.
                    potential = potential + xp.astype(efield_pot_lef, potential.dtype)
                if efield_node is not None:
                    # External field enters the latent electrostatic feedback:
                    # the update sees the total field (system + applied), so
                    # the dipoles/charges polarize self-consistently.  With
                    # the dU/dp = -E convention the applied field is SUBTRACTED.
                    field = field - xp.astype(efield_node, field.dtype)
                if sezm_fukui:
                    q, p, fukui = self.charge_net.update_polar_state(
                        update_descriptor,
                        q,
                        p,
                        local_p,
                        potential,
                        field,
                        return_fukui=True,
                    )
                else:
                    q, p = self.charge_net.update_polar_state(
                        update_descriptor, q, p, local_p, potential, field
                    )
                q = q * mask_q
                p = p * mask_q
                if sezm_fukui:
                    q = self._constrain_fukui(q, fukui, node_mask, frame_id, q_target)
                elif self.charge_constraint is not None:
                    q = self._constrain(
                        q, node_mask, frame_id, n_node, fparam, charge_spin
                    )
            q = xp.astype(q, q_output_dtype)
            p = xp.astype(p, p_output_dtype)

        q_src = xp.astype(q[src, 0], pairs.vec.dtype)
        q_dst = xp.astype(q[dst, 0], pairs.vec.dtype)
        p_src = xp.astype(p[src], pairs.vec.dtype)
        p_dst = xp.astype(p[dst], pairs.vec.dtype)
        p_src_dot_r = xp.sum(p_src * pairs.vec, axis=-1)
        p_dst_dot_r = xp.sum(p_dst * pairs.vec, axis=-1)
        pair_energy = (
            q_src * q_dst * kernel
            + (q_src * p_dst_dot_r - q_dst * p_src_dot_r) * a_coeff
            - a_coeff * xp.sum(p_src * p_dst, axis=-1)
            - c_coeff * p_src_dot_r * p_dst_dot_r
        )
        pair_energy = xp.where(pairs.mask, pair_energy, xp.zeros_like(pair_energy))
        pair_frame = xp.take(frame_id, dst, axis=0)
        frame_energy = segment_sum(pair_energy, pair_frame, n_node.shape[0])
        if self.use_self_pol:
            # Self-polarization spring energy 0.5 * theta^2 * |p|^2 with the
            # per-type raw stiffness theta (theta^2 is the spring constant,
            # so the energy is quadratic in theta and never divides by a
            # trainable quantity).
            stiffness = self.charge_net._parameter_array("pol_stiffness", pairs.vec)
            stiffness_atom = xp.take(stiffness, atype, axis=0) ** 2
            self_pol = (
                0.5
                * stiffness_atom
                * xp.sum(xp.astype(p, pairs.vec.dtype) ** 2, axis=-1)
                * xp.astype(node_mask, pairs.vec.dtype)
            )
            frame_energy = frame_energy + segment_sum(
                self_pol, frame_id, n_node.shape[0]
            )
        frame_energy = frame_energy * COULOMB_CONSTANT
        if efield_node is not None:
            # Dipole-field interaction energy -E . (sum_i p_i).
            mu_frame = segment_sum(
                xp.astype(p, frame_energy.dtype), frame_id, n_node.shape[0]
            )
            frame_energy = frame_energy - xp.sum(
                xp.astype(efield_frame, frame_energy.dtype) * mu_frame, axis=-1
            )
        count = xp.astype(n_node, frame_energy.dtype)
        count = xp.maximum(count, xp.ones_like(count))
        energy = xp.take(frame_energy / count, frame_id, axis=0)
        if efield_ext_pot is not None:
            # Charge-field energy sum_i q_i phi_ext(r_i) on the continuous
            # polarization branch, attributed per node directly (q in e,
            # phi_ext in V, so already eV).
            energy = energy + xp.reshape(q, (n,)) * xp.reshape(
                xp.astype(efield_ext_pot, energy.dtype), (n,)
            )
        if k_grid is not None:
            # The k-space energy is attributed per atom as (q phi + p . F) / 2.
            energy = energy + kspace_multipole_energy(q, p, k_grid, self.kernel)
        if mesh is not None:
            energy = energy + mesh_multipole_energy(
                q, p, mesh, frame_id, self.kernel, self.pme_order
            )
        ret = {
            "energy": xp.reshape(energy, (n, 1)),
            "latent_charge": q,
            "latent_dipole": p,
        }
        if efield_node is not None and not branch_mode:
            # ``force_only`` mode: the uniform field force q_i E on each real
            # atom rides as a side output for the model wrapper to ADD to the
            # autograd-assembled force; the charge-field potential is then
            # omitted.  In ``branch`` mode the potential is part of the
            # energy and its force contribution (explicit slope + charge
            # relaxation) comes from differentiating the energy w.r.t. the
            # coordinate leaf inside the model lower.
            ret["efield_force"] = (
                xp.astype(q, efield_node.dtype) * efield_node * node_mask[:, None]
            )
        return ret

    def _constrain_fukui(
        self,
        charge: Array,
        fukui: Array,
        node_mask: Array,
        frame_id: Array,
        target: Array,
    ) -> Array:
        """Fukui-weighted charge-constraint redistribution of the SeZM update.

        Conserves the per-frame, per-channel ``target`` totals by sharing each
        frame's deficit over the real atoms in proportion to the learned
        Fukui weights.
        """
        xp = array_api_compat.array_namespace(charge)
        nf = target.shape[0]
        weight = fukui[:, 0] * xp.astype(node_mask, charge.dtype)
        weight_total = segment_sum(weight, frame_id, nf)
        weight_total = xp.where(
            weight_total > 1e-12, weight_total, xp.full_like(weight_total, 1e-12)
        )
        weights = weight / xp.take(weight_total, frame_id, axis=0)
        deficit = target - segment_sum(charge, frame_id, nf)
        return charge + weights[:, None] * xp.take(deficit, frame_id, axis=0)

    def _pair_potential_field(
        self,
        q: Array,
        p: Array,
        update_vec: Array,
        update_mask: Array,
        update_k: Array,
        update_a: Array,
        update_c: Array,
        src: Array,
        dst: Array,
        n: int,
        xp: Any,
    ) -> tuple[Array, Array]:
        """The pair-sum potential and electric field of the current state."""
        q_src = q[src, 0]
        q_dst = q[dst, 0]
        p_src = p[src]
        p_dst = p[dst]
        p_src_dot_r = xp.sum(p_src * update_vec, axis=-1)
        p_dst_dot_r = xp.sum(p_dst * update_vec, axis=-1)
        potential_dst = (q_src * update_k - update_a * p_src_dot_r) * update_mask
        potential_src = (q_dst * update_k + update_a * p_dst_dot_r) * update_mask
        field_dst = (
            q_src[:, None] * update_a[:, None] * update_vec
            - update_a[:, None] * p_src
            - update_c[:, None] * p_src_dot_r[:, None] * update_vec
        ) * update_mask[:, None]
        field_src = (
            -q_dst[:, None] * update_a[:, None] * update_vec
            - update_a[:, None] * p_dst
            - update_c[:, None] * p_dst_dot_r[:, None] * update_vec
        ) * update_mask[:, None]
        potential = (
            segment_sum(potential_dst, dst, n) + segment_sum(potential_src, src, n)
        )[:, None]
        field = segment_sum(field_dst, dst, n) + segment_sum(field_src, src, n)
        return potential, field

    def _radial_coefficients(
        self, r_sq: Array, src: Array, n: int
    ) -> tuple[Array, Array, Array]:
        """The pair kernel's ``K``, ``K'/r`` and ``(K''-K'/r)/r²``."""
        return self.kernel.evaluate_multipole(
            r_sq, self._kernel_coefficients(r_sq, src, n)
        )

    def _kernel_coefficients(self, r_sq: Array, src: Array, n: int) -> dict[str, Array]:
        return self.kernel.coefficients(r_sq)

    def serialize(self) -> dict:
        data = {
            "@class": "LongRange",
            "type": "latent_multipole",
            "@version": 1,
            "charge_net": self.charge_net.serialize(),
            "kernel": self.kernel.serialize(),
            "charge_constraint": self.charge_constraint,
            "use_self_pol": self.use_self_pol,
        }
        if self.efield:
            data["efield"] = True
            if self.efield_potential != "force_only":
                data["efield_potential"] = self.efield_potential
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
    def deserialize(cls, data: dict) -> "LatentMultipoleLongRange":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        periodic = data.get("periodic", False)
        return cls(
            LatentMultipoleFitting.deserialize(data["charge_net"]),
            BaseInteractionKernel.deserialize(data["kernel"]),
            data["charge_constraint"],
            data["use_self_pol"],
            efield=bool(data.get("efield", False)),
            # Records without the key predate the charge-potential coupling
            # and keep the legacy force-only charge coupling.
            efield_potential=str(data.get("efield_potential", "force_only")),
            periodic=periodic,
            kspace_tolerance=data.get("kspace_tolerance", 1e-8),
            # A periodic record that names no mode predates the mesh and
            # evaluates the explicit wave numbers.
            k_grid_mode=data.get("k_grid_mode", "kgrid" if periodic else "pme"),
            pme_order=data.get("pme_order", 8),
            pme_mesh=data.get("pme_mesh"),
            pme_mesh_spacing=data.get("pme_mesh_spacing"),
        )
