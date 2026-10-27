# SPDX-License-Identifier: LGPL-3.0-or-later
"""Direct and reciprocal interaction of descriptor-predicted latent multipoles."""

from typing import Any

import array_api_compat

from deepmd.dpmodel.array_api import Array, xp_erf
from deepmd.dpmodel.common import get_xp_precision
from deepmd.dpmodel.fitting.latent_multipole import LatentMultipoleFitting
from deepmd.dpmodel.utils.neighbor_graph import (
    ALL_PAIRS,
    K_GRID,
    frame_id_from_n_node,
    segment_sum,
)
from deepmd.dpmodel.utils.seed import child_seed
from deepmd.utils.version import check_version_compatibility

from .base_long_range import BaseLongRange
from .kernel import COULOMB_CONSTANT, BaseInteractionKernel
from .latent_charge import (
    CHARGE_NET_NEURON,
    CHARGE_NET_SEED_INDEX,
    LatentChargeLongRange,
)
from .reciprocal import (
    kspace_multipole_energy,
    kspace_multipole_potential_field,
)

_SQRT_PI = 1.7724538509055160272981674833411451827975494561224


@BaseLongRange.register("latent_multipole")
class LatentMultipoleLongRange(LatentChargeLongRange):
    """LES charge--charge, charge--dipole and dipole--dipole direct energy.

    Unordered intra-frame pairs are counted once.  Recurrent updates and the
    single self-polarization term use the final latent state, with no detached
    tensors anywhere in the readout/update/energy path.
    """

    def __init__(
        self,
        charge_net: LatentMultipoleFitting,
        kernel: BaseInteractionKernel,
        charge_constraint: dict | None = None,
        use_self_pol: bool = True,
        periodic: bool = False,
        kspace_tolerance: float = 1e-8,
    ) -> None:
        if kernel.TYPE != "les":
            raise NotImplementedError(
                "latent multipoles currently require the LES kernel"
            )
        super().__init__(
            charge_net,
            kernel,
            charge_constraint,
            periodic=periodic,
            kspace_tolerance=kspace_tolerance,
        )
        self.use_self_pol = bool(use_self_pol)

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
            "activation_function": fitting.activation_function,
            "resnet_dt": fitting.resnet_dt,
            "precision": fitting.precision,
            "seed": child_seed(fitting_seed, CHARGE_NET_SEED_INDEX),
            **config.get("charge_net", {}),
        }
        kernel_config = {
            "precision": charge_net_config["precision"],
            **config["kernel"],
        }
        if kernel_config["type"] != "les":
            raise NotImplementedError(
                "latent multipoles currently require the LES kernel"
            )
        kernel_cls = BaseInteractionKernel.get_class_by_type(kernel_config.pop("type"))
        embedding_width = int(config.get("embedding_width", 0))
        if embedding_width == 0:
            embedding_width = int(descriptor.degree_channels[1])
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
            embedding_width=embedding_width,
            alpha_pol=config.get("alpha_pol"),
            lr_update_steps=int(config.get("lr_update_steps", 0)),
            lr_update_neuron=config.get("lr_update_neuron"),
            lr_update_damping=float(config.get("lr_update_damping", 0.2)),
            **charge_net_config,
        )
        return cls(
            charge_net,
            kernel_cls(**kernel_config),
            config.get("charge_constraint"),
            bool(config.get("use_self_pol", True)),
            periodic=bool(config.get("periodic", False)),
            kspace_tolerance=float(config.get("kspace_tolerance", 1e-8)),
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
        descriptor_context: dict[str, Array] | None = None,
    ) -> dict[str, Array]:
        xp = array_api_compat.array_namespace(descriptor)
        if descriptor_context is None or descriptor_context.get("gr") is None:
            raise RuntimeError("latent multipoles require DPA4C degree-one moments")
        n = descriptor.shape[0]
        pairs = topologies[ALL_PAIRS]
        k_grid = topologies.get(K_GRID) if self.periodic else None
        frame_id = frame_id_from_n_node(n_node, n_total=n)
        fparam_node = None if fparam is None else xp.take(fparam, frame_id, axis=0)
        initial = self.charge_net.call_graph(
            descriptor,
            atype,
            gr=descriptor_context["gr"],
            fparam=fparam_node,
            aparam=aparam,
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
            for _ in range(self.charge_net.lr_update_steps):
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
                    # The electric field of the k-space sum is -dE/dp.
                    phi_k, force_k = kspace_multipole_potential_field(
                        q, p, k_grid, self.kernel
                    )
                    potential = potential + phi_k
                    field = field - force_k
                q, p = self.charge_net.update_polar_state(
                    update_descriptor, q, p, local_p, potential, field
                )
                q = q * mask_q
                p = p * mask_q
                if self.charge_constraint is not None:
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
            alpha_pol = self.charge_net._parameter_array("alpha_pol", pairs.vec)
            alpha_pol_atom = xp.take(alpha_pol, atype, axis=0) ** 2
            self_pol = (
                xp.sum(xp.astype(p, pairs.vec.dtype) ** 2, axis=-1)
                / (2.0 * alpha_pol_atom)
                * xp.astype(node_mask, pairs.vec.dtype)
            )
            frame_energy = frame_energy + segment_sum(
                self_pol, frame_id, n_node.shape[0]
            )
        frame_energy = frame_energy * COULOMB_CONSTANT
        count = xp.astype(n_node, frame_energy.dtype)
        count = xp.maximum(count, xp.ones_like(count))
        energy = xp.take(frame_energy / count, frame_id, axis=0)
        if k_grid is not None:
            # The k-space energy is attributed per atom as (q phi + p . F) / 2.
            energy = energy + kspace_multipole_energy(q, p, k_grid, self.kernel)
        return {
            "energy": xp.reshape(energy, (n, 1)),
            "latent_charge": q,
            "latent_dipole": p,
        }

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
        """Stable ``K``, ``K'/r`` and ``(K''-K'/r)/r²`` of LES."""
        xp = array_api_compat.array_namespace(r_sq)
        alpha = self._kernel_coefficients(r_sq, src, n)["alpha"]
        r_sq_safe = xp.maximum(r_sq, xp.full_like(r_sq, xp.finfo(r_sq.dtype).eps))
        r = xp.sqrt(r_sq_safe)
        x_sq = alpha * alpha * r_sq
        exp_t = xp.exp(-x_sq)
        erf_t = xp_erf(alpha * r)
        k_analytic = erf_t / r
        a_analytic = (2.0 * alpha / _SQRT_PI) * exp_t / r_sq_safe - erf_t / (
            r_sq_safe * r
        )
        kpp = (
            -(4.0 * alpha**3 / _SQRT_PI) * exp_t
            - (4.0 * alpha / _SQRT_PI) * exp_t / r_sq_safe
            + 2.0 * erf_t / (r_sq_safe * r)
        )
        c_analytic = (kpp - a_analytic) / r_sq_safe
        alpha2 = alpha * alpha
        alpha3 = alpha * alpha2
        alpha5 = alpha3 * alpha2
        alpha7 = alpha5 * alpha2
        alpha9 = alpha7 * alpha2
        r4 = r_sq * r_sq
        r6 = r4 * r_sq
        k_series = (2.0 / _SQRT_PI) * (
            alpha - alpha3 * r_sq / 3.0 + alpha5 * r4 / 10.0 - alpha7 * r6 / 42.0
        )
        a_series = (
            -4.0 * alpha3 / (3.0 * _SQRT_PI)
            + 4.0 * alpha5 * r_sq / (5.0 * _SQRT_PI)
            - 2.0 * alpha7 * r4 / (7.0 * _SQRT_PI)
            + 2.0 * alpha9 * r6 / (27.0 * _SQRT_PI)
        )
        c_series = (
            8.0 * alpha5 / (5.0 * _SQRT_PI)
            - 8.0 * alpha7 * r_sq / (7.0 * _SQRT_PI)
            + 4.0 * alpha9 * r4 / (9.0 * _SQRT_PI)
        )
        small = x_sq < 0.04
        return (
            xp.where(small, k_series, k_analytic),
            xp.where(small, a_series, a_analytic),
            xp.where(small, c_series, c_analytic),
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
        if self.periodic:
            data["periodic"] = True
            data["kspace_tolerance"] = self.kspace_tolerance
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "LatentMultipoleLongRange":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        return cls(
            LatentMultipoleFitting.deserialize(data["charge_net"]),
            BaseInteractionKernel.deserialize(data["kernel"]),
            data["charge_constraint"],
            data["use_self_pol"],
            periodic=data.get("periodic", False),
            kspace_tolerance=data.get("kspace_tolerance", 1e-8),
        )
