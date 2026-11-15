# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for the long-range plugin and topology skeleton."""

import math

import numpy as np
import pytest

from deepmd.dpmodel.atomic_model import DPAtomicModel
from deepmd.dpmodel.descriptor import DescrptSeA
from deepmd.dpmodel.fitting import InvarFitting
from deepmd.dpmodel.long_range import (
    BaseInteractionKernel,
    BaseLongRange,
    LESKernel,
    SOGKernel,
)
from deepmd.dpmodel.model.model import get_standard_model
from deepmd.dpmodel.output_def import FittingOutputDef, OutputVariableDef
from deepmd.dpmodel.utils.neighbor_graph import (
    ALL_PAIRS,
    K_GRID,
    PME_MESH,
    build_topologies,
    cast_topology_geometry,
    flatten_topologies,
    unflatten_topologies,
)


@BaseLongRange.register("skeleton_test")
class SkeletonLongRange(BaseLongRange):
    """Minimal serializable plugin used to test composition, not its forward."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    @classmethod
    def from_config(
        cls, config, descriptor, fitting, fitting_seed=None
    ) -> "SkeletonLongRange":
        return cls(**config)

    def required_topologies(self) -> tuple[str, ...]:
        return (ALL_PAIRS,)

    def uses_charge_spin(self) -> bool:
        return False

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "latent_charge",
                    [1],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            ]
        )

    def call_graph(
        self,
        descriptor,
        atype,
        node_mask,
        n_node,
        topologies,
        fparam=None,
        aparam=None,
        charge_spin=None,
    ):
        return {
            "energy": np.zeros((atype.shape[0], 1)),
            "latent_charge": np.zeros((atype.shape[0], 1)),
        }

    def serialize(self) -> dict:
        return {
            "@class": "LongRange",
            "type": "skeleton_test",
            "@version": 1,
            "scale": self.scale,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SkeletonLongRange":
        return cls(scale=data["scale"])


def _atomic_model(long_range=None) -> DPAtomicModel:
    descriptor = DescrptSeA(6.0, 0.5, [4, 4])
    fitting = InvarFitting(
        "energy", 2, descriptor.get_dim_out(), 1, mixed_types=descriptor.mixed_types()
    )
    return DPAtomicModel(descriptor, fitting, ["A", "B"], long_range=long_range)


def test_les_kernel_math_and_round_trip() -> None:
    kernel = LESKernel(alpha=0.7)
    r = np.array([0.0, 0.5, 2.0])
    expected = np.array(
        [2 * 0.7 / math.sqrt(math.pi), math.erf(0.35) / 0.5, math.erf(1.4) / 2.0]
    )
    np.testing.assert_allclose(kernel(r**2), expected, rtol=1e-12)
    np.testing.assert_allclose(
        kernel.contact_value(r), expected[0], rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        kernel.kspace(np.array([4.0])), np.pi * np.exp(-1.0 / 0.49), rtol=1e-12
    )
    restored = BaseInteractionKernel.deserialize(kernel.serialize())
    np.testing.assert_allclose(restored(r**2), kernel(r**2), rtol=0.0, atol=0.0)


def test_sog_kernel_math_and_round_trip() -> None:
    kernel = SOGKernel(amplitude=[0.7, -0.2], width=[0.8, 1.7])
    r_sq = np.array([0.0, 1.0, 4.0])
    expected = 0.7 * np.exp(-r_sq / (2 * 0.8**2)) - 0.2 * np.exp(-r_sq / (2 * 1.7**2))
    np.testing.assert_allclose(kernel(r_sq), expected, rtol=1e-12)
    np.testing.assert_allclose(kernel.contact_value(r_sq), 0.5, rtol=1e-15, atol=0.0)
    restored = BaseInteractionKernel.deserialize(kernel.serialize())
    np.testing.assert_allclose(restored(r_sq), expected, rtol=0.0, atol=0.0)
    assert isinstance(restored, SOGKernel)


def test_topology_build_flatten_unflatten_and_cast() -> None:
    coord = np.arange(18, dtype=np.float64).reshape(6, 3)
    n_node = np.array([2, 4])
    atype = np.zeros(6, dtype=np.int64)
    box = np.stack([np.eye(3) * 8.0, np.eye(3) * 9.0])
    names = (ALL_PAIRS, K_GRID, PME_MESH)
    topologies = build_topologies(
        names,
        coord,
        n_node,
        box,
        atype=atype,
        topology_params={K_GRID: {"k_max": 1.0}, PME_MESH: {"spacing": 2.0}},
    )
    assert not topologies[ALL_PAIRS].mask.any()
    assert topologies[K_GRID].mask.any()
    assert topologies[PME_MESH].mask.all()

    flat = flatten_topologies(topologies)
    restored = unflatten_topologies(names, flat)
    for name in names:
        for original_value, restored_value in zip(
            flatten_topologies({name: topologies[name]}),
            flatten_topologies({name: restored[name]}),
            strict=True,
        ):
            np.testing.assert_array_equal(original_value, restored_value)
    cast = cast_topology_geometry(topologies[K_GRID], np.float32)
    assert cast.vec.dtype == np.float32
    assert cast.coord.dtype == np.float32
    assert cast.slot.dtype == np.int64

    nonperiodic = build_topologies((ALL_PAIRS,), coord, n_node, atype=atype)
    assert nonperiodic[ALL_PAIRS].mask.sum() == 7
    with pytest.raises(ValueError, match="flattened values"):
        unflatten_topologies(names, flat[:-1])


def test_atomic_model_without_long_range_keeps_version_two() -> None:
    model = _atomic_model()
    record = model.serialize()
    assert record["@version"] == 2
    assert "long_range" not in record
    assert model.required_topologies() == ()
    assert model.topology_params() == {}
    restored = DPAtomicModel.deserialize(record)
    assert restored.serialize()["@version"] == 2
    assert "long_range" not in restored.serialize()


def test_minimal_plugin_composes_and_round_trips() -> None:
    model = _atomic_model(SkeletonLongRange(scale=2.5))
    assert model.has_long_range()
    assert model.required_topologies() == (ALL_PAIRS,)
    record = model.serialize()
    assert record["@version"] == 3
    assert record["long_range"]["scale"] == 2.5
    restored = DPAtomicModel.deserialize(record)
    assert isinstance(restored.long_range, SkeletonLongRange)
    assert restored.long_range.scale == 2.5
    assert restored.serialize()["@version"] == 3
    with pytest.raises(NotImplementedError, match="skeleton"):
        model.forward_atomic(
            np.zeros((1, 1, 3)),
            np.zeros((1, 1), dtype=np.int64),
            np.full((1, 1, 1), -1, dtype=np.int64),
        )


def test_factory_constructs_registered_plugin() -> None:
    config = {
        "type_map": ["A", "B"],
        "descriptor": {"type": "se_e2_a", "rcut": 6.0, "rcut_smth": 0.5, "sel": [4, 4]},
        "fitting_net": {"type": "ener", "neuron": [4]},
        "long_range": {"type": "skeleton_test", "scale": 3.0},
    }
    model = get_standard_model(config)
    assert isinstance(model.atomic_model.long_range, SkeletonLongRange)
    assert model.atomic_model.long_range.scale == 3.0


def test_lef_dipole_field_matches_energy_gradient() -> None:
    """The LEF dipole field must be dU/dp with ONE consistent sign convention.

    Regression for a sign inconsistency: the direct pair sum
    (``_pair_potential_field``, inherited from the legacy DPA4C-LR
    recurrence) carries dU/dp = -E, while the k-space/PME generalized force
    (+dE/dp) was SUBTRACTED and the external field was ADDED as a physical
    E -- so the periodic LEF update mixed the two conventions.  The field
    assembled exactly as in ``LatentMultipoleLongRange.call_graph`` is
    compared against finite differences of the total (pair + k-space)
    energy.
    """
    from deepmd.dpmodel.long_range.kernel import (
        COULOMB_CONSTANT,
    )
    from deepmd.dpmodel.long_range.latent_multipole import (
        LatentMultipoleLongRange,
    )
    from deepmd.dpmodel.long_range.reciprocal import (
        kspace_multipole_energy,
        kspace_multipole_potential_field,
    )
    from deepmd.dpmodel.utils.neighbor_graph.reciprocal_graph import (
        build_k_grid,
    )

    rng = np.random.default_rng(20261115)
    natoms = 4
    box = np.eye(3)[None] * 6.0
    coord = rng.uniform(1.0, 5.0, (natoms, 3))
    n_node = np.array([natoms])
    q = rng.normal(0.0, 0.4, (natoms, 1))
    q = q - q.mean()
    p = rng.normal(0.0, 0.15, (natoms, 3))
    kernel = LESKernel(alpha=0.9)

    src, dst = np.triu_indices(natoms, k=1)
    vec = coord[dst] - coord[src]
    mask = np.ones(len(src), dtype=bool)
    r_sq = np.sum(vec * vec, axis=-1)
    k_val, a_coeff, c_coeff = kernel.evaluate_multipole(r_sq, kernel.coefficients(r_sq))

    def pair_energy(pp: np.ndarray) -> float:
        p_src, p_dst = pp[src], pp[dst]
        psr = np.sum(p_src * vec, axis=-1)
        pdr = np.sum(p_dst * vec, axis=-1)
        qs, qd = q[src, 0], q[dst, 0]
        u = (
            qs * qd * k_val
            + (qs * pdr - qd * psr) * a_coeff
            - a_coeff * np.sum(p_src * p_dst, axis=-1)
            - c_coeff * psr * pdr
        )
        return float(np.sum(u) * COULOMB_CONSTANT)

    k_grid = build_k_grid(box, 3.0, coord, n_node)

    def total_energy(pp: np.ndarray) -> float:
        return pair_energy(pp) + float(
            np.sum(kspace_multipole_energy(q, pp, k_grid, kernel))
        )

    # Assemble the LEF field exactly like LatentMultipoleLongRange.call_graph.
    _, field = LatentMultipoleLongRange._pair_potential_field(
        None, q, p, vec, mask, k_val, a_coeff, c_coeff, src, dst, natoms, np
    )
    _, force_k = kspace_multipole_potential_field(q, p, k_grid, kernel)
    field = field * COULOMB_CONSTANT + force_k  # all in the dU/dp convention

    eps = 1e-5
    numeric = np.zeros_like(p)
    for i in range(natoms):
        for d in range(3):
            dp_fd = np.zeros_like(p)
            dp_fd[i, d] = eps
            numeric[i, d] = (total_energy(p + dp_fd) - total_energy(p - dp_fd)) / (
                2 * eps
            )
    np.testing.assert_allclose(field, numeric, rtol=1e-4, atol=1e-6)
