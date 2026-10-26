# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch wrappers for the long-range skeleton."""

import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    ALL_PAIRS,
    K_GRID,
    PME_MESH,
    build_topologies,
)
from deepmd.pt_expt.long_range import BaseLongRange, LESKernel, SOGKernel
from deepmd.pt_expt.model.edge_transform_output import releaf_topologies
from deepmd.pt_expt.model.get_model import DPAtomicModel


def test_torch_kernel_registration_and_serialization() -> None:
    les = LESKernel(alpha=0.7, trainable=False)
    assert isinstance(les.log_alpha, torch.nn.Parameter)
    assert not les.log_alpha.requires_grad
    r_sq = torch.tensor([0.0, 1.0], dtype=torch.float64)
    restored = LESKernel.deserialize(les.serialize())
    torch.testing.assert_close(restored(r_sq), les(r_sq), rtol=0.0, atol=0.0)

    sog = SOGKernel(amplitude=[0.7, -0.2], width=[0.8, 1.7])
    assert sog.amplitude.requires_grad
    assert SOGKernel.deserialize(sog.serialize()).amplitude.shape == (2,)


def test_releaf_topologies_detaches_geometry_inputs() -> None:
    coord = torch.arange(18, dtype=torch.float64).reshape(6, 3).requires_grad_(True)
    n_node = torch.tensor([2, 4])
    atype = torch.zeros(6, dtype=torch.int64)
    box = torch.stack([torch.eye(3, dtype=torch.float64) * 8.0] * 2)
    names = (ALL_PAIRS, K_GRID, PME_MESH)
    topologies = build_topologies(
        names,
        coord,
        n_node,
        box,
        atype=atype,
        topology_params={K_GRID: {"k_max": 1.0}, PME_MESH: {"spacing": 2.0}},
    )
    releafed = releaf_topologies(topologies)
    for name, topology in releafed.items():
        for field in topology.LEAVES:
            value = getattr(topology, field)
            assert value.is_leaf
            assert value.requires_grad
    assert releafed[ALL_PAIRS].vec.data_ptr() == topologies[ALL_PAIRS].vec.data_ptr()
    assert DPAtomicModel.base_long_range_cls is BaseLongRange
